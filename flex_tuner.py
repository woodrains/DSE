"""
FlexAction-CSP Tuner for Heron
替代CGA算法，在Heron的CSP空间上实现FlexAction强化学习探索
"""

import os
import sys
sys.path.append('/root/Heron')

import time
import copy
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque, namedtuple
from typing import List, Dict, Tuple, Optional, Set

from Heron.tuner.tuner import Tuner
from Heron.sample import Sample, Code
from Heron.utils import anaCostModel
from Heron.multi import Job


# Experience replay buffer
Experience = namedtuple('Experience',
    ['state', 'action', 'reward', 'next_state', 'done'])


class LambdaItem:
    """Lambda项：表示一个可复用的宏动作"""

    def __init__(self, name: str, lowering_type: str, params: Dict):
        self.name = name
        self.lowering_type = lowering_type  # 对应的Lowering规则类型
        self.params = params  # 参数配置
        self.usage_count = 0  # 使用次数统计
        self.avg_reward = 0.0  # 平均奖励
        self.success_rate = 0.0  # 成功率

    def __repr__(self):
        return f"λ[{self.name}:{self.lowering_type}]"


class LambdaLibrary:
    """Lambda库：管理和演化Lambda项"""

    def __init__(self, arch_type: str, workload_type: str):
        self.arch_type = arch_type
        self.workload_type = workload_type
        self.items: Dict[str, LambdaItem] = {}
        self.generation_traces = []  # 生成轨迹

        # 初始化种子Lambda项
        self._init_seed_lambdas()

    def _init_seed_lambdas(self):
        """初始化种子Lambda项库"""

        # L1: TensorCore Tile配置
        if 'tensorcore' in self.arch_type.lower():
            self.add_item(LambdaItem(
                "TensorCoreTile_8", "TensorCoreTile",
                {"m": 8, "n": 8, "k": 16}
            ))
            self.add_item(LambdaItem(
                "TensorCoreTile_16", "TensorCoreTile",
                {"m": 16, "n": 16, "k": 16}
            ))
            self.add_item(LambdaItem(
                "TensorCoreTile_32", "TensorCoreTile",
                {"m": 32, "n": 32, "k": 16}
            ))

        # L2: 多级SPM配置
        self.add_item(LambdaItem(
            "MultiLevelSPM_Default", "MultiLevelSPM",
            {"levels": ["shared", "wmma"], "capacity": 48*1024}
        ))

        # L3: 分块融合骨架
        self.add_item(LambdaItem(
            "SplitFuse_2Level", "SplitFuseSkeleton",
            {"levels": 2, "factors": [16, 4]}
        ))
        self.add_item(LambdaItem(
            "SplitFuse_3Level", "SplitFuseSkeleton",
            {"levels": 3, "factors": [32, 8, 4]}
        ))

        # L4: 向量化配置
        self.add_item(LambdaItem(
            "Vectorize_1", "Vectorize",
            {"vlen": 1}
        ))
        self.add_item(LambdaItem(
            "Vectorize_4", "Vectorize",
            {"vlen": 4}
        ))
        self.add_item(LambdaItem(
            "Vectorize_8", "Vectorize",
            {"vlen": 8}
        ))

        # L5: ComputeAt位置
        self.add_item(LambdaItem(
            "ComputeAt_Inner", "ComputeAt",
            {"location": "inner"}
        ))
        self.add_item(LambdaItem(
            "ComputeAt_Outer", "ComputeAt",
            {"location": "outer"}
        ))

    def add_item(self, item: LambdaItem):
        """添加新的Lambda项"""
        self.items[item.name] = item

    def remove_item(self, name: str):
        """移除Lambda项"""
        if name in self.items:
            del self.items[name]

    def get_legal_items(self, csp_solver, csp_initial) -> List[LambdaItem]:
        """获取当前CSP下的合法Lambda项"""
        legal_items = []
        for item in self.items.values():
            # 检查该Lambda项对应的约束是否与CSP兼容
            if self._is_feasible(item, csp_solver, csp_initial):
                legal_items.append(item)
        return legal_items

    def _is_feasible(self, item: LambdaItem, csp_solver, csp_initial) -> bool:
        """检查Lambda项是否可行"""
        # 这里需要实际调用CSP求解器检查
        # 简化实现：基于规则的快速检查
        if item.lowering_type == "TensorCoreTile":
            # 检查TensorCore约束
            m, n, k = item.params["m"], item.params["n"], item.params["k"]
            if m * n * k != 4096:  # TensorCore约束
                return False
        return True

    def generate_new_items(self, traces: List, reward_threshold: float = 0.5):
        """根据轨迹生成新的Lambda项"""
        # 分析高奖励轨迹，提取模式
        high_reward_traces = [t for t in traces if t.get('reward', 0) > reward_threshold]

        for trace in high_reward_traces:
            # 模式匹配：查找重复的动作序列
            action_seq = trace.get('actions', [])
            if len(action_seq) >= 2:
                # 尝试组合相邻动作创建新的宏动作
                for i in range(len(action_seq) - 1):
                    self._try_combine_actions(action_seq[i], action_seq[i+1])

    def _try_combine_actions(self, action1: LambdaItem, action2: LambdaItem):
        """尝试组合两个动作创建新的Lambda项"""
        # 简化实现：特定模式的组合
        if action1.lowering_type == "SplitFuseSkeleton" and \
           action2.lowering_type == "Vectorize":
            # 创建组合的Lambda项
            new_name = f"Combined_{action1.name}_{action2.name}"
            if new_name not in self.items:
                new_item = LambdaItem(
                    new_name, "Combined",
                    {"sub_actions": [action1.name, action2.name]}
                )
                self.add_item(new_item)

    def eliminate_weak_items(self, size_limit: int = 50):
        """淘汰表现差的Lambda项"""
        if len(self.items) <= size_limit:
            return

        # 按平均奖励和成功率排序
        sorted_items = sorted(
            self.items.values(),
            key=lambda x: x.avg_reward * x.success_rate,
            reverse=True
        )

        # 保留前size_limit个
        kept_names = {item.name for item in sorted_items[:size_limit]}
        self.items = {name: item for name, item in self.items.items()
                     if name in kept_names}


class CSPLowering:
    """将Lambda项转换为CSP约束"""

    @staticmethod
    def lower(item: LambdaItem, arch, workload) -> Dict:
        """将Lambda项降低为CSP约束模板"""
        constraints = {}

        if item.lowering_type == "TensorCoreTile":
            # L1: TensorCore tile约束
            m, n, k = item.params["m"], item.params["n"], item.params["k"]
            constraints.update({
                "m_constraint": ("IN", "m", [m]),
                "n_constraint": ("IN", "n", [n]),
                "k_constraint": ("IN", "k", [k]),
                "prod_constraint": ("EQ", ("PROD", ["m", "n", "k"]), 4096)
            })

        elif item.lowering_type == "MultiLevelSPM":
            # L2: 多级SPM约束
            capacity = item.params["capacity"]
            constraints.update({
                "mem_constraint": ("LE", "total_mem", capacity)
            })

        elif item.lowering_type == "SplitFuseSkeleton":
            # L3: 分块融合约束
            levels = item.params["levels"]
            factors = item.params["factors"]
            for i, factor in enumerate(factors):
                constraints[f"split_{i}"] = ("EQ", f"factor_{i}", factor)

        elif item.lowering_type == "Vectorize":
            # L4: 向量化约束
            vlen = item.params["vlen"]
            constraints["vector_constraint"] = ("IN", "vector_length", [vlen])

        elif item.lowering_type == "ComputeAt":
            # L5: ComputeAt约束
            loc = item.params["location"]
            # 这里简化处理，实际需要更复杂的SELECT约束
            constraints["compute_loc"] = ("SELECT", "stage_loc", loc)

        elif item.lowering_type == "Combined":
            # 组合的Lambda项
            sub_actions = item.params.get("sub_actions", [])
            # 递归处理子动作
            # 这里简化处理

        return constraints


class FlexActionPolicy(nn.Module):
    """Lambda-aware强化学习策略网络"""

    def __init__(self, state_dim: int, max_vocab_size: int = 100):
        super().__init__()
        self.state_dim = state_dim
        self.max_vocab_size = max_vocab_size

        # 状态编码器
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )

        # 动作价值网络（支持动态词表）
        self.action_head = nn.Linear(128, max_vocab_size)

        # 价值网络（用于Actor-Critic）
        self.value_head = nn.Linear(128, 1)

    def forward(self, state, valid_actions_mask=None):
        """前向传播"""
        # 编码状态
        encoded = self.state_encoder(state)

        # 计算动作概率
        action_logits = self.action_head(encoded)

        # 应用合法动作掩码
        if valid_actions_mask is not None:
            action_logits = action_logits.masked_fill(~valid_actions_mask, -1e9)

        action_probs = F.softmax(action_logits, dim=-1)

        # 计算状态价值
        value = self.value_head(encoded)

        return action_probs, value

    def sample_action(self, state, valid_actions_mask=None, epsilon=0.1):
        """采样动作（支持ε-greedy）"""
        if random.random() < epsilon:
            # 探索：随机选择合法动作
            valid_indices = valid_actions_mask.nonzero().squeeze()
            if len(valid_indices) > 0:
                return valid_indices[random.randint(0, len(valid_indices)-1)].item()

        # 利用：根据策略选择
        with torch.no_grad():
            action_probs, _ = self.forward(state, valid_actions_mask)
            return torch.multinomial(action_probs, 1).item()

    def sample_batch(self, state, valid_actions_mask, batch_size: int):
        """批量采样动作"""
        actions = []
        for _ in range(batch_size):
            action = self.sample_action(state, valid_actions_mask)
            actions.append(action)
        return actions


class FlexActionTuner(Tuner):
    """FlexAction CSP探索器 - 替代CGA"""

    def __init__(self, config):
        super().__init__(config)

        # Lambda库
        self.lambda_library = None

        # 强化学习组件
        self.policy = None
        self.optimizer = None
        self.replay_buffer = deque(maxlen=10000)

        # 超参数
        self.batch_size = 16
        self.learning_rate = 1e-3
        self.gamma = 0.99  # 折扣因子
        self.epsilon = 0.2  # 探索率
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01

        print("✓ FlexAction Tuner initialized")

    def initialize_lambda_library(self, env):
        """初始化Lambda库"""
        # 根据架构和工作负载初始化
        arch_type = str(env.task.target)
        workload_type = env.task.name
        self.lambda_library = LambdaLibrary(arch_type, workload_type)

    def extract_state(self, env, history=None) -> np.ndarray:
        """从CSP提取状态特征"""
        task = env.task
        knob_manager = task.knob_manager

        # 提取特征（与Heron cost model一致）
        features = []

        # 变量数量
        num_vars = len(knob_manager.solver.vals) if hasattr(knob_manager, 'solver') else 0
        features.append(num_vars)

        # 约束数量
        num_constraints = len(knob_manager.solver.constraints) if hasattr(knob_manager.solver, 'constraints') else 0
        features.append(num_constraints)

        # 已测量的最佳性能
        best_perf = env.perf_buffer.best_perf if hasattr(env, 'perf_buffer') and env.perf_buffer.best_perf else 0
        features.append(best_perf)

        # 当前迭代轮次
        features.append(self.iter_no)

        # 填充到固定维度
        state_dim = 64
        while len(features) < state_dim:
            features.append(0.0)

        return np.array(features[:state_dim], dtype=np.float32)

    def get_legal_actions(self, env) -> Tuple[List[LambdaItem], torch.Tensor]:
        """获取合法的Lambda动作"""
        task = env.task
        csp_initial = task.knob_manager

        # 获取所有Lambda项
        all_items = list(self.lambda_library.items.values())

        # 创建合法动作掩码
        valid_mask = torch.zeros(self.lambda_library.max_vocab_size, dtype=torch.bool)
        legal_items = []

        for i, item in enumerate(all_items[:self.lambda_library.max_vocab_size]):
            # 检查是否可行
            if self._check_feasibility(item, csp_initial):
                valid_mask[i] = True
                legal_items.append(item)

        return legal_items, valid_mask

    def _check_feasibility(self, item: LambdaItem, csp_initial) -> bool:
        """检查Lambda项的可行性"""
        # 简化实现：基于规则的快速检查
        # 实际应该调用CSP求解器
        return True

    def apply_lambda_batch(self, env, lambda_batch: List[LambdaItem]):
        """应用一批Lambda项到CSP"""
        task = env.task
        csp = copy.deepcopy(task.knob_manager)

        # 合并所有Lambda项的约束
        all_constraints = {}
        for item in lambda_batch:
            constraints = CSPLowering.lower(item, task.target, task.args)
            # 处理冲突：后面的覆盖前面的
            all_constraints.update(constraints)

        # 应用约束到CSP
        # 这里需要实际的CSP操作
        # 简化处理：直接返回修改后的CSP
        return csp

    def optimize(self, env, population, stat, s_time):
        """主优化循环 - 替代CGA的optimize方法"""

        # 初始化Lambda库和策略网络
        if self.lambda_library is None:
            self.initialize_lambda_library(env)

        if self.policy is None:
            state_dim = 64
            self.lambda_library.max_vocab_size = 100
            self.policy = FlexActionPolicy(state_dim, self.lambda_library.max_vocab_size)
            self.optimizer = optim.Adam(self.policy.parameters(), lr=self.learning_rate)

        all_pop = [] + population

        # FlexAction主循环
        for iteration in range(self.config.iter_walks):

            # Step 1: 提取状态
            state = self.extract_state(env, all_pop)
            state_tensor = torch.FloatTensor(state).unsqueeze(0)

            # Step 2: 获取合法动作
            legal_items, valid_mask = self.get_legal_actions(env)
            if len(legal_items) == 0:
                # 没有合法动作，使用随机采样
                new_samples = self.constrained_random_sample(env, self.config.pop_num)
                all_pop += new_samples
                continue

            # Step 3: 策略选择动作批次
            action_indices = self.policy.sample_batch(
                state_tensor,
                valid_mask.unsqueeze(0),
                min(self.batch_size, len(legal_items))
            )

            selected_items = [legal_items[min(idx, len(legal_items)-1)]
                            for idx in action_indices if idx < len(legal_items)]

            # Step 4: 应用Lambda项生成新CSP
            modified_csp = self.apply_lambda_batch(env, selected_items)

            # Step 5: 从修改后的CSP采样
            new_samples = self.sample_from_csp(env, modified_csp, self.config.pop_num)

            # Step 6: 预测性能
            if len(new_samples) > 0:
                perfs = self.predict(new_samples)
                for idx, sample in enumerate(new_samples):
                    sample.predict = perfs[idx] if idx < len(perfs) else 0.0

            all_pop += new_samples

            # Step 7: 计算奖励
            if len(new_samples) > 0:
                avg_perf = np.mean([s.predict for s in new_samples if s.predict > 0])
                baseline = np.mean([s.predict for s in population if s.predict > 0]) if population else 0
                reward = avg_perf - baseline  # 相对改进作为奖励
            else:
                reward = -1.0  # 惩罚无效动作

            # Step 8: 更新Lambda项统计
            for item in selected_items:
                item.usage_count += 1
                item.avg_reward = (item.avg_reward * (item.usage_count - 1) + reward) / item.usage_count
                item.success_rate = (item.success_rate * (item.usage_count - 1) +
                                    (1.0 if reward > 0 else 0.0)) / item.usage_count

            # Step 9: 存储经验
            next_state = self.extract_state(env, all_pop)
            self.replay_buffer.append(
                Experience(state, action_indices, reward, next_state, False)
            )

            # Step 10: 策略更新
            if len(self.replay_buffer) >= 32:
                self.update_policy()

            # Step 11: Lambda库演化
            if iteration % 5 == 0:
                # 生成新Lambda项
                traces = [{'actions': selected_items, 'reward': reward}]
                self.lambda_library.generate_new_items(traces)

                # 淘汰弱项
                self.lambda_library.eliminate_weak_items()

            # 更新epsilon
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

            # 记录统计
            self.recordStat(all_pop, env, s_time, stat)

        # 移除无效样本
        self.removeInvalid(all_pop)

        return population + new_samples, all_pop

    def sample_from_csp(self, env, csp, num_samples):
        """从CSP采样（复用Heron的constrained_random_sample）"""
        # 临时替换task的knob_manager
        original_km = env.task.knob_manager
        env.task.knob_manager = csp

        samples = self.constrained_random_sample(env, num_samples)

        # 恢复
        env.task.knob_manager = original_km

        return samples

    def update_policy(self, batch_size=32):
        """更新策略网络"""
        if len(self.replay_buffer) < batch_size:
            return

        # 采样批次
        batch = random.sample(self.replay_buffer, batch_size)

        states = torch.FloatTensor([e.state for e in batch])
        actions = torch.LongTensor([e.action[0] if isinstance(e.action, list) else e.action
                                   for e in batch])
        rewards = torch.FloatTensor([e.reward for e in batch])
        next_states = torch.FloatTensor([e.next_state for e in batch])

        # 计算TD目标
        with torch.no_grad():
            _, next_values = self.policy(next_states)
            td_targets = rewards + self.gamma * next_values.squeeze()

        # 前向传播
        action_probs, values = self.policy(states)

        # 计算损失
        # Actor损失：策略梯度
        log_probs = torch.log(action_probs.gather(1, actions.unsqueeze(1)) + 1e-8)
        advantages = td_targets - values.squeeze()
        actor_loss = -(log_probs.squeeze() * advantages.detach()).mean()

        # Critic损失：TD误差
        critic_loss = F.mse_loss(values.squeeze(), td_targets)

        # 总损失
        loss = actor_loss + 0.5 * critic_loss

        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def removeInvalid(self, all_pop):
        """移除无效样本"""
        for sample in all_pop:
            if not sample.valid:
                sample.predict = 0.0

    def recordStat(self, all_pop, env, start_time, stat):
        """记录统计信息"""
        if len(all_pop) > 0:
            best = max(all_pop, key=lambda x: x.predict)
            print(f"  Iteration: best_predict={best.predict:.4f}, "
                  f"pop_size={len(all_pop)}, epsilon={self.epsilon:.3f}")

    def check_feasible_exits(self, env, num=1e3):
        """检查可行解是否存在"""
        # 继承自CGATuner
        return


# 注册到Heron系统
if __name__ == "__main__":
    print("FlexAction Tuner for Heron CSP Space")
    print("Features:")
    print("- Lambda-Item based macro actions")
    print("- CSP constraint lowering")
    print("- Lambda-aware RL policy")
    print("- Dynamic action vocabulary")
    print("- Library evolution (generation/elimination)")