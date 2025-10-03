#!/usr/bin/env python3
"""
FlexAction算法演示 - 无需硬件的简化版本
展示FlexAction在CSP空间中的核心算法流程
"""

import numpy as np
import random
from typing import Dict, List, Tuple
from dataclasses import dataclass
import json


@dataclass
class CSPConstraint:
    """CSP约束"""
    type: str  # IN, EQ, LE, PROD, SUM, SELECT
    params: tuple


@dataclass
class CSPSpace:
    """CSP搜索空间"""
    variables: Dict[str, List]  # 变量及其域
    constraints: List[CSPConstraint]  # 约束列表

    def is_valid(self, assignment: Dict) -> bool:
        """检查赋值是否满足所有约束"""
        for constraint in self.constraints:
            if not self._check_constraint(constraint, assignment):
                return False
        return True

    def _check_constraint(self, constraint: CSPConstraint, assignment: Dict) -> bool:
        """检查单个约束"""
        if constraint.type == "IN":
            var, values = constraint.params
            return assignment.get(var) in values

        elif constraint.type == "EQ":
            if len(constraint.params) == 2:
                if isinstance(constraint.params[0], tuple) and constraint.params[0][0] == "PROD":
                    # PROD约束
                    vars_list = constraint.params[0][1]
                    expected = constraint.params[1]
                    product = 1
                    for v in vars_list:
                        product *= assignment.get(v, 1)
                    return product == expected
                else:
                    var1, var2 = constraint.params
                    return assignment.get(var1) == assignment.get(var2)

        elif constraint.type == "LE":
            var, limit = constraint.params
            return assignment.get(var, 0) <= limit

        return True


class SimpleLambdaItem:
    """简化的Lambda项"""

    def __init__(self, name: str, constraint_template: Dict):
        self.name = name
        self.constraint_template = constraint_template
        self.usage_count = 0
        self.total_reward = 0
        self.avg_reward = 0

    def to_constraints(self) -> List[CSPConstraint]:
        """转换为CSP约束"""
        constraints = []
        for key, (ctype, *params) in self.constraint_template.items():
            constraints.append(CSPConstraint(ctype, tuple(params)))
        return constraints

    def update_stats(self, reward: float):
        """更新统计信息"""
        self.usage_count += 1
        self.total_reward += reward
        self.avg_reward = self.total_reward / self.usage_count


class FlexActionDemo:
    """FlexAction算法演示"""

    def __init__(self):
        # 初始化CSP空间（TensorCore GEMM示例）
        self.csp_initial = self._create_tensorcore_csp()

        # 初始化Lambda库
        self.lambda_library = self._init_lambda_library()

        # 性能历史
        self.performance_history = []
        self.best_performance = 0
        self.best_config = None

    def _create_tensorcore_csp(self) -> CSPSpace:
        """创建TensorCore的CSP空间"""
        variables = {
            'm': [8, 16, 32],
            'n': [8, 16, 32],
            'k': [16],
            'vector_length': [1, 2, 4, 8],
            'tile_i': list(range(1, 65)),
            'tile_j': list(range(1, 65)),
            'tile_r': list(range(1, 65)),
            'shared_mem': list(range(0, 48*1024, 1024))
        }

        constraints = [
            CSPConstraint("EQ", (("PROD", ['m', 'n', 'k']), 4096)),
            CSPConstraint("LE", ('shared_mem', 48*1024)),
            CSPConstraint("IN", ('vector_length', [1, 2, 4, 8]))
        ]

        return CSPSpace(variables, constraints)

    def _init_lambda_library(self) -> List[SimpleLambdaItem]:
        """初始化Lambda库"""
        library = []

        # L1: TensorCore配置
        library.append(SimpleLambdaItem(
            "TC_8x8x16",
            {'tc_m': ('IN', 'm', [8]),
             'tc_n': ('IN', 'n', [8]),
             'tc_k': ('IN', 'k', [16])}
        ))
        library.append(SimpleLambdaItem(
            "TC_16x16x16",
            {'tc_m': ('IN', 'm', [16]),
             'tc_n': ('IN', 'n', [16]),
             'tc_k': ('IN', 'k', [16])}
        ))
        library.append(SimpleLambdaItem(
            "TC_32x32x16",
            {'tc_m': ('IN', 'm', [32]),
             'tc_n': ('IN', 'n', [32]),
             'tc_k': ('IN', 'k', [16])}
        ))

        # L2: 向量化
        library.append(SimpleLambdaItem(
            "Vec_4",
            {'vec': ('IN', 'vector_length', [4])}
        ))
        library.append(SimpleLambdaItem(
            "Vec_8",
            {'vec': ('IN', 'vector_length', [8])}
        ))

        # L3: 内存优化
        library.append(SimpleLambdaItem(
            "Mem_Small",
            {'mem': ('LE', 'shared_mem', 16*1024)}
        ))
        library.append(SimpleLambdaItem(
            "Mem_Medium",
            {'mem': ('LE', 'shared_mem', 32*1024)}
        ))

        return library

    def select_lambda_batch(self, batch_size: int = 3) -> List[SimpleLambdaItem]:
        """选择一批Lambda项（简化的ε-greedy策略）"""
        epsilon = 0.2

        selected = []
        for _ in range(batch_size):
            if random.random() < epsilon:
                # 探索：随机选择
                item = random.choice(self.lambda_library)
            else:
                # 利用：选择平均奖励最高的
                if any(item.usage_count > 0 for item in self.lambda_library):
                    item = max(self.lambda_library, key=lambda x: x.avg_reward if x.usage_count > 0 else -1)
                else:
                    item = random.choice(self.lambda_library)

            selected.append(item)

        return selected

    def apply_lambda_batch(self, lambda_batch: List[SimpleLambdaItem]) -> CSPSpace:
        """应用Lambda批次，生成新的CSP"""
        # 深拷贝原始CSP
        new_csp = CSPSpace(
            self.csp_initial.variables.copy(),
            self.csp_initial.constraints.copy()
        )

        # 应用每个Lambda项的约束
        for item in lambda_batch:
            constraints = item.to_constraints()
            # 简化处理：覆盖相同类型的约束
            for new_constraint in constraints:
                # 检查是否与现有约束冲突
                conflict = False
                for i, existing in enumerate(new_csp.constraints):
                    if existing.type == new_constraint.type and \
                       len(existing.params) > 0 and len(new_constraint.params) > 0 and \
                       existing.params[0] == new_constraint.params[0]:
                        # 替换冲突的约束
                        new_csp.constraints[i] = new_constraint
                        conflict = True
                        break
                if not conflict:
                    new_csp.constraints.append(new_constraint)

        return new_csp

    def sample_from_csp(self, csp: CSPSpace, num_samples: int = 10) -> List[Dict]:
        """从CSP采样配置"""
        samples = []
        max_attempts = num_samples * 10

        for _ in range(max_attempts):
            if len(samples) >= num_samples:
                break

            # 随机采样
            assignment = {}
            for var, domain in csp.variables.items():
                if domain:
                    assignment[var] = random.choice(domain)

            # 检查是否满足约束
            if csp.is_valid(assignment):
                samples.append(assignment)

        return samples

    def simulate_performance(self, config: Dict) -> float:
        """模拟性能评估（替代实际硬件测量）"""
        # 简化的性能模型
        m, n, k = config.get('m', 16), config.get('n', 16), config.get('k', 16)
        vec_len = config.get('vector_length', 1)
        tile_i = config.get('tile_i', 16)
        tile_j = config.get('tile_j', 16)

        # 基础性能
        base_perf = 100.0

        # TensorCore加速
        if m * n * k == 4096:
            base_perf *= 10.0

        # 向量化加速
        base_perf *= (1 + vec_len * 0.1)

        # Tile优化
        if tile_i == tile_j and tile_i in [16, 32]:
            base_perf *= 1.2

        # 添加随机噪声
        noise = random.gauss(0, base_perf * 0.05)
        return max(0, base_perf + noise)

    def run_optimization(self, iterations: int = 20):
        """运行优化主循环"""
        print("Starting FlexAction Optimization")
        print("=" * 60)

        for iter_no in range(iterations):
            print(f"\nIteration {iter_no + 1}/{iterations}")

            # Step 1: 选择Lambda批次
            lambda_batch = self.select_lambda_batch(batch_size=2)
            print(f"Selected Lambda items: {[item.name for item in lambda_batch]}")

            # Step 2: 应用Lambda生成新CSP
            new_csp = self.apply_lambda_batch(lambda_batch)

            # Step 3: 从新CSP采样
            samples = self.sample_from_csp(new_csp, num_samples=5)
            print(f"Generated {len(samples)} valid samples")

            if not samples:
                print("No valid samples, trying random sampling...")
                samples = self.sample_from_csp(self.csp_initial, num_samples=5)

            # Step 4: 评估性能
            performances = [self.simulate_performance(s) for s in samples]
            best_sample_perf = max(performances) if performances else 0
            best_sample_idx = performances.index(best_sample_perf) if performances else 0

            print(f"Best sample performance: {best_sample_perf:.2f} GFLOPS")

            # Step 5: 计算奖励并更新Lambda统计
            baseline = self.best_performance if self.best_performance > 0 else 100
            reward = (best_sample_perf - baseline) / baseline

            for item in lambda_batch:
                item.update_stats(reward)

            # Step 6: 更新全局最佳
            if best_sample_perf > self.best_performance:
                self.best_performance = best_sample_perf
                self.best_config = samples[best_sample_idx] if samples else None
                print(f"New best performance: {self.best_performance:.2f} GFLOPS")

            self.performance_history.append(best_sample_perf)

            # Step 7: Lambda库演化（每5轮）
            if (iter_no + 1) % 5 == 0:
                self.evolve_lambda_library()

        print("\n" + "=" * 60)
        print("Optimization completed!")
        print(f"Best performance: {self.best_performance:.2f} GFLOPS")
        if self.best_config:
            print(f"Best configuration:")
            for key, value in self.best_config.items():
                if key in ['m', 'n', 'k', 'vector_length']:
                    print(f"  {key}: {value}")

    def evolve_lambda_library(self):
        """演化Lambda库"""
        # 按平均奖励排序
        sorted_items = sorted(self.lambda_library,
                            key=lambda x: x.avg_reward if x.usage_count > 0 else -float('inf'),
                            reverse=True)

        # 生成新的组合Lambda项
        if len(sorted_items) >= 2 and sorted_items[0].avg_reward > 0 and sorted_items[1].avg_reward > 0:
            # 组合表现最好的两个Lambda项
            item1, item2 = sorted_items[0], sorted_items[1]
            combined_name = f"Combined_{item1.name}_{item2.name}"

            # 检查是否已存在
            if not any(item.name == combined_name for item in self.lambda_library):
                combined_constraints = {}
                combined_constraints.update(item1.constraint_template)
                combined_constraints.update(item2.constraint_template)

                new_item = SimpleLambdaItem(combined_name, combined_constraints)
                self.lambda_library.append(new_item)
                print(f"Generated new Lambda item: {combined_name}")

        # 淘汰表现差的Lambda项（保持库大小）
        max_size = 15
        if len(self.lambda_library) > max_size:
            self.lambda_library = sorted_items[:max_size]
            print(f"Library size maintained at {len(self.lambda_library)} items")

    def save_results(self, filename: str = "flexaction_demo_results.json"):
        """保存结果"""
        results = {
            "best_performance": self.best_performance,
            "best_config": self.best_config,
            "performance_history": self.performance_history,
            "lambda_stats": [
                {
                    "name": item.name,
                    "usage_count": item.usage_count,
                    "avg_reward": item.avg_reward
                }
                for item in self.lambda_library
            ]
        }

        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\nResults saved to {filename}")


def main():
    """主函数"""
    print("=" * 60)
    print("FlexAction Algorithm Demonstration")
    print("CSP-based Optimization with Lambda Items")
    print("=" * 60)

    # 创建演示实例
    demo = FlexActionDemo()

    # 运行优化
    demo.run_optimization(iterations=20)

    # 显示Lambda项统计
    print("\n" + "=" * 60)
    print("Lambda Library Statistics:")
    for item in demo.lambda_library:
        if item.usage_count > 0:
            print(f"  {item.name}: used={item.usage_count}, avg_reward={item.avg_reward:.3f}")

    # 保存结果
    demo.save_results("/root/Heron104/flexaction_demo_results.json")

    # 绘制性能历史（文本图表）
    print("\n" + "=" * 60)
    print("Performance History:")
    history = demo.performance_history
    if history:
        max_perf = max(history)
        scale = 50 / max_perf if max_perf > 0 else 1
        for i, perf in enumerate(history):
            bar_len = int(perf * scale)
            bar = '█' * bar_len
            print(f"Iter {i+1:2d}: {bar} {perf:.1f}")

    print("\n" + "=" * 60)
    print("Demonstration completed successfully!")


if __name__ == "__main__":
    main()