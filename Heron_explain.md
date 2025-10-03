# Heron项目架构与实现说明文档

## 1. 项目概述

### 1.1 项目简介
Heron是一个面向深度学习加速器(DLA)的高性能库自动生成系统。该项目发表于ASPLOS 2023会议,旨在解决深度学习加速器程序自动生成中的核心挑战:如何在包含大量架构约束的情况下生成高质量的程序代码。

### 1.2 核心创新点
根据论文《Heron: Automatically Constrained High-Performance Library Generation for Deep Learning Accelerators》,Heron的主要创新包括:

1. **自动约束生成**: 通过静态分析自动生成精确的架构约束,而非手工编写
2. **约束遗传算法(CGA)**: 提出新型约束遗传算法,直接在约束满足问题(CSP)上进行演化操作
3. **高质量搜索空间**: 自动剪枝无效程序,生成高质量的约束搜索空间

### 1.3 支持的硬件平台
- **NVIDIA TensorCore** (V100, T4, A100)
- **Intel DL Boost** (Xeon处理器)
- **TVM VTA** (Versatile Tensor Accelerator)

### 1.4 性能表现
- 相比AutoTVM、Ansor、AMOS等方法,平均加速2.71倍
- 相比厂商优化库(cuDNN/cuBLAS/oneDNN),平均加速2.00倍

---

## 2. 项目架构

### 2.1 整体架构图

```
┌─────────────────────────────────────────────────────────┐
│                     Heron System                         │
├─────────────────────────────────────────────────────────┤
│                                                           │
│  ┌──────────────────────────────────────────────────┐  │
│  │          Generation Stage                         │  │
│  │  ┌────────────────────────────────────────────┐  │  │
│  │  │         Space Generator                     │  │  │
│  │  │  - Schedule Template Generation             │  │  │
│  │  │  - Constraint Generation                    │  │  │
│  │  │  - Generation Rules (S1-S3, C1-C6)         │  │  │
│  │  └────────────────────────────────────────────┘  │  │
│  └──────────────────────────────────────────────────┘  │
│                           ↓                              │
│              Constrained Search Space (CSP_initial)      │
│                           ↓                              │
│  ┌──────────────────────────────────────────────────┐  │
│  │          Exploration Stage                        │  │
│  │  ┌────────────────────────────────────────────┐  │  │
│  │  │   Space Explorer (CGA Algorithm)           │  │  │
│  │  │   - Constraint-based Crossover             │  │  │
│  │  │   - Constraint-based Mutation              │  │  │
│  │  │   - CSP Solver (OR-Tools)                  │  │  │
│  │  └────────────────────────────────────────────┘  │  │
│  │  ┌────────────────────────────────────────────┐  │  │
│  │  │   Cost Model (XGBoost)                     │  │  │
│  │  │   - Feature Extraction                     │  │  │
│  │  │   - Performance Prediction                 │  │  │
│  │  └────────────────────────────────────────────┘  │  │
│  │  ┌────────────────────────────────────────────┐  │  │
│  │  │   DLA Measurer                             │  │  │
│  │  │   - Hardware Performance Measurement       │  │  │
│  │  └────────────────────────────────────────────┘  │  │
│  └──────────────────────────────────────────────────┘  │
│                           ↓                              │
│              High-Performance Programs                    │
└─────────────────────────────────────────────────────────┘
```

### 2.2 核心模块关系

```
Input (Compute Description)
    ↓
┌─────────────────────┐
│  Task & Environment │ ← config.py (Configuration)
└─────────────────────┘
    ↓
┌─────────────────────────────────────┐
│    Schedule Generation              │
│  - primitives.py (Schedule Ops)     │
│  - constraints/ (Constraint Rules)  │
│  - context/ (Execution Context)     │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│    Tuner (Optimization)             │
│  - tuner.py (Base Tuner)            │
│  - ga_tuner.py (CGA Implementation) │
│  - Cost Model (XGBoost)             │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│    Runner & Measurement             │
│  - runner/ (Code Execution)         │
│  - perf/ (Performance Buffer)       │
└─────────────────────────────────────┘
    ↓
Optimized Program
```

---

## 3. 核心组件详解

### 3.1 配置系统 (`config.py`)

**文件位置**: `/root/Heron/config.py`

**核心功能**:
- 管理环境配置参数
- 设置优化算法参数(遗传算法、模拟退火等)
- 配置代价模型参数
- 管理并行化设置

**关键代码**:
```python
class Config:
    def __init__(self):
        # 环境配置
        self.target_name = None
        self.use_cost_model = True

        # 并行配置
        self.parallel = True
        self.parallel_num = 16

        # 遗传算法参数
        self.pop_num = 200  # 种群大小
        self.select_num = 100  # 选择数量
        self.crossover_key_ratio = 0.2  # 交叉关键变量比例
```

**支持的优化方法**:
- `CGA`: 约束遗传算法(核心方法)
- `GA`: 标准遗传算法
- `SA`: 模拟退火
- `CRAND`: 约束随机采样

### 3.2 环境管理 (`environment.py`)

**文件位置**: `/root/Heron/environment.py`

**核心功能**:
1. **任务管理**: 创建和管理优化任务
2. **调度生成**: 生成schedule模板
3. **性能测量**: 协调硬件测量
4. **调优流程**: 控制整体优化流程

**关键类和方法**:
```python
class Env:
    def __init__(self, measure_option, config):
        self.runner = Runner(measure_option)  # 硬件运行器
        self.task = None  # 当前任务
        self.tuner = None  # 优化器

    def createTask(self, name, opfunc, args, target):
        """创建优化任务"""
        task = Task(name, opfunc, args, target)
        # 根据配置选择tuner
        if self.config.opt_method == 'CGA':
            self.tuner = CGATuner(self.config)

    def tune(self, task_name, pretrained=False):
        """执行调优"""
        self.task.make_stage_schedules()  # 生成调度模板
        res = self.tuner.run(self)  # 运行优化器
        return res
```

**工作流程** (environment.py:31-43):
1. 初始化日志目录
2. 生成stage schedules
3. 创建测量批次
4. 运行tuner优化
5. 记录优化时间

### 3.3 调度原语 (`schedule/primitives.py`)

**文件位置**: `/root/Heron/schedule/primitives.py`

**核心功能**: 封装TVM的调度操作,提供约束感知的程序变换

**主要调度原语**:

1. **split** (primitives.py:5-37):
   - 功能: 分割循环,支持按因子或按份数分割
   - 约束更新: 自动更新循环长度依赖关系

2. **fuse** (primitives.py:53-76):
   - 功能: 融合多个循环
   - 约束更新: 更新融合后的循环长度约束

3. **tensorize** (primitives.py:175-198):
   - 功能: 使用硬件intrinsic替换计算
   - 支持: TensorCore wmma, DL Boost, VTA intrinsics

4. **cache_read/cache_write** (primitives.py:133-149):
   - 功能: 使用片上存储(shared memory)
   - 约束: 自动记录内存使用

**代码示例** (split操作):
```python
def split(ctx, stage, ax, knob_key, nparts=None, factor=None):
    """分割循环并更新约束依赖图"""
    if factor != None:
        axo, axi = stage.split(ax, factor=factor)
        # 更新父子关系
        ctx.knob_manager.updateAxisParents(stage.op.name, axo.var.name, [ax.var.name])
        ctx.knob_manager.updateAxisParents(stage.op.name, axi.var.name, [ax.var.name])
        # 添加约束
        ctx.knob_manager.addSplitFactor(ax_key, axo_key, axi_key, knob_key)
```

### 3.4 调优器系统 (`tuner/`)

#### 3.4.1 基础调优器 (`tuner/tuner.py`)

**文件位置**: `/root/Heron/tuner/tuner.py`

**核心类**: `Tuner` (tuner.py:12-257)

**主要方法**:

1. **run** (tuner.py:47-87): 主优化循环
   ```python
   def run(self, env, do_measure=True):
       """主优化循环"""
       for iter_no in range(total_iters):
           # 1. 更新种群
           initial_population = self.UpdatePopulation(env, population)
           # 2. 执行优化(遗传算法/模拟退火等)
           population, all_pop = self.optimize(env, initial_population, stat, start_time)
           # 3. ε-greedy选择
           pop = self.epsilon_select(env, all_pop, 0.6, select_num)
           # 4. 硬件测量
           self.measure(samples, env)
           # 5. 训练代价模型
           self.cost_model.train(env.perf_buffer)
   ```

2. **UpdatePopulation** (tuner.py:92-110): 种群更新策略
   - 约束随机采样
   - 历史top-k程序
   - 预测值重新评估

3. **采样方法**:
   - `constrained_random_sample` (tuner.py:200-209): 约束随机采样
   - `random_walk` (tuner.py:186-198): 随机游走
   - `history_topk_samples` (tuner.py:239-248): 历史最优

4. **选择策略**:
   - `epsilon_select` (tuner.py:288-298): ε-贪心选择
   - `RouletteWheelSelection` (tuner.py:146-153): 轮盘赌选择

#### 3.4.2 遗传算法调优器 (`tuner/ga_tuner.py`)

**文件位置**: `/root/Heron/tuner/ga_tuner.py`

**核心实现**: 约束遗传算法(CGA)

**关键特性**:
1. 染色体编码包含所有变量(可调参数+辅助变量)
2. 在CSP上进行交叉和变异,而非具体解
3. 使用CSP求解器生成有效染色体

**核心操作**:

1. **约束交叉** (constraint-based crossover):
   ```python
   # 从代价模型提取关键变量
   key_vars = extract_from_cost_model(model)

   # 为每个关键变量添加约束
   for v in key_vars:
       constraints.add(IN(v, [parent1.v, parent2.v]))

   # 生成新CSP
   new_CSP = CSP_initial + constraints
   ```

2. **约束变异** (constraint-based mutation):
   ```python
   # 随机移除一个交叉生成的约束
   constraints.remove_random_one()
   ```

3. **CSP求解**:
   - 使用OR-Tools求解器
   - 生成满足所有约束的具体参数值

### 3.5 代价模型 (`model/xgboost_cost_model.py`)

**文件位置**: `/root/Heron/model/xgboost_cost_model.py`

**核心类**: `XGBoostCostModel` (xgboost_cost_model.py:14-104)

**功能**: 快速预测程序性能,避免昂贵的硬件测量

**训练过程** (xgboost_cost_model.py:66-96):
```python
def train(self, perf_buffer):
    """训练XGBoost模型"""
    x_train = np.array(perf_buffer.data_x)  # 特征
    y_train = np.array(perf_buffer.data_y)  # 性能

    # 归一化
    y_train = y_train / (max(y_train) + 1e-8)

    # XGBoost训练
    self.bst = xgb.train(
        self.xgb_params,
        dtrain,
        num_boost_round=8000,
        callbacks=[early_stopping]
    )
```

**预测** (xgboost_cost_model.py:99-104):
```python
def predict(self, samples):
    """预测样本性能"""
    feas = np.array([s.point for s in samples])  # 参数点
    dtest = xgb.DMatrix(feas)
    return self.bst.predict(dtest)
```

**特征提取**:
- 使用约束生成过程中定义的变量作为特征
- 包括循环长度、tile大小、内存使用等
- 特征可无需编译直接获取

### 3.6 工具函数 (`utils.py`)

**文件位置**: `/root/Heron/utils.py`

**核心功能**:

1. **键生成** (utils.py:21-39):
   ```python
   def genKey(type_, stage_name, ax_name, param_name, var_name, others):
       """生成唯一键标识循环/参数/变量"""
       # L: 循环长度, P: 可调参数, V: 变量, O: 其他
   ```

2. **图遍历** (utils.py:89-112):
   - `getStageNamesOrdered`: 拓扑排序获取stage顺序
   - `getConsumers/getProducers`: 获取数据流依赖

3. **TensorCore工具** (utils.py:202-275):
   - `genTCShape`: 生成TensorCore计算shape
   - `genTCLoadAParams/genTCLoadBParams`: 生成加载参数
   - `genTCStoreParams`: 生成存储参数

4. **代价模型分析** (utils.py:319-332):
   ```python
   def anaCostModel(model, key_list):
       """分析代价模型,提取重要特征"""
       score_map = model.bst.get_score(importance_type='weight')
       # 按重要性排序
       score_list_sorted = sorted(score_list, key=lambda x: -x[1])
       selected_keys = [key_list[x] for x in selected_idxs]
       return selected_keys
   ```

### 3.7 样本管理 (`sample.py`)

**文件位置**: `/root/Heron/sample.py`

**核心类**: `Sample` (sample.py:8-35)

**属性**:
```python
class Sample:
    def __init__(self, task):
        self.valid = False  # 是否满足约束
        self.perf = 0.0  # 实测性能
        self.predict = 0  # 预测性能
        self.point = []  # 参数点
        self.knob_manager = copy.deepcopy(task.knob_manager)
```

**方法**:
- `fromCode`: 从编码解码参数
- `lower`: 生成底层IR

### 3.8 并行化支持 (`multi.py`)

**文件位置**: `/root/Heron/multi.py`

**核心类**: `Job` (multi.py:10-33)

**功能**:
- 支持多进程并行采样
- 超时控制
- 子进程管理

**使用示例**:
```python
class Job:
    def __init__(self, func, attach_info, timeout=10):
        self.func = func  # 要执行的函数
        self.timeout = timeout

    def start(self, inputs):
        """启动子进程"""
        self.process = multi.Process(target=exec, args=(self.func, self.queue, inputs))
        self.process.start()

    def get(self):
        """获取结果,超时则终止"""
        res = self.queue.get(timeout=self.timeout)
        return res
```

---

## 4. 约束生成机制

### 4.1 调度生成规则 (Schedule Generation Rules)

根据论文Table 6,Heron定义了3类调度生成规则:

**S1: Tensorize**
- 条件: `Tensorizable(S, i)` - 计算可张量化
- 操作: 使用硬件intrinsic加速计算
- 示例: TensorCore wmma, DL Boost vnni

**S2: Add Multi-Level SPM**
- 条件: `HasDataReuse(S,i) & HasMultiLevelCache(S,i)`
- 操作: 添加多层片上存储(shared memory, registers)
- 应用: TensorCore的shared memory → wmma fragments

**S3: Add Multi-Scope SPM**
- 条件: `HasDataReuse(S) & HasMultiScopeCache(S,i)`
- 操作: 为不同类型数据添加缓存
- 应用: 区分输入输出缓存

### 4.2 约束生成规则 (Constraint Generation Rules)

根据论文Table 8,定义了6类约束生成规则:

**C1: AddLoopSplit** (primitives.py中的split操作)
- 触发: `HasLoopSplit(P)` - 存在split操作
- 生成: `PROD(l, [lo, li])` - 循环长度乘积约束
- 示例: `l = lo × li`

**C2: AddLoopFuse**
- 触发: `HasLoopsFused(P)` - 存在fuse操作
- 生成: `PROD(l, [l1, l2])` - 融合循环长度约束

**C3: AddCandidates**
- 触发: `HasCandidates(P)` - 存在候选值
- 生成: `IN(v, [c1, ..., cn])` - 离散约束
- 示例: TensorCore的 `m ∈ {8, 16, 32}`

**C4: AddStageFuse**
- 触发: `HasStagesFused(P)` - stage被融合
- 生成: `SELECT(v, loc, [v1, ..., vk])` - 位置依赖约束
- 功能: 处理compute_at带来的依赖

**C5: AddMemLimit**
- 触发: `HasSPM(P)` - 使用片上存储
- 生成:
  - `PROD(mem_i, [l1, ..., ln])` - 每个张量内存
  - `SUM(total, [mem1, ..., memt])` - 总内存
  - `LE(total, limit)` - 容量限制
- 示例: TensorCore shared memory ≤ 48KB

**C6: AddDLASpecific**
- 触发: `HasSpecialArchConstraints(P)`
- 生成: 硬件特定约束
- 示例:
  - TensorCore: `vector_length ∈ {1,2,4,8}`
  - VTA: 访问周期约束

### 4.3 约束类型

根据论文Table 7,支持6种约束类型:

```python
# T1: 乘积约束
PROD(v, [v1, ..., vn])  # v = v1 × ... × vn

# T2: 求和约束
SUM(v, [v1, ..., vn])   # v = v1 + ... + vn

# T3: 相等约束
EQ(v1, v2)              # v1 = v2

# T4: 小于等于约束
LE(v1, v2)              # v1 ≤ v2

# T5: 离散值约束
IN(v, [c1, ..., cn])    # v ∈ {c1, ..., cn}

# T6: 选择约束
SELECT(v, u, [v1, ..., vn])  # v = v_u
```

---

## 5. 关键算法流程

### 5.1 约束空间生成 (Algorithm 1)

根据论文Section 4,空间生成分两步:

**Step 1: Schedule Template Generation** (Algorithm 1: lines 4-16)

```
Input: Program P, DAG
Output: CSP_initial

1. nodes ← post_order_traverse(DAG)
2. template ← []; nodes_scheduled ← ∅
3. while nodes ≠ ∅:
4.   node ← nodes.pop()
5.   for rule in predefined_schedule_rules:
6.     if rule.condition(P, node):
7.       P, DAG, primitives, next ← rule.apply(P, node)
8.       template.append(primitives)
9.       if next: break
10.  nodes_scheduled ← nodes_scheduled ∪ {node}
11.  nodes ← post_order_traverse(DAG) \ nodes_scheduled
```

**Step 2: Constraints Generation** (Algorithm 1: lines 17-23)

```
12. CSP_initial ← ∅
13. for primitives in template:
14.   for rule in predefined_constraint_rules:
15.     if rule.condition(schedule):
16.       constraints ← rule.apply(primitives)
17.       CSP_initial ← CSP_initial ∪ constraints
18. return CSP_initial
```

**实际示例** (论文Figure 4 - GEMM on TensorCore):

输入: `C[i,j] += A[i,r] * B[r,j]`, shape (64,64,64)

生成的变量和约束:
- 173个变量 (10个架构约束变量 + 82个循环长度 + 30个可调参数 + 51个辅助)
- 372个约束

关键约束:
```
# 架构约束 (C6)
m × n × k = 4096
m, n, k ∈ {8, 16, 32}
shared_mem ≤ 48K
vector_length ∈ {1, 2, 4, 8}

# 循环约束 (C1)
i = i_o × i6 (split)
i_o = i_oo × i5 (split)
i6 = m (tensorize)

# 内存约束 (C5)
mem_c_shared = PROD([i7, j7])
mem_c_shared ≤ 48K
```

### 5.2 约束空间探索 (Algorithm 2)

**主循环** (Algorithm 2: lines 3-17):

```
Input: CSP_initial, Trials, Generations
Output: Optimized program

1. D ← ∅; Pop ← ∅
2. for i ≤ Trials:
3.   # Step-1: 形成第一代种群
4.   Pop_random ← RandSAT(CSP_initial, solver)
5.   Pop ← Pop + Pop_random
6.
7.   # Step-2: 演化多代
8.   for j in [0, Generations-1]:
9.     Pop ← Select_rw(Pop, model)  # 轮盘赌选择
10.    CSPs ← constraint-based crossover and mutation
11.    Pop ← RandSAT(CSPs, solver) ∪ Pop
12.
13.  # Step-3: 选择测量
14.  programs ← Select_ε-greedy(Pop, model)
15.  perfs ← Measure(programs)
16.
17.  # Step-4: 更新模型
18.  D ← D + {perfs}; Update(model, D)
19.
20. return program with highest performance
```

### 5.3 约束交叉和变异 (Algorithm 3)

**核心创新**: 在CSP上操作,而非具体解

```
Input: Pop, model, N, CSP_initial
Output: CSPs

1. CSPs ← []
2. for i in [0, N-1]:
3.   constraints ← ∅
4.
5.   # Step-1: 关键变量提取
6.   V ← extract key variables from model by feature importance
7.
8.   # Step-2: 约束交叉
9.   c1, c2 ← two random chromosomes from Pop
10.  for v ∈ V:
11.    constraints ← constraints + {IN(cv, [c1.v, c2.v])}
12.
13.  # Step-3: 约束变异
14.  constraints ← remove one constraint from constraints randomly
15.
16.  CSP ← CSP_initial + constraints
17.  CSPs.append(CSP)
18.
19. return CSPs
```

**优势** (论文Section 5.1):
1. **保证有效性**: 新CSP生成的解必然满足CSP_initial
2. **保留优秀基因**: 交叉约束确保关键变量值来自父代
3. **增加多样性**: 变异移除约束,扩大搜索范围

**实例** (论文Figure 5):
```
初始CSP:
  Objective: 0.4x + 0.6y + 0.01z
  Constraints: x×y ≤ 8, x∈{1..5}, y∈{1..5}, z∈{0,1}

关键变量: x, y (从模型提取)

父代: c1=[1,4,z], c2=[2,3,z]

交叉生成约束:
  x ∈ {1, 2}  (来自c1.x和c2.x)
  y ∈ {3, 4}  (来自c1.y和c2.y)

变异(移除y约束):
  最终CSP = 初始CSP + {x ∈ {1,2}}

可能的解: [1,1..5,*], [2,1..5,*] (受x×y≤8约束)
能找到最优解: [2,4,*] → 0.4×2 + 0.6×4 = 3.2
```

---

## 6. 代码结构与关键文件

### 6.1 目录结构

```
Heron/
├── config.py              # 配置管理
├── environment.py         # 环境和任务管理
├── utils.py               # 工具函数
├── sample.py              # 样本表示
├── multi.py               # 并行化支持
├── schedule/              # 调度相关
│   ├── primitives.py      # 调度原语
│   ├── constraints/       # 约束生成
│   ├── context/           # 执行上下文
│   └── sched_op/          # 操作符调度
├── tuner/                 # 调优器
│   ├── tuner.py           # 基础调优器
│   ├── ga_tuner.py        # 遗传算法(CGA)
│   ├── sa_tuner.py        # 模拟退火
│   ├── random_tuner.py    # 随机搜索
│   └── idea_tuner.py      # IDEA算法
├── model/                 # 代价模型
│   └── xgboost_cost_model.py  # XGBoost模型
├── runner/                # 代码执行
├── perf/                  # 性能缓冲
├── task/                  # 任务定义
├── ops/                   # 算子库
│   ├── cuda/              # CUDA算子(TensorCore)
│   ├── x86/               # x86算子(DL Boost)
│   └── vta/               # VTA算子
├── pretrain/              # 预训练模型
└── tests/                 # 测试用例
```

### 6.2 核心文件说明

#### 6.2.1 schedule/primitives.py
**关键函数** (共240行):
- `split` (L5-37): 循环分割,更新约束图
- `fuse` (L53-76): 循环融合
- `bind` (L78-87): 线程绑定
- `tensorize` (L175-198): 张量化
- `cache_read/write` (L133-149): 缓存操作
- `compute_at` (L151-156): 计算位置

#### 6.2.2 tuner/tuner.py
**核心方法** (共386行):
- `run` (L47-87): 主优化循环
- `UpdatePopulation` (L92-110): 种群更新
- `constrained_random_sample` (L200-209): 约束随机采样
- `epsilon_select` (L288-298): ε-贪心选择
- `RouletteWheelSelection` (L146-153): 轮盘赌选择
- `random_walk` (L186-198): 随机游走

#### 6.2.3 tuner/ga_tuner.py
**实现细节**:
- 约束交叉: 提取关键变量,添加IN约束
- 约束变异: 随机移除一个约束
- CSP求解: 使用OR-Tools生成具体解

#### 6.2.4 model/xgboost_cost_model.py
**关键组件** (共290行):
- `XGBoostCostModel` (L14-104): 主模型类
- `fit` (L71-96): 训练过程
- `predict` (L99-104): 性能预测
- 自定义回调 (L106-214): early stopping
- 评估指标 (L218-289): recall@N, cover@N等

#### 6.2.5 utils.py
**工具函数** (共341行):
- `genKey` (L21-39): 生成唯一标识
- `getStageNamesOrdered` (L89-112): 拓扑排序
- `findStmtBufferSizes` (L114-125): 内存分析
- `genTC*Params` (L202-275): TensorCore参数生成
- `anaCostModel` (L319-332): 模型分析

### 6.3 算子实现

#### 6.3.1 CUDA算子 (ops/cuda/)
支持的操作:
- `dense.py`: 矩阵乘法(GEMM)
- `conv1d/2d/3d.py`: 1D/2D/3D卷积
- `batch_matmul.py`: 批量矩阵乘
- `conv2d_transposed.py`: 转置卷积
- `dil.py`: 膨胀卷积
- `gemv.py`: 矩阵向量乘
- `scan.py`: 扫描操作
- `var/mean.py`: 统计操作

#### 6.3.2 x86算子 (ops/x86/)
DL Boost特定:
- `dense.py`: 使用vnni指令的矩阵乘
- `conv*_int8.py`: INT8量化卷积
- `t2d.py`: 2D转换
- `util.py`: 工具函数

#### 6.3.3 VTA算子 (ops/vta/)
- `dense.py`: VTA矩阵乘
- `conv2d.py`: VTA 2D卷积

---

## 7. 工作流程详解

### 7.1 完整优化流程

```
1. 配置阶段
   ├── 读取config.json
   ├── 创建Environment
   └── 设置硬件平台(TensorCore/DL Boost/VTA)

2. 任务创建
   ├── 定义算子(opfunc)和参数(args)
   ├── 创建Task对象
   └── 选择优化方法(CGA/GA/SA)

3. 空间生成阶段
   ├── 生成naive program和DAG
   ├── 应用schedule生成规则(S1-S3)
   │   ├── Tensorize (S1)
   │   ├── Add Multi-Level SPM (S2)
   │   └── Add Multi-Scope SPM (S3)
   ├── 生成schedule template
   └── 应用约束生成规则(C1-C6)
       ├── AddLoopSplit (C1)
       ├── AddLoopFuse (C2)
       ├── AddCandidates (C3)
       ├── AddStageFuse (C4)
       ├── AddMemLimit (C5)
       └── AddDLASpecific (C6)
   → 输出: CSP_initial (约束搜索空间)

4. 空间探索阶段
   ├── 初始化种群
   │   ├── 约束随机采样
   │   └── 历史top-k
   │
   ├── 迭代优化 (for each round)
   │   ├── 约束交叉
   │   │   ├── 提取关键变量(从cost model)
   │   │   ├── 选择两个父代
   │   │   └── 生成IN约束
   │   ├── 约束变异
   │   │   └── 随机移除一个约束
   │   ├── CSP求解
   │   │   └── 使用OR-Tools生成有效程序
   │   ├── 性能预测(cost model)
   │   └── ε-greedy选择
   │
   ├── 硬件测量
   │   ├── 编译程序
   │   ├── 在硬件上运行
   │   └── 记录性能
   │
   └── 模型更新
       └── 用新测量数据训练XGBoost

5. 输出阶段
   └── 返回性能最优的程序
```

### 7.2 GEMM on TensorCore示例

**输入**:
```python
# 计算定义
C[i, j] += A[i, r] * B[r, j]
# 形状: M=64, N=64, K=64
```

**空间生成**:
```python
# Step 1: Schedule生成
input → C  # DAG顺序

# 应用规则S3: Add Multi-Scope SPM
C.wmma = cache_write(C, "wmma.accumulator")

# 应用规则S2: Add Multi-Level SPM
C.shared = cache_read(C.wmma, C, "shared")
A.shared = cache_read(A, C, "shared")
A.wmma = cache_read(A.shared, C, "wmma.matrix_a")
B.shared = cache_read(B, C, "shared")
B.wmma = cache_read(B.shared, C, "wmma.matrix_b")

# MultiLevelTiling
i_o, i6 = split(i, factor=m)
i_oo, i5 = split(i_o, factor=tile.i5)
...

# 应用规则S1: Tensorize
tensorize(i6, intrin_wmma_gemm(m, n, k))

# Step 2: 约束生成
# C1: split约束
PROD(i, [i_o, i6])
PROD(i_o, [i_oo, i5])
EQ(i6, m)
EQ(i5, tile.i5)

# C3: 架构约束
IN(m, [8, 16, 32])
IN(n, [8, 16, 32])
EQ(k, 16)
PROD(4096, [m, n, k])

# C5: 内存约束
PROD(mem_c_shared, [i7, j7])
LE(mem_c_shared, 48K)

# C6: TensorCore特定
IN(vector_length, [1, 2, 4, 8])
```

**探索过程**:
```python
# 初始种群
Pop = constrained_random_sample(200)  # 200个有效程序

# 第1轮
for gen in range(5):  # 5代演化
    # 提取关键变量: [m, n, tile.i5, tile.j5, ...]
    key_vars = cost_model.get_important_features()

    # 选择父代
    parents = RouletteWheelSelection(Pop, 100)

    # 交叉
    for p1, p2 in zip(parents[::2], parents[1::2]):
        CSP_new = CSP_initial.copy()
        for v in key_vars:
            CSP_new.add(IN(v, [p1.v, p2.v]))
        # 变异
        CSP_new.remove_random_constraint()
        # 求解
        offspring = OR_Tools_Solve(CSP_new)
        Pop.append(offspring)

# ε-greedy选择
top_k = sort(Pop, by=predict)[:k]  # 前k个
random_rest = RouletteWheelSelection(Pop[k:], select_num-k)
selected = top_k + random_rest

# 硬件测量
perfs = Measure(selected, V100)

# 更新模型
cost_model.train(perfs)
```

**最终输出**:
- 最优m=16, n=16, k=16
- 性能: 2.69×优于PyTorch

### 7.3 配置文件示例

**tests/quick_start/tensorcore/quick_start.json**:
```json
{
  "config": {
    "out_name": "out",
    "method": "CGA",
    "max_trials": 100,
    "runner_number": 4,
    "runner_repeat": 10,
    "runner_timeout": 10,
    "build_timeout": 10,
    "in_dtype": "float16",
    "out_dtype": "float32",
    "cases": [
      {"M": 64, "K": 64, "N": 64}
    ]
  }
}
```

**运行命令**:
```bash
cd tests/quick_start/tensorcore
python run.py -p tensorcore -c quick_start.json
```

---

## 8. 关键技术点

### 8.1 约束表示与求解

**约束满足问题(CSP)**:
```
CSP = (Variables, Domains, Constraints)

Variables = {
    # 可调参数
    tile.i5, tile.j5, tile.r5, ...

    # 辅助变量
    i, i_o, i_oo, i5, i6, ...

    # 架构变量
    m, n, k, wmma_m, wmma_n, wmma_k, ...
}

Domains = {
    tile.i5: [1..64],
    m: {8, 16, 32},
    ...
}

Constraints = {
    # 循环约束
    i = i_o × i6,
    i_o = i_oo × i5,

    # 架构约束
    m × n × k = 4096,
    m ∈ {8, 16, 32},

    # 内存约束
    mem ≤ 48K,
    ...
}
```

**求解器**: 使用Google OR-Tools
- 支持整数约束
- 支持离散域
- 支持逻辑约束

### 8.2 代价模型特征

**特征来源**: 从约束生成中定义的变量

**特征类别**:
1. **循环长度**: `stage.axis.length`
2. **Tile大小**: `tile.i5`, `tile.j5`等
3. **内存使用**: `mem_shared_A`, `mem_shared_B`
4. **向量长度**: `vector_length`
5. **计算位置**: `compute_at_location`
6. **并行度**: `blockIdx.x`, `threadIdx.x`范围

**优势**:
- 无需编译即可提取
- 直接反映架构约束
- 与性能高度相关

### 8.3 搜索空间质量

**对比AutoTVM** (论文Figure 11):

AutoTVM搜索空间:
- 简单约束: 矩形边界
- 大量无效程序
- 错失最优解

Heron搜索空间:
- 复杂约束: 不规则边界
- 几乎全部有效
- 包含最优解

**数据** (GEMM G1):
- AutoTVM: 95%程序无效
- Heron: <5%程序无效
- 平均性能提升: 1.55×

### 8.4 探索效率

**CGA vs 传统GA** (论文Figure 12, 13):

问题: 不规则约束空间

传统GA:
- 交叉/变异后常生成无效解
- 频繁重启,效率低
- 难以保留优秀基因

CGA:
- 在CSP上操作,保证有效性
- 关键变量约束保留优秀基因
- 变异扩大搜索范围

性能: CGA比GA快2-3×

---

## 9. 支持的算子与网络

### 9.1 算子支持

**矩阵运算**:
- GEMM (General Matrix Multiply)
- GEMV (General Matrix-Vector Multiply)
- BMM (Batch Matrix Multiply)

**卷积**:
- C1D (1D Convolution)
- C2D (2D Convolution)
- C3D (3D Convolution)
- T2D (Transposed 2D Convolution)
- DIL (Dilated Convolution)

**其他**:
- SCAN (Prefix Scan)
- VAR (Variance)
- MEAN (Mean)
- DEP (Depthwise)

### 9.2 网络支持

**图像分类**:
- ResNet-50
- VGG-16
- Inception-V3

**自然语言处理**:
- BERT (batch_size=16)

### 9.3 性能数据

**TensorCore (V100)**:
- vs AutoTVM: 1.55×
- vs Ansor: 2.85×
- vs AMOS: 1.52×
- vs PyTorch: 2.69×

**DL Boost (Xeon)**:
- vs AutoTVM: 2.93×
- vs Ansor: 12.0×
- vs AMOS: 2.71×
- vs oneDNN: 1.49×

**VTA**:
- vs AutoTVM: 2.32×

---

## 10. 扩展与定制

### 10.1 添加新硬件支持

**步骤**:

1. **定义架构约束**:
   ```python
   # 在ops/new_hw/下创建算子
   # 定义硬件intrinsic
   ```

2. **修改调度生成规则**:
   ```python
   # Rule-S1: 适配硬件intrinsic
   # Rule-S2/S3: 适配存储层次
   ```

3. **添加约束生成规则**:
   ```python
   # Rule-C6: 添加硬件特定约束
   # 例如: 计算单元大小,内存容量,访问模式
   ```

**示例: TensorCore适配** (论文Section 4):
```python
# S1: Tensorize规则
if pattern_match(compute, "C[i,j] += A[i,r] * B[r,j]"):
    map [i,j,r] to [m,n,k]
    tensorize(i6, intrin_wmma_gemm(m, n, k))

# C6: TensorCore约束
EQ(PROD([m, n, k]), 4096)
IN(m, {8, 16, 32})
IN(n, {8, 16, 32})
IN(k, {16})
LE(shared_mem, 48*1024)
IN(vector_length, {1, 2, 4, 8})
```

### 10.2 添加新算子

**步骤**:

1. **定义计算**:
   ```python
   def my_op(A, B, ...):
       # 使用TVM te定义计算
       C = te.compute(shape, lambda i, j: A[i] * B[j])
       return C
   ```

2. **定义调度方法**:
   ```python
   def schedule_my_op(cfg, s, C):
       # 应用调度原语
       # 生成约束
   ```

3. **注册**:
   ```python
   # 在ops/对应平台/下添加
   # 在sched_op中注册
   ```

### 10.3 定制优化策略

**修改遗传算法参数** (config.py):
```python
config.pop_num = 200  # 种群大小
config.select_num = 100  # 选择数量
config.crossover_key_ratio = 0.2  # 关键变量比例
config.iter_walks = 5  # 演化代数
```

**选择其他优化方法**:
```python
config.opt_method = 'SA'  # 模拟退火
config.opt_method = 'CRAND'  # 约束随机搜索
config.opt_method = 'GA'  # 标准遗传算法
```

**自定义选择策略**:
```python
# 在tuner.py中添加
def my_selection(self, env, all_samples, select_num):
    # 自定义选择逻辑
    return selected
```

---

## 11. 实验复现指南

### 11.1 环境准备

**依赖**:
```bash
# Python环境
Python 3.6.10

# 安装依赖
pip install -r requirements.txt

# 内容包括:
# - xgboost (代价模型)
# - ortools (CSP求解)
# - torch (辅助计算)
# - numpy, psutil等
```

**硬件要求**:

TensorCore:
- GPU: V100/T4/A100
- CUDA: 11.2
- TVM: GPU版本

DL Boost:
- CPU: Xeon Gold 6240或类似
- LLVM: 8.0.0
- TVM: CPU版本

### 11.2 快速开始

**TensorCore GEMM**:
```bash
cd tests/quick_start/tensorcore
python run.py -p tensorcore -c quick_start.json

# 预期输出:
# PASS
# Case [64, 64, 64], latency 0.002236 ms.
```

**DL Boost GEMM**:
```bash
cd tests/quick_start/dlboost
python run.py -p dlboost -c quick_start.json

# 预期输出:
# PASS
# Case [64, 64, 64], latency 0.002658 ms.
```

### 11.3 完整评估

**Figure 6 - TensorCore算子**:
```bash
cd tests/Figure6
python run.py -c tensorcore_ops.json
```

**Figure 7 - 其他GPU**:
```bash
# T4
cd tests/Figure7_T4
python run.py -c config.json

# A100
cd tests/Figure7_A100
python run.py -c config.json
```

**Figure 8 - DL Boost**:
```bash
cd tests/Figure8
python run.py -c dlboost_ops.json
```

**Figure 10 - 网络**:
```bash
cd tests/Figure10
python run.py -c networks.json
```

### 11.4 结果分析

**输出文件**:
```
out/task_name/
├── schedule.py        # 生成的调度代码
├── stat.txt           # 统计信息(每轮时间,性能)
└── logs/              # 详细日志
```

**stat.txt格式**:
```
iter_no, sample_time, measure_time, train_time, cur_time, best_perf
0, 0.5, 2.3, 0.1, 2.9, 1.234
1, 0.4, 2.1, 0.1, 5.5, 1.456
...
```

**性能指标**:
- TFLOPS (浮点运算性能)
- Latency (延迟,ms)
- Speedup (相对加速比)

---

## 12. 论文核心贡献总结

### 12.1 技术贡献

1. **自动约束生成框架** (Section 4):
   - 无需手工编写约束
   - 自动从调度原语推导
   - 支持6种约束类型

2. **约束遗传算法CGA** (Section 5):
   - 首个在CSP上演化的GA
   - 严格保持约束有效性
   - 通过关键变量保留优秀基因

3. **高质量搜索空间** (Section 7.3):
   - 精确约束消除95%无效程序
   - 包含最优或近优解
   - 相比手工约束,质量大幅提升

### 12.2 实验验证

**3个代表性DLA**:
- NVIDIA TensorCore (商业GPU)
- Intel DL Boost (商业CPU)
- TVM VTA (开源FPGA)

**大规模测试**:
- 9种算子 × 6-10种配置
- 4个神经网络
- 对比4种自动方法 + 4种手工库

**性能提升**:
- 自动方法: 1.52× - 12.0×
- 手工库: 1.02× - 8.89×

**编译开销**:
- 不引入额外开销
- CSP求解仅占23%时间
- 76%时间用于硬件测量

### 12.3 创新点

**vs AutoTVM/Ansor** (论文Section 2):
- 手工模板 → 自动约束生成
- 简单空间 → 复杂约束空间
- 通用架构 → DLA特化

**vs AMOS** (论文Section 7):
- 有限映射 → 完整约束空间
- 简单约束 → 精确架构建模
- 部分优化 → 全面优化

**vs Polyhedral (AKG)** (论文Section 7):
- 静态变换 → 动态探索
- 受限表达 → 灵活约束
- 固定策略 → 自适应搜索

---

## 13. 局限性与未来工作

### 13.1 当前局限

1. **约束求解开销**:
   - 复杂CSP求解时间长
   - 可能限制搜索空间大小

2. **代价模型精度**:
   - 依赖历史数据
   - 冷启动性能较差

3. **硬件覆盖**:
   - 需要为新硬件编写规则
   - 规则设计需要专业知识

### 13.2 潜在改进

1. **更高效的CSP求解**:
   - 使用启发式求解器
   - 增量求解技术
   - 近似满足

2. **迁移学习**:
   - 跨算子迁移
   - 跨硬件迁移
   - 元学习代价模型

3. **自动规则发现**:
   - 从硬件规格自动提取约束
   - 机器学习推断规则
   - 减少人工介入

### 13.3 研究方向

1. **动态DLA支持**:
   - 可重构架构
   - 动态张量形状
   - 稀疏计算

2. **端到端优化**:
   - 与图优化联合
   - 考虑数据布局
   - 算子融合

3. **多目标优化**:
   - 性能-能耗权衡
   - 延迟-吞吐量平衡
   - 精度-效率折中

---

## 14. 参考资源

### 14.1 论文信息

**标题**: Heron: Automatically Constrained High-Performance Library Generation for Deep Learning Accelerators

**会议**: ASPLOS 2023

**作者**: Jun Bi, Qi Guo, et al.

**机构**:
- 中科院计算所
- 中国科学技术大学
- 寒武纪科技

**DOI**: 10.1145/3582016.3582061

### 14.2 相关论文

**深度学习编译器**:
- TVM (OSDI'18)
- AutoTVM (NeurIPS'18)
- Ansor (OSDI'20)
- AMOS (ISCA'22)

**深度学习加速器**:
- DianNao (ASPLOS'14)
- TPU (ISCA'17)
- TensorCore (IEEE Micro'20)

**约束优化**:
- Genetic Algorithms for CSP (EC'94)
- Stochastic Ranking (EC'00)
- Multi-objective Optimization (Nature'08)

### 14.3 代码资源

**项目地址**: /root/Heron

**关键文件**:
- config.py: 配置系统
- environment.py: 环境管理
- tuner/ga_tuner.py: CGA实现
- model/xgboost_cost_model.py: 代价模型
- schedule/primitives.py: 调度原语

**测试用例**:
- tests/quick_start/: 快速开始
- tests/Figure*/: 论文实验复现

### 14.4 依赖工具

**TVM**: https://tvm.apache.org/
- 深度学习编译框架
- 提供IR和调度系统

**XGBoost**: https://xgboost.readthedocs.io/
- 梯度提升树
- 用于代价模型

**OR-Tools**: https://developers.google.com/optimization
- Google约束求解器
- CSP求解

---

## 15. 总结

Heron是一个创新的深度学习加速器程序自动生成系统,通过以下核心技术解决了DLA程序优化的关键挑战:

1. **自动约束生成**: 从调度操作自动推导精确约束,无需手工编写
2. **约束遗传算法**: 直接在CSP上演化,严格保持约束有效性
3. **高质量搜索空间**: 精确约束消除大量无效程序,包含最优解

实验表明,Heron在多个DLA平台上显著优于现有方法:
- 相比自动生成方法: 平均2.71×加速
- 相比手工优化库: 平均2.00×加速
- 不引入额外编译开销

本项目为DLA程序自动优化提供了新的思路,展示了约束编程与演化算法结合的强大潜力。通过理解Heron的设计思想和实现细节,可以为类似系统的开发提供重要参考。

---

**文档生成时间**: 2025-10-03
**项目版本**: ASPLOS 2023
**文档作者**: Based on Heron source code and paper analysis
