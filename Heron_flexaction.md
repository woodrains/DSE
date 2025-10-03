融合方案：
环境定义：Heron精心生成的CSP_initial将成为FlexAction中强化学习智能体的**“环境（Environment）”**。这个环境的特殊之处在于，它是一个“安全”的环境，任何状态（即一组参数配置）都天然满足硬件约束。
状态与动作：
状态（State）：环境中的一个具体状态就是CSP_initial中的一个合法解（一组完整的参数赋值）。
原子动作（Atomic Actions）：定义为能够修改当前状态（参数配置）的最小操作，例如“将循环i的tile_factor从16改为32”。至关重要的是，执行任何动作后的新状态都必须仍然满足CSP_initial的所有约束。
FlexAction的角色：FlexAction的RL智能体将在这个CSP环境中进行学习和探索。它的目标是学习一个策略（Policy），该策略能够指导其选择一系列动作（或学习到的宏操作），从而将当前状态（一个有效的程序配置）转变为一个性能更高的新状态（另一个有效的程序配置）。
优势与潜力：
解决定向搜索问题：这是最大的优势。FlexAction用目标驱动的强化学习替代了CGA的随机采样。RL智能体会根据历史经验（通过奖励信号）学习哪些操作序列（Lambda Items）能够持续带来性能提升，从而实现从“随机试探”到“策略寻优”的质变，这直接解决了CGA缺乏“定向引导”的核心问题。
提升复用性与自动化：FlexAction的设计初衷就是为了实现搜索策略的复用 。在Heron的框架下，FlexAction可以学习到针对某一类DLA体系结构（例如，具有两级SPM和Tensor-like单元的架构）通用的、高效的优化模式（例如，“对计算密集型循环进行多级分块，并同步进行数据搬移”），并将其固化为可复用的Lambda项。当面对一个新的、但架构类似的DLA时，这些学习到的宏操作可以被直接复用或微调，显著减少了Heron中需要手动修改规则的负担。
结构化与分层探索：Heron + FlexAction的组合实现了优势互补：
Heron定义“What”：它定义了什么是合法的搜索空间，为探索提供了一个高质量的起点。
FlexAction定义“How”：它学习如何高效地探索这个空间。通过Lambda演算的分层结构 ，智能体不仅能选择原子操作进行精细调优，还能在适当时机调用强大的宏操作实现大幅跨越，使得探索兼具广度与深度。
挑战与技术难点：
约束满足的挑战：这是集成过程中最大的技术难点。RL智能体在选择一个动作时，如何确保不会违反CSP_initial中的复杂约束（如PROD、SUM、LE等 ）？
主动方案（更优）：在每个状态下，动态地为智能体生成一个合法的动作空间。即，环境只向智能体提供那些执行后不会破坏CSP约束的动作选项。这需要一个能快速计算合法动作集的机制，对CSP求解器的能力提出了更高要求，但能极大地加速RL的收敛速度。
复杂性与训练成本：引入一个完整的RL框架（包含Lambda项的生成、消除、降低等管理机制 ）无疑会增加整个系统的复杂性。同时，RL在初期需要大量的探索来积累经验（冷启动问题），其初始效率可能低于CGA这种基于种群的并行搜索方法


一个**可替换 CGA、直接在 Heron 的 CSP 空间上运行的 FlexAction 算法**。它面向“**单一架构 × 单一 workload**”的场景（例如：V100/TensorCore 上的 GEMM-64×64×64），并且把 FlexAction 的 **Lambda-Items / Lowering / Generation / Eliminating / Lambda-aware RL** 完整嵌入 Heron 的两阶段流程中：**第 1 阶段仍由 Heron 生成 `CSP_initial`**（你截图里的 Algorithm 1），**第 2 阶段用本算法替换 CGA**。Heron 现有模块（空间生成器、CSP 求解器、Cost Model、DLA Measurer）全部复用，只替换 `ga_tuner.py` 的“探索器”。（Heron 的两阶段与模块边界见系统/模块图与 `environment.py` 的 `tune()` 调用栈。 ）

---

# Algorithm F — FlexAction-CSP Explorer (替换 CGA 的探索阶段)

```
Input:  P (program), DAG, A (architecture), W (workload), Budget
Output: Best feasible program for <A, W>

# Step 0: 构建环境（Heron 阶段一，保持不变）
1:  CSP_initial ← Constrained Search Space Generation(P, DAG, A)    # 见你图示 Algorithm 1
2:  # 说明：CSP_initial 内含变量/域与约束（PROD/SUM/EQ/LE/IN/SELECT），
3:  #      约束来自调度原语与架构特定规则（如 TensorCore tile/对齐/片上存储容量等）。

# Step 1: Lambda-Item 库初始化（可复用，可热启动）
4:  L ← LoadOrInitLambdaLibrary(A, W)
5:  # L0 种子：多级分块骨架、张量化-对齐宏、SPM（shared/warp/reg）成组约束、compute_at 模式等（见 Lowering 规则）

# Step 2: 形成状态与合法动作集（面向 CSP 的“可操作约束模板”）
6:  S_t ← ExtractState(CSP_initial, history) 
7:  # 特征包含：变量计数、域宽、内存/并行度/向量化候选集合、剩余容量等（与 Heron cost-model 输入一致）
8:  A_legal ← {}
9:  for λ ∈ L:
10:     ΔC(λ) ← Lowering(λ, A, W)           # 把 λ 映射成一簇 CSP 约束模板（见 Algorithm L）
11:     if IsFeasible(CSP_initial ⊕ ΔC(λ)): # 调用 CSP solver 做“合法性筛选”（增量/快速）
12:         A_legal ← A_legal ∪ {λ}

# Step 3: Lambda-aware RL 选动作（支持“增量词表”的动作空间）
13: π ← policy with dynamic vocabulary on L
14: B_t ← π.sample_batch(S_t, A_legal, batch_size=B)   # 批量选择一组 Lambda-Items（FlexAction 的“Batch”思想）
15: ΔC_batch ← ⨁_{λ∈B_t} ΔC(λ)                         # 合并约束模板（若冲突，按优先/回滚策略处理）

# Step 4: 生成候选、快速评估与真机测量
16: CSP' ← CSP_initial ⊕ ΔC_batch
17: Cand ← RandSAT(CSP', N)                            # 仍复用 Heron 的 RandSAT/OR-Tools 产生具体可行解
18: ŷ ← CostModelPredict(Cand)                         # 可与分析型仿真器混用（低成本路径）
19: Sel ← ε-greedy(ŷ, k, ε)                           # 同 Heron：Top-k + 探索
20: perf ← MeasureOnHardware(Sel, A)                   # DLA Measurer

# Step 5: 价值反馈、策略更新与库自进化
21: r_t ← Reward(perf, novelty(Cand), eval_cost)
22: π ← π.update(S_t, B_t, r_t, traces=solver/logs)
23: L ← Generation(L, traces) ∪ Initialization(seed)   # 依据 Trace 生成新 λ（模式匹配/α-应用/重复构型）
24: L ← Eliminating(L, scores=r_t, size/cvg budget)    # 淘汰“低效/冗余/冲突”的 λ
25: if Budget exhausted: goto 28
26: CSP_initial ← CSP'                                  # 保存“更优的可行域塑形”为下轮基线（可选）
27: goto Step 2

# Step 6: 返回最优程序
28: return argmax_{p∈AllMeasured} perf(p)
```

**要点解释**

* **与 Heron 的粘合位**：第 0 步由 Algorithm 1 产出 `CSP_initial`；第 4 步的 **RandSAT/求解器**、ε-greedy 选测与 **DLA Measurer** 都直接复用 Heron 现有实现；因此正确性边界与测量闭环保持稳定。&#x20;
* **FlexAction 的核心优势被“落地”**：
  (i) **批量宏动作（Lambda-Items）+ Lowering → 约束模板**，在**CSP 的“可行域”上做定向塑形**，再解算采样，替换 CGA 的交叉/变异；
  (ii) **Lambda-aware RL** 支持**动作集动态扩容**（新 λ 加入词表不破坏依赖）；
  (iii) **Generation/Eliminating** 形成**可复用宏步**库，迁移到相似架构/任务只需轻调 Lowering。  （FlexAction 方法思想亦见论文初稿摘要/方法与评估：）

---

# Algorithm L — Lambda → CSP Lowering（动作到约束模板的映射）

```
Input: λ (Lambda-Item), A (arch), W (workload)
Output: ΔC(λ)  # 可与 CSP_initial 合并的约束模板集合

Case L1: TensorCoreTile(m,n,k)
  ΔC ← { IN(m,{8,16,32}), IN(n,{8,16,32}), IN(k,{16}),
         EQ(PROD([m,n,k]), 4096) }
  # 解释：对齐张量核心指令形状与 tile 乘积恒等约束
  # 依据：C6(TensorCore) 规则与实例。 

Case L2: MultiLevelSPM({A.shared, B.shared, C.shared}, caps)
  ΔC ← { PROD(mem_t, [tile-factors...])  for each tensor t
         SUM(mem_total, [mem_A,mem_B,mem_C]),
         LE(mem_total, cap_shared) }
  # 解释：以乘积/求和/容量上界表达片上存储使用

Case L3: SplitFuseSkeleton(axes, factors)
  ΔC ← { PROD(ax, [ax_o,ax_i]), EQ(ax_i, factor) ... }
  # 解释：多级分块/融合骨架，以乘积/相等约束表达层次关系

Case L4: Vectorize(vlen_set)
  ΔC ← { IN(vector_length, vlen_set) }

Case L5: ComputeAt(stage, loc)
  ΔC ← { SELECT(v, loc, [v1,...,vk]) }
  # 解释：位置依赖选择，把 compute_at 带来的绑定关系转成 SELECT 约束
```

* **对应 Heron 的 6 类约束类型**（PROD/SUM/EQ/LE/IN/SELECT），保持与 **C1–C6** 规则一致；**Lowering 后的 ΔC** 与 `CSP_initial` 可并置求解，不破坏有效性。&#x20;
* **TensorCore 具体实例（GEMM-64×64×64）**：`IN(m,{8,16,32}), IN(n,{8,16,32}), IN(k,{16}), EQ(PROD([m,n,k]),4096)`；`LE(shared_mem, 48KB)`；`IN(vector_length,{1,2,4,8})`。这些在 L1/L2/L4 中一次性落下模板。

---

## 具体到“单一架构 × 单一 workload”的一次搜索流程（以 V100/TensorCore × GEMM-64³ 为例）

1. **阶段一（不变）**：Heron 依据调度生成与约束生成规则构造 `CSP_initial`（含 形状/循环/内存/对齐等约束）。此阶段输出的 CSP 是**高质量可行域**，求解器为 OR-Tools。

2. **阶段二（替换 CGA）**：调用 *Algorithm F*：

   * **初始化 `L`**：`TensorCoreTile(·)`、`MultiLevelSPM(·)`、`Vectorize({1,2,4,8})`、`SplitFuseSkeleton(·)`、`ComputeAt(·)`；
   * **构造 `A_legal`**：对每个 λ 执行 *Algorithm L* 得到 ΔC，保留 `CSP_initial ⊕ ΔC` 可满足者；
   * **策略选择 `B` 个 λ** 并合并模板：例如 `{TensorCoreTile(16,16,16), MultiLevelSPM, Vectorize(4)}`；
   * **采样/评估/测量**：对 `CSP'` 采样 N 个可行点，经 CostModel 预选（Heron 的 XGBoost 特征正好来自 CSP 变量/域），再 ε-greedy 测量并回传奖励；
   * **库自进化**：按 Trace 衍生“微调版骨架”（模式匹配/α-应用/重复步骤），剔除表现差与冲突项。 &#x20;

> **结果预期**：由于“先塑形可行域、再解算”的定向性，候选更集中于高质量簇；同时保持 Heron 的“求解器/ε-greedy/测量/CostModel”闭环不变，编译墙钟主要仍花在测量侧（Heron 实证中测量开销远大于搜索器本身）。

---

## 与 CGA 的差异对位（为何是“增量替换”，而非“推倒重来”）

* **操作对象**：CGA 在 **CSP** 上做“IN 交叉 + 随机变异”，本法在 **CSP** 上做“**模板叠加**”。两者都保证可行性，但**本法的动作是“可复用宏步”**，能迁移到相近架构/任务。&#x20;
* **探索驱动**：CGA 主要依赖代价模型的“关键变量”来定交叉，本法以 **Lambda-aware RL** 的**策略**来定向压缩/扩张可行域，减少 RandSAT 的随机性与重复采样。&#x20;
* **库效应**：CGA 无“动作库”，本法的 **Generation/Eliminating** 把“有效跨越步”固化为 λ，**下一次/下游算例可直接复用**。

---

## 代码落地建议（最小侵入）

* 新增 `flex_tuner.py`，接口与 `ga_tuner.py` 对齐；在 `environment.py:createTask()` 中允许 `opt_method='FLEXACTION'` 选择器（其余调用栈不动）。
* 复用 Heron 的 **CostModel** 与特征抽取；必要时引入**分析型仿真器**做二级代理（FlexAction 原生思路），以降低对单一 XGBoost 外推的依赖。&#x20;
* `Lowering()` 实现对接 Heron 的 **C1–C6 约束产生器**（本答的 Algorithm L 即映射规则清单）；求解器仍用 OR-Tools。

---

### 小结

* 以上 **Algorithm F + Algorithm L** 即为“**在 CSP 空间实现 FlexAction 并替换 CGA**”的完整步骤说明，与 Heron 的阶段边界严丝合缝；
* **核心优势**：策略化定向搜索、宏动作可复用、保留 CSP 严谨性与测量闭环；
* **工程成本**：仅在探索器侧新增一层 Lambda-Action 与策略器，生成/求解/测量等关键通道全部复用。
