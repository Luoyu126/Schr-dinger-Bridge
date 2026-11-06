# 相关工作怎么评估采样器（2026-11-06 调研）

文献都在 `/usr0/home/shengyuf/yunyi/ref/`：
- ASBS 正文：`papers/asbs.txt`（由 `papers/asbs.html` 转换）
- 其他论文：`lit_pdfs/<名字>.{pdf,txt}`
- Blessing 等人的评测代码：`variational_sampling_methods/`

下文的 `L<n>` 是对应 `.txt` 文件里的行号。

---

## 1. ASBS 本身（Liu 等，arXiv 2506.22565）

- **没有任何 MIS、离散或组合优化实验。**
  - 全部任务是连续的多粒子能量（`asbs.txt` L584–612）：MW-5、DW-4、LJ-13、LJ-55、丙氨酸二肽、摊销的分子构象生成。
  - 全文检索 "independent set""combinatorial""MIS" 都没有结果。
- **指标全是"与真值样本的分布距离"**（L624，L2755–2845）：
  - MW-5：Sinkhorn 距离。
  - DW-4 和 LJ：样本 W2 和能量直方图 W2，各 2000 个样本，3 次平均。
  - 丙氨酸二肽：5 个扭转角的 KL，以及 Ramachandran 图上的 W2。真值是 10^7 个分子动力学样本。
  - 构象生成：COV-R 和 AMR-R，只有召回率。
- **官方代码只算三个距离**：`energy_w2`、`eq_w2`、`dist_w2`（`adjoint_samplers/components/evaluator.py` L93–123）。
- **没有 best-of-K，也没有重要性权重、ESS 或 log Z。**
  - Table 1 特意强调目标函数"不依赖重要性权重"（L189）。
  - 附录 D.4（L3023–3063）写出了权重 dp*/dp^u，但对非 memoryless 的情形，说所需的比值 "not immediately available"，留作以后的工作。
- **参考过程**：所有任务都取 f=0。能量信息放在源分布上：DW、LJ 和构象用谐振子先验（L726–735），原文说它比高斯源 "slightly better"。
- **温度**：写在目标分布 $\nu\propto e^{-E/\tau}$ 里，按任务固定，训练中不退火。
  - DW 和 LJ 取 τ=1（L2733、L2743），丙氨酸二肽取 300K（L599）。
  - **摊销构象生成刻意取低温 τ≪1**（L604–612），因为目的是找能量极小点。评估看是否覆盖 CREST 的低能构象（COV-R、AMR-R），不看精确的分布距离。
  - ASBS 说沿用 Adjoint Sampling 的设定。Adjoint Sampling 代码中 τ=5e-3（`adjoint_sampling/configs/experiment/spice_cartesian.yaml:42`），力除以 τ（`adjoint_sampling/adjoint_sampling/energies/fairchem_energy.py:116`）。官方 ASBS 代码里没有这个任务的配置。
- **memoryless 的定义**：参考过程下 X0 与 X1 独立（L308–311）。按这个定义，我们的 B0（高斯源 + f=0）**也不是** memoryless，只有 AS（Dirac 源）是。

## 2. Adjoint Sampling（Havens 等，arXiv 2504.11713）

- **DW-4 和 LJ**：报 W2、能量 W2 和 path-ESS（Table 1，`2504.11713.txt` L470–484）。
  - ESS 的目标路径测度是 p* = p_base(X|X1) ν(X1)，源是 Dirac（X0=0），所以权重可以算出来（eq 2–3，L131–140；eq 74–75，L1551–1568）。
  - 每个方法的 ESS 都相对它自己的目标过程计算（L649–651）。
- **构象生成**：COV 和 AMR。表里的 ± 是测试集上跨分子的离散程度（L599）。

## 3. 做 MIS 和图组合优化的学习型采样器：只有优化指标

| 论文 | MIS 数据集 | 指标 | 每个实例的样本数 | 测试温度 |
|---|---|---|---|---|
| DiffUCO（ICML'24，`diffuco.txt`） | RB | 平均大小、best-of-8、相对误差 | 8 | 退火到 0 |
| SDDS（ICLR'25，`sdds.txt`） | RB | 30 个样本的平均、best-of-150、对比 Gurobi | 30 / 150 | 退火到 0 |
| VAG-CO（NeurIPS'23，`vagco.txt`） | RB、TU 数据集等 | best-of-8、30 个样本的平均 | 8 / 30 | 退火到 0 |
| GFlowNet-CO（NeurIPS'23，`gflow_co.txt`） | RB、SATLIB、ER-[700-800] | 大小、对 KaMIS 的差距、时间 | best-of-20 | 逆温度 500 |
| iSCO（ICML'23，`isco.txt`） | SATLIB、ER-[700-800]、ER-[9000-11000] | 大小、差距、时间 | MCMC 链 | 模拟退火 |
| DIFUSCO（有监督，`difusco.txt`） | SATLIB、ER-[700-800] | 大小、差距、时间；代码只记 best-of-K | 1 / 4 / 8 | — |
| RLSA / RLNN（ICML'25，`rld.txt`） | RB、ER | 平均大小、总时间 | 200 / 20 | RLSA 退火，RLNN 取 τ=1 |
| CAM（2026，`cam.txt`） | RB、ER | best-of-20、差距、时间 | 20 | 退火到 0 |

- **没有一篇在 MIS 上检查采样是否无偏。**
- **SDDS 明确写了指标与模式寻找的关系**（`sdds.txt` L550–556）：
  - 反向 KL 训练是模式寻找的，平均质量更高。
  - 正向 KL 训练是质量覆盖的，best-of-150 更好。
  - 我们观察到的"AS 的 MIS 更大但塌缩"就是这个规律。
- **用能量构造加噪过程的先例**，都只用优化指标来判断：
  - DiffUCO 的 Annealed Noise Distribution 用问题能量构造（Sec 3.2，L224–262）。在 100 节点的 RB 图上（Fig. 2，L493–513）：
    - 有退火时，它和均匀噪声差不多；
    - 不退火时，两者均值相近，均匀噪声的种子间方差更大。
    - 原文结论只是 "might be the preferable noise distribution when no or only a little annealing is used"。11-06 上午的版本把它写成了"它更好"，属于夸大，已更正。
  - CAM 复现 SDDS 时，按数据集在 Bernoulli 和 Annealing 两种噪声中选优，"mostly Annealing"（L1176–1178）。这句话紧跟在 TSP 基线列表之后，是否适用于 MIS 不确定。
- **PDNS**（ICLR'26，与 ASBS 有两位共同作者，`pdns.txt`）：
  - 做的是 MaxCut 的摊销采样，取 β=5，让分布集中到最优解上。
  - 报每张图的最大割和平均割，与 Gurobi 对比（L703–772）。
  - 同一篇论文里，Ising 和 Potts 模型则用分布指标评估。
- **温度**：组合优化论文都把 T 退火到 0，起始温度按方法分别调。DISCS 对所有采样器用同一个温度退火 schedule（温度随步数下降的曲线）（`discs.txt` L465）。

## 4. 采样质量的评估：只出现在自旋系统上

- **SDDS**（Sec 5.2，`sdds.txt` L385–403、L605–630）：
  - 用自归一化的神经重要性采样（SN-NIS）。路径权重 = 目标 × 固定加噪链 / 模型路径，和我们设想的权重是同一种构造。
  - 报 ESS，并把自由能、内能、熵与精确值对比。
  - ESS 很低：24×24 Ising 上，正向 KL 训练为 0.0102，反向 KL 为 0.00023；EA 模型上 48 万个样本里只有 1–4 个有效样本。
- **MDNS**（`mdns.txt`）：
  - 在 4×4 Ising 上精确枚举，算 TV、KL、χ²。
  - App D.2.1：ESS 看不出漏掉的模式，原话是 "only relying on ESS as the evaluation metric may be problematic"。
- **DIG-IN**（arXiv 2610.01739，`fixedpoint_discrete.txt`）：
  - 报 ESS、|Δ ln Z|，以及 TVD_mode（每个样本归到一个模式，再与参考的模式质量比 TV）。
  - 一个塌缩到单一模式的模型，ESS 仍有 0.983（Table 1）。
  - 非 memoryless 的离散 ASBS 不报 ESS 和 ln Z，因为没有可用的重要性权重（L2780–2782）。
- **DASBS**（离散 ASBS，`dasbs.txt`）：用样本统计量与 Swendsen-Wang 真值对比（磁化、两点关联、能量 W2），以此比较 memoryless 和非 memoryless 的参考过程（L669–718）。
- **Blessing 等 2024**（`2406.07423.txt`）：
  - 用模型自身样本求期望的指标（ELBO、反向 ESS、Δlog Z）对模式塌缩不敏感（L192–211）。
  - EUBO 和正向 ESS 需要目标样本，并沿固定参考过程反向传播（eq 7–8）。
  - W2 和 MMD 的结论大体一致。

## 5. 对我们评估的含义（建议，不是文献结论）

> **11-06 下午，用户决定主指标用 MIS 的优化指标**，并把我们的设定类比为 ASBS 的摊销构象生成（低温目标，看能否找到低能解），见 `docs/plan.md` §0.7。下面以分布距离为主的方案作废，只在需要论证"没有塌缩"时作为备用。

- **主指标按 ASBS 的惯例，看与参考样本的分布距离。**
  - 小图：符号模式上的 TV（已实现，带噪声下限），再补一个能量 W2。注意 TV 看不到同一个符号象限内部的形状误差。
  - n=100：用 PT-MALA 生成参考样本，再算能量 W2、解码后 MIS 大小的分布，以及按不同最大独立集统计的模式质量 TV。要用多条链和 R-hat 检查 PT-MALA 是否收敛。
- **重要性权重只作辅助。**
  - 用它自归一化地估计目标期望，与 PT-MALA 交叉核对。
  - ESS 只报告，不用来排名。原因有两个：它偏向 memoryless 的 AS，而且对塌缩不敏感。
  - 权重的尾部很重，要报 ESS·M 和 Pareto-k̂。
- **优化指标单独报告**（MIS 领域的惯例）：
  - 平均大小、best-of-K、与 KaMIS 的差距。
  - 所有方法用同一个 T 和同一个解码器，并报修复前的原始可行率。
  - 注明 SDDS 的结论：这些指标偏向模式寻找的方法。
- **摊销实验两类指标都报**：
  - 全部测试图报优化指标。
  - 少量建了参考的测试图报分布指标，给出平均值和最差的那张图。

## 6. 未解决

- n=100 和 n≈250 上 PT-MALA 能否收敛，还没有验证。
- 非 memoryless 参考过程的耦合下限有多大，还没有量化。可以用 KL(μ ‖ ∫ν Q(x0|x1)) 作下界估计，但没试过。
- 没有找到在 MIS 或 hard-core 模型上做无偏采样评估的神经采样器。这一点只基于检索，不能证明不存在。

## 7. 无监督 MIS 的范式，以及参考过程在哪里（11-06 下午调研）

原始报告在 `runs/scratch/landscape/`（`synthesis.md` 和四份 `report_*.md`，被 git 忽略）。下面只记结论。

### 7.1 范式分类

| 家族 | 代表方法 | 参考过程 | 参考里有没有能量 | MIS 结果（对 KaMIS 的差距） |
|---|---|---|---|---|
| 罚函数松弛，一次性输出 | EGN、Meta-EGN、PI-GNN | 无 | — | — |
| 构造式强化学习 | LwD、DIMES、VAG-CO | 无（熵项约等于对均匀策略的 KL） | 否 | ER-[700-800]：DIMES 6.3%，LwD 8.3% |
| GFlowNet | GFlowNet-CO | 反向策略 P_B，固定为均匀（`gflow_co.txt` L982–983） | 否 | ER-[700-800] 8.5% |
| 变分离散扩散 | DiffUCO、SDDS | 前向加噪过程，方向与生成相反 | DiffUCO 部分有（能量退火噪声，见 7.2）；SDDS 没有 | DiffUCO 2.0%，SDDS 3.5%（`cam.txt` L409–410） |
| 学习的局部搜索 | RLNN | 无 | — | 3.9% |
| 随机最优控制 / adjoint matching | **CAM** | base 过程，与生成同向；齐次翻转速率，实验中 λ=0，等价于每步每个坐标以 1/2 概率翻转；初始分布在代码里是 {0,1}^N 上的均匀分布 | 否 | **1.6%**（目前无监督最好） |
| 薛定谔桥 + corrector | ASBS（连续）、DASBS（离散） | 参考过程加源分布，同向 | ASBS 全用 f=0；DASBS 要求速率均匀（`dasbs.txt` L124） | **没有 MIS 实验** |

- AM 和 ASBS 不是独立的范式，而是随机最优控制 / 薛定谔桥框架里的两种训练算法。
- **只有参考过程和生成方向相同的框架（CAM、ASBS），能量进参考才等于"给采样器一个起点"**：网络输出为 0 时，采样器就是参考过程。扩散的前向加噪、GFlowNet 的 P_B 方向相反，能量只能进损失。

### 7.2 已被占掉的部分

- DiffUCO 的能量退火噪声（ICML'24，MIS 主实验的默认设置）：
  ```
  p(X_t | X_{t-1}) ∝ exp(-β·γ_t·X_t^T Q X_t),   γ_t = 1 - t/K
  ```
  - 它不依赖 X_{t-1}，方向与生成相反。
  - 代入损失后温度消掉，等于给每个中间步加一项能量惩罚 Σ_t γ_t·E[H(X_t)]，是奖励塑形，不是动力学先验（官方代码 `NoiseDistributions/AnnealedNoise.py` 核实）。
  - 证据只有最终误差，没有训练效率比较。
- 所以笼统的"用问题能量构造参考 / 噪声过程"不能当新意。能主张的是：**与生成同向、依赖状态、保持终点分布的能量参考过程，并给出训练效率证据**。
- 官方代码里还有一个论文没写的 "combined" 噪声（翻转核乘以能量倾斜，依赖 X_{t-1}），要注意区分。

### 7.3 审稿人一定会问的两个问题

- **"收益是不是只是降温？"**
  - 在没有 corrector 的框架里（CAM），按一阶近似，把能量放进参考过程约等于把终端代价放大、即降低温度，再加一个运行代价。
  - 在 ASBS 里，corrector 固定了两端的边缘分布，边界项是常数，只剩运行代价。所以参考过程改变的是"选哪条桥"，而不是目标分布。
  - 这是继续用 ASBS 做主线的一个理论理由。（推导在 `runs/scratch/landscape/synthesis.md` §4.3，还没有正式检查。）
- **"收益是不是只来自参数化？"**
  - 也就是"采样器 = 能量漂移 + 网络残差"，而参考过程仍是 f=0。这已有先例：PIS（`pis.txt` L311–314）、SCLD（`2412.07081.txt` L1636–1642）、MDNS 的 logit 预条件（`mdns.txt` L2218–2226）。
  - 以 f 为参考时，学到的残差在所有"f + 残差"的写法里控制能量最小。但"残差小"能否变成"训练快"，要靠实验。
- 所以训练效率实验至少要有四组配对：
  1. f=0；
  2. f=0 参考，但网络按 f 加残差参数化；
  3. 只降温；
  4. 能量参考（我们的方法）。

### 7.4 其他

- CAM 的代码在本机 `/usr0/home/shengyuf/CAM4CO`（只读）。
- NAAS（`2506.18165.txt`，作者含 ASBS 作者）在连续采样里用退火 Langevin 作参考过程，说"好的起点"来自参考动力学。但它不是组合优化，需要 Hessian，也没有参考强度的消融。

## 8. 连续空间无监督 MIS 的先例与新颖性（11-06 晚检索）

原始报告在 `runs/scratch/novelty_cont/`（`synth.md` 加三份分报告）。下面只记结论。

- **没找到先例的部分**：用 AS 或 ASBS（adjoint matching 或薛定谔桥）训练、多张图共用一个网络、只用能量的连续时间 SDE 采样器解 MIS。这只是检索结论。
- **"连续空间无监督 MIS"已有人做，不能说首创**：
  - **BloGDiT**（arXiv 2605.25129，2026）：在连续 logit 空间做高斯扩散，用路径反向 KL 训练，只有 5 步，τ 从 1 退火到 0，测了 MIS。它的朴素 DiT 版本在 MIS 上几乎完全失败，RB-large 上平均独立集大小只有 0.29（L519-520）。这从侧面说明连续扩散在 MIS 上难训。
  - **PQQA**（ICLR'25，2409.02135）：在同类松弛能量（QUBO 罚函数加推向顶点的项，λ=2）上跑退火 Langevin，不学习，ER-[700-800] 上 ApR 1.004–1.009，超过 KaMIS（L377-394）。这是最大的威胁：要在相同的能量梯度调用次数下和它比。
  - **SL-BQD**（ICML'25，2505.19438）：用连续的观测 SDE 采 MIS 的 Boltzmann 分布，漂移由离散 MCMC 估计，不学习。
  - **EGN、Meta-EGN、PI-GNN、CRA、pCQO、dNN**：连续松弛的点优化器，不是采样器。**Annealed Training**（2207.11542）是摊销的平均场变分分布，没有动力学。
- **和我们能量设计直接相关的两篇**：
  - **MiP-CRIM**（ICML'26，2606.30333）：给出"推向顶点的项"（包括双势阱）要满足的四个条件，其中 Robustness 条件要求它不能强到让不可行顶点也成为局部极小（L279）。这正是我们 smooth 和整轴双势阱的问题。
  - **pCQO**（ICML'25，2406.19532）：同样是二次罚函数，加不加补图团项，ER-[700-800] 上平均集合大小差约 10%（45.11 对 40.40）。说明能量设计本身对指标影响很大，要和采样器的效果分开报告。
- **可以主张的新意**（都要加"据我们检索"）：
  - S1，场景定位：首次把 AS/ASBS 这类连续时间 adjoint 采样器用于 MIS，只用能量，摊销。偏弱，只适合当定位。
  - S2，稳定性：系统刻画 AS/ASBS 在低温、带罚函数的组合优化能量上的训练失败机制，并给出针对性修复。单独拿出来容易被说成调参。
  - S3，参考过程：在连续薛定谔桥采样器的参考过程里放入组合优化的能量信息（主线）。
  - 建议写法：S1 作定位，S2 和 S3 合起来当主贡献。
- **审稿人会问**：
  - 为什么要连续：离散的 CAM 差距 1.58%，BloGDiT 的连续 DiT 崩了。
  - 学到的是不是只是摊销版的 PQQA。
  - 双势阱松弛下，x 上的目标不等于离散 Boltzmann 分布。
  - 收益是不是来自修复和贪心补全。
  - 裁剪、退火、加步数都是已知技巧。
