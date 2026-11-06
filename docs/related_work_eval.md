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
  - DiffUCO 的 Annealed Noise Distribution 用问题能量构造（Sec 3.1）。在 100 节点的 RB 图上，有退火时它和均匀噪声差不多，没有退火或退火很少时它更好（L500–516）。
  - CAM 复现基线时多数用 "Annealing" 噪声（L1176–1177）。
- **PDNS**（ICLR'26，与 ASBS 有两位共同作者，`pdns.txt`）：
  - 做的是 MaxCut 的摊销采样，取 β=5，让分布集中到最优解上。
  - 报每张图的最大割和平均割，与 Gurobi 对比（L703–772）。
  - 同一篇论文里，Ising 和 Potts 模型则用分布指标评估。
- **温度**：组合优化论文都把 T 退火到 0，起始温度按方法分别调。DISCS 对所有采样器用同一个退火日程（`discs.txt` L465）。

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
