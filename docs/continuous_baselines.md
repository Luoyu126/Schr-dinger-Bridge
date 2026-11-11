# 连续松弛基线：pCQO 和 PQQA 在 ER-[700-800] 上的复现

写于 2026-11-11。代码在 `sbmis/baselines/`（`pcqo.py`、`pqqa.py`），运行脚本 `scripts/baselines/continuous_mis.py`，测试 `tests/test_continuous_baselines.py`。结果在 `runs/baselines/<实验名>/result.json`（不进 git）。

## 1. 为什么复现、怎么复现

两篇都是免训练的 continuous relaxation 方法：把 x ∈ {0,1}ⁿ 放松到 [0,1]ⁿ，在可微的 QUBO energy 上做梯度下降，最后 rounding。它们在 ER-[700-800] 上超过所有学习方法和 KaMIS，是我们在 MIS 上绕不开的参照。

外部仓库的代码和数据文件（编译 pCQO 的 C++、反序列化它自带的图文件）没有直接运行；按官方源码在本仓库重写，跑在我们自己的 ER-[700-800] test 集（128 张图，`/home/shengyuf/yunyi/data/CAM_data/data_er/er_test`）上。

- **pCQO**（Alkhouri et al., ICML 2025，arXiv 2406.19532）：照 `ledenmat/pCQO-mis-benchmark` 的 `cpp_impl/pcqo_mis.cpp` 逐行重写。f(x) = −Σx + γΣ_E x_u x_v − γ'Σ_{E'} x_u x_v（E' 是补图的边），float16，动量投影梯度下降，速度不清零；每 450 步把 256 条解按 0.5 取整，记下最大的 maximal independent set，再以当前解为均值、std 2.25 重新采样。超参是官方 ER 设置：lr 9e-6，动量 0.9，γ=350，γ'=7，225000 步。
- **PQQA**（Ichikawa & Arai, ICLR 2025，arXiv 2409.02135）：照 `Yuma-Ichikawa/QQA4CO` 第一次公开的源码（commit b8721ae，`src/qqa.py`）重写，不用现在仓库里加的 restart、polish 等论文没有的东西。p = Clamp(w)（论文 App. C.1，**不是** sigmoid），AdamW（weight decay 0.01），Langevin 噪声 T=1e-3，lr=1；γ 从 −2 线性升到 0.1，α-entropy 取 α=4；并行通信项按论文式 (10) 的求和形式，α=0.1；MIS 的 QUBO 用代码里的写法 xᵀQx（Q_ii=−1，边上 Q_uv=Q_vu=3，罚项系数取自作者 ER 实例代码 commit 558a7ce）。
- 两者都把 128 张图补齐到同样大小、在一张 RTX 2080 Ti 上一起跑，所以时间是 128 张图的总时间，和论文的逐图顺序计时不完全可比。
- 小图测试：两者在 10–14 个点的图上都找到精确最优；向量化的 maximal-set 检查和 C++ 公式逐个一致。

## 2. 结果（128 张 test 图的平均独立集大小）

| 方法 | 设定 | 复现 | 用时（128 张图合计） | 论文 |
|---|---|---|---|---|
| PQQA | S=100，3000 步 | 44.94 | 41 秒 | 论文四组设定 ApR 1.004–1.009 |
| PQQA | S=100，30000 步 | 45.16 | 7.1 分钟 | |
| PQQA | S=1000，3000 步 | 45.16 | 5.9 分钟 | |
| PQQA | S=1000，30000 步 | **45.27** | 60 分钟 | 1.009 × 44.87 = 45.27，1.14 小时 |
| pCQO | 官方 ER 设置 | 45.16 | 15.6 分钟 | 45.11，54.8 分钟 |
| 参照：KaMIS | | 44.87 | | |
| 参照：CAMPrior（best.pt，种子 0） | 50 步 × 20 条链 | 44.18 | | |
| 参照：同预算 RLSA | 50 步 × 20 条链 | 39.94 | | |

- 两个方法都复现了论文数字。PQQA 的 ApR 在论文里是逐图比值的平均，我们只有 KaMIS 的整体均值，按均值比。
- PQQA 原代码返回的是 loss 最低的取整解；四组里它在 128 张图上都没有冲突，所以表里的数和原代码的输出一致。
- 质量随时间上升得很快：PQQA S=100 在约 21 秒（1500 步）就到 44.94 并停住；pCQO 第一次检查（450 步，1.9 秒）是 39.4，约 4 分钟到 44.87。

## 3. 对论文的含义（11-11 讨论）

- 在 MIS 上，免训练的 pCQO 和 PQQA 绝对质量都高于我们（45.2 对 44.2），而且两篇都已在 2025 年正式发表，审稿人有理由要求比较。
- 它们需要 continuous relaxation 和可微 energy，只适用于 QUBO 类问题；RLSA 只需要 ∇H，CAM 只需要翻转差值或一个 local search 启发式（TSP 用 2-OPT 构造替身代价）。
- 论文定位和对比口径见论文大纲文档（"适用范围和定位收束"一节和 5.1 节"对比口径"）。还没做：同一张卡、同样时间下 CAMPrior 和 PQQA 的对比。
