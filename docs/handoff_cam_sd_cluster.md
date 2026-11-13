# 交接：原集群（sd.lti.cs.cmu.edu）上的 CAM / CAMPrior 无监督实验

写于 2026-11-10 14:40 EST，分支 `mis-asbs-energy-base`。读者是上下文压缩之后接着干活的 Claude，也给用户核对用。

这台是原机器：CAM 臂最早在这里搭起来，ER-[700-800] 的种子 0 也在这里跑。补种子、消融和蒸馏热启动在 sa 集群，见 `docs/handoff_cam_sa_cluster.md`。本文写本集群上跑了什么、结果如何、还在跑什么。项目整体进度见 `docs/status.md`；本文和它冲突时，以 `status.md` 为准。

## 0. 先读：用户的硬性规则

1. `AGENTS.md`，以及用户在本集群上的要求：
   - **只在 `/usr0/home/shengyuf/yunyi/` 下面写文件**，包括缓存、临时文件、日志和 checkpoint。不写 `/tmp`，也不写家目录下的缓存。
   - 不往任何已有环境装包。CAM 的 conda 环境 `~/anaconda3/envs/CAM` 只读地用（用户 11-06 同意）。
   - 不碰别人的进程；停自己的进程要先得到用户确认。
   - 不复制 `/usr0/home/shengyuf/CAM4CO` 里的任何文件（里面的脚本有明文 API key）。
   - 提交里**绝不**出现 AI 署名：不加 `Co-Authored-By`，也不加 "Generated with"。
   - 不要提交 `runs/` 下的结果。
   - 全程用中文交流；内部代号（实验名、GPU 编号等）要用白话说明它指什么。
2. 做事方式：
   - 方法或评测上的决定，先查 ASBS 和相关工作怎么做，再按"文献怎么做 → 我建议什么 → 有什么风险"三段来问用户。
   - 每一步遇到问题就停下来问用户，不要自行换方案。
   - 分数掉下来不要停，也不要重启。

## 1. 现状一览（11-13 11:15 EST 更新）

| GPU | 任务 | 状态 |
|---|---|---|
| 0 | 最小支配集 BA-large，原版 CAM（`CAM_tau0.1_e100_s0`） | 训练中，11-13 10:41 启动，见 `docs/handoff_mds_ba_large.md` |
| 1 | 最小支配集 BA-large，CAMPrior 先验温度 0.1、d=20（`CAMPrior_p0.1_d20_e100_s0`） | 训练中，同上 |
| 2 | 最大团 RB-small，原版 CAM（`CAM_tau0.1_e100_s0`） | 训练中，59/100 个 epoch，见 8.7 |
| 3 | 最大团 RB-small，CAMPrior 先验温度 1、d=10（`CAMPrior_p1_d10_e100_s0`） | 训练中，56/100 个 epoch |
| 4 | 最大团 RB-large，原版 CAM（`CAM_tau0.1_e150_s0`） | 训练中，37/150 个 epoch |
| 5 | 最大团 RB-large，CAMPrior 先验温度 0.1、d=20（`CAMPrior_p0.1_d20_e150_s0`） | 训练中，36/150 个 epoch |
| 6 | 最小支配集 BA-large，CAMPrior 先验温度 1、d=20（`CAMPrior_p1_d20_e100_s0`） | 训练中，11-13 10:41 启动 |
| 7 | 别的用户（yixinhu2），约 10:48 起 | 不能用。原本给最小支配集的评测用，三条 test 曲线队列从第 0 个点起全部被拒绝 |

没有空卡。最大团的 test 评测已按用户要求暂停（11-12 22:40）。

## 2. 本集群环境

- 机器：sd.lti.cs.cmu.edu，8 张 RTX 2080 Ti（11 GB）。机器时间是 EST。
- 仓库：`/usr0/home/shengyuf/yunyi/Schr-dinger-Bridge`。
- 环境：`source cam/scripts/env.sh`。它用 `~/anaconda3/envs/CAM` 的解释器（只读），把所有缓存指到 `/usr0/home/shengyuf/yunyi/.cache/cam`，并关掉 wandb。
- 数据（只读）：
  - ER：`/data/shengyuf/MIS/data_er`，和 huggingface `shengyuf/CAM_data` 逐个核对过；
  - RB：`/data/shengyuf/MIS/RB_{small,large}_{train,valid,test}`，11000 个文件和 huggingface 的 MIS.zip 逐个核对过名字、大小和 CRC32，全部一致（核查报告 `runs/scratch/verify_rb/reports.md`）。
- 曲线网页（私有）：https://claude.ai/artifact/Wjg6CWcVdFc8mFiFqfvtBY ，目前只有 ER 种子 0 的五组；生成器 `runs/scratch/plots/val_curves.py`。

## 3. 已完成：ER-[700-800]，5 组，种子 0

设定：单卡、100 个 epoch（官方 1000 个的十分之一，温度退火和余弦学习率都按 100 个 epoch 走完）、官方 ER 超参，`cam/scripts/mis/mis_er700_cam.sh`。epoch 0 是训练前，epoch k 是训练 k 个 epoch 之后。

**validation**（64 张图，50 步 × 20 条链；同预算 RLSA 39.92）

| 实验 | 训练 1 个 epoch 后 | 最好（epoch） | 第 51–100 个 epoch 均值 ± 标准差 | 第 51–100 个 epoch 最低 |
|---|---|---|---|---|
| CAM τ0=0.1 | 37.77 | 44.22（33） | 42.46 ± 1.20 | 39.09 |
| CAM τ0=0.05 | 38.34 | 43.23（53） | 42.16 ± 0.95 | 39.23 |
| CAMPrior 先验 τ0=0.01 | 42.34 | 43.41（80） | 42.97 ± 0.23 | 42.23 |
| CAMPrior 先验 τ0=0.1 | 43.47 | 44.17（36） | 43.82 ± 0.26 | 42.98 |
| **CAMPrior 先验 τ0=1.0** | 43.02 | **44.41（71）** | **44.12 ± 0.22** | **43.53** |

**最终数字：按 validation 挑出的 `best.pt`**（结果在 `runs/cam/mis_er700_best/`）

| 实验 | test（128 张） | ER-[9000-11000]（16 张，200 步） |
|---|---|---|
| CAM τ0=0.1 | 44.04 | 379.3 |
| CAM τ0=0.05 | 42.75 | 371.6 |
| CAMPrior 先验 τ0=0.01 | 43.14 | 301.3 |
| CAMPrior 先验 τ0=0.1 | 44.05 | 379.9 |
| **CAMPrior 先验 τ0=1.0** | **44.18** | **383.3** |
| 参照：CAM 论文（1000 epoch） | 44.16 | 384.38 |
| 参照：KaMIS | 44.87 | 381.31 |
| 参照：同预算 RLSA | 39.94 | 339.69 |

**test 曲线**（128 张图，50 步 × 20 条链，每 10 个 epoch）

| 实验 | 0 | 10 | 20 | 30 | 40 | 50 | 60 | 70 | 80 | 90 | 100 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| CAM τ0=0.1 | 32.62 | 34.53 | 42.87 | 43.65 | 38.59 | 38.29 | 43.03 | 43.68 | 43.12 | 42.79 | 43.01 |
| CAM τ0=0.05 | 32.55 | 36.99 | 34.73 | 33.96 | 34.30 | 35.56 | 41.68 | 41.90 | 42.81 | 42.65 | 42.67 |
| CAMPrior τ0=0.01 | 40.10 | 42.62 | 42.90 | 43.27 | 43.04 | 42.96 | 43.02 | 43.03 | 43.14 | 43.09 | 42.69 |
| CAMPrior τ0=0.1 | 40.33 | 43.51 | 43.89 | 43.60 | 43.80 | 42.75 | 43.89 | 43.27 | 43.95 | 43.84 | 43.10 |
| CAMPrior τ0=1.0 | 38.56 | 43.77 | 43.97 | 43.84 | 44.04 | 44.03 | 43.98 | 44.07 | 44.20 | 44.14 | 43.80 |

**大图曲线**（16 张 ER-[9000-11000]，200 步 × 20 条链，不再训练）

| 实验 | 0 | 10 | 20 | 30 | 40 | 50 | 60 | 70 | 80 | 90 | 100 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| CAM τ0=0.1 | 268.4 | 284.7 | 366.9 | 378.0 | 365.4 | 361.1 | 377.8 | 378.1 | 380.5 | 378.9 | 380.6 |
| CAM τ0=0.05 | 268.2 | 302.6 | 286.7 | 280.6 | 283.9 | 363.6 | 364.7 | 366.1 | 372.4 | 372.4 | 374.1 |
| CAMPrior τ0=0.01 | 340.9 | 370.1 | 336.4 | 308.8 | 300.6 | 303.9 | 303.4 | 304.1 | 301.3 | 299.6 | 301.8 |
| CAMPrior τ0=0.1 | 345.0 | 374.5 | 380.9 | 379.4 | 380.1 | 371.9 | 381.4 | 375.9 | 383.7 | 381.8 | 306.7 |
| CAMPrior τ0=1.0 | 338.4 | 376.4 | 373.7 | 376.1 | 382.4 | 383.2 | 381.1 | 382.9 | 385.5 | 387.9 | 380.1 |

**读法**（1 个种子）
- CAMPrior 起点就在 RLSA 附近，训练 1 个 epoch 后 42.3–43.5；CAM 从随机网络起步（约 32.6），前期剧烈波动。
- 先验 τ0=1.0 最好、最稳：后期 validation 标准差 0.22，CAM 是 0.95–1.20。
- 先验 τ0=0.01（最硬）：87% 的坐标翻转概率饱和成 0 或 1，残差几乎学不动；大图泛化从第 20 个 epoch 起掉到约 300，原因未查。
- **最后一个 epoch 掉分**：训练温度接近 0 时，CAMPrior τ0=1.0 的 test 从 44.14 掉到 43.80，大图从 387.9 掉到 380.1；τ0=0.1 的大图从 381.8 掉到 306.7。推测是先验温度绑定在训练温度上，被放大最多 100 倍、变硬（`--prior_tau_min` 默认 1e-3）。sa 集群的"先验温度不绑定"消融不掉分，支持这个推测；还没有直接诊断。

## 4. 已完成：RB-small 和 RB-large

**为什么做**：把结论横向推广到 CAM 论文的另外两个 MIS 数据集。CAM 论文在 MIS 上训了 3 个独立模型：ER-[700-800]、RB-[200-300]、RB-[800-1200]；ER-[9000-11000] 只做迁移测试。

**设定**：沿用 ER 的协议，单卡、官方 epoch 数的十分之一、种子 0，其余照官方 RB 脚本。运行脚本 `cam/scripts/mis/mis_rb_cam.sh`（`SCALE=small|large`），启动器 `cam/scripts/mis/rb_runs.sh`，评测队列 `cam/scripts/mis/rb_curve.sh`。

| | RB-small | RB-large |
|---|---|---|
| 训练 epoch | 100 | 150 |
| CAM 组 | τ0=0.1 | τ0=0.1 |
| CAMPrior 组 | 先验 τ0=1.0，d=5 | 先验 τ0=1.0，**d=20** |

**先验步长 d 的选法（11-08 和用户商定）**：每个数据集在 64 张 validation 图上、按 CAM 的采样预算（50 步 × 20 条链）选，不看 test。
- RLSA 原论文在 RB 上用 d=5，但那是配 300–500 步、200 条链调的。
- RB-large 上 d=5 的先验只有 33.61，和随机网络差不多；d=20 是 34.41，同一批图配对比较高 0.80 ± 0.13。RB-small 上 d=5 最好。
- 扫描结果在 `runs/scratch/verify_rb/rlsa_{small,large}/`。

**同预算 RLSA 基线**（完整 test 集，`runs/cam/mis_rb_rlsa/`）

| | RB-small | RB-large |
|---|---|---|
| RLD 原设定（τ0=0.01，d=5） | 18.61 | 33.61 |
| validation 上调到最好 | 18.69（τ0=0.1，d=5） | 35.50（τ0=0.01，d=50） |

**RB-small test 曲线**（1000 张图，50 步 × 20 条链；CAM 论文 1000 epoch 19.91，KaMIS 20.10）

| 实验 | 0 | 1 | 2 | 5 | 10 | 20 | 30 | 40 | 50 | 60 | 70 | 80 | 90 | 100 | best.pt |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| CAM τ0=0.1 | 17.39 | 17.81 | 18.51 | 17.89 | 19.59 | 19.65 | 19.64 | 19.69 | 19.64 | 19.66 | 19.70 | 19.73 | 19.70 | 19.70 | 19.78（validation 最好 20.16，第 51 个） |
| CAMPrior d=5 | 18.23 | 18.98 | 19.37 | 19.62 | 19.81 | 19.84 | 19.86 | 19.92 | 19.91 | 19.93 | 19.91 | 19.92 | 19.93 | 19.93 | **19.93**（validation 最好 20.38，第 78 个） |

**RB-large test 曲线**（1000 张图；CAM 论文 1500 epoch 41.25，KaMIS 43.15）

| 实验 | 0 | 1 | 2 | 5 | 10 | 20 | 30 | 40 | 50 | 60 | 70 | 80 | 90 | 100 | 110 | 120 | 130 | 140 | 150 | best.pt |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| CAM τ0=0.1 | 34.20 | 34.32 | 34.31 | 32.96 | 33.30 | 34.63 | 40.04 | 40.74 | 40.79 | 40.57 | 40.99 | 40.42 | 40.93 | 40.89 | 40.97 | 40.99 | 40.74 | 40.75 | 40.78 | 41.07（validation 最好 40.77，第 97 个） |
| CAMPrior d=20 | 34.81 | 33.49 | 33.34 | 33.68 | 34.38 | 39.80 | 40.51 | 41.04 | 41.27 | 41.11 | 41.33 | 41.05 | 41.23 | 41.21 | 41.27 | 41.27 | 41.21 | 41.17 | 41.17 | **41.34**（validation 最好 41.03，第 93 个） |

**读法**（1 个种子）
- RB-small：CAMPrior 每个点都领先，后期稳定领先约 0.2，和 CAM 论文 1000 epoch 持平。RB-small 接近饱和，差别主要在训练早期（第 5 个 epoch 19.62 对 17.89）。
- RB-large：CAMPrior 起点只领先 0.6，前 2 个 epoch 小幅下滑，第 5 个 epoch 起每个点都领先；第 20 个 epoch 差距最大（39.80 对 34.63），第 40 个 epoch 起稳定在 41.0–41.3。最终 best.pt 41.34 对 41.07，第 150 个 epoch 41.17 对 40.78；CAM 论文训练 1500 epoch 是 41.25。
- RB-large 上 CAMPrior 没有出现 ER 上那种最后一个 epoch 掉分（第 140→150 个 epoch 41.17→41.17）。

## 5. 数据位置（本集群，`runs/` 不进 git）

- ER 训练日志：`runs/queue_cam_er700/mis_er700__<实验名>.log`；checkpoint：`runs/cam/mis_er700/<实验名>/`；训练前的初始权重：`runs/cam/mis_er700_init2/<实验名>/`。
- ER 评测：`runs/cam/mis_er700_test10/`、`runs/cam/mis_er700_large/`（`<实验名>/epKKK/test_log.txt`，第一行取负号）、`runs/cam/mis_er700_best/`；同预算 RLSA：`runs/cam/mis_er700_rlsa_{val,test,large}/`。
- RB 训练日志：`runs/queue_cam_rb/mis_rb_<规模>__<实验名>.log`；checkpoint：`runs/cam/mis_rb_{small,large}/<实验名>/`。
- RB 评测：`runs/cam/mis_rb_{small,large}_test/<实验名>/{epKKK,best}/`；同预算 RLSA：`runs/cam/mis_rb_rlsa/`。
- 实验名：ER `CAM_tau0.1_e100_s0`、`CAM_tau0.05_e100_s0`、`CAMPrior_p{0.01,0.1,1}_e100_s0`；RB-small `CAM_tau0.1_e100_s0`、`CAMPrior_p1_d5_e100_s0`；RB-large `CAM_tau0.1_e150_s0`、`CAMPrior_p1_d20_e150_s0`。

## 6. 本集群上遇到的问题和改动

- **评测卡被别的用户占用**：1 号卡（11-08）、3 号卡（11-09）、2 号卡（11-09 夜）先后被占。安全检查 `cam_require_gpu` 正确地拒绝了，但旧的 ER 评测队列被拒后立刻重试、空转。已改：新的 `rb_curve.sh` 同一个点最多重试 2 次，两次之间等待，然后放弃；被放弃的点都已补齐。
- **测试时先验设置不一致改为报错**：`cam_prior.py` 原来只打印警告，漏传 `--prior_d` 会悄悄用 d=20 评测 d=5 的模型；现在报错，`--prior_allow_mismatch` 才放行（`cam/UPSTREAM.md` 第 16 条）。
- **核查发现的上游问题**：官方多卡训练时余弦学习率实际推进快 8 倍（accelerate 包装后的调度器每个进程各走一步），8 卡下学习率来回 4 个周期；我们单卡是正常衰减一次。不影响 CAM 和 CAMPrior 之间的对比，和论文数字比较时要注意（`status.md` §2.6）。

## 7. 接下来

1. ~~RB-large 训完后的评测~~：已完成（11-11 16:15，在 6、7 号卡上）。
2. 种子 0 三个数据集的 best.pt 汇总表在 `docs/status.md` §2.6 末尾。
3. 待用户决定：
   - 最后一个 epoch 掉分的诊断：拿第 100 个 epoch 的 checkpoint，测试时把先验放大倍数限制住（`--prior_tau_ratio`），看能否回到第 90 个 epoch 的水平；不用训练，几分钟。
   - RB 两个规模补种子 1、2（8 个任务；RB-small 约 1.4 天，RB-large 约 3 天）。
   - 曲线网页加上 RB 和 sa 集群的种子曲线。
3. 论文大纲文档（https://claude.ai/artifact/RNCKAHkLcgJGsb9UJTAvSo ）按用户要求暂停，只写了第一节；相关工作定位要等文献调研的分工定下来再写。

## 8. 新任务：最大团（Maximum Clique，MCl），RB-[200-300] 和 RB-[800-1200]（11-13 写）

**为什么做**：RLSA 论文（arXiv 2502.00277，表 2）在 MIS 之外还比了最大团和 MaxCut。MaxCut 已在 sa 上做完；用户 11-12 要求把最大团也跑了，完全对上 RLSA 的设定。CAM 论文没做最大团，所以这一项只和 RLSA 论文比：RLSA 18.97 / 40.53，iSCO 18.96 / 40.35，RLNN 18.13 / 35.23，DiffUCO 16.22（只有小图），Gurobi 19.05（只有小图）。

**卡怎么分（用户定的）**：先用 2–7 号六张卡快速扫参，扫完后 4 张卡训练、1 张卡评测、1 张卡空着。0、1 号卡是别的用户的，不碰。

### 8.1 代码（已在 sa 上写好、测过、推送，提交 8bb9d81..3a91652）

- `cam/envs/mcl.py`：最大团 = 补图上的 MIS（RLSA 论文式 19），能量 −Σxᵢ + β·#(选中但不相邻的点对)。补图不显式建（RB-large 补图有约 47 万条边），用 A_c·x = (同图 x 之和 − x)/2 − A·x 隐式算，A 沿用数据集的 1/2 边权。**GNN 仍在原图上传消息**，只有能量和梯度用补图。解码：连续解码按概率贪心建团，离散解码同 MIS 的局部搜索。
- `cam/datasets/mcl.py`：直接复用 MIS 的读图（同一批 RB 图；RLSA 论文 MIS 和最大团用的也是同一套 RB 数据）。
- `cam/tests/test_mcl.py`：5 个测试，和显式补图上的 MIS 能量、梯度逐项对比（误差 1e-6），解码结果都是极大团，RLSA 能找到植入的 8-团。
- `mis_rb_cam.sh`、`rb_runs.sh`、`rb_curve.sh` 加了 `PROBLEM=mcl`（默认还是 mis，原来的命令不变）；最大团的结果目录是 `runs/cam/mcl_rb_<规模>/`。
- 新增 `cam/scripts/mis/rb_rlsa_sweep.sh`：多卡并行在 64 张 validation 图上扫 RLSA 的 (τ0, d)，预算同 CAM（50 步 × 20 条链）。
- sa 上冒烟测试都通过：扫参（8 张图，约 10 秒一个点），两个规模的 CAM / CAMPrior 各训 1 个 epoch，`rb_curve.sh` 评测（含 CAMPrior 读回先验设置）。
- 注意：`--label_mode sim|lin`（标签消融）只支持 MIS，对最大团会直接报错。

### 8.2 第一步：拉代码、跑测试

```bash
cd /usr0/home/shengyuf/yunyi/Schr-dinger-Bridge
git pull origin mis-asbs-energy-base
source cam/scripts/env.sh
$CAM_PYTHON cam/tests/test_mcl.py                                   # 5/5 passed（test_5 显示 skipped）
MCL_TEST_RB=/data/shengyuf/MIS/RB_small_valid $CAM_PYTHON cam/tests/test_mcl.py   # 5/5
$CAM_PYTHON cam/tests/test_cam_prior.py                             # 6/6，确认原来的路径没坏
```

### 8.3 第二步：六张卡扫参（预计 15–30 分钟）

```bash
mkdir -p runs/queue_cam_rb
PROBLEM=mcl GPUS="2 3 4 5 6 7" PAR=2 SCALES="small large" TAUS="0.1 1 4 10" DS="2 5 10 20 50" \
  nohup bash cam/scripts/mis/rb_rlsa_sweep.sh > runs/queue_cam_rb/mcl_rlsa_sweep.log 2>&1 &
```

两个规模各 20 个点，共 40 个；结果在 `runs/cam/mcl_rb_<规模>_rlsa_sweep/t<τ0>_d<d>/`，日志末尾按分数排好序。

**怎么选（和 MIS、MaxCut 的口径一致）**：
- CAMPrior 的先验温度固定 τ0 = 1（ER、RB、MaxCut 都是 1），**d 取 τ0 = 1 那一行 validation 最好的**。只有当 τ0 = 1 的最好值比整个网格的最好值低 0.3 以上时，才改用整个网格最好的 (τ0, d)，并在汇报里写明。
- 同预算 RLSA 基线用整个网格 validation 最好的 (τ0, d)。
- 同分时取更小的 d（更接近 RLSA 论文最大团的 d = 2）。
- 参考：sa 上 8 张图的冒烟结果，RB-small τ0=4,d=2 19.13、τ0=1,d=2/20 18.50；RB-large τ0=4,d=20 41.38、τ0=1,d=20 41.25、τ0=1,d=2 30.50。大图上 d=2 明显差（和 MIS RB-large 上 d=5 差的现象一样），只有 8 张图，以 64 张图为准。

### 8.4 第三步：四张卡训练（2–5 号卡）

协议同 MIS 的 RB：单卡、种子 0、RB-small 100 个 epoch、RB-large 150 个 epoch（官方 epoch 的 1/10，退火和余弦学习率按缩短后的长度走完），每个 epoch 存 checkpoint，其余超参照搬 MIS 的 RB 脚本，β = 1.02（RLSA 论文最大团也是 1.02）。把下面的 `<dS>`、`<dL>` 换成上一步选出的 d（如果 τ0 不是 1，组名里的 `p1` 也换掉）：

```bash
PROBLEM=mcl GPUS="2 3" SCALES=small RUN_GROUPS="CAM_tau0.1 CAMPrior_p1_d<dS>" DRY=1 bash cam/scripts/mis/rb_runs.sh   # 先看命令
PROBLEM=mcl GPUS="2 3" SCALES=small RUN_GROUPS="CAM_tau0.1 CAMPrior_p1_d<dS>" bash cam/scripts/mis/rb_runs.sh
PROBLEM=mcl GPUS="4 5" SCALES=large RUN_GROUPS="CAM_tau0.1 CAMPrior_p1_d<dL>" bash cam/scripts/mis/rb_runs.sh
```

训练日志 `runs/queue_cam_rb/mcl_rb_<规模>__<实验名>.log`，checkpoint `runs/cam/mcl_rb_<规模>/<实验名>/`。MIS 上 RB-small 约 20 分钟一个 epoch（约 1.4 天），RB-large 约 29 分钟一个 epoch（约 3 天），最大团的能量多一次散射求和，时间应差不多；启动后看第一个 epoch 的实际时间并汇报。

### 8.5 第四步：一张卡评测（6 号卡），7 号卡空着

test 曲线（1000 张 test 图，50 步 × 20 条链，第 0、1、2、5、10、20…个 epoch 加 best.pt），两个规模各一条队列，都在 6 号卡上：

```bash
mkdir -p runs/cam/mcl_rb_small_test runs/cam/mcl_rb_large_test
PROBLEM=mcl SCALE=small DEVICE=gpu GPU=6 PAR=1 EXPS="CAM_tau0.1_e100_s0 CAMPrior_p1_d<dS>_e100_s0" \
  nohup bash cam/scripts/mis/rb_curve.sh > runs/cam/mcl_rb_small_test/queue.log 2>&1 &
PROBLEM=mcl SCALE=large DEVICE=gpu GPU=6 PAR=1 EXPS="CAM_tau0.1_e150_s0 CAMPrior_p1_d<dL>_e150_s0" \
  nohup bash cam/scripts/mis/rb_curve.sh > runs/cam/mcl_rb_large_test/queue.log 2>&1 &
```

RLSA 基线（完整 test 集，6 号卡，训练开始后就能跑，不依赖训练）：

```bash
for cfg in "small|same|--num_t 50 --num_k 20 --tau0 <τS> --num_d <dS'>" \
           "large|same|--num_t 50 --num_k 20 --tau0 <τL> --num_d <dL'>" \
           "small|paper|--num_t 100 --num_k 200 --tau0 4 --num_d 2" \
           "large|paper|--num_t 500 --num_k 200 --tau0 4 --num_d 2"; do
  IFS="|" read -r sc name args <<< "$cfg"
  CUDA_VISIBLE_DEVICES=6 PROBLEM=mcl SCALE=$sc MODE=test PROJECT=mcl_rb_rlsa EXP=${sc}_$name \
    bash cam/scripts/mis/mis_rb_cam.sh --method RLSA $args > runs/queue_cam_rb/mcl_rlsa_${sc}_$name.log 2>&1
  echo "$sc $name $(head -1 runs/cam/mcl_rb_rlsa/${sc}_$name/test_log.txt)"
done
```

`<τS> <dS'>`、`<τL> <dL'>` 是扫参里整个网格最好的配置；`paper` 是 RLSA 论文最大团的设定（表 4：τ0 = 4，d = 2，200 条链，RB-small 100 步、RB-large 500 步，β = 1.02），用来核对我们的 RLSA 实现和数据能不能复现论文的 18.97 / 40.53。如果差得多（比如 > 0.3），先停下来告诉用户，可能是数据不同（RLSA 用 DiffUCO 的 RB 图，我们用 CAM 发布的 RB 图）。

### 8.6 汇报

- 扫参结束后：两张表（每个规模一个 τ0 × d 表），选出的 d，以及是否偏离 τ0 = 1。
- 训练第一天：每个 epoch 的时间，validation 曲线前几个点，有没有发散。
- 全部结束后：test 曲线表、best.pt、两个 RLSA 基线，和 RLSA 论文表 2 对比；同时更新 `docs/status.md`。

### 8.7 进度（11-13 11:15 EST）

**测试**：`test_mcl.py` 5/5（加上真实 RB 图也是 5/5），`test_cam_prior.py` 6/6（11-12 当时的版本）。

**扫参**（64 张 validation 图，50 步 × 20 条链，β=1.02，平均团大小，越大越好；结果在 `runs/cam/mcl_rb_<规模>_rlsa_sweep/`）

RB-small：

| τ0 \ d | 2 | 5 | 10 | 20 | 50 |
|---|---|---|---|---|---|
| 0.1 | 17.81 | **18.53** | 18.38 | 18.30 | 17.70 |
| 1 | 18.27 | 18.25 | **18.44** | 18.08 | 17.30 |
| 4 | 17.91 | 18.11 | 18.14 | 17.73 | 16.83 |
| 10 | 17.80 | 17.47 | 17.58 | 17.52 | 16.66 |

RB-large：

| τ0 \ d | 2 | 5 | 10 | 20 | 50 |
|---|---|---|---|---|---|
| 0.1 | 26.59 | 29.34 | 38.38 | **39.19** | 38.11 |
| 1 | 27.02 | 28.92 | 38.05 | **38.81** | 37.55 |
| 4 | 34.75 | 37.77 | 38.22 | 37.73 | 36.92 |
| 10 | 34.05 | 34.72 | 34.84 | 38.02 | 35.88 |

- **RB-small**：τ0=1 这一行最好是 d=10（18.44），比整个网格最好（τ0=0.1、d=5，18.53）只低 0.09，所以 CAMPrior 用先验温度 1、**d=10**，没有偏离 τ0=1。
- **RB-large**：τ0=1 这一行最好是 d=20（38.81），比整个网格最好（τ0=0.1、d=20，39.19）低 0.375，超过 0.3，按规则改用先验温度 **0.1**、**d=20**，**偏离了 τ0=1**。同一批 64 张图配对比较，两者差 +0.38 ± 0.53，在噪声内。
- **同预算 RLSA 基线**：RB-small 用 τ0=0.1、d=5；RB-large 用 τ0=0.1、d=20。
- 和 MIS 一样，大图上 d=2（RLSA 论文最大团的设定）在 50 步预算下明显差。

**训练**（种子 0，单卡；validation 64 张图，平均团大小，越大越好；训练前一列是 test 第 0 个点）

| 实验 | 卡 | 已跑 epoch | 每个 epoch | 预计结束 |
|---|---|---|---|---|
| RB-small 原版 CAM，`CAM_tau0.1_e100_s0` | 2 | 59 / 100 | 约 16.5 分钟 | 11-13 约 22:30 |
| RB-small CAMPrior 先验温度 1、d=10，`CAMPrior_p1_d10_e100_s0` | 3 | 56 / 100 | 约 17.5 分钟 | 11-14 约 0:00 |
| RB-large 原版 CAM，`CAM_tau0.1_e150_s0` | 4 | 37 / 150 | 约 25.5 分钟 | 11-15 约 11:00 |
| RB-large CAMPrior 先验温度 0.1、d=20，`CAMPrior_p0.1_d20_e150_s0` | 5 | 36 / 150 | 约 25.5 分钟 | 11-15 约 11:30 |

| 实验 | 训练前（test） | 1 | 2 | 5 | 10 | 20 | 30 | 40 | 50 | 最好（epoch） | 最近 5 个 epoch 均值 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| RB-small CAM | 16.89 | 15.66 | 11.67 | 17.64 | 16.00 | 16.56 | 15.27 | 17.39 | 12.36 | 18.20（6） | 15.22 |
| RB-small CAMPrior | 18.72 | 12.64 | 12.44 | 12.97 | 13.12 | 10.91 | 11.42 | 13.38 | 16.20 | 17.50（53） | 14.54 |
| RB-large CAM | 31.22 | 34.42 | 30.83 | 31.14 | 28.53 | 28.55 | 29.36 | | | 35.39（3） | 30.58 |
| RB-large CAMPrior | 38.83 | 34.95 | 34.48 | 35.05 | 24.39 | 27.58 | 32.00 | | | 39.59（11） | 30.86 |

- **没有报错，但训练在退化**：四组训练后都比各自训练前差，CAMPrior 掉得更多（它的起点就是 RLSA）；分数大幅来回跳（RB-large CAMPrior 在 23–39 之间）。MIS 的同样设定没有这种现象。
- RB-small CAMPrior 最近有回升（第 50 个 epoch 16.20、第 53 个 17.50），还没回到起点 18.72。
- 用户 11-12 同意继续跑、不停。

**诊断（11-12，用户同意，代码只在 `runs/scratch/mcl_diag/`）**
- 训练目标的大小（8 张 RB-small 训练图）：在 RLSA 链终点，最大团的 adjoint 目标平均 9.7，MIS 是 2.1；而且很不对称，没选中节点"不要加"的信号约 +10，已选中节点"保留"的信号只有约 −0.5。这会把策略推向"不翻 / 全关"。
- 训练温度试训（7 号卡，RB-small，每组 2 个半 epoch）：CAM 训练温度 0.1 / 0.5 / 2 的 validation 是 16.9 / 16.2 / 15.8；CAMPrior 是 16.3 / 12.5 / 16.5。提高训练温度不能稳定避免下滑，温度假说不成立。
- 文献：RLSA 论文图 2、表 3 也观察到只用无监督损失时最大团会退化（RLNN 不加正则 16.62，加正则 18.13）。CAM 论文附录 C.1 的"更新幅度正则"（代码里的 `--lambd`、`--num_d`，默认关）是可能的补救，还没试，等用户决定。

**评测**（1000 张 test 图，50 步 × 20 条链）

| 实验 | 第 0 个 epoch | 第 1 个 | 第 2 个 | 第 5 个 |
|---|---|---|---|---|
| RB-small CAM | 16.89 | 16.11 | 11.82 | 18.08 |
| RB-small CAMPrior | 18.72 | 12.76 | 12.17 | |
| RB-large CAM | 31.22 | 33.12 | 30.88 | |
| RB-large CAMPrior | 38.83 | 34.88 | 35.16 | |

- 之后的点没有评：6 号卡（11-12 约 21:00）和 7 号卡（22:40）先后被别的用户占用，评测按用户要求暂停。checkpoint 每个 epoch 都在，以后补。

**RLSA 基线**（1000 张 test 图，结果在 `runs/cam/mcl_rb_rlsa/`）

| | RB-small | RB-large |
|---|---|---|
| 同预算（50 步 × 20 条链；RB-small τ0=0.1、d=5，RB-large τ0=0.1、d=20） | 18.81 | 38.75 |
| 论文设定（τ0=4、d=2、200 条链；100 / 500 步） | 18.965 | 40.598 |
| RLSA 论文表 2 | 18.97 | 40.53 |

论文设定复现到 0.005 和 0.07 以内，说明我们的实现和数据（CAM 发布的 RB 图）都对得上。

**遇到的问题和改动**
- 6 号、7 号卡先后被别的用户占用，评测队列被拒绝；按用户要求暂停评测。
- 仓库里的代码和脚本没有改动；诊断脚本 `runs/scratch/mcl_diag/target_scale.py` 和 `pilot.sh` 只在本机，不进 git。

