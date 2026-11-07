# 交接：AS 训练稳定性小试（在另一个集群上运行）

写于 2026-11-06 晚，分支 `mis-asbs-energy-base`。读者是在另一个集群上接手这项任务的 Claude。

## 0. 先读

1. `AGENTS.md`：仓库的硬性规则。有两点在新集群上要变通：
   - 文件里写的路径 `/usr0/home/shengyuf/yunyi/` 是原机器的。在新集群上，**只在用户指定的工作目录里写文件**，包括缓存和临时文件。不要写 `/tmp`，也不要写家目录下的缓存。`scripts/pilots/as_pilots.sh` 已经把 `TMPDIR` 设到了仓库内的 `.cache/tmp`。
   - 其余规则照旧：
     - 不往任何已有环境里装包，自己建 venv；
     - 提交里绝不出现 AI 署名（不加 `Co-Authored-By`，也不加 "Generated with"）；
     - 和用户用中文交流，不要不加解释地使用内部代号。
2. 本文。
3. 背景（需要时再读）：
   - `docs/plan.md` §0.8，特别是"连续 v2 批次训崩的诊断"那一条；
   - 完整诊断在原机器的 `runs/scratch/collapse/synth.md`。runs/ 被 git 忽略，所以新集群上没有这个文件，本文第 2 节已经摘录了要点。

## 1. 任务是什么

项目在 MIS（最大独立集）上研究无监督扩散采样器：把 MIS 松弛到连续空间 x∈R^N，目标分布是 exp(−E/T)，用 Adjoint Sampling（AS）和 ASBS 训练。网络是在多张图上共享的 GNN。

用新协议（T=0.075、smooth 能量映射、逐节点裁剪 150、fp32、25 步 SDE）训练的一批连续采样器，在第 1–5 个 stage 内全部训崩，而且没有学到任何东西。

**这次的任务**：只用 AS（f=0，即参考过程为纯布朗运动），跑 4 个小试，判断是哪个原因导致训崩、哪个改动能稳住训练。然后视情况再跑第二轮的 2 个。

## 2. 诊断要点（假设，等着被小试验证）

1. **时间离散太粗**（置信度高）。
   - T=0.075 时，每个节点在能量里的刚度 k = 曲率/T 在 127–567 之间。显式 Euler 稳定要求 k·dt < 2，即至少约 64–280 步，而训练只用了 25 步。
   - 学到的拉回控制在 25 步网格上变成周期为 2 的抖振：每步来回移动 4–6，而 x 的合理范围宽度只有 2。
   - 同一个网络换成 200 步模拟，E_tgt 从几万降到个位数。
2. **裁剪阈值和步长不匹配**（置信度中高）。
   - 逐节点裁剪 α=150 照抄了 AS，但 AS 配的是 dt=0.001，每步最多移动 0.15；我们是 dt=0.04，每步最多移动 6。
   - 出界后回归目标被截成 ±150，网络学成"开关式"的全力控制，控制能量约等于 0.5·n·α²。
3. **空集陷阱**（置信度中）。第一次更新把所有节点一起往下推；smooth 映射在角点处梯度为 0，"全不选"角点成了一个平的陷阱。
4. **助因**：AS 迭代本身没有阻尼（完全拟合等于走单位步长）；AM 偏向众数。

**一个必须知道的陷阱：平均 MIS 会骗人。** 采样器塌到空集时，解码器的贪心补全仍然能给出一个看起来还行的 MIS。在 32 张验证图上，空集解码的平均值是 26.41，比未训练（u=0）的 26.05 还高；在小试用的前 16 张图上约为 25.69。所以平均 MIS 必须和"空集解码基线"一起看，`summarize_as_pilots.py` 会自动算出这个基线。

## 3. 准备

1. **代码**
   ```bash
   git clone git@github.com:Luoyu126/Schr-dinger-Bridge.git
   cd Schr-dinger-Bridge && git checkout mis-asbs-energy-base
   ```
2. **环境**：Python 3.9 的新 venv。原机器用的版本：
   ```bash
   python3.9 -m venv <工作目录>/envs/sb
   <工作目录>/envs/sb/bin/pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
   <工作目录>/envs/sb/bin/pip install numpy==2.0.2 scipy==1.13.1 networkx==3.2.1 pytest==8.4.2
   ```
   - 把 pip 缓存放在工作目录里：`PIP_CACHE_DIR=<工作目录>/.cache/pip`。
   - 不需要 PyG 或 CAM 的依赖。
   - 原机器的完整 `pip freeze` 在 `runs/scratch/transfer/requirements_sb.txt`，用户会和数据一起拷过来。
3. **数据**：`data/datasets/er_200_300/` 被 git 忽略，**不要在新集群上重新生成**（最好已知解要靠 MILP 求几个小时）。用户会拷来 `er_200_300.tar.gz`：
   - sha256 为 `d4cc903d2311f6c1be36058549dd2eaa7697a7fb7476e1c64e8397272d26d405`；
   - 在仓库根目录执行 `mkdir -p data/datasets && tar xzf er_200_300.tar.gz -C data/datasets`。
4. **检查**：`<venv>/bin/python -m pytest -q`，应该全部通过，原机器上是 111 个，加上 6 个标记为 slow 的被跳过。

## 4. 运行

**第一轮**：需要 4 张卡，一卡一个任务。fp32 下每个任务约占 7–8 GB 显存，2080 Ti 放不下两个。
```bash
PY=<venv>/bin/python GPUS="0 1 2 3" bash scripts/pilots/as_pilots.sh
```

| 小试 | 相对公共设置的改动 | 检验什么 | 规模（epoch 数 = am_epochs × stages） | 估计耗时 |
|---|---|---|---|---|
| P1_steps100 | 训练和评估都用 100 步，另报 200 步 | 原因 1：100 步够不够 | 15 × 3 | 约 80 分钟 |
| P2_alpha30 | 25 步，逐节点裁剪 α=30，另报 200 步 | 原因 2：只收紧裁剪行不行 | 15 × 6 | 约 75 分钟 |
| P3_lr1e-4_gc1 | 25 步，lr 1e-4，参数梯度范数裁剪 1，另报 200 步 | 原因 4：单靠优化器阻尼行不行 | 15 × 6 | 约 75 分钟 |
| P4_steps200 | 训练和评估都用 200 步 | 原因 1：是否需要约 200 步 | 10 × 3 | 约 85 分钟 |

- **公共设置**：AS，f=0，单点源，解析 corrector；T=0.075，κ_w=1，smooth 映射；逐节点裁剪 150；fp32；lr 3e-4；种子 0。每个 stage 在前 16 张验证图上评测一次（每张图 16 个样本），并存 checkpoint。完整命令见脚本。
- AS 只有 AM，"stage"只是评测间隔，不会重置任何东西。
- 耗时是按原机器的单个 epoch 时间估的。建议先看前 1–2 个 epoch 的实际速度，如果明显更慢，告诉用户。

**第二轮**：在第一轮的胜者上跑，需要 2 张卡。`WINNER` 填胜者相对公共设置的参数：
```bash
PY=<venv>/bin/python GPUS="0 1" ROUND=2 WINNER="--n_train_steps 100 --n_infer_steps 100 200" \
    STAGES=3 AM_EPOCHS=15 bash scripts/pilots/as_pilots.sh
```
- P5_clampout：加 `--pmap clamp --well out --kappa_w 2`，检验原因 3（空集陷阱）。
- P6_T0.25：加 `--temp 0.25`，检验低温下偏向众数的问题。

**输出**：在 `runs/pilots/as_v2/<名字>/`，包括 log.jsonl、epochs.jsonl、每个 stage 的 ckpt、meta.json、summary.json；标准输出在 `runs/pilots/as_v2/<名字>.out`。

## 5. 判据（开跑前已经写定，不要事后改）

用这条命令汇总，它会自动算空集解码基线：
```bash
<venv>/bin/python scripts/pilots/summarize_as_pilots.py runs/pilots/as_v2
```

**稳定**，以下全部满足：
- 每个 stage 的 `clip_node_rate`（被裁剪的节点比例）≤ 0.10，且第 1 个 stage 之后不再上升。崩掉的基线是 0.07 → 0.60。
- `ctrl_energy_stage`（控制能量）相邻 stage 之比 ≤ 2，且绝对值 ≤ 2e4。基线是 1.7e4 → 5.4e5。
- 训练网格上的 E_tgt ≤ 1e3，且和 200 步下的 E_tgt 相差不到 3 倍。基线是 455 对 6。

**学到了东西**，看 200 步下最后一个 stage 的结果，以下全部满足：
- E_tgt < 0：空集约为 +6，目标本身约为 −20；
- raw_size_mean（修复前选中的节点数）在 15–40 之间，不是 0，也不是全选；
- n_distinct（16 个样本里不同解的个数）≥ 8；
- 平均 MIS 高于空集解码基线；
- best-of-16 高于本小试自己 stage 0（u=0）时的值。

**怎么读结果**
- P1 或 P4 稳定，而 P2、P3 不稳：步数是必要条件。P1 不稳而 P4 稳，说明需要约 200 步以上，或者末端加细的非均匀时间网格（代码现在只有均匀网格）。
- P2 稳定：裁剪阈值必须和步长配套，大约 α ≲ 步数。但 α=30 低于目标平衡态的最大梯度（约 87），会让目标有偏，只能当过渡方案。
- P3 稳定：lr 1e-4 加梯度裁剪应该提前采用。
- 稳定但没学到东西（塌到空集，或者不同解少于 8）：跑第二轮。P5 好说明应该换 clampout 能量；P6 好而 T=0.075 不好，说明低温下偏向众数是瓶颈。

## 6. 交回什么

- 每轮结束后，把 `summarize_as_pilots.py` 的完整输出，加上你按第 5 节的判读，用中文写给用户。
- 把 `runs/pilots/as_v2/` 整个目录打包，让用户拷回原机器，原机器上会接着做 AS 和 ASBS 两臂的正式实验。
- 不要改 `sbmis/` 的代码或这两个小试脚本的设定。如果发现必须改，比如显存不够，先问用户，并在汇报里写明改了什么。
- 如果要提交：只提交代码和文档，不提交结果，提交里不带任何 AI 署名，推送到 `mis-asbs-energy-base`。原机器上的仓库也在往这个分支推送，先 `git pull` 再提交，避免冲突。
