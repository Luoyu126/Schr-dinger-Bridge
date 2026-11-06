# 交接文档（2026-11-06 下午更新，分支 `mis-asbs-energy-base`）

> **11-06 下午的决定（详见 `docs/plan.md` §0.7，与本文冲突时以它为准）**：
> - 主指标改为 MIS 领域的优化指标：与 best-known 的差距、best-of-K、平均大小。
> - 我们的设定对应 ASBS 的摊销分子构象生成：低温目标，看能否找到低能解。
> - 摊销主实验将改用更低的固定 T，具体值待小规模试跑决定。
> - 本文 §6 的"以分布距离为主"方案作废。

接手时按这个顺序读：
1. `AGENTS.md`：硬性规则，包括 11-06 新加的"先查文献"一节。
2. 本文。
3. `docs/related_work_eval.md`：ASBS 和相关工作怎么评估，带引用。
4. `docs/plan.md` §0.6：这一轮的全部结果与决定。

本文写的是交接时的状态。正在跑的实验以后会变，以 §4 里的检查命令为准。

---

## 1. 在做什么

我们在 MIS 上用无监督的 ASBS（Adjoint Schrödinger Bridge Sampler）。这一轮要验证的是：**更换参考过程本身有没有用**。这是用户说的"最核心要验证的事情"。
- **我们的方法**：把参考过程从 $f=0$ 换成基于能量二次部分的线性漂移，分两版：B1（原版）和 B1+。
- **对比对象**：
  - $f=0$（B0）；
  - 论文里的其他做法；
  - 只加漂移、不带能量信息的对照组；
  - 把能量信息放进源分布的对照组。

用户已经定下的事：
- **单实例只算初步验证**。单实例层是 `runs/main_ref`：3 张 U1 小图 g0–g2（12、16、14 个节点），加 3 个 100 节点的实例。
  - 用户原话：单实例只有一张图，训完肯定是把它"背下来"了。所以参考过程在单实例里没有优势，是预期之内的，**不要再拿单实例的结果去论证训练效率**。
  - 最终结论要靠**摊销模式**（U7），也就是大规模训练加大规模采样。用户认为，摊销时训练效率和跨图泛化成了瓶颈，能量先验的作用才会体现出来。
- **对比的重点不是 AS 的 MIS 大小**。AS 的 MIS 大，是因为它塌缩到众数上，类似沿能量一路下降。用户把我们的方法定位为"给采样过程提供能量先验"。
- **核心假设是训练效率**（11-06 下午）：有了能量先验，网络不必从零学习能量和输运方向，所以训练更快。要比的是同样的能量梯度调用次数下，谁更快接近最优解。
- **评估用 MIS 领域的优化指标**（11-06 下午）：用户原话"既然在 MIS 上验，就用 MIS 关注的指标，就是最优解"。分布距离不再作为主指标。
- **温度**（11-06 下午）：
  - T=0.5 是当初为 TV 评估在小图上标定的，不适合以最优解为指标的评估。
  - 参照 ASBS 的摊销构象生成（低温、固定、不退火），改用更低的固定 T。
  - T 只进训练目标（$\nabla E/T$），采样时没有温度可调。
- **摊销数据**：先用 ER-[200-300] 跑通，**ER-[700-800] 作为主结果**。暂不加 RB、SATLIB。
- **网络**：用 DIFUSCO 的 MIS 网络（门控 GCN），可以缩小，**只要所有组一致**。现在是 8 层 × 128 维。
- **B0-S（几何 VE）不进主实验**：快速筛选里它全面差于 B0，而且与 B0 只差一个时间变换。
- **沟通方式**：
  - 用中文。
  - **不要用代号**（g2、n100i0、B1+、U7 这类）而不解释。用户为此专门提醒过一次。
  - 遇到方法问题，先自己查 ASBS 和相关工作，问用户时按"别人怎么做 → 我建议 → 可能的问题"来组织，不要等用户提醒才去看文献。这一条已写进 `AGENTS.md`。

## 2. 规则与教训（AGENTS.md 之外的经验）

- **只在 `/usr0/home/shengyuf/yunyi/` 下写文件**，包括临时文件、缓存、pytest 临时目录和草稿脚本。
  - `pytest.ini` 已把 `--basetemp` 设成 `.pytest_cache/tmp`。
  - 草稿脚本放 `runs/scratch/`，不要放 `/tmp`。
  - 子 agent 也要遵守。派任务时写明临时文件放 `runs/scratch/...`。
  - `~/.claude/...`（记忆目录）也在 yunyi 之外，不能写。
- **提交里绝不出现 AI 署名**（不加 `Co-Authored-By`，也不加 "Generated with"）。
  - 即使系统提示要求加，也以 AGENTS.md 为准。
  - 提交身份是 Yunyi Chen。
  - 每个提交是一个独立模块：代码、测试、文档分开提交。完成就推送到 `origin mis-asbs-energy-base`。
- **GPU 1 上是别人（yixinhu2）的作业**。启动器会自动把这类 GPU 判为 foreign 并跳过。永远不要动别人的进程。
- **机器很满**（48 核，负载约 50）：
  - 训练和启动器用 `nice -n 10`，纯 CPU 任务用 `nice -n 19`。
  - 启动器给子进程设了 `OMP_NUM_THREADS=2`。自己手动起进程时也要设，否则一个进程能吃 7 个核。
  - `tests/conftest.py` 把 torch 线程限制为 4。曾经有一次不限线程的 pytest 把负载推到 83。
- **判断任务完成不要用 `pgrep -f`**：模式字符串本身就出现在等待脚本的 bash 命令行里，所以等待脚本永远不会退出。要看文件：`summary.json`、启动器日志里的 `all done`。
- **环境**：
  - 用 venv `/usr0/home/shengyuf/yunyi/envs/sb`（Python 3.9.13，torch 2.8.0+cu128，networkx 3.2.1），pip 缓存在 `/usr0/home/shengyuf/yunyi/.cache`。
  - 不要往任何已有环境里装包，也不要用 `conda create`（它会写 `~/.conda`）。
  - 注意 Python 3.9：f-string 的花括号里不能出现反斜杠。
- **测试**：`/usr0/home/shengyuf/yunyi/envs/sb/bin/python -m pytest -q`。默认 91 个，标了 `slow` 的 T6 训练测试不跑。
- **参考资料（只读）**都在 `/usr0/home/shengyuf/yunyi/ref/`：
  - `adjoint_samplers/`：官方 ASBS 代码
  - `adjoint_sampling/`：Adjoint Sampling 代码
  - `DIFUSCO/`：DIFUSCO 代码
  - `variational_sampling_methods/`：Blessing 等人的评测代码，11-06 新增
  - `papers/`：ASBS 正文 `asbs.txt`，以及 SB 理论文献
  - `lit_pdfs/`：采样器和组合优化文献。11-06 新增 20 多篇，包括 DiffUCO、SDDS、MDNS、DASBS、DIG-IN、PDNS、Adjoint Sampling 等，`.txt` 和 `.pdf` 都有
- **硬件**：8 × RTX 2080 Ti（11 GB）。

## 3. 代码地图

| 文件 | 内容 |
|---|---|
| `sbmis/energies/mis.py` | MIS 能量（惩罚 λ_p、双势阱 κ_w、温度 T），`e_tgt`、`grad_u`，二次部分 `M`、`c` |
| `sbmis/base/linear_energy_base.py` | 全部参考过程的闭式部件（转移、桥采样、lean adjoint、CM 目标、精确积分步）。`build_base(kind, ...)` 的 kind 有：`zero`（B0）、`linear_energy`（原版 B1）、`linear_energy_shift`（B1+）、`linear_energy_shift_rot`（rot）、`iso_ou`（`--iso_match var/m2`）、`vp`、`ve_geometric` |
| `sbmis/sources.py` | 源分布：`gauss` / `delta` / `energy_prior`（$\mu\propto\mathcal N(0,I)e^{-2\gamma E'_{quad}}$）。三种都是高斯（delta 是退化的高斯） |
| `sbmis/asbs.py` | 单实例 ASBS 训练器：AM/CM 交替、replay buffer、AS 模式（`fixed_corrector`）、`kl_ref`、`ctrl_energy_stage`，按调用次数存 checkpoint |
| `sbmis/models/nets.py`、`sbmis/models/difusco_gnn.py` | 单实例用的 GNN/MLP；摊销用的 DIFUSCO 门控 GCN（node-feature-only，自环，可选逐层 activation checkpointing） |
| `sbmis/amortized.py` | 摊销模式：`GraphPool`（填充批、逐图特征分解的 LRU 缓存 + GPU 参数库）、`BatchedLinearBase`、`AmortizedTrainer`、`decode_metrics` |
| `sbmis/datasets.py` | DIFUSCO 式 ER 生成器（n 和边来自同一条 RNG 流）、`GraphSet`、`load_split`、`load_opt` |
| `sbmis/evaluate.py`、`sbmis/reference.py`、`sbmis/graphs.py` | 解码、修复、指标（含 TV 噪声下限、熵、`p_exact_mis`、多样性）；PT-MALA 参考分布；图与 best-known MIS |
| `scripts/train_mis.py` | 单实例训练入口 |
| `scripts/train_amortized.py` | 摊销训练入口（`--dataset` 或 `--single_graph`） |
| `scripts/eval_amortized.py` | 测试集评测：K 个样本 × N 步。中间 checkpoint 可以在训练进行中评测；不同 N 的多次调用会合并进同一个 json |
| `scripts/launch.py` | GPU 队列。只用"所有进程都是我们的"GPU；`--per_gpu` 统计我们在这张卡上的全部进程；`--max_mine 1` 表示本启动器每张卡最多放一个任务 |
| `scripts/summarize.py` | 单实例结果按组汇总，给出相同调用次数下的均值±标准差 |
| `scripts/reference_prior.py` | u=0 诊断：参考过程本身的终点分布 |
| `scripts/mcmc_baseline.py` | MALA / 退火 MALA 对照 |
| `scripts/make_dataset.py` | 生成 ER 数据集，并用 MILP（时限 300 s）+ 局部搜索求 val/test 的 best-known MIS |
| `scripts/bench_amortized_gnn.py` | 网络吞吐和显存基准 |
| `scripts/u1_reference.py` | U1 小图的参考分布（`runs/u1_ref/*.pt`，g0–g2 的任务要用） |
| `scripts/sweeps/main_reference.py` | 单实例主实验的任务表，生成 `runs/main_ref/jobs.tsv` |
| `scripts/sweeps/amortized_er.py` | 摊销实验的任务表，生成 `runs/amortized/<dataset>/jobs_s<seeds>.tsv` |
| `docs/related_work_eval.md` | 11-06 新增：ASBS、Adjoint Sampling、MIS 采样器、自旋系统采样器各自怎么评估，以及对我们的含义 |

`runs/`、`data/datasets/`、`data/cache/` 都被 git 忽略，但里面有不可轻易重建的东西，**不要删**：
- `runs/u1_ref/` 是 U1 参考分布。
- `data/datasets/er_200_300/` 是数据集加最优解，求解耗时数小时。
- `data/cache/mis_er_100_0.15_{0,1,2}.json` 是 n100 实例的最优解，三个都已证明最优，大小均为 24。
- `runs/scratch/lit/` 是 11-06 文献调研的原始报告（`report_*.md`、`critic.md`、`verdicts.json`），论文已复制到 `ref/`。

## 4. 正在跑的实验（11-06 11:52 的状态）

| 启动器（pid） | 任务表 → 日志目录 | 内容 | 状态 / 预计 |
|---|---|---|---|
| 937543 | `runs/main_ref/jobs.tsv` → `runs/queue_main`（`--per_gpu 6`） | 单实例主实验，457 个任务。顺序：g2(72) → n100i0(67) → g0(72) → g1(72) → n100i1(67) → n100i2(67) → g2_lr(20) → n100i0_lr(20) | 74 个 done，没有 failed。g2（14 节点图）72/72 全部完成。n100i0（100 节点实例 0）已启动 40 个、完成 2 个。n100 单个 run 约 6–7 小时。照现在的顺序，n100i1/i2 大约 1.5–2 天后才有结果 |
| 1236443 | `runs/amortized/er_200_300/jobs_s0.tsv` → `runs/queue_amort_er200`（`--per_gpu 7 --max_mine 1`；输出在 `runs/queue_amort_er200.launcher.log`） | 摊销第一批：7 组 × 种子 0，ER-[200-300]，16 个 stage | 11-06 09:25 启动，GPU 0、2–7 各一个。**比原先估计的慢**，见表下说明 |
| 1019102 | `runs/amort_dev/jobs_v2.tsv` | 单图 sanity v2 | 已结束 |

**摊销第一批的实测速度（11:52）**：
- 开头的 CM 30 个 epoch 用了 0.7–0.9 小时。
- 之后一次 AM（30 个 epoch）约 1.4–1.6 小时，一次 CM（30 个 epoch）约 1.4 小时。所以一个完整 stage 约 2.8 小时，不是交接时估计的 1.2–1.7 小时。
- AS 只有 AM，每个 epoch 约 3.5 分钟。
- 按这个速度推算：
  - 第一次验证（stage 2）约 11-06 15:00–16:30。
  - 第一个 checkpoint `ckpt_30720.pt`（4 个 stage）约 11-06 21:00–23:30。
  - 全部完成约 11-08 上午。
- 以 `epochs.jsonl` 的实际进度为准。11:52 时各组只有 stage 0（u=0）的验证结果：平均 MIS 在 25.5–26.1 之间，best-of-16 在 28.3–28.4 之间。

输出位置：
- 单实例：`runs/main_ref/<block>/<group>_s<seed>/`，内含 `meta.json`、`log.jsonl`、`summary.json`、`ckpt_*.pt`。
- 摊销：`runs/amortized/er_200_300/<group>_s<seed>/`，多一个 `epochs.jsonl`，记录每个 epoch 的损失。

启动器怎么工作：
- 每个任务结束时写 `<logdir>/<name>.done` 或 `.failed`；运行中有 `<name>.pid`。
- 重启启动器是安全的：已 done 的任务跳过，pid 还活着的任务不会重复启动。
- 杀掉启动器不会杀掉它的子进程，但这些子进程结束后不会再有人写 `.done`。重启后需要按 `summary.json` 手工补 `.done`，否则下次会重跑。

检查命令：
```bash
cd /usr0/home/shengyuf/yunyi/Schr-dinger-Bridge
tail -n 20 runs/queue_amort_er200.launcher.log; grep -c done runs/queue_main/launcher.log
ls runs/queue_main | grep -c '\.done$'; ls runs/queue_main | grep '\.failed$'
# 摊销每组的进度和最新验证结果（每 2 个 stage 一次）
for d in runs/amortized/er_200_300/*_s0; do echo "$d $(tail -n1 $d/epochs.jsonl | cut -c1-120)"; echo "   $(grep mis_size_mean $d/log.jsonl | tail -n1 | cut -c1-200)"; done
# 单实例：各组 × 预算的紧凑表（runs/scratch/compact.py 不在 git 里；正式汇总用 summarize.py）
/usr0/home/shengyuf/yunyi/envs/sb/bin/python runs/scratch/compact.py g2 tv,p_exact_mis,entropy,kl_ref 153600,307200,614400
/usr0/home/shengyuf/yunyi/envs/sb/bin/python runs/scratch/compact.py n100i0 mis_size_mean,best_of_16,E_tgt_mean,kl_ref 0,92160,184320
/usr0/home/shengyuf/yunyi/envs/sb/bin/python scripts/summarize.py runs/main_ref/g2 --metrics tv kl_ref p_exact_mis entropy --at 153600 307200 614400
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
```

## 5. 到目前为止的结果（细节见 plan.md §0.6.6）

### 5.1 g2：14 节点小图，有参考分布，3 个种子，72 个 run 全部完成

g2 有 14 个节点、20 条边，只有一个最大独立集，大小为 8。$2^{14}$ 种选法可以全部列举，参考分布来自 1600 万个 PT-MALA 样本，自身误差约 0.005。

61.4 万次调用时的结果：

| 组 | TV | 落在最大独立集上的概率 | 熵 | kl_ref |
|---|---|---|---|---|
| **目标分布**（κ_w=2，T=0.5） | 0 | **0.206** | **4.56** | — |
| iso4m2（各向同性 OU，对齐二阶矩） | 0.102±0.021 | 0.203 | 4.51 | 22.6 |
| B1+ ρ=2 | 0.107±0.014 | 0.196 | 4.56 | 17.5 |
| iso4 | 0.113±0.013 | 0.228 | 4.30 | 24.4 |
| B0（f=0） | 0.126±0.014 | 0.169 | 4.86 | 23.8 |
| B1+ ρ=4 | 0.140±0.016 | 0.245 | 4.22 | 17.6 |
| B0+prior4（能量信息放进源分布） | 0.149±0.020 | 0.268 | 4.05 | 19.2 |
| AS | 0.390±0.028 | 0.532 | 2.46 | 22.7 |
| B1+ ρ=16 | 0.538±0.011 | 0.725 | 1.53 | 31.7 |
| MALA（1024 链） | 0.63–0.69 | — | 6.2–6.4 | — |

读法：
- **B1+ 小 ρ、各向同性 OU、B0 三者接近**。B1+ ρ=2 比 B0 好约 1.7 个标准误，和各向同性 OU 打平。B1+ ρ=2 的 MIS 概率和熵几乎正好等于目标值，B0 偏分散。
- **AS 和 B1+ ρ=16 往最大独立集上塌缩**。按"MIS 概率越大越好"排名，它们反而排第一，**说明 MIS 大小类指标奖励的是塌缩**，不能当采样质量用。
- **B1+ 只在 kl_ref 上明显更好**（17.5 对 23.8），但这个优势没有传到终点分布上。

### 5.2 n100 实例 0：100 节点，最优解 24，大多 1 个种子

18.4 万次调用时：

| 组 | 平均 MIS | best-of-16 | E_tgt_mean |
|---|---|---|---|
| AS | 22.27（2 个种子） | 23.98 | 19.8 |
| B0+prior8 / B0+prior4 | 21.10 / 21.08 | 23.3 / 23.4 | 23.1 / 22.9 |
| B1p4+prior4 | 20.97 | 23.25 | 21.0 |
| B0 | 20.92 | 23.42 | 25.3 |
| B1+ ρ=2 / ρ=4 | 20.80 / 20.70 | 23.27 / 23.14 | 23.8 / 23.1 |
| iso4、iso8m2、VP | 20.6–20.7 | 22.8–23.1 | 23.2–24.7 |

- B1+ 在 MIS 上没有超过 B0。差异在单种子的噪声以内。
- B1+ 的连续能量更低，kl_ref 也低得多（132–149 对 205），但解码后的 MIS 没有变大。
- **u=0 时，B1+ 参考过程本身解码出的 MIS 比 B0 小**：19.7–19.9 对 20.3。摊销实验的 stage 0 也一样：ER-[200-300] 上是 25.5–25.8 对 25.95。连续能量低很多（367–651 对 2104），但主要是因为终点方差变小了。
- n100 没有目标分布的参考值，所以**判断不了 AS 的 22.3 是采得准还是偏**。

### 5.3 单图摊销 sanity（n100 实例 0，摊销流程）

摊销流程里，每次梯度调用对应的网络更新比单实例少 16 倍。7.68 万次调用时，两个版本 v1 / v2 的平均 MIS：

| 组 | 平均 MIS |
|---|---|
| B1+ ρ=8 | 19.70 / 19.47 |
| B1+ ρ=4 | 19.11 / 19.05 |
| B0 | 19.19 / 19.00 |

- B0 在 u=0 时是 20.42，三组都还在 u=0 以下的回落段。
- 这和用户"更新越少、先验越有用"的说法方向一致，但只有 1 个种子，不能当结论。

### 5.4 摊销流程检查（交接前已完成）

- 单图 n=100 上，摊销流程与单实例流程的轨迹形状和 stage 0 数值都一致。回升更慢，因为每 epoch 只有 50×32 次网络更新，单实例是 100×512。两种模式只能按 stage 比，不能按调用次数比。
- ER-[200-300] 的 dry run 覆盖了第一批的全部 7 组、缓存淘汰和 eval 脚本，都正常。显存峰值约 2.7 GB。

## 6. 评估方式：文献调研结论与当前建议（11-06）

> **已被 11-06 下午的决定取代**：主指标改为优化指标，见 `docs/plan.md` §0.7。下面的文献结论仍然有效；"向用户提的主方案"（分布距离为主）作废，PT-MALA 参考样本不再是前置条件。

完整内容和引用见 `docs/related_work_eval.md`。要点：
- **ASBS 没有 MIS 或任何离散实验**。指标全是与真值样本的分布距离：样本 W2、能量直方图 W2、扭转角 KL、构象覆盖率。
  - 没有 best-of-K，也不用重要性权重。
  - 所有任务 $f=0$，能量信息放在源分布里（谐振子先验）。审稿人很可能问"为什么不像 ASBS 那样放进源分布"，所以摊销实验里**要赢过 B0+prior4 才有说服力**。
- **MIS 领域的论文只报优化指标**：平均大小、best-of-K、与 KaMIS 的差距，温度都退火到 0。SDDS 明确指出这类指标偏向模式寻找的方法。
- **采样质量的评估只出现在 Ising 和 Potts 上**：重要性权重 + ESS、小系统精确枚举、和长时间 MCMC 比统计量。
  - ESS 看不出模式塌缩：一个塌缩的模型 ESS 仍有 0.983。
  - 非 memoryless 的参考过程，相关论文干脆不报 ESS。
- **我曾向用户提过用重要性权重评估 n100，现在改成辅助手段**：
  - 权重公式：$\log w=-E(x_1)/T-\log Q_1(x_1)-\sum_k\log[p_\theta(x_{k+1}|x_k)/q(x_{k+1}|x_k)]$。目标路径测度是 $\nu(x_1)Q(\text{路径}|x_1)$。
  - 每一步的对数比值有闭式，因为带控制的一步就是参考转移加均值偏移、方差相同（`linear_energy_base.py:205`）。所有源都是高斯，$Q_1$ 也有闭式。
  - 只能用来做自归一化的目标期望估计，交叉核对用。**ESS 不能拿来跨方法排名**，原因有三：
    1. 按 ASBS 的定义，只有 AS（Dirac 源）是 memoryless，我们的 B0 也不是。非 memoryless 的参考过程即使训到最优，权重也有方差，ESS 会偏向 AS。
    2. ESS 看不出塌缩。
    3. 用 `--ve_euler` 时，$Q_1$ 要按离散步的方差算。
  - 不能用一个共同的 Dirac 参考来比所有方法：高斯源的采样器在 $x_0=0$ 上概率为零，权重没有定义。
- **向用户提的主方案**（用户还没答复）：
  - 主指标按 ASBS 的惯例，看与参考样本的分布距离。
    - 14 节点小图：已有 TV，再补能量 W2。
    - 100 节点图：用 PT-MALA 生成参考样本，比能量 W2、解码 MIS 大小的分布，以及各个最大独立集上的概率质量（仿 DIG-IN 的 TVD_mode）。
    - 摊销：全部测试图报优化指标；少量建了参考的测试图报分布指标，给平均值和最差的一张。
  - 优化指标单独报。所有方法用同一个 T 和同一个解码器，并报修复前的原始可行率。

## 7. 下一步（按优先级）

0. **（11-06 下午新增，最优先）选定低温 T**，见 `docs/plan.md` §0.7：
   - 在 ER-[200-300] 上用 B0 和 B1+ 小规模试跑 T∈{0.25, 0.1}，必要时更低。
   - 看三件事：训练是否稳定，未训练时与最优的差距，训练若干 stage 后与最优的差距。
   - 定下 T 后，用新温度重跑摊销实验的全部 7 组。T=0.5 的第一批不停，作为流程检查。
   - 还要决定 ρ 是否随 1/T 缩放。现在参考过程与 T 无关。
1. **等用户答复**：
   1. ~~要不要建 100 节点图的 PT-MALA 参考样本~~：11-06 下午主指标改为优化指标，这一项不再需要。
      - `sbmis/reference.py` 的 `pt_mala` 可以直接用，只是 100 维下不能做 $2^n$ 直方图，要改成保存样本或统计量。
      - 先在 14 节点图上验证整套指标：重要性加权后的 MIS 概率应该回到约 0.206。
   2. **要不要把 n100i1、n100i2 提到 g0、g1 前面**。用户关心结论能否推广到多个实例。如果要：
      1. 按新顺序重写 `runs/main_ref/jobs.tsv`，可以改 `scripts/sweeps/main_reference.py` 的 block 顺序后重新生成。
      2. 只杀主启动器（pid 937543），在同一个 `--logdir runs/queue_main` 用相同参数重启。
      3. 旧启动器留下的子进程结束后，按 `summary.json` 手工补 `.done`。
2. **摊销第一批的测试集评测**：每个 checkpoint 出现后通过启动器跑，不要手动挑 GPU。
   ```bash
   P=/usr0/home/shengyuf/yunyi/envs/sb/bin/python; R=runs/amortized/er_200_300
   for g in B0 AS VP B0+prior4 iso4m2 B1p_rho2 B1p_rho4; do
     printf "eval__%s__30720\t$P scripts/eval_amortized.py $R/%s_s0 --ckpt ckpt_30720.pt --split test --k 1 4 16 --n_infer_steps 25\n" $g $g
   done > $R/eval_jobs_30720.tsv
   nohup nice -n 10 $P scripts/launch.py $R/eval_jobs_30720.tsv --logdir runs/queue_amort_eval --gpus 0 2 3 4 5 6 7 --per_gpu 8 --max_mine 1 --min_free_mb 3000 > runs/queue_amort_eval.launcher.log 2>&1 &
   ```
   - 估计成本（有争用时）：K=16、N=25 约 5.4 s/图，128 张测试图约 12 分钟。
   - 最终 checkpoint 建议跑两次，结果会合并进同一个 json：
     - `--k 1 4 16 64 --n_infer_steps 25`，约 45 分钟
     - `--k 1 4 16 --n_infer_steps 100`，约 45 分钟
   - 同一次调用里的 K 子集来自同一批 K_max 个样本；K_max 不同，样本也就不同。
   - 读结果时用预先定好的判据：

     | 结果 | 结论 |
     |---|---|
     | B1+ > iso4m2 ≈ B0 | 能量信息有用 |
     | B1+ ≈ iso4m2 > B0 | 好处来自漂移和方差 |
     | 三者差不多 | 先验在摊销里也没用 |

     另外 B1+ 还要和 B0+prior4 比。前几个 stage 最关键，要等种子 1、2。
3. **摊销的种子 1、2**：
   ```bash
   P=/usr0/home/shengyuf/yunyi/envs/sb/bin/python
   $P scripts/sweeps/amortized_er.py --dataset er_200_300 --stages 16 --seeds 1 2
   ```
   - 生成 `jobs_s1_2.tsv`，用新的 logdir（如 `runs/queue_amort_er200_s12`），参数与第一批相同。单种子分不出 0.3 量级的差异。
   - 现在 7 张卡各有一个摊销任务，每张卡已用 7.3–9.5 GB。再加任务要先看显存，单个摊销任务峰值约 2.7 GB。
4. **主实验各 block 跑完后**：用 `summarize.py` 出正式表，更新 plan.md 的 §0.6.6。g2 已经全部完成，可以先出 g2 的正式表。
5. **ER-[700-800]**：根据第一批的收敛速度（多少个 stage 后验证集指标稳定），决定直接在 700-800 上训练，还是用 200-300 训练、在 700-800 上测试。
   - 生成数据：
     ```bash
     nice -n 19 $P scripts/make_dataset.py --name er_700_800 --n_min 700 --n_max 800 --workers 8 --time_limit 300
     ```
     这种规模下 MILP 基本证明不了最优，以局部搜索为主。
   - 显存：n≈750 时每批节点数约是现在的 3 倍。先用 `bench_amortized_gnn.py --n 750 --configs 8x128 --amp` 测，可能要把 `--train_batch` 和 `--sim_chunk` 调小。
   - 按目前的速度（一个 stage 约 2.8 小时），700-800 上 16 个 stage 可能要好几天，要提前算好时间。
   - 可选：在 yunyi 下编译 KaMIS 作为参考求解器。
6. **未解问题**：
   - AS 为什么塌缩。它在理论上是无偏的：Dirac 源天然满足 memoryless 条件。猜测是控制量大，AM 回放训练逐步偏向少数模式，还没验证。
   - B1+ 的 kl_ref 更低，为什么没有转化为终点分布或 MIS 指标上的优势。
   - 为什么 u=0 时 B1+ 参考过程解码出的 MIS 比 B0 小，而连续能量低很多。

## 8. 坑

- 摊销训练时，`--cache_size` 必须大于 buffer 覆盖的图数（约 1250，默认 3072 够用）。否则每个训练批都要重新做特征分解：pilot 里缓存 256 时，iso4m2 一个 stage 跑了 3 小时。
- `iso_ou` 只能配 `gauss` 源（代码里有断言）。rot 在摊销模式里不能学：每张图一个随机旋转，最优 u、h 就不是图的函数。
- 验证用的 `--eval_k 256` 很慢；训练中验证保持 `--eval_k 16 --n_infer_steps 25`。
- 摊销单任务显存峰值约 2.7 GB，单实例任务每个 0.5–1 GB。主启动器的 `--min_free_mb 1500` 会挡住把卡塞满的放置，但峰值叠加时仍可能 OOM。出现 `.failed` 先看日志尾部。
- 修改正在运行的脚本不会影响已经启动的进程（代码在启动时已载入）。
- 现有的 TV 是在符号模式上算的，也就是各个象限的概率质量，看不到同一象限内部的形状误差。
