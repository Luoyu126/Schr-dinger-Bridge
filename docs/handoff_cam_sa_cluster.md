# 交接：本集群（sa.lti.cs.cmu.edu）上的 CAM / CAMPrior 无监督实验

写于 2026-11-10 14:45 UTC，分支 `mis-asbs-energy-base`。读者是上下文压缩之后，在这个集群上接着干活的 Claude。

上一份交接 `docs/handoff_cam_seeds.md` 是从原机器同步过来时用的。训练设定、判断标准和交回清单都以它为准。本文接着写：本集群上实际跑了什么，现在在跑什么，接下来做什么。项目整体进度见 `docs/status.md`。

## 0. 先读：用户的硬性规则

1. `AGENTS.md`，加上用户在本集群上的要求：
   - **只在 `/home/shengyuf/yunyi/` 下面写文件**，包括环境、数据、缓存、临时文件和日志。不写 `/tmp`，也不写家目录下的缓存。
   - 不往任何已有环境装包。要装就自己建 venv，放在 `/home/shengyuf/yunyi/envs/`。只读地运行已有的解释器可以。
   - 不碰别人的进程。只停自己的进程，用 PID 停，见第 8 节第 3 条。
   - 提交里**绝不**出现 AI 署名：不加 `Co-Authored-By`，也不加 "Generated with"。
   - 提交用仓库主人的身份：`git -c user.name="Yunyi Chen" -c user.email="chenyuny24@mails.tsinghua.edu.cn" commit ...`。本机没有 git 配置。
   - 不要提交 `runs/` 下的结果。
   - 全程用中文交流。内部代号（实验名、GPU 编号等）要用白话说明它指什么。
2. 做事方式：
   - **每一步遇到问题就停下来问用户，不要自行换方案。**
   - 不要多卡并行，不要改任何超参数。不要改 `cam/` 的代码或训练设定，必须改的话先问。
   - 分数掉下来不要停，也不要重启。只有进程报错退出才处理，而且要先告诉用户。
   - 方法或评测上的决定，先查 ASBS 和相关工作怎么做（`AGENTS.md` 里写了文献位置），再按"文献怎么做 → 我建议什么 → 有什么风险"三段来问用户。
3. 用户问"进度如何"，就按组列表格：已训 epoch，validation 最近 5 个 epoch 的均值和最好值，test 和大图最近一个点的分数，预计完成时间。表格后面写一两句读法。

## 1. 现状一览（11-10 14:45 UTC）

| GPU | 任务 | 状态 |
|---|---|---|
| 0–3 | 4 个消融，种子 0 | **已训完 100 个 epoch**，卡空着 |
| 4 | 蒸馏热启动第二阶段 `CAM_tau0.1_distill_e100_s0` | 训练中，46/100 个 epoch，每个 epoch 约 25.6 分钟，预计 **11-11 13:30 UTC** 训完 |
| 5 | 评测队列（一个进程） | 已完成 164/176 个点，剩下的是蒸馏第二阶段第 50–100 个 epoch |
| 6、7、8 | CAM 种子 1、CAMPrior 种子 1、CAM 种子 2 | **已训完 100 个 epoch**，卡空着 |
| 9 | CAMPrior 种子 2 | 11-09 22:13 报错退出，**按用户决定停在 73 个 epoch**，见第 6 节。卡空着 |

除 GPU 4、5 外，其余 8 张卡都空着。用户还没说要用它们做什么，见第 8 节。

## 2. 本集群环境

- 机器：sa.lti.cs.cmu.edu，10 张 RTX 2080 Ti（11 GB），驱动 580.159.03。机器时间是 UTC。
- 仓库：`/home/shengyuf/yunyi/Schrödinger Bridge/Schr-dinger-Bridge`（路径里有空格和 ö，命令里要加引号）。
- 环境：每次开新 shell 先 `source /home/shengyuf/yunyi/envs/cam_env.sh`。它设置的变量：
  - `CAM_CONDA_BIN=/home/shengyuf/yunyi/envs/cam/bin`：Python 3.11 venv，torch 2.11.0+cu128，torch_scatter 2.1.2，torch_sparse 0.6.18，torch_geometric 2.8.0.post1；
  - `CAM_CACHE=/home/shengyuf/yunyi/.cache/cam`：`cam/scripts/env.sh` 会把 TMPDIR、XDG、torch、wandb 等缓存都指到这里；
  - `CAM_ER700_DIR=/home/shengyuf/yunyi/data/CAM_data/data_er`：校验值核对过；
  - `PIP_CACHE_DIR`；
  - `CUDA_DEVICE_ORDER=PCI_BUS_ID`。
- 系统的 `python3`（anaconda 3.9）可以只读地用来跑不依赖 torch 的小脚本，比如评测队列和汇报页面生成器。
- **限制**：
  - **推送**：用 `/home/shengyuf/yunyi/.ssh/id_ed25519_github` 这把 SSH key（11-10 建的，公钥由用户加到 GitHub），仓库的本地 git 配置已经指向它，直接 `git push origin mis-asbs-energy-base` 即可。细节见 `AGENTS.md` 的 "Pushing from the sa cluster"。
  - 本机没有 node，也没有浏览器，发布页面前没法看渲染效果。
  - `/var/log/kern.log` 和 `dmesg` 没有读权限，看不到 GPU 的 Xid 错误码。
  - 沙箱不允许下载并运行外部二进制，比如 NVIDIA 的 compute-sanitizer。

## 3. 已完成的训练与结果

所有设定和 `handoff_cam_seeds.md` 第 2 节一样：100 个 epoch，单卡，官方 ER 超参。启动脚本是 `cam/scripts/mis/seeds_er700.sh`，种子组和消融组（`SET=ablations`）都用它。

- 日志：`runs/queue_cam_er700/mis_er700__<实验名>.log`，搜 `eval/object`，取负号就是 validation；日志里第 N 个 epoch 对应训练 N+1 个 epoch 之后。
- checkpoint：`runs/cam/mis_er700/<实验名>/`。
- test 和大图评测：`runs/cam/mis_er700_test10/<实验名>/epKKK/` 和 `runs/cam/mis_er700_large/<实验名>/epKKK/`。

结果都是平均 MIS 大小，越大越好。test 是 128 张图、50 步 × 20 条链；大图是 16 张 ER-[9000-11000]、200 步 × 20 条链。validation 是 64 张图。

| 实验 | validation 最好（epoch） | test 第 100 个 epoch | 大图第 100 个 epoch | 备注 |
|---|---|---|---|---|
| CAM τ0=0.1 种子 1 | 44.05（29） | 42.52 | 377.8 | 一直在波动 |
| CAM τ0=0.1 种子 2 | 43.86（13） | 42.77 | 381.4 | 一直在波动 |
| CAMPrior 先验 τ0=1 种子 1 | 44.47（74） | 43.84 | 381.6 | 第 90 个 epoch 大图 388.0 |
| CAMPrior 先验 τ0=1 种子 2 | 44.42（53） | 第 70 个：44.12 | 第 70 个：383.7 | **只训到 73 个 epoch** |
| 只用先验参数化（原 warmstart），种子 0 | 44.20（17） | 43.70 | 374.1 | 波动大，第 30 个 epoch test 39.97 |
| refonly，种子 0 | 44.39（91） | 43.97 | 380.8 | 前 30 个 epoch 卡在 34–36，第 40 个 epoch 起追上来 |
| fixed，种子 0 | 44.36（42） | 43.30 | 384.4 | |
| p10，种子 0 | 44.20（19） | 43.79 | 376.9 | |

- 每 10 个 epoch 一个点的完整曲线：用汇报页面的生成器读（第 7 节）。也可以直接读各 `epKKK/test_log.txt` 的第一行，取负号。
- 消融各组的含义见 `handoff_cam_seeds.md` 第 7 节。`CAMPrior_p1_warmstart` 在文档和论文里改叫"只用先验参数化"，目录名不变。
- **还没做**：按 validation 选出的 `best.pt` 在 test 和大图上的评测，见第 8 节第 1 条。

## 4. 正在跑：蒸馏热启动（真正的热启动基线）

**为什么做**：用户指出，原来的 warmstart 消融在策略里一直加着 RLSA 的 logit，用的其实是我们自己的参数化，不是热启动。常规的热启动是：先把 RLSA 蒸馏进 CAM 的网络，再原样训练 CAM。CAMPrior 要超过这个基线，才能说明收益不只是"从 RLSA 出发"。设计、文献和风险写在 `docs/status.md` 第 4 节第 3 条。

**用户定的设定**：
- 蒸馏的是每一步的翻转概率；
- 教师是 RLSA 先验 τ0=1，和 CAMPrior p1 的起点相同；
- 固定蒸馏 20 个 epoch，按 validation 取最好。

**代码**：都已提交，单元测试是 `cam/tests/test_rlsa_distill.py`，5 个测试全部通过。
- `cam/solvers/rlsa_distill.py`：`RLSADistill` 求解器；
- `cam/distill_rlsa.py`：第一阶段的入口，`main.py` 本身没改；
- `cam/scripts/mis/mis_er700_distill.sh`：第一阶段启动器；
- `cam/scripts/mis/distill_warmstart_er700.sh`：两个阶段的总启动器，参数有 `GPUS`、`SEEDS`、`STAGE=both|1|2`、`DRY=1`。

**第一阶段**：`RLSA_distill_p1_s0`，GPU 4，11-09 13:30–18:53，已完成。
- 学生是不带先验的 CAM 网络（CAM 默认初始化），损失是前向 KL(教师‖学生)。
- 蒸馏出来的网络单独采样，validation 在教师附近：第 5–20 个 epoch 平均 38.96，教师是 38.88。最好的是第 2 个 epoch 的 39.94，主要是噪声。
- `best.pt` 就是 `epoch_0001.pt`，即训练 2 个 epoch 后的权重。

**第二阶段**：`CAM_tau0.1_distill_e100_s0`，GPU 4，11-09 18:53 开始，主进程 PID 1747429。
- 命令和 `seeds_er700.sh` 里 CAM τ0=0.1 的命令完全一样，只多了 `--checkpoint .../RLSA_distill_p1_s0/best.pt`。
- `--checkpoint` 只加载权重：优化器、学习率和温度都从头开始。
- 核对过：`epoch_init.pt` 和蒸馏的 best.pt 逐个张量完全相同。所以评测队列里第 0 个点就是蒸馏出来的网络本身。
- 目前结果：

  | epoch | test | 大图 |
  |---|---|---|
  | 0（蒸馏出的网络） | 39.97 | 341.6 |
  | 10 | 43.39 | 368.8 |
  | 20 | 43.30 | 377.6 |
  | 30 | 43.76 | 380.7 |
  | 40 | 43.83 | 376.6 |

  validation 最好是 44.12（第 22 个 epoch），中间有过 CAM 式的大幅波动。

**种子 1、2 还没跑，等用户决定**。我提过两种做法：现在用空着的卡跑，或者等卡空出来再跑。现在 8 张卡都空着。一个种子用一张卡，两个阶段在同一张卡上接着跑，共约 47 小时（第一阶段约 5.5 小时，第二阶段约 41 小时）：
```bash
source /home/shengyuf/yunyi/envs/cam_env.sh
GPUS="<卡1> <卡2>" SEEDS="1 2" DRY=1 bash cam/scripts/mis/distill_warmstart_er700.sh   # 先看命令
GPUS="<卡1> <卡2>" SEEDS="1 2" bash cam/scripts/mis/distill_warmstart_er700.sh
```
开跑后，把 `CAM_tau0.1_distill_e100_s1`、`CAM_tau0.1_distill_e100_s2` 加进评测队列的列表，见第 5 节。

## 5. 评测队列（GPU 5，只能有一个）

- 用户要求所有评测放在**一个**队列里，共用一张卡。不要再开第二个队列，也不要再用 `large_curve_er700.sh` 另起队列，否则会冲突。
- 脚本：`cam/scripts/mis/eval_queue_er700.py`（已提交）。命令、协议和输出目录都和 `large_curve_er700.sh` 相同。
- 进程：PID 1669330，用系统的 python3 运行，日志在 `runs/cam/eval_queue_er700.log`。
- 要评的实验列表在 `runs/cam/eval_queue_er700.exps`，一行一个，每轮都会重新读。**加实验只要往这个文件里加一行，不用重启。**
- 每个实验评第 0、10、…、100 个 epoch，test 和大图各一次。
  - 第 k 个点用 `epoch_{k-1}.pt`，第 0 个点用 `epoch_init.pt`。
  - 同时最多 6 个评测，其中大图最多 4 个。
  - checkpoint 写出满 60 秒才用。
  - 失败的点最多重试 2 次。
  - 别的进程正在评的点会跳过。
- 新的实验组要先在脚本的 `GROUP_FLAGS` 里登记评测参数。蒸馏第二阶段的组 `CAM_tau0.1_distill` 已经登记过。
- CAMPrior 种子 2 已经从列表里删了，它第 0–70 个 epoch 的结果都还在。现在列表里的点全评完后，队列会自己退出（日志最后一行写 all done）。
- 如果要重启（比如队列退出以后又加了新实验）：
  ```bash
  cd "/home/shengyuf/yunyi/Schrödinger Bridge/Schr-dinger-Bridge" && source /home/shengyuf/yunyi/envs/cam_env.sh
  python3 cam/scripts/mis/eval_queue_er700.py --gpu 5 --exps_file runs/cam/eval_queue_er700.exps --dry   # 先看状态
  nohup python3 cam/scripts/mis/eval_queue_er700.py --gpu 5 --exps_file runs/cam/eval_queue_er700.exps >> runs/cam/eval_queue_er700.log 2>&1 < /dev/null &
  ```
- 单个点的耗时：test 约 4 分钟，大图约 9 分钟（并发时）。

## 6. CAMPrior 种子 2 的报错（已结案）

- 11-09 22:13，第 74 个 epoch 训练完，在 validation 第 3 张图上报 `CUDA error: an illegal memory access was encountered`，进程退出。最后一个 checkpoint 是 `epoch_0072.pt`（训完 73 个 epoch）。
- 当时查到的情况：
  - 9 个训练日志、全部评测日志里只出过这一次 CUDA 错误；
  - 同一段代码之前在同样的 validation 图上正常跑过 73 次；
  - GPU 9 的硬件热降频累计约 18 分钟，比多数卡多，但 GPU 2 更多（约 59 分钟）且没崩；
  - 短时间的压力测试没有发现错误；
  - 内核日志读不到，所以没能确定原因。
- **用户决定（11-10）：不再调研，种子 2 就停在 73 个 epoch。** 不要重跑，也不要续训。
- 汇报和最后整理结果时都要注明：种子 2 只训练到 73 个 epoch，原因是 CUDA 报错退出。比较 100 个 epoch 的数字时，它只有第 0–70 个 epoch 的点；它的 `best.pt` 是在 73 个 epoch 里按 validation 选的。
- 排查时留下的文件在 `runs/diag/`（git 忽略），不用管。

## 7. 汇报页面

- 页面：https://claude.ai/artifact/XNy2bdbrWFxmVMHQrRJGuv ，标题"sa 集群 ER-700 结果"。
  - 格式照另一个集群的 "CAM ER-700 Validation Curves"（https://claude.ai/artifact/Wjg6CWcVdFc8mFiFqfvtBY）：Validation、Test、大图泛化三张图，加一张进度表。
  - 页面上用"复训 / 消融"两个标签切换，最下面有蒸馏的进度。
  - 页面明确标注"本集群结果 · sa.lti.cs.cmu.edu"，用户要求一定要标清楚。
- 生成器：`runs/plots/make_sa_report.py`，配合模板 `runs/plots/sa_report_template.html`。两个文件都没提交，在 git 忽略的 `runs/` 下。
  ```bash
  cd "/home/shengyuf/yunyi/Schrödinger Bridge/Schr-dinger-Bridge"
  python3 runs/plots/make_sa_report.py runs/plots/sa_er700_report.html
  ```
  然后用 Artifact 工具发布这个文件。新会话里要带上 `url` 参数，并先 read 一次，这样还是同一个链接。
- **下次更新前要改的地方**：
  - 生成器还不知道种子 2 已经停了，进度表会给出一个没意义的"预计训完"时间。要改成"停在 73（CUDA 报错）"。
  - 训完的实验要显示"已完成"。
  - 蒸馏第二阶段还没画进曲线。用户想看热启动和 CAMPrior 的对比时，把它加进"消融"那个标签或单开一个标签。
  - 页面上最新一版的数据是 11-09 18:41 的。

## 8. 接下来要做的事

1. **`best.pt` 评测**（`handoff_cam_seeds.md` 第 5 节的要求，还没做）。对 7 个已训完的实验和种子 2，各用 `best.pt` 在 test 和大图上评一次，结果放到 `runs/cam/mis_er700_best/<实验名>/`。
   - 卡现在都空着，但开工前先问用户用哪张卡。
   - 命令照评测队列的格式，把 `--checkpoint` 换成 `<run 目录>/best.pt`，`PROJECT` 换成 `mis_er700_best`。组参数见 `eval_queue_er700.py` 的 `GROUP_FLAGS`。
   - 举例（test；大图把 `er_test` 换成 `er_large_test`，`--num_t 50` 换成 `200`，输出子目录换成 `large`）：
     ```bash
     x=CAMPrior_p1_e100_s1
     CUDA_VISIBLE_DEVICES=<卡> PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True MODE=test PROJECT=mis_er700_best EXP=$x \
       bash cam/scripts/mis/mis_er700_cam.sh --method CAMPrior --prior_tau0 1 \
       --checkpoint "$PWD/runs/cam/mis_er700/$x/best.pt" --test_path "$CAM_ER700_DIR/er_test" --num_t 50 --num_k 20 \
       --log_file "$PWD/runs/cam/mis_er700_best/$x/test/test_log.txt"
     ```
   - CAMPrior 的 checkpoint 旁边有 `.prior.json`，里面存着测试时的外层温度比例。测试时的先验参数和训练不一致会直接报错，这是对的，不要加 `--prior_allow_mismatch`。
2. **蒸馏热启动种子 1、2**：等用户决定，命令见第 4 节。
3. **蒸馏第二阶段种子 0 训完后**（约 11-11 13:30 UTC）：等评测队列评完第 100 个 epoch，同样做 `best.pt` 评测，再向用户汇报热启动基线和 CAMPrior 的对比。
4. **打包结果**：按 `handoff_cam_seeds.md` 第 6 节的清单打包，再加上：
   - 蒸馏的两个阶段：`RLSA_distill_p1_s*`、`CAM_tau0.1_distill_e100_s*` 的日志、checkpoint 和评测目录；
   - `runs/cam/eval_queue_er700.log`；
   - `runs/cam/mis_er700_best/`。

   包放在 `/home/shengyuf/yunyi/` 下，由用户拷回原机器，图由原机器统一画。checkpoint 每个实验约 80 MB。
5. **文档**：结果全部出来后，把数字补进 `docs/status.md`。改之前先问用户要不要改、怎么写。
6. **推送**：用户把公钥加到 GitHub 之后，先 `ssh -T` 测试一次，再把本地多出来的提交推到 `mis-asbs-energy-base`。如果远程有新提交，先 `git pull --rebase`。

## 9. 踩过的坑

- **不要原地编辑正在被 bash 执行的脚本**，比如 `mis_er700_cam.sh`、`large_curve_er700.sh`。bash 是边读边执行的，改了会让正在跑的进程出错。要改就新建文件，或者等没有进程在用它再改。
- **停进程用 PID，不要用 `pkill -f <模式>`**。模式会匹配到执行这条命令的 shell 自己，把自己杀掉。要找 PID，用 `ps ... | grep "[g]pu_stress"` 这种写法，避免匹配到 grep 本身。
- 前台 `sleep` 会被拦。要等某件事，用后台命令加 `while ...; do sleep 20; done`，结束时会自动通知。
- 机器上没有中文字体，matplotlib 图里的标签用英文。汇报页面是 HTML，中文没问题。
- 训练进程启动后，`cam/` 下的代码又改过（远程的"测试时先验参数不一致就报错"那次提交），所以老进程报错时，栈里显示的源码行可能和行号对不上，要按进程启动时的代码版本对照。
