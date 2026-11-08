# 交接：CAM 与 CAMPrior 在 ER-[700-800] 上补两个种子（在另一个集群上运行）

写于 2026-11-08，分支 `mis-asbs-energy-base`。读者是在另一个集群上接手这项任务的 Claude。
项目整体进度见 `docs/status.md`。

## 0. 先读

1. `AGENTS.md`：仓库的硬性规则。在新集群上有两点要变通：
   - 文件里写的路径 `/usr0/home/shengyuf/yunyi/` 是原机器的。在新集群上，**只在用户指定的工作目录里写文件**，包括缓存、临时文件、数据和环境。不要写 `/tmp`，也不要写家目录下的缓存。
   - 其余规则照旧：
     - 不往任何已有环境里装包，自己建 venv；
     - 不碰别人的进程；
     - 提交里绝不出现 AI 署名（不加 `Co-Authored-By`，也不加 "Generated with"）；
     - 和用户用中文交流，不要不加解释地使用内部代号。
2. 本文。
3. 背景（需要时再读）：`docs/status.md`，包括 CAM 臂的设计、seed 0 的结果和未解决的问题。

## 1. 任务是什么

项目在 MIS（最大独立集）上研究无监督神经求解器，离散臂基于 CAM（Combinatorial Adjoint Matching，`cam/` 是它的官方代码，作者同意我们使用）。我们的方法 **CAMPrior** 把 CAM 里不含能量的参考过程（均匀随机翻转）换成 RLSA 的能量驱动翻转过程，网络只学相对它的残差。

原机器上，每组超参数各跑了 1 个种子（seed 0）。五组里，按 validation 选出两组最好的：
- **CAM τ0=0.1**
- **CAMPrior 先验 τ0=1.0**

单个种子下，CAMPrior 更好、也更稳。**这次的任务**：把这两组各补 seed 1 和 seed 2，**训练设定和 seed 0 完全一样**，用来检验结论是否稳健，并给出均值和方差。

一共 4 个训练任务，每个任务占一张卡。另外用一张卡在训练过程中评测 test 集和大图泛化。集群上剩下的卡**先不要用**，留给后面的消融实验，消融的说明之后另行给出。

## 2. 训练设定（必须和 seed 0 一致，不要改）

| 项目 | 设定 |
|---|---|
| 数据 | CAM 官方 ER-[700-800]：训练 16284 张图，validation 用 `er_valid` 里的 64 张（`--valid_sample 64`），test 是 `er_test` 的 128 张，大图泛化是 `er_large_test` 的 16 张 ER-[9000-11000] |
| 网络 | EncodeProcessDecode，8 层 × 64 维，无时间输入，约 18 万参数 |
| 超参数 | CAM 官方 ER 设定（`cam/scripts/mis/mis_er700_cam.sh`）：lr 5e-4，AdamW，weight decay 1e-4，cosine 学习率衰减，每个 epoch 1000 张训练图，batch 1 张图，mini-batch 512，训练和推理都是 50 步 × 20 条链 |
| 训练长度 | **100 epoch**。温度退火（`--N_anneal 100`）和学习率衰减都按 100 epoch 走完 |
| 温度 | 训练温度 τ 从 τ0 线性退火到 0。CAM 组 τ0=0.1。CAMPrior 组训练 τ0 用默认的 0.1，先验温度 τ0=1.0：内层沿采样步按 RLSA 方式降温，外层和训练温度成比例 |
| 卡数 | **每个任务 1 张卡**，不要多卡并行，原因见下 |
| 评测 | 每个 epoch 结束在 64 张 validation 图上评测一次（写在训练日志里），每个 epoch 存一个 checkpoint，训练前还会存一份未训练权重 `epoch_init.pt` |

**为什么不能多卡**：CAM 代码里每个 epoch 固定 1000 张图，优化步数按卡数平分（`cam/solvers/meta_solver.py:180`），多卡时每一步对各卡的梯度取平均。8 张卡等于 batch 变成 8、每个 epoch 只更新 125 次，这是另一个训练设定，结果不能和 seed 0 放在一起比。

完整命令（脚本会自动生成，这里只是让你看清楚）：
```
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PROJECT=mis_er700 EXP=<组名>_e100_s<种子> \
  bash cam/scripts/mis/mis_er700_cam.sh --epochs 100 --N_anneal 100 --save_every_epoch --seed <种子> <组参数>
# CAM 组参数：--tau0 0.1；CAMPrior 组参数：--method CAMPrior --prior_tau0 1
```

## 3. 准备

1. **停掉这个集群上之前的任务**，这是用户的要求，比如 AS 小试。停之前记下每个任务的状态，比如跑到哪个 epoch、结果在哪个目录，告诉用户。只停我们自己的进程。

2. **代码**
   ```bash
   git pull   # 或者重新 clone git@github.com:Luoyu126/Schr-dinger-Bridge.git
   git checkout mis-asbs-energy-base
   ```

3. **环境**：建一个 **Python 3.11** 的新 venv，放在工作目录下。原机器的版本见 `cam/scripts/requirements_env.txt`。关键的几个包：
   ```bash
   python3.11 -m venv <工作目录>/envs/cam
   export PIP_CACHE_DIR=<工作目录>/.cache/pip
   <venv>/bin/pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
   <venv>/bin/pip install torch_scatter==2.1.2 torch_sparse==0.6.18 -f https://data.pyg.org/whl/torch-2.11.0+cu128.html
   <venv>/bin/pip install torch-geometric==2.8.0.post1 accelerate==1.14.0 networkx==2.7.1 numpy==2.4.6 \
       scipy==1.17.1 scikit-learn==1.9.0 tqdm wandb Cython
   ```
   - `networkx` 必须是 2.7.x，因为数据是 `read_gpickle` 格式，3.x 里已经没有这个函数了。
   - 如果某个精确版本装不上，可以用相近的版本，但要告诉用户换了什么。
   - wandb 只是为了 import，`env.sh` 已经把它关掉了。

4. **数据**：从 huggingface 下载 CAM 官方数据 `MIS.zip`，约 5.9 GB：
   ```
   https://huggingface.co/datasets/shengyuf/CAM_data/resolve/main/MIS.zip
   ```
   - 解压到工作目录下，找到 `er_train`、`er_valid`、`er_test`、`er_large_test` 四个目录所在的父目录，记为 `<ER数据目录>`。
   - 如果这个集群访问不了 huggingface，告诉用户，由用户从原机器 `/data/shengyuf/MIS/data_er` 拷过来。
   - **核对校验值**，必须和原机器一致：
     ```bash
     cd <ER数据目录>; export LC_ALL=C
     for s in er_valid er_test er_large_test er_train; do (cd $s && echo "$s $(ls | grep '\.gpickle$' | wc -l) $(ls | grep '\.gpickle$' | sort | xargs md5sum | md5sum | cut -c1-32)"); done
     ```
     原机器的结果：

     | 目录 | 图数 | 校验值 |
     |---|---|---|
     | er_valid | 100 | deb753de7cede1bbbeac7d002d221f25 |
     | er_test | 128 | d4113ca08ada822ed269316f69bb3847 |
     | er_large_test | 16 | b8925aa7c29284eaf429411b637c7df7 |
     | er_train | 16284 | 9a5cb796efdeb05f446dc73d999a875b |

     `er_train/anno/`（KaMIS 标注）训练时用不到，有没有都行。

5. **环境变量**：每次开新 shell 都要设，或者写进一个只在工作目录里的脚本：
   ```bash
   export CAM_CONDA_BIN=<venv>/bin          # 里面要有 python 和 accelerate
   export CAM_CACHE=<工作目录>/.cache/cam     # 所有缓存和临时文件都放这里
   export CAM_ER700_DIR=<ER数据目录>
   ```

6. **检查**：
   ```bash
   $CAM_CONDA_BIN/python cam/tests/test_cam_prior.py      # CAMPrior 的单元测试，CPU 上跑，应该全部通过
   GPUS="0 1 2 3" SEEDS="1 2" DRY=1 bash cam/scripts/mis/seeds_er700.sh   # 只打印要跑的 4 条命令
   ```

## 4. 运行

1. **训练**，4 张卡，一卡一个任务，后台运行：
   ```bash
   GPUS="0 1 2 3" SEEDS="1 2" bash cam/scripts/mis/seeds_er700.sh
   ```
   - 日志写在 `runs/queue_cam_er700/mis_er700__<组名>_e100_s<种子>.log`；
   - checkpoint 写在 `runs/cam/mis_er700/<组名>_e100_s<种子>/`。
   - 如果 0–3 号卡被别人占用，换成空闲的卡号。`env.sh` 会拒绝使用正在跑别人进程的卡。

2. **评测队列**，用第 5 张卡，例如 4 号卡。两个队列放在同一张卡上，原机器也是这么做的：
   ```bash
   E="CAM_tau0.1_e100_s1 CAMPrior_p1_e100_s1 CAM_tau0.1_e100_s2 CAMPrior_p1_e100_s2"
   mkdir -p runs/cam/mis_er700_test10 runs/cam/mis_er700_large
   EXPS="$E" SPLIT=test GPU=4 PAR=2 nohup bash cam/scripts/mis/large_curve_er700.sh > runs/cam/mis_er700_test10/queue_s12.log 2>&1 &
   EXPS="$E" GPU=4 PAR=4 nohup bash cam/scripts/mis/large_curve_er700.sh > runs/cam/mis_er700_large/queue_s12.log 2>&1 &
   ```
   - 队列会在训练写出 checkpoint 后，依次评测第 0、10、20、…、100 个 epoch。第 k 个点用的是 `epoch_{k-1}.pt`，第 0 个点用的是未训练权重 `epoch_init.pt`。
   - test 集用 50 步 × 20 条链；大图泛化按 CAM 论文协议用 200 步 × 20 条链。
   - 每个检查点：test 约 2 分钟，大图约 8 分钟。

3. **时间**：原机器上（2080 Ti），每个 epoch 约 28 分钟，100 个 epoch 约 **2 天**。单个任务显存峰值约 10.2 GB。
   - 开跑后看一下前 1–2 个 epoch 的实际速度。如果明显慢于 28 分钟，告诉用户。
   - 如果出现显存不足，先检查卡上有没有别的进程，**不要**改 mini-batch 或其他超参数，先问用户。

## 5. 怎么判断跑得对不对

**环境检查**：第 0 个点，也就是未训练权重，在 test 集上的结果。
- CAMPrior 未训练时输出层初始化为 0，所以它就是 RLSA 先验本身，和种子关系很小。原机器上 seed 0 是 **38.56**。新种子如果偏离超过约 0.5，说明环境或数据可能有问题：先停下来告诉用户。
- CAM 未训练时是随机网络，seed 0 是 32.62。不同种子会有一些差别，偏离不大就行。

**seed 0 的参考曲线**：数值是平均 MIS 大小，越大越好。validation 是 64 张图，test 是 128 张，大图是 16 张。

| epoch | CAM τ0=0.1 val | test | 大图 | CAMPrior τ0=1.0 val | test | 大图 |
|---|---|---|---|---|---|---|
| 0（未训练） | 32.44 | 32.62 | 268.4 | 38.88 | 38.56 | 338.4 |
| 1 | 37.77 | | | 43.02 | | |
| 10 | 34.50 | 34.53 | 284.7 | 44.03 | 43.77 | 376.4 |
| 20 | 43.02 | 42.87 | 366.9 | 44.11 | 43.97 | 373.7 |
| 30 | 43.81 | 43.65 | 378.0 | 43.95 | 43.84 | 376.1 |
| 40 | 39.14 | 38.59 | 365.4 | 44.27 | 44.04 | 382.4 |
| 50 | 38.31 | 38.29 | 361.1 | 44.12 | 44.03 | 383.2 |
| 60 | 43.08 | 43.03 | 377.8 | 44.03 | 43.98 | 381.1 |
| 70 | 43.86 | 43.68 | 378.1 | 44.17 | 44.07 | 382.9 |
| 80 | 43.06 | 43.12 | 380.5 | 44.23 | 44.20 | 385.5 |

- seed 0 写这份文档时还在训练（约第 88 个 epoch），90 和 100 的值之后由原机器补上。
- 参考线：
  - test 集上，KaMIS 44.87，CAM 论文 44.16（训练 1000 epoch），同预算 RLSA 39.94；
  - 大图上，KaMIS 381.31，CAM 论文 384.38，同预算 RLSA 339.69。
- **seed 0 的波动**：
  - CAM 多次掉下来，第 11 个 epoch validation 跌到 26.48，第 40–50 个 epoch 在 38–39；
  - CAMPrior 也有一次，第 14 个 epoch 跌到 39.00，之后恢复。
  - 这种波动正是要用新种子来看的东西。**不要因为分数掉了就停掉或重启任务**，照常跑完 100 个 epoch。只有进程报错退出才需要处理：先告诉用户，不要自己改设定重跑。

## 6. 交回什么

1. **进度汇报**：用户问起时，从训练日志里读每个任务已完成的 epoch 和最近的 validation 分数，日志里搜 `eval/object`，分数取负号。再读评测队列已完成的点，用中文告诉用户。
2. **全部跑完后**，把下面这些打包，让用户拷回原机器。图由原机器统一画。
   ```
   runs/queue_cam_er700/mis_er700__*_s1.log  runs/queue_cam_er700/mis_er700__*_s2.log
   runs/cam/mis_er700/*_s1/  runs/cam/mis_er700/*_s2/          # checkpoint，每个任务约 80 MB
   runs/cam/mis_er700_test10/*_s1/  runs/cam/mis_er700_test10/*_s2/
   runs/cam/mis_er700_large/*_s1/   runs/cam/mis_er700_large/*_s2/
   runs/cam/mis_er700_test10/queue_s12.log  runs/cam/mis_er700_large/queue_s12.log
   ```
3. 不要改 `cam/` 的代码或训练设定。如果发现必须改，先问用户，并在汇报里写明改了什么、为什么改。
4. 如果要提交：只提交代码和文档，不提交结果（`runs/` 已被 git 忽略），提交里不带任何 AI 署名，推送到 `mis-asbs-energy-base`。原机器上的仓库也在往这个分支推送，先 `git pull` 再提交。
