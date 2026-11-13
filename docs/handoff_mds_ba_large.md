# 交接：最小支配集（MDS）BA-[800-1200] 的训练和评测（4 张卡）

写于 2026-11-13，分支 `mis-asbs-energy-base`。读者是在另一个集群上接手的 Claude，也给用户核对用。项目整体进度见 `docs/status.md`；本集群的环境（解释器、数据目录、哪些卡能用）见本集群自己的交接文档（sd 是 `docs/handoff_cam_sd_cluster.md`）。

## 0. 先读：用户的硬性规则

1. `AGENTS.md`，以及：
   - **只在 `/usr0/home/shengyuf/yunyi/` 下面写文件**（缓存、临时文件、日志、checkpoint、下载的数据都算）。
   - 不往任何已有环境装包；已有的 conda 环境只读地用。
   - 不碰别人的进程；停自己的进程要先问用户。
   - 不复制 `/usr0/home/shengyuf/CAM4CO` 里的任何文件（里面有明文 API key）。
   - 提交里**绝不**出现 AI 署名（不加 `Co-Authored-By`，不加 "Generated with"）；不提交 `runs/` 和数据。
   - 全程中文；内部代号（实验名、GPU 编号）要用白话说明。
2. 遇到和本文不符的情况（测试不过、数字差得多、显存不够）就停下来问用户，不要自行换方案。分数掉下来不要停，也不要重启。

## 1. 为什么做

把 CAMPrior 推广到一个**高阶**能量的问题上（MIS、MaxCut、最大团的能量都是二次的）。MDS 的能量来自 DiffUCO（表 2，A = 1，B = 1.01）：

E(x) = A·Σᵢ xᵢ + B·Σᵢ (1 − xᵢ)·Πⱼ∈N(i) (1 − xⱼ)

第二项对每个没被支配的点罚 B，是度数阶的多项式。它是多线性的，所以翻转一个点的能量差恰好等于 (1 − 2xᵢ)·∂E/∂xᵢ，CAM 的标签和 RLSA 的先验都能直接用，代价只是能量和梯度要多算一次邻居求和。

CAM 论文没有 MDS 的结果，RLSA 论文也没做 MDS；可比的基线来自 PT-Denoise（arXiv 2609.37323，表 2），数值越小越好：

| 方法 | BA-small（200–300 点） | BA-large（800–1200 点） |
|---|---|---|
| DiffUCO | 28.01 | 104.01 |
| SDDS | 27.93 | 103.90 |
| PT-Denoise | 27.90 | 103.84 |
| Gurobi | 27.89 | 104.01 |
| LTFT | 28.61 | 110.28 |

分工：BA-small 的 3 组训练已在 sa 集群上跑（种子 0，100 个 epoch），评测也在 sa 上做。**本文只管 BA-large**：3 组训练 + 它们的 test 曲线 + 同预算 RLSA 基线。

## 2. 代码（都已推送）

- `cam/envs/mds.py`：能量和梯度（用覆盖次数 c = x + 邻居和 算，不建高阶张量）；解码：连续解码按概率贪心补成支配集，再逐个删掉多余的点（结果是极小支配集），离散解码直接做删点。`--mds_a`、`--mds_b` 默认 1、1.01。
- `cam/datasets/mds.py`：读图同 MaxCut 的 BA 数据。
- `cam/tests/test_mds.py`：5 个测试（能量和暴力枚举对照、翻转差恰好等于梯度式、解码结果是极小支配集、贪心顺序、RLSA 在 n=16 的 BA 图上找到支配数）。
- `cam/scripts/mds/mds_ba_local.sh`：训练 / 测试入口，超参照搬 CAM 的 MaxCut BA 脚本（`num_l 6`、`mini_batch 256`、`lr 5e-4`，其余同 MIS）。数据目录由 `CAM_MDS_DIR` 指定。
- `cam/scripts/mds/mds_rlsa_sweep.sh`：validation 上扫 RLSA 的 (τ0, d)。
- `cam/scripts/mds/mds_curve.sh`：test 曲线队列（第 0、1、2、5、10、20…100 个 epoch 加 best.pt；checkpoint 还没出来就等）。

## 3. 第一步：拉代码、下数据、跑测试

```bash
cd <本集群的仓库目录>            # sd 上是 /usr0/home/shengyuf/yunyi/Schr-dinger-Bridge
git pull origin mis-asbs-energy-base
source cam/scripts/env.sh        # 按本集群交接文档的方式准备环境

# 数据：CAM 的 HF 发布里的 MDS.zip（约 200 MB，解压后 BA_{small,large}_{train,valid,test}）
mkdir -p /usr0/home/shengyuf/yunyi/data/CAM_data && cd /usr0/home/shengyuf/yunyi/data/CAM_data
curl -L -o MDS.zip https://huggingface.co/datasets/shengyuf/CAM_data/resolve/main/MDS.zip
unzip -q MDS.zip                 # 得到 MDS/BA_large_train（4001 项，含 anno 目录）、BA_large_valid（500）、BA_large_test（1000）等
cd -
export CAM_MDS_DIR=/usr0/home/shengyuf/yunyi/data/CAM_data/MDS    # 之后每个命令都要带上

$CAM_PYTHON cam/tests/test_mds.py                                        # 5/5（test_5 显示 skipped）
MDS_TEST_BA=$CAM_MDS_DIR/BA_large_valid $CAM_PYTHON cam/tests/test_mds.py # 5/5
$CAM_PYTHON cam/tests/test_cam_prior.py                                  # 8/8
```

**核对实现和数据**：在 sa 上，BA-large validation 前 64 张图、RLSA τ0 = 0.1、d = 20、50 步 × 20 条链的结果是 **105.469**。在本集群上重跑这一个点（几分钟），差 0.3 以上就停下来问用户：

```bash
mkdir -p runs/queue_cam_mds
GPUS="<一张卡>" PAR=1 SCALES=large TAUS=0.1 DS=20 bash cam/scripts/mds/mds_rlsa_sweep.sh > runs/queue_cam_mds/check_rlsa.log 2>&1
tail -3 runs/queue_cam_mds/check_rlsa.log
```

## 4. 第二步：三张卡训练

协议同 MaxCut BA-large 和 ER：单卡、种子 0、100 个 epoch（温度退火和余弦学习率按 100 个 epoch 走完）、每个 epoch 存 checkpoint。

**先验步长 d 怎么定的**（sa 上 64 张 validation 图、50 步 × 20 条链扫出来的，不看 test）：

| τ0 \ d | 1 | 2 | 5 | 10 | 20 | 40 | 80 |
|---|---|---|---|---|---|---|---|
| 0.01 | | | | | | 106.92 | 107.63 |
| 0.03 | | | | | | 105.92 | 107.28 |
| 0.1 | 114.39 | 110.77 | 107.91 | 106.77 | **105.47** | 105.64 | 107.34 |
| 0.3 | 112.72 | 109.78 | 108.14 | 106.83 | 105.92 | 105.98 | 107.81 |
| 1 | 118.64 | 115.34 | 113.69 | 112.31 | 110.75 | **110.25** | 110.38 |
| 3 | 128.11 | 125.91 | 123.98 | 123.28 | 122.11 | | |

整个网格最好的是 τ0 = 0.1、d = 20（105.47），这也是同预算 RLSA 基线的配置。τ0 = 1 这一行最好的是 d = 40（110.25），比网格最好差近 5（MIS、MaxCut、最大团上从没差这么多），按之前的口径应改用网格最好的 (τ0, d)。所以先验温度 0.1 和 1 两组都跑，**两组都用 d = 20**，只让先验温度一个变量不同（BA-small 也是两组同 d = 10）。三组：

| 实验名 | 内容 |
|---|---|
| `CAM_tau0.1_e100_s0` | 原版 CAM，训练温度 τ0 = 0.1 |
| `CAMPrior_p0.1_d20_e100_s0` | CAMPrior，先验温度 τ0 = 0.1，d = 20 |
| `CAMPrior_p1_d20_e100_s0` | CAMPrior，先验温度 τ0 = 1，d = 20 |

把 `<g1> <g2> <g3>` 换成三张空卡：

```bash
for cfg in "<g1>|CAM_tau0.1_e100_s0|--tau0 0.1" \
           "<g2>|CAMPrior_p0.1_d20_e100_s0|--method CAMPrior --prior_tau0 0.1 --prior_d 20" \
           "<g3>|CAMPrior_p1_d20_e100_s0|--method CAMPrior --prior_tau0 1 --prior_d 20"; do
  IFS="|" read -r g exp args <<< "$cfg"
  CUDA_VISIBLE_DEVICES=$g PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True SCALE=large PROJECT=mds_ba_large EXP=$exp \
    nohup bash cam/scripts/mds/mds_ba_local.sh --epochs 100 --N_anneal 100 --save_every_epoch --seed 0 $args \
    > runs/queue_cam_mds/mds_ba_large__$exp.log 2>&1 < /dev/null &
done
```

训练日志 `runs/queue_cam_mds/mds_ba_large__<实验名>.log`（每个 epoch 一行 `eval/object`，validation 64 张图，越小越好）；checkpoint `runs/cam/mds_ba_large/<实验名>/`。参考：MaxCut BA-large 在 sa 上约 20 分钟一个 epoch（约 1.4 天）；MDS 的能量多一次邻居求和，应该差不多或稍慢。启动后汇报第一个 epoch 的实际时间。

## 5. 第三步：第四张卡评测

**同预算 RLSA 基线**（完整 1000 张 test 图，不依赖训练，先跑）：

```bash
for cfg in "same_t0.1_d20|--num_t 50 --num_k 20 --tau0 0.1 --num_d 20" \
           "same_t1_d20|--num_t 50 --num_k 20 --tau0 1 --num_d 20"; do
  IFS="|" read -r name args <<< "$cfg"
  CUDA_VISIBLE_DEVICES=<g4> SCALE=large MODE=test PROJECT=mds_ba_large_rlsa EXP=$name \
    bash cam/scripts/mds/mds_ba_local.sh --method RLSA $args > runs/queue_cam_mds/rlsa_large_$name.log 2>&1
  echo "$name $(head -1 runs/cam/mds_ba_large_rlsa/$name/test_log.txt)"
done
```

第一个是 validation 上最好的配置（同预算 RLSA 基线），第二个是 CAMPrior τ0 = 1 那组训练前的先验本身。

**test 曲线**（三条队列都在第四张卡上，checkpoint 出来一个评一个；必须传训练时的 flag，CAMPrior 会核对 `.prior.json`）：

```bash
mkdir -p runs/cam/mds_ba_large_test
GPU=<g4> SCALE=large EXP=CAM_tau0.1_e100_s0 nohup bash cam/scripts/mds/mds_curve.sh --tau0 0.1 \
  > runs/cam/mds_ba_large_test/CAM_tau0.1.queue.log 2>&1 &
GPU=<g4> SCALE=large EXP=CAMPrior_p0.1_d20_e100_s0 nohup bash cam/scripts/mds/mds_curve.sh --method CAMPrior --prior_tau0 0.1 --prior_d 20 \
  > runs/cam/mds_ba_large_test/CAMPrior_p0.1.queue.log 2>&1 &
GPU=<g4> SCALE=large EXP=CAMPrior_p1_d20_e100_s0 nohup bash cam/scripts/mds/mds_curve.sh --method CAMPrior --prior_tau0 1 --prior_d 20 \
  > runs/cam/mds_ba_large_test/CAMPrior_p1.queue.log 2>&1 &
```

结果在 `runs/cam/mds_ba_large_test/<实验名>/{epKKK,best}/test_log.txt`，第一行 `Avg. obj` 是平均支配集大小（越小越好，不用取负号）。第 k 个 epoch 指训练 k 个 epoch 之后（`epoch_{k-1}.pt`；k = 0 是训练前的 `epoch_init.pt`）。

## 6. 汇报（中文，给用户看）

- 第一步之后：测试是否全过；RLSA 核对点是多少（对比 105.469）。
- 训练第一天：每个 epoch 的时间；三组 validation 曲线的前几个点；有没有发散或显存问题。
- 全部结束后：
  - 三组的 test 曲线表（第 0、1、2、5、10、20…100 个 epoch）和 best.pt；
  - 两个 RLSA 基线；
  - 和第 1 节 PT-Denoise 表 2 的 BA-large 列对比（注意那些方法的采样预算和我们不同，要写明我们是 50 步 × 20 条链）；
  - 按这些结果更新 `docs/status.md`，单独提交并推送（只提交文档，不提交 `runs/`）。
