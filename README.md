# Diffusion Policy · Push-T 视觉仿真复现

复现 RSS 2023 论文 **Diffusion Policy: Visuomotor Policy Learning via Action Diffusion** 的 Push-T 视觉策略：加载作者发布的 CNN + EMA 权重，在官方测试环境中执行闭环控制，统计逐回合得分并与官方日志比较。

## 复现结果

| 指标 | 官方记录 | WSL 复现 | Docker 复现 |
| --- | ---: | ---: | ---: |
| 50 回合平均得分 | 0.883584 | 0.918845 | 0.918845 |
| 得分差值 | — | +0.035261 | +0.035261 |
| 评测耗时 | — | 96.63 s | 92.38 s |

峰值 GPU 分配内存均为 1136.01 MiB。完整报告与逐回合数据见 [WSL 评测报告](records/evaluation/report.md)和 [Docker 评测报告](records/docker/evaluation/report.md)。

相同配置独立运行两次，平均得分均为 0.918845，逐回合最大绝对差为 2.22e-16。

Docker 在关闭网络、只挂载结果目录的条件下完成评测，与 WSL 结果的逐回合最大绝对差为 1.11e-16。4 段测试视频均已验证可解码。

## Docker 一键评测

镜像内置官方 EMA 权重、仿真环境和参考记录。导入交付文件后，在 Linux / WSL 中运行：

```bash
docker load -i diffusion-policy-pusht-epoch0500.tar.gz
mkdir -p results
docker run --gpus all --shm-size=1g --network none \
  --mount "type=bind,source=$PWD/results,target=/output" \
  laobiexl/diffusion-policy-reproduction:pusht-epoch0500
```

完成后查看 `results/report/report.md`，逐回合得分在 `results/report/episodes.csv`，视频在 `results/report/videos/`。Windows PowerShell 命令、源码构建和镜像导出见 [Docker 运行说明](docs/docker.md)。

## 实验设置

| 项目 | 配置 |
| --- | --- |
| 策略 | DiffusionPolicy-C，ResNet18 + 条件一维 U-Net |
| 权重 | train_0 / epoch=0500，EMA 参数 |
| 输入 | 最近 2 帧 96×96 图像、推动器二维位置 |
| 预测 / 执行时域 | 16 / 8 |
| 扩散采样 | DDPM，100 步，squaredcos_cap_v2 |
| 测试环境 | 50 个初始条件，每回合最多 300 步 |
| 测试种子 | 检查点内配置：4300000–4300049 |
| 官方参考得分 | 0.8835836184400558 |

策略每次生成一段动作，执行其中 8 步后重新读取观测并规划。评测指标为每回合最大归一化目标覆盖得分，再对 50 个测试回合求平均。

## 环境安装

已验证：Windows 11 + WSL2 Ubuntu 24.04、Python 3.10.20、PyTorch 2.11.0+cu128、RTX 5070 Laptop 8 GB。

准备 Linux / WSL、Python 3.10 和 C/C++ 编译器，执行：

```bash
git clone --recurse-submodules git@github.com:LAOBIEXL/diffusion-policy-reproduction.git
cd diffusion-policy-reproduction
python3.10 -m venv .venv
PUSHT_PYTHON="$PWD/.venv/bin/python" bash scripts/install_wsl.sh
source .venv/bin/activate
export SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy WANDB_MODE=disabled
```

本机独立环境位于 `/home/laobiexl/.local/share/diffusion-policy-reproduction/env`。网络较慢时，可为安装命令设置 `PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple`。完整依赖快照见 `records/environment/requirements-wsl.lock.txt`。

## 下载与评测

```bash
python scripts/fetch_assets.py config checkpoint train_log metrics
python scripts/audit_reference.py
python scripts/evaluate_checkpoint.py \
  --checkpoint data/checkpoints/official_train_0_epoch0500.ckpt \
  --output runs/eval-official-epoch0500
python scripts/compare_results.py --evaluation runs/eval-official-epoch0500/eval_log.json
```

下载程序支持断点续传，检查文件大小和检查点 ZIP CRC，并记录 SHA256。评测程序从权重读取模型、归一化参数和环境配置，使用 EMA 策略完成全部回合。

输出内容：

- `eval_log.json`：逐回合得分、平均分、耗时和显存。
- `checkpoint_config.yaml`、`evaluation_config.yaml`：实际使用的配置。
- `media/`：闭环控制视频。
- `records/evaluation/`：与官方结果的比较、逐回合 CSV 和配置记录。

## 仓库结构

| 路径 | 内容 |
| --- | --- |
| `third_party/diffusion_policy/` | 固定 commit 的官方实现及 MIT 许可证 |
| `scripts/` | 安装、下载、评测、结果比较和计算验证 |
| `references/` | 论文来源、引用信息和公开配置 |
| `records/` | 环境快照、数据校验、实验配置和结果 |
| `docs/reproduction_protocol.md` | 实验流程与指标定义 |
| `docs/experiment_log.md` | 实际执行记录 |
| `Dockerfile`、`docker/` | 镜像构建和固定依赖 |
| `docs/docker.md` | 镜像导入、评测、构建与导出命令 |
| `data/`、`runs/` | 本地数据、权重、运行日志和视频 |

## 参考资料

- [RSS 2023 论文](https://roboticsproceedings.org/rss19/p026.html)，DOI：`10.15607/RSS.2023.XIX.026`。
- [项目主页](https://diffusion-policy.cs.columbia.edu/)，RSS 版本对应 `arXiv:2303.04137v4`。
- [官方代码](https://github.com/real-stanford/diffusion_policy)。
- [官方 Push-T visual CNN 实验](https://diffusion-policy.cs.columbia.edu/data/experiments/image/pusht/diffusion_policy_cnn/)。
