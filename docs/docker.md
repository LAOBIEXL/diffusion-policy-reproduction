# Docker 镜像与评测

镜像：`laobiexl/diffusion-policy-reproduction:pusht-epoch0500`。

镜像内包含固定版本的官方代码、Python 3.10.20、PyTorch 2.11.0 + CUDA 12.8、官方 epoch=0500 EMA 权重和对应的参考得分。默认执行 50 个测试回合和 6 个训练初始条件的回合，生成得分、配置、视频和比较报告。

已在 Docker Desktop 4.69.0 / Engine 29.4.0、RTX 5070 Laptop 8 GB 上完成关闭网络的完整评测：平均得分 **0.918845**，耗时 **92.38 秒**，峰值 GPU 分配内存 **1136.01 MiB**。与 WSL 复现的逐回合最大绝对差为 **1.11e-16**，4 段测试视频全部解码通过。详见[容器评测报告](../records/docker/evaluation/report.md)和[验证记录](../records/docker/validation.json)。

## 导入并运行

主机使用 Linux x86_64 或 Windows + WSL2，具备 NVIDIA GPU 和支持 CUDA 12.8 的驱动。Windows 使用 Docker Desktop 的 WSL2 后端；Linux 使用已配置 NVIDIA Container Toolkit 的 Docker Engine。

导入交付的镜像文件：

```bash
docker load -i diffusion-policy-pusht-epoch0500.tar.gz
```

Linux / WSL，在希望保存结果的目录执行：

```bash
mkdir -p results
docker run --gpus all --shm-size=1g --network none \
  --mount "type=bind,source=$PWD/results,target=/output" \
  laobiexl/diffusion-policy-reproduction:pusht-epoch0500
```

Windows PowerShell：

```powershell
New-Item -ItemType Directory -Path results -Force | Out-Null
$resultPath = (Resolve-Path results).Path
docker run --gpus all --shm-size=1g --network none `
  --mount "type=bind,source=$resultPath,target=/output" `
  laobiexl/diffusion-policy-reproduction:pusht-epoch0500
```

运行时使用镜像内的权重和资源，网络关闭。默认 56 个并行环境；末尾追加 `--n-envs 8` 可调整并行数，仍会运行完整的测试种子集合。并行数会改变随机数消耗顺序，对照本仓库得分时使用默认配置。

输出文件：

| 路径 | 内容 |
| --- | --- |
| `results/eval_log.json` | 全部逐回合得分、平均分、GPU、耗时和显存 |
| `results/checkpoint_config.yaml` | 检查点配置 |
| `results/evaluation_config.yaml` | 本次环境配置 |
| `results/media/` | 仿真录像 |
| `results/report/report.md` | 与官方记录的比较报告 |
| `results/report/episodes.csv` | 50 个测试回合逐项比较 |
| `results/report/videos/` | 4 个测试回合录像 |

## 从仓库构建

以下为 Linux / WSL 命令。基础镜像固定为 Python 3.10.20 的 linux/amd64 digest，依赖固定在 `docker/requirements.lock.txt`，官方代码由 Git submodule 固定。

```bash
git clone --recurse-submodules https://github.com/LAOBIEXL/diffusion-policy-reproduction.git
cd diffusion-policy-reproduction

# 构建 Python、CUDA 和仿真依赖环境。
docker build --target runtime -t laobiexl/diffusion-policy-reproduction:runtime .

# 使用构建好的环境下载官方检查点和训练日志。
mkdir -p data
docker run --mount "type=bind,source=$PWD/data,target=/opt/diffusion-policy/data" \
  --entrypoint python laobiexl/diffusion-policy-reproduction:runtime \
  scripts/fetch_assets.py checkpoint train_log

# 导出 EMA 参数、配置、epoch 和对应的参考评测记录。
docker run --mount "type=bind,source=$PWD/data,target=/opt/diffusion-policy/data" \
  --entrypoint python laobiexl/diffusion-policy-reproduction:runtime \
  scripts/prepare_docker_assets.py

# 将权重装入最终镜像。
docker build -t laobiexl/diffusion-policy-reproduction:pusht-epoch0500 .
```

权重导出会校验官方文件 SHA256，逐项检查导出后的张量与原始 EMA 参数相等，并保存 `data/docker/manifest.json`。镜像中保留评测使用的 FP32 权重、归一化参数和配置，导出文件约 1.10 GB。

已有本仓库的 WSL 环境和下载文件时，可以直接运行：

```bash
python scripts/prepare_docker_assets.py
docker build -t laobiexl/diffusion-policy-reproduction:pusht-epoch0500 .
```

构建时可用 `--build-arg PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple` 指定 PyPI 镜像，用 `--build-arg APT_MIRROR=https://mirrors.tuna.tsinghua.edu.cn` 指定 Debian 镜像。若已有完整的 Linux Python 3.10 wheel 缓存，可通过 BuildKit 命名上下文复用：

```bash
docker buildx build --load \
  --build-context wheels=/absolute/path/to/wheels \
  --build-arg PIP_NO_INDEX=1 \
  -t laobiexl/diffusion-policy-reproduction:pusht-epoch0500 .
```

wheel 目录需包含锁文件中的全部依赖以及 `pip==23.3.2`、`setuptools==65.5.0`、`wheel==0.38.4`。Gym 0.21 的构建工具版本已固定。

## 导出与校验

```bash
python scripts/export_docker_image.py
docker load -i artifacts/diffusion-policy-pusht-epoch0500.tar.gz
```

导出程序生成 `.tar.gz`、同名 `.sha256` 校验文件，以及 `records/docker/image.json`，记录 image ID、镜像大小、平台和归档 SHA256。可用 `--output /path/to/image.tar.gz` 指定交付文件位置。

交付归档大小为 5,682,020,340 bytes，已通过 `docker load` 重新导入，image ID 与完成评测的镜像一致。校验信息见 [image.json](../records/docker/image.json)，导入记录见 [import-validation.json](../records/docker/import-validation.json)。

实际容器评测记录保存在 [records/docker](../records/docker/)，原始 WSL 评测记录保存在 [records/evaluation](../records/evaluation/)。
