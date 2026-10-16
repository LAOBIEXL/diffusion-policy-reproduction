# 实验日志

日期：2026-10-16，时区 Asia/Shanghai。

## 01 · 仓库与硬件

克隆 `git@github.com:LAOBIEXL/diffusion-policy-reproduction.git`，以 submodule 固定官方实现到 `5ba07ac6661db573af695b419a7947ecb704690f`。

- Windows 11，WSL2 Ubuntu 24.04。
- AMD Ryzen 9 8945HX，16 核 / 32 线程，主机内存 32 GB。
- RTX 5070 Laptop，8151 MiB 显存，驱动 582.05。
- WSL 内存约 15 GiB，swap 4 GiB；虚拟磁盘位于 D:/WSL/Ubuntu-24.04。
- 初始 D 盘剩余约 61 GB，硬件快照位于 `records/environment/windows.json`。

## 02 · 论文与公开材料

下载 RSS 2023 会议 PDF，核对标题、作者、会议、页数与附录。SHA256 为 `723a030b809be1958d03b06472ac79934fd64e5a890cdfeffd9508a353101a25`。

保存公开配置、train_0 日志和三种子聚合日志。`scripts/audit_reference.py` 统计到 512,400 条训练记录、61 次评测，每次包含 50 个测试初始条件。

- epoch=500 得分：0.8835836184400558。
- train_0 最佳：epoch=300，得分 0.9001179621471672。
- 三种子最佳检查点聚合：0.9066118835854854。
- 三种子最低训练损失 10 个检查点聚合：0.8363607066594249。

来源和哈希位于 `records/assets/`，统计结果位于 `records/reference/`。

## 03 · 独立环境安装

使用 Python 3.10.20 创建 venv：`/home/laobiexl/.local/share/diffusion-policy-reproduction/env`。

Conda repodata 下载出现 IncompleteRead，随后采用 venv。首次 CUDA 依赖下载发生截断，pip 哈希检查发现错误；通过已有完整 wheel 缓存完成 PyTorch 安装。后续依赖使用清华 PyPI 镜像，实测速度约 5–10 MB/s。

安装 CMake 3.24.3 后，`egl_probe` 构建成功。`scripts/install_wsl.sh` 完成安装，`pip check` 返回 `No broken requirements found.`。完整依赖见 `records/environment/requirements-wsl.lock.txt`，过程日志见 `records/setup/`。

CUDA 矩阵乘法通过：PyTorch 2.11.0+cu128、CUDA runtime 12.8、GPU 计算能力 (12, 0)。

## 04 · 数据与计算验证

下载并解压 `pusht.zip`，大小 30,988,725 bytes，SHA256 为 `63d52a114a3f010861f0181309d165b7d69133ccae426ece2fc94caed147bdf9`。

数据包含 206 条轨迹、25,650 帧。公开配置选择 90 条训练轨迹，其补集为 116 条验证轨迹，分别产生 10,726 / 13,482 个时序窗口。

`scripts/smoke_pusht.py` 完成仿真 reset / step / render、完整模型前向与反向以及 100 步 DDPM 采样。输出动作形状为 [1, 8, 2]，梯度数值正常。

- 扩散网络参数 251,511,938；视觉编码器参数 11,197,088。
- batch=2 前向反向约 2.22 s，峰值显存 2172.60 MiB。
- 单条观测 100 步采样约 2.58 s。

## 05 · 资源与训练流程检查

batch=64、FP32、GPU EMA 的完整优化步骤通过，AdamW 设置 foreach=false。峰值分配显存 5552.89 MiB，峰值保留 5784 MiB，两次步骤耗时 12.72 s 和 3.93 s。

训练 workspace 完成 3 次优化、2 个最多 16 步的回合、2 批验证、动作采样和检查点保存。计算验证结果位于 `records/validation/`。

梯度累积和 CPU EMA 辅助函数的数学对照：梯度最大绝对误差 8.94e-8，EMA 与上游实现对照误差为 0。本次模型评测使用 FP32 和 GPU 推理。

## 06 · 官方权重与评测配置

本次复现采用作者提供的 epoch=500 EMA 权重。文件通过 HTTP Range 分块下载、区间检查、完整 ZIP CRC 检查和 SHA256 计算。

- 大小：4,293,295,813 bytes。
- SHA256：`bac7221f7e34cd51162dc1972e1a39ffcddc87de1dc1780c44ffa61b88c4ff76`。
- 检查点测试种子：4300000–4300049；公开目录模板中的起始种子为 100000，评测按检查点配置执行。
- 56 个并行环境，包含 50 个测试初始条件和 6 个训练初始条件。
- 每回合最多 300 步，每次动作预测使用 100 步 DDPM。

命令：

```bash
python scripts/evaluate_checkpoint.py --checkpoint data/checkpoints/official_train_0_epoch0500.ckpt --output runs/eval-official-epoch0500
```

逐回合输出、视频和实际配置保存在 `runs/eval-official-epoch0500/`，比较报告保存到 `records/evaluation/`。

## 07 · 官方权重复现结果

50 个测试回合的本机平均得分为 0.9188447508779135，官方同一检查点记录为 0.8835836184400558，差值为 +0.03526113243785767。测试种子集合与官方日志完全一致。

评测耗时 96.63 秒，峰值 GPU 分配内存 1136.01 MiB。保存了 4 个测试回合和 2 个训练回合的视频。逐回合对比见 `records/evaluation/episodes.csv`，完整配置与结果见同目录报告。

相同配置再次独立评测，平均得分为 0.9188447508779135，耗时 92.40 秒。50 个回合与第一次结果的最大绝对差为 2.22e-16。复跑原始记录保存在 `records/evaluation/repeat_eval_log.json`。
