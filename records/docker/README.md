# Docker 评测记录

镜像标签：`laobiexl/diffusion-policy-reproduction:pusht-epoch0500`。

| 记录 | 内容 |
| --- | --- |
| [evaluation/report.md](evaluation/report.md) | 50 回合得分、官方比较和视频 |
| [validation.json](validation.json) | 容器退出码、网络配置、与 WSL 的逐回合比较、视频解码 |
| [weights.json](weights.json) | 原始检查点和导出 EMA 的 SHA256、参数相等性检查 |
| [image.json](image.json) | image ID、平台、归档大小与 SHA256 |
| [import-validation.json](import-validation.json) | 归档重新导入及 image ID 一致性验证 |
| [load.log](load.log) | `docker load` 输出 |
| [environment.json](environment.json) | Docker / 内核版本、基础镜像和上游代码版本 |
| [build.log](build.log) | 最终镜像构建日志 |
| [evaluate.log](evaluate.log) | 容器完整评测输出 |
| [pip-check.log](pip-check.log) | 镜像依赖检查 |

启动时使用 `--network none`，只挂载结果目录；代码、模型、仿真环境和官方参考记录均从镜像读取。
