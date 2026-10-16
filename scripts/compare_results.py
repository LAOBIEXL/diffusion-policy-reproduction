"""Compare the reproduced Push-T scores with the matching official checkpoint."""

import argparse
import csv
import json
import shutil
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EPISODE_PREFIX = "test/sim_max_reward_"


def reference_at_epoch(path, epoch):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            if record.get("epoch") == epoch and "test/mean_score" in record:
                return record
    raise ValueError(f"No evaluation at epoch {epoch} in {path}")


def episode_scores(record):
    return {int(key.removeprefix(EPISODE_PREFIX)): float(value)
            for key, value in record.items() if key.startswith(EPISODE_PREFIX)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--repeat-evaluation", type=Path, help="A second evaluation with the same configuration")
    parser.add_argument("--reference", type=Path, default=ROOT / "data/reference/train_0.logs.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "records/evaluation")
    args = parser.parse_args()
    actual = json.loads(args.evaluation.read_text(encoding="utf-8"))
    reference = reference_at_epoch(args.reference, actual["checkpoint_epoch"])
    actual_scores = episode_scores(actual)
    reference_scores = episode_scores(reference)
    if actual_scores.keys() != reference_scores.keys():
        raise ValueError("Evaluation seeds differ from the official checkpoint log")
    actual_mean = statistics.mean(actual_scores.values())
    reference_mean = statistics.mean(reference_scores.values())
    assert abs(actual_mean - actual["test/mean_score"]) < 1e-9
    assert abs(reference_mean - reference["test/mean_score"]) < 1e-9

    rows = [{"seed": seed, "official": reference_scores[seed],
             "reproduction": actual_scores[seed], "delta": actual_scores[seed] - reference_scores[seed]}
            for seed in sorted(actual_scores)]
    summary = {
        "checkpoint_epoch": actual["checkpoint_epoch"],
        "test_episodes": len(rows),
        "test_seed_start": min(actual_scores),
        "test_seed_end": max(actual_scores),
        "official_mean_score": reference_mean,
        "reproduction_mean_score": actual_mean,
        "mean_difference": actual_mean - reference_mean,
        "paired_mean_absolute_difference": statistics.mean(abs(row["delta"]) for row in rows),
        "evaluation_seconds": actual["evaluation_seconds"],
        "peak_gpu_allocated_mib": actual["peak_gpu_allocated_mib"],
        "environment": actual["environment"],
        "reference_log": str(args.reference),
        "evaluation_log": str(args.evaluation),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    repeat_section = ""
    if args.repeat_evaluation is not None:
        repeat = json.loads(args.repeat_evaluation.read_text(encoding="utf-8"))
        repeat_scores = episode_scores(repeat)
        if actual_scores.keys() != repeat_scores.keys():
            raise ValueError("Repeat evaluation uses different test seeds")
        repeatability = {
            "first_mean_score": actual_mean,
            "second_mean_score": statistics.mean(repeat_scores.values()),
            "max_episode_absolute_difference": max(abs(actual_scores[seed] - repeat_scores[seed]) for seed in actual_scores),
            "second_evaluation_seconds": repeat["evaluation_seconds"],
        }
        summary["repeatability"] = repeatability
        (args.output / "repeat_eval_log.json").write_bytes(args.repeat_evaluation.read_bytes())
        repeat_section = (f"\n## 重复运行\n\n相同配置独立运行两次，平均得分分别为 "
                          f"{actual_mean:.6f} 和 {repeatability['second_mean_score']:.6f}，"
                          f"逐回合最大绝对差为 {repeatability['max_episode_absolute_difference']:.3g}。\n")
    (args.output / "comparison.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (args.output / "eval_log.json").write_bytes(args.evaluation.read_bytes())
    for name in ["checkpoint_config.yaml", "evaluation_config.yaml"]:
        (args.output / name).write_bytes((args.evaluation.parent / name).read_bytes())
    video_links = []
    for key, value in actual.items():
        if key.startswith("test/sim_video_"):
            seed = key.removeprefix("test/sim_video_")
            target = args.output / "videos" / f"seed{seed}.mp4"
            target.parent.mkdir(exist_ok=True)
            shutil.copyfile(ROOT / value, target)
            video_links.append(f"- [回合 {seed}](videos/{target.name})")
    with (args.output / "episodes.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["seed", "official", "reproduction", "delta"])
        writer.writeheader()
        writer.writerows(rows)
    report = f"""# Push-T 视觉仿真复现结果

使用作者公开的 epoch={actual['checkpoint_epoch']} EMA 权重，在 {len(rows)} 个官方测试初始条件上运行闭环控制。

| 指标 | 结果 |
| --- | ---: |
| 官方平均得分 | {reference_mean:.6f} |
| 本机平均得分 | {actual_mean:.6f} |
| 得分差值（本机 − 官方） | {actual_mean - reference_mean:+.6f} |
| 测试回合数 | {len(rows)} |
| 评测用时 | {actual['evaluation_seconds']:.2f} s |
| 峰值 GPU 分配内存 | {actual['peak_gpu_allocated_mib']:.2f} MiB |

环境种子为 {min(actual_scores)}–{max(actual_scores)}，与官方日志逐项对齐。每回合上限 300 步，DDPM 采样 100 步，执行时域 8 步。得分为每回合最大归一化目标覆盖率的平均值。

运行环境：{actual['environment']['gpu']}，PyTorch {actual['environment']['torch']}，CUDA {actual['environment']['cuda']}，Python {actual['environment']['python']}。

官方日志记录训练期间的评测，本次从检查点配置的随机种子重新初始化采样。DDPM 动作采样的随机数状态及 PyTorch / CUDA 版本差异可能影响生成的轨迹与得分。

逐回合比较见 [episodes.csv](episodes.csv)，完整输出见 [eval_log.json](eval_log.json)，实际配置见 [evaluation_config.yaml](evaluation_config.yaml)。

## 回合视频

{chr(10).join(video_links)}
{repeat_section}
"""
    (args.output / "report.md").write_text(report.rstrip() + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
