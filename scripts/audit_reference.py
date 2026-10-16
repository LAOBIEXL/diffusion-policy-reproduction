"""Summarize checkpoint metrics in the published train_0 log."""

import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
path = ROOT / "data/reference/train_0.logs.jsonl"
evaluations = []
line_count = 0
with path.open(encoding="utf-8") as stream:
    for line in stream:
        record = json.loads(line)
        line_count += 1
        if "test/mean_score" in record:
            evaluations.append(record)

scores = [r["test/mean_score"] for r in evaluations]
best = max(evaluations, key=lambda r: r["test/mean_score"])
lowest_loss = sorted(evaluations, key=lambda r: r["train_loss"])[:10]
report = {
    "source_log": str(path.relative_to(ROOT)),
    "line_count": line_count,
    "evaluation_count": len(evaluations),
    "first_evaluation_epoch": evaluations[0]["epoch"],
    "last_evaluation_epoch": evaluations[-1]["epoch"],
    "best_epoch": best["epoch"],
    "best_score": best["test/mean_score"],
    "last_10_mean_score": statistics.mean(scores[-10:]),
    "k_min_train_loss_mean_score": statistics.mean(r["test/mean_score"] for r in lowest_loss),
    "k_min_train_loss_epochs": [r["epoch"] for r in lowest_loss],
    "best_evaluation_test_episode_count": sum(k.startswith("test/sim_max_reward_") for k in best),
    "evaluations": [
        {"epoch": r["epoch"], "score": r["test/mean_score"], "train_loss": r["train_loss"]}
        for r in evaluations
    ],
}
output = ROOT / "records/reference/train_0_summary.json"
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print(json.dumps({k: v for k, v in report.items() if k != "evaluations"}, indent=2))
