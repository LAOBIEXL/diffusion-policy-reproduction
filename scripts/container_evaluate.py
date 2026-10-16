"""Run all configured evaluation episodes and write a comparison report."""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("/output"))
    parser.add_argument("--n-envs", type=int, help="Parallel environments; the complete seed set is retained")
    args = parser.parse_args()
    evaluation = [sys.executable, str(ROOT / "scripts/evaluate_checkpoint.py"),
                  "--checkpoint", str(ROOT / "data/docker/ema.ckpt"),
                  "--output", str(args.output)]
    if args.n_envs is not None:
        evaluation.extend(["--n-envs", str(args.n_envs)])
    subprocess.run(evaluation, check=True, cwd=ROOT)
    subprocess.run([sys.executable, str(ROOT / "scripts/compare_results.py"),
                    "--evaluation", str(args.output / "eval_log.json"),
                    "--reference", str(ROOT / "data/docker/reference.jsonl"),
                    "--output", str(args.output / "report")], check=True, cwd=ROOT)
    print(f"Evaluation complete: {args.output / 'report/report.md'}")


if __name__ == "__main__":
    main()
