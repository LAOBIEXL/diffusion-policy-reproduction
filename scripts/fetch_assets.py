"""Download the public Push-T inputs and record their provenance."""

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import ProxyHandler, Request, build_opener
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
BASE = "https://diffusion-policy.cs.columbia.edu"
EXPERIMENT = f"{BASE}/data/experiments/image/pusht/diffusion_policy_cnn"
ASSETS = {
    "config": (f"{EXPERIMENT}/config.yaml", "references/official/config.yaml"),
    "dataset": (f"{BASE}/data/training/pusht.zip", "data/downloads/pusht.zip"),
    "train_log": (f"{EXPERIMENT}/train_0/logs.json.txt", "data/reference/train_0.logs.jsonl"),
    "metrics": (f"{EXPERIMENT}/metrics/logs.json.txt", "data/reference/metrics.logs.jsonl"),
    "checkpoint": (
        f"{EXPERIMENT}/train_0/checkpoints/epoch=0500-test_mean_score=0.884.ckpt",
        "data/checkpoints/official_train_0_epoch0500.ckpt",
    ),
}


def download(name, opener):
    url, relative_path = ASSETS[name]
    target = ROOT / relative_path
    target.parent.mkdir(parents=True, exist_ok=True)
    with opener.open(Request(url, method="HEAD"), timeout=60) as response:
        total = int(response.headers["Content-Length"])
    if target.exists():
        if target.stat().st_size != total:
            raise IOError(f"Incomplete file: {relative_path}")
        print(f"Using existing {relative_path}", flush=True)
    else:
        partial = target.with_suffix(target.suffix + ".part")
        if target.with_suffix(target.suffix + ".ranges.json").exists():
            raise RuntimeError("A range download is in progress; finish it before recording this asset")
        last_report = time.monotonic()
        size = partial.stat().st_size if partial.exists() else 0
        if size < total:
            request = Request(url, headers={"Range": f"bytes={size}-"})
            with opener.open(request, timeout=60) as response, partial.open("ab") as stream:
                if response.status != 206 or not response.headers["Content-Range"].startswith(f"bytes {size}-"):
                    raise IOError("Server did not honor the requested download range")
                while chunk := response.read(1024 * 1024):
                    stream.write(chunk)
                    size += len(chunk)
                    if time.monotonic() - last_report > 20:
                        print(f"{name}: {size / 2**20:.1f} / {total / 2**20:.1f} MiB", flush=True)
                        last_report = time.monotonic()
        if size != total:
            raise IOError(f"Interrupted download: {size} of {total} bytes; rerun to resume")
        if name == "dataset":
            with ZipFile(partial) as archive:
                bad_file = archive.testzip()
                if bad_file is not None:
                    raise IOError(f"ZIP checksum failed: {bad_file}")
        partial.rename(target)

    if name == "checkpoint":
        with ZipFile(target) as archive:
            bad_file = archive.testzip()
            if bad_file is not None:
                raise IOError(f"Checkpoint ZIP checksum failed: {bad_file}")
        print("Checkpoint ZIP CRC checks passed", flush=True)

    digest = hashlib.sha256()
    with target.open("rb") as stream:
        while chunk := stream.read(4 * 1024 * 1024):
            digest.update(chunk)
    record = {
        "name": name,
        "url": url,
        "path": relative_path,
        "bytes": target.stat().st_size,
        "expected_bytes": total,
        "sha256": digest.hexdigest(),
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    record_path = ROOT / "records" / "assets" / f"{name}.json"
    record_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record), flush=True)
    if name == "dataset":
        with ZipFile(target) as archive:
            archive.extractall(ROOT / "data")
        print("Extracted dataset under data/", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("assets", nargs="+", choices=ASSETS)
    parser.add_argument("--direct", action="store_true", help="Connect directly for this command, bypassing system proxy settings")
    args = parser.parse_args()
    opener = build_opener(ProxyHandler({})) if args.direct else build_opener()
    for name in args.assets:
        download(name, opener)
