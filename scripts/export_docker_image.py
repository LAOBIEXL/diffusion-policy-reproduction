"""Save a Docker image as a compressed archive and record its SHA256."""

import argparse
import gzip
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default="laobiexl/diffusion-policy-reproduction:pusht-epoch0500")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/diffusion-policy-pusht-epoch0500.tar.gz")
    parser.add_argument("--record", type=Path, default=ROOT / "records/docker/image.json")
    args = parser.parse_args()
    image = json.loads(subprocess.check_output(["docker", "image", "inspect", args.image]))[0]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    partial = args.output.with_suffix(args.output.suffix + ".part")
    with subprocess.Popen(["docker", "save", args.image], stdout=subprocess.PIPE) as process:
        with partial.open("wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", compresslevel=1, mtime=0) as archive:
            shutil.copyfileobj(process.stdout, archive, length=8 * 2**20)
        if process.wait() != 0:
            raise RuntimeError("docker save failed")
    partial.replace(args.output)
    digest = hashlib.sha256()
    with args.output.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 2**20), b""):
            digest.update(chunk)
    checksum = digest.hexdigest()
    args.output.with_name(args.output.name + ".sha256").write_text(f"{checksum}  {args.output.name}\n", encoding="utf-8")
    record = {
        "tag": args.image,
        "image_id": image["Id"],
        "image_created": image["Created"],
        "image_bytes": image["Size"],
        "platform": f"{image['Os']}/{image['Architecture']}",
        "archive": args.output.name,
        "archive_bytes": args.output.stat().st_size,
        "archive_sha256": checksum,
    }
    args.record.parent.mkdir(parents=True, exist_ok=True)
    args.record.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
