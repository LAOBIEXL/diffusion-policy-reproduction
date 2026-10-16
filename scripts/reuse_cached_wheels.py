"""Expose complete cached wheels to pip without downloading them again."""

import json
import os
from email.parser import Parser
from pathlib import Path
from zipfile import ZipFile, is_zipfile

cache = Path.home() / ".cache/pip/http-v2"
wheelhouse = Path.home() / ".local/share/diffusion-policy-reproduction/wheels"
wheelhouse.mkdir(parents=True, exist_ok=True)
records = []
for cached in cache.rglob("*.body"):
    if not is_zipfile(cached):
        continue
    with ZipFile(cached) as archive:
        metadata_paths = [p for p in archive.namelist() if p.count("/") == 1 and p.endswith(".dist-info/METADATA")]
        if len(metadata_paths) != 1:
            continue
        metadata_path = metadata_paths[0]
        wheel_path = metadata_path.replace("/METADATA", "/WHEEL")
        if wheel_path not in archive.namelist():
            continue
        metadata = Parser().parsestr(archive.read(metadata_path).decode())
        wheel = Parser().parsestr(archive.read(wheel_path).decode())
        tags = wheel.get_all("Tag")
        tag = next((tag for tag in tags if tag.startswith("py3-")), tags[0])
        name = metadata["Name"].replace("-", "_")
        filename = f"{name}-{metadata['Version']}-{tag}.whl"
    destination = wheelhouse / filename
    if not destination.exists():
        os.link(cached, destination)
    records.append({"wheel": filename, "bytes": cached.stat().st_size})

for cached in (Path.home() / ".cache/pip/wheels").rglob("*.whl"):
    destination = wheelhouse / cached.name
    if not destination.exists():
        os.link(cached, destination)
    records.append({"wheel": cached.name, "bytes": cached.stat().st_size})

root = Path(__file__).resolve().parents[1]
output = root / "records/setup/reused-wheels.json"
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
print(f"Exposed {len(records)} cached wheels by hard link at {wheelhouse}")
for record in records:
    if record["bytes"] > 100 * 2**20:
        print(record)
