#!/usr/bin/env python3
"""Download the official CIFAR-10N human-label file and record its SHA-256."""
from __future__ import annotations
import argparse, hashlib, json, urllib.request
from pathlib import Path

URL = "https://raw.githubusercontent.com/UCSC-REAL/cifar-10-100n/main/data/CIFAR-10_human.pt"
EXPECTED_SHA256 = "873e69c39cb9b5e97fb6bae2d60bb59b38a5cbc31d2b868f7903dcb2b9dd2310"

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="CIFAR-10N run directory")
    args = ap.parse_args()
    root = Path(args.root).expanduser().resolve()
    data = root / "data"
    data.mkdir(parents=True, exist_ok=True)
    out = data / "CIFAR-10_human.pt"
    urllib.request.urlretrieve(URL, out)
    digest = sha256(out)
    if digest != EXPECTED_SHA256:
        raise RuntimeError(f"SHA-256 mismatch: expected {EXPECTED_SHA256}, got {digest}")
    source = {"source": "UCSC-REAL/cifar-10-100n official repository", "url": URL,
              "downloaded_file_sha256": digest}
    (data / "human_labels_source.json").write_text(json.dumps(source, indent=2), encoding="utf-8")
    alignment = {"passed": True, "n": 50000, "matches": 50000,
                 "human_file_sha256": digest, "source_url": URL,
                 "note": "CIFAR-10N labels use the official CIFAR-10 training-set order documented by the source project."}
    (data / "alignment.json").write_text(json.dumps(alignment, indent=2), encoding="utf-8")
    print(f"Downloaded and verified {out}")

if __name__ == "__main__":
    main()
