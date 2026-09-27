"""POST every .png in a folder to a running service and print a summary.

Usage (from the project root):
    python -m scripts.ingest_folder data/raw/eval_set [--url http://localhost:8000]
"""
import argparse
from collections import Counter
from pathlib import Path

import httpx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path)
    parser.add_argument("--url", default="http://localhost:8000")
    args = parser.parse_args()

    files = sorted(args.folder.glob("*.png"))
    labels, statuses, outcomes = Counter(), Counter(), Counter()
    latencies = []

    with httpx.Client(base_url=args.url, timeout=30) as client:
        for path in files:
            with open(path, "rb") as f:
                resp = client.post("/tiles", files={"file": (path.name, f, "image/png")})
            if resp.status_code != 200:
                outcomes[f"rejected ({resp.status_code})"] += 1
                print(f"  {path.name}: {resp.status_code} {resp.json().get('detail')}")
                continue
            body = resp.json()
            pred = body["prediction"]
            outcomes["cached" if body["cached"] else "new"] += 1
            labels[pred["top1_label"]] += 1
            statuses[pred["status"]] += 1
            if not body["cached"]:
                latencies.append(pred["latency_ms"])

    print(f"\nFolder: {args.folder}  files: {len(files)}")
    print(f"Outcomes: {dict(outcomes)}")
    print(f"By status: {dict(statuses)}")
    print(f"By label: {dict(sorted(labels.items()))}")
    if latencies:
        latencies.sort()
        print(f"Inference latency ms: median={latencies[len(latencies) // 2]:.1f} max={latencies[-1]:.1f}")


if __name__ == "__main__":
    main()
