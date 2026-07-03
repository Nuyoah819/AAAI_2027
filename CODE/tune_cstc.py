from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path
import subprocess
import sys


CODE_ROOT = Path(__file__).resolve().parent
SEARCH_SPACE = {
    "temperature": [0.20, 0.24],
    "refine_repel": [0.15, 0.25],
    "target_hetero_ratio": [0.24, 0.32],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fixed-space CSTC tuning launcher.")
    parser.add_argument("--datasets", nargs="+", default=["acm", "texas", "squirrel"])
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = CODE_ROOT / "results" / "tuning"
    out_dir.mkdir(parents=True, exist_ok=True)
    records = []
    keys = list(SEARCH_SPACE)
    for values in itertools.product(*(SEARCH_SPACE[k] for k in keys)):
        trial = dict(zip(keys, values))
        records.append({"trial": len(records), **trial, "selection_signal": "diagnostics-only; no test metric used for final selection"})
    (out_dir / "cstc_fixed_search_space.json").write_text(json.dumps({"space": SEARCH_SPACE, "records": records}, indent=2), encoding="utf-8")
    with (out_dir / "cstc_tuning_plan.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    if args.dry_run:
        print(f"Wrote tuning plan with {len(records)} trials to {out_dir}")
        return
    print("Tuning plan is recorded. For integrity, this script does not auto-select by test labels.")
    print("Run individual trials only after defining a label-free diagnostic selection rule.")


if __name__ == "__main__":
    main()
