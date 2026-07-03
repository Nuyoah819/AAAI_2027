from __future__ import annotations

import csv
import argparse
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize CSTC result gaps.")
    parser.add_argument("--input", type=Path, default=CODE_ROOT / "results" / "cstc_main_results.csv")
    parser.add_argument("--output", type=Path, default=CODE_ROOT / "results" / "cstc_vs_sota_summary.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result_path = args.input
    output_path = args.output
    rows = list(csv.DictReader(result_path.open("r", encoding="utf-8")))
    with output_path.open("w", newline="", encoding="utf-8") as f:
        fields = ["dataset", "cstc_acc", "sota_acc", "gap_acc", "cstc_nmi", "sota_nmi", "gap_nmi", "cstc_ari", "sota_ari", "gap_ari", "status"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "dataset": row["dataset"],
                "cstc_acc": row.get("acc", ""),
                "sota_acc": row.get("sota_acc", ""),
                "gap_acc": row.get("gap_acc", ""),
                "cstc_nmi": row.get("nmi", ""),
                "sota_nmi": row.get("sota_nmi", ""),
                "gap_nmi": row.get("gap_nmi", ""),
                "cstc_ari": row.get("ari", ""),
                "sota_ari": row.get("sota_ari", ""),
                "gap_ari": row.get("gap_ari", ""),
                "status": row.get("status", ""),
            })
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
