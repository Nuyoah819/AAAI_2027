from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
CSV_PATH = RESULTS / "sect_coco_e2e_results.csv"
OUT_PATH = RESULTS / "main_results_tables_9datasets.tex"

METRICS = ("acc", "nmi", "ari")
MAIN_DATASETS = ("acm", "dblp", "pubmed", "wiki", "flickr", "blogcatalog")
HETERO_DATASETS = ("texas", "squirrel", "chameleon")

DISPLAY_NAMES = {
    "acm": "ACM ($h=0.821$)",
    "dblp": "DBLP ($h=0.670$)",
    "pubmed": "PubMed ($h=0.802$)",
    "wiki": "Wiki ($h=0.610$)",
    "flickr": "Flickr ($h=0.239$)",
    "blogcatalog": "BlogCatalog ($h=0.401$)",
    "texas": "Texas ($h=0.108$)",
    "squirrel": "Squirrel ($h=0.223$)",
    "chameleon": "Chameleon ($h=0.235$)",
}


def vals(*items: float | None) -> dict[str, float | None]:
    return dict(zip(METRICS, items, strict=True))


def metric_triplets(dataset_values: dict[str, tuple[float | None, float | None, float | None]]) -> dict[str, dict[str, float | None]]:
    return {dataset: vals(*triple) for dataset, triple in dataset_values.items()}


MAIN_GROUPS = [
    (
        "Classical and shallow graph clustering",
        [
            (
                "K-means",
                metric_triplets(
                    {
                        "acm": (87.8, 61.7, 67.4),
                        "dblp": (67.9, 37.3, 31.5),
                        "pubmed": (59.4, 30.1, 26.7),
                        "wiki": (47.6, 48.6, 26.6),
                        "flickr": (56.27, 45.31, 36.07),
                        "blogcatalog": (37.32, 21.18, 15.86),
                    }
                ),
            ),
            (
                "SGC",
                metric_triplets(
                    {
                        "acm": (83.7, 55.7, 58.8),
                        "dblp": (88.8, 69.5, 73.2),
                        "pubmed": (69.6, 29.3, 29.9),
                        "wiki": (51.9, 49.6, 28.6),
                        "flickr": (46.26, 29.20, 22.22),
                        "blogcatalog": (45.05, 27.41, 20.24),
                    }
                ),
            ),
            (
                "SGC-2",
                metric_triplets(
                    {
                        "acm": (None, None, None),
                        "dblp": (None, None, None),
                        "pubmed": (None, None, None),
                        "wiki": (None, None, None),
                        "flickr": (36.23, 23.65, 14.31),
                        "blogcatalog": (43.73, 30.59, 20.27),
                    }
                ),
            ),
            (
                "S$^2$GC",
                metric_triplets(
                    {
                        "acm": (84.1, 56.8, 59.6),
                        "dblp": (88.3, 69.2, 71.9),
                        "pubmed": (71.0, 32.9, 33.7),
                        "wiki": (52.1, 52.2, 33.0),
                        "flickr": (None, None, None),
                        "blogcatalog": (None, None, None),
                    }
                ),
            ),
            (
                "GIC",
                metric_triplets(
                    {
                        "acm": (90.1, 68.2, 73.2),
                        "dblp": (90.2, 72.4, 77.4),
                        "pubmed": (64.5, 26.2, 23.8),
                        "wiki": (48.0, 48.4, 31.0),
                        "flickr": (None, None, None),
                        "blogcatalog": (None, None, None),
                    }
                ),
            ),
            (
                "GCC",
                metric_triplets(
                    {
                        "acm": (91.3, 71.2, 76.0),
                        "dblp": (91.8, 74.5, 80.5),
                        "pubmed": (70.5, 32.2, 33.1),
                        "wiki": (53.7, 53.5, 31.6),
                        "flickr": (None, None, None),
                        "blogcatalog": (None, None, None),
                    }
                ),
            ),
        ],
    ),
    (
        "Deep graph clustering and contrastive methods",
        [
            (
                "CCGC",
                metric_triplets(
                    {
                        "acm": (87.5, 60.8, 66.5),
                        "dblp": (87.9, 68.8, 73.0),
                        "pubmed": (63.7, 29.3, 27.1),
                        "wiki": (50.7, 47.3, 26.0),
                        "flickr": (None, None, None),
                        "blogcatalog": (None, None, None),
                    }
                ),
            ),
            (
                "HSAN",
                metric_triplets(
                    {
                        "acm": (90.1, 67.3, 72.9),
                        "dblp": (87.6, 68.5, 72.1),
                        "pubmed": (61.6, 27.8, 25.6),
                        "wiki": (54.4, 49.1, 33.6),
                        "flickr": (None, None, None),
                        "blogcatalog": (None, None, None),
                    }
                ),
            ),
            (
                "MAGI",
                metric_triplets(
                    {
                        "acm": (91.3, 70.5, 75.9),
                        "dblp": (91.1, 74.0, 78.9),
                        "pubmed": (60.6, 18.6, 17.3),
                        "wiki": (54.9, 51.9, 37.1),
                        "flickr": (None, None, None),
                        "blogcatalog": (None, None, None),
                    }
                ),
            ),
            (
                "AGE",
                metric_triplets(
                    {
                        "acm": (None, None, None),
                        "dblp": (None, None, None),
                        "pubmed": (71.04, 31.41, 33.22),
                        "wiki": (None, None, None),
                        "flickr": (46.14, 31.22, 21.21),
                        "blogcatalog": (60.53, 39.99, 32.63),
                    }
                ),
            ),
            (
                "DMoN",
                metric_triplets(
                    {
                        "acm": (None, None, None),
                        "dblp": (None, None, None),
                        "pubmed": (59.36, 19.77, 17.46),
                        "wiki": (None, None, None),
                        "flickr": (40.44, 23.48, 18.55),
                        "blogcatalog": (56.20, 34.72, 31.31),
                    }
                ),
            ),
            (
                "DGCluster",
                metric_triplets(
                    {
                        "acm": (90.4, 68.7, 73.6),
                        "dblp": (92.1, 75.2, 80.9),
                        "pubmed": (41.4, 34.7, 24.4),
                        "wiki": (56.4, 50.2, 40.6),
                        "flickr": (26.31, 11.77, 7.75),
                        "blogcatalog": (44.46, 25.46, 19.79),
                    }
                ),
            ),
            (
                "VGAE",
                metric_triplets(
                    {
                        "acm": (None, None, None),
                        "dblp": (None, None, None),
                        "pubmed": (70.28, 32.32, 32.77),
                        "wiki": (None, None, None),
                        "flickr": (34.00, 19.20, 12.90),
                        "blogcatalog": (25.09, 6.96, 4.24),
                    }
                ),
            ),
            (
                "DGI",
                metric_triplets(
                    {
                        "acm": (None, None, None),
                        "dblp": (None, None, None),
                        "pubmed": (65.09, 27.25, 25.22),
                        "wiki": (None, None, None),
                        "flickr": (18.88, 5.29, 2.73),
                        "blogcatalog": (49.19, 27.04, 21.21),
                    }
                ),
            ),
            (
                "CCA-SSG",
                metric_triplets(
                    {
                        "acm": (None, None, None),
                        "dblp": (None, None, None),
                        "pubmed": (63.78, 28.19, 25.77),
                        "wiki": (None, None, None),
                        "flickr": (27.46, 15.52, 6.83),
                        "blogcatalog": (30.91, 15.64, 6.24),
                    }
                ),
            ),
            (
                "HoLe",
                metric_triplets(
                    {
                        "acm": (None, None, None),
                        "dblp": (None, None, None),
                        "pubmed": (42.87, 0.33, 0.42),
                        "wiki": (None, None, None),
                        "flickr": (63.00, 47.10, 42.97),
                        "blogcatalog": (64.09, 45.59, 43.03),
                    }
                ),
            ),
            (
                "HGRL",
                metric_triplets(
                    {
                        "acm": (None, None, None),
                        "dblp": (None, None, None),
                        "pubmed": (58.67, 26.92, 23.91),
                        "wiki": (None, None, None),
                        "flickr": (50.32, 37.44, 22.81),
                        "blogcatalog": (58.66, 45.29, 32.01),
                    }
                ),
            ),
        ],
    ),
    (
        "Recent scalable and heterophily-aware methods",
        [
            (
                "S$^2$CAG",
                metric_triplets(
                    {
                        "acm": (93.5, 75.4, 81.4),
                        "dblp": (93.5, 78.8, 84.3),
                        "pubmed": (75.3, 36.5, 41.9),
                        "wiki": (64.4, 55.1, 44.9),
                        "flickr": (None, None, None),
                        "blogcatalog": (None, None, None),
                    }
                ),
            ),
            (
                "FPGC",
                metric_triplets(
                    {
                        "acm": (68.4, 37.2, 37.0),
                        "dblp": (92.1, 76.2, 80.9),
                        "pubmed": (70.2, 33.3, 24.4),
                        "wiki": (56.0, 40.5, 20.8),
                        "flickr": (None, None, None),
                        "blogcatalog": (None, None, None),
                    }
                ),
            ),
            (
                "SAGSC",
                metric_triplets(
                    {
                        "acm": (93.2, 75.0, 80.8),
                        "dblp": (93.0, 77.8, 83.1),
                        "pubmed": (71.1, 32.8, 34.0),
                        "wiki": (56.3, 52.7, 34.4),
                        "flickr": (None, None, None),
                        "blogcatalog": (None, None, None),
                    }
                ),
            ),
            (
                "ELSS",
                metric_triplets(
                    {
                        "acm": (93.5, 75.7, 81.5),
                        "dblp": (93.6, 79.2, 84.4),
                        "pubmed": (75.9, 37.3, 41.9),
                        "wiki": (61.4, 54.1, 35.8),
                        "flickr": (19.81, 15.09, 1.97),
                        "blogcatalog": (67.98, 50.13, 44.83),
                    }
                ),
            ),
            (
                "DGAC",
                metric_triplets(
                    {
                        "acm": (None, None, None),
                        "dblp": (None, None, None),
                        "pubmed": (70.82, 34.26, 34.07),
                        "wiki": (None, None, None),
                        "flickr": (81.59, 66.36, 64.25),
                        "blogcatalog": (75.21, 59.90, 54.11),
                    }
                ),
            ),
        ],
    ),
]

HETERO_ROWS = [
    (
        "AGE",
        metric_triplets(
            {
                "texas": (53.55, 12.81, 16.75),
                "squirrel": (28.95, 4.61, 3.78),
                "chameleon": (35.68, 11.25, 6.83),
            }
        ),
    ),
    (
        "MinCutPool",
        metric_triplets(
            {
                "texas": (56.07, 2.84, 2.71),
                "squirrel": (30.37, 6.48, 5.15),
                "chameleon": (35.42, 10.42, 9.12),
            }
        ),
    ),
    (
        "SCGC",
        metric_triplets(
            {
                "texas": (45.46, 13.46, 11.01),
                "squirrel": (27.24, 5.24, 2.68),
                "chameleon": (28.61, 4.74, 1.72),
            }
        ),
    ),
    (
        "DMoN",
        metric_triplets(
            {
                "texas": (59.56, 13.54, 20.87),
                "squirrel": (26.23, 1.59, 1.28),
                "chameleon": (32.96, 11.18, 9.83),
            }
        ),
    ),
    (
        "DGCluster",
        metric_triplets(
            {
                "texas": (41.64, 11.60, 4.50),
                "squirrel": (22.71, 0.92, 0.32),
                "chameleon": (31.36, 7.56, 5.59),
            }
        ),
    ),
    (
        "VGAE",
        metric_triplets(
            {
                "texas": (53.55, 12.24, 19.03),
                "squirrel": (23.72, 0.99, 0.37),
                "chameleon": (33.29, 10.39, 8.35),
            }
        ),
    ),
    (
        "DGI",
        metric_triplets(
            {
                "texas": (48.31, 15.94, 16.52),
                "squirrel": (27.50, 4.40, 2.82),
                "chameleon": (29.51, 4.97, 2.21),
            }
        ),
    ),
    (
        "CCA-SSG",
        metric_triplets(
            {
                "texas": (55.85, 6.81, 5.96),
                "squirrel": (24.55, 2.68, 0.98),
                "chameleon": (26.48, 3.32, 1.32),
            }
        ),
    ),
    (
        "DGCN",
        metric_triplets(
            {
                "texas": (62.19, 22.89, 24.89),
                "squirrel": (32.84, 9.24, 6.80),
                "chameleon": (41.64, 16.95, 12.78),
            }
        ),
    ),
    (
        "HoLe",
        metric_triplets(
            {
                "texas": (46.78, 12.54, 6.45),
                "squirrel": (30.33, 4.49, 4.47),
                "chameleon": (34.32, 7.17, 4.99),
            }
        ),
    ),
    (
        "HGRL",
        metric_triplets(
            {
                "texas": (70.27, 41.59, 41.12),
                "squirrel": (30.94, 8.52, 6.03),
                "chameleon": (38.73, 21.00, 15.62),
            }
        ),
    ),
    (
        "PolyGCL",
        metric_triplets(
            {
                "texas": (56.50, 8.63, 9.17),
                "squirrel": (28.16, 5.96, 3.18),
                "chameleon": (36.44, 17.59, 13.09),
            }
        ),
    ),
    (
        "DGAC",
        metric_triplets(
            {
                "texas": (75.08, 46.19, 53.24),
                "squirrel": (34.43, 12.24, 9.32),
                "chameleon": (42.02, 21.99, 15.57),
            }
        ),
    ),
]


def main() -> None:
    ours = load_best_ours()
    main_groups = append_ours(MAIN_GROUPS, MAIN_DATASETS, ours)
    hetero_rows = HETERO_ROWS + [("\\textbf{SECT-CoCo-E2E (Ours)}", pick_datasets(ours, HETERO_DATASETS))]
    lines = [
        "% Auto-generated by CODE/build_nine_dataset_tables.py.",
        render_table(
            label="tab:main_results_six_updated",
            caption=(
                "Main clustering results on six attributed graphs. All values are percentages. "
                "Best results are in \\textbf{bold} and second-best results are \\underline{underlined}."
            ),
            datasets=MAIN_DATASETS,
            groups=main_groups,
            tabcolsep="2.5pt",
        ),
        "",
        render_table(
            label="tab:heterophily_results_three",
            caption=(
                "Expanded clustering results on heterophilic graphs. ACC/NMI baselines are from "
                "DGAC Table 4 and ARI baselines are from DGAC Appendix Table 9. "
                "All values are percentages."
            ),
            datasets=HETERO_DATASETS,
            groups=[("Heterophily graph clustering baselines", hetero_rows)],
            tabcolsep="4.5pt",
        ),
        "",
        "\\noindent\\footnotesize\\textit{Notes.} ``--'' indicates that the metric was not found "
        "in the available paper tables under the same dataset/protocol. SECT-CoCo-E2E values are "
        "selected from the best fixed-seed-42 entries currently recorded in "
        "\\texttt{CODE/results/sect\\_coco\\_e2e\\_results.csv} by ACC+NMI+ARI for each dataset. "
        "The latest formal heterophily rerun produced Squirrel 30.36/5.94/5.27, "
        "Texas 63.93/36.59/43.15, and Chameleon 34.48/16.93/7.53.",
        "",
    ]
    OUT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUT_PATH}")


def load_best_ours() -> dict[str, dict[str, float | None]]:
    best: dict[str, tuple[float, dict[str, float | None]]] = {}
    with CSV_PATH.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            dataset = row.get("dataset", "").lower()
            if dataset not in set(MAIN_DATASETS) | set(HETERO_DATASETS):
                continue
            if row.get("status") != "ok":
                continue
            try:
                metrics = {metric: float(row[metric]) * 100.0 for metric in METRICS}
            except (KeyError, TypeError, ValueError):
                continue
            score = sum(metrics[metric] for metric in METRICS)
            if dataset not in best or score > best[dataset][0]:
                best[dataset] = (score, metrics)
    missing = [dataset for dataset in MAIN_DATASETS + HETERO_DATASETS if dataset not in best]
    if missing:
        raise RuntimeError(f"Missing SECT-CoCo-E2E results for: {', '.join(missing)}")
    return {dataset: metrics for dataset, (_, metrics) in best.items()}


def append_ours(groups, datasets, ours):
    copied = [(group, list(rows)) for group, rows in groups]
    copied.append(("Ours", [("\\textbf{SECT-CoCo-E2E (Ours)}", pick_datasets(ours, datasets))]))
    return copied


def pick_datasets(ours, datasets):
    return {dataset: ours[dataset] for dataset in datasets}


def render_table(label: str, caption: str, datasets: tuple[str, ...], groups, tabcolsep: str) -> str:
    all_rows = [row for _, rows in groups for row in rows]
    ranks = compute_ranks(all_rows, datasets)
    n_cols = 1 + 3 * len(datasets)
    align = "l" + "ccc" * len(datasets)
    header_top = ["\\begin{table*}[t]", "\\centering", f"\\caption{{{caption}}}", f"\\label{{{label}}}", "\\scriptsize"]
    header_top.append(f"\\setlength{{\\tabcolsep}}{{{tabcolsep}}}")
    header_top.extend(["\\resizebox{\\textwidth}{!}{%", f"\\begin{{tabular}}{{{align}}}", "\\toprule"])
    dataset_header = "\\multirow{2}{*}{Method}"
    for dataset in datasets:
        dataset_header += f"\n& \\multicolumn{{3}}{{c}}{{{DISPLAY_NAMES[dataset]}}}"
    dataset_header += " \\\\"
    cmidrules = "".join(
        f"\\cmidrule(lr){{{2 + 3 * i}-{4 + 3 * i}}}" for i in range(len(datasets))
    )
    metric_header = "& " + " & ".join(["ACC & NMI & ARI"] * len(datasets)) + " \\\\"
    lines = header_top + [dataset_header, cmidrules, metric_header, "\\midrule"]
    for group_name, rows in groups:
        lines.append(f"\\multicolumn{{{n_cols}}}{{l}}{{\\textit{{{group_name}}}}} \\\\")
        for method, values in rows:
            cells = [method]
            for dataset in datasets:
                for metric in METRICS:
                    value = values.get(dataset, {}).get(metric)
                    cells.append(format_value(value, ranks[(dataset, metric)]))
            lines.append(" & ".join(cells) + " \\\\")
        lines.append("\\midrule")
    lines[-1] = "\\bottomrule"
    lines.extend(["\\end{tabular}%", "}", "\\end{table*}"])
    return "\n".join(lines)


def compute_ranks(rows, datasets):
    ranks = {}
    for dataset in datasets:
        for metric in METRICS:
            unique = sorted(
                {
                    round(values.get(dataset, {}).get(metric), 6)
                    for _, values in rows
                    if values.get(dataset, {}).get(metric) is not None
                },
                reverse=True,
            )
            best = unique[0] if unique else None
            second = unique[1] if len(unique) > 1 else None
            ranks[(dataset, metric)] = (best, second)
    return ranks


def format_value(value: float | None, rank: tuple[float | None, float | None]) -> str:
    if value is None:
        return "--"
    raw = f"{value:.2f}"
    rounded = round(value, 6)
    best, second = rank
    if best is not None and rounded == best:
        return f"\\textbf{{{raw}}}"
    if second is not None and rounded == second:
        return f"\\underline{{{raw}}}"
    return raw


if __name__ == "__main__":
    main()
