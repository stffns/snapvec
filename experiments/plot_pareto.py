"""Pareto plot for the unified competitive benchmark.

Reads the rows currently published in docs/benchmarks.md's head-to-head
table (hard-coded here so the plot generator has no external deps
beyond matplotlib).  Emits a single PNG showing recall@10 vs p50
latency, with point size proportional to on-disk footprint and
colour-coded by backend family.  Pareto-dominant rows are annotated
inline.

Run with: python experiments/plot_pareto.py docs/_static/pareto.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch


# (label, family, recall, p50_us, disk_mb, marker)
ROWS = [
    ("sqlite-vec exact",            "sqlite",  1.000, 13401, 91.1,  "D"),
    ("hnswlib M=32",                "hnsw",    0.994,   524, 104.5, "s"),
    ("snapvec IVFPQ + fp16 rerank", "snapvec", 0.945,   345,  56.9, "*"),
    ("FAISS IVFPQ M=192",           "faiss",   0.906,   457,  12.7, "o"),
    ("snapvec IVFPQ M=192",         "snapvec", 0.895,   319,  12.6, "o"),
    ("snapvec IVFPQ M=192 + OPQ",   "snapvec", 0.895,   368,  13.2, "^"),
    ("snapvec SnapIndex 4-bit",     "snapvec", 0.854,  2764,  15.4, "v"),
    ("snapvec IVFPQ M=48 + OPQ",    "snapvec", 0.649,   263,   4.9, "^"),
    ("FAISS IVFPQ M=48",            "faiss",   0.603,   144,   4.4, "o"),
    ("snapvec IVFPQ M=48",          "snapvec", 0.549,   241,   4.3, "o"),
]

FAMILY_COLOR = {
    "snapvec": "#1f77b4",  # blue
    "faiss":   "#ff7f0e",  # orange
    "hnsw":    "#2ca02c",  # green
    "sqlite":  "#7f7f7f",  # gray
}

# Rows to label inline (Pareto-dominant + flagship markers).
HIGHLIGHT = {
    "snapvec IVFPQ + fp16 rerank",
    "snapvec IVFPQ M=192",
    "snapvec IVFPQ M=48 + OPQ",
    "FAISS IVFPQ M=48",
    "hnswlib M=32",
    "sqlite-vec exact",
}


def main(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))

    # Draw a shaded 'flagship corner' (recall >= 0.89, p50 <= 500us).
    ax.axvspan(100, 500, ymin=0, ymax=1, color="#fff4e0", alpha=0.4, zorder=0)

    for label, family, recall, p50, disk, marker in ROWS:
        # Point area scales with disk (sqrt so the sqlite/hnswlib rows
        # don't overwhelm).  Minimum size so small rows stay visible.
        size = max(60, 40 * (disk ** 0.5))
        ax.scatter(
            p50, recall,
            s=size,
            marker=marker,
            c=FAMILY_COLOR[family],
            edgecolors="black",
            linewidths=0.6,
            alpha=0.85,
            zorder=3,
        )
        if label in HIGHLIGHT:
            offset = (10, 8)
            if label == "sqlite-vec exact":
                offset = (-140, 6)
            elif label == "FAISS IVFPQ M=48":
                offset = (-8, -18)
            elif label == "snapvec IVFPQ M=48 + OPQ":
                offset = (10, -6)
            ax.annotate(
                label,
                (p50, recall),
                xytext=offset,
                textcoords="offset points",
                fontsize=8,
                color="#333",
            )

    ax.set_xscale("log")
    ax.set_xlabel("p50 latency per query (us, log scale)")
    ax.set_ylabel("recall@10 vs brute-force float32")
    ax.set_xlim(100, 20000)
    ax.set_ylim(0.5, 1.01)
    ax.grid(True, which="major", ls="--", lw=0.4, alpha=0.6)
    ax.set_title(
        "ANN Pareto on BEIR FIQA (N=57,638, BGE-small, 200 queries, M4 Pro)\n"
        "point area proportional to on-disk footprint",
        fontsize=10,
    )

    # Legend: one entry per family + shape legend.
    family_handles = [
        Patch(color=c, label=name) for name, c in [
            ("snapvec", FAMILY_COLOR["snapvec"]),
            ("FAISS",   FAMILY_COLOR["faiss"]),
            ("hnswlib", FAMILY_COLOR["hnsw"]),
            ("sqlite-vec", FAMILY_COLOR["sqlite"]),
        ]
    ]
    first_legend = ax.legend(
        handles=family_handles, loc="lower left", title="backend",
        fontsize=8, title_fontsize=8, frameon=True,
    )
    ax.add_artist(first_legend)

    # Annotate the 'flagship corner' band
    ax.text(
        205, 0.515,
        "sub-500 us flagship corner",
        fontsize=7.5, color="#aa7700", style="italic",
    )

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "docs/_static/pareto.png")
    main(out)
