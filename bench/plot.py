"""Plot median end_to_end and compute time against problem size from bench/results/*.json.

Needs matplotlib (not a project dependency; it is preinstalled on Kaggle):
    python -m bench.plot --out bench/results/timings.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

from bench.make_table import load_records


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("bench/results/timings.png"))
    a = p.parse_args(argv)
    records = load_records()
    if not records:
        print("No GPU results in bench/results/; nothing to plot.")
        return 1
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, stage in zip(axes, ("compute", "end_to_end"), strict=True):
        for rec in records:
            for impl in ("cpu", "gpu_torch", "gpu_cuda", "gpu_cuda_tiled"):
                rows = sorted(
                    (
                        r
                        for r in rec["results"]
                        if r["impl"] == impl and not r.get("error")  # e.g. out of memory
                    ),
                    key=lambda r: r["cells"],
                )
                where = "host CPU" if impl == "cpu" else rec["gpu"]["name"]
                if rows:
                    ax.plot(
                        [r["cells"] for r in rows],
                        [r["stages_ms"][stage] for r in rows],
                        marker="o",
                        label=f"{impl} ({where})",
                    )
                for r in rec["results"]:
                    if r["impl"] == impl and r.get("error"):
                        ax.axvline(r["cells"], color="0.6", linestyle=":", linewidth=1)
                        ax.annotate(
                            f"{impl}: {r['error'].split(':')[0].lower()}",
                            (r["cells"], 0.02),
                            xycoords=("data", "axes fraction"),
                            rotation=90,
                            fontsize=7,
                            color="0.35",
                            ha="right",
                            va="bottom",
                        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("cells (scenario × location × day)")
        ax.set_title(f"median {stage}")
    axes[0].set_ylabel("ms")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=150)
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
