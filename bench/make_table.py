"""Render the README benchmark section from bench/results/*.json.

Only files in bench/results/ are read, and only records that name a GPU are used. With no
results the section says so. Run ``python -m bench.make_table`` to update README.md in place
between the ``bench-table`` markers.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

RESULTS_DIR = Path(__file__).parent / "results"
README = Path(__file__).parent.parent / "README.md"
START = "<!-- bench-table:start -->"
END = "<!-- bench-table:end -->"

NOT_MEASURED = (
    "**Not yet measured.** No benchmark has been run on a GPU yet, so there are no numbers here. "
    "No speedup is claimed until it is measured."
)


def load_records(results_dir: Path = RESULTS_DIR) -> list[dict[str, Any]]:
    records = []
    for path in sorted(results_dir.glob("*.json")):
        rec = json.loads(path.read_text())
        if rec.get("dry_run") or not (rec.get("gpu") or {}).get("name"):
            continue
        rec["_file"] = path.name
        records.append(rec)
    return records


def _fmt(ms: float | None) -> str:
    return "" if ms is None else f"{ms:.3f}"


def render(records: list[dict[str, Any]]) -> str:
    if not records:
        return NOT_MEASURED
    out = [
        "Generated from `bench/results/*.json` by `python -m bench.make_table`. Times are medians "
        "in milliseconds. A GPU result slower than `cpu` is reported as measured. "
        "No speedup is claimed beyond these measurements.",
        "",
    ]
    for rec in records:
        gpu, sw, cfg = rec["gpu"], rec["software"], rec["config"]
        out.append(
            f"**{gpu['name']}** (device {gpu['device_index']}, driver {gpu.get('driver')}), "
            f"torch {sw.get('torch')}, cuda-core {sw.get('cuda_core')}, {cfg['dtype']}, "
            f"fma={'on' if cfg['fma'] else 'off'}, {cfg['reps']} reps, "
            f"{rec['created_utc']}, `{rec['_file']}`"
        )
        out.append("")
        out.append(
            "| impl | size (S×L×D) | h2d | layout_prep | compute | d2h | end_to_end "
            "| matches cpu |"
        )
        out.append("|---|---|---|---|---|---|---|---|")
        for r in rec["results"]:
            size = f"{r['scenarios']}×{r['locations']}×{r['days']}"
            if r.get("error"):
                short = r["error"].split(":")[0]
                out.append(f"| {r['impl']} | {size} | {short} |  |  |  |  | n/a |")
                continue
            st = r["stages_ms"]
            out.append(
                f"| {r['impl']} | {size} | {_fmt(st.get('h2d'))} | {_fmt(st.get('layout_prep'))} "
                f"| {_fmt(st.get('compute'))} | {_fmt(st.get('d2h'))} "
                f"| {_fmt(st.get('end_to_end'))} | {'yes' if r['matches_cpu'] else '**no**'} |"
            )
        out.append("")
    return "\n".join(out).rstrip()


def update_readme(text: str, section: str) -> str:
    if START not in text or END not in text:
        raise ValueError("README is missing the bench-table markers")
    head, rest = text.split(START, 1)
    _, tail = rest.split(END, 1)
    return f"{head}{START}\n{section}\n{END}{tail}"


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    readme = Path(args[0]) if args else README
    section = render(load_records())
    readme.write_text(update_readme(readme.read_text(), section))
    print(section)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
