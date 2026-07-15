#!/usr/bin/env python
"""Build the standalone interactive cheetah gait explorer.

Reads the export written by ``benchmarks/export_cheetah_gait.py`` and inlines it
into ``benchmarks/gait_explorer.template.html``, producing one self-contained
HTML file: phase portraits, Poincare return maps and the metric battery, live
across (time base x mode x leg joint), plus the math behind each metric.

The page must be self-contained (it is published as an artifact, where a strict
CSP blocks every external request), so the data is inlined rather than fetched.
The raw export is ~2.9 MB, most of it trajectory samples at far higher rate than
a ~320 px plot can show, so it is compacted first:

  * the ``time`` array is dropped   - neither plot uses it
  * each (theta, theta_dot) trace is decimated to <= MAX_PTS samples. The traces
    are resampled at ~100 Hz (regular) / ~500 Hz (irregular); a ~3 Hz stride
    needs nothing like that to draw. Metrics are NOT recomputed here - they come
    from the export, where they were computed on the full-rate signals.
  * theta -> 4 dp, theta_dot -> 3 dp

    python -m benchmarks.build_gait_explorer
"""
from __future__ import annotations

import json
import os

MAX_PTS = 1100
SRC = "results/cheetah_gait_data/gait_data.json"
OUT = "results/cheetah_gait_explorer.html"
_HERE = os.path.dirname(os.path.abspath(__file__))
TEMPLATE = os.path.join(_HERE, "gait_explorer.template.html")
# Pre-rendered KaTeX + its inlined font faces. Static (the formulas do not depend
# on the data) and committed, so this build stays pure Python and needs no node.
# Regenerate only when the math changes: cd benchmarks && npm ci && npm run render
MATH = os.path.join(_HERE, "gait_math.generated.html")
PLACEHOLDER = "__GAIT_DATA__"
MATH_PLACEHOLDER = "__GAIT_MATH__"


def decimate(a, nd):
    """Decimate to <= MAX_PTS samples, keeping the last one so the loop closes."""
    k = max(1, -(-len(a) // MAX_PTS))          # ceil division
    out = list(a[::k])
    if out[-1] != a[-1]:
        out.append(a[-1])
    return [round(float(x), nd) for x in out]


def compact(d):
    out = {"meta": d["meta"], "data": {}}
    n_before = n_after = 0
    for tb, modes in d["data"].items():
        out["data"][tb] = {}
        for mode, blk in modes.items():
            joints = {}
            for j, jb in blk["joints"].items():
                if jb is None:
                    joints[j] = None
                    continue
                n_before += len(jb["theta"])
                theta = decimate(jb["theta"], 4)
                n_after += len(theta)
                joints[j] = {
                    "theta": theta,
                    "theta_dot": decimate(jb["theta_dot"], 3),
                    "section_theta": jb["section_theta"],
                    "cross_theta_dot": [round(float(x), 3)
                                        for x in jb["cross_theta_dot"]],
                    "metrics": jb["metrics"],
                }
            out["data"][tb][mode] = {
                "label": blk["label"],
                "seed": blk["seed"],
                "auto_ref_joint": blk["auto_ref_joint"],
                "aggregate": {k: (round(v, 5) if isinstance(v, (int, float)) else v)
                              for k, v in blk["aggregate"].items()},
                "joints": joints,
            }
    return out, n_before, n_after


def main():
    if not os.path.exists(SRC):
        raise SystemExit(f"{SRC} not found - run `python -m benchmarks.export_cheetah_gait` first")
    with open(SRC) as f:
        raw = json.load(f)
    data, n_before, n_after = compact(raw)

    with open(TEMPLATE) as f:
        tpl = f.read()
    for ph in (PLACEHOLDER, MATH_PLACEHOLDER):
        if ph not in tpl:
            raise SystemExit(f"{ph} missing from {TEMPLATE}")
    if not os.path.exists(MATH):
        raise SystemExit(f"{MATH} not found - run `node benchmarks/render_gait_math.js`")
    with open(MATH) as f:
        math_html = f.read()

    html = tpl.replace(PLACEHOLDER, json.dumps(data, separators=(",", ":")))
    html = html.replace(MATH_PLACEHOLDER, math_html)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        f.write(html)

    print(f"trace points {n_before} -> {n_after}")
    print(f"wrote {OUT} "
          f"({os.path.getsize(SRC)/1e6:.2f} MB export -> {os.path.getsize(OUT)/1e6:.2f} MB page)")


if __name__ == "__main__":
    main()
