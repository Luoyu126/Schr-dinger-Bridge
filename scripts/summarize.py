"""Aggregate runs: mean ± std over seeds at matched energy-gradient-call budgets.

python scripts/summarize.py runs/screen_u1 --metrics tv p_exact_mis entropy --at 76800 153600 307200
Run directories need meta.json and log.jsonl (written by train_mis.py).
Runs are grouped by every CLI argument except --seed and --out.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

IGNORE = {"seed", "out"}


def load(run):
    meta = json.loads((run / "meta.json").read_text())
    recs = [json.loads(l) for l in (run / "log.jsonl").read_text().splitlines() if l.strip()]
    summ = json.loads((run / "summary.json").read_text()) if (run / "summary.json").exists() else None
    return meta, recs, summ


def at_budget(recs, calls, metric):
    """Last eval record with grad_calls <= calls (None if the run diverged before calls)."""
    div = [r for r in recs if "diverged" in r]
    if div and div[0]["grad_calls"] <= calls:
        return "div"
    if not recs or max(r["grad_calls"] for r in recs) < calls:
        return None  # budget not reached (still running, or a shorter run)
    ok = [r for r in recs if metric in r and r["grad_calls"] <= calls]
    return ok[-1][metric] if ok else None


def fmt(vals):
    good = [v for v in vals if isinstance(v, (int, float))]
    nd = sum(v == "div" for v in vals)
    s = "—" if not good else (f"{np.mean(good):.3f}±{np.std(good):.3f}" if len(good) > 1
                               else f"{good[0]:.3f}")
    return s + (f" ({nd}div)" if nd else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--metrics", nargs="+", default=["tv", "p_exact_mis", "entropy"])
    ap.add_argument("--at", nargs="+", type=int, default=None)
    ap.add_argument("--keys", nargs="+", default=None, help="args shown in the group label")
    a = ap.parse_args()
    groups = defaultdict(list)
    for run in sorted(Path(a.root).glob("*/")):
        if not (run / "meta.json").exists() or not (run / "log.jsonl").exists():
            continue
        meta, recs, summ = load(run)
        args = meta["args"]
        key = tuple(sorted((k, str(v)) for k, v in args.items() if k not in IGNORE))
        groups[key].append((args["seed"], recs, summ))
    budgets = a.at or sorted({r["grad_calls"] for g in groups.values() for _, rs, _ in g
                              for r in rs if a.metrics[0] in r})[-1:]
    varying = set()
    keys = list(groups)
    for k in keys:
        for kk, v in k:
            if any(dict(o).get(kk) != v for o in keys):
                varying.add(kk)
    show = a.keys or sorted(varying)
    print("| " + " | ".join(show) + " | seeds | calls | " + " | ".join(a.metrics) + " |")
    print("|" + "---|" * (len(show) + 2 + len(a.metrics)))
    for key in sorted(groups, key=lambda k: [dict(k).get(s, "") for s in show]):
        d = dict(key)
        runs = groups[key]
        for b in budgets:
            cells = [fmt([at_budget(rs, b, m) for _, rs, _ in runs]) for m in a.metrics]
            print("| " + " | ".join(d.get(s, "") for s in show) +
                  f" | {len(runs)} | {b} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
