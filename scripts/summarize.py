"""Aggregate runs: mean ± std over seeds at matched energy-gradient-call budgets.

python scripts/summarize.py runs/screen_u1 --metrics tv p_exact_mis entropy --at 76800 153600 307200
Run directories need meta.json and log.jsonl (written by train_mis.py).
Runs are grouped by every CLI argument except --seed and --out.
"""
import argparse
import json
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np

IGNORE = {"seed", "out"}


def running_out_dirs():
    """--out directories of train_mis.py processes that are alive right now."""
    ps = subprocess.run(["ps", "-eo", "args="], capture_output=True, text=True).stdout
    outs = set()
    for line in ps.splitlines():
        if "train_mis.py" in line and "--out" in line:
            outs.add(line.split("--out", 1)[1].split()[0].rstrip("/"))
    return outs


def load(run):
    meta = json.loads((run / "meta.json").read_text())
    recs = [json.loads(l) for l in (run / "log.jsonl").read_text().splitlines() if l.strip()]
    summ = json.loads((run / "summary.json").read_text()) if (run / "summary.json").exists() else None
    return meta, recs, summ


def at_budget(recs, calls, metric, summ=None):
    """Value of `metric` at the last eval with grad_calls <= calls.

    Returns a number, or one of the markers
      "div"     the run diverged at or before this budget,
      "run"     still running (no summary.json yet) and the budget is not reached,
      "crash"   finished without summary.json (crashed / killed) before the budget,
      "short"   finished normally but its total budget is below `calls`.
    """
    div = [r for r in recs if "diverged" in r]
    if div and div[0]["grad_calls"] <= calls:
        return "div"
    if not recs or max(r["grad_calls"] for r in recs) < calls:
        if summ is None:
            return "run"
        return "crash" if summ.get("status") == "crash" else "short"
    ok = [r for r in recs if metric in r and r["grad_calls"] <= calls]
    return ok[-1][metric] if ok else None


def fmt(vals):
    """mean ± std over seeds that have a value; the contributing count is always shown
    when some seed is missing, and every missing seed is labelled by reason."""
    good = [v for v in vals if isinstance(v, (int, float)) and not isinstance(v, bool)]
    s = "—" if not good else (f"{np.mean(good):.3f}±{np.std(good):.3f}" if len(good) > 1
                               else f"{good[0]:.3f}")
    if len(good) < len(vals):
        tags = [f"{sum(v == k for v in vals)}{k}" for k in ("div", "run", "crash", "short")
                if any(v == k for v in vals)]
        s += f" [n={len(good)}/{len(vals)}" + ("; " + ", ".join(tags) if tags else "") + "]"
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--metrics", nargs="+", default=["tv", "p_exact_mis", "entropy"])
    ap.add_argument("--at", nargs="+", type=int, default=None)
    ap.add_argument("--keys", nargs="+", default=None, help="args shown in the group label")
    a = ap.parse_args()
    groups = defaultdict(list)
    live = running_out_dirs()
    for run in sorted(Path(a.root).glob("*/")):
        if not (run / "meta.json").exists() or not (run / "log.jsonl").exists():
            continue
        meta, recs, summ = load(run)
        if summ is None and str(run).rstrip("/") not in live:
            summ = {"status": "crash"}  # no summary and no live process writing to it
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
            cells = [fmt([at_budget(rs, b, m, sm) for _, rs, sm in runs]) for m in a.metrics]
            print("| " + " | ".join(d.get(s, "") for s in show) +
                  f" | {len(runs)} | {b} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
