"""Compare annealed AS pilots with the parallel-tempering reference of the target (pt_ladder.py).

For every evaluated stage of each pilot, the training temperature of that stage is looked up on
the PT ladder (log-linear interpolation, averaged over the PT graphs, which should be the
pilots' validation graphs) and the sampler's metrics at 200 inference steps are printed next to
the target's, for both PT starts (N(0, I) / greedy MIS).

python scripts/pilots/compare_pt_ladder.py --pt runs/pilots/pt_ladder/g*.json \
    --runs runs/pilots/as_v2/H1_Tgeo50 runs/pilots/as_v2/H3_Tgeo50_B1p_slow
"""
import argparse
import json
from pathlib import Path

import numpy as np

KEYS = ("raw_size_mean", "raw_feasible_frac", "n_distinct", "mis_size_mean", "best_of_16",
        "E_tgt_mean")
SHORT = ("raw", "feas", "ndist", "mis", "best16", "E")


def load_pt(paths):
    graphs = {}
    for p in paths:
        graphs.update(json.loads(Path(p).read_text())["graphs"])
    T = np.array([r["T"] for r in next(iter(graphs.values()))["levels"]])
    tab = {st: {k: np.mean([[r[st][k] for r in g["levels"]] for g in graphs.values()], 0)
                for k in KEYS + ("raw_empty_frac",)} for st in ("rand", "greedy")}
    return T, tab, graphs


def at(T, col, t):
    return float(np.interp(np.log(t), np.log(T), col))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt", nargs="+", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--temp", type=float, default=0.075, help="final temperature of the pilots")
    a = ap.parse_args()
    T, tab, graphs = load_pt(a.pt)
    print(f"PT: {len(graphs)} graphs {sorted(map(int, graphs))}, T {T[0]:.3f}..{T[-1]:.1f}, "
          f"{len(T)} levels; min swap acceptance "
          f"{min(min(g['swap_accept']) for g in graphs.values()):.3f}")
    hdr = " ".join(f"{s:>6}" for s in SHORT)
    for run in a.runs:
        print(f"\n== {Path(run).name}\n{'stage':>5} {'T':>7} | sampler {hdr} | PT rand {hdr} | "
              f"PT greedy {hdr}")
        for line in (Path(run) / "log.jsonl").read_text().splitlines():
            r = json.loads(line)
            if r["stage"] == 0:
                continue
            t = r.get("temp_train", a.temp)
            smp = " ".join(f"{r.get(k + '@N200', float('nan')):6.2f}" for k in KEYS)
            pt = [" ".join(f"{at(T, tab[st][k], t):6.2f}" for k in KEYS) for st in ("rand", "greedy")]
            print(f"{r['stage']:5d} {t:7.3f} |         {smp} |         {pt[0]} |           {pt[1]}")
    print("\nPT along the ladder (mean over graphs; rand / greedy start)")
    print(f"{'T':>7} " + " ".join(f"{s:>13}" for s in SHORT + ("empty",)))
    for i in range(0, len(T), 3):
        print(f"{T[i]:7.3f} " + " ".join(
            f"{tab['rand'][k][i]:6.2f}/{tab['greedy'][k][i]:6.2f}" for k in KEYS + ("raw_empty_frac",)))


if __name__ == "__main__":
    main()
