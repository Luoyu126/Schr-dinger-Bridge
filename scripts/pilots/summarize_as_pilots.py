"""Summarise the AS stability pilots (scripts/pilots/as_pilots.sh) against the pre-registered
criteria of docs/handoff_as_pilots.md.

Per pilot and stage: clip rates, control energy, and the validation metrics on the training
grid and at 200 steps (key suffix "@N200" unless 200 is the pilot's first --n_infer_steps).
Verdicts:
  stable  : clip_node_rate <= 0.10 and not rising after stage 1; control energy ratio between
            consecutive stages <= 2 and <= 2e4; E_tgt on the training grid <= 1e3 and within 3x
            of E_tgt at 200 steps.
  learned : at 200 steps E_tgt < 0, 15 <= raw_size_mean <= 40, n_distinct >= 8,
            mis_size_mean > empty-set decode baseline, best_of_16 > the pilot's own stage 0.

python scripts/pilots/summarize_as_pilots.py [runs/pilots/as_v2]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sbmis.amortized import GraphPool, ProblemSpec, decode_metrics  # noqa: E402
from sbmis.datasets import load_opt, load_split  # noqa: E402


def empty_set_baseline(meta: dict, n_graphs: int) -> float:
    """Decoder output from the all-off corner on the first n_graphs val graphs."""
    spec = ProblemSpec(**meta["spec"])
    val, opt = load_split("er_200_300", "val"), load_opt("er_200_300", "val")
    pool = GraphPool(val, spec, device="cpu", cache_size=len(val) + 1, n_max=meta["n_max"])
    b = pool.batch(np.arange(n_graphs))
    x = -torch.ones(b.S, b.mask.shape[1]) * b.mask
    return float(decode_metrics(pool, b, x, 1, opt)["mis_size_mean"])


def get(r: dict, key: str, sfx: str):
    return r.get(key + sfx, r.get(key) if sfx == "" else None)


def main(root: Path):
    runs = sorted(p for p in root.iterdir() if (p / "log.jsonl").exists())
    if not runs:
        raise SystemExit(f"no pilot logs under {root}")
    base = None
    for run in runs:
        meta = json.loads((run / "meta.json").read_text())
        steps = meta["args"]["n_infer_steps"]
        sfx200 = "" if steps[0] == 200 else "@N200"
        if base is None:
            n = meta["args"]["eval_graphs"] or 32
            base = empty_set_baseline(meta, n)
            print(f"empty-set decode baseline on the first {n} val graphs: {base:.3f}\n")
        recs = [json.loads(l) for l in (run / "log.jsonl").read_text().splitlines()]
        evals = [r for r in recs if "mis_size_mean" in r]
        print(f"== {run.name}  (train steps {meta['args']['n_train_steps']}, eval steps {steps})")
        print("st  clip_s clip_n  ctrlE      | E@train   E@200   raw@200 ndist@200 mean@200 best16@200")
        b0 = get(evals[0], "best_of_16", sfx200) if evals else None
        prev_ce, prev_cn, stable, learned = None, None, True, False
        for r in evals:
            cs, cn, ce = r.get("clip_rate"), r.get("clip_node_rate"), r.get("ctrl_energy_stage")
            e_tr, e200 = r.get("E_tgt_mean"), get(r, "E_tgt_mean", sfx200)
            raw, nd = get(r, "raw_size_mean", sfx200), get(r, "n_distinct", sfx200)
            mean, b16 = get(r, "mis_size_mean", sfx200), get(r, "best_of_16", sfx200)
            f = lambda v, w=7, p=2: f"{v:{w}.{p}f}" if isinstance(v, (int, float)) else " " * (w - 1) + "-"
            print(f"{r['stage']:2d}  {f(cs,6)} {f(cn,6)}  {f(ce,10,0)} | {f(e_tr,8,1)} {f(e200,7,1)} "
                  f"{f(raw,7,1)} {f(nd,8,1)} {f(mean,8,2)} {f(b16,9,2)}")
            if r["stage"] >= 1:
                if cn is None or cn > 0.10 or (prev_cn is not None and r["stage"] > 1 and cn > prev_cn + 0.02):
                    stable = False
                if ce is None or ce > 2e4 or (prev_ce and ce / prev_ce > 2):
                    stable = False
                if e_tr is None or e200 is None or e_tr > 1e3 or e_tr > 3 * max(abs(e200), 1.0):
                    stable = False
                learned = (e200 is not None and e200 < 0 and raw is not None and 15 <= raw <= 40
                           and nd is not None and nd >= 8 and mean is not None and mean > base
                           and b16 is not None and b0 is not None and b16 > b0)
                prev_ce, prev_cn = ce, cn
        status = json.loads((run / "summary.json").read_text())["status"] if (run / "summary.json").exists() else "running"
        print(f"-> status={status}  stable={stable}  learned(last stage)={learned}\n")


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "runs/pilots/as_v2"))
