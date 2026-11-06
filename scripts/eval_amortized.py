"""Evaluate trained amortized samplers on a test split (large-scale sampling).

For every run directory and checkpoint: K_max samples per test graph at each number of
inference steps N; metrics for every K in --k (best-of-K uses the first K samples), plus
sampling wall-clock per graph. Writes <run>/eval_<split>_<ckpt>.json. Intermediate
checkpoints (--ckpt ckpt_30720.pt ...) can be evaluated while the run is still training.

python scripts/eval_amortized.py runs/amortized/er_200_300/*/ --split test --k 1 4 16 64 \
    --n_infer_steps 10 25 50 100
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sbmis.amortized import (AmortizedConfig, AmortizedTrainer, AnalyticCorrectorAmortized,  # noqa: E402
                             GraphPool, ProblemSpec, decode_metrics)
from sbmis.datasets import load_opt, load_split  # noqa: E402
from sbmis.models.difusco_gnn import DifuscoGNN  # noqa: E402


def evaluate_run(run: Path, ckpt: str, split: str, ks, steps, graphs_per_batch, max_graphs):
    meta = json.loads((run / "meta.json").read_text())
    a = meta["args"]
    spec = ProblemSpec(**meta["spec"])
    gset, opt = load_split(a["dataset"], split), load_opt(a["dataset"], split)
    n = len(gset) if max_graphs is None else min(max_graphs, len(gset))
    pool = GraphPool(gset, spec, device="cuda", cache_size=len(gset) + 1,
                     n_max=max(int(gset.n.max()), meta["n_max"]))
    u = DifuscoGNN(hidden=a["hidden"], n_layers=a["layers"]).cuda()
    h = AnalyticCorrectorAmortized(pool) if a["analytic_corrector"] else \
        DifuscoGNN(hidden=a["hidden"], n_layers=a["layers"]).cuda()
    sd = torch.load(run / ckpt, map_location="cuda")
    u.load_state_dict(sd["u"])
    if not a["analytic_corrector"]:
        h.load_state_dict(sd["h"])
    cfg = AmortizedConfig(**{k: v for k, v in meta["cfg"].items()
                             if k in AmortizedConfig.__dataclass_fields__})
    cfg.save_at = ()
    tr = AmortizedTrainer(pool, u, h, cfg, device="cuda", log=lambda *_: None)
    torch.manual_seed(12345)
    kmax = max(ks)
    out = {"run": str(run), "ckpt": ckpt, "grad_calls": sd.get("grad_calls"), "split": split,
           "n_graphs": n, "results": {}}
    for N in steps:
        xs, bs, secs = [], [], 0.0
        for i in range(0, n, graphs_per_batch):
            ids = np.arange(i, min(i + graphs_per_batch, n))
            torch.cuda.synchronize()
            t0 = time.time()
            b, x = tr.sample_graphs(ids, kmax, N)
            torch.cuda.synchronize()
            secs += time.time() - t0
            xs.append((b, x))
        res = {"sample_sec_per_graph": secs / n}
        for K in ks:
            rows = []
            for b, x in xs:
                # first K of the kmax samples of each graph
                G = len(b.gids) // kmax
                sel = (np.arange(G)[:, None] * kmax + np.arange(K)[None, :]).reshape(-1)
                sub = type(b)(**{**vars(b), "gids": b.gids[sel]})
                rows.append((decode_metrics(pool, sub, x[torch.as_tensor(sel, device=x.device)], K, opt), G))
            tot = sum(w for _, w in rows)
            res[f"K{K}"] = {k: sum(r[k] * w for r, w in rows) / tot for k in rows[0][0]
                            if k != "n_graphs"}
        e = torch.cat([pool.e_tgt(x, b) for b, x in xs])
        res["E_tgt_mean"] = float(e.mean())
        out["results"][f"N{N}"] = res
        print(json.dumps({"run": run.name, "ckpt": ckpt, "N": N,
                          **{f"K{K}_ratio_mean": round(res[f'K{K}']['ratio_mean'], 4) for K in ks},
                          **{f"K{K}_ratio_best": round(res[f'K{K}']['ratio_best'], 4) for K in ks}}),
              flush=True)
    (run / f"eval_{split}_{Path(ckpt).stem}.json").write_text(json.dumps(out, indent=1))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--ckpt", nargs="+", default=["final.pt"])
    ap.add_argument("--split", default="test")
    ap.add_argument("--k", type=int, nargs="+", default=[1, 4, 16, 64])
    ap.add_argument("--n_infer_steps", type=int, nargs="+", default=[25, 100])
    ap.add_argument("--graphs_per_batch", type=int, default=4)
    ap.add_argument("--max_graphs", type=int, default=None)
    a = ap.parse_args()
    for r in a.runs:
        run = Path(r)
        for c in a.ckpt:
            if ready(run, c):
                evaluate_run(run, c, a.split, a.k, a.n_infer_steps, a.graphs_per_batch, a.max_graphs)
            else:
                print(f"skip {run}/{c}: missing checkpoint or unfinished run", flush=True)


def ready(run: Path, ckpt: str, settle_sec: float = 60.0) -> bool:
    """final.pt only after the run finished; an intermediate ckpt_<calls>.pt once it has been
    on disk for settle_sec (it is written in one torch.save while training goes on)."""
    f = run / ckpt
    if not f.exists():
        return False
    if ckpt == "final.pt":
        return (run / "summary.json").exists()
    return time.time() - f.stat().st_mtime > settle_sec


if __name__ == "__main__":
    main()
