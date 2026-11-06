"""Train an amortized ASBS sampler on a graph dataset (plan U7).

python scripts/train_amortized.py --dataset er_200_300 --base linear_energy_shift --rho 4 \
    --out runs/amortized/er_200_300/B1p_rho4_s0

--single_graph er:100:0.15:0 trains on that one graph instead (sanity check against the
single-instance runs). Validation: every graph of the val split, --eval_k samples each, at
every --n_infer_steps (the first one is reported without suffix).
"""
import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sbmis.amortized import (AmortizedConfig, AmortizedTrainer, AnalyticCorrectorAmortized,  # noqa: E402
                             GraphPool, ProblemSpec, decode_metrics)
from sbmis.datasets import GraphSet, load_opt, load_split  # noqa: E402
from sbmis.graphs import best_known_mis, get_graph  # noqa: E402
from sbmis.models.difusco_gnn import DifuscoGNN  # noqa: E402

BASES = ["zero", "linear_energy", "linear_energy_shift", "linear_energy_shift_rot", "iso_ou",
         "vp", "ve_geometric"]


def graphset_from_adj(adj):
    n = adj.shape[0]
    iu = np.triu_indices(n, 1)
    e = np.stack(iu, 1)[adj[iu] > 0].astype(np.int32)
    return GraphSet(np.array([n], dtype=np.int32), np.array([0, len(e)]), e)


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="er_200_300")
    ap.add_argument("--single_graph", default=None)
    # problem / reference process / source (same meaning as train_mis.py)
    ap.add_argument("--base", choices=BASES, default="zero")
    ap.add_argument("--rho", type=float, default=0.0)
    ap.add_argument("--sigma", type=float, default=1.0)
    ap.add_argument("--sigma_min", type=float, default=0.001)
    ap.add_argument("--rot_seed", type=int, default=None, help="default: --seed")
    ap.add_argument("--iso_match_rho", type=float, default=None)
    ap.add_argument("--iso_match", choices=["var", "m2"], default="var")
    ap.add_argument("--vp_beta0", type=float, default=20.0)
    ap.add_argument("--vp_beta1", type=float, default=0.1)
    ap.add_argument("--source", choices=["gauss", "delta", "energy_prior"], default="gauss")
    ap.add_argument("--prior_gamma", type=float, default=4.0)
    ap.add_argument("--analytic_corrector", action="store_true")
    ap.add_argument("--lambda_p", type=float, default=2.0)
    ap.add_argument("--kappa_w", type=float, default=2.0)
    ap.add_argument("--temp", type=float, default=0.5)
    ap.add_argument("--alpha_max", type=float, default=100.0)
    # network (DIFUSCO gated GCN, shared by u and h)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--no_amp", action="store_true")
    # schedule
    ap.add_argument("--graphs_per_epoch", type=int, default=32)
    ap.add_argument("--samples_per_graph", type=int, default=8)
    ap.add_argument("--n_train_steps", type=int, default=25)
    ap.add_argument("--train_batch", type=int, default=64)
    ap.add_argument("--itr_per_epoch", type=int, default=50)
    ap.add_argument("--buffer_size", type=int, default=10_000)
    ap.add_argument("--am_epochs", type=int, default=30)
    ap.add_argument("--cm_epochs", type=int, default=30)
    ap.add_argument("--init", choices=["cm_first", "am_first"], default="cm_first")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--grad_clip", type=float, default=None)
    ap.add_argument("--stages", type=int, default=20)
    ap.add_argument("--max_grad_calls", type=int, default=None)
    ap.add_argument("--cache_size", type=int, default=3072)
    # evaluation
    ap.add_argument("--eval_every", type=int, default=1)
    ap.add_argument("--eval_k", type=int, default=16)
    ap.add_argument("--eval_graphs", type=int, default=None, help="default: whole val split")
    ap.add_argument("--n_infer_steps", type=int, nargs="+", default=[25, 100])
    ap.add_argument("--save_at", type=int, nargs="*", default=[])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    return ap.parse_args()


def main():
    a = parse()
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    dev = "cuda"
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for f in ("log.jsonl", "summary.json", "epochs.jsonl", "final.pt"):
        (out / f).unlink(missing_ok=True)
    for f in out.glob("ckpt_*.pt"):
        f.unlink()
    if a.single_graph:
        adj = get_graph(a.single_graph)
        train = val = graphset_from_adj(adj)
        val_opt = [{"size": best_known_mis(a.single_graph, adj)["size"]}]
    else:
        train, val = load_split(a.dataset, "train"), load_split(a.dataset, "val")
        val_opt = load_opt(a.dataset, "val")
    spec = ProblemSpec(lambda_p=a.lambda_p, kappa_w=a.kappa_w, temp=a.temp, base=a.base,
                       rho=a.rho, sigma=a.sigma, rot_seed=a.seed if a.rot_seed is None else a.rot_seed,
                       iso_match_rho=a.iso_match_rho, iso_match=a.iso_match,
                       sigma_min=a.sigma_min, vp_beta0=a.vp_beta0, vp_beta1=a.vp_beta1,
                       source=a.source, prior_gamma=a.prior_gamma)
    n_max = int(max(train.n.max(), val.n.max()))
    pool = GraphPool(train, spec, device=dev, cache_size=a.cache_size, n_max=n_max)
    vpool = GraphPool(val, spec, device=dev, cache_size=len(val) + 1, n_max=n_max)
    u = DifuscoGNN(hidden=a.hidden, n_layers=a.layers)
    if a.analytic_corrector:
        assert a.source == "delta"
        h = AnalyticCorrectorAmortized(pool)
    else:
        h = DifuscoGNN(hidden=a.hidden, n_layers=a.layers)
    cfg = AmortizedConfig(
        graphs_per_epoch=a.graphs_per_epoch, samples_per_graph=a.samples_per_graph,
        n_train_steps=a.n_train_steps, train_batch=a.train_batch, itr_per_epoch=a.itr_per_epoch,
        buffer_size=a.buffer_size, am_epochs=a.am_epochs, cm_epochs=a.cm_epochs,
        init_stage="corrector" if a.init == "cm_first" else "adjoint", lr=a.lr,
        grad_clip=a.grad_clip, alpha_max=a.alpha_max, max_stages=a.stages,
        max_grad_calls=a.max_grad_calls, eval_every_stages=a.eval_every,
        fixed_corrector=a.analytic_corrector, amp=not a.no_amp, save_at=tuple(a.save_at),
        seed=a.seed)
    eval_ids = np.arange(len(val) if a.eval_graphs is None else min(a.eval_graphs, len(val)))

    def eval_fn(tr):
        res = {}
        pool_train = tr.pool
        tr.pool = vpool  # sample on validation graphs with the same networks
        try:
            for N in a.n_infer_steps:
                rows = []
                for i in range(0, len(eval_ids), 8):
                    b, x = tr.sample_graphs(eval_ids[i:i + 8], a.eval_k, N)
                    m = decode_metrics(vpool, b, x, a.eval_k, val_opt)
                    m["E_tgt_mean"] = float(vpool.e_tgt(x, b).mean())
                    rows.append((m, len(eval_ids[i:i + 8])))
                tot = sum(w for _, w in rows)
                m = {k: sum(r[k] * w for r, w in rows) / tot for k in rows[0][0] if k != "n_graphs"}
                sfx = "" if N == a.n_infer_steps[0] else f"@N{N}"
                res.update({k + sfx: v for k, v in m.items()})
        finally:
            tr.pool = pool_train
        return res

    meta = {"args": vars(a), "cfg": asdict(cfg), "spec": asdict(spec), "n_max": n_max,
            "n_train_graphs": len(train), "n_val_graphs": len(val),
            "n_params": sum(p.numel() for p in u.parameters())}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta), flush=True)
    tr = AmortizedTrainer(pool, u, h, cfg, device=dev, out_dir=out, eval_fn=eval_fn)
    print(json.dumps(tr.run()), flush=True)


if __name__ == "__main__":
    main()
