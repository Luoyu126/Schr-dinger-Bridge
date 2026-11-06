"""Train ASBS on a single MIS instance.

Examples
  # U1 (small graph, compared to the PT-MALA reference)
  python scripts/train_mis.py --graph g2 --base linear_energy --rho 1 \
      --ref runs/u1_ref/g2_k2_T0.5.pt --out runs/u1/g2_rho1_s0
  # n = 100
  python scripts/train_mis.py --graph er:100:0.15:0 --base zero --out runs/n100/b0_s0
"""
import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sbmis.asbs import ASBSConfig, ASBSTrainer  # noqa: E402
from sbmis.base.linear_energy_base import build_base  # noqa: E402
from sbmis.energies.mis import MISEnergy  # noqa: E402
from sbmis.evaluate import Decoder, discrete_metrics, mis_metrics, tv_noise_floor  # noqa: E402
from sbmis.graphs import exact_mis_size, get_graph  # noqa: E402
from sbmis.models.nets import build_net  # noqa: E402


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", default="g2")
    ap.add_argument("--base", choices=["zero", "linear_energy", "ve_geometric"], default="zero")
    ap.add_argument("--rho", type=float, default=0.0)
    ap.add_argument("--sigma", type=float, default=1.0)
    ap.add_argument("--sigma_min", type=float, default=0.01, help="ve_geometric only")
    ap.add_argument("--lambda_p", type=float, default=2.0)
    ap.add_argument("--kappa_w", type=float, default=2.0)
    ap.add_argument("--temp", type=float, default=0.5)
    ap.add_argument("--alpha_max", type=float, default=100.0)
    ap.add_argument("--no_alpha_sqrt_n", action="store_true")
    ap.add_argument("--n_infer_steps", type=int, nargs="+", default=[100])
    ap.add_argument("--n_train_steps", type=int, default=100)
    ap.add_argument("--net", choices=["gnn", "mlp"], default="gnn")
    ap.add_argument("--hidden", type=int, default=None)
    ap.add_argument("--layers", type=int, default=None)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--grad_clip", type=float, default=None)
    ap.add_argument("--batch_size", type=int, default=512, help="fresh samples per epoch")
    ap.add_argument("--train_batch", type=int, default=512)
    ap.add_argument("--itr_per_epoch", type=int, default=100)
    ap.add_argument("--buffer_size", type=int, default=10_000)
    ap.add_argument("--am_epochs", type=int, default=30)
    ap.add_argument("--cm_epochs", type=int, default=30)
    ap.add_argument("--init", choices=["cm_first", "am_first"], default="cm_first")
    ap.add_argument("--stages", type=int, default=20)
    ap.add_argument("--max_grad_calls", type=int, default=None)
    ap.add_argument("--eval_every", type=int, default=1, help="stages")
    ap.add_argument("--eval_samples", type=int, default=None)
    ap.add_argument("--ref", default=None, help="U1 reference .pt (small graphs)")
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
    for f in ("log.jsonl", "summary.json"):
        (out / f).unlink(missing_ok=True)

    adj = get_graph(a.graph)
    n = adj.shape[0]
    en = MISEnergy(adj, a.lambda_p, a.kappa_w, a.temp, device=dev)
    base = build_base(a.base, en, sigma=a.sigma, rho=a.rho, device=dev, sigma_min=a.sigma_min)
    kw = {k: v for k, v in (("hidden", a.hidden), ("layers", a.layers)) if v is not None}
    u, h = build_net(a.net, adj, **kw), build_net(a.net, adj, **kw)
    cfg = ASBSConfig(
        n_train_steps=a.n_train_steps, resample_size=a.batch_size,
        sample_batch=min(a.batch_size, 1024), train_batch=a.train_batch,
        itr_per_epoch=a.itr_per_epoch, buffer_size=a.buffer_size,
        am_epochs=a.am_epochs, cm_epochs=a.cm_epochs,
        init_stage="corrector" if a.init == "cm_first" else "adjoint",
        lr=a.lr, grad_clip=a.grad_clip, alpha_max=a.alpha_max,
        alpha_sqrt_n_scale=not a.no_alpha_sqrt_n,
        max_stages=a.stages, max_grad_calls=a.max_grad_calls, eval_every_stages=a.eval_every)

    dec = Decoder(adj, device=dev)
    ref = None
    if a.ref:
        ref = torch.load(a.ref)
        s = ref["summary"]
        assert s["graph"] == a.graph and s["kappa_w"] == a.kappa_w and s["temp"] == a.temp
        n_eval = a.eval_samples or 20_000
        ref["tv_floor"] = tv_noise_floor(ref["p"], n_eval)
        opt = s["mis_size"]
    else:
        n_eval = a.eval_samples or 1024
        opt = exact_mis_size(adj)[0] if n <= 400 else None

    def eval_fn(tr):
        res = {}
        for N in a.n_infer_steps:
            x = tr.sample(n_eval, N)
            m = mis_metrics(x, en, dec, opt_size=opt)
            if ref is not None:
                m.update(discrete_metrics(x, ref["p"], ref["mis_codes"].to(dev),
                                          ref["feasible"].to(dev)))
            sfx = "" if N == a.n_infer_steps[0] else f"@N{N}"
            res.update({k + sfx: v for k, v in m.items()})
        return res

    meta = {"args": vars(a), "cfg": asdict(cfg), "n": n, "n_edges": en.n_edges, "opt_size": opt,
            "rho_effective": base.rho, "lam": base.lam, "alpha_eff": None,
            "n_params": sum(p.numel() for p in u.parameters())}
    if ref is not None:
        meta.update(ref_summary=ref["summary"], tv_floor=ref["tv_floor"])
    tr = ASBSTrainer(en, base, u, h, cfg, device=dev, out_dir=out, eval_fn=eval_fn)
    meta["alpha_eff"] = tr.alpha_eff
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta))
    print(json.dumps(tr.run()))


if __name__ == "__main__":
    main()
