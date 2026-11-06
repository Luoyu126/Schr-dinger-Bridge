"""Quick tuning screens that re-check last round's conclusions (plan §0.5.5).

S1  U1 g2  network: {gnn, mlp} x rho {0, 4} x 2 seeds
S2  U1 g2  lr {1e-4, 3e-4, 1e-3} x rho {0, 1, 4} x 3 seeds (GNN)
S3  U1 g2  rho = 4 stabilisation: am_precond (EXPERIMENTAL) and grad-norm clip, 3 seeds
S4  n=100  lr {1e-4, 3e-4, 1e-3} x rho {0, 2} x 2 seeds, plus lr 1e-3 + grad clip

Common: AM:CM = 30:30 epochs, 512 fresh samples / epoch, alpha_max = 100,
CM first, sigma = 1. Writes runs/<screen>/jobs.tsv for scripts/launch.py.

python scripts/sweeps/quick_screen.py --kappa 2 --temp 0.5
"""
import argparse
from pathlib import Path

PY = "/usr0/home/shengyuf/yunyi/envs/sb/bin/python"


def job(root, name, **kw):
    args = " ".join(f"--{k} {v}" if v is not True else f"--{k}" for k, v in kw.items())
    return f"{name}\t{PY} scripts/train_mis.py {args} --out {root}/{name}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kappa", type=float, default=2.0)
    ap.add_argument("--temp", type=float, default=0.5)
    ap.add_argument("--u1_graph", default="g2")
    ap.add_argument("--u1_stages", type=int, default=20)
    ap.add_argument("--n100_stages", type=int, default=10)
    a = ap.parse_args()
    ref = f"runs/u1_ref/{a.u1_graph}_k{a.kappa:g}_T{a.temp:g}.pt"
    u1 = dict(graph=a.u1_graph, kappa_w=a.kappa, temp=a.temp, ref=ref, stages=a.u1_stages,
              eval_every=2)
    screens = {}

    def base_kw(rho):
        return dict(base="zero") if rho == 0 else dict(base="linear_energy", rho=rho)

    r = "runs/s1_net"
    screens[r] = [job(r, f"{net}_rho{rho}_s{s}", net=net, lr=3e-4, seed=s, **u1, **base_kw(rho))
                  for net in ["gnn", "mlp"] for rho in [0, 4] for s in [0, 1]]
    r = "runs/s2_lr"
    screens[r] = [job(r, f"lr{lr:g}_rho{rho}_s{s}", net="gnn", lr=lr, seed=s, **u1, **base_kw(rho))
                  for lr in [1e-4, 3e-4, 1e-3] for rho in [0, 1, 4] for s in [0, 1, 2]]
    r = "runs/s3_rho4"
    screens[r] = (
        [job(r, f"precond_lr{lr:g}_s{s}", net="gnn", lr=lr, seed=s, am_precond=True, **u1,
             **base_kw(4)) for lr in [3e-4, 1e-3] for s in [0, 1, 2]]
        + [job(r, f"gclip1_lr1e-3_s{s}", net="gnn", lr=1e-3, seed=s, grad_clip=1.0, **u1,
               **base_kw(4)) for s in [0, 1, 2]])
    n100 = dict(graph="er:100:0.15:0", kappa_w=a.kappa, temp=a.temp, stages=a.n100_stages,
                n_infer_steps="20 100", eval_every=1)
    r = "runs/s4_n100_lr"
    screens[r] = (
        [job(r, f"lr{lr:g}_rho{rho}_s{s}", net="gnn", lr=lr, seed=s, **n100, **base_kw(rho))
         for lr in [1e-4, 3e-4, 1e-3] for rho in [0, 2] for s in [0, 1]]
        + [job(r, f"gclip1_lr1e-3_rho{rho}_s{s}", net="gnn", lr=1e-3, grad_clip=1.0, seed=s,
               **n100, **base_kw(rho)) for rho in [0, 2] for s in [0, 1]])
    for root, jobs in screens.items():
        Path(root).mkdir(parents=True, exist_ok=True)
        (Path(root) / "jobs.tsv").write_text("\n".join(jobs) + "\n")
        print(root, len(jobs))


if __name__ == "__main__":
    main()
