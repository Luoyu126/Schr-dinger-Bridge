"""T6: end-to-end ASBS with f != 0 on a Gaussian toy (plan §6).

mu = N(0, I), f(x) = -theta0 x, nu = N(2, 0.5^2 I). After training, terminal
samples must match nu: |mean err| < 0.05, |std rel err| < 5%.

Slow (trains a model): `pytest -m slow tests/test_t6_toy.py`, or run as a
script: `python tests/test_t6_toy.py --dim 2 --theta0 2`.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sbmis.asbs import ASBSConfig, ASBSTrainer  # noqa: E402
from sbmis.base.linear_energy_base import LinearEnergyBase  # noqa: E402
from sbmis.energies.mis import GaussianEnergy  # noqa: E402
from sbmis.models.nets import FourierMLP  # noqa: E402

M_NU, S_NU = 2.0, 0.5


def run_t6(dim, theta0, stages=10, seed=0, device="cuda", lr=3e-4, quiet=True,
           grad_clip=None, out_dir=None):
    torch.manual_seed(seed)
    np.random.seed(seed)
    en = GaussianEnergy(dim, M_NU, S_NU, device=device)
    base = LinearEnergyBase(np.eye(dim), np.zeros(dim), sigma=1.0, lam=theta0, device=device)
    cfg = ASBSConfig(resample_size=512, am_epochs=20, cm_epochs=20, itr_per_epoch=100,
                     lr=lr, max_stages=stages, alpha_sqrt_n_scale=False, init_stage="adjoint",
                     grad_clip=grad_clip)
    if out_dir:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        for f in Path(out_dir).glob("*.jsonl"):
            f.unlink()
    tr = ASBSTrainer(en, base, FourierMLP(dim, channels=128), FourierMLP(dim, channels=128),
                     cfg, device=device, out_dir=out_dir,
                     log=(lambda *_: None) if quiet else print)
    summ = tr.run()
    x = tr.sample(50_000, 100)
    mean, std = x.mean(0), x.std(0)
    return {
        "dim": dim, "theta0": theta0, "seed": seed, "lr": lr, "grad_clip": grad_clip,
        "stages": stages, "status": summ["status"],
        "max_mean_err": (mean - M_NU).abs().max().item(),
        "max_std_relerr": ((std - S_NU).abs() / S_NU).max().item(),
        "mean": mean.tolist(), "std": std.tolist(),
    }


@pytest.mark.slow
@pytest.mark.parametrize("dim", [1, 2])
@pytest.mark.parametrize("theta0", [-1.0, 0.5, 2.0])
def test_t6(dim, theta0):
    r = run_t6(dim, theta0)
    assert r["max_mean_err"] < 0.05 and r["max_std_relerr"] < 0.05, r


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, default=1)
    ap.add_argument("--theta0", type=float, default=2.0)
    ap.add_argument("--stages", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--grad_clip", type=float, default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    r = run_t6(a.dim, a.theta0, a.stages, a.seed, lr=a.lr, quiet=False,
               grad_clip=a.grad_clip, out_dir=a.out)
    r["pass"] = r["max_mean_err"] < 0.05 and r["max_std_relerr"] < 0.05
    print("RESULT", json.dumps(r))
