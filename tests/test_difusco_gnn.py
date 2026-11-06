"""DIFUSCO network port: node-permutation equivariance, independence of batched samples,
zero-initialised output (u = 0 at start) and the fp16 path."""
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from bench_amortized_gnn import make_batch  # noqa: E402
from sbmis.models.difusco_gnn import DifuscoGNN  # noqa: E402


def _net(seed=0):
    torch.manual_seed(seed)
    net = DifuscoGNN(hidden=32, n_layers=3)
    for m in net.modules():  # give the zero-initialised layers weights for the checks
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.normal_(m.weight, std=0.3)
    return net.double()


def test_zero_initialised_output():
    b = make_batch(20, 3, device="cpu")
    out = DifuscoGNN(hidden=32, n_layers=3)(torch.rand(3), torch.randn(b.n_nodes), b)
    assert out.shape == (b.n_nodes,) and (out == 0).all()


def test_permutation_equivariance():
    net = _net()
    b = make_batch(15, 1, device="cpu")
    x, t = torch.randn(b.n_nodes, dtype=torch.float64), torch.rand(1)
    perm = torch.randperm(b.n_nodes)
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(b.n_nodes)
    bp = SimpleNamespace(src=inv[b.src], dst=inv[b.dst], node2sample=b.node2sample)
    out, outp = net(t, x, b), net(t, x[perm], bp)
    assert (outp - out[perm]).abs().max() < 1e-10


def test_batched_samples_are_independent():
    net = _net()
    b = make_batch(12, 4, device="cpu")
    x, t = torch.randn(b.n_nodes, dtype=torch.float64), torch.rand(4)
    out = net(t, x, b)
    for s in range(4):
        m = b.node2sample == s
        idx = torch.nonzero(m).squeeze(1)
        keep = m[b.src] & m[b.dst]
        sub = SimpleNamespace(src=b.src[keep] - idx[0], dst=b.dst[keep] - idx[0],
                              node2sample=torch.zeros(len(idx), dtype=torch.long))
        assert (net(t[s:s + 1], x[m], sub) - out[m]).abs().max() < 1e-10


def test_fp16_autocast_runs():
    if not torch.cuda.is_available():
        return
    net = _net().float().cuda()
    b = make_batch(30, 2, device="cuda")
    with torch.autocast("cuda", dtype=torch.float16):
        out = net(torch.rand(2, device="cuda"), torch.randn(b.n_nodes, device="cuda"), b)
    assert torch.isfinite(out).all()
