"""Bounded encodings p(x) of the MIS energy (sbmis/energies/mis.py, `pmap`)."""
import numpy as np
import pytest
import torch

from sbmis.energies.mis import MISEnergy
from sbmis.graphs import er_graph

F64 = torch.float64
PMAPS = ["lin", "clamp", "smooth"]


def _en(pmap, n=12, seed=0, well="full"):
    return MISEnergy(er_graph(n, 0.35, seed=seed), lambda_p=2.0, kappa_w=2.0, temp=0.1,
                     device="cpu", dtype=F64, pmap=pmap, well=well)


def test_corners_agree_across_pmaps():
    g = torch.Generator().manual_seed(0)
    x = torch.randint(0, 2, (64, 12), generator=g).to(F64) * 2 - 1
    ref = _en("lin").e_tgt(x)
    for pm in PMAPS[1:]:
        assert (_en(pm).e_tgt(x) - ref).abs().max() < 1e-12
    assert (_en("clamp", well="out").e_tgt(x) - ref).abs().max() < 1e-12


@pytest.mark.parametrize("pmap,well", [(p, "full") for p in PMAPS] + [("clamp", "out")])
def test_gradient_matches_autodiff(pmap, well):
    en = _en(pmap, well=well)
    g = torch.Generator().manual_seed(1)
    x = 1.4 * torch.randn(32, en.n, generator=g, dtype=F64)
    x = torch.where((x.abs() - 1).abs() < 1e-3, x + 1e-2, x)  # stay off the kinks at |x| = 1
    x.requires_grad_(True)
    want, = torch.autograd.grad(en.e_tgt(x).sum(), x)
    assert (want - en.grad_tgt(x.detach())).abs().max() < 1e-9
    assert (en.grad_u(x.detach(), count=False) - want / en.temp).abs().max() < 1e-8


@pytest.mark.parametrize("pmap", ["clamp", "smooth"])
def test_bounded_p_removes_the_negative_p_discount(pmap):
    """Pushing an unselected node below -1 must not lower E_quad for a bounded pmap
    (with lin it does whenever the node has selected neighbours)."""
    en_lin, en_b = _en("lin"), _en(pmap)
    A = en_lin.A64.numpy()
    i = int(np.argmax(A.sum(1)))
    x = -torch.ones(1, en_b.n, dtype=F64)
    x[0, np.nonzero(A[i])[0]] = 1.0          # neighbours of i selected (conflicts allowed)
    y = x.clone()
    y[0, i] = -1.35
    assert en_lin.e_quad(y) < en_lin.e_quad(x)
    assert en_b.e_quad(y) == en_b.e_quad(x)


def test_reference_matrices_do_not_depend_on_pmap():
    a, b = _en("lin"), _en("smooth")
    assert torch.equal(a.M64, b.M64) and torch.equal(a.c64, b.c64)


def test_clamp_with_outside_well_is_multilinear_in_the_box():
    """Inside [-1, 1]^n the clamp + outside-well energy is affine in each coordinate, so it has
    no interior local minima; its only local minima are corners."""
    en = _en("clamp", well="out")
    g = torch.Generator().manual_seed(2)
    x = (2 * torch.rand(16, en.n, generator=g, dtype=F64) - 1) * 0.9
    i = 3
    vals = []
    for t in (-0.9, 0.0, 0.9):
        y = x.clone(); y[:, i] = t
        vals.append(en.e_tgt(y))
    assert ((vals[1] - 0.5 * (vals[0] + vals[2])).abs() < 1e-10).all()


def test_conflicting_corner_is_not_a_local_minimum_with_outside_well():
    """At a corner where a selected node has a selected neighbour, moving that node into the box
    lowers the energy (lambda_p > 1); with the full double well this direction is blocked."""
    en_out, en_full = _en("clamp", well="out"), _en("clamp", well="full")
    A = en_out.A64.numpy()
    i = int(np.argmax(A.sum(1))); j = int(np.nonzero(A[i])[0][0])
    x = -torch.ones(1, en_out.n, dtype=F64)
    x[0, i] = x[0, j] = 1.0
    y = x.clone(); y[0, i] = 0.9
    assert en_out.e_tgt(y) < en_out.e_tgt(x)
    assert en_full.e_tgt(y) > en_full.e_tgt(x)
