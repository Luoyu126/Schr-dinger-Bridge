"""First-order adjoint labels for CAMPrior (local addition, see cam/UPSTREAM.md).

Derivation: docs/label_correction.md, section 3. Along one sampled trajectory X_0..X_K of the
policy p = sigmoid(a + r) (a = RLSA prior logit, r = network residual), the adjoint
alpha_j(i) = J_j(X_j + e_i) - J_j(X_j) is propagated backwards with
    alpha_K(i) = g(X_K + e_i) - g(X_K)
    alpha_j(i) = tau * dKL_j(i) + (1 - pi_i^(i)) alpha_{j+1}(i) [+ sum_{l: A_li != 0} pi_l^(i) alpha_{j+1}(l)]
where toggling i at step j changes the prior logits by
    da_l = beta_p A_li (1 - 2 x_i)(2 x_l - 1) / T_j   (neighbours),   da_i = -2 delta_i / T_j   (i itself)
(residual and top-d threshold held fixed), p'_l = sigmoid(z_l + da_l), and pi_l is the probability
that node l's flip at step j differs between the two coupled paths, given the observed flip F_l:
    F_l = 1: pi_l = max(0, 1 - p'_l / p_l),    F_l = 0: pi_l = max(0, 1 - (1 - p'_l)/(1 - p_l)).
dKL_j(i) = sum over l in {i} + N(i) of kappa(a_l + da_l, r_l) - kappa(a_l, r_l),
kappa(a, r) = KL(Ber(sigmoid(a + r)) || Ber(sigmoid(a))). The label of node i at step k is
    T_i(k) = -s_i(k) alpha_{k+1}(i),  s_i(k) = +1 if i was not flipped at step k, -1 if it was,
the same convention as the simulated labels (solvers/sim_labels.py).

The bracketed neighbour-reaction term (neighbors=True) is the discrete analogue of NAAS's Hessian
term. It makes the recursion explode: each step multiplies alpha by (1 - pi_i) + sum_l pi_l, which is
1.1-1.35 on average on the n = 12 check graphs, so labels reach 1e5-1e8 after 50 steps (first-order
additivity counts every cascade path separately, while the true difference set saturates). The
default (neighbors=False) keeps only i's own perturbation, absorbed with probability pi_i; its gain
is <= 1. On the small-graph check it has correlation 0.89-0.94 with the exact optimal label
(docs/label_correction.md, section 3.8).
"""
import torch
import torch.nn.functional as F


def kl_bern(z, a):
    """KL(Ber(sigmoid(z)) || Ber(sigmoid(a))), elementwise."""
    p = torch.sigmoid(z)
    return p*(F.logsigmoid(z) - F.logsigmoid(a)) + (1 - p)*(F.logsigmoid(-z) - F.logsigmoid(-a))


def differ_prob(z, dz, flipped):
    """P(flip decision differs | observed flip), coupled uniforms, logits z and z + dz."""
    one = torch.clamp(1 - torch.exp(F.logsigmoid(z + dz) - F.logsigmoid(z)), min=0)
    zero = torch.clamp(1 - torch.exp(F.logsigmoid(-z - dz) - F.logsigmoid(-z)), min=0)
    return torch.where(flipped > 0, one, zero)


def _step(alpha, A, b, x, z, a, flipped, temp, beta_prior, tau, neighbors=False):
    """alpha_j from alpha_{j+1}; x, z, a, flipped: n x C at step j; A dense n x n."""
    xt, zt, at, ft, al = x.T, z.T, a.T, flipped.T, alpha.T                  # C x n
    r = zt - at
    # i itself: delta_i changes sign
    grad = 2*beta_prior*(A@x) + b                                           # n x C
    delta = (grad*(2*x - 1)/2).T                                            # C x n
    d_own = -2*delta/temp
    pi_own = differ_prob(zt, d_own, ft)
    kl0 = kl_bern(zt, at)                                                   # C x n
    dkl_own = kl_bern(zt + d_own, at + d_own) - kl0
    # neighbours: [c, l, i]
    da = beta_prior*A.unsqueeze(0)*(2*xt - 1).unsqueeze(2)*(1 - 2*xt).unsqueeze(1)/temp
    zl, al_, fl = zt.unsqueeze(2), at.unsqueeze(2), ft.unsqueeze(2)
    dkl = ((kl_bern(zl + da, al_ + da) - kl0.unsqueeze(2))*(A.unsqueeze(0) != 0)).sum(1)
    out = tau*(dkl_own + dkl) + (1 - pi_own)*al
    if neighbors:
        pi = differ_prob(zl, da, fl)*(A.unsqueeze(0) != 0)
        out = out + (pi*al.unsqueeze(2)).sum(1)                             # C x n (over l)
    return out.T


@torch.no_grad()
def linear_labels(A, b, X, z, a, temps, beta_prior, beta, tau, chunk=5, neighbors=False):
    """First-order labels T_i(k) for all nodes and steps.

    A      (n, n) dense adjacency;  b (n, 1)
    X      (K+1, n, C) states X_0..X_K;  z, a (K, n, C) policy and prior logits at X_0..X_{K-1}
    temps  K prior temperatures T_j;  tau = tau_train (weight of the KL term)
    returns (K, n, C)
    """
    K, n, C = z.shape
    b = b.reshape(n, 1).to(X.dtype)
    out = torch.empty(K, n, C, device=X.device, dtype=X.dtype)
    for c0 in range(0, C, chunk):
        cs = slice(c0, min(C, c0 + chunk))
        xK = X[K][:, cs]
        alpha = (1 - 2*xK)*(2*beta*(A@xK) + b)
        for j in range(K - 1, -1, -1):
            flipped = (X[j + 1][:, cs] != X[j][:, cs]).to(X.dtype)
            out[j][:, cs] = -(1 - 2*flipped)*alpha
            if j > 0:
                alpha = _step(alpha, A, b, X[j][:, cs], z[j][:, cs], a[j][:, cs], flipped,
                              temps[j], beta_prior, tau, neighbors)
    return out
