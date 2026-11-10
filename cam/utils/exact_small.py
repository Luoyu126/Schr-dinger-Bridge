"""Exact quantities of the RLSA base process on small graphs (local addition, see cam/UPSTREAM.md).

For n <= ~12 nodes all 2^n states fit in memory, so the base (RLSA prior) transition of one step,
    P_j(x' | x) = prod_i q_i^{f_i} (1 - q_i)^{1 - f_i},  f = x XOR x',  q = RLSA flip probs at x, step j,
can be applied exactly. State index s = sum_i x_i 2^i.

    expected_energy   m_j(x) = E_base[g(X_K) | X_j = x]               (what one simulated label estimates)
    soft_value        V_j(x) = -tau log E_base[exp(-g(X_K)/tau) | X_j = x]
                      (the optimal cost-to-go of min E[g(X_K)] + tau KL(p || base), Feynman-Kac)
The flip label of node i for the move into x (= X_{k+1}) is then
    L(x, i) = F(x with i at its old value) - F(x with i flipped),  F = m_{k+1} or V_{k+1}.
"""
import torch
import torch.nn.functional as F


def all_states(n, dtype=torch.float64):
    s = torch.arange(2**n)
    return ((s.unsqueeze(1) >> torch.arange(n)) & 1).to(dtype)        # 2^n x n


def energies(A, b, beta, X):
    """g(x) = sum_i x_i (beta (A x)_i + b_i) for each row x of X."""
    return (X*(beta*(X@A.T) + b.reshape(1, -1))).sum(1)


def flip_logits(A, b, X, temp, d, eps, beta_prior):
    """RLSA flip logits at every state (rows of X), as solvers/rlsa.py and cam_prior.prior_logits."""
    grad = 2*beta_prior*(X@A.T) + b.reshape(1, -1)
    delta = grad*(2*X - 1)/2
    thr = torch.topk(delta, min(d, X.shape[1]), dim=1).values[:, -1:]
    return (delta - thr + eps)/temp


def _step_logprob(A, b, X, temp, d, eps, beta_prior):
    """log P(x -> x XOR f) as a 2^n x 2^n matrix (rows x, columns f)."""
    z = flip_logits(A, b, X, temp, d, eps, beta_prior)
    lp1, lp0 = F.logsigmoid(z), F.logsigmoid(-z)
    return lp0.sum(1, keepdim=True) + (lp1 - lp0)@X.T                    # X doubles as the flip masks


def backward_tables(A, b, temps, d, eps, beta_prior, beta, tau=None):
    """[F_0, ..., F_K] over all states: expected energy (tau None) or soft value at temperature tau."""
    n = A.shape[0]
    A, b = A.double(), b.double().reshape(-1)
    X = all_states(n).to(A.device)
    idx = torch.arange(2**n, device=A.device)
    xor = idx.unsqueeze(1) ^ idx.unsqueeze(0)                            # [x, f] -> x XOR f
    g = energies(A, b, beta, X)
    K = len(temps)
    tabs = [None]*(K + 1)
    tabs[K] = g if tau is None else -g/tau                              # log of exp(-g/tau) for the soft value
    for j in range(K - 1, -1, -1):
        lp = _step_logprob(A, b, X, temps[j], d, eps, beta_prior)
        nxt = tabs[j + 1][xor]
        tabs[j] = (lp.exp()*nxt).sum(1) if tau is None else torch.logsumexp(lp + nxt, dim=1)
    if tau is not None:
        tabs = [-tau*t for t in tabs]
    return tabs


def state_index(x):
    """x: (n,) or (n, C) 0/1 -> index (C,)"""
    x = torch.as_tensor(x).long()
    if x.dim() == 1:
        x = x.unsqueeze(1)
    return (x << torch.arange(x.shape[0]).unsqueeze(1)).sum(0)


def flip_label(table, x_next, sel, sign):
    """Exact label sign * (F(x_next) - F(x_next with node sel toggled)) for items (columns of x_next)."""
    s = state_index(x_next)                                              # C
    sel = torch.as_tensor(sel).long()                                    # C x m
    s_t = s.unsqueeze(1) ^ (1 << sel)
    return torch.as_tensor(sign, dtype=torch.float64)*(table[s].unsqueeze(1) - table[s_t])
