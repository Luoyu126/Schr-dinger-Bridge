"""Throughput of the DIFUSCO network on ER graphs, to size the amortized ASBS runs.

One ASBS AM epoch = simulate `resample` trajectories with `n_steps` network evaluations each
(no grad) + `itr` optimiser steps on minibatches of `train_batch` samples (u: fwd + bwd).
CM epochs cost about the same. Reported per config: inference and training time per sample,
peak memory, and the implied minutes per AM epoch.

CUDA_VISIBLE_DEVICES=0 python scripts/bench_amortized_gnn.py --n 250 750
"""
import argparse
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import networkx as nx
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sbmis.models.difusco_gnn import DifuscoGNN  # noqa: E402


def make_batch(n, S, p=0.15, seed=0, device="cuda"):
    """S samples on S independent ER(n, p) graphs, as one disjoint union."""
    src, dst, n2s, off = [], [], [], 0
    for s in range(S):
        g = nx.fast_gnp_random_graph(n, p, seed=seed + s)
        e = np.array(g.edges(), dtype=np.int64).reshape(-1, 2) + off
        src += [e[:, 0], e[:, 1]]
        dst += [e[:, 1], e[:, 0]]
        n2s.append(np.full(n, s))
        off += n
    cat = lambda a: torch.tensor(np.concatenate(a), device=device)
    return SimpleNamespace(src=cat(src), dst=cat(dst), node2sample=cat(n2s), n_nodes=off,
                           n_edges=sum(len(a) for a in src))


def timeit(fn, reps):
    fn()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / reps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, nargs="+", default=[250, 750])
    ap.add_argument("--configs", nargs="+", default=["12x256", "12x128", "8x128", "6x64"])
    ap.add_argument("--amp", action="store_true", help="fp16 autocast")
    ap.add_argument("--resample", type=int, default=512)
    ap.add_argument("--n_steps", type=int, default=100)
    ap.add_argument("--itr", type=int, default=100)
    a = ap.parse_args()
    dev = "cuda"
    print("| n | config | amp | infer ms/sample | train ms/sample | infer S | train S | peak GB |"
          " min / AM epoch (bs=64) |")
    print("|---|---|---|---|---|---|---|---|---|")
    for n in a.n:
        for cfg in a.configs:
            L, H = (int(v) for v in cfg.split("x"))
            net = DifuscoGNN(hidden=H, n_layers=L).to(dev)
            res = {}
            for mode in ("infer", "train"):
                S = 64 if n <= 300 else 16
                while S >= 1:
                    try:
                        torch.cuda.empty_cache()
                        torch.cuda.reset_peak_memory_stats()
                        b = make_batch(n, S, device=dev)
                        x = torch.randn(b.n_nodes, device=dev)
                        t = torch.rand(S, device=dev)

                        def step():
                            with torch.autocast("cuda", dtype=torch.float16, enabled=a.amp):
                                if mode == "infer":
                                    with torch.no_grad():
                                        net(t, x, b)
                                else:
                                    out = net(t, x, b)
                                    out.float().pow(2).mean().backward()
                        dt = timeit(step, 3)
                        res[mode] = (dt / S, S, torch.cuda.max_memory_allocated() / 2**30)
                        break
                    except torch.OutOfMemoryError:
                        S //= 2
                if mode not in res:
                    res[mode] = (float("nan"), 0, float("nan"))
            inf, tr = res["infer"], res["train"]
            epoch_min = (a.resample * a.n_steps * inf[0] + a.itr * 64 * tr[0]) / 60
            print(f"| {n} | {cfg} | {a.amp} | {inf[0]*1e3:.2f} | {tr[0]*1e3:.2f} | {inf[1]} |"
                  f" {tr[1]} | {max(inf[2], tr[2]):.2f} | {epoch_min:.1f} |", flush=True)
            del net
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
