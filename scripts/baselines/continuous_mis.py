"""Run the pCQO or PQQA reimplementation (sbmis/baselines) on a directory of MIS graphs.

  CUDA_VISIBLE_DEVICES=6 python scripts/baselines/continuous_mis.py pcqo \
      --data /home/shengyuf/yunyi/data/CAM_data/data_er/er_test --out runs/baselines/pcqo_er700
  CUDA_VISIBLE_DEVICES=7 python scripts/baselines/continuous_mis.py pqqa --runs 100 --steps 3000 \
      --data /home/shengyuf/yunyi/data/CAM_data/data_er/er_test --out runs/baselines/pqqa_er700_S100_3k

Defaults are the published ER-[700-800] settings (docs/continuous_baselines.md). Graphs are read as
networkx gpickles (nx.read_gpickle where it exists, i.e. networkx 2.x, else pickle), sorted by file
name, and run `--chunk` at a time on one GPU. Writes <out>/result.json (config, per-graph sizes,
mean, wall time, curves) and <out>/sets.npz (best set per graph). Refuses an existing <out>.
"""
import argparse
import glob
import json
import os
import pickle
import sys
import time

import networkx as nx
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from sbmis.baselines.pcqo import pcqo  # noqa: E402
from sbmis.baselines.pqqa import pqqa  # noqa: E402

KAMIS_ER700 = 44.87     # KaMIS mean on the 128 ER-[700-800] test graphs (DIMES / CAM tables)


def load_graph(path):
    if hasattr(nx, "read_gpickle"):
        G = nx.read_gpickle(path)
    else:
        with open(path, "rb") as f:
            G = pickle.load(f)
    n = G.number_of_nodes()
    assert sorted(G.nodes()) == list(range(n)), path
    return nx.to_numpy_array(G, nodelist=range(n), dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("method", choices=["pcqo", "pqqa"])
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_graphs", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=128, help="graphs per GPU batch")
    ap.add_argument("--seed", type=int, default=None)
    # pCQO (cpp_impl/pcqo_mis.cpp ER settings)
    ap.add_argument("--lr", type=float, default=None, help="pcqo 9e-6, pqqa 1.0")
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--steps", type=int, default=None, help="pcqo 225000, pqqa 3000")
    ap.add_argument("--steps_per_batch", type=int, default=450)
    ap.add_argument("--gamma", type=float, default=350.0)
    ap.add_argument("--gamma_prime", type=float, default=7.0)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--std", type=float, default=2.25)
    ap.add_argument("--fp32", action="store_true", help="pcqo in float32 instead of the C++ float16")
    # PQQA (qqa.optimize, paper App. C)
    ap.add_argument("--runs", type=int, default=100)
    ap.add_argument("--temp", type=float, default=1e-3)
    ap.add_argument("--min_bg", type=float, default=-2.0)
    ap.add_argument("--max_bg", type=float, default=0.1)
    ap.add_argument("--div_param", type=float, default=0.1)
    ap.add_argument("--penalty", type=float, default=3.0)
    ap.add_argument("--record_every", type=int, default=100)
    args = ap.parse_args()

    if os.path.exists(args.out):
        sys.exit(f"error: {args.out} exists")
    files = sorted(glob.glob(os.path.join(args.data, "*.gpickle")))
    if args.max_graphs:
        files = files[:args.max_graphs]
    if not files:
        sys.exit(f"error: no gpickles in {args.data}")
    adjs = [load_graph(f) for f in files]
    os.makedirs(args.out)
    device = "cuda"
    if args.method == "pcqo":
        cfg = dict(lr=args.lr if args.lr is not None else 9e-6, momentum=args.momentum,
                   steps=args.steps or 225_000, steps_per_batch=args.steps_per_batch, gamma=args.gamma,
                   gamma_prime=args.gamma_prime, batch=args.batch, std=args.std,
                   seed=113 if args.seed is None else args.seed)
        dtype = torch.float32 if args.fp32 else torch.float16
        run = lambda a: pcqo(a, dtype=dtype, device=device, log_every=50, **cfg)
    else:
        cfg = dict(runs=args.runs, steps=args.steps or 3000, lr=args.lr if args.lr is not None else 1.0,
                   temp=args.temp, min_bg=args.min_bg, max_bg=args.max_bg, div_param=args.div_param,
                   penalty=args.penalty, seed=0 if args.seed is None else args.seed,
                   record_every=args.record_every)
        run = lambda a: pqqa(a, device=device, log_every=max(1, cfg["steps"]//10), **cfg)

    res = {"method": args.method, "config": dict(cfg), "data": args.data, "files": [os.path.basename(f) for f in files],
           "chunk": args.chunk, "gpu": torch.cuda.get_device_name(0), "time": 0.0, "curve_chunks": []}
    if args.method == "pcqo":
        res["config"]["dtype"] = "float32" if args.fp32 else "float16"
    sets, per_graph = [], {}
    t0 = time.time()
    for c0 in range(0, len(adjs), args.chunk):
        out = run(adjs[c0:c0 + args.chunk])
        print(f"graphs {c0}-{c0 + len(out['best_set']) - 1} done in {out['time']:.0f}s", flush=True)
        res["time"] += out["time"]
        res["curve_chunks"].append(out["curve"])
        sets += [s.numpy() for s in out["best_set"]]
        for k, v in out.items():
            if k not in ("best_set", "curve", "time"):
                per_graph.setdefault(k, []).extend(v.tolist())
    res["per_graph"] = per_graph
    key = "best_size" if args.method == "pcqo" else "best_feasible"
    res["mean_size"] = float(np.mean(per_graph[key]))
    res["ratio_to_kamis_mean"] = res["mean_size"]/KAMIS_ER700 if "er_test" in args.data else None
    res["total_time_incl_loading"] = time.time() - t0
    with open(os.path.join(args.out, "result.json"), "w") as f:
        json.dump(res, f)
    np.savez_compressed(os.path.join(args.out, "sets.npz"), *sets)
    print(json.dumps({"method": args.method, "mean_size": res["mean_size"],
                      "ratio_to_kamis_mean": res["ratio_to_kamis_mean"], "time_s": res["time"]}))


if __name__ == "__main__":
    main()
