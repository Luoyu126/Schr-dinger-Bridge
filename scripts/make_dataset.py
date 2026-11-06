"""Generate an ER graph dataset and best-known MIS sizes for its evaluation splits.

python scripts/make_dataset.py --name er_200_300 --n_min 200 --n_max 300 \
    --train 16384 --val 32 --test 128 --workers 8 --time_limit 300

Writes data/datasets/<name>/{train,val,test}.npz, meta.json and {val,test}_opt.json. Each
opt entry: best size = max(local search, MILP incumbent) and whether the MILP proved it.
"""
import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sbmis.datasets import SPLIT_SEED, GraphSet, dataset_dir  # noqa: E402
from sbmis.graphs import exact_mis_size, local_search_mis  # noqa: E402


def best_known(args):
    i, A, time_limit = args
    ls = int(local_search_mis(A, restarts=10, iters=20_000, seed=i).sum())
    size, _, proven = exact_mis_size(A, time_limit=time_limit)
    best = max(ls, int(size))
    return {"index": i, "size": best, "proven_optimal": bool(proven and size >= ls),
            "milp_size": int(size), "ls_size": ls}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--n_min", type=int, required=True)
    ap.add_argument("--n_max", type=int, required=True)
    ap.add_argument("--p", type=float, default=0.15)
    ap.add_argument("--train", type=int, default=16384)
    ap.add_argument("--val", type=int, default=32)
    ap.add_argument("--test", type=int, default=128)
    ap.add_argument("--solve", nargs="*", default=["val", "test"])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--time_limit", type=float, default=300.0)
    a = ap.parse_args()
    out = dataset_dir(a.name)
    out.mkdir(parents=True, exist_ok=True)
    meta = {k: getattr(a, k) for k in ("n_min", "n_max", "p", "train", "val", "test")}
    meta.update(split_seed=SPLIT_SEED, generator="DIFUSCO ErdosRenyi: randint(n_min, n_max), "
                "networkx.erdos_renyi_graph(n, p), one seed per graph")
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    sets = {}
    for split in ("train", "val", "test"):
        f = out / f"{split}.npz"
        if not f.exists():
            GraphSet.generate_er(a.n_min, a.n_max, a.p, getattr(a, split), SPLIT_SEED[split]).save(f)
        sets[split] = GraphSet.load(f)
        g = sets[split]
        print(split, len(g), "graphs, n in", g.n.min(), g.n.max(), "mean edges",
              round(len(g.edges) / len(g), 1), flush=True)
    for split in a.solve:
        g = sets[split]
        jobs = [(i, g.adj(i), a.time_limit) for i in range(len(g))]
        with Pool(a.workers) as pool:
            res = sorted(pool.map(best_known, jobs, chunksize=1), key=lambda r: r["index"])
        (out / f"{split}_opt.json").write_text(json.dumps(res))
        sizes = [r["size"] for r in res]
        print(split, "best-known MIS mean", sum(sizes) / len(sizes), "proven",
              sum(r["proven_optimal"] for r in res), "/", len(res), flush=True)


if __name__ == "__main__":
    main()
