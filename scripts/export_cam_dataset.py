"""Export an sbmis graph dataset to the on-disk format read by the CAM code (cam/datasets/mis.py).

Must run with the CAM conda env (networkx 2.x + PyG), which is used read-only:

  PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 nice -n 10 \
      /usr0/home/shengyuf/anaconda3/envs/CAM/bin/python -s scripts/export_cam_dataset.py \
      --name er_200_300 --workers 8

Why that env: CAM reads graphs with nx.read_gpickle, which only exists in networkx < 3 (CAM pins
2.7.1), and a graph pickled by networkx 3.x is not guaranteed to unpickle under 2.7. Building the
graphs with the same networkx that will read them avoids the version issue; the input .npz files
(plain edge lists) are version-neutral.

Writes, for data/datasets/<name>/{train,val,test}.npz:
  data/cam/<name>/ER_train/<name>_train_00000.gpickle, ...
  data/cam/<name>/ER_valid/<name>_val_00000.gpickle, ...  + best_known.json
  data/cam/<name>/ER_test/<name>_test_00000.gpickle, ...  + best_known.json
  data/cam/<name>/manifest.json
File k is graph k of the split (zero-padded, so CAM's sorted glob keeps the npz order). Each
gpickle is an undirected nx.Graph on nodes 0..n-1 (added in order, no attributes) with the npz
edges. No anno/*_unweighted.result labels are written: CAM's dataset puts labels into data.x,
which only the supervised solvers (DIFUSCO, FM) read, and our best-known solutions exist only as
sizes ({val,test}_opt.json). CAM's evaluation reports only the mean objective (-MIS size); the
per-graph best-known sizes go to best_known.json (not matched by CAM's '*gpickle' glob) for
computing gaps offline.

Verification: every exported graph of the selected splits (default: all) is loaded back with
CAM's own Dataset class and its node count, edge count and edge set are checked against the npz.
"""
import argparse
import hashlib
import importlib.util
import json
import os
import pickle
import shutil
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import networkx as nx
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from sbmis.datasets import dataset_dir, load_opt, load_split  # noqa: E402

CAM_SPLIT = {"train": "train", "val": "valid", "test": "test"}
CAM_DATASET_FILE = REPO / "cam" / "datasets" / "mis.py"


def file_name(name, split, i):
    return f"{name}_{split}_{i:05d}.gpickle"


def load_cam_dataset_class():
    # load cam/datasets/mis.py under a private name (a bare `datasets` import could pick up the
    # HuggingFace package installed in the env)
    spec = importlib.util.spec_from_file_location("cam_datasets_mis", CAM_DATASET_FILE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.Dataset


# ---- workers (fork: globals set in the parent before the pool starts) ----
_G = {}


def _write(i):
    n, e = _G["gs"].graph(i)
    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from(e.tolist())
    nx.write_gpickle(G, _G["out"] / file_name(_G["name"], _G["split"], i), protocol=pickle.HIGHEST_PROTOCOL)
    return i


def _verify(i):
    import torch
    torch.set_num_threads(1)
    gs, ds = _G["gs"], _G["ds"]
    n, e = gs.graph(i)
    errs = []
    if os.path.basename(ds.files[i]) != file_name(_G["name"], _G["split"], i):
        errs.append(f"file order: {ds.files[i]}")
    d = ds[i]
    ei = d.edge_index.numpy()
    if d.num_nodes != n:
        errs.append(f"num_nodes {d.num_nodes} != {n}")
    if ei.shape[1] != 2 * len(e):
        errs.append(f"directed edges {ei.shape[1]} != 2*{len(e)}")
    und = np.unique(np.stack([ei.min(0), ei.max(0)], 1), axis=0)
    ref = np.unique(e.astype(np.int64), axis=0)
    if und.shape != ref.shape or not np.array_equal(und, ref):
        errs.append("edge set differs")
    if tuple(d.x.shape) != (n, 1) or not bool((d.x == 1).all()) or not bool((d.b == -1).all()):
        errs.append("x/b not the label-free defaults")
    if not bool((d.edge_weight == 0.5).all()):
        errs.append("edge_weight != 0.5")
    return i, n, len(e), errs


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--name", default="er_200_300")
    ap.add_argument("--root", default=str(REPO / "data" / "datasets"))
    ap.add_argument("--out", default=str(REPO / "data" / "cam"))
    ap.add_argument("--prefix", default="ER", help="CAM split dir prefix: <prefix>_{train,valid,test}")
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"], choices=list(CAM_SPLIT))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--verify_train", type=int, default=-1,
                    help="number of train graphs to verify (-1 = all); val/test are always all")
    ap.add_argument("--verify_only", action="store_true", help="skip writing, only verify")
    ap.add_argument("--overwrite", action="store_true", help="replace existing split dirs")
    args = ap.parse_args()

    if int(nx.__version__.split(".")[0]) >= 3:
        sys.exit(f"networkx {nx.__version__}: run this with the CAM env python "
                 "(/usr0/home/shengyuf/anaconda3/envs/CAM/bin/python), whose networkx 2.7 reads the files")

    out_root = Path(args.out).resolve() / args.name
    Dataset = load_cam_dataset_class()
    manifest_path = out_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    manifest.update({"source": str(dataset_dir(args.name, args.root).resolve()),
                     "networkx": nx.__version__, "python": sys.executable,
                     "script": "scripts/export_cam_dataset.py",
                     "file_pattern": f"{args.prefix}_<train|valid|test>/{args.name}_<split>_<idx:05d>.gpickle",
                     "labels": "none (no anno/); best-known sizes in best_known.json of valid/test"})
    manifest.setdefault("splits", {})

    for split in args.splits:
        t0 = time.time()
        gs = load_split(args.name, split, args.root)
        out = out_root / f"{args.prefix}_{CAM_SPLIT[split]}"
        _G.update(gs=gs, out=out, name=args.name, split=split)
        if not args.verify_only:
            if out.exists() and any(out.iterdir()):
                if not args.overwrite:
                    sys.exit(f"{out} is not empty; pass --overwrite to replace it")
                assert out.is_relative_to(Path(args.out).resolve()), out
                shutil.rmtree(out)
            out.mkdir(parents=True, exist_ok=True)
            with Pool(args.workers) as pool:
                for _ in pool.imap_unordered(_write, range(len(gs)), chunksize=64):
                    pass
            if split in ("val", "test"):
                opt = load_opt(args.name, split, args.root)
                assert len(opt) == len(gs) and [o["index"] for o in opt] == list(range(len(gs)))
                ref = [{"file": file_name(args.name, split, o["index"]), **o} for o in opt]
                (out / "best_known.json").write_text(json.dumps(ref, indent=0))
        t_write = time.time() - t0

        ds = Dataset(str(out))
        assert len(ds) == len(gs), f"{split}: CAM sees {len(ds)} files, npz has {len(gs)}"
        assert ds.data_label_path is None, "unexpected anno/ dir"
        _G["ds"] = ds
        idx = list(range(len(gs)))
        if split == "train" and 0 <= args.verify_train < len(gs):
            rng = np.random.default_rng(0)
            idx = sorted({0, len(gs) - 1, *rng.choice(len(gs), args.verify_train, replace=False).tolist()})
        bad = []
        with Pool(args.workers) as pool:
            for i, n, m, errs in pool.imap_unordered(_verify, idx, chunksize=16):
                if errs:
                    bad.append((i, errs))
        if bad:
            sys.exit(f"{split}: {len(bad)} graphs failed verification, e.g. {bad[:3]}")
        npz = dataset_dir(args.name, args.root) / f"{split}.npz"
        manifest["splits"][split] = {
            "dir": out.name, "graphs": len(gs), "verified": len(idx),
            "nodes_total": int(gs.n.sum()), "edges_total": int(len(gs.edges)),
            "npz_sha256": sha256(npz)}
        print(f"{split}: {len(gs)} graphs -> {out} (write {t_write:.1f}s); verified {len(idx)} with "
              f"CAM's Dataset in {time.time() - t0 - t_write:.1f}s: OK", flush=True)

    manifest["created_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    out_root.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
