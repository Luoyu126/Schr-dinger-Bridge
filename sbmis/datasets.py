"""Graph datasets for the amortized setting (plan U7).

ER-[n_min-n_max] follows DIFUSCO's generator (data/mis-benchmark-framework, ErdosRenyi):
n ~ uniform{n_min, ..., n_max}, then networkx.erdos_renyi_graph(n, p) with p = 0.15. Graphs
are not forced to be connected (DIFUSCO does not either). Every graph is generated from its
own seed, so a split is reproducible from (n_min, n_max, p, base_seed, count).

A split is stored as one .npz: n (G,), edge_ptr (G+1,), edges (E, 2) int32 with i < j.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import numpy as np

SPLIT_SEED = {"train": 0, "val": 10_000_000, "test": 20_000_000}


def er_dataset_graph(n_min: int, n_max: int, p: float, seed: int) -> tuple[int, np.ndarray]:
    rng = random.Random(seed)
    n = rng.randint(n_min, n_max)
    # edges from the continuing stream (n and the edges independent, as in DIFUSCO's
    # single global stream); seeding networkx with `seed` again would tie edge (0, 1) to n
    g = nx.erdos_renyi_graph(n, p, seed=rng)
    e = np.array(sorted((min(a, b), max(a, b)) for a, b in g.edges()), dtype=np.int32)
    return n, e.reshape(-1, 2)


@dataclass
class GraphSet:
    n: np.ndarray          # (G,)
    edge_ptr: np.ndarray   # (G+1,)
    edges: np.ndarray      # (E, 2), local node ids, i < j

    def __len__(self):
        return len(self.n)

    def graph(self, i: int) -> tuple[int, np.ndarray]:
        return int(self.n[i]), self.edges[self.edge_ptr[i]:self.edge_ptr[i + 1]]

    def adj(self, i: int) -> np.ndarray:
        n, e = self.graph(i)
        A = np.zeros((n, n))
        A[e[:, 0], e[:, 1]] = A[e[:, 1], e[:, 0]] = 1.0
        return A

    def save(self, path):
        np.savez_compressed(path, n=self.n, edge_ptr=self.edge_ptr, edges=self.edges)

    @classmethod
    def load(cls, path):
        d = np.load(path)
        return cls(d["n"], d["edge_ptr"], d["edges"])

    @classmethod
    def generate_er(cls, n_min, n_max, p, count, base_seed):
        ns, es = [], []
        for k in range(count):
            n, e = er_dataset_graph(n_min, n_max, p, base_seed + k)
            ns.append(n)
            es.append(e)
        ptr = np.concatenate([[0], np.cumsum([len(e) for e in es])]).astype(np.int64)
        return cls(np.array(ns, dtype=np.int32), ptr, np.concatenate(es).astype(np.int32))


def dataset_dir(name: str, root: str = "data/datasets") -> Path:
    return Path(root) / name


def load_split(name: str, split: str, root: str = "data/datasets") -> GraphSet:
    return GraphSet.load(dataset_dir(name, root) / f"{split}.npz")


def load_opt(name: str, split: str, root: str = "data/datasets") -> list[dict]:
    return json.loads((dataset_dir(name, root) / f"{split}_opt.json").read_text())
