"""Main experiment: which reference process helps ASBS? (plan §7, reference-process study)

Groups (same GNN, lr, AM:CM = 30:30, 512 samples / epoch, alpha_max, budget in
energy-gradient calls; only the reference process / source changes):

  paper versions      B0 (f = 0, sigma = 1); AS (Dirac source, analytic corrector, AM only)
                      on the same constant-sigma base. B0-S (geometric VE) is not run: the
                      quick screen (runs/s5_ve) had it clearly behind B0 for every
                      (sigma_min, sigma_max) tried, independent of the number of steps.
  drift, no info      VP (official VPSDE, beta 20 -> 0.1), iso4 / iso8 (f = -theta x with the
                      u = 0 terminal variance of B1+ at rho = 4 / 8), rot4 / rot8 (B1+
                      conjugated by a random rotation: same spectrum, graph alignment lost)
  info in the source  B0+prior4 / B0+prior8 (f = 0, mu ∝ N(0,I) exp(-2 gamma E'_quad),
                      gamma_rel = 4 / 8), B1p4+prior4 (both)
  negative control    B1_rho-2
  ours                B1_rho{1,2,4} (plan version, indefinite M),
                      B1p_rho{2,4,8,16} (B1+: M - lambda_min I, PSD)
  non-learned (U8)    MALA and annealed MALA at the same budgets (scripts/mcmc_baseline.py)

U1 graphs g0, g1, g2: lr 1e-4, 40 stages (614400 calls), TV to the PT-MALA reference.
n = 100 instances er:100:0.15:{0,1,2}: lr 3e-4, 20 stages (307200 calls), plus B1_rho8 with
one seed (plan U2). 3 seeds. Order: g2, n100 instance 0, g0, g1, n100 instances 1, 2;
within a block seed 0 of every group first.

python scripts/sweeps/main_reference.py   -> runs/main_ref/jobs.tsv
"""
from pathlib import Path

PY = "/usr0/home/shengyuf/yunyi/envs/sb/bin/python"
ROOT = "runs/main_ref"

GROUPS = {
    "B0": dict(base="zero"),
    "AS": dict(base="zero", source="delta", analytic_corrector=True),
    "VP": dict(base="vp"),
    "iso4": dict(base="iso_ou", iso_match_rho=4),
    "iso8": dict(base="iso_ou", iso_match_rho=8),
    "rot4": dict(base="linear_energy_shift_rot", rho=4),
    "rot8": dict(base="linear_energy_shift_rot", rho=8),
    "B0+prior4": dict(base="zero", source="energy_prior", prior_gamma=4),
    "B0+prior8": dict(base="zero", source="energy_prior", prior_gamma=8),
    "B1p4+prior4": dict(base="linear_energy_shift", rho=4, source="energy_prior", prior_gamma=4),
    "B1_rho-2": dict(base="linear_energy", rho=-2),
    **{f"B1_rho{r}": dict(base="linear_energy", rho=r) for r in (1, 2, 4)},
    **{f"B1p_rho{r}": dict(base="linear_energy_shift", rho=r) for r in (2, 4, 8, 16)},
}
SEEDS = (0, 1, 2)
U1 = dict(kappa_w=2, temp=0.5, net="gnn", lr=1e-4, stages=40, eval_every=2,
          n_infer_steps="100 20", save_at="153600 307200 614400")
N100 = dict(kappa_w=2, temp=0.5, net="gnn", lr=3e-4, stages=20, eval_every=2,
            n_infer_steps="20 100", save_at="76800 153600 307200")
U1_BUDGETS = "30720 76800 153600 307200 460800 614400"
N100_BUDGETS = "30720 76800 153600 230400 307200"


def fmt(kw):
    return " ".join(f"--{k}" if v is True else f"--{k} {v}" for k, v in kw.items())


def asbs_jobs(tag, graph, common, extra_ref=""):
    jobs = []
    for s in SEEDS:
        for name, g in GROUPS.items():
            out = f"{ROOT}/{tag}/{name}_s{s}"
            jobs.append((f"{tag}__{name}__s{s}",
                         f"{PY} scripts/train_mis.py --graph {graph} {fmt(common)} {fmt(g)}"
                         f" --seed {s}{extra_ref} --out {out}"))
    return jobs


def mala_jobs(tag, graph, budgets, chains, extra_ref=""):
    jobs = []
    for s in SEEDS:
        for c in chains:
            for af in (0.0, 0.5):
                name = f"MALA_c{c}" + ("_anneal" if af else "")
                jobs.append((f"{tag}__{name}__s{s}",
                             f"{PY} scripts/mcmc_baseline.py --graph {graph} --chains {c}"
                             f" --anneal_frac {af} --budgets {budgets} --seed {s}{extra_ref}"
                             f" --out {ROOT}/{tag}/{name}_s{s}"))
    return jobs


def main():
    jobs = []
    order = ["g2", "n100i0", "g0", "g1", "n100i1", "n100i2"]
    for tag in order:
        if tag.startswith("g"):
            ref = f" --ref runs/u1_ref/{tag}_k2_T0.5.pt"
            jobs += asbs_jobs(tag, tag, U1, ref)
            jobs += mala_jobs(tag, tag, U1_BUDGETS, (1024, 20000), ref)
        else:
            graph = f"er:100:0.15:{tag[-1]}"
            jobs += asbs_jobs(tag, graph, N100)
            jobs.append((f"{tag}__B1_rho8__s0",
                         f"{PY} scripts/train_mis.py --graph {graph} {fmt(N100)}"
                         f" --base linear_energy --rho 8 --seed 0 --out {ROOT}/{tag}/B1_rho8_s0"))
            jobs += mala_jobs(tag, graph, N100_BUDGETS, (1024,))
    Path(ROOT).mkdir(parents=True, exist_ok=True)
    (Path(ROOT) / "jobs.tsv").write_text("".join(f"{n}\t{c}\n" for n, c in jobs))
    print(len(jobs), "jobs ->", Path(ROOT) / "jobs.tsv")


if __name__ == "__main__":
    main()
