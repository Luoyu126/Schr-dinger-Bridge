"""Run a list of jobs on free GPUs without touching other users' jobs.

Jobs file: one job per line, "<name>\t<command>". A job is skipped if
<logdir>/<name>.done exists. A GPU is eligible only if every compute process
on it is one of ours (command line contains OUR_ENV) and it has at least
--min_free_mb free memory. Never kills or signals anything it did not start.

python scripts/launch.py jobs.tsv --logdir runs/sweep --gpus 0 2 3 4 5 6 7 --per_gpu 4
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

OUR_ENV = "/usr0/home/shengyuf/yunyi/envs/sb/"


def gpu_state():
    q = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,memory.total,memory.used",
                        "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
    gpus = {}
    for line in q.strip().splitlines():
        idx, uuid, tot, used = [s.strip() for s in line.split(",")]
        gpus[uuid] = {"index": int(idx), "free": int(tot) - int(used), "foreign": False}
    apps = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid",
                           "--format=csv,noheader"], capture_output=True, text=True).stdout
    for line in apps.strip().splitlines():
        if not line.strip():
            continue
        uuid, pid = [s.strip() for s in line.split(",")]
        cmd = subprocess.run(["ps", "-o", "args=", "-p", pid], capture_output=True,
                             text=True).stdout
        if OUR_ENV not in cmd and uuid in gpus:
            gpus[uuid]["foreign"] = True
    return {g["index"]: g for g in gpus.values()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jobs")
    ap.add_argument("--logdir", required=True)
    ap.add_argument("--gpus", nargs="+", type=int, default=[0, 2, 3, 4, 5, 6, 7])
    ap.add_argument("--per_gpu", type=int, default=4)
    ap.add_argument("--min_free_mb", type=int, default=2500)
    ap.add_argument("--poll", type=float, default=20.0)
    a = ap.parse_args()
    logdir = Path(a.logdir)
    logdir.mkdir(parents=True, exist_ok=True)
    jobs = []
    for line in Path(a.jobs).read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            name, cmd = line.split("\t", 1)
            if not (logdir / f"{name}.done").exists():
                jobs.append((name, cmd))
    running = {}  # name -> (Popen, gpu)
    print(f"{len(jobs)} jobs pending", flush=True)
    while jobs or running:
        for name, (p, g) in list(running.items()):
            if p.poll() is not None:
                (logdir / f"{name}.done").write_text(str(p.returncode))
                print(f"[done] {name} gpu{g} rc={p.returncode}", flush=True)
                del running[name]
        if jobs:
            st = gpu_state()
            load = {g: 0 for g in a.gpus}
            for _, g in running.values():
                load[g] += 1
            for g in sorted(a.gpus, key=lambda g: load[g]):
                if not jobs:
                    break
                s = st.get(g)
                if s is None or s["foreign"] or s["free"] < a.min_free_mb or load[g] >= a.per_gpu:
                    continue
                name, cmd = jobs.pop(0)
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g), PYTHONUNBUFFERED="1")
                log = open(logdir / f"{name}.log", "w")
                p = subprocess.Popen(cmd, shell=True, stdout=log, stderr=subprocess.STDOUT, env=env)
                running[name] = (p, g)
                load[g] += 1
                print(f"[start] {name} gpu{g}", flush=True)
                time.sleep(2)  # let memory usage register before the next placement
        time.sleep(a.poll)
    print("all done", flush=True)


if __name__ == "__main__":
    sys.exit(main())
