"""Run a list of jobs on free GPUs without touching other users' jobs.

Jobs file: one job per line, "<name>\t<command>". On exit a job writes
<logdir>/<name>.done (rc == 0) or <name>.failed (rc != 0); only .done jobs are
skipped on a later launch, so crashed jobs are retried. While a job runs,
<logdir>/<name>.pid holds its pid; a job whose pid is still alive is never
started a second time (e.g. after the launcher itself was restarted).

A GPU is eligible only if every compute process on it is one of ours (command
line contains OUR_ENV; an unreadable process counts as foreign) and it has at
least --min_free_mb free memory. GPU state is re-queried right before every
placement. The per-GPU cap counts all of our processes on the GPU, so several
launchers can share the machine. Never kills or signals any process.

python scripts/launch.py jobs.tsv --logdir runs/sweep --gpus 0 2 3 4 5 6 7 --per_gpu 4
--max_mine 1 spreads long jobs one per GPU (they wait instead of doubling up).
"""
import argparse
import fcntl
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
        gpus[uuid] = {"index": int(idx), "free": int(tot) - int(used), "foreign": False, "ours": 0}
    apps = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid",
                           "--format=csv,noheader"], capture_output=True, text=True).stdout
    for line in apps.strip().splitlines():
        if not line.strip():
            continue
        uuid, pid = [s.strip() for s in line.split(",")]
        if uuid not in gpus:
            continue
        cmd = subprocess.run(["ps", "-o", "args=", "-p", pid], capture_output=True,
                             text=True).stdout
        if OUR_ENV in cmd:
            gpus[uuid]["ours"] += 1
        else:
            gpus[uuid]["foreign"] = True
    return {g["index"]: g for g in gpus.values()}


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)  # signal 0: existence check only, sends nothing
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jobs")
    ap.add_argument("--logdir", required=True)
    ap.add_argument("--gpus", nargs="+", type=int, default=[0, 2, 3, 4, 5, 6, 7])
    ap.add_argument("--per_gpu", type=int, default=4)
    ap.add_argument("--min_free_mb", type=int, default=2500)
    ap.add_argument("--max_mine", type=int, default=None,
                    help="at most this many of this launcher's jobs per GPU (default: no limit)")
    ap.add_argument("--poll", type=float, default=20.0)
    a = ap.parse_args()
    logdir = Path(a.logdir)
    logdir.mkdir(parents=True, exist_ok=True)
    lock = open(logdir / ".launch.lock", "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        sys.exit(f"another launcher is already using {logdir}")

    jobs = []
    for line in Path(a.jobs).read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        name, cmd = line.split("\t", 1)
        if (logdir / f"{name}.done").exists():
            continue
        pidf = logdir / f"{name}.pid"
        if pidf.exists() and pid_alive(int(pidf.read_text() or 0)):
            print(f"[skip] {name}: still running as pid {pidf.read_text()}", flush=True)
            continue
        jobs.append((name, cmd))
    running = {}  # name -> (Popen, gpu)
    print(f"{len(jobs)} jobs pending", flush=True)
    while jobs or running:
        for name, (p, g) in list(running.items()):
            if p.poll() is not None:
                ext = "done" if p.returncode == 0 else "failed"
                (logdir / f"{name}.{ext}").write_text(str(p.returncode))
                (logdir / f"{name}.pid").unlink(missing_ok=True)
                print(f"[{ext}] {name} gpu{g} rc={p.returncode}", flush=True)
                del running[name]
        placed = True
        while jobs and placed:
            placed = False
            mine = {g: 0 for g in a.gpus}
            for _, g in running.values():
                mine[g] += 1
            for g in sorted(a.gpus, key=lambda g: mine[g]):
                s = gpu_state().get(g)  # fresh snapshot for every placement decision
                if s is None or s["foreign"] or s["free"] < a.min_free_mb:
                    continue
                if max(s["ours"], mine[g]) >= a.per_gpu:
                    continue
                if a.max_mine is not None and mine[g] >= a.max_mine:
                    continue
                name, cmd = jobs.pop(0)
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g), PYTHONUNBUFFERED="1",
                           OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
                (logdir / f"{name}.failed").unlink(missing_ok=True)
                log = open(logdir / f"{name}.log", "w")
                p = subprocess.Popen(cmd, shell=True, stdout=log, stderr=subprocess.STDOUT, env=env)
                (logdir / f"{name}.pid").write_text(str(p.pid))
                running[name] = (p, g)
                print(f"[start] {name} gpu{g}", flush=True)
                placed = True
                time.sleep(5)  # let the new process register on the GPU before re-checking
                break
        time.sleep(a.poll)
    print("all done", flush=True)


if __name__ == "__main__":
    sys.exit(main())
