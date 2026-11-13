#!/usr/bin/env python3
"""One evaluation queue for all ER-[700-800] runs on one GPU (local addition, see cam/UPSTREAM.md).

Instead of one copy of large_curve_er700.sh per split and per set of runs (several independent
queues that can pile up on a GPU), a single process watches every (run, split, epoch) point,
starts its evaluation once the checkpoint is written, and keeps at most --max_jobs evaluations on
the GPU, at most --max_large of them on the large graphs. Command, protocol and output paths are
those of large_curve_er700.sh, so results continue its directories:
  test   128 ER-[700-800] graphs, 50 steps x 20 samples     runs/cam/mis_er700_test10/<exp>/epKKK/
  large  16 ER-[9000-11000] graphs, 200 steps x 20 samples  runs/cam/mis_er700_large/<exp>/epKKK/
Epoch k (after k epochs) is runs/cam/mis_er700/<exp>/epoch_{k-1}.pt; k = 0 is epoch_init.pt (from
runs/cam/mis_er700_init2/<exp>/ if present). A point is done when its test_log.txt.per_graph.csv
exists. A point whose evaluation is already running in another process (e.g. a queue being
replaced) is left alone and counted as load. A checkpoint is used only once it is --min_age seconds
old (not while it is being written). A failed point is retried at most --retries times. The run
list is re-read from --exps_file at every pass, so runs can be added without a restart.

From the repository root, with CAM_ER700_DIR set (e.g. source /home/shengyuf/yunyi/envs/cam_env.sh):
  nohup python3 cam/scripts/mis/eval_queue_er700.py --gpu 5 --exps_file runs/cam/eval_queue_er700.exps \\
      >> runs/cam/eval_queue_er700.log 2>&1 &
  ... --dry      print the state and the next commands, start nothing
"""
import argparse
import os
import re
import subprocess
import sys
import time

# Flags of each run group (as large_curve_er700.sh), group = run name without _e<E>_s<seed>.
GROUP_FLAGS = {
    'CAM_tau0.1': ['--tau0', '0.1'],
    'CAM_tau0.05': ['--tau0', '0.05'],
    'CAMPrior_p0.01': ['--method', 'CAMPrior', '--prior_tau0', '0.01'],
    'CAMPrior_p0.1': ['--method', 'CAMPrior', '--prior_tau0', '0.1'],
    'CAMPrior_p1': ['--method', 'CAMPrior', '--prior_tau0', '1'],
    'CAMPrior_p1_warmstart': ['--method', 'CAMPrior', '--prior_tau0', '1', '--prior_kl_ref', 'uniform'],
    'CAMPrior_p1_refonly': ['--method', 'CAMPrior', '--prior_tau0', '1', '--prior_policy', 'off'],
    'CAMPrior_p1_fixed': ['--method', 'CAMPrior', '--prior_tau0', '1', '--prior_fixed'],
    'CAMPrior_p10': ['--method', 'CAMPrior', '--prior_tau0', '10'],
    'CAM_tau0.1_distill': ['--tau0', '0.1'],     # stage 2 of the distillation warm start: plain CAM
    # stage 2 of the distillation warm start with the KL reference kept at RLSA (prior only in the reference)
    'CAMPrior_p1_distill_refrlsa': ['--method', 'CAMPrior', '--prior_tau0', '1', '--prior_policy', 'off'],
    # training-label ablation: the label only changes training, sampling is plain CAMPrior p1
    'CAMPrior_p1_simlab0.1': ['--method', 'CAMPrior', '--prior_tau0', '1'],
    'CAMPrior_p1_camlab0.1': ['--method', 'CAMPrior', '--prior_tau0', '1'],
}
# dense-label and segment-label runs (labels_er700.sh SET=dense / clips): sampling is plain CAMPrior p1
for _lam in ('0', '0.3', '0.5', '0.7', '0.8', '0.9', '0.95', '0.97', '0.99'):
    GROUP_FLAGS[f'CAMPrior_p1_dense{_lam}'] = ['--method', 'CAMPrior', '--prior_tau0', '1']
for _c in ('2', '5', '10', '25', '50'):
    GROUP_FLAGS[f'CAMPrior_p1_clips{_c}'] = ['--method', 'CAMPrior', '--prior_tau0', '1']
SPLITS = {  # name: (test dir, num_t, output dir)
    'test': ('er_test', 50, 'runs/cam/mis_er700_test10'),
    'large': ('er_large_test', 200, 'runs/cam/mis_er700_large'),
}


def log(msg):
    print(f'{time.strftime("%F %T")} {msg}', flush=True)


def group_of(exp):
    return re.sub(r'_e\d+_s\d+$', '', exp)


def read_exps(path):
    with open(path) as f:
        return [w for line in f for w in line.split('#')[0].split()]


def points(exps, splits, every, last):
    for k in range(0, last + 1, every):
        for split in splits:
            for x in exps:
                out = SPLITS[split][2]
                if k == 0:
                    ck = f'runs/cam/mis_er700_init2/{x}/epoch_init.pt'
                    if not os.path.exists(ck):
                        ck = f'runs/cam/mis_er700/{x}/epoch_init.pt'
                else:
                    ck = f'runs/cam/mis_er700/{x}/epoch_{k - 1:04d}.pt'
                yield (x, split, k, ck, f'{out}/{x}/ep{k:03d}/test_log.txt')


def running_evals():
    """log paths (relative to the repository) of all evaluations running on this machine"""
    found = set()
    for pid in os.listdir('/proc'):
        if not pid.isdigit():
            continue
        try:
            with open(f'/proc/{pid}/cmdline', 'rb') as f:
                args = f.read().decode(errors='replace').split('\0')
        except OSError:
            continue
        if '--log_file' in args:
            lf = args[args.index('--log_file') + 1] if args.index('--log_file') + 1 < len(args) else ''
            m = re.search(r'runs/cam/mis_er700_(test10|large)/.*/test_log\.txt$', lf)
            if m:
                found.add(m.group(0))
    return found


def command(a, x, split, ck, logp, repo):
    testdir, num_t, _ = SPLITS[split]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(a.gpu), PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',
               MODE='test', PROJECT='mis_er700_eval', EXP=x)
    argv = (['bash', 'cam/scripts/mis/mis_er700_cam.sh'] + GROUP_FLAGS[group_of(x)]
            + ['--checkpoint', f'{repo}/{ck}', '--test_path', f'{a.data}/{testdir}', '--num_t', str(num_t),
               '--num_k', '20', '--log_file', f'{repo}/{logp}'])
    return env, argv


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--gpu', type=int, required=True)
    ap.add_argument('--exps_file', required=True, help='run names, whitespace or one per line, # comments')
    ap.add_argument('--splits', nargs='+', default=['test', 'large'], choices=list(SPLITS))
    ap.add_argument('--every', type=int, default=10)
    ap.add_argument('--last', type=int, default=100)
    ap.add_argument('--max_jobs', type=int, default=6)
    ap.add_argument('--max_large', type=int, default=4)
    ap.add_argument('--retries', type=int, default=2)
    ap.add_argument('--min_age', type=float, default=60.0, help='seconds since the checkpoint was written')
    ap.add_argument('--poll', type=float, default=60.0)
    ap.add_argument('--status_every', type=float, default=600.0)
    ap.add_argument('--forever', action='store_true', help='keep watching after every point is done')
    ap.add_argument('--dry', action='store_true')
    a = ap.parse_args()
    repo = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
    os.chdir(repo)
    a.data = os.environ.get('CAM_ER700_DIR')
    if not a.data:
        sys.exit('set CAM_ER700_DIR (e.g. source /home/shengyuf/yunyi/envs/cam_env.sh)')

    running = {}            # log path -> (Popen, x, split, k, start time)
    fails = {}              # log path -> number of failed attempts
    last_status, last_line = 0.0, None
    log(f'queue on GPU {a.gpu}: splits {a.splits}, epochs 0..{a.last} every {a.every}, '
        f'max {a.max_jobs} jobs ({a.max_large} large), runs from {a.exps_file}')
    while True:
        for lp, (p, x, split, k, t0) in list(running.items()):
            rc = p.poll()
            if rc is None:
                continue
            del running[lp]
            if rc == 0 and os.path.exists(lp + '.per_graph.csv'):
                with open(lp) as f:
                    first = f.readline().strip()
                log(f'done {x} {split} ep{k} ({(time.time() - t0) / 60:.0f} min) {first}')
            else:
                fails[lp] = fails.get(lp, 0) + 1
                log(f'FAILED {x} {split} ep{k} (exit {rc}, attempt {fails[lp]}), see {lp}.out')

        exps = read_exps(a.exps_file)
        unknown = [x for x in exps if group_of(x) not in GROUP_FLAGS]
        if unknown:
            log(f'skipping runs of unknown groups {unknown}: add their flags to GROUP_FLAGS')
            exps = [x for x in exps if x not in unknown]
        elsewhere = running_evals() - set(running)   # e.g. started by a queue this one replaces
        pts = list(points(exps, a.splits, a.every, a.last))
        n_done, ready, waiting, gave_up = 0, [], 0, 0
        now = time.time()
        for x, split, k, ck, lp in pts:
            if os.path.exists(lp + '.per_graph.csv'):
                n_done += 1
            elif lp in running or lp in elsewhere:
                continue
            elif fails.get(lp, 0) > a.retries:
                gave_up += 1
            elif os.path.exists(ck) and now - os.path.getmtime(ck) >= a.min_age:
                ready.append((x, split, k, ck, lp))
            else:
                waiting += 1
        n_large = sum(1 for _, _, s, _, _ in running.values() if s == 'large') + \
            sum(1 for lp in elsewhere if 'mis_er700_large/' in lp)
        n_jobs = len(running) + len(elsewhere)
        started = []
        for x, split, k, ck, lp in ready:
            if n_jobs >= a.max_jobs:
                break
            if split == 'large' and n_large >= a.max_large:
                continue
            env, argv = command(a, x, split, ck, lp, repo)
            if a.dry:
                print('  would run:', ' '.join(f'{k}={env[k]}' for k in ('CUDA_VISIBLE_DEVICES', 'MODE', 'PROJECT', 'EXP')),
                      ' '.join(f"'{w}'" if ' ' in w else w for w in argv), f'> {lp}.out')
            else:
                os.makedirs(os.path.dirname(lp), exist_ok=True)
                with open(lp + '.out', 'w') as out:
                    p = subprocess.Popen(argv, env=env, stdout=out, stderr=subprocess.STDOUT,
                                         stdin=subprocess.DEVNULL, start_new_session=True)
                running[lp] = (p, x, split, k, time.time())
            started.append(f'{x} {split} ep{k}')
            n_jobs += 1
            n_large += split == 'large'
        if started:
            log('start ' + '; '.join(started))
        line = (f'done {n_done}/{len(pts)}, running {len(running)} here + {len(elsewhere)} elsewhere, '
                f'ready {len(ready) - len(started)}, waiting for checkpoints {waiting}'
                + (f', gave up {gave_up}' if gave_up else ''))
        if a.dry or line != last_line or time.time() - last_status >= a.status_every:
            log(line)
            last_line, last_status = line, time.time()
        if a.dry:
            return
        if n_done == len(pts) and not running and not a.forever:
            log('all done')
            return
        time.sleep(a.poll)


if __name__ == '__main__':
    main()
