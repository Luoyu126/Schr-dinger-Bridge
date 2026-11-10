"""Simulated adjoint labels for CAMPrior (local addition, see cam/UPSTREAM.md).

CAM's training label for the flip decision of node i at step k is the lean adjoint
    target_i = grad_i(X_end) * (2 x_i(k) - 1)                            (solvers/cam.py)
i.e. "flip i now and the perturbation reaches the end unchanged". This is exact only for a
homogeneous base process (CAM paper, Eqs. 10-11 and 16-23). With the RLSA prior as the base
(solvers/cam_prior.py) the flip probabilities depend on the state, so flipping i changes how the
neighbours (and, through the top-d threshold, all nodes) move afterwards. The lean label ignores
this.

The simulated label measures it directly, by rolling the base process out from the two
candidate next states (DAM, Prop. 2.3 / Eq. 12: roll the base CTMC out from the counterfactual):
    x0 = X_{k+1} with node i at its value before step k   (i not flipped at step k)
    x1 = X_{k+1} with node i flipped at step k
    label_i(k) = g(Z(x0)) - g(Z(x1)),   Z(x) = RLSA run from x over the steps k+1 .. K-1
g is the training energy (envs/<problem>.py), K = num_tp, and RLSA uses the prior of CAMPrior at
those steps (same d, eps, energy and temperatures tau_rlsa(j) * ratio). A positive label favours
flipping, like CAM's target. The two rollouts share their random numbers (common random numbers;
one uniform per node and step), so a node whose flip decision is the same in both runs moves the
same way. One rollout per label estimates E_base[g(Z(x0))] - E_base[g(Z(x1))], the cost-to-go
difference of the base policy; the soft-min form of the exact optimal-control value is not used.

Items. One item = one stored state (chain c, step k) and m sampled nodes. It runs 1 + m rollouts
from X_{k+1}: the factual next state, and X_{k+1} with one sampled node toggled. With
s_i = +1 if i was not flipped at step k and -1 if it was,
    label_i = s_i * (g(Z(X_{k+1})) - g(Z(X_{k+1} with i toggled))).
All rollouts of an item use the uniforms of that item's own seeded CPU generator, so a label does
not depend on how items are grouped into tasks or which device computes them (a retried task gives
the same numbers).

SimLabelPool computes items on several GPUs, one worker process per GPU, and survives failures:
a task that raises is retried on another worker (an out-of-memory task is split in two); a worker
that dies, hangs past the timeout or keeps failing is restarted, and after max_restarts it is
dropped and its share goes to the others; with no worker left the trainer's own device computes
the labels. A task that fails max_attempts times stops training with an error: labels never fall
back to CAM's lean target.
"""
import os
import queue
import time
import traceback

import numpy as np
import torch
import torch.multiprocessing as mp


# ---------------------------------------------------------------------------- computation
def item_uniforms(seeds, starts, K, n):
    """Per-item uniforms, U[p, j, :] for steps j = start_p .. K-1 (zeros before), on the CPU."""
    U = torch.zeros(len(seeds), K, n)
    gen = torch.Generator()
    for p, (s, j0) in enumerate(zip(seeds.tolist(), starts.tolist())):
        if j0 < K:
            gen.manual_seed(int(s))
            U[p, j0:] = torch.rand(K - j0, n, generator=gen)
    return U


def dense_adjacency(edge_index, n, device, edge_weight=None, dtype=torch.float32):
    A = torch.zeros(n, n, device=device, dtype=dtype)
    ei = torch.as_tensor(edge_index, device=device, dtype=torch.long)
    w = 1.0 if edge_weight is None else torch.as_tensor(edge_weight, device=device, dtype=dtype).reshape(-1)
    A[ei[0], ei[1]] = w
    return A


def rlsa_cf_labels(A, b, x_next, sel, sign, starts, seeds, temps, d, eps, beta_prior, beta):
    """Simulated labels of a set of items, on A's device.

    A          (n, n) adjacency (dense or sparse), the graph's A of envs/<problem>.py
    b          (n, 1) linear term of the energy (MIS: -1)
    x_next     (n, C) X_{k+1} of each item, 0/1
    sel        (C, m) sampled node indices of each item
    sign       (C, m) +1 if the node was not flipped at step k, -1 if it was
    starts     (C,)   first RLSA step to run, k + 1 (K = len(temps): no step)
    seeds      (C,)   per-item seeds of the CPU generator of the uniforms
    temps      K temperatures, tau_rlsa(j) * ratio
    returns    (C, m) labels, float32
    """
    dev = A.device
    n, C = x_next.shape
    m = sel.shape[1]
    K = len(temps)
    starts = torch.as_tensor(starts, dtype=torch.long)
    order = torch.argsort(starts, stable=True)
    s_sorted = starts[order]
    sel_s = torch.as_tensor(sel, dtype=torch.long)[order].to(dev)
    sign_s = torch.as_tensor(sign, dtype=torch.float32)[order].to(dev)
    U = item_uniforms(torch.as_tensor(seeds)[order], s_sorted, K, n).to(dev)   # C x K x n

    w = m + 1                                     # columns per item: factual, then m toggles
    X = torch.as_tensor(x_next, dtype=torch.float32)[:, order].to(dev).repeat_interleave(w, dim=1)
    rows = sel_s.reshape(-1)
    cols = (torch.arange(C, device=dev).unsqueeze(1)*w + 1 + torch.arange(m, device=dev)).reshape(-1)
    X[rows, cols] = 1 - X[rows, cols]
    b = b.to(dev).reshape(n, 1).float()
    dd = min(int(d), n)

    for j in range(int(s_sorted.min()) if C else K, K):
        P = int(torch.searchsorted(s_sorted, torch.tensor(j), right=True))   # items with start <= j
        if P == 0:
            continue
        x = X[:, :P*w]
        grad = 2*beta_prior*(A@x) + b
        delta = grad*(2*x - 1)/2
        thr = -torch.kthvalue(-delta, dd, dim=0, keepdim=True).values
        diff = delta - thr
        if eps != 0:
            diff = diff + eps
        prob = torch.sigmoid(diff/temps[j])
        u = U[:P, j, :].T.repeat_interleave(w, dim=1)                        # n x P*w
        X[:, :P*w] = torch.where(u < prob, 1 - x, x)

    energy = (X*(beta*(A@X) + b)).sum(0).reshape(C, w)                       # g of every rollout
    lab_sorted = sign_s*(energy[:, :1] - energy[:, 1:])
    out = torch.empty_like(lab_sorted)
    out[order.to(dev)] = lab_sorted
    return out.float()


def compute_task(task, device, A_cache=None):
    """Run one task (dict of numpy arrays, see SimLabelPool.compute) on device; returns numpy."""
    n = int(task['n'])
    key = task['graph_key']
    if A_cache is not None and A_cache.get('key') == key:
        A = A_cache['A']
    else:
        A = dense_adjacency(task['edge_index'], n, device, task.get('edge_weight'))
        if A_cache is not None:
            A_cache.clear()
            A_cache.update(key=key, A=A)
    lab = rlsa_cf_labels(A, torch.as_tensor(task['b']), torch.as_tensor(task['x_next']),
                         torch.as_tensor(task['sel']), torch.as_tensor(task['sign']),
                         torch.as_tensor(task['starts']), torch.as_tensor(task['seeds']),
                         task['temps'], task['d'], task['eps'], task['beta_prior'], task['beta'])
    return lab.cpu().numpy()


def _worker_main(slot, device, task_q, result_q, nthreads, fault):
    """Worker loop. fault (tests only): {'raise': n} | {'exit': n} | {'hang': n} on its first n tasks."""
    torch.set_num_threads(nthreads)
    dev = torch.device(device)
    if dev.type == 'cuda':
        torch.cuda.set_device(dev)
    cache, done = {}, 0
    result_q.put((slot, None, 'ready', os.getpid()))
    while True:
        msg = task_q.get()
        if msg is None:
            break
        tid, task = msg
        try:
            if fault and done < next(iter(fault.values())):
                kind = next(iter(fault))
                done += 1
                if kind == 'exit':
                    os._exit(3)
                if kind == 'hang':
                    time.sleep(10**6)
                raise RuntimeError(f'injected fault on slot {slot}')
            done += 1
            lab = compute_task(task, dev, cache)
            result_q.put((slot, tid, 'ok', lab))
        except Exception as e:  # report every error; the pool decides what to do
            oom = isinstance(e, getattr(torch.cuda, 'OutOfMemoryError', ())) or 'out of memory' in str(e)
            cache.clear()
            if dev.type == 'cuda':
                torch.cuda.empty_cache()
            result_q.put((slot, tid, 'oom' if oom else 'err', traceback.format_exc(limit=5)))


# ---------------------------------------------------------------------------- pool
class SimLabelError(RuntimeError):
    pass


class SimLabelPool:
    """Fault-tolerant pool of label workers, one process per device.

    devices   list of physical GPU ids (int or str) or 'cpu'
    fallback  torch.device used in-process when no worker is left (None: raise instead)
    """

    def __init__(self, devices, fallback=None, timeout=900.0, max_restarts=2, max_attempts=4,
                 tasks_per_worker=2, nthreads=2, log=print, faults=None, start_timeout=180.0):
        self.ctx = mp.get_context('spawn')
        self.devices = [str(d) for d in devices]
        self.fallback = fallback
        self.timeout = float(timeout)
        self.max_restarts = int(max_restarts)
        self.max_attempts = int(max_attempts)
        self.tasks_per_worker = int(tasks_per_worker)
        self.nthreads = int(nthreads)
        self.log = log
        self.faults = faults or {}
        self.result_q = self.ctx.Queue()
        self.slots = []
        self.stats = dict(tasks=0, retries=0, splits=0, restarts=0, dropped=0, fallback=0)
        self._tid = 0
        self._fallback_cache = {}
        for i, d in enumerate(self.devices):
            self.slots.append(dict(device=d, proc=None, q=None, alive=False, restarts=0,
                                   errors=0, inflight={}))
            self._start(i)
        self._wait_ready(start_timeout)

    # -- worker life cycle
    def _start(self, i):
        s = self.slots[i]
        d = s['device']
        s['q'] = self.ctx.Queue()
        old = os.environ.get('CUDA_VISIBLE_DEVICES')
        os.environ['CUDA_VISIBLE_DEVICES'] = '' if d == 'cpu' else d   # the child sees one GPU
        try:
            p = self.ctx.Process(target=_worker_main,
                                 args=(i, 'cpu' if d == 'cpu' else 'cuda:0', s['q'], self.result_q,
                                       self.nthreads, self.faults.get(i) if s['restarts'] == 0 else None),
                                 daemon=True)
            p.start()
        finally:
            if old is None:
                os.environ.pop('CUDA_VISIBLE_DEVICES', None)
            else:
                os.environ['CUDA_VISIBLE_DEVICES'] = old
        s.update(proc=p, alive=True, errors=0, ready=False, inflight={})

    def _wait_ready(self, limit):
        t0 = time.time()
        while any(s['alive'] and not s['ready'] for s in self.slots) and time.time() - t0 < limit:
            try:
                slot, tid, status, payload = self.result_q.get(timeout=1.0)
            except queue.Empty:
                for i, s in enumerate(self.slots):
                    if s['alive'] and not s['proc'].is_alive():
                        self._drop_or_restart(i, 'died during start-up')
                continue
            if status == 'ready':
                self.slots[slot]['ready'] = True
        for i, s in enumerate(self.slots):
            if s['alive'] and not s['ready']:
                self._kill(i)
                s['alive'] = False
                self.stats['dropped'] += 1
                self.log(f'[sim-labels] worker {i} (GPU {s["device"]}) did not start; dropped')
        alive = [s['device'] for s in self.slots if s['alive']]
        self.log(f'[sim-labels] workers ready on {alive}')
        if not alive and self.fallback is None:
            raise SimLabelError('no label worker started')

    def _kill(self, i):
        p = self.slots[i]['proc']
        if p is not None and p.is_alive():
            p.kill()
            p.join(timeout=10)

    def _drop_or_restart(self, i, why):
        """Restart worker i (its in-flight tasks are returned) or drop it after max_restarts."""
        s = self.slots[i]
        lost = list(s['inflight'].values())
        self._kill(i)
        if s['restarts'] < self.max_restarts:
            s['restarts'] += 1
            self.stats['restarts'] += 1
            self.log(f'[sim-labels] worker {i} (GPU {s["device"]}): {why}; restart '
                     f'{s["restarts"]}/{self.max_restarts}')
            self._start(i)
            self._wait_one_ready(i)
        else:
            s['alive'] = False
            s['inflight'] = {}
            self.stats['dropped'] += 1
            self.log(f'[sim-labels] worker {i} (GPU {s["device"]}): {why}; dropped, its share goes '
                     f'to the other workers')
        return lost

    def _wait_one_ready(self, i, limit=180.0):
        t0 = time.time()
        stash = []
        while time.time() - t0 < limit:
            try:
                msg = self.result_q.get(timeout=1.0)
            except queue.Empty:
                if not self.slots[i]['proc'].is_alive():
                    break
                continue
            if msg[0] == i and msg[2] == 'ready':
                self.slots[i]['ready'] = True
                for m in stash:
                    self.result_q.put(m)
                return
            stash.append(msg)
        for m in stash:
            self.result_q.put(m)
        self._kill(i)
        self.slots[i]['alive'] = False
        self.stats['dropped'] += 1
        self.log(f'[sim-labels] worker {i} did not come back after a restart; dropped')

    def close(self):
        for s in self.slots:
            if s['alive'] and s['proc'] is not None and s['proc'].is_alive():
                try:
                    s['q'].put(None)
                except Exception:
                    pass
        for i, s in enumerate(self.slots):
            if s['proc'] is not None:
                s['proc'].join(timeout=10)
                self._kill(i)
            s['alive'] = False

    def alive_slots(self):
        return [i for i, s in enumerate(self.slots) if s['alive']]

    # -- computing
    @staticmethod
    def split_items(cost, parts):
        """Greedy longest-first assignment of items (by cost) to `parts` balanced groups."""
        order = np.argsort(-cost, kind='stable')
        load = np.zeros(parts)
        groups = [[] for _ in range(parts)]
        for it in order:
            g = int(np.argmin(load))
            groups[g].append(int(it))
            load[g] += cost[it]
        return [np.sort(np.array(g, dtype=np.int64)) for g in groups if g]

    def compute(self, common, x_next, sel, sign, starts, seeds):
        """Labels (C, m) for C items; common = dict(edge_index, n, b, temps, d, eps, beta_prior,
        beta, graph_key); the per-item arrays are numpy with items along the first axis (x_next:
        n x C)."""
        C = len(starts)
        K = len(common['temps'])
        cost = (sel.shape[1] + 1)*np.maximum(K - starts, 0) + 1.0
        alive = self.alive_slots()
        parts = max(1, len(alive))*self.tasks_per_worker
        pending = [dict(items=g, attempts=0, bad=set()) for g in self.split_items(cost, parts)]
        out = np.full((C, sel.shape[1]), np.nan, dtype=np.float32)
        tasks = {}

        def make(t):
            it = t['items']
            return dict(common, x_next=x_next[:, it], sel=sel[it], sign=sign[it],
                        starts=starts[it], seeds=seeds[it])

        def finish(t, lab):
            if lab.shape != (len(t['items']), sel.shape[1]) or not np.isfinite(lab).all():
                return False
            out[t['items']] = lab
            return True

        def requeue(t, slot, why):
            t['attempts'] += 1
            t['bad'].add(slot)
            self.stats['retries'] += 1
            if t['attempts'] >= self.max_attempts:
                raise SimLabelError(f'label task of {len(t["items"])} items failed {t["attempts"]} '
                                    f'times; last: {why}')
            pending.append(t)

        while pending or any(s['inflight'] for s in self.slots):
            alive = self.alive_slots()
            if not alive:
                self._run_fallback(pending, make, finish)
                pending = []
                break
            # dispatch: least-loaded alive worker that has not failed this task, if possible
            while pending:
                free = [i for i in alive if len(self.slots[i]['inflight']) < self.tasks_per_worker]
                if not free:
                    break
                t = pending.pop(0)
                cand = [i for i in free if i not in t['bad']] or free
                i = min(cand, key=lambda j: len(self.slots[j]['inflight']))
                self._tid += 1
                t['t0'] = time.time()
                tasks[self._tid] = t
                self.slots[i]['inflight'][self._tid] = t
                self.slots[i]['q'].put((self._tid, make(t)))
                self.stats['tasks'] += 1
            try:
                slot, tid, status, payload = self.result_q.get(timeout=1.0)
            except queue.Empty:
                slot = None
            if slot is not None and status != 'ready' and tid in self.slots[slot]['inflight']:
                t = self.slots[slot]['inflight'].pop(tid)
                if status == 'ok' and finish(t, payload):
                    self.slots[slot]['errors'] = 0
                elif status == 'oom' and len(t['items']) > 1:
                    self.stats['splits'] += 1
                    h = len(t['items'])//2
                    self.log(f'[sim-labels] worker {slot}: out of memory on {len(t["items"])} items; split')
                    pending += [dict(items=t['items'][:h], attempts=t['attempts'], bad=set(t['bad'])),
                                dict(items=t['items'][h:], attempts=t['attempts'], bad=set(t['bad']))]
                else:
                    why = payload if status != 'ok' else 'bad shape or non-finite labels'
                    self.log(f'[sim-labels] worker {slot} (GPU {self.slots[slot]["device"]}) task failed: '
                             f'{str(why).strip().splitlines()[-1] if why else why}')
                    self.slots[slot]['errors'] += 1
                    requeue(t, slot, why)
                    if self.slots[slot]['errors'] >= 2:
                        for lt in self._drop_or_restart(slot, '2 failures in a row'):
                            requeue(lt, slot, 'lost with its worker')
            # dead or hung workers
            now = time.time()
            for i in self.alive_slots():
                s = self.slots[i]
                hung = any(now - t['t0'] > self.timeout for t in s['inflight'].values())
                if not s['proc'].is_alive() or hung:
                    for lt in self._drop_or_restart(i, 'process died' if not s['proc'].is_alive()
                                                    else f'task over {self.timeout:.0f}s'):
                        requeue(lt, i, 'lost with its worker')
        if np.isnan(out).any():
            raise SimLabelError('some labels were not computed')
        return out

    def _run_fallback(self, pending, make, finish):
        if self.fallback is None:
            raise SimLabelError('all label workers are gone and no fallback device is set')
        self.stats['fallback'] += len(pending)
        self.log(f'[sim-labels] no worker left: computing {len(pending)} task(s) on {self.fallback}')
        for t in pending:
            if not finish(t, compute_task(make(t), self.fallback, self._fallback_cache)):
                raise SimLabelError('fallback produced bad labels')


def simulated_labels(pool, device, common, x_next, sel, sign, starts, seeds):
    """Labels through the pool, or in-process on device when pool is None."""
    if pool is None:
        task = dict(common, x_next=x_next, sel=sel, sign=sign, starts=starts, seeds=seeds)
        return compute_task(task, device)
    return pool.compute(common, x_next, sel, sign, starts, seeds)
