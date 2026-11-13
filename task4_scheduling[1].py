import math, time
import multiprocessing as mp
from harness import emit_verify_tag
from labcfg import *

def work(i):
    K = 1 + (i // 40) ** 2
    x = i * 1e-3 + 0.5
    for _ in range(K):
        x = math.sqrt(abs(math.sin(x)) + abs(math.cos(x))) + 0.1
    return x

def sched_worker(bar, mode, t, p, N, C, counter, tw):
    bar.wait()
    t0 = time.perf_counter()
    acc = 0.0
    if mode == "block":
        for i in range(t * N // p, (t + 1) * N // p):
            acc += work(i)
    elif mode == "cyclic":
        for i in range(t, N, p):
            acc += work(i)
    else:
        while True:
            with counter.get_lock():
                s = counter.value
                counter.value = s + C
            if s >= N:
                break
            for i in range(s, min(s + C, N)):
                acc += work(i)
    tw[t] = time.perf_counter() - t0

def run_once(mode, p, N, C=1):
    counter = mp.Value('i', 0)
    tw = mp.Array('d', p)
    twall = run_workers(sched_worker, [(mode, t, p, N, C, counter, tw) for t in range(p)])
    ts = list(tw)
    lam = max(ts) / (sum(ts) / p)
    return twall, lam

def measure(mode, p, N, C=1):
    for _ in range(WARM):
        run_once(mode, p, N, C)
    res = sorted(run_once(mode, p, N, C) for _ in range(RUNS))
    return res[RUNS // 2]                      # (T_wall, lambda) медианного прогона

if __name__ == "__main__":
    pphys, sid = get_args()
    p = pphys
    N = int(8000 * SCALE) if not QUICK else 800
    print(f"N = {N}, p = {p}")
    print("strategy            | T_wall ms | lambda")
    tb, lb = measure("block", p, N)
    print(f"block static        | {tb:9.1f} | {lb:6.3f}")
    tc, lc = measure("cyclic", p, N)
    print(f"cyclic static       | {tc:9.1f} | {lc:6.3f}")
    best = (None, float("inf"))
    for C in (1, 8, 64, 512):
        td, ld = measure("dynamic", p, N, C)
        print(f"dynamic C={C:<4d}      | {td:9.1f} | {ld:6.3f}")
        if td < best[1]:
            best = (C, td)
    print(f"\nC* = {best[0]}")
    print(emit_verify_tag("PHASE4", sid, tb, best[1]))
