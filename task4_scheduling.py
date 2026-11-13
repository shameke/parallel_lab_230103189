import math
import multiprocessing as mp
import time

from harness import run_benchmark, emit_verify_tag

STUDENT_ID = 230103189
P_PHYS = 14
N = 12_000
CHUNKS = [1, 4, 16, 64, 256, 1024]
WARMUP, TIMED = 1, 3


def work(i: int) -> float:
    k = 1 + (i * i) // 3600            # K(i) = 1 + floor((i/60)^2)
    x = i * 1e-3 + 0.5
    acc = 0.0
    sin, cos, sqrt = math.sin, math.cos, math.sqrt
    for _ in range(k):
        x += 1e-3
        acc += sqrt(abs(sin(x)) + abs(cos(x)))
    return acc


def block_worker(t, p, times):
    t0 = time.perf_counter_ns()
    for i in range(t * N // p, (t + 1) * N // p):
        work(i)
    times[t] = (time.perf_counter_ns() - t0) / 1e6


def cyclic_worker(t, p, times):
    t0 = time.perf_counter_ns()
    for i in range(t, N, p):
        work(i)
    times[t] = (time.perf_counter_ns() - t0) / 1e6


def dynamic_worker(t, p, times, counter, chunk):
    t0 = time.perf_counter_ns()
    while True:
        with counter.get_lock():
            start = counter.value
            counter.value = start + chunk
        if start >= N:
            break
        for i in range(start, min(start + chunk, N)):
            work(i)
    times[t] = (time.perf_counter_ns() - t0) / 1e6


def run_once(kind, chunk, times):
    p = P_PHYS
    counter = mp.Value("i", 0)
    for t in range(p):
        times[t] = 0.0
    if kind == "block":
        args, target = (lambda t: (t, p, times)), block_worker
    elif kind == "cyclic":
        args, target = (lambda t: (t, p, times)), cyclic_worker
    else:
        args, target = (lambda t: (t, p, times, counter, chunk)), dynamic_worker
    procs = [mp.Process(target=target, args=args(t)) for t in range(p)]
    for pr in procs:
        pr.start()
    for pr in procs:
        pr.join()


def lam(times):
    tw = list(times)
    return max(tw) / (sum(tw) / len(tw))


def measure(kind, chunk=0):
    times = mp.Array("d", P_PHYS, lock=False)
    lams = []

    def fn():
        run_once(kind, chunk, times)
        lams.append(lam(times))

    st = run_benchmark(fn, warmup_runs=WARMUP, timed_runs=TIMED)
    lams = sorted(lams[-TIMED:])
    return st.median_ms, lams[len(lams) // 2]


def main():
    rows = []
    print("| strategy | C | T_wall_ms | lambda |")
    print("|---|---|---|---|")

    t, l = measure("block")
    rows.append(("block", 0, t, l))
    print(f"| 1 block | - | {t:.4f} | {l:.4f} |", flush=True)
    t_block = t

    t, l = measure("cyclic")
    print(f"| 2 cyclic | - | {t:.4f} | {l:.4f} |", flush=True)

    best_c, best_t = None, float("inf")
    for c in CHUNKS:
        t, l = measure("dynamic", c)
        print(f"| 3 dynamic | {c} | {t:.4f} | {l:.4f} |", flush=True)
        if t < best_t:
            best_c, best_t = c, t

    print(f"\nC* = {best_c}")
    print(f"hash inputs: T_wall(S1)={t_block:.4f}, T_wall(S3,C*)={best_t:.4f}")
    print(emit_verify_tag("PHASE4", STUDENT_ID, t_block, best_t))


if __name__ == "__main__":
    main()
