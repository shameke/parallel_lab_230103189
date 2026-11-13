import math
import statistics
import multiprocessing as mp

from harness import run_benchmark, emit_verify_tag
import time

STUDENT_ID = 230103189
P_PHYS = 14
P_LOG = 20
N_STRONG = 1_600_000
N0_WEAK = 250_000

stage_a_times = []  # время Stage A в каждом прогоне (мс)


def stage_a(n: int) -> float:
    """Строго последовательная часть: 0.05*N элементов, 12 трансцендентных операций на элемент."""
    m = int(0.05 * n)
    acc = 0.0
    for i in range(m):
        x = i * 1e-3 + 0.5
        for _ in range(6):
            x = math.sin(x) + 1.5
            acc += math.sqrt(x)
    return acc


def stage_b_chunk(bounds) -> float:
    """Параллельное ядро: 35 итераций multiply-add + sin на элемент, без общих записей."""
    start, end = bounds
    total = 0.0
    sin = math.sin
    for i in range(start, end):
        x = i * 1e-6 + 0.1
        for _ in range(35):
            x = x * 0.999 + 0.001
            total += sin(x)
    return total


def make_chunks(n: int, p: int):
    base, rem = divmod(n, p)
    chunks, s = [], 0
    for t in range(p):
        e = s + base + (1 if t < rem else 0)
        chunks.append((s, e))
        s = e
    return chunks


def run_trial(pool, n: int, p: int):
    t0 = time.perf_counter_ns()
    stage_a(n)
    stage_a_times.append((time.perf_counter_ns() - t0) / 1e6)
    pool.map(stage_b_chunk, make_chunks(n, p), chunksize=1)


def measure(n: int, p: int):
    stage_a_times.clear()
    with mp.Pool(processes=p) as pool:   # пул создаётся вне замера
        pool.map(stage_b_chunk, [(0, 10)] * p, chunksize=1)  # прогрев процессов
        st = run_benchmark(lambda: run_trial(pool, n, p),
                           warmup_runs=1, timed_runs=5)
    ta = statistics.median(stage_a_times[-5:])
    return st.median_ms, ta


def main():
    ps = list(range(1, P_LOG + 1))

    # ---------- 2A: strong scaling ----------
    print(f"\n### Experiment 2A: strong scaling, N = {N_STRONG}")
    print("| p | T_ms | S_p | E_p | marginal dS |")
    print("|---|---|---|---|---|")
    t1 = ta1 = None
    prev_s = 0.0
    s_strong_last = None
    for p in ps:
        t, ta = measure(N_STRONG, p)
        if p == 1:
            t1, ta1 = t, ta
        sp = t1 / t
        print(f"| {p} | {t:.1f} | {sp:.3f} | {sp/p:.3f} | {sp-prev_s:+.3f} |", flush=True)
        prev_s = sp
        s_strong_last = sp

    s = ta1 / t1
    s_inf = 1 / s
    s_amdahl_plog = 1 / (s + (1 - s) / P_LOG)
    print(f"\nserial fraction s = {ta1:.1f} / {t1:.1f} = {s:.5f}")
    print(f"Amdahl S_inf = {s_inf:.2f}")
    print(f"Amdahl S(p={P_LOG}) = {s_amdahl_plog:.3f}, observed = {s_strong_last:.3f}")

    # ---------- 2B: weak scaling ----------
    print(f"\n### Experiment 2B: weak scaling, N0 = {N0_WEAK}")
    print("| p | N | T_ms | E_weak | alpha | S_Gustafson |")
    print("|---|---|---|---|---|---|")
    tw1 = None
    s_gust_last = None
    for p in ps:
        n = p * N0_WEAK
        t, ta = measure(n, p)
        if p == 1:
            tw1 = t
        ew = tw1 / t
        alpha = ta / t
        sg = p - alpha * (p - 1)
        print(f"| {p} | {n} | {t:.1f} | {ew:.3f} | {alpha:.4f} | {sg:.3f} |", flush=True)
        s_gust_last = sg

    print()
    print(emit_verify_tag("PHASE2", STUDENT_ID, s_strong_last, s_gust_last))


if __name__ == "__main__":
    main()