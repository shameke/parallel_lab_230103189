import math, statistics
import multiprocessing as mp
from harness import run_benchmark, emit_verify_tag
from labcfg import *

def stage_a(n):                      # серийно: 10 math-операций на элемент
    acc = 0.0
    for i in range(n):
        x = i * 1e-3 + 1.0
        for _ in range(5):
            x = math.sqrt(math.sin(x) + 2.0)
        acc += x
    return acc

def stage_b(rng):                    # параллельно: 30 FMA/trig операций на элемент
    lo, hi = rng
    acc = 0.0
    for i in range(lo, hi):
        x = i * 1e-6 + 0.5
        for _ in range(10):
            x = x * 1.000001 + 0.25          # FMA
            y = math.sin(x)                  # trig
            x = y * 0.5 + x * 0.5            # FMA
        acc += x
    return acc

def measure(p, N):
    pool = mp.Pool(processes=p)                      # вне таймируемого цикла
    pool.map(stage_b, [(0, 10)] * p, chunksize=1)    # прогрев пула
    tAs = []
    chunks = [(k * N // p, (k + 1) * N // p) for k in range(p)]
    def trial():
        t0 = time.perf_counter_ns()
        stage_a(int(0.05 * N))
        tAs.append((time.perf_counter_ns() - t0) / 1e6)
        pool.map(stage_b, chunks, chunksize=1)
    st = run_benchmark(trial, WARM, RUNS)
    tA = statistics.median(tAs[-RUNS:])
    pool.close(); pool.join()
    return st.median_ms, tA

if __name__ == "__main__":
    pphys, sid = get_args()
    plog = mp.cpu_count()
    import time

    N = int(1.2e6 * SCALE)
    print(f"== Strong scaling, N = {N} ==")
    print("  p |   T_p ms |   S_p |   E_p |  marginal dS")
    Ts, Ss, tA1 = {}, {}, None
    for p in range(1, plog + 1):
        T, tA = measure(p, N)
        Ts[p] = T
        if p == 1:
            tA1 = tA
        Ss[p] = Ts[1] / T
        d = Ss[p] - Ss.get(p - 1, 0.0) if p > 1 else float("nan")
        print(f"{p:3d} | {T:8.1f} | {Ss[p]:5.2f} | {Ss[p]/p:5.2f} | {d:7.2f}")
    s = tA1 / Ts[1]
    print(f"serial fraction s = {s:.4f}  -> Amdahl cap S_inf = {1/s:.2f};  observed S(P_log) = {Ss[plog]:.2f}")
    if pphys + 1 <= plog:
        print(f"marginal speedup: P_phys-1 -> P_phys: {Ss[pphys]-Ss[pphys-1]:.2f}, "
              f"P_phys -> P_phys+1: {Ss[pphys+1]-Ss[pphys]:.2f}")

    N0 = int(2.0e5 * SCALE)
    print(f"\n== Weak scaling, N0 = {N0} per worker ==")
    print("  p |   T_p ms | E_weak | alpha | S_Gustafson")
    Tw, Sg = {}, {}
    for p in range(1, plog + 1):
        T, tA = measure(p, p * N0)
        Tw[p] = T
        alpha = tA / T
        Sg[p] = p - alpha * (p - 1)
        print(f"{p:3d} | {T:8.1f} | {Tw[1]/T:6.2f} | {alpha:5.3f} | {Sg[p]:7.2f}")

    print()
    print(emit_verify_tag("PHASE2", sid, Ss[plog], Sg[plog]))
