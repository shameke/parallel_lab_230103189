import time
import multiprocessing as mp
from harness import emit_verify_tag

STUDENT_ID = 230103189
N_CALLS = 2 * 10**6


def main():
    p_log = mp.cpu_count()
    clock = time.perf_counter_ns

    # прогрев
    for _ in range(10_000):
        clock()

    t0 = clock()
    for _ in range(N_CALLS):
        clock()
    t1 = clock()

    latency_ns = (t1 - t0) / N_CALLS

    print(f"mp.cpu_count() = {p_log}")
    print(f"Clock reads    = {N_CALLS}")
    print(f"Avg call latency = {latency_ns:.2f} ns")
    print(emit_verify_tag("PHASE0", STUDENT_ID, latency_ns, float(p_log)))


if __name__ == "__main__":
    main()