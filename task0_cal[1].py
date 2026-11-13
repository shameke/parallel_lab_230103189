import sys, time
import multiprocessing as mp
from harness import emit_verify_tag

if __name__ == "__main__":
    sid = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    plog = mp.cpu_count()
    n = 10**6
    for _ in range(1000):
        time.perf_counter_ns()
    t0 = time.perf_counter_ns()
    for _ in range(n):
        time.perf_counter_ns()
    avg_ns = (time.perf_counter_ns() - t0) / n
    print(f"P_log = {plog}")
    print(f"avg perf_counter_ns call latency = {avg_ns:.2f} ns")
    print(emit_verify_tag("PHASE0", sid, float(plog), avg_ns))
