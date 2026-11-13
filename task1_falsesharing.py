import ctypes
import multiprocessing as mp
from multiprocessing import shared_memory

from harness import run_benchmark, emit_verify_tag

STUDENT_ID = 230103189
P_PHYS = 14
P_LOG = 20
M = 24_000_000
SHM_SIZE = 4096


def fs_worker(shm_name: str, slot_idx: int, iters: int) -> None:
    shm = shared_memory.SharedMemory(name=shm_name)
    arr = (ctypes.c_uint64 * 512).from_buffer(shm.buf)
    acc = arr[slot_idx]
    mask = 0xFFFFFFFFFFFFFFFF
    mult = 6364136223846793005
    for i in range(iters):
        acc = (acc * mult + (i | 1)) & mask
        arr[slot_idx] = acc
    del arr
    shm.close()


def run_once(shm, p: int, stride: int) -> None:
    shm.buf[:SHM_SIZE] = bytes(SHM_SIZE)  # обнуляем буфер перед запуском
    iters = M // p
    procs = [
        mp.Process(target=fs_worker, args=(shm.name, t * stride, iters))
        for t in range(p)
    ]
    for pr in procs:
        pr.start()
    for pr in procs:
        pr.join()


def bench(shm, p: int, stride: int):
    return run_benchmark(lambda: run_once(shm, p, stride),
                         warmup_runs=2, timed_runs=7)


def karp_flatt(sp: float, p: int) -> float:
    return (1 / sp - 1 / p) / (1 - 1 / p)


def main():
    shm = shared_memory.SharedMemory(create=True, size=SHM_SIZE)
    try:
        # ---- Таблица 1A: stride sweep при p = P_PHYS ----
        print(f"\n### Table 1A (p = {P_PHYS})")
        print("| stride | bytes | median_ms | IQR_ms | Mops/s |")
        print("|---|---|---|---|---|")
        med_at_pphys = {}
        for stride in [1, 2, 4, 8, 16, 32]:
            st = bench(shm, P_PHYS, stride)
            mops = M / (st.median_ms * 1e3)
            med_at_pphys[stride] = st.median_ms
            print(f"| {stride} | {stride*8} | {st.median_ms:.2f} | "
                  f"{st.iqr_ms:.2f} | {mops:.2f} |", flush=True)

        # ---- Таблица 1B: масштабирование по p ----
        ps = [1, 2, 4, 8, 14, 16, 20]
        for stride in [1, 16]:
            print(f"\n### Table 1B, stride = {stride}")
            print("| p | median_ms | S_p | E_p | e(p) |")
            print("|---|---|---|---|---|")
            t1 = None
            for p in ps:
                st = bench(shm, p, stride)
                if p == 1:
                    t1 = st.median_ms
                sp = t1 / st.median_ms
                ep = sp / p
                e = f"{karp_flatt(sp, p):.4f}" if p >= 2 else "-"
                print(f"| {p} | {st.median_ms:.2f} | {sp:.3f} | "
                      f"{ep:.3f} | {e} |", flush=True)

        print()
        print(emit_verify_tag("PHASE1", STUDENT_ID,
                              med_at_pphys[1], med_at_pphys[16]))
    finally:
        shm.close()
        shm.unlink()


if __name__ == "__main__":
    main()