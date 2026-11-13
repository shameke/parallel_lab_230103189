import ctypes
import multiprocessing as mp
from multiprocessing import shared_memory
from harness import emit_verify_tag
from labcfg import *

MASK = 0xFFFFFFFFFFFFFFFF

def fs_worker(bar, shm_name, slot_idx, iters):
    shm = shared_memory.SharedMemory(name=shm_name)
    arr = (ctypes.c_uint64 * 512).from_buffer(shm.buf)
    acc = arr[slot_idx]
    bar.wait()
    for i in range(iters):
        acc = (acc * 6364136223846793005 + (i | 1)) & MASK
        arr[slot_idx] = acc          # store в общую кэш-линию
    del arr
    shm.close()

if __name__ == "__main__":
    pphys, sid = get_args()
    plog = mp.cpu_count()
    M = int(1.6e7 * SCALE)
    shm = shared_memory.SharedMemory(create=True, size=4096)
    try:
        def measure(p, stride):
            if p * stride > 512:
                return None
            iters = M // p
            args = [(shm.name, t * stride, iters) for t in range(p)]
            return bench_ms(lambda: run_workers(fs_worker, args))

        print(f"== Stride sweep, p = P_phys = {pphys}, M = {M} ==")
        print("stride | bytes | median_ms |  IQR_ms | Mops/s")
        med = {}
        for stride in (1, 2, 4, 8, 16, 32):
            st = measure(pphys, stride)
            if st is None:
                print(f"{stride:6d} | skipped (p*stride > 512 slots)"); continue
            med[stride] = st.median_ms
            print(f"{stride:6d} | {stride*8:5d} | {st.median_ms:9.1f} | {st.iqr_ms:7.1f} | {M/(st.median_ms/1e3)/1e6:7.2f}")

        print(f"\n== Process scaling (p up to P_log = {plog}) ==")
        for stride in (1, 16):
            print(f"-- stride = {stride} --")
            print("  p | median_ms |   S_p |   E_p |  e(p)")
            T1 = None
            for p in p_list(plog):
                st = measure(p, stride)
                if st is None:
                    print(f"{p:3d} | skipped"); continue
                T = st.median_ms
                if p == 1:
                    T1 = T
                S = T1 / T
                kf = f"{karp_flatt(S, p):6.3f}" if p >= 2 else "   -  "
                print(f"{p:3d} | {T:9.1f} | {S:5.2f} | {S/p:5.2f} | {kf}")
            if stride == 16 and pphys not in p_list(plog):
                pass

        m1 = med.get(1, float("nan")); m2 = med.get(16, float("nan"))
        print()
        print(emit_verify_tag("PHASE1", sid, m1, m2))
    finally:
        shm.close(); shm.unlink()
