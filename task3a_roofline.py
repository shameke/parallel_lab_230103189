import multiprocessing as mp
from multiprocessing import shared_memory

from harness import run_benchmark

P_PHYS = 14
SIZE = 96 * 10**6          # 96 MB на буфер
FMA_FLOPS_PER_WORKER = 80_000_000
ITERS = FMA_FLOPS_PER_WORKER // (2 * 16)   # 16 FMA за итерацию, 2 FLOP на FMA

_mv_a = _mv_b = None
_shms = []


def init_copy(name_a, name_b):
    global _mv_a, _mv_b
    sa = shared_memory.SharedMemory(name=name_a)
    sb = shared_memory.SharedMemory(name=name_b)
    _shms.extend([sa, sb])
    _mv_a, _mv_b = sa.buf, sb.buf


def copy_slice(bounds):
    s, e = bounds
    _mv_a[s:e] = _mv_b[s:e]


def fma_worker(iters):
    a = 0.999999
    b = 1e-6
    x0 = 0.1; x1 = 0.2; x2 = 0.3; x3 = 0.4
    x4 = 0.5; x5 = 0.6; x6 = 0.7; x7 = 0.8
    x8 = 0.9; x9 = 1.0; x10 = 1.1; x11 = 1.2
    x12 = 1.3; x13 = 1.4; x14 = 1.5; x15 = 1.6
    for _ in range(iters):
        x0 = x0 * a + b
        x1 = x1 * a + b
        x2 = x2 * a + b
        x3 = x3 * a + b
        x4 = x4 * a + b
        x5 = x5 * a + b
        x6 = x6 * a + b
        x7 = x7 * a + b
        x8 = x8 * a + b
        x9 = x9 * a + b
        x10 = x10 * a + b
        x11 = x11 * a + b
        x12 = x12 * a + b
        x13 = x13 * a + b
        x14 = x14 * a + b
        x15 = x15 * a + b
    return (x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7 +
            x8 + x9 + x10 + x11 + x12 + x13 + x14 + x15)


def main():
    # ---------- 3A.1: DRAM bandwidth ----------
    sa = shared_memory.SharedMemory(create=True, size=SIZE)
    sb = shared_memory.SharedMemory(create=True, size=SIZE)
    try:
        sa.buf[:SIZE] = bytes(SIZE)
        sb.buf[:SIZE] = b"\x01" * SIZE        # физически занимаем страницы
        step = SIZE // P_PHYS
        bounds = [(t * step, SIZE if t == P_PHYS - 1 else (t + 1) * step)
                  for t in range(P_PHYS)]
        with mp.Pool(P_PHYS, initializer=init_copy,
                     initargs=(sa.name, sb.name)) as pool:
            st = run_benchmark(lambda: pool.map(copy_slice, bounds, chunksize=1),
                               warmup_runs=3, timed_runs=9)
        bw = 2 * SIZE / (st.median_ms * 1e-3 * 1e9)
        print(f"Copy median = {st.median_ms:.4f} ms (IQR {st.iqr_ms:.4f})")
        print(f"BW_sustained = {bw:.4f} GB/s")
    finally:
        sa.close(); sa.unlink()
        sb.close(); sb.unlink()

    # ---------- 3A.2: compute ceiling ----------
    with mp.Pool(P_PHYS) as pool:
        st = run_benchmark(
            lambda: pool.map(fma_worker, [ITERS] * P_PHYS, chunksize=1),
            warmup_runs=2, timed_runs=7)
    gflops = P_PHYS * FMA_FLOPS_PER_WORKER / (st.median_ms * 1e-3 * 1e9)
    print(f"FMA median = {st.median_ms:.4f} ms (IQR {st.iqr_ms:.4f})")
    print(f"P_sustained = {gflops:.4f} GFLOP/s")

    # ---------- 3A.3: ridge point ----------
    print(f"Ridge I* = {gflops / bw:.6f} FLOP/Byte")


if __name__ == "__main__":
    main()
