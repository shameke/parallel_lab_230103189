import random
import multiprocessing as mp
from multiprocessing import shared_memory
from harness import emit_verify_tag
from labcfg import *

BYTES = 64_000_000

# ---------- 3A: roofline ----------
def bw_worker(bar, na, nb, s, e):
    sa = shared_memory.SharedMemory(name=na); sb = shared_memory.SharedMemory(name=nb)
    mv_a, mv_b = sa.buf, sb.buf
    bar.wait()
    mv_a[s:e] = mv_b[s:e]
    del mv_a, mv_b
    sa.close(); sb.close()

def fma_worker(bar, iters):
    a = 1.0000001; b = 1e-9
    x0=0.1; x1=0.2; x2=0.3; x3=0.4; x4=0.5; x5=0.6; x6=0.7; x7=0.8
    x8=0.9; x9=1.0; x10=1.1; x11=1.2; x12=1.3; x13=1.4; x14=1.5; x15=1.6
    bar.wait()
    for _ in range(iters):
        x0=x0*a+b; x1=x1*a+b; x2=x2*a+b; x3=x3*a+b
        x4=x4*a+b; x5=x5*a+b; x6=x6*a+b; x7=x7*a+b
        x8=x8*a+b; x9=x9*a+b; x10=x10*a+b; x11=x11*a+b
        x12=x12*a+b; x13=x13*a+b; x14=x14*a+b; x15=x15*a+b

# ---------- 3B: GEMM ----------
def gemm_worker(bar, variant, nA, nB, nC, N, r0, r1, BS):
    sA = shared_memory.SharedMemory(name=nA); sB = shared_memory.SharedMemory(name=nB)
    sC = shared_memory.SharedMemory(name=nC)
    A = sA.buf.cast('d'); B = sB.buf.cast('d'); C = sC.buf.cast('d')
    bar.wait()
    if variant == 1:                                   # i-j-k, шаг N по B
        for i in range(r0, r1):
            ib = i * N
            for j in range(N):
                s = 0.0
                for k in range(N):
                    s += A[ib + k] * B[k * N + j]
                C[ib + j] = s
    elif variant == 2:                                 # i-k-j, unit-stride
        for i in range(r0, r1):
            ib = i * N
            for j in range(N):
                C[ib + j] = 0.0
            for k in range(N):
                a_ik = A[ib + k]; kb = k * N
                for j in range(N):
                    C[ib + j] += a_ik * B[kb + j]
    else:                                              # 6 циклов, блоки BS
        for i in range(r0, r1):
            for j in range(N):
                C[i * N + j] = 0.0
        for ii in range(r0, r1, BS):
            for jj in range(0, N, BS):
                for kk in range(0, N, BS):
                    for i in range(ii, min(ii + BS, r1)):
                        ib = i * N
                        for k in range(kk, min(kk + BS, N)):
                            a_ik = A[ib + k]; kb = k * N
                            for j in range(jj, min(jj + BS, N)):
                                C[ib + j] += a_ik * B[kb + j]
    A.release(); B.release(); C.release()
    sA.close(); sB.close(); sC.close()

if __name__ == "__main__":
    pphys, sid = get_args()
    p = pphys

    # ---- 3A ----
    sa = shared_memory.SharedMemory(create=True, size=BYTES)
    sb = shared_memory.SharedMemory(create=True, size=BYTES)
    sa.buf[:] = bytes(range(256)) * (BYTES // 256) + bytes(BYTES % 256)
    sb.buf[:] = bytes(BYTES)
    bounds = [(k * BYTES // p, (k + 1) * BYTES // p) for k in range(p)]
    st = bench_ms(lambda: run_workers(bw_worker, [(sa.name, sb.name, s, e) for s, e in bounds]))
    BW = 2 * BYTES / ((st.median_ms / 1e3) * 1e9)
    sa.close(); sa.unlink(); sb.close(); sb.unlink()
    print(f"[3A] DRAM copy: median {st.median_ms:.1f} ms (IQR {st.iqr_ms:.1f}) -> BW_sustained = {BW:.2f} GB/s")

    iters = int(6.0e7 * SCALE) // 32                  # 16 FMA * 2 FLOP = 32 FLOP/итерация
    flops_w = iters * 32
    st = bench_ms(lambda: run_workers(fma_worker, [(iters,)] * p))
    P_sus = p * flops_w / ((st.median_ms / 1e3) * 1e9)
    print(f"[3A] FMA: median {st.median_ms:.1f} ms -> P_sustained = {P_sus:.4f} GFLOP/s (суммарно, {p} процессов)")
    print(f"[3A] Ridge point I* = P/BW = {P_sus/BW:.5f} FLOP/byte")

    # ---- 3B ----
    N = 256 if not QUICK else 64
    rng = random.Random(sid)
    shs = [shared_memory.SharedMemory(create=True, size=N * N * 8) for _ in range(3)]
    mvs = [s.buf.cast('d') for s in shs]
    for idx in range(N * N):
        mvs[0][idx] = rng.random(); mvs[1][idx] = rng.random()
    names = [s.name for s in shs]
    rows = [(k * N // p, (k + 1) * N // p) for k in range(p)]

    def run_variant(v, BS=0):
        return bench_ms(lambda: run_workers(
            gemm_worker, [(v, *names, N, r0, r1, BS) for r0, r1 in rows]))
    def gflops(ms):
        return 2 * N**3 / ((ms / 1e3) * 1e9)

    st1 = run_variant(1); g1 = gflops(st1.median_ms)
    print(f"\n[3B] V1 naive i-j-k : {st1.median_ms:9.1f} ms  {g1:.4f} GFLOP/s")
    st2 = run_variant(2); g2 = gflops(st2.median_ms)
    C2 = list(mvs[2])
    print(f"[3B] V2 i-k-j       : {st2.median_ms:9.1f} ms  {g2:.4f} GFLOP/s")

    print("[3B] V3 tile sweep:  BS | median_ms | GFLOP/s | W_tile KiB | I_tiled")
    best = (None, -1.0, None)
    for BS in (8, 16, 32, 64, 96, 128):
        st3 = run_variant(3, BS); g3 = gflops(st3.median_ms)
        C3 = list(mvs[2])
        err = max(abs(x - y) for x, y in zip(C3, C2))
        flag = "OK" if err < 1e-9 else "MISMATCH!"
        print(f"                  {BS:4d} | {st3.median_ms:9.1f} | {g3:7.4f} | {3*BS*BS*8/1024:9.1f}  | {BS/12:6.2f}   max|C3-C2|={err:.1e} {flag}")
        if g3 > best[1]:
            best = (BS, g3, st3.median_ms)
    BSs, g3s, _ = best
    print(f"\nBS* = {BSs}, W_tile = {3*BSs*BSs*8/1024:.1f} KiB (сравните с L1d / L2), "
          f"I_tiled ≈ {BSs/12:.2f} FLOP/B")
    I = BSs / 12
    print(f"Roofline-потолок при I={I:.2f}: min(P_sus, I*BW) = {min(P_sus, I*BW):.4f} GFLOP/s; "
          f"достигнуто V3 = {g3s:.4f}, V1 = {g1:.4f}")
    print()
    print(emit_verify_tag("PHASE3", sid, g1, g3s))

    for m in mvs: m.release()
    for s in shs: s.close(); s.unlink()
