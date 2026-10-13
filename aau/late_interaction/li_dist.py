"""Distanze di interazione tardiva fra insiemi di token (f_i, a_i): Chamfer simmetrico pesato e OT entropico.

Funzioni torch su GPU, usate da ``li_tokens.py run`` (insiemi interi) e ``li_cost.py`` (una coppia alla volta).
Costo di terra c(f, g) = ||f - g||_2 sulle feature grezze (nessuna normalizzazione L2: REGOLE.md, sez. 2), calcolato
come sqrt(max(|f|^2 + |g|^2 - 2 f.g, 0)) in float32 SENZA TF32 (il processo parte con
``TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0``: il container NGC lo mette a 1 e forzerebbe il TF32 in ogni prodotto di cuBLAS;
qui ``tf32(False)`` intorno a ogni prodotto), sulle feature centrate sulla media dell'insieme (le distanze non cambiano,
l'errore di cancellazione si riduce). Controllo in float64 su CPU: ``check_pairs``.

  * ``chamfer``: d(A, B) = 1/2 [sum_i a_i min_j c(f_i, g_j) + sum_j b_j min_i c(f_i, g_j)], pesi a, b a somma 1;
  * ``sinkhorn``: OT entropico con masse a, b, costo c, epsilon fisso; valore = <P, C> del piano entropico (senza il
    termine d'entropia, come il GW entropico di ``gp_gw``). Sinkhorn-Knopp a blocchi di coppie (``sinkhorn_batch``:
    l'iterazione di ``ot.sinkhorn``; ripiego nel dominio dei logaritmi, ``sinkhorn_log_batch``, dove il nucleo va in
    underflow), controllato contro ``ot.sinkhorn(..., method="sinkhorn_log")`` in float64 (``check_pot``); arresto
    quando l'errore L1 sul marginale delle righe e' sotto ``SK_TOL`` (colonne esatte dopo l'aggiornamento di v) o a
    ``SK_MAX_ITER`` iterazioni (la coppia resta, contata come non convergente).
"""

from __future__ import annotations

import contextlib
import time

import numpy as np
import torch

import li_common as lc


@contextlib.contextmanager
def tf32(on: bool):
    """TF32 di cuBLAS e cuDNN acceso / spento nel blocco (richiede ``TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0``)."""
    prev = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = on
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = prev


def check_no_tf32(device) -> float:
    """Errore relativo di un prodotto float32 contro float64 con TF32 spento: ~1e-7 (TF32 darebbe ~1e-3)."""
    g = torch.Generator(device="cpu").manual_seed(0)
    A, B = torch.randn(512, 256, generator=g, dtype=torch.float64), torch.randn(256, 512, generator=g, dtype=torch.float64)
    with tf32(False):
        P = (A.float().to(device) @ B.float().to(device)).double().cpu()
    err = float(((P - A @ B).abs().max() / (A @ B).abs().max()))
    if err > 1e-5:
        raise SystemExit(f"prodotto float32 con errore {err:.2e}: TF32 ancora attivo (TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0?)")
    return err


# ------------------------------------------------------------------------------------------ Chamfer

def half_sqdist(X: torch.Tensor, Y: torch.Tensor, sqY: torch.Tensor) -> torch.Tensor:
    """|y|^2 - 2 x.y (n, m) in un solo prodotto (manca |x|^2, costante per riga: si aggiunge dopo il minimo di riga)."""
    with tf32(False):
        return torch.addmm(sqY[None, :].expand(len(X), -1), X, Y.T, alpha=-2.0)


def chamfer_pair(X: torch.Tensor, a: torch.Tensor, Y: torch.Tensor, b: torch.Tensor, rows: int = 8192) -> float:
    """Chamfer simmetrico pesato fra due insiemi (una coppia; ``li_cost`` e controllo di ``chamfer_set``)."""
    sqX, sqY = (X * X).sum(1), (Y * Y).sum(1)
    colmin = torch.full((len(Y),), float("inf"), device=Y.device)
    s = torch.zeros((), dtype=torch.float64, device=X.device)
    for r in range(0, len(X), rows):
        H = half_sqdist(X[r:r + rows], Y, sqY)
        s += (a[r:r + rows].double() * (H.amin(1) + sqX[r:r + rows]).clamp_(min=0.0).sqrt().double()).sum()
        colmin = torch.minimum(colmin, H.add_(sqX[r:r + rows, None]).amin(0))
    return float(0.5 * (s + (b.double() * colmin.clamp_(min=0.0).sqrt().double()).sum()))


def chamfer_set(T: torch.Tensor, w: torch.Tensor, off: np.ndarray, subj: np.ndarray, stop: np.ndarray | None = None,
                rows: int = 8192, cols: int = 262144, log=None) -> np.ndarray:
    """Chamfer di TUTTE le coppie di mesh di soggetti diversi di un insieme (token concatenati ``T`` (N, d), pesi ``w``
    (N) a somma 1 per mesh, ``off`` (M + 1) inizi delle mesh, ``subj`` (M) soggetti, mesh dello stesso soggetto
    contigue). Per la mesh A, le mesh B dei soggetti successivi (fino a ``stop[A]`` escluso, se
    dato: es. il generatore degli held-out) sono una fetta contigua di ``T``: un prodotto a blocchi (``rows`` righe di
    A x ``cols`` colonne, mesh intere) da' entrambi i versi. Uscita (M, M), NaN dove non calcolata, diagonale 0."""
    M = len(off) - 1
    out = np.full((M, M), np.nan)
    np.fill_diagonal(out, 0.0)
    sq = (T * T).sum(1)
    cut = np.flatnonzero(subj[1:] != subj[:-1]) + 1                # fine dei blocchi dello stesso soggetto
    if len(set(subj[np.r_[0, cut]])) != len(cut) + 1:
        raise SystemExit("chamfer_set: mesh dello stesso soggetto non contigue")
    t0 = time.time()
    for k in range(M):
        nxt = int(cut[np.searchsorted(cut, k, side="right")]) if k < (cut[-1] if len(cut) else 0) else M
        end = M if stop is None else int(stop[k])
        if nxt >= end:
            continue
        X, a, sqX = T[off[k]:off[k + 1]], w[off[k]:off[k + 1]], sq[off[k]:off[k + 1]]
        acc_ab = torch.zeros(end - nxt, dtype=torch.float64, device=T.device)
        acc_ba = torch.zeros(end - nxt, dtype=torch.float64, device=T.device)
        l0 = nxt
        while l0 < end:                                             # blocchi di colonne = mesh intere
            l1 = l0 + 1
            while l1 < end and off[l1 + 1] - off[l0] <= cols:
                l1 += 1
            c0, c1 = int(off[l0]), int(off[l1])
            colmin = torch.full((c1 - c0,), float("inf"), device=T.device)
            rowmin = torch.empty((len(X), l1 - l0), device=T.device)
            for r in range(0, len(X), rows):
                H = half_sqdist(X[r:r + rows], T[c0:c1], sq[c0:c1])
                for q, l in enumerate(range(l0, l1)):
                    rowmin[r:r + rows, q] = H[:, off[l] - c0:off[l + 1] - c0].amin(1)
                rowmin[r:r + rows] += sqX[r:r + rows, None]
                colmin = torch.minimum(colmin, H.add_(sqX[r:r + rows, None]).amin(0))
                del H
            acc_ab[l0 - nxt:l1 - nxt] = (a.double()[:, None] * rowmin.clamp_(min=0.0).sqrt().double()).sum(0)
            cw = w[c0:c1].double() * colmin.clamp_(min=0.0).sqrt().double()
            seg = torch.as_tensor(np.repeat(np.arange(l1 - l0), np.diff(off[l0:l1 + 1])), device=T.device)
            acc_ba[l0 - nxt:l1 - nxt] = torch.zeros(l1 - l0, dtype=torch.float64, device=T.device).index_add_(0, seg, cw)
            l0 = l1
        d = (0.5 * (acc_ab + acc_ba)).cpu().numpy()
        out[k, nxt:end] = out[nxt:end, k] = d
        if log and (k % 50 == 0 or k == M - 1):
            log(f"chamfer: mesh {k + 1}/{M}, {time.time() - t0:.0f}s")
    return out


# ------------------------------------------------------------------------------- sottocampione e OT

def cost_batch(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """Costo c (B, n, m) fra i token di un blocco di coppie (X (B, n, d), Y (B, m, d))."""
    with tf32(False):
        G = torch.bmm(X, Y.transpose(1, 2))
    sqX, sqY = (X * X).sum(2), (Y * Y).sum(2)
    return G.mul_(-2.0).add_(sqX[:, :, None]).add_(sqY[:, None, :]).clamp_(min=0.0).sqrt_()


def chamfer_batch(C: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return 0.5 * ((a.double() * C.amin(2).double()).sum(1) + (b.double() * C.amin(1).double()).sum(1))


def sinkhorn_log_batch(C: torch.Tensor, a: torch.Tensor, b: torch.Tensor, eps: float, tol: float = lc.SK_TOL,
                       max_iter: int = lc.SK_MAX_ITER, check: int = lc.SK_CHECK):
    """Ripiego nel dominio dei logaritmi (``ot.bregman.sinkhorn_log``) per le coppie dove il nucleo va in underflow:
    (<P, C> (B,), iterazioni (B,), errore L1 finale sulle righe (B,)). Potenziali divisi per eps."""
    K = C / (-eps)
    la, lb = a.log(), b.log()
    f, g = torch.zeros_like(la), torch.zeros_like(lb)
    it_done, err = np.full(len(C), max_iter), np.full(len(C), np.nan)
    active = torch.ones(len(C), dtype=torch.bool, device=C.device)
    for it in range(1, max_iter + 1):
        f = torch.where(active[:, None], -torch.logsumexp(K + (g + lb)[:, None, :], dim=2), f)
        g = torch.where(active[:, None], -torch.logsumexp(K + (f + la)[:, :, None], dim=1), g)
        if it % check == 0 or it == max_iter:
            e = ((torch.logsumexp(K + (g + lb)[:, None, :], dim=2) + f + la).exp() - a).abs().sum(1)
            err = e.cpu().numpy()
            it_done[(active & (e < tol)).cpu().numpy()] = it
            active &= ~(e < tol)
            if not bool(active.any()):
                break
    val = ((K + (f + la)[:, :, None] + (g + lb)[:, None, :]).exp().double() * C.double()).sum((1, 2))
    return val, it_done, err


def sinkhorn_batch(C: torch.Tensor, a: torch.Tensor, b: torch.Tensor, eps: float, tol: float = lc.SK_TOL,
                   max_iter: int = lc.SK_MAX_ITER, check: int = lc.SK_CHECK) -> tuple[torch.Tensor, np.ndarray, np.ndarray]:
    """OT entropico di un blocco di coppie: (<P, C> (B,), iterazioni (B,), errore L1 finale sulle righe (B,)).
    Sinkhorn-Knopp sul nucleo K = exp(-C / eps) (``ot.sinkhorn``, metodo ``sinkhorn``): u = a / K v, v = b / K^T u,
    P = diag(u) K diag(v); le coppie convergenti si fermano alla loro iterazione (aggiornamenti congelati). Coppie con
    valori non finiti (underflow del nucleo): ripiego ``sinkhorn_log_batch``, segnato con iterazioni negative."""
    K = torch.exp(C / (-eps))
    u, v = torch.ones_like(a), torch.ones_like(b)
    it_done, err = np.full(len(C), max_iter), np.full(len(C), np.nan)
    active = torch.ones(len(C), dtype=torch.bool, device=C.device)
    with tf32(False):
        for it in range(1, max_iter + 1):
            u = torch.where(active[:, None], a / torch.bmm(K, v[:, :, None])[:, :, 0], u)
            v = torch.where(active[:, None], b / torch.bmm(u[:, None, :], K)[:, 0, :], v)
            if it % check == 0 or it == max_iter:
                e = (u * torch.bmm(K, v[:, :, None])[:, :, 0] - a).abs().sum(1)
                e = torch.where(torch.isfinite(e), e, torch.zeros_like(e))   # i non finiti si riprendono dopo
                err = e.cpu().numpy()
                it_done[(active & (e < tol)).cpu().numpy()] = it
                active &= ~(e < tol)
                if not bool(active.any()):
                    break
    val = ((u[:, :, None] * K * v[:, None, :]).double() * C.double()).sum((1, 2))
    bad = ~torch.isfinite(val) | ~torch.isfinite(u).all(1) | ~torch.isfinite(v).all(1)
    if bool(bad.any()):
        k = torch.nonzero(bad)[:, 0]
        v2, it2, e2 = sinkhorn_log_batch(C[k], a[k], b[k], eps, tol, max_iter, check)
        val[k] = v2
        kk = k.cpu().numpy()
        it_done[kk], err[kk] = -it2, e2
    return val, it_done, err


def sub_pairs(S: torch.Tensor, m: torch.Tensor, i: np.ndarray, j: np.ndarray, eps: float, block: int = 128,
              variants=("chamfer_sub", "sinkhorn"), log=None) -> dict:
    """Varianti sul sottocampione per le coppie (i, j): token ``S`` (M, n, d), masse ``m`` (M, n). Uscite per coppia:
    ``chamfer_sub``, ``sinkhorn``, ``sk_iter``, ``sk_err``; secondi totali per variante (``seconds``)."""
    out = {v: np.full(len(i), np.nan) for v in variants}
    out.update(sk_iter=np.zeros(len(i), np.int64), sk_err=np.full(len(i), np.nan))
    sec = {v: 0.0 for v in variants}
    t0 = time.time()
    for s in range(0, len(i), block):
        ii = torch.as_tensor(i[s:s + block], device=S.device)
        jj = torch.as_tensor(j[s:s + block], device=S.device)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        C = cost_batch(S[ii], S[jj])
        torch.cuda.synchronize()
        tc = time.perf_counter() - t1
        if "chamfer_sub" in variants:
            t1 = time.perf_counter()
            out["chamfer_sub"][s:s + block] = chamfer_batch(C, m[ii], m[jj]).cpu().numpy()
            torch.cuda.synchronize()
            sec["chamfer_sub"] += tc + time.perf_counter() - t1
        if "sinkhorn" in variants:
            t1 = time.perf_counter()
            v, it, e = sinkhorn_batch(C, m[ii], m[jj], eps)
            out["sinkhorn"][s:s + block], out["sk_iter"][s:s + block], out["sk_err"][s:s + block] = v.cpu().numpy(), it, e
            torch.cuda.synchronize()
            sec["sinkhorn"] += tc + time.perf_counter() - t1
        del C
        if log and (s // block) % 100 == 0:
            log(f"sottocampione: {s + len(ii)}/{len(i)} coppie, {time.time() - t0:.0f}s")
    out["seconds"] = sec
    return out


# ------------------------------------------------------------------------------------------ controlli

def chamfer_f64(X: np.ndarray, a: np.ndarray, Y: np.ndarray, b: np.ndarray) -> float:
    """Chamfer esatto in float64 su CPU (a blocchi), per il controllo."""
    ra, cb = np.empty(len(X)), np.full(len(Y), np.inf)
    for r in range(0, len(X), 2048):
        D = ((X[r:r + 2048, None, :] - Y[None, :, :]) ** 2).sum(-1) if len(Y) <= 4096 else \
            np.maximum((X[r:r + 2048] ** 2).sum(1)[:, None] + (Y ** 2).sum(1)[None] - 2 * X[r:r + 2048] @ Y.T, 0)
        ra[r:r + 2048] = D.min(1)
        cb = np.minimum(cb, D.min(0))
    return float(0.5 * ((a * np.sqrt(ra)).sum() + (b * np.sqrt(cb)).sum()))


def check_pot(C: np.ndarray, a: np.ndarray, b: np.ndarray, eps: float) -> float:
    """<P, C> di ``ot.sinkhorn`` (``sinkhorn_log``, float64, CPU) con arresto molto stretto: il riferimento POT."""
    import ot
    P = ot.sinkhorn(a, b, C, eps, method="sinkhorn_log", numItermax=20000, stopThr=1e-9)
    return float((P * C).sum())
