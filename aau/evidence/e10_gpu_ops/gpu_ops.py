#!/usr/bin/env python3
"""E10: operatori DiffusionNet su GPU, stessa convenzione di ``prepass_ops.ops_areanorm``.

Riproduce ``diffusion_net.geometry.compute_operators`` passo per passo, con gli stessi dtype:
  - geometria: mesh centrata ad area 1 in fp64 (numpy, le stesse righe di ops_areanorm), poi verts fp32;
  - Laplaciano cotangente e massa = pp3d.cotan_laplacian(denom_eps=1e-10) e pp3d.vertex_areas
    (+1e-8 * media), in fp64 sui verts fp32, con scatter-add;
  - normali: facce fp32 normalizzate con eps 1e-6 (geometry.normalize), somma per vertice fp64,
    cast fp32; frame tangenti e vettori tangenti degli archi in fp32;
  - gradienti in fp64 con la formula chiusa di e9_bench/grad_vec.py (struttura: L piu' la diagonale);
  - autovettori di L phi = lambda M phi in forma standard A = M^-1/2 (L + 1e-8 I) M^-1/2,
    phi = M^-1/2 psi (M-ortonormali come quelli di eigsh), autovalori < 0 tagliati a 0.

Batch: B mesh concatenate (indici globali) per tutta la geometria; per gli autovettori A e' diagonale
a blocchi su un layout con padding (B, Vmax): le righe di padding sono vuote e i blocchi densi vi
restano a zero, quindi non entrano mai nel sottospazio.

Risolutori (``EIG``; lo spettro di A arriva a lambda_max ~ 1e8 contro lambda_128 ~ 1.4e3, vedi E10.md):
  dense   eigh denso (torch.linalg.eigh = cusolver syevd), una mesh alla volta; fp32 sbaglia;
  lobpcg  torch.lobpcg su A con precondizionatore di Jacobi, una mesh alla volta;
  clobpcg cupyx.scipy.sparse.linalg.lobpcg, precondizionatore di Jacobi;
  chfsi   subspace iteration con filtro di Chebyshev (Zhou-Saad 2007) su A: non converge oltre ~3k vertici;
  si      shift-invert (A + tau I)^-1 fattorizzata con cuDSS + filtro di Chebyshev, batch di mesh;
  bk      shift-invert con cuDSS + Krylov a blocchi, batch di mesh: la via scelta (equivalente alla CPU
          sugli embedding e la piu' veloce).
"""
from __future__ import annotations

import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[2]
sys.path.insert(0, str(REPO_ROOT / "v2_work" / "potential"))
sys.path.insert(0, str(REPO_ROOT / "diffusion-net" / "src"))

EPS_MASS = 1e-8      # compute_operators: massvec += eps * mean, e L + eps I in eigsh
DENOM_EPS = 1e-10    # pp3d.cotan_laplacian(denom_eps=1e-10)
EPS_REG = 1e-5       # geometry.build_grad


# --- ingresso / uscita -------------------------------------------------------------------------

def prepare(src: Path) -> dict:
    """== prepass_ops.ops_areanorm fino a ``Vt`` (senza transform: il run e108 e' nocanon)."""
    from areanorm_operators import total_area
    from potential_operators import load_mesh

    V, F = load_mesh(src)
    A = total_area(V, F)
    if not np.isfinite(A) or A <= 0:
        raise ValueError(f"area non valida: {A}")
    V = (V - V.mean(0)) / np.sqrt(A)
    return {"name": Path(src).name, "verts": V.astype(np.float32), "faces": F.astype(np.int32)}


def save_npz(data: dict, out: Path) -> None:
    """Stesse chiavi e dtype di ops_areanorm (npz non compresso, indici int32)."""
    tmp = out.with_name(f".{out.name}.{os.getpid()}.tmp.npz")
    np.savez(tmp, **data)
    os.replace(tmp, out)


# --- geometria ---------------------------------------------------------------------------------

def _sync() -> None:
    """Attesa sullo stream corrente soltanto: piu' lavoratori su stream diversi non si serializzano."""
    torch.cuda.current_stream().synchronize()


def _cross(a, b):
    return torch.cross(a, b, dim=-1)


def geometry_batch(meshes: list[dict], dev: torch.device) -> dict:
    """Laplaciano, massa, frame e gradienti di B mesh concatenate.

    Restituisce i tensori globali e ``off`` (offset dei vertici); le voci sparse sono coalesced
    (ordine riga-colonna), quindi quelle di ogni mesh sono contigue.
    """
    nv = [int(m["verts"].shape[0]) for m in meshes]
    nf = [int(m["faces"].shape[0]) for m in meshes]
    off = np.concatenate([[0], np.cumsum(nv)])
    N = int(off[-1])
    v32 = torch.from_numpy(np.concatenate([m["verts"] for m in meshes])).to(dev, non_blocking=True)
    F = torch.from_numpy(np.concatenate([m["faces"].astype(np.int64) + off[i]
                                         for i, m in enumerate(meshes)])).to(dev, non_blocking=True)
    mesh_of_v = torch.repeat_interleave(torch.arange(len(meshes), device=dev),
                                        torch.tensor(nv, device=dev))
    vd = v32.double()

    # Laplaciano cotangente == pp3d.cotan_laplacian (stesse 4 voci per angolo), fp64
    ii, jj, vals = [], [], []
    for c in range(3):
        fi, fj, fk = F[:, c], F[:, (c + 1) % 3], F[:, (c + 2) % 3]
        eki, ekj = vd[fi] - vd[fk], vd[fj] - vd[fk]
        cot = 0.5 * (eki * ekj).sum(1) / (torch.linalg.norm(_cross(eki, ekj), dim=1) + DENOM_EPS)
        ii += [fi, fj, fi, fj]
        jj += [fi, fj, fj, fi]
        vals += [cot, cot, -cot, -cot]
    L = torch.sparse_coo_tensor(torch.stack([torch.cat(ii), torch.cat(jj)]), torch.cat(vals),
                                (N, N)).coalesce()
    if torch.isnan(L.values()).any():
        raise RuntimeError("NaN Laplace matrix")

    # massa == pp3d.vertex_areas + eps * media (per mesh)
    area = 0.5 * torch.linalg.norm(_cross(vd[F[:, 1]] - vd[F[:, 0]], vd[F[:, 2]] - vd[F[:, 0]]), dim=1)
    mass = torch.zeros(N, dtype=torch.float64, device=dev)
    for c in range(3):
        mass.index_add_(0, F[:, c], area)
    mass /= 3.0
    msum = torch.zeros(len(meshes), dtype=torch.float64, device=dev).index_add_(0, mesh_of_v, mass)
    mass += EPS_MASS * (msum / torch.tensor(nv, device=dev, dtype=torch.float64))[mesh_of_v]

    # normali == geometry.mesh_vertex_normals (facce fp32 con normalize eps 1e-6, somma fp64)
    fn = _cross(v32[F[:, 1]] - v32[F[:, 0]], v32[F[:, 2]] - v32[F[:, 0]])
    fn = (fn / (torch.linalg.norm(fn, dim=-1) + 1e-6).unsqueeze(-1)).double()
    vn = torch.zeros(N, 3, dtype=torch.float64, device=dev)
    for c in range(3):
        vn.index_add_(0, F[:, c], fn)
    vn = (vn / torch.linalg.norm(vn, dim=-1, keepdim=True)).float()
    if torch.isnan(vn).any():
        # compute_operators qui fa il "wiggle"; non replicato: lo segnala il chiamante
        raise ValueError("normali NaN (vertici non referenziati o facce degeneri)")

    # frame == geometry.build_tangent_frames, fp32
    c1 = torch.tensor([1.0, 0.0, 0.0], device=dev).expand(N, -1)
    c2 = torch.tensor([0.0, 1.0, 0.0], device=dev).expand(N, -1)
    bx = torch.where((torch.abs((vn * c1).sum(-1)) < 0.9).unsqueeze(-1), c1, c2)
    bx = bx - vn * (bx * vn).sum(-1, keepdim=True)
    bx = bx / (torch.linalg.norm(bx, dim=-1) + 1e-6).unsqueeze(-1)
    by = _cross(vn, bx)
    frames = torch.stack((bx, by, vn), dim=-2)

    # vettori tangenti degli archi (fp32) sugli archi di L, poi gradiente fp64 == grad_vec
    li = L.indices()
    keep = li[0] != li[1]
    tail, tip = li[0, keep], li[1, keep]
    ev = v32[tip] - v32[tail]
    tx = (ev * bx[tail]).sum(-1).double()
    ty = (ev * by[tail]).sum(-1).double()
    z = lambda: torch.zeros(N, dtype=torch.float64, device=dev)  # noqa: E731
    a = z().index_add_(0, tail, tx * tx) + EPS_REG
    b = z().index_add_(0, tail, tx * ty)
    d = z().index_add_(0, tail, ty * ty) + EPS_REG
    det = a * d - b * b
    cx = (d[tail] * tx - b[tail] * ty) / det[tail]
    cy = (a[tail] * ty - b[tail] * tx) / det[tail]
    ar = torch.arange(N, device=dev)
    gidx = torch.stack([torch.cat([ar, tail]), torch.cat([ar, tip])])
    G = torch.sparse_coo_tensor(gidx, torch.stack([torch.cat([-z().index_add_(0, tail, cx), cx]),
                                                   torch.cat([-z().index_add_(0, tail, cy), cy])], 1),
                                (N, N, 2)).coalesce()
    return {"N": N, "off": off, "nv": nv, "nf": nf, "v32": v32, "F": F, "L": L, "mass": mass,
            "frames": frames, "G": G}


def _seg(idx: torch.Tensor, off: np.ndarray) -> np.ndarray:
    """Confini delle voci (coalesced, righe ordinate) di ogni mesh."""
    return torch.searchsorted(idx[0].contiguous(), torch.tensor(off, device=idx.device)).cpu().numpy()


# --- autovettori -------------------------------------------------------------------------------

def standard_form(geo: dict, vmax: int | None = None, dtype=torch.float64):
    """A = M^-1/2 (L + eps I) M^-1/2 in layout con padding (B * vmax), CSR; dinv = M^-1/2 (B, vmax)."""
    L, mass, off, nv = geo["L"], geo["mass"], geo["off"], geo["nv"]
    B = len(nv)
    vmax = vmax or max(nv)
    dev = mass.device
    idx = L.indices()
    val = L.values().clone()
    val[idx[0] == idx[1]] += EPS_MASS
    mesh_of_v = torch.repeat_interleave(torch.arange(B, device=dev), torch.tensor(nv, device=dev))
    pad_of_v = mesh_of_v * vmax + (torch.arange(geo["N"], device=dev)
                                   - torch.tensor(off[:-1], device=dev)[mesh_of_v])
    dinv = mass.rsqrt()
    val = val * dinv[idx[0]] * dinv[idx[1]]
    A = torch.sparse_coo_tensor(pad_of_v[idx], val.to(dtype), (B * vmax, B * vmax)).coalesce()
    dpad = torch.zeros(B * vmax, dtype=torch.float64, device=dev)
    dpad[pad_of_v] = dinv
    return A.to_sparse_csr(), dpad.view(B, vmax), pad_of_v


def gershgorin_upper(A_csr: torch.Tensor, B: int, vmax: int) -> torch.Tensor:
    """Limite superiore dello spettro per mesh: max_i sum_j |A_ij| (A simmetrica)."""
    crow = A_csr.crow_indices()
    rows = torch.repeat_interleave(torch.arange(B * vmax, device=crow.device), crow[1:] - crow[:-1])
    s = torch.zeros(B * vmax, dtype=torch.float64, device=crow.device)
    s.index_add_(0, rows, A_csr.values().abs().double())
    return s.view(B, vmax).amax(1)


def eig_dense(geo: dict, k: int, dtype=torch.float64) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """(b) eigh denso, una mesh alla volta (A densa V x V in ``dtype``)."""
    out = []
    A, dinv, _ = standard_form(geo, dtype=dtype)
    vmax = dinv.shape[1]
    for b, n in enumerate(geo["nv"]):
        crow = A.crow_indices()[b * vmax: b * vmax + n + 1]
        lo, hi = int(crow[0]), int(crow[-1])
        sub = torch.sparse_csr_tensor(crow - lo, A.col_indices()[lo:hi] - b * vmax, A.values()[lo:hi],
                                      (n, n)).to_dense()
        w, U = torch.linalg.eigh(sub)
        out.append((w[:k].double(), U[:, :k].double() * dinv[b, :n, None]))
        del sub, U
    return out


def eig_lobpcg(geo: dict, k: int, tol: float = 1e-6, niter: int = 2000, nb: int | None = None,
               dtype=torch.float64, seed: int = 0) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """(a) torch.lobpcg sulla forma standard, precondizionatore di Jacobi (iK = diag(1/A_ii))."""
    out = []
    A, dinv, _ = standard_form(geo, dtype=dtype)
    vmax = dinv.shape[1]
    g = torch.Generator(device=A.device).manual_seed(seed)
    for b, n in enumerate(geo["nv"]):
        crow = A.crow_indices()[b * vmax: b * vmax + n + 1]
        lo, hi = int(crow[0]), int(crow[-1])
        sub = torch.sparse_csr_tensor(crow - lo, A.col_indices()[lo:hi] - b * vmax, A.values()[lo:hi],
                                      (n, n)).to_sparse_coo().coalesce()
        dg = torch.zeros(n, dtype=dtype, device=A.device)
        si = sub.indices()
        dm = si[0] == si[1]
        dg[si[0, dm]] = sub.values()[dm]
        iK = torch.sparse_coo_tensor(torch.arange(n, device=A.device).expand(2, -1), 1.0 / dg, (n, n))
        X0 = torch.randn(n, nb or k, dtype=dtype, device=A.device, generator=g)
        w, U = torch.lobpcg(sub, k=k, X=X0, iK=iK, niter=niter, tol=tol, largest=False)
        o = torch.argsort(w)
        out.append((w[o].double(), U[:, o].double() * dinv[b, :n, None]))
    return out


def eig_clobpcg(geo: dict, k: int, tol: float = 1e-6, maxiter: int = 2000, nb: int | None = None,
                seed: int = 0) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """(a) cupyx lobpcg sulla forma standard fp64, precondizionatore di Jacobi."""
    import cupy as cp
    import cupyx.scipy.sparse as csp
    from cupyx.scipy.sparse.linalg import LinearOperator, lobpcg

    out = []
    A, dinv, _ = standard_form(geo, dtype=torch.float64)
    vmax = dinv.shape[1]
    rs = cp.random.RandomState(seed)
    for b, n in enumerate(geo["nv"]):
        crow = A.crow_indices()[b * vmax: b * vmax + n + 1]
        lo, hi = int(crow[0]), int(crow[-1])
        sub = csp.csr_matrix((cp.asarray(A.values()[lo:hi]), cp.asarray(A.col_indices()[lo:hi] - b * vmax),
                              cp.asarray(crow - lo)), shape=(n, n))
        idg = 1.0 / sub.diagonal()
        P = LinearOperator((n, n), matvec=lambda x, d=idg: d * x.ravel(),
                           matmat=lambda X, d=idg: d[:, None] * X, dtype=cp.float64)
        X0 = rs.standard_normal((n, nb or k))
        w, U = lobpcg(sub, X0, M=P, tol=tol, maxiter=maxiter, largest=False)
        w, U = torch.as_tensor(w, device=A.device), torch.as_tensor(U, device=A.device)
        o = torch.argsort(w)[:k]
        out.append((w[o], U[:, o] * dinv[b, :n, None]))
    return out


def _inner(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """X_b @ Y_b^T per b, X (B, p, n), Y (B, q, n): un mm per mesh. Con n lungo la GEMM batched di
    cuBLAS in fp64 e' ~12x piu' lenta (misurato su V100: 10.1 contro 0.85 ms a B=2, n=60k)."""
    out = torch.empty(X.shape[0], X.shape[1], Y.shape[1], dtype=X.dtype, device=X.device)
    for b in range(X.shape[0]):
        torch.mm(X[b], Y[b].T, out=out[b])
    return out


def _cholqr(X: torch.Tensor, rr_dtype) -> torch.Tensor:
    """X (B, n, m) -> base ortonormale dello stesso span: CholeskyQR2 in rr_dtype, colonne prima
    normalizzate; dove Cholesky fallisce (span quasi degenere) Householder QR."""
    Xd = X.to(rr_dtype)
    Xd = Xd / torch.linalg.norm(Xd, dim=1, keepdim=True).clamp_min(1e-300)
    for _ in range(2):
        Xt = Xd.transpose(1, 2)
        G = _inner(Xt, Xt)
        R, info = torch.linalg.cholesky_ex(G, upper=True)
        eye = torch.eye(R.shape[-1], dtype=R.dtype, device=R.device).expand_as(R)
        bad = info != 0
        Xn = Xd @ torch.linalg.solve_triangular(R, eye, upper=True)
        if bool(bad.any()):                  # Householder sul blocco PRIMA di R^-1 (R e' spazzatura)
            Xn[bad] = torch.linalg.qr(Xd[bad])[0]
        Xd = Xn
    return Xd


def _spmm(A: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
    B, n, m = X.shape
    return (A @ X.reshape(B * n, m)).view(B, n, m)


def cheb_filter(A, X, deg: int, a, ub, a0):
    """Filtro di Chebyshev scalato (Zhou-Saad 2007, alg. 4.1): smorza [a, ub], amplifica [a0, a).
    a, ub, a0: (B,) per mesh; X (B, n, m) nel dtype di A."""
    dt = X.dtype
    e = ((ub - a) / 2).to(dt)[:, None, None]
    c = ((ub + a) / 2).to(dt)[:, None, None]
    sigma = e / (a0.to(dt)[:, None, None] - c)
    tau = 2.0 / sigma
    Y = (_spmm(A, X) - c * X) * (sigma / e)
    for _ in range(2, deg + 1):
        sigma_new = 1.0 / (tau - sigma)
        Yn = _spmm(A, Y)
        Yn.sub_(c * Y).mul_(2.0 * sigma_new / e).sub_((sigma * sigma_new) * X)
        X, Y, sigma = Y, Yn, sigma_new
    return Y


def eig_chfsi(geo: dict, k: int, nb: int | None = None, deg: int = 20, tol: float = 1e-5,
              max_outer: int = 30, dtype=torch.float32, rr_dtype=torch.float64, seed: int = 0,
              vmax: int | None = None, stats: dict | None = None):
    """(c) subspace iteration con filtro di Chebyshev, B mesh insieme.

    nb colonne (default 1.5 k + 16); taglio iniziale da Weyl (area 1: lambda_n ~ 4 pi n), poi il
    massimo valore di Ritz. Filtro in ``dtype``, ortonormalizzazione e Rayleigh-Ritz in ``rr_dtype``.
    Convergenza: residuo ||A x - theta x|| <= tol * theta_nb per le prime k coppie di ogni mesh.
    """
    nb = nb or (3 * k) // 2 + 16
    A64, dinv, _ = standard_form(geo, vmax=vmax, dtype=torch.float64)
    A = A64 if dtype == torch.float64 else A64.to(dtype)
    B, n = dinv.shape
    dev = A.device
    nvt = torch.tensor(geo["nv"], device=dev)
    mask = (torch.arange(n, device=dev)[None, :] < nvt[:, None]).to(dtype)[..., None]
    ub = gershgorin_upper(A64, B, n) * 1.01
    a = torch.full((B,), 4 * math.pi * nb, dtype=torch.float64, device=dev)
    a0 = torch.zeros(B, dtype=torch.float64, device=dev)
    g = torch.Generator(device=dev).manual_seed(seed)
    X = torch.randn(B, n, nb, dtype=dtype, device=dev, generator=g) * mask
    n_spmm, it = 0, 0
    for it in range(1, max_outer + 1):
        X = cheb_filter(A, X, deg, a, ub, a0)
        n_spmm += deg
        Q = _cholqr(X, rr_dtype) * mask.to(rr_dtype)
        AQ = _spmm(A64 if rr_dtype == torch.float64 else A, Q)
        n_spmm += 1
        H = Q.transpose(1, 2) @ AQ
        theta, S = torch.linalg.eigh((H + H.transpose(1, 2)) / 2)
        Q, AQ = Q @ S, AQ @ S
        res = torch.linalg.norm(AQ - Q * theta[:, None, :], dim=1)        # (B, nb)
        conv = (res[:, :k] <= tol * theta[:, -1:]).all(1)
        if bool(conv.all()):
            break
        a = torch.minimum(theta[:, -1], a * 4)  # theta_nb >= lambda_nb: limite inferiore della parte smorzata
        X = Q.to(dtype)
    if stats is not None:
        stats.update({"outer": it, "n_spmm": n_spmm, "nb": nb, "deg": deg,
                      "max_res_rel": float((res[:, :k] / theta[:, -1:]).max()),
                      "converged": bool(conv.all())})
    out = []
    for b, nvb in enumerate(geo["nv"]):
        out.append((theta[b, :k].double(), Q[b, :nvb, :k].double() * dinv[b, :nvb, None]))
    return out


# --- shift-invert con cuDSS ---------------------------------------------------------------------

class CuDSS:
    """Cholesky sparso su GPU (cuDSS 0.8 via ctypes, wheel nvidia-cudss-cu12 in .venv_e10).

    Una fattorizzazione SPD di una matrice CSR (triangolo superiore, indici int32, valori fp64) e
    soluzioni con piu' termini noti, sullo stream corrente di torch."""

    R_64F, R_32I = 1, 10
    MTYPE_SPD, MVIEW_UPPER, BASE_ZERO, COL_MAJOR = 3, 2, 0, 0
    PHASE_ANALYSIS, PHASE_FACTORIZATION, PHASE_SOLVE = 3, 4, 0x3F0

    _lib = None

    @classmethod
    def lib(cls):
        if cls._lib is None:
            import ctypes
            import glob
            hits = [p for d in sys.path for p in glob.glob(os.path.join(d, "nvidia/cu12/lib/libcudss.so.0"))]
            if not hits:
                raise RuntimeError("libcudss.so.0 non trovata: usare VENV=.venv_e10 (vedi README)")
            cls._lib = ctypes.CDLL(hits[0])
        return cls._lib

    def _ok(self, st, what):
        if st != 0:
            raise RuntimeError(f"cuDSS {what}: stato {st}")

    CFG_REORDERING_ALG, CFG_HOST_NTHREADS = 0, 14
    REORDER = {"default": 0, "amd": 3, "nd": 4}

    def __init__(self, crow: torch.Tensor, col: torch.Tensor, val: torch.Tensor, n: int,
                 reorder: str = "default", host_threads: int = 0):
        import ctypes as C
        import glob
        self.C, L = C, self.lib()
        self.L = L
        vp, i64 = C.c_void_p, C.c_int64
        self.h, self.cfg, self.data = vp(), vp(), vp()
        self._ok(L.cudssCreate(C.byref(self.h)), "create")
        self._ok(L.cudssSetStream(self.h, vp(torch.cuda.current_stream().cuda_stream)), "stream")
        self._ok(L.cudssConfigCreate(C.byref(self.cfg)), "config")
        if reorder != "default":
            v = C.c_int(self.REORDER[reorder])
            self._ok(L.cudssConfigSet(self.cfg, C.c_int(self.CFG_REORDERING_ALG), C.byref(v), C.c_size_t(4)), "reorder")
        if host_threads:
            mt = glob.glob(os.path.join(os.path.dirname(L._name), "libcudss_mtlayer_gomp.so*"))[0]
            self._ok(L.cudssSetThreadingLayer(self.h, mt.encode()), "threading layer")
            v = C.c_int(host_threads)
            self._ok(L.cudssConfigSet(self.cfg, C.c_int(self.CFG_HOST_NTHREADS), C.byref(v), C.c_size_t(4)), "nthreads")
        self._ok(L.cudssDataCreate(self.h, C.byref(self.data)), "data")
        self.keep = (crow.int().contiguous(), col.int().contiguous(), val.double().contiguous())
        self.n = n
        self.A = vp()
        self._ok(L.cudssMatrixCreateCsr(
            C.byref(self.A), i64(n), i64(n), i64(self.keep[1].numel()), vp(self.keep[0].data_ptr()), vp(None),
            vp(self.keep[1].data_ptr()), vp(self.keep[2].data_ptr()), C.c_int(self.R_32I), C.c_int(self.R_32I),
            C.c_int(self.R_64F), C.c_int(self.MTYPE_SPD), C.c_int(self.MVIEW_UPPER), C.c_int(self.BASE_ZERO)),
            "matrix")
        self.nrhs = 0
        self.x, self.b = vp(), vp()

    def _dense(self, nrhs: int, xb: torch.Tensor, bb: torch.Tensor):
        C, L = self.C, self.L
        if self.nrhs:
            L.cudssMatrixDestroy(self.x)
            L.cudssMatrixDestroy(self.b)
        for m, t in ((self.x, xb), (self.b, bb)):
            self._ok(L.cudssMatrixCreateDn(C.byref(m), C.c_int64(self.n), C.c_int64(nrhs), C.c_int64(self.n),
                                           C.c_void_p(t.data_ptr()), C.c_int(self.R_64F), C.c_int(self.COL_MAJOR)),
                     "dense")
        self.nrhs = nrhs

    def _exec(self, phase, what):
        self._ok(self.L.cudssExecute(self.h, self.C.c_int(phase), self.cfg, self.data, self.A, self.x, self.b), what)

    def factor(self) -> None:
        dummy = torch.zeros(self.n, 1, dtype=torch.float64, device="cuda")
        self._dense(1, dummy, dummy)
        self._exec(self.PHASE_ANALYSIS, "analysis")
        self._exec(self.PHASE_FACTORIZATION, "factorization")
        self._dummy = dummy

    def solve(self, Bt: torch.Tensor) -> torch.Tensor:
        """Bt (nrhs, n) contiguo = n x nrhs column-major; restituisce K^-1 B nello stesso formato."""
        C, L = self.C, self.L
        Bt = Bt.contiguous()
        Xt = torch.empty_like(Bt)
        if self.nrhs != Bt.shape[0]:
            self._dense(Bt.shape[0], Xt, Bt)
        else:
            self._ok(L.cudssMatrixSetValues(self.x, C.c_void_p(Xt.data_ptr())), "set x")
            self._ok(L.cudssMatrixSetValues(self.b, C.c_void_p(Bt.data_ptr())), "set b")
        self._exec(self.PHASE_SOLVE, "solve")
        return Xt

    def close(self) -> None:
        L = self.L
        if self.nrhs:
            L.cudssMatrixDestroy(self.x)
            L.cudssMatrixDestroy(self.b)
        L.cudssMatrixDestroy(self.A)
        L.cudssDataDestroy(self.h, self.data)
        L.cudssConfigDestroy(self.cfg)
        L.cudssDestroy(self.h)


def shift_factor(geo: dict, tau: float, reorder: str = "default", host_threads: int = 8):
    """A (forma standard, padding) e la fattorizzazione cuDSS di K = A + tau I (triangolo superiore,
    righe di padding = identita': i blocchi vi restano a zero). Restituisce (A, dinv, solver, tempi)."""
    tm = {}
    t0 = time.perf_counter()
    A, dinv, _ = standard_form(geo, dtype=torch.float64)
    B, n = dinv.shape
    N = B * n
    dev = A.device
    nvt = torch.tensor(geo["nv"], device=dev)
    real = (torch.arange(n, device=dev)[None, :] < nvt[:, None]).reshape(-1)
    Ac = A.to_sparse_coo().coalesce()
    idx, val = Ac.indices(), Ac.values()
    up = idx[1] >= idx[0]
    idx, val = idx[:, up], val[up].clone()
    val[idx[0] == idx[1]] += tau
    pad = torch.nonzero(~real).flatten()
    K = torch.sparse_coo_tensor(torch.cat([idx, pad.expand(2, -1)], 1),
                                torch.cat([val, torch.ones(pad.numel(), dtype=val.dtype, device=dev)]),
                                (N, N)).coalesce().to_sparse_csr()
    _sync()
    tm["build_s"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    solver = CuDSS(K.crow_indices(), K.col_indices(), K.values(), N, reorder=reorder, host_threads=host_threads)
    solver.factor()
    _sync()
    tm["factor_s"] = time.perf_counter() - t0
    return A, dinv, solver, tm


def _ritz(A, Q, k: int):
    """Rayleigh-Ritz su A nel sottospazio ortonormale Q (B, n, m): le prime k coppie e i residui."""
    AQ = _spmm(A, Q)
    H = _inner(Q.transpose(1, 2), AQ.transpose(1, 2))
    theta, Sv = torch.linalg.eigh((H + H.transpose(1, 2)) / 2)
    X, AX = Q @ Sv[:, :, :k], AQ @ Sv[:, :, :k]
    res = torch.linalg.norm(AX - X * theta[:, None, :k], dim=1)
    return theta, X, res


def eig_si(geo: dict, k: int, nb: int | None = None, tau: float = 1.0, deg: int = 4, tol: float = 1e-5,
           max_outer: int = 20, seed: int = 0, reorder: str = "default", host_threads: int = 8,
           stats: dict | None = None):
    """(c1) shift-invert + Chebyshev: S = (A + tau I)^-1 fattorizzata una volta con cuDSS (B mesh
    insieme, A diagonale a blocchi); filtro di Chebyshev di grado ``deg`` su -S (smorza
    [-1/(theta_nb + tau), 0], cioe' gli autovalori oltre il taglio), ortonormalizzazione,
    Rayleigh-Ritz su A in fp64. Convergenza: ||A q - theta q|| <= tol * theta_k per le prime k coppie."""
    nb = nb or (3 * k) // 2 + 16
    A, dinv, solver, tm = shift_factor(geo, tau, reorder, host_threads)
    B, n = dinv.shape
    N = B * n
    dev = A.device
    t0 = time.perf_counter()
    nvt = torch.tensor(geo["nv"], device=dev)
    maskt = (torch.arange(n, device=dev)[None, :] < nvt[:, None]).view(1, B, n).double()  # (nb, B, n) = col-major
    g = torch.Generator(device=dev).manual_seed(seed)
    Xt = torch.randn(nb, B, n, dtype=torch.float64, device=dev, generator=g) * maskt
    lam_cut = torch.full((B,), 4 * math.pi * nb, dtype=torch.float64, device=dev)   # Weyl, area 1
    n_solve, it = 0, 0
    T = lambda Z: -solver.solve(Z.reshape(nb, N)).view(nb, B, n)  # noqa: E731
    for it in range(1, max_outer + 1):
        # Chebyshev su T = -S: smorza [-1/(lam_cut + tau), 0], a0 = -1/tau
        a = (-1.0 / (lam_cut + tau))[None, :, None]
        e = -a / 2
        c = a / 2
        sigma = e / (-1.0 / tau - c)
        tau2 = 2.0 / sigma
        Y = (T(Xt) - c * Xt) * (sigma / e)
        n_solve += 1
        for _ in range(2, deg + 1):
            s_new = 1.0 / (tau2 - sigma)
            Yn = (T(Y) - c * Y) * (2.0 * s_new / e) - (sigma * s_new) * Xt
            n_solve += 1
            Xt, Y, sigma = Y, Yn, s_new
        Q = _cholqr(Y.permute(1, 2, 0), torch.float64) * maskt.view(B, n, 1)    # (B, n, nb)
        theta, Q, res = _ritz(A, Q, nb)
        conv = (res[:, :k] <= tol * theta[:, k - 1:k]).all(1)
        if bool(conv.all()):
            break
        lam_cut = theta[:, -1]
        Xt = Q.permute(2, 0, 1).contiguous()
    solver.close()
    _sync()
    tm["iter_s"] = time.perf_counter() - t0
    if stats is not None:
        stats.update({"outer": it, "n_solve": n_solve, "rhs_cols": n_solve * nb, "nb": nb, "deg": deg,
                      "tau": tau, **tm, "max_res_rel": float((res[:, :k] / theta[:, k - 1:k]).max()),
                      "converged": bool(conv.all())})
    return [(theta[b, :k], Q[b, :nvb, :k] * dinv[b, :nvb, None]) for b, nvb in enumerate(geo["nv"])]


def _cholqr_t(Wt: torch.Tensor) -> torch.Tensor:
    """Come _cholqr ma sui blocchi trasposti Wt (B, m, n) (righe = vettori): CholeskyQR2 in fp64."""
    Wt = Wt / torch.linalg.norm(Wt, dim=2, keepdim=True).clamp_min(1e-300)
    for _ in range(2):
        G = _inner(Wt, Wt)
        R, info = torch.linalg.cholesky_ex(G)                      # G = R R^T, R triangolare inferiore
        # R^-1 esplicita (b x b) e bmm: trsm batched con n termini noti e' lentissimo su cuBLAS
        eye = torch.eye(R.shape[-1], dtype=R.dtype, device=R.device).expand_as(R)
        bad = info != 0
        Wn = torch.linalg.solve_triangular(R, eye, upper=False) @ Wt
        if bool(bad.any()):                  # Householder sul blocco PRIMA di R^-1 (R e' spazzatura)
            Wn[bad] = torch.linalg.qr(Wt[bad].transpose(1, 2))[0].transpose(1, 2)
        Wt = Wn
    return Wt


def eig_bk(geo: dict, k: int, bs: int = 16, m: int | None = None, tau: float = 1.0, tol: float = 1e-5,
           max_restart: int = 2, seed: int = 0, reorder: str = "default", host_threads: int = 8,
           stats: dict | None = None):
    """(c2) shift-invert + Krylov a blocchi: base ortonormale di K_s(S, S X0) = [S X0, ..., S^s X0]
    con blocchi di ``bs`` colonne (CGS2 contro tutta la base), poi Rayleigh-Ritz su A. ``m`` = colonne
    totali (default max(4k, k + 256): con bs 16 converge in una passata su ICT 9.4k-24k e BFM 60k, per
    k 64 e 128; a parita' di colonne i blocchi piccoli danno un polinomio di grado piu' alto, quindi
    convergono meglio). tol 1e-5: sotto ~2e-6 il residuo tocca il pavimento di fp64 con lambda_max ~1e8
    (misurato su ICT up60k: residui 1-6e-6 anche sulle prime coppie, ben separate).
    Se le prime k coppie non convergono (||A q - theta q|| <= tol * theta_k),
    riparte tenendo i primi ~k + bs vettori di Ritz e riempie il resto della base con blocchi di Krylov
    nuovi, a partire da S applicato agli ultimi bs vettori tenuti.
    La base e' memorizzata trasposta, Qt (B, m, n): ogni fetta Qt[:, :j] e' contigua per le GEMM."""
    m = m or max(4 * k, k + 256)
    m = math.ceil(m / bs) * bs
    A, dinv, solver, tm = shift_factor(geo, tau, reorder, host_threads)
    B, n = dinv.shape
    N = B * n
    dev = A.device
    t0 = time.perf_counter()
    nvt = torch.tensor(geo["nv"], device=dev)
    mask = (torch.arange(n, device=dev)[None, :] < nvt[:, None]).double()[:, None, :]   # (B, 1, n)
    g = torch.Generator(device=dev).manual_seed(seed)
    prof = {"solve_s": 0.0, "orth_s": 0.0, "rr_s": 0.0}
    profile = stats is not None and stats.get("profile")

    def tick(key, t):
        if profile:
            _sync()
            prof[key] += time.perf_counter() - t

    def S(Wt):          # (B, b, n) -> (B, b, n); cuDSS vuole (b, B*n) = (B*n) x b column-major
        t = time.perf_counter()
        out = solver.solve(Wt.transpose(0, 1).reshape(Wt.shape[1], N)).view(-1, B, n).transpose(0, 1)
        tick("solve_s", t)
        return out

    Qt = torch.zeros(B, m, n, dtype=torch.float64, device=dev)
    j0 = 0                                                  # righe gia' nella base (riavvio: Ritz)
    # primo blocco S X0, non X0: un blocco casuale porterebbe nella base le frequenze alte (fino a
    # lambda_max ~ 1e8) senza contribuire ai primi k
    Wt = S(torch.randn(B, bs, n, dtype=torch.float64, device=dev, generator=g) * mask)
    n_solve, rhs_cols, restarts = 1, bs, 0
    while True:
        j = j0
        while j < m:
            t = time.perf_counter()
            if j > 0:
                P = Qt[:, :j]
                for _ in range(2):          # CGS2 contro la base
                    Wt = Wt - _inner(Wt, P) @ P
            Wt = _cholqr_t(Wt) * mask
            Qt[:, j:j + bs] = Wt
            tick("orth_s", t)
            j += bs
            if j < m:
                Wt = S(Wt)
                n_solve += 1
                rhs_cols += bs
        t = time.perf_counter()
        theta, X, res = _ritz(A, Qt.transpose(1, 2), m)    # X (B, n, m)
        tick("rr_s", t)
        conv = (res[:, :k] <= tol * theta[:, k - 1:k]).all(1)
        if bool(conv.all()) or restarts >= max_restart:
            break
        # riavvio: i primi j0 vettori di Ritz restano, il blocco successivo e' S applicato agli ultimi bs
        restarts += 1
        j0 = min(m - bs, math.ceil((k + bs) / bs) * bs)
        Qt[:, :j0] = X[:, :, :j0].transpose(1, 2)
        Wt = S(Qt[:, j0 - bs:j0])
        n_solve += 1
        rhs_cols += bs
    solver.close()
    _sync()
    tm["iter_s"] = time.perf_counter() - t0
    if stats is not None:
        stats.update({"n_solve": n_solve, "rhs_cols": rhs_cols, "bs": bs, "m": m, "tau": tau, "restarts": restarts,
                      **tm, **(prof if profile else {}),
                      "max_res_rel": float((res[:, :k] / theta[:, k - 1:k]).max()),
                      "converged": bool(conv.all())})
    return [(theta[b, :k], X[b, :nvb, :k] * dinv[b, :nvb, None]) for b, nvb in enumerate(geo["nv"])]


EIG = {"dense": eig_dense, "lobpcg": eig_lobpcg, "clobpcg": eig_clobpcg, "chfsi": eig_chfsi, "si": eig_si,
       "bk": eig_bk}


# --- tutto insieme -----------------------------------------------------------------------------

def compute_batch(meshes: list[dict], k: int, method: str = "bk", dev: torch.device | None = None,
                  eig_opts: dict | None = None, timings: dict | None = None) -> list[dict]:
    """Operatori di B mesh preparate (``prepare``): lista di dict con le chiavi del npz di ops_areanorm."""
    dev = dev or torch.device("cuda")
    sync = _sync if dev.type == "cuda" else (lambda: None)
    t0 = time.perf_counter()
    geo = geometry_batch(meshes, dev)
    sync()
    t1 = time.perf_counter()
    eig = EIG[method](geo, k, **(eig_opts or {}))
    sync()
    t2 = time.perf_counter()
    # copia su host in un colpo per tensore (indici locali calcolati sulla GPU), poi fette numpy
    off, nv = geo["off"], geo["nv"]
    li, gi = geo["L"].indices(), geo["G"].indices()
    sl, sg = _seg(li, off), _seg(gi, off)
    offt = torch.tensor(off[:-1], device=dev)
    loc = lambda idx, seg: (idx - torch.repeat_interleave(offt, torch.tensor(np.diff(seg), device=dev))).int()  # noqa: E731
    h = {"L_idx": loc(li, sl), "G_idx": loc(gi, sg), "L_val": geo["L"].values().float(),
         "G_val": geo["G"].values().float(), "mass": geo["mass"].float(),
         "evals": torch.stack([w.clamp_min(0.0) for w, _ in eig]).float(),
         "evecs": torch.cat([U for _, U in eig]).float()}
    h = {key: v.cpu().numpy() for key, v in h.items()}
    out = []
    for b, m in enumerate(meshes):
        o, nvb = int(off[b]), nv[b]
        shape = np.array([nvb, nvb])
        Gidx = h["G_idx"][:, sg[b]:sg[b + 1]]
        out.append({"verts": m["verts"], "faces": m["faces"], "mass": h["mass"][o:o + nvb],
                    "evals": h["evals"][b], "evecs": h["evecs"][o:o + nvb],
                    "L_indices": h["L_idx"][:, sl[b]:sl[b + 1]], "L_values": h["L_val"][sl[b]:sl[b + 1]],
                    "L_shape": shape,
                    "gradX_indices": Gidx, "gradX_values": h["G_val"][sg[b]:sg[b + 1], 0], "gradX_shape": shape,
                    "gradY_indices": Gidx, "gradY_values": h["G_val"][sg[b]:sg[b + 1], 1], "gradY_shape": shape})
    if timings is not None:
        sync()
        timings.update({"geom_s": t1 - t0, "eig_s": t2 - t1, "d2h_s": time.perf_counter() - t2})
    return out
