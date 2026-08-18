"""Sanity checks for v2_work/phase0/measure_distances.py. Run: .conda_env/bin/python this_file."""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from measure_distances import (DEFAULT_SIGMAS, currents_distance, mesh_measure,  # noqa: E402
                               pairwise_distances, varifold_distance)

DATA = Path(__file__).resolve().parents[2] / "datasets/REMESH/npz_data_topo_500"
A_ORIG = DATA / "id0000_GTready_original.npz"
A_DOWN = DATA / "id0000_GTready_down8k.npz"
B_ORIG = DATA / "id0001_GTready_original.npz"


def main():
    m_a = mesh_measure(A_ORIG)
    m_a2 = mesh_measure(A_ORIG)          # same mesh, independent object
    m_a_down = mesh_measure(A_DOWN)
    m_b = mesh_measure(B_ORIG)

    d_same_subject = varifold_distance(m_a, m_a_down)
    d_diff_subject = varifold_distance(m_a, m_b)
    d_identical = varifold_distance(m_a, m_a2)
    print(f"varifold identical mesh        : {d_identical:.6e}")
    print(f"varifold id0000 orig vs down8k : {d_same_subject:.6e}")
    print(f"varifold id0000 vs id0001      : {d_diff_subject:.6e}")
    print(f"gap ratio (diff/same)          : {d_diff_subject / d_same_subject:.3f}")

    assert d_identical < 1e-4 * d_diff_subject, f"identical mesh not ~0: {d_identical}"
    assert d_same_subject < d_diff_subject, "cross-topology >= cross-subject (metric is useless)"

    # Currents on the same three. NOT asserted: over 8 subjects the currents
    # cross-subject/cross-topology ratio is 0.97 -- retopology noise matches the identity
    # signal, so the ordering is a coin flip. Use varifold for identity work.
    c_same = currents_distance(m_a, m_a_down)
    c_diff = currents_distance(m_a, m_b)
    print(f"currents same/diff subject     : {c_same:.6e} / {c_diff:.6e} (not asserted)")

    # Orientation: flipped winding -> normals negate. Currents blows up, varifold ignores it.
    V, F = np.load(A_ORIG)["V"], np.load(A_ORIG)["F"]
    m_flip = mesh_measure((V, F[:, ::-1].copy()))
    c_flip = currents_distance(m_a, m_flip)
    v_flip = varifold_distance(m_a, m_flip)
    print(f"flipped orientation cur / var  : {c_flip:.6e} / {v_flip:.6e}")
    assert c_flip > d_diff_subject, "currents did not react to flipped orientation"
    assert v_flip < 1e-4 * d_diff_subject, f"varifold not orientation-invariant: {v_flip}"

    # pairwise_distances plumbing.
    out = pairwise_distances([m_a, m_a_down, m_b], [(0, 1), (0, 2)], kind="varifold")
    assert np.allclose(out, [d_same_subject, d_diff_subject], rtol=1e-5), out

    # Finding, not an assertion: the repo's max-abs normalization is topology-dependent
    # and destroys the ordering above. Kept visible so the number is not forgotten.
    q_a, q_ad, q_b = (mesh_measure(p, normalize="maxabs") for p in (A_ORIG, A_DOWN, B_ORIG))
    q_same, q_diff = varifold_distance(q_a, q_ad), varifold_distance(q_a, q_b)
    print(f"[maxabs norm] same={q_same:.6e} diff={q_diff:.6e} gap={q_diff / q_same:.3f}")

    # Timing (cold measures each time so the self-inner cache does not hide the cost).
    for max_tris in (2000, 4000):
        pa = mesh_measure(A_ORIG, max_tris=max_tris)
        pb = mesh_measure(B_ORIG, max_tris=max_tris)
        varifold_distance(pa, pb)                     # warm-up + fill self-inner cache
        t0 = time.perf_counter()
        for _ in range(3):
            varifold_distance(pa, pb)
        ms = (time.perf_counter() - t0) / 3 * 1e3
        print(f"max_tris={max_tris:5d}: {ms:7.1f} ms/pair (cross term only, self-terms cached)")

    print(f"sigmas = {DEFAULT_SIGMAS}")
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
