import argparse
import fnmatch
import multiprocessing as mp
import os
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

# === PATH alla libreria DiffusionNet ===
if "/equilibrium/lpampaloni/diffusion-net/src" not in sys.path:
    sys.path.append("/equilibrium/lpampaloni/diffusion-net/src")
from diffusion_net.geometry import compute_operators


DEFAULT_INPUT_DIR = "/equilibrium/lpampaloni/WBES-FaceEmbedding/datasets/REMESH/npz_data_topo_500"
DEFAULT_OUTPUT_DIR = "/equilibrium/lpampaloni/WBES-FaceEmbedding/datasets/REMESH/npz_data_topo_500_withops"
DEFAULT_K_EIG = 128
DEFAULT_N_CORES = 4
MESH_SUFFIXES = {".ply", ".obj", ".off", ".stl"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Precompute DiffusionNet operators from .npz or triangle-mesh files."
    )
    parser.add_argument("--input-dir", default=DEFAULT_INPUT_DIR, help="Directory with input meshes or npz files.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directory where .npz with operators will be saved.")
    parser.add_argument(
        "--input-kind",
        choices=("auto", "npz", "mesh"),
        default="auto",
        help="Interpret inputs as npz, mesh, or infer from extension.",
    )
    parser.add_argument(
        "--pattern",
        default=None,
        help="Optional filename glob, e.g. '*.ply' or '001_*.ply'.",
    )
    parser.add_argument(
        "--stem-suffix",
        default=None,
        help="Keep only files whose stem ends with this suffix, e.g. '_01'.",
    )
    parser.add_argument("--k-eig", type=int, default=DEFAULT_K_EIG, help="Number of eigenpairs for DiffusionNet operators.")
    parser.add_argument("--n-cores", type=int, default=DEFAULT_N_CORES, help="Worker processes to use.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files.")
    return parser.parse_args()


def sparse_to_coo_dict(sparse_tensor, base_name):
    coalesced = sparse_tensor.coalesce()
    indices = coalesced.indices().detach().cpu().numpy()
    values = coalesced.values().detach().cpu().numpy()
    shape = coalesced.shape
    return {
        f"{base_name}_indices": indices,
        f"{base_name}_values": values,
        f"{base_name}_shape": np.array(shape, dtype=np.int64),
    }


def infer_input_kind(path):
    suffix = path.suffix.lower()
    if suffix == ".npz":
        return "npz"
    if suffix in MESH_SUFFIXES:
        return "mesh"
    raise ValueError(f"Unsupported input extension for {path.name}")


def is_candidate_file(path, input_kind, pattern=None, stem_suffix=None):
    if not path.is_file():
        return False
    if pattern and not fnmatch.fnmatch(path.name, pattern):
        return False
    if stem_suffix and not path.stem.endswith(stem_suffix):
        return False

    suffix = path.suffix.lower()
    if input_kind == "npz":
        return suffix == ".npz"
    if input_kind == "mesh":
        return suffix in MESH_SUFFIXES
    return suffix == ".npz" or suffix in MESH_SUFFIXES


def list_input_files(input_dir, input_kind, pattern=None, stem_suffix=None):
    root = Path(input_dir)
    if not root.exists():
        raise FileNotFoundError(f"Input directory does not exist: {root}")

    files = [
        path for path in sorted(root.iterdir())
        if is_candidate_file(path, input_kind=input_kind, pattern=pattern, stem_suffix=stem_suffix)
    ]
    return files


def load_geometry_from_npz(path):
    with np.load(path, allow_pickle=False) as data:
        if "V" in data and "F" in data:
            V_np = data["V"]
            F_np = data["F"]
        elif "verts" in data and "faces" in data:
            V_np = data["verts"]
            F_np = data["faces"]
        else:
            raise KeyError(f"{path.name} is missing V/F or verts/faces")
    return np.asarray(V_np), np.asarray(F_np)


def load_geometry_from_mesh(path):
    try:
        import igl
    except ImportError as exc:
        raise ImportError(
            "Mesh input requires `igl`. Run this script with an environment that has pyigl/libigl installed."
        ) from exc

    V_np, F_np = igl.read_triangle_mesh(str(path))
    if V_np.size == 0 or F_np.size == 0:
        raise ValueError(f"Empty mesh: {path.name}")
    return np.asarray(V_np), np.asarray(F_np)


def get_output_path(input_path, output_dir):
    if input_path.suffix.lower() == ".npz":
        out_name = input_path.name
    else:
        out_name = f"{input_path.stem}.npz"
    return output_dir / out_name


def effective_k_eig(requested_k, num_verts):
    if num_verts <= 2:
        return 1
    return max(1, min(int(requested_k), int(num_verts) - 2))


def process_file(task):
    input_path_str, output_dir_str, input_kind, k_eig, overwrite = task
    input_path = Path(input_path_str)
    output_dir = Path(output_dir_str)
    output_path = get_output_path(input_path, output_dir)

    if output_path.exists() and not overwrite:
        return "[skip]", input_path.name

    try:
        sample_kind = input_kind if input_kind != "auto" else infer_input_kind(input_path)
        if sample_kind == "npz":
            V_np, F_np = load_geometry_from_npz(input_path)
        else:
            V_np, F_np = load_geometry_from_mesh(input_path)

        V_np = np.asarray(V_np, dtype=np.float32)
        F_np = np.asarray(F_np, dtype=np.int64)
        V = torch.from_numpy(V_np)
        F = torch.from_numpy(F_np)

        k_eff = effective_k_eig(k_eig, V_np.shape[0])
        ops = compute_operators(V, F, k_eig=k_eff)

        new_data = {
            "verts": V_np,
            "faces": F_np,
            "mass": ops[1].cpu().numpy(),
            "evals": ops[3].cpu().numpy(),
            "evecs": ops[4].cpu().numpy(),
        }
        new_data.update(sparse_to_coo_dict(ops[2], "L"))
        new_data.update(sparse_to_coo_dict(ops[5], "gradX"))
        new_data.update(sparse_to_coo_dict(ops[6], "gradY"))

        np.savez_compressed(output_path, **new_data)
        extra = "" if k_eff == k_eig else f" (k_eig={k_eff})"
        return "[ok]", f"{input_path.name} -> {output_path.name}{extra}"

    except Exception as exc:
        return "[fail]", f"{input_path.name}: {exc}"


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = list_input_files(
        input_dir=args.input_dir,
        input_kind=args.input_kind,
        pattern=args.pattern,
        stem_suffix=args.stem_suffix,
    )
    if not files:
        raise RuntimeError(
            f"No matching input files found in {args.input_dir} "
            f"(input_kind={args.input_kind}, pattern={args.pattern}, stem_suffix={args.stem_suffix})"
        )

    n_cores = max(1, int(args.n_cores))
    print(f"Using {n_cores} core(s)")
    print(f"Input:  {args.input_dir}")
    print(f"Output: {output_dir}")
    print(f"Files:  {len(files)} | input_kind={args.input_kind} | stem_suffix={args.stem_suffix}")

    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    tasks = [
        (str(path), str(output_dir), args.input_kind, args.k_eig, args.overwrite)
        for path in files
    ]

    ok = skip = fail = 0
    failures = []

    with mp.Pool(n_cores) as pool:
        with tqdm(total=len(tasks), dynamic_ncols=True, desc="Computing operators") as pbar:
            for status, msg in pool.imap_unordered(process_file, tasks):
                if status == "[ok]":
                    ok += 1
                elif status == "[skip]":
                    skip += 1
                else:
                    fail += 1
                    failures.append(msg)
                pbar.set_postfix(ok=ok, skip=skip, fail=fail)
                pbar.update(1)

    print(f"\nDone. Saved: {ok} | Skipped: {skip} | Failed: {fail}")
    if failures:
        print("Failures:")
        for msg in failures[:20]:
            print(f"  - {msg}")


if __name__ == "__main__":
    main()
