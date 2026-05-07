import re
import os
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from typing import Optional, List, Union, Tuple, Any
from ..config import PipelineConfig
from facebench.utils.pipeline import run_pipeline, parse_pipeline_config
from .warnings import warn_if_scale_mismatch, warn_if_shape_mismatch, warn_if_landmarks_invalid

def get_subject_ids(gmesh_dir: str) -> List[int]:
    """
    Scans a directory and returns a sorted list of available subject IDs based on file names like idXXXX.txt

    Parameters
    ----------
    gmesh_dir : str
        Path to the directory containing mesh files (e.g., Gmeshes or Rmeshes)

    Returns
    -------
    List[int]
        List of subject IDs (integers)
    """
    pattern = re.compile(r"id(\d{4})\.txt$")
    subj_ids = []

    for filename in os.listdir(gmesh_dir):
        match = pattern.match(filename)
        if match:
            subj_ids.append(int(match.group(1)))

    return sorted(subj_ids)


def process_subject_from_id(
        subj_id: int,
        r_path: str,
        g_path: str,
        g_lmks_path: str,
        config: PipelineConfig,
        mm_data: dict
) -> Tuple[np.ndarray, np.ndarray]:
    r_file = os.path.join(r_path, f"id{subj_id:04d}.txt")
    g_file = os.path.join(g_path, f"id{subj_id:04d}.txt")
    g_lmk_file = os.path.join(g_lmks_path, f"id{subj_id:04d}.lmks")
    return process_subject(r_file, g_file, g_lmk_file, config=config, mm_data=mm_data)


def process_subject(
        r_path: str,
        g_path: str,
        g_lmks_path: str,
        config: PipelineConfig,
        mm_data: dict
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run the evaluation pipeline for a single subject.

    Parameters
    ----------
    r_path : str
        Path to reconstructed mesh file.
    g_path : str
        Path to ground truth mesh file.
    g_lmks_path : str
        Path to ground truth landmarks file.
    config : PipelineConfig
        Pipeline configuration.
    mm_data : dict
        Morphable model info dictionary with lmk_indices.

    Returns
    -------
    np.ndarray
    Array degli errori per ogni vertice (in metri).
    """
    R = np.loadtxt(r_path)
    Rlmks = R[mm_data["lmk_indices"]]
    G = np.loadtxt(g_path)
    Glmks = np.loadtxt(g_lmks_path)

    warn_if_landmarks_invalid(Rlmks, Glmks)
    warn_if_shape_mismatch(R, G)
    warn_if_scale_mismatch(R, G)

    error, R_aligned = run_pipeline(R, G, Rlmks, Glmks, mm=mm_data, config=config)
    return np.atleast_1d(error), R_aligned


def run_pipeline_batch(
        rec_methods: Union[str, List[str]],
        g_path: str,
        g_lmks_path: str,
        config: Union[PipelineConfig, dict],
        mm_data: dict,
        base_r_path: str,
        max_cores: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, List[int], List[str]]:
    """
    Evaluates the full pipeline across all specified subjects and reconstruction methods.

    For each subject-method pair, it returns both:
    - Per-vertex errors (after alignment and correction)
    - Aligned reconstructed meshes (to enable visualization)

    Parameters
    ----------
    rec_methods : str or List[str]
        Name or list of reconstruction methods to evaluate.
    g_path : str
        Path to the folder containing ground-truth meshes.
    g_lmks_path : str
        Path to the folder containing ground-truth landmarks.
    config : PipelineConfig
        Full pipeline configuration (aligners, correspondences, metrics, etc.).
    mm_data : dict
        Dictionary with morphable model metadata (e.g., landmark indices).
    base_r_path : str
        Base path containing Rmeshes/<method> folders.
    max_cores : int, optional
        Number of parallel processes to use. Defaults to all available cores.

    Returns
    -------
    errors_stacked : np.ndarray
        Array of shape (num_subjects, num_methods, num_vertices) with per-vertex errors (in meters).
    vertices_stacked : np.ndarray
        Array of shape (num_subjects, num_methods, num_vertices, 3) with aligned reconstructed meshes.
    subject_ids : List[int]
        List of evaluated subject IDs.
    method_names : List[str]
        List of reconstruction method names (in order).
    """
    config = parse_pipeline_config(config)

    if isinstance(rec_methods, str):
        rec_methods = [rec_methods]

    subj_ids = get_subject_ids(os.path.join(base_r_path, rec_methods[0]))
    num_subj = len(subj_ids)
    errors_list = []
    vertices_list = []

    for method in rec_methods:
        subject_dir = os.path.join(base_r_path, method)
        r_path = subject_dir

        args = [(subj_id, r_path, g_path, g_lmks_path, config, mm_data) for subj_id in subj_ids]

        num_cores = max_cores or cpu_count()
        print(f"\n🚀 Evaluating {num_subj} subjects using {num_cores} cores for method: {method}")
        with Pool(processes=num_cores) as pool:
            results = list(tqdm(pool.starmap(process_subject_from_id, args), total=num_subj))

        # Separate errors and aligned vertices
        errors, aligned = zip(*results)
        errors_list.append(np.stack(errors))  # (num_subj, num_vertices)
        vertices_list.append(np.stack(aligned))  # (num_subj, num_vertices, 3)

    errors_stacked = np.stack(errors_list, axis=1)  # (num_subj, num_methods, num_vertices)
    vertices_stacked = np.stack(vertices_list, axis=1)  # (num_subj, num_methods, num_vertices, 3)

    return errors_stacked, vertices_stacked, subj_ids, rec_methods

    # === Output structure explanation ===
    # The function returns two 3D arrays:
    #
    #   - errors_stacked:   shape (num_subjects, num_methods, num_vertices)
    #   - vertices_stacked: shape (num_subjects, num_methods, num_vertices, 3)
    #
    # Each cell errors_stacked[s, m, v] contains the reconstruction error (in meters)
    # for vertex v of subject s and method m.
    #
    # The corresponding vertex 3D coordinate is found at:
    #   vertices_stacked[s, m, v, :]  ←→  error at errors_stacked[s, m, v]
    #
    # This design uses *parallel arrays* instead of a fused (error, vertex) structure
    # to enable efficient slicing, aggregation, and compatibility with NumPy/Plotly/Vedo.
    #
    # Example usage:
    #   errors = errors_stacked[0, 1]        # all vertex errors for subject 0, method 1
    #   coords = vertices_stacked[0, 1]      # corresponding vertex coordinates
    #
    #   for err, pt in zip(errors, coords):
    #       print(f"Vertex {pt} → error: {err * 1000:.2f} mm")
