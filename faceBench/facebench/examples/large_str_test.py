
import facebench as fb
import time
import json

if __name__ == "__main__":
    # Load Morphable Model Info
    start = time.time()
    with open("../../info/BFM-p23470.json") as f:
        mm_data = json.load(f)

    mm_data["lmk_indices"] = mm_data.get("lmk_indices", [])
    # Setup
    rec_methods = ["Deep3DFace-m", "3DDFAv2-m", "3DIv2-m", "INORig-m"]
    g_path = "../../data/BFMsynth/Gmeshes"
    g_lmks_path = g_path

    # Pipeline config
    config = {
        "rigid_aligner": "rigid",
        "rigid_prealign": "landmark",

        "nonrigid_aligner": "elastic",
        "nonrigid_ref_lmk_indices": mm_data["lmk_indices"],

        "corr_establisher": "chamfer",
        "distance_computer": "p2p",

        "corrector": "topology_consistency"
    }

    # Evaluate all reconstruction methods in one call
    errors, aligned_vertices, subject_ids, methods = fb.run_pipeline_batch(
        rec_methods=rec_methods,
        g_path=g_path,
        g_lmks_path=g_lmks_path,
        config=config,
        mm_data=mm_data,
        base_r_path="../../data/BFMsynth/Rmeshes/BFM/p23470/"
    )

    fb.plot_face_3d_error_interactive(aligned_vertices[1, 2], errors[1, 2])

    fb.generate_results_table(errors, method_names=rec_methods)
