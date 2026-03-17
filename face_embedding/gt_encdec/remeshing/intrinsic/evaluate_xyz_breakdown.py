#!/usr/bin/env python3
"""
Light entrypoint for post-hoc XYZ robustness breakdown.

The implementation now lives under `intrinsic/robustness/`, while this script
keeps compatibility re-exports for existing local tooling.
"""

try:
    from robustness.data_utils import (
        EvalSampleRecord,
        MeshTopologySignature,
        build_eval_plan,
        build_sample_eval_records,
        maybe_cache_chamfer_vertices_on_device,
        preload_eval_samples,
        rebuild_subject_split,
        summarize_sample_records,
    )
    from robustness.eval_utils import (
        ALLOWED_AGGREGATION_LEVELS,
        ALLOWED_METRICS,
        ALLOWED_PAIR_MODES,
        PairEvalContext,
        aggregate_pair_observations,
        build_pair_eval_context,
        build_perturbation_descriptor,
        build_perturbation_reference,
        build_sigma_grid,
        compute_pairwise_chamfer_values,
        evaluate_at_sigma_cached,
        evaluate_robustness_grid_tqdm,
        finite_nanmean,
        first_sigma_below,
        parse_ratio_thresholds,
        summarize_eval_plan,
        summarize_pack,
        symmetric_chamfer_same_shape_batch,
        symmetric_chamfer_two_sets,
        threshold_label,
        write_grid_csv,
        write_summary_md,
    )
    from robustness.noise import PerturbationParams, apply_xyz_perturbation, parse_noise_modes
    from robustness.posthoc_runner import (
        ScenarioSpec,
        infer_run_dir,
        load_checkpoint_bundle,
        load_json,
        main,
        merge_run_args,
        parse_args,
        resolve_checkpoint_path,
        resolve_runtime_args,
        run_posthoc_evaluation,
    )
except ImportError:  # pragma: no cover - package import fallback
    from .robustness.data_utils import (
        EvalSampleRecord,
        MeshTopologySignature,
        build_eval_plan,
        build_sample_eval_records,
        maybe_cache_chamfer_vertices_on_device,
        preload_eval_samples,
        rebuild_subject_split,
        summarize_sample_records,
    )
    from .robustness.eval_utils import (
        ALLOWED_AGGREGATION_LEVELS,
        ALLOWED_METRICS,
        ALLOWED_PAIR_MODES,
        PairEvalContext,
        aggregate_pair_observations,
        build_pair_eval_context,
        build_perturbation_descriptor,
        build_perturbation_reference,
        build_sigma_grid,
        compute_pairwise_chamfer_values,
        evaluate_at_sigma_cached,
        evaluate_robustness_grid_tqdm,
        finite_nanmean,
        first_sigma_below,
        parse_ratio_thresholds,
        summarize_eval_plan,
        summarize_pack,
        symmetric_chamfer_same_shape_batch,
        symmetric_chamfer_two_sets,
        threshold_label,
        write_grid_csv,
        write_summary_md,
    )
    from .robustness.noise import PerturbationParams, apply_xyz_perturbation, parse_noise_modes
    from .robustness.posthoc_runner import (
        ScenarioSpec,
        infer_run_dir,
        load_checkpoint_bundle,
        load_json,
        main,
        merge_run_args,
        parse_args,
        resolve_checkpoint_path,
        resolve_runtime_args,
        run_posthoc_evaluation,
    )


__all__ = [
    "ALLOWED_AGGREGATION_LEVELS",
    "ALLOWED_METRICS",
    "ALLOWED_PAIR_MODES",
    "EvalSampleRecord",
    "MeshTopologySignature",
    "PairEvalContext",
    "PerturbationParams",
    "ScenarioSpec",
    "aggregate_pair_observations",
    "apply_xyz_perturbation",
    "build_eval_plan",
    "build_pair_eval_context",
    "build_perturbation_descriptor",
    "build_perturbation_reference",
    "build_sample_eval_records",
    "build_sigma_grid",
    "compute_pairwise_chamfer_values",
    "evaluate_at_sigma_cached",
    "evaluate_robustness_grid_tqdm",
    "finite_nanmean",
    "first_sigma_below",
    "infer_run_dir",
    "load_checkpoint_bundle",
    "load_json",
    "main",
    "maybe_cache_chamfer_vertices_on_device",
    "merge_run_args",
    "parse_args",
    "parse_noise_modes",
    "parse_ratio_thresholds",
    "preload_eval_samples",
    "rebuild_subject_split",
    "resolve_checkpoint_path",
    "resolve_runtime_args",
    "run_posthoc_evaluation",
    "summarize_eval_plan",
    "summarize_pack",
    "summarize_sample_records",
    "symmetric_chamfer_same_shape_batch",
    "symmetric_chamfer_two_sets",
    "threshold_label",
    "write_grid_csv",
    "write_summary_md",
]


if __name__ == "__main__":
    main()
