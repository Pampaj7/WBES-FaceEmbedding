#!/usr/bin/env python3
"""
Light entrypoint for robustness-aware intrinsic training.

The implementation now lives under `intrinsic/robustness/`, while this script
keeps a small compatibility surface for older imports.
"""

try:
    from robustness.data_utils import sample_to_device
    from robustness.eval_utils import (
        SubjectEvalContext,
        build_sigma_grid as _build_sigma_grid,
        evaluate_subject_robustness_grid as evaluate_robustness_grid,
        evaluate_subjects_at_sigma as evaluate_at_sigma,
        ratio_auc as _ratio_auc,
    )
    from robustness.model_helpers import SpectrumMLPBaseline, build_model, forward_model, smooth_term_from_model
    from robustness.noise import (
        PerturbationParams,
        apply_xyz_perturbation,
        parse_noise_modes,
        sample_log_uniform_sigma,
    )
    from robustness.train_runner import main, make_run_dir, parse_args, run_training
except ImportError:  # pragma: no cover - package import fallback
    from .robustness.data_utils import sample_to_device
    from .robustness.eval_utils import (
        SubjectEvalContext,
        build_sigma_grid as _build_sigma_grid,
        evaluate_subject_robustness_grid as evaluate_robustness_grid,
        evaluate_subjects_at_sigma as evaluate_at_sigma,
        ratio_auc as _ratio_auc,
    )
    from .robustness.model_helpers import SpectrumMLPBaseline, build_model, forward_model, smooth_term_from_model
    from .robustness.noise import (
        PerturbationParams,
        apply_xyz_perturbation,
        parse_noise_modes,
        sample_log_uniform_sigma,
    )
    from .robustness.train_runner import main, make_run_dir, parse_args, run_training


__all__ = [
    "PerturbationParams",
    "SpectrumMLPBaseline",
    "SubjectEvalContext",
    "_build_sigma_grid",
    "_ratio_auc",
    "apply_xyz_perturbation",
    "build_model",
    "evaluate_at_sigma",
    "evaluate_robustness_grid",
    "forward_model",
    "main",
    "make_run_dir",
    "parse_args",
    "parse_noise_modes",
    "run_training",
    "sample_log_uniform_sigma",
    "sample_to_device",
    "smooth_term_from_model",
]


if __name__ == "__main__":
    main()
