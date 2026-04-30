"""Diagnostic helpers for FUSION runs."""

from __future__ import annotations

import numpy as np


def weight_stability_across_seeds(weights_list: list[np.ndarray]) -> np.ndarray:
    """Per-member standard deviation across multiple seeded runs.

    A common sanity check: rerun the pipeline with several
    ``inference.subsample.seed`` values and pass the resulting
    per-member weight vectors here. Large σ on a member means the
    weight is sensitive to which 20k pixels happened to be drawn.
    """
    stacked = np.stack(weights_list)
    return stacked.std(axis=0)
