"""Validate benchmark data without importing optional runtime features."""
import math


def validate_l2_config(mode="off", min_score=2.0, margin=0.1,
                       demotion_budget=2, prefill_budget=8):
    if mode not in ("off", "fixed", "dynamic"):
        raise ValueError("l2_mode must be off, fixed or dynamic")
    for value, name in ((min_score, "l2_min_score"), (margin, "l2_margin")):
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    for value, name in ((demotion_budget, "l2_demotion_budget"), (prefill_budget, "l2_prefill_budget")):
        if type(value) is not int or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
