"""Approved analysis tool: inverse-variance (fixed-effects) meta-analysis.

Sensitive action: callers must obtain human approval before invoking
(see medai/approvals.py). Pure Python; only stdlib math.
"""
from __future__ import annotations

from math import exp, log, sqrt


def meta_analysis_fixed(effects: list[dict]) -> dict:
    """Pool effect estimates {y, se} by the inverse-variance method.

    Returns {k, pooled_point, ci_lo, ci_hi, z, p_value_approx, Q, I2}.
    pooled_point is exponentiated back for ratio measures (rr/or) when all
    inputs are ratio measures; otherwise returned on the raw scale.
    """
    if not effects or any(e["se"] <= 0 for e in effects):
        raise ValueError("meta-analysis requires at least one effect with se > 0")
    w = [1.0 / e["se"] ** 2 for e in effects]
    y_hat = sum(wi * e["y"] for wi, e in zip(w, effects)) / sum(w)
    se_hat = sqrt(1.0 / sum(w))
    z = y_hat / se_hat
    ci_lo_y, ci_hi_y = y_hat - 1.96 * se_hat, y_hat + 1.96 * se_hat

    q = sum(wi * (e["y"] - y_hat) ** 2 for wi, e in zip(w, effects))
    df = len(effects) - 1
    i2 = max(0.0, (q - df) / q) * 100 if q > 0 and df > 0 else 0.0

    measures = {e["measure"] for e in effects}
    ratio = measures <= {"rr", "or", "hr"}
    scale = lambda v: exp(v) if ratio else v  # noqa: E731

    return dict(
        k=len(effects),
        effect_measure=("ratio" if ratio else "mean_difference"),
        pooled_point=round(scale(y_hat), 4),
        ci_lo=round(scale(ci_lo_y), 4),
        ci_hi=round(scale(ci_hi_y), 4),
        z=round(z, 4),
        p_value=round(_norm_sf(abs(z)) * 2, 5),
        Q=round(q, 4),
        I2=round(i2, 2),
        cited_papers=sorted({e["paper_id"] for e in effects}),
    )


def _norm_sf(x: float) -> float:
    """Survival function of the standard normal (Abramowitz-Stegun 7.1.26)."""
    from math import erf, sqrt as _s
    return 0.5 * (1.0 - erf(x / _s(2.0)))


def valid_analysis(result) -> bool:
    return (isinstance(result, dict)
            and {"k", "pooled_point", "ci_lo", "ci_hi", "Q", "I2", "cited_papers"} <= set(result)
            and isinstance(result["k"], int) and result["k"] > 0
            and all(isinstance(result[k], (int, float)) for k in ("pooled_point", "ci_lo", "ci_hi")))
