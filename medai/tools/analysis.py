"""Approved analysis tool: meta-analysis with automatic model selection.

Two estimators over the same effect units {y, se}:
  * fixed-effect  (inverse variance) — assumes one true effect;
  * random-effects (DerSimonian-Laird) — adds between-study variance tau^2,
    giving wider CIs when studies genuinely disagree.

Model selection: when heterogeneity is substantial (I^2 >= 50%), the
random-effects estimate becomes the primary result — pooling heterogeneous
studies with a fixed-effect model overstates precision. Both estimates and
tau^2 are always returned. This is a sensitive action: callers must obtain
human approval before invoking (see medai/approvals.py).
"""
from __future__ import annotations

from math import erf, exp, sqrt


def _fixed(effects: list[dict]) -> tuple[float, float]:
    w = [1.0 / e["se"] ** 2 for e in effects]
    y = sum(wi * e["y"] for wi, e in zip(w, effects)) / sum(w)
    return y, sqrt(1.0 / sum(w))


def _derSimonian_laird_tau2(effects: list[dict], y_fixed: float) -> float:
    w = [1.0 / e["se"] ** 2 for e in effects]
    k = len(effects)
    q = sum(wi * (e["y"] - y_fixed) ** 2 for wi, e in zip(w, effects))
    df = k - 1
    if df <= 0 or q <= df:
        return 0.0
    sum_w, sum_w2 = sum(w), sum(wi * wi for wi in w)
    return max(0.0, (q - df) / (sum_w - sum_w2 / sum_w))


def _random(effects: list[dict], tau2: float) -> tuple[float, float]:
    w = [1.0 / (e["se"] ** 2 + tau2) for e in effects]
    y = sum(wi * e["y"] for wi, e in zip(w, effects)) / sum(w)
    return y, sqrt(1.0 / sum(w))


def _norm_sf(x: float) -> float:
    return 0.5 * (1.0 - erf(x / sqrt(2.0)))


def meta_analysis(effects: list[dict], model: str = "auto") -> dict:
    """Pool effect estimates {y, se, measure, paper_id}.

    model: "auto" (default — random-effects when I^2 >= 50%), "fixed", or
    "random". Returns the primary estimate plus both models, tau^2, Q, I^2.
    """
    if not effects or any(e["se"] <= 0 for e in effects):
        raise ValueError("meta-analysis requires at least one effect with se > 0")

    y_f, se_f = _fixed(effects)
    q = sum((1.0 / e["se"] ** 2) * (e["y"] - y_f) ** 2 for e in effects)
    df = len(effects) - 1
    i2 = max(0.0, (q - df) / q) * 100 if q > 0 and df > 0 else 0.0
    tau2 = _derSimonian_laird_tau2(effects, y_f)
    y_r, se_r = _random(effects, tau2)

    if model == "fixed" or (model == "auto" and i2 < 50):
        chosen, use = "fixed", (y_f, se_f)
    else:
        chosen, use = "random-effects", (y_r, se_r)
    y_hat, se_hat = use
    z = y_hat / se_hat

    measures = {e["measure"] for e in effects}
    ratio = measures <= {"rr", "or", "hr"}
    scale = (lambda v: exp(v)) if ratio else (lambda v: v)

    return dict(
        k=len(effects),
        effect_measure=("ratio" if ratio else "mean_difference"),
        model=chosen,
        heterogeneity=round(i2, 2),       # explicit threshold-driven switch
        tau2=round(tau2, 6),
        pooled_point=round(scale(y_hat), 4),
        ci_lo=round(scale(y_hat - 1.96 * se_hat), 4),
        ci_hi=round(scale(y_hat + 1.96 * se_hat), 4),
        z=round(z, 4),
        p_value=round(_norm_sf(abs(z)) * 2, 5),
        Q=round(q, 4),
        I2=round(i2, 2),
        fixed_effect={"point": round(scale(y_f), 4),
                      "ci": [round(scale(y_f - 1.96 * se_f), 4),
                             round(scale(y_f + 1.96 * se_f), 4)]},
        random_effects={"point": round(scale(y_r), 4),
                        "ci": [round(scale(y_r - 1.96 * se_r), 4),
                               round(scale(y_r + 1.96 * se_r), 4)],
                        "tau2": round(tau2, 6)},
        cited_papers=sorted({e["paper_id"] for e in effects}),
    )


def valid_analysis(result) -> bool:
    if not (isinstance(result, dict)
            and {"k", "pooled_point", "ci_lo", "ci_hi", "Q", "I2", "model",
                 "cited_papers"} <= set(result)
            and isinstance(result["k"], int) and result["k"] > 0
            and result["model"] in ("fixed", "random-effects")
            and all(isinstance(result[kk], (int, float))
                    for kk in ("pooled_point", "ci_lo", "ci_hi"))):
        return False
    return True


# back-compat alias for callers that specifically want the fixed-effect model
def meta_analysis_fixed(effects: list[dict]) -> dict:
    return meta_analysis(effects, model="fixed")


def egger_test(effects: list[dict]) -> dict | None:
    """Egger's regression for funnel-plot asymmetry (small-study effects).

    Regresses the standard normal deviate (y/se) on precision (1/se): a slope
    significantly different from zero indicates asymmetry, i.e. possible
    publication bias. Returns None with fewer than 3 studies. The two-sided
    p-value uses a normal approximation (adequate for k >= 5; interpret
    cautiously for k = 3-4)."""
    k = len(effects)
    if k < 3:
        return None
    z = [e["y"] / e["se"] for e in effects]
    prec = [1.0 / e["se"] for e in effects]
    n = float(k)
    mx, mz = sum(prec) / n, sum(z) / n
    sxx = sum((p - mx) ** 2 for p in prec)
    if sxx <= 0:
        return None
    sxy = sum((p - mx) * (zi - mz) for p, zi in zip(prec, z))
    slope = sxy / sxx
    intercept = mz - slope * mx
    sse = sum((zi - (intercept + slope * p)) ** 2 for p, zi in zip(prec, z))
    se_slope = sqrt(max(sse / (n - 2.0), 0.0) / sxx)
    t_stat = slope / se_slope if se_slope > 0 else 0.0
    p_val = min(1.0, 2.0 * _norm_sf(abs(t_stat)))
    return {"intercept": round(intercept, 4), "slope": round(slope, 4),
            "se_slope": round(se_slope, 4), "t": round(t_stat, 4),
            "p_value": round(p_val, 5), "n": k,
            "asymmetry": bool(p_val < 0.05)}


def leave_one_out(effects: list[dict]) -> list[dict]:
    """Influence analysis: the pooled estimate excluding each study in turn
    (fixed-effect inverse variance). Values are on the display scale
    (ratios exponentiated). A row whose CI changes significance vs the
    full-cohort result flags an influential study."""
    ratio = {e["measure"] for e in effects} <= {"rr", "or", "hr"}
    scale = (lambda v: exp(v)) if ratio else (lambda v: v)
    out = []
    for i, e in enumerate(effects):
        rest = [x for j, x in enumerate(effects) if j != i]
        if not rest:
            continue
        y, se = _fixed(rest)
        out.append({"excluded_paper_id": e["paper_id"],
                    "point": round(scale(y), 4),
                    "ci_lo": round(scale(y - 1.96 * se), 4),
                    "ci_hi": round(scale(y + 1.96 * se), 4),
                    "k": len(rest)})
    return out
