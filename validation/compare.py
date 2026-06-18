"""Side-by-side validation: pinned full_model.py vs ice-fusion.

Driver for ice-fusion v1 Task 15 (see
``dev-docs/plans/implementation_plan.md``). Runs both stacks on the same
reference inputs and diffs at three layers:

1. **Prepared arrays** (``y_obs``, ``sigma_obs``, ``F``, ``speed``,
   ``n_dhdt``, ``n_vel``) — bit-exact. The first place a quiet port
   divergence shows up.
2. **Per-member plug-in log-likelihood** — cross-implementation:
   ice-fusion's ``plug_in_weights`` vs the prototype's
   ``compute_model_weights``, both at the same canonical posterior
   (ice-fusion's). Gated by ``rtol`` rather than exact equality, since
   the prototype computes in float32 and ice-fusion in float64; the
   tolerance is set well below an O(N) scaling/formula divergence.
3. **Posterior summaries** (``sigma_base_*``, ``beta_*``, weights ``w``)
   — ``rtol=1e-3``. MCMC reproducibility across processes is fragile;
   these are the loosest comparisons.

Writes ``validation/reports/<YYYY-MM-DD>.md`` with sign-off slots.

Run from the ice-fusion repo root::

    uv run python -m validation.compare

Reference inputs are expected at ``validation/data/`` (gitignored — see
``validation/baseline/README.md`` for the layout).
"""

from __future__ import annotations

import datetime
import os
import sys
from pathlib import Path

import arviz as az
import numpy as np

import fusion
from fusion.config import (
    Config,
    EnsembleConfig,
    GridConfig,
    InferenceConfig,
    MetricConfig,
    ObservationsConfig,
    ProjectionConfig,
    StreamWeights,
    SubsampleConfig,
)
from fusion.data.prepare import PreparedData
from fusion.pipeline import plug_in_weights
from validation.baseline import full_model as fm

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

VALIDATION_DIR = Path(__file__).resolve().parent
DATA_DIR = VALIDATION_DIR / "data"
ENS_DIR = DATA_DIR / "ensemble"
OBS_VERSION = "ref_2026_04_30"
OBS_ROOT = DATA_DIR / "obs"
ELEV_DIR = OBS_ROOT / OBS_VERSION / "elevation"
VEL_DIR = OBS_ROOT / OBS_VERSION / "velocity"
REPORTS_DIR = VALIDATION_DIR / "reports"

# ---------------------------------------------------------------------
# Shared inference settings — pinned to the prototype's hardcoded values.
# ---------------------------------------------------------------------

SUBSAMPLE_SEED = 42
SUBSAMPLE_SIZE = 20_000
DRAWS = 500
TUNE = 1000
CHAINS = 4
TARGET_ACCEPT = 0.95
ALPHA_THICK = 0.5
ALPHA_VEL = 0.5

POSTERIOR_VARS = ("sigma_base_thick", "sigma_base_vel", "beta_thick", "beta_vel")
RTOL_POSTERIOR = 1e-3
# Layer 2 gate. The float32 (prototype) vs float64 (ice-fusion) rounding gap
# is ~1e-9 relative; a scaling/formula divergence is O(N) ≈ 1e4. 1e-6 sits
# comfortably between the two.
RTOL_LOGLIK = 1e-6


# ---------------------------------------------------------------------
# Prototype side — call fm.main() with env-var-overridden paths/seed.
# ---------------------------------------------------------------------


def run_prototype():
    """Run the pinned prototype against ``validation/data/``.

    Sets the env vars our patched baseline reads (paths + seed + size),
    then calls ``fm.main()``. Returns ``(data_dict, trace)``.
    """
    os.environ["FUSION_OBS_THICKNESS_DIR"] = str(ELEV_DIR)
    os.environ["FUSION_OBS_VELOCITY_DIR"] = str(VEL_DIR)
    os.environ["FUSION_MODEL_DIR"] = str(ENS_DIR)
    os.environ["FUSION_SUBSAMPLE_SEED"] = str(SUBSAMPLE_SEED)
    os.environ["FUSION_SUBSAMPLE_SIZE"] = str(SUBSAMPLE_SIZE)

    print("=" * 70)
    print("PROTOTYPE: fm.main()")
    print("=" * 70)
    trace, _, data, _ = fm.main()
    return data, trace


# ---------------------------------------------------------------------
# ice-fusion side — point load_observations at validation/data/obs/.
# ---------------------------------------------------------------------


def run_fusion():
    """Run the fusion pipeline step-by-step with phase markers.

    Decomposes ``fusion.run(cfg)`` so we can print between phases —
    data load + prepare are otherwise silent, leaving the user staring
    at no output for ~30 s–2 min before the MCMC progress bar appears.
    """
    from types import SimpleNamespace

    os.environ["FUSION_CACHE"] = str(OBS_ROOT)

    cfg = Config(
        ensemble=EnsembleConfig(path=ENS_DIR, adapter="psuism"),
        observations=ObservationsConfig(source="source-coop", version=OBS_VERSION),
        grid=GridConfig(target="obs_8km", method="bilinear"),
        metric=MetricConfig(type="pixelwise_gaussian"),
        inference=InferenceConfig(
            obs_alpha=0.5,
            stream_weights=StreamWeights(thick=ALPHA_THICK, vel=ALPHA_VEL),
            subsample=SubsampleConfig(size=SUBSAMPLE_SIZE, seed=SUBSAMPLE_SEED),
            draws=DRAWS,
            tune=TUNE,
            chains=CHAINS,
            target_accept=TARGET_ACCEPT,
        ),
        projection=ProjectionConfig(target_year=2100, quantity="grounded_ice_volume"),
    )
    print("\n" + "=" * 70)
    print("ICE-FUSION: pipeline")
    print("=" * 70)

    print("[1/3] loading data (obs + ensemble)...", flush=True)
    data = fusion.load_data(cfg)
    print(
        f"      ensemble members={data['ensemble'].sizes['member']}, "
        f"time={data['ensemble'].sizes['time']}",
        flush=True,
    )

    print("[2/3] preparing arrays (rate-of-change + flatten + mask + subsample)...", flush=True)
    prepared = fusion.prepare(cfg, data)
    print(
        f"      n_obs={prepared.y_obs.size:,} "
        f"(n_dhdt={prepared.n_dhdt:,}, n_vel={prepared.n_vel:,})",
        flush=True,
    )

    print("[3/3] sampling (PyMC progress follows)...", flush=True)
    trace = fusion.sample(cfg, prepared)
    return SimpleNamespace(prepared=prepared, trace=trace)


# ---------------------------------------------------------------------
# Diff layers
# ---------------------------------------------------------------------


def diff_prepared(data_proto: dict, prepared_fus: PreparedData) -> dict:
    """Bit-exact comparison of the inputs the PyMC model consumes."""
    out: dict = {
        "n_obs_proto": int(data_proto["n_obs"]),
        "n_obs_fus": int(prepared_fus.y_obs.size),
        "n_dhdt_match": int(data_proto["n_dhdt"]) == int(prepared_fus.n_dhdt),
        "n_vel_match": int(data_proto["n_vel"]) == int(prepared_fus.n_vel),
    }
    out["n_obs_match"] = out["n_obs_proto"] == out["n_obs_fus"]

    if not out["n_obs_match"]:
        out["arrays_compared"] = False
        return out

    out["arrays_compared"] = True
    pairs = {
        "y_obs": (data_proto["y_obs"], prepared_fus.y_obs),
        "sigma_obs": (data_proto["sigma_obs"], prepared_fus.sigma_obs),
        "speed": (data_proto["speed"], prepared_fus.speed),
        "F": (data_proto["F"], prepared_fus.F),
    }
    for name, (a, b) in pairs.items():
        a = np.asarray(a)
        b = np.asarray(b)
        if a.shape != b.shape:
            out[f"{name}_match"] = False
            out[f"{name}_max_abs_diff"] = float("nan")
            out[f"{name}_shape_proto"] = a.shape
            out[f"{name}_shape_fus"] = b.shape
            continue
        out[f"{name}_match"] = bool(np.array_equal(a, b, equal_nan=True))
        diff = np.abs(a.astype(float) - b.astype(float))
        finite = np.isfinite(diff)
        out[f"{name}_max_abs_diff"] = float(diff[finite].max()) if finite.any() else float("nan")
    return out


def diff_loglik(data_proto: dict, prepared_fus: PreparedData, trace_fus) -> dict:
    """Cross-implementation check of the per-member plug-in log-likelihood.

    ``ll_proto`` comes from the prototype's own ``compute_model_weights``;
    ``ll_fus`` from ice-fusion's ``plug_in_weights``. Both are evaluated at
    the *same* canonical posterior (ice-fusion's trace), so the only thing
    that can differ is the two log-likelihood implementations themselves —
    a formula/scaling divergence (e.g. a missing ``1/N`` factor) shows up
    here even when the prepared arrays match Layer 1.

    Exact equality is not attainable: the prototype computes in the file
    dtype (float32 obs arrays) while ice-fusion promotes to float64, so
    ``sigma_obs**2`` and friends round differently. The gate is therefore a
    relative tolerance — tight enough to catch a scaling/formula error
    (those are O(N) ≈ 10^4 relative) but loose enough to admit the
    float32-vs-float64 rounding gap (~1e-9 relative in practice)."""
    if prepared_fus.y_obs.size != int(data_proto["n_obs"]):
        return {
            "loglik_compared": False,
            "reason": "prepared shapes differ; skipping loglik diff",
        }

    # Prototype's loglik, via its own weight routine (returns scaled loglik).
    _, ll_proto = fm.compute_model_weights(trace_fus, data_proto)
    # ice-fusion's loglik, via its own routine.
    _, ll_fus = plug_in_weights(prepared_fus, trace_fus)
    ll_proto = np.asarray(ll_proto, dtype=float)
    ll_fus = np.asarray(ll_fus, dtype=float)

    abs_diff = np.abs(ll_proto - ll_fus)
    rel_diff = abs_diff / np.maximum(np.abs(ll_proto), 1e-30)
    return {
        "loglik_compared": True,
        "loglik_match": bool(np.array_equal(ll_proto, ll_fus)),
        "loglik_within_rtol": bool((rel_diff <= RTOL_LOGLIK).all()),
        "loglik_rtol": RTOL_LOGLIK,
        "loglik_max_abs_diff": float(abs_diff.max()),
        "loglik_max_rel_diff": float(rel_diff.max()),
        "loglik_proto": ll_proto.tolist(),
        "loglik_fus": ll_fus.tolist(),
    }


def diff_posterior(trace_proto, trace_fus, rtol: float = RTOL_POSTERIOR) -> dict:
    """``rtol=1e-3`` comparison of posterior means for the sigma/beta
    scalars and the deterministic ``w`` vector."""
    out: dict = {"rtol": rtol, "vars": {}}
    sum_proto = az.summary(trace_proto, var_names=list(POSTERIOR_VARS), hdi_prob=0.95)
    sum_fus = az.summary(trace_fus, var_names=list(POSTERIOR_VARS), hdi_prob=0.95)
    for v in POSTERIOR_VARS:
        mean_p = float(sum_proto.loc[v, "mean"])
        mean_f = float(sum_fus.loc[v, "mean"])
        rel = abs(mean_p - mean_f) / max(abs(mean_p), 1e-30)
        out["vars"][v] = {
            "mean_proto": mean_p,
            "mean_fus": mean_f,
            "rel_diff": rel,
            "within_rtol": rel <= rtol,
        }

    w_proto = trace_proto.posterior["w"].mean(dim=("chain", "draw")).values
    w_fus = trace_fus.posterior["w"].mean(dim=("chain", "draw")).values
    if w_proto.shape == w_fus.shape:
        rel_w = np.abs(w_proto - w_fus) / np.maximum(np.abs(w_proto), 1e-30)
        out["w"] = {
            "max_rel_diff": float(rel_w.max()),
            "all_within_rtol": bool((rel_w <= rtol).all()),
            "w_proto": w_proto.tolist(),
            "w_fus": w_fus.tolist(),
        }
    else:
        out["w"] = {"shape_mismatch": True}
    return out


# ---------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------


def write_report(prep_diff: dict, ll_diff: dict, post_diff: dict, path: Path) -> None:
    today = datetime.date.today().isoformat()
    lines = [f"# ice-fusion ↔ full_model.py validation — {today}", ""]
    lines.append(
        "Side-by-side run on the reference PSU-ISM ensemble in `validation/data/`. "
        "Driver: `validation/compare.py`. Oracle: `validation/baseline/full_model.py`."
    )
    lines.append("")

    lines.append("## Layer 1 — Prepared arrays (bit-exact)")
    lines.append("")
    lines.append(f"- n_obs (proto): {prep_diff['n_obs_proto']:,}")
    lines.append(f"- n_obs (fus):   {prep_diff['n_obs_fus']:,}")
    lines.append(f"- n_dhdt match:  {prep_diff['n_dhdt_match']}")
    lines.append(f"- n_vel match:   {prep_diff['n_vel_match']}")
    if prep_diff.get("arrays_compared"):
        for name in ("y_obs", "sigma_obs", "speed", "F"):
            match = prep_diff[f"{name}_match"]
            mad = prep_diff[f"{name}_max_abs_diff"]
            lines.append(f"- {name}: match={match}, max_abs_diff={mad:.3e}")
    else:
        lines.append("- Arrays not compared (length mismatch).")
    lines.append("")

    lines.append("## Layer 2 — Per-member plug-in log-likelihood (ice-fusion vs prototype)")
    lines.append("")
    if ll_diff.get("loglik_compared"):
        lines.append(
            f"- within rtol ({ll_diff['loglik_rtol']:.0e}): {ll_diff['loglik_within_rtol']}"
        )
        lines.append(f"- exact match: {ll_diff['loglik_match']}")
        lines.append(f"- max_abs_diff: {ll_diff['loglik_max_abs_diff']:.3e}")
        lines.append(f"- max_rel_diff: {ll_diff['loglik_max_rel_diff']:.3e}")
        lines.append("")
        lines.append("| member | proto loglik | fus loglik |")
        lines.append("|---|---|---|")
        for i, (a, b) in enumerate(
            zip(ll_diff["loglik_proto"], ll_diff["loglik_fus"], strict=False)
        ):
            lines.append(f"| {i} | {a:.6f} | {b:.6f} |")
    else:
        lines.append(f"- Skipped: {ll_diff.get('reason')}")
    lines.append("")

    lines.append(f"## Layer 3 — Posterior summaries (rtol={post_diff['rtol']:.0e})")
    lines.append("")
    lines.append("| var | mean (proto) | mean (fus) | rel diff | within rtol |")
    lines.append("|---|---|---|---|---|")
    for v, d in post_diff["vars"].items():
        lines.append(
            f"| {v} | {d['mean_proto']:.6f} | {d['mean_fus']:.6f} | "
            f"{d['rel_diff']:.3e} | {d['within_rtol']} |"
        )
    lines.append("")
    if "shape_mismatch" not in post_diff["w"]:
        lines.append(
            f"- w max_rel_diff: {post_diff['w']['max_rel_diff']:.3e}, "
            f"all_within_rtol: {post_diff['w']['all_within_rtol']}"
        )
    lines.append("")

    lines.append("## Sign-off")
    lines.append("")
    lines.append("- [ ] Max")
    lines.append("- [ ] Sara")
    lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))
    print(f"\nReport written to {path}")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main() -> int:
    if not DATA_DIR.exists():
        raise FileNotFoundError(
            f"{DATA_DIR} not found. Drop the reference inputs there per "
            f"validation/baseline/README.md."
        )

    data_proto, trace_proto = run_prototype()
    result = run_fusion()

    print("\n" + "=" * 70)
    print("DIFFING")
    print("=" * 70)
    prep_diff = diff_prepared(data_proto, result.prepared)
    print("Layer 1:", prep_diff)

    ll_diff = diff_loglik(data_proto, result.prepared, result.trace)
    print(
        "Layer 2:",
        {k: v for k, v in ll_diff.items() if k not in {"loglik_proto", "loglik_fus"}},
    )

    post_diff = diff_posterior(trace_proto, result.trace)
    print("Layer 3:", {v: d["rel_diff"] for v, d in post_diff["vars"].items()})

    today = datetime.date.today().isoformat()
    write_report(prep_diff, ll_diff, post_diff, REPORTS_DIR / f"{today}.md")

    layer1_ok = (
        prep_diff.get("arrays_compared")
        and all(prep_diff[f"{n}_match"] for n in ("y_obs", "sigma_obs", "speed", "F"))
        and prep_diff["n_dhdt_match"]
        and prep_diff["n_vel_match"]
    )
    layer2_ok = ll_diff.get("loglik_compared") and ll_diff.get("loglik_within_rtol")
    layer3_ok = all(d["within_rtol"] for d in post_diff["vars"].values())

    print("\nSummary:")
    print(f"  Layer 1 (prepared, bit-exact):  {'PASS' if layer1_ok else 'FAIL'}")
    print(f"  Layer 2 (loglik, rtol={RTOL_LOGLIK:.0e}):   {'PASS' if layer2_ok else 'FAIL'}")
    print(f"  Layer 3 (posterior, rtol=1e-3): {'PASS' if layer3_ok else 'FAIL'}")
    return 0 if (layer1_ok and layer2_ok and layer3_ok) else 1


if __name__ == "__main__":
    sys.exit(main())
