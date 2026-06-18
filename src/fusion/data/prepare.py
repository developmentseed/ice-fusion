"""Rate-of-change preparation for the v1 metric. **Sara's territory.**

Direct port of the relevant prototype helpers (see
``https://github.com/sc-peters/PSUISM_HBM_V1/blob/main/push%204_13_26/full_model.py``):

- ``compute_model_dhdt`` / ``compute_model_velocity_change``: finite
  differences of the modelled fields ``h``, ``ua``, ``va``.
- ``compute_obs_dhdt_on_model_intervals`` /
  ``compute_obs_dvdt_on_model_intervals``: align obs years with model
  interval endpoints (via ``snap_model_year_to_obs_year``) and propagate
  uncertainty as ``sqrt(σ_t1² + σ_t2²) / dt``.
- ``flatten_and_mask_combined``: drop non-finite + above-threshold
  pixels, concatenate dh/dt then dv/dt blocks, build a per-pixel
  observed-speed vector for the speed-dependent uncertainty inflation
  in the PyMC model.

Final 20k subsample is seeded — same seed in ice-fusion and the patched
prototype produces identical indices, which is what makes the
validation harness bit-exact.

Per-stream uncertainty thresholds default to the prototype's values and
are surfaced on ``MetricConfig`` (``thick_unc_threshold`` /
``vel_unc_threshold``).
"""

from dataclasses import dataclass

import numpy as np
import xarray as xr

from fusion._array_types import Float32Array, FloatArray
from fusion.config import InferenceConfig, MetricConfig
from fusion.data.time_utils import snap_model_year_to_obs_year


@dataclass
class PreparedData:
    """Output of :func:`prepare` — exactly the inputs the v1 PyMC model consumes.

    All arrays share the same first axis, of length ``n_dhdt + n_vel``.
    The first ``n_dhdt`` entries are thickness-rate observations; the
    remainder are velocity-rate observations (``vx`` then ``vy``
    concatenated, per-interval).
    """

    y_obs: FloatArray
    sigma_obs: FloatArray
    F: FloatArray  # (M, N)
    speed: FloatArray
    member_ids: list[str]
    n_dhdt: int
    n_vel: int


def prepare(
    obs: dict[str, xr.Dataset],
    ensemble: xr.Dataset,
    inference_cfg: InferenceConfig,
    metric_cfg: MetricConfig | None = None,
) -> PreparedData:
    """Build the flattened (y, σ, F, speed) arrays the PyMC model expects.

    Parameters
    ----------
    obs
        ``{"elevation": ds, "velocity": ds}`` from
        :func:`fusion.data.obs.load_observations`. Each dataset has
        coords ``year, y, x`` and the prototype's variable names
        (``height`` + ``absolute_elevation_rmse`` for elevation;
        ``VX, VY, ERRX, ERRY`` for velocity).
    ensemble
        The PSU-ISM ensemble with dims ``(member, time, y, x)`` and
        variables ``h, ua, va``. ``time`` is decimal years.
    inference_cfg
        Used here only for the subsample size and seed.
    metric_cfg
        Supplies the per-stream uncertainty thresholds. Defaults to
        ``MetricConfig(type="pixelwise_gaussian")`` (prototype values).

    Returns
    -------
    PreparedData
        See class docstring.
    """
    if metric_cfg is None:
        metric_cfg = MetricConfig(type="pixelwise_gaussian")

    members = [str(m) for m in ensemble["member"].values]

    # Reference time grid: take from the first member; assume all members
    # share the same time axis (the adapter concatenates with join="outer",
    # but for v1 we expect identical axes).
    t0: FloatArray = ensemble["time"].values.astype(np.float64)
    if t0.size < 2:
        raise ValueError("Ensemble needs ≥ 2 time steps to compute dh/dt")
    dt: FloatArray = np.diff(t0)
    if np.any(dt == 0):
        raise ValueError("Duplicate ensemble time values")

    dhdt_models = [_model_dhdt(ensemble.sel(member=m), dt) for m in members]
    dvxdt_models = []
    dvydt_models = []
    for m in members:
        dvx, dvy = _model_dvdt(ensemble.sel(member=m), dt)
        dvxdt_models.append(dvx)
        dvydt_models.append(dvy)

    obs_dhdt, obs_dhdt_unc = _obs_dhdt_on_intervals(obs["elevation"], t0, dt)
    obs_dhdt_unc = _fill_thickness_unc(obs_dhdt_unc)
    obs_dvxdt, obs_dvydt, obs_uncx, obs_uncy = _obs_dvdt_on_intervals(obs["velocity"], t0, dt)
    speed_mean = _mean_obs_speed(obs["velocity"])

    y, sigma, F, speed, n_dhdt, n_vel = _flatten_and_mask_combined(
        obs_dhdt,
        obs_dhdt_unc,
        dhdt_models,
        obs_dvxdt,
        obs_dvydt,
        obs_uncx,
        obs_uncy,
        dvxdt_models,
        dvydt_models,
        speed_mean,
        thick_unc_threshold=metric_cfg.thick_unc_threshold,
        vel_unc_threshold=metric_cfg.vel_unc_threshold,
    )

    # Final subsample (matches prototype: drawn from the combined
    # n_dhdt + n_vel array; n_dhdt_new is the number of indices that
    # land in the thickness block).
    sub = inference_cfg.subsample
    n_total = y.size
    if sub.size < n_total:
        rng = np.random.default_rng(sub.seed)
        idx = rng.choice(n_total, size=sub.size, replace=False)
        y = y[idx]
        sigma = sigma[idx]
        F = F[:, idx]
        speed = speed[idx]
        n_dhdt_new = int(np.sum(idx < n_dhdt))
        n_vel_new = int(sub.size - n_dhdt_new)
    else:
        n_dhdt_new = n_dhdt
        n_vel_new = n_vel

    # Promote to float64 once at the boundary. Inference previously did
    # this via `.astype(float)`; doing it here avoids the per-stream
    # dtype mix and keeps PreparedData consumers consistent.
    return PreparedData(
        y_obs=y.astype(np.float64, copy=False),
        sigma_obs=sigma.astype(np.float64, copy=False),
        F=F.astype(np.float64, copy=False),
        speed=speed.astype(np.float64, copy=False),
        member_ids=members,
        n_dhdt=n_dhdt_new,
        n_vel=n_vel_new,
    )


# ---------------------------------------------------------------------
# Model-side rates
# ---------------------------------------------------------------------


def _member_rate(arr_3d: Float32Array, dt: FloatArray) -> FloatArray:
    """Finite-difference of a (time, y, x) array along time, divided by dt.

    ``arr_3d`` is storage-precision float32; dividing by the float64 ``dt``
    promotes the result to float64.
    """
    rate: FloatArray = (arr_3d[1:] - arr_3d[:-1]) / dt[:, None, None]
    return rate


def _model_dhdt(member_ds: xr.Dataset, dt: FloatArray) -> FloatArray:
    return _member_rate(member_ds["h"].values.astype("float32"), dt)


def _model_dvdt(member_ds: xr.Dataset, dt: FloatArray) -> tuple[FloatArray, FloatArray]:
    dvx = _member_rate(member_ds["ua"].values.astype("float32"), dt)
    dvy = _member_rate(member_ds["va"].values.astype("float32"), dt)
    return dvx, dvy


# ---------------------------------------------------------------------
# Obs-side rates on model intervals
# ---------------------------------------------------------------------


def _obs_rate_on_intervals(
    field: xr.DataArray,
    sigma: xr.DataArray,
    model_t: FloatArray,
    model_dt: FloatArray,
) -> tuple[Float32Array, Float32Array]:
    """Snap obs years to model interval endpoints, return (rate, unc).

    ``field`` and ``sigma`` are per-year DataArrays with dim ``year``.
    Both outputs are float32 arrays of shape ``(n_intervals, ny, nx)``.
    """
    years = field["year"].values.astype(int)
    sample_shape = field.isel(year=0).shape
    n_intervals = len(model_t) - 1
    rate = np.full((n_intervals,) + sample_shape, np.nan, dtype="float32")
    unc = np.full_like(rate, np.nan)
    for i in range(n_intervals):
        y1 = snap_model_year_to_obs_year(model_t[i], years)
        y2 = snap_model_year_to_obs_year(model_t[i + 1], years)
        if y1 is None or y2 is None:
            continue
        f1 = field.sel(year=y1).values.astype("float32")
        f2 = field.sel(year=y2).values.astype("float32")
        s1 = sigma.sel(year=y1).values.astype("float32")
        s2 = sigma.sel(year=y2).values.astype("float32")
        dt_i = float(model_dt[i])
        rate[i] = (f2 - f1) / dt_i
        unc[i] = np.sqrt(s1**2 + s2**2) / dt_i
    return rate, unc


def _obs_dhdt_on_intervals(
    elev: xr.Dataset, model_t: FloatArray, model_dt: FloatArray
) -> tuple[Float32Array, Float32Array]:
    return _obs_rate_on_intervals(
        elev["height"], elev["absolute_elevation_rmse"], model_t, model_dt
    )


def _obs_dvdt_on_intervals(
    vel: xr.Dataset, model_t: FloatArray, model_dt: FloatArray
) -> tuple[Float32Array, Float32Array, Float32Array, Float32Array]:
    dvxdt, uncx = _obs_rate_on_intervals(vel["VX"], vel["ERRX"], model_t, model_dt)
    dvydt, uncy = _obs_rate_on_intervals(vel["VY"], vel["ERRY"], model_t, model_dt)
    return dvxdt, dvydt, uncx, uncy


def _fill_thickness_unc(unc: Float32Array) -> Float32Array:
    """Match the prototype's NaN-fill for the dh/dt obs uncertainty.

    Direct port of ``prepare_for_inference`` (full_model.py): all-NaN
    falls back to a constant ``20 m/yr``; partial-NaN falls back to the
    median of finite values. The reference dataset has
    ``absolute_elevation_rmse`` 100% NaN, so the constant-20 branch is
    what fires in practice.
    """
    if not np.any(np.isfinite(unc)):
        return np.full_like(unc, 20.0)
    if np.any(np.isnan(unc)):
        fill = float(np.nanmedian(unc))
        return np.where(np.isfinite(unc), unc, fill).astype(unc.dtype)
    return unc


def _mean_obs_speed(vel: xr.Dataset) -> Float32Array:
    # Cast to float32 to match the prototype: load_obs_velocity_yearly
    # downcasts VX/VY at read time, so the per-year speed (sqrt of squared
    # sums) and the across-year mean are computed in float32. Without this,
    # ice-fusion stays in file-dtype float64 and Layer 1 of the validation
    # harness shows a ~4e-4 m/yr divergence on `speed` alone.
    vx = vel["VX"].values.astype("float32")
    vy = vel["VY"].values.astype("float32")
    speed: Float32Array = np.nanmean(np.sqrt(vx**2 + vy**2), axis=0)
    return speed


# ---------------------------------------------------------------------
# Flatten + mask
# ---------------------------------------------------------------------


def _flatten_and_mask_combined(
    obs_dhdt: Float32Array,
    obs_dhdt_unc: Float32Array,
    dhdt_models: list[FloatArray],
    obs_dvxdt: Float32Array,
    obs_dvydt: Float32Array,
    obs_uncx: Float32Array,
    obs_uncy: Float32Array,
    dvxdt_models: list[FloatArray],
    dvydt_models: list[FloatArray],
    speed_mean: Float32Array,
    *,
    thick_unc_threshold: float,
    vel_unc_threshold: float,
) -> tuple[Float32Array, Float32Array, FloatArray, Float32Array, int, int]:
    M = len(dhdt_models)

    # ---- thickness block ----
    y_t = obs_dhdt.reshape(-1)
    s_t = obs_dhdt_unc.reshape(-1)
    F_t = np.stack([m.reshape(-1) for m in dhdt_models], axis=0)

    mask_t = np.isfinite(y_t) & np.isfinite(s_t) & (s_t < thick_unc_threshold)
    mask_t &= np.isfinite(F_t).all(axis=0)
    y_t = y_t[mask_t]
    s_t = s_t[mask_t]
    F_t = F_t[:, mask_t]
    speed_t = np.zeros_like(y_t, dtype=float)
    n_dhdt = y_t.size

    # ---- velocity block (per interval, vx then vy) ----
    n_intervals = obs_dvxdt.shape[0]
    speed_flat = speed_mean.reshape(-1)
    # Pre-stack model velocity arrays once; shape (M, n_intervals, ny*nx).
    dvxdt_stack = np.stack([m.reshape(m.shape[0], -1) for m in dvxdt_models], axis=0)
    dvydt_stack = np.stack([m.reshape(m.shape[0], -1) for m in dvydt_models], axis=0)
    y_v_list, s_v_list, F_v_list, sp_v_list = [], [], [], []
    for i in range(n_intervals):
        dvx_obs = obs_dvxdt[i].reshape(-1)
        dvy_obs = obs_dvydt[i].reshape(-1)
        dvx_err = obs_uncx[i].reshape(-1)
        dvy_err = obs_uncy[i].reshape(-1)
        y_int = np.concatenate([dvx_obs, dvy_obs])
        s_int = np.concatenate([dvx_err, dvy_err])
        speed_int = np.concatenate([speed_flat, speed_flat])
        # Slice per-interval from the pre-stacked arrays, concatenated
        # vx + vy along the per-pixel axis. Cast to float64 to preserve
        # the original `np.zeros((M, ...))` default dtype.
        F_int = np.concatenate([dvxdt_stack[:, i, :], dvydt_stack[:, i, :]], axis=1).astype(
            np.float64
        )

        mask_v = (
            np.isfinite(y_int)
            & np.isfinite(s_int)
            & np.isfinite(speed_int)
            & (s_int < vel_unc_threshold)
        )
        mask_v &= np.isfinite(F_int).all(axis=0)

        y_v_list.append(y_int[mask_v])
        s_v_list.append(s_int[mask_v])
        F_v_list.append(F_int[:, mask_v])
        sp_v_list.append(speed_int[mask_v])

    if y_v_list:
        y_v = np.concatenate(y_v_list)
        s_v = np.concatenate(s_v_list)
        F_v = np.concatenate(F_v_list, axis=1)
        speed_v = np.concatenate(sp_v_list)
    else:
        y_v = np.zeros(0)
        s_v = np.zeros(0)
        F_v = np.zeros((M, 0))
        speed_v = np.zeros(0)
    n_vel = y_v.size

    # ---- combine ----
    y = np.concatenate([y_t, y_v])
    sigma = np.concatenate([s_t, s_v])
    F = np.concatenate([F_t, F_v], axis=1)
    speed = np.concatenate([speed_t, speed_v])

    if not np.all(np.isfinite(speed)):
        raise RuntimeError("speed vector contains NaNs after masking")

    return y, sigma, F, speed, n_dhdt, n_vel
