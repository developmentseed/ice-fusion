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

Thresholds and the per-stream constants are hardcoded to match the
prototype; promoting them to config is a v1.1 item (see plan §open
questions).
"""

from dataclasses import dataclass

import numpy as np
import xarray as xr

from fusion.config import InferenceConfig
from fusion.data.time_utils import snap_model_year_to_obs_year

THICK_UNC_THRESHOLD = 50.0
VEL_UNC_THRESHOLD = 10.0


@dataclass
class PreparedData:
    """Output of :func:`prepare` — exactly the inputs the v1 PyMC model consumes.

    All arrays share the same first axis, of length ``n_dhdt + n_vel``.
    The first ``n_dhdt`` entries are thickness-rate observations; the
    remainder are velocity-rate observations (``vx`` then ``vy``
    concatenated, per-interval).
    """

    y_obs: np.ndarray
    sigma_obs: np.ndarray
    F: np.ndarray  # (M, N)
    speed: np.ndarray
    member_ids: list[str]
    n_dhdt: int
    n_vel: int


def prepare(
    obs: dict[str, xr.Dataset],
    ensemble: xr.Dataset,
    inference_cfg: InferenceConfig,
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

    Returns
    -------
    PreparedData
        See class docstring.
    """
    members = [str(m) for m in ensemble["member"].values]

    # Reference time grid: take from the first member; assume all members
    # share the same time axis (the adapter concatenates with join="outer",
    # but for v1 we expect identical axes).
    t0 = ensemble["time"].values.astype(np.float64)
    if t0.size < 2:
        raise ValueError("Ensemble needs ≥ 2 time steps to compute dh/dt")
    dt = np.diff(t0)
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

    return PreparedData(
        y_obs=y,
        sigma_obs=sigma,
        F=F,
        speed=speed,
        member_ids=members,
        n_dhdt=n_dhdt_new,
        n_vel=n_vel_new,
    )


# ---------------------------------------------------------------------
# Model-side rates
# ---------------------------------------------------------------------


def _model_dhdt(member_ds: xr.Dataset, dt: np.ndarray) -> np.ndarray:
    h = member_ds["h"].values.astype("float32")
    h0 = h[:-1]
    h1 = h[1:]
    return (h1 - h0) / dt[:, None, None]


def _model_dvdt(member_ds: xr.Dataset, dt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ua = member_ds["ua"].values.astype("float32")
    va = member_ds["va"].values.astype("float32")
    dvx = (ua[1:] - ua[:-1]) / dt[:, None, None]
    dvy = (va[1:] - va[:-1]) / dt[:, None, None]
    return dvx, dvy


# ---------------------------------------------------------------------
# Obs-side rates on model intervals
# ---------------------------------------------------------------------


def _obs_dhdt_on_intervals(
    elev: xr.Dataset, model_t: np.ndarray, model_dt: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    years = elev["year"].values.astype(int)
    H = {int(y): elev["height"].sel(year=y).values.astype("float32") for y in years}
    RMSE = {
        int(y): elev["absolute_elevation_rmse"].sel(year=y).values.astype("float32") for y in years
    }

    sample_shape = next(iter(H.values())).shape
    n_intervals = len(model_t) - 1
    dhdt = np.full((n_intervals,) + sample_shape, np.nan, dtype="float32")
    unc = np.full_like(dhdt, np.nan)

    for i in range(n_intervals):
        y1 = snap_model_year_to_obs_year(model_t[i], years)
        y2 = snap_model_year_to_obs_year(model_t[i + 1], years)
        if y1 is None or y2 is None or y1 not in H or y2 not in H:
            continue
        dt_i = float(model_dt[i])
        dhdt[i] = (H[y2] - H[y1]) / dt_i
        unc[i] = np.sqrt(RMSE[y1] ** 2 + RMSE[y2] ** 2) / dt_i
    return dhdt, unc


def _obs_dvdt_on_intervals(
    vel: xr.Dataset, model_t: np.ndarray, model_dt: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    years = vel["year"].values.astype(int)
    VX = {int(y): vel["VX"].sel(year=y).values.astype("float32") for y in years}
    VY = {int(y): vel["VY"].sel(year=y).values.astype("float32") for y in years}
    EX = {int(y): vel["ERRX"].sel(year=y).values.astype("float32") for y in years}
    EY = {int(y): vel["ERRY"].sel(year=y).values.astype("float32") for y in years}

    sample_shape = next(iter(VX.values())).shape
    n_intervals = len(model_t) - 1
    dvxdt = np.full((n_intervals,) + sample_shape, np.nan, dtype="float32")
    dvydt = np.full_like(dvxdt, np.nan)
    uncx = np.full_like(dvxdt, np.nan)
    uncy = np.full_like(dvxdt, np.nan)

    for i in range(n_intervals):
        y1 = snap_model_year_to_obs_year(model_t[i], years)
        y2 = snap_model_year_to_obs_year(model_t[i + 1], years)
        if y1 is None or y2 is None:
            continue
        dt_i = float(model_dt[i])
        dvxdt[i] = (VX[y2] - VX[y1]) / dt_i
        dvydt[i] = (VY[y2] - VY[y1]) / dt_i
        uncx[i] = np.sqrt(EX[y1] ** 2 + EX[y2] ** 2) / dt_i
        uncy[i] = np.sqrt(EY[y1] ** 2 + EY[y2] ** 2) / dt_i
    return dvxdt, dvydt, uncx, uncy


def _mean_obs_speed(vel: xr.Dataset) -> np.ndarray:
    speeds = np.sqrt(vel["VX"].values ** 2 + vel["VY"].values ** 2)
    return np.nanmean(speeds, axis=0)


# ---------------------------------------------------------------------
# Flatten + mask
# ---------------------------------------------------------------------


def _flatten_and_mask_combined(
    obs_dhdt: np.ndarray,
    obs_dhdt_unc: np.ndarray,
    dhdt_models: list[np.ndarray],
    obs_dvxdt: np.ndarray,
    obs_dvydt: np.ndarray,
    obs_uncx: np.ndarray,
    obs_uncy: np.ndarray,
    dvxdt_models: list[np.ndarray],
    dvydt_models: list[np.ndarray],
    speed_mean: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int]:
    M = len(dhdt_models)

    # ---- thickness block ----
    y_t = obs_dhdt.reshape(-1)
    s_t = obs_dhdt_unc.reshape(-1)
    F_t = np.stack([m.reshape(-1) for m in dhdt_models], axis=0)

    mask_t = np.isfinite(y_t) & np.isfinite(s_t) & (s_t < THICK_UNC_THRESHOLD)
    for m in range(M):
        mask_t &= np.isfinite(F_t[m])
    y_t = y_t[mask_t]
    s_t = s_t[mask_t]
    F_t = F_t[:, mask_t]
    speed_t = np.zeros_like(y_t, dtype=float)
    n_dhdt = y_t.size

    # ---- velocity block (per interval, vx then vy) ----
    n_intervals = obs_dvxdt.shape[0]
    speed_flat = speed_mean.reshape(-1)
    y_v_list, s_v_list, F_v_list, sp_v_list = [], [], [], []
    for i in range(n_intervals):
        dvx_obs = obs_dvxdt[i].reshape(-1)
        dvy_obs = obs_dvydt[i].reshape(-1)
        dvx_err = obs_uncx[i].reshape(-1)
        dvy_err = obs_uncy[i].reshape(-1)
        y_int = np.concatenate([dvx_obs, dvy_obs])
        s_int = np.concatenate([dvx_err, dvy_err])
        speed_int = np.concatenate([speed_flat, speed_flat])
        F_int = np.zeros((M, y_int.size))
        for m in range(M):
            F_int[m] = np.concatenate(
                [dvxdt_models[m][i].reshape(-1), dvydt_models[m][i].reshape(-1)]
            )

        mask_v = (
            np.isfinite(y_int)
            & np.isfinite(s_int)
            & np.isfinite(speed_int)
            & (s_int < VEL_UNC_THRESHOLD)
        )
        for m in range(M):
            mask_v &= np.isfinite(F_int[m])

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
