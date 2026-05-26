#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DXB 03: ACF/Fit Manager v1.4 Package Editor
=======================================

目的
----
- 02_bin 以下の images_*.h5 を入力として ACF + Fit を実行する
- 結果は今後の解析拡張に合わせて 03_analysis/acf_fit/run_.../ 形式で保存する
- split は平均せず split01 / split02 ... として別々に保存する
- ACF Viewer / Pixel Inspector が追跡しやすいよう manifest を必ず保存する

想定入力H5
----------
/entry/data/images                              shape=(T,H,W)
/entry/instrument/detector/mask                 optional, 1=valid,0=invalid 推奨

出力構造
--------
<channel_root>/03_analysis/acf_fit/<run_id>/
├─ analysis_manifest.json
├─ recipe.json
├─ split_manifest.json
├─ outputs/
│  ├─ split01/
│  │  ├─ acf.h5
│  │  ├─ fit.csv
│  │  ├─ grid_meta.npz
│  │  └─ summary.json
│  └─ split02/
│     ├─ acf.h5
│     ├─ fit.csv
│     ├─ grid_meta.npz
│     └─ summary.json
└─ logs/
   └─ run.log

起動
----
streamlit run DXB_03_ACF_Fit_Manager_v1_4_package_editor.py
"""

from __future__ import annotations

import json
import math
import os
import re
import time
import traceback
import warnings
import multiprocessing as mp
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import h5py
import hdf5plugin  # noqa: F401  # required for bitshuffle/lz4 HDF5 filters
import numpy as np
import pandas as pd
import streamlit as st
from numba import get_num_threads, njit, prange, set_num_threads
from scipy.optimize import OptimizeWarning, curve_fit

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=OptimizeWarning)
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

# =============================================================================
# Config
# =============================================================================

APP_TITLE = "DXB 03 ACF/Fit Manager"
IMAGE_DATASET = "/entry/data/images"
MASK_DATASET = "/entry/instrument/detector/mask"
PACKAGES_FILENAME = "dxb_acf_fit_packages.json"


@dataclass
class AcfFitRecipe:
    package_name: str = "Package 1"
    display_label: str = ""
    analysis_type: str = "acf_fit"

    # ACF
    acf_method: str = "Symmetric"  # Normal / variance_normalized / Symmetric / Arai_legacy
    detrend_mode: str = "none"     # none / linear / quadratic
    n_splits: int = 2
    max_acf_time: int | None = None
    pixels_per_batch: int = 16000
    frame_dtype: str = "float32"
    numba_threads: int | None = None

    # Mask
    mask_input_semantics: str = "one_is_valid"  # one_is_valid / zero_is_valid
    use_all_pixels_if_no_mask: bool = True

    # Fit
    run_fit: bool = True
    fit_model: str = "stretched"
    max_lag_fit: int | None = None
    frame_time_sec: float | None = 0.5
    fit_processes: int = max(1, (os.cpu_count() or 2) - 1)
    use_weights: bool = True
    maxfev_fit: int = 20000

    # Fit bounds
    a_min: float = 0.0
    a_max: float = 1e6
    gamma_min: float = 1e-10
    gamma_max: float = 1e6
    g_min: float = 0.05
    g_max: float = 10.0

    # flags/selection
    r2_flag_low: float = 0.10
    select_r2_min: float = 0.10
    select_a_min: float = 0.0
    select_gamma_min: float = 0.0
    select_g_min: float = 0.0

    # output
    save_acf_h5: bool = True
    overwrite_existing_run: bool = False


@dataclass
class H5Info:
    path: str
    rel_path: str
    dataset_id: str
    data_label: str
    channel_root: str
    shape: tuple[int, int, int] | None = None
    dtype: str | None = None
    has_mask: bool = False
    bin_label: str = ""
    status: str = "unknown"
    error: str = ""


@dataclass
class QueueItem:
    input_h5: str
    rel_path: str
    dataset_id: str
    data_label: str
    channel_root: str
    package_name: str
    recipe: dict[str, Any]
    run_id: str
    out_dir: str
    status: str = "pending"


# =============================================================================
# Small utilities
# =============================================================================

def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def now_run_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def sanitize_token(s: str, max_len: int = 80) -> str:
    s = str(s or "").strip()
    s = re.sub(r"[^0-9A-Za-z_.\-\u3040-\u30ff\u3400-\u9fff]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return (s or "run")[:max_len]


def read_json(path: str | Path, default=None):
    p = Path(path)
    if not p.exists():
        return default
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return default


def write_json_atomic(path: str | Path, obj: Any):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, p)


def rel_to(path: str | Path, base: str | Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(Path(base).resolve()))
    except Exception:
        return str(path)


def seconds_to_hms(x: float | None) -> str:
    if x is None or not np.isfinite(x):
        return "--:--:--"
    x = max(0, int(x))
    h = x // 3600
    m = (x % 3600) // 60
    s = x % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


# =============================================================================
# H5 discovery
# =============================================================================

def infer_dataset_label(root: Path, h5_path: Path) -> tuple[str, str, Path]:
    """Return dataset_id, data_label, channel_root.

    channel_root is the directory that contains 01_h5 / 02_bin / 03_analysis.
    For .../<dataset>/<label>/02_bin/.../images.h5, returns:
      dataset=<dataset>, label=<label>, channel_root=.../<dataset>/<label>
    For .../<dataset>/02_bin/.../images.h5, returns label=default.
    """
    parts = list(h5_path.resolve().parts)
    try:
        idx = parts.index("02_bin")
    except ValueError:
        parent = h5_path.parent
        return parent.name, "default", parent

    channel_root = Path(*parts[:idx])
    try:
        rel_parts = list(channel_root.relative_to(root.resolve()).parts)
    except Exception:
        rel_parts = list(channel_root.parts[-2:])

    if len(rel_parts) >= 2:
        dataset_id = rel_parts[0]
        data_label = rel_parts[1]
    elif len(rel_parts) == 1:
        dataset_id = rel_parts[0]
        data_label = "default"
    else:
        dataset_id = channel_root.name
        data_label = "default"
    return dataset_id, data_label, channel_root


@st.cache_data(show_spinner=False)
def scan_binned_h5(analysis_root: str, refresh_token: int = 0) -> list[dict[str, Any]]:
    root = Path(analysis_root)
    if not root.exists():
        return []
    files = sorted(root.glob("**/02_bin/**/images*.h5"))
    out: list[dict[str, Any]] = []
    for p in files:
        if p.name.endswith(".tmp"):
            continue
        dataset_id, data_label, channel_root = infer_dataset_label(root, p)
        info = H5Info(
            path=str(p),
            rel_path=rel_to(p, root),
            dataset_id=dataset_id,
            data_label=data_label,
            channel_root=str(channel_root),
            bin_label="/".join(p.parent.relative_to(channel_root / "02_bin").parts) if (channel_root / "02_bin") in p.parents else p.parent.name,
        )
        try:
            with h5py.File(p, "r") as hf:
                ds = hf[IMAGE_DATASET]
                if ds.ndim != 3:
                    raise ValueError(f"{IMAGE_DATASET} is not 3D: {ds.shape}")
                info.shape = tuple(map(int, ds.shape))
                info.dtype = str(ds.dtype)
                info.has_mask = MASK_DATASET in hf
                info.status = "ready"
        except Exception as e:
            info.status = "error"
            info.error = str(e)
        out.append(asdict(info))
    return out


# =============================================================================
# Mask utilities
# =============================================================================

def normalize_mask(raw_mask: np.ndarray, semantics: str) -> np.ndarray:
    if semantics == "one_is_valid":
        return (raw_mask > 0).astype(np.uint8)
    if semantics == "zero_is_valid":
        return (raw_mask == 0).astype(np.uint8)
    raise ValueError("mask_input_semantics must be one_is_valid or zero_is_valid")


def read_images_and_mask(h5_path: str, recipe: AcfFitRecipe):
    with h5py.File(h5_path, "r") as hf:
        if IMAGE_DATASET not in hf:
            raise KeyError(f"{IMAGE_DATASET} not found: {h5_path}")
        dset = hf[IMAGE_DATASET]
        T, H, W = map(int, dset.shape)
        if MASK_DATASET in hf:
            raw_mask = hf[MASK_DATASET][:]
            if raw_mask.shape != (H, W):
                raise ValueError(f"mask shape {raw_mask.shape} != image shape {(H, W)}")
            mask = normalize_mask(raw_mask, recipe.mask_input_semantics)
            mask_source = MASK_DATASET
        else:
            if not recipe.use_all_pixels_if_no_mask:
                raise RuntimeError("mask not found and use_all_pixels_if_no_mask=False")
            mask = np.ones((H, W), dtype=np.uint8)
            mask_source = "generated_all_valid"
    return T, H, W, mask, mask_source


# =============================================================================
# ACF kernels
# =============================================================================

@njit(cache=True)
def detrend_none_1d(arr):
    out = np.empty_like(arr)
    for i in range(arr.shape[0]):
        out[i] = arr[i]
    return out


@njit(cache=True)
def detrend_linear_preserve_mean_1d(arr):
    N = arr.shape[0]
    out = np.empty_like(arr)
    if N <= 2:
        for i in range(N):
            out[i] = arr[i]
        return out
    S_t = S_y = S_tt = S_ty = 0.0
    for i in range(N):
        t = float(i)
        y = float(arr[i])
        S_t += t
        S_y += y
        S_tt += t * t
        S_ty += t * y
    denom = N * S_tt - S_t * S_t
    if denom == 0.0:
        for i in range(N):
            out[i] = arr[i]
        return out
    a = (N * S_ty - S_t * S_y) / denom
    b = (S_y - a * S_t) / N
    mean_trend = 0.0
    for i in range(N):
        mean_trend += a * i + b
    mean_trend /= N
    for i in range(N):
        out[i] = arr[i] - ((a * i + b) - mean_trend)
    return out


@njit(cache=True)
def solve_3x3(A, b):
    M = np.empty((3, 4), dtype=np.float64)
    for i in range(3):
        for j in range(3):
            M[i, j] = A[i, j]
        M[i, 3] = b[i]
    for col in range(3):
        pivot = col
        max_abs = abs(M[pivot, col])
        for r in range(col + 1, 3):
            v = abs(M[r, col])
            if v > max_abs:
                max_abs = v
                pivot = r
        if max_abs < 1e-15:
            return False, np.zeros(3, dtype=np.float64)
        if pivot != col:
            for c in range(col, 4):
                tmp = M[col, c]
                M[col, c] = M[pivot, c]
                M[pivot, c] = tmp
        piv = M[col, col]
        for c in range(col, 4):
            M[col, c] /= piv
        for r in range(3):
            if r == col:
                continue
            fac = M[r, col]
            for c in range(col, 4):
                M[r, c] -= fac * M[col, c]
    x = np.empty(3, dtype=np.float64)
    for i in range(3):
        x[i] = M[i, 3]
    return True, x


@njit(cache=True)
def detrend_quadratic_preserve_mean_1d(arr):
    N = arr.shape[0]
    out = np.empty_like(arr)
    if N <= 3:
        for i in range(N):
            out[i] = arr[i]
        return out
    S0 = float(N)
    S1 = S2 = S3 = S4 = 0.0
    T0 = T1 = T2 = 0.0
    for i in range(N):
        t = float(i)
        y = float(arr[i])
        t2 = t * t
        S1 += t
        S2 += t2
        S3 += t2 * t
        S4 += t2 * t2
        T0 += y
        T1 += t * y
        T2 += t2 * y
    A = np.empty((3, 3), dtype=np.float64)
    b = np.empty(3, dtype=np.float64)
    A[0, 0] = S0
    A[0, 1] = S1
    A[0, 2] = S2
    A[1, 0] = S1
    A[1, 1] = S2
    A[1, 2] = S3
    A[2, 0] = S2
    A[2, 1] = S3
    A[2, 2] = S4
    b[0] = T0
    b[1] = T1
    b[2] = T2
    ok, coef = solve_3x3(A, b)
    if not ok:
        for i in range(N):
            out[i] = arr[i]
        return out
    c0, c1, c2 = coef[0], coef[1], coef[2]
    mean_trend = 0.0
    for i in range(N):
        t = float(i)
        mean_trend += c0 + c1 * t + c2 * t * t
    mean_trend /= N
    for i in range(N):
        t = float(i)
        out[i] = arr[i] - ((c0 + c1 * t + c2 * t * t) - mean_trend)
    return out


@njit(cache=True)
def apply_detrend_1d(arr, detrend_id):
    if detrend_id == 0:
        return detrend_none_1d(arr)
    if detrend_id == 1:
        return detrend_linear_preserve_mean_1d(arr)
    return detrend_quadratic_preserve_mean_1d(arr)


@njit(parallel=True, cache=True)
def compute_acf_batch_exact(traces, ACF_Time, method_id, detrend_id):
    N_batch, T = traces.shape
    out = np.zeros((N_batch, ACF_Time), dtype=np.float32)
    for p in prange(N_batch):
        work = apply_detrend_1d(traces[p], detrend_id)
        mean_val = 0.0
        for i in range(T):
            mean_val += work[i]
        mean_val /= T
        if mean_val == 0.0:
            continue

        if method_id == 0:  # Normal
            denom = mean_val * mean_val
            for b in range(ACF_Time):
                upper = T - b
                if upper <= 0:
                    continue
                s = 0.0
                for i in range(upper):
                    s += work[i] * work[i + b]
                out[p, b] = (s / upper) / denom

        elif method_id == 1:  # variance_normalized
            var_val = 0.0
            for i in range(T):
                d = work[i] - mean_val
                var_val += d * d
            var_val /= T
            if var_val <= 0.0:
                continue
            for b in range(ACF_Time):
                upper = T - b
                if upper <= 0:
                    continue
                s = 0.0
                for i in range(upper):
                    s += (work[i] - mean_val) * (work[i + b] - mean_val)
                out[p, b] = s / (upper * var_val)

        elif method_id == 2:  # Symmetric
            for b in range(ACF_Time):
                upper = T - b
                if upper <= 0:
                    out[p, b] = 1.0
                    continue
                s_prod = s_f = s_b = 0.0
                for i in range(upper):
                    vf = work[i]
                    vb = work[i + b]
                    s_prod += vf * vb
                    s_f += vf
                    s_b += vb
                mf = s_f / upper
                mb = s_b / upper
                denom = mf * mb
                if abs(denom) > 1e-12:
                    out[p, b] = (s_prod / upper) / denom
                else:
                    out[p, b] = 1.0

        else:  # Arai_legacy
            var_val = 0.0
            for i in range(T):
                d = work[i] - mean_val
                var_val += d * d
            var_val /= T
            if var_val <= 0.0:
                continue
            std_val = np.sqrt(var_val)
            denom = mean_val * mean_val
            L2 = 2 * ACF_Time
            if L2 > T:
                L2 = T
            for b in range(ACF_Time):
                upper = L2 - b
                if upper <= 0:
                    out[p, b] = 1.0
                    continue
                s = 0.0
                for i in range(upper):
                    z0 = (work[i] - mean_val) / std_val
                    z1 = (work[i + b] - mean_val) / std_val
                    s += z0 * z1
                out[p, b] = (s / upper) / denom + 1.0
    return out


def method_to_id(name: str) -> int:
    table = {"Normal": 0, "variance_normalized": 1, "Symmetric": 2, "Arai_legacy": 3}
    if name not in table:
        raise ValueError(f"Unknown ACF method: {name}")
    return table[name]


def detrend_to_id(name: str) -> int:
    table = {"none": 0, "linear": 1, "quadratic": 2}
    if name not in table:
        raise ValueError(f"Unknown detrend mode: {name}")
    return table[name]


def warmup_numba(recipe: AcfFitRecipe):
    dummy = np.random.rand(8, 64).astype(np.float32)
    _ = compute_acf_batch_exact(dummy, 32, method_to_id(recipe.acf_method), detrend_to_id(recipe.detrend_mode))


def extract_traces(frames, x_coords, y_coords, start, end):
    xb = x_coords[start:end]
    yb = y_coords[start:end]
    traces = frames[:, yb, xb].T
    if traces.dtype != np.float32:
        traces = traces.astype(np.float32, copy=False)
    return np.ascontiguousarray(traces)


# =============================================================================
# Fit utilities
# =============================================================================

_FIT_CFG: dict[str, Any] = {}
_FIT_H5 = None
_FIT_ACF = None
_FIT_X = None
_FIT_Y = None
_FIT_TAU = None
_FIT_SIGMA = None


def model_stretched(tau, A, Gamma, g):
    return 1.0 + A * np.exp(-np.power(Gamma * tau, g))


def calc_r2(y_true, y_fit):
    resid = y_true - y_fit
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot == 0:
        return np.nan
    return 1.0 - ss_res / ss_tot


def estimate_initial_params(gdat, tau, cfg):
    amp0 = max(float(np.nanmax(gdat) - 1.0), 1e-6)
    target = 1.0 + amp0 / 2.0
    idx = np.where(gdat <= target)[0]
    if idx.size > 0:
        tau_half = float(tau[idx[0]])
        gamma0 = max(np.log(2.0) / max(tau_half, 1e-12), 1e-6)
    else:
        gamma0 = max(1.0 / max(float(tau[-1]), 1.0), 1e-6)
    return amp0, min(max(gamma0, cfg["gamma_min"] * 10), cfg["gamma_max"] / 10), 1.0


def fit_one_trace(gdat_full, tau, sigma, cfg):
    gdat = np.asarray(gdat_full[1:1 + tau.size], dtype=np.float64)  # skip lag0
    if gdat.size < 6:
        return [np.nan] * 7 + ["too_short", "too_short"]
    if not np.isfinite(gdat).all():
        return [np.nan] * 7 + ["invalid", "nonfinite_input"]
    if float(np.nanmax(gdat)) <= 1.0:
        return [np.nan] * 7 + ["flat", "flat_or_below_1"]

    pre_flags = []
    if gdat[0] <= 1.0:
        pre_flags.append("first_point_not_above_1")
    if np.nanmedian(gdat[:min(5, gdat.size)]) <= np.nanmedian(gdat[-min(5, gdat.size):]):
        pre_flags.append("no_clear_decay")

    try:
        p0 = estimate_initial_params(gdat, tau, cfg)
        popt, pcov = curve_fit(
            model_stretched,
            tau,
            gdat,
            p0=p0,
            sigma=sigma,
            absolute_sigma=False,
            bounds=([
                cfg["a_min"], cfg["gamma_min"], cfg["g_min"]
            ], [
                cfg["a_max"], cfg["gamma_max"], cfg["g_max"]
            ]),
            maxfev=int(cfg["maxfev_fit"]),
            method="trf",
        )
        y_fit = model_stretched(tau, *popt)
        r2 = calc_r2(gdat, y_fit)
        if pcov is not None and np.all(np.isfinite(pcov)):
            perr = np.sqrt(np.diag(pcov))
        else:
            perr = np.array([np.nan, np.nan, np.nan])
        A, Gamma, g = map(float, popt)
        A_se, Gamma_se, g_se = map(float, perr)
        flags = list(pre_flags)
        if not np.isfinite(r2) or r2 < cfg["r2_flag_low"]:
            flags.append("r2_low")
        if not np.isfinite(A) or A <= 1e-6:
            flags.append("A_small")
        if not np.isfinite(Gamma) or Gamma <= 0:
            flags.append("Gamma_nonpositive")
        if not np.isfinite(g):
            flags.append("g_nan")
        status = "ok_stretched" if not flags else "ok_stretched_flagged"
        return [A, A_se, Gamma, Gamma_se, g, g_se, r2, status, "|".join(flags)]
    except Exception as e:
        return [np.nan] * 7 + ["fit_fail", type(e).__name__]


def _fit_worker_init(acf_h5_path: str, tau: np.ndarray, sigma, cfg: dict[str, Any]):
    global _FIT_H5, _FIT_ACF, _FIT_X, _FIT_Y, _FIT_TAU, _FIT_SIGMA, _FIT_CFG
    _FIT_H5 = h5py.File(acf_h5_path, "r")
    _FIT_ACF = _FIT_H5["ACF"]
    _FIT_X = _FIT_H5["x_coords"][:]
    _FIT_Y = _FIT_H5["y_coords"][:]
    _FIT_TAU = tau
    _FIT_SIGMA = sigma
    _FIT_CFG = cfg


def _fit_one_index_from_h5(i: int):
    vals = fit_one_trace(_FIT_ACF[i, :], _FIT_TAU, _FIT_SIGMA, _FIT_CFG)
    return [int(_FIT_X[i]), int(_FIT_Y[i])] + vals


def build_fit_tau_sigma(n_lags_full: int, recipe: AcfFitRecipe):
    if recipe.max_lag_fit is None:
        n_lags = n_lags_full
    else:
        n_lags = int(min(n_lags_full, recipe.max_lag_fit))
    n_lags = max(6, n_lags)

    tau = np.arange(1, n_lags, dtype=np.float64)
    tau_unit = "frame"
    if recipe.frame_time_sec is not None and recipe.frame_time_sec > 0:
        tau = tau * float(recipe.frame_time_sec)
        tau_unit = "sec"

    sigma = None
    if recipe.use_weights:
        eff = (n_lags_full - np.arange(1, n_lags, dtype=np.float64)).astype(np.float64)
        eff[eff <= 0] = 1.0
        sigma = 1.0 / np.sqrt(eff)
    return tau, sigma, tau_unit


def recipe_fit_cfg(recipe: AcfFitRecipe) -> dict[str, Any]:
    return {
        "a_min": recipe.a_min,
        "a_max": recipe.a_max,
        "gamma_min": recipe.gamma_min,
        "gamma_max": recipe.gamma_max,
        "g_min": recipe.g_min,
        "g_max": recipe.g_max,
        "r2_flag_low": recipe.r2_flag_low,
        "maxfev_fit": recipe.maxfev_fit,
        "select_r2_min": recipe.select_r2_min,
        "select_a_min": recipe.select_a_min,
        "select_gamma_min": recipe.select_gamma_min,
        "select_g_min": recipe.select_g_min,
    }


def save_fit_outputs(df: pd.DataFrame, acf_reader, out_dir: Path, tau_unit: str, recipe: AcfFitRecipe):
    df["fit_type"] = recipe.fit_model
    df["Gamma_unit"] = f"{tau_unit}^-1"
    df[f"Gamma({tau_unit}^-1)"] = df["Gamma"]
    df[f"log10_Decay constant [{tau_unit}-1]"] = np.log10(
        pd.to_numeric(df["Gamma"], errors="coerce").clip(lower=1e-30)
    )
    df[f"tau({tau_unit})"] = 1.0 / pd.to_numeric(df["Gamma"], errors="coerce")
    df[f"tau({tau_unit})"] = df[f"tau({tau_unit})"].replace([np.inf, -np.inf], np.nan)
    df[f"log10_tau({tau_unit})"] = np.log10(df[f"tau({tau_unit})"].clip(lower=1e-30))

    if tau_unit == "sec":
        df["Gamma [s^-1]"] = df["Gamma"]
        df["log10 Gamma [s^-1]"] = np.log10(pd.to_numeric(df["Gamma"], errors="coerce").clip(lower=1e-30))
        df["tau [s]"] = 1.0 / pd.to_numeric(df["Gamma"], errors="coerce")
        df["tau [s]"] = df["tau [s]"].replace([np.inf, -np.inf], np.nan)
        df["log10 tau [s]"] = np.log10(df["tau [s]"].clip(lower=1e-30))
    else:
        df["Gamma [frame^-1]"] = df["Gamma"]
        df["log10 Gamma [frame^-1]"] = np.log10(pd.to_numeric(df["Gamma"], errors="coerce").clip(lower=1e-30))
        df["tau [frame]"] = 1.0 / pd.to_numeric(df["Gamma"], errors="coerce")
        df["tau [frame]"] = df["tau [frame]"].replace([np.inf, -np.inf], np.nan)
        df["log10 tau [frame]"] = np.log10(df["tau [frame]"].clip(lower=1e-30))

    tmp_csv = out_dir / "fit.csv.tmp"
    final_csv = out_dir / "fit.csv"
    df.to_csv(tmp_csv, index=False, encoding="utf-8-sig")
    os.replace(tmp_csv, final_csv)

    selected = df[
        (df["A"] > recipe.select_a_min)
        & (df["Gamma"] > recipe.select_gamma_min)
        & (df["g"] > recipe.select_g_min)
        & (df["R2"] >= recipe.select_r2_min)
    ]
    if len(selected) > 0:
        idx = selected.index.to_numpy(dtype=int)
        idx_sorted = np.sort(idx)
        sel_acf_sorted = np.asarray(acf_reader[idx_sorted, :], dtype=np.float64)
        pos = {int(v): j for j, v in enumerate(idx_sorted)}
        sel_acf = np.asarray([sel_acf_sorted[pos[int(v)]] for v in idx], dtype=np.float64)
        lag_index = np.arange(sel_acf.shape[1], dtype=int)
        tau_full = lag_index.astype(float)
        if recipe.frame_time_sec is not None and recipe.frame_time_sec > 0:
            tau_full = tau_full * float(recipe.frame_time_sec)
        stats = pd.DataFrame({
            "lag_index": lag_index,
            "tau": tau_full,
            "mean_acf": np.mean(sel_acf, axis=0),
            "std_acf": np.std(sel_acf, axis=0),
            "median_acf": np.median(sel_acf, axis=0),
            "n_selected": np.full(sel_acf.shape[1], len(idx), dtype=int),
        })
    else:
        stats = pd.DataFrame(columns=["lag_index", "tau", "mean_acf", "std_acf", "median_acf", "n_selected"])
    tmp_stats = out_dir / "selected_acf_stats.csv.tmp"
    final_stats = out_dir / "selected_acf_stats.csv"
    stats.to_csv(tmp_stats, index=False, encoding="utf-8-sig")
    os.replace(tmp_stats, final_stats)
    return final_csv


def fit_acf_h5(acf_h5_path: Path, out_dir: Path, recipe: AcfFitRecipe, progress_cb=None):
    with h5py.File(acf_h5_path, "r") as hf:
        n_pixels = int(hf["ACF"].shape[0])
        n_lags_full = int(hf["ACF"].shape[1])
    tau, sigma, tau_unit = build_fit_tau_sigma(n_lags_full, recipe)
    cfg = recipe_fit_cfg(recipe)

    columns = ["x", "y", "A", "A_se", "Gamma", "Gamma_se", "g", "g_se", "R2", "status", "flags"]
    rows = []
    t0 = time.time()

    if recipe.fit_processes <= 1:
        with h5py.File(acf_h5_path, "r") as hf:
            acf = hf["ACF"]
            x = hf["x_coords"][:]
            y = hf["y_coords"][:]
            for i in range(n_pixels):
                vals = fit_one_trace(acf[i, :], tau, sigma, cfg)
                rows.append([int(x[i]), int(y[i])] + vals)
                if progress_cb and (i == 0 or (i + 1) % 500 == 0 or i + 1 == n_pixels):
                    elapsed = time.time() - t0
                    fps = (i + 1) / max(elapsed, 1e-9)
                    progress_cb("fit", i + 1, n_pixels, f"Fit {i+1}/{n_pixels}", fps)
            df = pd.DataFrame(rows, columns=columns)
            return save_fit_outputs(df, acf, out_dir, tau_unit, recipe)

    ctx = mp.get_context("spawn") if os.name == "nt" else mp.get_context()
    with ctx.Pool(
        processes=int(recipe.fit_processes),
        initializer=_fit_worker_init,
        initargs=(str(acf_h5_path), tau, sigma, cfg),
    ) as pool:
        for i, r in enumerate(pool.imap(_fit_one_index_from_h5, range(n_pixels), chunksize=1000), start=1):
            rows.append(r)
            if progress_cb and (i == 1 or i % 1000 == 0 or i == n_pixels):
                elapsed = time.time() - t0
                fps = i / max(elapsed, 1e-9)
                progress_cb("fit", i, n_pixels, f"Fit {i}/{n_pixels}", fps)

    df = pd.DataFrame(rows, columns=columns)
    with h5py.File(acf_h5_path, "r") as hf:
        return save_fit_outputs(df, hf["ACF"], out_dir, tau_unit, recipe)


# =============================================================================
# ACF/Fit processing
# =============================================================================

def log_line(log_path: Path, msg: str):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"[{now_iso()}] {msg}\n")


def make_progress_callback(status_box, progress_bar, log_area, log_messages: list[str], log_path: Path,
                           job_context: dict[str, Any] | None = None):
    """Render a compact progress dashboard instead of a raw log stream."""
    start_time = time.time()
    state = {
        "last_ui": 0.0,
        "last_log": 0.0,
        "last_stage": None,
        "stage_start": time.time(),
    }
    ctx = job_context or {}

    stage_names = {
        "read": "Read input",
        "read_split": "Read split frames",
        "warmup": "Numba warmup",
        "split": "Split setup",
        "acf": "ACF calculation",
        "write_acf": "Write ACF H5",
        "fit": "Fit",
        "done": "Done",
        "failed": "Failed",
    }

    def cb(stage: str, current: int, total: int, message: str = "", rate: float | None = None):
        now = time.time()
        if stage != state["last_stage"]:
            state["last_stage"] = stage
            state["stage_start"] = now

        # Keep Streamlit responsive; do not rerender for every pixel batch.
        if now - state["last_ui"] < 0.35 and current < total:
            return
        state["last_ui"] = now

        total_safe = max(int(total), 0)
        current_safe = max(int(current), 0)
        frac = 0.0 if total_safe <= 0 else min(1.0, max(0.0, current_safe / total_safe))
        elapsed_job = now - start_time
        elapsed_stage = now - state["stage_start"]
        eta = None
        if current_safe > 0 and total_safe > current_safe:
            eta = elapsed_stage * (total_safe / current_safe - 1.0)

        job_idx = ctx.get("job_index")
        job_total = ctx.get("job_total")
        job_prefix = f"Job {job_idx}/{job_total}" if job_idx and job_total else "Job"
        dataset = ctx.get("dataset_id", "")
        label = ctx.get("data_label", "")
        rel_path = ctx.get("rel_path", "")
        stage_label = stage_names.get(stage, stage)
        rate_txt = "-" if rate is None else f"{rate:,.1f}/s"
        eta_txt = seconds_to_hms(eta)
        updated_txt = datetime.now().strftime("%H:%M:%S")
        msg = str(message or "")

        status_box.markdown(
            f"""
<div style="border:1px solid #e5e7eb;border-radius:14px;padding:14px 16px;background:#fbfbfd;margin-bottom:8px;">
  <div style="display:flex;justify-content:space-between;gap:12px;align-items:flex-start;">
    <div>
      <div style="font-size:0.86rem;color:#6b7280;">{job_prefix}</div>
      <div style="font-size:1.05rem;font-weight:700;color:#111827;">{dataset} / {label}</div>
      <div style="font-size:0.80rem;color:#6b7280;word-break:break-all;">{rel_path}</div>
    </div>
    <div style="text-align:right;min-width:160px;">
      <div style="font-size:0.86rem;color:#6b7280;">current stage</div>
      <div style="font-size:1.05rem;font-weight:700;color:#2563eb;">{stage_label}</div>
    </div>
  </div>
  <hr style="border:none;border-top:1px solid #e5e7eb;margin:10px 0;"/>
  <div style="display:grid;grid-template-columns:repeat(6,minmax(0,1fr));gap:10px;">
    <div><div style="font-size:0.75rem;color:#6b7280;">stage progress</div><div style="font-weight:700;">{current_safe:,}/{total_safe:,}</div></div>
    <div><div style="font-size:0.75rem;color:#6b7280;">percent</div><div style="font-weight:700;">{frac*100:.1f}%</div></div>
    <div><div style="font-size:0.75rem;color:#6b7280;">speed</div><div style="font-weight:700;">{rate_txt}</div></div>
    <div><div style="font-size:0.75rem;color:#6b7280;">stage ETA</div><div style="font-weight:700;">{eta_txt}</div></div>
    <div><div style="font-size:0.75rem;color:#6b7280;">job elapsed</div><div style="font-weight:700;">{seconds_to_hms(elapsed_job)}</div></div>
    <div><div style="font-size:0.75rem;color:#6b7280;">last update</div><div style="font-weight:700;">{updated_txt}</div></div>
  </div>
  <div style="margin-top:10px;font-size:0.90rem;color:#374151;word-break:break-all;">{msg}</div>
</div>
""",
            unsafe_allow_html=True,
        )
        progress_bar.progress(frac)

        # Keep the visible log readable by sampling ACF/Fit progress.
        should_log = bool(message) and (
            stage not in {"acf", "fit"}
            or current_safe >= total_safe
            or now - state["last_log"] >= 8.0
        )
        if should_log:
            state["last_log"] = now
            line = f"{datetime.now().strftime('%H:%M:%S')} | {stage_label} | {current_safe:,}/{total_safe:,} | {frac*100:.1f}% | {msg}"
            if not log_messages or log_messages[-1] != line:
                log_messages.append(line)
                log_messages[:] = log_messages[-120:]
                log_area.code("\n".join(log_messages[-60:]), language="text")
                log_line(log_path, line)
    return cb


def write_acf_h5_atomic(out_path: Path, acf: np.ndarray, x_coords: np.ndarray, y_coords: np.ndarray,
                        mask: np.ndarray, attrs: dict[str, Any]):
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()
    with h5py.File(tmp, "w", libver="latest") as hf:
        hf.create_dataset("ACF", data=acf, dtype=np.float32, chunks=(min(4096, acf.shape[0]), acf.shape[1]))
        hf.create_dataset("x_coords", data=x_coords, dtype=np.int32)
        hf.create_dataset("y_coords", data=y_coords, dtype=np.int32)
        hf.create_dataset("grid_mask", data=mask.astype(np.uint8), dtype=np.uint8)
        for k, v in attrs.items():
            if v is None:
                hf.attrs[k] = "None"
            else:
                hf.attrs[k] = v
    os.replace(tmp, out_path)


def save_grid_meta(out_dir: Path, mask: np.ndarray, H: int, W: int, source_h5: str):
    tmp = out_dir / "grid_meta.npz.tmp"
    final = out_dir / "grid_meta.npz"
    np.savez_compressed(
        tmp,
        mask=mask.astype(np.uint8),
        height=np.array([H], dtype=np.int32),
        width=np.array([W], dtype=np.int32),
        mask_semantics=np.array(["1=valid,0=invalid"]),
        source_h5=np.array([source_h5]),
    )
    # np.savez appends .npz if suffix does not end with npz
    actual_tmp = Path(str(tmp) + ".npz") if not tmp.exists() else tmp
    os.replace(actual_tmp, final)
    return final


def save_summary(out_dir: Path, summary: dict[str, Any]):
    write_json_atomic(out_dir / "summary.json", summary)


def compute_split_acf(h5_path: str, out_dir: Path, split_name: str, split_index: int, frame_start: int, frame_end: int,
                      mask: np.ndarray, x_coords: np.ndarray, y_coords: np.ndarray, recipe: AcfFitRecipe,
                      input_info: dict[str, Any], progress_cb=None):
    frame_dtype = np.float32 if recipe.frame_dtype == "float32" else np.float64
    method_id = method_to_id(recipe.acf_method)
    detrend_id = detrend_to_id(recipe.detrend_mode)

    if progress_cb:
        progress_cb("read_split", 0, 1, f"{split_name}: reading frames {frame_start}-{frame_end-1}")
    with h5py.File(h5_path, "r") as hf:
        dset = hf[IMAGE_DATASET]
        frames = dset[frame_start:frame_end].astype(frame_dtype, copy=False)
        frames = np.ascontiguousarray(frames)
    if progress_cb:
        progress_cb("read_split", 1, 1, f"{split_name}: loaded {frame_end-frame_start} frames")

    T_seg = int(frames.shape[0])
    acf_time = T_seg // 2
    if recipe.max_acf_time is not None:
        acf_time = min(acf_time, int(recipe.max_acf_time))
    if acf_time < 2:
        raise RuntimeError(f"ACF_Time too short: {acf_time}")

    n_pixels = int(x_coords.size)
    acf = np.zeros((n_pixels, acf_time), dtype=np.float32)
    t0 = time.time()
    batch = max(1, int(recipe.pixels_per_batch))

    for p0 in range(0, n_pixels, batch):
        p1 = min(p0 + batch, n_pixels)
        traces = extract_traces(frames, x_coords, y_coords, p0, p1)
        acf[p0:p1] = compute_acf_batch_exact(traces, acf_time, method_id, detrend_id)
        if progress_cb:
            elapsed = time.time() - t0
            rate = p1 / max(elapsed, 1e-9)
            progress_cb("acf", p1, n_pixels, f"{split_name}: pixels {p1}/{n_pixels}", rate)

    attrs = {
        "analysis_type": "acf_fit",
        "source_images_h5": str(h5_path),
        "split_name": split_name,
        "split_index": int(split_index),
        "frame_start": int(frame_start),
        "frame_end": int(frame_end),
        "frame_range_semantics": "[frame_start, frame_end)",
        "frame_time_sec": recipe.frame_time_sec if recipe.frame_time_sec is not None else "None",
        "acf_method": recipe.acf_method,
        "detrend_mode": recipe.detrend_mode,
        "fit_model": recipe.fit_model,
        "grid_height": int(input_info["H"]),
        "grid_width": int(input_info["W"]),
        "mask_semantics": "1=valid,0=invalid",
    }
    acf_path = out_dir / "acf.h5"
    if progress_cb:
        progress_cb("write_acf", 0, 1, f"{split_name}: writing acf.h5")
    write_acf_h5_atomic(acf_path, acf, x_coords, y_coords, mask, attrs)
    if progress_cb:
        progress_cb("write_acf", 1, 1, f"{split_name}: acf.h5 saved")
    del frames
    return acf_path, acf_time, T_seg


def run_acf_fit_job(item: QueueItem, status_box, progress_bar, log_area, job_context: dict[str, Any] | None = None) -> dict[str, Any]:
    recipe = AcfFitRecipe(**item.recipe)
    input_h5 = item.input_h5
    out_dir = Path(item.out_dir)
    outputs_dir = out_dir / "outputs"
    logs_dir = out_dir / "logs"
    log_path = logs_dir / "run.log"
    running_marker = out_dir / ".running.json"
    failed_marker = out_dir / ".failed.json"

    if failed_marker.exists():
        failed_marker.unlink()
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(running_marker, {
        "status": "running",
        "started_at": now_iso(),
        "input_h5": input_h5,
        "out_dir": str(out_dir),
        "run_id": item.run_id,
    })

    log_messages: list[str] = []
    progress_cb = make_progress_callback(status_box, progress_bar, log_area, log_messages, log_path, job_context=job_context)
    t_job = time.time()

    try:
        log_line(log_path, f"Start job: {item.run_id}")
        write_json_atomic(out_dir / "recipe.json", asdict(recipe))

        progress_cb("read", 0, 1, "Reading input H5 metadata and mask")
        T, H, W, mask, mask_source = read_images_and_mask(input_h5, recipe)
        progress_cb("read", 1, 1, f"Loaded shape=({T},{H},{W}), mask={mask_source}")
        input_info = {"T": T, "H": H, "W": W, "mask_source": mask_source}
        y_coords, x_coords = np.where(mask > 0)
        x_coords = x_coords.astype(np.int32, copy=False)
        y_coords = y_coords.astype(np.int32, copy=False)
        n_pixels = int(x_coords.size)
        if n_pixels == 0:
            raise RuntimeError("No valid pixels after mask normalization")

        if recipe.numba_threads is not None and recipe.numba_threads > 0:
            set_num_threads(int(recipe.numba_threads))
        progress_cb("warmup", 0, 1, f"Numba threads: {get_num_threads()}")
        warmup_numba(recipe)
        progress_cb("warmup", 1, 1, "Numba warmup done")

        frames_per_split = T // int(recipe.n_splits)
        extra = T % int(recipe.n_splits)
        split_records = []
        split_start = 0

        for si in range(int(recipe.n_splits)):
            split_end = split_start + frames_per_split + (1 if si < extra else 0)
            if split_end <= split_start:
                continue
            split_name = f"split{si + 1:02d}"
            split_dir = outputs_dir / split_name
            split_dir.mkdir(parents=True, exist_ok=True)
            write_json_atomic(split_dir / ".running.json", {
                "status": "running",
                "split_name": split_name,
                "frame_start": int(split_start),
                "frame_end": int(split_end),
                "started_at": now_iso(),
            })
            progress_cb("split", si + 1, int(recipe.n_splits), f"Start {split_name}: frames {split_start}-{split_end-1}")
            grid_meta = save_grid_meta(split_dir, mask, H, W, input_h5)
            acf_path, acf_time, T_seg = compute_split_acf(
                input_h5, split_dir, split_name, si + 1, split_start, split_end,
                mask, x_coords, y_coords, recipe, input_info, progress_cb=progress_cb,
            )

            fit_csv = None
            if recipe.run_fit:
                fit_csv = fit_acf_h5(acf_path, split_dir, recipe, progress_cb=progress_cb)
                df_fit = pd.read_csv(fit_csv)
                status_counts = df_fit["status"].value_counts(dropna=False).to_dict() if "status" in df_fit else {}
                flag_counter = Counter()
                if "flags" in df_fit:
                    for s in df_fit["flags"].fillna(""):
                        for token in str(s).split("|"):
                            token = token.strip()
                            if token:
                                flag_counter[token] += 1
            else:
                status_counts = {}
                flag_counter = Counter()

            summary = {
                "status": "done",
                "split_name": split_name,
                "source_images_h5": str(input_h5),
                "shape": {"T_total": int(T), "H": int(H), "W": int(W), "T_seg": int(T_seg)},
                "frame_start": int(split_start),
                "frame_end": int(split_end),
                "ACF_Time": int(acf_time),
                "valid_pixels": int(n_pixels),
                "mask_source": mask_source,
                "mask_semantics": "1=valid,0=invalid",
                "acf_method": recipe.acf_method,
                "detrend_mode": recipe.detrend_mode,
                "fit_model": recipe.fit_model,
                "run_fit": bool(recipe.run_fit),
                "status_counts": status_counts,
                "flag_counts": dict(flag_counter.most_common()),
                "created_at": now_iso(),
            }
            save_summary(split_dir, summary)
            running_split = split_dir / ".running.json"
            if running_split.exists():
                running_split.unlink()

            split_records.append({
                "split_name": split_name,
                "split_index": si + 1,
                "frame_start": int(split_start),
                "frame_end": int(split_end),
                "frame_range_semantics": "[frame_start, frame_end)",
                "T_seg": int(T_seg),
                "ACF_Time": int(acf_time),
                "acf_h5": str(Path("outputs") / split_name / "acf.h5"),
                "fit_csv": str(Path("outputs") / split_name / "fit.csv") if fit_csv else None,
                "grid_meta": str(Path("outputs") / split_name / "grid_meta.npz"),
                "summary": str(Path("outputs") / split_name / "summary.json"),
            })
            split_start = split_end

        split_manifest = {
            "status": "done",
            "created_at": now_iso(),
            "n_splits": int(recipe.n_splits),
            "source_images_h5": str(input_h5),
            "frame_time_sec": recipe.frame_time_sec,
            "splits": split_records,
        }
        write_json_atomic(out_dir / "split_manifest.json", split_manifest)

        analysis_manifest = {
            "analysis_type": "acf_fit",
            "run_id": item.run_id,
            "status": "done",
            "created_at": now_iso(),
            "dataset_id": item.dataset_id,
            "data_label": item.data_label,
            "input_h5": str(input_h5),
            "input_h5_rel_to_run": rel_to(input_h5, out_dir),
            "input_level": "bin",
            "input_shape": [int(T), int(H), int(W)],
            "valid_pixels": int(n_pixels),
            "recipe": "recipe.json",
            "split_manifest": "split_manifest.json",
            "outputs_dir": "outputs",
            "logs_dir": "logs",
            "package_name": recipe.package_name,
            "acf_method": recipe.acf_method,
            "detrend_mode": recipe.detrend_mode,
            "fit_model": recipe.fit_model,
            "n_splits": int(recipe.n_splits),
            "frame_time_sec": recipe.frame_time_sec,
            "mask_semantics": "1=valid,0=invalid",
            "elapsed_sec": float(time.time() - t_job),
        }
        write_json_atomic(out_dir / "analysis_manifest.json", analysis_manifest)

        # Update channel catalog lightly
        catalog_path = Path(item.channel_root) / "03_analysis" / "analysis_catalog.json"
        catalog = read_json(catalog_path, default=[])
        if not isinstance(catalog, list):
            catalog = []
        catalog = [r for r in catalog if r.get("run_dir") != str(out_dir)]
        catalog.append({
            "analysis_type": "acf_fit",
            "run_id": item.run_id,
            "status": "done",
            "input_h5": str(input_h5),
            "run_dir": str(out_dir),
            "manifest": str(out_dir / "analysis_manifest.json"),
            "created_at": now_iso(),
        })
        write_json_atomic(catalog_path, catalog)

        if running_marker.exists():
            running_marker.unlink()
        progress_cb("done", 1, 1, f"Done: {item.run_id}")
        return {"status": "done", "run_id": item.run_id, "out_dir": str(out_dir)}

    except Exception as e:
        tb = traceback.format_exc()
        write_json_atomic(failed_marker, {
            "status": "failed",
            "failed_at": now_iso(),
            "error": str(e),
            "traceback": tb,
            "input_h5": input_h5,
            "run_id": item.run_id,
        })
        if running_marker.exists():
            running_marker.unlink()
        log_line(log_path, "FAILED: " + str(e))
        log_line(log_path, tb)
        status_box.error(f"Failed: {e}")
        return {"status": "failed", "run_id": item.run_id, "out_dir": str(out_dir), "error": str(e)}


# =============================================================================
# UI recipes / state
# =============================================================================


def recipe_from_dict(obj: dict[str, Any] | AcfFitRecipe) -> AcfFitRecipe:
    """Build AcfFitRecipe while tolerating older/extra JSON keys."""
    if isinstance(obj, AcfFitRecipe):
        return obj
    obj = dict(obj or {})
    allowed = set(AcfFitRecipe.__dataclass_fields__.keys())
    clean = {k: v for k, v in obj.items() if k in allowed}
    return AcfFitRecipe(**clean)


def package_display_name(recipe: AcfFitRecipe) -> str:
    label = str(getattr(recipe, "display_label", "") or "").strip()
    return f"{recipe.package_name} - {label}" if label else recipe.package_name


def packages_path(analysis_root: str | Path) -> Path:
    return Path(analysis_root) / PACKAGES_FILENAME


def load_project_packages(analysis_root: str | Path) -> dict[str, AcfFitRecipe]:
    """Load editable package definitions from AnalysisRoot.

    If the project file does not exist, defaults are returned. The file is intentionally
    project-local so different PCs/projects can keep different defaults.
    """
    defaults = default_packages()
    p = packages_path(analysis_root)
    obj = read_json(p, default=None)
    if not obj:
        return defaults
    raw_list = obj.get("packages", []) if isinstance(obj, dict) else []
    out: dict[str, AcfFitRecipe] = {}
    for raw in raw_list:
        try:
            rec = recipe_from_dict(raw)
            if not rec.package_name:
                continue
            out[rec.package_name] = rec
        except Exception:
            continue
    return out or defaults


def save_project_packages(analysis_root: str | Path, packages: dict[str, AcfFitRecipe]) -> Path:
    p = packages_path(analysis_root)
    data = {
        "schema": "dxb_acf_fit_packages.v1",
        "updated_at": now_iso(),
        "packages": [asdict(packages[k]) for k in sorted(packages.keys())],
    }
    write_json_atomic(p, data)
    return p

def default_packages() -> dict[str, AcfFitRecipe]:
    cpu = os.cpu_count() or 2
    return {
        "Package 1": AcfFitRecipe(
            package_name="Package 1",
            display_label="Standard",
            acf_method="Symmetric",
            detrend_mode="none",
            n_splits=2,
            max_acf_time=None,
            frame_time_sec=0.5,
            run_fit=True,
            fit_processes=max(1, cpu - 1),
            pixels_per_batch=16000,
        ),
        "Package 2": AcfFitRecipe(
            package_name="Package 2",
            display_label="Quick",
            acf_method="Symmetric",
            detrend_mode="none",
            n_splits=1,
            max_acf_time=200,
            frame_time_sec=0.5,
            run_fit=True,
            fit_processes=max(1, cpu - 1),
            pixels_per_batch=16000,
        ),
        "Package 3": AcfFitRecipe(
            package_name="Package 3",
            display_label="ACF only",
            acf_method="Symmetric",
            detrend_mode="none",
            n_splits=2,
            max_acf_time=None,
            frame_time_sec=0.5,
            run_fit=False,
            fit_processes=1,
            pixels_per_batch=16000,
        ),
    }


def init_state():
    st.session_state.setdefault("acf_scan_token", 0)
    st.session_state.setdefault("acf_selected_h5", [])
    st.session_state.setdefault("acf_recipe", asdict(default_packages()["Package 1"]))
    st.session_state.setdefault("acf_queue", [])
    st.session_state.setdefault("acf_is_running", False)
    st.session_state.setdefault("acf_last_results", [])


def make_run_id(input_h5: str, recipe: AcfFitRecipe) -> str:
    stem = sanitize_token(Path(input_h5).stem)
    pkg = sanitize_token(recipe.package_name)
    return f"run_{now_run_stamp()}_{stem}_{pkg}"


def output_dir_for(input_h5: str, channel_root: str, run_id: str) -> Path:
    return Path(channel_root) / "03_analysis" / "acf_fit" / run_id


def build_queue_items(selected_infos: list[dict[str, Any]], recipe_dict: dict[str, Any], overwrite: bool) -> list[dict[str, Any]]:
    recipe = recipe_from_dict(recipe_dict)
    items = []
    for info in selected_infos:
        run_id = make_run_id(info["path"], recipe)
        out_dir = output_dir_for(info["path"], info["channel_root"], run_id)
        if out_dir.exists() and not overwrite:
            # unique run_id by suffix if same second collision
            run_id = f"{run_id}_{int(time.time() * 1000) % 100000}"
            out_dir = output_dir_for(info["path"], info["channel_root"], run_id)
        item = QueueItem(
            input_h5=info["path"],
            rel_path=info["rel_path"],
            dataset_id=info["dataset_id"],
            data_label=info["data_label"],
            channel_root=info["channel_root"],
            package_name=recipe.package_name,
            recipe=asdict(recipe),
            run_id=run_id,
            out_dir=str(out_dir),
        )
        items.append(asdict(item))
    return items




# =============================================================================
# UI organization helpers
# =============================================================================

def is_1x1_bin_label(bin_label: str) -> bool:
    s = str(bin_label or "").replace("\\", "/").lower().strip()
    return s.startswith("1x1/t1") or s == "1x1" or s.startswith("1/t1")


def bin_sort_key(label: str):
    s = str(label or "")
    m = re.search(r"(\d+)x\d+", s)
    b = int(m.group(1)) if m else 10**9
    mt = re.search(r"/t(\d+)", s.replace("\\", "/"))
    t = int(mt.group(1)) if mt else 10**9
    return (b, t, s)


def fmt_shape(shape) -> str:
    if shape is None:
        return ""
    try:
        return "×".join(str(int(x)) for x in shape)
    except Exception:
        return str(shape)


@st.cache_data(show_spinner=False)
def scan_acf_run_manifests(analysis_root: str, refresh_token: int = 0) -> list[dict[str, Any]]:
    root = Path(analysis_root)
    rows: list[dict[str, Any]] = []
    for m in sorted(root.glob("**/03_analysis/acf_fit/run_*/analysis_manifest.json")):
        obj = read_json(m, default={}) or {}
        inp = obj.get("input_h5", "")
        try:
            inp_key = str(Path(inp).resolve()) if inp else ""
        except Exception:
            inp_key = str(inp)
        rows.append({
            "dataset_id": obj.get("dataset_id", ""),
            "data_label": obj.get("data_label", ""),
            "run_id": obj.get("run_id", m.parent.name),
            "status": obj.get("status", ""),
            "package_name": obj.get("package_name", ""),
            "input_h5": str(inp),
            "input_key": inp_key,
            "input_shape": obj.get("input_shape"),
            "created_at": obj.get("created_at", ""),
            "manifest": str(m),
        })
    return rows


def build_run_summary_by_input(run_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    d: dict[str, dict[str, Any]] = {}
    for r in run_rows:
        key = r.get("input_key") or r.get("input_h5") or ""
        if not key:
            continue
        cur = d.setdefault(key, {"count": 0, "done": 0, "failed": 0, "packages": set(), "latest": ""})
        cur["count"] += 1
        if r.get("status") == "done":
            cur["done"] += 1
        if r.get("status") == "failed":
            cur["failed"] += 1
        if r.get("package_name"):
            cur["packages"].add(str(r.get("package_name")))
        if str(r.get("created_at", "")) > str(cur.get("latest", "")):
            cur["latest"] = r.get("created_at", "")
    for cur in d.values():
        cur["packages"] = ", ".join(sorted(cur["packages"])) if cur["packages"] else ""
    return d


def h5_key(path: str) -> str:
    try:
        return str(Path(path).resolve())
    except Exception:
        return str(path)


def add_existing_run_columns(infos: list[dict[str, Any]], run_summary: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for x in infos:
        y = dict(x)
        s = run_summary.get(h5_key(y.get("path", "")), {})
        y["existing_runs"] = int(s.get("count", 0) or 0)
        y["existing_done"] = int(s.get("done", 0) or 0)
        y["existing_failed"] = int(s.get("failed", 0) or 0)
        y["existing_packages"] = s.get("packages", "")
        y["shape_text"] = fmt_shape(y.get("shape"))
        out.append(y)
    return out


def make_bin_matrix(infos: list[dict[str, Any]], include_1x1: bool = False) -> pd.DataFrame:
    ready = [x for x in infos if x.get("status") == "ready"]
    if not include_1x1:
        ready = [x for x in ready if not is_1x1_bin_label(x.get("bin_label", ""))]
    if not ready:
        return pd.DataFrame()
    bins = sorted({x.get("bin_label", "") for x in ready}, key=bin_sort_key)
    rows = []
    for (ds, lab), grp in sorted(pd.DataFrame(ready).groupby(["dataset_id", "data_label"]), key=lambda kv: (str(kv[0][0]), str(kv[0][1]))):
        rec = {"dataset_id": ds, "data_label": lab}
        have = set(grp["bin_label"].astype(str).tolist())
        for b in bins:
            rec[b] = "✓" if b in have else ""
        rows.append(rec)
    return pd.DataFrame(rows)


def filter_h5_infos(
    infos: list[dict[str, Any]],
    datasets: list[str] | None = None,
    labels: list[str] | None = None,
    bins: list[str] | None = None,
    include_1x1: bool = False,
    ready_only: bool = True,
    missing_only: bool = False,
) -> list[dict[str, Any]]:
    out = []
    for x in infos:
        if ready_only and x.get("status") != "ready":
            continue
        if not include_1x1 and is_1x1_bin_label(x.get("bin_label", "")):
            continue
        if datasets and x.get("dataset_id") not in datasets:
            continue
        if labels and x.get("data_label") not in labels:
            continue
        if bins and x.get("bin_label") not in bins:
            continue
        if missing_only and int(x.get("existing_runs", 0) or 0) > 0:
            continue
        out.append(x)
    return out


def default_bin_filter(bin_labels: list[str]) -> list[str]:
    non1 = [b for b in bin_labels if not is_1x1_bin_label(b)]
    for preferred in ["10x10/t1", "8x8/t1", "5x5/t1", "3x3/t1"]:
        if preferred in non1:
            return [preferred]
    return non1[:1] if non1 else bin_labels[:1]


# =============================================================================
# Main UI
# =============================================================================


def package_description(recipe: AcfFitRecipe) -> str:
    fit = "Fit:on" if recipe.run_fit else "Fit:off"
    maxacf = "all" if recipe.max_acf_time is None else str(recipe.max_acf_time)
    return f"{recipe.acf_method} / {fit} / split={recipe.n_splits} / frame={recipe.frame_time_sec}s / maxACF={maxacf}"



def render_package_editor(analysis_root: str):
    st.subheader("Package editor")
    st.caption("ACF/Fit packageをこの画面で編集できます。保存先は Analysis root 直下の dxb_acf_fit_packages.json です。")

    packages = load_project_packages(analysis_root)
    path = packages_path(analysis_root)
    st.code(str(path), language="text")

    if not packages:
        packages = default_packages()

    package_names = list(packages.keys())
    selected_name = st.selectbox("Edit package", package_names, key="acf_pkg_editor_select")
    rec0 = packages[selected_name]

    with st.form("acf_package_editor_form"):
        st.markdown("#### Basic")
        b1, b2, b3, b4 = st.columns(4)
        with b1:
            package_name = st.text_input("Package name", value=rec0.package_name, key="pkg_editor_name")
        with b2:
            display_label = st.text_input("Display label", value=rec0.display_label, key="pkg_editor_label")
        with b3:
            acf_method = st.selectbox(
                "ACF method",
                ["Symmetric", "Normal", "variance_normalized", "Arai_legacy"],
                index=["Symmetric", "Normal", "variance_normalized", "Arai_legacy"].index(rec0.acf_method)
                if rec0.acf_method in ["Symmetric", "Normal", "variance_normalized", "Arai_legacy"] else 0,
                key="pkg_editor_acf_method",
            )
        with b4:
            detrend_mode = st.selectbox(
                "Detrend",
                ["none", "linear", "quadratic"],
                index=["none", "linear", "quadratic"].index(rec0.detrend_mode) if rec0.detrend_mode in ["none", "linear", "quadratic"] else 0,
                key="pkg_editor_detrend",
            )

        st.markdown("#### ACF")
        a1, a2, a3, a4 = st.columns(4)
        with a1:
            n_splits = st.number_input("n_splits", min_value=1, max_value=50, value=int(rec0.n_splits), step=1, key="pkg_editor_nsplits")
        with a2:
            max_acf_val = 0 if rec0.max_acf_time is None else int(rec0.max_acf_time)
            max_acf_time_tmp = st.number_input("max_acf_time (0=None)", min_value=0, max_value=5_000_000, value=max_acf_val, step=10, key="pkg_editor_maxacf")
        with a3:
            pixels_per_batch = st.number_input("Pixels / ACF batch", min_value=1000, max_value=500000, value=int(rec0.pixels_per_batch), step=1000, key="pkg_editor_pixbatch")
        with a4:
            cpu_max = max(1, os.cpu_count() or 2)
            numba_val = 0 if rec0.numba_threads is None else int(rec0.numba_threads)
            numba_threads_tmp = st.number_input("Numba threads (0=auto)", min_value=0, max_value=cpu_max, value=numba_val, step=1, key="pkg_editor_numba")

        st.markdown("#### Time / mask")
        t1, t2, t3, t4 = st.columns(4)
        with t1:
            frame_time_sec = st.number_input("frame_time_sec", min_value=0.0, max_value=1e9, value=float(rec0.frame_time_sec or 0.5), step=0.1, format="%.6f", key="pkg_editor_frametime")
        with t2:
            frame_dtype = st.selectbox("frame dtype", ["float32", "float64"], index=0 if rec0.frame_dtype == "float32" else 1, key="pkg_editor_framedtype")
        with t3:
            mask_input_semantics = st.selectbox(
                "Mask input",
                ["one_is_valid", "zero_is_valid"],
                index=["one_is_valid", "zero_is_valid"].index(rec0.mask_input_semantics) if rec0.mask_input_semantics in ["one_is_valid", "zero_is_valid"] else 0,
                key="pkg_editor_masksem",
            )
        with t4:
            use_all_pixels_if_no_mask = st.checkbox("Use all pixels if no mask", value=bool(rec0.use_all_pixels_if_no_mask), key="pkg_editor_nomask")

        st.markdown("#### Fit")
        f1, f2, f3, f4 = st.columns(4)
        with f1:
            run_fit = st.checkbox("Run Fit", value=bool(rec0.run_fit), key="pkg_editor_runfit")
        with f2:
            fit_model = st.selectbox("Fit model", ["stretched"], index=0, key="pkg_editor_model")
        with f3:
            cpu_max = max(1, os.cpu_count() or 2)
            fit_processes = st.number_input("Fit processes", min_value=1, max_value=cpu_max, value=int(min(rec0.fit_processes, cpu_max)), step=1, key="pkg_editor_fitproc")
        with f4:
            max_lag_val = 0 if rec0.max_lag_fit is None else int(rec0.max_lag_fit)
            max_lag_fit_tmp = st.number_input("max_lag_fit (0=None)", min_value=0, max_value=5_000_000, value=max_lag_val, step=10, key="pkg_editor_maxlag")

        f5, f6, f7, f8 = st.columns(4)
        with f5:
            use_weights = st.checkbox("Use weights", value=bool(rec0.use_weights), key="pkg_editor_weights")
        with f6:
            maxfev_fit = st.number_input("maxfev_fit", min_value=100, max_value=1_000_000, value=int(rec0.maxfev_fit), step=1000, key="pkg_editor_maxfev")
        with f7:
            r2_flag_low = st.number_input("R2 flag low", min_value=-1.0, max_value=1.0, value=float(rec0.r2_flag_low), step=0.01, format="%.4f", key="pkg_editor_r2flag")
        with f8:
            save_acf_h5 = st.checkbox("Save ACF H5", value=bool(rec0.save_acf_h5), key="pkg_editor_saveacf")

        with st.expander("Bounds / selection thresholds", expanded=False):
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                a_min = st.number_input("A min", value=float(rec0.a_min), format="%.6g", key="pkg_editor_amin")
                a_max = st.number_input("A max", value=float(rec0.a_max), format="%.6g", key="pkg_editor_amax")
            with c2:
                gamma_min = st.number_input("Gamma min", value=float(rec0.gamma_min), format="%.6g", key="pkg_editor_gammamin")
                gamma_max = st.number_input("Gamma max", value=float(rec0.gamma_max), format="%.6g", key="pkg_editor_gammamax")
            with c3:
                g_min = st.number_input("g min", value=float(rec0.g_min), format="%.6g", key="pkg_editor_gmin")
                g_max = st.number_input("g max", value=float(rec0.g_max), format="%.6g", key="pkg_editor_gmax")
            with c4:
                select_r2_min = st.number_input("Select R2 min", value=float(rec0.select_r2_min), format="%.6g", key="pkg_editor_selr2")
                select_a_min = st.number_input("Select A min", value=float(rec0.select_a_min), format="%.6g", key="pkg_editor_sela")
                select_gamma_min = st.number_input("Select Gamma min", value=float(rec0.select_gamma_min), format="%.6g", key="pkg_editor_selgamma")
                select_g_min = st.number_input("Select g min", value=float(rec0.select_g_min), format="%.6g", key="pkg_editor_selg")

        st.markdown("#### Save")
        save_col1, save_col2, save_col3 = st.columns(3)
        save_current = save_col1.form_submit_button("Save package")
        save_as_new = save_col2.form_submit_button("Save as new package")
        reset_defaults = save_col3.form_submit_button("Reset all to defaults")

    new_rec = AcfFitRecipe(
        package_name=package_name.strip() or selected_name,
        display_label=display_label.strip(),
        acf_method=acf_method,
        detrend_mode=detrend_mode,
        n_splits=int(n_splits),
        max_acf_time=None if int(max_acf_time_tmp) == 0 else int(max_acf_time_tmp),
        pixels_per_batch=int(pixels_per_batch),
        frame_dtype=frame_dtype,
        numba_threads=None if int(numba_threads_tmp) == 0 else int(numba_threads_tmp),
        mask_input_semantics=mask_input_semantics,
        use_all_pixels_if_no_mask=bool(use_all_pixels_if_no_mask),
        run_fit=bool(run_fit),
        fit_model=fit_model,
        max_lag_fit=None if int(max_lag_fit_tmp) == 0 else int(max_lag_fit_tmp),
        frame_time_sec=float(frame_time_sec),
        fit_processes=int(fit_processes),
        use_weights=bool(use_weights),
        maxfev_fit=int(maxfev_fit),
        a_min=float(a_min),
        a_max=float(a_max),
        gamma_min=float(gamma_min),
        gamma_max=float(gamma_max),
        g_min=float(g_min),
        g_max=float(g_max),
        r2_flag_low=float(r2_flag_low),
        select_r2_min=float(select_r2_min),
        select_a_min=float(select_a_min),
        select_gamma_min=float(select_gamma_min),
        select_g_min=float(select_g_min),
        save_acf_h5=bool(save_acf_h5),
        overwrite_existing_run=False,
    )

    if reset_defaults:
        save_project_packages(analysis_root, default_packages())
        st.success("Default packages restored.")
        st.rerun()

    if save_current:
        updated = dict(packages)
        # If package_name was renamed, remove old key.
        if selected_name in updated and selected_name != new_rec.package_name:
            updated.pop(selected_name, None)
        updated[new_rec.package_name] = new_rec
        save_project_packages(analysis_root, updated)
        st.success(f"Saved: {package_display_name(new_rec)}")
        st.rerun()

    if save_as_new:
        updated = dict(packages)
        base_name = new_rec.package_name or "Package"
        if base_name in updated:
            i = 1
            while f"{base_name}_{i}" in updated:
                i += 1
            new_rec.package_name = f"{base_name}_{i}"
        updated[new_rec.package_name] = new_rec
        save_project_packages(analysis_root, updated)
        st.success(f"Saved as new: {package_display_name(new_rec)}")
        st.rerun()

    st.markdown("#### Current packages")
    rows = []
    for rec in packages.values():
        rows.append({
            "package": rec.package_name,
            "label": rec.display_label,
            "ACF": rec.acf_method,
            "split": rec.n_splits,
            "frame_time_sec": rec.frame_time_sec,
            "max_acf_time": "all" if rec.max_acf_time is None else rec.max_acf_time,
            "fit": "on" if rec.run_fit else "off",
            "fit_processes": rec.fit_processes,
            "description": package_description(rec),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    init_state()

    st.title(APP_TITLE)
    st.caption("Bin済みH5を選んで ACF/Fit を実行します。詳細な検出表やmanifest確認は Advanced にまとめています。")

    with st.sidebar:
        st.header("Project")
        analysis_root = st.text_input(
            "Analysis root",
            value=st.session_state.get("acf_analysis_root", ""),
            key="acf_simple_analysis_root_input",
        )
        if st.button("Refresh", key="acf_simple_refresh_h5"):
            st.session_state.acf_scan_token += 1
            st.session_state.acf_analysis_root = analysis_root
            scan_binned_h5.clear()
            scan_acf_run_manifests.clear()
        st.session_state.acf_analysis_root = analysis_root

        st.markdown("---")
        st.header("Current")
        st.metric("Selected", len(st.session_state.acf_selected_h5))
        if st.session_state.acf_is_running:
            st.warning("Running")

    if not analysis_root or not Path(analysis_root).exists():
        st.info("Analysis root を指定してください。")
        return

    h5_infos_raw = scan_binned_h5(analysis_root, st.session_state.acf_scan_token)
    run_rows = scan_acf_run_manifests(analysis_root, st.session_state.acf_scan_token)
    run_summary = build_run_summary_by_input(run_rows)
    h5_infos = add_existing_run_columns(h5_infos_raw, run_summary)
    ready_infos = [x for x in h5_infos if x.get("status") == "ready"]

    tab_run, tab_results, tab_advanced = st.tabs(["Run ACF/Fit", "Results", "Advanced"])

    with tab_run:
        if not ready_infos:
            st.warning("readyな入力H5がありません。先にBinを作成してください。")
            return

        # Lightweight project summary
        datasets_all = sorted({x.get("dataset_id", "") for x in ready_infos})
        bin_labels_all = sorted({x.get("bin_label", "") for x in ready_infos}, key=bin_sort_key)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Datasets", len(datasets_all))
        c2.metric("Ready H5", len(ready_infos))
        c3.metric("Bin labels", len(bin_labels_all))
        c4.metric("Existing runs", len(run_rows))

        st.markdown("### 1. Select input")
        packages = load_project_packages(analysis_root)

        with st.form("acf_simple_select_form"):
            s1, s2, s3 = st.columns([1.2, 1.2, 1.0])
            with s1:
                dataset_choice = st.selectbox("Dataset", ["all"] + datasets_all, index=0, key="acf_simple_dataset")
            with s2:
                visible_bins_default = [b for b in bin_labels_all if not is_1x1_bin_label(b)]
                preferred = default_bin_filter(bin_labels_all)
                default_bin = preferred[0] if preferred else (visible_bins_default[0] if visible_bins_default else bin_labels_all[0])
                bin_choices = visible_bins_default if visible_bins_default else bin_labels_all
                if default_bin not in bin_choices and bin_choices:
                    default_bin = bin_choices[0]
                bin_choice = st.selectbox(
                    "Bin",
                    bin_choices,
                    index=bin_choices.index(default_bin) if default_bin in bin_choices else 0,
                    key="acf_simple_bin",
                )
            with s3:
                hide_existing = st.checkbox("既存runなしだけ", value=False, key="acf_simple_missing_only")
                show_1x1 = st.checkbox("1x1/t1も使う", value=False, key="acf_simple_show_1x1")

            # Rebuild candidate list using the values currently in the form.
            datasets_filter = [] if dataset_choice == "all" else [dataset_choice]
            bins_filter = [bin_choice] if bin_choice else []
            candidates = filter_h5_infos(
                h5_infos,
                datasets=datasets_filter,
                labels=None,
                bins=bins_filter,
                include_1x1=show_1x1,
                ready_only=True,
                missing_only=hide_existing,
            )
            # Avoid showing paths in the main selector.
            option_map: dict[str, dict[str, Any]] = {}
            for x in candidates:
                run_note = f"runs:{x.get('existing_runs', 0)}" if x.get("existing_runs", 0) else "no run"
                label = f"{x['dataset_id']} / {x['data_label']}  ({x.get('shape_text','')}, {run_note})"
                # If duplicate label occurs, append bin/path short tail.
                if label in option_map:
                    label = f"{label} | {Path(x['rel_path']).parent}"
                option_map[label] = x

            st.caption(f"候補: {len(option_map)} inputs")
            current_selected_labels = [k for k, v in option_map.items() if v["path"] in st.session_state.acf_selected_h5]
            selected_labels = st.multiselect(
                "Targets",
                list(option_map.keys()),
                default=current_selected_labels,
                key="acf_simple_targets",
            )

            st.markdown("### 2. Analysis package")
            pkg_keys = list(packages.keys()) + ["Custom"]
            pkg_display = [package_display_name(packages[k]) if k != "Custom" else "Custom" for k in pkg_keys]
            pkg_display_choice = st.radio(
                "Package",
                pkg_display,
                index=0,
                horizontal=True,
                key="acf_simple_pkg_radio",
            )
            pkg_name = pkg_keys[pkg_display.index(pkg_display_choice)]
            if pkg_name == "Custom":
                base = recipe_from_dict(st.session_state.acf_recipe)
            else:
                base = packages[pkg_name]

            st.info(package_description(base))

            with st.expander("Custom / advanced recipe", expanded=(pkg_name == "Custom")):
                r1, r2, r3, r4 = st.columns(4)
                with r1:
                    base.package_name = st.text_input("Package name", value=base.package_name, key="acf_simple_pkg_name")
                    base.acf_method = st.selectbox(
                        "ACF method",
                        ["Symmetric", "Normal", "variance_normalized", "Arai_legacy"],
                        index=["Symmetric", "Normal", "variance_normalized", "Arai_legacy"].index(base.acf_method),
                        key="acf_simple_method",
                    )
                    base.detrend_mode = st.selectbox(
                        "Detrend",
                        ["none", "linear", "quadratic"],
                        index=["none", "linear", "quadratic"].index(base.detrend_mode),
                        key="acf_simple_detrend",
                    )
                with r2:
                    base.n_splits = st.number_input("n_splits", min_value=1, max_value=20, value=int(base.n_splits), step=1, key="acf_simple_nsplits")
                    max_acf_val = 0 if base.max_acf_time is None else int(base.max_acf_time)
                    tmp_max_acf = st.number_input("max_acf_time (0=None)", min_value=0, max_value=1_000_000, value=int(max_acf_val), step=10, key="acf_simple_maxacf")
                    base.max_acf_time = None if tmp_max_acf == 0 else int(tmp_max_acf)
                    base.frame_time_sec = st.number_input("frame_time_sec", min_value=0.0, max_value=1e9, value=float(base.frame_time_sec or 0.5), step=0.1, format="%.6f", key="acf_simple_frametime")
                with r3:
                    base.run_fit = st.checkbox("Run Fit", value=bool(base.run_fit), key="acf_simple_runfit")
                    cpu_max = max(1, os.cpu_count() or 2)
                    base.fit_processes = st.number_input("Fit processes", min_value=1, max_value=cpu_max, value=int(min(base.fit_processes, cpu_max)), step=1, key="acf_simple_fitproc")
                    base.pixels_per_batch = st.number_input("Pixels / ACF batch", min_value=1000, max_value=200000, value=int(base.pixels_per_batch), step=1000, key="acf_simple_pixbatch")
                with r4:
                    base.mask_input_semantics = st.selectbox("Mask input", ["one_is_valid", "zero_is_valid"], index=["one_is_valid", "zero_is_valid"].index(base.mask_input_semantics), key="acf_simple_masksem")
                    cpu_max = max(1, os.cpu_count() or 2)
                    numba_threads_val = 0 if base.numba_threads is None else int(base.numba_threads)
                    tmp_threads = st.number_input("Numba threads (0=auto)", min_value=0, max_value=cpu_max, value=numba_threads_val, step=1, key="acf_simple_numba")
                    base.numba_threads = None if tmp_threads == 0 else int(tmp_threads)

            b1, b2, b3 = st.columns(3)
            apply_selected = b1.form_submit_button("Apply selected", disabled=st.session_state.acf_is_running)
            select_all = b2.form_submit_button("Select all targets", disabled=st.session_state.acf_is_running)
            clear_selected = b3.form_submit_button("Clear", disabled=st.session_state.acf_is_running)

        if apply_selected:
            st.session_state.acf_selected_h5 = [option_map[k]["path"] for k in selected_labels]
            st.session_state.acf_recipe = asdict(base)
            st.success(f"Applied: {len(st.session_state.acf_selected_h5)} inputs / {base.package_name}")
        if select_all:
            st.session_state.acf_selected_h5 = [x["path"] for x in candidates]
            st.session_state.acf_recipe = asdict(base)
            st.success(f"Selected all targets: {len(candidates)} inputs / {base.package_name}")
        if clear_selected:
            st.session_state.acf_selected_h5 = []
            st.session_state.acf_recipe = asdict(base)
            st.info("Selection cleared")

        selected_infos = [x for x in ready_infos if x["path"] in st.session_state.acf_selected_h5]
        recipe_now = recipe_from_dict(st.session_state.acf_recipe)
        estimated_outputs = len(selected_infos) * int(recipe_now.n_splits)

        st.markdown("### 3. Run")
        c1, c2, c3 = st.columns(3)
        c1.metric("Selected inputs", len(selected_infos))
        c2.metric("Split outputs", estimated_outputs)
        c3.metric("Package", recipe_now.package_name)

        if selected_infos:
            with st.expander("Selected details", expanded=False):
                sel_df = pd.DataFrame(selected_infos)
                view_cols = ["dataset_id", "data_label", "bin_label", "shape_text", "dtype", "has_mask", "existing_runs", "existing_packages", "rel_path"]
                st.dataframe(sel_df[[c for c in view_cols if c in sel_df.columns]], use_container_width=True, height=min(320, 68 + 28 * len(sel_df)))
        else:
            st.info("Step 1で対象を選んでください。")

        st.markdown("#### Progress")
        overall_box = st.empty()
        overall_bar = st.progress(0.0)
        status_box = st.empty()
        progress_bar = st.progress(0.0)
        with st.expander("Detailed log", expanded=False):
            log_area = st.empty()
        run_clicked = st.button(
            "Run selected",
            type="primary",
            disabled=st.session_state.acf_is_running or not selected_infos,
            key="acf_simple_run_selected",
        )

        if run_clicked:
            # Queue is internal in Simple UI.
            st.session_state.acf_is_running = True
            queue_items = build_queue_items(selected_infos, st.session_state.acf_recipe, overwrite=False)
            st.session_state.acf_queue = queue_items
            results = []
            try:
                total_jobs = len(queue_items)
                for i, item_dict in enumerate(queue_items, start=1):
                    item = QueueItem(**item_dict)
                    overall_bar.progress((i - 1) / max(total_jobs, 1))
                    overall_box.info(
                        f"Overall: job {i}/{total_jobs} | done {i-1}/{total_jobs} | "
                        f"current: {item.dataset_id} / {item.data_label}"
                    )
                    ctx = {
                        "job_index": i,
                        "job_total": total_jobs,
                        "dataset_id": item.dataset_id,
                        "data_label": item.data_label,
                        "rel_path": item.rel_path,
                    }
                    res = run_acf_fit_job(item, status_box, progress_bar, log_area, job_context=ctx)
                    results.append(res)
                    overall_bar.progress(i / max(total_jobs, 1))
                    overall_box.success(f"Overall: completed {i}/{total_jobs} jobs")
            finally:
                st.session_state.acf_is_running = False
                st.session_state.acf_last_results = results
                st.session_state.acf_queue = []
                scan_acf_run_manifests.clear()
            st.success("Run finished")
            st.rerun()

    with tab_results:
        st.subheader("Results")
        if st.session_state.acf_last_results:
            st.markdown("#### Last run")
            st.dataframe(pd.DataFrame(st.session_state.acf_last_results), use_container_width=True)

        if run_rows:
            rdf = pd.DataFrame(run_rows)
            # Simple filters for results.
            r1, r2 = st.columns([1, 1])
            with r1:
                ds_opts = ["all"] + sorted(rdf["dataset_id"].dropna().astype(str).unique().tolist())
                ds_res = st.selectbox("Dataset", ds_opts, index=0, key="acf_results_dataset")
            with r2:
                status_opts = ["all"] + sorted(rdf["status"].dropna().astype(str).unique().tolist())
                status_res = st.selectbox("Status", status_opts, index=0, key="acf_results_status")
            show = rdf.copy()
            if ds_res != "all":
                show = show[show["dataset_id"].astype(str) == ds_res]
            if status_res != "all":
                show = show[show["status"].astype(str) == status_res]
            cols = ["dataset_id", "data_label", "run_id", "status", "package_name", "input_shape", "created_at", "manifest"]
            st.dataframe(show[[c for c in cols if c in show.columns]], use_container_width=True, height=520)
            st.caption("Viewerでは manifest または runフォルダを入口にします。")
        else:
            st.info("ACF/Fit runはまだ見つかりません。")

    with tab_advanced:
        st.subheader("Advanced")
        render_package_editor(analysis_root)
        st.markdown("---")
        st.caption("普段は触らなくてよい詳細情報です。検出表、matrix、内部Queue、保存形式を確認できます。")

        adv_tab1, adv_tab2, adv_tab3, adv_tab4 = st.tabs(["Detected", "Matrix", "Runs", "Storage"])
        with adv_tab1:
            st.markdown("#### Full detected H5 table")
            if h5_infos:
                df = pd.DataFrame(h5_infos)
                cols = ["dataset_id", "data_label", "bin_label", "shape_text", "dtype", "has_mask", "existing_runs", "existing_packages", "status", "rel_path", "error"]
                st.dataframe(df[[c for c in cols if c in df.columns]], use_container_width=True, height=520)
            else:
                st.info("検出されたH5がありません。")

        with adv_tab2:
            st.markdown("#### Dataset × Bin matrix")
            include_1x1_adv = st.checkbox("1x1/t1もMatrixに表示", value=False, key="acf_adv_show_1x1")
            matrix = make_bin_matrix(h5_infos, include_1x1=include_1x1_adv)
            if matrix.empty:
                st.info("表示対象がありません。")
            else:
                st.dataframe(matrix, use_container_width=True, height=min(600, 84 + 32 * len(matrix)))

        with adv_tab3:
            st.markdown("#### Existing run manifests")
            if run_rows:
                st.dataframe(pd.DataFrame(run_rows), use_container_width=True, height=520)
            else:
                st.info("既存runはありません。")

            st.markdown("#### Internal queue")
            if st.session_state.acf_queue:
                st.dataframe(pd.DataFrame(st.session_state.acf_queue), use_container_width=True, height=320)
            else:
                st.caption("Queue is empty. Simple UIではRun時に内部で作成します。")

        with adv_tab4:
            st.subheader("保存形式")
            st.code(
                """<channel_root>/03_analysis/acf_fit/<run_id>/
├─ analysis_manifest.json
├─ recipe.json
├─ split_manifest.json
├─ outputs/
│  ├─ split01/
│  │  ├─ acf.h5
│  │  ├─ fit.csv
│  │  ├─ grid_meta.npz
│  │  └─ summary.json
│  └─ split02/
│     ├─ acf.h5
│     ├─ fit.csv
│     ├─ grid_meta.npz
│     └─ summary.json
└─ logs/
   └─ run.log""",
                language="text",
            )
            st.markdown(
                """
- `analysis_manifest.json` がViewer/Compareの入口です。
- `split_manifest.json` に `frame_start/frame_end` と `acf.h5/fit.csv/grid_meta.npz` の対応を保存します。
- splitは平均しません。
- 1x1/t1 は重くなりやすいので標準では隠しています。
- Fitが不安定な場合は `Fit processes=1` で確認してください。
                """
            )


if __name__ == "__main__":
    mp.freeze_support()
    main()
