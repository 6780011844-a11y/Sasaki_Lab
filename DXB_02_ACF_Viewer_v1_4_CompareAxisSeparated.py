#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DXB 04: ACF/Fit Viewer v1.4 CompareAxisSeparated
=========================================

For the new DXB_03_ACF_Fit_Manager output format:

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

Features
--------
- Select an ACF/Fit run by analysis_manifest.json
- Heatmap + histogram from fit.csv and grid_meta.npz
- Split selection and split comparison
- Pixel inspector: I(t), ACF, ACF+fit for x,y
- Uses frame_start/frame_end from split_manifest.json
- Standard display labels are short: Γ, τ, R², A, g
- Exports PNG / CSV / JSON
- Time metadata aware: uses effective_frame_time_sec when available
- Faster run browser refresh via cached manifest scan
- Compare View v1.4: separated heatmap color range and distribution x-axis range
- Compare View v1.4: y-axis mode and separate distribution layout

Start:
    streamlit run DXB_02_ACF_Viewer_v1_3_TimeMeta_FastUI.py
"""

from __future__ import annotations

import io
import os
import re
import json
import math
import copy
from pathlib import Path
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import h5py
import hdf5plugin  # noqa: F401  # needed for Bitshuffle/LZ4 HDF5 files
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize, LogNorm
import streamlit as st


# =============================================================================
# App config
# =============================================================================

APP_TITLE = "DXB 04 ACF/Fit Viewer v1.4"
DEFAULT_FRAME_TIME_SEC = 0.5
DEFAULT_CMAP = "physics"
DEFAULT_FIG_H = 4.8

st.set_page_config(page_title=APP_TITLE, layout="wide")


# =============================================================================
# Style
# =============================================================================

st.markdown(
    """
<style>
.block-container { padding-top: 1.0rem; padding-bottom: 1rem; }
[data-testid="stMetricValue"] { font-size: 1.1rem; }
.small-caption { color: #666; font-size: 0.85rem; }
.card {
  padding: 0.7rem 0.85rem;
  border: 1px solid rgba(49, 51, 63, 0.15);
  border-radius: 0.7rem;
  background: rgba(250, 250, 252, 0.72);
  margin-bottom: 0.6rem;
}
.warn-card {
  padding: 0.7rem 0.85rem;
  border: 1px solid rgba(255, 170, 0, 0.35);
  border-radius: 0.7rem;
  background: rgba(255, 245, 220, 0.72);
  margin-bottom: 0.6rem;
}
</style>
""",
    unsafe_allow_html=True,
)


# =============================================================================
# Color maps
# =============================================================================

PHYSICS_CMAP = LinearSegmentedColormap.from_list(
    "physics_custom",
    [
        "#1100ff", "#1a4cff", "#00b7ff", "#00f0ff", "#27ff88",
        "#b7ff00", "#fff200", "#ffb300", "#ff5a00", "#d40000",
    ],
    N=256,
)

CMAP_PRESETS: dict[str, Any] = {
    "physics": PHYSICS_CMAP,
    "viridis": "viridis",
    "plasma": "plasma",
    "inferno": "inferno",
    "magma": "magma",
    "turbo": "turbo",
    "gray": "gray",
    "coolwarm": "coolwarm",
}


def get_cmap(name: str, reverse: bool = False):
    cmap = CMAP_PRESETS.get(name, PHYSICS_CMAP)
    if isinstance(cmap, str):
        obj = plt.get_cmap(cmap)
    else:
        obj = cmap
    obj = copy.copy(obj)
    if reverse:
        obj = obj.reversed()
    try:
        obj.set_bad(color=(1, 1, 1, 0))
    except Exception:
        pass
    return obj


# =============================================================================
# Data structures / utilities
# =============================================================================

@dataclass
class RunBundle:
    run_dir: Path
    analysis_manifest: dict[str, Any]
    recipe: dict[str, Any]
    split_manifest: dict[str, Any]


@dataclass
class SplitFiles:
    split_name: str
    split_index: int
    frame_start: int
    frame_end: int
    acf_h5: Path
    fit_csv: Path | None
    grid_meta: Path
    summary: Path | None


def clean_path_text(s: str) -> str:
    return str(s or "").strip().strip("\"'")


def read_json(path: str | Path, default=None):
    p = Path(path)
    if not p.exists():
        return default
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return default


def rel_resolve(base: Path, maybe_rel: str | None) -> Path | None:
    if not maybe_rel:
        return None
    p = Path(str(maybe_rel))
    if p.is_absolute():
        return p
    return (base / p).resolve()


def safe_file_label(p: str | Path) -> str:
    pp = Path(str(p))
    parts = list(pp.parts)
    if len(parts) >= 4:
        return str(Path(*parts[-4:]))
    return str(pp)


def _float_or_none(v):
    try:
        if v is None:
            return None
        if isinstance(v, (bytes, bytearray)):
            v = v.decode("utf-8", errors="replace")
        if isinstance(v, np.ndarray):
            if v.size == 0:
                return None
            v = np.ravel(v)[0]
        if isinstance(v, str):
            vv = v.strip()
            if not vv or vv.lower() in ["none", "nan", "null", "-"]:
                return None
            v = vv
        x = float(v)
        if not np.isfinite(x) or x <= 0:
            return None
        return x
    except Exception:
        return None


def _int_or_none(v):
    try:
        if v is None:
            return None
        if isinstance(v, (bytes, bytearray)):
            v = v.decode("utf-8", errors="replace")
        if isinstance(v, np.ndarray):
            if v.size == 0:
                return None
            v = np.ravel(v)[0]
        if isinstance(v, str):
            vv = v.strip()
            if not vv or vv.lower() in ["none", "nan", "null", "-"]:
                return None
            v = vv
        x = int(float(v))
        return x if x > 0 else None
    except Exception:
        return None


def _first_meta_value(dicts: list[dict[str, Any]], keys: list[str]):
    for d in dicts:
        if not isinstance(d, dict):
            continue
        for k in keys:
            if k in d:
                return d.get(k)
    return None


def resolve_time_meta(analysis_manifest: dict[str, Any] | None, recipe: dict[str, Any] | None = None,
                      split_entry: dict[str, Any] | None = None) -> dict[str, Any]:
    """Resolve the frame time that should be used for display axes.

    New DXB_01 writes effective_frame_time_sec to manifests after applying time_bin.
    Viewer must not re-multiply by time_bin, because fit.csv was already generated
    using the effective time. This function only selects and displays the stored value.
    """
    am = analysis_manifest if isinstance(analysis_manifest, dict) else {}
    rc = recipe if isinstance(recipe, dict) else {}
    sp = split_entry if isinstance(split_entry, dict) else {}
    sources = [sp, am, rc]

    explicit_effective = _float_or_none(_first_meta_value(sources, [
        "effective_frame_time_sec",
        "frame_interval_effective_sec",
        "dt_effective_sec",
    ]))
    frame_time = explicit_effective
    if frame_time is None:
        frame_time = _float_or_none(_first_meta_value(sources, ["frame_time_sec", "dt_sec"]))
    if frame_time is None:
        frame_time = float(DEFAULT_FRAME_TIME_SEC)

    raw_frame_time = _float_or_none(_first_meta_value(sources, [
        "raw_frame_time_sec",
        "source_frame_time_sec",
        "frame_time_raw_sec",
    ]))
    time_bin = _int_or_none(_first_meta_value(sources, ["time_bin", "temporal_bin", "t_bin"]))
    package_frame_time = _float_or_none(_first_meta_value(sources, [
        "package_frame_time_sec",
        "recipe_frame_time_sec",
    ]))
    if package_frame_time is None:
        package_frame_time = _float_or_none(rc.get("frame_time_sec"))

    frame_time_source = str(_first_meta_value(sources, ["frame_time_source", "time_meta_source"]) or "")
    has_explicit_effective = explicit_effective is not None
    if not frame_time_source:
        frame_time_source = "effective_frame_time_sec" if has_explicit_effective else "frame_time_sec fallback"

    return {
        "frame_time_sec": float(frame_time),
        "effective_frame_time_sec": float(frame_time),
        "raw_frame_time_sec": raw_frame_time,
        "time_bin": time_bin,
        "package_frame_time_sec": package_frame_time,
        "frame_time_source": frame_time_source,
        "has_explicit_effective_frame_time": bool(has_explicit_effective),
    }


def _fmt_num(x, digits: int = 6) -> str:
    try:
        if x is None or pd.isna(x):
            return "-"
        return f"{float(x):.{digits}g}"
    except Exception:
        return "-"


def time_meta_caption(meta_or_row: Any) -> str:
    try:
        get = meta_or_row.get
    except Exception:
        return "effective frame: -"
    eff = get("effective_frame_time_sec", get("frame_time_sec", None))
    raw = get("raw_frame_time_sec", None)
    tb = get("time_bin", None)
    src = str(get("frame_time_source", "") or "")
    core = f"effective={_fmt_num(eff)} s"
    if raw is not None and tb is not None and not pd.isna(raw):
        core += f" / raw={_fmt_num(raw)} s × t{int(tb)}"
    if src:
        core += f" / source={src}"
    return core


def selected_split_manifest_entry(bundle: 'RunBundle', split_name: str) -> dict[str, Any]:
    try:
        for s in bundle.split_manifest.get("splits", []):
            if str(s.get("split_name", "")) == str(split_name):
                return s if isinstance(s, dict) else {}
    except Exception:
        pass
    return {}


@st.cache_data(show_spinner=False)
def find_run_manifest_strings_cached(root: str, refresh_token: int = 0) -> list[str]:
    r = Path(root)
    if not r.exists():
        return []
    return [str(p) for p in sorted(r.glob("**/03_analysis/acf_fit/run_*/analysis_manifest.json"), key=lambda p: str(p))]


def find_run_manifests(root: str | Path, refresh_token: int = 0) -> list[Path]:
    return [Path(x) for x in find_run_manifest_strings_cached(str(root), int(refresh_token))]


def _bin_label_from_manifest(m: dict[str, Any]) -> str:
    sb = m.get("space_bin", None)
    tb = m.get("time_bin", None)
    try:
        if sb is not None and tb is not None:
            return f"{int(sb)}x{int(sb)}/t{int(tb)}"
    except Exception:
        pass
    txt = str(m.get("input_h5", ""))
    mm = re.search(r"(\d+)x\1[\\/]+t(\d+)", txt)
    if mm:
        return f"{mm.group(1)}x{mm.group(1)}/t{mm.group(2)}"
    return "unknown"


def _short_package_name(name: str) -> str:
    txt = str(name or "").strip()
    m = re.search(r"package\s*(\d+)", txt, flags=re.I)
    if m:
        return f"P{m.group(1)}"
    return txt or "-"


def run_label(manifest_path: Path) -> str:
    """Concise fallback label for a run manifest."""
    m = read_json(manifest_path, default={}) or {}
    ds = m.get("dataset_id", "dataset")
    data_label = m.get("data_label", "default")
    bin_label = _bin_label_from_manifest(m)
    pkg = _short_package_name(str(m.get("package_name", "")))
    return f"{ds} / {data_label} / {bin_label} / {pkg}"


def _fmt_time_from_mtime(ts: float) -> str:
    try:
        import datetime as _dt
        return _dt.datetime.fromtimestamp(float(ts)).strftime("%Y-%m-%d %H:%M")
    except Exception:
        return ""


def notes_file_for_root(root: str | Path) -> Path:
    return Path(root) / "dxb_acf_run_notes.json"


def load_run_notes(root: str | Path) -> dict[str, Any]:
    p = notes_file_for_root(root)
    d = read_json(p, default={}) or {}
    return d if isinstance(d, dict) else {}


def save_run_notes(root: str | Path, notes: dict[str, Any]):
    p = notes_file_for_root(root)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(notes, ensure_ascii=False, indent=2), encoding="utf-8")


@st.cache_data(show_spinner=False)
def build_run_catalog_cached(manifest_path_strings: tuple[str, ...]) -> pd.DataFrame:
    """Build a human-readable run catalog. One row per analysis run."""
    rows: list[dict[str, Any]] = []
    for mp_str in manifest_path_strings:
        mp = Path(mp_str)
        run_dir = mp.parent
        am = read_json(mp, default={}) or {}
        recipe = read_json(run_dir / str(am.get("recipe", "recipe.json")), default={}) or {}
        sm = read_json(run_dir / str(am.get("split_manifest", "split_manifest.json")), default={}) or {}
        mtime = mp.stat().st_mtime if mp.exists() else 0.0
        dataset_id = str(am.get("dataset_id", "dataset"))
        data_label = str(am.get("data_label", "default"))
        package_name = str(am.get("package_name", recipe.get("package_name", "")))
        bin_label = _bin_label_from_manifest(am)
        run_id = str(am.get("run_id", run_dir.name))
        status = str(am.get("status", "unknown"))
        n_splits = len(sm.get("splits", [])) if isinstance(sm, dict) else int(am.get("n_splits", 0) or 0)
        tm = resolve_time_meta(am, recipe)
        frame_time = tm["frame_time_sec"]
        acf_method = str(am.get("acf_method", recipe.get("acf_method", "")))
        fit_model = str(am.get("fit_model", recipe.get("fit_model", "")))
        created = str(am.get("created_at", "")) or _fmt_time_from_mtime(mtime)
        rows.append({
            "manifest": mp_str,
            "run_dir": str(run_dir),
            "dataset_id": dataset_id,
            "data_label": data_label,
            "bin_label": bin_label,
            "package": package_name,
            "package_short": _short_package_name(package_name),
            "status": status,
            "run_id": run_id,
            "created": created,
            "created_mtime": float(mtime),
            "n_splits": int(n_splits),
            "frame_time_sec": frame_time,
            "effective_frame_time_sec": tm["effective_frame_time_sec"],
            "raw_frame_time_sec": tm["raw_frame_time_sec"],
            "time_bin": tm["time_bin"],
            "package_frame_time_sec": tm["package_frame_time_sec"],
            "frame_time_source": tm["frame_time_source"],
            "has_explicit_effective_frame_time": tm["has_explicit_effective_frame_time"],
            "acf_method": acf_method,
            "fit_model": fit_model,
            "input_h5": str(am.get("input_h5", "")),
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["dataset_id", "data_label", "bin_label", "created_mtime"], ascending=[True, True, True, False], kind="stable").reset_index(drop=True)
    return df


def apply_run_notes_to_catalog(df: pd.DataFrame, notes: dict[str, Any]) -> pd.DataFrame:
    out = df.copy()
    if out.empty:
        return out
    run_notes = []
    favs = []
    labels = []
    for _, r in out.iterrows():
        n = notes.get(str(r["manifest"]), {}) if isinstance(notes, dict) else {}
        if not isinstance(n, dict):
            n = {}
        note = str(n.get("note", ""))
        fav = bool(n.get("favorite", False))
        run_notes.append(note)
        favs.append(fav)
        star = "★ " if fav else ""
        note_txt = f" — {note}" if note else ""
        labels.append(f"{star}{r['dataset_id']} / {r['bin_label']} / {r['package_short']} / {r['created']}{note_txt}")
    out["run_note"] = run_notes
    out["favorite"] = favs
    out["display_label"] = labels
    return out


def newest_runs_per_group(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    group_cols = ["dataset_id", "data_label", "bin_label", "package"]
    d = df.sort_values("created_mtime", ascending=False, kind="stable")
    return d.drop_duplicates(group_cols, keep="first").sort_values(["dataset_id", "data_label", "bin_label"], kind="stable")


@st.cache_data(show_spinner=False)
def load_run_bundle_cached(manifest_path_str: str) -> dict[str, Any]:
    manifest_path = Path(manifest_path_str)
    run_dir = manifest_path.parent
    analysis_manifest = read_json(manifest_path, default={}) or {}
    recipe = read_json(run_dir / str(analysis_manifest.get("recipe", "recipe.json")), default={}) or {}
    split_manifest = read_json(run_dir / str(analysis_manifest.get("split_manifest", "split_manifest.json")), default={}) or {}
    return {
        "run_dir": str(run_dir),
        "analysis_manifest": analysis_manifest,
        "recipe": recipe,
        "split_manifest": split_manifest,
    }


def load_run_bundle(manifest_path: Path) -> RunBundle:
    d = load_run_bundle_cached(str(manifest_path))
    return RunBundle(Path(d["run_dir"]), d["analysis_manifest"], d["recipe"], d["split_manifest"])


def get_splits(bundle: RunBundle) -> list[SplitFiles]:
    out: list[SplitFiles] = []
    for i, s in enumerate(bundle.split_manifest.get("splits", []), start=1):
        split_name = str(s.get("split_name", f"split{i:02d}"))
        acf = rel_resolve(bundle.run_dir, s.get("acf_h5"))
        fit = rel_resolve(bundle.run_dir, s.get("fit_csv"))
        meta = rel_resolve(bundle.run_dir, s.get("grid_meta"))
        summary = rel_resolve(bundle.run_dir, s.get("summary"))
        if acf is None or meta is None:
            continue
        out.append(
            SplitFiles(
                split_name=split_name,
                split_index=int(s.get("split_index", i)),
                frame_start=int(s.get("frame_start", 0)),
                frame_end=int(s.get("frame_end", 0)),
                acf_h5=acf,
                fit_csv=fit if fit and fit.exists() else None,
                grid_meta=meta,
                summary=summary if summary and summary.exists() else None,
            )
        )
    return out


@st.cache_data(show_spinner=False)
def load_grid_meta_cached(path_str: str):
    z = np.load(path_str, allow_pickle=True)
    if "mask" not in z:
        raise ValueError(f"grid_metaに mask がありません: {path_str}")
    mask = z["mask"].astype(np.uint8)
    if "height" in z and "width" in z:
        H = int(np.ravel(z["height"])[0])
        W = int(np.ravel(z["width"])[0])
    elif "H" in z and "W" in z:
        H = int(np.ravel(z["H"])[0])
        W = int(np.ravel(z["W"])[0])
    else:
        H, W = mask.shape
    return H, W, mask


@st.cache_data(show_spinner=False)
def load_fit_csv_cached(path_str: str) -> pd.DataFrame:
    df = pd.read_csv(path_str)
    df.columns = df.columns.str.strip()
    return df


@st.cache_data(show_spinner=False)
def load_acf_coords_cached(acf_h5_path: str):
    with h5py.File(acf_h5_path, "r") as hf:
        x = hf["x_coords"][:].astype(np.int32)
        y = hf["y_coords"][:].astype(np.int32)
        n_lags = int(hf["ACF"].shape[1])
        attrs = {k: hf.attrs[k] for k in hf.attrs.keys()}
    return x, y, n_lags, attrs


@st.cache_data(show_spinner=False)
def load_acf_trace_cached(acf_h5_path: str, pixel_index: int) -> np.ndarray:
    with h5py.File(acf_h5_path, "r") as hf:
        return np.asarray(hf["ACF"][int(pixel_index), :], dtype=np.float64)


@st.cache_data(show_spinner=False)
def load_image_trace_cached(h5_path: str, frame_start: int, frame_end: int, x: int, y: int, radius: int = 0) -> np.ndarray:
    with h5py.File(h5_path, "r") as hf:
        d = hf["/entry/data/images"]
        T, H, W = d.shape
        fs = max(0, min(int(frame_start), T))
        fe = max(fs, min(int(frame_end), T))
        xi = int(x)
        yi = int(y)
        r = max(0, int(radius))
        if r <= 0:
            return np.asarray(d[fs:fe, yi, xi], dtype=np.float64)
        x0 = max(0, xi - r)
        x1 = min(W, xi + r + 1)
        y0 = max(0, yi - r)
        y1 = min(H, yi + r + 1)
        block = np.asarray(d[fs:fe, y0:y1, x0:x1], dtype=np.float64)
        return np.nanmean(block, axis=(1, 2))


def get_source_images_h5(bundle: RunBundle, split: SplitFiles) -> str | None:
    # Prefer ACF attrs, then analysis_manifest input_h5
    try:
        _, _, _, attrs = load_acf_coords_cached(str(split.acf_h5))
        p = attrs.get("source_images_h5", None)
        if p is not None:
            if isinstance(p, bytes):
                p = p.decode("utf-8", errors="replace")
            return str(p)
    except Exception:
        pass
    p = bundle.analysis_manifest.get("input_h5", None)
    if p:
        return str(p)
    return None


# =============================================================================
# Fit data helpers
# =============================================================================


def add_display_columns(df: pd.DataFrame, frame_time_sec: float | None) -> pd.DataFrame:
    out = df.copy()
    for c in ["x", "y", "A", "Gamma", "g", "R2"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    ft = frame_time_sec if frame_time_sec is not None and float(frame_time_sec) > 0 else None

    # Best-effort Gamma columns
    if "Gamma" in out.columns:
        gamma = pd.to_numeric(out["Gamma"], errors="coerce")
        unit = str(out["Gamma_unit"].dropna().iloc[0]) if "Gamma_unit" in out.columns and out["Gamma_unit"].dropna().size else ""
        if "sec" in unit or "s^-1" in unit or "s-1" in unit:
            gamma_s = gamma
            gamma_frame = gamma_s * float(ft) if ft else np.nan
        elif "frame" in unit:
            gamma_frame = gamma
            gamma_s = gamma_frame / float(ft) if ft else np.nan
        else:
            # If frame_time was set in recipe, current Manager fit Gamma is usually sec^-1.
            if ft:
                gamma_s = gamma
                gamma_frame = gamma_s * float(ft)
            else:
                gamma_frame = gamma
                gamma_s = np.nan

        out["Γ [s⁻¹]"] = gamma_s
        out["log₁₀ Γ [s⁻¹]"] = np.log10(pd.Series(gamma_s).clip(lower=1e-30))
        tau_s = 1.0 / pd.Series(gamma_s).replace(0, np.nan)
        out["τ [s]"] = tau_s.replace([np.inf, -np.inf], np.nan)
        out["log₁₀ τ [s]"] = np.log10(out["τ [s]"].clip(lower=1e-30))

        out["Γ [frame⁻¹]"] = gamma_frame
        out["log₁₀ Γ [frame⁻¹]"] = np.log10(pd.Series(gamma_frame).clip(lower=1e-30))
        tau_frame = 1.0 / pd.Series(gamma_frame).replace(0, np.nan)
        out["τ [frame]"] = tau_frame.replace([np.inf, -np.inf], np.nan)
        out["log₁₀ τ [frame]"] = np.log10(out["τ [frame]"].clip(lower=1e-30))

    rename_map = {
        "R2": "R²",
        "A": "A",
        "g": "g",
    }
    for old, new in rename_map.items():
        if old in out.columns and new not in out.columns:
            out[new] = pd.to_numeric(out[old], errors="coerce")
    return out


def candidate_display_columns(df: pd.DataFrame, show_frame_cols: bool = False, show_detail_cols: bool = False) -> list[str]:
    base = ["log₁₀ Γ [s⁻¹]", "Γ [s⁻¹]", "τ [s]", "log₁₀ τ [s]", "R²", "A", "g"]
    frame = ["log₁₀ Γ [frame⁻¹]", "Γ [frame⁻¹]", "τ [frame]", "log₁₀ τ [frame]"]
    cols = [c for c in base if c in df.columns]
    if show_frame_cols:
        cols += [c for c in frame if c in df.columns]
    if show_detail_cols:
        for c in df.columns:
            if c in ["x", "y"] or c in cols:
                continue
            s = pd.to_numeric(df[c], errors="coerce")
            if s.notna().any():
                cols.append(c)
    return cols


def build_heatmap_grid(df_filtered: pd.DataFrame, value_col: str, H: int, W: int, mask: np.ndarray, fill_zero: bool = False) -> np.ndarray:
    grid = np.full((H, W), np.nan, dtype=float)
    if fill_zero:
        grid[mask > 0] = 0.0
    if df_filtered.empty:
        return grid
    for _, row in df_filtered.iterrows():
        try:
            xi = int(row["x"])
            yi = int(row["y"])
            val = float(row[value_col])
        except Exception:
            continue
        if 0 <= yi < H and 0 <= xi < W and mask[yi, xi] > 0 and np.isfinite(val):
            grid[yi, xi] = val
    grid[mask == 0] = np.nan
    return grid


def finite_values(grid: np.ndarray) -> np.ndarray:
    vals = grid[np.isfinite(grid)]
    return vals.astype(float) if vals.size else vals


def auto_range(vals: np.ndarray, low: float = 1.0, high: float = 99.0, fallback=(0.0, 1.0)) -> tuple[float, float]:
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return fallback
    vmin = float(np.percentile(vals, low))
    vmax = float(np.percentile(vals, high))
    if not np.isfinite(vmin):
        vmin = float(np.nanmin(vals))
    if not np.isfinite(vmax):
        vmax = float(np.nanmax(vals))
    if vmax <= vmin:
        vmax = vmin + 1.0
    return vmin, vmax


def make_norm(vmin: float, vmax: float, scale: str = "linear"):
    if scale == "log" and vmin > 0 and vmax > vmin:
        return LogNorm(vmin=vmin, vmax=vmax)
    return Normalize(vmin=vmin, vmax=vmax)


def fig_to_png_bytes(fig, image_only: bool = False, dpi: int = 220) -> bytes:
    buf = io.BytesIO()
    if image_only:
        fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", pad_inches=0, transparent=True)
    else:
        fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def make_heatmap_fig(
    grid: np.ndarray,
    title: str,
    cmap_name: str,
    reverse_cmap: bool,
    vmin: float,
    vmax: float,
    scale: str,
    label: str,
    fig_h: float = DEFAULT_FIG_H,
    show_axes: bool = True,
    marker_xy: tuple[int, int] | None = None,
):
    H, W = grid.shape
    fig_w = max(3.2, fig_h * W / max(H, 1))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    norm = make_norm(vmin, vmax, scale=scale)
    im = ax.imshow(grid, origin="upper", interpolation="nearest", cmap=get_cmap(cmap_name, reverse_cmap), norm=norm)
    if show_axes:
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(label)
    else:
        ax.set_axis_off()
    if marker_xy is not None:
        x, y = marker_xy
        ax.axvline(x, linewidth=0.8, color="white", alpha=0.9)
        ax.axhline(y, linewidth=0.8, color="white", alpha=0.9)
        ax.axvline(x, linewidth=0.35, color="black", alpha=0.9)
        ax.axhline(y, linewidth=0.35, color="black", alpha=0.9)
        ax.plot([x], [y], marker="o", markersize=3.5, markerfacecolor="none", markeredgecolor="black", markeredgewidth=1.1)
        ax.plot([x], [y], marker="o", markersize=2.2, markerfacecolor="white", markeredgecolor="white")
    fig.tight_layout()
    return fig


def make_hist_fig(vals: np.ndarray, title: str, xlabel: str, vmin: float, vmax: float, bins: int = 60):
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    vals = vals[np.isfinite(vals)]
    if vals.size:
        ax.hist(vals, bins=int(bins), range=(float(vmin), float(vmax)))
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")
    fig.tight_layout()
    return fig


def apply_value_filters(df: pd.DataFrame, r2_min: float | None = None, value_col: str | None = None, vmin=None, vmax=None) -> pd.DataFrame:
    out = df.copy()
    if r2_min is not None and "R²" in out.columns:
        out = out[pd.to_numeric(out["R²"], errors="coerce") >= float(r2_min)]
    if value_col and value_col in out.columns and vmin is not None and vmax is not None:
        lo, hi = min(float(vmin), float(vmax)), max(float(vmin), float(vmax))
        s = pd.to_numeric(out[value_col], errors="coerce")
        out = out[s.between(lo, hi, inclusive="both")]
    return out


def coord_to_index(x_arr: np.ndarray, y_arr: np.ndarray) -> dict[tuple[int, int], int]:
    return {(int(x), int(y)): int(i) for i, (x, y) in enumerate(zip(x_arr, y_arr))}


def nearest_valid_xy(mask: np.ndarray, x: int, y: int) -> tuple[int, int]:
    H, W = mask.shape
    x = max(0, min(int(x), W - 1))
    y = max(0, min(int(y), H - 1))
    if mask[y, x] > 0:
        return x, y
    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        return x, y
    d2 = (xs - x) ** 2 + (ys - y) ** 2
    k = int(np.argmin(d2))
    return int(xs[k]), int(ys[k])


def parse_xy(text: str, default=(0, 0)) -> tuple[int, int]:
    nums = re.findall(r"-?\d+", str(text))
    if len(nums) >= 2:
        return int(nums[0]), int(nums[1])
    return default


def model_stretched(tau, A, Gamma, g):
    return 1.0 + A * np.exp(-np.power(np.maximum(Gamma * tau, 0), g))


# =============================================================================
# Compare view helpers
# =============================================================================

STANDARD_COMPARE_VALUES = [
    "log₁₀ Γ [s⁻¹]", "Γ [s⁻¹]", "τ [s]", "log₁₀ τ [s]", "R²", "A", "g",
    "log₁₀ Γ [frame⁻¹]", "Γ [frame⁻¹]", "τ [frame]", "log₁₀ τ [frame]",
]


def _safe_rel(base: Path, p: str | None) -> str | None:
    rr = rel_resolve(base, p)
    return str(rr) if rr is not None else None


@st.cache_data(show_spinner=False)
def build_compare_index_cached(manifest_path_strings: tuple[str, ...]) -> pd.DataFrame:
    """Build one row per run/split for Compare View."""
    rows: list[dict[str, Any]] = []
    for mp_str in manifest_path_strings:
        mp = Path(mp_str)
        run_dir = mp.parent
        am = read_json(mp, default={}) or {}
        recipe_path = run_dir / str(am.get("recipe", "recipe.json"))
        split_manifest_path = run_dir / str(am.get("split_manifest", "split_manifest.json"))
        recipe = read_json(recipe_path, default={}) or {}
        sm = read_json(split_manifest_path, default={}) or {}
        run_tm = resolve_time_meta(am, recipe)
        frame_time = run_tm["frame_time_sec"]
        dataset_id = str(am.get("dataset_id", "dataset"))
        data_label = str(am.get("data_label", "default"))
        run_id = str(am.get("run_id", run_dir.name))
        package_name = str(am.get("package_name", recipe.get("package_name", "")))
        sb = am.get("space_bin", None)
        tb = am.get("time_bin", None)
        try:
            bin_label = f"{int(sb)}x{int(sb)}/t{int(tb)}" if sb is not None and tb is not None else "unknown"
        except Exception:
            bin_label = "unknown"
        input_h5 = str(am.get("input_h5", ""))
        mtime = mp.stat().st_mtime if mp.exists() else 0.0
        for i, s in enumerate(sm.get("splits", []), start=1):
            split_name = str(s.get("split_name", f"split{i:02d}"))
            fit_csv = _safe_rel(run_dir, s.get("fit_csv"))
            grid_meta = _safe_rel(run_dir, s.get("grid_meta"))
            acf_h5 = _safe_rel(run_dir, s.get("acf_h5"))
            if not fit_csv or not grid_meta or not Path(fit_csv).exists() or not Path(grid_meta).exists():
                continue
            short_label = f"{dataset_id} / {data_label} / {bin_label} / {split_name}"
            if package_name:
                short_label += f" / {package_name}"
            tm = resolve_time_meta(am, recipe, s)
            frame_time = tm["frame_time_sec"]
            rows.append({
                "item_id": f"{mp_str}::{split_name}",
                "label": short_label,
                "dataset_id": dataset_id,
                "data_label": data_label,
                "bin_label": bin_label,
                "split": split_name,
                "run_id": run_id,
                "package": package_name,
                "manifest": mp_str,
                "run_dir": str(run_dir),
                "input_h5": input_h5,
                "fit_csv": fit_csv,
                "grid_meta": grid_meta,
                "acf_h5": acf_h5,
                "frame_time_sec": frame_time,
                "effective_frame_time_sec": tm["effective_frame_time_sec"],
                "raw_frame_time_sec": tm["raw_frame_time_sec"],
                "time_bin": tm["time_bin"],
                "frame_time_source": tm["frame_time_source"],
                "has_explicit_effective_frame_time": tm["has_explicit_effective_frame_time"],
                "created_mtime": float(mtime),
            })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["dataset_id", "data_label", "bin_label", "split", "created_mtime"], kind="stable").reset_index(drop=True)
    return df


@st.cache_data(show_spinner=False)
def load_compare_grid_cached(fit_csv: str, grid_meta: str, frame_time_sec_item: float, value_col: str, r2_min: float, fill_zero: bool):
    df0 = load_fit_csv_cached(fit_csv)
    H0, W0, mask0 = load_grid_meta_cached(grid_meta)
    df1 = add_display_columns(df0, frame_time_sec_item)
    if value_col not in df1.columns:
        raise KeyError(f"{value_col} が fit.csv にありません: {Path(fit_csv).name}")
    for c in ["x", "y", value_col, "R²"]:
        if c in df1.columns:
            df1[c] = pd.to_numeric(df1[c], errors="coerce")
    fdf = apply_value_filters(df1, r2_min=float(r2_min))
    grid = build_heatmap_grid(fdf, value_col, H0, W0, mask0, fill_zero=bool(fill_zero))
    vals = finite_values(grid)
    return grid, vals, int(H0), int(W0), int(mask0.sum()), int(len(fdf))


def newest_per_group(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if df.empty:
        return df
    d = df.sort_values("created_mtime", ascending=False, kind="stable")
    return d.drop_duplicates(group_cols, keep="first").sort_index()


def compare_label_from_row(row: pd.Series, compact: bool = True) -> str:
    if compact:
        return f"{row['dataset_id']}\n{row['bin_label']} | {row['split']}"
    return str(row.get("label", "item"))


def prepare_compare_items(rows_df: pd.DataFrame, value_col: str, r2_min: float, fill_zero: bool, label_mode: str) -> tuple[list[dict[str, Any]], list[str]]:
    items: list[dict[str, Any]] = []
    errors: list[str] = []
    compact = label_mode == "compact"
    for _, row in rows_df.iterrows():
        try:
            grid, vals, H0, W0, mask_sum, n_rows = load_compare_grid_cached(
                str(row["fit_csv"]), str(row["grid_meta"]), float(row.get("frame_time_sec", DEFAULT_FRAME_TIME_SEC)),
                value_col, float(r2_min), bool(fill_zero)
            )
            items.append({
                "label": compare_label_from_row(row, compact=compact),
                "full_label": str(row.get("label", "")),
                "dataset_id": str(row.get("dataset_id", "")),
                "data_label": str(row.get("data_label", "")),
                "bin_label": str(row.get("bin_label", "")),
                "split": str(row.get("split", "")),
                "run_id": str(row.get("run_id", "")),
                "grid": grid,
                "vals": vals,
                "H": H0,
                "W": W0,
                "mask_sum": mask_sum,
                "n_rows": n_rows,
                "manifest": str(row.get("manifest", "")),
                "fit_csv": str(row.get("fit_csv", "")),
            })
        except Exception as e:
            errors.append(f"{row.get('label', 'item')}: {e}")
    return items, errors


def _clean_compare_vals(items: list[dict[str, Any]], positive_only: bool = False) -> np.ndarray:
    """Collect finite compare values from all items."""
    all_vals = []
    for it in items:
        vals = np.asarray(it.get("vals", []), dtype=float)
        vals = vals[np.isfinite(vals)]
        if positive_only:
            vals = vals[vals > 0]
        if vals.size:
            all_vals.append(vals)
    if not all_vals:
        return np.asarray([], dtype=float)
    return np.concatenate(all_vals).astype(float, copy=False)


def _valid_range(vmin: float, vmax: float, fallback=(0.0, 1.0)) -> tuple[float, float]:
    try:
        lo = float(vmin)
        hi = float(vmax)
        if not np.isfinite(lo) or not np.isfinite(hi):
            return fallback
        if hi <= lo:
            hi = lo + 1.0
        return lo, hi
    except Exception:
        return fallback


def calc_compare_color_range(
    items: list[dict[str, Any]],
    mode: str,
    manual_vmin: float,
    manual_vmax: float,
    scale: str,
    pct_low: float = 1.0,
    pct_high: float = 99.0,
):
    """Heatmap color range only. Distribution x-axis is calculated separately."""
    fallback = (1e-6, 1.0) if scale == "log" else (0.0, 1.0)
    if mode == "Manual global":
        lo, hi = _valid_range(manual_vmin, manual_vmax, fallback=fallback)
        if scale == "log":
            lo = max(lo, 1e-30)
            hi = max(hi, lo * 10.0)
        return lo, hi

    vals = _clean_compare_vals(items, positive_only=(scale == "log"))
    if vals.size == 0:
        return fallback

    if mode == "Percentile global":
        lo_pct = max(0.0, min(float(pct_low), 100.0))
        hi_pct = max(0.0, min(float(pct_high), 100.0))
        if hi_pct <= lo_pct:
            hi_pct = min(100.0, lo_pct + 1.0)
        return auto_range(vals, lo_pct, hi_pct, fallback=fallback)

    # Auto global and Auto individual both need a global fallback/range for colorbar and metrics.
    return auto_range(vals, 1, 99, fallback=fallback)


def calc_compare_dist_range(
    items: list[dict[str, Any]],
    mode: str,
    heatmap_vmin: float,
    heatmap_vmax: float,
    manual_xmin: float | None,
    manual_xmax: float | None,
    pct_low: float = 1.0,
    pct_high: float = 99.0,
) -> tuple[float | None, float | None]:
    """Distribution x-axis range. Kept independent from heatmap color Min/Max."""
    if mode == "Same as heatmap":
        return _valid_range(heatmap_vmin, heatmap_vmax)
    if mode == "Manual distribution":
        return _valid_range(float(manual_xmin), float(manual_xmax))

    vals = _clean_compare_vals(items, positive_only=False)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return None, None

    if mode == "Auto distribution":
        lo = float(np.nanmin(vals))
        hi = float(np.nanmax(vals))
        return _valid_range(lo, hi)

    # Percentile distribution
    lo_pct = max(0.0, min(float(pct_low), 100.0))
    hi_pct = max(0.0, min(float(pct_high), 100.0))
    if hi_pct <= lo_pct:
        hi_pct = min(100.0, lo_pct + 1.0)
    return auto_range(vals, lo_pct, hi_pct, fallback=(float(np.nanmin(vals)), float(np.nanmax(vals))))

def _save_fig_bytes(fig, fmt: str = "png", dpi: int = 240) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format=fmt, dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    return buf.getvalue()


def _filter_dist_vals(vals: np.ndarray, xmin: float | None, xmax: float | None) -> np.ndarray:
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if xmin is not None and xmax is not None and xmax > xmin:
        vals = vals[(vals >= float(xmin)) & (vals <= float(xmax))]
    return vals


def _apply_dist_axis_style(ax, dist_type: str, xmin: float | None, xmax: float | None, y_mode: str):
    if xmin is not None and xmax is not None and xmax > xmin and dist_type != "Raincloud":
        ax.set_xlim(float(xmin), float(xmax))
    if y_mode == "log density" and dist_type in ["Histogram", "KDE"]:
        ax.set_yscale("log")
    if y_mode == "log count" and dist_type == "Histogram":
        ax.set_yscale("log")


def draw_distribution(
    ax,
    vals: np.ndarray,
    dist_type: str,
    xmin: float | None,
    xmax: float | None,
    bins: int,
    label: str = "",
    y_mode: str = "density",
):
    vals = _filter_dist_vals(vals, xmin, xmax)
    if vals.size == 0:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        ax.set_yticks([])
        return
    hist_density = y_mode in ["density", "log density"]
    if dist_type == "Histogram":
        ax.hist(vals, bins=int(bins), range=(xmin, xmax) if xmin is not None and xmax is not None and xmax > xmin else None, density=hist_density, alpha=0.85)
        ax.set_ylabel("density" if hist_density else "count")
    elif dist_type == "KDE":
        try:
            from scipy.stats import gaussian_kde
            lo = float(xmin) if xmin is not None else float(np.nanpercentile(vals, 1))
            hi = float(xmax) if xmax is not None else float(np.nanpercentile(vals, 99))
            if hi <= lo:
                hi = lo + 1.0
            xs = np.linspace(lo, hi, 300)
            kde = gaussian_kde(vals)
            ys = kde(xs)
            ax.plot(xs, ys, linewidth=1.6)
            ax.fill_between(xs, ys, alpha=0.18)
            ax.set_ylabel("density")
        except Exception:
            ax.hist(vals, bins=int(bins), density=True, alpha=0.85)
            ax.set_ylabel("density")
    elif dist_type == "ECDF":
        xs = np.sort(vals)
        ys = np.arange(1, xs.size + 1, dtype=float) / xs.size
        ax.plot(xs, ys, linewidth=1.6)
        ax.set_ylabel("ECDF")
        ax.set_ylim(0, 1)
    elif dist_type == "Raincloud":
        parts = ax.violinplot([vals], positions=[0.0], vert=False, widths=0.7, showmeans=False, showmedians=False, showextrema=False)
        for pc in parts.get("bodies", []):
            pc.set_alpha(0.35)
        ax.boxplot([vals], positions=[0.0], vert=False, widths=0.18, showfliers=False, patch_artist=True)
        rng = np.random.default_rng(0)
        n = min(vals.size, 800)
        sample = vals if vals.size <= n else rng.choice(vals, size=n, replace=False)
        y = rng.normal(loc=-0.32, scale=0.035, size=sample.size)
        ax.scatter(sample, y, s=3, alpha=0.18)
        ax.set_yticks([])
        ax.set_ylim(-0.55, 0.55)
    if label:
        ax.set_title(label, fontsize=ax.title.get_fontsize())
    _apply_dist_axis_style(ax, dist_type, xmin, xmax, y_mode)


def draw_combined_distribution(
    ax,
    items: list[dict[str, Any]],
    dist_type: str,
    xmin: float | None,
    xmax: float | None,
    bins: int,
    value_col: str,
    y_mode: str = "density",
):
    """Draw one combined distribution panel for all compare items."""
    if dist_type == "None":
        ax.set_axis_off()
        return

    def clean_vals(vals):
        return _filter_dist_vals(vals, xmin, xmax)

    plotted = 0
    hist_density = y_mode in ["density", "log density"]
    if dist_type == "Histogram":
        for it in items:
            vals = clean_vals(it.get("vals", []))
            if vals.size == 0:
                continue
            ax.hist(
                vals,
                bins=int(bins),
                range=(xmin, xmax) if xmin is not None and xmax is not None and xmax > xmin else None,
                density=hist_density,
                histtype="step",
                linewidth=1.5,
                label=str(it.get("label", "item")).replace("\n", " | "),
            )
            plotted += 1
        ax.set_ylabel("density" if hist_density else "count")
    elif dist_type == "KDE":
        try:
            from scipy.stats import gaussian_kde
            lo = float(xmin) if xmin is not None else np.nan
            hi = float(xmax) if xmax is not None else np.nan
            if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                vals_for_range = [clean_vals(it.get("vals", [])) for it in items]
                vals_for_range = [v for v in vals_for_range if v.size]
                allv = np.concatenate(vals_for_range) if vals_for_range else np.asarray([], dtype=float)
                lo, hi = auto_range(allv, 1, 99, fallback=(0.0, 1.0))
            xs = np.linspace(lo, hi, 400)
            rng = np.random.default_rng(0)
            for it in items:
                vals = clean_vals(it.get("vals", []))
                if vals.size < 3:
                    continue
                if vals.size > 60000:
                    vals = rng.choice(vals, size=60000, replace=False)
                kde = gaussian_kde(vals)
                ys = kde(xs)
                ax.plot(xs, ys, linewidth=1.7, label=str(it.get("label", "item")).replace("\n", " | "))
                plotted += 1
            ax.set_ylabel("density")
        except Exception:
            for it in items:
                vals = clean_vals(it.get("vals", []))
                if vals.size:
                    ax.hist(vals, bins=int(bins), density=True, histtype="step", linewidth=1.5, label=str(it.get("label", "item")).replace("\n", " | "))
                    plotted += 1
            ax.set_ylabel("density")
    elif dist_type == "ECDF":
        for it in items:
            vals = clean_vals(it.get("vals", []))
            if vals.size == 0:
                continue
            xs = np.sort(vals)
            ys = np.arange(1, xs.size + 1, dtype=float) / xs.size
            ax.plot(xs, ys, linewidth=1.7, label=str(it.get("label", "item")).replace("\n", " | "))
            plotted += 1
        ax.set_ylabel("ECDF")
        ax.set_ylim(0, 1)
    elif dist_type == "Raincloud":
        data = []
        labels = []
        rng = np.random.default_rng(0)
        for it in items:
            vals = clean_vals(it.get("vals", []))
            if vals.size == 0:
                continue
            if vals.size > 50000:
                vals = rng.choice(vals, size=50000, replace=False)
            data.append(vals)
            labels.append(str(it.get("label", "item")).replace("\n", " | "))
        if data:
            positions = np.arange(len(data), 0, -1)
            parts = ax.violinplot(data, positions=positions, vert=False, widths=0.75, showmeans=False, showmedians=False, showextrema=False)
            for pc in parts.get("bodies", []):
                pc.set_alpha(0.32)
            ax.boxplot(data, positions=positions, vert=False, widths=0.18, showfliers=False, patch_artist=True)
            ax.set_yticks(positions)
            ax.set_yticklabels(labels, fontsize=max(6, plt.rcParams.get("font.size", 9) - 1))
            plotted = len(data)
        ax.set_ylabel("")
    else:
        ax.text(0.5, 0.5, "unknown distribution", ha="center", va="center", transform=ax.transAxes)

    _apply_dist_axis_style(ax, dist_type, xmin, xmax, y_mode)
    ax.set_xlabel(value_col)
    if plotted:
        if dist_type != "Raincloud":
            ax.legend(fontsize=max(6, plt.rcParams.get("font.size", 9) - 1), frameon=False, ncols=1, loc="best")
    else:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)

def _imshow_origin_from_y_axis_mode(y_axis_mode: str) -> str:
    return "lower" if y_axis_mode == "Cartesian: y=0 bottom" else "upper"


def _format_map_axis(ax, y_axis_mode: str, show_map_axes: bool):
    if y_axis_mode == "Invert current":
        ax.invert_yaxis()
    if show_map_axes:
        ax.set_xlabel("x", labelpad=1)
        ax.set_ylabel("y", labelpad=1)
    else:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")


def make_compare_figure(
    items: list[dict[str, Any]],
    value_col: str,
    cmap_name: str,
    reverse_cmap: bool,
    scale: str,
    color_mode: str,
    manual_vmin: float,
    manual_vmax: float,
    distribution: str,
    layout: str,
    bins: int,
    ncols: int,
    fig_width: float,
    map_height: float,
    font_size: float,
    paper_labels: bool,
    heatmap_pct_low: float = 1.0,
    heatmap_pct_high: float = 99.0,
    dist_x_mode: str = "Same as heatmap",
    dist_manual_xmin: float | None = None,
    dist_manual_xmax: float | None = None,
    dist_pct_low: float = 1.0,
    dist_pct_high: float = 99.0,
    dist_y_mode: str = "density",
    y_axis_mode: str = "Image: y=0 top",
    show_map_axes: bool = False,
):
    """Compare figure with separated heatmap color range and distribution x-axis range."""
    if not items:
        raise ValueError("Compare items are empty.")
    plt.rcParams.update({"font.size": float(font_size)})
    n = len(items)
    ncols = max(1, min(int(ncols), n))
    nrows = int(math.ceil(n / ncols))
    cmap = get_cmap(cmap_name, reverse_cmap)
    global_vmin, global_vmax = calc_compare_color_range(
        items, color_mode, manual_vmin, manual_vmax, scale,
        pct_low=float(heatmap_pct_low), pct_high=float(heatmap_pct_high),
    )
    if scale == "log":
        global_vmin = max(global_vmin, 1e-30)

    dist_xmin, dist_xmax = calc_compare_dist_range(
        items,
        dist_x_mode,
        global_vmin,
        global_vmax,
        dist_manual_xmin,
        dist_manual_xmax,
        pct_low=float(dist_pct_low),
        pct_high=float(dist_pct_high),
    )

    def item_range(it):
        if color_mode in ["Auto global", "Manual global", "Percentile global"]:
            return global_vmin, global_vmax
        vals = np.asarray(it.get("vals", []), dtype=float)
        vals = vals[np.isfinite(vals)]
        if scale == "log":
            vals = vals[vals > 0]
        return auto_range(vals, 1, 99, fallback=(global_vmin, global_vmax))

    use_dist = distribution != "None" and layout != "Maps only"
    separate_dist = use_dist and layout == "Maps + separate distributions"
    combined_dist = use_dist and layout == "Maps + combined distribution"
    origin = _imshow_origin_from_y_axis_mode(y_axis_mode)

    if separate_dist:
        total_rows = nrows * 2
        height_ratios = []
        for _ in range(nrows):
            height_ratios.extend([1.0, 0.42])
        fig_h = max(2.4, nrows * (float(map_height) + 1.0) + 0.35)
    else:
        total_rows = nrows + (1 if combined_dist else 0)
        height_ratios = [1.0] * nrows + ([0.78] if combined_dist else [])
        dist_height = 2.2 if combined_dist else 0.0
        fig_h = max(2.4, nrows * float(map_height) + dist_height + 0.35)

    fig_w = max(float(fig_width), ncols * 2.5)
    fig = plt.figure(figsize=(fig_w, fig_h), constrained_layout=False)
    gs = fig.add_gridspec(total_rows, ncols, height_ratios=height_ratios, hspace=0.42, wspace=0.20)

    map_axes = []
    first_im = None
    for i in range(nrows * ncols):
        r = i // ncols
        c = i % ncols
        map_row = r * 2 if separate_dist else r
        ax = fig.add_subplot(gs[map_row, c])
        if i >= n:
            ax.set_axis_off()
            if separate_dist:
                dax_empty = fig.add_subplot(gs[map_row + 1, c])
                dax_empty.set_axis_off()
            continue
        it = items[i]
        map_axes.append(ax)
        grid = np.array(it["grid"], dtype=float, copy=True)
        vmin, vmax = item_range(it)
        if scale == "log":
            finite = np.isfinite(grid)
            grid[finite & (grid <= 0)] = max(float(vmin), 1e-30)
            vmin = max(float(vmin), 1e-30)
        im = ax.imshow(grid, origin=origin, interpolation="nearest", cmap=cmap, norm=make_norm(float(vmin), float(vmax), scale=scale))
        if first_im is None:
            first_im = im
        letter = f"{chr(65+i)}. " if paper_labels and i < 26 else ""
        title = str(it.get("label", "item")).replace("\n", " | ")
        ax.set_title(letter + title, fontsize=float(font_size), loc="left")
        _format_map_axis(ax, y_axis_mode, show_map_axes)

        if separate_dist:
            dax = fig.add_subplot(gs[map_row + 1, c])
            draw_distribution(
                dax,
                it.get("vals", []),
                distribution,
                dist_xmin,
                dist_xmax,
                bins,
                label="",
                y_mode=dist_y_mode,
            )
            dax.set_xlabel(value_col, labelpad=1)
            dax.tick_params(axis="both", labelsize=max(6, float(font_size) - 1.5))

    if first_im is not None and map_axes and color_mode in ["Auto global", "Manual global", "Percentile global"]:
        cbar = fig.colorbar(first_im, ax=map_axes, fraction=0.028, pad=0.012)
        cbar.set_label(value_col)

    if combined_dist:
        dax = fig.add_subplot(gs[nrows, :])
        draw_combined_distribution(dax, items, distribution, dist_xmin, dist_xmax, bins, value_col, y_mode=dist_y_mode)
        dax.set_title("Combined distribution", loc="left", fontsize=float(font_size))

    fig.tight_layout()
    return fig


# =============================================================================
# Sidebar: run browser
# =============================================================================

st.title(APP_TITLE)
st.caption("ACF/Fit Managerの `analysis_manifest.json` を入口にして可視化します。")

with st.sidebar:
    st.header("Project")
    analysis_root = st.text_input(
        "Analysis root",
        value=st.session_state.get("viewer_analysis_root", ""),
        placeholder=r"例: D:\analysis_project",
        key="viewer_root_input",
    )
    analysis_root = clean_path_text(analysis_root)
    st.session_state["viewer_analysis_root"] = analysis_root
    if "viewer_refresh_token" not in st.session_state:
        st.session_state["viewer_refresh_token"] = 0
    b1, b2 = st.columns(2)
    with b1:
        if st.button("Refresh", key="refresh_runs"):
            st.session_state["viewer_refresh_token"] += 1
    with b2:
        if st.button("Clear cache", key="clear_run_cache"):
            st.cache_data.clear()
            st.session_state["viewer_refresh_token"] += 1

    if analysis_root and Path(analysis_root).exists():
        manifests = find_run_manifests(analysis_root, st.session_state.get("viewer_refresh_token", 0))
    else:
        manifests = []

    st.metric("ACF/Fit runs", len(manifests))

selected_manifest: Path | None = None
run_catalog = pd.DataFrame()
run_notes: dict[str, Any] = {}

if manifests:
    run_notes = load_run_notes(analysis_root)
    run_catalog = build_run_catalog_cached(tuple(str(p) for p in manifests))
    run_catalog = apply_run_notes_to_catalog(run_catalog, run_notes)

    with st.sidebar:
        st.subheader("Run Browser")
        f1, f2 = st.columns(2)
        with f1:
            latest_only_sidebar = st.checkbox("Latest only", value=True, key="rb_latest_only")
        with f2:
            favorites_only_sidebar = st.checkbox("★ only", value=False, key="rb_fav_only")

        df_browser = run_catalog.copy()
        if latest_only_sidebar:
            df_browser = newest_runs_per_group(df_browser)
        if favorites_only_sidebar and not df_browser.empty:
            df_browser = df_browser[df_browser["favorite"] == True]

        datasets = sorted(df_browser["dataset_id"].dropna().unique().tolist()) if not df_browser.empty else []
        bins = sorted(df_browser["bin_label"].dropna().unique().tolist()) if not df_browser.empty else []
        packages = sorted([x for x in df_browser["package"].dropna().unique().tolist() if str(x)]) if not df_browser.empty else []
        statuses = sorted([x for x in df_browser["status"].dropna().unique().tolist() if str(x)]) if not df_browser.empty else []

        dataset_filter_rb = st.selectbox("Dataset", ["all"] + datasets, index=0, key="rb_dataset")
        bin_filter_rb = st.selectbox("Bin", ["all"] + bins, index=0, key="rb_bin")
        package_filter_rb = st.selectbox("Package", ["all"] + packages, index=0, key="rb_package")
        status_filter_rb = st.selectbox("Status", ["all"] + statuses, index=0, key="rb_status")

        if dataset_filter_rb != "all":
            df_browser = df_browser[df_browser["dataset_id"] == dataset_filter_rb]
        if bin_filter_rb != "all":
            df_browser = df_browser[df_browser["bin_label"] == bin_filter_rb]
        if package_filter_rb != "all":
            df_browser = df_browser[df_browser["package"] == package_filter_rb]
        if status_filter_rb != "all":
            df_browser = df_browser[df_browser["status"] == status_filter_rb]

        st.caption(f"Showing {len(df_browser)} runs")
        if df_browser.empty:
            st.info("条件に合うrunがありません。")
        else:
            manifest_options = df_browser["manifest"].tolist()
            current_manifest = st.session_state.get("selected_run_manifest", manifest_options[0])
            default_idx = manifest_options.index(current_manifest) if current_manifest in manifest_options else 0
            label_map = dict(zip(df_browser["manifest"], df_browser["display_label"]))
            selected_manifest_str = st.selectbox(
                "Select run",
                manifest_options,
                index=default_idx,
                format_func=lambda x: label_map.get(x, str(x)),
                key="run_browser_select",
            )
            st.session_state["selected_run_manifest"] = selected_manifest_str
            selected_manifest = Path(selected_manifest_str)

            sel_row = df_browser[df_browser["manifest"] == selected_manifest_str].iloc[0]
            _time_caption = time_meta_caption(sel_row)
            st.markdown(
                f"""
<div class="card">
<b>{sel_row['dataset_id']} / {sel_row['bin_label']} / {sel_row['package_short']}</b><br>
<span class="small-caption">{sel_row['created']} | {sel_row['status']} | split={sel_row['n_splits']} | {_time_caption}</span><br>
<span class="small-caption">note: {sel_row.get('run_note', '') or '-'}</span>
</div>
""",
                unsafe_allow_html=True,
            )

            with st.expander("Edit note / favorite", expanded=False):
                note_key = f"note_{abs(hash(selected_manifest_str))}"
                fav_key = f"fav_{abs(hash(selected_manifest_str))}"
                note_value = str(sel_row.get("run_note", ""))
                fav_value = bool(sel_row.get("favorite", False))
                new_note = st.text_input("Run note", value=note_value, key=note_key)
                new_fav = st.checkbox("Favorite", value=fav_value, key=fav_key)
                if st.button("Save note", key=f"save_note_{abs(hash(selected_manifest_str))}"):
                    run_notes[selected_manifest_str] = {"note": new_note, "favorite": bool(new_fav)}
                    save_run_notes(analysis_root, run_notes)
                    st.session_state["viewer_refresh_token"] = st.session_state.get("viewer_refresh_token", 0) + 1
                    st.success("Saved")

            with st.expander("Run table", expanded=False):
                show_cols = ["favorite", "dataset_id", "data_label", "bin_label", "package_short", "status", "created", "run_note"]
                st.dataframe(df_browser[show_cols], use_container_width=True, hide_index=True)
else:
    with st.sidebar:
        st.info("Analysis root内に `03_analysis/acf_fit/run_*/analysis_manifest.json` が見つかりません。")

if not selected_manifest:
    st.stop()

bundle = load_run_bundle(Path(selected_manifest))
splits = get_splits(bundle)
if not splits:
    st.error("split_manifest.json からsplit情報を読めませんでした。")
    st.stop()

# Short run summary
am = bundle.analysis_manifest
recipe = bundle.recipe
_selected_note = ""
_selected_fav = False
try:
    _n = run_notes.get(str(selected_manifest), {}) if isinstance(run_notes, dict) else {}
    _selected_note = str(_n.get("note", "")) if isinstance(_n, dict) else ""
    _selected_fav = bool(_n.get("favorite", False)) if isinstance(_n, dict) else False
except Exception:
    pass
_star = "★ " if _selected_fav else ""
st.markdown(
    f"""
<div class="card">
<b>{_star}{am.get('dataset_id', '')} / {am.get('data_label', '')} / {_bin_label_from_manifest(am)} / {_short_package_name(str(am.get('package_name', recipe.get('package_name', ''))))}</b><br>
<span class="small-caption">run: {am.get('run_id', bundle.run_dir.name)} | status: {am.get('status', '')} | input: {safe_file_label(am.get('input_h5', ''))}</span><br>
<span class="small-caption">note: {_selected_note or '-'}</span>
</div>
""",
    unsafe_allow_html=True,
)

# =============================================================================
# Tabs
# =============================================================================

map_tab, pixel_tab, compare_tab, export_tab, advanced_tab = st.tabs(["Map", "Pixel", "Compare", "Export", "Advanced"])


# Common split selection
split_names = [s.split_name for s in splits]
# Keep existing split widgets synchronized before resolving time metadata.
for _split_key in ["map_split_select", "pixel_split_select"]:
    if st.session_state.get(_split_key) in split_names:
        st.session_state["viewer_split_name"] = st.session_state[_split_key]
if "viewer_split_name" not in st.session_state or st.session_state["viewer_split_name"] not in split_names:
    st.session_state["viewer_split_name"] = split_names[0]

selected_split_name = st.session_state["viewer_split_name"]
selected_split = next(s for s in splits if s.split_name == selected_split_name)

selected_split_meta = selected_split_manifest_entry(bundle, selected_split_name)
time_meta = resolve_time_meta(bundle.analysis_manifest, bundle.recipe, selected_split_meta)
frame_time_sec = float(time_meta["frame_time_sec"])

st.caption("Time metadata: " + time_meta_caption(time_meta))
if not time_meta.get("has_explicit_effective_frame_time", False):
    st.warning(
        "effective_frame_time_sec がmanifestに明示されていません。"
        "このrunは frame_time_sec fallback で表示しています。"
        "t1以外のbinでは、TimeMeta対応版DXB_01で再解析してください。"
    )


# =============================================================================
# Load selected split data helper
# =============================================================================

@st.cache_data(show_spinner=False)
def split_data_payload(fit_csv: str, grid_meta: str, frame_time: float):
    df0 = load_fit_csv_cached(fit_csv)
    H0, W0, mask0 = load_grid_meta_cached(grid_meta)
    df1 = add_display_columns(df0, frame_time)
    return df1, H0, W0, mask0


def load_current_split(split: SplitFiles):
    if split.fit_csv is None:
        raise RuntimeError(f"{split.split_name} に fit.csv がありません。ACF onlyのrunです。")
    return split_data_payload(str(split.fit_csv), str(split.grid_meta), float(frame_time_sec))


# =============================================================================
# Map tab
# =============================================================================

with map_tab:
    top = st.columns([1.1, 1.1, 1.1, 0.9, 0.9, 0.9])
    with top[0]:
        selected_split_name = st.selectbox("Split", split_names, index=split_names.index(st.session_state["viewer_split_name"]), key="map_split_select")
        st.session_state["viewer_split_name"] = selected_split_name
        selected_split = next(s for s in splits if s.split_name == selected_split_name)
    try:
        df, H, W, mask = load_current_split(selected_split)
    except Exception as e:
        st.error(str(e))
        st.stop()

    with top[1]:
        show_frame_cols = st.checkbox("frame列", value=False, key="map_show_frame")
    with top[2]:
        show_detail_cols = st.checkbox("詳細列", value=False, key="map_show_detail")
    cols = candidate_display_columns(df, show_frame_cols=show_frame_cols, show_detail_cols=show_detail_cols)
    if not cols:
        st.error("表示できる数値列がありません。")
        st.stop()
    default_col = "log₁₀ Γ [s⁻¹]" if "log₁₀ Γ [s⁻¹]" in cols else cols[0]
    with top[3]:
        display_col = st.selectbox("Value", cols, index=cols.index(default_col), key="map_display_col")
    with top[4]:
        r2_min = st.number_input("R² min", value=0.0, step=0.05, format="%.3f", key="map_r2min")
    with top[5]:
        fig_h = st.number_input("高さ", value=float(DEFAULT_FIG_H), min_value=2.0, max_value=9.0, step=0.2, key="map_fig_h")

    settings_cols = st.columns([1, 1, 1, 1, 1, 1])
    with settings_cols[0]:
        cmap_name = st.selectbox("Cmap", list(CMAP_PRESETS.keys()), index=list(CMAP_PRESETS.keys()).index(DEFAULT_CMAP), key="map_cmap")
    with settings_cols[1]:
        reverse_cmap = st.checkbox("反転", value=False, key="map_reverse")
    with settings_cols[2]:
        color_scale = st.selectbox("Scale", ["linear", "log"], index=0, key="map_scale")
    with settings_cols[3]:
        fill_zero = st.checkbox("未選択0", value=False, key="map_fill_zero")
    with settings_cols[4]:
        manual_range = st.checkbox("範囲手動", value=False, key="map_manual_range")
    with settings_cols[5]:
        hist_sync = st.checkbox("Hist同期", value=True, key="map_hist_sync")

    # Optional value filtering
    with st.expander("Value filter", expanded=False):
        vals0 = pd.to_numeric(df[display_col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().values
        vf_auto_min, vf_auto_max = auto_range(vals0, 0, 100)
        vf_enabled = st.checkbox("Value filter ON", value=False, key="map_value_filter_on")
        vf1, vf2 = st.columns(2)
        with vf1:
            vf_min = st.number_input("Value min", value=float(vf_auto_min), format="%.6g", key="map_vf_min")
        with vf2:
            vf_max = st.number_input("Value max", value=float(vf_auto_max), format="%.6g", key="map_vf_max")

    df_num = df.copy()
    for c in ["x", "y", display_col, "R²"]:
        if c in df_num.columns:
            df_num[c] = pd.to_numeric(df_num[c], errors="coerce")
    filtered = apply_value_filters(
        df_num,
        r2_min=float(r2_min),
        value_col=display_col if vf_enabled else None,
        vmin=vf_min if vf_enabled else None,
        vmax=vf_max if vf_enabled else None,
    )
    grid = build_heatmap_grid(filtered, display_col, H, W, mask, fill_zero=fill_zero)
    plot_vals = finite_values(grid)

    if color_scale == "log":
        positive = plot_vals[plot_vals > 0]
        range_vals = positive if positive.size else plot_vals
        fallback = (1e-6, 1.0)
    else:
        range_vals = plot_vals
        fallback = (0.0, 1.0)
    auto_vmin, auto_vmax = auto_range(range_vals, 1, 99, fallback=fallback)

    if manual_range:
        cr1, cr2 = st.columns(2)
        with cr1:
            map_vmin = st.number_input("Map min", value=float(auto_vmin), format="%.6g", key="map_vmin")
        with cr2:
            map_vmax = st.number_input("Map max", value=float(auto_vmax), format="%.6g", key="map_vmax")
    else:
        map_vmin, map_vmax = auto_vmin, auto_vmax

    if color_scale == "log":
        grid_plot = grid.copy()
        finite = np.isfinite(grid_plot)
        grid_plot[finite & (grid_plot <= 0)] = max(float(map_vmin), 1e-30)
        map_vmin = max(float(map_vmin), 1e-30)
    else:
        grid_plot = grid

    left, right = st.columns([1.35, 1.0])
    with left:
        fig = make_heatmap_fig(
            grid_plot,
            title=f"{display_col} | {selected_split.split_name}",
            cmap_name=cmap_name,
            reverse_cmap=reverse_cmap,
            vmin=float(map_vmin),
            vmax=float(map_vmax),
            scale=color_scale,
            label=display_col,
            fig_h=float(fig_h),
            show_axes=True,
        )
        st.pyplot(fig, use_container_width=True)
    with right:
        vals = pd.to_numeric(filtered[display_col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().values
        if hist_sync:
            hist_vmin, hist_vmax = float(map_vmin), float(map_vmax)
        else:
            ha, hb = auto_range(vals, 1, 99)
            hc1, hc2 = st.columns(2)
            with hc1:
                hist_vmin = st.number_input("Hist min", value=float(ha), format="%.6g", key="hist_min")
            with hc2:
                hist_vmax = st.number_input("Hist max", value=float(hb), format="%.6g", key="hist_max")
        hist_bins = st.number_input("Bins", value=60, min_value=5, max_value=500, step=5, key="hist_bins")
        hfig = make_hist_fig(vals, title=display_col, xlabel=display_col, vmin=float(hist_vmin), vmax=float(hist_vmax), bins=int(hist_bins))
        st.pyplot(hfig, use_container_width=True)
        m1, m2, m3 = st.columns(3)
        m1.metric("Rows", len(filtered))
        m2.metric("Valid px", int(mask.sum()))
        m3.metric("NaN", int(np.isnan(grid).sum()))

    st.session_state["viewer_last_map"] = {
        "split": selected_split.split_name,
        "display_col": display_col,
        "r2_min": float(r2_min),
        "map_vmin": float(map_vmin),
        "map_vmax": float(map_vmax),
        "cmap": cmap_name,
        "reverse": bool(reverse_cmap),
        "scale": color_scale,
    }


# =============================================================================
# Pixel tab
# =============================================================================

with pixel_tab:
    # Reload using session split to keep tabs synchronized
    selected_split = next(s for s in splits if s.split_name == st.session_state.get("viewer_split_name", split_names[0]))
    try:
        df, H, W, mask = load_current_split(selected_split)
    except Exception as e:
        st.error(str(e))
        st.stop()

    x_arr, y_arr, n_lags, acf_attrs = load_acf_coords_cached(str(selected_split.acf_h5))
    idx_map = coord_to_index(x_arr, y_arr)

    ctop = st.columns([1.0, 1.0, 1.0, 1.0])
    with ctop[0]:
        split_for_pixel = st.selectbox("Split", split_names, index=split_names.index(selected_split.split_name), key="pixel_split_select")
        st.session_state["viewer_split_name"] = split_for_pixel
        selected_split = next(s for s in splits if s.split_name == split_for_pixel)
    with ctop[1]:
        neighborhood = st.selectbox("I(t)", ["single", "3×3 mean", "5×5 mean", "7×7 mean"], index=0, key="pixel_neighborhood")
    with ctop[2]:
        acf_xscale = st.selectbox("ACF x", ["log", "linear"], index=0, key="acf_xscale")
    with ctop[3]:
        acf_y_mode = st.selectbox("ACF y", ["ACF - 1", "ACF"], index=0, key="acf_y_mode")

    df, H, W, mask = load_current_split(selected_split)
    x_arr, y_arr, n_lags, acf_attrs = load_acf_coords_cached(str(selected_split.acf_h5))
    idx_map = coord_to_index(x_arr, y_arr)

    if "pixel_xy_text" not in st.session_state:
        # Pick first valid coordinate from fit csv if possible
        if len(df) > 0 and "x" in df.columns and "y" in df.columns:
            st.session_state["pixel_xy_text"] = f"{int(df.iloc[0]['x'])},{int(df.iloc[0]['y'])}"
        else:
            st.session_state["pixel_xy_text"] = "0,0"

    with st.form("pixel_xy_form"):
        xy_text = st.text_input("Coordinate x,y", value=st.session_state["pixel_xy_text"], key="pixel_xy_input")
        submitted = st.form_submit_button("Update")
        if submitted:
            st.session_state["pixel_xy_text"] = xy_text

    x0, y0 = parse_xy(st.session_state.get("pixel_xy_text", "0,0"), default=(0, 0))
    x, y = nearest_valid_xy(mask, x0, y0)
    if (x, y) != (x0, y0):
        st.info(f"指定座標はinvalidだったため、最寄りvalid pixelに補正しました: ({x0},{y0}) → ({x},{y})")

    if (x, y) not in idx_map:
        st.error(f"ACF座標に ({x},{y}) がありません。")
        st.stop()
    pix_idx = idx_map[(x, y)]

    row = df[(pd.to_numeric(df["x"], errors="coerce") == x) & (pd.to_numeric(df["y"], errors="coerce") == y)]
    fit_row = row.iloc[0] if len(row) else None

    # metric cards
    mcols = st.columns(6)
    def fmt_metric(v, fmt=".4g"):
        try:
            if pd.isna(v):
                return "nan"
            return format(float(v), fmt)
        except Exception:
            return "-"

    mcols[0].metric("x,y", f"{x},{y}")
    if fit_row is not None:
        mcols[1].metric("A", fmt_metric(fit_row.get("A")))
        mcols[2].metric("Γ [s⁻¹]", fmt_metric(fit_row.get("Γ [s⁻¹]")))
        mcols[3].metric("τ [s]", fmt_metric(fit_row.get("τ [s]")))
        mcols[4].metric("g", fmt_metric(fit_row.get("g")))
        mcols[5].metric("R²", fmt_metric(fit_row.get("R²"), ".3f"))
    else:
        for i, lab in enumerate(["A", "Γ [s⁻¹]", "τ [s]", "g", "R²"], start=1):
            mcols[i].metric(lab, "-")

    # source image H5
    source_h5 = get_source_images_h5(bundle, selected_split)
    with st.expander("Path / manual source H5", expanded=False):
        st.write("ACF H5:", str(selected_split.acf_h5))
        st.write("Fit CSV:", str(selected_split.fit_csv))
        st.write("Source images.h5:", source_h5 or "not found")
        manual_source = st.text_input("Manual source images.h5", value=source_h5 or "", key="manual_source_h5")
    source_h5 = clean_path_text(manual_source)

    # load traces
    acf = load_acf_trace_cached(str(selected_split.acf_h5), int(pix_idx))
    lag = np.arange(acf.size, dtype=float)
    tau = lag * float(frame_time_sec) if frame_time_sec else lag
    tau_label = "time [s]" if frame_time_sec else "lag [frame]"

    radius = {"single": 0, "3×3 mean": 1, "5×5 mean": 2, "7×7 mean": 3}[neighborhood]
    trace = None
    trace_t = None
    if source_h5 and Path(source_h5).exists():
        try:
            trace = load_image_trace_cached(source_h5, selected_split.frame_start, selected_split.frame_end, x, y, radius=radius)
            trace_t = np.arange(trace.size, dtype=float) * float(frame_time_sec) if frame_time_sec else np.arange(trace.size, dtype=float)
        except Exception as e:
            st.warning(f"I(t)を読めませんでした: {e}")
    else:
        st.warning("source images.h5 が見つかりません。I(t)は表示できません。")

    # plots
    p1, p2 = st.columns([1.0, 1.0])
    with p1:
        fig, ax = plt.subplots(figsize=(5.8, 3.6))
        if trace is not None:
            ax.plot(trace_t, trace, linewidth=0.9)
        ax.set_title(f"I(t) | {neighborhood}")
        ax.set_xlabel("time [s]" if frame_time_sec else "frame")
        ax.set_ylabel("intensity")
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
    with p2:
        fig, ax = plt.subplots(figsize=(5.8, 3.6))
        start_idx = 1  # hide lag0 by default
        xx = tau[start_idx:]
        yy_raw = acf[start_idx:]
        yy = yy_raw - 1.0 if acf_y_mode == "ACF - 1" else yy_raw
        ax.plot(xx, yy, marker="o", markersize=2.5, linewidth=0.8, label="ACF")
        if fit_row is not None and all(k in fit_row for k in ["A", "Gamma", "g"]):
            try:
                A = float(fit_row["A"])
                G = float(fit_row["Gamma"])
                gg = float(fit_row["g"])
                # fit tau uses seconds when frame_time_sec was set by Manager. Otherwise frames.
                fit_y_raw = model_stretched(xx, A, G, gg)
                fit_y = fit_y_raw - 1.0 if acf_y_mode == "ACF - 1" else fit_y_raw
                ax.plot(xx, fit_y, linewidth=1.5, label="fit")
            except Exception:
                pass
        if acf_xscale == "log":
            ax.set_xscale("log")
        ax.set_title("ACF + fit")
        ax.set_xlabel(tau_label)
        ax.set_ylabel(acf_y_mode)
        ax.legend(loc="best")
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)

    # locator map
    st.markdown("#### Locator")
    loc_cols = st.columns([1.0, 1.0])
    with loc_cols[0]:
        # Use current map display if available, else R2
        loc_col = st.session_state.get("viewer_last_map", {}).get("display_col", "R²")
        if loc_col not in df.columns:
            loc_col = "R²" if "R²" in df.columns else candidate_display_columns(df)[0]
        grid_loc = build_heatmap_grid(df, loc_col, H, W, mask, fill_zero=False)
        vals = finite_values(grid_loc)
        lvmin, lvmax = auto_range(vals, 1, 99)
        loc_fig = make_heatmap_fig(
            grid_loc,
            title=f"{loc_col}",
            cmap_name=st.session_state.get("viewer_last_map", {}).get("cmap", DEFAULT_CMAP),
            reverse_cmap=st.session_state.get("viewer_last_map", {}).get("reverse", False),
            vmin=lvmin,
            vmax=lvmax,
            scale="linear",
            label=loc_col,
            fig_h=3.2,
            show_axes=True,
            marker_xy=(x, y),
        )
        st.pyplot(loc_fig, use_container_width=True)
    with loc_cols[1]:
        st.write("Selected pixel")
        st.code(f"x={x}, y={y}\nframe range=[{selected_split.frame_start}, {selected_split.frame_end})\npixel_index={pix_idx}")

    # save last pixel data for export
    st.session_state["viewer_last_pixel"] = {
        "x": int(x),
        "y": int(y),
        "split": selected_split.split_name,
        "acf_h5": str(selected_split.acf_h5),
        "source_h5": source_h5,
        "frame_start": int(selected_split.frame_start),
        "frame_end": int(selected_split.frame_end),
        "neighborhood": neighborhood,
    }


# =============================================================================
# Compare tab
# =============================================================================

with compare_tab:
    st.subheader("Compare")
    st.caption("複数run/splitを、共通色軸のmapグリッド + 1つの分布図で比較します。")

    compare_df = build_compare_index_cached(tuple(str(p) for p in manifests)) if manifests else pd.DataFrame()
    if compare_df.empty:
        st.info("比較できるACF/Fit runが見つかりません。")
    else:
        # --- 1. Select candidates -------------------------------------------------
        st.markdown("#### 1. Select")
        top = st.columns([0.9, 1.0, 1.0, 1.0, 0.8, 0.8])
        with top[0]:
            compare_mode = st.radio("Mode", ["Dataset", "Bin", "Split", "Custom"], horizontal=True, key="cmp_mode3")
        with top[1]:
            latest_only = st.checkbox("最新runだけ", value=True, key="cmp_latest_only3")
        with top[2]:
            hide_1x1 = st.checkbox("1x1/t1を隠す", value=True, key="cmp_hide_1x1_3")
        with top[3]:
            r2_min_cmp = st.number_input("R² min", value=0.0, min_value=-1.0, max_value=1.0, step=0.05, format="%.3f", key="cmp_r2_min3")
        with top[4]:
            label_mode = st.selectbox("Label", ["compact", "full"], index=0, key="cmp_label_mode3")
        with top[5]:
            max_default = st.number_input("初期選択", min_value=1, max_value=20, value=4, step=1, key="cmp_default_n3")

        df_cmp = compare_df.copy()
        if hide_1x1:
            df_cmp = df_cmp[df_cmp["bin_label"] != "1x1/t1"]
        if latest_only:
            df_cmp = newest_per_group(df_cmp, ["dataset_id", "data_label", "bin_label", "split", "package"])

        all_datasets = sorted(df_cmp["dataset_id"].dropna().unique().tolist())
        all_bins = sorted(df_cmp["bin_label"].dropna().unique().tolist())
        all_splits = sorted(df_cmp["split"].dropna().unique().tolist())
        all_packages = sorted([x for x in df_cmp["package"].dropna().unique().tolist() if str(x)])

        fcols = st.columns([1, 1, 1, 1])
        with fcols[0]:
            dataset_filter = st.selectbox("Dataset", ["all"] + all_datasets, index=0, key="cmp_dataset_filter3")
        with fcols[1]:
            default_bin_idx = all_bins.index("10x10/t1") + 1 if "10x10/t1" in all_bins else 0
            bin_filter = st.selectbox("Bin", ["all"] + all_bins, index=default_bin_idx if compare_mode == "Dataset" else 0, key="cmp_bin_filter3")
        with fcols[2]:
            split_default = 1 if compare_mode in ["Dataset", "Bin"] and all_splits else 0
            split_filter = st.selectbox("Split", ["all"] + all_splits, index=split_default, key="cmp_split_filter3")
        with fcols[3]:
            package_filter = st.selectbox("Package", ["all"] + all_packages, index=0, key="cmp_package_filter3")

        filtered_cmp = df_cmp.copy()
        if dataset_filter != "all":
            filtered_cmp = filtered_cmp[filtered_cmp["dataset_id"] == dataset_filter]
        if bin_filter != "all":
            filtered_cmp = filtered_cmp[filtered_cmp["bin_label"] == bin_filter]
        if split_filter != "all":
            filtered_cmp = filtered_cmp[filtered_cmp["split"] == split_filter]
        if package_filter != "all":
            filtered_cmp = filtered_cmp[filtered_cmp["package"] == package_filter]

        mode_help = {
            "Dataset": "同じBin / splitで、dataset違いを横並び比較します。",
            "Bin": "同じdataset / splitで、bin違いを比較します。",
            "Split": "同じdataset / binで、split違いを比較します。",
            "Custom": "任意のrun/splitを自由に選びます。",
        }
        st.caption(mode_help.get(compare_mode, ""))

        candidate_options = filtered_cmp["item_id"].tolist()
        candidate_map = {row["item_id"]: row["label"] for _, row in filtered_cmp.iterrows()}
        default_selection = candidate_options[: min(int(max_default), len(candidate_options))]
        selected_ids = st.multiselect(
            "Compare items",
            candidate_options,
            default=default_selection,
            format_func=lambda x: candidate_map.get(x, x),
            key="cmp_items_select3",
        )
        selected_rows = filtered_cmp[filtered_cmp["item_id"].isin(selected_ids)].copy()
        st.caption(f"Selected: {len(selected_rows)} / Candidates: {len(filtered_cmp)}")
        with st.expander("Selected item details", expanded=False):
            show_cols = ["dataset_id", "data_label", "bin_label", "split", "package", "run_id", "fit_csv"]
            st.dataframe(selected_rows[show_cols], use_container_width=True, hide_index=True)

        # --- 2. Display settings --------------------------------------------------
        st.markdown("#### 2. Display settings")

        with st.expander("2-A. Compare target / value", expanded=True):
            vc = st.columns([1.2, 1.0, 1.0])
            with vc[0]:
                value_col_cmp = st.selectbox("Value", STANDARD_COMPARE_VALUES, index=0, key="cmp_value_col3")
            with vc[1]:
                st.metric("Selected", len(selected_rows))
            with vc[2]:
                st.metric("Candidates", len(filtered_cmp))
            st.caption("ここでは比較する値だけを選びます。色範囲や分布図の横軸は下のパネルで別々に調整します。")

        with st.expander("2-B. Heatmap settings", expanded=True):
            hc1 = st.columns([1.1, 0.75, 0.75, 0.75, 0.75, 0.9])
            with hc1[0]:
                color_mode = st.selectbox(
                    "Color range",
                    ["Auto global", "Percentile global", "Manual global", "Auto individual"],
                    index=0,
                    key="cmp_color_mode3",
                )
            with hc1[1]:
                manual_vmin = st.number_input("Heatmap min", value=0.0, format="%.6g", key="cmp_manual_vmin3")
            with hc1[2]:
                manual_vmax = st.number_input("Heatmap max", value=1.0, format="%.6g", key="cmp_manual_vmax3")
            with hc1[3]:
                heatmap_pct_low = st.number_input("Pctl low", min_value=0.0, max_value=99.0, value=1.0, step=0.5, format="%.2f", key="cmp_heatmap_pct_low3")
            with hc1[4]:
                heatmap_pct_high = st.number_input("Pctl high", min_value=1.0, max_value=100.0, value=99.0, step=0.5, format="%.2f", key="cmp_heatmap_pct_high3")
            with hc1[5]:
                scale_cmp = st.selectbox("Scale", ["linear", "log"], index=0, key="cmp_scale3")

            hc2 = st.columns([0.9, 0.7, 1.0, 0.8, 0.9])
            with hc2[0]:
                cmap_cmp = st.selectbox("Cmap", list(CMAP_PRESETS.keys()), index=list(CMAP_PRESETS.keys()).index(DEFAULT_CMAP), key="cmp_cmap3")
            with hc2[1]:
                rev_cmp = st.checkbox("反転", value=False, key="cmp_rev3")
            with hc2[2]:
                y_axis_mode_cmp = st.selectbox(
                    "Y-axis",
                    ["Image: y=0 top", "Cartesian: y=0 bottom", "Invert current"],
                    index=0,
                    key="cmp_y_axis_mode3",
                )
            with hc2[3]:
                show_map_axes_cmp = st.checkbox("軸目盛表示", value=False, key="cmp_show_map_axes3")
            with hc2[4]:
                fill_zero_cmp = st.checkbox("未選択を0", value=False, key="cmp_fillzero3")

        with st.expander("2-C. Histogram / KDE settings", expanded=True):
            dc1 = st.columns([0.95, 1.1, 1.05, 0.75, 0.75])
            with dc1[0]:
                distribution = st.selectbox("Distribution", ["Histogram", "KDE", "ECDF", "Raincloud", "None"], index=0, key="cmp_distribution3")
            with dc1[1]:
                dist_layout_choice = st.selectbox("Distribution layout", ["Combined overlay", "Separate under each map"], index=0, key="cmp_dist_layout_choice3")
            with dc1[2]:
                dist_x_mode = st.selectbox(
                    "X range",
                    ["Same as heatmap", "Auto distribution", "Percentile distribution", "Manual distribution"],
                    index=0,
                    key="cmp_dist_x_mode3",
                )
            with dc1[3]:
                hist_bins_cmp = st.number_input("Bins", min_value=5, max_value=400, value=60, step=5, key="cmp_bins3")
            with dc1[4]:
                dist_y_mode = st.selectbox("Y mode", ["density", "count", "log density", "log count"], index=0, key="cmp_dist_y_mode3")

            dc2 = st.columns([0.75, 0.75, 0.75, 0.75])
            with dc2[0]:
                dist_manual_xmin = st.number_input("Dist X min", value=0.0, format="%.6g", key="cmp_dist_xmin3")
            with dc2[1]:
                dist_manual_xmax = st.number_input("Dist X max", value=1.0, format="%.6g", key="cmp_dist_xmax3")
            with dc2[2]:
                dist_pct_low = st.number_input("Dist Pctl low", min_value=0.0, max_value=99.0, value=1.0, step=0.5, format="%.2f", key="cmp_dist_pct_low3")
            with dc2[3]:
                dist_pct_high = st.number_input("Dist Pctl high", min_value=1.0, max_value=100.0, value=99.0, step=0.5, format="%.2f", key="cmp_dist_pct_high3")
            st.caption("Heatmap min/max は色だけ、Dist X min/max はヒストグラム/KDE/ECDF/Raincloud の横軸だけに効きます。")

        with st.expander("2-D. Layout / export style", expanded=False):
            lc = st.columns([0.8, 0.8, 0.8, 0.8, 0.7])
            with lc[0]:
                grid_cols_cmp = st.number_input("Map columns", min_value=1, max_value=6, value=3, step=1, key="cmp_grid_cols3")
            with lc[1]:
                fig_width_cmp = st.number_input("Fig width", min_value=5.0, max_value=24.0, value=11.0, step=0.5, key="cmp_fig_width3")
            with lc[2]:
                map_h_cmp = st.number_input("Map height", min_value=1.6, max_value=6.0, value=2.7, step=0.1, key="cmp_map_h3")
            with lc[3]:
                font_size_cmp = st.number_input("Font size", min_value=6.0, max_value=18.0, value=9.0, step=0.5, key="cmp_font3")
            with lc[4]:
                paper_labels = st.checkbox("A,B,C", value=True, key="cmp_paper_labels3")

        if selected_rows.empty:
            st.info("比較対象を選択してください。")
        else:
            items, errors = prepare_compare_items(selected_rows, value_col_cmp, float(r2_min_cmp), bool(fill_zero_cmp), label_mode)
            if errors:
                with st.expander("Skipped / load errors", expanded=True):
                    for e in errors[:30]:
                        st.warning(e)
                    if len(errors) > 30:
                        st.warning(f"... and {len(errors) - 30} more")
            if not items:
                st.error("表示できる比較対象がありません。Value列やR²条件を確認してください。")
            else:
                g_vmin, g_vmax = calc_compare_color_range(
                    items,
                    color_mode,
                    float(manual_vmin),
                    float(manual_vmax),
                    scale_cmp,
                    pct_low=float(heatmap_pct_low),
                    pct_high=float(heatmap_pct_high),
                )
                dist_xmin, dist_xmax = calc_compare_dist_range(
                    items,
                    dist_x_mode,
                    float(g_vmin),
                    float(g_vmax),
                    float(dist_manual_xmin),
                    float(dist_manual_xmax),
                    pct_low=float(dist_pct_low),
                    pct_high=float(dist_pct_high),
                )
                m1, m2, m3, m4, m5, m6, m7 = st.columns(7)
                m1.metric("Items", len(items))
                m2.metric("R² min", f"{float(r2_min_cmp):.3f}")
                m3.metric("Color min", f"{g_vmin:.4g}")
                m4.metric("Color max", f"{g_vmax:.4g}")
                m5.metric("Dist X min", "-" if dist_xmin is None else f"{dist_xmin:.4g}")
                m6.metric("Dist X max", "-" if dist_xmax is None else f"{dist_xmax:.4g}")
                m7.metric("Pixels", int(sum(len(it["vals"]) for it in items)))

                if distribution == "None":
                    layout_internal = "Maps only"
                elif dist_layout_choice == "Separate under each map":
                    layout_internal = "Maps + separate distributions"
                else:
                    layout_internal = "Maps + combined distribution"

                fig_cmp = make_compare_figure(
                    items,
                    value_col=value_col_cmp,
                    cmap_name=cmap_cmp,
                    reverse_cmap=rev_cmp,
                    scale=scale_cmp,
                    color_mode=color_mode,
                    manual_vmin=float(manual_vmin),
                    manual_vmax=float(manual_vmax),
                    distribution=distribution,
                    layout=layout_internal,
                    bins=int(hist_bins_cmp),
                    ncols=int(grid_cols_cmp),
                    fig_width=float(fig_width_cmp),
                    map_height=float(map_h_cmp),
                    font_size=float(font_size_cmp),
                    paper_labels=bool(paper_labels),
                    heatmap_pct_low=float(heatmap_pct_low),
                    heatmap_pct_high=float(heatmap_pct_high),
                    dist_x_mode=dist_x_mode,
                    dist_manual_xmin=float(dist_manual_xmin),
                    dist_manual_xmax=float(dist_manual_xmax),
                    dist_pct_low=float(dist_pct_low),
                    dist_pct_high=float(dist_pct_high),
                    dist_y_mode=dist_y_mode,
                    y_axis_mode=y_axis_mode_cmp,
                    show_map_axes=bool(show_map_axes_cmp),
                )
                st.pyplot(fig_cmp, use_container_width=True)

                st.markdown("#### Export comparison")
                ecols = st.columns([1, 1, 1, 1])
                def _new_cmp_fig():
                    return make_compare_figure(
                        items,
                        value_col=value_col_cmp,
                        cmap_name=cmap_cmp,
                        reverse_cmap=rev_cmp,
                        scale=scale_cmp,
                        color_mode=color_mode,
                        manual_vmin=float(manual_vmin),
                        manual_vmax=float(manual_vmax),
                        distribution=distribution,
                        layout=layout_internal,
                        bins=int(hist_bins_cmp),
                        ncols=int(grid_cols_cmp),
                        fig_width=float(fig_width_cmp),
                        map_height=float(map_h_cmp),
                        font_size=float(font_size_cmp),
                        paper_labels=bool(paper_labels),
                        heatmap_pct_low=float(heatmap_pct_low),
                        heatmap_pct_high=float(heatmap_pct_high),
                        dist_x_mode=dist_x_mode,
                        dist_manual_xmin=float(dist_manual_xmin),
                        dist_manual_xmax=float(dist_manual_xmax),
                        dist_pct_low=float(dist_pct_low),
                        dist_pct_high=float(dist_pct_high),
                        dist_y_mode=dist_y_mode,
                        y_axis_mode=y_axis_mode_cmp,
                        show_map_axes=bool(show_map_axes_cmp),
                    )
                with ecols[0]:
                    st.download_button("PNG", _save_fig_bytes(_new_cmp_fig(), "png", dpi=300), file_name="compare_figure.png", mime="image/png")
                with ecols[1]:
                    st.download_button("SVG", _save_fig_bytes(_new_cmp_fig(), "svg", dpi=300), file_name="compare_figure.svg", mime="image/svg+xml")
                with ecols[2]:
                    st.download_button("PDF", _save_fig_bytes(_new_cmp_fig(), "pdf", dpi=300), file_name="compare_figure.pdf", mime="application/pdf")
                with ecols[3]:
                    cmp_set = {
                        "mode": compare_mode,
                        "value": value_col_cmp,
                        "items": selected_rows[["manifest", "split", "label", "fit_csv", "grid_meta"]].to_dict(orient="records"),
                        "filters": {"dataset": dataset_filter, "bin": bin_filter, "split": split_filter, "package": package_filter, "r2_min": float(r2_min_cmp)},
                        "heatmap": {
                            "color_mode": color_mode,
                            "scale": scale_cmp,
                            "cmap": cmap_cmp,
                            "reverse_cmap": bool(rev_cmp),
                            "vmin": float(g_vmin),
                            "vmax": float(g_vmax),
                            "manual_vmin": float(manual_vmin),
                            "manual_vmax": float(manual_vmax),
                            "percentile_low": float(heatmap_pct_low),
                            "percentile_high": float(heatmap_pct_high),
                            "y_axis_mode": y_axis_mode_cmp,
                            "show_map_axes": bool(show_map_axes_cmp),
                        },
                        "distribution": {
                            "type": distribution,
                            "layout": layout_internal,
                            "x_range_mode": dist_x_mode,
                            "xmin": None if dist_xmin is None else float(dist_xmin),
                            "xmax": None if dist_xmax is None else float(dist_xmax),
                            "manual_xmin": float(dist_manual_xmin),
                            "manual_xmax": float(dist_manual_xmax),
                            "percentile_low": float(dist_pct_low),
                            "percentile_high": float(dist_pct_high),
                            "y_mode": dist_y_mode,
                            "bins": int(hist_bins_cmp),
                        },
                        "layout": {
                            "map_columns": int(grid_cols_cmp),
                            "fig_width": float(fig_width_cmp),
                            "map_height": float(map_h_cmp),
                            "font_size": float(font_size_cmp),
                            "paper_labels": bool(paper_labels),
                        },
                    }
                    st.download_button("Set JSON", json.dumps(cmp_set, indent=2, ensure_ascii=False).encode("utf-8"), file_name="comparison_set.json", mime="application/json")

                with st.expander("Compare value table", expanded=False):
                    summary_rows = []
                    for it in items:
                        vals = np.asarray(it["vals"], dtype=float)
                        vals = vals[np.isfinite(vals)]
                        summary_rows.append({
                            "label": it["full_label"],
                            "n": int(vals.size),
                            "mean": float(np.nanmean(vals)) if vals.size else np.nan,
                            "median": float(np.nanmedian(vals)) if vals.size else np.nan,
                            "std": float(np.nanstd(vals)) if vals.size else np.nan,
                            "p05": float(np.nanpercentile(vals, 5)) if vals.size else np.nan,
                            "p95": float(np.nanpercentile(vals, 95)) if vals.size else np.nan,
                        })
                    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)
# =============================================================================
# Export tab
# =============================================================================

with export_tab:
    st.subheader("Export")
    # Rebuild current map export
    try:
        current_split = next(s for s in splits if s.split_name == st.session_state.get("viewer_split_name", split_names[0]))
        df, H, W, mask = load_current_split(current_split)
        last_map = st.session_state.get("viewer_last_map", {})
        export_col = last_map.get("display_col", "log₁₀ Γ [s⁻¹]")
        if export_col not in df.columns:
            export_col = candidate_display_columns(df)[0]
        export_filtered = apply_value_filters(df, r2_min=float(last_map.get("r2_min", 0.0)))
        export_grid = build_heatmap_grid(export_filtered, export_col, H, W, mask)
        ev = finite_values(export_grid)
        evmin = float(last_map.get("map_vmin", auto_range(ev)[0]))
        evmax = float(last_map.get("map_vmax", auto_range(ev)[1]))
        efig = make_heatmap_fig(
            export_grid,
            title=f"{export_col} | {current_split.split_name}",
            cmap_name=last_map.get("cmap", DEFAULT_CMAP),
            reverse_cmap=last_map.get("reverse", False),
            vmin=evmin,
            vmax=evmax,
            scale=last_map.get("scale", "linear"),
            label=export_col,
            fig_h=5.0,
            show_axes=True,
        )
        efig_image = make_heatmap_fig(
            export_grid,
            title="",
            cmap_name=last_map.get("cmap", DEFAULT_CMAP),
            reverse_cmap=last_map.get("reverse", False),
            vmin=evmin,
            vmax=evmax,
            scale=last_map.get("scale", "linear"),
            label=export_col,
            fig_h=5.0,
            show_axes=False,
        )
        b1, b2, b3 = st.columns(3)
        with b1:
            st.download_button("PNG heatmap", fig_to_png_bytes(efig), file_name=f"heatmap_{current_split.split_name}_{export_col}.png".replace(os.sep, "_"), mime="image/png")
        with b2:
            st.download_button("PNG image only", fig_to_png_bytes(efig_image, image_only=True), file_name=f"heatmap_image_{current_split.split_name}_{export_col}.png".replace(os.sep, "_"), mime="image/png")
        with b3:
            st.download_button("CSV filtered", export_filtered.to_csv(index=False).encode("utf-8-sig"), file_name=f"filtered_{current_split.split_name}.csv", mime="text/csv")
    except Exception as e:
        st.warning(f"Map exportはまだ準備できていません: {e}")

    # Pixel export
    st.markdown("#### Pixel data")
    px = st.session_state.get("viewer_last_pixel")
    if px:
        try:
            split = next(s for s in splits if s.split_name == px["split"])
            x, y = int(px["x"]), int(px["y"])
            x_arr, y_arr, _, _ = load_acf_coords_cached(str(split.acf_h5))
            idx = coord_to_index(x_arr, y_arr)[(x, y)]
            acf = load_acf_trace_cached(str(split.acf_h5), idx)
            lag = np.arange(acf.size)
            tau = lag.astype(float) * float(frame_time_sec) if frame_time_sec else lag.astype(float)
            acf_df = pd.DataFrame({"lag_index": lag, "tau": tau, "ACF": acf, "ACF_minus_1": acf - 1.0})
            st.download_button("CSV ACF", acf_df.to_csv(index=False).encode("utf-8-sig"), file_name=f"acf_{px['split']}_x{x}_y{y}.csv", mime="text/csv")
            if px.get("source_h5") and Path(px["source_h5"]).exists():
                radius = {"single": 0, "3×3 mean": 1, "5×5 mean": 2, "7×7 mean": 3}.get(px.get("neighborhood", "single"), 0)
                trace = load_image_trace_cached(px["source_h5"], int(px["frame_start"]), int(px["frame_end"]), x, y, radius=radius)
                tt = np.arange(trace.size, dtype=float) * float(frame_time_sec) if frame_time_sec else np.arange(trace.size, dtype=float)
                trace_df = pd.DataFrame({"index": np.arange(trace.size), "time": tt, "intensity": trace})
                st.download_button("CSV I(t)", trace_df.to_csv(index=False).encode("utf-8-sig"), file_name=f"trace_{px['split']}_x{x}_y{y}.csv", mime="text/csv")
        except Exception as e:
            st.warning(f"Pixel exportに失敗: {e}")
    else:
        st.info("Pixelタブで座標をUpdateすると、ACF/I(t)のCSV保存が有効になります。")

    settings = {
        "run_dir": str(bundle.run_dir),
        "analysis_manifest": str(selected_manifest),
        "selected_split": st.session_state.get("viewer_split_name"),
        "time_meta": time_meta,
        "last_map": st.session_state.get("viewer_last_map", {}),
        "last_pixel": st.session_state.get("viewer_last_pixel", {}),
    }
    st.download_button("JSON settings", json.dumps(settings, indent=2, ensure_ascii=False).encode("utf-8"), file_name="viewer_settings.json", mime="application/json")


# =============================================================================
# Advanced tab
# =============================================================================

with advanced_tab:
    st.subheader("Run details")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Splits", len(splits))
    c2.metric("Effective frame [s]", _fmt_num(frame_time_sec))
    c3.metric("Status", str(bundle.analysis_manifest.get("status", "")))
    c4.metric("Package", str(bundle.analysis_manifest.get("package_name", "")))
    st.caption("Time metadata: " + time_meta_caption(time_meta))

    with st.expander("analysis_manifest.json", expanded=False):
        st.json(bundle.analysis_manifest)
    with st.expander("recipe.json", expanded=False):
        st.json(bundle.recipe)
    with st.expander("split_manifest.json", expanded=False):
        st.json(bundle.split_manifest)

    rows = []
    for s in splits:
        rows.append({
            "split": s.split_name,
            "frame_start": s.frame_start,
            "frame_end": s.frame_end,
            "acf_h5": str(s.acf_h5),
            "fit_csv": str(s.fit_csv) if s.fit_csv else None,
            "grid_meta": str(s.grid_meta),
            "summary": str(s.summary) if s.summary else None,
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    log_path = bundle.run_dir / "logs" / "run.log"
    if log_path.exists():
        with st.expander("run.log", expanded=False):
            txt = log_path.read_text(encoding="utf-8", errors="replace")
            st.text_area("", value=txt[-20000:], height=300, key="log_text")
