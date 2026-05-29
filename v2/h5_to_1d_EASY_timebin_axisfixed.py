#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
images.h5 -> 1D integration with PONI + MASK preview + multiprocessing

Features
--------
- GUI (tkinter) for selecting:
    * images.h5
    * poni file
    * external mask tif (optional)
    * output folder
- Preview:
    * representative frame image
    * external mask / internal h5 mask / final mask overlay
    * geometric contour preview from PONI (q or 2theta map)
    * beam-center estimate marker from geometry map minimum
- 1D integration:
    * frame summation / averaging before integrate
    * start/end frame
    * radial points
    * unit selection
    * optional use of H5 internal mask
    * optional mask inversion for external/internal mask
    * multiprocessing by accumulation group
- Output:
    * per-group TXT only
    * ALLmerged Origin/Trans DAT files for downstream analysis
    * preview PNG
    * config JSON

Expected HDF5 structure
-----------------------
- /entry/data/images
- optional /entry/instrument/detector/mask

Notes on mask convention
------------------------
pyFAI expects mask==1 for masked/invalid pixels.
Many DXB files store valid=1, invalid=0 for internal H5 mask.
So this GUI keeps separate inversion switches for external and internal masks.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
import multiprocessing as mp

import h5py
import numpy as np
import tifffile
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

# matplotlib backend must be selected before pyplot import in some environments
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pyFAI


# worker globals for persistent multiprocessing
_G_H5 = None
_G_DSET = None
_G_AI = None
_G_MASK = None
_G_CFG = {}
_G_SECTOR_MASKS = None


def worker_init_persistent(h5_path, image_dataset, poni_path, final_mask, accum_frames,
                           accum_mode, npt, unit, method_text, pol_factor, dummy_value, delta_dummy,
                           sector_enable, sector_centers, sector_width_deg, paired_180, angle_map,
                           shape_hw, poni_space_bin):
    global _G_H5, _G_DSET, _G_AI, _G_MASK, _G_CFG, _G_SECTOR_MASKS
    _G_H5 = h5py.File(h5_path, "r", rdcc_nbytes=256 * 1024 * 1024, rdcc_nslots=1_000_003)
    _G_DSET = _G_H5[image_dataset]
    _G_AI = load_ai_with_optional_binning(poni_path, shape_hw=shape_hw, poni_space_bin=poni_space_bin)
    _G_MASK = final_mask
    _G_CFG = {
        "accum_frames": int(accum_frames),
        "accum_mode": accum_mode,
        "npt": int(npt),
        "unit": unit,
        "method": parse_method(method_text),
        "pol_factor": float(pol_factor),
        "dummy_value": dummy_value,
        "delta_dummy": delta_dummy,
        "sector_enable": bool(sector_enable),
        "sector_centers": list(sector_centers),
        "sector_width_deg": float(sector_width_deg),
        "paired_180": bool(paired_180),
    }
    if sector_enable:
        base_mask = None if final_mask is None else (np.asarray(final_mask) != 0)
        _G_SECTOR_MASKS = []
        for cdeg in sector_centers:
            inside = build_sector_inside_mask(angle_map, cdeg, sector_width_deg, paired_180)
            sector_invalid = ~inside
            if base_mask is not None:
                sector_invalid = np.logical_or(base_mask, sector_invalid)
            _G_SECTOR_MASKS.append((float(cdeg), sector_invalid.astype(np.uint8)))
    else:
        _G_SECTOR_MASKS = None


def integrate_group_range_worker_persistent(group_starts):
    global _G_DSET, _G_AI, _G_MASK, _G_CFG, _G_SECTOR_MASKS

    out = []
    accum_frames = _G_CFG["accum_frames"]

    for item in group_starts:
        if isinstance(item, (tuple, list)) and len(item) >= 3:
            gi, start, end = int(item[0]), int(item[1]), int(item[2])
        else:
            start = int(item)
            end = start + accum_frames
            gi = start // accum_frames

        block = _G_DSET[start:end]
        if block.dtype != np.float32:
            block = block.astype(np.float32, copy=False)

        if _G_CFG["accum_mode"] == "mean":
            img = np.mean(block, axis=0, dtype=np.float32)
        else:
            img = np.sum(block, axis=0, dtype=np.float32)

        common = dict(
            data=img,
            npt=_G_CFG["npt"],
            unit=_G_CFG["unit"],
            polarization_factor=_G_CFG["pol_factor"],
            method=_G_CFG["method"],
        )
        if _G_CFG["dummy_value"] != "":
            common["dummy"] = float(_G_CFG["dummy_value"])
        if _G_CFG["delta_dummy"] != "":
            common["delta_dummy"] = float(_G_CFG["delta_dummy"])

        if _G_CFG["sector_enable"]:
            for cdeg, sector_mask in _G_SECTOR_MASKS:
                kwargs = dict(common, mask=sector_mask)
                res = _G_AI.integrate1d(**kwargs)
                radial = np.asarray(res.radial, dtype=np.float32)
                intensity = np.asarray(res.intensity, dtype=np.float32)
                out.append((gi, start, end - 1, radial, intensity, float(cdeg)))
        else:
            kwargs = dict(common, mask=_G_MASK)
            res = _G_AI.integrate1d(**kwargs)
            radial = np.asarray(res.radial, dtype=np.float32)
            intensity = np.asarray(res.intensity, dtype=np.float32)
            out.append((gi, start, end - 1, radial, intensity, None))

    return out


# -----------------------------
# dataclasses / config
# -----------------------------
@dataclass
class IntegrationConfig:
    h5_path: str
    poni_path: str
    mask_tif_path: str
    output_dir: str
    image_dataset: str = "/entry/data/images"
    internal_mask_dataset: str = "/entry/instrument/detector/mask"
    use_internal_h5_mask: bool = True
    invert_internal_h5_mask: bool = True
    # External mask default is DXB-style: 1/255 = valid/use, 0 = invalid/exclude.
    # pyFAI needs the opposite, so default invert=True.
    invert_external_mask: bool = True
    accum_frames: int = 1
    accum_mode: str = "sum"          # sum / mean
    # If >0, split selected frame range into exactly this many temporal bins.
    # Example: 100 means produce 100 1D profiles across start_frame..end_frame.
    # This overrides fixed accum_frames for integration ranges.
    target_time_bins: int = 0
    # Time binning mode:
    #   frames   : use exactly accum_frames frames per 1D profile
    #   duration : use profile_duration_value/profile_duration_unit to calculate frames per profile
    #   profiles : split selected frame range into target_time_bins profiles
    time_bin_mode: str = "frames"
    profile_duration_value: float = 1.0
    profile_duration_unit: str = "min"  # sec / min
    include_partial_last_bin: bool = False
    npt: int = 2000
    unit: str = "q_A^-1"
    method: str = "bbox_csr_cython"
    polarization_factor: float = 0.0
    dummy_value: str = ""
    delta_dummy: str = ""
    start_frame: int = 0
    end_frame_inclusive: int = -1
    preview_frame: int = 0
    preview_percentile_low: float = 1.0
    preview_percentile_high: float = 99.5
    contour_levels: int = 6
    processes: int = 1
    groups_per_task: int = 16
    sector_enable: bool = False
    ref_angle_deg: float = 0.0
    sector_step_deg: float = 10.0
    sector_width_deg: float = 10.0
    paired_180: bool = True
    # Time / binned-H5 metadata for ALLTrans and geometry scaling
    # effective_frame_time_sec is the time per frame in THIS H5 after temporal binning.
    effective_frame_time_sec: float = 1.0
    time_axis_unit: str = "frame"     # frame / sec / min
    # If the input H5 image is spatially binned but the PONI was made for the original image,
    # set poni_space_bin to the spatial bin factor, e.g. 10 for 10x10.
    # If the PONI already matches the binned image, leave this at 1.
    poni_space_bin: int = 1
    time_bin_label: int = 1
    # Output time labels for ALLMerge-like files.
    # auto: use seconds for <60 s/profile, minutes for >=60 s/profile.
    # sec/min: force unit.
    allmerge_axis_unit: str = "auto"
    # Manual legacy minute step. 0 means use the actual time width.
    # Non-zero is a label-only override and is kept only for legacy compatibility.
    allmerge_step_min: float = 0.0


# -----------------------------
# utility functions
# -----------------------------
def parse_method(method_text: str):
    txt = method_text.strip()
    if not txt:
        return None
    if "," in txt:
        return tuple(x.strip() for x in txt.split(",") if x.strip())
    if "_" in txt:
        return tuple(x.strip() for x in txt.split("_") if x.strip())
    return txt


def ensure_dir(path: str | Path):
    Path(path).mkdir(parents=True, exist_ok=True)


def load_h5_shape(h5_path: str, dataset: str):
    with h5py.File(h5_path, "r") as hf:
        d = hf[dataset]
        return tuple(d.shape), str(d.dtype)


def read_frame(h5_path: str, dataset: str, idx: int):
    with h5py.File(h5_path, "r") as hf:
        d = hf[dataset]
        idx = max(0, min(int(idx), d.shape[0] - 1))
        return d[idx].astype(np.float32, copy=False)


def read_internal_mask(h5_path: str, dataset: str, shape_hw):
    with h5py.File(h5_path, "r") as hf:
        if dataset not in hf:
            return None
        arr = hf[dataset][:]
    if arr.shape != tuple(shape_hw):
        raise ValueError(f"Internal H5 mask shape mismatch: {arr.shape} != {shape_hw}")
    return arr


def read_external_mask(mask_path: str, shape_hw):
    if not mask_path:
        return None
    arr = tifffile.imread(mask_path)
    if arr.shape != tuple(shape_hw):
        raise ValueError(f"External mask shape mismatch: {arr.shape} != {shape_hw}")
    return arr


def to_pyfai_mask(mask_arr: np.ndarray | None, invert: bool) -> np.ndarray | None:
    if mask_arr is None:
        return None
    m = np.asarray(mask_arr)
    m = (m != 0)
    if invert:
        m = ~m
    return m.astype(np.uint8)


def combine_masks(mask_a: np.ndarray | None, mask_b: np.ndarray | None) -> np.ndarray | None:
    if mask_a is None and mask_b is None:
        return None
    if mask_a is None:
        return mask_b.astype(np.uint8)
    if mask_b is None:
        return mask_a.astype(np.uint8)
    return np.logical_or(mask_a != 0, mask_b != 0).astype(np.uint8)


def robust_limits(img: np.ndarray, low=1.0, high=99.5):
    finite = np.asarray(img[np.isfinite(img)], dtype=np.float32)
    if finite.size == 0:
        return 0.0, 1.0
    vmin = float(np.percentile(finite, low))
    vmax = float(np.percentile(finite, high))
    if not np.isfinite(vmin):
        vmin = float(np.nanmin(finite))
    if not np.isfinite(vmax):
        vmax = float(np.nanmax(finite))
    if vmax <= vmin:
        vmax = vmin + 1.0
    return vmin, vmax


def load_ai_with_optional_binning(poni_path: str, shape_hw=None, poni_space_bin: int | float = 1):
    """Load PONI and optionally scale detector pixel size for spatially binned H5 images.

    Use poni_space_bin > 1 only when the input H5 is spatially binned but the PONI
    file was generated for the original unbinned detector image. Physical PONI1/PONI2
    stay in meters; only detector pixel size is multiplied by the bin factor.
    """
    ai = pyFAI.load(poni_path)
    b = float(poni_space_bin or 1)
    if b <= 0:
        raise ValueError("poni_space_bin must be >= 1")
    if abs(b - 1.0) < 1e-12:
        return ai

    det = ai.detector
    old_p1 = float(getattr(det, "pixel1"))
    old_p2 = float(getattr(det, "pixel2"))

    def _set_pixel(obj, attr, value):
        try:
            setattr(obj, attr, value)
            return
        except Exception:
            pass
        setter = getattr(obj, f"set_{attr}", None)
        if setter is not None:
            setter(value)
            return
        # last-resort for older pyFAI Detector classes
        try:
            setattr(obj, "_" + attr, value)
        except Exception as e:
            raise RuntimeError(f"Could not set detector {attr} for binned PONI scaling") from e

    _set_pixel(det, "pixel1", old_p1 * b)
    _set_pixel(det, "pixel2", old_p2 * b)

    if shape_hw is not None:
        shape_hw = tuple(map(int, shape_hw))
        for attr in ["max_shape", "shape"]:
            try:
                setattr(det, attr, shape_hw)
            except Exception:
                pass

    # Clear cached geometry arrays after detector changes.
    for meth in ["reset", "reset_cache"]:
        fn = getattr(ai, meth, None)
        if fn is not None:
            try:
                fn()
            except Exception:
                pass
    return ai


def get_geometry_map(ai, shape_hw, unit: str):
    unit_low = unit.lower()
    if "q" in unit_low:
        if hasattr(ai, "qArray"):
            return ai.qArray(shape_hw)
    if "2th" in unit_low or "2theta" in unit_low:
        if hasattr(ai, "twoThetaArray"):
            arr = ai.twoThetaArray(shape_hw)
            if "deg" in unit_low:
                arr = np.rad2deg(arr)
            return arr
    if hasattr(ai, "rArray"):
        return ai.rArray(shape_hw)
    raise RuntimeError("Could not build geometry map from PONI with this pyFAI version.")


def estimate_center_from_geometry_map(geom_map: np.ndarray):
    iy, ix = np.unravel_index(np.nanargmin(geom_map), geom_map.shape)
    return int(ix), int(iy)


def save_preview_png(
    out_png: str,
    frame: np.ndarray,
    ext_mask: np.ndarray | None,
    int_mask: np.ndarray | None,
    final_mask: np.ndarray | None,
    geom_map: np.ndarray,
    center_xy,
    unit_label: str,
    preview_percentile_low: float,
    preview_percentile_high: float,
    contour_levels: int,
    sector_enable: bool = False,
    ref_angle_deg: float = 0.0,
    sector_step_deg: float = 10.0,
    sector_width_deg: float = 10.0,
    paired_180: bool = True,
):
    frame = np.asarray(frame, dtype=np.float32)
    disp = np.log10(np.clip(frame, 1e-6, None))
    vmin, vmax = robust_limits(disp, preview_percentile_low, preview_percentile_high)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
    axs = axes.ravel()

    def draw_base(ax, title):
        im = ax.imshow(disp, origin="lower", cmap="gray", vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="log10(intensity)")

    draw_base(axs[0], "Raw frame (log)")
    draw_base(axs[1], "Mask overlay")
    draw_base(axs[2], f"PONI geometry preview ({unit_label})")
    draw_base(axs[3], "Final mask only")

    if ext_mask is not None:
        rgba = np.zeros((*ext_mask.shape, 4), dtype=np.float32)
        rgba[..., 0] = 1.0  # red
        rgba[..., 3] = 0.20 * (ext_mask != 0)
        axs[1].imshow(rgba, origin="lower")
    if int_mask is not None:
        rgba = np.zeros((*int_mask.shape, 4), dtype=np.float32)
        rgba[..., 1] = 1.0  # green
        rgba[..., 3] = 0.20 * (int_mask != 0)
        axs[1].imshow(rgba, origin="lower")
    if final_mask is not None:
        rgba = np.zeros((*final_mask.shape, 4), dtype=np.float32)
        rgba[..., 2] = 1.0  # blue
        rgba[..., 3] = 0.28 * (final_mask != 0)
        axs[1].imshow(rgba, origin="lower")
        axs[3].imshow((final_mask != 0).astype(np.float32), origin="lower", cmap="Blues", alpha=0.85)

    finite = geom_map[np.isfinite(geom_map)]
    if finite.size > 0:
        g0 = float(np.percentile(finite, 10))
        g1 = float(np.percentile(finite, 90))
        if g1 > g0 and contour_levels >= 2:
            levels = np.linspace(g0, g1, contour_levels)
            cs = axs[2].contour(geom_map, levels=levels, linewidths=0.8)
            axs[2].clabel(cs, inline=True, fontsize=7, fmt="%.3g")

    cx, cy = center_xy
    for ax in axs[:3]:
        ax.plot(cx, cy, marker="+", markersize=12, markeredgewidth=1.8)

    if sector_enable:
        h, w = frame.shape
        r = 0.48 * math.hypot(h, w)
        centers = make_sector_centers(ref_angle_deg, sector_step_deg, paired_180)
        show_centers = centers[: min(6, len(centers))]
        for cdeg in show_centers:
            for add180 in ([0.0, 180.0] if paired_180 else [0.0]):
                ang = math.radians(normalize_deg(cdeg + add180))
                x2 = cx + r * math.cos(ang)
                y2 = cy + r * math.sin(ang)
                axs[2].plot([cx, x2], [cy, y2], "-", linewidth=1.2)
            # width boundary for the first side
            for bound in [cdeg - sector_width_deg/2.0, cdeg + sector_width_deg/2.0]:
                ang = math.radians(normalize_deg(bound))
                x2 = cx + r * math.cos(ang)
                y2 = cy + r * math.sin(ang)
                axs[2].plot([cx, x2], [cy, y2], "--", linewidth=0.8)
        axs[2].text(
            0.01, 0.99,
            f"Sector mode\nref={ref_angle_deg:.2f}°  step={sector_step_deg:.2f}°\nwidth={sector_width_deg:.2f}°  paired180={paired_180}",
            transform=axs[2].transAxes, va="top", ha="left",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
        )

    axs[1].text(0.01, 0.99, "Red: external mask\nGreen: internal H5 mask\nBlue: final mask",
                transform=axs[1].transAxes, va="top", ha="left",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    fig.suptitle("PONI / MASK preview", fontsize=14)
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


# -----------------------------
# integration workers
# -----------------------------
def integrate_group_range_worker(args):
    (
        h5_path,
        image_dataset,
        poni_path,
        final_mask,
        group_indices,
        accum_frames,
        accum_mode,
        npt,
        unit,
        method_text,
        pol_factor,
        dummy_value,
        delta_dummy,
        sector_enable,
        sector_centers,
        sector_width_deg,
        paired_180,
        angle_map,
        shape_hw,
        poni_space_bin,
    ) = args

    ai = load_ai_with_optional_binning(poni_path, shape_hw=shape_hw, poni_space_bin=poni_space_bin)
    method = parse_method(method_text)
    base_mask = None if final_mask is None else (np.asarray(final_mask) != 0)
    sector_masks = None
    if sector_enable:
        sector_masks = []
        for cdeg in sector_centers:
            inside = build_sector_inside_mask(angle_map, cdeg, sector_width_deg, paired_180)
            sector_invalid = ~inside
            if base_mask is not None:
                sector_invalid = np.logical_or(base_mask, sector_invalid)
            sector_masks.append((float(cdeg), sector_invalid.astype(np.uint8)))

    out = []
    with h5py.File(h5_path, "r") as hf:
        dset = hf[image_dataset]

        for item in group_indices:
            if isinstance(item, (tuple, list)) and len(item) >= 3:
                gi, start, end = int(item[0]), int(item[1]), int(item[2])
            else:
                start = int(item)
                end = start + accum_frames
                gi = start // accum_frames
            block = dset[start:end].astype(np.float32, copy=False)
            if accum_mode == "mean":
                img = np.mean(block, axis=0, dtype=np.float32)
            else:
                img = np.sum(block, axis=0, dtype=np.float32)

            common = dict(
                data=img,
                npt=int(npt),
                unit=unit,
                polarization_factor=pol_factor,
                method=method,
            )
            if dummy_value != "":
                common["dummy"] = float(dummy_value)
            if delta_dummy != "":
                common["delta_dummy"] = float(delta_dummy)

            if sector_enable:
                for cdeg, sector_mask in sector_masks:
                    res = ai.integrate1d(**dict(common, mask=sector_mask))
                    radial = np.asarray(res.radial, dtype=np.float32)
                    intensity = np.asarray(res.intensity, dtype=np.float32)
                    out.append((gi, start, end - 1, radial, intensity, float(cdeg)))
            else:
                res = ai.integrate1d(**dict(common, mask=final_mask))
                radial = np.asarray(res.radial, dtype=np.float32)
                intensity = np.asarray(res.intensity, dtype=np.float32)
                out.append((gi, start, end - 1, radial, intensity, None))

    return out


def chunked(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


def build_time_bin_specs(
    start_frame: int,
    end_frame: int,
    accum_frames: int,
    target_time_bins: int = 0,
    time_bin_mode: str = "frames",
    effective_frame_time_sec: float = 1.0,
    profile_duration_value: float = 1.0,
    profile_duration_unit: str = "min",
    include_partial_last_bin: bool = False,
):
    """Return list of (group_index, start_frame, end_exclusive) for integration.

    Modes:
      frames   : fixed number of frames per profile, using accum_frames.
      duration : fixed real-time width per profile; frames/profile is calculated from
                 profile_duration / effective_frame_time_sec.
      profiles : exactly target_time_bins profiles across the selected frame range.

    The default for real analysis should usually be frames or duration.  The old
    "Target profiles=100" behavior is still available as mode="profiles".
    """
    start_frame = int(start_frame)
    end_frame = int(end_frame)
    frame_count = int(end_frame - start_frame + 1)
    if frame_count <= 0:
        return []

    mode = str(time_bin_mode or "frames").strip().lower()
    if mode in ["fixed_frames", "frame", "frames", "accum"]:
        mode = "frames"
    elif mode in ["fixed_duration", "duration", "time", "minutes", "seconds"]:
        mode = "duration"
    elif mode in ["target_profiles", "profiles", "target", "nprofiles"]:
        mode = "profiles"
    else:
        raise ValueError(f"Unknown time_bin_mode: {time_bin_mode}")

    if mode == "profiles":
        target = int(target_time_bins or 0)
        if target <= 0:
            raise ValueError("time_bin_mode=profiles requires target_time_bins > 0")
        n_bins = min(target, frame_count)
        edges = np.linspace(start_frame, end_frame + 1, n_bins + 1)
        edges = np.rint(edges).astype(np.int64)
        specs = []
        for gi in range(n_bins):
            s0 = int(edges[gi])
            s1 = int(edges[gi + 1])
            if s1 <= s0:
                s1 = s0 + 1
            s1 = min(s1, end_frame + 1)
            specs.append((int(gi), int(s0), int(s1)))
        return specs

    if mode == "duration":
        ft = float(effective_frame_time_sec)
        if not np.isfinite(ft) or ft <= 0:
            raise ValueError("effective_frame_time_sec must be > 0 for duration mode")
        unit = str(profile_duration_unit or "min").strip().lower()
        mult = 60.0 if unit.startswith("min") else 1.0
        duration_sec = float(profile_duration_value) * mult
        if not np.isfinite(duration_sec) or duration_sec <= 0:
            raise ValueError("profile duration must be > 0")
        accum_frames = int(round(duration_sec / ft))
        if accum_frames <= 0:
            accum_frames = 1
    else:
        accum_frames = max(1, int(accum_frames))

    specs = []
    gi = 0
    pos = start_frame
    while pos + accum_frames <= end_frame + 1:
        specs.append((int(gi), int(pos), int(pos + accum_frames)))
        gi += 1
        pos += accum_frames
    if include_partial_last_bin and pos <= end_frame:
        specs.append((int(gi), int(pos), int(end_frame + 1)))
    return specs


def seconds_for_profile_duration(value: float, unit: str) -> float:
    unit = str(unit or "min").strip().lower()
    return float(value) * (60.0 if unit.startswith("min") else 1.0)


def estimate_frames_per_profile_from_cfg(cfg: IntegrationConfig) -> int:
    mode = str(getattr(cfg, "time_bin_mode", "frames") or "frames").strip().lower()
    if mode in ["fixed_duration", "duration", "time", "minutes", "seconds"]:
        sec = seconds_for_profile_duration(cfg.profile_duration_value, cfg.profile_duration_unit)
        return max(1, int(round(sec / float(cfg.effective_frame_time_sec))))
    return max(1, int(cfg.accum_frames))

def describe_time_bin_specs(specs):
    sizes = [int(e - s) for _, s, e in specs]
    if not sizes:
        return "no bins"
    unique = sorted(set(sizes))
    return f"{len(specs)} bins, frames/bin={unique[0]}" + (f"..{unique[-1]}" if len(unique) > 1 else "")



def angle_diff_deg(a, b):
    return (a - b + 180.0) % 360.0 - 180.0


def normalize_deg(a):
    return a % 360.0


def make_sector_centers(ref_angle_deg: float, step_deg: float, paired_180: bool):
    step = float(step_deg)
    if step <= 0:
        raise ValueError("sector_step_deg must be > 0")
    span = 180.0 if paired_180 else 360.0
    n = max(1, int(round(span / step)))
    return [normalize_deg(ref_angle_deg + k * step) for k in range(n)]


def estimate_center_and_angle_map_from_poni(cfg: IntegrationConfig, shape_hw):
    ai = load_ai_with_optional_binning(cfg.poni_path, shape_hw=shape_hw, poni_space_bin=cfg.poni_space_bin)
    geom_map = get_geometry_map(ai, shape_hw, cfg.unit)
    center_xy = estimate_center_from_geometry_map(geom_map)
    H, W = shape_hw
    yy, xx = np.indices((H, W), dtype=np.float32)
    cx, cy = float(center_xy[0]), float(center_xy[1])
    angle_map = (np.degrees(np.arctan2(yy - cy, xx - cx)) + 360.0) % 360.0
    return center_xy, angle_map, geom_map


def build_sector_inside_mask(angle_map: np.ndarray, center_deg: float, width_deg: float, paired_180: bool):
    half = float(width_deg) / 2.0
    inside = np.abs(angle_diff_deg(angle_map, center_deg)) <= half
    if paired_180:
        inside |= np.abs(angle_diff_deg(angle_map, center_deg + 180.0)) <= half
    return inside


def format_angle_label(a: float):
    a = normalize_deg(a)
    if abs(a - round(a)) < 1e-6:
        return f"{int(round(a)):03d}"
    return f"{a:06.2f}".replace(".", "p")
# -----------------------------
# high-level pipeline
# -----------------------------
def build_final_mask(cfg: IntegrationConfig, shape_hw):
    ext_raw = read_external_mask(cfg.mask_tif_path, shape_hw) if cfg.mask_tif_path else None
    int_raw = read_internal_mask(cfg.h5_path, cfg.internal_mask_dataset, shape_hw) if cfg.use_internal_h5_mask else None

    ext_pyfai = to_pyfai_mask(ext_raw, cfg.invert_external_mask)
    int_pyfai = to_pyfai_mask(int_raw, cfg.invert_internal_h5_mask)
    final_mask = combine_masks(ext_pyfai, int_pyfai)
    return ext_pyfai, int_pyfai, final_mask


def run_preview(cfg: IntegrationConfig):
    ensure_dir(cfg.output_dir)

    shape, dtype = load_h5_shape(cfg.h5_path, cfg.image_dataset)
    if len(shape) != 3:
        raise ValueError(f"Image dataset must be 3D (T,H,W), got {shape}")
    _, H, W = shape

    frame = read_frame(cfg.h5_path, cfg.image_dataset, cfg.preview_frame)
    ext_mask, int_mask, final_mask = build_final_mask(cfg, (H, W))

    center_xy, angle_map, geom_map = estimate_center_and_angle_map_from_poni(cfg, (H, W))

    out_png = str(Path(cfg.output_dir) / "poni_mask_preview.png")
    save_preview_png(
        out_png=out_png,
        frame=frame,
        ext_mask=ext_mask,
        int_mask=int_mask,
        final_mask=final_mask,
        geom_map=geom_map,
        center_xy=center_xy,
        unit_label=cfg.unit,
        preview_percentile_low=cfg.preview_percentile_low,
        preview_percentile_high=cfg.preview_percentile_high,
        contour_levels=cfg.contour_levels,
        sector_enable=cfg.sector_enable,
        ref_angle_deg=cfg.ref_angle_deg,
        sector_step_deg=cfg.sector_step_deg,
        sector_width_deg=cfg.sector_width_deg,
        paired_180=cfg.paired_180,
    )

    preview_info = {
        "dataset_shape": list(shape),
        "dataset_dtype": dtype,
        "preview_frame": int(max(0, min(cfg.preview_frame, shape[0]-1))),
        "beam_center_estimate_xy": [int(center_xy[0]), int(center_xy[1])],
        "unit": cfg.unit,
        "output_preview_png": out_png,
        "sector_enable": bool(cfg.sector_enable),
        "reference_angle_deg": float(cfg.ref_angle_deg),
        "sector_step_deg": float(cfg.sector_step_deg),
        "sector_width_deg": float(cfg.sector_width_deg),
        "paired_180": bool(cfg.paired_180),
        "angle_convention": "right=0deg, counterclockwise",
    }
    with open(Path(cfg.output_dir) / "preview_info.json", "w", encoding="utf-8") as f:
        json.dump(preview_info, f, indent=2, ensure_ascii=False)

    return out_png, center_xy



def _safe_col_label(text: str) -> str:
    """Return a compact label safe for TSV/DAT headers."""
    return str(text).replace("\t", "_").replace(" ", "_").replace("/", "-")


def _group_label(start: int, end: int, gi: int, frame_time_sec: float = 1.0, time_axis_unit: str = "frame") -> str:
    """Label used as spectrum/time column in ALLmerged Origin files."""
    unit = str(time_axis_unit or "frame").lower()
    if unit == "sec":
        t = float(start) * float(frame_time_sec)
        return f"t{t:.6g}s"
    if unit == "min":
        t = float(start) * float(frame_time_sec) / 60.0
        return f"t{t:.6g}min"
    if int(start) == int(end):
        return f"f{int(start):06d}"
    return f"f{int(start):06d}-{int(end):06d}"


def _time_axis_header(time_axis_unit: str) -> str:
    unit = str(time_axis_unit or "frame").lower()
    if unit == "sec":
        return "time_sec"
    if unit == "min":
        return "time_min"
    return "frame_start"


def _time_axis_value(start: int, frame_time_sec: float, time_axis_unit: str):
    unit = str(time_axis_unit or "frame").lower()
    if unit == "sec":
        return float(start) * float(frame_time_sec)
    if unit == "min":
        return float(start) * float(frame_time_sec) / 60.0
    return int(start)


def _sector_file_token(sector_center):
    if sector_center is None:
        return "all"
    return f"sector_{format_angle_label(float(sector_center))}deg"


def _allmerged_basename(sector_center) -> str:
    token = _sector_file_token(sector_center)
    if token == "all":
        return "ALLmerged_1D"
    return f"ALLmerged_1D_{token}"


def _format_num(x) -> str:
    try:
        return f"{float(x):.8e}"
    except Exception:
        return "nan"


def _format_min_label(x: float, suffix: bool = False) -> str:
    """Format minute values like the legacy ALLMerge script: 0, 6, 12 or 0.5."""
    try:
        v = float(x)
        if abs(v - round(v)) < 1e-9:
            s = str(int(round(v)))
        else:
            s = (f"{v:.8g}").rstrip("0").rstrip(".")
        return s + ("min" if suffix else "")
    except Exception:
        return "nan" + ("min" if suffix else "")


def _allmerge_radial_axis_and_label(base_radial: np.ndarray, unit: str):
    """Return a legacy ALLMerge radial axis.

    Original ALLMerge assumes Q(nm-1). pyFAI's common q_A^-1 output is converted
    from A^-1 to nm^-1 by multiplying by 10. Non-q units are kept as-is.
    """
    u = str(unit or "").strip()
    ul = u.lower().replace(" ", "")
    r = np.asarray(base_radial, dtype=np.float64)
    if "q" in ul and ("a^-1" in ul or "a-1" in ul or "å" in ul or "ang" in ul):
        return r * 10.0, "Q(nm-1)"
    if "q" in ul and ("nm^-1" in ul or "nm-1" in ul):
        return r, "Q(nm-1)"
    return r, (u or "radial")


def _legacy_file_step_token(step_min: float) -> str:
    return _format_min_label(step_min, suffix=False).replace(".", "p")


def _format_time_axis_value(value: float, unit: str, suffix: bool = True) -> str:
    """Format time labels without forcing minutes. unit is 'sec' or 'min'."""
    unit = str(unit or "sec").strip().lower()
    v = float(value)
    if abs(v - round(v)) < 1e-9:
        txt = str(int(round(v)))
    else:
        txt = (f"{v:.8g}").rstrip("0").rstrip(".")
    if not suffix:
        return txt
    return txt + ("s" if unit.startswith("sec") or unit == "s" else "min")


def _axis_unit_from_step(step_sec: float, requested: str = "auto") -> str:
    req = str(requested or "auto").strip().lower()
    if req in ["sec", "s", "second", "seconds"]:
        return "sec"
    if req in ["min", "m", "minute", "minutes"]:
        return "min"
    # auto: second labels are easier for sub-minute profiles.
    return "sec" if float(step_sec) < 60.0 else "min"


def _axis_file_step_token(step_value: float, unit: str) -> str:
    txt = _format_time_axis_value(step_value, unit, suffix=False).replace(".", "p")
    return txt + ("s" if str(unit).startswith("sec") else "min")


def write_allmerged_dat_outputs(output_dir: str | Path, results: list, unit: str,
                                frame_time_sec: float = 1.0, time_axis_unit: str = "frame",
                                poni_space_bin: int = 1, time_bin_label: int = 1,
                                accum_frames: int = 1, allmerge_step_min: float = 0.0,
                                allmerge_axis_unit: str = "auto"):
    """Write DAT matrix outputs directly from integration results.

    Two formats are written:
    1) Rich format: preserves pyFAI unit and actual frame/time metadata.
    2) Legacy ALLMerge-compatible format:
       - Origin first column = Q(nm-1) when q_A^-1 is selected, converted by x10
       - Origin time columns = 0min, stepmin, 2*stepmin ...
       - Trans first column header = min

    Sector integration creates one pair of files per sector for each format.
    """
    all_dir = Path(output_dir) / "ALLmerged"
    ensure_dir(all_dir)

    grouped = {}
    for gi, start, end, radial, intensity, sector_center in results:
        key = None if sector_center is None else float(sector_center)
        grouped.setdefault(key, []).append((int(gi), int(start), int(end), np.asarray(radial), np.asarray(intensity), sector_center))

    written = []
    index_rows = []

    for sector_key, rows in sorted(grouped.items(), key=lambda kv: (-1.0 if kv[0] is None else float(kv[0]))):
        rows = sorted(rows, key=lambda r: (r[1], r[0]))
        if not rows:
            continue

        base_radial = np.asarray(rows[0][3], dtype=np.float64)
        base_name = _allmerged_basename(sector_key)
        origin_path = all_dir / f"{base_name}_Origin.dat"
        trans_path = all_dir / f"{base_name}_Trans.dat"

        labels = [_safe_col_label(_group_label(start, end, gi, frame_time_sec, time_axis_unit)) for gi, start, end, _, _, _ in rows]
        spectra = []
        for gi, start, end, radial, intensity, sector_center in rows:
            r = np.asarray(radial, dtype=np.float64)
            y = np.asarray(intensity, dtype=np.float64)
            if r.shape != base_radial.shape or not np.allclose(r, base_radial, rtol=1e-6, atol=1e-12, equal_nan=True):
                # pyFAI normally returns the same radial axis for every frame/sector.
                # If it ever changes slightly, interpolate so the matrix remains rectangular.
                finite = np.isfinite(r) & np.isfinite(y)
                if np.count_nonzero(finite) < 2:
                    y2 = np.full_like(base_radial, np.nan, dtype=np.float64)
                else:
                    y2 = np.interp(base_radial, r[finite], y[finite], left=np.nan, right=np.nan)
                spectra.append(y2)
            else:
                spectra.append(y)

            index_rows.append({
                "sector_center_deg": "" if sector_key is None else f"{float(sector_key):.8g}",
                "group_index": int(gi),
                "start_frame": int(start),
                "end_frame": int(end),
                "label": _group_label(start, end, gi, frame_time_sec, time_axis_unit),
                "time_axis_value": _time_axis_value(start, frame_time_sec, time_axis_unit),
                "time_axis_unit": _time_axis_header(time_axis_unit),
                "start_time_sec": float(start) * float(frame_time_sec),
                "end_time_sec": float(end) * float(frame_time_sec),
                "center_time_sec": 0.5 * (float(start) + float(end)) * float(frame_time_sec),
                "origin_file": str(origin_path),
                "trans_file": str(trans_path),
            })

        # Origin: radial as first column, each spectrum as a time/frame column.
        origin_arr = np.column_stack([base_radial] + spectra)
        np.savetxt(
            origin_path,
            origin_arr,
            fmt="%.8e",
            delimiter="\t",
            header="\t".join([unit] + labels),
            comments="",
            encoding="utf-8",
        )

        # Trans: rows are spectra/time, columns are radial positions.
        time_header = _time_axis_header(time_axis_unit)
        with open(trans_path, "w", encoding="utf-8", newline="\n") as f:
            f.write(time_header + "\t" + "\t".join(_format_num(x) for x in base_radial) + "\n")
            for (gi, start, end, radial, intensity, sector_center), y in zip(rows, spectra):
                tv = _time_axis_value(start, frame_time_sec, time_axis_unit)
                tv_text = str(int(tv)) if time_header == "frame_start" else _format_num(tv)
                f.write(tv_text + "\t" + "\t".join(_format_num(v) for v in y) + "\n")

        # ALLMerge-like files with real, readable time labels.
        # If allmerge_step_min > 0, keep the old manual minute override. Otherwise
        # labels are based on actual start frame time and use sec for sub-minute bins.
        legacy_radial, legacy_radial_label = _allmerge_radial_axis_and_label(base_radial, unit)
        manual_step_min = float(allmerge_step_min) if float(allmerge_step_min or 0.0) > 0 else 0.0
        if manual_step_min > 0:
            axis_unit = "min"
            axis_step_value = manual_step_min
            legacy_time_vals = [i * manual_step_min for i in range(len(rows))]
        else:
            step_sec = float(frame_time_sec) * max(1, int(accum_frames))
            axis_unit = _axis_unit_from_step(step_sec, allmerge_axis_unit)
            if axis_unit == "sec":
                axis_step_value = step_sec
                legacy_time_vals = [float(start) * float(frame_time_sec) for gi, start, end, _, _, _ in rows]
            else:
                axis_step_value = step_sec / 60.0
                legacy_time_vals = [float(start) * float(frame_time_sec) / 60.0 for gi, start, end, _, _, _ in rows]
        step_token = _axis_file_step_token(axis_step_value, axis_unit)
        if sector_key is None:
            legacy_base = f"ALLmerged_{step_token}"
        else:
            legacy_base = f"ALLmerged_{step_token}_{_sector_file_token(sector_key)}"
        legacy_origin_path = all_dir / f"{legacy_base}_Origin.dat"
        legacy_trans_path = all_dir / f"{legacy_base}_Trans.dat"

        legacy_headers = [legacy_radial_label] + [_format_time_axis_value(v, axis_unit, suffix=True) for v in legacy_time_vals]
        with open(legacy_origin_path, "w", encoding="utf-8", newline="\n") as f:
            f.write(" ".join(legacy_headers) + "\n")
            for j, qv in enumerate(legacy_radial):
                vals = [spectra[i][j] if j < len(spectra[i]) else np.nan for i in range(len(spectra))]
                f.write(_format_num(qv) + " " + " ".join(_format_num(v) for v in vals) + "\n")

        with open(legacy_trans_path, "w", encoding="utf-8", newline="\n") as f:
            f.write(("sec" if axis_unit == "sec" else "min") + " " + " ".join(_format_num(x) for x in legacy_radial) + "\n")
            for i, y in enumerate(spectra):
                f.write(_format_time_axis_value(legacy_time_vals[i], axis_unit, suffix=False) + " " + " ".join(_format_num(v) for v in y) + "\n")

        written.append({
            "sector_center_deg": None if sector_key is None else float(sector_key),
            "origin_dat": str(origin_path),
            "trans_dat": str(trans_path),
            "legacy_allmerge_origin_dat": str(legacy_origin_path),
            "legacy_allmerge_trans_dat": str(legacy_trans_path),
            "axis_step_value": float(axis_step_value),
            "axis_unit": axis_unit,
            "manual_step_min_override": float(manual_step_min),
            "legacy_radial_label": legacy_radial_label,
            "n_spectra": int(len(rows)),
            "n_radial": int(base_radial.size),
        })

    index_path = all_dir / "ALLmerged_index.tsv"
    with open(index_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("sector_center_deg\tgroup_index\tstart_frame\tend_frame\tlabel\ttime_axis_value\ttime_axis_unit\tstart_time_sec\tend_time_sec\tcenter_time_sec\torigin_file\ttrans_file\n")
        for r in index_rows:
            f.write(
                f'{r["sector_center_deg"]}\t{r["group_index"]}\t{r["start_frame"]}\t{r["end_frame"]}\t'
                f'{r["label"]}\t{r["time_axis_value"]}\t{r["time_axis_unit"]}\t'
                f'{r["start_time_sec"]}\t{r["end_time_sec"]}\t{r["center_time_sec"]}\t'
                f'{r["origin_file"]}\t{r["trans_file"]}\n'
            )

    manifest_path = all_dir / "ALLmerged_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump({
            "format": "ALLMerge+Trans-like",
            "unit": unit,
            "time_axis_unit": _time_axis_header(time_axis_unit),
            "effective_frame_time_sec": float(frame_time_sec),
            "poni_space_bin": int(poni_space_bin),
            "time_bin_label": int(time_bin_label),
            "accum_frames": int(accum_frames),
            "allmerge_axis_unit": str(allmerge_axis_unit),
            "allmerge_step_min": float(allmerge_step_min),
            "index_tsv": str(index_path),
            "files": written,
            "notes": [
                "Rich Origin: first column is radial axis in the selected pyFAI unit; following columns are spectra.",
                "Rich Trans: first column is frame_start, time_sec, or time_min according to time_axis_unit; following columns are intensities at each radial bin.",
                "ALLMerge-like Origin: first column follows Q(nm-1) when q_A^-1 is selected; following columns use actual time labels such as 0s, 1s, 2s or 0min, 1min.",
                "ALLMerge-like Trans: first column header is sec or min according to the actual label unit.",
                "For sector integration, one Origin/Trans pair is written per sector."
            ],
        }, f, indent=2, ensure_ascii=False)

    return all_dir, written, index_path, manifest_path


def _parse_optional_float(text):
    txt = str(text or "").strip()
    if txt == "" or txt.lower() in ["none", "nan", "null"]:
        return None
    try:
        return float(txt)
    except Exception:
        return None


def rebuild_allmerged_from_existing_txt(cfg: IntegrationConfig):
    """Rebuild only ALLmerged Origin/Trans DAT files from existing per-group TXT outputs.

    This is intentionally lightweight: it does NOT read images.h5, does NOT load PONI,
    and does NOT run pyFAI integration. It only rewrites axis labels / time axis /
    legacy ALLMerge files using txt/group_*.txt and integration_summary.tsv.
    """
    out_dir = Path(cfg.output_dir)
    summary_tsv = out_dir / "integration_summary.tsv"
    if not summary_tsv.exists():
        raise FileNotFoundError(
            f"integration_summary.tsv not found: {summary_tsv}\n"
            "先に一度だけ Run 1D Integration を実行してください。"
        )

    import csv
    results = []
    detected_unit = None
    detected_accum_frames = None
    missing = []

    with open(summary_tsv, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            try:
                gi = int(float(row.get("group_index", 0)))
                start = int(float(row.get("start_frame", 0)))
                end = int(float(row.get("end_frame", start)))
            except Exception:
                continue

            if detected_unit is None:
                detected_unit = str(row.get("unit", "") or "").strip() or None
            if detected_accum_frames is None:
                try:
                    detected_accum_frames = int(float(row.get("accum_frames", cfg.accum_frames)))
                except Exception:
                    detected_accum_frames = int(cfg.accum_frames)

            sec_val = _parse_optional_float(row.get("sector_center_deg", ""))
            sector_center = None if sec_val is None else float(sec_val)

            txt_raw = str(row.get("txt_path", "") or "").strip()
            if not txt_raw:
                # Standard fallback from group index / sector.
                if sector_center is None:
                    txt_path = out_dir / "txt" / f"group_{gi:05d}.txt"
                else:
                    txt_path = out_dir / "txt" / f"group_{gi:05d}_sector_{format_angle_label(sector_center)}deg.txt"
            else:
                txt_path = Path(txt_raw)
                if not txt_path.exists():
                    # Old summary may contain absolute paths from a different machine.
                    alt = out_dir / "txt" / txt_path.name
                    txt_path = alt if alt.exists() else txt_path

            if not txt_path.exists():
                missing.append(str(txt_path))
                continue

            try:
                arr = np.loadtxt(txt_path, skiprows=1, comments="#")
                arr = np.asarray(arr, dtype=np.float64)
                if arr.ndim == 1:
                    arr = arr.reshape(1, -1)
                if arr.shape[1] < 2:
                    raise ValueError(f"TXT has <2 columns: {txt_path}")
                radial = arr[:, 0]
                intensity = arr[:, 1]
                results.append((gi, start, end, radial, intensity, sector_center))
            except Exception as e:
                raise RuntimeError(f"Could not read integrated TXT: {txt_path}\n{e}") from e

    if missing:
        preview = "\n".join(missing[:8])
        more = "" if len(missing) <= 8 else f"\n... and {len(missing)-8} more"
        raise FileNotFoundError(f"Some integrated TXT files are missing:\n{preview}{more}")
    if not results:
        raise RuntimeError("No integrated TXT rows were found. Run 1D Integration first.")

    unit = detected_unit or cfg.unit
    accum_frames = detected_accum_frames or int(cfg.accum_frames)

    # Important: radial-axis-changing parameters are already baked into each TXT.
    # This rebuild intentionally uses the radial values saved in TXT and cannot
    # alter PONI/bin scaling, unit, npt, sector geometry, mask, frame range, etc.
    allmerged_dir, allmerged_files, allmerged_index_tsv, allmerged_manifest = write_allmerged_dat_outputs(
        cfg.output_dir,
        results,
        unit,
        frame_time_sec=cfg.effective_frame_time_sec,
        time_axis_unit=cfg.time_axis_unit,
        poni_space_bin=cfg.poni_space_bin,
        time_bin_label=cfg.time_bin_label,
        accum_frames=accum_frames,
        allmerge_step_min=cfg.allmerge_step_min,
        allmerge_axis_unit=getattr(cfg, "allmerge_axis_unit", "auto"),
    )

    # Record a small rebuild marker so it is clear that the heavy integration was not rerun.
    rebuild_info = {
        "mode": "axis_only_rebuild_from_existing_txt",
        "summary_tsv": str(summary_tsv),
        "n_profiles": int(len(results)),
        "unit": unit,
        "effective_frame_time_sec": float(cfg.effective_frame_time_sec),
        "time_axis_unit": cfg.time_axis_unit,
        "poni_space_bin": int(cfg.poni_space_bin),
        "time_bin_label": int(cfg.time_bin_label),
        "allmerge_axis_unit": str(getattr(cfg, "allmerge_axis_unit", "auto")),
        "allmerge_step_min": float(cfg.allmerge_step_min),
        "allmerged_dir": str(allmerged_dir),
        "allmerged_index_tsv": str(allmerged_index_tsv),
        "allmerged_manifest": str(allmerged_manifest),
        "allmerged_files": allmerged_files,
    }
    with open(out_dir / "ALLmerged_axis_rebuild_info.json", "w", encoding="utf-8") as f:
        json.dump(rebuild_info, f, indent=2, ensure_ascii=False)

    return allmerged_dir, len(results), allmerged_files


def run_integration(cfg: IntegrationConfig):
    ensure_dir(cfg.output_dir)

    shape, dtype = load_h5_shape(cfg.h5_path, cfg.image_dataset)
    if len(shape) != 3:
        raise ValueError(f"Image dataset must be 3D (T,H,W), got {shape}")
    T, H, W = shape

    start_frame = max(0, int(cfg.start_frame))
    end_frame = T - 1 if int(cfg.end_frame_inclusive) < 0 else min(T - 1, int(cfg.end_frame_inclusive))
    if end_frame < start_frame:
        raise ValueError("end_frame_inclusive must be >= start_frame")

    frame_count = end_frame - start_frame + 1
    accum_frames = max(1, int(cfg.accum_frames))
    target_time_bins = max(0, int(getattr(cfg, "target_time_bins", 0) or 0))
    group_specs = build_time_bin_specs(
        start_frame,
        end_frame,
        accum_frames,
        target_time_bins,
        time_bin_mode=getattr(cfg, "time_bin_mode", "frames"),
        effective_frame_time_sec=float(cfg.effective_frame_time_sec),
        profile_duration_value=float(getattr(cfg, "profile_duration_value", 1.0)),
        profile_duration_unit=getattr(cfg, "profile_duration_unit", "min"),
        include_partial_last_bin=bool(getattr(cfg, "include_partial_last_bin", False)),
    )
    n_groups = len(group_specs)
    if n_groups <= 0:
        raise ValueError("Selected frame range is smaller than the requested bin settings.")
    first_bin_frames = int(group_specs[0][2] - group_specs[0][1])
    effective_accum_for_axis = float(first_bin_frames)

    ext_mask, int_mask, final_mask = build_final_mask(cfg, (H, W))
    center_xy, angle_map, _ = estimate_center_and_angle_map_from_poni(cfg, (H, W))
    sector_centers = make_sector_centers(cfg.ref_angle_deg, cfg.sector_step_deg, cfg.paired_180) if cfg.sector_enable else []

    # warmup / first integration to determine radial axis
    ai = load_ai_with_optional_binning(cfg.poni_path, shape_hw=(H, W), poni_space_bin=cfg.poni_space_bin)
    method = parse_method(cfg.method)
    warm_frame = read_frame(cfg.h5_path, cfg.image_dataset, start_frame)
    warm_mask = final_mask
    if cfg.sector_enable:
        warm_inside = build_sector_inside_mask(angle_map, sector_centers[0], cfg.sector_width_deg, cfg.paired_180)
        warm_invalid = ~warm_inside
        if final_mask is not None:
            warm_invalid = np.logical_or(final_mask != 0, warm_invalid)
        warm_mask = warm_invalid.astype(np.uint8)

    warm_kwargs = dict(
        data=warm_frame,
        npt=int(cfg.npt),
        unit=cfg.unit,
        mask=warm_mask,
        polarization_factor=float(cfg.polarization_factor),
        method=method,
    )
    if cfg.dummy_value != "":
        warm_kwargs["dummy"] = float(cfg.dummy_value)
    if cfg.delta_dummy != "":
        warm_kwargs["delta_dummy"] = float(cfg.delta_dummy)
    warm_res = ai.integrate1d(**warm_kwargs)
    radial_axis = np.asarray(warm_res.radial, dtype=np.float32)

    # group specs respect arbitrary start_frame and optional target_time_bins.
    # Each spec is (group_index, start_frame, end_exclusive).
    groups_per_task = max(1, int(cfg.groups_per_task))
    tasks = list(chunked(group_specs, groups_per_task))
    processes = max(1, int(cfg.processes))

    summary_rows = []

    worker_args = []
    for group_chunk in tasks:
        worker_args.append((
            cfg.h5_path,
            cfg.image_dataset,
            cfg.poni_path,
            final_mask,
            group_chunk,
            accum_frames,
            cfg.accum_mode,
            cfg.npt,
            cfg.unit,
            cfg.method,
            float(cfg.polarization_factor),
            cfg.dummy_value,
            cfg.delta_dummy,
            cfg.sector_enable,
            sector_centers,
            cfg.sector_width_deg,
            cfg.paired_180,
            angle_map,
            (H, W),
            cfg.poni_space_bin,
        ))

    if processes == 1:
        results_nested = [integrate_group_range_worker(a) for a in worker_args]
    else:
        ctx = mp.get_context("spawn")
        group_chunks = [args[4] for args in worker_args]
        initargs = (
            cfg.h5_path,
            cfg.image_dataset,
            cfg.poni_path,
            final_mask,
            accum_frames,
            cfg.accum_mode,
            cfg.npt,
            cfg.unit,
            cfg.method,
            float(cfg.polarization_factor),
            cfg.dummy_value,
            cfg.delta_dummy,
            cfg.sector_enable,
            sector_centers,
            cfg.sector_width_deg,
            cfg.paired_180,
            angle_map,
            (H, W),
            cfg.poni_space_bin,
        )
        with ctx.Pool(processes=processes, initializer=worker_init_persistent, initargs=initargs) as pool:
            results_nested = list(pool.imap_unordered(integrate_group_range_worker_persistent, group_chunks, chunksize=1))

    results = []
    for lst in results_nested:
        results.extend(lst)
    results.sort(key=lambda x: x[0])

    txt_dir = Path(cfg.output_dir) / "txt"
    ensure_dir(txt_dir)

    # save radial axis once
    axis_txt = Path(cfg.output_dir) / "radial_axis.txt"
    np.savetxt(
        axis_txt,
        radial_axis.reshape(-1, 1),
        fmt="%.8e",
        header=cfg.unit,
        comments="",
        encoding="utf-8",
    )

    for gi, start, end, radial, intensity, sector_center in results:
        if sector_center is None:
            txt_path = txt_dir / f"group_{gi:05d}.txt"
        else:
            txt_path = txt_dir / f"group_{gi:05d}_sector_{format_angle_label(sector_center)}deg.txt"
        summary_rows.append({
            "group_index": int(gi),
            "start_frame": int(start),
            "end_frame": int(end),
            "accum_frames": int(end - start + 1),
            "target_time_bins": int(target_time_bins),
            "time_bin_mode": str(getattr(cfg, "time_bin_mode", "frames")),
            "profile_duration_value": float(getattr(cfg, "profile_duration_value", 0.0)),
            "profile_duration_unit": str(getattr(cfg, "profile_duration_unit", "")),
            "include_partial_last_bin": bool(getattr(cfg, "include_partial_last_bin", False)),
            "accum_mode": cfg.accum_mode,
            "npt": int(cfg.npt),
            "unit": cfg.unit,
            "sector_center_deg": None if sector_center is None else float(sector_center),
            "paired_180": bool(cfg.paired_180) if sector_center is not None else False,
            "effective_frame_time_sec": float(cfg.effective_frame_time_sec),
            "start_time_sec": float(start) * float(cfg.effective_frame_time_sec),
            "end_time_sec": float(end) * float(cfg.effective_frame_time_sec),
            "center_time_sec": 0.5 * (float(start) + float(end)) * float(cfg.effective_frame_time_sec),
            "poni_space_bin": int(cfg.poni_space_bin),
            "time_bin_label": int(cfg.time_bin_label),
            "txt_path": str(txt_path),
        })

        arr = np.column_stack([radial, intensity])
        header = f"{cfg.unit}\tintensity"
        if sector_center is not None:
            header += f"\n# angle_convention: right=0deg, counterclockwise\n# sector_center_deg={float(sector_center):.6f}\n# sector_width_deg={float(cfg.sector_width_deg):.6f}\n# paired_180={bool(cfg.paired_180)}"
        np.savetxt(
            txt_path,
            arr,
            fmt="%.8e",
            delimiter="\t",
            header=header,
            comments="",
            encoding="utf-8",
        )


    # ALLMerge+Trans-like matrix outputs
    allmerged_dir, allmerged_files, allmerged_index_tsv, allmerged_manifest = write_allmerged_dat_outputs(
        cfg.output_dir,
        results,
        cfg.unit,
        frame_time_sec=cfg.effective_frame_time_sec,
        time_axis_unit=cfg.time_axis_unit,
        poni_space_bin=cfg.poni_space_bin,
        time_bin_label=cfg.time_bin_label,
        accum_frames=max(1, int(round(effective_accum_for_axis))),
        allmerge_step_min=cfg.allmerge_step_min,
        allmerge_axis_unit=getattr(cfg, "allmerge_axis_unit", "auto"),
    )

    # summary as txt-only TSV
    summary_tsv = Path(cfg.output_dir) / "integration_summary.tsv"
    with open(summary_tsv, "w", encoding="utf-8") as f:
        f.write("group_index\tstart_frame\tend_frame\tstart_time_sec\tend_time_sec\tcenter_time_sec\teffective_frame_time_sec\taccum_frames\ttarget_time_bins\taccum_mode\tnpt\tunit\tponi_space_bin\ttime_bin_label\tsector_center_deg\tpaired_180\ttxt_path\n")
        for r in summary_rows:
            f.write(
                f'{r["group_index"]}\t{r["start_frame"]}\t{r["end_frame"]}\t'
                f'{r["start_time_sec"]}\t{r["end_time_sec"]}\t{r["center_time_sec"]}\t{r["effective_frame_time_sec"]}\t'
                f'{r["accum_frames"]}\t{r["target_time_bins"]}\t{r["accum_mode"]}\t{r["npt"]}\t{r["unit"]}\t'
                f'{r["poni_space_bin"]}\t{r["time_bin_label"]}\t'
                f'{r["sector_center_deg"]}\t{r["paired_180"]}\t{r["txt_path"]}\n'
            )

    with open(Path(cfg.output_dir) / "integration_config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2, ensure_ascii=False)

    info = {
        "dataset_shape": list(shape),
        "dataset_dtype": dtype,
        "used_frame_range": [int(group_specs[0][1]), int(group_specs[-1][2] - 1)],
        "frame_count_used": int(sum(e - s for _, s, e in group_specs)),
        "accum_frames": int(accum_frames),
        "target_time_bins": int(target_time_bins),
        "time_bin_mode": str(getattr(cfg, "time_bin_mode", "frames")),
        "profile_duration_value": float(getattr(cfg, "profile_duration_value", 0.0)),
        "profile_duration_unit": str(getattr(cfg, "profile_duration_unit", "")),
        "include_partial_last_bin": bool(getattr(cfg, "include_partial_last_bin", False)),
        "frames_per_profile_first_bin": int(first_bin_frames),
        "profile_duration_sec_first_bin": float(first_bin_frames) * float(cfg.effective_frame_time_sec),
        "time_bin_specs": describe_time_bin_specs(group_specs),
        "effective_frame_time_sec": float(cfg.effective_frame_time_sec),
        "time_axis_unit": cfg.time_axis_unit,
        "poni_space_bin": int(cfg.poni_space_bin),
        "time_bin_label": int(cfg.time_bin_label),
        "n_groups": int(n_groups),
        "processes": int(processes),
        "groups_per_task": int(groups_per_task),
        "txt_dir": str(txt_dir),
        "radial_axis_txt": str(axis_txt),
        "allmerged_dir": str(allmerged_dir),
        "allmerged_index_tsv": str(allmerged_index_tsv),
        "allmerged_manifest": str(allmerged_manifest),
        "allmerged_files": allmerged_files,
        "summary_tsv": str(summary_tsv),
        "sector_enable": bool(cfg.sector_enable),
        "reference_angle_deg": float(cfg.ref_angle_deg),
        "sector_step_deg": float(cfg.sector_step_deg),
        "sector_width_deg": float(cfg.sector_width_deg),
        "paired_180": bool(cfg.paired_180),
        "angle_convention": "right=0deg, counterclockwise",
        "beam_center_estimate_xy": [int(center_xy[0]), int(center_xy[1])],
    }
    with open(Path(cfg.output_dir) / "run_info.json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)

    return txt_dir, len(summary_rows)


# -----------------------------
# tkinter gui
# -----------------------------
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("H5 -> 1D (PONI/MASK + ALLMerge axis rebuild)")
        self.geometry("1120x920")

        self.vars = {}
        self._build()

    def _sv(self, key, value=""):
        v = tk.StringVar(value=value)
        self.vars[key] = v
        return v

    def _bv(self, key, value=False):
        v = tk.BooleanVar(value=value)
        self.vars[key] = v
        return v

    def _iv(self, key, value=0):
        v = tk.IntVar(value=value)
        self.vars[key] = v
        return v

    def _build(self):
        """Build a simplified UI.

        The old UI exposed every internal parameter in one long page.  This version
        keeps the normal workflow on the first tab and moves rarely-used pyFAI /
        sector / speed settings to Advanced.  Variable names are kept identical so
        the existing processing code below can be reused unchanged.
        """
        pad = {"padx": 6, "pady": 4}

        # ------------------------------------------------------------------
        # Create all variables once.  Easy and Advanced tabs share them.
        # ------------------------------------------------------------------
        defaults_s = {
            "h5_path": "",
            "poni_path": "",
            "mask_tif_path": "",
            "output_dir": "",
            "image_dataset": "/entry/data/images",
            "internal_mask_dataset": "/entry/instrument/detector/mask",
            "external_mask_semantics": "dxb_valid",
            "accum_frames": "200",
            "accum_mode": "mean",
            "target_time_bins": "100",
            "time_bin_mode": "frames",
            "profile_duration_value": "1.0",
            "profile_duration_unit": "min",
            "effective_frame_time_sec": "0.05",
            "time_axis_unit": "min",
            "poni_space_bin": "1",
            "time_bin_label": "1",
            "allmerge_axis_unit": "auto",
            "allmerge_step_min": "0",
            "npt": "2000",
            "unit": "q_A^-1",
            "method": "bbox_csr_cython",
            "polarization_factor": "0.0",
            "dummy_value": "",
            "delta_dummy": "",
            "start_frame": "0",
            "end_frame_inclusive": "-1",
            "preview_frame": "0",
            "preview_percentile_low": "1.0",
            "preview_percentile_high": "99.5",
            "contour_levels": "6",
            "processes": str(max(1, (os.cpu_count() or 2) // 2)),
            "groups_per_task": "4",
            "ref_angle_deg": "0.0",
            "sector_step_deg": "10.0",
            "sector_width_deg": "10.0",
            "input_preset": "01_h5 original / 未Bin画像",
        }
        defaults_b = {
            "use_internal_h5_mask": True,
            "invert_internal_h5_mask": True,
            "invert_external_mask": True,
            "sector_enable": False,
            "paired_180": True,
            "include_partial_last_bin": False,
        }
        for k, v in defaults_s.items():
            self._sv(k, v)
        for k, v in defaults_b.items():
            self._bv(k, v)

        root = ttk.Frame(self)
        root.pack(fill="both", expand=True)
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=0)
        root.rowconfigure(1, weight=0)
        root.rowconfigure(2, weight=1)

        nb = ttk.Notebook(root)
        nb.grid(row=0, column=0, sticky="nsew", padx=6, pady=6)
        easy = ttk.Frame(nb)
        adv = ttk.Frame(nb)
        nb.add(easy, text="Easy / 通常ここだけ")
        nb.add(adv, text="Advanced / 詳細")

        # ------------------------------------------------------------------
        # Small UI helpers
        # ------------------------------------------------------------------
        def path_row(parent, row, label, key, kind, button_text="Browse", clear=False):
            ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", **pad)
            ttk.Entry(parent, textvariable=self.vars[key], width=82).grid(row=row, column=1, sticky="ew", **pad)
            box = ttk.Frame(parent)
            box.grid(row=row, column=2, sticky="ew", **pad)
            ttk.Button(box, text=button_text, command=lambda: self._browse(key, kind)).pack(side="left", padx=(0, 4))
            if clear:
                ttk.Button(box, text="Clear", command=lambda: self.vars[key].set("")).pack(side="left")
            return row + 1

        def entry_row(parent, row, label, key, hint="", width=12):
            ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", **pad)
            ttk.Entry(parent, textvariable=self.vars[key], width=width).grid(row=row, column=1, sticky="w", **pad)
            if hint:
                ttk.Label(parent, text=hint, foreground="#666").grid(row=row, column=2, sticky="w", **pad)
            return row + 1

        def section(parent, row, title, subtitle=""):
            ttk.Label(parent, text=title, font=("TkDefaultFont", 10, "bold")).grid(row=row, column=0, columnspan=3, sticky="w", padx=6, pady=(10, 2))
            if subtitle:
                ttk.Label(parent, text=subtitle, foreground="#555").grid(row=row+1, column=0, columnspan=3, sticky="w", padx=6, pady=(0, 4))
                return row + 2
            return row + 1

        # ------------------------------------------------------------------
        # Easy tab
        # ------------------------------------------------------------------
        r = 0
        easy.columnconfigure(1, weight=1)
        r = section(easy, r, "1. ファイル選択", "必須は H5 / PONI / 出力先。外部MASKは必要な時だけ選択。")
        r = path_row(easy, r, "images.h5", "h5_path", "file_h5")
        r = path_row(easy, r, "PONI file", "poni_path", "file_poni")
        r = path_row(easy, r, "External MASK TIF", "mask_tif_path", "file_tif", button_text="Browse mask", clear=True)
        r = path_row(easy, r, "Output folder", "output_dir", "dir")

        r = section(easy, r, "2. 入力H5の種類", "ここがq軸の最重要設定。Bin済みH5を使うなら、元PONIに合わせて空間Bin倍率を指定。")
        ttk.Label(easy, text="Input preset").grid(row=r, column=0, sticky="w", **pad)
        preset = ttk.Combobox(
            easy,
            textvariable=self.vars["input_preset"],
            values=[
                "01_h5 original / 未Bin画像",
                "02_bin 5x5 / Bin済み",
                "02_bin 8x8 / Bin済み",
                "02_bin 10x10 / Bin済み",
                "02_bin 12x12 / Bin済み",
                "Custom / 手入力",
            ],
            width=30,
            state="readonly",
        )
        preset.grid(row=r, column=1, sticky="w", **pad)
        preset.bind("<<ComboboxSelected>>", lambda _e: self._apply_input_preset())
        ttk.Label(easy, text="選ぶだけで PONI space bin factor を設定", foreground="#006400").grid(row=r, column=2, sticky="w", **pad)
        r += 1
        r = entry_row(easy, r, "PONI space bin factor", "poni_space_bin", "01_h5なら1。02_bin/10x10なら10。変更後はRun必要。")

        r = section(easy, r, "3. 時間方向のまとめ方", "ここが一番重要。『何フレームごと』または『何分ごと』で1本の1Dを作るかを指定します。")
        ttk.Label(easy, text="Effective frame time [s/frame]").grid(row=r, column=0, sticky="w", **pad)
        ttk.Entry(easy, textvariable=self.vars["effective_frame_time_sec"], width=12).grid(row=r, column=1, sticky="w", **pad)
        ttk.Label(easy, text="例: 1 frame = 0.05 s なら 0.05。時間bin計算の基準。", foreground="#8a4b00").grid(row=r, column=2, sticky="w", **pad)
        r += 1
        ttk.Label(easy, text="Time bin mode").grid(row=r, column=0, sticky="w", **pad)
        ttk.Combobox(
            easy,
            textvariable=self.vars["time_bin_mode"],
            values=["frames", "duration", "profiles"],
            width=12,
            state="readonly",
        ).grid(row=r, column=1, sticky="w", **pad)
        ttk.Label(easy, text="frames=何枚ごと / duration=何分ごと / profiles=全体をN分割", foreground="#555").grid(row=r, column=2, sticky="w", **pad)
        r += 1
        r = entry_row(easy, r, "Frames/profile", "accum_frames", "mode=frames の時に使用。例: 200 frames/profile → 15000 framesなら75本。")
        ttk.Label(easy, text="Duration/profile").grid(row=r, column=0, sticky="w", **pad)
        durbox = ttk.Frame(easy)
        durbox.grid(row=r, column=1, sticky="w", **pad)
        ttk.Entry(durbox, textvariable=self.vars["profile_duration_value"], width=8).pack(side="left")
        ttk.Combobox(durbox, textvariable=self.vars["profile_duration_unit"], values=["sec", "min"], width=6, state="readonly").pack(side="left", padx=(4, 0))
        ttk.Label(easy, text="mode=duration の時に使用。例: 1 min/profile。", foreground="#555").grid(row=r, column=2, sticky="w", **pad)
        r += 1
        r = entry_row(easy, r, "Target profiles", "target_time_bins", "mode=profiles の時だけ使用。旧100分割方式。通常は使わない。")
        ttk.Checkbutton(easy, text="最後の端数binも出す / include partial last bin", variable=self.vars["include_partial_last_bin"]).grid(row=r, column=0, columnspan=3, sticky="w", **pad)
        r += 1
        ttk.Label(easy, text="Intensity mode").grid(row=r, column=0, sticky="w", **pad)
        ttk.Combobox(easy, textvariable=self.vars["accum_mode"], values=["mean", "sum"], width=10, state="readonly").grid(row=r, column=1, sticky="w", **pad)
        ttk.Label(easy, text="時間bin比較なら mean 推奨。sum は積算強度。", foreground="#006400").grid(row=r, column=2, sticky="w", **pad)
        r += 1
        btns_time = ttk.Frame(easy)
        btns_time.grid(row=r, column=0, columnspan=3, sticky="ew", padx=6, pady=(2, 8))
        ttk.Button(btns_time, text="Check time-bin plan / 時間bin確認", command=self.on_check_time_bin_plan).pack(side="left", padx=(0, 6))
        ttk.Button(btns_time, text="Set 200 frames/profile", command=self._set_recommended_200frames).pack(side="left", padx=(0, 6))
        ttk.Button(btns_time, text="Set 1 sec/profile", command=self._set_recommended_1sec).pack(side="left", padx=(0, 6))
        ttk.Button(btns_time, text="Set 1 min/profile", command=self._set_recommended_1min).pack(side="left")
        r += 1

        r = section(easy, r, "4. 出力軸", "軸ラベルだけ。変更後は『Rebuild ALLMerge axis only』でOK。基本は auto のまま。")
        ttk.Label(easy, text="Output time label unit").grid(row=r, column=0, sticky="w", **pad)
        ttk.Combobox(easy, textvariable=self.vars["allmerge_axis_unit"], values=["auto", "sec", "min"], width=10, state="readonly").grid(row=r, column=1, sticky="w", **pad)
        ttk.Label(easy, text="auto: 1sなど60秒未満は 0s,1s... / 1分以上は 0min,1min...", foreground="#006400").grid(row=r, column=2, sticky="w", **pad)
        r += 1
        ttk.Label(easy, text="Detailed Trans axis").grid(row=r, column=0, sticky="w", **pad)
        ttk.Combobox(easy, textvariable=self.vars["time_axis_unit"], values=["frame", "sec", "min"], width=10, state="readonly").grid(row=r, column=1, sticky="w", **pad)
        ttk.Label(easy, text="詳細版Trans用。通常はOutput time label unitだけ見ればOK。", foreground="#555").grid(row=r, column=2, sticky="w", **pad)
        r += 1
        r = entry_row(easy, r, "Manual label step [min/profile]", "allmerge_step_min", "通常0。ここに1や6を入れると実時間を無視してminラベルを強制するので注意。")
        axis_btns = ttk.Frame(easy)
        axis_btns.grid(row=r, column=0, columnspan=3, sticky="ew", padx=6, pady=(0, 4))
        ttk.Button(axis_btns, text="Actual labels / 実時間ラベル", command=self._set_allmerge_actual_time).pack(side="left", padx=(0, 6))
        ttk.Button(axis_btns, text="Force 1 min labels / 非推奨", command=self._set_allmerge_1min_labels).pack(side="left", padx=(0, 6))
        ttk.Button(axis_btns, text="Force 6 min labels / 非推奨", command=self._set_allmerge_6min_labels).pack(side="left")
        r += 1
        r = entry_row(easy, r, "Time bin label / t値メモ", "time_bin_label", "メタ情報用。時間計算には使いません。")

        r = section(easy, r, "5. MASK", "DXB標準は 白=使う / 黒=捨てる。pyFAIには内部で反転します。")
        ttk.Checkbutton(easy, text="Use internal H5 mask / H5内マスクを使う", variable=self.vars["use_internal_h5_mask"]).grid(row=r, column=0, sticky="w", **pad)
        ttk.Checkbutton(easy, text="Internal H5 is DXB style: 1=use,0=exclude → invert for pyFAI", variable=self.vars["invert_internal_h5_mask"]).grid(row=r, column=1, columnspan=2, sticky="w", **pad)
        r += 1
        ttk.Label(easy, text="External MASK meaning").grid(row=r, column=0, sticky="w", **pad)
        ttk.Radiobutton(easy, text="DXB style: white/255/1 = USE, black/0 = EXCLUDE", variable=self.vars["external_mask_semantics"], value="dxb_valid", command=self._sync_external_mask_semantics).grid(row=r, column=1, columnspan=2, sticky="w", **pad)
        r += 1
        ttk.Radiobutton(easy, text="pyFAI style: white/255/1 = EXCLUDE, black/0 = USE", variable=self.vars["external_mask_semantics"], value="pyfai_invalid", command=self._sync_external_mask_semantics).grid(row=r, column=1, columnspan=2, sticky="w", **pad)
        r += 1
        self.mask_hint_var = tk.StringVar(value="Selected: DXB style external mask. white=USE, black=EXCLUDE. It will be inverted for pyFAI automatically.")
        ttk.Label(easy, textvariable=self.mask_hint_var, foreground="#555").grid(row=r, column=0, columnspan=2, sticky="w", padx=6, pady=2)
        ttk.Button(easy, text="Check MASK / 向きを確認", command=self.on_check_mask).grid(row=r, column=2, sticky="ew", **pad)
        r += 1

        # Clear guidance box
        guide = ttk.LabelFrame(easy, text="どのボタンを押すか")
        guide.grid(row=r, column=0, columnspan=3, sticky="ew", padx=6, pady=(8, 6))
        ttk.Label(guide, text="Run 1D Integration：H5 / PONI / MASK / PONI bin / 時間bin設定 / frame範囲 / npt / unit を変えた時", foreground="#8a4b00").grid(row=0, column=0, sticky="w", padx=8, pady=3)
        ttk.Label(guide, text="Rebuild ALLMerge axis only：Effective frame time / Trans axis / ALLMerge step だけ変えた時", foreground="#006400").grid(row=1, column=0, sticky="w", padx=8, pady=3)
        r += 1

        # ------------------------------------------------------------------
        # Advanced tab
        # ------------------------------------------------------------------
        a = 0
        adv.columnconfigure(1, weight=1)
        a = section(adv, a, "Dataset paths", "通常は変更不要。H5構造が違う場合だけ変更。")
        a = entry_row(adv, a, "Image dataset", "image_dataset", "標準: /entry/data/images", width=42)
        a = entry_row(adv, a, "Internal mask dataset", "internal_mask_dataset", "標準: /entry/instrument/detector/mask", width=42)

        a = section(adv, a, "pyFAI integration", "q点数・単位・アルゴリズム。変更後はRun必要。")
        a = entry_row(adv, a, "npt", "npt", "q/2θ方向の点数。")
        ttk.Label(adv, text="unit").grid(row=a, column=0, sticky="w", **pad)
        ttk.Combobox(adv, textvariable=self.vars["unit"], values=["q_A^-1", "q_nm^-1", "2th_deg", "2th_rad", "r_mm"], width=12).grid(row=a, column=1, sticky="w", **pad)
        ttk.Label(adv, text="AllMerge互換では q_A^-1 は Q(nm-1) に×10変換", foreground="#555").grid(row=a, column=2, sticky="w", **pad)
        a += 1
        a = entry_row(adv, a, "pyFAI method", "method", "通常: bbox_csr_cython", width=22)
        a = entry_row(adv, a, "polarization_factor", "polarization_factor", "通常0.0。必要時のみ変更。")
        a = entry_row(adv, a, "dummy value", "dummy_value", "無効値扱いする画素値。空欄で未使用。")
        a = entry_row(adv, a, "delta_dummy", "delta_dummy", "dummy許容幅。空欄で未使用。")

        a = section(adv, a, "Frame range / Preview", "開始・終了フレームを限定したい時だけ変更。")
        a = entry_row(adv, a, "Start frame", "start_frame", "0始まり。")
        a = entry_row(adv, a, "End frame inclusive", "end_frame_inclusive", "-1なら最後まで。")
        a = entry_row(adv, a, "Preview frame", "preview_frame", "プレビューのみ。")
        a = entry_row(adv, a, "Contour levels", "contour_levels", "プレビューのみ。")
        ttk.Label(adv, text="Preview percentile low/high").grid(row=a, column=0, sticky="w", **pad)
        pv = ttk.Frame(adv)
        pv.grid(row=a, column=1, sticky="w")
        ttk.Entry(pv, textvariable=self.vars["preview_percentile_low"], width=8).pack(side="left", padx=4)
        ttk.Entry(pv, textvariable=self.vars["preview_percentile_high"], width=8).pack(side="left", padx=4)
        ttk.Label(adv, text="表示の明るさのみ。積分結果には影響しません。", foreground="#555").grid(row=a, column=2, sticky="w", **pad)
        a += 1

        a = section(adv, a, "Sector integration", "方位角別に1D化したい時だけON。通常OFF。")
        ttk.Checkbutton(adv, text="Enable sector integration", variable=self.vars["sector_enable"]).grid(row=a, column=0, sticky="w", **pad)
        ttk.Checkbutton(adv, text="Pair opposite sector θ and θ+180°", variable=self.vars["paired_180"]).grid(row=a, column=1, sticky="w", **pad)
        a += 1
        a = entry_row(adv, a, "Reference angle [deg]", "ref_angle_deg", "right=0°, counterclockwise")
        a = entry_row(adv, a, "Sector step [deg]", "sector_step_deg", "例: 10")
        a = entry_row(adv, a, "Sector width [deg]", "sector_width_deg", "例: 10")

        a = section(adv, a, "Speed", "結果には影響しません。重い場合だけ調整。")
        a = entry_row(adv, a, "Processes", "processes", "5950Xなら8〜16程度。HDDなら上げすぎ注意。")
        a = entry_row(adv, a, "Groups per task", "groups_per_task", "通常4〜16。")

        # ------------------------------------------------------------------
        # Common buttons + log
        # ------------------------------------------------------------------
        btnrow = ttk.Frame(root)
        btnrow.grid(row=1, column=0, sticky="ew", padx=6, pady=(0, 4))
        btnrow.columnconfigure(0, weight=1)
        btnrow.columnconfigure(1, weight=1)
        btnrow.columnconfigure(2, weight=1)
        btnrow.columnconfigure(3, weight=1)
        ttk.Button(btnrow, text="Check H5 shape", command=self.on_check_shape).grid(row=0, column=0, sticky="ew", padx=4, pady=2)
        ttk.Button(btnrow, text="Create Preview", command=self.on_preview).grid(row=0, column=1, sticky="ew", padx=4, pady=2)
        ttk.Button(btnrow, text="Run 1D Integration / 重い再計算", command=self.on_run).grid(row=0, column=2, sticky="ew", padx=4, pady=2)
        ttk.Button(btnrow, text="Rebuild ALLMerge axis only / 軸だけ再出力", command=self.on_rebuild_allmerge_axis).grid(row=0, column=3, sticky="ew", padx=4, pady=2)

        logbox = ttk.LabelFrame(root, text="Log")
        logbox.grid(row=2, column=0, sticky="nsew", padx=6, pady=(0, 6))
        logbox.columnconfigure(0, weight=1)
        logbox.rowconfigure(0, weight=1)
        self.log = tk.Text(logbox, height=12, wrap="word")
        self.log.grid(row=0, column=0, sticky="nsew", padx=6, pady=6)

        self._sync_external_mask_semantics()
        self._apply_input_preset()
        self._append_log("Easy UI loaded. Recommended workflow: select H5/PONI/output -> Check MASK -> Create Preview -> Run 1D Integration.")
        self._append_log("Default is 200 frames/profile, frame time=0.05 s, intensity mode=mean. This means each profile is 10 s = 0.1666667 min unless you force manual labels.")

    def _apply_input_preset(self):
        txt = self.vars.get("input_preset").get() if "input_preset" in self.vars else ""
        mapping = {
            "01_h5 original / 未Bin画像": 1,
            "02_bin 5x5 / Bin済み": 5,
            "02_bin 8x8 / Bin済み": 8,
            "02_bin 10x10 / Bin済み": 10,
            "02_bin 12x12 / Bin済み": 12,
        }
        if txt in mapping:
            self.vars["poni_space_bin"].set(str(mapping[txt]))
        try:
            if hasattr(self, "log"):
                self._append_log(f"Input preset: {txt} -> PONI space bin factor = {self.vars['poni_space_bin'].get()}")
        except Exception:
            pass

    def _set_recommended_100(self):
        # Backward-compatible button target from the previous UI.
        self.vars["time_bin_mode"].set("profiles")
        self.vars["target_time_bins"].set("100")
        self.vars["accum_mode"].set("mean")
        self.vars["time_axis_unit"].set("sec")
        if "allmerge_axis_unit" in self.vars:
            self.vars["allmerge_axis_unit"].set("auto")
        self.vars["allmerge_step_min"].set("0")
        self.vars["external_mask_semantics"].set("dxb_valid")
        self.vars["use_internal_h5_mask"].set(True)
        self.vars["invert_internal_h5_mask"].set(True)
        self._sync_external_mask_semantics()
        self._append_log("Set profiles mode: split selected range into 100 profiles. This is not fixed-duration binning.")

    def _set_recommended_200frames(self):
        self.vars["time_bin_mode"].set("frames")
        self.vars["accum_frames"].set("200")
        self.vars["accum_mode"].set("mean")
        self.vars["time_axis_unit"].set("min")
        self.vars["allmerge_step_min"].set("0")
        self.vars["include_partial_last_bin"].set(False)
        self._append_log("Set frames mode: 200 frames/profile. With 15000 frames this produces 75 profiles exactly.")
        self.on_check_time_bin_plan()


    def _set_recommended_1sec(self):
        self.vars["time_bin_mode"].set("duration")
        self.vars["profile_duration_value"].set("1")
        self.vars["profile_duration_unit"].set("sec")
        self.vars["accum_mode"].set("mean")
        self.vars["time_axis_unit"].set("sec")
        if "allmerge_axis_unit" in self.vars:
            self.vars["allmerge_axis_unit"].set("auto")
        self.vars["allmerge_step_min"].set("0")
        self.vars["include_partial_last_bin"].set(False)
        self._append_log("Set duration mode: 1 sec/profile. At 0.05 s/frame this means 20 frames/profile.")
        self.on_check_time_bin_plan()

    def _set_recommended_1min(self):
        self.vars["time_bin_mode"].set("duration")
        self.vars["profile_duration_value"].set("1")
        self.vars["profile_duration_unit"].set("min")
        self.vars["accum_mode"].set("mean")
        self.vars["time_axis_unit"].set("min")
        if "allmerge_axis_unit" in self.vars:
            self.vars["allmerge_axis_unit"].set("auto")
        self.vars["allmerge_step_min"].set("0")
        self.vars["include_partial_last_bin"].set(False)
        self._append_log("Set duration mode: 1 min/profile. Frames/profile is calculated from frame time.")
        self.on_check_time_bin_plan()

    def on_check_time_bin_plan(self):
        try:
            cfg = self._cfg()
            if not cfg.h5_path:
                # Still show arithmetic if no H5 is selected, using the common example.
                T = 15000
                start_frame = 0
                end_frame = T - 1
                self._append_log("No H5 selected: showing example calculation with 15000 frames.")
            else:
                shape, dtype = load_h5_shape(cfg.h5_path, cfg.image_dataset)
                T = int(shape[0])
                start_frame = max(0, int(cfg.start_frame))
                end_frame = T - 1 if int(cfg.end_frame_inclusive) < 0 else min(T - 1, int(cfg.end_frame_inclusive))
            specs = build_time_bin_specs(
                start_frame,
                end_frame,
                int(cfg.accum_frames),
                int(cfg.target_time_bins),
                time_bin_mode=cfg.time_bin_mode,
                effective_frame_time_sec=float(cfg.effective_frame_time_sec),
                profile_duration_value=float(cfg.profile_duration_value),
                profile_duration_unit=cfg.profile_duration_unit,
                include_partial_last_bin=bool(cfg.include_partial_last_bin),
            )
            if not specs:
                self._append_log("Time-bin plan: no output profiles. Check settings.")
                return
            sizes = [e - s for _, s, e in specs]
            first_sec = sizes[0] * float(cfg.effective_frame_time_sec)
            total_frames = int(end_frame - start_frame + 1)
            total_sec = total_frames * float(cfg.effective_frame_time_sec)
            self._append_log("=== Time-bin plan ===")
            self._append_log(f"Selected frames: {total_frames} frames ({total_sec:.3f} s = {total_sec/60:.3f} min)")
            self._append_log(f"Mode: {cfg.time_bin_mode} / frame time: {float(cfg.effective_frame_time_sec):.6g} s/frame")
            self._append_log(f"Output profiles: {len(specs)} / frames per profile: {min(sizes)}" + (f"..{max(sizes)}" if min(sizes) != max(sizes) else ""))
            actual_step_min = first_sec / 60.0
            manual_step = float(cfg.allmerge_step_min or 0.0)
            if manual_step > 0:
                axis_unit = "min"
                axis_step = manual_step
                axis_text = f"0min, {_format_time_axis_value(axis_step, axis_unit, True)}, {_format_time_axis_value(axis_step*2, axis_unit, True)} ... (manual override)"
            else:
                axis_unit = _axis_unit_from_step(first_sec, getattr(cfg, "allmerge_axis_unit", "auto"))
                axis_step = first_sec if axis_unit == "sec" else actual_step_min
                axis_text = f"0{'s' if axis_unit=='sec' else 'min'}, {_format_time_axis_value(axis_step, axis_unit, True)}, {_format_time_axis_value(axis_step*2, axis_unit, True)} ... (actual time)"
            self._append_log(f"First profile width: {first_sec:.3f} s = {actual_step_min:.6g} min")
            self._append_log(f"Output column labels: {axis_text}")
            if manual_step > 0 and abs(manual_step - actual_step_min) > max(1e-9, abs(actual_step_min) * 1e-6):
                self._append_log(f"WARNING: manual label step ({manual_step:.6g} min) != actual bin width ({actual_step_min:.6g} min).")
            if abs(float(cfg.effective_frame_time_sec) - 0.05) < 1e-12 and int(cfg.accum_frames) == 200 and str(cfg.time_bin_mode).lower() == "frames":
                self._append_log("Note: 200 frames × 0.05 s = 10 s. 1 sec needs 20 frames; true 1 min needs 1200 frames.")
            if str(cfg.time_bin_mode).lower() == "duration" and str(cfg.profile_duration_unit).lower().startswith("sec"):
                self._append_log(f"Duration mode: Frames/profile is ignored. Calculated frames/profile = round({float(cfg.profile_duration_value)} sec / {float(cfg.effective_frame_time_sec)} sec) = {sizes[0]}.")
        except Exception as e:
            messagebox.showerror("Time-bin plan error", str(e))
            self._append_log("Time-bin plan error: " + repr(e))

    def _set_allmerge_actual_time(self):
        self.vars["allmerge_step_min"].set("0")
        if "allmerge_axis_unit" in self.vars:
            self.vars["allmerge_axis_unit"].set("auto")
        self._append_log("Output labels set to actual time. auto uses seconds for sub-minute bins, minutes for >=60 s bins.")
        self.on_check_time_bin_plan()

    def _set_allmerge_1min_labels(self):
        self.vars["allmerge_step_min"].set("1")
        self._append_log("ALLMerge labels forced to 1 min/profile: columns become 0min, 1min, 2min... This changes labels only, not the integrated data.")
        self.on_check_time_bin_plan()

    def _set_allmerge_6min_labels(self):
        self.vars["allmerge_step_min"].set("6")
        self._append_log("ALLMerge labels forced to 6 min/profile: columns become 0min, 6min, 12min... This changes labels only, not the integrated data.")
        self.on_check_time_bin_plan()

    def _sync_external_mask_semantics(self):
        """Keep the user-facing radio choice and the backend pyFAI invert flag synchronized."""
        mode = self.vars.get("external_mask_semantics").get() if "external_mask_semantics" in self.vars else "dxb_valid"
        # to_pyfai_mask(): invert=True means raw nonzero becomes USE instead of EXCLUDE.
        # DXB mask: 1=use -> invert=True. pyFAI mask: 1=exclude -> invert=False.
        if "invert_external_mask" in self.vars:
            self.vars["invert_external_mask"].set(mode == "dxb_valid")
        if hasattr(self, "mask_hint_var"):
            if mode == "dxb_valid":
                self.mask_hint_var.set("Selected: DXB style external mask. white/255/1 = USE, black/0 = EXCLUDE. It will be inverted for pyFAI automatically.")
            else:
                self.mask_hint_var.set("Selected: pyFAI style external mask. white/255/1 = EXCLUDE, black/0 = USE. No inversion will be applied.")

    def _clear_mask_path(self):
        if "mask_tif_path" in self.vars:
            self.vars["mask_tif_path"].set("")

    def _browse(self, key, kind):
        if kind == "file_h5":
            p = filedialog.askopenfilename(filetypes=[("HDF5", "*.h5 *.hdf5"), ("All", "*.*")])
        elif kind == "file_poni":
            p = filedialog.askopenfilename(filetypes=[("PONI", "*.poni"), ("All", "*.*")])
        elif kind == "file_tif":
            p = filedialog.askopenfilename(title="Select external MASK TIF", filetypes=[("TIF mask", "*.tif *.tiff"), ("All", "*.*")])
        elif kind == "dir":
            p = filedialog.askdirectory()
        else:
            p = ""
        if p:
            self.vars[key].set(p)

    def _append_log(self, msg):
        self.log.insert("end", str(msg) + "\n")
        self.log.see("end")
        self.update_idletasks()

    def _cfg(self) -> IntegrationConfig:
        return IntegrationConfig(
            h5_path=self.vars["h5_path"].get().strip(),
            poni_path=self.vars["poni_path"].get().strip(),
            mask_tif_path=self.vars["mask_tif_path"].get().strip(),
            output_dir=self.vars["output_dir"].get().strip(),
            image_dataset=self.vars["image_dataset"].get().strip(),
            internal_mask_dataset=self.vars["internal_mask_dataset"].get().strip(),
            use_internal_h5_mask=bool(self.vars["use_internal_h5_mask"].get()),
            invert_internal_h5_mask=bool(self.vars["invert_internal_h5_mask"].get()),
            invert_external_mask=bool(self.vars["invert_external_mask"].get()),
            accum_frames=int(self.vars["accum_frames"].get()),
            accum_mode=self.vars["accum_mode"].get().strip(),
            target_time_bins=int(self.vars["target_time_bins"].get()),
            time_bin_mode=self.vars["time_bin_mode"].get().strip(),
            profile_duration_value=float(self.vars["profile_duration_value"].get()),
            profile_duration_unit=self.vars["profile_duration_unit"].get().strip(),
            include_partial_last_bin=bool(self.vars["include_partial_last_bin"].get()),
            effective_frame_time_sec=float(self.vars["effective_frame_time_sec"].get()),
            time_axis_unit=self.vars["time_axis_unit"].get().strip(),
            poni_space_bin=int(self.vars["poni_space_bin"].get()),
            time_bin_label=int(self.vars["time_bin_label"].get()),
            allmerge_axis_unit=self.vars.get("allmerge_axis_unit", tk.StringVar(value="auto")).get().strip(),
            allmerge_step_min=float(self.vars["allmerge_step_min"].get()),
            npt=int(self.vars["npt"].get()),
            unit=self.vars["unit"].get().strip(),
            method=self.vars["method"].get().strip(),
            polarization_factor=float(self.vars["polarization_factor"].get()),
            dummy_value=self.vars["dummy_value"].get().strip(),
            delta_dummy=self.vars["delta_dummy"].get().strip(),
            start_frame=int(self.vars["start_frame"].get()),
            end_frame_inclusive=int(self.vars["end_frame_inclusive"].get()),
            preview_frame=int(self.vars["preview_frame"].get()),
            preview_percentile_low=float(self.vars["preview_percentile_low"].get()),
            preview_percentile_high=float(self.vars["preview_percentile_high"].get()),
            contour_levels=int(self.vars["contour_levels"].get()),
            processes=int(self.vars["processes"].get()),
            groups_per_task=int(self.vars["groups_per_task"].get()),
            sector_enable=bool(self.vars["sector_enable"].get()),
            ref_angle_deg=float(self.vars["ref_angle_deg"].get()),
            sector_step_deg=float(self.vars["sector_step_deg"].get()),
            sector_width_deg=float(self.vars["sector_width_deg"].get()),
            paired_180=bool(self.vars["paired_180"].get()),
        )

    def on_check_mask(self):
        """Inspect external/internal/final mask counts so the user can catch inverted masks."""
        try:
            # Make sure radio state is reflected in invert_external_mask.
            self._sync_external_mask_semantics()
            cfg = self._cfg()

            if not cfg.h5_path:
                messagebox.showwarning("MASK check", "H5 fileを先に選択してください。画像shapeと照合します。")
                return

            shape, dtype = load_h5_shape(cfg.h5_path, cfg.image_dataset)
            if len(shape) != 3:
                raise ValueError(f"Image dataset must be 3D: {shape}")
            _, H, W = shape
            shape_hw = (H, W)
            self._append_log("=== MASK check ===")
            self._append_log(f"Image shape: H={H}, W={W}, dtype={dtype}")

            ext_raw = read_external_mask(cfg.mask_tif_path, shape_hw) if cfg.mask_tif_path else None
            int_raw = read_internal_mask(cfg.h5_path, cfg.internal_mask_dataset, shape_hw) if cfg.use_internal_h5_mask else None

            ext_pyfai = to_pyfai_mask(ext_raw, cfg.invert_external_mask)
            int_pyfai = to_pyfai_mask(int_raw, cfg.invert_internal_h5_mask)
            final_mask = combine_masks(ext_pyfai, int_pyfai)

            total = H * W
            mode = self.vars.get("external_mask_semantics").get() if "external_mask_semantics" in self.vars else "dxb_valid"
            mode_txt = "DXB style: white=USE" if mode == "dxb_valid" else "pyFAI style: white=EXCLUDE"
            self._append_log(f"External mask convention: {mode_txt}")

            def _stats_raw(name, arr):
                if arr is None:
                    self._append_log(f"{name}: not used")
                    return
                nz = int(np.count_nonzero(arr))
                z = int(arr.size - nz)
                self._append_log(f"{name} raw: nonzero/white={nz:,} ({100*nz/arr.size:.2f}%), zero/black={z:,} ({100*z/arr.size:.2f}%)")

            def _stats_pyfai(name, arr):
                if arr is None:
                    return
                masked = int(np.count_nonzero(arr))
                used = int(arr.size - masked)
                self._append_log(f"{name} pyFAI: masked/excluded={masked:,} ({100*masked/arr.size:.2f}%), used={used:,} ({100*used/arr.size:.2f}%)")

            _stats_raw("External", ext_raw)
            _stats_pyfai("External", ext_pyfai)
            _stats_raw("Internal H5", int_raw)
            _stats_pyfai("Internal H5", int_pyfai)
            _stats_pyfai("FINAL", final_mask)

            if final_mask is not None:
                masked = int(np.count_nonzero(final_mask))
                used = int(total - masked)
                if used <= 0:
                    self._append_log("WARNING: final mask uses 0 pixels. Mask direction is probably wrong.")
                    messagebox.showwarning("MASK check", "最終MASKで使える画素が0です。外部MASKの白黒設定が逆の可能性が高いです。")
                elif masked <= 0:
                    self._append_log("NOTE: final mask excludes 0 pixels. MASKが効いていない可能性があります。")
                elif used < total * 0.05:
                    self._append_log("WARNING: used pixels < 5%. Mask may be too strict or inverted.")
                    messagebox.showwarning("MASK check", "使える画素が5%未満です。MASKが厳しすぎる、または向きが逆かもしれません。")
                else:
                    messagebox.showinfo("MASK check", f"MASK確認OK候補です。\nused={used:,} / {total:,} pixels")
            else:
                messagebox.showinfo("MASK check", "外部/内部MASKとも未使用です。全画素で積分します。")

        except Exception as e:
            messagebox.showerror("MASK check error", str(e))
            self._append_log("ERROR: " + repr(e))

    def on_check_shape(self):
        try:
            cfg = self._cfg()
            shape, dtype = load_h5_shape(cfg.h5_path, cfg.image_dataset)
            self._append_log(f"shape={shape}, dtype={dtype}")
            self._append_log(f"Current metadata: effective_frame_time_sec={cfg.effective_frame_time_sec}, time_axis_unit={cfg.time_axis_unit}, poni_space_bin={cfg.poni_space_bin}, time_bin_label={cfg.time_bin_label}")
            messagebox.showinfo("H5 shape", f"shape = {shape}\ndtype = {dtype}")
        except Exception as e:
            self._append_log(f"ERROR: {e}")
            messagebox.showerror("Error", str(e))

    def on_preview(self):
        try:
            cfg = self._cfg()
            out_png, center_xy = run_preview(cfg)
            self._append_log(f"Preview saved: {out_png}")
            self._append_log(f"Estimated center xy: {center_xy}")
            messagebox.showinfo("Preview complete", f"Saved:\n{out_png}\n\nEstimated center xy: {center_xy}")
        except Exception as e:
            self._append_log(f"ERROR: {e}")
            messagebox.showerror("Error", str(e))


    def on_rebuild_allmerge_axis(self):
        try:
            cfg = self._cfg()
            if not cfg.output_dir:
                messagebox.showwarning("Rebuild ALLMerge", "Output folderを選択してください。")
                return
            self._append_log("Rebuilding ALLMerge/Trans axis files only... (no H5/PONI integration)")
            self._append_log("Axis-only can change: Effective frame time, Trans time axis, Time bin label, ALLMerge step.")
            self._append_log("Axis-only cannot change: PONI space bin, unit, npt, mask, sector, frame range, accum frames, target time bins. Use Run 1D Integration for those.")
            all_dir, n_profiles, files = rebuild_allmerged_from_existing_txt(cfg)
            self._append_log(f"Axis-only rebuild done: {all_dir} (profiles={n_profiles}, file groups={len(files)})")
            self._append_log("No 1D integration was rerun. H5/PONI/MASK were not read for calculation.")
            messagebox.showinfo(
                "ALLMerge axis rebuild complete",
                f"軸だけ再出力しました。\n1D積分は再計算していません。\n\nSaved in:\n{all_dir}\n\nProfiles: {n_profiles}"
            )
        except Exception as e:
            self._append_log(f"ERROR: {e}")
            messagebox.showerror("Rebuild ALLMerge error", str(e))

    def on_run(self):
        try:
            cfg = self._cfg()
            self._append_log("Running integration...")
            out_dir, n = run_integration(cfg)
            self._append_log(f"Integration done: {out_dir} (files={n})")
            messagebox.showinfo("Integration complete", f"Saved in:\n{out_dir}\n\nFiles: {n}")
        except Exception as e:
            self._append_log(f"ERROR: {e}")
            messagebox.showerror("Error", str(e))


def main():
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
