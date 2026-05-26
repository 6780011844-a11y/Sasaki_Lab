#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DXB 00: Project Import + Bin Manager v5.8 Integrated Worker CoarseProgress
=======================================

目的
- 初回だけ Raw データ(TIF stack / headerless RAW binary stack / EIGER master.h5 / existing images.h5)を canonical H5 化する。
- 2回目以降は Analysis root 以下の 01_h5/images.h5 を起点に、Binだけ追加・再作成する。
- data_label は単なるフォルダ由来ラベルとして扱う。
- TIF/RAW/H5/EIGER は source type、data_label は任意名として分離する。
- UIは「初回Import」と「後からBin」を分ける。

起動
    streamlit run DXB_00_Project_ImportBin_Manager_v5_5_safety_resume.py

出力例
AnalysisRoot/
  01_AAA/
    00_dataset_manifest.json
    SAXS/
      00_channel_manifest.json
      01_h5/images.h5
      01_h5/images_manifest.json
      02_bin/8x8/t1/images_8x8_t1.h5
      02_bin/8x8/t1/index.json
    WAXS/
      00_channel_manifest.json
      01_h5/images.h5
      02_bin/5x5/t1/images_5x5_t1.h5

DXB mask convention
- H5内 /entry/instrument/detector/mask は 1=valid, 0=invalid を標準とする。
- pyFAIに渡すときは別工程で 1=invalid に変換する。
"""

from __future__ import annotations

import os
import re
import json
import time
import shutil
import sys
import subprocess
import uuid
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import h5py
import numpy as np
import pandas as pd
import streamlit as st
import tifffile as tiff

try:
    import hdf5plugin  # noqa: F401
    from hdf5plugin import Bitshuffle
    HAS_HDF5PLUGIN = True
except Exception:
    Bitshuffle = None
    HAS_HDF5PLUGIN = False


APP_TITLE = "DXB Project Import / Bin Manager v5.8 Integrated Worker"
DATASET_MANIFEST = "00_dataset_manifest.json"
CHANNEL_MANIFEST = "00_channel_manifest.json"
IMAGES_MANIFEST = "images_manifest.json"
BIN_MANIFEST = "bin_manifest.json"
CATALOG_JSON = "dxb_project_catalog.json"
CATALOG_CSV = "dxb_project_catalog.csv"

SCAN_MODES = ["Fast", "Normal", "Full verify"]
IMPORT_MODES = ["Build if missing", "Reuse canonical H5", "Rebuild canonical H5"]
RUN_MODES = ["Skip existing", "Overwrite selected", "Force rebuild selected"]
SOURCE_TYPES = ["tif_stack", "raw_binary_stack", "eiger_master_h5", "existing_images_h5"]

# Headerless RAW binary defaults.
# Rawバイナリ画像読み込み用.txt に合わせた初期値:
#   width=487, height=407, dtype=<i4, offset=0
RAW_BINARY_DEFAULT_WIDTH = 487
RAW_BINARY_DEFAULT_HEIGHT = 407
RAW_BINARY_DEFAULT_DTYPE = "<i4"
RAW_BINARY_DEFAULT_OFFSET_BYTES = 0
RAW_BINARY_DTYPE_OPTIONS = ["<i4", "<u2", "<i2", "<u4", "<f4", ">i4", ">u2", ">i2", ">u4", ">f4"]
COMPRESSION_MODES = ["bitshuffle_lz4", "none", "gzip"]
PREVIEW_MODES = ["none", "first_only", "first_last", "first_last_mask"]
SPEED_PRESETS = {
    "Safe": {
        "parallel_jobs": 1,
        "tif_workers": 4,
        "tif_batch": 32,
        "eiger_batch": 512,
        "h5_chunk_mib": 64.0,
        "bin_chunk_factor": 256,
        "preview_mode": "first_only",
        "compression_mode": "bitshuffle_lz4",
    },
    "Balanced": {
        "parallel_jobs": 2,
        "tif_workers": 8,
        "tif_batch": 64,
        "eiger_batch": 512,
        "h5_chunk_mib": 64.0,
        "bin_chunk_factor": 512,
        "preview_mode": "first_only",
        "compression_mode": "bitshuffle_lz4",
    },
    "Fast NVMe": {
        "parallel_jobs": 3,
        "tif_workers": 12,
        "tif_batch": 128,
        "eiger_batch": 1024,
        "h5_chunk_mib": 128.0,
        "bin_chunk_factor": 1024,
        "preview_mode": "none",
        "compression_mode": "bitshuffle_lz4",
    },
    "Conservative Laptop": {
        "parallel_jobs": 1,
        "tif_workers": 2,
        "tif_batch": 16,
        "eiger_batch": 256,
        "h5_chunk_mib": 32.0,
        "bin_chunk_factor": 128,
        "preview_mode": "first_only",
        "compression_mode": "bitshuffle_lz4",
    },
    "Auto": {},
    "Custom": {},
}


# =============================================================================
# Small utilities
# =============================================================================

def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def clean_path_text(s: str) -> str:
    return str(s or "").strip().strip('"\'')


def safe_name(s: str, default: str = "unknown") -> str:
    s = str(s or default).strip()
    s = re.sub(r"[<>:\"/\\|?*\x00-\x1f]", "_", s)
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"_+", "_", s)
    return s.strip("._ ") or default


def natural_key(path_or_name) -> Tuple[Any, str]:
    name = Path(path_or_name).name
    nums = re.findall(r"\d+", name)
    if nums:
        return tuple(int(x) for x in nums), name
    return (10**12,), name


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_json(path: Path, default=None):
    if default is None:
        default = {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def save_json(path: Path, obj: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


# =============================================================================
# Safety / resume helpers
# =============================================================================
DONE_STATUSES = {"done", "done_fused", "done_alias"}


def incomplete_dir_for(final_h5: Path) -> Path:
    return Path(final_h5).parent / "_incomplete"


def tmp_h5_path(final_h5: Path) -> Path:
    final_h5 = Path(final_h5)
    inc = incomplete_dir_for(final_h5)
    inc.mkdir(parents=True, exist_ok=True)
    return inc / (final_h5.name + ".part")


def running_json_path(final_h5: Path) -> Path:
    final_h5 = Path(final_h5)
    inc = incomplete_dir_for(final_h5)
    inc.mkdir(parents=True, exist_ok=True)
    return inc / (final_h5.name + ".running.json")


def failed_json_path(final_h5: Path) -> Path:
    final_h5 = Path(final_h5)
    inc = incomplete_dir_for(final_h5)
    inc.mkdir(parents=True, exist_ok=True)
    return inc / (final_h5.name + ".failed.json")


def _safe_unlink(p: Path):
    try:
        p = Path(p)
        if p.exists() or p.is_symlink():
            p.unlink()
    except Exception:
        pass


def start_running_marker(final_h5: Path, info: Dict[str, Any]) -> Path:
    marker = running_json_path(final_h5)
    payload = dict(info or {})
    payload.update({
        "status": "running",
        "final_h5": str(final_h5),
        "tmp_h5": str(tmp_h5_path(final_h5)),
        "started_at": now_iso(),
        "updated_at": now_iso(),
        "current": 0,
        "total": int(payload.get("total", 0) or 0),
    })
    save_json(marker, payload)
    _safe_unlink(failed_json_path(final_h5))
    return marker


def update_running_marker(final_h5: Path, current: int = 0, total: int = 0, stage: str = "running", message: str = ""):
    marker = running_json_path(final_h5)
    payload = load_json(marker, default={})
    payload.update({
        "status": "running",
        "stage": stage,
        "message": message,
        "current": int(current),
        "total": int(total),
        "updated_at": now_iso(),
    })
    save_json(marker, payload)


def finish_running_marker(final_h5: Path):
    _safe_unlink(running_json_path(final_h5))
    _safe_unlink(failed_json_path(final_h5))


def fail_running_marker(final_h5: Path, error: BaseException):
    marker = running_json_path(final_h5)
    payload = load_json(marker, default={})
    payload.update({
        "status": "failed",
        "failed_at": now_iso(),
        "updated_at": now_iso(),
        "error": repr(error),
        "error_type": type(error).__name__,
    })
    save_json(failed_json_path(final_h5), payload)
    _safe_unlink(marker)


def commit_tmp_h5(tmp_h5: Path, final_h5: Path):
    tmp_h5 = Path(tmp_h5)
    final_h5 = Path(final_h5)
    if not tmp_h5.exists():
        raise FileNotFoundError(f"temporary H5 not found: {tmp_h5}")
    if final_h5.exists():
        final_h5.unlink()
    os.replace(str(tmp_h5), str(final_h5))


def validate_images_h5(path: Path, expected_shape: Optional[List[int]] = None) -> Tuple[bool, str]:
    path = Path(path)
    if not path.exists():
        return False, "missing h5"
    try:
        with h5py.File(path, "r") as hf:
            if "entry/data/images" not in hf:
                return False, "missing /entry/data/images"
            shape = list(map(int, hf["entry/data/images"].shape))
            if expected_shape is not None and list(map(int, expected_shape)) != shape:
                return False, f"shape mismatch: got {shape}, expected {expected_shape}"
        return True, "ok"
    except Exception as e:
        return False, f"cannot open h5: {type(e).__name__}: {e}"


def validate_done_index(index_path: Path, output_h5: Optional[Path] = None) -> Tuple[bool, str, Dict[str, Any]]:
    info = load_json(index_path, default={})
    if not info:
        return False, "missing index.json", {}
    status = str(info.get("status", ""))
    if status not in DONE_STATUSES and not status.startswith("skipped"):
        return False, f"index status is not done: {status}", info
    if output_h5 is None:
        output_h5 = Path(info.get("output_h5", "")) if info.get("output_h5") else None
    if output_h5 is not None and str(output_h5):
        expected = info.get("shape_binned") or None
        ok, msg = validate_images_h5(Path(output_h5), expected_shape=expected)
        if not ok and status not in ["skipped_bin1_t1", "skipped_existing"]:
            return False, msg, info
    return True, "ok", info


def is_complete_bin_output(time_dir: Path, out_h5: Path) -> Tuple[bool, str, Dict[str, Any]]:
    return validate_done_index(Path(time_dir) / "index.json", Path(out_h5))


def clean_incomplete_for_output(final_h5: Path, remove_final: bool = False):
    final_h5 = Path(final_h5)
    inc = incomplete_dir_for(final_h5)
    _safe_unlink(tmp_h5_path(final_h5))
    _safe_unlink(running_json_path(final_h5))
    _safe_unlink(failed_json_path(final_h5))
    # Backward compatibility: remove old same-folder tmp/marker names too.
    _safe_unlink(final_h5.with_name(final_h5.name + ".tmp"))
    _safe_unlink(final_h5.with_name(final_h5.name + ".running.json"))
    _safe_unlink(final_h5.with_name(final_h5.name + ".failed.json"))
    try:
        if inc.exists() and not any(inc.iterdir()):
            inc.rmdir()
    except Exception:
        pass
    if remove_final:
        _safe_unlink(final_h5)


def scan_incomplete_outputs(analysis_root: Path) -> List[Dict[str, Any]]:
    root = Path(analysis_root)
    if not root.exists():
        return []
    rows: List[Dict[str, Any]] = []
    seen = set()
    artifacts = (
        list(root.rglob("*.h5.tmp")) +
        list(root.rglob("*.h5.part")) +
        list(root.rglob("*.running.json")) +
        list(root.rglob("*.failed.json"))
    )
    for p in artifacts:
        final_h5 = p
        kind = "unknown"
        # v5.8 writes temporary files under 01_h5/_incomplete/.
        in_inc = p.parent.name == "_incomplete"
        base_dir = p.parent.parent if in_inc else p.parent
        if p.name.endswith(".h5.part"):
            final_h5 = base_dir / p.name[:-5]  # remove .part
            kind = "part"
        elif p.name.endswith(".h5.tmp"):
            final_h5 = base_dir / p.name[:-4]  # remove .tmp
            kind = "tmp"
        elif p.name.endswith(".running.json"):
            final_h5 = base_dir / p.name[:-13]  # remove .running.json
            kind = "running_marker"
        elif p.name.endswith(".failed.json"):
            final_h5 = base_dir / p.name[:-12]  # remove .failed.json
            kind = "failed_marker"
        key = str(final_h5)
        if key in seen:
            continue
        seen.add(key)
        rows.append({
            "kind": kind,
            "final_h5": str(final_h5),
            "artifact": str(p),
            "mtime": datetime.fromtimestamp(p.stat().st_mtime).isoformat(timespec="seconds") if p.exists() else "",
        })
    # Find bin H5 with no valid index.json.
    for h5 in root.rglob("02_bin/*x*/t*/images_*x*_t*.h5"):
        ok, msg, _ = is_complete_bin_output(h5.parent, h5)
        if not ok:
            key = str(h5)
            if key not in seen:
                seen.add(key)
                rows.append({"kind": "h5_without_valid_index", "final_h5": str(h5), "artifact": str(h5.parent / "index.json"), "mtime": datetime.fromtimestamp(h5.stat().st_mtime).isoformat(timespec="seconds"), "reason": msg})
    return rows


def parse_int_list(text: str) -> List[int]:
    vals: List[int] = []
    for token in re.split(r"[,\s]+", str(text or "")):
        token = token.strip()
        if not token:
            continue
        vals.append(int(token))
    return sorted(set(vals))


def get_filters(compression_mode: str = "bitshuffle_lz4") -> Dict[str, Any]:
    """Return h5py create_dataset filter kwargs.

    compression_mode:
      - bitshuffle_lz4: fast and compact when hdf5plugin is available; gzip fallback otherwise
      - none: fastest write, largest files
      - gzip: portable but slower
    """
    mode = str(compression_mode or "bitshuffle_lz4").lower().strip()
    if mode == "none":
        return {}
    if mode == "gzip":
        return {"compression": "gzip", "compression_opts": 4}
    if HAS_HDF5PLUGIN:
        return Bitshuffle(cname="lz4")
    return {"compression": "gzip", "compression_opts": 4}


def compression_label(compression_mode: str) -> str:
    mode = str(compression_mode or "bitshuffle_lz4").lower().strip()
    if mode == "none":
        return "none"
    if mode == "gzip" or not HAS_HDF5PLUGIN:
        return "gzip"
    return "bitshuffle(lz4)"


def normalize_mask_array(raw_mask: np.ndarray, semantics: str) -> np.ndarray:
    """Return DXB canonical mask: 1=valid, 0=invalid."""
    if semantics == "one_is_valid":
        return (raw_mask > 0).astype(np.uint8)
    if semantics == "zero_is_valid":
        return (raw_mask == 0).astype(np.uint8)
    raise ValueError("mask semantics must be one_is_valid or zero_is_valid")


def sorted_tif_paths(folder: Path) -> List[Path]:
    paths = list(folder.glob("*.tif")) + list(folder.glob("*.tiff"))
    # maskは画像stackから除外
    paths = [p for p in paths if "mask" not in p.name.lower()]
    return sorted(paths, key=natural_key)


def bounded_walk(root: Path, max_depth: int = 5):
    root = root.resolve()
    for current, dirs, files in os.walk(root):
        cur = Path(current)
        try:
            depth = len(cur.relative_to(root).parts)
        except Exception:
            depth = 0
        if depth >= max_depth:
            dirs[:] = []
        yield cur, dirs, files


def channel_root_for(analysis_root: Path, dataset_id: str, channel: str) -> Path:
    ds_root = analysis_root / safe_name(dataset_id, "dataset")
    ch = safe_name(channel, "default")
    if ch.lower() in ["", "default", "none"]:
        return ds_root
    return ds_root / ch


def canonical_h5_path(analysis_root: Path, dataset_id: str, channel: str) -> Path:
    return channel_root_for(analysis_root, dataset_id, channel) / "01_h5" / "images.h5"


def infer_dataset_channel_from_canonical(images_h5: Path, analysis_root: Path) -> Tuple[str, str, Path]:
    channel_root = images_h5.parent.parent
    try:
        parts = channel_root.resolve().relative_to(analysis_root.resolve()).parts
    except Exception:
        parts = channel_root.parts
    if len(parts) >= 2:
        return parts[0], "/".join(parts[1:]), channel_root
    if len(parts) == 1:
        return parts[0], "default", channel_root
    return channel_root.name, "default", channel_root


def infer_channel(dataset_dir: Path, primary_path: Path) -> str:
    """Infer channel label from path. SAXS/WAXS are labels, not source types."""
    try:
        rel_parts = primary_path.resolve().relative_to(dataset_dir.resolve()).parts
    except Exception:
        rel_parts = primary_path.parts
    # Priority patterns
    joined = "/".join(rel_parts).upper()
    for key in ["GISAXS", "GIWAXS", "SAXS", "WAXS"]:
        if key in joined:
            return key
    # Common detector/location labels; only use if meaningful
    for part in rel_parts:
        up = part.upper()
        if up in ["PL", "PL_1M", "EIGER", "PILATUS"]:
            return safe_name(part)
    return "default"


def calc_chunk_frames(h: int, w: int, bpp: int, target_mib: float) -> int:
    frame_bytes = max(1, h * w * bpp)
    target = int(float(target_mib) * 1024 * 1024)
    return max(1, target // frame_bytes)


# =============================================================================
# Source detection
# =============================================================================

@dataclass
class SourceCandidate:
    dataset_id: str
    channel: str
    source_type: str
    score: int
    source_root: str
    primary_path: str
    selected_sources: List[str]
    mask_path: str = ""
    frames: Optional[int] = None
    height: Optional[int] = None
    width: Optional[int] = None
    dtype: str = ""
    notes: str = ""


def looks_like_images_h5(path: Path) -> bool:
    try:
        with h5py.File(path, "r") as hf:
            return "entry/data/images" in hf
    except Exception:
        return False


def inspect_images_h5(path: Path) -> Dict[str, Any]:
    info: Dict[str, Any] = {"frames": None, "height": None, "width": None, "dtype": "", "has_mask": False, "mask_valid": None, "mask_total": None, "ok": False, "error": ""}
    try:
        with h5py.File(path, "r") as hf:
            if "entry/data/images" not in hf:
                raise KeyError("/entry/data/images not found")
            d = hf["entry/data/images"]
            T, H, W = map(int, d.shape)
            info.update({"frames": T, "height": H, "width": W, "dtype": str(d.dtype), "ok": True})
            try:
                info.update(get_h5_time_metadata(hf, fallback_texts=[str(path)]))
            except Exception:
                info.update({"raw_frame_time_sec": None, "time_bin": None, "effective_frame_time_sec": None, "frame_time_source": "inspect_failed"})
            if "entry/instrument/detector/mask" in hf:
                m = hf["entry/instrument/detector/mask"][:]
                info.update({"has_mask": True, "mask_valid": int(np.count_nonzero(m > 0)), "mask_total": int(m.size)})
    except Exception as e:
        info["error"] = f"{type(e).__name__}: {e}"
    return info


def find_eiger_blocks(fin: h5py.File):
    g = fin.get("/entry/data", None)
    if g is None:
        return []
    blocks = []
    for k in sorted(list(g.keys()), key=natural_key):
        try:
            ds = g[k]
            if isinstance(ds, h5py.Dataset) and ds.ndim == 3:
                blocks.append((k, ds))
        except Exception:
            continue
    return blocks


def inspect_eiger_master(path: Path) -> Dict[str, Any]:
    info: Dict[str, Any] = {"frames": None, "height": None, "width": None, "dtype": "", "has_mask": False, "ok": False, "error": ""}
    cwd0 = os.getcwd()
    try:
        os.chdir(str(path.parent))
        with h5py.File(path, "r") as fin:
            blocks = find_eiger_blocks(fin)
            if not blocks:
                raise RuntimeError("no 3D /entry/data/data_* blocks")
            frames = int(sum(int(ds.shape[0]) for _, ds in blocks))
            info.update({
                "frames": frames,
                "height": int(blocks[0][1].shape[1]),
                "width": int(blocks[0][1].shape[2]),
                "dtype": str(blocks[0][1].dtype),
                "has_mask": bool("/entry/instrument/detector/detectorSpecific/pixel_mask" in fin),
                "ok": True,
            })
    except Exception as e:
        info["error"] = f"{type(e).__name__}: {e}"
    finally:
        try:
            os.chdir(cwd0)
        except Exception:
            pass
    return info


def candidate_mask_near(dataset_dir: Path, primary_dir: Path, max_depth: int = 4) -> str:
    candidates: List[Path] = []
    # prefer nearby masks
    search_roots = []
    for p in [primary_dir, primary_dir.parent, dataset_dir]:
        if p.exists() and p.is_dir() and p not in search_roots:
            search_roots.append(p)
    for root in search_roots:
        for cur, _, files in bounded_walk(root, max_depth=max_depth):
            for f in files:
                lf = f.lower()
                if ("mask" in lf or "blemish" in lf) and lf.endswith((".tif", ".tiff")):
                    candidates.append(cur / f)
    if not candidates:
        return ""
    candidates = sorted(set(candidates), key=lambda p: (0 if primary_dir in p.parents or p.parent == primary_dir else 1, len(str(p)), str(p)))
    return str(candidates[0])


def _raw_binary_dtype(raw_dtype: str) -> np.dtype:
    """Return numpy dtype for headerless raw binary images."""
    txt = str(raw_dtype or RAW_BINARY_DEFAULT_DTYPE).strip()
    try:
        return np.dtype(txt)
    except Exception as e:
        raise ValueError(f"Invalid RAW dtype: {raw_dtype!r}") from e


def raw_binary_expected_bytes(width: int, height: int, raw_dtype: str, offset_bytes: int = 0) -> int:
    dt = _raw_binary_dtype(raw_dtype)
    return int(offset_bytes) + int(width) * int(height) * int(dt.itemsize)


def is_raw_binary_candidate_file(path: Path, width: int, height: int, raw_dtype: str, offset_bytes: int = 0, allow_raw_tif_like: bool = False) -> bool:
    """Detect a headerless RAW image file.

    Primary target is extensionless files. For safety, .tif/.tiff files are only
    treated as RAW when tifffile cannot read them and the byte size exactly
    matches width*height*dtype + offset.
    """
    p = Path(path)
    if not p.is_file() or p.name.startswith("."):
        return False
    name_low = p.name.lower()
    if any(key in name_low for key in ["mask", "blemish", "dark", "flat"]):
        return False
    try:
        expected = raw_binary_expected_bytes(width, height, raw_dtype, offset_bytes)
        if int(p.stat().st_size) != expected:
            return False
    except Exception:
        return False

    # Main case: no extension.
    if p.suffix == "":
        return True

    # Optional rescue: files originally named .tif but actually headerless RAW.
    if allow_raw_tif_like and p.suffix.lower() in [".tif", ".tiff"]:
        try:
            _ = tiff.imread(str(p))
            return False
        except Exception:
            return True
    return False


def sorted_raw_binary_paths(folder: Path, width: int, height: int, raw_dtype: str, offset_bytes: int = 0) -> List[Path]:
    p = Path(folder)
    if not p.exists() or not p.is_dir():
        return []
    paths = [
        q for q in p.iterdir()
        if is_raw_binary_candidate_file(q, width, height, raw_dtype, offset_bytes)
    ]
    return sorted(paths, key=natural_key)


def _read_raw_binary_one_for_import(args):
    """Read one headerless RAW binary frame."""
    path_str, H, W, raw_dtype, offset_bytes = args
    p = Path(path_str)
    dt = _raw_binary_dtype(raw_dtype)
    expected_count = int(H) * int(W)
    with open(p, "rb") as fh:
        if int(offset_bytes) > 0:
            fh.seek(int(offset_bytes))
        arr = np.fromfile(fh, dtype=dt, count=expected_count)
    if arr.size != expected_count:
        raise ValueError(f"raw size mismatch: {p.name} got={arr.size}, expected={expected_count}")
    arr = arr.reshape(int(H), int(W))
    try:
        ts = float(p.stat().st_mtime)
    except Exception:
        ts = time.time()
    return arr, ts, p.name


def read_raw_binary_batch(paths: List[Path], H: int, W: int, raw_dtype: str, offset_bytes: int = 0, workers: int = 1):
    workers = max(1, int(workers))
    args = [(str(p), H, W, raw_dtype, int(offset_bytes)) for p in paths]
    if workers <= 1 or len(args) <= 1:
        res = [_read_raw_binary_one_for_import(a) for a in args]
    else:
        with ThreadPoolExecutor(max_workers=min(workers, len(args))) as ex:
            res = list(ex.map(_read_raw_binary_one_for_import, args))
    arrs = [r[0] for r in res]
    ts = [r[1] for r in res]
    names = [r[2] for r in res]
    return arrs, ts, names


def detect_raw_binary_stacks(dataset_dir: Path, scan_mode: str, max_depth: int = 5,
                             width: int = RAW_BINARY_DEFAULT_WIDTH,
                             height: int = RAW_BINARY_DEFAULT_HEIGHT,
                             raw_dtype: str = RAW_BINARY_DEFAULT_DTYPE,
                             offset_bytes: int = RAW_BINARY_DEFAULT_OFFSET_BYTES,
                             min_count: int = 1) -> List[SourceCandidate]:
    """Detect headerless RAW binary frame stacks by exact byte size."""
    out: List[SourceCandidate] = []
    for cur, _, files in bounded_walk(dataset_dir, max_depth=max_depth):
        # Avoid scanning very large sibling trees more than needed; candidate check is stat-only.
        paths = [cur / f for f in files]
        raw_paths = [p for p in paths if is_raw_binary_candidate_file(p, width, height, raw_dtype, offset_bytes)]
        if len(raw_paths) < int(min_count):
            continue
        raw_paths = sorted(raw_paths, key=natural_key)
        notes = (
            f"headerless RAW binary; width={int(width)}, height={int(height)}, "
            f"dtype={str(raw_dtype)}, offset={int(offset_bytes)} bytes; "
            f"expected_file_size={raw_binary_expected_bytes(width, height, raw_dtype, offset_bytes)} bytes"
        )
        if scan_mode == "Full verify":
            # Verify a few frames can be reshaped.
            try:
                for idx in sorted(set([0, len(raw_paths)//2, len(raw_paths)-1])):
                    arr, _, _ = _read_raw_binary_one_for_import((str(raw_paths[idx]), int(height), int(width), str(raw_dtype), int(offset_bytes)))
                    if arr.shape != (int(height), int(width)):
                        notes += f" shape mismatch at {raw_paths[idx].name};"
            except Exception as e:
                notes += f" verify failed: {e!r}"
        ch = infer_channel(dataset_dir, cur)
        mask_path = candidate_mask_near(dataset_dir, cur, max_depth=3)
        score = 88 + min(8, len(raw_paths) // 100)
        out.append(SourceCandidate(
            dataset_id=safe_name(dataset_dir.name, "dataset"),
            channel=ch,
            source_type="raw_binary_stack",
            score=score,
            source_root=str(dataset_dir),
            primary_path=str(cur),
            selected_sources=[str(cur)],
            mask_path=mask_path,
            frames=len(raw_paths),
            height=int(height),
            width=int(width),
            dtype=str(raw_dtype),
            notes=notes,
        ))
    return sorted(out, key=lambda c: (-c.score, c.channel, c.primary_path))[:12]


def detect_tif_stacks(dataset_dir: Path, scan_mode: str, max_depth: int = 5, min_count: int = 2) -> List[SourceCandidate]:
    out: List[SourceCandidate] = []
    for cur, _, files in bounded_walk(dataset_dir, max_depth=max_depth):
        tif_files = [f for f in files if f.lower().endswith((".tif", ".tiff"))]
        image_like = [f for f in tif_files if "mask" not in f.lower()]
        if len(image_like) < min_count:
            continue
        frames = len(image_like)
        H = W = None
        dtype = ""
        notes = ""
        if scan_mode in ["Normal", "Full verify"]:
            try:
                paths = sorted([cur / f for f in image_like], key=natural_key)
                arr = tiff.imread(str(paths[0]))
                if arr.ndim != 2:
                    notes = f"first tif ndim={arr.ndim}; skipped"
                    continue
                H, W = map(int, arr.shape)
                dtype = str(arr.dtype)
                if scan_mode == "Full verify":
                    for idx in sorted(set([0, len(paths)//2, len(paths)-1])):
                        a = tiff.imread(str(paths[idx]))
                        if a.shape != arr.shape:
                            notes += f" shape mismatch at {paths[idx].name};"
            except Exception as e:
                notes = f"inspect failed: {e!r}"
        ch = infer_channel(dataset_dir, cur)
        mask_path = candidate_mask_near(dataset_dir, cur, max_depth=3)
        score = 70 + min(20, frames // 100)
        out.append(SourceCandidate(
            dataset_id=safe_name(dataset_dir.name, "dataset"),
            channel=ch,
            source_type="tif_stack",
            score=score,
            source_root=str(dataset_dir),
            primary_path=str(cur),
            selected_sources=[str(cur)],
            mask_path=mask_path,
            frames=frames,
            height=H,
            width=W,
            dtype=dtype,
            notes=notes,
        ))
    return sorted(out, key=lambda c: (-c.score, c.channel, c.primary_path))[:8]


def detect_h5_candidates(dataset_dir: Path, scan_mode: str, max_depth: int = 5) -> List[SourceCandidate]:
    out: List[SourceCandidate] = []
    h5_paths: List[Path] = []
    for cur, _, files in bounded_walk(dataset_dir, max_depth=max_depth):
        for f in files:
            if f.lower().endswith((".h5", ".hdf5")):
                h5_paths.append(cur / f)
    if not h5_paths:
        return out

    # EIGER masters: group by parent folder, not whole dataset, so SAXS/WAXS folders remain separate.
    master_paths = [p for p in h5_paths if p.name.lower().endswith("_master.h5") or "master" in p.name.lower()]
    by_parent: Dict[Path, List[Path]] = {}
    for p in master_paths:
        by_parent.setdefault(p.parent, []).append(p)
    for parent, masters in sorted(by_parent.items(), key=lambda kv: str(kv[0])):
        masters = sorted(masters, key=natural_key)
        info: Dict[str, Any] = {}
        frames = None
        if scan_mode in ["Normal", "Full verify"]:
            total = 0
            ok = 0
            for mp in masters:
                ii = inspect_eiger_master(mp)
                if ii.get("frames") is not None:
                    total += int(ii["frames"])
                    ok += 1
                if not info and ii.get("ok"):
                    info = ii
            frames = total if ok else None
        ch = infer_channel(dataset_dir, parent)
        out.append(SourceCandidate(
            dataset_id=safe_name(dataset_dir.name, "dataset"),
            channel=ch,
            source_type="eiger_master_h5",
            score=95,
            source_root=str(dataset_dir),
            primary_path=str(masters[0]),
            selected_sources=[str(p) for p in masters],
            frames=frames,
            height=info.get("height"),
            width=info.get("width"),
            dtype=str(info.get("dtype") or ""),
            notes=f"masters={len(masters)}; mask={'yes' if info.get('has_mask') else 'unknown/none'}",
        ))

    # existing images.h5-like files
    for p in sorted(h5_paths, key=lambda x: (0 if x.name.lower() == "images.h5" else 1, len(str(x)), str(x))):
        if p in master_paths:
            continue
        is_images = False
        info: Dict[str, Any] = {}
        if p.name.lower().startswith("images"):
            is_images = True if scan_mode == "Fast" else looks_like_images_h5(p)
        elif scan_mode in ["Normal", "Full verify"]:
            is_images = looks_like_images_h5(p)
        if not is_images:
            continue
        if scan_mode in ["Normal", "Full verify"]:
            info = inspect_images_h5(p)
        ch = infer_channel(dataset_dir, p.parent)
        out.append(SourceCandidate(
            dataset_id=safe_name(dataset_dir.name, "dataset"),
            channel=ch,
            source_type="existing_images_h5",
            score=90 if p.name.lower() == "images.h5" else 82,
            source_root=str(dataset_dir),
            primary_path=str(p),
            selected_sources=[str(p)],
            frames=info.get("frames"),
            height=info.get("height"),
            width=info.get("width"),
            dtype=str(info.get("dtype") or ""),
            notes=f"mask={'yes' if info.get('has_mask') else 'unknown/none'}",
        ))
    return sorted(out, key=lambda c: (-c.score, c.channel, c.primary_path))[:12]


def scan_source_root(source_root: Path, scan_mode: str = "Normal", max_depth: int = 5,
                     raw_binary_width: int = RAW_BINARY_DEFAULT_WIDTH,
                     raw_binary_height: int = RAW_BINARY_DEFAULT_HEIGHT,
                     raw_binary_dtype: str = RAW_BINARY_DEFAULT_DTYPE,
                     raw_binary_offset_bytes: int = RAW_BINARY_DEFAULT_OFFSET_BYTES,
                     include_raw_binary: bool = False) -> pd.DataFrame:
    if not source_root.exists():
        raise FileNotFoundError(source_root)
    dataset_dirs = [p for p in sorted(source_root.iterdir(), key=natural_key) if p.is_dir()]
    # Include source root itself if data are directly inside it.
    direct = (
        detect_h5_candidates(source_root, scan_mode=scan_mode, max_depth=0)
        + detect_tif_stacks(source_root, scan_mode=scan_mode, max_depth=0)
    )
    # RAW binary is intentionally excluded from recursive auto-scan by default.
    # Headerless files have no reliable signature, so recursive size matching over
    # a large source tree is expensive and can make Streamlit look frozen.
    if include_raw_binary:
        direct += detect_raw_binary_stacks(
            source_root, scan_mode=scan_mode, max_depth=0,
            width=int(raw_binary_width), height=int(raw_binary_height),
            raw_dtype=str(raw_binary_dtype), offset_bytes=int(raw_binary_offset_bytes),
        )
    if direct:
        dataset_dirs = [source_root] + dataset_dirs
    rows: List[Dict[str, Any]] = []
    for ds_dir in dataset_dirs:
        candidates = (
            detect_h5_candidates(ds_dir, scan_mode=scan_mode, max_depth=max_depth)
            + detect_tif_stacks(ds_dir, scan_mode=scan_mode, max_depth=max_depth)
        )
        if include_raw_binary:
            candidates += detect_raw_binary_stacks(
                ds_dir, scan_mode=scan_mode, max_depth=max_depth,
                width=int(raw_binary_width), height=int(raw_binary_height),
                raw_dtype=str(raw_binary_dtype), offset_bytes=int(raw_binary_offset_bytes),
            )
        candidates = sorted(candidates, key=lambda c: (-c.score, c.channel, c.source_type, c.primary_path))
        if not candidates:
            continue
        # default run: rank1 per dataset+channel only
        seen = set()
        for ci, c in enumerate(candidates):
            key = (c.dataset_id, c.channel)
            default_run = key not in seen
            seen.add(key)
            rows.append({
                "run": default_run,
                "dataset_id": c.dataset_id if ds_dir != source_root else safe_name(source_root.name, "dataset"),
                "channel": c.channel,
                "candidate_rank": ci + 1,
                "score": c.score,
                "source_type": c.source_type,
                "primary_path": c.primary_path,
                "selected_sources": "\n".join(c.selected_sources),
                "mask_path": c.mask_path,
                "frames": c.frames,
                "height": c.height,
                "width": c.width,
                "dtype": c.dtype,
                "notes": c.notes,
                "raw_frame_time_sec": infer_frame_time_sec_from_text(c.primary_path, c.source_root, c.notes),
                "source_root": c.source_root,
            })
    return pd.DataFrame(rows)


# =============================================================================
# Manifest / catalog
# =============================================================================

def channel_manifest_path(analysis_root: Path, dataset_id: str, channel: str) -> Path:
    return channel_root_for(analysis_root, dataset_id, channel) / CHANNEL_MANIFEST


def dataset_manifest_path(analysis_root: Path, dataset_id: str) -> Path:
    return analysis_root / safe_name(dataset_id, "dataset") / DATASET_MANIFEST


def update_dataset_manifest_index(analysis_root: Path, dataset_id: str, channel: str):
    ds_root = analysis_root / safe_name(dataset_id, "dataset")
    ensure_dir(ds_root)
    mp = ds_root / DATASET_MANIFEST
    m = load_json(mp, default={})
    m.setdefault("schema_version", 3)
    m.setdefault("dataset_id", safe_name(dataset_id, "dataset"))
    m.setdefault("analysis_dir", str(ds_root))
    channels = m.get("channels", [])
    if channel not in channels:
        channels.append(channel)
    m["channels"] = sorted(channels)
    m["updated_at"] = now_iso()
    m.setdefault("created_at", now_iso())
    save_json(mp, m)


def make_or_update_channel_manifest(analysis_root: Path, dataset_id: str, channel: str, source_type: str, source_root: str, selected_sources: List[str], mask_path: str = "") -> Dict[str, Any]:
    update_dataset_manifest_index(analysis_root, dataset_id, channel)
    ch_root = channel_root_for(analysis_root, dataset_id, channel)
    ensure_dir(ch_root)
    mp = ch_root / CHANNEL_MANIFEST
    m = load_json(mp, default={})
    m.update({
        "schema_version": 3,
        "dataset_id": safe_name(dataset_id, "dataset"),
        "channel": safe_name(channel, "default"),
        "channel_root": str(ch_root),
        "source_type": source_type,
        "source_root": source_root,
        "selected_sources": selected_sources,
        "mask_path": mask_path,
        "canonical_images_h5": str(canonical_h5_path(analysis_root, dataset_id, channel)),
        "updated_at": now_iso(),
    })
    m.setdefault("created_at", now_iso())
    m.setdefault("import_status", "pending")
    m.setdefault("bin_recipes", [])
    save_json(mp, m)
    return m


def update_channel_import_status(analysis_root: Path, dataset_id: str, channel: str, status: str, extra: Optional[Dict[str, Any]] = None):
    mp = channel_manifest_path(analysis_root, dataset_id, channel)
    m = load_json(mp, default={})
    m["import_status"] = status
    m["updated_at"] = now_iso()
    if extra:
        m.update(extra)
    save_json(mp, m)


def append_bin_manifest(channel_root: Path, info: Dict[str, Any]):
    manifest_path = channel_root / "02_bin" / BIN_MANIFEST
    manifest = load_json(manifest_path, default={})
    manifest.setdefault("schema_version", 3)
    manifest.setdefault("created_at", now_iso())
    manifest["updated_at"] = now_iso()
    manifest["channel_root"] = str(channel_root)
    recipes = manifest.get("bin_recipes", [])
    key = (int(info.get("space_bin", -1)), int(info.get("time_bin", -1)), str(info.get("mask_bin_mode", "")))
    replaced = False
    for i, r in enumerate(recipes):
        k2 = (int(r.get("space_bin", -1)), int(r.get("time_bin", -1)), str(r.get("mask_bin_mode", "")))
        if k2 == key:
            recipes[i] = info
            replaced = True
            break
    if not replaced:
        recipes.append(info)
    manifest["bin_recipes"] = recipes
    save_json(manifest_path, manifest)

    # mirror into channel manifest
    cm_path = channel_root / CHANNEL_MANIFEST
    cm = load_json(cm_path, default={})
    cm_recipes = cm.get("bin_recipes", [])
    replaced = False
    for i, r in enumerate(cm_recipes):
        k2 = (int(r.get("space_bin", -1)), int(r.get("time_bin", -1)), str(r.get("mask_bin_mode", "")))
        if k2 == key:
            cm_recipes[i] = info
            replaced = True
            break
    if not replaced:
        cm_recipes.append(info)
    cm["bin_recipes"] = cm_recipes
    cm["updated_at"] = now_iso()
    save_json(cm_path, cm)


# =============================================================================
# Import to canonical H5
# =============================================================================

def decide_target_dtype(sample: np.ndarray, mode: str):
    orig_dtype = sample.dtype
    orig_min = float(np.min(sample))
    orig_max = float(np.max(sample))
    target_dtype = orig_dtype
    clip_min = clip_max = None
    if mode == "force_uint8":
        target_dtype = np.uint8
        clip_min, clip_max = 0, 255
    elif mode == "force_uint16":
        target_dtype = np.uint16
        clip_min, clip_max = 0, 65535
    elif mode == "auto_uint":
        if orig_min >= 0 and orig_max <= 255:
            target_dtype = np.uint8
            clip_min, clip_max = 0, 255
        elif orig_min >= 0 and orig_max <= 65535:
            target_dtype = np.uint16
            clip_min, clip_max = 0, 65535
    elif mode == "keep":
        pass
    else:
        raise ValueError(f"Unknown pixel dtype mode: {mode}")
    return np.dtype(target_dtype), clip_min, clip_max, orig_min, orig_max


def cast_to_target(arr: np.ndarray, target_dtype, clip_min, clip_max):
    if clip_min is not None:
        arr = np.clip(arr, clip_min, clip_max)
    if arr.dtype != target_dtype:
        arr = arr.astype(target_dtype, copy=False)
    return arr



def _read_tif_one_for_import(args):
    """Read one TIF for import. Kept top-level for safe threaded/multiprocess-like execution."""
    path_str, H, W = args
    p = Path(path_str)
    arr = tiff.imread(str(p))
    if arr.shape != (H, W):
        raise ValueError(f"shape mismatch: {p.name} got={arr.shape}, expected={(H, W)}")
    try:
        ts = float(p.stat().st_mtime)
    except Exception:
        ts = time.time()
    return arr, ts, p.name


def read_tif_batch(paths: List[Path], H: int, W: int, workers: int = 1):
    """Read a batch of TIFs. Uses threads by default because Streamlit + Windows spawn multiprocessing is fragile."""
    workers = max(1, int(workers))
    args = [(str(p), H, W) for p in paths]
    if workers <= 1 or len(args) <= 1:
        res = [_read_tif_one_for_import(a) for a in args]
    else:
        with ThreadPoolExecutor(max_workers=min(workers, len(args))) as ex:
            # map preserves input order
            res = list(ex.map(_read_tif_one_for_import, args))
    arrs = [r[0] for r in res]
    ts = [r[1] for r in res]
    names = [r[2] for r in res]
    return arrs, ts, names

def import_tif_stack_to_h5(tif_folder: Path, out_h5: Path, mask_path: str = "", mask_semantics: str = "one_is_valid", pixel_dtype_mode: str = "force_uint16", target_chunk_mib: float = 32.0, batch_frames: int = 32, progress_cb=None, tif_read_workers: int = 1, compression_mode: str = "bitshuffle_lz4", raw_frame_time_sec: Optional[float] = None, frame_time_source: str = "user_input") -> Dict[str, Any]:
    paths = sorted_tif_paths(tif_folder)
    if not paths:
        raise RuntimeError(f"No tif images found: {tif_folder}")
    sample = tiff.imread(str(paths[0]))
    if sample.ndim != 2:
        raise ValueError(f"Only 2D tif is supported: got {sample.shape}")
    H, W = map(int, sample.shape)
    target_dtype, clip_min, clip_max, sample_min, sample_max = decide_target_dtype(sample, pixel_dtype_mode)
    chunk_frames = calc_chunk_frames(H, W, np.dtype(target_dtype).itemsize, target_chunk_mib)

    mask = None
    if mask_path:
        mp = Path(mask_path)
        if mp.exists() and mp.suffix.lower() in [".tif", ".tiff"]:
            raw_mask = tiff.imread(str(mp))
            if raw_mask.shape != (H, W):
                raise ValueError(f"mask shape {raw_mask.shape} != image shape {(H, W)}")
            mask = normalize_mask_array(raw_mask, mask_semantics)

    ensure_dir(out_h5.parent)
    clean_incomplete_for_output(out_h5, remove_final=out_h5.exists())
    tmp_h5 = tmp_h5_path(out_h5)
    start_running_marker(out_h5, {"operation": "import_tif_stack", "source_tif_folder": str(tif_folder), "output_h5": str(out_h5), "total": int(len(paths))})
    filters = get_filters(compression_mode)
    str_dtype = h5py.string_dtype(encoding="utf-8")
    total = len(paths)

    with h5py.File(tmp_h5, "w", libver="latest") as hf:
        g_entry = hf.require_group("entry")
        g_data = g_entry.require_group("data")
        g_det = g_entry.require_group("instrument").require_group("detector")
        g_meta = g_entry.require_group("meta")
        d_img = g_data.create_dataset("images", shape=(total, H, W), dtype=target_dtype, chunks=(min(chunk_frames, total), H, W), **filters)
        d_ts = g_data.create_dataset("timestamps", shape=(total,), dtype=np.float64)
        d_src = g_data.create_dataset("source_filenames", shape=(total,), dtype=str_dtype)
        d_fid = g_data.create_dataset("frame_id", shape=(total,), dtype=np.int64)
        d_exp = g_data.create_dataset("exposure_time", shape=(total,), dtype=np.float32)
        if mask is not None:
            g_det.create_dataset("mask", data=mask.astype(np.uint8), dtype=np.uint8)
            g_det["mask"].attrs["mask_semantics"] = "1=valid,0=invalid"
        d_img.attrs["source_format"] = "tif_stack"
        d_img.attrs["pixel_dtype_mode"] = pixel_dtype_mode
        d_img.attrs["dtype_out"] = str(np.dtype(target_dtype))
        d_img.attrs["clip_min"] = "None" if clip_min is None else int(clip_min)
        d_img.attrs["clip_max"] = "None" if clip_max is None else int(clip_max)
        d_img.attrs["compression"] = compression_label(compression_mode)
        if _to_optional_positive_float(raw_frame_time_sec) is None:
            raw_frame_time_sec = infer_frame_time_sec_from_text(str(tif_folder), str(mask_path))
            if raw_frame_time_sec is not None:
                frame_time_source = "inferred_from_path"
        time_meta = apply_time_metadata_to_images_dataset(d_img, raw_frame_time_sec, time_bin=1, frame_time_source=frame_time_source)
        g_meta.create_dataset("software", data="DXB_00_Project_ImportBin_Manager_v5.6 tif->images.h5", dtype=str_dtype)
        g_meta.create_dataset("created_at", data=now_iso(), dtype=str_dtype)
        g_meta.create_dataset("source_tif_folder", data=str(tif_folder), dtype=str_dtype)
        write_time_metadata_to_meta_group(g_meta, time_meta)
        wptr = 0
        for k in range(0, total, int(batch_frames)):
            sub = paths[k:k + int(batch_frames)]
            arrs, ts, names = read_tif_batch(sub, H, W, workers=int(tif_read_workers))
            batch = cast_to_target(np.stack(arrs, axis=0), target_dtype, clip_min, clip_max)
            n = batch.shape[0]
            d_img[wptr:wptr+n] = batch
            d_ts[wptr:wptr+n] = np.asarray(ts, dtype=np.float64)
            d_src[wptr:wptr+n] = np.asarray(names, dtype=object)
            d_fid[wptr:wptr+n] = np.arange(wptr, wptr+n, dtype=np.int64)
            d_exp[wptr:wptr+n] = np.nan
            wptr += n
            if progress_cb:
                progress_cb(wptr, total)
            update_running_marker(out_h5, wptr, total, stage="import_tif", message=f"{wptr}/{total} frames")

    update_running_marker(out_h5, total, total, stage="commit", message="renaming tmp to final")
    commit_tmp_h5(tmp_h5, out_h5)
    info = {
        "source_type": "tif_stack",
        "source_tif_folder": str(tif_folder),
        "output_h5": str(out_h5),
        "frames": int(total), "height": int(H), "width": int(W),
        "dtype_out": str(np.dtype(target_dtype)),
        "pixel_dtype_mode": pixel_dtype_mode,
        "mask": bool(mask is not None),
        "mask_semantics": "1=valid,0=invalid" if mask is not None else "no_mask",
        "created_at": now_iso(),
        "compression": compression_label(compression_mode),
        "raw_frame_time_sec": time_meta.get("raw_frame_time_sec"),
        "time_bin": time_meta.get("time_bin"),
        "effective_frame_time_sec": time_meta.get("effective_frame_time_sec"),
        "frame_time_source": time_meta.get("frame_time_source"),
        "tif_read_workers": int(tif_read_workers),
        "batch_frames": int(batch_frames),
    }
    save_json(out_h5.parent / IMAGES_MANIFEST, info)
    finish_running_marker(out_h5)
    return info


def import_raw_binary_stack_to_h5(raw_folder: Path, out_h5: Path, mask_path: str = "", mask_semantics: str = "one_is_valid",
                                  pixel_dtype_mode: str = "force_uint16", target_chunk_mib: float = 32.0,
                                  batch_frames: int = 32, progress_cb=None, raw_read_workers: int = 1,
                                  compression_mode: str = "bitshuffle_lz4", raw_frame_time_sec: Optional[float] = None,
                                  frame_time_source: str = "user_input", raw_width: int = RAW_BINARY_DEFAULT_WIDTH,
                                  raw_height: int = RAW_BINARY_DEFAULT_HEIGHT, raw_dtype: str = RAW_BINARY_DEFAULT_DTYPE,
                                  raw_offset_bytes: int = RAW_BINARY_DEFAULT_OFFSET_BYTES) -> Dict[str, Any]:
    """Import extensionless/headerless RAW binary frames into canonical DXB images.h5."""
    raw_width = int(raw_width)
    raw_height = int(raw_height)
    raw_offset_bytes = int(raw_offset_bytes)
    raw_dtype = str(raw_dtype)
    paths = sorted_raw_binary_paths(raw_folder, raw_width, raw_height, raw_dtype, raw_offset_bytes)
    if not paths:
        expected = raw_binary_expected_bytes(raw_width, raw_height, raw_dtype, raw_offset_bytes)
        raise RuntimeError(f"No RAW binary frames found: {raw_folder} (expected file size={expected} bytes)")

    sample, _, _ = _read_raw_binary_one_for_import((str(paths[0]), raw_height, raw_width, raw_dtype, raw_offset_bytes))
    H, W = map(int, sample.shape)
    target_dtype, clip_min, clip_max, sample_min, sample_max = decide_target_dtype(sample, pixel_dtype_mode)
    chunk_frames = calc_chunk_frames(H, W, np.dtype(target_dtype).itemsize, target_chunk_mib)

    mask = None
    if mask_path:
        mp = Path(mask_path)
        if mp.exists() and mp.suffix.lower() in [".tif", ".tiff"]:
            raw_mask = tiff.imread(str(mp))
            if raw_mask.shape != (H, W):
                raise ValueError(f"mask shape {raw_mask.shape} != image shape {(H, W)}")
            mask = normalize_mask_array(raw_mask, mask_semantics)

    ensure_dir(out_h5.parent)
    clean_incomplete_for_output(out_h5, remove_final=out_h5.exists())
    tmp_h5 = tmp_h5_path(out_h5)
    start_running_marker(out_h5, {"operation": "import_raw_binary_stack", "source_raw_folder": str(raw_folder), "output_h5": str(out_h5), "total": int(len(paths))})
    filters = get_filters(compression_mode)
    str_dtype = h5py.string_dtype(encoding="utf-8")
    total = len(paths)

    try:
        with h5py.File(tmp_h5, "w", libver="latest") as hf:
            g_entry = hf.require_group("entry")
            g_data = g_entry.require_group("data")
            g_det = g_entry.require_group("instrument").require_group("detector")
            g_meta = g_entry.require_group("meta")
            d_img = g_data.create_dataset("images", shape=(total, H, W), dtype=target_dtype, chunks=(min(chunk_frames, total), H, W), **filters)
            d_ts = g_data.create_dataset("timestamps", shape=(total,), dtype=np.float64)
            d_src = g_data.create_dataset("source_filenames", shape=(total,), dtype=str_dtype)
            d_fid = g_data.create_dataset("frame_id", shape=(total,), dtype=np.int64)
            d_exp = g_data.create_dataset("exposure_time", shape=(total,), dtype=np.float32)
            if mask is not None:
                g_det.create_dataset("mask", data=mask.astype(np.uint8), dtype=np.uint8)
                g_det["mask"].attrs["mask_semantics"] = "1=valid,0=invalid"
            d_img.attrs["source_format"] = "raw_binary_stack"
            d_img.attrs["raw_binary_width"] = int(raw_width)
            d_img.attrs["raw_binary_height"] = int(raw_height)
            d_img.attrs["raw_binary_dtype"] = str(raw_dtype)
            d_img.attrs["raw_binary_offset_bytes"] = int(raw_offset_bytes)
            d_img.attrs["pixel_dtype_mode"] = pixel_dtype_mode
            d_img.attrs["dtype_in"] = str(_raw_binary_dtype(raw_dtype))
            d_img.attrs["dtype_out"] = str(np.dtype(target_dtype))
            d_img.attrs["clip_min"] = "None" if clip_min is None else int(clip_min)
            d_img.attrs["clip_max"] = "None" if clip_max is None else int(clip_max)
            d_img.attrs["compression"] = compression_label(compression_mode)
            if _to_optional_positive_float(raw_frame_time_sec) is None:
                raw_frame_time_sec = infer_frame_time_sec_from_text(str(raw_folder), str(mask_path))
                if raw_frame_time_sec is not None:
                    frame_time_source = "inferred_from_path"
            time_meta = apply_time_metadata_to_images_dataset(d_img, raw_frame_time_sec, time_bin=1, frame_time_source=frame_time_source)
            g_meta.create_dataset("software", data="DXB_00_Project_ImportBin_Manager_v5.7 raw_binary->images.h5", dtype=str_dtype)
            g_meta.create_dataset("created_at", data=now_iso(), dtype=str_dtype)
            g_meta.create_dataset("source_raw_folder", data=str(raw_folder), dtype=str_dtype)
            g_meta.create_dataset("raw_binary_width", data=int(raw_width))
            g_meta.create_dataset("raw_binary_height", data=int(raw_height))
            g_meta.create_dataset("raw_binary_dtype", data=str(raw_dtype), dtype=str_dtype)
            g_meta.create_dataset("raw_binary_offset_bytes", data=int(raw_offset_bytes))
            write_time_metadata_to_meta_group(g_meta, time_meta)

            wptr = 0
            for k in range(0, total, int(batch_frames)):
                sub = paths[k:k + int(batch_frames)]
                arrs, ts, names = read_raw_binary_batch(sub, H, W, raw_dtype, raw_offset_bytes, workers=int(raw_read_workers))
                batch = cast_to_target(np.stack(arrs, axis=0), target_dtype, clip_min, clip_max)
                n = batch.shape[0]
                d_img[wptr:wptr+n] = batch
                d_ts[wptr:wptr+n] = np.asarray(ts, dtype=np.float64)
                d_src[wptr:wptr+n] = np.asarray(names, dtype=object)
                d_fid[wptr:wptr+n] = np.arange(wptr, wptr+n, dtype=np.int64)
                d_exp[wptr:wptr+n] = np.nan
                wptr += n
                if progress_cb:
                    progress_cb(wptr, total, stage="import_raw_binary", message=f"{wptr}/{total} frames")
                update_running_marker(out_h5, wptr, total, stage="import_raw_binary", message=f"{wptr}/{total} frames")
                if wptr % max(1, int(batch_frames) * 4) == 0:
                    try:
                        hf.flush()
                    except Exception:
                        pass

        update_running_marker(out_h5, total, total, stage="commit", message="renaming tmp to final")
        commit_tmp_h5(tmp_h5, out_h5)
        info = {
            "source_type": "raw_binary_stack",
            "source_raw_folder": str(raw_folder),
            "output_h5": str(out_h5),
            "frames": int(total), "height": int(H), "width": int(W),
            "dtype_in": str(_raw_binary_dtype(raw_dtype)),
            "dtype_out": str(np.dtype(target_dtype)),
            "pixel_dtype_mode": pixel_dtype_mode,
            "raw_binary_width": int(raw_width),
            "raw_binary_height": int(raw_height),
            "raw_binary_dtype": str(raw_dtype),
            "raw_binary_offset_bytes": int(raw_offset_bytes),
            "expected_file_size": raw_binary_expected_bytes(raw_width, raw_height, raw_dtype, raw_offset_bytes),
            "mask": bool(mask is not None),
            "mask_semantics": "1=valid,0=invalid" if mask is not None else "no_mask",
            "created_at": now_iso(),
            "compression": compression_label(compression_mode),
            "raw_frame_time_sec": time_meta.get("raw_frame_time_sec"),
            "time_bin": time_meta.get("time_bin"),
            "effective_frame_time_sec": time_meta.get("effective_frame_time_sec"),
            "frame_time_source": time_meta.get("frame_time_source"),
            "raw_read_workers": int(raw_read_workers),
            "batch_frames": int(batch_frames),
        }
        save_json(out_h5.parent / IMAGES_MANIFEST, info)
        finish_running_marker(out_h5)
        return info
    except Exception as e:
        fail_running_marker(out_h5, e)
        raise


def read_scalar(fin: h5py.File, path: str, default=None):
    try:
        if path in fin:
            v = fin[path][()]
            if isinstance(v, (bytes, bytearray)):
                return v.decode("utf-8", errors="replace")
            return v
    except Exception:
        pass
    return default


def _to_optional_positive_float(value) -> Optional[float]:
    """Return a positive float or None. Accepts blanks, numpy scalars, strings."""
    if value is None:
        return None
    try:
        if isinstance(value, (bytes, bytearray)):
            value = value.decode("utf-8", errors="replace")
        if isinstance(value, str):
            txt = value.strip()
            if not txt or txt.lower() in {"none", "nan", "null", "-"}:
                return None
            value = txt
        v = float(value)
        if np.isfinite(v) and v > 0:
            return v
    except Exception:
        return None
    return None


def infer_frame_time_sec_from_text(*texts) -> Optional[float]:
    """Best-effort extraction of frame interval from names like 250ms, 500 ms, 0.5s."""
    joined = " ".join(str(t or "") for t in texts)
    # Prefer explicit units. Avoid unitless numbers because sample IDs often contain dates/IDs.
    matches = re.findall(r"(?<![A-Za-z0-9])(\d+(?:\.\d+)?)\s*(ms|msec|millisecond|milliseconds|s|sec|second|seconds)(?![A-Za-z])", joined, flags=re.I)
    if not matches:
        return None
    # Use the last explicit time-like token; paths often contain several date-like parts but only one with unit.
    val_txt, unit = matches[-1]
    try:
        v = float(val_txt)
    except Exception:
        return None
    unit = unit.lower()
    if unit.startswith("m"):
        return v / 1000.0
    return v


def get_h5_time_metadata(hf: h5py.File, fallback_texts: Optional[List[str]] = None) -> Dict[str, Any]:
    """Read DXB time metadata from an open H5.

    Canonical convention:
      raw_frame_time_sec       = original unbinned frame interval
      time_bin                 = temporal bin factor
      effective_frame_time_sec = raw_frame_time_sec * time_bin
    """
    fallback_texts = fallback_texts or []
    dset = hf.get("/entry/data/images", None)
    raw = None
    effective = None
    time_bin = None
    source = "missing"

    if dset is not None:
        raw = _to_optional_positive_float(dset.attrs.get("raw_frame_time_sec", None))
        effective = _to_optional_positive_float(dset.attrs.get("effective_frame_time_sec", None))
        time_bin = _to_optional_positive_float(dset.attrs.get("time_bin", None))
        src_attr = dset.attrs.get("frame_time_source", None)
        if src_attr is not None:
            try:
                source = src_attr.decode("utf-8", errors="replace") if isinstance(src_attr, (bytes, bytearray)) else str(src_attr)
            except Exception:
                source = "dataset_attrs"

    # HDF5 meta group fallbacks
    for key in ["raw_frame_time_sec", "frame_time_sec", "frame_time"]:
        if raw is None:
            raw = _to_optional_positive_float(read_scalar(hf, f"/entry/meta/{key}", None))
            if raw is not None and source == "missing":
                source = f"/entry/meta/{key}"
    if effective is None:
        effective = _to_optional_positive_float(read_scalar(hf, "/entry/meta/effective_frame_time_sec", None))
        if effective is not None and source == "missing":
            source = "/entry/meta/effective_frame_time_sec"
    if time_bin is None:
        time_bin = _to_optional_positive_float(read_scalar(hf, "/entry/meta/time_bin", None))

    # EIGER-style detector metadata fallback
    if raw is None:
        raw = _to_optional_positive_float(read_scalar(hf, "/entry/instrument/detector/frame_time", None))
        if raw is not None and source == "missing":
            source = "/entry/instrument/detector/frame_time"

    if raw is None:
        raw = infer_frame_time_sec_from_text(*fallback_texts)
        if raw is not None and source == "missing":
            source = "inferred_from_path"

    if time_bin is None:
        time_bin = 1.0
    try:
        time_bin_i = int(round(float(time_bin)))
    except Exception:
        time_bin_i = 1
    if time_bin_i <= 0:
        time_bin_i = 1

    if effective is None and raw is not None:
        effective = float(raw) * int(time_bin_i)
        if source == "missing":
            source = "computed_raw_times_time_bin"

    return {
        "raw_frame_time_sec": raw,
        "time_bin": int(time_bin_i),
        "effective_frame_time_sec": effective,
        "frame_time_source": source,
    }


def apply_time_metadata_to_images_dataset(dset, raw_frame_time_sec: Optional[float], time_bin: int = 1, frame_time_source: str = "user_input") -> Dict[str, Any]:
    """Write canonical time metadata to /entry/data/images attrs and return serializable metadata."""
    raw = _to_optional_positive_float(raw_frame_time_sec)
    tb = max(1, int(time_bin))
    effective = float(raw) * tb if raw is not None else None
    dset.attrs["time_bin"] = int(tb)
    if raw is not None:
        dset.attrs["raw_frame_time_sec"] = float(raw)
        dset.attrs["effective_frame_time_sec"] = float(effective)
        dset.attrs["frame_time_source"] = str(frame_time_source)
    else:
        dset.attrs["frame_time_source"] = "missing"
    return {
        "raw_frame_time_sec": raw,
        "time_bin": int(tb),
        "effective_frame_time_sec": effective,
        "frame_time_source": str(frame_time_source) if raw is not None else "missing",
    }


def write_time_metadata_to_meta_group(g_meta, meta: Dict[str, Any]):
    """Mirror time metadata into /entry/meta for easier inspection."""
    str_dtype = h5py.string_dtype("utf-8")
    for key in ["raw_frame_time_sec", "effective_frame_time_sec"]:
        val = meta.get(key)
        if val is not None:
            try:
                if key in g_meta:
                    del g_meta[key]
                g_meta.create_dataset(key, data=float(val))
            except Exception:
                pass
    if "time_bin" in meta:
        try:
            if "time_bin" in g_meta:
                del g_meta["time_bin"]
            g_meta.create_dataset("time_bin", data=int(meta.get("time_bin", 1)))
        except Exception:
            pass
    try:
        if "frame_time_source" in g_meta:
            del g_meta["frame_time_source"]
        g_meta.create_dataset("frame_time_source", data=str(meta.get("frame_time_source", "missing")), dtype=str_dtype)
    except Exception:
        pass


def build_valid_mask_from_eiger(fin: h5py.File, h: int, w: int):
    p = "/entry/instrument/detector/detectorSpecific/pixel_mask"
    if p not in fin:
        return None
    pm = fin[p][:]
    if pm.shape != (h, w):
        return None
    return (pm == 0).astype(np.uint8)


def merge_masks(mask_list: List[np.ndarray], mode: str) -> Optional[np.ndarray]:
    if not mask_list:
        return None
    mode = mode.lower().strip()
    if mode == "first":
        return mask_list[0].astype(np.uint8)
    if mode == "and":
        m = mask_list[0].astype(bool)
        for mm in mask_list[1:]:
            m &= mm.astype(bool)
        return m.astype(np.uint8)
    if mode == "or":
        m = mask_list[0].astype(bool)
        for mm in mask_list[1:]:
            m |= mm.astype(bool)
        return m.astype(np.uint8)
    raise ValueError("mask merge mode must be first/and/or")


def import_eiger_masters_to_h5(master_paths: List[Path], out_h5: Path, out_dtype_mode: str = "keep", mask_merge_mode: str = "first", target_chunk_mib: float = 64.0, batch_frames: int = 512, progress_cb=None, compression_mode: str = "bitshuffle_lz4", raw_frame_time_sec: Optional[float] = None, frame_time_source: str = "user_input") -> Dict[str, Any]:
    if not master_paths:
        raise RuntimeError("No master paths")
    masters = sorted([Path(p).resolve() for p in master_paths], key=natural_key)
    ensure_dir(out_h5.parent)
    clean_incomplete_for_output(out_h5, remove_final=out_h5.exists())
    tmp_h5 = tmp_h5_path(out_h5)
    start_running_marker(out_h5, {"operation": "import_eiger_masters", "source_masters": [str(p) for p in masters], "output_h5": str(out_h5)})
    total_frames = 0
    H = W = None
    in_dtype = None
    masks: List[np.ndarray] = []
    master_infos = []
    meta_first: Dict[str, Any] = {}
    cwd0 = os.getcwd()
    try:
        for mp in masters:
            os.chdir(str(mp.parent))
            with h5py.File(mp, "r") as fin:
                blocks = find_eiger_blocks(fin)
                if not blocks:
                    raise RuntimeError(f"No EIGER blocks in {mp}")
                h, w = int(blocks[0][1].shape[1]), int(blocks[0][1].shape[2])
                dt = blocks[0][1].dtype
                if H is None:
                    H, W, in_dtype = h, w, dt
                    meta_first = {
                        "frame_time": read_scalar(fin, "/entry/instrument/detector/frame_time", None),
                        "count_time": read_scalar(fin, "/entry/instrument/detector/count_time", None),
                        "incident_wavelength": read_scalar(fin, "/entry/instrument/beam/incident_wavelength", None),
                    }
                else:
                    if (h, w) != (H, W):
                        raise RuntimeError(f"shape mismatch: {mp} got={(h, w)} expected={(H, W)}")
                    if dt != in_dtype:
                        raise RuntimeError(f"dtype mismatch: {mp} got={dt} expected={in_dtype}")
                n_frames = int(sum(int(ds.shape[0]) for _, ds in blocks))
                mv = build_valid_mask_from_eiger(fin, H, W)
                if mv is not None:
                    masks.append(mv)
                master_infos.append({"master_h5": str(mp), "blocks": [k for k, _ in blocks], "frames": n_frames, "frame_start": total_frames, "frame_end": total_frames + n_frames})
                total_frames += n_frames
        if out_dtype_mode == "uint16":
            out_dtype = np.uint16
            clip_min, clip_max = 0, 65535
        elif out_dtype_mode == "keep":
            out_dtype = in_dtype
            clip_min = clip_max = None
        else:
            raise ValueError("EIGER dtype mode must be keep or uint16")
        mask_valid = merge_masks(masks, mask_merge_mode) if masks else None
        chunk_frames = calc_chunk_frames(H, W, np.dtype(out_dtype).itemsize, target_chunk_mib)
        filters = get_filters(compression_mode)
        str_dtype = h5py.string_dtype(encoding="utf-8")
        with h5py.File(tmp_h5, "w", libver="latest") as fout:
            g_entry = fout.require_group("entry")
            g_data = g_entry.require_group("data")
            g_det = g_entry.require_group("instrument").require_group("detector")
            g_meta = g_entry.require_group("meta")
            d_img = g_data.create_dataset("images", shape=(total_frames, H, W), dtype=out_dtype, chunks=(min(chunk_frames, total_frames), H, W), **filters)
            g_meta.create_dataset("source_master_h5_list", data=np.array([x["master_h5"] for x in master_infos], dtype=str_dtype))
            g_meta.create_dataset("source_master_frame_ranges_json", data=json.dumps(master_infos, ensure_ascii=False).encode("utf-8"))
            g_meta.create_dataset("software", data="DXB_00_Project_ImportBin_Manager_v5 eiger->images.h5", dtype=str_dtype)
            g_meta.create_dataset("created_at", data=now_iso(), dtype=str_dtype)
            for key, val in meta_first.items():
                if val is not None:
                    try:
                        g_meta.create_dataset(key, data=float(val))
                    except Exception:
                        g_meta.create_dataset(key, data=str(val), dtype=str_dtype)
            if mask_valid is not None:
                g_det.create_dataset("mask", data=mask_valid.astype(np.uint8), dtype=np.uint8)
                g_det["mask"].attrs["mask_semantics"] = "1=valid,0=invalid"
            d_img.attrs["source_format"] = "EIGER master.h5 merged"
            d_img.attrs["dtype_in"] = str(in_dtype)
            d_img.attrs["dtype_out"] = str(np.dtype(out_dtype))
            d_img.attrs["out_dtype_mode"] = out_dtype_mode
            d_img.attrs["compression"] = compression_label(compression_mode)
            d_img.attrs["mask_merge_mode"] = mask_merge_mode
            raw_for_meta = _to_optional_positive_float(raw_frame_time_sec)
            source_for_meta = frame_time_source
            if raw_for_meta is None:
                raw_for_meta = _to_optional_positive_float(meta_first.get("frame_time"))
                source_for_meta = "eiger_detector_frame_time" if raw_for_meta is not None else "missing"
            time_meta = apply_time_metadata_to_images_dataset(d_img, raw_for_meta, time_bin=1, frame_time_source=source_for_meta)
            write_time_metadata_to_meta_group(g_meta, time_meta)
            wpos = 0
            last_flush_t = time.perf_counter()
            for mp in masters:
                os.chdir(str(mp.parent))
                with h5py.File(mp, "r") as fin:
                    for block_key, ds in find_eiger_blocks(fin):
                        n = int(ds.shape[0])
                        # Coarse progress mode: read/write one EIGER ExternalLink block as one set.
                        # This keeps UI/progress I/O small and avoids the Streamlit app spending time
                        # updating logs every small frame slice. Stop requests are checked after each set.
                        msg_read = f"{mp.name}/{block_key} -> out {wpos}-{wpos + n - 1}"
                        if progress_cb:
                            progress_cb(wpos, total_frames, stage="read_eiger", message=msg_read)

                        arr = np.asarray(ds[:, :, :])
                        if clip_min is not None:
                            arr = np.clip(arr, clip_min, clip_max)
                        if arr.dtype != out_dtype:
                            arr = arr.astype(out_dtype, copy=False)
                        d_img[wpos:wpos + n] = arr

                        wpos += n
                        # Flush only at completed EIGER block boundaries. This is much lighter than
                        # flushing every frame/batch but still keeps the output safer during long jobs.
                        fout.flush()

                        done_msg = f"{mp.name}/{block_key} done | {wpos}/{total_frames} frames"
                        if progress_cb:
                            progress_cb(wpos, total_frames, stage="import_eiger", message=done_msg)
                        update_running_marker(out_h5, wpos, total_frames, stage="import_eiger", message=done_msg)
                        del arr
        update_running_marker(out_h5, total_frames, total_frames, stage="commit", message="renaming tmp to final")
        commit_tmp_h5(tmp_h5, out_h5)
        info = {"source_type": "eiger_master_h5", "source_masters": [str(p) for p in masters], "output_h5": str(out_h5), "frames": int(total_frames), "height": int(H), "width": int(W), "dtype_in": str(in_dtype), "dtype_out": str(np.dtype(out_dtype)), "mask": bool(mask_valid is not None), "compression": compression_label(compression_mode), "raw_frame_time_sec": time_meta.get("raw_frame_time_sec"), "time_bin": time_meta.get("time_bin"), "effective_frame_time_sec": time_meta.get("effective_frame_time_sec"), "frame_time_source": time_meta.get("frame_time_source"), "batch_frames": int(batch_frames), "created_at": now_iso()}
        save_json(out_h5.parent / IMAGES_MANIFEST, info)
        finish_running_marker(out_h5)
        return info
    except Exception as e:
        fail_running_marker(out_h5, e)
        if progress_cb:
            try:
                progress_cb(0, max(1, int(total_frames or 1)), stage="error", message=f"{type(e).__name__}: {e}")
            except Exception:
                pass
        raise
    finally:
        try:
            os.chdir(cwd0)
        except Exception:
            pass


def annotate_images_h5_time_metadata(h5_path: Path, raw_frame_time_sec: Optional[float] = None, frame_time_source: str = "user_input") -> Dict[str, Any]:
    """Ensure canonical images.h5 carries raw/effective frame-time metadata."""
    h5_path = Path(h5_path)
    with h5py.File(h5_path, "r+") as hf:
        if "/entry/data/images" not in hf:
            return {"raw_frame_time_sec": None, "time_bin": 1, "effective_frame_time_sec": None, "frame_time_source": "missing_images_dataset"}
        dset = hf["/entry/data/images"]
        existing = get_h5_time_metadata(hf, fallback_texts=[str(h5_path)])
        raw = _to_optional_positive_float(raw_frame_time_sec)
        src = frame_time_source
        if raw is None:
            raw = existing.get("raw_frame_time_sec")
            src = existing.get("frame_time_source", "existing_h5_metadata") if raw is not None else "missing"
        meta = apply_time_metadata_to_images_dataset(dset, raw, time_bin=1, frame_time_source=src)
        g_meta = hf.require_group("entry").require_group("meta")
        write_time_metadata_to_meta_group(g_meta, meta)
    return meta


def copy_existing_images_h5(source_h5: Path, out_h5: Path, overwrite: bool = False, raw_frame_time_sec: Optional[float] = None, frame_time_source: str = "user_input") -> Dict[str, Any]:
    source_h5 = Path(source_h5)
    out_h5 = Path(out_h5)
    ensure_dir(out_h5.parent)
    if source_h5.resolve() == out_h5.resolve():
        time_meta = annotate_images_h5_time_metadata(out_h5, raw_frame_time_sec=raw_frame_time_sec, frame_time_source=frame_time_source)
        info = inspect_images_h5(out_h5)
        info.update({"source_type": "existing_images_h5", "source_h5": str(source_h5), "output_h5": str(out_h5), "copied": False, "raw_frame_time_sec": time_meta.get("raw_frame_time_sec"), "time_bin": time_meta.get("time_bin"), "effective_frame_time_sec": time_meta.get("effective_frame_time_sec"), "frame_time_source": time_meta.get("frame_time_source"), "created_at": now_iso()})
        save_json(out_h5.parent / IMAGES_MANIFEST, info)
        return info
    if out_h5.exists() and not overwrite:
        time_meta = annotate_images_h5_time_metadata(out_h5, raw_frame_time_sec=raw_frame_time_sec, frame_time_source=frame_time_source)
        info = inspect_images_h5(out_h5)
        info.update({"source_type": "existing_images_h5", "source_h5": str(source_h5), "output_h5": str(out_h5), "copied": False, "note": "existing canonical reused", "raw_frame_time_sec": time_meta.get("raw_frame_time_sec"), "time_bin": time_meta.get("time_bin"), "effective_frame_time_sec": time_meta.get("effective_frame_time_sec"), "frame_time_source": time_meta.get("frame_time_source"), "created_at": now_iso()})
        save_json(out_h5.parent / IMAGES_MANIFEST, info)
        return info
    clean_incomplete_for_output(out_h5, remove_final=out_h5.exists())
    tmp_h5 = tmp_h5_path(out_h5)
    start_running_marker(out_h5, {"operation": "copy_existing_images_h5", "source_h5": str(source_h5), "output_h5": str(out_h5), "total": 1})
    shutil.copy2(str(source_h5), str(tmp_h5))
    commit_tmp_h5(tmp_h5, out_h5)
    finish_running_marker(out_h5)
    time_meta = annotate_images_h5_time_metadata(out_h5, raw_frame_time_sec=raw_frame_time_sec, frame_time_source=frame_time_source)
    info = inspect_images_h5(out_h5)
    info.update({"source_type": "existing_images_h5", "source_h5": str(source_h5), "output_h5": str(out_h5), "copied": True, "raw_frame_time_sec": time_meta.get("raw_frame_time_sec"), "time_bin": time_meta.get("time_bin"), "effective_frame_time_sec": time_meta.get("effective_frame_time_sec"), "frame_time_source": time_meta.get("frame_time_source"), "created_at": now_iso()})
    save_json(out_h5.parent / IMAGES_MANIFEST, info)
    return info


def ensure_canonical_h5(row: pd.Series, analysis_root: Path, import_mode: str, pixel_dtype_mode: str, tif_mask_semantics: str, eiger_dtype_mode: str, target_chunk_mib: float, batch_frames_tif: int, batch_frames_eiger: int, tif_read_workers: int = 1, compression_mode: str = "bitshuffle_lz4", progress_cb=None, default_raw_frame_time_sec: Optional[float] = None,
                        raw_binary_width: int = RAW_BINARY_DEFAULT_WIDTH, raw_binary_height: int = RAW_BINARY_DEFAULT_HEIGHT,
                        raw_binary_dtype: str = RAW_BINARY_DEFAULT_DTYPE, raw_binary_offset_bytes: int = RAW_BINARY_DEFAULT_OFFSET_BYTES) -> Dict[str, Any]:
    dataset_id = safe_name(row["dataset_id"], "dataset")
    channel = safe_name(row.get("channel", "default"), "default")
    source_type = str(row["source_type"])
    out_h5 = canonical_h5_path(analysis_root, dataset_id, channel)
    selected_sources = [s.strip() for s in str(row["selected_sources"]).splitlines() if s.strip()]
    mask_path = clean_path_text(str(row.get("mask_path", "")))
    raw_frame_time_sec = _to_optional_positive_float(row.get("raw_frame_time_sec", None))
    frame_time_source = "candidate_table" if raw_frame_time_sec is not None else "missing"
    if raw_frame_time_sec is None:
        raw_frame_time_sec = _to_optional_positive_float(default_raw_frame_time_sec)
        frame_time_source = "default_import_setting" if raw_frame_time_sec is not None else "missing"
    if raw_frame_time_sec is None:
        raw_frame_time_sec = infer_frame_time_sec_from_text(row.get("selected_sources", ""), row.get("primary_path", ""), row.get("notes", ""))
        frame_time_source = "inferred_from_path" if raw_frame_time_sec is not None else "missing"
    make_or_update_channel_manifest(analysis_root, dataset_id, channel, source_type, str(row.get("source_root", "")), selected_sources, mask_path)

    if import_mode == "Reuse canonical H5":
        if not out_h5.exists():
            raise FileNotFoundError(f"canonical images.h5 not found: {out_h5}")
        info = inspect_images_h5(out_h5)
        update_channel_import_status(analysis_root, dataset_id, channel, "done", {"images_info": info})
        return info
    if out_h5.exists() and import_mode == "Build if missing":
        info = inspect_images_h5(out_h5)
        update_channel_import_status(analysis_root, dataset_id, channel, "done", {"images_info": info})
        return info

    update_channel_import_status(analysis_root, dataset_id, channel, "running")
    if source_type == "tif_stack":
        if not selected_sources:
            raise RuntimeError("No tif folder selected")
        info = import_tif_stack_to_h5(Path(selected_sources[0]), out_h5, mask_path=mask_path, mask_semantics=tif_mask_semantics, pixel_dtype_mode=pixel_dtype_mode, target_chunk_mib=target_chunk_mib, batch_frames=batch_frames_tif, progress_cb=progress_cb, tif_read_workers=tif_read_workers, compression_mode=compression_mode, raw_frame_time_sec=raw_frame_time_sec, frame_time_source=frame_time_source)
    elif source_type == "raw_binary_stack":
        if not selected_sources:
            raise RuntimeError("No RAW binary folder selected")
        info = import_raw_binary_stack_to_h5(
            Path(selected_sources[0]), out_h5,
            mask_path=mask_path, mask_semantics=tif_mask_semantics,
            pixel_dtype_mode=pixel_dtype_mode, target_chunk_mib=target_chunk_mib,
            batch_frames=batch_frames_tif, progress_cb=progress_cb,
            raw_read_workers=tif_read_workers, compression_mode=compression_mode,
            raw_frame_time_sec=raw_frame_time_sec, frame_time_source=frame_time_source,
            raw_width=int(raw_binary_width), raw_height=int(raw_binary_height),
            raw_dtype=str(raw_binary_dtype), raw_offset_bytes=int(raw_binary_offset_bytes),
        )
    elif source_type == "eiger_master_h5":
        masters = [Path(p) for p in selected_sources]
        info = import_eiger_masters_to_h5(masters, out_h5, out_dtype_mode=eiger_dtype_mode, mask_merge_mode="first", target_chunk_mib=target_chunk_mib, batch_frames=batch_frames_eiger, progress_cb=progress_cb, compression_mode=compression_mode, raw_frame_time_sec=raw_frame_time_sec, frame_time_source=frame_time_source)
    elif source_type == "existing_images_h5":
        if not selected_sources:
            raise RuntimeError("No images.h5 selected")
        info = copy_existing_images_h5(Path(selected_sources[0]), out_h5, overwrite=(import_mode == "Rebuild canonical H5"), raw_frame_time_sec=raw_frame_time_sec, frame_time_source=frame_time_source)
    else:
        raise ValueError(f"Unknown source_type: {source_type}")
    update_channel_import_status(analysis_root, dataset_id, channel, "done", {"images_info": info})
    return info


# =============================================================================
# Canonical H5 discovery and bin core
# =============================================================================

def discover_existing_bins(channel_root: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    bin_root = channel_root / "02_bin"
    if not bin_root.exists():
        return rows
    for idx_path in sorted(bin_root.glob("*x*/t*/index.json")):
        info = load_json(idx_path, default={})
        if info:
            out_h5 = Path(info.get("output_h5", "")) if info.get("output_h5") else None
            ok, msg, _ = validate_done_index(idx_path, out_h5)
            if not ok:
                info = dict(info)
                info["status"] = "incomplete"
                info["validation_message"] = msg
            rows.append(info)
    known = {str(Path(r.get("output_h5", "")).resolve()) for r in rows if r.get("output_h5")}
    for h5 in sorted(bin_root.glob("*x*/t*/images_*.h5")):
        if str(h5.resolve()) in known:
            continue
        rows.append({"status": "found_no_index", "output_h5": str(h5), "space_bin": "", "time_bin": ""})
    return rows


def scan_analysis_root(analysis_root: Path) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    if not analysis_root.exists():
        return pd.DataFrame()
    for h5 in sorted(analysis_root.rglob("01_h5/images.h5")):
        dataset_id, channel, channel_root = infer_dataset_channel_from_canonical(h5, analysis_root)
        info = inspect_images_h5(h5)
        bins = discover_existing_bins(channel_root)
        labels = []
        for b in bins:
            sb, tb = b.get("space_bin", ""), b.get("time_bin", "")
            if sb != "" and tb != "":
                labels.append(f"{sb}x{sb}/t{tb}")
            elif b.get("output_h5"):
                labels.append(str(Path(str(b["output_h5"])).parent.parent.name + "/" + Path(str(b["output_h5"])).parent.name))
        rows.append({
            "run": False,
            "dataset_id": dataset_id,
            "channel": channel,
            "images_h5": str(h5),
            "channel_root": str(channel_root),
            "frames": info.get("frames"),
            "height": info.get("height"),
            "width": info.get("width"),
            "dtype": info.get("dtype"),
            "mask": "yes" if info.get("has_mask") else "no",
            "mask_valid": info.get("mask_valid"),
            "raw_frame_time_sec": info.get("raw_frame_time_sec"),
            "effective_frame_time_sec": info.get("effective_frame_time_sec"),
            "frame_time_source": info.get("frame_time_source"),
            "existing_bins": ", ".join(labels),
            "n_bins": len(bins),
            "status": "ready" if info.get("ok") else "error",
            "error": info.get("error", ""),
        })
    return pd.DataFrame(rows)


def decide_bin_out_dtype(in_dtype, max_val, out_dtype_mode: str, max_amp: int = 108):
    if out_dtype_mode == "same":
        return np.dtype(in_dtype)
    max_after = float(max_val) * float(max_amp)
    if max_after < np.iinfo(np.int32).max:
        return np.int32
    return np.int64


def spatial_bin_block(block: np.ndarray, b: int, out_dtype):
    N, H_eff, W_eff = block.shape
    H_bin = H_eff // b
    W_bin = W_eff // b
    resh = block.reshape(N, H_bin, b, W_bin, b)
    return resh.sum(axis=(2, 4), dtype=out_dtype)


def spatial_bin_mask(mask: np.ndarray, b: int, mode: str = "all_valid", valid_ratio_threshold: float = 0.5):
    H, W = mask.shape
    H_eff = (H // b) * b
    W_eff = (W // b) * b
    sub = mask[:H_eff, :W_eff].astype(bool)
    H_bin = H_eff // b
    W_bin = W_eff // b
    resh = sub.reshape(H_bin, b, W_bin, b)
    if mode == "any_valid":
        out = resh.any(axis=(1, 3))
    elif mode == "all_valid":
        out = resh.all(axis=(1, 3))
    elif mode == "valid_ratio":
        valid_count = resh.sum(axis=(1, 3))
        out = (valid_count / float(b * b)) >= float(valid_ratio_threshold)
    else:
        raise ValueError(f"Unknown mask bin mode: {mode}")
    return out.astype(np.uint8)


def write_preview(out_h5: Path, preview_dir: Path, preview_mode: str = "first_last_mask", max_each: int = 2):
    """Write lightweight preview TIFs.

    preview_mode:
      none            : do nothing
      first_only      : first_0000.tif only
      first_last      : first and last frames
      first_last_mask : first/last + binned mask
    """
    mode = str(preview_mode or "none").lower().strip()
    if mode == "none":
        return
    ensure_dir(preview_dir)
    with h5py.File(out_h5, "r") as hf:
        imgs = hf["entry/data/images"]
        T = int(imgs.shape[0])
        n_first = 1 if mode == "first_only" else min(max_each, T)
        for i in range(min(n_first, T)):
            tiff.imwrite(str(preview_dir / f"first_{i:04d}.tif"), imgs[i].astype(np.uint32))
        if mode in ["first_last", "first_last_mask"]:
            for i in range(min(max_each, T)):
                idx = T - 1 - i
                tiff.imwrite(str(preview_dir / f"last_{i:04d}.tif"), imgs[idx].astype(np.uint32))
        if mode == "first_last_mask" and "entry/instrument/detector/mask" in hf:
            out_mask = hf["entry/instrument/detector/mask"][:]
            tiff.imwrite(str(preview_dir / "mask_binned.tif"), (out_mask * 255).astype(np.uint8))


def estimate_bin_memory_gb(source_h5: Path, space_bin: int, time_bin: int, chunk_factor: int) -> float:
    try:
        with h5py.File(source_h5, "r") as hf:
            d = hf["entry/data/images"]
            T, H, W = map(int, d.shape)
            bpp = np.dtype(d.dtype).itemsize
        chunk_size = max(1, int(time_bin) * int(chunk_factor))
        frames = min(T, chunk_size)
        return float(frames * H * W * bpp) / (1024 ** 3)
    except Exception:
        return float("nan")


def _relpath_for_h5_link(target: Path, link_file_dir: Path) -> str:
    """Return a portable path string for HDF5 ExternalLink."""
    try:
        rel = os.path.relpath(str(Path(target).resolve()), str(Path(link_file_dir).resolve()))
        return rel.replace(os.sep, "/")
    except Exception:
        return str(Path(target).resolve())



def create_alias_bin_h5(source_h5: Path, out_h5: Path, channel_root: Path, dataset_id: str, channel: str, preview_mode: str = "first_only", overwrite: bool = False) -> Dict[str, Any]:
    """
    Create a tiny alias H5 for 1x1/t1 using ExternalLink.
    Safety: write to *.tmp, then atomically rename to final H5. running/failed markers are used.
    """
    source_h5 = Path(source_h5)
    out_h5 = Path(out_h5)
    ensure_dir(out_h5.parent)
    if out_h5.exists() and not overwrite:
        ok, msg, info = is_complete_bin_output(out_h5.parent, out_h5)
        if ok and info:
            append_bin_manifest(channel_root, info)
            return info
        clean_incomplete_for_output(out_h5, remove_final=True)
    elif out_h5.exists():
        clean_incomplete_for_output(out_h5, remove_final=True)
    else:
        clean_incomplete_for_output(out_h5, remove_final=False)

    tmp_h5 = tmp_h5_path(out_h5)
    clean_incomplete_for_output(out_h5, remove_final=False)
    start_running_marker(out_h5, {
        "operation": "alias_bin_h5",
        "dataset_id": str(dataset_id),
        "channel": str(channel),
        "source_h5": str(source_h5),
        "output_h5": str(out_h5),
        "space_bin": 1,
        "time_bin": 1,
        "total": 1,
    })
    try:
        with h5py.File(source_h5, "r") as hf_src:
            if "entry/data/images" not in hf_src:
                raise KeyError(f"{source_h5}: /entry/data/images not found")
            d = hf_src["entry/data/images"]
            T, H, W = map(int, d.shape)
            in_dtype = d.dtype
            has_mask = "entry/instrument/detector/mask" in hf_src
            source_time_meta = get_h5_time_metadata(hf_src, fallback_texts=[str(source_h5)])
            alias_time_meta = {
                "raw_frame_time_sec": source_time_meta.get("raw_frame_time_sec"),
                "time_bin": 1,
                "effective_frame_time_sec": source_time_meta.get("raw_frame_time_sec"),
                "frame_time_source": source_time_meta.get("frame_time_source", "source_h5"),
            }

        link_name = _relpath_for_h5_link(source_h5, tmp_h5.parent)
        with h5py.File(tmp_h5, "w", libver="latest") as hf_out:
            g_entry = hf_out.require_group("entry")
            g_data = g_entry.require_group("data")
            g_inst = g_entry.require_group("instrument")
            g_det = g_inst.require_group("detector")
            g_meta = g_entry.require_group("meta")
            g_data["images"] = h5py.ExternalLink(link_name, "/entry/data/images")
            if has_mask:
                g_det["mask"] = h5py.ExternalLink(link_name, "/entry/instrument/detector/mask")
            str_dtype = h5py.string_dtype("utf-8")
            g_meta.create_dataset("software", data="DXB_00_Project_ImportBin_Manager_v5.5 alias 1x1/t1", dtype=str_dtype)
            g_meta.create_dataset("created_at", data=now_iso(), dtype=str_dtype)
            g_meta.create_dataset("alias_source_h5", data=str(source_h5), dtype=str_dtype)
            g_meta.create_dataset("alias_external_link", data=link_name, dtype=str_dtype)
            g_meta.attrs["is_alias_h5"] = True
            g_meta.attrs["space_bin"] = 1
            g_meta.attrs["time_bin"] = 1
            write_time_metadata_to_meta_group(g_meta, alias_time_meta)
        commit_tmp_h5(tmp_h5, out_h5)
        update_running_marker(out_h5, 1, 1, stage="index", message="writing index")
        info = {
            "schema_version": 6,
            "status": "done_alias",
            "is_alias_h5": True,
            "alias_mode": "external_link",
            "dataset_id": str(dataset_id),
            "channel": str(channel),
            "source_h5": str(source_h5),
            "output_h5": str(out_h5),
            "space_bin": 1,
            "time_bin": 1,
            "shape_original": [int(T), int(H), int(W)],
            "shape_binned": [int(T), int(H), int(W)],
            "frames": int(T),
            "height": int(H),
            "width": int(W),
            "dtype_in": str(in_dtype),
            "dtype_out": str(in_dtype),
            "filter": "alias_external_link_no_copy",
            "out_dtype_mode": "alias_no_copy",
            "mask_input_semantics": "linked_from_source" if has_mask else "no_mask",
            "mask_bin_mode": "linked_from_source" if has_mask else "no_mask",
            "mask_semantics_output": "1=valid,0=invalid" if has_mask else "no_mask",
            "preview_mode": str(preview_mode),
            "preview_written": bool(preview_mode != "none"),
            "raw_frame_time_sec": alias_time_meta.get("raw_frame_time_sec"),
            "effective_frame_time_sec": alias_time_meta.get("effective_frame_time_sec"),
            "frame_time_source": alias_time_meta.get("frame_time_source"),
            "safety": "tmp_then_atomic_rename",
            "created_at": now_iso(),
        }
        save_json(out_h5.parent / "index.json", info)
        append_bin_manifest(channel_root, info)
        if str(preview_mode) != "none":
            update_running_marker(out_h5, 1, 1, stage="preview", message="writing preview")
            write_preview(out_h5, out_h5.parent / "preview", preview_mode=preview_mode)
        finish_running_marker(out_h5)
        return info
    except Exception as e:
        fail_running_marker(out_h5, e)
        _safe_unlink(tmp_h5)
        raise



def bin_images_h5(source_h5: Path, channel_root: Path, dataset_id: str, channel: str, space_bin: int, time_bin: int, mask_input_semantics: str = "one_is_valid", mask_bin_mode: str = "all_valid", valid_ratio_threshold: float = 0.5, out_dtype_mode: str = "auto", chunk_factor: int = 256, preview_mode: str = "first_only", overwrite: bool = False, compression_mode: str = "bitshuffle_lz4", bin1_t1_mode: str = "alias", progress_cb=None) -> Dict[str, Any]:
    """Create one binned H5 safely.

    Safety behavior:
    - write to images_...h5.tmp first
    - keep images_...h5.running.json during processing
    - atomically rename tmp -> final only after HDF5 write succeeds
    - mark failed in images_...h5.failed.json if an exception occurs
    - skip only when index.json + H5 shape validation succeed
    """
    source_h5 = Path(source_h5)
    channel_root = Path(channel_root)
    space_bin = int(space_bin)
    time_bin = int(time_bin)
    time_dir = channel_root / "02_bin" / f"{space_bin}x{space_bin}" / f"t{time_bin}"
    ensure_dir(time_dir)
    out_h5 = time_dir / f"images_{space_bin}x{space_bin}_t{time_bin}.h5"
    bin1_t1_mode = str(bin1_t1_mode or "alias").lower().strip()

    if space_bin == 1 and time_bin == 1:
        if bin1_t1_mode in ["skip", "skip_alias"]:
            info = {"schema_version": 6, "status": "skipped_bin1_t1", "dataset_id": str(dataset_id), "channel": str(channel), "source_h5": str(source_h5), "output_h5": str(out_h5), "space_bin": 1, "time_bin": 1, "is_alias_h5": False, "created_at": now_iso()}
            save_json(time_dir / "index.json", info)
            append_bin_manifest(channel_root, info)
            return info
        if bin1_t1_mode in ["alias", "alias_no_copy", "external_link"]:
            return create_alias_bin_h5(source_h5, out_h5, channel_root, dataset_id, channel, preview_mode=preview_mode, overwrite=overwrite)
        # otherwise fall through to copy behavior.

    if out_h5.exists() and not overwrite:
        ok, msg, info = is_complete_bin_output(time_dir, out_h5)
        if ok and info:
            append_bin_manifest(channel_root, info)
            return info
        clean_incomplete_for_output(out_h5, remove_final=True)
    elif out_h5.exists():
        clean_incomplete_for_output(out_h5, remove_final=True)
    else:
        clean_incomplete_for_output(out_h5, remove_final=False)

    tmp_h5 = tmp_h5_path(out_h5)
    filters = get_filters(compression_mode)
    start_running_marker(out_h5, {
        "operation": "bin_images_h5",
        "dataset_id": str(dataset_id),
        "channel": str(channel),
        "source_h5": str(source_h5),
        "output_h5": str(out_h5),
        "space_bin": int(space_bin),
        "time_bin": int(time_bin),
    })
    try:
        with h5py.File(source_h5, "r") as hf_in:
            if "entry/data/images" not in hf_in:
                raise KeyError(f"{source_h5}: /entry/data/images not found")
            d_in = hf_in["entry/data/images"]
            T, H, W = map(int, d_in.shape)
            in_dtype = d_in.dtype
            source_time_meta = get_h5_time_metadata(hf_in, fallback_texts=[str(source_h5)])
            raw_frame_time_sec = source_time_meta.get("raw_frame_time_sec")
            effective_frame_time_sec = float(raw_frame_time_sec) * int(time_bin) if raw_frame_time_sec is not None else None
            frame_time_source = source_time_meta.get("frame_time_source", "missing")
            update_running_marker(out_h5, 0, T, stage="scan", message="sampling dtype/range")
            max_val = d_in[: min(200, T)].max()
            out_base_dtype = decide_bin_out_dtype(in_dtype, max_val, out_dtype_mode=out_dtype_mode)
            H_eff = (H // space_bin) * space_bin
            W_eff = (W // space_bin) * space_bin
            T_eff = (T // time_bin) * time_bin
            if H_eff == 0 or W_eff == 0 or T_eff == 0:
                raise RuntimeError(f"Invalid bin settings: b={space_bin}, t={time_bin}, source shape={(T,H,W)}")
            H_bin, W_bin, T_bin = H_eff // space_bin, W_eff // space_bin, T_eff // time_bin
            mask_bin = None
            if "entry/instrument/detector/mask" in hf_in:
                update_running_marker(out_h5, 0, T_eff, stage="mask", message="binning mask")
                raw_mask = hf_in["entry/instrument/detector/mask"][:]
                mask = normalize_mask_array(raw_mask, mask_input_semantics)
                mask_bin = spatial_bin_mask(mask, space_bin, mode=mask_bin_mode, valid_ratio_threshold=valid_ratio_threshold)
            with h5py.File(tmp_h5, "w", libver="latest") as hf_out:
                g_entry = hf_out.require_group("entry")
                g_data = g_entry.require_group("data")
                g_det = g_entry.require_group("instrument").require_group("detector")
                g_meta = g_entry.require_group("meta")
                d_out = g_data.create_dataset("images", shape=(T_bin, H_bin, W_bin), dtype=out_base_dtype, chunks=(min(32, T_bin), H_bin, W_bin), **filters)
                if mask_bin is not None:
                    g_det.create_dataset("mask", data=mask_bin.astype(np.uint8), dtype=np.uint8)
                    g_det["mask"].attrs["mask_semantics"] = "1=valid,0=invalid"
                if space_bin == 1 and time_bin == 1:
                    update_running_marker(out_h5, 0, T_eff, stage="copy", message="copying 1x1/t1")
                    d_out[...] = d_in[:T_eff]
                    if progress_cb:
                        progress_cb(T_eff, T_eff)
                    update_running_marker(out_h5, T_eff, T_eff, stage="copy", message="copy complete")
                else:
                    out_idx = 0
                    chunk_size = time_bin * int(chunk_factor)
                    if chunk_size > T_eff:
                        chunk_size = T_eff
                    for start in range(0, T_eff, chunk_size):
                        end = min(start + chunk_size, T_eff)
                        n = end - start
                        block = d_in[start:end, :H_eff, :W_eff]
                        block_sp = spatial_bin_block(block, space_bin, out_base_dtype)
                        groups = n // time_bin
                        block_tb = block_sp.reshape(groups, time_bin, H_bin, W_bin).sum(axis=1, dtype=out_base_dtype)
                        d_out[out_idx: out_idx + groups] = block_tb
                        out_idx += groups
                        if progress_cb:
                            progress_cb(end, T_eff)
                        update_running_marker(out_h5, end, T_eff, stage="processing", message=f"{end}/{T_eff} frames")
                d_out.attrs["source_h5"] = str(source_h5)
                d_out.attrs["dataset_id"] = str(dataset_id)
                d_out.attrs["channel"] = str(channel)
                d_out.attrs["space_bin"] = int(space_bin)
                d_out.attrs["time_bin"] = int(time_bin)
                time_meta = apply_time_metadata_to_images_dataset(d_out, raw_frame_time_sec, time_bin=int(time_bin), frame_time_source=frame_time_source)
                d_out.attrs["dtype_in"] = str(in_dtype)
                d_out.attrs["dtype_out"] = str(np.dtype(out_base_dtype))
                d_out.attrs["mask_input_semantics"] = mask_input_semantics if mask_bin is not None else "no_mask"
                d_out.attrs["mask_bin_mode"] = mask_bin_mode if mask_bin is not None else "no_mask"
                d_out.attrs["mask_semantics_output"] = "1=valid,0=invalid"
                d_out.attrs["created_at"] = now_iso()
                g_meta.create_dataset("software", data="DXB_00_Project_ImportBin_Manager_v5.6 safe bin", dtype=h5py.string_dtype("utf-8"))
                g_meta.create_dataset("created_at", data=now_iso(), dtype=h5py.string_dtype("utf-8"))
                write_time_metadata_to_meta_group(g_meta, time_meta)
        update_running_marker(out_h5, T_eff, T_eff, stage="commit", message="renaming tmp to final")
        commit_tmp_h5(tmp_h5, out_h5)
        info = {"schema_version": 6, "status": "done", "dataset_id": str(dataset_id), "channel": str(channel), "source_h5": str(source_h5), "output_h5": str(out_h5), "space_bin": int(space_bin), "time_bin": int(time_bin), "raw_frame_time_sec": time_meta.get("raw_frame_time_sec"), "effective_frame_time_sec": time_meta.get("effective_frame_time_sec"), "frame_time_source": time_meta.get("frame_time_source"), "shape_original": [int(T), int(H), int(W)], "shape_binned": [int(T_bin), int(H_bin), int(W_bin)], "frames": int(T_bin), "height": int(H_bin), "width": int(W_bin), "dtype_in": str(in_dtype), "dtype_out": str(np.dtype(out_base_dtype)), "filter": compression_label(compression_mode), "out_dtype_mode": out_dtype_mode, "mask_input_semantics": mask_input_semantics if mask_bin is not None else "no_mask", "mask_bin_mode": mask_bin_mode if mask_bin is not None else "no_mask", "valid_ratio_threshold": float(valid_ratio_threshold), "mask_semantics_output": "1=valid,0=invalid", "preview_mode": str(preview_mode), "preview_written": bool(preview_mode != "none"), "chunk_factor": int(chunk_factor), "compression": compression_label(compression_mode), "safety": "tmp_then_atomic_rename", "created_at": now_iso()}
        save_json(time_dir / "index.json", info)
        append_bin_manifest(channel_root, info)
        if str(preview_mode) != "none":
            update_running_marker(out_h5, T_eff, T_eff, stage="preview", message="writing preview")
            write_preview(out_h5, time_dir / "preview", preview_mode=preview_mode)
        finish_running_marker(out_h5)
        return info
    except Exception as e:
        fail_running_marker(out_h5, e)
        _safe_unlink(tmp_h5)
        raise


# -----------------------------------------------------------------------------
# Fused binning: make multiple space bins from the same source H5 in one read pass
# -----------------------------------------------------------------------------

def _call_progress(progress_cb, current: int, total: int, stage: str = "running", message: str = "", extra: Optional[Dict[str, Any]] = None):
    """Call old/new progress callbacks without breaking compatibility."""
    if progress_cb is None:
        return
    try:
        progress_cb(int(current), int(total), stage=stage, message=message, extra=extra or {})
    except TypeError:
        progress_cb(int(current), int(total))


def _same_fused_group_key(r: Dict[str, Any]) -> Tuple[str, str, str, str, int, str, float, str, str, str, str]:
    """Group recipes that can be produced in one read pass.

    We fuse only recipes sharing the same source H5, channel root, time bin and relevant modes.
    Different space bins are allowed. This avoids reading the same source H5 repeatedly.
    """
    return (
        str(r.get("images_h5", "")),
        str(r.get("channel_root", "")),
        str(r.get("dataset_id", "")),
        str(r.get("channel", "")),
        int(r.get("time_bin", 1)),
        str(r.get("mask_bin_mode", "all_valid")),
        float(r.get("valid_ratio", 0.5)),
        str(r.get("bin_dtype_mode", "auto")),
        str(r.get("preview_mode", "first_only")),
        str(r.get("compression_mode", "bitshuffle_lz4")),
        str(r.get("bin1_t1_mode", "alias")),
    )


def group_bin_jobs_for_fused(jobs: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    groups: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = {}
    for r in jobs:
        groups.setdefault(_same_fused_group_key(r), []).append(r)
    # Keep deterministic ordering and largest groups first for clearer progress.
    out = list(groups.values())
    out.sort(key=lambda g: (str(g[0].get("dataset_id", "")), str(g[0].get("channel", "")), int(g[0].get("time_bin", 1))))
    return out


def bin_images_h5_multi_same_source(recipes: List[Dict[str, Any]], progress_cb=None) -> List[Dict[str, Any]]:
    """Create multiple binned H5 files from one source H5 in a single read pass.

    This is intended for common packages such as 8/10/12 x t1. It opens the source
    only once and processes each frame block once, then writes all requested outputs.
    1x1/t1 alias/skip recipes are handled without reading image data.

    If the recipes cannot be fused safely, call bin_images_h5 individually instead.
    """
    if not recipes:
        return []

    first = recipes[0]
    source_h5 = Path(str(first["images_h5"]))
    channel_root = Path(str(first["channel_root"]))
    dataset_id = str(first.get("dataset_id", ""))
    channel = str(first.get("channel", "default"))
    time_bin = int(first.get("time_bin", 1))
    mask_bin_mode = str(first.get("mask_bin_mode", "all_valid"))
    valid_ratio_threshold = float(first.get("valid_ratio", 0.5))
    out_dtype_mode = str(first.get("bin_dtype_mode", "auto"))
    preview_mode = str(first.get("preview_mode", "first_only"))
    compression_mode = str(first.get("compression_mode", "bitshuffle_lz4"))
    bin1_t1_mode = str(first.get("bin1_t1_mode", "alias"))
    chunk_factor = max(int(r.get("chunk_factor", 256)) for r in recipes)

    results: List[Dict[str, Any]] = []
    real_recipes: List[Dict[str, Any]] = []

    # Handle 1x1/t1 alias/skip/copy. copy falls through to real pass.
    for r in recipes:
        b = int(r.get("space_bin", 1))
        tb = int(r.get("time_bin", 1))
        if b == 1 and tb == 1 and bin1_t1_mode in ["alias", "alias_no_copy", "external_link", "skip", "skip_alias"]:
            _call_progress(progress_cb, 0, 1, stage="alias", message=f"{dataset_id}/{channel} 1x1 t1")
            info = bin_images_h5(
                source_h5, channel_root, dataset_id, channel, b, tb,
                mask_input_semantics="one_is_valid",
                mask_bin_mode=mask_bin_mode,
                valid_ratio_threshold=valid_ratio_threshold,
                out_dtype_mode=out_dtype_mode,
                chunk_factor=chunk_factor,
                preview_mode=preview_mode,
                overwrite=bool(r.get("overwrite", False)),
                compression_mode=compression_mode,
                bin1_t1_mode=bin1_t1_mode,
                progress_cb=None,
            )
            results.append(info)
        else:
            real_recipes.append(r)

    if not real_recipes:
        _call_progress(progress_cb, 1, 1, stage="done", message="alias/skip only")
        return results

    # Remove existing outputs or skip existing.
    active: List[Dict[str, Any]] = []
    for r in real_recipes:
        b = int(r["space_bin"])
        tb = int(r["time_bin"])
        time_dir = channel_root / "02_bin" / f"{b}x{b}" / f"t{tb}"
        ensure_dir(time_dir)
        out_h5 = time_dir / f"images_{b}x{b}_t{tb}.h5"
        r = dict(r)
        r["_out_h5"] = out_h5
        r["_tmp_h5"] = tmp_h5_path(out_h5)
        r["_time_dir"] = time_dir
        if out_h5.exists() and not bool(r.get("overwrite", False)):
            ok, msg, info = is_complete_bin_output(time_dir, out_h5)
            if ok and info:
                append_bin_manifest(channel_root, info)
                results.append(info)
            else:
                clean_incomplete_for_output(out_h5, remove_final=True)
                active.append(r)
        else:
            clean_incomplete_for_output(out_h5, remove_final=out_h5.exists())
            active.append(r)

    if not active:
        _call_progress(progress_cb, 1, 1, stage="done", message="all existing/skipped")
        return results

    filters = get_filters(compression_mode)
    open_outputs = []
    try:
        with h5py.File(source_h5, "r") as hf_in:
            if "entry/data/images" not in hf_in:
                raise KeyError(f"{source_h5}: /entry/data/images not found")
            d_in = hf_in["entry/data/images"]
            T, H, W = map(int, d_in.shape)
            in_dtype = d_in.dtype
            source_time_meta = get_h5_time_metadata(hf_in, fallback_texts=[str(source_h5)])
            raw_frame_time_sec = source_time_meta.get("raw_frame_time_sec")
            frame_time_source = source_time_meta.get("frame_time_source", "missing")
            max_val = d_in[: min(200, T)].max()
            out_base_dtype = decide_bin_out_dtype(in_dtype, max_val, out_dtype_mode=out_dtype_mode)

            raw_mask = None
            mask = None
            if "entry/instrument/detector/mask" in hf_in:
                raw_mask = hf_in["entry/instrument/detector/mask"][:]
                # canonical H5 should be 1=valid; keep rescue point for old data.
                mask = normalize_mask_array(raw_mask, "one_is_valid")

            prepared: List[Dict[str, Any]] = []
            for r in active:
                b = int(r["space_bin"])
                tb = int(r["time_bin"])
                H_eff = (H // b) * b
                W_eff = (W // b) * b
                T_eff = (T // tb) * tb
                if H_eff == 0 or W_eff == 0 or T_eff == 0:
                    raise RuntimeError(f"Invalid bin settings: b={b}, t={tb}, source shape={(T,H,W)}")
                H_bin, W_bin, T_bin = H_eff // b, W_eff // b, T_eff // tb
                mask_bin = spatial_bin_mask(mask, b, mode=mask_bin_mode, valid_ratio_threshold=valid_ratio_threshold) if mask is not None else None

                start_running_marker(r["_out_h5"], {
                    "operation": "fused_bin",
                    "dataset_id": dataset_id,
                    "channel": channel,
                    "source_h5": str(source_h5),
                    "output_h5": str(r["_out_h5"]),
                    "tmp_h5": str(r["_tmp_h5"]),
                    "space_bin": b,
                    "time_bin": tb,
                    "total": int(T_eff),
                })
                hf_out = h5py.File(r["_tmp_h5"], "w", libver="latest")
                open_outputs.append(hf_out)
                g_entry = hf_out.require_group("entry")
                g_data = g_entry.require_group("data")
                g_det = g_entry.require_group("instrument").require_group("detector")
                g_meta = g_entry.require_group("meta")
                d_out = g_data.create_dataset(
                    "images",
                    shape=(T_bin, H_bin, W_bin),
                    dtype=out_base_dtype,
                    chunks=(min(32, T_bin), H_bin, W_bin),
                    **filters,
                )
                if mask_bin is not None:
                    g_det.create_dataset("mask", data=mask_bin.astype(np.uint8), dtype=np.uint8)
                    g_det["mask"].attrs["mask_semantics"] = "1=valid,0=invalid"
                d_out.attrs["source_h5"] = str(source_h5)
                d_out.attrs["dataset_id"] = dataset_id
                d_out.attrs["channel"] = channel
                d_out.attrs["space_bin"] = b
                d_out.attrs["time_bin"] = tb
                time_meta = apply_time_metadata_to_images_dataset(d_out, raw_frame_time_sec, time_bin=tb, frame_time_source=frame_time_source)
                d_out.attrs["dtype_in"] = str(in_dtype)
                d_out.attrs["dtype_out"] = str(np.dtype(out_base_dtype))
                d_out.attrs["mask_input_semantics"] = "one_is_valid" if mask_bin is not None else "no_mask"
                d_out.attrs["mask_bin_mode"] = mask_bin_mode if mask_bin is not None else "no_mask"
                d_out.attrs["mask_semantics_output"] = "1=valid,0=invalid"
                d_out.attrs["created_at"] = now_iso()
                g_meta.create_dataset("software", data="DXB_00_Project_ImportBin_Manager_v5.6 fused bin", dtype=h5py.string_dtype("utf-8"))
                g_meta.create_dataset("created_at", data=now_iso(), dtype=h5py.string_dtype("utf-8"))
                write_time_metadata_to_meta_group(g_meta, time_meta)
                prepared.append(dict(r, _hf_out=hf_out, _d_out=d_out, _out_idx=0, _T_eff=T_eff, _H_eff=H_eff, _W_eff=W_eff, _H_bin=H_bin, _W_bin=W_bin, _T_bin=T_bin, _mask_bin=mask_bin, _time_meta=time_meta))

            common_T_eff = min(int(p["_T_eff"]) for p in prepared)
            # Fused pass supports same time_bin group. This is enforced by grouping key.
            chunk_size = max(1, int(time_bin) * int(chunk_factor))
            if chunk_size > common_T_eff:
                chunk_size = common_T_eff

            _call_progress(progress_cb, 0, common_T_eff, stage="start", message=f"{dataset_id}/{channel}: {len(prepared)} bins from one H5")
            for start in range(0, common_T_eff, chunk_size):
                end = min(start + chunk_size, common_T_eff)
                n = end - start
                block = d_in[start:end, :, :]
                for pinfo in prepared:
                    b = int(pinfo["space_bin"])
                    tb = int(pinfo["time_bin"])
                    H_eff = int(pinfo["_H_eff"])
                    W_eff = int(pinfo["_W_eff"])
                    H_bin = int(pinfo["_H_bin"])
                    W_bin = int(pinfo["_W_bin"])
                    # Since all tb are same within group, n is divisible except possibly final chunk.
                    groups = n // tb
                    if groups <= 0:
                        continue
                    use_n = groups * tb
                    sub = block[:use_n, :H_eff, :W_eff]
                    if b == 1:
                        block_sp = sub.astype(out_base_dtype, copy=False)
                    else:
                        block_sp = spatial_bin_block(sub, b, out_base_dtype)
                    block_tb = block_sp.reshape(groups, tb, H_bin, W_bin).sum(axis=1, dtype=out_base_dtype)
                    out_idx = int(pinfo["_out_idx"])
                    pinfo["_d_out"][out_idx: out_idx + groups] = block_tb
                    pinfo["_out_idx"] = out_idx + groups
                for pinfo in prepared:
                    update_running_marker(pinfo["_out_h5"], end, common_T_eff, stage="processing", message=f"fused {end}/{common_T_eff} frames")
                _call_progress(progress_cb, end, common_T_eff, stage="processing", message=f"{dataset_id}/{channel}: {len(prepared)} bins")

            # Close outputs before preview reads.
            for hf_out in list(open_outputs):
                try:
                    hf_out.close()
                except Exception:
                    pass
                open_outputs.remove(hf_out)

            for pinfo in prepared:
                update_running_marker(pinfo["_out_h5"], common_T_eff, common_T_eff, stage="commit", message="renaming tmp to final")
                commit_tmp_h5(pinfo["_tmp_h5"], pinfo["_out_h5"])

            for pinfo in prepared:
                b = int(pinfo["space_bin"])
                tb = int(pinfo["time_bin"])
                info = {
                    "schema_version": 6,
                    "status": "done_fused",
                    "dataset_id": dataset_id,
                    "channel": channel,
                    "source_h5": str(source_h5),
                    "output_h5": str(pinfo["_out_h5"]),
                    "space_bin": b,
                    "time_bin": tb,
                    "raw_frame_time_sec": pinfo["_time_meta"].get("raw_frame_time_sec"),
                    "effective_frame_time_sec": pinfo["_time_meta"].get("effective_frame_time_sec"),
                    "frame_time_source": pinfo["_time_meta"].get("frame_time_source"),
                    "shape_original": [int(T), int(H), int(W)],
                    "shape_binned": [int(pinfo["_T_bin"]), int(pinfo["_H_bin"]), int(pinfo["_W_bin"])],
                    "frames": int(pinfo["_T_bin"]),
                    "height": int(pinfo["_H_bin"]),
                    "width": int(pinfo["_W_bin"]),
                    "dtype_in": str(in_dtype),
                    "dtype_out": str(np.dtype(out_base_dtype)),
                    "filter": compression_label(compression_mode),
                    "out_dtype_mode": out_dtype_mode,
                    "mask_input_semantics": "one_is_valid" if pinfo["_mask_bin"] is not None else "no_mask",
                    "mask_bin_mode": mask_bin_mode if pinfo["_mask_bin"] is not None else "no_mask",
                    "valid_ratio_threshold": float(valid_ratio_threshold),
                    "mask_semantics_output": "1=valid,0=invalid",
                    "preview_mode": preview_mode,
                    "preview_written": bool(preview_mode != "none"),
                    "chunk_factor": int(chunk_factor),
                    "compression": compression_label(compression_mode),
                    "fused_pass": True,
                    "safety": "tmp_then_atomic_rename",
                    "created_at": now_iso(),
                }
                save_json(Path(pinfo["_time_dir"]) / "index.json", info)
                append_bin_manifest(channel_root, info)
                results.append(info)

            if preview_mode != "none":
                _call_progress(progress_cb, common_T_eff, common_T_eff, stage="preview", message="writing preview")
                for pinfo in prepared:
                    update_running_marker(pinfo["_out_h5"], common_T_eff, common_T_eff, stage="preview", message="writing preview")
                    write_preview(Path(pinfo["_out_h5"]), Path(pinfo["_time_dir"]) / "preview", preview_mode=preview_mode)

            for pinfo in prepared:
                finish_running_marker(pinfo["_out_h5"])
            _call_progress(progress_cb, common_T_eff, common_T_eff, stage="done", message=f"finished {len(prepared)} bins")
            return results
    finally:
        for hf_out in list(open_outputs):
            try:
                hf_out.close()
            except Exception:
                pass


def rebuild_project_catalog(analysis_root: Path) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    base = scan_analysis_root(analysis_root)
    for _, r in base.iterrows():
        rows.append({"row_type": "images_h5", "dataset_id": r["dataset_id"], "channel": r["channel"], "space_bin": "", "time_bin": "", "status": r["status"], "frames": r["frames"], "height": r["height"], "width": r["width"], "dtype": r["dtype"], "path": r["images_h5"]})
        for b in discover_existing_bins(Path(str(r["channel_root"]))):
            rows.append({"row_type": "bin", "dataset_id": r["dataset_id"], "channel": r["channel"], "space_bin": b.get("space_bin", ""), "time_bin": b.get("time_bin", ""), "status": b.get("status", ""), "frames": b.get("frames", ""), "height": b.get("height", ""), "width": b.get("width", ""), "dtype": b.get("dtype_out", ""), "path": b.get("output_h5", "")})
    df = pd.DataFrame(rows)
    if not df.empty:
        save_json(analysis_root / CATALOG_JSON, rows)
        df.to_csv(analysis_root / CATALOG_CSV, index=False, encoding="utf-8-sig")
    return df


# =============================================================================
# UI helpers
# =============================================================================

def inject_css():
    st.markdown("""
    <style>
    .block-container {padding-top: 1rem; padding-bottom: 2rem; max-width: 1500px;}
    div[data-testid="stMetric"] {background:#fafafa; border:1px solid #eee; padding:8px 10px; border-radius:12px;}
    .dxb-card {background:#fafafa; border:1px solid #ececec; border-radius:14px; padding:12px 14px; margin-bottom:0.6rem;}
    .muted {color:#666; font-size:0.9rem;}
    </style>
    """, unsafe_allow_html=True)


def _fmt_seconds(sec: float) -> str:
    if not np.isfinite(sec) or sec < 0:
        return "--:--"
    sec = int(sec)
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def progress_bar(label: str):
    """Streamlit progress monitor with FPS, ETA, phase and last update.

    The returned callback accepts both the old style cb(done, total) and
    the new style cb(done, total, stage=..., message=..., extra={...}).
    """
    title_slot = st.empty()
    bar = st.progress(0.0)
    metric_slot = st.empty()
    log_slot = st.empty()
    start_t = time.perf_counter()
    last_update = [start_t]
    last_done = [0]
    logs: List[str] = []

    def cb(done, total, stage: str = "running", message: str = "", extra: Optional[Dict[str, Any]] = None):
        now = time.perf_counter()
        total_i = max(1, int(total))
        done_i = max(0, min(int(done), total_i))
        critical_stage = str(stage).lower() in {"start", "scan", "commit", "preview", "done", "error", "failed", "index"}
        if (now - last_update[0]) < 0.25 and done_i < total_i and not critical_stage:
            return
        p = min(1.0, max(0.0, float(done_i) / float(total_i)))
        elapsed = now - start_t
        fps_total = done_i / elapsed if elapsed > 0 and done_i > 0 else 0.0
        eta = (total_i - done_i) / fps_total if fps_total > 0 else float("nan")
        delta_t = now - last_update[0]
        delta_n = done_i - last_done[0]
        fps_recent = delta_n / delta_t if delta_t > 0 and delta_n > 0 else fps_total
        title_slot.markdown(f"**{label}**  ·  `{stage}`  {message}")
        bar.progress(p)
        metric_slot.caption(
            f"frames {done_i:,} / {total_i:,}  |  {p*100:.1f}%  |  "
            f"speed {fps_recent:,.1f} fps  |  elapsed {_fmt_seconds(elapsed)}  |  ETA {_fmt_seconds(eta)}  |  "
            f"last update {datetime.now().strftime('%H:%M:%S')}"
        )
        if stage or message:
            line = f"[{datetime.now().strftime('%H:%M:%S')}] {stage}: {message} ({done_i}/{total_i})"
            if not logs or logs[-1] != line:
                logs.append(line)
                del logs[:-8]
                log_slot.code("\n".join(logs), language="text")
        last_update[0] = now
        last_done[0] = done_i
    return cb


def existing_overview(analysis_root: Path, refresh_token: int = 0):
    df = cached_scan_analysis_root(str(analysis_root), int(refresh_token))
    if df.empty:
        st.info("まだ canonical H5 が見つかりません。初回は Import H5 タブで作成してください。")
        return df
    c1, c2, c3 = st.columns(3)
    c1.metric("channels", len(df))
    c2.metric("ready H5", int((df["status"] == "ready").sum()))
    c3.metric("bins", int(df["n_bins"].sum()))
    show_cols = ["dataset_id", "channel", "frames", "height", "width", "dtype", "mask", "raw_frame_time_sec", "effective_frame_time_sec", "frame_time_source", "existing_bins", "status", "images_h5"]
    show_cols = [c for c in show_cols if c in df.columns]
    st.dataframe(df[show_cols], width="stretch", height=360)
    return df


def build_bin_queue_from_table(df: pd.DataFrame, space_bins: List[int], time_bins: List[int], mask_bin_mode: str, valid_ratio: float, bin_dtype_mode: str, preview_mode: str, overwrite: bool, chunk_factor: int, compression_mode: str, bin1_t1_mode: str = "alias") -> pd.DataFrame:
    rows = []
    if df is None or df.empty:
        return pd.DataFrame()
    selected = df[df.get("run", False) == True].copy()  # noqa: E712
    for _, r in selected.iterrows():
        images_h5 = Path(str(r["images_h5"]))
        channel_root = Path(str(r["channel_root"]))
        for b in space_bins:
            for t in time_bins:
                out_h5 = channel_root / "02_bin" / f"{b}x{b}" / f"t{t}" / f"images_{b}x{b}_t{t}.h5"
                status = "exists" if out_h5.exists() else "missing"
                rows.append({
                    "run": True,
                    "dataset_id": r.get("dataset_id", ""),
                    "channel": r.get("channel", "default"),
                    "images_h5": str(images_h5),
                    "channel_root": str(channel_root),
                    "space_bin": int(b),
                    "time_bin": int(t),
                    "mask_bin_mode": str(mask_bin_mode),
                    "valid_ratio": float(valid_ratio),
                    "bin_dtype_mode": str(bin_dtype_mode),
                    "preview_mode": str(preview_mode),
                    "chunk_factor": int(chunk_factor),
                    "compression_mode": str(compression_mode),
                    "bin1_t1_mode": str(bin1_t1_mode),
                    "overwrite": bool(overwrite),
                    "current_status": status,
                    "est_block_gb": 0.0 if (int(b) == 1 and int(t) == 1 and str(bin1_t1_mode).lower().startswith("alias")) else round(estimate_bin_memory_gb(images_h5, int(b), int(t), int(chunk_factor)), 3),
                    "output_h5": str(out_h5),
                })
    return pd.DataFrame(rows)


def run_bin_job_from_dict(r: Dict[str, Any]) -> Tuple[str, Optional[Dict[str, Any]], Optional[str]]:
    label = f"{r.get('dataset_id')} / {r.get('channel')} / {r.get('space_bin')}x{r.get('space_bin')} t{r.get('time_bin')}"
    try:
        info = bin_images_h5(
            Path(str(r["images_h5"])),
            Path(str(r["channel_root"])),
            str(r["dataset_id"]),
            str(r["channel"]),
            int(r["space_bin"]),
            int(r["time_bin"]),
            mask_input_semantics="one_is_valid",
            mask_bin_mode=str(r["mask_bin_mode"]),
            valid_ratio_threshold=float(r["valid_ratio"]),
            out_dtype_mode=str(r["bin_dtype_mode"]),
            chunk_factor=int(r.get("chunk_factor", 256)),
            preview_mode=str(r.get("preview_mode", "first_only")),
            overwrite=bool(r.get("overwrite", False)),
            compression_mode=str(r.get("compression_mode", "bitshuffle_lz4")),
            bin1_t1_mode=str(r.get("bin1_t1_mode", "alias")),
            progress_cb=None,
        )
        return label, info, None
    except Exception as e:
        return label, None, f"{type(e).__name__}: {e}"


def run_fused_group_job_from_list(group: List[Dict[str, Any]]) -> Tuple[str, Optional[List[Dict[str, Any]]], Optional[str]]:
    """Run one fused source-H5 group in a worker thread.

    Used by Execution mode = Auto when multiple independent source H5 groups are queued.
    It intentionally does not update Streamlit widgets from the worker thread; the UI gets
    job-level progress when each group finishes.
    """
    if not group:
        return "empty group", [], None
    bins_txt = ", ".join([f"{int(r['space_bin'])}x{int(r['space_bin'])}/t{int(r['time_bin'])}" for r in group])
    label = f"{group[0].get('dataset_id')} / {group[0].get('channel')} · {bins_txt}"
    try:
        infos = bin_images_h5_multi_same_source(group, progress_cb=None)
        return label, infos, None
    except Exception as e:
        return label, None, f"{type(e).__name__}: {e}"


def auto_perf_values() -> Dict[str, Any]:
    """Conservative auto settings for unknown PCs.

    Uses available CPU count only; memory/disk are intentionally not probed deeply.
    Users can override everything in Settings.
    """
    cpu = os.cpu_count() or 4
    if cpu >= 24:
        base = dict(SPEED_PRESETS["Fast NVMe"])
        base["parallel_jobs"] = 2  # safer default; fused pass is preferred over many parallel readers
        base["tif_workers"] = min(12, cpu)
    elif cpu >= 12:
        base = dict(SPEED_PRESETS["Balanced"])
        base["parallel_jobs"] = 2
        base["tif_workers"] = min(8, cpu)
    else:
        base = dict(SPEED_PRESETS["Safe"])
        base["parallel_jobs"] = 1
        base["tif_workers"] = min(4, cpu)
    return base


def default_perf() -> Dict[str, Any]:
    if "perf" not in st.session_state:
        st.session_state["perf"] = auto_perf_values()
    return dict(st.session_state["perf"])


def set_perf_from_preset(preset: str):
    if preset == "Auto":
        st.session_state["perf"] = auto_perf_values()
    elif preset in SPEED_PRESETS and preset != "Custom":
        vals = dict(SPEED_PRESETS[preset])
        cpu = os.cpu_count() or 4
        vals["tif_workers"] = min(int(vals["tif_workers"]), max(1, cpu))
        vals["parallel_jobs"] = min(int(vals["parallel_jobs"]), max(1, max(1, cpu // 2)))
        st.session_state["perf"] = vals

@st.cache_data(show_spinner=False)
def cached_scan_incomplete_outputs(analysis_root_str: str, refresh_token: int = 0) -> List[Dict[str, Any]]:
    return scan_incomplete_outputs(Path(clean_path_text(analysis_root_str)))


@st.cache_data(show_spinner=False)
def cached_scan_analysis_root(analysis_root_str: str, refresh_token: int = 0) -> pd.DataFrame:
    return scan_analysis_root(Path(clean_path_text(analysis_root_str)))


@st.cache_data(show_spinner=False)
def cached_scan_source_root(source_root_str: str, scan_mode: str, max_depth: int,
                            raw_binary_width: int = RAW_BINARY_DEFAULT_WIDTH,
                            raw_binary_height: int = RAW_BINARY_DEFAULT_HEIGHT,
                            raw_binary_dtype: str = RAW_BINARY_DEFAULT_DTYPE,
                            raw_binary_offset_bytes: int = RAW_BINARY_DEFAULT_OFFSET_BYTES,
                            include_raw_binary: bool = False,
                            refresh_token: int = 0) -> pd.DataFrame:
    return scan_source_root(
        Path(clean_path_text(source_root_str)), scan_mode=scan_mode, max_depth=int(max_depth),
        raw_binary_width=int(raw_binary_width), raw_binary_height=int(raw_binary_height),
        raw_binary_dtype=str(raw_binary_dtype), raw_binary_offset_bytes=int(raw_binary_offset_bytes),
        include_raw_binary=bool(include_raw_binary),
    )




# =============================================================================
# Background worker jobs (v5.8)
# =============================================================================
class WorkerCancelled(RuntimeError):
    pass


def _json_safe(obj):
    """Convert pandas/numpy/path objects to JSON-safe Python objects."""
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return None if not np.isfinite(v) else v
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, float):
        return None if not np.isfinite(obj) else obj
    if pd.isna(obj):
        return None
    return obj


def _job_root(analysis_root: Path) -> Path:
    return Path(analysis_root) / ".dxb_jobs"


def _read_job_progress(job_dir: Path) -> Dict[str, Any]:
    return load_json(Path(job_dir) / "progress.json", default={})


def _write_job_progress(job_dir: Path, payload: Dict[str, Any]):
    payload = dict(payload or {})
    payload["updated_at"] = now_iso()
    save_json(Path(job_dir) / "progress.json", payload)


def _append_job_log(job_dir: Path, line: str, max_lines: int = 80):
    p = Path(job_dir) / "worker_events.log"
    try:
        old = p.read_text(encoding="utf-8").splitlines() if p.exists() else []
        old.append(f"[{datetime.now().strftime('%H:%M:%S')}] {line}")
        p.write_text("\n".join(old[-max_lines:]) + "\n", encoding="utf-8")
    except Exception:
        pass


def _job_cancel_requested(job_dir: Path) -> bool:
    return (Path(job_dir) / "cancel.flag").exists()


def _make_worker_progress_cb(job_dir: Path, item_label: str, item_index: int, item_total: int):
    last_written = {"current": None, "stage": None, "message": None}
    def _cb(current, total, stage="running", message=""):
        # Stop is cooperative: checked only at completed sets/blocks.
        if _job_cancel_requested(job_dir):
            raise WorkerCancelled("Stop requested")
        cur = int(current or 0)
        tot = int(total or 0)
        msg = str(message or "")
        stg = str(stage or "running")
        # Coarse writes only: write if current/stage/message changed.
        if (last_written["current"], last_written["stage"], last_written["message"]) == (cur, stg, msg):
            return
        last_written.update({"current": cur, "stage": stg, "message": msg})
        _write_job_progress(job_dir, {
            "status": "running",
            "stage": stg,
            "message": msg,
            "current": cur,
            "total": tot,
            "item_label": item_label,
            "item_index": int(item_index),
            "item_total": int(item_total),
        })
        if msg:
            _append_job_log(job_dir, msg)
    return _cb


def _worker_run_import_job(job_json_path: str) -> int:
    job_path = Path(job_json_path)
    job_dir = job_path.parent
    job = load_json(job_path, default={})
    current_out_h5 = None
    try:
        rows = job.get("rows", []) or []
        analysis_root = Path(job["analysis_root"])
        settings = job.get("settings", {}) or {}
        _write_job_progress(job_dir, {
            "status": "running",
            "stage": "start",
            "message": f"Starting import job: {len(rows)} item(s)",
            "current": 0,
            "total": 0,
            "item_index": 0,
            "item_total": len(rows),
        })
        _append_job_log(job_dir, f"worker started pid={os.getpid()}")

        for idx, row_dict in enumerate(rows, start=1):
            if _job_cancel_requested(job_dir):
                raise WorkerCancelled("Stop requested before next item")
            row = pd.Series(row_dict)
            dataset_id = safe_name(row.get("dataset_id", "dataset"), "dataset")
            channel = safe_name(row.get("channel", "default"), "default")
            item_label = f"{dataset_id} / {channel} / {row.get('source_type', '')}"
            current_out_h5 = canonical_h5_path(analysis_root, dataset_id, channel)
            _write_job_progress(job_dir, {
                "status": "running",
                "stage": "item_start",
                "message": item_label,
                "current": 0,
                "total": int(row.get("frames") or 0),
                "item_label": item_label,
                "item_index": idx,
                "item_total": len(rows),
            })
            _append_job_log(job_dir, f"start {item_label}")
            cb = _make_worker_progress_cb(job_dir, item_label, idx, len(rows))
            info = ensure_canonical_h5(
                row=row,
                analysis_root=analysis_root,
                import_mode=str(settings.get("import_mode", "Build if missing")),
                pixel_dtype_mode=str(settings.get("pixel_dtype_mode", "force_uint16")),
                tif_mask_semantics=str(settings.get("tif_mask_semantics", "one_is_valid")),
                eiger_dtype_mode=str(settings.get("eiger_dtype_mode", "keep")),
                target_chunk_mib=float(settings.get("target_chunk_mib", 64.0)),
                batch_frames_tif=int(settings.get("batch_frames_tif", 64)),
                batch_frames_eiger=int(settings.get("batch_frames_eiger", 512)),
                tif_read_workers=int(settings.get("tif_read_workers", 4)),
                compression_mode=str(settings.get("compression_mode", "bitshuffle_lz4")),
                progress_cb=cb,
                default_raw_frame_time_sec=_to_optional_positive_float(settings.get("default_raw_frame_time_sec", None)),
                raw_binary_width=int(settings.get("raw_binary_width", RAW_BINARY_DEFAULT_WIDTH)),
                raw_binary_height=int(settings.get("raw_binary_height", RAW_BINARY_DEFAULT_HEIGHT)),
                raw_binary_dtype=str(settings.get("raw_binary_dtype", RAW_BINARY_DEFAULT_DTYPE)),
                raw_binary_offset_bytes=int(settings.get("raw_binary_offset_bytes", RAW_BINARY_DEFAULT_OFFSET_BYTES)),
            )
            _append_job_log(job_dir, f"done {item_label}: {info.get('output_h5', '')}")
            _write_job_progress(job_dir, {
                "status": "running",
                "stage": "item_done",
                "message": f"done: {item_label}",
                "current": int(info.get("frames") or 0),
                "total": int(info.get("frames") or 0),
                "item_label": item_label,
                "item_index": idx,
                "item_total": len(rows),
            })

        rebuild_project_catalog(analysis_root)
        _write_job_progress(job_dir, {
            "status": "done",
            "stage": "done",
            "message": "Import job completed",
            "current": len(rows),
            "total": len(rows),
            "item_index": len(rows),
            "item_total": len(rows),
            "finished_at": now_iso(),
        })
        _append_job_log(job_dir, "worker done")
        return 0
    except WorkerCancelled as e:
        if current_out_h5 is not None:
            try:
                clean_incomplete_for_output(Path(current_out_h5), remove_final=False)
                _append_job_log(job_dir, f"cleaned incomplete: {current_out_h5}")
            except Exception as ce:
                _append_job_log(job_dir, f"cleanup failed: {type(ce).__name__}: {ce}")
        _write_job_progress(job_dir, {"status": "cancelled", "stage": "cancelled", "message": str(e), "finished_at": now_iso()})
        _append_job_log(job_dir, "worker cancelled")
        return 2
    except Exception as e:
        _write_job_progress(job_dir, {"status": "error", "stage": "error", "message": f"{type(e).__name__}: {e}", "finished_at": now_iso()})
        _append_job_log(job_dir, f"worker error: {type(e).__name__}: {e}")
        return 1


def start_import_worker_job(analysis_root: Path, rows: list[dict[str, Any]], settings: Dict[str, Any]) -> Path:
    root = _job_root(Path(analysis_root))
    ensure_dir(root)
    job_id = datetime.now().strftime("import_%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:8]
    job_dir = root / job_id
    ensure_dir(job_dir)
    job = {
        "job_id": job_id,
        "job_type": "import",
        "created_at": now_iso(),
        "analysis_root": str(analysis_root),
        "rows": rows,
        "settings": settings,
        "script": str(Path(__file__).resolve()),
    }
    job_path = job_dir / "job.json"
    save_json(job_path, job)
    _write_job_progress(job_dir, {"status": "queued", "stage": "queued", "message": "queued", "item_total": len(rows)})

    cmd = [sys.executable, str(Path(__file__).resolve()), "--dxb-worker", str(job_path)]
    popen_kwargs = {}
    if os.name == "nt":
        # Keep the worker in a separate process group so Streamlit stays responsive.
        popen_kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    stdout_path = job_dir / "worker_stdout.log"
    stderr_path = job_dir / "worker_stderr.log"
    with open(stdout_path, "ab") as out, open(stderr_path, "ab") as err:
        proc = subprocess.Popen(cmd, stdout=out, stderr=err, **popen_kwargs)
    save_json(job_dir / "pid.json", {"pid": int(proc.pid), "started_at": now_iso(), "cmd": cmd})
    _append_job_log(job_dir, f"started worker pid={proc.pid}")
    return job_dir


def render_import_worker_panel(analysis_root: Optional[Path]):
    if not analysis_root:
        return
    job_dir_txt = st.session_state.get("active_import_job_dir", "")
    if not job_dir_txt:
        return
    job_dir = Path(job_dir_txt)
    if not job_dir.exists():
        st.session_state.pop("active_import_job_dir", None)
        return
    prog = _read_job_progress(job_dir)
    status = str(prog.get("status", "unknown"))
    st.markdown("### Import worker")
    st.caption(str(job_dir))
    c1, c2, c3, c4 = st.columns([1, 1, 1, 2])
    with c1:
        st.metric("status", status)
    with c2:
        st.metric("item", f"{prog.get('item_index', 0)}/{prog.get('item_total', 0)}")
    with c3:
        current = int(prog.get("current") or 0)
        total = int(prog.get("total") or 0)
        st.metric("frames", f"{current}/{total}" if total else str(current))
    with c4:
        st.write(prog.get("message", ""))
    total = int(prog.get("total") or 0)
    current = int(prog.get("current") or 0)
    if total > 0:
        st.progress(min(1.0, max(0.0, current / total)))
    cols = st.columns([1, 1, 1, 3])
    with cols[0]:
        if st.button("Refresh worker", width="stretch", key="worker_refresh_btn"):
            pass
    with cols[1]:
        if status in ["queued", "running"] and st.button("Stop after set", width="stretch", type="secondary", key="worker_stop_btn"):
            (job_dir / "cancel.flag").write_text(now_iso(), encoding="utf-8")
            st.warning("Stop requested. It will stop after the current block/set finishes.")
    with cols[2]:
        if status not in ["queued", "running"] and st.button("Clear worker", width="stretch", key="worker_clear_btn"):
            st.session_state.pop("active_import_job_dir", None)
    with cols[3]:
        st.caption("進捗更新はEIGER block / TIF・RAW batch完了ごとだけです。")
    logp = job_dir / "worker_events.log"
    if logp.exists():
        with st.expander("Worker log", expanded=False):
            st.code(logp.read_text(encoding="utf-8", errors="replace")[-5000:])


def bump_project_refresh_tokens():
    for key in ["overview_refresh_token", "analysis_scan_refresh_token", "catalog_refresh_token"]:
        st.session_state[key] = int(st.session_state.get(key, 0)) + 1


def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    inject_css()
    st.title(APP_TITLE)
    st.caption("初回はImport H5だけ。Binはあとから同じ画面でPackage作成・実行できます。Auto/Fused/Parallel実行に対応しました。")

    with st.sidebar:
        st.header("Project")
        source_root_txt = st.text_area("Source root", value=st.session_state.get("source_root_txt", ""), height=70, placeholder=r"F:\20260417_SPring-8_BL40XU", key="sidebar_source_root")
        analysis_root_txt = st.text_area("Analysis root", value=st.session_state.get("analysis_root_txt", ""), height=70, placeholder=r"D:\20260417_BL40XU_解析", key="sidebar_analysis_root")
        st.session_state["source_root_txt"] = source_root_txt
        st.session_state["analysis_root_txt"] = analysis_root_txt
        st.divider()
        st.caption("左バーはProject選択だけ。細かい設定は各タブ内にあります。")

    source_root = Path(clean_path_text(source_root_txt)) if clean_path_text(source_root_txt) else None
    analysis_root = Path(clean_path_text(analysis_root_txt)) if clean_path_text(analysis_root_txt) else None

    # Performance settings must be initialized before Import/Bin tabs use `perf`.
    # In v5 this was only initialized in the Settings tab, so opening Import H5 first
    # caused: NameError: name 'perf' is not defined.
    perf = default_perf()
    for _tok in ["overview_refresh_token", "analysis_scan_refresh_token", "raw_scan_refresh_token", "catalog_refresh_token"]:
        st.session_state.setdefault(_tok, 0)

    tab_overview, tab_import, tab_bin, tab_catalog, tab_settings = st.tabs(["Overview", "Import H5", "Bin Manager", "Catalog", "Settings"])

    with tab_overview:
        st.subheader("Project overview")
        if not analysis_root:
            st.info("Analysis root を指定してください。")
        else:
            existing_overview(analysis_root, st.session_state.get("analysis_scan_refresh_token", 0))
            st.markdown("---")
            st.subheader("Safety / resume")
            incomplete = cached_scan_incomplete_outputs(str(analysis_root), st.session_state.get("overview_refresh_token", 0))
            if incomplete:
                st.warning(f"Incomplete / failed outputs detected: {len(incomplete)}")
                inc_df = pd.DataFrame(incomplete)
                st.dataframe(inc_df, width="stretch", hide_index=True)
                c_clean, c_note = st.columns([1, 3])
                with c_clean:
                    if st.button("Clean incomplete", type="secondary", width="stretch", key="overview_clean_incomplete"):
                        removed = 0
                        for row in incomplete:
                            final_h5 = Path(row.get("final_h5", ""))
                            if final_h5:
                                clean_incomplete_for_output(final_h5, remove_final=(row.get("kind") == "h5_without_valid_index"))
                                removed += 1
                        st.session_state["overview_refresh_token"] = int(st.session_state.get("overview_refresh_token", 0)) + 1
                        st.session_state["analysis_scan_refresh_token"] = int(st.session_state.get("analysis_scan_refresh_token", 0)) + 1
                        st.success(f"Cleaned {removed} incomplete item(s). Press Clean again only if items remain.")
                with c_note:
                    st.caption(".tmp / .running.json / .failed.json を検出します。index.jsonが不完全なBin H5も検出します。")
            else:
                st.success("No incomplete outputs detected.")

    with tab_import:
        st.subheader("Import H5 - 初回だけ使う")
        st.markdown("""
        <div class="dxb-card">
        Rawフォルダを調査し、各 dataset/channel に対して canonical <code>01_h5/images.h5</code> を作成します。<br>
        ここではBinを作らなくてもOKです。後日 <b>Bin</b> タブから追加できます。
        </div>
        """, unsafe_allow_html=True)

        render_import_worker_panel(analysis_root)

        scan_c1, scan_c2, scan_c3, scan_c4 = st.columns([1, 1, 1, 2])
        with scan_c1:
            scan_mode = st.selectbox("Scan mode", SCAN_MODES, index=1, key="import_scan_mode")
        with scan_c2:
            max_depth = st.number_input("Max depth", min_value=1, max_value=12, value=5, step=1, key="import_max_depth")
        with scan_c3:
            scan_btn = st.button("Scan raw", type="primary", width="stretch", key="import_scan_raw_btn")
        with scan_c4:
            st.caption("dataset直下のサブフォルダ名は data_label として推定します。推定結果は表で修正できます。")

        raw_c1, raw_c2, raw_c3, raw_c4 = st.columns([1, 1, 1, 1])
        with raw_c1:
            raw_binary_width = st.number_input("RAW width", min_value=1, max_value=20000, value=RAW_BINARY_DEFAULT_WIDTH, step=1, key="import_raw_binary_width")
        with raw_c2:
            raw_binary_height = st.number_input("RAW height", min_value=1, max_value=20000, value=RAW_BINARY_DEFAULT_HEIGHT, step=1, key="import_raw_binary_height")
        with raw_c3:
            raw_binary_dtype = st.selectbox("RAW dtype", RAW_BINARY_DTYPE_OPTIONS, index=RAW_BINARY_DTYPE_OPTIONS.index(RAW_BINARY_DEFAULT_DTYPE), key="import_raw_binary_dtype")
        with raw_c4:
            raw_binary_offset_bytes = st.number_input("RAW offset [bytes]", min_value=0, max_value=1048576, value=RAW_BINARY_DEFAULT_OFFSET_BYTES, step=1, key="import_raw_binary_offset_bytes")
        raw_scan_c1, raw_scan_c2 = st.columns([1, 3])
        with raw_scan_c1:
            include_raw_binary_scan = st.checkbox("Include RAW in recursive scan", value=False, key="import_include_raw_binary_scan")
        with raw_scan_c2:
            st.caption("推奨: OFF。RAWは拡張子がなく自動判定が重いため、下の Manual RAW folder で1フォルダだけ追加する方が安全です。")

        manual_raw_c1, manual_raw_c2 = st.columns([3, 1])
        with manual_raw_c1:
            manual_raw_folder_txt = st.text_input("Manual RAW folder", value=st.session_state.get("manual_raw_folder_txt", ""), placeholder=r"F:\...\raw_frames", key="manual_raw_folder_input")
            st.session_state["manual_raw_folder_txt"] = manual_raw_folder_txt
        with manual_raw_c2:
            add_manual_raw_btn = st.button("Add RAW folder", type="secondary", width="stretch", key="add_manual_raw_folder_btn")
        if add_manual_raw_btn:
            raw_dir = Path(clean_path_text(manual_raw_folder_txt))
            if not raw_dir.exists() or not raw_dir.is_dir():
                st.error("Manual RAW folder が見つかりません。")
            else:
                raw_paths = sorted_raw_binary_paths(raw_dir, int(raw_binary_width), int(raw_binary_height), str(raw_binary_dtype), int(raw_binary_offset_bytes))
                if not raw_paths:
                    expected = raw_binary_expected_bytes(int(raw_binary_width), int(raw_binary_height), str(raw_binary_dtype), int(raw_binary_offset_bytes))
                    st.error(f"RAW候補が見つかりません。expected file size = {expected} bytes")
                else:
                    dataset_guess = safe_name(raw_dir.parent.name or raw_dir.name, "dataset")
                    channel_guess = infer_channel(raw_dir.parent if raw_dir.parent.exists() else raw_dir, raw_dir)
                    row = {
                        "run": True,
                        "dataset_id": dataset_guess,
                        "channel": channel_guess,
                        "candidate_rank": 1,
                        "score": 99,
                        "source_type": "raw_binary_stack",
                        "primary_path": str(raw_dir),
                        "selected_sources": str(raw_dir),
                        "mask_path": "",
                        "frames": int(len(raw_paths)),
                        "height": int(raw_binary_height),
                        "width": int(raw_binary_width),
                        "dtype": str(raw_binary_dtype),
                        "notes": f"manual RAW folder; expected_file_size={raw_binary_expected_bytes(int(raw_binary_width), int(raw_binary_height), str(raw_binary_dtype), int(raw_binary_offset_bytes))} bytes; offset={int(raw_binary_offset_bytes)}",
                        "source_root": str(raw_dir.parent),
                        "raw_frame_time_sec": None,
                    }
                    old = st.session_state.get("scan_df")
                    if old is None or getattr(old, "empty", True):
                        st.session_state["scan_df"] = pd.DataFrame([row])
                    else:
                        st.session_state["scan_df"] = pd.concat([old, pd.DataFrame([row])], ignore_index=True)
                    st.success(f"RAW folder added: {len(raw_paths)} frames")

        if scan_btn:
            if not source_root or not source_root.exists():
                st.error("Source root が見つかりません。")
            else:
                st.session_state["raw_scan_refresh_token"] = int(st.session_state.get("raw_scan_refresh_token", 0)) + 1
                with st.spinner("Scanning raw source..."):
                    df = cached_scan_source_root(
                        str(source_root), scan_mode=scan_mode, max_depth=int(max_depth),
                        raw_binary_width=int(raw_binary_width), raw_binary_height=int(raw_binary_height),
                        raw_binary_dtype=str(raw_binary_dtype), raw_binary_offset_bytes=int(raw_binary_offset_bytes),
                        include_raw_binary=bool(include_raw_binary_scan),
                        refresh_token=st.session_state.get("raw_scan_refresh_token", 0),
                    )
                st.session_state["scan_df"] = df
                st.success(f"Detected {len(df)} candidates")

        df0 = st.session_state.get("scan_df")
        if df0 is not None and not df0.empty:
            if "raw_frame_time_sec" not in df0.columns:
                df0 = df0.copy()
                df0["raw_frame_time_sec"] = None
            st.markdown("### Candidates")
            st.caption("run, dataset_id, channel, selected_sources, mask_path, raw_frame_time_sec は必要に応じて編集してください。raw_frame_time_sec は例: 250ms→0.25, 500ms→0.50。")
            edited = st.data_editor(
                df0,
                width="stretch",
                height=360,
                column_config={
                    "run": st.column_config.CheckboxColumn("run"),
                    "dataset_id": st.column_config.TextColumn("dataset"),
                    "channel": st.column_config.TextColumn("channel"),
                    "raw_frame_time_sec": st.column_config.NumberColumn("raw_frame_time_sec [s]", min_value=0.0, step=0.001, format="%.6f"),
                    "selected_sources": st.column_config.TextColumn("selected_sources", width="large"),
                    "primary_path": st.column_config.TextColumn("primary_path", width="large"),
                    "mask_path": st.column_config.TextColumn("mask_path", width="large"),
                    "notes": st.column_config.TextColumn("notes", width="medium"),
                },
                disabled=["candidate_rank", "score", "source_type", "frames", "height", "width", "dtype", "source_root"],
                key="scan_editor_v3",
            )
            st.session_state["scan_edited"] = edited

            with st.expander("Import settings", expanded=True):
                a1, a2, a3, a4 = st.columns(4)
                with a1:
                    import_mode = st.selectbox("H5 mode", IMPORT_MODES, index=0, key="import_h5_mode")
                with a2:
                    pixel_dtype_mode = st.selectbox("TIF/RAW dtype", ["force_uint16", "auto_uint", "force_uint8", "keep"], index=0, key="import_tif_dtype")
                with a3:
                    tif_mask_semantics = st.selectbox("TIF mask", ["one_is_valid", "zero_is_valid"], index=0, help="one_is_valid: 白/非0がvalid。zero_is_valid: 黒/0がvalid。canonical H5では1=validに統一します。", key="import_tif_mask_semantics")
                with a4:
                    eiger_dtype_mode = st.selectbox("EIGER dtype", ["keep", "uint16"], index=0, key="import_eiger_dtype")
                b1, b2, b3, b4 = st.columns(4)
                with b1:
                    target_chunk_mib = st.number_input("H5 chunk MiB", min_value=1.0, max_value=1024.0, value=float(perf.get("h5_chunk_mib", 64.0)), step=8.0, key="import_h5_chunk_mib")
                with b2:
                    batch_frames_tif = st.number_input("TIF/RAW batch", min_value=1, max_value=2048, value=int(perf.get("tif_batch", 64)), step=1, key="import_tif_batch")
                with b3:
                    batch_frames_eiger = st.number_input("EIGER batch", min_value=1, max_value=8192, value=int(perf.get("eiger_batch", 512)), step=64, key="import_eiger_batch")
                with b4:
                    tif_read_workers = st.number_input("TIF/RAW workers", min_value=1, max_value=64, value=int(perf.get("tif_workers", min(8, os.cpu_count() or 8))), step=1, help="TIF/RAW読み込みだけスレッド並列化。HDF5書き込みは1 writerです。", key="import_tif_workers")
                t1, t2 = st.columns([1, 3])
                with t1:
                    default_raw_frame_time_sec_txt = st.text_input("Default raw frame time [s]", value="", placeholder="例: 0.25 / 0.50", key="import_default_raw_frame_time_sec")
                with t2:
                    st.caption("候補表の raw_frame_time_sec が空の行だけ、この値を使います。空欄ならパス中の 250ms/500ms を自動推定し、見つからなければ missing として保存します。")

            cimp1, cimp2 = st.columns([1, 3])
            with cimp1:
                import_compression_mode = st.selectbox("Import compression", COMPRESSION_MODES, index=COMPRESSION_MODES.index(str(perf.get("compression_mode", "bitshuffle_lz4"))) if str(perf.get("compression_mode", "bitshuffle_lz4")) in COMPRESSION_MODES else 0, key="import_compression")
            with cimp2:
                st.caption("速度優先なら none。保存容量とのバランスなら bitshuffle_lz4。別PCでは Settings タブで変更できます。")

            run_import = st.button("Start import worker", type="primary", width="stretch", key="import_selected_h5_btn")
            if run_import:
                if not analysis_root:
                    st.error("Analysis root を指定してください。")
                else:
                    selected = edited[edited["run"] == True].copy()  # noqa: E712
                    if selected.empty:
                        st.warning("run=True の候補がありません。")
                    else:
                        # Launch heavy import in a separate worker process. Streamlit only monitors progress.json.
                        rows = [
                            {str(k): _json_safe(v) for k, v in row.items()}
                            for row in selected.to_dict(orient="records")
                        ]
                        settings = {
                            "import_mode": str(import_mode),
                            "pixel_dtype_mode": str(pixel_dtype_mode),
                            "tif_mask_semantics": str(tif_mask_semantics),
                            "eiger_dtype_mode": str(eiger_dtype_mode),
                            "target_chunk_mib": float(target_chunk_mib),
                            "batch_frames_tif": int(batch_frames_tif),
                            "batch_frames_eiger": int(batch_frames_eiger),
                            "tif_read_workers": int(tif_read_workers),
                            "compression_mode": str(import_compression_mode),
                            "default_raw_frame_time_sec": _to_optional_positive_float(default_raw_frame_time_sec_txt),
                            "raw_binary_width": int(raw_binary_width),
                            "raw_binary_height": int(raw_binary_height),
                            "raw_binary_dtype": str(raw_binary_dtype),
                            "raw_binary_offset_bytes": int(raw_binary_offset_bytes),
                        }
                        try:
                            job_dir = start_import_worker_job(Path(analysis_root), rows, settings)
                            st.session_state["active_import_job_dir"] = str(job_dir)
                            st.success(f"Import worker started: {job_dir.name}")
                            st.info("進捗はこのタブの Import worker で確認してください。Stop after set は現在のblock/batch完了後に止まります。")
                        except Exception as e:
                            st.error(f"worker start failed: {type(e).__name__}: {e}")
        else:
            st.info("Source root を指定して Scan raw を押してください。")

    with tab_bin:
        st.subheader("Bin - 後から追加・再作成")
        if not analysis_root:
            st.info("Analysis root を指定してください。")
        else:
            df = cached_scan_analysis_root(str(analysis_root), st.session_state.get("analysis_scan_refresh_token", 0))
            if df.empty:
                st.info("canonical 01_h5/images.h5 がありません。先にImport H5を実行してください。")
            else:
                st.markdown("### Select H5 channels")
                df_sel = df.copy()
                # useful default: select all ready rows
                df_sel["run"] = df_sel["status"].eq("ready")
                edited_h5 = st.data_editor(
                    df_sel,
                    width="stretch",
                    height=300,
                    column_config={"run": st.column_config.CheckboxColumn("run")},
                    disabled=[c for c in df_sel.columns if c != "run"],
                    key="h5_bin_select_v3",
                )
                st.markdown("### Bin recipe")
                preset = st.radio("Package", ["Package 1", "Package 2", "Package 3", "Custom"], horizontal=True, index=0, key="bin_preset")
                if preset == "Package 1":
                    default_space, default_time = "8,10,12", "1"
                elif preset == "Package 2":
                    default_space, default_time = "5", "1"
                elif preset == "Package 3":
                    default_space, default_time = "1", "1"
                else:
                    default_space, default_time = "8", "1"
                r1, r2, r3, r4 = st.columns(4)
                with r1:
                    space_bins_txt = st.text_input("Space bins", value=default_space, key="bin_space_bins")
                with r2:
                    time_bins_txt = st.text_input("Time bins", value=default_time, key="bin_time_bins")
                with r3:
                    mask_bin_mode = st.selectbox("Mask bin", ["all_valid", "any_valid", "valid_ratio"], index=0, key="bin_mask_mode")
                with r4:
                    valid_ratio = st.number_input("Valid ratio", min_value=0.0, max_value=1.0, value=0.5, step=0.05, key="bin_valid_ratio")
                s1, s2, s3, s4 = st.columns(4)
                with s1:
                    bin_dtype_mode = st.selectbox("Bin dtype", ["auto", "same"], index=0, key="bin_dtype_mode")
                with s2:
                    preview_mode = st.selectbox("Preview", PREVIEW_MODES, index=PREVIEW_MODES.index(str(perf.get("preview_mode", "first_only"))) if str(perf.get("preview_mode", "first_only")) in PREVIEW_MODES else 1, key="bin_preview_mode")
                with s3:
                    bin_chunk_factor = st.number_input("Chunk factor", min_value=16, max_value=4096, value=int(perf.get("bin_chunk_factor", 512)), step=16, help="大きいほど速いことがありますが一時メモリが増えます。", key="bin_chunk_factor")
                with s4:
                    bin_compression_mode = st.selectbox("Compression", COMPRESSION_MODES, index=COMPRESSION_MODES.index(str(perf.get("compression_mode", "bitshuffle_lz4"))) if str(perf.get("compression_mode", "bitshuffle_lz4")) in COMPRESSION_MODES else 0, key="bin_compression")
                s5, _s6 = st.columns([1, 3])
                with s5:
                    bin1_t1_mode = st.selectbox("1x1/t1", ["alias", "copy", "skip"], index=0, help="alias: 画像をコピーせず 01_h5/images.h5 へのExternalLinkを作ります。copy: 従来通り丸ごとコピー。skip: 1x1/t1は作りません。", key="bin1_t1_mode")
                t1, t2 = st.columns([1, 3])
                with t1:
                    overwrite_default = st.checkbox("overwrite default", value=False, key="bin_overwrite_default")
                with t2:
                    add_queue = st.button("Add to queue", type="primary", width="stretch", key="bin_add_queue_btn")
                if add_queue:
                    try:
                        q = build_bin_queue_from_table(edited_h5, parse_int_list(space_bins_txt), parse_int_list(time_bins_txt), mask_bin_mode, float(valid_ratio), bin_dtype_mode, str(preview_mode), bool(overwrite_default), int(bin_chunk_factor), str(bin_compression_mode), str(bin1_t1_mode))
                        if q.empty:
                            st.warning("Queueに追加できる行がありません。")
                        else:
                            if "bin_queue" in st.session_state and isinstance(st.session_state["bin_queue"], pd.DataFrame):
                                st.session_state["bin_queue"] = pd.concat([st.session_state["bin_queue"], q], ignore_index=True).drop_duplicates(subset=["images_h5", "space_bin", "time_bin", "mask_bin_mode"], keep="last")
                            else:
                                st.session_state["bin_queue"] = q
                            st.success(f"Added {len(q)} recipes to queue")
                    except Exception as e:
                        st.error(f"Queue作成に失敗: {e}")

        st.markdown("---")
        st.subheader("Queue / Run")
        q = st.session_state.get("bin_queue")
        if q is None or q.empty:
            st.info("上の Package / recipe を選んで Add to queue を押してください。")
        else:
            st.caption("Queueはこの画面内で最終確認できます。overwrite列だけ必要なら変更してください。")
            edited_q = st.data_editor(
                q,
                width="stretch",
                height=300,
                column_config={"run": st.column_config.CheckboxColumn("run"), "overwrite": st.column_config.CheckboxColumn("overwrite")},
                disabled=[c for c in q.columns if c not in ["run", "overwrite"]],
                key="queue_editor_v4_unified",
            )
            st.session_state["bin_queue"] = edited_q
            selected_count = int((edited_q["run"] == True).sum()) if "run" in edited_q else 0  # noqa: E712
            source_count = int(edited_q[edited_q.get("run", False) == True]["images_h5"].nunique()) if selected_count else 0  # noqa: E712
            m1, m2, m3 = st.columns(3)
            m1.metric("selected recipes", selected_count)
            m2.metric("source H5 groups", source_count)
            m3.metric("queue rows", len(edited_q))

            c1, c2, c3, c4 = st.columns([1, 1, 1.2, 1.8])
            with c1:
                run_queue = st.button("Run selected", type="primary", width="stretch", key="run_selected_btn_v4")
            with c2:
                clear_queue = st.button("Clear queue", width="stretch", key="run_clear_queue_btn_v4")
            with c3:
                parallel_jobs = st.number_input("Parallel jobs", min_value=1, max_value=8, value=int(perf.get("parallel_jobs", 1)), step=1, help="Auto/Parallel modeで使います。別source H5のグループを並列化します。", key="run_parallel_jobs_v4")
            with c4:
                execution_mode = st.selectbox(
                    "Execution mode",
                    ["Auto", "Fused detailed", "Parallel jobs"],
                    index=0,
                    help="Auto: 同じsourceはfused、sourceが複数あればジョブ並列。Fused detailed: 詳細frame進捗を優先して順番処理。Parallel jobs: recipe単位で並列、進捗は簡易。",
                    key="run_execution_mode_v4",
                )
                st.caption("Rawは読みません。01_h5/images.h5 からBinします。")
            if clear_queue:
                st.session_state.pop("bin_queue", None)
                st.rerun()
            if run_queue:
                run_df = edited_q[edited_q["run"] == True].copy()  # noqa: E712
                if run_df.empty:
                    st.warning("run=True のrecipeがありません。")
                else:
                    jobs = [dict(r) for _, r in run_df.iterrows()]
                    run_start = time.perf_counter()
                    summary_rows = []

                    if execution_mode == "Fused detailed":
                        groups = group_bin_jobs_for_fused(jobs)
                        st.info(f"Fused detailed: {len(jobs)} recipes -> {len(groups)} source-H5 group(s). 同じsourceは1回読みで処理します。")
                        overall = st.progress(0.0)
                        for gi, group in enumerate(groups, 1):
                            bins_txt = ", ".join([f"{int(r['space_bin'])}x{int(r['space_bin'])}/t{int(r['time_bin'])}" for r in group])
                            label = f"{group[0]['dataset_id']} / {group[0]['channel']}  ·  {bins_txt}"
                            st.markdown(f"#### {label}")
                            cb = progress_bar("Fused binning")
                            try:
                                infos = bin_images_h5_multi_same_source(group, progress_cb=cb)
                                for info in infos:
                                    summary_rows.append({"status": info.get("status"), "bin": f"{info.get('space_bin')}x{info.get('space_bin')} t{info.get('time_bin')}", "output": info.get("output_h5")})
                                st.success(f"done: {len(infos)} output(s)")
                            except Exception as e:
                                st.error(f"{label}: {type(e).__name__}: {e}")
                                summary_rows.append({"status": "failed", "bin": bins_txt, "output": str(e)})
                            overall.progress(gi / max(1, len(groups)))

                    elif execution_mode == "Auto":
                        groups = group_bin_jobs_for_fused(jobs)
                        st.info(f"Auto: {len(jobs)} recipes -> {len(groups)} source-H5 group(s). 同じsource内はfused、sourceが複数なら最大 {int(parallel_jobs)} 並列です。")
                        if len(groups) == 1 or int(parallel_jobs) <= 1:
                            # One source group: keep detailed progress because there is nothing useful to parallelize.
                            overall = st.progress(0.0)
                            for gi, group in enumerate(groups, 1):
                                bins_txt = ", ".join([f"{int(r['space_bin'])}x{int(r['space_bin'])}/t{int(r['time_bin'])}" for r in group])
                                label = f"{group[0]['dataset_id']} / {group[0]['channel']}  ·  {bins_txt}"
                                st.markdown(f"#### {label}")
                                cb = progress_bar("Auto fused binning")
                                try:
                                    infos = bin_images_h5_multi_same_source(group, progress_cb=cb)
                                    for info in infos:
                                        summary_rows.append({"status": info.get("status"), "bin": f"{info.get('space_bin')}x{info.get('space_bin')} t{info.get('time_bin')}", "output": info.get("output_h5")})
                                    st.success(f"done: {len(infos)} output(s)")
                                except Exception as e:
                                    st.error(f"{label}: {type(e).__name__}: {e}")
                                    summary_rows.append({"status": "failed", "bin": bins_txt, "output": str(e)})
                                overall.progress(gi / max(1, len(groups)))
                        else:
                            st.warning("Auto parallel uses source-group progress only. 詳細frame進捗を見たい場合は Fused detailed を使ってください。")
                            prog = st.progress(0.0)
                            status_box = st.empty()
                            done = 0
                            with ThreadPoolExecutor(max_workers=int(parallel_jobs)) as ex:
                                futs = [ex.submit(run_fused_group_job_from_list, group) for group in groups]
                                for fut in as_completed(futs):
                                    label, infos, err = fut.result()
                                    done += 1
                                    prog.progress(done / max(1, len(groups)))
                                    status_box.caption(f"completed source groups {done} / {len(groups)} at {datetime.now().strftime('%H:%M:%S')}")
                                    if err:
                                        st.error(f"{label}: {err}")
                                        summary_rows.append({"status": "failed", "bin": label, "output": err})
                                    else:
                                        st.success(f"{label}: done {len(infos or [])} output(s)")
                                        for info in infos or []:
                                            summary_rows.append({"status": info.get("status"), "bin": f"{info.get('space_bin')}x{info.get('space_bin')} t{info.get('time_bin')}", "output": info.get("output_h5")})

                    else:  # Parallel jobs
                        if int(parallel_jobs) <= 1:
                            st.info("Parallel jobs modeですが Parallel jobs=1 なので順番処理します。")
                        else:
                            st.warning("Parallel jobs mode uses recipe-level progress only. 同じsource H5を複数回読む可能性があります。通常はAuto推奨です。")
                        prog = st.progress(0.0)
                        status_box = st.empty()
                        done = 0
                        if int(parallel_jobs) <= 1:
                            iterator = []
                            for r in jobs:
                                iterator.append(run_bin_job_from_dict(r))
                            for label, info, err in iterator:
                                done += 1
                                prog.progress(done / max(1, len(jobs)))
                                status_box.caption(f"completed {done} / {len(jobs)} at {datetime.now().strftime('%H:%M:%S')}")
                                if err:
                                    st.error(f"{label}: {err}")
                                    summary_rows.append({"status": "failed", "bin": label, "output": err})
                                else:
                                    st.success(f"{label}: {info.get('status')} {info.get('output_h5')}")
                                    summary_rows.append({"status": info.get("status"), "bin": label, "output": info.get("output_h5")})
                        else:
                            with ThreadPoolExecutor(max_workers=int(parallel_jobs)) as ex:
                                futs = [ex.submit(run_bin_job_from_dict, r) for r in jobs]
                                for fut in as_completed(futs):
                                    label, info, err = fut.result()
                                    done += 1
                                    prog.progress(done / max(1, len(jobs)))
                                    status_box.caption(f"completed recipes {done} / {len(jobs)} at {datetime.now().strftime('%H:%M:%S')}")
                                    if err:
                                        st.error(f"{label}: {err}")
                                        summary_rows.append({"status": "failed", "bin": label, "output": err})
                                    else:
                                        st.success(f"{label}: {info.get('status')} {info.get('output_h5')}")
                                        summary_rows.append({"status": info.get("status"), "bin": label, "output": info.get("output_h5")})

                    if summary_rows:
                        st.dataframe(pd.DataFrame(summary_rows), width="stretch", height=min(320, 38 * (len(summary_rows) + 1)))
                    if analysis_root:
                        rebuild_project_catalog(analysis_root)
                        bump_project_refresh_tokens()
                    st.success(f"Run complete | elapsed {_fmt_seconds(time.perf_counter() - run_start)}")

    with tab_catalog:
        st.subheader("Catalog")
        if not analysis_root:
            st.info("Analysis root を指定してください。")
        else:
            c1, c2 = st.columns([1, 3])
            with c1:
                if st.button("Rebuild catalog", type="primary", width="stretch", key="catalog_rebuild_btn"):
                    dfcat = rebuild_project_catalog(analysis_root)
                    bump_project_refresh_tokens()
                    st.session_state["catalog_df"] = dfcat
                    st.success(f"catalog rows: {len(dfcat)}")
            dfcat = st.session_state.get("catalog_df")
            if dfcat is None:
                if (analysis_root / CATALOG_CSV).exists():
                    try:
                        dfcat = pd.read_csv(analysis_root / CATALOG_CSV)
                    except Exception:
                        dfcat = pd.DataFrame()
                else:
                    dfcat = rebuild_project_catalog(analysis_root)
            if dfcat is not None and not dfcat.empty:
                st.dataframe(dfcat, width="stretch", height=420)
                st.caption(f"{analysis_root / CATALOG_CSV}")
            else:
                st.info("catalog が空です。")

    with tab_settings:
        st.subheader("Performance / PC settings")
        st.caption("別PCで使う場合はここを調整してください。安全側にしたい場合は Safe / Conservative Laptop、NVMe + 大容量メモリなら Fast NVMe が目安です。")

        preset_current = st.selectbox("Speed preset", list(SPEED_PRESETS.keys()), index=list(SPEED_PRESETS.keys()).index("Balanced"), key="settings_speed_preset")
        if st.button("Apply preset", type="primary", key="settings_apply_preset_btn"):
            set_perf_from_preset(preset_current)
            st.success(f"Applied preset: {preset_current}")

        perf_now = default_perf()
        p1, p2, p3, p4 = st.columns(4)
        with p1:
            perf_now["parallel_jobs"] = st.number_input("Parallel bin jobs", min_value=1, max_value=8, value=int(perf_now.get("parallel_jobs", 1)), step=1, key="settings_parallel_bin_jobs")
        with p2:
            perf_now["tif_workers"] = st.number_input("TIF/RAW workers", min_value=1, max_value=64, value=int(perf_now.get("tif_workers", 8)), step=1, key="settings_tif_workers")
        with p3:
            perf_now["tif_batch"] = st.number_input("TIF/RAW batch", min_value=1, max_value=2048, value=int(perf_now.get("tif_batch", 64)), step=1, key="settings_tif_batch")
        with p4:
            perf_now["eiger_batch"] = st.number_input("EIGER batch", min_value=1, max_value=8192, value=int(perf_now.get("eiger_batch", 512)), step=64, key="settings_eiger_batch")

        p5, p6, p7, p8 = st.columns(4)
        with p5:
            perf_now["h5_chunk_mib"] = st.number_input("H5 chunk MiB", min_value=1.0, max_value=1024.0, value=float(perf_now.get("h5_chunk_mib", 64.0)), step=8.0, key="settings_h5_chunk_mib")
        with p6:
            perf_now["bin_chunk_factor"] = st.number_input("Bin chunk factor", min_value=16, max_value=4096, value=int(perf_now.get("bin_chunk_factor", 512)), step=16, key="settings_bin_chunk_factor")
        with p7:
            perf_now["preview_mode"] = st.selectbox("Preview mode", PREVIEW_MODES, index=PREVIEW_MODES.index(str(perf_now.get("preview_mode", "first_only"))) if str(perf_now.get("preview_mode", "first_only")) in PREVIEW_MODES else 1, key="settings_preview_mode")
        with p8:
            perf_now["compression_mode"] = st.selectbox("Compression", COMPRESSION_MODES, index=COMPRESSION_MODES.index(str(perf_now.get("compression_mode", "bitshuffle_lz4"))) if str(perf_now.get("compression_mode", "bitshuffle_lz4")) in COMPRESSION_MODES else 0, key="settings_compression")

        st.session_state["perf"] = dict(perf_now)

        st.markdown("### Notes")
        st.markdown("""
        - **Import H5** は初回用。Rawから canonical `01_h5/images.h5` を作るだけでもよいです。
        - **Bin** は後から何度でも追加できます。Rawは読みません。
        - `Parallel bin jobs` は別出力H5のジョブを並列化します。ディスクが遅いPCでは1が安全です。
        - `Bin chunk factor` は大きいほど速くなることがありますが、一時メモリ使用量が増えます。Queueの `est_block_gb` を目安にしてください。
        - `Compression=none` は最速ですがファイルが大きくなります。標準は `bitshuffle_lz4` です。
        - data_label は任意のラベル。`TIF/EIGER/existing H5` は source type。
        - TIF mask は Import時に canonical mask `1=valid, 0=invalid` に統一します。
        - 1D integration は別Managerに分ける想定です。catalogから `01_h5` または `02_bin` を選べるようにします。
        """)
        st.json({
            "canonical_mask": "1=valid,0=invalid",
            "pyfai_mask_note": "pyFAI integrationでは1=invalidへ変換する",
            "recommended_flow": ["Import H5", "Bin later", "ACF/Fit", "1D integration as separate module"],
            "active_performance": perf_now,
        })


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "--dxb-worker":
        raise SystemExit(_worker_run_import_job(sys.argv[2]))
    main()
