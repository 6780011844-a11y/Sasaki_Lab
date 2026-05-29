#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DXB H5 AlbulaLite Viewer + Selective TIF Export v3
================================================

目的
----
大量フレームの images.h5 を、Dectris Albula のように「見ながら」確認し、
必要な範囲だけ TIF へ軽く保存するための viewer/exporter です。

主な機能
--------
- H5ファイル/フォルダを選択し、/entry/data/images などの3D datasetを表示
- フレーム再生、スライダー移動、FPS/step指定、contrast/colormap調整
- 表示は軽量化: 画面サイズに合わせて間引き表示、uint8 preview変換
- 保存は分割:
    1) current frame: 現在表示フレームだけ保存
    2) range frames: start/end/step の各フレームを保存
    3) sample every N: N枚ごとに1枚保存
    4) aggregate groups: 1000枚ごと等で mean/sum/max/first/last を1枚に集約保存
    5) split whole into K: 全体を9分割等して、各区間を mean/sum/max/first/last 保存
- 出力dtype: raw / uint16 clip / uint8 display scaling / RGB display color
- H5が Bitshuffle/LZ4 の場合に備えて hdf5plugin を import
- 書き出しはworker threadで実行、Cancel対応

想定H5
------
- canonical DXB: /entry/data/images shape=(T,H,W)
- optional: /entry/data/timestamps, /entry/data/source_filenames, /entry/data/exposure_time

起動
----
python DXB_H5_AlbulaLite_Viewer_Export.py

必要ライブラリ
--------------
pip install h5py hdf5plugin tifffile pillow numpy

メモ
----
全フレームTIF出力は重くなりがちなので、まず viewer で範囲を見つけてから
aggregate / split / sample で必要最小限を書き出す設計にしています。
"""

from __future__ import annotations

import os
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import json
import math
import queue
import re
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import h5py
try:
    import hdf5plugin  # noqa: F401  # needed for bitshuffle/lz4
except Exception:
    hdf5plugin = None

import tifffile as tiff

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

try:
    from PIL import Image, ImageTk
except Exception as e:
    raise RuntimeError("Pillow is required. Install with: pip install pillow") from e


# =============================================================================
# Config
# =============================================================================

APP_TITLE = "DXB H5 AlbulaLite Viewer / Exporter v3"
DEFAULT_DATASET = "/entry/data/images"
MAX_TREE_ITEMS = 3000

# Preview colormap presets.  No matplotlib dependency: LUTs are interpolated here.
COLORMAP_NAMES = [
    "gray", "gray_r", "fire", "viridis", "plasma", "turbo", "physics", "cyan_hot"
]
COLORMAP_ANCHORS = {
    "gray":      ["#000000", "#ffffff"],
    "gray_r":    ["#ffffff", "#000000"],
    "fire":      ["#000000", "#240000", "#7a0000", "#e95a00", "#ffd64a", "#ffffff"],
    "viridis":   ["#440154", "#482878", "#3e4989", "#31688e", "#26828e", "#1f9e89", "#35b779", "#6ece58", "#b5de2b", "#fde725"],
    "plasma":    ["#0d0887", "#46039f", "#7201a8", "#9c179e", "#bd3786", "#d8576b", "#ed7953", "#fb9f3a", "#fdca26", "#f0f921"],
    "turbo":     ["#30123b", "#4145ab", "#4685f9", "#1bcfd4", "#2df09d", "#a4fc3c", "#f2e627", "#ff9b21", "#e0440e", "#7a0403"],
    "physics":   ["#1100ff", "#1a4cff", "#00b7ff", "#00f0ff", "#27ff88", "#b7ff00", "#fff200", "#ffb300", "#ff5a00", "#d40000"],
    "cyan_hot":  ["#000000", "#001a40", "#004cff", "#00d5ff", "#f7ff00", "#ff6a00", "#ffffff"],
}
_LUT_CACHE: dict[tuple[str, bool], np.ndarray] = {}


def _hex_to_rgb01(x: str) -> tuple[float, float, float]:
    x = x.strip().lstrip("#")
    return tuple(int(x[i:i+2], 16) / 255.0 for i in (0, 2, 4))


def colormap_lut(name: str, reverse: bool = False) -> np.ndarray:
    name = str(name or "gray").strip()
    if name not in COLORMAP_ANCHORS:
        name = "gray"
    key = (name, bool(reverse))
    if key in _LUT_CACHE:
        return _LUT_CACHE[key]
    colors = np.array([_hex_to_rgb01(c) for c in COLORMAP_ANCHORS[name]], dtype=np.float32)
    if reverse:
        colors = colors[::-1]
    x = np.linspace(0.0, 1.0, colors.shape[0], dtype=np.float32)
    xi = np.linspace(0.0, 1.0, 256, dtype=np.float32)
    lut = np.empty((256, 3), dtype=np.uint8)
    for ch in range(3):
        lut[:, ch] = np.clip(np.interp(xi, x, colors[:, ch]) * 255.0 + 0.5, 0, 255).astype(np.uint8)
    _LUT_CACHE[key] = lut
    return lut


def apply_colormap_to_u8(u8: np.ndarray, name: str, reverse: bool = False) -> np.ndarray:
    arr = np.asarray(u8, dtype=np.uint8)
    lut = colormap_lut(name, reverse=reverse)
    return lut[arr]


# =============================================================================
# Small utilities
# =============================================================================

def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def safe_name(s: str, default: str = "item") -> str:
    s = str(s or default)
    s = re.sub(r"[<>:\"/\\|?*\x00-\x1f]", "_", s)
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"_+", "_", s)
    return s.strip("._ ") or default


def tiff_ascii(s: Any, max_len: int = 32000) -> str:
    """Return a TIFF-safe ASCII string for ImageDescription/Software tags.

    tifffile encodes standard TIFF string tags as 7-bit ASCII.
    Japanese paths / filenames are kept in export_manifest.json, while the TIFF
    tag receives an ASCII-safe escaped version to avoid export failures.
    """
    txt = str(s if s is not None else "")
    txt = txt.replace("\x00", " ")
    txt = txt.encode("ascii", errors="backslashreplace").decode("ascii", errors="ignore")
    if len(txt) > int(max_len):
        txt = txt[: int(max_len) - 20] + "\n...truncated..."
    return txt


def natural_key(path_or_name: Any):
    name = Path(str(path_or_name)).name
    parts = re.split(r"(\d+)", name)
    return [int(p) if p.isdigit() else p.lower() for p in parts]


def clamp_int(x: Any, lo: int, hi: int, default: int = 0) -> int:
    try:
        v = int(float(str(x).strip()))
    except Exception:
        v = default
    return max(lo, min(hi, v))


def parse_float_or_none(x: Any) -> Optional[float]:
    try:
        s = str(x).strip()
        if s == "" or s.lower() in ["none", "nan", "auto"]:
            return None
        v = float(s)
        if not np.isfinite(v):
            return None
        return v
    except Exception:
        return None


def unique_path(p: Path) -> Path:
    p = Path(p)
    if not p.exists():
        return p
    stem = p.stem
    suf = p.suffix or ".tif"
    parent = p.parent
    k = 1
    while True:
        q = parent / f"{stem}_dup{k}{suf}"
        if not q.exists():
            return q
        k += 1


def dt_str_from_epoch(ts: float) -> str:
    return datetime.fromtimestamp(float(ts)).strftime("%Y:%m:%d %H:%M:%S")


def read_meta_scalar(ds, idx: int, default: Any):
    if ds is None:
        return default
    try:
        v = ds[idx:idx + 1][0]
        if isinstance(v, (bytes, bytearray)):
            return v.decode("utf-8", errors="replace")
        if isinstance(v, np.generic):
            return v.item()
        return v
    except Exception:
        return default


def scan_3d_datasets(h5_path: Path) -> list[tuple[str, tuple[int, ...], str]]:
    out: list[tuple[str, tuple[int, ...], str]] = []
    try:
        with h5py.File(h5_path, "r") as hf:
            def visitor(name: str, obj: Any):
                if isinstance(obj, h5py.Dataset) and obj.ndim == 3:
                    out.append(("/" + name if not name.startswith("/") else name,
                                tuple(map(int, obj.shape)), str(obj.dtype)))
            hf.visititems(visitor)
    except Exception:
        return []
    out.sort(key=lambda x: (0 if x[0] == DEFAULT_DATASET else 1, x[0]))
    return out


def inspect_h5_file(h5_path: Path) -> dict[str, Any]:
    info = {
        "path": str(h5_path),
        "name": h5_path.name,
        "ok": False,
        "dataset": "",
        "shape": "",
        "dtype": "",
        "datasets": [],
        "error": "",
    }
    try:
        dsets = scan_3d_datasets(h5_path)
        info["datasets"] = dsets
        if not dsets:
            info["error"] = "no 3D dataset"
            return info
        best = None
        for ds in dsets:
            if ds[0] == DEFAULT_DATASET:
                best = ds
                break
        if best is None:
            best = dsets[0]
        info.update({
            "ok": True,
            "dataset": best[0],
            "shape": str(best[1]),
            "dtype": best[2],
        })
    except Exception as e:
        info["error"] = f"{type(e).__name__}: {e}"
    return info


def collect_h5_files(root: Path, recursive: bool = True, max_items: int = MAX_TREE_ITEMS) -> list[Path]:
    root = Path(root)
    if root.is_file() and root.suffix.lower() in [".h5", ".hdf5", ".nxs"]:
        return [root]
    if not root.exists() or not root.is_dir():
        return []
    pat = "**/*" if recursive else "*"
    files = [p for p in root.glob(pat) if p.is_file() and p.suffix.lower() in [".h5", ".hdf5", ".nxs"]]
    files = sorted(files, key=natural_key)
    return files[:max_items]


def robust_percentile_limits(img: np.ndarray, low: float, high: float) -> tuple[float, float]:
    a = np.asarray(img)
    if a.size == 0:
        return 0.0, 1.0
    # 大きい画像ではサンプルして高速化
    flat = a.ravel()
    if flat.size > 1_000_000:
        step = max(1, flat.size // 1_000_000)
        flat = flat[::step]
    flat = flat[np.isfinite(flat)]
    if flat.size == 0:
        return 0.0, 1.0
    vmin = float(np.percentile(flat, low))
    vmax = float(np.percentile(flat, high))
    if not np.isfinite(vmin):
        vmin = float(np.nanmin(flat))
    if not np.isfinite(vmax):
        vmax = float(np.nanmax(flat))
    if vmax <= vmin:
        vmax = vmin + 1.0
    return vmin, vmax


def preview_to_uint8(img: np.ndarray, vmin: Optional[float], vmax: Optional[float],
                     p_low: float, p_high: float, log_view: bool = False) -> tuple[np.ndarray, float, float]:
    arr = np.asarray(img)
    if log_view:
        arrf = arr.astype(np.float32, copy=False)
        minv = np.nanmin(arrf) if np.isfinite(arrf).any() else 0.0
        if minv < 0:
            arrf = arrf - minv
        arr = np.log1p(arrf)
    if vmin is None or vmax is None or vmax <= vmin:
        vmin2, vmax2 = robust_percentile_limits(arr, p_low, p_high)
    else:
        vmin2, vmax2 = float(vmin), float(vmax)
    out = (arr.astype(np.float32, copy=False) - vmin2) / max(1e-12, (vmax2 - vmin2))
    out = np.clip(out, 0.0, 1.0)
    return (out * 255.0 + 0.5).astype(np.uint8), vmin2, vmax2


def cast_export_frame(img: np.ndarray, mode: str, display_vmin: Optional[float] = None,
                      display_vmax: Optional[float] = None) -> np.ndarray:
    mode = str(mode).lower().strip()
    if mode == "raw":
        return np.asarray(img)
    if mode == "uint16_clip":
        return np.clip(img, 0, 65535).astype(np.uint16, copy=False)
    if mode == "uint8_clip":
        return np.clip(img, 0, 255).astype(np.uint8, copy=False)
    if mode == "uint8_display":
        vmin = 0.0 if display_vmin is None else float(display_vmin)
        vmax = float(np.nanmax(img)) if display_vmax is None else float(display_vmax)
        if vmax <= vmin:
            vmax = vmin + 1.0
        arr = (img.astype(np.float32, copy=False) - vmin) / (vmax - vmin)
        return (np.clip(arr, 0, 1) * 255.0 + 0.5).astype(np.uint8)
    raise ValueError(f"unknown export dtype mode: {mode}")


def make_export_frame(img: np.ndarray, mode: str, display_vmin: Optional[float] = None,
                      display_vmax: Optional[float] = None, cmap_name: str = "gray",
                      cmap_reverse: bool = False, log_view: bool = False) -> tuple[np.ndarray, str]:
    """Return (array, photometric) for tifffile.imwrite."""
    mode = str(mode).lower().strip()
    if mode == "rgb_display":
        # Save exactly the display-style image: contrast range + log option + colormap.
        u8, _, _ = preview_to_uint8(
            img,
            vmin=display_vmin,
            vmax=display_vmax,
            p_low=1.0,
            p_high=99.5,
            log_view=log_view,
        )
        return apply_colormap_to_u8(u8, cmap_name, reverse=cmap_reverse), "rgb"
    return cast_export_frame(img, mode, display_vmin, display_vmax), "minisblack"


def aggregate_block(block: np.ndarray, method: str) -> np.ndarray:
    method = str(method).lower().strip()
    if block.ndim == 2:
        return block
    if method == "first":
        return block[0]
    if method == "last":
        return block[-1]
    if method == "mean":
        return np.mean(block.astype(np.float32, copy=False), axis=0, dtype=np.float32)
    if method == "sum":
        return np.sum(block.astype(np.float64, copy=False), axis=0, dtype=np.float64)
    if method == "max":
        return np.max(block, axis=0)
    if method == "median":
        return np.median(block.astype(np.float32, copy=False), axis=0)
    raise ValueError(f"unknown aggregation method: {method}")


# =============================================================================
# H5 reader with small LRU cache
# =============================================================================

class LRUFrameCache:
    def __init__(self, max_items: int = 32):
        self.max_items = max(1, int(max_items))
        self._d: OrderedDict[int, np.ndarray] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: int) -> Optional[np.ndarray]:
        with self._lock:
            if key not in self._d:
                return None
            val = self._d.pop(key)
            self._d[key] = val
            return val

    def put(self, key: int, val: np.ndarray):
        with self._lock:
            if key in self._d:
                self._d.pop(key)
            self._d[key] = val
            while len(self._d) > self.max_items:
                self._d.popitem(last=False)

    def clear(self):
        with self._lock:
            self._d.clear()


class H5Stack:
    def __init__(self):
        self.path: Optional[Path] = None
        self.dataset_path: str = DEFAULT_DATASET
        self.hf: Optional[h5py.File] = None
        self.ds = None
        self.ts_ds = None
        self.src_ds = None
        self.exp_ds = None
        self.cache = LRUFrameCache(max_items=48)
        self.shape = (0, 0, 0)
        self.dtype = ""

    def close(self):
        self.cache.clear()
        if self.hf is not None:
            try:
                self.hf.close()
            except Exception:
                pass
        self.path = None
        self.hf = None
        self.ds = None
        self.ts_ds = None
        self.src_ds = None
        self.exp_ds = None
        self.shape = (0, 0, 0)
        self.dtype = ""

    def open(self, path: str | Path, dataset_path: str = DEFAULT_DATASET):
        self.close()
        self.path = Path(path)
        self.dataset_path = dataset_path
        self.hf = h5py.File(self.path, "r", rdcc_nbytes=512 * 1024 * 1024, rdcc_nslots=1_000_003)
        if dataset_path not in self.hf:
            raise KeyError(f"dataset not found: {dataset_path}")
        self.ds = self.hf[dataset_path]
        if self.ds.ndim != 3:
            raise ValueError(f"dataset is not 3D: {self.ds.shape}")
        self.shape = tuple(map(int, self.ds.shape))
        self.dtype = str(self.ds.dtype)
        # Metadata is only standardized for /entry/data/images, but try common paths.
        base = "/entry/data"
        self.ts_ds = self.hf.get(base + "/timestamps", None)
        self.src_ds = self.hf.get(base + "/source_filenames", None)
        self.exp_ds = self.hf.get(base + "/exposure_time", None)

    @property
    def is_open(self) -> bool:
        return self.ds is not None

    def read_frame(self, idx: int, use_cache: bool = True) -> np.ndarray:
        if self.ds is None:
            raise RuntimeError("no H5 open")
        T = self.shape[0]
        idx = max(0, min(T - 1, int(idx)))
        if use_cache:
            hit = self.cache.get(idx)
            if hit is not None:
                return hit
        arr = self.ds[idx]
        if use_cache:
            self.cache.put(idx, arr)
        return arr

    def read_block(self, start: int, end_exclusive: int) -> np.ndarray:
        if self.ds is None:
            raise RuntimeError("no H5 open")
        T = self.shape[0]
        s = max(0, min(T, int(start)))
        e = max(s, min(T, int(end_exclusive)))
        return self.ds[s:e]

    def meta_for_frame(self, idx: int) -> dict[str, Any]:
        default_name = f"{self.path.stem if self.path else 'images'}_frame_{idx:06d}.tif"
        base = read_meta_scalar(self.src_ds, idx, default_name)
        if not str(base).lower().endswith((".tif", ".tiff")):
            base = f"{base}.tif"
        ts = read_meta_scalar(self.ts_ds, idx, time.time())
        try:
            tsf = float(ts)
        except Exception:
            tsf = time.time()
        exp = read_meta_scalar(self.exp_ds, idx, None)
        try:
            expf = float(exp)
            if not np.isfinite(expf):
                expf = None
        except Exception:
            expf = None
        return {"source_filename": str(base), "timestamp": tsf, "exposure_time": expf}


# =============================================================================
# Export worker
# =============================================================================

@dataclass
class ExportConfig:
    mode: str
    out_dir: str
    start: int
    end: int
    step: int = 1
    every_n: int = 1000
    group_size: int = 1000
    split_k: int = 9
    agg_method: str = "mean"
    dtype_mode: str = "uint16_clip"
    batch_read: int = 64
    display_vmin: Optional[float] = None
    display_vmax: Optional[float] = None
    cmap_name: str = "gray"
    cmap_reverse: bool = False
    log_view: bool = False
    prefix: str = "export"


class ExportWorker(threading.Thread):
    def __init__(self, h5_path: Path, dataset_path: str, cfg: ExportConfig,
                 q: queue.Queue, cancel_event: threading.Event):
        super().__init__(daemon=True)
        self.h5_path = Path(h5_path)
        self.dataset_path = dataset_path
        self.cfg = cfg
        self.q = q
        self.cancel_event = cancel_event

    def _progress(self, current: int, total: int, message: str):
        self.q.put({"type": "progress", "current": int(current), "total": int(total), "message": str(message)})

    def _write_tif(self, path: Path, image: np.ndarray, desc: str, dt_epoch: Optional[float] = None):
        arr, photometric = make_export_frame(
            image,
            self.cfg.dtype_mode,
            self.cfg.display_vmin,
            self.cfg.display_vmax,
            cmap_name=self.cfg.cmap_name,
            cmap_reverse=self.cfg.cmap_reverse,
            log_view=self.cfg.log_view,
        )
        kwargs = dict(
            photometric=photometric,
            description=tiff_ascii(desc),
            software=tiff_ascii("DXB H5 AlbulaLite"),
        )
        if dt_epoch is not None:
            try:
                kwargs["datetime"] = dt_str_from_epoch(dt_epoch)
            except Exception:
                pass
        tiff.imwrite(str(path), arr, **kwargs)
        if dt_epoch is not None:
            try:
                os.utime(path, (float(dt_epoch), float(dt_epoch)))
            except Exception:
                pass

    def _iter_ranges(self, T: int) -> list[tuple[int, int, str]]:
        cfg = self.cfg
        start = max(0, min(T - 1, int(cfg.start)))
        end = max(start, min(T - 1, int(cfg.end)))
        if cfg.mode == "current frame":
            return [(start, start + 1, f"frame_{start:06d}")]
        if cfg.mode == "range frames":
            step = max(1, int(cfg.step))
            return [(i, i + 1, f"frame_{i:06d}") for i in range(start, end + 1, step)]
        if cfg.mode == "sample every n":
            n = max(1, int(cfg.every_n))
            return [(i, i + 1, f"sample_{i:06d}") for i in range(start, end + 1, n)]
        if cfg.mode == "aggregate groups":
            g = max(1, int(cfg.group_size))
            ranges = []
            k = 0
            for s in range(start, end + 1, g):
                e = min(end + 1, s + g)
                ranges.append((s, e, f"grp{k:04d}_{s:06d}-{e-1:06d}_{cfg.agg_method}"))
                k += 1
            return ranges
        if cfg.mode == "split whole into k":
            k = max(1, int(cfg.split_k))
            edges = np.linspace(start, end + 1, k + 1).round().astype(int)
            ranges = []
            for i in range(k):
                s = int(edges[i])
                e = int(edges[i + 1])
                if e <= s:
                    continue
                ranges.append((s, e, f"split{i+1:02d}_of_{k:02d}_{s:06d}-{e-1:06d}_{cfg.agg_method}"))
            return ranges
        raise ValueError(f"unknown mode: {cfg.mode}")

    def run(self):
        try:
            out_dir = Path(self.cfg.out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            manifest: dict[str, Any] = {
                "created_at": now_iso(),
                "source_h5": str(self.h5_path),
                "dataset_path": self.dataset_path,
                "export_config": asdict(self.cfg),
                "outputs": [],
            }
            with h5py.File(self.h5_path, "r", rdcc_nbytes=512 * 1024 * 1024, rdcc_nslots=1_000_003) as hf:
                if self.dataset_path not in hf:
                    raise KeyError(f"dataset not found: {self.dataset_path}")
                ds = hf[self.dataset_path]
                T, H, W = map(int, ds.shape)
                ts_ds = hf.get("/entry/data/timestamps", None)
                src_ds = hf.get("/entry/data/source_filenames", None)
                exp_ds = hf.get("/entry/data/exposure_time", None)

                ranges = self._iter_ranges(T)
                total = len(ranges)
                self._progress(0, total, f"start export: {total} item(s)")

                for j, (s, e, tag) in enumerate(ranges, start=1):
                    if self.cancel_event.is_set():
                        raise RuntimeError("cancelled by user")
                    prefix = safe_name(self.cfg.prefix or self.h5_path.stem, "export")
                    out_name = f"{prefix}_{safe_name(tag)}.tif"
                    out_path = unique_path(out_dir / out_name)

                    if e - s == 1:
                        img = ds[s]
                        meta_name = read_meta_scalar(src_ds, s, out_name)
                        tsf = read_meta_scalar(ts_ds, s, None)
                        try:
                            tsf = float(tsf) if tsf is not None else None
                        except Exception:
                            tsf = None
                        exp = read_meta_scalar(exp_ds, s, None)
                        desc = [
                            f"source_h5={self.h5_path}",
                            f"dataset={self.dataset_path}",
                            f"frame_index={s}",
                            f"source_filename={meta_name}",
                            f"export_mode={self.cfg.mode}",
                        ]
                        if exp is not None:
                            desc.append(f"exposure_time={exp}")
                        self._write_tif(out_path, img, "\n".join(desc), dt_epoch=tsf)
                    else:
                        # Read grouped block in smaller chunks for memory, then aggregate incrementally for sum/mean/max.
                        method = self.cfg.agg_method.lower().strip()
                        batch = max(1, int(self.cfg.batch_read))
                        if method in ["first", "last"]:
                            idx = s if method == "first" else e - 1
                            img = ds[idx]
                        elif method in ["sum", "mean"]:
                            acc = None
                            count = 0
                            for b0 in range(s, e, batch):
                                if self.cancel_event.is_set():
                                    raise RuntimeError("cancelled by user")
                                b1 = min(e, b0 + batch)
                                block = ds[b0:b1].astype(np.float64 if method == "sum" else np.float32, copy=False)
                                part = np.sum(block, axis=0, dtype=np.float64 if method == "sum" else np.float32)
                                if acc is None:
                                    acc = part
                                else:
                                    acc += part
                                count += (b1 - b0)
                            img = acc / max(1, count) if method == "mean" else acc
                        elif method == "max":
                            acc = None
                            for b0 in range(s, e, batch):
                                if self.cancel_event.is_set():
                                    raise RuntimeError("cancelled by user")
                                b1 = min(e, b0 + batch)
                                part = np.max(ds[b0:b1], axis=0)
                                acc = part if acc is None else np.maximum(acc, part)
                            img = acc
                        elif method == "median":
                            # Median needs full block. Warn by behavior: use smaller groups for huge data.
                            block = ds[s:e]
                            img = np.median(block.astype(np.float32, copy=False), axis=0)
                        else:
                            raise ValueError(f"unknown aggregation method: {method}")

                        desc = [
                            f"source_h5={self.h5_path}",
                            f"dataset={self.dataset_path}",
                            f"frame_start={s}",
                            f"frame_end_inclusive={e-1}",
                            f"n_frames={e-s}",
                            f"export_mode={self.cfg.mode}",
                            f"aggregation={method}",
                        ]
                        self._write_tif(out_path, img, "\n".join(desc), dt_epoch=None)

                    manifest["outputs"].append({
                        "path": str(out_path),
                        "frame_start": int(s),
                        "frame_end_exclusive": int(e),
                        "label": tag,
                    })
                    self._progress(j, total, f"saved {j}/{total}: {out_path.name}")

            manifest_path = out_dir / f"{safe_name(self.cfg.prefix or self.h5_path.stem)}_export_manifest.json"
            manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
            self.q.put({"type": "done", "message": f"done: {len(manifest['outputs'])} tif(s)", "manifest": str(manifest_path)})
        except Exception as e:
            self.q.put({"type": "error", "message": f"{type(e).__name__}: {e}"})


# =============================================================================
# GUI
# =============================================================================

class AlbulaLiteApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title(APP_TITLE)
        self.root.geometry("1500x940")
        self.root.minsize(1150, 720)

        self.stack = H5Stack()
        self.current_frame = 0
        self.playing = False
        self.photo = None
        self.last_display_limits = (None, None)
        self.export_q: queue.Queue = queue.Queue()
        self.export_cancel = threading.Event()
        self.export_worker: Optional[ExportWorker] = None

        self.h5_infos: list[dict[str, Any]] = []
        self.selected_h5_path: Optional[Path] = None

        self._build_vars()
        self._build_ui()
        self._poll_export_queue()

    # ------------------------------------------------------------------
    # UI setup
    # ------------------------------------------------------------------
    def _build_vars(self):
        self.path_var = tk.StringVar(value="")
        self.recursive_var = tk.BooleanVar(value=True)
        self.dataset_var = tk.StringVar(value=DEFAULT_DATASET)
        self.info_var = tk.StringVar(value="No H5 loaded")

        self.frame_var = tk.IntVar(value=0)
        self.fps_var = tk.DoubleVar(value=8.0)
        self.play_step_var = tk.IntVar(value=1)
        self.log_view_var = tk.BooleanVar(value=False)
        self.cmap_var = tk.StringVar(value="gray")
        self.cmap_reverse_var = tk.BooleanVar(value=False)
        self.auto_contrast_var = tk.BooleanVar(value=True)
        self.p_low_var = tk.DoubleVar(value=1.0)
        self.p_high_var = tk.DoubleVar(value=99.5)
        self.vmin_var = tk.StringVar(value="")
        self.vmax_var = tk.StringVar(value="")
        self.flip_y_var = tk.BooleanVar(value=False)
        self.transpose_var = tk.BooleanVar(value=False)
        self.max_preview_px_var = tk.IntVar(value=1100)

        self.export_preset_var = tk.StringVar(value="今見ている1枚だけ")
        self.export_mode_var = tk.StringVar(value="current frame")
        self.export_out_var = tk.StringVar(value="")
        self.export_start_var = tk.IntVar(value=0)
        self.export_end_var = tk.IntVar(value=0)
        self.export_step_var = tk.IntVar(value=1)
        self.export_every_var = tk.IntVar(value=1000)
        self.export_group_var = tk.IntVar(value=1000)
        self.export_split_var = tk.IntVar(value=9)
        self.export_agg_var = tk.StringVar(value="mean")
        self.export_dtype_var = tk.StringVar(value="uint16_clip")
        self.export_dtype_hint_var = tk.StringVar(value="uint16_clip: 解析・再利用向け。0-65535にクリップして保存。")
        self.export_batch_var = tk.IntVar(value=32)
        self.export_prefix_var = tk.StringVar(value="")
        self.status_var = tk.StringVar(value="Ready")

    def _build_ui(self):
        main = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main.pack(fill=tk.BOTH, expand=True)

        left = ttk.Frame(main, padding=8)
        center = ttk.Frame(main, padding=8)
        right = ttk.Frame(main, padding=8)
        main.add(left, weight=1)
        main.add(center, weight=4)
        main.add(right, weight=2)

        self._build_left(left)
        self._build_center(center)
        self._build_right(right)

        status = ttk.Label(self.root, textvariable=self.status_var, anchor="w")
        status.pack(side=tk.BOTTOM, fill=tk.X)

    def _build_left(self, parent: ttk.Frame):
        ttk.Label(parent, text="1. H5 source", font=("TkDefaultFont", 11, "bold")).pack(anchor="w")
        row = ttk.Frame(parent)
        row.pack(fill=tk.X, pady=(4, 4))
        ttk.Entry(row, textvariable=self.path_var).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(row, text="File", command=self.choose_file).pack(side=tk.LEFT, padx=(4, 0))
        ttk.Button(row, text="Folder", command=self.choose_folder).pack(side=tk.LEFT, padx=(4, 0))
        ttk.Checkbutton(parent, text="recursive scan", variable=self.recursive_var).pack(anchor="w")
        ttk.Button(parent, text="Scan / Refresh", command=self.scan_source).pack(fill=tk.X, pady=(4, 8))

        cols = ("name", "shape", "dtype", "dataset")
        self.h5_tree = ttk.Treeview(parent, columns=cols, show="headings", height=18)
        self.h5_tree.heading("name", text="H5")
        self.h5_tree.heading("shape", text="shape")
        self.h5_tree.heading("dtype", text="dtype")
        self.h5_tree.heading("dataset", text="dataset")
        self.h5_tree.column("name", width=170)
        self.h5_tree.column("shape", width=120)
        self.h5_tree.column("dtype", width=70)
        self.h5_tree.column("dataset", width=180)
        self.h5_tree.pack(fill=tk.BOTH, expand=True)
        self.h5_tree.bind("<<TreeviewSelect>>", self.on_h5_selected)
        self.h5_tree.bind("<Double-1>", lambda e: self.open_selected_h5())

        ds_row = ttk.Frame(parent)
        ds_row.pack(fill=tk.X, pady=(8, 4))
        ttk.Label(ds_row, text="Dataset").pack(side=tk.LEFT)
        self.dataset_combo = ttk.Combobox(ds_row, textvariable=self.dataset_var, state="readonly", values=[DEFAULT_DATASET])
        self.dataset_combo.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(6, 0))
        ttk.Button(parent, text="Open selected H5", command=self.open_selected_h5).pack(fill=tk.X, pady=(2, 4))

        ttk.Label(parent, text="Info", font=("TkDefaultFont", 10, "bold")).pack(anchor="w", pady=(8, 0))
        ttk.Label(parent, textvariable=self.info_var, wraplength=430, justify="left").pack(anchor="w", fill=tk.X)

    def _build_center(self, parent: ttk.Frame):
        ttk.Label(parent, text="2. Viewer", font=("TkDefaultFont", 11, "bold")).pack(anchor="w")
        self.canvas = tk.Canvas(parent, bg="#111111", highlightthickness=1, highlightbackground="#444")
        self.canvas.pack(fill=tk.BOTH, expand=True, pady=(4, 8))
        self.canvas.bind("<Configure>", lambda e: self.display_current_frame())

        nav = ttk.Frame(parent)
        nav.pack(fill=tk.X)
        ttk.Button(nav, text="⏮", width=4, command=lambda: self.goto_frame(0)).pack(side=tk.LEFT)
        ttk.Button(nav, text="◀", width=4, command=lambda: self.step_frame(-self.play_step_var.get())).pack(side=tk.LEFT)
        self.play_btn = ttk.Button(nav, text="▶ Play", width=10, command=self.toggle_play)
        self.play_btn.pack(side=tk.LEFT, padx=4)
        ttk.Button(nav, text="▶", width=4, command=lambda: self.step_frame(self.play_step_var.get())).pack(side=tk.LEFT)
        ttk.Button(nav, text="⏭", width=4, command=self.goto_last).pack(side=tk.LEFT)
        ttk.Label(nav, text="Frame").pack(side=tk.LEFT, padx=(12, 3))
        self.frame_spin = ttk.Spinbox(nav, from_=0, to=0, textvariable=self.frame_var, width=10, command=self.on_frame_spin)
        self.frame_spin.pack(side=tk.LEFT)
        ttk.Button(nav, text="Go", command=self.on_frame_spin).pack(side=tk.LEFT, padx=(3, 8))
        ttk.Label(nav, text="FPS").pack(side=tk.LEFT)
        ttk.Spinbox(nav, from_=0.2, to=60.0, increment=0.5, textvariable=self.fps_var, width=5).pack(side=tk.LEFT, padx=(3, 8))
        ttk.Label(nav, text="step").pack(side=tk.LEFT)
        ttk.Spinbox(nav, from_=1, to=10000, increment=1, textvariable=self.play_step_var, width=7).pack(side=tk.LEFT, padx=(3, 8))

        self.slider = ttk.Scale(parent, from_=0, to=0, orient=tk.HORIZONTAL, command=self.on_slider)
        self.slider.pack(fill=tk.X, pady=(6, 0))

        opts = ttk.LabelFrame(parent, text="Display")
        opts.pack(fill=tk.X, pady=(8, 0))
        c1 = ttk.Frame(opts)
        c1.pack(fill=tk.X, padx=6, pady=4)
        ttk.Checkbutton(c1, text="auto range", variable=self.auto_contrast_var, command=self.display_current_frame).pack(side=tk.LEFT)
        ttk.Label(c1, text="low %").pack(side=tk.LEFT, padx=(8, 2))
        ttk.Spinbox(c1, from_=0.0, to=50.0, increment=0.1, textvariable=self.p_low_var, width=6, command=self.display_current_frame).pack(side=tk.LEFT)
        ttk.Label(c1, text="high %").pack(side=tk.LEFT, padx=(8, 2))
        ttk.Spinbox(c1, from_=50.0, to=100.0, increment=0.1, textvariable=self.p_high_var, width=6, command=self.display_current_frame).pack(side=tk.LEFT)
        ttk.Label(c1, text="vmin").pack(side=tk.LEFT, padx=(12, 2))
        ttk.Entry(c1, textvariable=self.vmin_var, width=9).pack(side=tk.LEFT)
        ttk.Label(c1, text="vmax").pack(side=tk.LEFT, padx=(6, 2))
        ttk.Entry(c1, textvariable=self.vmax_var, width=9).pack(side=tk.LEFT)
        ttk.Button(c1, text="Apply", command=self.display_current_frame).pack(side=tk.LEFT, padx=6)
        ttk.Button(c1, text="Use shown", command=self.use_current_as_manual_range).pack(side=tk.LEFT, padx=(0, 4))
        ttk.Checkbutton(c1, text="log", variable=self.log_view_var, command=self.display_current_frame).pack(side=tk.LEFT, padx=(8, 0))

        c2 = ttk.Frame(opts)
        c2.pack(fill=tk.X, padx=6, pady=(0, 4))
        ttk.Label(c2, text="colormap").pack(side=tk.LEFT)
        cmap_combo = ttk.Combobox(c2, textvariable=self.cmap_var, values=COLORMAP_NAMES, state="readonly", width=11)
        cmap_combo.pack(side=tk.LEFT, padx=(4, 4))
        cmap_combo.bind("<<ComboboxSelected>>", lambda e: self.display_current_frame())
        ttk.Checkbutton(c2, text="reverse", variable=self.cmap_reverse_var, command=self.display_current_frame).pack(side=tk.LEFT)
        ttk.Button(c2, text="soft", command=lambda: self.set_contrast_preset(1.0, 99.5)).pack(side=tk.LEFT, padx=(12, 2))
        ttk.Button(c2, text="strong", command=lambda: self.set_contrast_preset(0.1, 99.9)).pack(side=tk.LEFT, padx=2)
        ttk.Button(c2, text="narrow", command=lambda: self.adjust_manual_range(0.70)).pack(side=tk.LEFT, padx=2)
        ttk.Button(c2, text="wide", command=lambda: self.adjust_manual_range(1.40)).pack(side=tk.LEFT, padx=2)
        ttk.Checkbutton(c2, text="flip Y", variable=self.flip_y_var, command=self.display_current_frame).pack(side=tk.LEFT, padx=(12, 0))
        ttk.Checkbutton(c2, text="transpose", variable=self.transpose_var, command=self.display_current_frame).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Label(c2, text="max px").pack(side=tk.LEFT, padx=(12, 2))
        ttk.Spinbox(c2, from_=300, to=3000, increment=100, textvariable=self.max_preview_px_var, width=6, command=self.display_current_frame).pack(side=tk.LEFT)

    def _build_right(self, parent: ttk.Frame):
        ttk.Label(parent, text="3. Selective export", font=("TkDefaultFont", 11, "bold")).pack(anchor="w")

        preset_box = ttk.LabelFrame(parent, text="Quick preset")
        preset_box.pack(fill=tk.X, pady=(4, 4))
        pr = ttk.Frame(preset_box)
        pr.pack(fill=tk.X, padx=6, pady=4)
        preset_values = ["今見ている1枚だけ", "範囲を間引き保存", "1000枚ごと平均", "全体9分割平均", "範囲を全フレーム保存"]
        ttk.Combobox(pr, textvariable=self.export_preset_var, values=preset_values, state="readonly").pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(pr, text="Apply", command=self.apply_export_preset).pack(side=tk.LEFT, padx=(4, 0))

        out_row = ttk.Frame(parent)
        out_row.pack(fill=tk.X, pady=(4, 6))
        ttk.Entry(out_row, textvariable=self.export_out_var).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(out_row, text="Out", command=self.choose_output_folder).pack(side=tk.LEFT, padx=(4, 0))

        box = ttk.LabelFrame(parent, text="Export mode")
        box.pack(fill=tk.X, pady=4)
        modes = [
            ("current frame", "current frame  / 今の1枚"),
            ("range frames", "range frames  / 範囲をstepごと"),
            ("sample every n", "sample every N  / N枚ごとに1枚"),
            ("aggregate groups", "aggregate groups  / N枚を1枚に平均など"),
            ("split whole into k", "split whole into K  / 全体をK分割"),
        ]
        for value, label in modes:
            ttk.Radiobutton(box, text=label, variable=self.export_mode_var, value=value, command=self.update_export_hint).pack(anchor="w", padx=8, pady=1)

        rng = ttk.LabelFrame(parent, text="Frame range")
        rng.pack(fill=tk.X, pady=4)
        grid = ttk.Frame(rng)
        grid.pack(fill=tk.X, padx=6, pady=4)
        ttk.Label(grid, text="start").grid(row=0, column=0, sticky="w")
        ttk.Spinbox(grid, from_=0, to=0, textvariable=self.export_start_var, width=10).grid(row=0, column=1, sticky="ew", padx=4)
        ttk.Label(grid, text="end").grid(row=0, column=2, sticky="w")
        ttk.Spinbox(grid, from_=0, to=0, textvariable=self.export_end_var, width=10).grid(row=0, column=3, sticky="ew", padx=4)
        ttk.Button(grid, text="use current", command=self.set_export_current).grid(row=1, column=0, columnspan=2, sticky="ew", pady=(4, 0))
        ttk.Button(grid, text="use all", command=self.set_export_all).grid(row=1, column=2, columnspan=2, sticky="ew", pady=(4, 0))
        for i in range(4):
            grid.columnconfigure(i, weight=1)

        params = ttk.LabelFrame(parent, text="Mode parameters")
        params.pack(fill=tk.X, pady=4)
        pg = ttk.Frame(params)
        pg.pack(fill=tk.X, padx=6, pady=4)
        ttk.Label(pg, text="range step").grid(row=0, column=0, sticky="w")
        ttk.Spinbox(pg, from_=1, to=100000, textvariable=self.export_step_var, width=10).grid(row=0, column=1, sticky="ew", padx=4)
        ttk.Label(pg, text="sample every N").grid(row=1, column=0, sticky="w")
        ttk.Spinbox(pg, from_=1, to=10000000, textvariable=self.export_every_var, width=10).grid(row=1, column=1, sticky="ew", padx=4)
        ttk.Label(pg, text="group size").grid(row=2, column=0, sticky="w")
        ttk.Spinbox(pg, from_=1, to=10000000, textvariable=self.export_group_var, width=10).grid(row=2, column=1, sticky="ew", padx=4)
        ttk.Label(pg, text="split K").grid(row=3, column=0, sticky="w")
        ttk.Spinbox(pg, from_=1, to=10000, textvariable=self.export_split_var, width=10).grid(row=3, column=1, sticky="ew", padx=4)
        ttk.Label(pg, text="aggregation").grid(row=4, column=0, sticky="w")
        ttk.Combobox(pg, textvariable=self.export_agg_var, values=["mean", "sum", "max", "first", "last", "median"], state="readonly", width=12).grid(row=4, column=1, sticky="ew", padx=4)
        ttk.Label(pg, text="save type").grid(row=5, column=0, sticky="w")
        dtype_combo = ttk.Combobox(pg, textvariable=self.export_dtype_var, values=["raw", "uint16_clip", "uint8_clip", "uint8_display", "rgb_display"], state="readonly", width=12)
        dtype_combo.grid(row=5, column=1, sticky="ew", padx=4)
        dtype_combo.bind("<<ComboboxSelected>>", lambda e: self.update_dtype_hint())
        ttk.Label(pg, text="read batch").grid(row=6, column=0, sticky="w")
        ttk.Spinbox(pg, from_=1, to=4096, textvariable=self.export_batch_var, width=10).grid(row=6, column=1, sticky="ew", padx=4)
        ttk.Label(pg, text="prefix").grid(row=7, column=0, sticky="w")
        ttk.Entry(pg, textvariable=self.export_prefix_var).grid(row=7, column=1, sticky="ew", padx=4)
        pg.columnconfigure(1, weight=1)

        ttk.Label(parent, textvariable=self.export_dtype_hint_var, wraplength=420, justify="left").pack(anchor="w", fill=tk.X, pady=(2, 2))
        ttk.Button(parent, text="Update estimate", command=self.update_export_hint).pack(fill=tk.X, pady=(2, 2))

        self.export_hint_var = tk.StringVar(value="current frame only")
        ttk.Label(parent, textvariable=self.export_hint_var, wraplength=420, justify="left").pack(anchor="w", fill=tk.X, pady=(4, 4))

        buttons = ttk.Frame(parent)
        buttons.pack(fill=tk.X, pady=(4, 6))
        ttk.Button(buttons, text="Export", command=self.start_export).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(buttons, text="Cancel", command=self.cancel_export).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(6, 0))

        self.progress = ttk.Progressbar(parent, orient=tk.HORIZONTAL, mode="determinate")
        self.progress.pack(fill=tk.X, pady=(2, 2))
        self.export_log = tk.Text(parent, height=15, wrap="word")
        self.export_log.pack(fill=tk.BOTH, expand=True, pady=(4, 0))
        self.log("Ready. Open H5 and play/preview before export.")

    def set_contrast_preset(self, low: float, high: float):
        self.auto_contrast_var.set(True)
        self.p_low_var.set(float(low))
        self.p_high_var.set(float(high))
        self.display_current_frame()

    def use_current_as_manual_range(self):
        vmin, vmax = self.last_display_limits
        if vmin is None or vmax is None:
            self.display_current_frame()
            vmin, vmax = self.last_display_limits
        if vmin is None or vmax is None:
            return
        self.auto_contrast_var.set(False)
        self.vmin_var.set(f"{float(vmin):.6g}")
        self.vmax_var.set(f"{float(vmax):.6g}")
        self.display_current_frame()

    def adjust_manual_range(self, factor: float):
        # factor < 1 narrows contrast; factor > 1 widens contrast around current center.
        vmin, vmax = self.last_display_limits
        if vmin is None or vmax is None:
            self.display_current_frame()
            vmin, vmax = self.last_display_limits
        if vmin is None or vmax is None:
            return
        c = 0.5 * (float(vmin) + float(vmax))
        half = 0.5 * (float(vmax) - float(vmin)) * float(factor)
        if half <= 0:
            half = 1.0
        self.auto_contrast_var.set(False)
        self.vmin_var.set(f"{c - half:.6g}")
        self.vmax_var.set(f"{c + half:.6g}")
        self.display_current_frame()

    def apply_export_preset(self):
        preset = self.export_preset_var.get()
        if preset == "今見ている1枚だけ":
            self.export_mode_var.set("current frame")
            self.set_export_current()
        elif preset == "範囲を間引き保存":
            self.export_mode_var.set("sample every n")
            self.export_every_var.set(1000)
        elif preset == "1000枚ごと平均":
            self.export_mode_var.set("aggregate groups")
            self.export_group_var.set(1000)
            self.export_agg_var.set("mean")
        elif preset == "全体9分割平均":
            self.export_mode_var.set("split whole into k")
            self.export_split_var.set(9)
            self.export_agg_var.set("mean")
            self.set_export_all()
        elif preset == "範囲を全フレーム保存":
            self.export_mode_var.set("range frames")
            self.export_step_var.set(1)
        self.update_export_hint()

    def update_dtype_hint(self):
        m = self.export_dtype_var.get()
        hints = {
            "raw": "raw: 値をそのまま保存。解析向きだがファイルが大きく、viewerによって表示が暗い場合があります。",
            "uint16_clip": "uint16_clip: 解析・再利用向け。0-65535にクリップして保存。通常はこれが安全。",
            "uint8_clip": "uint8_clip: 0-255にクリップ。軽いが強度情報はかなり失われます。",
            "uint8_display": "uint8_display: 今の表示レンジを使って8bit保存。見た目確認用。強度値は保存されません。",
            "rgb_display": "rgb_display: 今の表示レンジ＋colormapをRGBで保存。PowerPoint/報告図向け。強度値は保存されません。",
        }
        self.export_dtype_hint_var.set(hints.get(m, ""))

    # ------------------------------------------------------------------
    # File source
    # ------------------------------------------------------------------
    def choose_file(self):
        p = filedialog.askopenfilename(title="Select H5", filetypes=[("HDF5", "*.h5 *.hdf5 *.nxs"), ("All", "*.*")])
        if p:
            self.path_var.set(p)
            self.scan_source()

    def choose_folder(self):
        p = filedialog.askdirectory(title="Select H5 folder")
        if p:
            self.path_var.set(p)
            self.scan_source()

    def choose_output_folder(self):
        p = filedialog.askdirectory(title="Select output folder")
        if p:
            self.export_out_var.set(p)

    def scan_source(self):
        path_txt = self.path_var.get().strip().strip('"\'')
        if not path_txt:
            messagebox.showwarning("No path", "H5 file or folderを選択してください")
            return
        root = Path(path_txt)
        files = collect_h5_files(root, recursive=self.recursive_var.get())
        self.h5_infos = []
        self.h5_tree.delete(*self.h5_tree.get_children())
        if not files:
            self.status_var.set("No H5 files found")
            return
        self.status_var.set(f"Scanning {len(files)} H5 file(s)...")
        self.root.update_idletasks()
        for p in files:
            info = inspect_h5_file(p)
            self.h5_infos.append(info)
            vals = (info["name"], info["shape"], info["dtype"], info["dataset"] if info["ok"] else info["error"])
            self.h5_tree.insert("", tk.END, values=vals)
        self.status_var.set(f"Found {sum(1 for x in self.h5_infos if x['ok'])}/{len(self.h5_infos)} readable H5 stack(s)")
        if self.h5_infos:
            first_ok = next((i for i, x in enumerate(self.h5_infos) if x["ok"]), 0)
            child = self.h5_tree.get_children()[first_ok]
            self.h5_tree.selection_set(child)
            self.h5_tree.see(child)
            self.on_h5_selected(None)

    def on_h5_selected(self, _event):
        sel = self.h5_tree.selection()
        if not sel:
            return
        idx = self.h5_tree.index(sel[0])
        if idx < 0 or idx >= len(self.h5_infos):
            return
        info = self.h5_infos[idx]
        self.selected_h5_path = Path(info["path"])
        dsets = info.get("datasets", []) or []
        values = [d[0] for d in dsets]
        if not values:
            values = [DEFAULT_DATASET]
        self.dataset_combo["values"] = values
        self.dataset_var.set(info.get("dataset", values[0]))
        self.info_var.set(f"selected: {info['path']}\nshape={info.get('shape')} dtype={info.get('dtype')}\ndataset={self.dataset_var.get()}")

    def open_selected_h5(self):
        if self.selected_h5_path is None:
            messagebox.showwarning("No H5", "H5を選択してください")
            return
        ds_path = self.dataset_var.get().strip()
        try:
            self.stack.open(self.selected_h5_path, ds_path)
            T, H, W = self.stack.shape
            self.current_frame = 0
            self.frame_var.set(0)
            self.frame_spin.config(to=max(0, T - 1))
            self.slider.config(to=max(0, T - 1))
            self.export_start_var.set(0)
            self.export_end_var.set(max(0, T - 1))
            self.export_prefix_var.set(safe_name(self.selected_h5_path.stem))
            if not self.export_out_var.get().strip():
                self.export_out_var.set(str(self.selected_h5_path.parent / "tif_export_selective"))
            self.info_var.set(
                f"open: {self.selected_h5_path}\n"
                f"dataset={ds_path}\nshape=(T={T}, H={H}, W={W}) dtype={self.stack.dtype}\n"
                f"chunks={getattr(self.stack.ds, 'chunks', None)}"
            )
            self.status_var.set(f"Opened {self.selected_h5_path.name}: {T} frames")
            self.display_current_frame(force=True)
            self.update_export_hint()
        except Exception as e:
            messagebox.showerror("Open failed", f"{type(e).__name__}: {e}")

    # ------------------------------------------------------------------
    # Viewer
    # ------------------------------------------------------------------
    def goto_frame(self, idx: int):
        if not self.stack.is_open:
            return
        T = self.stack.shape[0]
        idx = max(0, min(T - 1, int(idx)))
        self.current_frame = idx
        self.frame_var.set(idx)
        self.slider.set(idx)
        self.display_current_frame()

    def goto_last(self):
        if self.stack.is_open:
            self.goto_frame(self.stack.shape[0] - 1)

    def step_frame(self, delta: int):
        if not self.stack.is_open:
            return
        self.goto_frame(self.current_frame + int(delta))

    def on_frame_spin(self):
        self.goto_frame(self.frame_var.get())

    def on_slider(self, value):
        if not self.stack.is_open:
            return
        try:
            idx = int(float(value))
        except Exception:
            return
        if idx != self.current_frame:
            self.current_frame = idx
            self.frame_var.set(idx)
            self.display_current_frame()

    def toggle_play(self):
        if not self.stack.is_open:
            return
        self.playing = not self.playing
        self.play_btn.config(text="⏸ Pause" if self.playing else "▶ Play")
        if self.playing:
            self._play_loop()

    def _play_loop(self):
        if not self.playing or not self.stack.is_open:
            return
        T = self.stack.shape[0]
        step = max(1, int(self.play_step_var.get()))
        nxt = self.current_frame + step
        if nxt >= T:
            nxt = 0
        self.goto_frame(nxt)
        fps = max(0.2, float(self.fps_var.get()))
        delay = int(max(1, 1000.0 / fps))
        self.root.after(delay, self._play_loop)

    def _display_downsample(self, img: np.ndarray) -> np.ndarray:
        arr = img
        if self.flip_y_var.get():
            arr = arr[::-1, :]
        if self.transpose_var.get():
            arr = arr.T
        max_px = max(200, int(self.max_preview_px_var.get()))
        h, w = arr.shape[:2]
        stride = max(1, int(math.ceil(max(h, w) / max_px)))
        if stride > 1:
            arr = arr[::stride, ::stride]
        return arr

    def display_current_frame(self, force: bool = False):
        if not self.stack.is_open:
            self.canvas.delete("all")
            return
        try:
            img = self.stack.read_frame(self.current_frame, use_cache=True)
            disp = self._display_downsample(img)
            auto = self.auto_contrast_var.get()
            vmin = None if auto else parse_float_or_none(self.vmin_var.get())
            vmax = None if auto else parse_float_or_none(self.vmax_var.get())
            u8, used_min, used_max = preview_to_uint8(
                disp,
                vmin=vmin,
                vmax=vmax,
                p_low=float(self.p_low_var.get()),
                p_high=float(self.p_high_var.get()),
                log_view=self.log_view_var.get(),
            )
            self.last_display_limits = (used_min, used_max)
            rgb = apply_colormap_to_u8(u8, self.cmap_var.get(), reverse=self.cmap_reverse_var.get())
            pil = Image.fromarray(rgb, mode="RGB")
            cw = max(1, self.canvas.winfo_width())
            ch = max(1, self.canvas.winfo_height())
            iw, ih = pil.size
            scale = min(cw / iw, ch / ih, 1.0 if max(iw, ih) > 700 else min(cw / iw, ch / ih))
            nw = max(1, int(iw * scale))
            nh = max(1, int(ih * scale))
            if (nw, nh) != pil.size:
                pil = pil.resize((nw, nh), resample=Image.Resampling.NEAREST)
            self.photo = ImageTk.PhotoImage(pil)
            self.canvas.delete("all")
            self.canvas.create_image(cw // 2, ch // 2, image=self.photo, anchor="center")
            self.canvas.create_text(
                10, 10, anchor="nw", fill="#eeeeee",
                text=f"frame {self.current_frame}/{self.stack.shape[0]-1}  shape={self.stack.shape[1:]}  preview={u8.shape}  v=[{used_min:.4g},{used_max:.4g}]  cmap={self.cmap_var.get()}",
                font=("Consolas", 10),
            )
            self.status_var.set(f"frame {self.current_frame}/{self.stack.shape[0]-1} | vmin={used_min:.4g} vmax={used_max:.4g}")
        except Exception as e:
            self.status_var.set(f"Display error: {type(e).__name__}: {e}")

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------
    def set_export_current(self):
        self.export_start_var.set(int(self.current_frame))
        self.export_end_var.set(int(self.current_frame))
        self.update_export_hint()

    def set_export_all(self):
        if self.stack.is_open:
            self.export_start_var.set(0)
            self.export_end_var.set(self.stack.shape[0] - 1)
        self.update_export_hint()

    def update_export_hint(self):
        if not self.stack.is_open:
            self.export_hint_var.set("Open H5 first.")
            return
        T = self.stack.shape[0]
        s = clamp_int(self.export_start_var.get(), 0, T - 1, 0)
        e = clamp_int(self.export_end_var.get(), 0, T - 1, T - 1)
        if e < s:
            s, e = e, s
        mode = self.export_mode_var.get()
        n = e - s + 1
        if mode == "current frame":
            txt = f"現在の start={s} 1枚だけ保存します。"
        elif mode == "range frames":
            step = max(1, int(self.export_step_var.get()))
            cnt = len(range(s, e + 1, step))
            txt = f"start-end内を step={step} で {cnt}枚保存します。全出しに近いので注意。"
        elif mode == "sample every n":
            every = max(1, int(self.export_every_var.get()))
            cnt = len(range(s, e + 1, every))
            txt = f"{every}枚ごとに1枚サンプル保存します。出力 {cnt}枚。"
        elif mode == "aggregate groups":
            g = max(1, int(self.export_group_var.get()))
            cnt = math.ceil(n / g)
            txt = f"{g}枚ごとに {self.export_agg_var.get()} で集約し、出力 {cnt}枚。1000枚ごとの軽量保存に向きます。"
        elif mode == "split whole into k":
            k = max(1, int(self.export_split_var.get()))
            txt = f"範囲 {n} frames を {k}分割し、各区間を {self.export_agg_var.get()} で集約。出力は最大 {k}枚。"
        else:
            txt = ""
        txt += f"\n保存タイプ: {self.export_dtype_var.get()} / colormap: {self.cmap_var.get()}"
        self.export_hint_var.set(txt)
        self.update_dtype_hint()

    def _make_export_config(self) -> ExportConfig:
        if not self.stack.is_open or self.stack.path is None:
            raise RuntimeError("H5 is not open")
        out = self.export_out_var.get().strip().strip('"\'')
        if not out:
            raise RuntimeError("Output folder is empty")
        T = self.stack.shape[0]
        s = clamp_int(self.export_start_var.get(), 0, T - 1, 0)
        e = clamp_int(self.export_end_var.get(), 0, T - 1, T - 1)
        if e < s:
            s, e = e, s
        if self.export_mode_var.get() == "current frame":
            s = e = int(self.current_frame)
        vmin, vmax = self.last_display_limits
        return ExportConfig(
            mode=self.export_mode_var.get(),
            out_dir=out,
            start=s,
            end=e,
            step=max(1, int(self.export_step_var.get())),
            every_n=max(1, int(self.export_every_var.get())),
            group_size=max(1, int(self.export_group_var.get())),
            split_k=max(1, int(self.export_split_var.get())),
            agg_method=self.export_agg_var.get(),
            dtype_mode=self.export_dtype_var.get(),
            batch_read=max(1, int(self.export_batch_var.get())),
            display_vmin=vmin,
            display_vmax=vmax,
            cmap_name=self.cmap_var.get(),
            cmap_reverse=bool(self.cmap_reverse_var.get()),
            log_view=bool(self.log_view_var.get()),
            prefix=safe_name(self.export_prefix_var.get() or (self.stack.path.stem if self.stack.path else "export")),
        )

    def start_export(self):
        if self.export_worker is not None and self.export_worker.is_alive():
            messagebox.showwarning("Export running", "既に書き出し中です")
            return
        try:
            cfg = self._make_export_config()
            if cfg.mode == "range frames":
                count = len(range(cfg.start, cfg.end + 1, max(1, cfg.step)))
                if count > 5000:
                    ok = messagebox.askyesno(
                        "Large export",
                        f"{count}枚のTIFを書き出します。かなり重い可能性があります。続行しますか？"
                    )
                    if not ok:
                        return
            self.export_cancel.clear()
            self.progress.config(value=0, maximum=1)
            self.log(f"START export mode={cfg.mode}, range={cfg.start}-{cfg.end}, out={cfg.out_dir}")
            self.export_worker = ExportWorker(self.stack.path, self.stack.dataset_path, cfg, self.export_q, self.export_cancel)
            self.export_worker.start()
        except Exception as e:
            messagebox.showerror("Export failed", f"{type(e).__name__}: {e}")

    def cancel_export(self):
        self.export_cancel.set()
        self.log("Cancel requested. It will stop at the next batch boundary.")

    def _poll_export_queue(self):
        try:
            while True:
                msg = self.export_q.get_nowait()
                typ = msg.get("type")
                if typ == "progress":
                    cur = int(msg.get("current", 0))
                    tot = max(1, int(msg.get("total", 1)))
                    self.progress.config(maximum=tot, value=cur)
                    self.status_var.set(msg.get("message", ""))
                    self.log(msg.get("message", ""))
                elif typ == "done":
                    self.status_var.set(msg.get("message", "done"))
                    self.log(msg.get("message", "done"))
                    self.log(f"manifest: {msg.get('manifest', '')}")
                    messagebox.showinfo("Export done", msg.get("message", "done"))
                elif typ == "error":
                    self.status_var.set(msg.get("message", "error"))
                    self.log("ERROR: " + msg.get("message", ""))
                    messagebox.showerror("Export error", msg.get("message", "error"))
        except queue.Empty:
            pass
        self.root.after(200, self._poll_export_queue)

    def log(self, text: str):
        try:
            self.export_log.insert(tk.END, f"[{datetime.now().strftime('%H:%M:%S')}] {text}\n")
            self.export_log.see(tk.END)
        except Exception:
            pass

    def on_close(self):
        self.playing = False
        self.export_cancel.set()
        self.stack.close()
        self.root.destroy()


def main():
    root = tk.Tk()
    app = AlbulaLiteApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()


if __name__ == "__main__":
    main()
