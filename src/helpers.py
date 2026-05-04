#!/usr/bin/env python3
"""Convert labeled OptiTrack C3D mocap to AMASS-style SMPL-X NPZ with Torch.

This is a Python 3 / PyTorch replacement for the chumpy MoSh++ path used by
``convert_optitrack_to_amass.py``.  It keeps the same high-level two-stage idea:

1. Stage I fits subject shape and a MoSh-like marker layout from sparse frames.
2. Stage II fits per-frame root orientation, body pose and translation.

The optimization itself is done with ``smplx`` and ``torch.optim`` so it can run
on CUDA.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import re
import struct
import sys
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation


SMPLX_FULLPOSE_DOF = 165
SMPLX_BODY_DOF = 63
SMPLX_HAND_DOF = 45
ARM_MARKER_LABELS = {
    "LBSH", "LSHOUP", "LUPA", "LELB", "LIWR", "LOWR", "LFIN",
    "RBSH", "RSHOUP", "RUPA", "RELB", "RIWR", "ROWR", "RFIN",
}
_SMPLX_NO_EYEBALL_VIDS: Optional[np.ndarray] = None


def _parse_c3d_parameters_raw(mocap_fname: Path) -> Dict[Tuple[str, str], object]:
    data = mocap_fname.read_bytes()
    param_start = (data[0] - 1) * 512
    pos = param_start + 4
    groups: Dict[int, str] = {}
    params: Dict[Tuple[str, str], object] = {}
    while pos + 2 <= len(data):
        name_len = struct.unpack("b", data[pos:pos + 1])[0]
        group_id = struct.unpack("b", data[pos + 1:pos + 2])[0]
        if name_len == 0:
            break
        name_start = pos + 2
        offset_pos = name_start + abs(name_len)
        if offset_pos + 2 > len(data):
            break
        name = data[name_start:offset_pos].decode("latin1", errors="replace").strip().upper()
        offset = struct.unpack("<h", data[offset_pos:offset_pos + 2])[0]
        value_pos = offset_pos + 2
        next_pos = offset_pos + offset
        if group_id < 0:
            groups[-group_id] = name
        elif group_id > 0 and value_pos + 2 <= len(data):
            group_name = groups.get(group_id, str(group_id)).upper()
            ptype = struct.unpack("b", data[value_pos:value_pos + 1])[0]
            ndim = data[value_pos + 1]
            dims = list(data[value_pos + 2:value_pos + 2 + ndim])
            raw_start = value_pos + 2 + ndim
            count = int(np.prod(dims)) if dims else 1
            if ptype == -1:
                raw = data[raw_start:raw_start + count]
                if ndim == 2:
                    width, nitems = dims
                    value = [
                        raw[i * width:(i + 1) * width].decode("latin1", errors="replace").strip()
                        for i in range(nitems)
                    ]
                else:
                    value = raw.decode("latin1", errors="replace").strip()
            elif ptype == 2:
                raw = data[raw_start:raw_start + 2 * count]
                arr = np.frombuffer(raw, dtype="<i2", count=count).copy()
                value = arr if dims else int(arr[0])
            elif ptype == 4:
                raw = data[raw_start:raw_start + 4 * count]
                arr = np.frombuffer(raw, dtype="<f4", count=count).copy()
                value = arr if dims else float(arr[0])
            else:
                value = None
            params[(group_name, name)] = value
        if next_pos <= pos:
            break
        pos = next_pos
    return params


def _read_c3d_raw(mocap_fname: Path) -> Tuple[np.ndarray, List[str], float, Optional[str]]:
    params = _parse_c3d_parameters_raw(mocap_fname)
    used = int(params[("POINT", "USED")])
    scale = float(params[("POINT", "SCALE")])
    rate = float(params[("POINT", "RATE")])
    data_start = int(params[("POINT", "DATA_START")])
    labels = [str(x).replace(" ", "") for x in params[("POINT", "LABELS")]]
    units_value = params.get(("POINT", "UNITS"))
    unit = str(units_value).strip().lower() if units_value is not None else None
    frames_value = params.get(("POINT", "LONG_FRAMES"), params.get(("POINT", "FRAMES")))
    frames = int(round(float(np.asarray(frames_value).reshape(-1)[0])))
    byte_offset = (data_start - 1) * 512
    values_per_point = 4
    count = frames * used * values_per_point
    dtype = "<f4" if scale < 0.0 else "<i2"
    raw = np.fromfile(mocap_fname, dtype=dtype, count=count, offset=byte_offset)
    if raw.size != count:
        frames = raw.size // (used * values_per_point)
        raw = raw[:frames * used * values_per_point]
    points = raw.reshape(frames, used, values_per_point).astype(np.float64)
    markers = points[:, :, :3]
    if scale > 0.0:
        markers *= scale
    return markers, labels, rate, unit if unit in {"mm", "cm", "m"} else None


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _sanitize_stem(stem: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("_")


def _ensure_parent(fname: Path) -> Path:
    fname.parent.mkdir(parents=True, exist_ok=True)
    return fname


def _smplx_no_eyeball_vids(num_vertices: int) -> Optional[np.ndarray]:
    """Match MoSh++ TransformedCoeffs, which omits SMPL-X eyeball vertices."""

    global _SMPLX_NO_EYEBALL_VIDS
    if num_vertices != 10475:
        return None
    if _SMPLX_NO_EYEBALL_VIDS is not None:
        return _SMPLX_NO_EYEBALL_VIDS
    eyeball_fname = _repo_root() / "moshpp" / "support_data" / "smplx_eyeballs.npz"
    if not eyeball_fname.exists():
        return None
    eyeballs = np.load(eyeball_fname)["eyeballs"].astype(np.int64)
    mask = np.ones(num_vertices, dtype=bool)
    mask[eyeballs] = False
    _SMPLX_NO_EYEBALL_VIDS = np.flatnonzero(mask).astype(np.int64)
    return _SMPLX_NO_EYEBALL_VIDS


def _torch_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def _read_json(path: Optional[str]) -> Optional[Dict[str, str]]:
    if not path:
        return None
    fname = Path(path).expanduser().resolve()
    if not fname.exists():
        raise FileNotFoundError(f"labels_map_json not found: {fname}")
    data = json.loads(fname.read_text())
    return {str(k): str(v) for k, v in data.items() if v is not None}


def _read_c3d_unit(mocap_fname: Path) -> Optional[str]:
    import ezc3d

    try:
        c3d = ezc3d.c3d(str(mocap_fname), ignore_bad_formatting=True)
        units = c3d["parameters"]["POINT"]["UNITS"]["value"]
        if not units:
            return None
        unit = str(units[0]).strip().lower()
        return unit if unit in {"mm", "cm", "m"} else None
    except ValueError:
        return _read_c3d_raw(mocap_fname)[3]


def _resolve_mocap_unit(mocap_fname: Path, unit_arg: str) -> str:
    if unit_arg != "auto":
        return unit_arg
    return _read_c3d_unit(mocap_fname) or "mm"


def _marker_availability(markers: np.ndarray) -> np.ndarray:
    return np.logical_and(np.isnan(markers).sum(-1) == 0, (markers == 0).sum(-1) != 3)


@dataclass
class MocapData:
    markers: np.ndarray
    labels: List[str]
    frame_rate: float
    unit: str

    @property
    def time_length(self) -> float:
        return float(self.markers.shape[0]) / float(self.frame_rate)


def load_mocap(
    mocap_fname: Path,
    mocap_unit: str,
    labels_map: Optional[Dict[str, str]] = None,
    start_fidx: int = 0,
    end_fidx: int = -1,
    ds_rate: int = 1,
) -> MocapData:
    import ezc3d

    if ds_rate < 1:
        raise ValueError(f"ds_rate must be >= 1, got {ds_rate}")

    try:
        c3d = ezc3d.c3d(str(mocap_fname), ignore_bad_formatting=True)
        markers = c3d["data"]["points"][:3].transpose(2, 1, 0).astype(np.float64)
        labels = [str(x).replace(" ", "") for x in c3d["parameters"]["POINT"]["LABELS"]["value"]]
        frame_rate = float(c3d["parameters"]["POINT"]["RATE"]["value"][0]) / float(ds_rate)
    except ValueError:
        markers, labels, raw_frame_rate, _ = _read_c3d_raw(mocap_fname)
        frame_rate = raw_frame_rate / float(ds_rate)
    labels = [label.split(":")[-1] for label in labels]
    if len(labels) < markers.shape[1]:
        labels.extend([f"*{i}" for i in range(len(labels), markers.shape[1])])
    if labels_map:
        labels = [labels_map.get(label, label) for label in labels]

    scale = {"mm": 1000.0, "cm": 100.0, "m": 1.0}[mocap_unit]
    markers = markers / scale
    available = _marker_availability(markers)
    markers[~available] = np.nan

    nframes = markers.shape[0]
    start = max(0, int(start_fidx))
    end = nframes if end_fidx == -1 else min(int(end_fidx), nframes)
    if start >= nframes:
        raise ValueError(f"start_fidx={start} is out of range for sequence length={nframes}")
    if end <= start:
        raise ValueError(f"Invalid frame window: start_fidx={start}, end_fidx={end}")
    markers = markers[start:end:ds_rate]

    return MocapData(markers=markers, labels=labels, frame_rate=frame_rate, unit=mocap_unit)


def auto_resolve_headside_label(mocap_fname: Path, labels_map: Optional[Dict[str, str]]) -> Optional[Dict[str, str]]:
    if not labels_map:
        return labels_map
    if labels_map.get("HeadSide") not in {"AUTO_HEADSIDE", "AUTO_SIDE", "AUTO"}:
        return labels_map

    import ezc3d

    try:
        c3d = ezc3d.c3d(str(mocap_fname), ignore_bad_formatting=True)
        labels = [str(x).replace(" ", "").split(":")[-1] for x in c3d["parameters"]["POINT"]["LABELS"]["value"]]
        pts = c3d["data"]["points"][:3]
    except ValueError:
        markers_raw, labels_raw, _, _ = _read_c3d_raw(mocap_fname)
        labels = [str(x).replace(" ", "").split(":")[-1] for x in labels_raw]
        pts = markers_raw.transpose(2, 1, 0)
    idx = {label: i for i, label in enumerate(labels)}
    required = ["HeadSide", "HeadFront", "LShoulderBack", "RShoulderBack"]
    if any(label not in idx for label in required):
        labels_map["HeadSide"] = "LBHD"
        return labels_map

    hs = pts[:, idx["HeadSide"], :]
    hf = pts[:, idx["HeadFront"], :]
    ls = pts[:, idx["LShoulderBack"], :]
    rs = pts[:, idx["RShoulderBack"], :]
    valid = (
        (np.linalg.norm(hs, axis=0) > 0)
        & (np.linalg.norm(hf, axis=0) > 0)
        & (np.linalg.norm(ls, axis=0) > 0)
        & (np.linalg.norm(rs, axis=0) > 0)
        & np.isfinite(hs).all(axis=0)
        & np.isfinite(hf).all(axis=0)
        & np.isfinite(ls).all(axis=0)
        & np.isfinite(rs).all(axis=0)
    )
    if valid.sum() < 10:
        labels_map["HeadSide"] = "LBHD"
        return labels_map
    score = ((ls - rs)[:, valid] * (hs - hf)[:, valid]).sum(axis=0)
    labels_map["HeadSide"] = "LBHD" if float(np.median(score)) >= 0.0 else "RBHD"
    return labels_map


def load_moshpp_marker_database() -> Tuple[Dict[str, Dict[str, int]], Dict[str, List[str]]]:
    """Load MoSh++ marker vertex tables without importing chumpy-era deps."""

    marker_vids_py = _repo_root() / "moshpp" / "src" / "moshpp" / "marker_layout" / "marker_vids.py"
    support_dir = _repo_root() / "moshpp" / "support_data"
    if not marker_vids_py.exists():
        raise FileNotFoundError(f"MoSh++ marker table not found: {marker_vids_py}")
    mapping_fname = support_dir / "smplx_fit2_smplh.npz"
    smplh2smplx_map = np.load(mapping_fname)["smh2smhf"]

    def smplh2smplx(vids):
        if isinstance(vids, int):
            return int(smplh2smplx_map[vids])
        return [int(smplh2smplx_map[int(vid)]) for vid in vids]

    text = marker_vids_py.read_text()
    start = text.index("all_marker_vids")
    namespace = {"np": np, "smplh2smplx": smplh2smplx}
    exec(text[start:], namespace)  # noqa: S102 - trusted local project data table.
    return namespace["all_marker_vids"], namespace["marker_type_labels"]


def marker_labels_to_layout(
    chosen_markers: Sequence[str],
    marker_layout_fname: Path,
    surface_model_type: str = "smplx",
    wrist_markers_on_stick: bool = False,
    separate_types: Sequence[str] = ("body", "face", "finger"),
) -> Dict:
    all_marker_vids, marker_type_labels = load_moshpp_marker_database()
    if surface_model_type not in all_marker_vids:
        raise ValueError(f"No marker database for surface_model_type={surface_model_type}")

    mean_dist_from_skin = {
        "wrist": 0.039,
        "body": 0.0095,
        "face": 0.0002,
        "finger_right": 0.0002,
        "finger_left": 0.0002,
    }
    has_face = surface_model_type in {"smplx", "flame"} and "face" in separate_types
    has_finger = surface_model_type in {"smplh", "smplx", "mano"} and "finger" in separate_types
    has_body = surface_model_type not in {"mano", "flame"}

    marker_vids = OrderedDict()
    unknown = []
    for label in sorted(set(chosen_markers)):
        if label.startswith("*") or label.startswith("Unlabeled"):
            continue
        vid = all_marker_vids[surface_model_type].get(label)
        if vid is None:
            unknown.append(label)
        else:
            marker_vids[label] = int(vid)
    if unknown:
        print(f"Warning: skipped unknown marker labels: {unknown}", file=sys.stderr)
    if not marker_vids:
        raise RuntimeError("No known markers remain after applying the marker database.")

    marker_type_mask: Dict[str, np.ndarray] = {}
    if has_face:
        marker_type_mask["face"] = np.zeros(len(marker_vids), dtype=bool)
    if has_finger:
        marker_type_mask["finger_left"] = np.zeros(len(marker_vids), dtype=bool)
        marker_type_mask["finger_right"] = np.zeros(len(marker_vids), dtype=bool)
    if has_body:
        marker_type_mask["body"] = np.zeros(len(marker_vids), dtype=bool)
    if wrist_markers_on_stick:
        marker_type_mask["wrist"] = np.zeros(len(marker_vids), dtype=bool)

    for lid, label in enumerate(marker_vids):
        if has_face and label in marker_type_labels["face"]:
            marker_type_mask["face"][lid] = True
        elif has_finger and label in marker_type_labels["finger_left"]:
            marker_type_mask["finger_left"][lid] = True
        elif has_finger and label in marker_type_labels["finger_right"]:
            marker_type_mask["finger_right"][lid] = True
        elif wrist_markers_on_stick and label in marker_type_labels["wrist"]:
            marker_type_mask["wrist"][lid] = True
        elif has_body:
            marker_type_mask["body"][lid] = True
        else:
            raise ValueError(f"Marker {label} could not be assigned to any marker type.")

    markersets = []
    for marker_type, mask in marker_type_mask.items():
        if not mask.any():
            continue
        labels = np.array(list(marker_vids.keys()))[mask]
        markersets.append(
            {
                "indices": {label: int(marker_vids[label]) for label in labels},
                "distance_from_skin": float(mean_dist_from_skin[marker_type]),
                "type": marker_type,
            }
        )
    layout = {"surface_model_type": surface_model_type, "markersets": markersets}
    _ensure_parent(marker_layout_fname).write_text(json.dumps(layout, indent=2, sort_keys=True))
    return load_marker_layout(marker_layout_fname)


def load_marker_layout(marker_layout_fname: Path) -> Dict:
    data = json.loads(marker_layout_fname.read_text())
    marker_vids = OrderedDict()
    marker_type = OrderedDict()
    marker_type_mask = OrderedDict()
    m2b_distance = {}
    for markerset in sorted(data["markersets"], key=lambda item: item["type"]):
        mtype = markerset["type"]
        m2b_distance[mtype] = float(markerset.get("distance_from_skin", 0.0095))
        for label in sorted(markerset["indices"]):
            if label in marker_vids:
                raise ValueError(f"Duplicate marker label in layout: {label}")
            marker_vids[label] = int(markerset["indices"][label])
            marker_type[label] = mtype
    labels = list(marker_vids.keys())
    for mtype in sorted(m2b_distance):
        marker_type_mask[mtype] = np.array([marker_type[label] == mtype for label in labels], dtype=bool)
    return {
        "marker_vids": marker_vids,
        "marker_type": marker_type,
        "marker_type_mask": marker_type_mask,
        "m2b_distance": m2b_distance,
        "surface_model_type": data.get("surface_model_type", "smplx"),
        "marker_layout_fname": str(marker_layout_fname),
    }


def resolve_model_file(model_base_dir: Path, support_base_dir: Path, gender: str) -> Path:
    candidates = [
        model_base_dir / "smplx" / gender / "model.pkl",
        model_base_dir / gender / "model.pkl",
        support_base_dir / "smplx" / gender / "model.pkl",
        _repo_root() / "support_files" / "smplx" / gender / "model.pkl",
        _repo_root() / "smplx_models" / gender / "model.pkl",
    ]
    for candidate in candidates:
        if candidate.exists():
            try:
                from smplx.body_models import SMPLX

                SMPLX(str(candidate), gender=gender, batch_size=1, num_betas=16, use_pca=False, flat_hand_mean=False, ext="pkl")
                return candidate
            except Exception:
                continue
    raise FileNotFoundError(
        "Could not find a SMPL-X model.pkl loadable by the smplx library. "
        "The files under smplx_models appear to be MoSh++-style and may miss hand PCA keys; "
        "support_files/smplx/<gender>/model.pkl works in this workspace."
    )


def create_smplx_model(model_file: Path, batch_size: int, gender: str, num_betas: int, device: torch.device):
    from smplx.body_models import SMPLX

    model = SMPLX(
        str(model_file),
        gender=gender,
        batch_size=batch_size,
        num_betas=num_betas,
        use_pca=False,
        flat_hand_mean=False,
        ext="pkl",
    )
    return model.to(device)


def smplx_hand_mean(model, batch_size: int) -> torch.Tensor:
    hand_mean = torch.cat([model.left_hand_mean, model.right_hand_mean], dim=0)
    return hand_mean.reshape(1, -1).expand(batch_size, -1)


def make_face_tensor(model, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(np.asarray(model.faces, dtype=np.int64), dtype=torch.long, device=device)


def vertex_normals(vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    tris = vertices[:, faces]  # B, F, 3, 3
    face_normals = torch.cross(tris[:, :, 1] - tris[:, :, 0], tris[:, :, 2] - tris[:, :, 0], dim=-1)
    normals = torch.zeros_like(vertices)
    for corner in range(3):
        idx = faces[:, corner][None, :, None].expand(vertices.shape[0], -1, 3)
        normals.scatter_add_(1, idx, face_normals)
    return F.normalize(normals, dim=-1, eps=1e-8)


def build_neighbor_anchors(faces: np.ndarray, marker_vids: Sequence[int]) -> np.ndarray:
    neighbors: Dict[int, List[int]] = {}
    for tri in faces.astype(np.int64):
        a, b, c = [int(x) for x in tri]
        neighbors.setdefault(a, []).extend([b, c])
        neighbors.setdefault(b, []).extend([a, c])
        neighbors.setdefault(c, []).extend([a, b])
    anchor = []
    for vid in marker_vids:
        unique = sorted(set(neighbors.get(int(vid), [])))
        if not unique:
            anchor.append(int(vid))
        else:
            anchor.append(int(unique[0]))
    return np.asarray(anchor, dtype=np.int64)


def vertex_normals_np(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    tris = vertices[faces]
    face_normals = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    normals = np.zeros_like(vertices)
    for corner in range(3):
        np.add.at(normals, faces[:, corner], face_normals)
    norm = np.linalg.norm(normals, axis=1, keepdims=True)
    return normals / np.maximum(norm, 1e-8)


def reanchor_marker_coeffs(
    canonical_vertices: np.ndarray,
    faces: np.ndarray,
    markers_latent: np.ndarray,
    labels: Optional[Sequence[str]] = None,
    initial_anchor_vids: Optional[Sequence[int]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    distances = np.linalg.norm(canonical_vertices[None, :, :] - markers_latent[:, None, :], axis=-1)
    anchor_vids = []
    for marker_idx in range(markers_latent.shape[0]):
        cur_dist = distances[marker_idx].copy()
        if labels is not None and initial_anchor_vids is not None:
            label = labels[marker_idx]
            if label.startswith(("L", "R")):
                layout_x = canonical_vertices[int(initial_anchor_vids[marker_idx]), 0]
                if abs(layout_x) > 1e-5:
                    same_side = np.sign(canonical_vertices[:, 0]) == np.sign(layout_x)
                    if same_side.any():
                        cur_dist[~same_side] = np.inf
        anchor_vids.append(int(np.argmin(cur_dist)))
    anchor_vids = np.asarray(anchor_vids, dtype=np.int64)
    tangent_vids = build_neighbor_anchors(faces, anchor_vids)
    normals = vertex_normals_np(canonical_vertices, faces)
    v0 = canonical_vertices[anchor_vids]
    n0 = normals[anchor_vids]
    tangent = canonical_vertices[tangent_vids] - v0
    tangent = tangent - np.sum(tangent * n0, axis=-1, keepdims=True) * n0
    f1 = tangent / np.maximum(np.linalg.norm(tangent, axis=-1, keepdims=True), 1e-8)
    f2 = np.cross(n0, f1)
    f2 = f2 / np.maximum(np.linalg.norm(f2, axis=-1, keepdims=True), 1e-8)
    diff = markers_latent - v0
    coeffs = np.stack(
        [
            np.sum(diff * f1, axis=-1),
            np.sum(diff * f2, axis=-1),
            np.sum(diff * n0, axis=-1),
        ],
        axis=1,
    ).astype(np.float64)
    return anchor_vids, tangent_vids, coeffs


def reconstruct_markers(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    marker_vids: torch.Tensor,
    tangent_vids: torch.Tensor,
    coeffs: torch.Tensor,
) -> torch.Tensor:
    normals = vertex_normals(vertices, faces)
    v0 = vertices[:, marker_vids]
    n0 = normals[:, marker_vids]
    tangent = vertices[:, tangent_vids] - v0
    tangent = tangent - (tangent * n0).sum(dim=-1, keepdim=True) * n0
    f1 = F.normalize(tangent, dim=-1, eps=1e-8)
    f2 = F.normalize(torch.cross(n0, f1, dim=-1), dim=-1, eps=1e-8)
    f3 = n0
    return v0 + coeffs[None, :, 0:1] * f1 + coeffs[None, :, 1:2] * f2 + coeffs[None, :, 2:3] * f3


def marker_coeffs_from_latent(
    canonical_vertices: torch.Tensor,
    faces: torch.Tensor,
    marker_vids: torch.Tensor,
    tangent_vids: torch.Tensor,
    markers_latent: torch.Tensor,
) -> torch.Tensor:
    if canonical_vertices.ndim == 3:
        canonical_vertices = canonical_vertices[0]
    normals = vertex_normals(canonical_vertices[None], faces)[0]
    v0 = canonical_vertices[marker_vids]
    n0 = normals[marker_vids]
    tangent = canonical_vertices[tangent_vids] - v0
    tangent = tangent - (tangent * n0).sum(dim=-1, keepdim=True) * n0
    f1 = F.normalize(tangent, dim=-1, eps=1e-8)
    f2 = F.normalize(torch.cross(n0, f1, dim=-1), dim=-1, eps=1e-8)
    diff = markers_latent - v0
    return torch.stack(
        [
            (diff * f1).sum(dim=-1),
            (diff * f2).sum(dim=-1),
            (diff * n0).sum(dim=-1),
        ],
        dim=-1,
    )


def marker_coeffs_from_latent_nn(
    canonical_vertices: torch.Tensor,
    markers_latent: torch.Tensor,
    num_neighbors: int = 8,
    exclude_eyeballs: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """MoSh++ TransformedCoeffs analogue using nearest-neighbour local frames."""

    if canonical_vertices.ndim == 3:
        canonical_vertices = canonical_vertices[0]
    vertex_ids_np = _smplx_no_eyeball_vids(int(canonical_vertices.shape[0])) if exclude_eyeballs else None
    if vertex_ids_np is None:
        search_vertices = canonical_vertices
        vertex_ids = None
    else:
        vertex_ids = torch.as_tensor(vertex_ids_np, dtype=torch.long, device=canonical_vertices.device)
        search_vertices = canonical_vertices[vertex_ids]
    nearest_local = torch.topk(
        torch.cdist(markers_latent[None], search_vertices[None])[0],
        k=min(max(3, int(num_neighbors)), int(search_vertices.shape[0])),
        largest=False,
        dim=1,
    ).indices
    nearest = vertex_ids[nearest_local] if vertex_ids is not None else nearest_local
    v0 = canonical_vertices[nearest[:, 0]]
    e1 = canonical_vertices[nearest[:, 1]] - v0
    e2 = canonical_vertices[nearest[:, 2]] - v0
    for candidate_idx in range(3, nearest.shape[1]):
        candidate = canonical_vertices[nearest[:, candidate_idx]] - v0
        cross_norm = torch.linalg.norm(torch.cross(e1, e2, dim=-1), dim=-1)
        replace = cross_norm < 1e-10
        if not bool(replace.any().detach().cpu()):
            break
        e2 = torch.where(replace[:, None], candidate, e2)
        nearest = nearest.clone()
        nearest[:, 2] = torch.where(replace, nearest[:, candidate_idx], nearest[:, 2])
    f1 = F.normalize(e1, dim=-1, eps=1e-8)
    f2 = F.normalize(torch.cross(e1, e2, dim=-1), dim=-1, eps=1e-8)
    f3 = torch.cross(f1, f2, dim=-1)
    diff = markers_latent - v0
    coeffs = torch.stack(
        [
            (diff * f1).sum(dim=-1),
            (diff * f2).sum(dim=-1),
            (diff * f3).sum(dim=-1),
        ],
        dim=-1,
    )
    return coeffs, nearest


def marker_coeffs_from_latent_fixed_nn(
    canonical_vertices: torch.Tensor,
    markers_latent: torch.Tensor,
    nearest_vids: torch.Tensor,
) -> torch.Tensor:
    if canonical_vertices.ndim == 3:
        canonical_vertices = canonical_vertices[0]
    v0 = canonical_vertices[nearest_vids[:, 0]]
    e1 = canonical_vertices[nearest_vids[:, 1]] - v0
    e2 = canonical_vertices[nearest_vids[:, 2]] - v0
    f1 = F.normalize(e1, dim=-1, eps=1e-8)
    f2 = F.normalize(torch.cross(e1, e2, dim=-1), dim=-1, eps=1e-8)
    f3 = torch.cross(f1, f2, dim=-1)
    diff = markers_latent - v0
    return torch.stack(
        [
            (diff * f1).sum(dim=-1),
            (diff * f2).sum(dim=-1),
            (diff * f3).sum(dim=-1),
        ],
        dim=-1,
    )


def reconstruct_markers_nn(
    vertices: torch.Tensor,
    nearest_vids: torch.Tensor,
    coeffs: torch.Tensor,
) -> torch.Tensor:
    v0 = vertices[:, nearest_vids[:, 0]]
    e1 = vertices[:, nearest_vids[:, 1]] - v0
    e2 = vertices[:, nearest_vids[:, 2]] - v0
    f1 = F.normalize(e1, dim=-1, eps=1e-8)
    f2 = F.normalize(torch.cross(e1, e2, dim=-1), dim=-1, eps=1e-8)
    f3 = torch.cross(f1, f2, dim=-1)
    return v0 + coeffs[None, :, 0:1] * f1 + coeffs[None, :, 1:2] * f2 + coeffs[None, :, 2:3] * f3


def latent_surface_distance_vertex_normal(
    canonical_vertices: torch.Tensor,
    faces: torch.Tensor,
    markers_latent: torch.Tensor,
    desired_distances: torch.Tensor,
) -> torch.Tensor:
    """Approximate MoSh++ PtsToMesh signed distance with nearest vertex normals."""

    if canonical_vertices.ndim == 3:
        canonical_vertices = canonical_vertices[0]
    nearest = torch.argmin(torch.cdist(markers_latent[None], canonical_vertices[None])[0], dim=1)
    normals = vertex_normals(canonical_vertices[None], faces)[0]
    v0 = canonical_vertices[nearest]
    n0 = normals[nearest]
    signed = torch.sum((markers_latent - v0) * n0, dim=-1)
    return signed - desired_distances


def latent_surface_distance_fixed_vertex_normal(
    canonical_vertices: torch.Tensor,
    faces: torch.Tensor,
    markers_latent: torch.Tensor,
    desired_distances: torch.Tensor,
    nearest: torch.Tensor,
) -> torch.Tensor:
    if canonical_vertices.ndim == 3:
        canonical_vertices = canonical_vertices[0]
    normals = vertex_normals(canonical_vertices[None], faces)[0]
    v0 = canonical_vertices[nearest]
    n0 = normals[nearest]
    signed = torch.sum((markers_latent - v0) * n0, dim=-1)
    return signed - desired_distances


def latent_surface_distance_triangles(
    canonical_vertices: torch.Tensor,
    faces: torch.Tensor,
    markers_latent: torch.Tensor,
    desired_distances: torch.Tensor,
) -> torch.Tensor:
    """Closest point-to-triangle signed distance, closer to MoSh++ PtsToMesh."""

    if canonical_vertices.ndim == 3:
        canonical_vertices = canonical_vertices[0]
    tris = canonical_vertices[faces]
    a = tris[:, 0]
    b = tris[:, 1]
    c = tris[:, 2]
    ab = b - a
    ac = c - a
    bc = c - b
    face_normals = F.normalize(torch.cross(ab, ac, dim=-1), dim=-1, eps=1e-8)
    vert_normals = vertex_normals(canonical_vertices[None], faces)[0]
    vn_a = vert_normals[faces[:, 0]]
    vn_b = vert_normals[faces[:, 1]]
    vn_c = vert_normals[faces[:, 2]]

    p = markers_latent[:, None, :]
    ap = p - a[None]
    d1 = torch.sum(ab[None] * ap, dim=-1)
    d2 = torch.sum(ac[None] * ap, dim=-1)

    bp = p - b[None]
    d3 = torch.sum(ab[None] * bp, dim=-1)
    d4 = torch.sum(ac[None] * bp, dim=-1)

    cp = p - c[None]
    d5 = torch.sum(ab[None] * cp, dim=-1)
    d6 = torch.sum(ac[None] * cp, dim=-1)

    vc = d1 * d4 - d3 * d2
    vb = d5 * d2 - d1 * d6
    va = d3 * d6 - d5 * d4

    closest = torch.empty_like(p.expand(-1, faces.shape[0], -1))
    normal = face_normals[None].expand_as(closest)

    # Face region fallback.
    denom = (va + vb + vc).clamp_min(1e-12)
    v = vb / denom
    w = vc / denom
    closest_face = a[None] + ab[None] * v[..., None] + ac[None] * w[..., None]
    closest.copy_(closest_face)

    # Edge BC.
    edge_bc = (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0)
    w_bc = ((d4 - d3) / ((d4 - d3) + (d5 - d6)).clamp_min(1e-12)).clamp(0.0, 1.0)
    closest_bc = b[None] + bc[None] * w_bc[..., None]
    normal_bc = F.normalize(vn_b[None] + vn_c[None], dim=-1, eps=1e-8)
    closest = torch.where(edge_bc[..., None], closest_bc, closest)
    normal = torch.where(edge_bc[..., None], normal_bc.expand_as(normal), normal)

    # Edge AC.
    edge_ac = (vb <= 0) & (d2 >= 0) & (d6 <= 0)
    w_ac = (d2 / (d2 - d6).clamp_min(1e-12)).clamp(0.0, 1.0)
    closest_ac = a[None] + ac[None] * w_ac[..., None]
    normal_ac = F.normalize(vn_a[None] + vn_c[None], dim=-1, eps=1e-8)
    closest = torch.where(edge_ac[..., None], closest_ac, closest)
    normal = torch.where(edge_ac[..., None], normal_ac.expand_as(normal), normal)

    # Edge AB.
    edge_ab = (vc <= 0) & (d1 >= 0) & (d3 <= 0)
    w_ab = (d1 / (d1 - d3).clamp_min(1e-12)).clamp(0.0, 1.0)
    closest_ab = a[None] + ab[None] * w_ab[..., None]
    normal_ab = F.normalize(vn_a[None] + vn_b[None], dim=-1, eps=1e-8)
    closest = torch.where(edge_ab[..., None], closest_ab, closest)
    normal = torch.where(edge_ab[..., None], normal_ab.expand_as(normal), normal)

    # Vertex regions.
    vert_a = (d1 <= 0) & (d2 <= 0)
    vert_b = (d3 >= 0) & (d4 <= d3)
    vert_c = (d6 >= 0) & (d5 <= d6)
    closest = torch.where(vert_a[..., None], a[None].expand_as(closest), closest)
    normal = torch.where(vert_a[..., None], vn_a[None].expand_as(normal), normal)
    closest = torch.where(vert_b[..., None], b[None].expand_as(closest), closest)
    normal = torch.where(vert_b[..., None], vn_b[None].expand_as(normal), normal)
    closest = torch.where(vert_c[..., None], c[None].expand_as(closest), closest)
    normal = torch.where(vert_c[..., None], vn_c[None].expand_as(normal), normal)

    diff = p - closest
    dist_sq = torch.sum(diff * diff, dim=-1)
    best = torch.argmin(dist_sq, dim=1)
    row = torch.arange(markers_latent.shape[0], device=markers_latent.device)
    best_diff = diff[row, best]
    best_normal = normal[row, best]
    signed = torch.linalg.norm(best_diff, dim=-1) * torch.sign(torch.sum(best_diff * best_normal, dim=-1))
    return signed - desired_distances


def latent_surface_residual(
    canonical_vertices: torch.Tensor,
    faces: torch.Tensor,
    markers_latent: torch.Tensor,
    desired_distances: torch.Tensor,
    mode: str,
) -> torch.Tensor:
    if mode == "triangles":
        return latent_surface_distance_triangles(canonical_vertices, faces, markers_latent, desired_distances)
    return latent_surface_distance_vertex_normal(canonical_vertices, faces, markers_latent, desired_distances)


def smplx_forward(
    model,
    betas: torch.Tensor,
    body_pose: torch.Tensor,
    global_orient: torch.Tensor,
    transl: torch.Tensor,
    batch_size: int,
) -> torch.Tensor:
    zeros_hand = torch.zeros((batch_size, SMPLX_HAND_DOF), dtype=body_pose.dtype, device=body_pose.device)
    zeros_3 = torch.zeros((batch_size, 3), dtype=body_pose.dtype, device=body_pose.device)
    output = model(
        betas=betas,
        body_pose=body_pose,
        global_orient=global_orient,
        transl=transl,
        left_hand_pose=zeros_hand,
        right_hand_pose=zeros_hand,
        jaw_pose=zeros_3,
        leye_pose=zeros_3,
        reye_pose=zeros_3,
        return_verts=True,
    )
    return output.vertices


class BodyPosePrior:
    def __init__(self, prior_fname: Optional[Path], device: torch.device, dtype: torch.dtype) -> None:
        self.valid = False
        if prior_fname is None or not prior_fname.exists():
            return
        with open(prior_fname, "rb") as f:
            data = pickle.load(f, encoding="latin1")
        covars = np.asarray(data["covars"], dtype=np.float64)
        means = np.asarray(data["means"], dtype=np.float64)
        weights = np.asarray(data["weights"], dtype=np.float64)
        npose = SMPLX_BODY_DOF
        covars = covars[:, :npose, :npose]
        means = means[:, :npose]
        covars = covars + np.eye(covars.shape[-1])[None] * 1e-6
        precisions = np.linalg.inv(covars)
        chols = np.linalg.cholesky(precisions)
        sqrdets = np.asarray([np.sqrt(np.linalg.det(cov)) for cov in covars], dtype=np.float64)
        const = (2.0 * np.pi) ** (npose / 2.0)
        adjusted_weights = weights / (const * (sqrdets / sqrdets.min()))
        adjusted_weights = np.maximum(adjusted_weights, 1e-300)
        self.precisions = torch.as_tensor(precisions, dtype=dtype, device=device)
        self.chols = torch.as_tensor(chols, dtype=dtype, device=device)
        self.means = torch.as_tensor(means, dtype=dtype, device=device)
        self.log_weights = torch.as_tensor(np.log(adjusted_weights), dtype=dtype, device=device)
        self.valid = True

    def __call__(self, body_pose: torch.Tensor) -> torch.Tensor:
        return self.sse(body_pose) / max(1, body_pose.shape[0])

    def sse(self, body_pose: torch.Tensor) -> torch.Tensor:
        if not self.valid:
            return torch.sum(body_pose * body_pose)
        diff = body_pose[:, None, :] - self.means[None, :, :]
        whitened = torch.einsum("bki,kij->bkj", diff, self.chols)
        residual_sse = 0.5 * torch.sum(whitened * whitened, dim=-1) - self.log_weights[None, :]
        return torch.sum(torch.min(residual_sse, dim=1).values)

    def residual(self, body_pose: torch.Tensor) -> torch.Tensor:
        if not self.valid:
            return body_pose.reshape(-1)
        diff = body_pose[:, None, :] - self.means[None, :, :]
        whitened = torch.einsum("bki,kij->bkj", diff, self.chols)
        residual_sse = 0.5 * torch.sum(whitened * whitened, dim=-1) - self.log_weights[None, :]
        comp = torch.argmin(residual_sse, dim=1)
        batch = torch.arange(body_pose.shape[0], device=body_pose.device)
        selected = math.sqrt(0.5) * whitened[batch, comp]
        weight_res = torch.sqrt(torch.clamp(-self.log_weights[comp], min=0.0))[:, None]
        return torch.cat([selected.reshape(-1), weight_res.reshape(-1)], dim=0)


def robust_gmof(residual: torch.Tensor, sigma: float) -> torch.Tensor:
    sq = residual * residual
    sig2 = sigma * sigma
    return (sig2 * sq) / (sig2 + sq)


def marker_huber_loss(distances: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma <= 0:
        return distances * distances
    quad = 0.5 * distances * distances / sigma
    linear = distances - 0.5 * sigma
    return torch.where(distances < sigma, quad, linear)


def masked_marker_loss(
    pred: torch.Tensor,
    obs: torch.Tensor,
    mask: torch.Tensor,
    sigma: float,
    marker_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    residual = pred - torch.nan_to_num(obs, nan=0.0)
    distances = torch.linalg.norm(residual, dim=-1)
    losses = marker_huber_loss(distances, sigma=sigma)
    if marker_weights is not None:
        losses = losses * marker_weights[None]
        denom = (mask.float() * marker_weights[None]).sum()
    else:
        denom = mask.float().sum()
    if float(denom.detach().cpu()) <= 0:
        return pred.sum() * 0.0
    return (losses * mask.float()).sum() / denom


def masked_marker_mse(
    pred: torch.Tensor,
    obs: torch.Tensor,
    mask: torch.Tensor,
    marker_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    residual = pred - torch.nan_to_num(obs, nan=0.0)
    sq = (residual * residual).sum(dim=-1)
    if marker_weights is not None:
        sq = sq * marker_weights[None]
        denom = (mask.float() * marker_weights[None]).sum() * 3.0
    else:
        denom = mask.float().sum() * 3.0
    if float(denom.detach().cpu()) <= 0:
        return pred.sum() * 0.0
    return (sq * mask.float()).sum() / denom


def masked_marker_sse(
    pred: torch.Tensor,
    obs: torch.Tensor,
    mask: torch.Tensor,
    marker_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    residual = pred - torch.nan_to_num(obs, nan=0.0)
    sq = (residual * residual).sum(dim=-1)
    if marker_weights is not None:
        sq = sq * marker_weights[None]
    return (sq * mask.float()).sum()


def pose_velocity_extrap_sse(
    pose_seq: torch.Tensor,
    prev_pose: Optional[np.ndarray],
    prev_prev_pose: Optional[np.ndarray],
) -> torch.Tensor:
    if pose_seq.shape[0] == 0:
        return pose_seq.sum() * 0.0
    residuals = []
    if prev_pose is not None and prev_prev_pose is not None:
        prev = torch.as_tensor(prev_pose, dtype=pose_seq.dtype, device=pose_seq.device)
        prev_prev = torch.as_tensor(prev_prev_pose, dtype=pose_seq.dtype, device=pose_seq.device)
        residuals.append(pose_seq[0] - (prev + (prev - prev_prev)))
    if pose_seq.shape[0] >= 2 and prev_pose is not None:
        prev = torch.as_tensor(prev_pose, dtype=pose_seq.dtype, device=pose_seq.device)
        residuals.append(pose_seq[1] - (pose_seq[0] + (pose_seq[0] - prev)))
    if pose_seq.shape[0] >= 3:
        residuals.append(pose_seq[2:] - (pose_seq[1:-1] + (pose_seq[1:-1] - pose_seq[:-2])))
    if not residuals:
        return pose_seq.sum() * 0.0
    flat = torch.cat([r.reshape(-1) for r in residuals])
    return torch.sum(flat * flat)


def single_pose_velocity_extrap_sse(
    pose_vec: torch.Tensor,
    prev_pose: Optional[np.ndarray],
    prev_prev_pose: Optional[np.ndarray],
) -> torch.Tensor:
    if prev_pose is None or prev_prev_pose is None:
        return pose_vec.sum() * 0.0
    prev = torch.as_tensor(prev_pose, dtype=pose_vec.dtype, device=pose_vec.device)
    prev_prev = torch.as_tensor(prev_prev_pose, dtype=pose_vec.dtype, device=pose_vec.device)
    target = prev + (prev - prev_prev)
    return torch.sum((pose_vec - target) ** 2)


def mask_toe_pose(body_pose: torch.Tensor, freeze_toes: bool) -> torch.Tensor:
    if not freeze_toes:
        return body_pose
    masked = body_pose.clone()
    masked[:, 27:33] = 0.0
    return masked


def marker_weight_tensor(labels: Sequence[str], arm_marker_weight: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    weights = np.ones(len(labels), dtype=np.float32)
    if arm_marker_weight != 1.0:
        weights[[label in ARM_MARKER_LABELS for label in labels]] = float(arm_marker_weight)
    return torch.as_tensor(weights, dtype=dtype, device=device)


def parse_anneal_factors(text: str) -> Tuple[float, ...]:
    factors = tuple(float(part) for part in re.split(r"[,\s]+", text.strip()) if part)
    if not factors:
        raise ValueError("At least one Stage-I anneal factor is required.")
    if any(f <= 0.0 for f in factors):
        raise ValueError(f"Stage-I anneal factors must be positive, got {factors}.")
    return factors


def fullpose_from_parts(root_orient: np.ndarray, body_pose: np.ndarray, hand_pose: np.ndarray) -> np.ndarray:
    n = root_orient.shape[0]
    fullpose = np.zeros((n, SMPLX_FULLPOSE_DOF), dtype=np.float64)
    fullpose[:, :3] = root_orient
    fullpose[:, 3:66] = body_pose
    fullpose[:, 75:] = hand_pose
    return fullpose


def kabsch_init(sim: np.ndarray, obs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    valid = np.isfinite(obs).all(axis=1)
    if valid.sum() < 3:
        return np.zeros(3, dtype=np.float64), np.zeros(3, dtype=np.float64)
    a = sim[valid].T
    b = obs[valid].T
    a_mean = a.mean(axis=1, keepdims=True)
    b_mean = b.mean(axis=1, keepdims=True)
    u, _, vt = np.linalg.svd((a - a_mean) @ (b - b_mean).T, full_matrices=False)
    rmat = vt.T @ u.T
    if np.linalg.det(rmat) < 0:
        vt[-1] *= -1
        rmat = vt.T @ u.T
    trans = (b_mean - rmat @ a_mean).reshape(3)
    return Rotation.from_matrix(rmat).as_rotvec(), trans


def kabsch_init_batch(sim: np.ndarray, obs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorized rigid initialization from one rest marker set to many frames."""

    obs = np.asarray(obs, dtype=np.float64)
    sim = np.asarray(sim, dtype=np.float64)
    valid = np.isfinite(obs).all(axis=-1)
    counts = valid.sum(axis=1)
    good = counts >= 3
    root = np.zeros((obs.shape[0], 3), dtype=np.float64)
    trans = np.zeros((obs.shape[0], 3), dtype=np.float64)
    if not good.any():
        return root, trans

    weights = valid[good].astype(np.float64)
    denom = np.maximum(counts[good].astype(np.float64), 1.0)[:, None]
    obs_good = np.nan_to_num(obs[good], nan=0.0)
    sim_good = np.broadcast_to(sim[None], obs_good.shape)
    a_mean = (sim_good * weights[:, :, None]).sum(axis=1) / denom
    b_mean = (obs_good * weights[:, :, None]).sum(axis=1) / denom
    a_centered = (sim_good - a_mean[:, None, :]) * weights[:, :, None]
    b_centered = (obs_good - b_mean[:, None, :]) * weights[:, :, None]
    cov = np.einsum("bmi,bmj->bij", a_centered, b_centered)
    u, _, vt = np.linalg.svd(cov, full_matrices=False)
    rmat = vt.transpose(0, 2, 1) @ u.transpose(0, 2, 1)
    bad = np.linalg.det(rmat) < 0
    if bad.any():
        vt[bad, -1, :] *= -1
        rmat = vt.transpose(0, 2, 1) @ u.transpose(0, 2, 1)
    trans_good = b_mean - np.einsum("bij,bj->bi", rmat, a_mean)
    root[good] = Rotation.from_matrix(rmat).as_rotvec()
    trans[good] = trans_good
    return root, trans


def make_observation_tensors(
    mocap: MocapData,
    latent_labels: Sequence[str],
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    label_to_idx = {label: i for i, label in enumerate(mocap.labels)}
    obs = np.full((mocap.markers.shape[0], len(latent_labels), 3), np.nan, dtype=np.float32)
    for lid, label in enumerate(latent_labels):
        midx = label_to_idx.get(label)
        if midx is not None:
            obs[:, lid] = mocap.markers[:, midx].astype(np.float32)
    mask_np = np.isfinite(obs).all(axis=-1)
    return (
        torch.as_tensor(obs, dtype=dtype, device=device),
        torch.as_tensor(mask_np, dtype=torch.bool, device=device),
        obs,
    )


def select_stagei_frames(
    mask_np: np.ndarray,
    num_frames: int,
    least_avail_markers: float,
    seed: int,
    manual_frame_ids: Optional[str] = None,
) -> np.ndarray:
    availability = mask_np.mean(axis=1)
    if manual_frame_ids:
        frame_ids = np.asarray([int(part) for part in re.split(r"[,\s]+", manual_frame_ids.strip()) if part], dtype=np.int64)
        if len(frame_ids) == 0:
            raise ValueError("--stagei-frame-ids was provided but no frame ids were parsed.")
        bad = frame_ids[(frame_ids < 0) | (frame_ids >= len(availability))]
        if len(bad):
            raise ValueError(f"Stage-I frame ids out of range: {bad.tolist()}")
        bad_avail = frame_ids[availability[frame_ids] < least_avail_markers]
        if len(bad_avail):
            raise ValueError(
                "Manual Stage-I frame ids below availability threshold: "
                + ", ".join(f"{int(fid)}={availability[fid]:.3f}" for fid in bad_avail)
            )
        return frame_ids[:num_frames]

    candidates = np.flatnonzero(availability >= least_avail_markers)
    if len(candidates) == 0:
        best = float(availability.max()) if len(availability) else 0.0
        raise RuntimeError(
            f"No stage-I frames satisfy least_avail_markers={least_avail_markers}; "
            f"best frame has availability={best:.3f}."
        )

    # Match MoSh++ frame_picker.load_marker_sessions_random_strict: seed legacy
    # NumPy RNG, shuffle all valid candidates, take the first num_frames, then
    # apply a second random choice over those selected frames.
    np.random.seed(seed=seed)
    picked = []
    for fidx in np.random.choice(np.arange(len(availability), dtype=np.int64), len(availability), replace=False):
        if availability[int(fidx)] >= least_avail_markers:
            picked.append(int(fidx))
        if len(picked) >= num_frames:
            break
    if len(picked) < num_frames:
        raise RuntimeError(f"Only found {len(picked)} valid Stage-I frames; requested {num_frames}.")
    ids = np.random.choice(len(picked), num_frames, replace=False)
    return np.asarray([picked[int(i)] for i in ids], dtype=np.int64)


def marker_distances_from_layout(marker_meta: Dict, latent_labels: Sequence[str]) -> np.ndarray:
    out = []
    for label in latent_labels:
        mtype = marker_meta["marker_type"][label]
        out.append(marker_meta["m2b_distance"][mtype])
    return np.asarray(out, dtype=np.float32)


def numpy_marker_meta(marker_meta: Dict) -> Dict:
    return {
        **marker_meta,
        "marker_vids": OrderedDict(marker_meta["marker_vids"]),
        "marker_type": OrderedDict(marker_meta["marker_type"]),
        "marker_type_mask": OrderedDict((k, np.asarray(v)) for k, v in marker_meta["marker_type_mask"].items()),
    }
