#!/usr/bin/env python3
"""Quantitatively compare a Torch fit against a reference MoSh++ fit."""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from torch_optitrack_to_amass.convert_optitrack_to_amass_torch import (  # noqa: E402
    SMPLX_BODY_DOF,
    create_smplx_model,
)


ARM_LABELS = {
    "LBSH", "LSHOUP", "LUPA", "LELB", "LIWR", "LOWR", "LFIN",
    "RBSH", "RSHOUP", "RUPA", "RELB", "RIWR", "ROWR", "RFIN",
}


def load_npz(path: Path) -> dict:
    return {k: v for k, v in np.load(path, allow_pickle=True).items()}


def marker_errors(stageii_pkl: Path, labels_subset: set[str] | None = None, max_frames: int = -1) -> np.ndarray:
    data = pickle.load(open(stageii_pkl, "rb"))
    dbg = data["stageii_debug_details"]
    chunks = []
    iterator = zip(dbg["markers_sim"], dbg["markers_obs"], dbg["labels_obs"])
    for frame_idx, (sim_frame, obs_frame, labels) in enumerate(iterator):
        if max_frames > 0 and frame_idx >= max_frames:
            break
        sim = np.asarray(sim_frame, dtype=np.float64)
        obs = np.asarray(obs_frame, dtype=np.float64)
        valid = np.isfinite(obs).all(axis=1)
        if labels_subset is not None:
            valid &= np.asarray([label in labels_subset for label in labels], dtype=bool)
        if valid.any():
            chunks.append(np.linalg.norm(sim[valid] - obs[valid], axis=1))
    return np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float64)


def summarize_mm(values_m: np.ndarray) -> str:
    values = values_m * 1000.0
    return (
        f"mean={values.mean():.2f}mm median={np.median(values):.2f}mm "
        f"p95={np.percentile(values, 95):.2f}mm max={values.max():.2f}mm"
    )


def rotation_geodesic_deg(a_axis_angle: np.ndarray, b_axis_angle: np.ndarray) -> np.ndarray:
    a = Rotation.from_rotvec(a_axis_angle.reshape(-1, 3))
    b = Rotation.from_rotvec(b_axis_angle.reshape(-1, 3))
    return (a.inv() * b).magnitude().reshape(a_axis_angle.shape[:-1]) * 180.0 / np.pi


def smplx_joints(npz_data: dict, model_file: Path, gender: str, num_betas: int, batch_size: int, device: torch.device) -> np.ndarray:
    nframes = npz_data["poses"].shape[0]
    chunks = []
    for start in range(0, nframes, batch_size):
        end = min(start + batch_size, nframes)
        bsz = end - start
        model = create_smplx_model(model_file, bsz, gender, num_betas, device)
        dtype = torch.float32
        poses = npz_data["poses"][start:end]
        betas = torch.as_tensor(npz_data["betas"][:num_betas][None], dtype=dtype, device=device).expand(bsz, -1)
        body_pose = torch.as_tensor(poses[:, 3:66], dtype=dtype, device=device)
        global_orient = torch.as_tensor(poses[:, :3], dtype=dtype, device=device)
        transl = torch.as_tensor(npz_data["trans"][start:end], dtype=dtype, device=device)
        left_hand = torch.as_tensor(poses[:, 75:120], dtype=dtype, device=device)
        right_hand = torch.as_tensor(poses[:, 120:165], dtype=dtype, device=device)
        zeros_3 = torch.zeros((bsz, 3), dtype=dtype, device=device)
        with torch.no_grad():
            out = model(
                betas=betas,
                body_pose=body_pose,
                global_orient=global_orient,
                transl=transl,
                left_hand_pose=left_hand,
                right_hand_pose=right_hand,
                jaw_pose=zeros_3,
                leye_pose=zeros_3,
                reye_pose=zeros_3,
                return_verts=False,
            )
        chunks.append(out.joints[:, :55].detach().cpu().numpy().astype(np.float64))
        del model
    return np.concatenate(chunks, axis=0)


def elbow_angles_deg(joints: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    def angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
        u = a - b
        v = c - b
        denom = np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1)
        cos = np.sum(u * v, axis=1) / np.maximum(denom, 1e-8)
        return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))

    left = angle(joints[:, 16], joints[:, 18], joints[:, 20])
    right = angle(joints[:, 17], joints[:, 19], joints[:, 21])
    return left, right


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-npz", required=True, type=Path)
    parser.add_argument("--reference-pkl", required=True, type=Path)
    parser.add_argument("--candidate-npz", required=True, type=Path)
    parser.add_argument("--candidate-pkl", required=True, type=Path)
    parser.add_argument("--model-file", default="support_files/smplx/female/model.pkl", type=Path)
    parser.add_argument("--gender", default="female")
    parser.add_argument("--num-betas", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-frames", type=int, default=-1)
    args = parser.parse_args()

    ref = load_npz(args.reference_npz)
    cand = load_npz(args.candidate_npz)
    nframes = min(ref["poses"].shape[0], cand["poses"].shape[0])
    if args.max_frames > 0:
        nframes = min(nframes, args.max_frames)
    for data in (ref, cand):
        for key in ("poses", "trans", "root_orient", "pose_body", "pose_hand"):
            if key in data and data[key].ndim > 1:
                data[key] = data[key][:nframes]

    print(f"frames={nframes}")

    ref_marker = marker_errors(args.reference_pkl, max_frames=nframes)
    cand_marker = marker_errors(args.candidate_pkl, max_frames=nframes)
    print(f"reference marker reprojection: {summarize_mm(ref_marker)}")
    print(f"candidate marker reprojection: {summarize_mm(cand_marker)}")
    print(f"candidate arm marker reprojection: {summarize_mm(marker_errors(args.candidate_pkl, ARM_LABELS, nframes))}")

    root_deg = rotation_geodesic_deg(ref["root_orient"], cand["root_orient"])
    body_ref = ref["pose_body"].reshape(nframes, 21, 3)
    body_cand = cand["pose_body"].reshape(nframes, 21, 3)
    body_deg = rotation_geodesic_deg(body_ref, body_cand)
    arm_ids = [13, 14, 15, 16, 17, 18, 19, 20]
    trans_err = np.linalg.norm(ref["trans"] - cand["trans"], axis=1)
    pose_vel_l2 = np.linalg.norm(np.diff(ref["pose_body"], axis=0) - np.diff(cand["pose_body"], axis=0), axis=1)

    print(
        f"root rotation difference: mean={root_deg.mean():.2f}deg "
        f"median={np.median(root_deg):.2f}deg p95={np.percentile(root_deg, 95):.2f}deg"
    )
    print(
        f"body joint rotation difference: mean={body_deg.mean():.2f}deg "
        f"median={np.median(body_deg):.2f}deg p95={np.percentile(body_deg, 95):.2f}deg"
    )
    print(
        f"arm joint rotation difference: mean={body_deg[:, arm_ids].mean():.2f}deg "
        f"median={np.median(body_deg[:, arm_ids]):.2f}deg p95={np.percentile(body_deg[:, arm_ids], 95):.2f}deg"
    )
    print(f"translation difference: {summarize_mm(trans_err)}")
    print(
        f"body pose velocity difference L2: mean={pose_vel_l2.mean():.4f} "
        f"median={np.median(pose_vel_l2):.4f} p95={np.percentile(pose_vel_l2, 95):.4f}"
    )

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    ref_joints = smplx_joints(ref, args.model_file, args.gender, args.num_betas, args.batch_size, device)
    cand_joints = smplx_joints(cand, args.model_file, args.gender, args.num_betas, args.batch_size, device)
    joint_err = np.linalg.norm(ref_joints[:, :22] - cand_joints[:, :22], axis=2)
    joint_err_root_aligned = np.linalg.norm(
        (ref_joints[:, :22] - ref_joints[:, :1]) - (cand_joints[:, :22] - cand_joints[:, :1]),
        axis=2,
    )
    print(f"absolute body MPJPE vs reference: {summarize_mm(joint_err.reshape(-1))}")
    print(f"root-aligned body MPJPE vs reference: {summarize_mm(joint_err_root_aligned.reshape(-1))}")

    ref_l_elbow, ref_r_elbow = elbow_angles_deg(ref_joints)
    cand_l_elbow, cand_r_elbow = elbow_angles_deg(cand_joints)
    elbow_diff = np.concatenate([np.abs(ref_l_elbow - cand_l_elbow), np.abs(ref_r_elbow - cand_r_elbow)])
    print(
        f"elbow angle difference: mean={elbow_diff.mean():.2f}deg "
        f"median={np.median(elbow_diff):.2f}deg p95={np.percentile(elbow_diff, 95):.2f}deg"
    )
    print(
        f"max elbow extension ref/candidate: "
        f"L {ref_l_elbow.max():.1f}/{cand_l_elbow.max():.1f}deg, "
        f"R {ref_r_elbow.max():.1f}/{cand_r_elbow.max():.1f}deg"
    )


if __name__ == "__main__":
    main()
