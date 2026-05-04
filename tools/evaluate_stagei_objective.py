#!/usr/bin/env python3
"""Evaluate Torch Stage-I objective terms for one or more Stage-I pickles."""

from __future__ import annotations

import argparse
import pickle
import re
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from torch_optitrack_to_amass.convert_optitrack_to_amass_torch import (  # noqa: E402
    SMPLX_BODY_DOF,
    BodyPosePrior,
    _read_json,
    _resolve_mocap_unit,
    _torch_device,
    auto_resolve_headside_label,
    create_smplx_model,
    latent_surface_residual,
    load_marker_layout,
    load_mocap,
    make_face_tensor,
    make_observation_tensors,
    marker_coeffs_from_latent_nn,
    marker_distances_from_layout,
    marker_weight_tensor,
    masked_marker_mse,
    masked_marker_sse,
    reconstruct_markers_nn,
    resolve_model_file,
    smplx_forward,
)


def stagei_frame_ids(stagei_data: dict) -> np.ndarray:
    dbg = stagei_data["stagei_debug_details"]
    ids = dbg.get("stagei_frame_ids")
    if ids is not None:
        return np.asarray(ids, dtype=np.int64)
    fnames = dbg.get("stagei_fnames", [])
    return np.asarray([int(re.search(r"_(\d+)$", str(fname)).group(1)) for fname in fnames], dtype=np.int64)


def pose_and_trans(stagei_data: dict) -> tuple[np.ndarray, np.ndarray]:
    dbg = stagei_data["stagei_debug_details"]
    pose = np.asarray(dbg["opt_models_pose"], dtype=np.float32)
    trans = np.asarray(dbg["opt_models_trans"], dtype=np.float32)
    if pose.ndim != 2 or pose.shape[1] < 66:
        raise ValueError("Stage-I pickle does not contain opt_models_pose with at least 66 columns.")
    return pose, trans


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stagei_pkls", nargs="+", type=Path)
    parser.add_argument("--mocap", required=True)
    parser.add_argument("--marker-layout", required=True)
    parser.add_argument("--labels-map-json", default=None)
    parser.add_argument("--support-base-dir", default="support_files")
    parser.add_argument("--model-base-dir", default="support_files")
    parser.add_argument("--gender", default="female")
    parser.add_argument("--mocap-unit", default="auto")
    parser.add_argument("--num-betas", type=int, default=16)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--pose-body-prior", default="support_files/smplx/pose_body_prior.pkl")
    parser.add_argument("--surface-distance", choices=["vertex", "triangles"], default="vertex")
    parser.add_argument("--arm-marker-weight", type=float, default=1.0)
    args = parser.parse_args()

    device = _torch_device(args.device)
    dtype = torch.float32
    mocap_fname = Path(args.mocap).expanduser().resolve()
    labels_map = _read_json(args.labels_map_json)
    labels_map = auto_resolve_headside_label(mocap_fname, labels_map)
    mocap_unit = _resolve_mocap_unit(mocap_fname, args.mocap_unit)
    mocap = load_mocap(mocap_fname, mocap_unit, labels_map)
    marker_meta = load_marker_layout(Path(args.marker_layout))
    model_file = resolve_model_file(Path(args.model_base_dir), Path(args.support_base_dir), args.gender)
    pose_prior = BodyPosePrior(Path(args.pose_body_prior).expanduser().resolve(), device, dtype)
    anneals = (1.0, 0.5, 0.25, 0.125)

    for stagei_pkl in args.stagei_pkls:
        with open(stagei_pkl, "rb") as f:
            data = pickle.load(f, encoding="latin1")
        labels = list(data["latent_labels"])
        frame_ids = stagei_frame_ids(data)
        pose, trans = pose_and_trans(data)
        obs_all, mask_all, _ = make_observation_tensors(mocap, labels, device, dtype)
        obs = obs_all[frame_ids]
        mask = mask_all[frame_ids]
        k = len(frame_ids)
        model = create_smplx_model(model_file, batch_size=k, gender=args.gender, num_betas=args.num_betas, device=device)
        can_model = create_smplx_model(model_file, batch_size=1, gender=args.gender, num_betas=args.num_betas, device=device)
        faces = make_face_tensor(can_model, device)
        betas = torch.as_tensor(np.asarray(data["betas"][: args.num_betas], dtype=np.float32)[None], dtype=dtype, device=device)
        body_pose = torch.as_tensor(pose[:, 3:66], dtype=dtype, device=device)
        root = torch.as_tensor(pose[:, :3], dtype=dtype, device=device)
        transl = torch.as_tensor(trans, dtype=dtype, device=device)
        markers_latent = torch.as_tensor(np.asarray(data["markers_latent"], dtype=np.float32), dtype=dtype, device=device)
        marker_weights = marker_weight_tensor(labels, args.arm_marker_weight, device, dtype)
        init_dist = torch.as_tensor(marker_distances_from_layout(marker_meta, labels), dtype=dtype, device=device)

        with torch.no_grad():
            verts = smplx_forward(model, betas.expand(k, -1), body_pose, root, transl, k)
            can_verts = smplx_forward(
                can_model,
                betas,
                torch.zeros((1, SMPLX_BODY_DOF), dtype=dtype, device=device),
                torch.zeros((1, 3), dtype=dtype, device=device),
                torch.zeros((1, 3), dtype=dtype, device=device),
                1,
            )
            coeffs, nn_vids = marker_coeffs_from_latent_nn(can_verts, markers_latent)
            pred = reconstruct_markers_nn(verts, nn_vids, coeffs)
            init_betas = torch.zeros_like(betas)
            init_can_verts = smplx_forward(
                can_model,
                init_betas,
                torch.zeros((1, SMPLX_BODY_DOF), dtype=dtype, device=device),
                torch.zeros((1, 3), dtype=dtype, device=device),
                torch.zeros((1, 3), dtype=dtype, device=device),
                1,
            )
            init_latent = torch.as_tensor(
                init_can_verts[0, [marker_meta["marker_vids"][label] for label in labels]].detach().cpu().numpy(),
                dtype=dtype,
                device=device,
            )
            # This mirrors prepare_mosh_markers_latent closely enough for diagnostics.
            from torch_optitrack_to_amass.convert_optitrack_to_amass_torch import vertex_normals  # local import

            normals = vertex_normals(init_can_verts, faces)[0, [marker_meta["marker_vids"][label] for label in labels]]
            init_latent = init_latent + normals * init_dist[:, None]
            data_sse = masked_marker_sse(pred, obs, mask, marker_weights=marker_weights)
            data_mse = masked_marker_mse(pred, obs, mask, marker_weights=marker_weights)
            init_sse = torch.sum((markers_latent - init_latent) ** 2)
            surf = latent_surface_residual(can_verts, faces, markers_latent, init_dist, args.surface_distance)
            surf_sse = torch.sum(surf * surf)
            pose_sse = pose_prior.sse(body_pose)
            beta_sse = torch.sum(betas * betas)

        print(f"\n{stagei_pkl}")
        print(f"frames: {frame_ids.tolist()}")
        print(
            "raw: "
            f"data_sse={float(data_sse):.6g} data_mse={float(data_mse):.6g} "
            f"init_sse={float(init_sse):.6g} surf_sse={float(surf_sse):.6g} "
            f"pose_sse={float(pose_sse):.6g} beta_sse={float(beta_sse):.6g}"
        )
        for anneal in anneals:
            wt_data = (75.0 / anneal) * (46.0 / len(labels)) * 0.01
            wt_init = 300.0 * anneal * 0.01
            wt_surf = 10000.0 * 0.01
            wt_pose = 3.0 * anneal * 0.01
            wt_beta = 10.0 * anneal * 0.01
            total = (
                (wt_data * wt_data) * data_sse
                + (wt_init * wt_init) * init_sse
                + (wt_surf * wt_surf) * surf_sse
                + (wt_pose * wt_pose) * pose_sse
                + (wt_beta * wt_beta) * beta_sse
            )
            print(f"anneal={anneal:.3f} torch_scaled_loss={float(total):.6g}")


if __name__ == "__main__":
    main()
