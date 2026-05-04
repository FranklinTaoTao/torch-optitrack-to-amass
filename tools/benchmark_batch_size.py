#!/usr/bin/env python3
"""Benchmark stage-II batch sizes for the Torch OptiTrack converter."""

from __future__ import annotations

import argparse
import gc
import pickle
import sys
import time
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from torch_optitrack_to_amass.convert_optitrack_to_amass_torch import (
    _read_json,
    _resolve_mocap_unit,
    _torch_device,
    auto_resolve_headside_label,
    load_marker_layout,
    load_mocap,
    optimize_stageii,
    resolve_model_file,
)


def marker_error_mm(stageii_data: dict) -> tuple[float, float, float]:
    errs = []
    dbg = stageii_data["stageii_debug_details"]
    for sim_frame, obs_frame in zip(dbg["markers_sim"], dbg["markers_obs"]):
        sim = np.asarray(sim_frame, dtype=np.float64)
        obs = np.asarray(obs_frame, dtype=np.float64)
        valid = np.isfinite(obs).all(axis=1)
        if valid.any():
            errs.append(np.linalg.norm(sim[valid] - obs[valid], axis=1))
    values = np.concatenate(errs) * 1000.0
    return float(values.mean()), float(np.median(values)), float(np.percentile(values, 95))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mocap", required=True)
    parser.add_argument("--stagei-pkl", required=True)
    parser.add_argument("--marker-layout", required=True)
    parser.add_argument("--support-base-dir", default="support_files")
    parser.add_argument("--model-base-dir", default="support_files")
    parser.add_argument("--labels-map-json", default=None)
    parser.add_argument("--gender", default="female")
    parser.add_argument("--mocap-unit", default="auto")
    parser.add_argument("--start-fidx", type=int, default=0)
    parser.add_argument("--end-fidx", type=int, default=1024)
    parser.add_argument("--ds-rate", type=int, default=1)
    parser.add_argument("--num-betas", type=int, default=16)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--stageii-epochs", type=int, default=40)
    parser.add_argument("--stageii-lr", type=float, default=0.025)
    parser.add_argument("--robust-sigma", type=float, default=0.04)
    parser.add_argument("--stageii-data-weight", type=float, default=1.0)
    parser.add_argument("--stageii-pose-prior-weight", type=float, default=0.002)
    parser.add_argument("--stageii-pose-l2-weight", type=float, default=0.0005)
    parser.add_argument("--stageii-velocity-weight", type=float, default=0.05)
    parser.add_argument("--batch-sizes", type=int, nargs="+", required=True)
    args = parser.parse_args()

    mocap_fname = Path(args.mocap).expanduser().resolve()
    support_base_dir = Path(args.support_base_dir).expanduser().resolve()
    model_base_dir = Path(args.model_base_dir).expanduser().resolve()
    labels_map = _read_json(args.labels_map_json)
    labels_map = auto_resolve_headside_label(mocap_fname, labels_map)
    mocap_unit = _resolve_mocap_unit(mocap_fname, args.mocap_unit)
    mocap = load_mocap(mocap_fname, mocap_unit, labels_map, args.start_fidx, args.end_fidx, args.ds_rate)
    marker_meta = load_marker_layout(Path(args.marker_layout))
    stagei_data = pickle.load(open(args.stagei_pkl, "rb"))
    model_file = resolve_model_file(model_base_dir, support_base_dir, args.gender)
    pose_prior = support_base_dir / "smplx" / "pose_body_prior.pkl"
    device = _torch_device(args.device)

    print(f"device={device} frames={mocap.markers.shape[0]} epochs={args.stageii_epochs}")
    print("batch_size,seconds,frames_per_second,peak_cuda_gb,mean_mm,median_mm,p95_mm")
    for batch_size in args.batch_sizes:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device) if device.type == "cuda" else None
        bench_args = Namespace(
            mocap=str(mocap_fname),
            gender=args.gender,
            num_betas=args.num_betas,
            pose_body_prior=str(pose_prior) if pose_prior.exists() else None,
            batch_size=batch_size,
            stageii_epochs=args.stageii_epochs,
            stageii_lr=args.stageii_lr,
            robust_sigma=args.robust_sigma,
            stageii_data_weight=args.stageii_data_weight,
            stageii_pose_prior_weight=args.stageii_pose_prior_weight,
            stageii_pose_l2_weight=args.stageii_pose_l2_weight,
            stageii_velocity_weight=args.stageii_velocity_weight,
            verbose=False,
        )
        started = time.perf_counter()
        try:
            out = optimize_stageii(bench_args, model_file, marker_meta, mocap, stagei_data, device)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - started
            mean_mm, median_mm, p95_mm = marker_error_mm(out)
            peak_gb = torch.cuda.max_memory_allocated(device) / (1024**3) if device.type == "cuda" else 0.0
            fps = mocap.markers.shape[0] / elapsed
            print(f"{batch_size},{elapsed:.3f},{fps:.2f},{peak_gb:.3f},{mean_mm:.3f},{median_mm:.3f},{p95_mm:.3f}", flush=True)
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                print(f"{batch_size},OOM,OOM,OOM,OOM,OOM,OOM", flush=True)
            else:
                raise
        finally:
            gc.collect()
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
