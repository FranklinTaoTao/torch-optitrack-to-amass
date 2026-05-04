#!/usr/bin/env python3
"""Compare MoSh-style pickle outputs by marker reprojection statistics."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Iterable

import numpy as np


def marker_errors(stageii_pkl: Path, frames: int | None = None) -> np.ndarray:
    data = pickle.load(open(stageii_pkl, "rb"))
    dbg = data["stageii_debug_details"]
    sims: Iterable[np.ndarray] = dbg["markers_sim"]
    obs: Iterable[np.ndarray] = dbg["markers_obs"]
    if frames is not None:
        sims = list(sims)[:frames]
        obs = list(obs)[:frames]
    errs = []
    for sim_frame, obs_frame in zip(sims, obs):
        sim = np.asarray(sim_frame, dtype=np.float64)
        target = np.asarray(obs_frame, dtype=np.float64)
        valid = np.isfinite(target).all(axis=1)
        if valid.any():
            errs.append(np.linalg.norm(sim[valid] - target[valid], axis=1))
    if not errs:
        return np.asarray([], dtype=np.float64)
    return np.concatenate(errs)


def summarize(name: str, errs: np.ndarray) -> None:
    if errs.size == 0:
        print(f"{name}: no valid marker errors")
        return
    print(
        f"{name}: n={errs.size} "
        f"mean={errs.mean() * 1000:.2f}mm "
        f"median={np.median(errs) * 1000:.2f}mm "
        f"p95={np.percentile(errs, 95) * 1000:.2f}mm "
        f"max={errs.max() * 1000:.2f}mm"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("outputs", nargs="+", type=Path)
    parser.add_argument("--frames", type=int, default=None)
    args = parser.parse_args()
    for output in args.outputs:
        summarize(str(output), marker_errors(output, args.frames))


if __name__ == "__main__":
    main()
