from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import run_conversion

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_SUPPORT_DIR = _PROJECT_ROOT / "support_files"
_DEFAULT_LABEL_MAP = _PROJECT_ROOT / "examples" / "optitrack_to_amass_label_map_suggested.json"
_DEFAULT_MARKER_LAYOUT = _PROJECT_ROOT / "examples" / "smplx_marker_layout_41.json"

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            "Convert labeled OptiTrack C3D mocap to AMASS-style SMPL-X output. "
            "Defaults use the current high-accuracy GPU recipe: MoSh-style Stage I "
            "and Stage II dogleg with the fast finite-difference Jacobian."
        ),
    )
    p.add_argument("--mocap", required=True, help="Path to labeled .c3d file")
    p.add_argument("--support-base-dir", default=str(_DEFAULT_SUPPORT_DIR), help="Path to support assets")
    p.add_argument("--model-base-dir", default=str(_DEFAULT_SUPPORT_DIR), help="Folder containing SMPL-X model files")
    p.add_argument("--model-file", default=None, help="Direct path to SMPL-X model.pkl loadable by smplx")
    p.add_argument("--output-dir", default=None, help="Write all generated files directly into this directory")
    p.add_argument("--work-base-dir", default=None, help="Legacy MoSh-style output root: writes into <work-base-dir>/workspace/<input-folder>")
    p.add_argument("--surface-model-type", default="smplx", choices=["smplx"])
    p.add_argument("--gender", default="neutral", choices=["male", "female", "neutral"])
    p.add_argument("--mocap-unit", default="auto", choices=["auto", "mm", "cm", "m"])
    p.add_argument("--labels-map-json", default=str(_DEFAULT_LABEL_MAP),
                   help="C3D label-name map. Defaults to the bundled OptiTrack Baseline (41) suggestion.")
    p.add_argument("--marker-layout", default=str(_DEFAULT_MARKER_LAYOUT),
                   help="SMPL-X marker layout JSON. Defaults to the bundled OptiTrack Baseline (41) layout.")
    p.add_argument("--stagei-pkl", default=None, help="Optional existing Torch stage-I pickle; skips stage-I optimization.")
    p.add_argument("--pose-body-prior", default="auto", help="Path, 'auto', or 'none'")
    p.add_argument("--wrist-markers-on-stick", action="store_true")
    p.add_argument("--stagei-only", action="store_true")
    p.add_argument("--start-fidx", type=int, default=0)
    p.add_argument("--end-fidx", type=int, default=-1)
    p.add_argument("--ds-rate", type=int, default=1)
    p.add_argument("--stagei-num-frames", type=int, default=120)
    p.add_argument("--stagei-frame-ids", default=None,
                   help="Comma/space separated absolute mocap frame ids for Stage-I; overrides random frame picking.")
    p.add_argument("--stagei-least-avail-markers", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=100)
    p.add_argument("--num-betas", type=int, default=16)
    p.add_argument("--device", default="auto")
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--stagei-epochs", type=int, default=800)
    p.add_argument("--stagei-bodyfit-epochs", type=int, default=0,
                   help="Stage-I epochs that keep marker offsets fixed so betas/pose must explain the layout first.")
    p.add_argument("--stagei-optimize-latent-markers", action=argparse.BooleanOptionalAction, default=True,
                   help="MoSh-style Stage-I: optimize canonical latent marker positions and derive local coefficients from the current body.")
    p.add_argument("--stagei-latent-basis", choices=["layout", "nn"], default="nn",
                   help="Basis used when optimizing latent markers. 'nn' matches MoSh++ TransformedCoeffs but is less stable with Adam.")
    p.add_argument("--stagei-nn-neighbors", type=int, default=3,
                   help="Nearest-neighbor candidate count for the Stage-I NN latent marker basis.")
    p.add_argument("--stagei-nn-include-eyeballs", action=argparse.BooleanOptionalAction, default=True,
                   help="Include SMPL-X eyeball vertices in the Stage-I NN marker search. MoSh++ excludes them.")
    p.add_argument("--stagei-surface-distance", choices=["vertex", "triangles"], default="vertex",
                   help="Surface residual for NN latent Stage-I. 'triangles' is closer to MoSh++ PtsToMesh but slower.")
    p.add_argument("--stagei-freeze-toes", action=argparse.BooleanOptionalAction, default=True,
                   help="Keep SMPL-X toe pose dimensions at zero during Stage I, matching MoSh++ optimize_toes=false.")
    p.add_argument("--stagei-mosh-anneal", action=argparse.BooleanOptionalAction, default=True,
                   help="Use MoSh++-style four-step Stage-I annealing: increasing marker data weight and decreasing pose/beta/init priors.")
    p.add_argument("--stagei-mosh-anneal-factors", default="1,0.5,0.25,0.125",
                   help="Comma/space separated Stage-I annealing factors. MoSh++ default is 1,0.5,0.25,0.125.")
    p.add_argument("--stagei-mosh-weight-scale", type=float, default=0.01,
                   help="Common scale applied to MoSh++ residual weights before squaring them in the Torch scalar objective.")
    p.add_argument("--stagei-mosh-data-mult", type=float, default=1.0)
    p.add_argument("--stagei-mosh-init-mult", type=float, default=1.5)
    p.add_argument("--stagei-mosh-surf-mult", type=float, default=1.0)
    p.add_argument("--stagei-mosh-pose-mult", type=float, default=2.0)
    p.add_argument("--stagei-mosh-beta-mult", type=float, default=1.0)
    p.add_argument("--stagei-lbfgs-refine", action="store_true",
                   help="Run MoSh-style Stage-I LBFGS refinement after Adam, useful for the nearest-neighbor latent marker basis.")
    p.add_argument("--stagei-lbfgs-phases", type=int, default=4)
    p.add_argument("--stagei-lbfgs-iters", type=int, default=40)
    p.add_argument("--stagei-lbfgs-lr", type=float, default=1.0)
    p.add_argument("--stagei-lbfgs-history", type=int, default=50)
    p.add_argument("--stagei-dogleg-refine", action="store_true",
                   help="Experimental Stage-I trust-region dogleg refinement after Adam.")
    p.add_argument("--stagei-dogleg-full-pose", action="store_true",
                   help="Also optimize selected-frame pose/trans in Stage-I dogleg. Default only refines shape and marker layout.")
    p.add_argument("--stagei-dogleg-jacobian", choices=["auto", "loop-jvp"], default="auto",
                   help="Jacobian builder for Stage-I dogleg. 'loop-jvp' is slow but avoids dense vectorized OOM for full-pose solves.")
    p.add_argument("--stagei-dogleg-anneal-factors", default="0.25,0.125",
                   help="Comma/space separated anneal factors used by Stage-I dogleg refinement.")
    p.add_argument("--stagei-dogleg-iters", type=int, default=2)
    p.add_argument("--stagei-dogleg-delta", type=float, default=0.05)
    p.add_argument("--stagei-dogleg-max-delta", type=float, default=0.5)
    p.add_argument("--stagei-dogleg-min-delta", type=float, default=1e-7)
    p.add_argument("--stagei-dogleg-eta", type=float, default=0.01)
    p.add_argument("--stagei-dogleg-grad-tol", type=float, default=1e-6)
    p.add_argument("--stagei-dogleg-step-tol", type=float, default=1e-6)
    p.add_argument("--stagei-dogleg-solve-damping", type=float, default=1e-6)
    p.add_argument("--stageii-epochs", type=int, default=160)
    p.add_argument("--stagei-lr", type=float, default=0.035)
    p.add_argument("--stageii-lr", type=float, default=0.025)
    p.add_argument("--robust-sigma", type=float, default=0.04)
    p.add_argument("--arm-marker-weight", type=float, default=1.0,
                   help="Multiplier for shoulder/upper-arm/elbow/wrist/hand marker data terms.")
    p.add_argument("--max-marker-offset", type=float, default=0.08)
    p.add_argument("--stagei-data-weight", type=float, default=1.0)
    p.add_argument("--stagei-marker-init-weight", type=float, default=0.05)
    p.add_argument("--stagei-surface-weight", type=float, default=5.0)
    p.add_argument("--stagei-pose-prior-weight", type=float, default=0.002)
    p.add_argument("--stagei-beta-prior-weight", type=float, default=0.001)
    p.add_argument("--stageii-data-weight", type=float, default=1.0)
    p.add_argument("--stageii-pose-prior-weight", type=float, default=0.002)
    p.add_argument("--stageii-pose-l2-weight", type=float, default=0.0005)
    p.add_argument("--stageii-velocity-weight", type=float, default=0.05)
    p.add_argument("--stageii-wrist-velocity-weight", type=float, default=20.0,
                   help="Extra robust Stage-II temporal residual on SMPL-X wrist joints 19/20. Hidden development knob.")
    p.add_argument("--stageii-wrist-velocity-sigma", type=float, default=0.7,
                   help="Pseudo-Huber sigma in radians for the wrist-only temporal residual.")
    p.add_argument("--stageii-mosh-loss", action=argparse.BooleanOptionalAction, default=True,
                   help="Use MoSh++-style Stage-II residual blocks: marker SSE, pose-prior SSE, and pose extrapolation velocity.")
    p.add_argument("--stageii-mosh-weight-scale", type=float, default=1e-4,
                   help="Common scalar multiplier for the MoSh++-style Stage-II residual objective.")
    p.add_argument("--stageii-freeze-toes", action=argparse.BooleanOptionalAction, default=True,
                   help="Keep SMPL-X toe pose dimensions at zero, matching the default MoSh++ optimize_toes=false behavior.")
    p.add_argument("--stageii-sequential-lbfgs", action=argparse.BooleanOptionalAction, default=True,
                   help="Process Stage II one frame at a time on GPU with LBFGS, MoSh++-style warm starts, and first-frame prior passes.")
    p.add_argument(
        "--stageii-solver",
        choices=[
            "lbfgs", "lm", "dogleg",
            "torchmin-bfgs", "torchmin-lbfgs", "torchmin-dogleg", "torchmin-trf",
        ],
        default="dogleg",
                   help="Sequential Stage-II solver. 'lm', 'dogleg', and torchmin variants are experimental.")
    p.add_argument("--stageii-block-size", type=int, default=1,
                   help="Experimental dogleg mode: optimize this many consecutive Stage-II frames jointly.")
    p.add_argument("--stageii-independent-block-size", type=int, default=1,
                   help="Experimental dogleg mode: optimize this many frames in parallel with block-diagonal per-frame Jacobians.")
    p.add_argument("--stageii-lbfgs-iters", type=int, default=20)
    p.add_argument("--stageii-lbfgs-first-iters", type=int, default=20)
    p.add_argument("--stageii-lbfgs-passes", type=int, default=2,
                   help="Number of regular LBFGS solves per sequential Stage-II frame after first-frame setup.")
    p.add_argument("--stageii-lbfgs-lr", type=float, default=1.0)
    p.add_argument("--stageii-lbfgs-history", type=int, default=20)
    p.add_argument("--stageii-lbfgs-line-search", choices=["strong_wolfe", "none"], default="strong_wolfe",
                   help="LBFGS line search. 'none' is faster but may change convergence quality.")
    p.add_argument("--stageii-fallback-strong-wolfe", action="store_true",
                   help="When using --stageii-lbfgs-line-search none, rerun frames with large marker error using strong-Wolfe LBFGS.")
    p.add_argument("--stageii-fallback-mean-marker-error", type=float, default=0.05,
                   help="Mean per-frame marker error threshold in meters for strong-Wolfe fallback.")
    p.add_argument("--stageii-fallback-max-marker-error", type=float, default=0.20,
                   help="Max per-frame marker error threshold in meters for strong-Wolfe fallback.")
    p.add_argument("--stageii-lm-iters", type=int, default=6)
    p.add_argument("--stageii-lm-first-iters", type=int, default=8)
    p.add_argument("--stageii-lm-damping", type=float, default=1e-3)
    p.add_argument("--stageii-lm-step-tol", type=float, default=1e-5)
    p.add_argument("--stageii-dogleg-iters", type=int, default=1)
    p.add_argument("--stageii-dogleg-first-iters", type=int, default=3)
    p.add_argument("--stageii-dogleg-delta", type=float, default=0.5)
    p.add_argument("--stageii-dogleg-max-delta", type=float, default=5.0)
    p.add_argument("--stageii-dogleg-min-delta", type=float, default=1e-7)
    p.add_argument("--stageii-dogleg-eta", type=float, default=0.01)
    p.add_argument("--stageii-dogleg-grad-tol", type=float, default=1e-6)
    p.add_argument("--stageii-dogleg-solve-damping", type=float, default=1e-6)
    p.add_argument("--stageii-dogleg-jacobian-mode", choices=["autograd", "fd"], default="fd",
                   help="Jacobian builder for sequential dogleg. 'fd' uses batched finite differences as a fast derivative proxy.")
    p.add_argument("--stageii-dogleg-fd-eps", type=float, default=5e-4)
    p.add_argument("--stageii-dogleg-jacobian-refresh", type=int, default=4,
                   help="Refresh the dogleg Jacobian every N iterations. Values above 1 reuse J^T J for cheap local steps.")
    p.add_argument("--stageii-dogleg-adaptive-refresh", action=argparse.BooleanOptionalAction, default=True,
                   help="Treat --stageii-dogleg-jacobian-refresh as a max stale-step cap and refresh early after weak dogleg steps.")
    p.add_argument("--stageii-dogleg-refresh-rho", type=float, default=0.5,
                   help="In adaptive refresh mode, refresh the Jacobian after accepted steps with rho below this value.")
    p.add_argument("--stageii-dogleg-merge-passes", action=argparse.BooleanOptionalAction, default=True,
                   help="Merge regular dogleg passes into one multi-iteration solve so Jacobian reuse can reduce work.")
    p.add_argument("--stageii-torch-compile", action="store_true",
                   help="Compile the sequential Stage-II SMPL-X forward with torch.compile. Experimental; can add warmup overhead.")
    p.add_argument("--stageii-compile-mode", default="max-autotune-no-cudagraphs",
                   choices=["default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"])
    p.add_argument("--stageii-compile-fullgraph", action="store_true")
    p.add_argument("--stageii-profile", action="store_true",
                   help="Synchronize CUDA and collect detailed timing for sequential Stage-II solver sections.")
    p.add_argument("--stageii-profile-frames", type=int, default=300,
                   help="Number of sequential Stage-II frames to profile. Use 0 to profile all frames.")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--log-every", type=int, default=50)

    important_args = {
        "help", "mocap", "output_dir", "work_base_dir", "support_base_dir", "model_base_dir",
        "model_file", "marker_layout", "labels_map_json", "gender", "stagei_pkl",
        "stagei_only", "start_fidx", "end_fidx", "ds_rate", "device",
        "stagei_num_frames", "stageii_block_size", "verbose", "log_every",
    }
    for action in p._actions:
        if action.dest not in important_args:
            action.help = argparse.SUPPRESS
    return p


def main() -> None:
    run_conversion(build_parser().parse_args())
