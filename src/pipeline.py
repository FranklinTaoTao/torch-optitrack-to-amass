from __future__ import annotations

from .helpers import *
from .helpers import _read_json, _repo_root, _resolve_mocap_unit, _sanitize_stem, _torch_device
from .io import save_stagei, save_stageii
from .stagei import optimize_stagei
from .stageii import optimize_stageii

def run_conversion(args: argparse.Namespace) -> None:
    mocap_fname = Path(args.mocap).expanduser().resolve()
    support_base_dir = Path(args.support_base_dir).expanduser().resolve()
    model_base_dir = Path(args.model_base_dir).expanduser().resolve()
    work_base_dir = Path(args.work_base_dir).expanduser().resolve()
    if args.surface_model_type != "smplx":
        raise NotImplementedError("The Torch converter currently supports SMPL-X only.")
    if not mocap_fname.exists():
        raise FileNotFoundError(f"Mocap file not found: {mocap_fname}")

    labels_map = _read_json(args.labels_map_json)
    labels_map = auto_resolve_headside_label(mocap_fname, labels_map)
    mocap_unit = _resolve_mocap_unit(mocap_fname, args.mocap_unit)
    mocap = load_mocap(mocap_fname, mocap_unit, labels_map, args.start_fidx, args.end_fidx, args.ds_rate)
    labels = [label for label in mocap.labels if label and not label.startswith("*")]
    if not labels:
        raise RuntimeError("No labeled markers found in mocap.")

    marker_layout_dir = work_base_dir / "marker_layouts"
    marker_layout_fname = Path(args.marker_layout) if args.marker_layout else marker_layout_dir / f"{mocap_fname.stem}_{args.surface_model_type}.json"
    if marker_layout_fname.exists():
        marker_meta = load_marker_layout(marker_layout_fname)
    else:
        marker_meta = marker_labels_to_layout(
            labels,
            marker_layout_fname,
            surface_model_type=args.surface_model_type,
            wrist_markers_on_stick=args.wrist_markers_on_stick,
        )

    if args.pose_body_prior == "auto":
        prior_candidate = support_base_dir / args.surface_model_type / "pose_body_prior.pkl"
        if not prior_candidate.exists():
            prior_candidate = _repo_root() / "support_files" / args.surface_model_type / "pose_body_prior.pkl"
        args.pose_body_prior = str(prior_candidate) if prior_candidate.exists() else None
    elif args.pose_body_prior == "none":
        args.pose_body_prior = None

    model_file = Path(args.model_file).expanduser().resolve() if args.model_file else resolve_model_file(model_base_dir, support_base_dir, args.gender)
    device = _torch_device(args.device)
    print(f"Using device: {device}")
    print(f"Using SMPL-X model: {model_file}")
    print(f"Resolved mocap unit: {mocap_unit}; frames after window/stride: {mocap.markers.shape[0]}")
    print(f"Marker layout: {marker_layout_fname} ({len(marker_meta['marker_vids'])} markers)")
    if args.pose_body_prior:
        print(f"Using body pose prior: {args.pose_body_prior}")
    else:
        print("Body pose prior disabled.")

    stem = _sanitize_stem(mocap_fname.stem)
    out_dir = work_base_dir / "workspace" / mocap_fname.parent.name
    stagei_pkl = out_dir / f"{stem}_{args.gender}_stagei.pkl"
    stagei_npz = stagei_pkl.with_suffix(".npz")
    stageii_pkl = out_dir / f"{stem}_stageii.pkl"
    stageii_npz = stageii_pkl.with_suffix(".npz")

    cfg = {
        "mocap": {
            "fname": str(mocap_fname),
            "unit": mocap_unit,
            "start_fidx": args.start_fidx,
            "end_fidx": args.end_fidx,
            "ds_rate": args.ds_rate,
        },
        "surface_model": {
            "type": args.surface_model_type,
            "gender": args.gender,
            "fname": str(model_file),
            "num_betas": args.num_betas,
        },
        "moshpp": {
            "optimize_betas": True,
            "optimize_dynamics": False,
            "optimize_face": False,
            "optimize_fingers": False,
            "pose_body_prior_fname": args.pose_body_prior,
        },
        "torch": vars(args).copy(),
    }

    if args.stagei_pkl:
        with open(Path(args.stagei_pkl).expanduser().resolve(), "rb") as f:
            stagei_data = pickle.load(f)
        print(f"Loaded Stage-I input: {Path(args.stagei_pkl).expanduser().resolve()}")
    else:
        stagei_data = optimize_stagei(args, model_file, marker_meta, mocap, device)
        save_stagei(stagei_data, stagei_pkl, stagei_npz, args, cfg)
        print(f"Stage-I output: {stagei_pkl}")
        print(f"Stage-I AMASS-style NPZ: {stagei_npz}")

    if args.stagei_only:
        return

    stageii_data = optimize_stageii(args, model_file, marker_meta, mocap, stagei_data, device)
    save_stageii(stageii_data, stageii_pkl, stageii_npz, args, cfg)
    print(f"Stage-II output: {stageii_pkl}")
    print(f"AMASS-compatible NPZ: {stageii_npz}")
