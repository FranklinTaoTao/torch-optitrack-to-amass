from __future__ import annotations

from .helpers import *
from .helpers import _ensure_parent

def save_stagei(stagei_data: Dict, pkl_fname: Path, npz_fname: Path, args, cfg: Dict) -> None:
    stagei_data["stagei_debug_details"]["cfg"] = cfg
    with open(_ensure_parent(pkl_fname), "wb") as f:
        pickle.dump(stagei_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    np.savez(
        _ensure_parent(npz_fname),
        gender=args.gender,
        surface_model_type=args.surface_model_type,
        markers_latent=stagei_data["markers_latent"],
        latent_labels=np.asarray(stagei_data["latent_labels"]),
        markers_latent_vids=stagei_data["markers_latent_vids"],
        betas=stagei_data["betas"][: args.num_betas],
        num_betas=args.num_betas,
    )


def save_stageii(stageii_data: Dict, pkl_fname: Path, npz_fname: Path, args, cfg: Dict) -> None:
    stageii_data["stageii_debug_details"]["cfg"] = cfg
    stageii_data["stagei_debug_details"]["cfg"] = cfg
    with open(_ensure_parent(pkl_fname), "wb") as f:
        pickle.dump(stageii_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    fullpose = stageii_data["fullpose"]
    np.savez(
        _ensure_parent(npz_fname),
        gender=args.gender,
        surface_model_type=args.surface_model_type,
        mocap_frame_rate=float(stageii_data["stageii_debug_details"]["mocap_frame_rate"]),
        mocap_time_length=float(stageii_data["stageii_debug_details"]["mocap_time_length"]),
        markers_latent=stageii_data["markers_latent"],
        latent_labels=np.asarray(stageii_data["latent_labels"]),
        markers_latent_vids=stageii_data["markers_latent_vids"],
        trans=stageii_data["trans"],
        poses=fullpose,
        betas=stageii_data["betas"][: args.num_betas],
        num_betas=args.num_betas,
        root_orient=fullpose[:, :3],
        pose_body=fullpose[:, 3:66],
        pose_hand=fullpose[:, 75:],
        pose_jaw=fullpose[:, 66:69],
        pose_eye=fullpose[:, 69:75],
    )
