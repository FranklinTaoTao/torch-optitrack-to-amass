from __future__ import annotations

from .helpers import *

def optimize_stageii(args, model_file: Path, marker_meta: Dict, mocap: MocapData, stagei_data: Dict, device: torch.device) -> Dict:
    dtype = torch.float32
    latent_labels = list(stagei_data["latent_labels"])
    obs_all, mask_all, obs_np = make_observation_tensors(mocap, latent_labels, device, dtype)
    nframes = obs_all.shape[0]
    betas_np = np.asarray(stagei_data["betas"][: args.num_betas], dtype=np.float32)[None]
    if "torch_marker_coeffs" not in stagei_data:
        # Original MoSh++ pickles do not carry Torch marker bases. Rebuild the
        # closest analogue from the optimized latent markers in canonical space.
        basis_model = create_smplx_model(model_file, batch_size=1, gender=args.gender, num_betas=args.num_betas, device=device)
        basis_faces = np.asarray(basis_model.faces, dtype=np.int64)
        with torch.no_grad():
            basis_verts = smplx_forward(
                basis_model,
                betas=torch.as_tensor(betas_np, dtype=dtype, device=device),
                body_pose=torch.zeros((1, SMPLX_BODY_DOF), dtype=dtype, device=device),
                global_orient=torch.zeros((1, 3), dtype=dtype, device=device),
                transl=torch.zeros((1, 3), dtype=dtype, device=device),
                batch_size=1,
            )
            latent_t = torch.as_tensor(np.asarray(stagei_data["markers_latent"], dtype=np.float32), dtype=dtype, device=device)
            nn_coeffs, nn_vids = marker_coeffs_from_latent_nn(basis_verts, latent_t)
        can_verts_np = basis_verts[0].detach().cpu().numpy().astype(np.float64)
        marker_vids_guess = np.asarray(
            [stagei_data.get("markers_latent_vids", {}).get(label, marker_meta["marker_vids"][label]) for label in latent_labels],
            dtype=np.int64,
        )
        anchor_vids_np, tangent_vids_np, _ = reanchor_marker_coeffs(
            can_verts_np,
            basis_faces,
            np.asarray(stagei_data["markers_latent"], dtype=np.float64),
            labels=latent_labels,
            initial_anchor_vids=marker_vids_guess,
        )
        stagei_data = dict(stagei_data)
        stagei_data["torch_marker_anchor_vids"] = {label: int(vid) for label, vid in zip(latent_labels, anchor_vids_np)}
        stagei_data["torch_marker_tangent_vids"] = {label: int(vid) for label, vid in zip(latent_labels, tangent_vids_np)}
        stagei_data["torch_marker_coeffs"] = nn_coeffs.detach().cpu().numpy().astype(np.float64)
        stagei_data["torch_marker_nn_vids"] = nn_vids.detach().cpu().numpy().astype(np.int64)
    anchor_vids = stagei_data.get("torch_marker_anchor_vids", stagei_data.get("markers_latent_vids", {}))
    marker_vids_np = np.asarray([anchor_vids.get(label, marker_meta["marker_vids"][label]) for label in latent_labels], dtype=np.int64)
    tangent_vids_np = np.asarray([stagei_data["torch_marker_tangent_vids"][label] for label in latent_labels], dtype=np.int64)
    coeffs_np = np.asarray(stagei_data["torch_marker_coeffs"], dtype=np.float32)
    nn_vids_data = stagei_data.get("torch_marker_nn_vids")
    nn_vids_np = np.asarray(nn_vids_data, dtype=np.int64) if nn_vids_data is not None else None

    root_all = np.zeros((nframes, 3), dtype=np.float64)
    body_all = np.zeros((nframes, SMPLX_BODY_DOF), dtype=np.float64)
    trans_all = np.zeros((nframes, 3), dtype=np.float64)
    markers_sim_debug: List[np.ndarray] = []
    markers_obs_debug: List[np.ndarray] = []
    labels_obs_debug: List[List[str]] = []
    errs: List[float] = []
    pose_prior = BodyPosePrior(Path(args.pose_body_prior).expanduser().resolve() if args.pose_body_prior else None, device, dtype)
    marker_weights = marker_weight_tensor(latent_labels, args.arm_marker_weight, device, dtype)
    started = time.time()

    init_model = create_smplx_model(model_file, batch_size=1, gender=args.gender, num_betas=args.num_betas, device=device)
    init_faces = make_face_tensor(init_model, device)
    init_marker_vids = torch.as_tensor(marker_vids_np, dtype=torch.long, device=device)
    init_tangent_vids = torch.as_tensor(tangent_vids_np, dtype=torch.long, device=device)
    init_coeffs = torch.as_tensor(coeffs_np, dtype=dtype, device=device)
    init_nn_vids = torch.as_tensor(nn_vids_np, dtype=torch.long, device=device) if nn_vids_np is not None else None
    init_betas = torch.as_tensor(betas_np, dtype=dtype, device=device)
    with torch.no_grad():
        rest_verts = smplx_forward(
            init_model,
            betas=init_betas,
            body_pose=torch.zeros((1, SMPLX_BODY_DOF), dtype=dtype, device=device),
            global_orient=torch.zeros((1, 3), dtype=dtype, device=device),
            transl=torch.zeros((1, 3), dtype=dtype, device=device),
            batch_size=1,
        )
        if init_nn_vids is not None:
            rest_markers = reconstruct_markers_nn(rest_verts, init_nn_vids, init_coeffs)[0].cpu().numpy()
        else:
            rest_markers = reconstruct_markers(rest_verts, init_faces, init_marker_vids, init_tangent_vids, init_coeffs)[0].cpu().numpy()
        hand_mean_np = smplx_hand_mean(init_model, nframes).detach().cpu().numpy().astype(np.float64)

    rigid_started = time.time()
    rigid_root_init, rigid_trans_init = kabsch_init_batch(rest_markers, obs_np)
    if args.verbose:
        print(f"stage II rigid initialization for {nframes} frames: {time.time() - rigid_started:.3f}s")

    if args.stageii_sequential_lbfgs:
        model = create_smplx_model(model_file, batch_size=1, gender=args.gender, num_betas=args.num_betas, device=device)
        faces = make_face_tensor(model, device)
        marker_vids = torch.as_tensor(marker_vids_np, dtype=torch.long, device=device)
        tangent_vids = torch.as_tensor(tangent_vids_np, dtype=torch.long, device=device)
        coeffs = torch.as_tensor(coeffs_np, dtype=dtype, device=device)
        nn_vids = torch.as_tensor(nn_vids_np, dtype=torch.long, device=device) if nn_vids_np is not None else None
        betas = torch.as_tensor(betas_np, dtype=dtype, device=device)
        prev_pose_vec_seq: Optional[np.ndarray] = None
        prev_prev_pose_vec_seq: Optional[np.ndarray] = None
        prev_root_seq = rigid_root_init[0].copy() if nframes else np.zeros(3, dtype=np.float64)
        prev_body_seq = np.zeros(SMPLX_BODY_DOF, dtype=np.float64)
        prev_trans_seq = rigid_trans_init[0].copy() if nframes else np.zeros(3, dtype=np.float64)
        fallback_frames: List[int] = []
        profile_times: Dict[str, float] = {}
        profile_counts: Dict[str, int] = {}

        def profile_active(frame_idx: int) -> bool:
            return bool(args.stageii_profile) and (args.stageii_profile_frames <= 0 or frame_idx < args.stageii_profile_frames)

        def profile_sync() -> None:
            if device.type == "cuda":
                torch.cuda.synchronize(device)

        def profile_now(active: bool) -> float:
            if active:
                profile_sync()
            return time.perf_counter()

        def profile_add(active: bool, label: str, started_at: float) -> None:
            if not active:
                return
            profile_sync()
            profile_times[label] = profile_times.get(label, 0.0) + (time.perf_counter() - started_at)
            profile_counts[label] = profile_counts.get(label, 0) + 1

        def stageii_forward(
            betas_tensor: torch.Tensor,
            body_pose_tensor: torch.Tensor,
            global_orient_tensor: torch.Tensor,
            transl_tensor: torch.Tensor,
        ) -> torch.Tensor:
            return smplx_forward(model, betas_tensor, body_pose_tensor, global_orient_tensor, transl_tensor, 1)

        if args.stageii_torch_compile:
            stageii_forward = torch.compile(  # type: ignore[assignment]
                stageii_forward,
                mode=args.stageii_compile_mode,
                fullgraph=args.stageii_compile_fullgraph,
            )

        fd_basis = torch.eye(69, dtype=dtype, device=device)
        fd_model = None
        fd_betas = None
        if args.stageii_dogleg_jacobian_mode == "fd":
            fd_model = create_smplx_model(model_file, batch_size=70, gender=args.gender, num_betas=args.num_betas, device=device)
            fd_betas = betas.expand(70, -1)

        def apply_pose_mask(body_pose_tensor: torch.Tensor) -> torch.Tensor:
            if args.stageii_freeze_toes or args.stageii_mosh_loss:
                masked = body_pose_tensor.clone()
                masked[:, 27:33] = 0.0
                return masked
            return body_pose_tensor

        wrist_pose_slice = slice(3 + 19 * 3, 3 + 21 * 3)

        def robust_vector_residual(diff: torch.Tensor, sigma: float) -> torch.Tensor:
            if sigma <= 0.0:
                return diff
            sigma_t = torch.as_tensor(float(sigma), dtype=diff.dtype, device=diff.device)
            sq = torch.sum(diff * diff, dim=-1, keepdim=True)
            rho2 = 2.0 * sigma_t * sigma_t * (torch.sqrt(1.0 + sq / (sigma_t * sigma_t)) - 1.0)
            scale = torch.sqrt(rho2 / sq.clamp_min(1e-12))
            return diff * scale

        def wrist_temporal_residual_values(diff: torch.Tensor) -> torch.Tensor:
            weight = float(args.stageii_wrist_velocity_weight)
            if weight <= 0.0:
                return diff[..., :0]
            wrist_diff = diff[..., wrist_pose_slice].reshape(*diff.shape[:-1], 2, 3)
            return weight * robust_vector_residual(wrist_diff, float(args.stageii_wrist_velocity_sigma)).reshape(*diff.shape[:-1], 6)

        def wrist_temporal_residual_from_diff(diff: torch.Tensor) -> torch.Tensor:
            return wrist_temporal_residual_values(diff).reshape(-1)

        def single_wrist_velocity_residual(pose_vec: torch.Tensor) -> torch.Tensor:
            if prev_pose_vec_seq is None or prev_prev_pose_vec_seq is None:
                return pose_vec[:0]
            prev = torch.as_tensor(prev_pose_vec_seq, dtype=pose_vec.dtype, device=pose_vec.device)
            prev_prev = torch.as_tensor(prev_prev_pose_vec_seq, dtype=pose_vec.dtype, device=pose_vec.device)
            return wrist_temporal_residual_from_diff(pose_vec - (prev + (prev - prev_prev)))

        def single_wrist_velocity_sse(pose_vec: torch.Tensor) -> torch.Tensor:
            res = single_wrist_velocity_residual(pose_vec)
            return torch.sum(res * res)

        def pose_prior_residual_rows(body_pose_tensor: torch.Tensor) -> torch.Tensor:
            if not pose_prior.valid:
                return body_pose_tensor
            diff = body_pose_tensor[:, None, :] - pose_prior.means[None, :, :]
            whitened = torch.einsum("bki,kij->bkj", diff, pose_prior.chols)
            residual_sse = 0.5 * torch.sum(whitened * whitened, dim=-1) - pose_prior.log_weights[None, :]
            comp = torch.argmin(residual_sse, dim=1)
            batch = torch.arange(body_pose_tensor.shape[0], device=body_pose_tensor.device)
            selected = math.sqrt(0.5) * whitened[batch, comp]
            weight_res = torch.sqrt(torch.clamp(-pose_prior.log_weights[comp], min=0.0))[:, None]
            return torch.cat([selected, weight_res], dim=1)

        if args.stageii_independent_block_size > 1:
            if args.stageii_solver != "dogleg":
                raise ValueError("--stageii-independent-block-size > 1 currently supports --stageii-solver dogleg only.")
            from torch.func import jacfwd, vmap

            iblock_size = max(1, int(args.stageii_independent_block_size))
            imodel = create_smplx_model(model_file, batch_size=1, gender=args.gender, num_betas=args.num_betas, device=device)
            if nn_vids is None:
                raise ValueError("--stageii-independent-block-size currently requires NN marker coefficients.")
            ibetas = betas
            mask_scale = math.sqrt(args.stageii_mosh_weight_scale)

            def single_independent_residual(
                x_i: torch.Tensor,
                obs_i: torch.Tensor,
                mask_i: torch.Tensor,
                velo_target_i: torch.Tensor,
                has_velo_i: torch.Tensor,
                pose_weight_i: torch.Tensor,
            ) -> torch.Tensor:
                root_i = x_i[:3].reshape(1, 3)
                body_raw_i = x_i[3:66].reshape(1, SMPLX_BODY_DOF)
                trans_i = x_i[66:69].reshape(1, 3)
                body_i = apply_pose_mask(body_raw_i)
                verts_i = smplx_forward(imodel, ibetas, body_i, root_i, trans_i, 1)
                pred_i = reconstruct_markers_nn(verts_i, nn_vids, coeffs)[0]
                marker_count = mask_i.sum().clamp_min(1.0)
                wt_data = 400.0 * (46.0 / marker_count)
                obs_clean = torch.nan_to_num(obs_i, nan=0.0)
                data_res = wt_data * (pred_i - obs_clean) * mask_i[:, None]
                pose_res = (1.6 * pose_weight_i) * pose_prior.residual(body_i)
                pose_vec = torch.cat([root_i.reshape(-1), body_i.reshape(-1)])
                velo_diff = pose_vec - velo_target_i
                velo_res = has_velo_i * 2.5 * velo_diff
                wrist_res = has_velo_i.reshape(-1) * wrist_temporal_residual_from_diff(velo_diff)
                return mask_scale * torch.cat([data_res.reshape(-1), pose_res.reshape(-1), velo_res.reshape(-1), wrist_res], dim=0)

            batched_residual = vmap(single_independent_residual, in_dims=(0, 0, 0, 0, 0, None))
            batched_jacobian = vmap(jacfwd(single_independent_residual), in_dims=(0, 0, 0, 0, 0, None))

            def make_velocity_targets(bsz: int) -> Tuple[torch.Tensor, torch.Tensor]:
                targets = np.zeros((bsz, 66), dtype=np.float32)
                has = np.zeros((bsz, 1), dtype=np.float32)
                if prev_pose_vec_seq is not None and prev_prev_pose_vec_seq is not None:
                    delta = prev_pose_vec_seq - prev_prev_pose_vec_seq
                    for j in range(bsz):
                        targets[j] = (prev_pose_vec_seq + (j + 1) * delta).astype(np.float32)
                        has[j, 0] = 1.0
                return (
                    torch.as_tensor(targets, dtype=dtype, device=device),
                    torch.as_tensor(has, dtype=dtype, device=device),
                )

            def run_independent_dogleg(
                x: torch.Tensor,
                obs_block: torch.Tensor,
                mask_block: torch.Tensor,
                velo_targets: torch.Tensor,
                has_velo: torch.Tensor,
                pose_weight_multiplier: float,
                max_iter: int,
            ) -> Tuple[torch.Tensor, float]:
                bsz = x.shape[0]
                trust_radius = torch.full((bsz,), float(args.stageii_dogleg_delta), dtype=dtype, device=device)
                max_trust_radius = float(args.stageii_dogleg_max_delta)
                eye = torch.eye(69, dtype=dtype, device=device).expand(bsz, 69, 69)
                pose_weight = torch.tensor(float(pose_weight_multiplier), dtype=dtype, device=device)
                best_loss = math.inf
                jac_refresh = max(1, int(args.stageii_dogleg_jacobian_refresh))
                jac: Optional[torch.Tensor] = None
                jtj: Optional[torch.Tensor] = None
                force_refresh = True
                jac_age = jac_refresh
                for iter_idx in range(max_iter):
                    residual = batched_residual(x, obs_block, mask_block.float(), velo_targets, has_velo, pose_weight)
                    loss_per = 0.5 * torch.sum(residual * residual, dim=1)
                    best_loss = float((2.0 * torch.sum(loss_per)).detach().cpu())
                    refresh_due = force_refresh or jac is None or jtj is None
                    if args.stageii_dogleg_adaptive_refresh:
                        refresh_due = refresh_due or jac_age >= jac_refresh
                    else:
                        refresh_due = refresh_due or iter_idx % jac_refresh == 0
                    if refresh_due:
                        jac = batched_jacobian(x, obs_block, mask_block.float(), velo_targets, has_velo, pose_weight).detach()
                        jtj = torch.bmm(jac.transpose(1, 2), jac)
                        force_refresh = False
                        jac_age = 0
                    grad = torch.bmm(jac.transpose(1, 2), residual.detach()[:, :, None])[:, :, 0]
                    try:
                        p_gn = torch.linalg.solve(jtj + args.stageii_dogleg_solve_damping * eye, -grad)
                    except RuntimeError:
                        p_gn = torch.linalg.lstsq(jtj + args.stageii_dogleg_solve_damping * eye, -grad[:, :, None]).solution[:, :, 0]
                    p_gn_norm = torch.linalg.norm(p_gn, dim=1)
                    grad_norm = torch.linalg.norm(grad, dim=1).clamp_min(1e-12)
                    bg = torch.bmm(jtj, grad[:, :, None])[:, :, 0]
                    denom = torch.sum(grad * bg, dim=1)
                    alpha = torch.where(denom > 1e-12, torch.sum(grad * grad, dim=1) / denom.clamp_min(1e-12), trust_radius / grad_norm)
                    p_u = -alpha[:, None] * grad
                    p_u_norm = torch.linalg.norm(p_u, dim=1).clamp_min(1e-12)
                    full_gn = p_gn_norm <= trust_radius
                    steepest = p_u_norm >= trust_radius
                    step = torch.empty_like(x)
                    step[full_gn] = p_gn[full_gn]
                    step[~full_gn & steepest] = trust_radius[~full_gn & steepest, None] * p_u[~full_gn & steepest] / p_u_norm[~full_gn & steepest, None]
                    mid = ~full_gn & ~steepest
                    if bool(mid.any()):
                        dogleg_dir = p_gn[mid] - p_u[mid]
                        a = torch.sum(dogleg_dir * dogleg_dir, dim=1)
                        b = 2.0 * torch.sum(p_u[mid] * dogleg_dir, dim=1)
                        c = torch.sum(p_u[mid] * p_u[mid], dim=1) - trust_radius[mid] * trust_radius[mid]
                        disc = torch.clamp(b * b - 4.0 * a * c, min=0.0)
                        tau = (-b + torch.sqrt(disc)) / (2.0 * a).clamp_min(1e-12)
                        step[mid] = p_u[mid] + tau[:, None] * dogleg_dir
                    pred_reduction = -(torch.sum(grad * step, dim=1) + 0.5 * torch.sum(step * torch.bmm(jtj, step[:, :, None])[:, :, 0], dim=1))
                    trial = x.detach() + step
                    trial_res = batched_residual(trial, obs_block, mask_block.float(), velo_targets, has_velo, pose_weight)
                    trial_loss_per = 0.5 * torch.sum(trial_res * trial_res, dim=1)
                    rho = (loss_per - trial_loss_per) / pred_reduction.clamp_min(1e-12)
                    finite = torch.isfinite(trial_loss_per) & torch.isfinite(rho)
                    accept = finite & (rho > args.stageii_dogleg_eta)
                    x = torch.where(accept[:, None], trial.detach(), x.detach())
                    best_loss = float((2.0 * torch.sum(torch.where(accept, trial_loss_per, loss_per))).detach().cpu())
                    trust_radius = torch.where(rho < 0.25, trust_radius * 0.25, trust_radius)
                    expand = (rho > 0.75) & (torch.abs(torch.linalg.norm(step, dim=1) - trust_radius) <= 1e-5 * torch.clamp(trust_radius, min=1.0))
                    trust_radius = torch.where(expand, torch.clamp(2.0 * trust_radius, max=max_trust_radius), trust_radius)
                    jac_age += 1
                    if bool((~accept).any()) or (args.stageii_dogleg_adaptive_refresh and float(torch.min(rho.detach()).cpu()) < args.stageii_dogleg_refresh_rho):
                        force_refresh = True
                return x.detach(), best_loss

            for block_start in range(0, nframes, iblock_size):
                block_end = min(nframes, block_start + iblock_size)
                bsz = block_end - block_start
                obs = obs_all[block_start:block_end]
                mask = mask_all[block_start:block_end]
                root_init = rigid_root_init[block_start:block_end].copy()
                body_init = np.repeat(prev_body_seq[None], bsz, axis=0)
                trans_init = rigid_trans_init[block_start:block_end].copy()
                if block_start > 0:
                    root_init[0] = prev_root_seq
                    trans_init[0] = prev_trans_seq
                x = torch.cat(
                    [
                        torch.as_tensor(root_init, dtype=dtype, device=device),
                        torch.as_tensor(body_init, dtype=dtype, device=device),
                        torch.as_tensor(trans_init, dtype=dtype, device=device),
                    ],
                    dim=1,
                )
                velo_targets, has_velo = make_velocity_targets(bsz)
                last_loss = math.inf
                if block_start == 0:
                    for mult in (10.0, 5.0, 1.0):
                        x, last_loss = run_independent_dogleg(x, obs, mask, velo_targets, has_velo, mult, args.stageii_dogleg_first_iters)
                regular_iters = max(1, args.stageii_lbfgs_passes) * max(1, args.stageii_dogleg_iters) if args.stageii_dogleg_merge_passes else args.stageii_dogleg_iters
                x, last_loss = run_independent_dogleg(x, obs, mask, velo_targets, has_velo, 1.0, regular_iters)
                if not args.stageii_dogleg_merge_passes:
                    for _ in range(max(0, args.stageii_lbfgs_passes - 1)):
                        x, last_loss = run_independent_dogleg(x, obs, mask, velo_targets, has_velo, 1.0, args.stageii_dogleg_iters)

                with torch.no_grad():
                    root_t = x[:, :3]
                    body_t = apply_pose_mask(x[:, 3:66])
                    trans_t = x[:, 66:69]
                    verts = smplx_forward(
                        create_smplx_model(model_file, batch_size=bsz, gender=args.gender, num_betas=args.num_betas, device=device),
                        betas.expand(bsz, -1),
                        body_t,
                        root_t,
                        trans_t,
                        bsz,
                    )
                    pred = reconstruct_markers_nn(verts, nn_vids, coeffs)
                root_np = root_t.detach().cpu().numpy().astype(np.float64)
                body_np = body_t.detach().cpu().numpy().astype(np.float64)
                trans_np = trans_t.detach().cpu().numpy().astype(np.float64)
                root_all[block_start:block_end] = root_np
                body_all[block_start:block_end] = body_np
                trans_all[block_start:block_end] = trans_np
                for j in range(bsz):
                    if prev_pose_vec_seq is not None:
                        prev_prev_pose_vec_seq = prev_pose_vec_seq.copy()
                    prev_pose_vec_seq = np.concatenate([root_np[j], body_np[j]]).astype(np.float64)
                    prev_root_seq, prev_body_seq, prev_trans_seq = root_np[j], body_np[j], trans_np[j]
                pred_np = pred.detach().cpu().numpy().astype(np.float64)
                obs_np_block = obs.detach().cpu().numpy().astype(np.float64)
                markers_sim_debug.extend([x_frame for x_frame in pred_np])
                markers_obs_debug.extend([x_frame for x_frame in obs_np_block])
                labels_obs_debug.extend(
                    [[label for lid, label in enumerate(latent_labels) if bool(mask[j, lid])] for j in range(bsz)]
                )
                errs.extend([last_loss] * bsz)
                if args.verbose and (block_start == 0 or block_end % args.log_every == 0 or block_end == nframes):
                    print(
                        f"stage II independent block {block_start + 1:05d}-{block_end:05d}/{nframes}: "
                        f"loss={last_loss:.6f}"
                    )

            fullpose = fullpose_from_parts(root_all, body_all, hand_mean_np)
            stageii_data = {
                "fullpose": fullpose,
                "trans": trans_all,
                "stageii_debug_details": {
                    "stageii_errs": {"loss": np.asarray(errs, dtype=np.float64)},
                    "stageii_fallback_frames": np.asarray(fallback_frames, dtype=np.int64),
                    "markers_sim": markers_sim_debug,
                    "markers_obs": markers_obs_debug,
                    "labels_obs": labels_obs_debug,
                    "markers_orig": obs_np,
                    "labels_orig": mocap.labels,
                    "mocap_fname": args.mocap,
                    "mocap_frame_rate": mocap.frame_rate,
                    "mocap_time_length": nframes / mocap.frame_rate,
                    "stageii_elapsed_time": time.time() - started,
                    "stageii_profile_times": dict(profile_times),
                    "stageii_profile_counts": dict(profile_counts),
                    "cfg": vars(args),
                },
                "betas": stagei_data["betas"],
                "markers_latent": stagei_data["markers_latent"],
                "latent_labels": latent_labels,
                "marker_meta": numpy_marker_meta(marker_meta),
                "markers_latent_vids": stagei_data["markers_latent_vids"],
                "stagei_debug_details": stagei_data.get("stagei_debug_details", {}),
            }
            for key in ("torch_marker_anchor_vids", "torch_marker_coeffs", "torch_marker_tangent_vids", "torch_marker_nn_vids"):
                if key in stagei_data:
                    stageii_data[key] = stagei_data[key]
            return stageii_data

        if args.stageii_block_size > 1:
            if args.stageii_solver != "dogleg":
                raise ValueError("--stageii-block-size > 1 currently supports --stageii-solver dogleg only.")

            block_size = max(1, int(args.stageii_block_size))
            block_models = {
                block_size: create_smplx_model(
                    model_file,
                    batch_size=block_size,
                    gender=args.gender,
                    num_betas=args.num_betas,
                    device=device,
                )
            }
            block_model = block_models[block_size]
            block_faces = make_face_tensor(block_model, device)
            fd_block_models = {}

            def get_block_model(bsz: int):
                if bsz not in block_models:
                    block_models[bsz] = create_smplx_model(
                        model_file,
                        batch_size=bsz,
                        gender=args.gender,
                        num_betas=args.num_betas,
                        device=device,
                    )
                return block_models[bsz]

            def get_fd_block_model(total_batch: int):
                if total_batch not in fd_block_models:
                    fd_block_models[total_batch] = create_smplx_model(
                        model_file,
                        batch_size=total_batch,
                        gender=args.gender,
                        num_betas=args.num_betas,
                        device=device,
                    )
                return fd_block_models[total_batch]

            def block_forward(
                bsz: int,
                body_pose_tensor: torch.Tensor,
                global_orient_tensor: torch.Tensor,
                transl_tensor: torch.Tensor,
            ) -> torch.Tensor:
                return smplx_forward(
                    get_block_model(bsz),
                    betas.expand(bsz, -1),
                    body_pose_tensor,
                    global_orient_tensor,
                    transl_tensor,
                    bsz,
                )

            def block_velocity_residual(pose_vecs: torch.Tensor) -> torch.Tensor:
                residuals = []
                bsz = pose_vecs.shape[0]
                if bsz == 0:
                    return pose_vecs.reshape(-1)
                if prev_pose_vec_seq is not None and prev_prev_pose_vec_seq is not None:
                    prev = torch.as_tensor(prev_pose_vec_seq, dtype=pose_vecs.dtype, device=pose_vecs.device)
                    prev_prev = torch.as_tensor(prev_prev_pose_vec_seq, dtype=pose_vecs.dtype, device=pose_vecs.device)
                    residuals.append(pose_vecs[0] - (prev + (prev - prev_prev)))
                if bsz >= 2 and prev_pose_vec_seq is not None:
                    prev = torch.as_tensor(prev_pose_vec_seq, dtype=pose_vecs.dtype, device=pose_vecs.device)
                    residuals.append(pose_vecs[1] - (pose_vecs[0] + (pose_vecs[0] - prev)))
                if bsz >= 3:
                    residuals.append(pose_vecs[2:] - (pose_vecs[1:-1] + (pose_vecs[1:-1] - pose_vecs[:-2])))
                if not residuals:
                    return pose_vecs[:0].reshape(-1)
                return torch.cat([r.reshape(-1) for r in residuals], dim=0)

            def block_wrist_velocity_residual(pose_vecs: torch.Tensor) -> torch.Tensor:
                residuals = []
                bsz = pose_vecs.shape[0]
                if bsz == 0 or float(args.stageii_wrist_velocity_weight) <= 0.0:
                    return pose_vecs[:0].reshape(-1)
                if prev_pose_vec_seq is not None and prev_prev_pose_vec_seq is not None:
                    prev = torch.as_tensor(prev_pose_vec_seq, dtype=pose_vecs.dtype, device=pose_vecs.device)
                    prev_prev = torch.as_tensor(prev_prev_pose_vec_seq, dtype=pose_vecs.dtype, device=pose_vecs.device)
                    residuals.append(wrist_temporal_residual_from_diff(pose_vecs[0] - (prev + (prev - prev_prev))))
                if bsz >= 2 and prev_pose_vec_seq is not None:
                    prev = torch.as_tensor(prev_pose_vec_seq, dtype=pose_vecs.dtype, device=pose_vecs.device)
                    residuals.append(wrist_temporal_residual_from_diff(pose_vecs[1] - (pose_vecs[0] + (pose_vecs[0] - prev))))
                if bsz >= 3:
                    residuals.append(wrist_temporal_residual_from_diff(pose_vecs[2:] - (pose_vecs[1:-1] + (pose_vecs[1:-1] - pose_vecs[:-2]))))
                if not residuals:
                    return pose_vecs[:0].reshape(-1)
                return torch.cat([r.reshape(-1) for r in residuals], dim=0)

            def unpack_block_state(x: torch.Tensor, bsz: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                root_x = x[: bsz * 3].reshape(bsz, 3)
                body_x_raw = x[bsz * 3: bsz * (3 + SMPLX_BODY_DOF)].reshape(bsz, SMPLX_BODY_DOF)
                trans_x = x[bsz * (3 + SMPLX_BODY_DOF):].reshape(bsz, 3)
                return root_x, body_x_raw, trans_x

            def block_residual(
                x: torch.Tensor,
                bsz: int,
                obs_block: torch.Tensor,
                mask_block: torch.Tensor,
                pose_weight_multiplier: float = 1.0,
            ) -> torch.Tensor:
                root_x, body_x_raw, trans_x = unpack_block_state(x, bsz)
                body_x = apply_pose_mask(body_x_raw)
                verts_x = block_forward(bsz, body_x, root_x, trans_x)
                if nn_vids is not None:
                    pred_x = reconstruct_markers_nn(verts_x, nn_vids, coeffs)
                else:
                    pred_x = reconstruct_markers(verts_x, block_faces, marker_vids, tangent_vids, coeffs)
                marker_count = mask_block.float().sum(dim=1).clamp_min(1.0)
                wt_data = 400.0 * (46.0 / marker_count)
                data_res = (wt_data[:, None, None] * (pred_x - torch.nan_to_num(obs_block, nan=0.0)))
                data_res = data_res[mask_block[:, :, None].expand_as(data_res)]
                pose_res = (1.6 * pose_weight_multiplier) * pose_prior.residual(body_x)
                pose_vecs = torch.cat([root_x, body_x], dim=1)
                velo_res = 2.5 * block_velocity_residual(pose_vecs)
                wrist_res = block_wrist_velocity_residual(pose_vecs)
                scale = math.sqrt(args.stageii_mosh_weight_scale)
                return scale * torch.cat([data_res.reshape(-1), pose_res, velo_res.reshape(-1), wrist_res], dim=0)

            def block_residual_rows(
                x_rows: torch.Tensor,
                bsz: int,
                obs_block: torch.Tensor,
                mask_block: torch.Tensor,
                pose_weight_multiplier: float = 1.0,
            ) -> torch.Tensor:
                nrows = x_rows.shape[0]
                root_x = x_rows[:, : bsz * 3].reshape(nrows, bsz, 3)
                body_x_raw = x_rows[:, bsz * 3: bsz * (3 + SMPLX_BODY_DOF)].reshape(nrows, bsz, SMPLX_BODY_DOF)
                trans_x = x_rows[:, bsz * (3 + SMPLX_BODY_DOF):].reshape(nrows, bsz, 3)
                body_x = apply_pose_mask(body_x_raw.reshape(nrows * bsz, SMPLX_BODY_DOF)).reshape(nrows, bsz, SMPLX_BODY_DOF)
                total_batch = nrows * bsz
                fd_model_local = get_fd_block_model(total_batch)
                verts_x = smplx_forward(
                    fd_model_local,
                    betas.expand(total_batch, -1),
                    body_x.reshape(total_batch, SMPLX_BODY_DOF),
                    root_x.reshape(total_batch, 3),
                    trans_x.reshape(total_batch, 3),
                    total_batch,
                ).reshape(nrows, bsz, -1, 3)
                if nn_vids is not None:
                    pred_x = reconstruct_markers_nn(verts_x.reshape(total_batch, -1, 3), nn_vids, coeffs).reshape(nrows, bsz, -1, 3)
                else:
                    pred_x = reconstruct_markers(
                        verts_x.reshape(total_batch, -1, 3),
                        block_faces,
                        marker_vids,
                        tangent_vids,
                        coeffs,
                    ).reshape(nrows, bsz, -1, 3)
                marker_count = mask_block.float().sum(dim=1).clamp_min(1.0)
                wt_data = 400.0 * (46.0 / marker_count)
                data_res = wt_data[None, :, None, None] * (pred_x - torch.nan_to_num(obs_block, nan=0.0)[None])
                data_res = data_res * mask_block.float()[None, :, :, None]
                pose_res = (1.6 * pose_weight_multiplier) * pose_prior_residual_rows(
                    body_x.reshape(total_batch, SMPLX_BODY_DOF)
                ).reshape(nrows, bsz, -1)
                pose_vecs = torch.cat([root_x, body_x], dim=2)
                velo_parts = []
                if prev_pose_vec_seq is not None and prev_prev_pose_vec_seq is not None:
                    prev = torch.as_tensor(prev_pose_vec_seq, dtype=x_rows.dtype, device=x_rows.device)
                    prev_prev = torch.as_tensor(prev_prev_pose_vec_seq, dtype=x_rows.dtype, device=x_rows.device)
                    velo_parts.append(pose_vecs[:, 0] - (prev + (prev - prev_prev))[None])
                if bsz >= 2 and prev_pose_vec_seq is not None:
                    prev = torch.as_tensor(prev_pose_vec_seq, dtype=x_rows.dtype, device=x_rows.device)
                    velo_parts.append(pose_vecs[:, 1] - (pose_vecs[:, 0] + (pose_vecs[:, 0] - prev[None])))
                if bsz >= 3:
                    velo_parts.append((pose_vecs[:, 2:] - (pose_vecs[:, 1:-1] + (pose_vecs[:, 1:-1] - pose_vecs[:, :-2]))).reshape(nrows, -1))
                if velo_parts:
                    velo_res = torch.cat([v.reshape(nrows, -1) for v in velo_parts], dim=1)
                else:
                    velo_res = x_rows[:, :0]
                wrist_parts = []
                if float(args.stageii_wrist_velocity_weight) > 0.0:
                    if prev_pose_vec_seq is not None and prev_prev_pose_vec_seq is not None:
                        prev = torch.as_tensor(prev_pose_vec_seq, dtype=x_rows.dtype, device=x_rows.device)
                        prev_prev = torch.as_tensor(prev_prev_pose_vec_seq, dtype=x_rows.dtype, device=x_rows.device)
                        wrist_parts.append(wrist_temporal_residual_values(pose_vecs[:, 0] - (prev + (prev - prev_prev))[None]))
                    if bsz >= 2 and prev_pose_vec_seq is not None:
                        prev = torch.as_tensor(prev_pose_vec_seq, dtype=x_rows.dtype, device=x_rows.device)
                        wrist_parts.append(wrist_temporal_residual_values(pose_vecs[:, 1] - (pose_vecs[:, 0] + (pose_vecs[:, 0] - prev[None]))))
                    if bsz >= 3:
                        wrist_parts.append(
                            wrist_temporal_residual_values(
                                pose_vecs[:, 2:] - (pose_vecs[:, 1:-1] + (pose_vecs[:, 1:-1] - pose_vecs[:, :-2]))
                            ).reshape(nrows, -1)
                        )
                if wrist_parts:
                    wrist_res = torch.cat([w.reshape(nrows, -1) for w in wrist_parts], dim=1)
                else:
                    wrist_res = x_rows[:, :0]
                scale = math.sqrt(args.stageii_mosh_weight_scale)
                return scale * torch.cat(
                    [
                        data_res.reshape(nrows, -1),
                        pose_res.reshape(nrows, -1),
                        2.5 * velo_res.reshape(nrows, -1),
                        wrist_res.reshape(nrows, -1),
                    ],
                    dim=1,
                )

            def finite_difference_block_jacobian(
                x: torch.Tensor,
                bsz: int,
                obs_block: torch.Tensor,
                mask_block: torch.Tensor,
                pose_weight_multiplier: float = 1.0,
            ) -> Tuple[torch.Tensor, torch.Tensor]:
                eps = float(args.stageii_dogleg_fd_eps)
                nvars = int(x.numel())
                basis = torch.eye(nvars, dtype=x.dtype, device=x.device)
                residual_rows = block_residual_rows(
                    torch.cat([x[None], x[None] + eps * basis], dim=0),
                    bsz,
                    obs_block,
                    mask_block,
                    pose_weight_multiplier,
                )
                residual = residual_rows[0]
                jac = (residual_rows[1:] - residual[None]) / eps
                return residual, jac.transpose(0, 1).contiguous()

            def run_block_dogleg(
                x: torch.Tensor,
                bsz: int,
                obs_block: torch.Tensor,
                mask_block: torch.Tensor,
                pose_weight_multiplier: float,
                max_iter: int,
            ) -> Tuple[torch.Tensor, float]:
                trust_radius = float(args.stageii_dogleg_delta) * math.sqrt(float(bsz))
                max_trust_radius = float(args.stageii_dogleg_max_delta) * math.sqrt(float(bsz))
                nvars = int(x.numel())
                eye = torch.eye(nvars, dtype=dtype, device=device)
                best_loss = math.inf
                jac_refresh = max(1, int(args.stageii_dogleg_jacobian_refresh))
                jac: Optional[torch.Tensor] = None
                jtj: Optional[torch.Tensor] = None
                force_refresh = True
                jac_age = jac_refresh
                for iter_idx in range(max_iter):
                    x_req = x.detach().requires_grad_(True)

                    def residual_for_jac(y: torch.Tensor) -> torch.Tensor:
                        return block_residual(y, bsz, obs_block, mask_block, pose_weight_multiplier)

                    if args.stageii_dogleg_jacobian_mode == "fd":
                        residual = block_residual_rows(
                            x.detach()[None],
                            bsz,
                            obs_block,
                            mask_block,
                            pose_weight_multiplier,
                        )[0]
                    else:
                        residual = residual_for_jac(x_req)
                    loss_value = 0.5 * torch.sum(residual * residual)
                    best_loss = float((2.0 * loss_value).detach().cpu())
                    refresh_due = force_refresh or jac is None or jtj is None
                    if args.stageii_dogleg_adaptive_refresh:
                        refresh_due = refresh_due or jac_age >= jac_refresh
                    else:
                        refresh_due = refresh_due or iter_idx % jac_refresh == 0
                    refreshed_jac = False
                    if refresh_due:
                        if args.stageii_dogleg_jacobian_mode == "fd":
                            residual, jac = finite_difference_block_jacobian(
                                x.detach(),
                                bsz,
                                obs_block,
                                mask_block,
                                pose_weight_multiplier,
                            )
                            loss_value = 0.5 * torch.sum(residual * residual)
                            best_loss = float((2.0 * loss_value).detach().cpu())
                        else:
                            try:
                                jac = torch.autograd.functional.jacobian(
                                    residual_for_jac,
                                    x_req,
                                    vectorize=True,
                                    strategy="forward-mode",
                                )
                            except Exception:
                                jac = torch.autograd.functional.jacobian(residual_for_jac, x_req, vectorize=True)
                        jac = jac.detach()
                        jtj = jac.transpose(0, 1) @ jac
                        force_refresh = False
                        jac_age = 0
                        refreshed_jac = True
                    grad = jac.transpose(0, 1) @ residual.detach()
                    grad_norm = torch.linalg.norm(grad)
                    if float(grad_norm.detach().cpu()) < args.stageii_dogleg_grad_tol:
                        break
                    try:
                        p_gn = torch.linalg.solve(jtj + args.stageii_dogleg_solve_damping * eye, -grad)
                    except RuntimeError:
                        p_gn = torch.linalg.lstsq(jtj + args.stageii_dogleg_solve_damping * eye, -grad[:, None]).solution[:, 0]
                    p_gn_norm = torch.linalg.norm(p_gn)
                    if float(p_gn_norm.detach().cpu()) <= trust_radius:
                        step = p_gn
                    else:
                        bg = jtj @ grad
                        denom = torch.dot(grad, bg)
                        if float(denom.detach().cpu()) <= 1e-12:
                            p_u = -trust_radius * grad / grad_norm.clamp_min(1e-12)
                        else:
                            p_u = -(torch.dot(grad, grad) / denom) * grad
                        p_u_norm = torch.linalg.norm(p_u)
                        if float(p_u_norm.detach().cpu()) >= trust_radius:
                            step = trust_radius * p_u / p_u_norm.clamp_min(1e-12)
                        else:
                            dogleg_dir = p_gn - p_u
                            a = torch.dot(dogleg_dir, dogleg_dir)
                            b = 2.0 * torch.dot(p_u, dogleg_dir)
                            c = torch.dot(p_u, p_u) - trust_radius * trust_radius
                            disc = torch.clamp(b * b - 4.0 * a * c, min=0.0)
                            tau = (-b + torch.sqrt(disc)) / (2.0 * a).clamp_min(1e-12)
                            step = p_u + tau * dogleg_dir

                    pred_reduction = -(torch.dot(grad, step) + 0.5 * torch.dot(step, jtj @ step))
                    if float(pred_reduction.detach().cpu()) <= 1e-12:
                        trust_radius *= 0.25
                        if trust_radius < args.stageii_dogleg_min_delta:
                            break
                        continue
                    trial = x.detach() + step
                    trial_residual = block_residual(trial, bsz, obs_block, mask_block, pose_weight_multiplier)
                    trial_loss = 0.5 * torch.sum(trial_residual * trial_residual)
                    rho = (loss_value - trial_loss) / pred_reduction
                    rho_value = float(rho.detach().cpu()) if torch.isfinite(rho) else -math.inf
                    step_norm = float(torch.linalg.norm(step).detach().cpu())
                    if rho_value < 0.25:
                        trust_radius *= 0.25
                        force_refresh = True
                    elif rho_value > 0.75 and abs(step_norm - trust_radius) <= 1e-5 * max(1.0, trust_radius):
                        trust_radius = min(2.0 * trust_radius, max_trust_radius)
                    if rho_value > args.stageii_dogleg_eta and torch.isfinite(trial_loss):
                        x = trial.detach()
                        best_loss = float((2.0 * trial_loss).detach().cpu())
                        jac_age += 1
                        if args.stageii_dogleg_adaptive_refresh:
                            if rho_value < args.stageii_dogleg_refresh_rho:
                                force_refresh = True
                        elif rho_value < 0.5:
                            force_refresh = True
                        if step_norm < args.stageii_lm_step_tol:
                            break
                    elif trust_radius < args.stageii_dogleg_min_delta:
                        break
                    else:
                        force_refresh = True
                        if not refreshed_jac:
                            jac_age += 1
                return x.detach(), best_loss

            for block_start in range(0, nframes, block_size):
                block_end = min(nframes, block_start + block_size)
                bsz = block_end - block_start
                obs = obs_all[block_start:block_end]
                mask = mask_all[block_start:block_end]
                root_init = rigid_root_init[block_start:block_end].copy()
                body_init = np.repeat(prev_body_seq[None], bsz, axis=0)
                trans_init = rigid_trans_init[block_start:block_end].copy()
                if block_start > 0:
                    root_init[0] = prev_root_seq
                    trans_init[0] = prev_trans_seq
                x = torch.cat(
                    [
                        torch.as_tensor(root_init, dtype=dtype, device=device).reshape(-1),
                        torch.as_tensor(body_init, dtype=dtype, device=device).reshape(-1),
                        torch.as_tensor(trans_init, dtype=dtype, device=device).reshape(-1),
                    ],
                    dim=0,
                )
                last_loss = math.inf
                if block_start == 0:
                    for mult in (10.0, 5.0, 1.0):
                        x, last_loss = run_block_dogleg(x, bsz, obs, mask, mult, args.stageii_dogleg_first_iters)
                if args.stageii_dogleg_merge_passes:
                    regular_iters = max(1, args.stageii_lbfgs_passes) * max(1, args.stageii_dogleg_iters)
                    x, last_loss = run_block_dogleg(x, bsz, obs, mask, 1.0, regular_iters)
                else:
                    for _ in range(max(1, args.stageii_lbfgs_passes)):
                        x, last_loss = run_block_dogleg(x, bsz, obs, mask, 1.0, args.stageii_dogleg_iters)

                with torch.no_grad():
                    root_t, body_raw_t, trans_t = unpack_block_state(x, bsz)
                    body_t = apply_pose_mask(body_raw_t)
                    verts = block_forward(bsz, body_t, root_t, trans_t)
                    if nn_vids is not None:
                        pred = reconstruct_markers_nn(verts, nn_vids, coeffs)
                    else:
                        pred = reconstruct_markers(verts, block_faces, marker_vids, tangent_vids, coeffs)

                root_np = root_t.detach().cpu().numpy().astype(np.float64)
                body_np = body_t.detach().cpu().numpy().astype(np.float64)
                trans_np = trans_t.detach().cpu().numpy().astype(np.float64)
                root_all[block_start:block_end] = root_np
                body_all[block_start:block_end] = body_np
                trans_all[block_start:block_end] = trans_np
                for j in range(bsz):
                    if prev_pose_vec_seq is not None:
                        prev_prev_pose_vec_seq = prev_pose_vec_seq.copy()
                    prev_pose_vec_seq = np.concatenate([root_np[j], body_np[j]]).astype(np.float64)
                    prev_root_seq, prev_body_seq, prev_trans_seq = root_np[j], body_np[j], trans_np[j]
                pred_np = pred.detach().cpu().numpy().astype(np.float64)
                obs_np_block = obs.detach().cpu().numpy().astype(np.float64)
                markers_sim_debug.extend([x_frame for x_frame in pred_np])
                markers_obs_debug.extend([x_frame for x_frame in obs_np_block])
                labels_obs_debug.extend(
                    [[label for lid, label in enumerate(latent_labels) if bool(mask[j, lid])] for j in range(bsz)]
                )
                errs.extend([last_loss] * bsz)
                if args.verbose and (block_start == 0 or block_end % args.log_every == 0 or block_end == nframes):
                    print(
                        f"stage II block {block_start + 1:05d}-{block_end:05d}/{nframes}: "
                        f"loss={last_loss:.6f}"
                    )

            fullpose = fullpose_from_parts(root_all, body_all, hand_mean_np)
            stageii_data = {
                "fullpose": fullpose,
                "trans": trans_all,
                "stageii_debug_details": {
                    "stageii_errs": {"loss": np.asarray(errs, dtype=np.float64)},
                    "stageii_fallback_frames": np.asarray(fallback_frames, dtype=np.int64),
                    "markers_sim": markers_sim_debug,
                    "markers_obs": markers_obs_debug,
                    "labels_obs": labels_obs_debug,
                    "markers_orig": obs_np,
                    "labels_orig": mocap.labels,
                    "mocap_fname": args.mocap,
                    "mocap_frame_rate": mocap.frame_rate,
                    "mocap_time_length": nframes / mocap.frame_rate,
                    "stageii_elapsed_time": time.time() - started,
                    "stageii_profile_times": dict(profile_times),
                    "stageii_profile_counts": dict(profile_counts),
                    "cfg": vars(args),
                },
                "betas": stagei_data["betas"],
                "markers_latent": stagei_data["markers_latent"],
                "latent_labels": latent_labels,
                "marker_meta": numpy_marker_meta(marker_meta),
                "markers_latent_vids": stagei_data["markers_latent_vids"],
                "stagei_debug_details": stagei_data.get("stagei_debug_details", {}),
            }
            for key in ("torch_marker_anchor_vids", "torch_marker_coeffs", "torch_marker_tangent_vids", "torch_marker_nn_vids"):
                if key in stagei_data:
                    stageii_data[key] = stagei_data[key]
            return stageii_data

        for fidx in range(nframes):
            prof = profile_active(fidx)
            frame_started = profile_now(prof)
            if fidx == 0:
                root_init = rigid_root_init[fidx:fidx + 1]
                body_init = np.zeros((1, SMPLX_BODY_DOF), dtype=np.float64)
                trans_init = rigid_trans_init[fidx:fidx + 1]
            else:
                root_init = prev_root_seq[None]
                body_init = prev_body_seq[None]
                trans_init = prev_trans_seq[None]

            obs = obs_all[fidx:fidx + 1]
            mask = mask_all[fidx:fidx + 1]
            global_orient = torch.nn.Parameter(torch.as_tensor(root_init, dtype=dtype, device=device))
            body_pose = torch.nn.Parameter(torch.as_tensor(body_init, dtype=dtype, device=device))
            transl = torch.nn.Parameter(torch.as_tensor(trans_init, dtype=dtype, device=device))
            profile_add(prof, "frame_setup", frame_started)

            def frame_loss(pose_weight_multiplier: float = 1.0) -> torch.Tensor:
                body_eval = apply_pose_mask(body_pose)
                if args.stageii_torch_compile and hasattr(torch.compiler, "cudagraph_mark_step_begin"):
                    torch.compiler.cudagraph_mark_step_begin()
                verts = stageii_forward(betas, body_eval, global_orient, transl)
                if nn_vids is not None:
                    pred = reconstruct_markers_nn(verts, nn_vids, coeffs)
                else:
                    pred = reconstruct_markers(verts, faces, marker_vids, tangent_vids, coeffs)
                marker_count = mask.float().sum(dim=1).clamp_min(1.0)
                wt_data = 400.0 * (46.0 / marker_count)
                residual = pred - torch.nan_to_num(obs, nan=0.0)
                data_sse = torch.sum((wt_data[:, None] * wt_data[:, None]) * (residual * residual).sum(dim=-1) * mask.float())
                pose_sse = ((1.6 * pose_weight_multiplier) ** 2) * pose_prior.sse(body_eval)
                pose_vec = torch.cat([global_orient.reshape(-1), body_eval.reshape(-1)])
                velo_sse = (2.5 * 2.5) * single_pose_velocity_extrap_sse(
                    pose_vec,
                    prev_pose_vec_seq,
                    prev_prev_pose_vec_seq,
                )
                wrist_sse = single_wrist_velocity_sse(pose_vec)
                return args.stageii_mosh_weight_scale * (data_sse + pose_sse + velo_sse + wrist_sse)

            def frame_loss_x(x: torch.Tensor, pose_weight_multiplier: float = 1.0) -> torch.Tensor:
                root_x = x[:3].reshape(1, 3)
                body_x_raw = x[3:66].reshape(1, SMPLX_BODY_DOF)
                trans_x = x[66:69].reshape(1, 3)
                body_x = apply_pose_mask(body_x_raw)
                verts_x = stageii_forward(betas, body_x, root_x, trans_x)
                if nn_vids is not None:
                    pred_x = reconstruct_markers_nn(verts_x, nn_vids, coeffs)
                else:
                    pred_x = reconstruct_markers(verts_x, faces, marker_vids, tangent_vids, coeffs)
                marker_count = mask.float().sum(dim=1).clamp_min(1.0)
                wt_data = 400.0 * (46.0 / marker_count)
                residual = pred_x - torch.nan_to_num(obs, nan=0.0)
                data_sse = torch.sum((wt_data[:, None] * wt_data[:, None]) * (residual * residual).sum(dim=-1) * mask.float())
                pose_sse = ((1.6 * pose_weight_multiplier) ** 2) * pose_prior.sse(body_x)
                pose_vec = torch.cat([root_x.reshape(-1), body_x.reshape(-1)])
                velo_sse = (2.5 * 2.5) * single_pose_velocity_extrap_sse(
                    pose_vec,
                    prev_pose_vec_seq,
                    prev_prev_pose_vec_seq,
                )
                wrist_sse = single_wrist_velocity_sse(pose_vec)
                return args.stageii_mosh_weight_scale * (data_sse + pose_sse + velo_sse + wrist_sse)

            def lm_residual(x: torch.Tensor, pose_weight_multiplier: float = 1.0) -> torch.Tensor:
                root_x = x[:3].reshape(1, 3)
                body_x_raw = x[3:66].reshape(1, SMPLX_BODY_DOF)
                trans_x = x[66:69].reshape(1, 3)
                body_x = apply_pose_mask(body_x_raw)
                verts_x = stageii_forward(betas, body_x, root_x, trans_x)
                if nn_vids is not None:
                    pred_x = reconstruct_markers_nn(verts_x, nn_vids, coeffs)
                else:
                    pred_x = reconstruct_markers(verts_x, faces, marker_vids, tangent_vids, coeffs)
                marker_count = mask.float().sum().clamp_min(1.0)
                wt_data = 400.0 * (46.0 / marker_count)
                valid = mask[0]
                data_res = (wt_data * (pred_x[0, valid] - obs[0, valid])).reshape(-1)
                pose_res = (1.6 * pose_weight_multiplier) * pose_prior.residual(body_x)
                pose_vec = torch.cat([root_x.reshape(-1), body_x.reshape(-1)])
                if prev_pose_vec_seq is None or prev_prev_pose_vec_seq is None:
                    velo_res = pose_vec[:0]
                else:
                    prev = torch.as_tensor(prev_pose_vec_seq, dtype=x.dtype, device=x.device)
                    prev_prev = torch.as_tensor(prev_prev_pose_vec_seq, dtype=x.dtype, device=x.device)
                    velo_diff = pose_vec - (prev + (prev - prev_prev))
                    velo_res = 2.5 * velo_diff
                    wrist_res = wrist_temporal_residual_from_diff(velo_diff)
                if prev_pose_vec_seq is None or prev_prev_pose_vec_seq is None:
                    wrist_res = pose_vec[:0]
                scale = math.sqrt(args.stageii_mosh_weight_scale)
                return scale * torch.cat([data_res, pose_res, velo_res, wrist_res], dim=0)

            def lm_residual_rows(x_rows: torch.Tensor, pose_weight_multiplier: float = 1.0) -> torch.Tensor:
                nrows = x_rows.shape[0]
                root_x = x_rows[:, :3]
                body_x_raw = x_rows[:, 3:66]
                trans_x = x_rows[:, 66:69]
                body_x = apply_pose_mask(body_x_raw)
                verts_x = smplx_forward(fd_model, fd_betas, body_x, root_x, trans_x, nrows)
                if nn_vids is not None:
                    pred_x = reconstruct_markers_nn(verts_x, nn_vids, coeffs)
                else:
                    pred_x = reconstruct_markers(verts_x, faces, marker_vids, tangent_vids, coeffs)
                marker_count = mask.float().sum().clamp_min(1.0)
                wt_data = 400.0 * (46.0 / marker_count)
                data_res = wt_data * (pred_x - torch.nan_to_num(obs, nan=0.0))
                data_res = data_res * mask.float()[0, :, None]
                pose_res = (1.6 * pose_weight_multiplier) * pose_prior_residual_rows(body_x)
                pose_vec = torch.cat([root_x, body_x], dim=1)
                if prev_pose_vec_seq is None or prev_prev_pose_vec_seq is None:
                    velo_res = torch.zeros_like(pose_vec)
                    wrist_res = x_rows[:, :0]
                else:
                    prev = torch.as_tensor(prev_pose_vec_seq, dtype=x_rows.dtype, device=x_rows.device)
                    prev_prev = torch.as_tensor(prev_prev_pose_vec_seq, dtype=x_rows.dtype, device=x_rows.device)
                    velo_diff = pose_vec - (prev + (prev - prev_prev))[None]
                    velo_res = 2.5 * velo_diff
                    wrist_res = wrist_temporal_residual_values(velo_diff)
                scale = math.sqrt(args.stageii_mosh_weight_scale)
                return scale * torch.cat([data_res.reshape(nrows, -1), pose_res, velo_res, wrist_res], dim=1)

            def finite_difference_jacobian(x: torch.Tensor, pose_weight_multiplier: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
                eps = float(args.stageii_dogleg_fd_eps)
                x_rows = torch.cat([x[None], x[None] + eps * fd_basis], dim=0)
                residual_rows = lm_residual_rows(x_rows, pose_weight_multiplier)
                residual = residual_rows[0]
                jac = (residual_rows[1:] - residual[None]) / eps
                return residual, jac.transpose(0, 1).contiguous()

            def run_lm(pose_weight_multiplier: float, max_iter: int) -> float:
                damping = float(args.stageii_lm_damping)
                eye = torch.eye(69, dtype=dtype, device=device)
                x = torch.cat([global_orient.detach().reshape(-1), body_pose.detach().reshape(-1), transl.detach().reshape(-1)])
                best_loss = math.inf
                for _ in range(max_iter):
                    x_req = x.detach().requires_grad_(True)

                    def residual_for_jac(y: torch.Tensor) -> torch.Tensor:
                        return lm_residual(y, pose_weight_multiplier)

                    residual_started = profile_now(prof)
                    residual = residual_for_jac(x_req)
                    loss_value = torch.sum(residual * residual)
                    best_loss = float(loss_value.detach().cpu())
                    profile_add(prof, "lm_residual_current", residual_started)
                    jac_started = profile_now(prof)
                    try:
                        jac = torch.autograd.functional.jacobian(
                            residual_for_jac,
                            x_req,
                            vectorize=True,
                            strategy="forward-mode",
                        )
                    except Exception:
                        jac = torch.autograd.functional.jacobian(residual_for_jac, x_req, vectorize=True)
                    profile_add(prof, "lm_jacobian", jac_started)
                    normal_started = profile_now(prof)
                    jtj = jac.transpose(0, 1) @ jac
                    g = jac.transpose(0, 1) @ residual.detach()
                    profile_add(prof, "lm_normal_equations", normal_started)
                    accepted = False
                    for _attempt in range(6):
                        solve_started = profile_now(prof)
                        try:
                            step = torch.linalg.solve(jtj + damping * eye, -g)
                        except RuntimeError:
                            damping *= 10.0
                            profile_add(prof, "lm_linear_solve", solve_started)
                            continue
                        profile_add(prof, "lm_linear_solve", solve_started)
                        trial_started = profile_now(prof)
                        trial = x.detach() + step
                        trial_residual = lm_residual(trial, pose_weight_multiplier)
                        trial_loss = torch.sum(trial_residual * trial_residual)
                        profile_add(prof, "lm_trial_residual", trial_started)
                        if torch.isfinite(trial_loss) and float(trial_loss.detach().cpu()) < best_loss:
                            x = trial.detach()
                            best_loss = float(trial_loss.detach().cpu())
                            damping = max(damping * 0.5, 1e-8)
                            accepted = True
                            break
                        damping *= 2.0
                    if not accepted or float(torch.linalg.norm(step).detach().cpu()) < args.stageii_lm_step_tol:
                        break
                with torch.no_grad():
                    copy_started = profile_now(prof)
                    global_orient.copy_(x[:3].reshape(1, 3))
                    body_pose.copy_(x[3:66].reshape(1, SMPLX_BODY_DOF))
                    transl.copy_(x[66:69].reshape(1, 3))
                    profile_add(prof, "lm_copy_solution", copy_started)
                return best_loss

            def run_dogleg(pose_weight_multiplier: float, max_iter: int) -> float:
                trust_radius = float(args.stageii_dogleg_delta)
                max_trust_radius = float(args.stageii_dogleg_max_delta)
                eye = torch.eye(69, dtype=dtype, device=device)
                x = torch.cat([global_orient.detach().reshape(-1), body_pose.detach().reshape(-1), transl.detach().reshape(-1)])
                best_loss = math.inf
                jac_refresh = max(1, int(args.stageii_dogleg_jacobian_refresh))
                jac: Optional[torch.Tensor] = None
                jtj: Optional[torch.Tensor] = None
                force_refresh = True
                jac_age = jac_refresh

                for iter_idx in range(max_iter):
                    x_req = x.detach().requires_grad_(True)

                    def residual_for_jac(y: torch.Tensor) -> torch.Tensor:
                        return lm_residual(y, pose_weight_multiplier)

                    residual_started = profile_now(prof)
                    if args.stageii_dogleg_jacobian_mode == "fd":
                        residual = lm_residual_rows(x.detach()[None].expand(70, -1), pose_weight_multiplier)[0]
                    else:
                        residual = residual_for_jac(x_req)
                    loss_value = 0.5 * torch.sum(residual * residual)
                    best_loss = float((2.0 * loss_value).detach().cpu())
                    profile_add(prof, "dogleg_residual_current", residual_started)
                    refresh_due = force_refresh or jac is None or jtj is None
                    if args.stageii_dogleg_adaptive_refresh:
                        refresh_due = refresh_due or jac_age >= jac_refresh
                    else:
                        refresh_due = refresh_due or iter_idx % jac_refresh == 0
                    refreshed_jac = False
                    if refresh_due:
                        jac_started = profile_now(prof)
                        if args.stageii_dogleg_jacobian_mode == "fd":
                            residual, jac = finite_difference_jacobian(x.detach(), pose_weight_multiplier)
                            loss_value = 0.5 * torch.sum(residual * residual)
                            best_loss = float((2.0 * loss_value).detach().cpu())
                        else:
                            try:
                                jac = torch.autograd.functional.jacobian(
                                    residual_for_jac,
                                    x_req,
                                    vectorize=True,
                                    strategy="forward-mode",
                                )
                            except Exception:
                                jac = torch.autograd.functional.jacobian(residual_for_jac, x_req, vectorize=True)
                        jac = jac.detach()
                        jtj = jac.transpose(0, 1) @ jac
                        force_refresh = False
                        jac_age = 0
                        refreshed_jac = True
                        profile_add(prof, "dogleg_jacobian", jac_started)
                    else:
                        profile_counts["dogleg_jacobian_reused"] = profile_counts.get("dogleg_jacobian_reused", 0) + 1

                    normal_started = profile_now(prof)
                    grad = jac.transpose(0, 1) @ residual.detach()
                    grad_norm = torch.linalg.norm(grad)
                    profile_add(prof, "dogleg_normal_equations", normal_started)
                    if float(grad_norm.detach().cpu()) < args.stageii_dogleg_grad_tol:
                        break

                    step_started = profile_now(prof)
                    try:
                        p_gn = torch.linalg.solve(jtj + args.stageii_dogleg_solve_damping * eye, -grad)
                    except RuntimeError:
                        p_gn = torch.linalg.lstsq(jtj + args.stageii_dogleg_solve_damping * eye, -grad[:, None]).solution[:, 0]

                    p_gn_norm = torch.linalg.norm(p_gn)
                    if float(p_gn_norm.detach().cpu()) <= trust_radius:
                        step = p_gn
                    else:
                        bg = jtj @ grad
                        denom = torch.dot(grad, bg)
                        if float(denom.detach().cpu()) <= 1e-12:
                            p_u = -trust_radius * grad / grad_norm.clamp_min(1e-12)
                        else:
                            alpha = torch.dot(grad, grad) / denom
                            p_u = -alpha * grad
                        p_u_norm = torch.linalg.norm(p_u)
                        if float(p_u_norm.detach().cpu()) >= trust_radius:
                            step = trust_radius * p_u / p_u_norm.clamp_min(1e-12)
                        else:
                            dogleg_dir = p_gn - p_u
                            a = torch.dot(dogleg_dir, dogleg_dir)
                            b = 2.0 * torch.dot(p_u, dogleg_dir)
                            c = torch.dot(p_u, p_u) - trust_radius * trust_radius
                            disc = torch.clamp(b * b - 4.0 * a * c, min=0.0)
                            tau = (-b + torch.sqrt(disc)) / (2.0 * a).clamp_min(1e-12)
                            step = p_u + tau * dogleg_dir
                    profile_add(prof, "dogleg_linear_solve_and_step", step_started)

                    trust_started = profile_now(prof)
                    pred_reduction = -(torch.dot(grad, step) + 0.5 * torch.dot(step, jtj @ step))
                    if float(pred_reduction.detach().cpu()) <= 1e-12:
                        trust_radius *= 0.25
                        profile_add(prof, "dogleg_trust_region_update", trust_started)
                        if trust_radius < args.stageii_dogleg_min_delta:
                            break
                        continue
                    profile_add(prof, "dogleg_trust_region_update", trust_started)

                    trial_started = profile_now(prof)
                    trial = x.detach() + step
                    trial_residual = lm_residual(trial, pose_weight_multiplier)
                    trial_loss = 0.5 * torch.sum(trial_residual * trial_residual)
                    profile_add(prof, "dogleg_trial_residual", trial_started)
                    accept_started = profile_now(prof)
                    actual_reduction = loss_value - trial_loss
                    rho = actual_reduction / pred_reduction
                    rho_value = float(rho.detach().cpu()) if torch.isfinite(rho) else -math.inf
                    step_norm = float(torch.linalg.norm(step).detach().cpu())

                    if rho_value < 0.25:
                        trust_radius *= 0.25
                        force_refresh = True
                    elif rho_value > 0.75 and abs(step_norm - trust_radius) <= 1e-5 * max(1.0, trust_radius):
                        trust_radius = min(2.0 * trust_radius, max_trust_radius)

                    if rho_value > args.stageii_dogleg_eta and torch.isfinite(trial_loss):
                        x = trial.detach()
                        best_loss = float((2.0 * trial_loss).detach().cpu())
                        jac_age += 1
                        if args.stageii_dogleg_adaptive_refresh:
                            if rho_value < args.stageii_dogleg_refresh_rho:
                                force_refresh = True
                        elif rho_value < 0.5:
                            force_refresh = True
                        if step_norm < args.stageii_lm_step_tol:
                            profile_add(prof, "dogleg_accept_update", accept_started)
                            break
                    elif trust_radius < args.stageii_dogleg_min_delta:
                        profile_add(prof, "dogleg_accept_update", accept_started)
                        break
                    else:
                        force_refresh = True
                        if not refreshed_jac:
                            jac_age += 1
                    profile_add(prof, "dogleg_accept_update", accept_started)

                with torch.no_grad():
                    copy_started = profile_now(prof)
                    global_orient.copy_(x[:3].reshape(1, 3))
                    body_pose.copy_(x[3:66].reshape(1, SMPLX_BODY_DOF))
                    transl.copy_(x[66:69].reshape(1, 3))
                    profile_add(prof, "dogleg_copy_solution", copy_started)
                return best_loss

            def run_lbfgs(pose_weight_multiplier: float, max_iter: int, line_search: str) -> float:
                optimizer = torch.optim.LBFGS(
                    [global_orient, body_pose, transl],
                    lr=args.stageii_lbfgs_lr,
                    max_iter=max_iter,
                    max_eval=max_iter * 2,
                    tolerance_grad=1e-7,
                    tolerance_change=1e-9,
                    history_size=args.stageii_lbfgs_history,
                    line_search_fn=None if line_search == "none" else "strong_wolfe",
                )

                def closure():
                    optimizer.zero_grad(set_to_none=True)
                    loss_value = frame_loss(pose_weight_multiplier)
                    loss_value.backward()
                    return loss_value

                return float(optimizer.step(closure).detach().cpu())

            def run_torchmin(pose_weight_multiplier: float, max_iter: int, line_search: str) -> float:
                try:
                    from torchmin import minimize
                except ImportError as exc:
                    raise ImportError("Install pytorch-minimize to use --stageii-solver torchmin-* options.") from exc

                method_map = {
                    "torchmin-bfgs": "bfgs",
                    "torchmin-lbfgs": "l-bfgs",
                    "torchmin-dogleg": "dogleg",
                }
                method = method_map[args.stageii_solver]
                x0 = torch.cat([global_orient.detach().reshape(-1), body_pose.detach().reshape(-1), transl.detach().reshape(-1)])
                if method in {"bfgs", "l-bfgs"}:
                    options = {
                        "line_search": "strong-wolfe" if line_search == "strong_wolfe" else line_search,
                        "gtol": 1e-7,
                        "xtol": 1e-9,
                    }
                    if method == "l-bfgs":
                        options["history_size"] = args.stageii_lbfgs_history
                else:
                    options = {
                        "initial_trust_radius": args.stageii_dogleg_delta,
                        "max_trust_radius": args.stageii_dogleg_max_delta,
                        "eta": args.stageii_dogleg_eta,
                        "gtol": args.stageii_dogleg_grad_tol,
                    }

                try:
                    result = minimize(
                        lambda x: frame_loss_x(x, pose_weight_multiplier),
                        x0,
                        method=method,
                        max_iter=max_iter,
                        options=options,
                    )
                except RuntimeError as exc:
                    if method == "dogleg" and "positive-definite" in str(exc):
                        raise RuntimeError(
                            "torchmin-dogleg failed because the exact scalar-loss Hessian "
                            "is not positive definite. The built-in torchmin dogleg is a "
                            "Newton trust-region method, unlike the residual Gauss-Newton "
                            "dogleg used by this converter."
                        ) from exc
                    raise
                x = result.x.detach()
                with torch.no_grad():
                    global_orient.copy_(x[:3].reshape(1, 3))
                    body_pose.copy_(x[3:66].reshape(1, SMPLX_BODY_DOF))
                    transl.copy_(x[66:69].reshape(1, 3))
                fun = result.fun
                return float(fun.detach().cpu()) if torch.is_tensor(fun) else float(fun)

            def run_torchmin_trf(pose_weight_multiplier: float, max_nfev: int) -> float:
                try:
                    from torchmin import least_squares
                except ImportError as exc:
                    raise ImportError("Install pytorch-minimize to use --stageii-solver torchmin-trf.") from exc

                x0 = torch.cat([global_orient.detach().reshape(-1), body_pose.detach().reshape(-1), transl.detach().reshape(-1)])
                result = least_squares(
                    lambda x: lm_residual(x, pose_weight_multiplier),
                    x0,
                    method="trf",
                    tr_solver="exact",
                    max_nfev=max_nfev,
                    ftol=1e-8,
                    xtol=1e-8,
                    gtol=1e-8,
                    verbose=0,
                )
                x = result.x.detach()
                with torch.no_grad():
                    global_orient.copy_(x[:3].reshape(1, 3))
                    body_pose.copy_(x[3:66].reshape(1, SMPLX_BODY_DOF))
                    transl.copy_(x[66:69].reshape(1, 3))
                cost = result.cost
                return float((2.0 * cost).detach().cpu()) if torch.is_tensor(cost) else float(2.0 * cost)

            def run_schedule(line_search: str) -> float:
                loss_value = math.inf
                if fidx == 0:
                    for mult in (10.0, 5.0, 1.0):
                        if args.stageii_solver == "dogleg":
                            loss_value = run_dogleg(mult, args.stageii_dogleg_first_iters)
                        elif args.stageii_solver == "lm":
                            loss_value = run_lm(mult, args.stageii_lm_first_iters)
                        elif args.stageii_solver == "torchmin-trf":
                            loss_value = run_torchmin_trf(mult, args.stageii_lm_first_iters)
                        elif args.stageii_solver.startswith("torchmin-"):
                            loss_value = run_torchmin(mult, args.stageii_lbfgs_first_iters, line_search)
                        else:
                            loss_value = run_lbfgs(mult, args.stageii_lbfgs_first_iters, line_search)
                if args.stageii_solver == "dogleg" and args.stageii_dogleg_merge_passes:
                    regular_iters = max(1, args.stageii_lbfgs_passes) * max(1, args.stageii_dogleg_iters)
                    loss_value = run_dogleg(1.0, regular_iters)
                else:
                    for _ in range(max(1, args.stageii_lbfgs_passes)):
                        if args.stageii_solver == "dogleg":
                            loss_value = run_dogleg(1.0, args.stageii_dogleg_iters)
                        elif args.stageii_solver == "lm":
                            loss_value = run_lm(1.0, args.stageii_lm_iters)
                        elif args.stageii_solver == "torchmin-trf":
                            loss_value = run_torchmin_trf(1.0, args.stageii_lm_iters)
                        elif args.stageii_solver.startswith("torchmin-"):
                            loss_value = run_torchmin(1.0, args.stageii_lbfgs_iters, line_search)
                        else:
                            loss_value = run_lbfgs(1.0, args.stageii_lbfgs_iters, line_search)
                return loss_value

            root_start = global_orient.detach().clone()
            body_start = body_pose.detach().clone()
            trans_start = transl.detach().clone()

            def restore_frame_start() -> None:
                with torch.no_grad():
                    global_orient.copy_(root_start)
                    body_pose.copy_(body_start)
                    transl.copy_(trans_start)

            def current_marker_stats() -> Tuple[float, float]:
                with torch.no_grad():
                    body_eval_stats = apply_pose_mask(body_pose)
                    if args.stageii_torch_compile and hasattr(torch.compiler, "cudagraph_mark_step_begin"):
                        torch.compiler.cudagraph_mark_step_begin()
                    verts_stats = stageii_forward(betas, body_eval_stats, global_orient, transl)
                    if nn_vids is not None:
                        pred_stats = reconstruct_markers_nn(verts_stats, nn_vids, coeffs)
                    else:
                        pred_stats = reconstruct_markers(verts_stats, faces, marker_vids, tangent_vids, coeffs)
                    distances = torch.linalg.norm(pred_stats - torch.nan_to_num(obs, nan=0.0), dim=-1)
                    valid_distances = distances[mask]
                    if valid_distances.numel() == 0:
                        return 0.0, 0.0
                    return float(valid_distances.mean().detach().cpu()), float(valid_distances.max().detach().cpu())

            schedule_started = profile_now(prof)
            last_loss = run_schedule(args.stageii_lbfgs_line_search)
            profile_add(prof, "schedule_total", schedule_started)
            if args.stageii_fallback_strong_wolfe and args.stageii_solver == "lbfgs" and args.stageii_lbfgs_line_search == "none":
                mean_err, max_err = current_marker_stats()
                if (
                    (not math.isfinite(last_loss))
                    or mean_err > args.stageii_fallback_mean_marker_error
                    or max_err > args.stageii_fallback_max_marker_error
                ):
                    fallback_frames.append(fidx)
                    restore_frame_start()
                    last_loss = run_schedule("strong_wolfe")

            with torch.no_grad():
                final_forward_started = profile_now(prof)
                body_eval = apply_pose_mask(body_pose)
                if args.stageii_torch_compile and hasattr(torch.compiler, "cudagraph_mark_step_begin"):
                    torch.compiler.cudagraph_mark_step_begin()
                verts = stageii_forward(betas, body_eval, global_orient, transl)
                if nn_vids is not None:
                    pred = reconstruct_markers_nn(verts, nn_vids, coeffs)
                else:
                    pred = reconstruct_markers(verts, faces, marker_vids, tangent_vids, coeffs)
                profile_add(prof, "final_forward_markers", final_forward_started)

            cpu_started = profile_now(prof)
            root_np = global_orient.detach().cpu().numpy().astype(np.float64)
            body_np = body_eval.detach().cpu().numpy().astype(np.float64)
            trans_np = transl.detach().cpu().numpy().astype(np.float64)
            root_all[fidx:fidx + 1] = root_np
            body_all[fidx:fidx + 1] = body_np
            trans_all[fidx:fidx + 1] = trans_np
            if prev_pose_vec_seq is not None:
                prev_prev_pose_vec_seq = prev_pose_vec_seq.copy()
            prev_pose_vec_seq = np.concatenate([root_np[0], body_np[0]]).astype(np.float64)
            prev_root_seq, prev_body_seq, prev_trans_seq = root_np[0], body_np[0], trans_np[0]
            pred_np = pred.detach().cpu().numpy().astype(np.float64)
            obs_np_frame = obs.detach().cpu().numpy().astype(np.float64)
            profile_add(prof, "cpu_transfer_debug", cpu_started)
            markers_sim_debug.append(pred_np[0])
            markers_obs_debug.append(obs_np_frame[0])
            labels_obs_debug.append([label for lid, label in enumerate(latent_labels) if bool(mask[0, lid])])
            errs.append(last_loss)
            profile_add(prof, "frame_total", frame_started)
            if args.verbose and (fidx == 0 or (fidx + 1) % args.log_every == 0 or fidx + 1 == nframes):
                print(f"stage II sequential {fidx + 1:05d}/{nframes}: loss={last_loss:.6f}")

        if args.stageii_profile and args.verbose:
            profiled_frames = min(nframes, args.stageii_profile_frames if args.stageii_profile_frames > 0 else nframes)
            profile_sum = profile_times.get("frame_total", sum(profile_times.values()))
            print(f"stage II profile over {profiled_frames} frame(s):")
            for key, value in sorted(profile_times.items(), key=lambda kv: kv[1], reverse=True):
                pct = 100.0 * value / profile_sum if profile_sum > 0 else 0.0
                count = profile_counts.get(key, 0)
                mean_ms = (value / max(1, count)) * 1000.0
                print(f"  {key}: {value:.6f}s ({pct:.1f}%), count={count}, mean={mean_ms:.3f}ms")

        fullpose = fullpose_from_parts(root_all, body_all, hand_mean_np)
        stageii_data = {
            "fullpose": fullpose,
            "trans": trans_all,
            "stageii_debug_details": {
                "stageii_errs": {"loss": np.asarray(errs, dtype=np.float64)},
                "stageii_fallback_frames": np.asarray(fallback_frames, dtype=np.int64),
                "markers_sim": markers_sim_debug,
                "markers_obs": markers_obs_debug,
                "labels_obs": labels_obs_debug,
                "markers_orig": obs_np,
                "labels_orig": mocap.labels,
                "mocap_fname": args.mocap,
                "mocap_frame_rate": mocap.frame_rate,
                "mocap_time_length": nframes / mocap.frame_rate,
                "stageii_elapsed_time": time.time() - started,
                "stageii_profile_times": dict(profile_times),
                "stageii_profile_counts": dict(profile_counts),
                "cfg": vars(args),
            },
            "betas": stagei_data["betas"],
            "markers_latent": stagei_data["markers_latent"],
            "latent_labels": latent_labels,
            "marker_meta": numpy_marker_meta(marker_meta),
            "markers_latent_vids": stagei_data["markers_latent_vids"],
            "stagei_debug_details": stagei_data.get("stagei_debug_details", {}),
        }
        for key in ("torch_marker_anchor_vids", "torch_marker_coeffs", "torch_marker_tangent_vids", "torch_marker_nn_vids"):
            if key in stagei_data:
                stageii_data[key] = stagei_data[key]
        return stageii_data

    prev_root = np.zeros(3, dtype=np.float64)
    prev_body = np.zeros(SMPLX_BODY_DOF, dtype=np.float64)
    prev_trans = np.zeros(3, dtype=np.float64)
    prev_pose_vec: Optional[np.ndarray] = None
    prev_prev_pose_vec: Optional[np.ndarray] = None
    batches = list(range(0, nframes, args.batch_size))
    for batch_idx, start in enumerate(batches):
        end = min(start + args.batch_size, nframes)
        bsz = end - start
        root_init = np.tile(prev_root, (bsz, 1))
        body_init = np.tile(prev_body, (bsz, 1))
        trans_init = np.tile(prev_trans, (bsz, 1))
        root_init[:] = rigid_root_init[start:end]
        trans_init[:] = rigid_trans_init[start:end]
        if args.verbose:
            print(f"stage II batch {batch_idx + 1:04d}/{len(batches)} frames {start}:{end}: optimizing...")

        model = create_smplx_model(model_file, batch_size=bsz, gender=args.gender, num_betas=args.num_betas, device=device)
        faces = make_face_tensor(model, device)
        marker_vids = torch.as_tensor(marker_vids_np, dtype=torch.long, device=device)
        tangent_vids = torch.as_tensor(tangent_vids_np, dtype=torch.long, device=device)
        coeffs = torch.as_tensor(coeffs_np, dtype=dtype, device=device)
        nn_vids = torch.as_tensor(nn_vids_np, dtype=torch.long, device=device) if nn_vids_np is not None else None
        betas = torch.as_tensor(betas_np, dtype=dtype, device=device).expand(bsz, -1)
        obs = obs_all[start:end]
        mask = mask_all[start:end]

        global_orient = torch.nn.Parameter(torch.as_tensor(root_init, dtype=dtype, device=device))
        body_pose = torch.nn.Parameter(torch.as_tensor(body_init, dtype=dtype, device=device))
        transl = torch.nn.Parameter(torch.as_tensor(trans_init, dtype=dtype, device=device))
        optimizer = torch.optim.Adam([global_orient, body_pose, transl], lr=args.stageii_lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.stageii_epochs), eta_min=args.stageii_lr * 0.05)

        last_loss = math.inf
        for epoch in range(args.stageii_epochs):
            optimizer.zero_grad(set_to_none=True)
            if args.stageii_mosh_loss or args.stageii_freeze_toes:
                body_pose_eval = body_pose.clone()
                body_pose_eval[:, 27:33] = 0.0
            else:
                body_pose_eval = body_pose
            verts = smplx_forward(model, betas, body_pose_eval, global_orient, transl, bsz)
            if nn_vids is not None:
                pred = reconstruct_markers_nn(verts, nn_vids, coeffs)
            else:
                pred = reconstruct_markers(verts, faces, marker_vids, tangent_vids, coeffs)
            if args.stageii_mosh_loss:
                marker_count = mask.float().sum(dim=1).clamp_min(1.0)
                per_frame_wt_data = 400.0 * (46.0 / marker_count)
                residual = pred - torch.nan_to_num(obs, nan=0.0)
                sq = (residual * residual).sum(dim=-1) * mask.float()
                data_loss = torch.sum((per_frame_wt_data * per_frame_wt_data) * sq.sum(dim=1))
                pose_loss = (1.6 * 1.6) * pose_prior.sse(body_pose_eval)
                pose_seq = torch.cat([global_orient, body_pose_eval], dim=1)
                velo_loss = (2.5 * 2.5) * pose_velocity_extrap_sse(pose_seq, prev_pose_vec, prev_prev_pose_vec)
                loss = args.stageii_mosh_weight_scale * (data_loss + pose_loss + velo_loss)
            else:
                data_loss = masked_marker_loss(pred, obs, mask, sigma=args.robust_sigma, marker_weights=marker_weights)
                pose_loss = pose_prior(body_pose_eval)
                smooth_loss = torch.tensor(0.0, dtype=dtype, device=device)
                if bsz > 1 and args.stageii_velocity_weight > 0:
                    smooth_loss = torch.mean((body_pose_eval[1:] - body_pose_eval[:-1]) ** 2) + torch.mean((global_orient[1:] - global_orient[:-1]) ** 2)
                loss = (
                    args.stageii_data_weight * data_loss
                    + args.stageii_pose_prior_weight * pose_loss
                    + args.stageii_pose_l2_weight * torch.mean(body_pose_eval * body_pose_eval)
                    + args.stageii_velocity_weight * smooth_loss
                )
            loss.backward()
            optimizer.step()
            scheduler.step()
            last_loss = float(loss.detach().cpu())

        with torch.no_grad():
            if args.stageii_mosh_loss or args.stageii_freeze_toes:
                body_pose_eval = body_pose.clone()
                body_pose_eval[:, 27:33] = 0.0
            else:
                body_pose_eval = body_pose
            verts = smplx_forward(model, betas, body_pose_eval, global_orient, transl, bsz)
            if nn_vids is not None:
                pred = reconstruct_markers_nn(verts, nn_vids, coeffs)
            else:
                pred = reconstruct_markers(verts, faces, marker_vids, tangent_vids, coeffs)
        root_np = global_orient.detach().cpu().numpy().astype(np.float64)
        body_np = body_pose_eval.detach().cpu().numpy().astype(np.float64)
        trans_np = transl.detach().cpu().numpy().astype(np.float64)
        root_all[start:end] = root_np
        body_all[start:end] = body_np
        trans_all[start:end] = trans_np
        if bsz >= 2:
            prev_prev_pose_vec = np.concatenate([root_np[-2], body_np[-2]]).astype(np.float64)
        elif prev_pose_vec is not None:
            prev_prev_pose_vec = prev_pose_vec.copy()
        prev_pose_vec = np.concatenate([root_np[-1], body_np[-1]]).astype(np.float64)
        prev_root, prev_body, prev_trans = root_np[-1], body_np[-1], trans_np[-1]
        pred_np = pred.detach().cpu().numpy().astype(np.float64)
        obs_batch_np = obs.detach().cpu().numpy().astype(np.float64)
        markers_sim_debug.extend([x for x in pred_np])
        markers_obs_debug.extend([x for x in obs_batch_np])
        labels_obs_debug.extend([[label for lid, label in enumerate(latent_labels) if bool(mask[j, lid])] for j in range(bsz)])
        errs.append(last_loss)
        if args.verbose:
            print(f"stage II batch {batch_idx + 1:04d}/{len(batches)} frames {start}:{end}: loss={last_loss:.6f}")

    fullpose = fullpose_from_parts(root_all, body_all, hand_mean_np)
    stageii_data = {
        "fullpose": fullpose,
        "trans": trans_all,
        "stageii_debug_details": {
            "stageii_errs": {"loss": np.asarray(errs, dtype=np.float64)},
            "markers_sim": markers_sim_debug,
            "markers_obs": markers_obs_debug,
            "labels_obs": labels_obs_debug,
            "markers_orig": mocap.markers,
            "labels_orig": mocap.labels,
            "mocap_fname": str(Path(args.mocap).expanduser().resolve()),
            "mocap_frame_rate": float(mocap.frame_rate),
            "mocap_time_length": float(mocap.time_length),
            "stageii_elapsed_time": time.time() - started,
        },
        "betas": stagei_data["betas"],
        "markers_latent": stagei_data["markers_latent"],
        "latent_labels": stagei_data["latent_labels"],
        "marker_meta": stagei_data["marker_meta"],
        "markers_latent_vids": stagei_data["markers_latent_vids"],
        "torch_marker_anchor_vids": stagei_data.get("torch_marker_anchor_vids", stagei_data["markers_latent_vids"]),
        "torch_marker_coeffs": stagei_data["torch_marker_coeffs"],
        "torch_marker_tangent_vids": stagei_data["torch_marker_tangent_vids"],
        "stagei_debug_details": stagei_data["stagei_debug_details"],
    }
    return stageii_data
