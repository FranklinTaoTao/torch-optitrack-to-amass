from __future__ import annotations

from .helpers import *

def optimize_stagei(args, model_file: Path, marker_meta: Dict, mocap: MocapData, device: torch.device) -> Dict:
    dtype = torch.float32
    stagei_anneals = parse_anneal_factors(args.stagei_mosh_anneal_factors)
    use_nn_latent_basis = args.stagei_optimize_latent_markers and args.stagei_latent_basis == "nn"
    latent_labels = list(marker_meta["marker_vids"].keys())
    obs_all, mask_all, obs_np = make_observation_tensors(mocap, latent_labels, device, dtype)
    stagei_ids = select_stagei_frames(
        mask_all.detach().cpu().numpy(),
        args.stagei_num_frames,
        args.stagei_least_avail_markers,
        args.seed,
        args.stagei_frame_ids,
    )
    k = len(stagei_ids)

    model = create_smplx_model(model_file, batch_size=k, gender=args.gender, num_betas=args.num_betas, device=device)
    faces = make_face_tensor(model, device)
    marker_vids_np = np.asarray([marker_meta["marker_vids"][label] for label in latent_labels], dtype=np.int64)
    tangent_vids_np = build_neighbor_anchors(np.asarray(model.faces), marker_vids_np)
    marker_vids = torch.as_tensor(marker_vids_np, dtype=torch.long, device=device)
    tangent_vids = torch.as_tensor(tangent_vids_np, dtype=torch.long, device=device)
    init_coeffs_np = np.zeros((len(latent_labels), 3), dtype=np.float32)
    init_coeffs_np[:, 2] = marker_distances_from_layout(marker_meta, latent_labels)
    init_coeffs = torch.as_tensor(init_coeffs_np, dtype=dtype, device=device)

    betas = torch.nn.Parameter(torch.zeros((1, args.num_betas), dtype=dtype, device=device))
    coeffs = torch.nn.Parameter(init_coeffs.clone())
    can_model = create_smplx_model(model_file, batch_size=1, gender=args.gender, num_betas=args.num_betas, device=device)
    can_faces = make_face_tensor(can_model, device)
    with torch.no_grad():
        init_can_verts = smplx_forward(
            can_model,
            betas=torch.zeros((1, args.num_betas), dtype=dtype, device=device),
            body_pose=torch.zeros((1, SMPLX_BODY_DOF), dtype=dtype, device=device),
            global_orient=torch.zeros((1, 3), dtype=dtype, device=device),
            transl=torch.zeros((1, 3), dtype=dtype, device=device),
            batch_size=1,
        )
        init_latent = reconstruct_markers(init_can_verts, can_faces, marker_vids, tangent_vids, init_coeffs)[0]
        init_nn_coeffs, init_nn_vids = marker_coeffs_from_latent_nn(
            init_can_verts,
            init_latent,
            num_neighbors=args.stagei_nn_neighbors,
            exclude_eyeballs=not args.stagei_nn_include_eyeballs,
        )
    markers_latent = torch.nn.Parameter(init_latent.clone())
    root_init_np, trans_init_np = kabsch_init_batch(init_latent.detach().cpu().numpy(), obs_np[stagei_ids])
    body_pose = torch.nn.Parameter(torch.zeros((k, SMPLX_BODY_DOF), dtype=dtype, device=device))
    global_orient = torch.nn.Parameter(torch.as_tensor(root_init_np, dtype=dtype, device=device))
    transl = torch.nn.Parameter(torch.as_tensor(trans_init_np, dtype=dtype, device=device))

    obs = obs_all[stagei_ids]
    mask = mask_all[stagei_ids]
    marker_weights = marker_weight_tensor(latent_labels, args.arm_marker_weight, device, dtype)
    pose_prior = BodyPosePrior(Path(args.pose_body_prior).expanduser().resolve() if args.pose_body_prior else None, device, dtype)
    bodyfit_epochs = min(max(0, int(args.stagei_bodyfit_epochs)), args.stagei_epochs)
    coeffs_active = bodyfit_epochs == 0
    if args.stagei_optimize_latent_markers:
        coeffs.requires_grad_(False)
        markers_latent.requires_grad_(coeffs_active)
        opt_params = [betas, body_pose, global_orient, transl] + ([markers_latent] if coeffs_active else [])
    else:
        coeffs.requires_grad_(coeffs_active)
        markers_latent.requires_grad_(False)
        opt_params = [betas, body_pose, global_orient, transl] + ([coeffs] if coeffs_active else [])
    optimizer = torch.optim.Adam(opt_params, lr=args.stagei_lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, math.ceil(args.stagei_epochs / 4) if args.stagei_mosh_anneal else (args.stagei_epochs if coeffs_active else bodyfit_epochs)),
        eta_min=args.stagei_lr * 0.05,
    )

    last_loss = math.inf
    last_mosh_phase = 0
    started = time.time()
    for epoch in range(args.stagei_epochs):
        if (not coeffs_active) and epoch == bodyfit_epochs:
            coeffs_active = True
            if args.stagei_optimize_latent_markers:
                markers_latent.requires_grad_(True)
                optimizer = torch.optim.Adam([betas, markers_latent, body_pose, global_orient, transl], lr=args.stagei_lr)
            else:
                coeffs.requires_grad_(True)
                optimizer = torch.optim.Adam([betas, coeffs, body_pose, global_orient, transl], lr=args.stagei_lr)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max(1, args.stagei_epochs - epoch),
                eta_min=args.stagei_lr * 0.05,
            )
        if args.stagei_mosh_anneal:
            mosh_phase = min(len(stagei_anneals) - 1, int(epoch * len(stagei_anneals) / max(1, args.stagei_epochs)))
            if mosh_phase != last_mosh_phase:
                if args.stagei_optimize_latent_markers:
                    params = [betas, body_pose, global_orient, transl] + ([markers_latent] if coeffs_active else [])
                else:
                    params = [betas, body_pose, global_orient, transl] + ([coeffs] if coeffs_active else [])
                optimizer = torch.optim.Adam(params, lr=args.stagei_lr)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=max(1, math.ceil(args.stagei_epochs / 4)),
                    eta_min=args.stagei_lr * 0.05,
                )
                last_mosh_phase = mosh_phase
        optimizer.zero_grad(set_to_none=True)
        body_pose_eval = mask_toe_pose(body_pose, args.stagei_freeze_toes)
        verts = smplx_forward(
            model,
            betas=betas.expand(k, -1),
            body_pose=body_pose_eval,
            global_orient=global_orient,
            transl=transl,
            batch_size=k,
        )
        if use_nn_latent_basis:
            can_verts = smplx_forward(
                can_model,
                betas=betas,
                body_pose=torch.zeros((1, SMPLX_BODY_DOF), dtype=dtype, device=device),
                global_orient=torch.zeros((1, 3), dtype=dtype, device=device),
                transl=torch.zeros((1, 3), dtype=dtype, device=device),
                batch_size=1,
            )
            cur_coeffs, cur_nn_vids = marker_coeffs_from_latent_nn(
                can_verts,
                markers_latent,
                num_neighbors=args.stagei_nn_neighbors,
                exclude_eyeballs=not args.stagei_nn_include_eyeballs,
            )
            init_target = reconstruct_markers_nn(can_verts, init_nn_vids, init_nn_coeffs)[0]
            pred = reconstruct_markers_nn(verts, cur_nn_vids, cur_coeffs)
        elif args.stagei_optimize_latent_markers:
            can_verts = smplx_forward(
                can_model,
                betas=betas,
                body_pose=torch.zeros((1, SMPLX_BODY_DOF), dtype=dtype, device=device),
                global_orient=torch.zeros((1, 3), dtype=dtype, device=device),
                transl=torch.zeros((1, 3), dtype=dtype, device=device),
                batch_size=1,
            )
            cur_coeffs = marker_coeffs_from_latent(can_verts, can_faces, marker_vids, tangent_vids, markers_latent)
            init_target = reconstruct_markers(can_verts, can_faces, marker_vids, tangent_vids, init_coeffs)[0]
            pred = reconstruct_markers(verts, faces, marker_vids, tangent_vids, cur_coeffs)
        else:
            cur_coeffs = coeffs
            init_target = init_coeffs
            pred = reconstruct_markers(verts, faces, marker_vids, tangent_vids, cur_coeffs)
        robust_data_loss = masked_marker_loss(pred, obs, mask, sigma=args.robust_sigma, marker_weights=marker_weights)
        data_mse = masked_marker_mse(pred, obs, mask, marker_weights=marker_weights)
        data_sse = masked_marker_sse(pred, obs, mask, marker_weights=marker_weights)
        if args.stagei_optimize_latent_markers:
            coeff_init_loss = torch.mean((markers_latent - init_target) ** 2)
            coeff_init_sse = torch.sum((markers_latent - init_target) ** 2)
            normal_excess = F.relu(torch.abs(cur_coeffs[:, 2] - init_coeffs[:, 2]) - args.max_marker_offset)
            signed_surface_loss = torch.mean((cur_coeffs[:, 2] - init_coeffs[:, 2]) ** 2)
            if use_nn_latent_basis:
                surface_residual = latent_surface_residual(
                    can_verts, can_faces, markers_latent, init_coeffs[:, 2], args.stagei_surface_distance
                )
                signed_surface_sse = torch.sum(surface_residual * surface_residual)
            else:
                signed_surface_sse = torch.sum((cur_coeffs[:, 2] - init_coeffs[:, 2]) ** 2)
        else:
            coeff_init_loss = torch.mean((cur_coeffs - init_coeffs) ** 2)
            coeff_init_sse = torch.sum((cur_coeffs - init_coeffs) ** 2)
            normal_excess = F.relu(torch.abs(cur_coeffs[:, 2]) - args.max_marker_offset)
            signed_surface_loss = torch.mean((cur_coeffs[:, 2] - init_coeffs[:, 2]) ** 2)
            signed_surface_sse = torch.sum((cur_coeffs[:, 2] - init_coeffs[:, 2]) ** 2)
        tangent_loss = torch.mean(cur_coeffs[:, :2] ** 2)
        surf_loss = torch.mean(normal_excess * normal_excess) + 0.1 * tangent_loss
        pose_loss = pose_prior(body_pose_eval)
        pose_sse = pose_prior.sse(body_pose_eval)
        beta_loss = torch.mean(betas * betas)
        if args.stagei_mosh_anneal:
            phase = min(len(stagei_anneals) - 1, int(epoch * len(stagei_anneals) / max(1, args.stagei_epochs)))
            anneal = stagei_anneals[phase]
            scale = args.stagei_mosh_weight_scale
            wt_data = (75.0 / anneal) * (46.0 / max(1, len(latent_labels))) * scale * args.stagei_mosh_data_mult
            wt_init = 300.0 * anneal * scale * args.stagei_mosh_init_mult
            wt_surf = 10000.0 * scale * args.stagei_mosh_surf_mult
            wt_pose = 3.0 * anneal * scale * args.stagei_mosh_pose_mult
            wt_beta = 10.0 * anneal * scale * args.stagei_mosh_beta_mult
            surf_mosh_loss = signed_surface_sse
            loss = (
                (wt_data * wt_data) * data_sse
                + (wt_init * wt_init) * coeff_init_sse
                + (wt_surf * wt_surf) * surf_mosh_loss
                + (wt_pose * wt_pose) * pose_sse
                + (wt_beta * wt_beta) * torch.sum(betas * betas)
            )
        else:
            data_loss = robust_data_loss
            loss = (
                args.stagei_data_weight * data_loss
                + args.stagei_marker_init_weight * coeff_init_loss
                + args.stagei_surface_weight * surf_loss
                + args.stagei_pose_prior_weight * pose_loss
                + args.stagei_beta_prior_weight * beta_loss
            )
        loss.backward()
        optimizer.step()
        scheduler.step()
        last_loss = float(loss.detach().cpu())
        if args.verbose and (epoch == 0 or (epoch + 1) % args.log_every == 0 or epoch + 1 == args.stagei_epochs):
            print(
                f"stage I {epoch + 1:04d}/{args.stagei_epochs}: "
                f"loss={last_loss:.6f} data={float(data_mse.detach().cpu() if args.stagei_mosh_anneal else robust_data_loss.detach().cpu()):.6f} "
                f"marker_offsets={'on' if coeffs_active else 'fixed'}"
            )

    if args.stagei_dogleg_refine:
        dogleg_factors = parse_anneal_factors(args.stagei_dogleg_anneal_factors)
        optimize_pose_in_dogleg = args.stagei_dogleg_full_pose
        dogleg_nn_vids: Optional[torch.Tensor] = None
        dogleg_surface_vids: Optional[torch.Tensor] = None
        if use_nn_latent_basis:
            with torch.no_grad():
                dogleg_can_verts = smplx_forward(
                    can_model,
                    betas=betas.detach(),
                    body_pose=torch.zeros((1, SMPLX_BODY_DOF), dtype=dtype, device=device),
                    global_orient=torch.zeros((1, 3), dtype=dtype, device=device),
                    transl=torch.zeros((1, 3), dtype=dtype, device=device),
                    batch_size=1,
                )
                _, dogleg_nn_vids = marker_coeffs_from_latent_nn(
                    dogleg_can_verts,
                    markers_latent.detach(),
                    num_neighbors=args.stagei_nn_neighbors,
                    exclude_eyeballs=not args.stagei_nn_include_eyeballs,
                )
                dogleg_surface_vids = torch.argmin(
                    torch.cdist(markers_latent.detach()[None], dogleg_can_verts)[0],
                    dim=1,
                )

        def pack_stagei_state() -> torch.Tensor:
            parts = [betas.detach().reshape(-1)]
            parts.append((markers_latent if args.stagei_optimize_latent_markers else coeffs).detach().reshape(-1))
            if optimize_pose_in_dogleg:
                parts.extend([body_pose.detach().reshape(-1), global_orient.detach().reshape(-1), transl.detach().reshape(-1)])
            return torch.cat(parts)

        def unpack_stagei_state(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            idx = 0
            betas_l = x[idx:idx + args.num_betas].reshape(1, args.num_betas)
            idx += args.num_betas
            marker_var = x[idx:idx + len(latent_labels) * 3].reshape(len(latent_labels), 3)
            idx += len(latent_labels) * 3
            if optimize_pose_in_dogleg:
                body_l = x[idx:idx + k * SMPLX_BODY_DOF].reshape(k, SMPLX_BODY_DOF)
                idx += k * SMPLX_BODY_DOF
                root_l = x[idx:idx + k * 3].reshape(k, 3)
                idx += k * 3
                trans_l = x[idx:idx + k * 3].reshape(k, 3)
            else:
                body_l = body_pose.detach()
                root_l = global_orient.detach()
                trans_l = transl.detach()
            return betas_l, marker_var, body_l, root_l, trans_l

        def stagei_residual_for_state(x: torch.Tensor, anneal: float) -> torch.Tensor:
            betas_l, marker_var_l, body_l_raw, root_l, trans_l = unpack_stagei_state(x)
            body_l = mask_toe_pose(body_l_raw, args.stagei_freeze_toes)
            verts_l = smplx_forward(
                model,
                betas=betas_l.expand(k, -1),
                body_pose=body_l,
                global_orient=root_l,
                transl=trans_l,
                batch_size=k,
            )
            if use_nn_latent_basis:
                can_verts_l = smplx_forward(
                    can_model,
                    betas=betas_l,
                    body_pose=torch.zeros((1, SMPLX_BODY_DOF), dtype=dtype, device=device),
                    global_orient=torch.zeros((1, 3), dtype=dtype, device=device),
                    transl=torch.zeros((1, 3), dtype=dtype, device=device),
                    batch_size=1,
                )
                cur_nn_vids_l = dogleg_nn_vids
                cur_coeffs_l = marker_coeffs_from_latent_fixed_nn(can_verts_l, marker_var_l, cur_nn_vids_l)
                init_target_l = reconstruct_markers_nn(can_verts_l, init_nn_vids, init_nn_coeffs)[0]
                pred_l = reconstruct_markers_nn(verts_l, cur_nn_vids_l, cur_coeffs_l)
            elif args.stagei_optimize_latent_markers:
                can_verts_l = smplx_forward(
                    can_model,
                    betas=betas_l,
                    body_pose=torch.zeros((1, SMPLX_BODY_DOF), dtype=dtype, device=device),
                    global_orient=torch.zeros((1, 3), dtype=dtype, device=device),
                    transl=torch.zeros((1, 3), dtype=dtype, device=device),
                    batch_size=1,
                )
                cur_coeffs_l = marker_coeffs_from_latent(can_verts_l, can_faces, marker_vids, tangent_vids, marker_var_l)
                init_target_l = reconstruct_markers(can_verts_l, can_faces, marker_vids, tangent_vids, init_coeffs)[0]
                pred_l = reconstruct_markers(verts_l, faces, marker_vids, tangent_vids, cur_coeffs_l)
            else:
                cur_coeffs_l = marker_var_l
                can_verts_l = None
                init_target_l = init_coeffs
                pred_l = reconstruct_markers(verts_l, faces, marker_vids, tangent_vids, cur_coeffs_l)

            if args.stagei_mosh_anneal:
                scale = args.stagei_mosh_weight_scale
                wt_data = (75.0 / anneal) * (46.0 / max(1, len(latent_labels))) * scale * args.stagei_mosh_data_mult
                wt_init = 300.0 * anneal * scale * args.stagei_mosh_init_mult
                wt_surf = 10000.0 * scale * args.stagei_mosh_surf_mult
                wt_pose = 3.0 * anneal * scale * args.stagei_mosh_pose_mult
                wt_beta = 10.0 * anneal * scale * args.stagei_mosh_beta_mult
                data_weight = torch.sqrt(marker_weights).reshape(1, -1, 1)
                data_res = wt_data * data_weight * (pred_l - torch.nan_to_num(obs, nan=0.0))
                data_res = data_res[mask[:, :, None].expand_as(data_res)]
                if args.stagei_optimize_latent_markers:
                    init_res = wt_init * (marker_var_l - init_target_l).reshape(-1)
                    if use_nn_latent_basis:
                        if args.stagei_surface_distance == "vertex":
                            surf_residual = latent_surface_distance_fixed_vertex_normal(
                                can_verts_l, can_faces, marker_var_l, init_coeffs[:, 2], dogleg_surface_vids
                            )
                        else:
                            surf_residual = latent_surface_residual(
                                can_verts_l, can_faces, marker_var_l, init_coeffs[:, 2], args.stagei_surface_distance
                            )
                        surf_res = wt_surf * surf_residual.reshape(-1)
                    else:
                        surf_res = wt_surf * (cur_coeffs_l[:, 2] - init_coeffs[:, 2]).reshape(-1)
                else:
                    init_res = wt_init * (cur_coeffs_l - init_coeffs).reshape(-1)
                    surf_res = wt_surf * (cur_coeffs_l[:, 2] - init_coeffs[:, 2]).reshape(-1)
                pose_res = wt_pose * pose_prior.residual(body_l)
                beta_res = wt_beta * betas_l.reshape(-1)
                return torch.cat([data_res.reshape(-1), init_res, surf_res, pose_res, beta_res], dim=0)

            data_weight = torch.sqrt(marker_weights).reshape(1, -1, 1)
            data_res = math.sqrt(args.stagei_data_weight) * data_weight * (pred_l - torch.nan_to_num(obs, nan=0.0))
            data_res = data_res[mask[:, :, None].expand_as(data_res)]
            if args.stagei_optimize_latent_markers:
                init_res = math.sqrt(args.stagei_marker_init_weight) * (marker_var_l - init_target_l).reshape(-1)
                normal_excess_l = F.relu(torch.abs(cur_coeffs_l[:, 2] - init_coeffs[:, 2]) - args.max_marker_offset)
            else:
                init_res = math.sqrt(args.stagei_marker_init_weight) * (cur_coeffs_l - init_coeffs).reshape(-1)
                normal_excess_l = F.relu(torch.abs(cur_coeffs_l[:, 2]) - args.max_marker_offset)
            surf_res = math.sqrt(args.stagei_surface_weight) * torch.cat(
                [normal_excess_l.reshape(-1), math.sqrt(0.1) * cur_coeffs_l[:, :2].reshape(-1)],
                dim=0,
            )
            pose_res = math.sqrt(args.stagei_pose_prior_weight) * pose_prior.residual(body_l)
            beta_res = math.sqrt(args.stagei_beta_prior_weight) * betas_l.reshape(-1)
            return torch.cat([data_res.reshape(-1), init_res, surf_res, pose_res, beta_res], dim=0)

        def run_stagei_dogleg_phase(x: torch.Tensor, anneal: float) -> Tuple[torch.Tensor, float]:
            trust_radius = float(args.stagei_dogleg_delta)
            max_trust_radius = float(args.stagei_dogleg_max_delta)
            nvars = int(x.numel())
            eye = torch.eye(nvars, dtype=dtype, device=device)
            best_loss = math.inf

            def jacobian_loop_jvp(residual_for_jac, x_base: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
                residual_base = residual_for_jac(x_base).detach()
                cols = []
                for col_idx in range(nvars):
                    basis = torch.zeros_like(x_base)
                    basis[col_idx] = 1.0
                    _, j_col = torch.autograd.functional.jvp(
                        residual_for_jac,
                        (x_base,),
                        (basis,),
                        create_graph=False,
                        strict=False,
                    )
                    cols.append(j_col.detach())
                return residual_base, torch.stack(cols, dim=1)

            for _ in range(max(1, args.stagei_dogleg_iters)):
                x_req = x.detach().requires_grad_(True)

                def residual_for_jac(y: torch.Tensor) -> torch.Tensor:
                    return stagei_residual_for_state(y, anneal)

                if args.stagei_dogleg_jacobian == "loop-jvp":
                    residual, jac = jacobian_loop_jvp(residual_for_jac, x_req)
                    loss_value = 0.5 * torch.sum(residual * residual)
                    best_loss = float((2.0 * loss_value).detach().cpu())
                else:
                    residual = residual_for_jac(x_req)
                    loss_value = 0.5 * torch.sum(residual * residual)
                    best_loss = float((2.0 * loss_value).detach().cpu())
                    try:
                        jac = torch.autograd.functional.jacobian(
                            residual_for_jac,
                            x_req,
                            vectorize=True,
                            strategy="forward-mode",
                        )
                    except Exception:
                        jac = torch.autograd.functional.jacobian(residual_for_jac, x_req, vectorize=True)
                jtj = jac.transpose(0, 1) @ jac
                grad = jac.transpose(0, 1) @ residual.detach()
                grad_norm = torch.linalg.norm(grad)
                if float(grad_norm.detach().cpu()) < args.stagei_dogleg_grad_tol:
                    break
                try:
                    p_gn = torch.linalg.solve(jtj + args.stagei_dogleg_solve_damping * eye, -grad)
                except RuntimeError:
                    p_gn = torch.linalg.lstsq(jtj + args.stagei_dogleg_solve_damping * eye, -grad[:, None]).solution[:, 0]
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
                        d = p_gn - p_u
                        a = torch.dot(d, d)
                        b = 2.0 * torch.dot(p_u, d)
                        c = torch.dot(p_u, p_u) - trust_radius * trust_radius
                        tau = (-b + torch.sqrt(torch.clamp(b * b - 4.0 * a * c, min=0.0))) / (2.0 * a).clamp_min(1e-12)
                        step = p_u + tau * d
                pred_reduction = -(torch.dot(grad, step) + 0.5 * torch.dot(step, jtj @ step))
                if float(pred_reduction.detach().cpu()) <= 1e-12:
                    trust_radius *= 0.25
                    continue
                trial = x.detach() + step
                trial_residual = stagei_residual_for_state(trial, anneal)
                trial_loss = 0.5 * torch.sum(trial_residual * trial_residual)
                rho = (loss_value - trial_loss) / pred_reduction
                rho_value = float(rho.detach().cpu()) if torch.isfinite(rho) else -math.inf
                step_norm = float(torch.linalg.norm(step).detach().cpu())
                if rho_value < 0.25:
                    trust_radius *= 0.25
                elif rho_value > 0.75 and abs(step_norm - trust_radius) <= 1e-5 * max(1.0, trust_radius):
                    trust_radius = min(2.0 * trust_radius, max_trust_radius)
                if rho_value > args.stagei_dogleg_eta and torch.isfinite(trial_loss):
                    x = trial.detach()
                    best_loss = float((2.0 * trial_loss).detach().cpu())
                if step_norm < args.stagei_dogleg_step_tol or trust_radius < args.stagei_dogleg_min_delta:
                    break
            return x.detach(), best_loss

        dogleg_started = time.time()
        dogleg_x = pack_stagei_state()
        for phase_idx, anneal in enumerate(dogleg_factors):
            dogleg_x, last_loss = run_stagei_dogleg_phase(dogleg_x, anneal)
            if args.verbose:
                print(
                    f"stage I dogleg phase {phase_idx + 1:02d}/{len(dogleg_factors)}: "
                    f"anneal={anneal:.3f} loss={last_loss:.6f}"
                )
        with torch.no_grad():
            betas_l, marker_var_l, body_l, root_l, trans_l = unpack_stagei_state(dogleg_x)
            betas.copy_(betas_l)
            if args.stagei_optimize_latent_markers:
                markers_latent.copy_(marker_var_l)
            else:
                coeffs.copy_(marker_var_l)
            if optimize_pose_in_dogleg:
                body_pose.copy_(body_l)
                global_orient.copy_(root_l)
                transl.copy_(trans_l)
        if args.verbose:
            print(f"stage I dogleg elapsed: {time.time() - dogleg_started:.3f}s")

    if args.stagei_lbfgs_refine:
        if args.stagei_optimize_latent_markers:
            markers_latent.requires_grad_(True)
            lbfgs_params = [betas, markers_latent, body_pose, global_orient, transl]
        else:
            coeffs.requires_grad_(True)
            lbfgs_params = [betas, coeffs, body_pose, global_orient, transl]
        n_phases = max(1, int(args.stagei_lbfgs_phases))

        def lbfgs_stagei_loss(phase_idx: int) -> torch.Tensor:
            body_pose_eval_l = mask_toe_pose(body_pose, args.stagei_freeze_toes)
            verts = smplx_forward(
                model,
                betas=betas.expand(k, -1),
                body_pose=body_pose_eval_l,
                global_orient=global_orient,
                transl=transl,
                batch_size=k,
            )
            if use_nn_latent_basis:
                can_verts_l = smplx_forward(
                    can_model,
                    betas=betas,
                    body_pose=torch.zeros((1, SMPLX_BODY_DOF), dtype=dtype, device=device),
                    global_orient=torch.zeros((1, 3), dtype=dtype, device=device),
                    transl=torch.zeros((1, 3), dtype=dtype, device=device),
                    batch_size=1,
                )
                cur_coeffs_l, cur_nn_vids_l = marker_coeffs_from_latent_nn(
                    can_verts_l,
                    markers_latent,
                    num_neighbors=args.stagei_nn_neighbors,
                    exclude_eyeballs=not args.stagei_nn_include_eyeballs,
                )
                init_target_l = reconstruct_markers_nn(can_verts_l, init_nn_vids, init_nn_coeffs)[0]
                pred_l = reconstruct_markers_nn(verts, cur_nn_vids_l, cur_coeffs_l)
            elif args.stagei_optimize_latent_markers:
                can_verts_l = smplx_forward(
                    can_model,
                    betas=betas,
                    body_pose=torch.zeros((1, SMPLX_BODY_DOF), dtype=dtype, device=device),
                    global_orient=torch.zeros((1, 3), dtype=dtype, device=device),
                    transl=torch.zeros((1, 3), dtype=dtype, device=device),
                    batch_size=1,
                )
                cur_coeffs_l = marker_coeffs_from_latent(can_verts_l, can_faces, marker_vids, tangent_vids, markers_latent)
                init_target_l = reconstruct_markers(can_verts_l, can_faces, marker_vids, tangent_vids, init_coeffs)[0]
                pred_l = reconstruct_markers(verts, faces, marker_vids, tangent_vids, cur_coeffs_l)
            else:
                cur_coeffs_l = coeffs
                init_target_l = init_coeffs
                pred_l = reconstruct_markers(verts, faces, marker_vids, tangent_vids, cur_coeffs_l)

            if args.stagei_mosh_anneal:
                phase = min(len(stagei_anneals) - 1, phase_idx)
                anneal = stagei_anneals[phase]
                scale = args.stagei_mosh_weight_scale
                wt_data = (75.0 / anneal) * (46.0 / max(1, len(latent_labels))) * scale * args.stagei_mosh_data_mult
                wt_init = 300.0 * anneal * scale * args.stagei_mosh_init_mult
                wt_surf = 10000.0 * scale * args.stagei_mosh_surf_mult
                wt_pose = 3.0 * anneal * scale * args.stagei_mosh_pose_mult
                wt_beta = 10.0 * anneal * scale * args.stagei_mosh_beta_mult
                data_term = masked_marker_sse(pred_l, obs, mask, marker_weights=marker_weights)
                if args.stagei_optimize_latent_markers:
                    init_term = torch.sum((markers_latent - init_target_l) ** 2)
                    if use_nn_latent_basis:
                        surface_residual_l = latent_surface_residual(
                            can_verts_l, can_faces, markers_latent, init_coeffs[:, 2], args.stagei_surface_distance
                        )
                        surf_term = torch.sum(surface_residual_l * surface_residual_l)
                    else:
                        surf_term = torch.sum((cur_coeffs_l[:, 2] - init_coeffs[:, 2]) ** 2)
                else:
                    init_term = torch.sum((cur_coeffs_l - init_coeffs) ** 2)
                    surf_term = torch.sum((cur_coeffs_l[:, 2] - init_coeffs[:, 2]) ** 2)
                return (
                    (wt_data * wt_data) * data_term
                    + (wt_init * wt_init) * init_term
                    + (wt_surf * wt_surf) * surf_term
                    + (wt_pose * wt_pose) * pose_prior.sse(body_pose_eval_l)
                    + (wt_beta * wt_beta) * torch.sum(betas * betas)
                )

            data_term = masked_marker_loss(pred_l, obs, mask, sigma=args.robust_sigma, marker_weights=marker_weights)
            if args.stagei_optimize_latent_markers:
                init_term = torch.mean((markers_latent - init_target_l) ** 2)
                normal_excess_l = F.relu(torch.abs(cur_coeffs_l[:, 2] - init_coeffs[:, 2]) - args.max_marker_offset)
            else:
                init_term = torch.mean((cur_coeffs_l - init_coeffs) ** 2)
                normal_excess_l = F.relu(torch.abs(cur_coeffs_l[:, 2]) - args.max_marker_offset)
            surf_term = torch.mean(normal_excess_l * normal_excess_l) + 0.1 * torch.mean(cur_coeffs_l[:, :2] ** 2)
            return (
                args.stagei_data_weight * data_term
                + args.stagei_marker_init_weight * init_term
                + args.stagei_surface_weight * surf_term
                + args.stagei_pose_prior_weight * pose_prior(body_pose_eval_l)
                + args.stagei_beta_prior_weight * torch.mean(betas * betas)
            )

        for phase_idx in range(n_phases):
            optimizer = torch.optim.LBFGS(
                lbfgs_params,
                lr=args.stagei_lbfgs_lr,
                max_iter=args.stagei_lbfgs_iters,
                max_eval=args.stagei_lbfgs_iters * 2,
                tolerance_grad=1e-7,
                tolerance_change=1e-9,
                history_size=args.stagei_lbfgs_history,
                line_search_fn="strong_wolfe",
            )

            def closure():
                optimizer.zero_grad(set_to_none=True)
                loss_value = lbfgs_stagei_loss(phase_idx)
                loss_value.backward()
                return loss_value

            last_loss = float(optimizer.step(closure).detach().cpu())
            if args.verbose:
                print(f"stage I LBFGS phase {phase_idx + 1:02d}/{n_phases}: loss={last_loss:.6f}")

    with torch.no_grad():
        body_pose_eval = mask_toe_pose(body_pose, args.stagei_freeze_toes)
        verts = smplx_forward(
            model,
            betas=betas.expand(k, -1),
            body_pose=body_pose_eval,
            global_orient=global_orient,
            transl=transl,
            batch_size=k,
        )
        if use_nn_latent_basis:
            can_verts = smplx_forward(
                can_model,
                betas=betas,
                body_pose=torch.zeros((1, SMPLX_BODY_DOF), dtype=dtype, device=device),
                global_orient=torch.zeros((1, 3), dtype=dtype, device=device),
                transl=torch.zeros((1, 3), dtype=dtype, device=device),
                batch_size=1,
            )
            final_coeffs, final_nn_vids = marker_coeffs_from_latent_nn(
                can_verts,
                markers_latent,
                num_neighbors=args.stagei_nn_neighbors,
                exclude_eyeballs=not args.stagei_nn_include_eyeballs,
            )
            pred = reconstruct_markers_nn(verts, final_nn_vids, final_coeffs)
            latent_markers = markers_latent
        elif args.stagei_optimize_latent_markers:
            can_verts = smplx_forward(
                can_model,
                betas=betas,
                body_pose=torch.zeros((1, SMPLX_BODY_DOF), dtype=dtype, device=device),
                global_orient=torch.zeros((1, 3), dtype=dtype, device=device),
                transl=torch.zeros((1, 3), dtype=dtype, device=device),
                batch_size=1,
            )
            final_coeffs = marker_coeffs_from_latent(can_verts, can_faces, marker_vids, tangent_vids, markers_latent)
            pred = reconstruct_markers(verts, faces, marker_vids, tangent_vids, final_coeffs)
            latent_markers = markers_latent
        else:
            final_coeffs = coeffs
            pred = reconstruct_markers(verts, faces, marker_vids, tangent_vids, final_coeffs)
        can_verts = smplx_forward(
            can_model,
            betas=betas,
            body_pose=torch.zeros((1, SMPLX_BODY_DOF), dtype=dtype, device=device),
            global_orient=torch.zeros((1, 3), dtype=dtype, device=device),
            transl=torch.zeros((1, 3), dtype=dtype, device=device),
            batch_size=1,
        )
        if not args.stagei_optimize_latent_markers:
            latent_markers = reconstruct_markers(can_verts, can_faces, marker_vids, tangent_vids, final_coeffs)[0]
        final_data_sse = float(masked_marker_sse(pred, obs, mask, marker_weights=marker_weights).detach().cpu())
        final_data_mse = float(masked_marker_mse(pred, obs, mask, marker_weights=marker_weights).detach().cpu())
        if args.stagei_optimize_latent_markers:
            if use_nn_latent_basis:
                init_target_final = reconstruct_markers_nn(can_verts, init_nn_vids, init_nn_coeffs)[0]
                surface_residual_final = latent_surface_residual(
                    can_verts, can_faces, latent_markers, init_coeffs[:, 2], args.stagei_surface_distance
                )
                final_surface_sse = float(torch.sum(surface_residual_final * surface_residual_final).detach().cpu())
            else:
                init_target_final = reconstruct_markers(can_verts, can_faces, marker_vids, tangent_vids, init_coeffs)[0]
                final_surface_sse = float(torch.sum((final_coeffs[:, 2] - init_coeffs[:, 2]) ** 2).detach().cpu())
            final_init_sse = float(torch.sum((latent_markers - init_target_final) ** 2).detach().cpu())
        else:
            final_init_sse = float(torch.sum((final_coeffs - init_coeffs) ** 2).detach().cpu())
            final_surface_sse = float(torch.sum((final_coeffs[:, 2] - init_coeffs[:, 2]) ** 2).detach().cpu())
        final_pose_sse = float(pose_prior.sse(body_pose_eval).detach().cpu())
        final_beta_sse = float(torch.sum(betas * betas).detach().cpu())
        latent_np = latent_markers.detach().cpu().numpy().astype(np.float64)
        if use_nn_latent_basis:
            nn_vids_np = final_nn_vids.detach().cpu().numpy().astype(np.int64)
            anchor_vids_np = nn_vids_np[:, 0]
            tangent_vids_np = nn_vids_np[:, 1]
            coeffs_np = final_coeffs.detach().cpu().numpy().astype(np.float64)
        else:
            can_verts_np = can_verts[0].detach().cpu().numpy().astype(np.float64)
            faces_np = np.asarray(can_model.faces, dtype=np.int64)
            anchor_vids_np, tangent_vids_np, coeffs_np = reanchor_marker_coeffs(
                can_verts_np,
                faces_np,
                latent_np,
                labels=latent_labels,
                initial_anchor_vids=marker_vids_np,
            )
            nn_vids_np = None

    betas_np = np.zeros(400, dtype=np.float64)
    betas_np[: args.num_betas] = betas.detach().cpu().numpy().reshape(-1).astype(np.float64)
    stagei_data = {
        "betas": betas_np,
        "markers_latent": latent_np,
        "latent_labels": latent_labels,
        "marker_meta": numpy_marker_meta(marker_meta),
        "markers_latent_vids": {label: int(vid) for label, vid in zip(latent_labels, anchor_vids_np)},
        "torch_marker_anchor_vids": {label: int(vid) for label, vid in zip(latent_labels, anchor_vids_np)},
        "torch_marker_coeffs": coeffs_np,
        "torch_marker_tangent_vids": {label: int(vid) for label, vid in zip(latent_labels, tangent_vids_np)},
        "torch_marker_nn_vids": nn_vids_np,
        "stagei_debug_details": {
            "stagei_frame_ids": stagei_ids,
            "stagei_frames": [{label: obs_np[fidx, lid] for lid, label in enumerate(latent_labels) if np.isfinite(obs_np[fidx, lid]).all()} for fidx in stagei_ids],
            "stagei_markers_sim": [x for x in pred.detach().cpu().numpy().astype(np.float64)],
            "stagei_markers_obs": [x for x in obs.detach().cpu().numpy().astype(np.float64)],
            "stagei_labels_obs": [[label for lid, label in enumerate(latent_labels) if bool(mask[j, lid])] for j in range(k)],
            "opt_models_trans": transl.detach().cpu().numpy().astype(np.float64),
            "opt_models_pose": fullpose_from_parts(
                global_orient.detach().cpu().numpy().astype(np.float64),
                body_pose_eval.detach().cpu().numpy().astype(np.float64),
                np.zeros((k, 90), dtype=np.float64),
            ),
            "stagei_errs": {
                "loss": last_loss,
                "data_sse": final_data_sse,
                "data_mse": final_data_mse,
                "init_sse": final_init_sse,
                "pose_sse": final_pose_sse,
                "beta_sse": final_beta_sse,
                "surf_sse": final_surface_sse,
            },
            "stagei_elapsed_time": time.time() - started,
        },
    }
    return stagei_data
