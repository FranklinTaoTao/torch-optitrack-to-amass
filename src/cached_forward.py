"""Native Stage-II forward with fixed-shape caching.

Preserves SMPL-X tensor dimensions, operation ordering, pose/shape deformations
and full matrix products. CPU indices avoid CUDA synchronization per joint.
Only valid for the converter's fixed-shape, axis-angle Stage-II calls.
"""
import weakref

import torch
from torch.nn import functional as F
from smplx.lbs import blend_shapes, vertices2joints, batch_rodrigues, transform_mat


class CachedStageIIForward:
    """Cache fixed shape; optionally specialize the internal marker readout.

    ``repeated_skin`` is only for explicitly repeated current-residual rows,
    never finite-difference perturbations. ``vertex_ids`` selects vertices after
    the full pose-blend and skinning products, preserving their accumulation.
    The model must outlive this backend; its weak reference avoids a cycle.
    """

    def __init__(self, model, repeated_skin=False, vertex_ids=None):
        self.model = weakref.proxy(model)
        self.repeated_skin = repeated_skin
        self.vertex_ids = vertex_ids
        self.parents_cpu = model.parents.detach().cpu().tolist()
        self.cache = None
        self.signature = None

    def __call__(self, betas, body_pose, global_orient, transl, batch_size):
        if betas.requires_grad:
            raise ValueError(
                'Stage-II cache requires fixed shape; use native forward when optimizing betas.'
            )
        m = self.model
        signature = (
            betas.data_ptr(), betas._version,
            m.expression.data_ptr(), m.expression._version, batch_size,
        )
        if self.cache is None or self.signature != signature:
            with torch.no_grad():
                components = torch.cat((betas.expand(batch_size, -1), m.expression[:batch_size]), -1)
                shapedirs = torch.cat((m.shapedirs, m.expr_dirs), -1)
                shaped = m.v_template + blend_shapes(components, shapedirs)
                joints = vertices2joints(m.J_regressor, shaped)
                self.cache = shaped, joints
                self.signature = signature
        shaped, joints = self.cache
        zeros3 = body_pose.new_zeros(batch_size, 9)
        hands = body_pose.new_zeros(batch_size, 90)
        pose = torch.cat((global_orient, body_pose, zeros3, hands), -1) + m.pose_mean
        rotations = batch_rodrigues(pose.reshape(-1, 3)).reshape(batch_size, -1, 3, 3)
        eye = torch.eye(3, device=pose.device, dtype=pose.dtype)
        feature = (rotations[:, 1:] - eye).reshape(batch_size, -1)
        offsets = (feature @ m.posedirs).reshape(batch_size, -1, 3)
        expanded_joints = joints[..., None]
        relative = expanded_joints.clone()
        relative[:, 1:] -= expanded_joints[:, m.parents[1:]]
        local = transform_mat(
            rotations.reshape(-1, 3, 3), relative.reshape(-1, 3, 1)
        ).reshape(batch_size, -1, 4, 4)
        chain = [local[:, 0]]
        for j in range(1, len(self.parents_cpu)):
            chain.append(torch.matmul(chain[self.parents_cpu[j]], local[:, j]))
        global_ = torch.stack(chain, dim=1)
        transforms = global_ - F.pad(
            global_ @ F.pad(expanded_joints, (0, 0, 0, 1)),
            (3, 0, 0, 0, 0, 0, 0, 0),
        )
        # Internal current-residual path: all input rows are explicitly repeated.
        # Keep full70 pose blending and joint correction; shrink only the
        # independently verified skin and final vertex multiply operations.
        output_batch = batch_size
        if self.repeated_skin:
            transforms, offsets, shaped, transl = (
                transforms[:1], offsets[:1], shaped[:1], transl[:1]
            )
            batch_size = 1
        skin = (
            m.lbs_weights[None].expand(batch_size, -1, -1)
            @ transforms.reshape(batch_size, len(self.parents_cpu), 16)
        ).reshape(batch_size, -1, 4, 4)
        # Marker-only readout: full pose/skin GEMMs stay native. Only the final
        # independent per-vertex operations skip vertices unused by NN markers.
        if self.vertex_ids is not None:
            skin = skin.index_select(1, self.vertex_ids)
            shaped = shaped.index_select(1, self.vertex_ids)
            offsets = offsets.index_select(1, self.vertex_ids)
        positioned = shaped + offsets
        homogeneous = torch.cat(
            (positioned, positioned.new_ones(batch_size, positioned.shape[1], 1)), -1
        )
        result = (skin @ homogeneous[..., None])[..., :3, 0] + transl[:, None]
        return result.expand(output_batch, -1, -1) if self.repeated_skin else result
