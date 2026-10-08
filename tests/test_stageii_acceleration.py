"""Portable regression tests for the Stage-II acceleration backends.

Run: SMPLX_MODEL_DIR=/path/to/smplx python -m unittest discover -s tests -v
GPU tests skip when CUDA or the licensed SMPL-X assets are unavailable.
"""
import os
from pathlib import Path
import unittest

import torch

from src.cached_forward import CachedStageIIForward
from src.cli import build_parser
from src.cuda_graph_forward import GraphedStageIIForward
from src.helpers import create_smplx_model, smplx_forward
from src.residual_graph import ResidualGraph


MODEL_DIR = Path(os.environ.get(
    "SMPLX_MODEL_DIR",
    Path(__file__).resolve().parents[1] / "support_files" / "smplx",
))
MODELS_AVAILABLE = all(
    (MODEL_DIR / gender / "model.pkl").is_file()
    for gender in ("female", "male", "neutral")
)


class AccelerationOptionsTest(unittest.TestCase):
    def test_defaults_and_opt_outs(self):
        options = (
            "stageii-cache-forward", "stageii-cuda-graphs",
            "stageii-graph-markers", "stageii-graph-residuals",
            "stageii-repeated-skin", "stageii-select-marker-vertices",
        )
        parser = build_parser()
        defaults = parser.parse_args(["--mocap", "placeholder.c3d"])
        for option in options:
            with self.subTest(option=option):
                self.assertTrue(getattr(defaults, option.replace("-", "_")))
                disabled = parser.parse_args([
                    "--mocap", "placeholder.c3d", "--no-" + option,
                ])
                self.assertFalse(getattr(disabled, option.replace("-", "_")))


@unittest.skipUnless(torch.cuda.is_available() and MODELS_AVAILABLE,
                     "CUDA and licensed SMPL-X models required")
class AccelerationBackendTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(4728)

    def make_inputs(self, gender, batch):
        model = create_smplx_model(
            MODEL_DIR / gender / "model.pkl", batch, gender, 16,
            torch.device("cuda"),
        )
        beta = torch.randn(1, 16, device="cuda") * .2
        body = torch.randn(batch, 63, device="cuda") * .4
        root = torch.randn(batch, 3, device="cuda") * .3
        trans = torch.randn(batch, 3, device="cuda") * .5
        return model, beta, body, root, trans

    def test_replay_invalidation_gradients_and_output_ownership(self):
        for gender in ("female", "male", "neutral"):
            for batch in (1, 70):
                with self.subTest(gender=gender, batch=batch):
                    model, beta, body, root, trans = self.make_inputs(gender, batch)
                    backend = GraphedStageIIForward(model)
                    previous = snapshot = None
                    for case in ("initial", "replay", "beta", "expression", "new_beta", "grad", "after_grad"):
                        if case == "beta":
                            beta.add_(.05)
                        if case == "expression":
                            with torch.no_grad():
                                model.expression.add_(.03)
                        if case == "new_beta":
                            beta = beta.clone() - .08
                        body = (body.detach() + .001).requires_grad_(case == "grad")
                        with torch.set_grad_enabled(case == "grad"):
                            expected = smplx_forward(model, beta, body, root, trans, batch)
                            actual = backend(beta, body, root, trans, batch)
                            self.assertTrue(torch.equal(expected, actual), case)
                            if case == "grad":
                                expected_grad = torch.autograd.grad(expected.sum(), body)[0]
                                actual_grad = torch.autograd.grad(actual.sum(), body)[0]
                                self.assertTrue(torch.equal(expected_grad, actual_grad))
                        if previous is not None:
                            self.assertTrue(torch.equal(previous, snapshot))
                        previous = actual.detach()
                        snapshot = previous.clone()

    def test_fixed_shape_guard(self):
        model, beta, body, root, trans = self.make_inputs("female", 1)
        with self.assertRaises(ValueError):
            CachedStageIIForward(model)(beta.requires_grad_(), body, root, trans, 1)

    def test_selected_vertices_and_repeated_current_row(self):
        for gender in ("female", "male", "neutral"):
            with self.subTest(gender=gender):
                model, beta, body, root, trans = self.make_inputs(gender, 70)
                indices = torch.arange(0, len(model.v_template), 83, device="cuda")
                for repeated in (False, True):
                    b = body[:1].expand(70, -1) if repeated else body
                    r = root[:1].expand(70, -1) if repeated else root
                    t = trans[:1].expand(70, -1) if repeated else trans
                    backend = GraphedStageIIForward(
                        model, repeated_skin=repeated, vertex_ids=indices,
                    )
                    with torch.no_grad():
                        expected = smplx_forward(model, beta, b, r, t, 70).index_select(1, indices)
                        actual = backend(beta, b, r, t, 70)
                    # Repeated specialization is used ONLY for current row0;
                    # perturbed FD rows always use the full distinct-row path.
                    self.assertTrue(torch.equal(expected[:1] if repeated else expected,
                                                actual[:1] if repeated else actual))

    def test_residual_inputs_are_refreshed(self):
        def residual(prediction, observation, weight):
            return weight * (prediction - observation)

        inputs = (torch.randn(70, 41, 3, device="cuda"),
                  torch.randn(1, 41, 3, device="cuda"),
                  torch.tensor(2.0, device="cuda"))
        backend = ResidualGraph(residual, inputs)
        previous = backend(inputs)
        snapshot = previous.clone()
        updated = tuple(value + .1 for value in inputs)
        self.assertTrue(torch.equal(backend(updated), residual(*updated)))
        self.assertTrue(torch.equal(previous, snapshot))


if __name__ == "__main__":
    unittest.main()
