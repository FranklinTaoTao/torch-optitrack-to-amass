"""Native-operation CUDA graph replay for fixed-shape finite-difference fitting."""
import torch
import os

from .cached_forward import CachedStageIIForward


class GraphedStageIIForward:
    """Own captured inputs/constants and return independent output snapshots.

    Fixed shape/expression changes rebuild the graph. CPU or autograd calls
    use the eager backend; its cache may change without freeing graph constants.
    """

    def __init__(self, model, readout=None, repeated_skin=False, vertex_ids=None):
        self.eager = CachedStageIIForward(
            model, repeated_skin=repeated_skin, vertex_ids=vertex_ids
        )
        self.readout = readout
        self.signature = None
        self.graph = None

    def __call__(self, betas, body, root, trans, batch):
        if not body.is_cuda or (
            torch.is_grad_enabled()
            and any(x.requires_grad for x in (betas, body, root, trans))
        ):
            value = self.eager(betas, body, root, trans, batch)
            return self.readout(value) if self.readout is not None else value
        model = self.eager.model
        signature = (
            betas.data_ptr(), betas._version,
            model.expression.data_ptr(), model.expression._version, batch,
        )
        if self.graph is None or self.signature != signature:
            with torch.no_grad():
                self.inputs = tuple(x.detach().clone() for x in (betas, body, root, trans))
                self.eager(*self.inputs, batch)
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(5):
                        self.eager(*self.inputs, batch)
                torch.cuda.current_stream().wait_stream(stream)
                # Graphs keep raw addresses, not ownership of external constants.
                # An eager/autograd fallback can replace this backend's cache.
                self.capture_constants = self.eager.cache
                self.graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(self.graph):
                    value = self.eager(*self.inputs, batch)
                    self.output = self.readout(value) if self.readout is not None else value
                # Capture records kernels; the first returned output needs replay.
                self.graph.replay()
                self.signature = signature
        else:
            with torch.no_grad():
                for target, source in zip(self.inputs[1:], (body, root, trans)):
                    target.copy_(source)
                self.graph.replay()
        # Every returned value owns its data; later replay cannot mutate it.
        result = self.output.clone()
        if os.environ.get('MOSH_GRAPH_CHECK') == '1':
            with torch.no_grad():
                reference = self.eager(betas, body, root, trans, batch)
                if self.readout is not None:
                    reference = self.readout(reference)
                assert torch.equal(reference, result), (
                    batch, float((reference - result).abs().max())
                )
        return result
