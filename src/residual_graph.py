"""Native residual-tail graph with explicit dynamic tensor inputs."""
import torch


class ResidualGraph:
    """Replay one fixed residual layout without retaining stale frame data.

    The caller keys instances by shape, dtype and temporal layout. All changing
    observations, masks, history and coefficients must be supplied as inputs.
    ``function`` owns the fixed prior/readout tensors referenced by the graph.
    """

    def __init__(self, function, inputs):
        self.function = function
        self.inputs = tuple(x.detach().clone() for x in inputs)
        with torch.no_grad():
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(5):
                    function(*self.inputs)
            torch.cuda.current_stream().wait_stream(stream)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self.output = function(*self.inputs)
            self.graph.replay()

    def __call__(self, inputs):
        with torch.no_grad():
            for target, source in zip(self.inputs, inputs):
                target.copy_(source)
            self.graph.replay()
            return self.output.clone()
