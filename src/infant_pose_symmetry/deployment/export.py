from __future__ import annotations

from pathlib import Path

import torch


class PoseNet(torch.nn.Module):
    """Exportable MMPose backbone, neck and head wrapper."""

    def __init__(self, estimator):
        super().__init__()
        self.estimator = estimator

    def forward(self, x):
        output = self.estimator.head.forward(self.estimator.extract_feat(x))
        return tuple(output) if isinstance(output, (tuple, list)) else (output,)


def export_onnx(
    model,
    output_path: str | Path,
    height: int,
    width: int,
    device: str = "cuda",
    opset: int = 17,
) -> Path:
    """Export an MMPose pose network to ONNX with dynamic batch size."""
    import onnx

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    network = PoseNet(model).eval()
    dummy = torch.randn(1, 3, height, width, device=device)

    with torch.no_grad():
        output_count = len(network(dummy))
    output_names = [f"out{i}" for i in range(output_count)]
    dynamic_axes = {
        "input": {0: "batch"},
        **{name: {0: "batch"} for name in output_names},
    }
    kwargs = dict(
        input_names=["input"],
        output_names=output_names,
        opset_version=opset,
        dynamic_axes=dynamic_axes,
    )

    try:
        torch.onnx.export(
            network,
            dummy,
            str(output_path),
            dynamo=False,
            **kwargs,
        )
    except TypeError:
        torch.onnx.export(network, dummy, str(output_path), **kwargs)

    onnx.checker.check_model(str(output_path))
    return output_path
