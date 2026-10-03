import torch

from infant_pose_symmetry.deployment.export import PoseNet


class DummyHead(torch.nn.Module):
    def forward(self, features):
        return features.mean(dim=(2, 3), keepdim=True)


class DummyEstimator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.head = DummyHead()

    def extract_feat(self, x):
        return x


def test_posenet_always_returns_tuple():
    model = PoseNet(DummyEstimator())
    output = model(torch.ones(2, 3, 4, 4))
    assert isinstance(output, tuple)
    assert len(output) == 1
    assert output[0].shape == (2, 3, 1, 1)
