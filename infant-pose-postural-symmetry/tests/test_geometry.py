import numpy as np

from infant_pose_symmetry.symmetry.geometry import (
    angle_class,
    canonicalize,
    limb_angles,
)


def test_angle_class_boundaries():
    values = np.array([0, 29.9, 30, 59.9, 60, 120])
    assert angle_class(values).tolist() == [0, 0, 1, 1, 2, 2]


def test_canonicalize_shape():
    keypoints = np.zeros((2, 17, 2), dtype=float)
    keypoints[:, 5] = [0, 0]
    keypoints[:, 6] = [1, 0]
    keypoints[:, 11] = [0, 2]
    keypoints[:, 12] = [1, 2]
    normalized, theta, scale = canonicalize(keypoints)
    assert normalized.shape == (2, 17, 2)
    assert theta.shape == (2,)
    assert scale.shape == (2,)


def test_limb_angles_shape():
    points = np.random.default_rng(0).normal(size=(3, 17, 2))
    angles = limb_angles(points)
    assert angles.shape == (3, 4)
