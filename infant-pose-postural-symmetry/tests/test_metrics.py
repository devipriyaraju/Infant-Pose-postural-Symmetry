import numpy as np

from infant_pose_symmetry.evaluation.metrics import pck


def test_pck_perfect_predictions():
    ground_truth = np.zeros((2, 17, 3), dtype=float)
    ground_truth[:, :, 2] = 2
    predictions = ground_truth[:, :, :2].copy()
    boxes = np.array([[0, 0, 100, 100], [0, 0, 100, 100]], dtype=float)
    hits = pck(predictions, ground_truth, boxes, alpha=0.05)
    assert np.nanmean(hits) == 1.0
