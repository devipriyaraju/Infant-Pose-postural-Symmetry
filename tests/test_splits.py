import pandas as pd

from infant_pose_symmetry.data.splits import grouped_pose_folds


def test_grouped_split_keeps_group_together():
    meta = pd.DataFrame(
        {
            "posture": ["A", "A", "B", "B", "A", "A", "B", "B"],
            "group": ["g1", "g1", "g2", "g2", "g3", "g3", "g4", "g4"],
        }
    )
    folds = grouped_pose_folds(meta, n_splits=2, seed=0)
    meta["fold"] = folds
    assert meta.groupby("group").fold.nunique().max() == 1
