import numpy as np
from sklearn.model_selection import StratifiedGroupKFold


def grouped_pose_folds(meta, n_splits=5, seed=0):
    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed,
    )
    folds = np.full(len(meta), -1, dtype=int)
    for fold, (_, validation_index) in enumerate(
        splitter.split(
            np.zeros(len(meta)),
            meta.posture,
            groups=meta.group,
        )
    ):
        folds[validation_index] = fold

    group_fold_count = (
        meta.assign(fold=folds)
        .groupby("group")
        .fold.nunique()
        .max()
    )
    if group_fold_count != 1:
        raise RuntimeError("A source video leaked across folds")

    return folds
