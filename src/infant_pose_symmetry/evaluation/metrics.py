import numpy as np
from sklearn.metrics import (
    cohen_kappa_score,
    f1_score,
    roc_auc_score,
)


def pck(keypoints, ground_truth, boxes_xywh, alpha=0.05):
    output = []
    for pred, gt, box in zip(keypoints, ground_truth, boxes_xywh):
        size = max(box[2], box[3])
        distance = np.linalg.norm(
            pred - gt[:, :2],
            axis=1,
        ) / size
        output.append(
            np.where(
                gt[:, 2] > 0,
                distance <= alpha,
                np.nan,
            )
        )
    return np.asarray(output)


def symmetry_metrics(y_sym, y_angle, score_sym, pred_sym, pred_angle):
    return {
        "AUROC sym": roc_auc_score(y_sym, score_sym),
        "kappa sym": cohen_kappa_score(y_sym, pred_sym),
        "weighted kappa angle": cohen_kappa_score(
            y_angle,
            pred_angle,
            weights="quadratic",
        ),
        "macro F1 angle": f1_score(
            y_angle,
            pred_angle,
            average="macro",
        ),
    }


def cluster_bootstrap_auc(
    y_true,
    scores,
    groups,
    n_boot=1000,
    seed=0,
):
    rng = np.random.default_rng(seed)
    unique_groups = np.unique(groups)
    members = {
        group: np.where(groups == group)[0]
        for group in unique_groups
    }

    estimates = []
    for _ in range(n_boot):
        sampled_groups = rng.choice(
            unique_groups,
            len(unique_groups),
        )
        index = np.concatenate(
            [members[group] for group in sampled_groups]
        )
        if y_true[index].min() == y_true[index].max():
            continue
        estimates.append(
            roc_auc_score(y_true[index], scores[index])
        )

    return np.percentile(estimates, [2.5, 97.5])


def paired_cluster_bootstrap_auc(
    y_true,
    scores_a,
    scores_b,
    groups,
    n_boot=2000,
    seed=0,
):
    rng = np.random.default_rng(seed)
    unique_groups = np.unique(groups)
    members = {
        group: np.where(groups == group)[0]
        for group in unique_groups
    }

    differences = []
    for _ in range(n_boot):
        sampled_groups = rng.choice(
            unique_groups,
            len(unique_groups),
        )
        index = np.concatenate(
            [members[group] for group in sampled_groups]
        )
        if y_true[index].min() == y_true[index].max():
            continue
        auc_a = roc_auc_score(y_true[index], scores_a[index])
        auc_b = roc_auc_score(y_true[index], scores_b[index])
        differences.append(auc_a - auc_b)

    differences = np.asarray(differences)
    return {
        "mean_delta": float(differences.mean()),
        "ci_low": float(np.percentile(differences, 2.5)),
        "ci_high": float(np.percentile(differences, 97.5)),
        "one_sided_p": float((differences <= 0).mean()),
    }
