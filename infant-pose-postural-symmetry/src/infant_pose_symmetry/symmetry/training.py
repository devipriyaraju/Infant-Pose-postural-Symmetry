from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from infant_pose_symmetry.models.symmetry_gnn import (
    SymmetryGNN,
    augment_batch,
    flip_batch,
    make_graph_inputs,
)
from infant_pose_symmetry.symmetry.geometry import (
    angle_class,
    limb_angles,
    row_features,
)


def make_logistic_regression():
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=0.5,
            max_iter=3000,
            class_weight="balanced",
        ),
    )


def make_mlp(seed=0):
    return make_pipeline(
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=(64, 32),
            alpha=1e-2,
            learning_rate_init=1e-3,
            max_iter=600,
            random_state=seed,
        ),
    )


def cross_validate_sklearn(model_factory, features, y_sym, y_angle, row_folds):
    score = np.zeros(len(y_sym), dtype=float)
    pred_sym = np.zeros(len(y_sym), dtype=int)
    pred_angle = np.zeros(len(y_sym), dtype=int)

    for fold in np.unique(row_folds):
        train = row_folds != fold
        test = row_folds == fold

        sym_model = model_factory().fit(
            features[train],
            y_sym[train],
        )
        angle_model = model_factory().fit(
            features[train],
            y_angle[train],
        )

        score[test] = sym_model.predict_proba(
            features[test]
        )[:, 1]
        pred_sym[test] = (
            score[test] > 0.5
        ).astype(int)
        pred_angle[test] = angle_model.predict_proba(
            features[test]
        ).argmax(1)

    return {
        "score_sym": score,
        "pred_sym": pred_sym,
        "pred_ang": pred_angle,
    }


def threshold_baseline(angles, y_sym, row_folds, thresholds=None):
    if thresholds is None:
        thresholds = np.arange(0, 90.01, 0.1)

    flat_angles = angles.reshape(-1)
    predictions = np.zeros_like(y_sym)

    for fold in np.unique(row_folds):
        train = row_folds != fold
        test = row_folds == fold

        best_threshold = None
        best_kappa = -np.inf
        for threshold in thresholds:
            candidate = (
                flat_angles[train] < threshold
            ).astype(int)

            observed = (
                candidate == y_sym[train]
            ).mean()
            prevalence_y = y_sym[train].mean()
            prevalence_p = candidate.mean()
            expected = (
                prevalence_y * prevalence_p
                + (1 - prevalence_y) * (1 - prevalence_p)
            )
            kappa = (
                observed - expected
            ) / (1 - expected + 1e-12)

            if kappa > best_kappa:
                best_kappa = kappa
                best_threshold = threshold

        predictions[test] = (
            flat_angles[test] < best_threshold
        ).astype(int)

    return {
        "score_sym": -flat_angles,
        "pred_sym": predictions,
        "pred_ang": angle_class(flat_angles),
    }


def train_gnn(
    node_features,
    geometry,
    y_sym,
    y_angle,
    train_images,
    layer="gat",
    use_geometry=True,
    seed=0,
    epochs=150,
    flip_loss_weight=0.5,
    batch_size=64,
    mirror=True,
    device=None,
):
    device = device or (
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)

    model = SymmetryGNN(
        layer=layer,
        use_geometry=use_geometry,
        mirror=mirror,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=2e-3,
        weight_decay=1e-3,
    )
    steps = epochs * math.ceil(
        len(train_images) / batch_size
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=2e-3,
        total_steps=steps,
    )

    train_images = np.asarray(train_images)
    y_sym_tensor = torch.tensor(
        y_sym,
        dtype=torch.float32,
    )
    y_angle_tensor = torch.tensor(
        y_angle,
        dtype=torch.long,
    )

    positive_rate = float(
        y_sym[train_images].mean()
    )
    positive_weight = torch.tensor(
        (1 - positive_rate) / positive_rate,
        dtype=torch.float32,
        device=device,
    )

    counts = np.bincount(
        y_angle[train_images].reshape(-1),
        minlength=3,
    )
    class_weight = torch.tensor(
        counts.sum() / (3 * counts + 1),
        dtype=torch.float32,
        device=device,
    )

    for _ in range(epochs):
        model.train()
        permutation = torch.as_tensor(
            train_images
        )[
            torch.randperm(
                len(train_images),
                generator=generator,
            )
        ]

        for batch in permutation.split(batch_size):
            x_batch, g_batch = augment_batch(
                node_features[batch],
                geometry[batch],
                generator,
            )
            x_batch = x_batch.to(device)
            g_batch = g_batch.to(device)
            y_batch = y_sym_tensor[batch].to(device)
            a_batch = y_angle_tensor[batch].to(device)

            sym_logit, angle_logit = model(
                x_batch,
                g_batch,
            )
            loss = F.binary_cross_entropy_with_logits(
                sym_logit,
                y_batch,
                pos_weight=positive_weight,
            )
            loss = loss + F.cross_entropy(
                angle_logit.reshape(-1, 3),
                a_batch.reshape(-1),
                weight=class_weight,
            )

            if flip_loss_weight > 0:
                flipped_x, flipped_g = flip_batch(
                    x_batch.cpu(),
                    g_batch.cpu(),
                )
                flipped_sym, flipped_angle = model(
                    flipped_x.to(device),
                    flipped_g.to(device),
                )
                flip_loss = F.mse_loss(
                    torch.sigmoid(sym_logit),
                    torch.sigmoid(flipped_sym),
                )
                flip_loss = flip_loss + F.mse_loss(
                    angle_logit.softmax(-1),
                    flipped_angle.softmax(-1),
                )
                loss = loss + (
                    flip_loss_weight * flip_loss
                )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

    return model


@torch.no_grad()
def predict_gnn(model, node_features, geometry, image_indices, device=None):
    device = device or (
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    model.eval()
    index = torch.as_tensor(
        np.asarray(image_indices)
    )
    x = node_features[index].to(device)
    g = geometry[index].to(device)

    sym_a, angle_a = model(x, g)

    flipped_x, flipped_g = flip_batch(
        x.cpu(),
        g.cpu(),
    )
    sym_b, angle_b = model(
        flipped_x.to(device),
        flipped_g.to(device),
    )

    sym = (
        torch.sigmoid(sym_a)
        + torch.sigmoid(sym_b)
    ) / 2
    angle = (
        angle_a.softmax(-1)
        + angle_b.softmax(-1)
    ) / 2

    return sym.cpu().numpy(), angle.cpu().numpy()


def cross_validate_gnn(
    keypoints,
    scores,
    y_sym,
    y_angle,
    folds,
    layer="gat",
    use_geometry=True,
    seeds=(0, 1, 2),
    epochs=150,
    flip_loss_weight=0.5,
    mirror=True,
    device=None,
):
    node_features, geometry = make_graph_inputs(
        keypoints,
        scores,
    )
    y_sym_image = y_sym.reshape(len(folds), 4)
    y_angle_image = y_angle.reshape(len(folds), 4)

    score = np.zeros((len(folds), 4), dtype=float)
    angle_probability = np.zeros(
        (len(folds), 4, 3),
        dtype=float,
    )

    for fold in np.unique(folds):
        train_images = np.where(folds != fold)[0]
        test_images = np.where(folds == fold)[0]

        for seed in seeds:
            model = train_gnn(
                node_features,
                geometry,
                y_sym_image,
                y_angle_image,
                train_images,
                layer=layer,
                use_geometry=use_geometry,
                seed=seed,
                epochs=epochs,
                flip_loss_weight=flip_loss_weight,
                mirror=mirror,
                device=device,
            )
            sym, angle = predict_gnn(
                model,
                node_features,
                geometry,
                test_images,
                device=device,
            )
            score[test_images] += (
                sym / len(seeds)
            )
            angle_probability[test_images] += (
                angle / len(seeds)
            )

    score = score.reshape(-1)
    angle_probability = angle_probability.reshape(-1, 3)

    return {
        "score_sym": score,
        "pred_sym": (score > 0.5).astype(int),
        "pred_ang": angle_probability.argmax(1),
    }


def evaluate_pose_source(
    keypoints,
    scores,
    y_sym,
    y_angle,
    folds,
    seeds=(0, 1, 2),
    gnn_epochs=150,
    device=None,
):
    row_folds = np.repeat(folds, 4)
    angles = limb_angles(keypoints)
    features = row_features(keypoints, scores)

    results = {
        "angle threshold": threshold_baseline(
            angles,
            y_sym,
            row_folds,
        ),
        "LogReg (geo)": cross_validate_sklearn(
            make_logistic_regression,
            features,
            y_sym,
            y_angle,
            row_folds,
        ),
    }

    mlp_runs = [
        cross_validate_sklearn(
            lambda seed=seed: make_mlp(seed),
            features,
            y_sym,
            y_angle,
            row_folds,
        )
        for seed in seeds
    ]
    mlp_score = np.mean(
        [run["score_sym"] for run in mlp_runs],
        axis=0,
    )
    mlp_angle = np.round(
        np.mean(
            [run["pred_ang"] for run in mlp_runs],
            axis=0,
        )
    ).astype(int)
    results["MLP (geo)"] = {
        "score_sym": mlp_score,
        "pred_sym": (mlp_score > 0.5).astype(int),
        "pred_ang": mlp_angle,
    }

    for name, layer, use_geometry in [
        ("GCN", "gcn", False),
        ("GAT", "gat", False),
        ("GAT + geo", "gat", True),
    ]:
        results[name] = cross_validate_gnn(
            keypoints,
            scores,
            y_sym,
            y_angle,
            folds,
            layer=layer,
            use_geometry=use_geometry,
            seeds=seeds,
            epochs=gnn_epochs,
            device=device,
        )

    return results
