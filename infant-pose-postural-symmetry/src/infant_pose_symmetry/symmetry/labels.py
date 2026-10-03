import numpy as np


def krippendorff_alpha(data, metric="nominal"):
    data = np.asarray(data, dtype=float)
    values = np.unique(data[~np.isnan(data)])
    counts = np.stack([(data == value).sum(0) for value in values], axis=1).astype(float)
    per_unit = counts.sum(1)
    counts = counts[per_unit >= 2]
    per_unit = per_unit[per_unit >= 2]

    weighted = counts / (per_unit - 1)[:, None]
    coincidence = weighted.T @ counts - np.diag(weighted.sum(0))
    marginals = coincidence.sum(1)
    total = marginals.sum()

    if metric == "nominal":
        distance = 1.0 - np.eye(len(values))
    elif metric == "interval":
        distance = (values[:, None] - values[None, :]) ** 2
    elif metric == "ordinal":
        cumulative = np.concatenate([[0], np.cumsum(marginals)])
        distance = np.zeros((len(values), len(values)))
        for a in range(len(values)):
            for b in range(len(values)):
                low = min(a, b)
                high = max(a, b)
                distance[a, b] = (
                    cumulative[high + 1]
                    - cumulative[low]
                    - (marginals[a] + marginals[b]) / 2
                ) ** 2
    else:
        raise ValueError(f"Unknown metric: {metric}")

    observed = (coincidence * distance).sum() / total
    expected = (np.outer(marginals, marginals) * distance).sum()
    expected = expected / (total * (total - 1))
    return 1 - observed / expected


def raykar_em(
    ratings,
    prior_sensitivity=(2, 2),
    prior_specificity=(2, 2),
    prior_prevalence=(2, 2),
    n_iter=200,
    tol=1e-8,
):
    ratings = np.asarray(ratings, dtype=float)
    posterior = ratings.mean(1)
    a1, a2 = prior_sensitivity
    b1, b2 = prior_specificity
    p1, p2 = prior_prevalence
    eps = 1e-9

    for _ in range(n_iter):
        sensitivity = (a1 - 1 + posterior @ ratings) / (
            a1 + a2 - 2 + posterior.sum()
        )
        specificity = (b1 - 1 + (1 - posterior) @ (1 - ratings)) / (
            b1 + b2 - 2 + (1 - posterior).sum()
        )
        sensitivity = np.clip(sensitivity, eps, 1 - eps)
        specificity = np.clip(specificity, eps, 1 - eps)

        prevalence = (p1 - 1 + posterior.sum()) / (
            p1 + p2 - 2 + len(posterior)
        )
        prevalence = np.clip(prevalence, eps, 1 - eps)

        positive_log_likelihood = (
            ratings @ np.log(sensitivity)
            + (1 - ratings) @ np.log(1 - sensitivity)
        )
        negative_log_likelihood = (
            (1 - ratings) @ np.log(specificity)
            + ratings @ np.log(1 - specificity)
        )

        updated = 1 / (
            1
            + np.exp(
                -(
                    positive_log_likelihood
                    + np.log(prevalence)
                    - negative_log_likelihood
                    - np.log(1 - prevalence)
                )
            )
        )
        if np.abs(updated - posterior).max() < tol:
            posterior = updated
            break
        posterior = updated

    return posterior, sensitivity, specificity, prevalence
