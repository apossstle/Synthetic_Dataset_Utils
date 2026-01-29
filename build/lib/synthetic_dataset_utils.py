import numpy as np
import pandas as pd


def generate_linear_regression(n=500, slope=1.0, intercept=0.0, noise_std=1.0, x_range=(0, 10), seed=None):
    rng = np.random.default_rng(seed)
    x = rng.uniform(x_range[0], x_range[1], size=n)
    noise = rng.normal(0, noise_std, size=n)
    y = slope * x + intercept + noise
    return pd.DataFrame({"x": x, "y": y})


def generate_gaussian_mixture(n=500, components=((0, 1), (5, 1)), weights=None, seed=None):
    rng = np.random.default_rng(seed)
    k = len(components)
    weights = weights or [1 / k] * k
    labels = rng.choice(k, size=n, p=weights)
    values = np.zeros(n)

    for i, (mu, sigma) in enumerate(components):
        mask = labels == i
        values[mask] = rng.normal(mu, sigma, size=mask.sum())

    return pd.DataFrame({"value": values, "component": labels})


def summary_statistics(values):
    arr = np.asarray(values)
    arr = arr[~np.isnan(arr)]
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()) if arr.size else None,
        "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "min": float(arr.min()) if arr.size else None,
        "max": float(arr.max()) if arr.size else None,
    }


def mse(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float(((y_true - y_pred) ** 2).mean())


def mae(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float(np.abs(y_true - y_pred).mean())


def ks_statistic(sample_a, sample_b):
    a, b = np.sort(sample_a), np.sort(sample_b)
    data = np.concatenate([a, b])
    cdf_a = np.searchsorted(a, data, side="right") / len(a)
    cdf_b = np.searchsorted(b, data, side="right") / len(b)
    return float(np.max(np.abs(cdf_a - cdf_b)))
