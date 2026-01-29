import numpy as np
import pandas as pd

def generate_regression_linear(n=500, x_range=(0, 10), slope=2.0, intercept=0.0, noise_std=1.0, random_state=None):
    rng = np.random.default_rng(random_state)
    x = rng.uniform(x_range[0], x_range[1], size=n)
    noise = rng.normal(0, noise_std, size=n)
    y = slope * x + intercept + noise
    return pd.DataFrame({'x': x, 'y': y})

def generate_gaussian_mixture(n=500, components=[(0,1),(5,1)], weights=None, random_state=None):
    rng = np.random.default_rng(random_state)
    k = len(components)
    if weights is None:
        weights = np.ones(k)/k
    comp_ids = rng.choice(k, size=n, p=weights)
    samples = np.empty(n, dtype=float)
    for i, (mu, sigma) in enumerate(components):
        idx = (comp_ids == i)
        samples[idx] = rng.normal(mu, sigma, size=idx.sum())
    return pd.DataFrame({'value': samples, 'component': comp_ids})

def generate_categorical(n=500, categories=('A','B','C'), probs=None, random_state=None, noise_level=0.0):
    rng = np.random.default_rng(random_state)
    k = len(categories)
    if probs is None:
        probs = np.ones(k)/k
    cats = rng.choice(categories, size=n, p=probs)
    if noise_level > 0:
        flip_mask = rng.random(n) < noise_level
        cats[flip_mask] = rng.choice(categories, size=flip_mask.sum())
    return pd.DataFrame({'category': cats})

def inject_outliers(series, frac=0.01, multiplier=10.0, random_state=None):
    rng = np.random.default_rng(random_state)
    s = series.copy().astype(float)
    n = len(s)
    idx = rng.choice(n, size=int(np.ceil(n*frac)), replace=False)
    s.iloc[idx] = s.iloc[idx] * multiplier
    return s, idx

def inject_missing(df, frac=0.05, cols=None, random_state=None):
    rng = np.random.default_rng(random_state)
    df = df.copy()
    n = len(df)
    if cols is None:
        cols = df.columns
    for c in cols:
        mask = rng.random(n) < frac
        df.loc[mask, c] = np.nan
    return df


def summary_stats(arr):
    a = np.asarray(arr)[~np.isnan(arr)]
    return {
        'count': int(a.size),
        'mean': float(np.mean(a)) if a.size>0 else None,
        'median': float(np.median(a)) if a.size>0 else None,
        'std': float(np.std(a, ddof=1)) if a.size>1 else 0.0,
        'min': float(np.min(a)) if a.size>0 else None,
        'max': float(np.max(a)) if a.size>0 else None
    }

def mse(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.mean((y_true - y_pred)**2))

def mae(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))

def two_sample_ks_stat(a, b):
    a = np.sort(np.asarray(a))
    b = np.sort(np.asarray(b))
    n = a.size
    m = b.size
    data_all = np.concatenate([a, b])
    cdf_a = np.searchsorted(a, data_all, side='right') / n
    cdf_b = np.searchsorted(b, data_all, side='right') / m
    return float(np.max(np.abs(cdf_a - cdf_b)))

def chi2_stat_from_counts(observed_counts, expected_probs):
    obs = np.asarray(observed_counts, dtype=float)
    total = obs.sum()
    exp = np.asarray(expected_probs, dtype=float) * total
    eps = 1e-12
    exp = np.maximum(exp, eps)
    return float(((obs - exp)**2 / exp).sum())

def kl_divergence(p, q):
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    eps = 1e-12
    p = p + eps
    q = q + eps
    p = p / p.sum()
    q = q / q.sum()
    return float((p * np.log(p/q)).sum())


def plot_hist_and_ecdf(values, bins=30, filename=None, show=False):
    import matplotlib.pyplot as plt
    v = np.asarray(values)[~np.isnan(values)]
    fig, ax = plt.subplots(figsize=(6,4))
    ax.hist(v, bins=bins)
    ax.set_title('Histogram')
    fig2, ax2 = plt.subplots(figsize=(6,4))
    s = np.sort(v)
    ecdf = np.arange(1, len(s)+1) / len(s)
    ax2.step(s, ecdf)
    ax2.set_title('Empirical CDF')
    if filename:
        fig.savefig(filename + '_hist.png', bbox_inches='tight')
        fig2.savefig(filename + '_ecdf.png', bbox_inches='tight')
    if show:
        plt.show()
    plt.close(fig); plt.close(fig2)
    return None
