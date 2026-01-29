import numpy as np, pandas as pd, matplotlib.pyplot as plt
from synthetic_data_utils import generate_regression_linear, generate_categorical, generate_gaussian_mixture, inject_outliers, summary_stats, mse, mae, two_sample_ks_stat, chi2_stat_from_counts, kl_divergence, plot_hist_and_ecdf

def main():
    df = generate_regression_linear(n=400, x_range=(0, 20), slope=3.14, intercept=1.0, noise_std=2.0, random_state=42)
    rng = np.random.default_rng(1)
    y_pred = df['y'] * 0.98 + rng.normal(0, 1.5, size=len(df))
    df['y_pred'] = y_pred
    print("Regression summary stats for y")
    print(summary_stats(df['y']))
    print("MSE between true and pred", mse(df['y'], df['y_pred']))
    print("MAE between true and pred", mae(df['y'], df['y_pred']))

    gm = generate_gaussian_mixture(n=600, components=[(0,1),(5,1.5),(10,2)], weights=[0.4,0.4,0.2], random_state=7)
    print("Gaussian mixture head")
    print(gm.head())

    a = gm.loc[gm['component']==0, 'value']
    b = gm.loc[gm['component']==1, 'value']
    print("Two-sample KS stat between component 0 and 1", two_sample_ks_stat(a, b))

    catA = generate_categorical(n=500, categories=('A','B','C'), probs=[0.6,0.3,0.1], random_state=10)
    catB = generate_categorical(n=500, categories=('A','B','C'), probs=[0.55,0.35,0.1], random_state=11, noise_level=0.05)
    obs_counts = catA['category'].value_counts().reindex(['A','B','C']).fillna(0).values
    exp_probs = np.array([0.6,0.3,0.1])
    print("Chi2 stat for categorical distribution deviation", chi2_stat_from_counts(obs_counts, exp_probs))
    p = obs_counts / obs_counts.sum()
    q = np.array(catB['category'].value_counts().reindex(['A','B','C']).fillna(0).values)
    q = q / q.sum()
    print("KL divergence between two categorical samples", kl_divergence(p, q))

    s, idx = inject_outliers(df['y'], frac=0.02, multiplier=8.0, random_state=3)
    df['y_with_outliers'] = s
    print("Injected outliers indices example", idx[:10])

    sample = df.sample(50, random_state=0).reset_index(drop=True)
    sample.to_csv('sample_dataset.csv', index=False)
    print('Saved sample_dataset.csv')

    plot_hist_and_ecdf(df['y'], filename='regression_y')
    print('Saved regression_y_hist.png and regression_y_ecdf.png')

    try:
        import caas_jupyter_tools as cjt
        cjt.display_dataframe_to_user("sample_dataset", sample)
    except Exception:
        print(sample.head())

if __name__ == "__main__":
    main()
