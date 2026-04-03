import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error


def parity_plot(y_true, y_pred, outname):
    """Create a parity plot with hexbin-based density."""
    # Compute regression statistics
    r, _ = pearsonr(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)

    # Create plot
    plt.figure(figsize=(6, 6))

    hb = plt.hexbin(y_true, y_pred, gridsize=50, mincnt=1,)
    cb = plt.colorbar(
        hb,
        fraction=0.04,  # width relative to axes
        pad=0.02,  # gap between plot and colorbar
        shrink=0.85,  # height scaling
    )
    cb.set_label("# of Compounds", fontsize=11)

    # 1:1 line
    lims = [
        min(y_true.min(), y_pred.min()),
        max(y_true.max(), y_pred.max()),
    ]
    plt.plot(lims, lims, color="red", linestyle="--", linewidth=1.5)

    # Labels and title
    plt.xlabel("Measured log(solubility) [mol/L]", fontsize=12)
    plt.ylabel("Predicted log(solubility) [mol/L]", fontsize=12)
    plt.title("Parity Plot", fontsize=14)

    # Annotation with statistics
    stats_text = f"$r$ = {r:.3f}\n" f"RMSE = {rmse:.3f}\n" f"MAE = {mae:.3f}"
    plt.text(
        0.05,
        0.95,
        stats_text,
        transform=plt.gca().transAxes,
        fontsize=11,
        verticalalignment="top",
        horizontalalignment="left",
        bbox=dict(facecolor="white", edgecolor="gray", boxstyle="round,pad=0.3"),
    )

    # Aesthetics
    plt.gca().set_aspect("equal", adjustable="box")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(outname, dpi=300)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr


def bootstrap_model_comparison_plot(df, target_col='logS', n_bootstraps=2000, outname="model_comparison.png"):
    models = [c for c in df.columns if c.endswith('_pred')]
    results = {}

    y_true = df[target_col].values
    indices = np.arange(len(df))

    # Fisher transform
    def fisher_z(r):
        r = np.clip(r, -0.999999, 0.999999)
        return 0.5 * np.log((1 + r) / (1 - r))

    def inv_fisher(z):
        return (np.exp(2*z) - 1) / (np.exp(2*z) + 1)

    # --- 1. Base metrics ---
    print(f"{'Model':<15} | {'RMSE':<10} | {'Pearson r':<10}")
    print("-" * 45)

    for model in models:
        y_pred = df[model].values
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r, _ = pearsonr(y_true, y_pred)

        results[model] = {
            'rmse': rmse,
            'r': r,
            'boot_rmse': [],
            'boot_z': []
        }

        print(f"{model:<15} | {rmse:.4f} | {r:.4f}")

    # --- 2. Bootstrap ---
    print(f"\nRunning {n_bootstraps} bootstraps...")

    for _ in range(n_bootstraps):
        boot_idx = np.random.choice(indices, size=len(indices), replace=True)
        y_boot = y_true[boot_idx]

        for model in models:
            y_pred_boot = df[model].values[boot_idx]

            # RMSE
            mse = mean_squared_error(y_boot, y_pred_boot)
            results[model]['boot_rmse'].append(np.sqrt(mse))

            # Pearson r
            r_boot, _ = pearsonr(y_boot, y_pred_boot)
            results[model]['boot_z'].append(fisher_z(r_boot))

    # --- 3. Compute CIs ---
    for model in models:
        rmse_arr = np.array(results[model]['boot_rmse'])
        z_arr = np.array(results[model]['boot_z'])

        results[model]['rmse_ci'] = np.percentile(rmse_arr, [2.5, 97.5])

        z_ci = np.percentile(z_arr, [2.5, 97.5])
        results[model]['r_ci'] = [inv_fisher(z_ci[0]), inv_fisher(z_ci[1])]

    # --- 4. Pairwise significance ---
    print(f"\n{'Comparison':<30} | {'ΔRMSE (CI)':<25} | {'Δr (CI)':<25}")
    print("-" * 90)

    significance = {}

    for i, m1 in enumerate(models):
        for m2 in models[i+1:]:

            rmse_diff = np.array(results[m1]['boot_rmse']) - np.array(results[m2]['boot_rmse'])
            z_diff = np.array(results[m1]['boot_z']) - np.array(results[m2]['boot_z'])

            # RMSE
            rmse_ci = np.percentile(rmse_diff, [2.5, 97.5])
            rmse_sig = (rmse_ci[0] > 0) or (rmse_ci[1] < 0)

            # r
            z_ci = np.percentile(z_diff, [2.5, 97.5])
            r_ci = [inv_fisher(z_ci[0]), inv_fisher(z_ci[1])]
            r_sig = (r_ci[0] > 0) or (r_ci[1] < 0)

            significance[(m1, m2)] = (rmse_sig, r_sig)

            print(
                f"{m1} vs {m2:<22} | "
                f"{np.mean(rmse_diff):+.4f} [{rmse_ci[0]:+.4f}, {rmse_ci[1]:+.4f}] | "
                f"{inv_fisher(np.mean(z_diff)):+.4f} [{r_ci[0]:+.4f}, {r_ci[1]:+.4f}]"
            )

    # --- 5. Plot ---
    plt.figure(figsize=(7, 6))

    for model in models:
        rmse = results[model]['rmse']
        r = results[model]['r']

        rmse_ci = results[model]['rmse_ci']
        r_ci = results[model]['r_ci']

        # error bars
        plt.errorbar(
            rmse,
            r,
            xerr=[[rmse - rmse_ci[0]], [rmse_ci[1] - rmse]],
            yerr=[[r - r_ci[0]], [r_ci[1] - r]],
            fmt='o',
            capsize=4,
            label=model
        )

        plt.text(rmse, r, f" {model}", fontsize=9, va='center')

    plt.xlabel("RMSE (↓ better)")
    plt.ylabel("Pearson r (↑ better)")
    plt.title("Model Comparison with Bootstrap CIs")

    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outname, dpi=300)

    print(f"\nPlot saved to {outname}")

if __name__ == "__main__":
    biogen_df = pd.read_csv("biogen_pred.csv")
    print("Biogen:")
    bootstrap_model_comparison_plot(biogen_df, outname="biogen_model_comparison.png")
    ochem_df = pd.read_csv("ochem_pred.csv")
    print("\nOChem:")
    bootstrap_model_comparison_plot(ochem_df, outname="ochem_model_comparison.png")
    for model in ("rf", "esol", "symantic", "chemeleon"):
        parity_plot(ochem_df["logS"], ochem_df[f"{model}_pred"], f"../results/{model}_ochem_parity.png")
        parity_plot(biogen_df["logS"], biogen_df[f"{model}_pred"], f"../results/{model}_biogen_parity.png")