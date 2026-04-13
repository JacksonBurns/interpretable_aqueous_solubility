"""
I've uploaded you some start code and the actual data for a machine learning experiment I am running. Write additional code to generate the following plots:

1. Parity Plots

For the models run on the entire dataset (the last column in the _pred CSVs, called modelname_pred), make a parity plot with the existing parity plot function.

2. Simultaneous comparison

Also for the full dataset, make a plot with two columns (one for each dataset) showing the performance of each model in terms of RMSE on the horiztonal axis and spearman rho on the vertical axis. Organize the plot as a pareto front, sharing the dominated region.

You should include statistical tests and errors bars in this plot. Using whatever method you deem appropriate, such as 2000 iteration bootstrapping and a simultaneous comparison (you should choose the best, publiation quality method), any model which is indistribuishable from the best on the respsective metric should have solid errors bars, whereas other should be dashed. 

3. Performance trajectory

Finally, the other columns. The other present columns follow the pattern modelname_number of training points_repeition_pred, where the dataset has been downsampled to the indicated size 5 total times. You should make a plot of the log10 number of training points on the horiztonal axis and the performance on the vertical axis. lay out the plots in the 2x2 grid, with columns for the dataset and rows for the RMSE and spearman metrics, respectively. 

This plot should also include statistics - using statsmodels, check which model is the best on each downsample size according to the tukey HSD test, then color that marker solid. Any model which is indistruishable from that model should also be colored solid with others hollow. For the final full dataset, where all points are used, re-use the results of the bootstrapping from plot 2 to determine coloring.

Save all plots as PDFs at 300 dpi to the directory "../results/" with an appropriate name. Ensure you call plt.close() after aevery plotting call, to reduce memory onsumption. Keep the code concise and readable, relying on external libraries wherever possible.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error
from statsmodels.stats.multicomp import pairwise_tukeyhsd


RESULT_DIR = "../results/"
os.makedirs(RESULT_DIR, exist_ok=True)


_df = pd.read_csv("../data/aqsoldbc_no_overlap.csv")

REASONABLE_MIN = _df["logS"].min()
REASONABLE_MAX = _df["logS"].max()

MODEL_COLORS = {
    "esol": "#1f77b4",  # blue
    "symantic_utopia": "#ff7f0e",  # orange
    "pysr_utopia": "#2ca02c",  # green
    "chemeleon": "#d62728",  # red
    "symantic_greedy": "#ffbb78",  # light orange
    "pysr_greedy": "#98df8a",  # light green
    "rf": "#9467bd",  # purple
}


def replace_errors(df):
    # columnwise, replace nan with mean, inf with max, -inf with min
    for col in df.columns:
        if df[col].dtype in [np.float64, np.float32, np.float16]:
            # set +/- inf to nan
            df[col] = df[col].replace(np.inf, np.nan)
            df[col] = df[col].replace(-np.inf, np.nan)

            col_mean = df[col].mean(skipna=True)
            col_max = df[col].max(skipna=True)
            col_min = df[col].min(skipna=True)
            df[col] = df[col].replace(np.nan, col_mean)
            df[col] = df[col].replace(np.inf, col_max)
            df[col] = df[col].replace(-np.inf, col_min)

            # finally, clip values at reasonable bounds based on the training data
            df[col] = df[col].clip(lower=REASONABLE_MIN, upper=REASONABLE_MAX)
    return df


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
    plt.close()



if __name__ == "__main__":
    biogen_df = pd.read_csv("biogen_pred.csv")
    biogen_df = replace_errors(biogen_df)
    ochem_df = pd.read_csv("ochem_pred.csv")
    ochem_df = replace_errors(ochem_df)

    models = list(MODEL_COLORS.keys())
    datasets = {'OCHEM': ochem_df, 'BIOGEN': biogen_df}

    # --- 1. Parity Plots ---
    for ds_name, df in datasets.items():
        y_true = df['logS']
        for model in models:
            y_pred = df[f"{model}_pred"]
            outname = f"../results/parity_{ds_name}_{model}.pdf"
            parity_plot(y_true, y_pred, outname)
            plt.close()

    # --- 2. Simultaneous Comparison ---
    n_boot = 2000
    boot_results = {}

    # Bootstrapping loop to calculate distributions of RMSE and Spearman Rho
    for ds_name, df in datasets.items():
        y_true = df['logS'].values
        n_samples = len(y_true)
        boot_results[ds_name] = {m: {'rmse': [], 'spearman': []} for m in models}
        
        np.random.seed(42)
        indices = np.random.randint(0, n_samples, size=(n_boot, n_samples))
        
        for i in range(n_boot):
            idx = indices[i]
            y_t = y_true[idx]
            for m in models:
                y_p = df[f"{m}_pred"].values[idx]
                rmse = np.sqrt(mean_squared_error(y_t, y_p))
                rho, _ = spearmanr(y_t, y_p)
                boot_results[ds_name][m]['rmse'].append(rmse)
                boot_results[ds_name][m]['spearman'].append(rho)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    indistinguishable_full = {} 

    for ax, ds_name in zip(axes, datasets.keys()):
        indistinguishable_full[ds_name] = {'rmse': [], 'spearman': []}
        means = {m: {'rmse': np.mean(boot_results[ds_name][m]['rmse']), 
                    'spearman': np.mean(boot_results[ds_name][m]['spearman'])} for m in models}
        
        # Selecting the best performing reference points
        best_rmse_model = min(models, key=lambda m: means[m]['rmse'])
        best_spearman_model = max(models, key=lambda m: means[m]['spearman'])
        
        indist_rmse = [best_rmse_model]
        indist_spearman = [best_spearman_model]
        
        best_rmse_boot = np.array(boot_results[ds_name][best_rmse_model]['rmse'])
        best_spearman_boot = np.array(boot_results[ds_name][best_spearman_model]['spearman'])
        
        # 95% Confidence Interval tests across metrics
        for m in models:
            if m != best_rmse_model:
                diff = np.array(boot_results[ds_name][m]['rmse']) - best_rmse_boot
                ci = np.percentile(diff, [2.5, 97.5])
                if ci[0] <= 0 <= ci[1]:
                    indist_rmse.append(m)
            if m != best_spearman_model:
                diff = best_spearman_boot - np.array(boot_results[ds_name][m]['spearman'])
                ci = np.percentile(diff, [2.5, 97.5])
                if ci[0] <= 0 <= ci[1]:
                    indist_spearman.append(m)
                    
        indistinguishable_full[ds_name]['rmse'] = indist_rmse
        indistinguishable_full[ds_name]['spearman'] = indist_spearman
        
        for m in models:
            r_mean = means[m]['rmse']
            s_mean = means[m]['spearman']
            r_ci = np.percentile(boot_results[ds_name][m]['rmse'], [2.5, 97.5])
            s_ci = np.percentile(boot_results[ds_name][m]['spearman'], [2.5, 97.5])
            
            r_err = [[r_mean - r_ci[0]], [r_ci[1] - r_mean]]
            s_err = [[s_mean - s_ci[0]], [s_ci[1] - s_mean]]
            
            r_ls = 'solid' if m in indist_rmse else 'dashed'
            s_ls = 'solid' if m in indist_spearman else 'dashed'
            
            ax.errorbar(r_mean, s_mean, xerr=r_err, color=MODEL_COLORS[m], linestyle=r_ls, alpha=0.7, zorder=1)
            ax.errorbar(r_mean, s_mean, yerr=s_err, color=MODEL_COLORS[m], linestyle=s_ls, alpha=0.7, zorder=1)
            ax.scatter(r_mean, s_mean, color=MODEL_COLORS[m], label=m, zorder=5)

        ax.set_title(f"{ds_name} Dataset")
        ax.set_xlabel("RMSE")
        ax.set_ylabel("Spearman Rho")

    axes[0].legend()
    plt.tight_layout()
    plt.savefig('../results/simultaneous_comparison.pdf', dpi=300)
    plt.close()

    # --- 3. Performance Trajectory ---
    sizes = [10, 22, 50, 112, 251, 561, 1257]
    metrics = ['rmse', 'spearman']

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot lines
    for c, (ds_name, df) in enumerate(datasets.items()):
        y_true = df['logS'].values
        
        for r, metric in enumerate(metrics):
            ax = axes[r, c]
            
            for m in models:
                x_vals = []
                y_vals = []
                
                for size in sizes:
                    reps = []
                    for rep in range(5):
                        col = f"{m}_{size}_{rep}_pred"
                        if col in df.columns:
                            y_pred = df[col].values
                            if metric == 'rmse':
                                reps.append(np.sqrt(mean_squared_error(y_true, y_pred)))
                            else:
                                rho, _ = spearmanr(y_true, y_pred)
                                reps.append(rho)
                    if len(reps) > 0:
                        x_vals.append(size)
                        y_vals.append(np.mean(reps))
                        
                if len(x_vals) > 0:
                    ax.plot(np.log10(x_vals), y_vals, color=MODEL_COLORS[m], zorder=1)

    # Overlay Statistical tests
    for c, (ds_name, df) in enumerate(datasets.items()):
        y_true = df['logS'].values
        for r, metric in enumerate(metrics):
            ax = axes[r, c]
            
            for size in sizes:
                all_vals = []
                groups = []
                for m in models:
                    for rep in range(5):
                        col = f"{m}_{size}_{rep}_pred"
                        if col in df.columns:
                            y_pred = df[col].values
                            if metric == 'rmse':
                                val = np.sqrt(mean_squared_error(y_true, y_pred))
                            else:
                                val, _ = spearmanr(y_true, y_pred)
                            all_vals.append(val)
                            groups.append(m)
                
                if len(all_vals) > 0:
                    # Tukey HSD tests directly per subset 
                    tukey = pairwise_tukeyhsd(all_vals, groups, alpha=0.05)
                    means = pd.DataFrame({'val': all_vals, 'group': groups}).groupby('group').mean()
                    best_m = means['val'].idxmin() if metric == 'rmse' else means['val'].idxmax()
                    
                    indist = [best_m]
                    for row in tukey.summary().data[1:]:
                        g1, g2, meandiff, p_adj, lower, upper, reject = row
                        if best_m in [g1, g2] and not reject:
                            other = g1 if g2 == best_m else g2
                            if other not in indist:
                                indist.append(other)
                                
                    for m in models:
                        if m in means.index:
                            mean_val = means.loc[m, 'val']
                            marker = 'o'
                            facecolor = MODEL_COLORS[m] if m in indist else 'white'
                            ax.scatter(np.log10(size), mean_val, edgecolor=MODEL_COLORS[m], facecolor=facecolor, marker=marker, zorder=5)

            for m in models:
                mean_val = np.mean(boot_results[ds_name][m][metric])
                indist = indistinguishable_full[ds_name][metric]
                facecolor = MODEL_COLORS[m] if m in indist else 'white'
                ax.scatter(np.log10(biogen_df.shape[0] if ds_name == "biogen" else ochem_df.shape[0]), mean_val, edgecolor=MODEL_COLORS[m], facecolor=facecolor, marker='*', s=150, zorder=5)

            if r == 0:
                ax.set_title(ds_name)
            if r == 1:
                ax.set_xlabel("log10|Training Points|")
            if c == 0:
                ax.set_ylabel(metric.upper() if metric == 'rmse' else "Spearman Rho")

            if r == 1 and c == 1:
                for m in models:
                    ax.scatter([], [], color=MODEL_COLORS[m], label=m)
                ax.legend(fontsize=8, loc='best')
                
    plt.tight_layout()
    plt.savefig('../results/performance_trajectory.pdf', dpi=300)
    plt.close()