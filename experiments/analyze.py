import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error

import os
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

PRETTY_NAME = {
    "esol": "ESOL (refit)",
    "symantic_utopia": "SyMANTIC (utopia)",
    "pysr_utopia": "PySR (utopia)",
    "chemeleon": "CheMeleon",
    "symantic_greedy": "SyMANTIC (greedy)",
    "pysr_greedy": "PySR (greedy)",
    "rf": "Random Forest",
    "biogen": "Biogen",
    "ochem": "OChem",
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

    # Labels
    plt.xlabel("Measured log(solubility) [mol/L]", fontsize=12)
    plt.ylabel("Predicted log(solubility) [mol/L]", fontsize=12)

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

    # --- 2. Simultaneous Comparison ---
    n_boot = 2000
    boot_results = {}

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    indistinguishable_full = {} 

    for ax, (ds_name, df) in zip(axes, datasets.items()):
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

        indistinguishable_full[ds_name] = {'rmse': [], 'spearman': []}
        means = {m: {'rmse': np.mean(boot_results[ds_name][m]['rmse']), 
                    'spearman': np.mean(boot_results[ds_name][m]['spearman'])} for m in models}
        
        best_rmse_model = min(models, key=lambda m: means[m]['rmse'])
        best_spearman_model = max(models, key=lambda m: means[m]['spearman'])
        
        indist_rmse = [best_rmse_model]
        indist_spearman = [best_spearman_model]
        
        best_rmse_boot = np.array(boot_results[ds_name][best_rmse_model]['rmse'])
        best_spearman_boot = np.array(boot_results[ds_name][best_spearman_model]['spearman'])
        
        for m in models:
            if m != best_rmse_model:
                diff = np.array(boot_results[ds_name][m]['rmse']) - best_rmse_boot
                ci = np.percentile(diff, [0.416, 99.584])  # 7-way bonferroni correction
                if ci[0] <= 0 <= ci[1]: indist_rmse.append(m)
            if m != best_spearman_model:
                diff = best_spearman_boot - np.array(boot_results[ds_name][m]['spearman'])
                ci = np.percentile(diff, [0.416, 99.584])  # 7-way bonferroni correction
                if ci[0] <= 0 <= ci[1]: indist_spearman.append(m)
                    
        indistinguishable_full[ds_name]['rmse'] = indist_rmse
        indistinguishable_full[ds_name]['spearman'] = indist_spearman
        
        points = []
        for m in models:
            r_mean = means[m]['rmse']
            s_mean = means[m]['spearman']
            points.append((r_mean, s_mean))
            r_ci = np.percentile(boot_results[ds_name][m]['rmse'], [0.416, 99.584])  # 7-way bonferroni correction
            s_ci = np.percentile(boot_results[ds_name][m]['spearman'], [0.416, 99.584])  # 7-way bonferroni correction
            
            r_err = [[r_mean - r_ci[0]], [r_ci[1] - r_mean]]
            s_err = [[s_mean - s_ci[0]], [s_ci[1] - s_mean]]
            
            r_ls = 'solid' if m in indist_rmse else 'dashed'
            s_ls = 'solid' if m in indist_spearman else 'dashed'

            eb = ax.errorbar(r_mean, s_mean, xerr=r_err, color=MODEL_COLORS[m], alpha=0.7, zorder=2)
            eb[-1][0].set_linestyle(r_ls)
            eb = ax.errorbar(r_mean, s_mean, yerr=s_err, color=MODEL_COLORS[m], alpha=0.7, zorder=2)
            eb[-1][0].set_linestyle(s_ls)
            ax.scatter(r_mean, s_mean, color=MODEL_COLORS[m], label=PRETTY_NAME[m], zorder=5)

        # Shade Pareto / Dominated Region
        pareto_points = []
        for i, (r1, s1) in enumerate(points):
            dominated = False
            for j, (r2, s2) in enumerate(points):
                if i != j:
                    if r2 <= r1 and s2 >= s1 and (r2 < r1 or s2 > s1):
                        dominated = True
                        break
            if not dominated:
                pareto_points.append((r1, s1))
                
        pareto_points.sort(key=lambda x: x[0]) 
        
        if pareto_points:
            rs = [p[0] for p in pareto_points]
            ss = [p[1] for p in pareto_points]
            
            r_lim_min, r_lim_max = ax.get_xlim()
            s_lim_min, s_lim_max = ax.get_ylim()
            
            r_max = max(max(rs)*1.1, r_lim_max) + 1.0
            s_min = min(min(ss)*0.9, s_lim_min) - 1.0
            
            shade_r = [r_max, r_max]
            shade_s = [s_min, ss[-1]]
            
            for i in range(len(rs)-1, 0, -1):
                shade_r.append(rs[i])
                shade_s.append(ss[i])
                shade_r.append(rs[i])
                shade_s.append(ss[i-1])
                
            shade_r.append(rs[0])
            shade_s.append(ss[0])
            shade_r.append(rs[0])
            shade_s.append(s_min)
            
            ax.fill(shade_r, shade_s, color='lightgray', alpha=0.5, zorder=0)

        ax.invert_xaxis() 
        # Reset limits gracefully after shading off-screen boundary polygons
        ax.set_xlim(r_lim_max + (r_lim_max-r_lim_min)*0.05, r_lim_min - (r_lim_max-r_lim_min)*0.05)
        ax.set_ylim(s_lim_min - (s_lim_max-s_lim_min)*0.05, s_lim_max + (s_lim_max-s_lim_min)*0.05)

        ax.grid(True, alpha=0.3)
        ax.set_title(PRETTY_NAME[ds_name.lower()])
        ax.set_xlabel("RMSE $\\rightarrow$ (Better)")
        ax.set_ylabel("Spearman Rho $\\rightarrow$ (Better)")
        ax.grid(True)

    axes[0].legend(loc='lower right', fontsize=12)
    plt.tight_layout()
    plt.savefig('../results/simultaneous_comparison.pdf', dpi=300)
    plt.close()

    # --- 3. Performance trajectory ---
    # leaves the greedy traces out for the sake of being easier to read
    sizes = [10, 22, 50, 112, 251, 561, 1257]
    full_size = 2815
    metrics = ['rmse', 'spearman']

    fig, axes = plt.subplots(2, 2, figsize=(8, 6))

    for c, (ds_name, df) in enumerate(datasets.items()):
        y_true = df['logS'].values
        
        for r, metric in enumerate(metrics):
            ax = axes[r, c]
            ax.grid(True, alpha=0.3)
            
            for m in models:
                x_vals = []
                y_vals = []
                y_errs = []

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
                        y_errs.append(np.std(reps))
                        
                if len(x_vals) > 0:
                    full_val = np.mean(boot_results[ds_name][m][metric])
                    full_err = np.std(boot_results[ds_name][m][metric])
                    
                    x_vals.append(full_size)
                    y_vals.append(full_val)
                    y_errs.append(full_err)
                    
                    if "greedy" not in m:
                        ax.plot(np.log10(x_vals), y_vals, color=MODEL_COLORS[m], zorder=2)
                        # ax.errorbar(np.log10(x_vals), y_vals, yerr=y_errs, fmt='none', color=MODEL_COLORS[m], alpha=0.2, zorder=1) <-- not needed, marker filledness shows this

    for c, (ds_name, df) in enumerate(datasets.items()):
        y_true = df['logS'].values
        
        for r, metric in enumerate(metrics):
            ax = axes[r, c]

            y_max = 0.0
            y_min = np.inf
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
                    tukey = pairwise_tukeyhsd(all_vals, groups, alpha=0.05)
                    means = pd.DataFrame({'val': all_vals, 'group': groups}).groupby('group').mean()
                    if metric == 'rmse':
                        best_m = means['val'].idxmin()
                    else:
                        best_m = means['val'].idxmax()
                    
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
                            if "greedy" not in m:
                                ax.scatter(np.log10(size), mean_val, edgecolor=MODEL_COLORS[m], facecolor=facecolor, marker=marker, zorder=5)
                                if mean_val > y_max:
                                    y_max = mean_val
                                if mean_val < y_min:
                                    y_min = mean_val
            for m in models:
                mean_val = np.mean(boot_results[ds_name][m][metric])
                indist = indistinguishable_full[ds_name][metric]
                facecolor = MODEL_COLORS[m] if m in indist else 'white'
                if "greedy" not in m:
                    ax.scatter(np.log10(full_size), mean_val, edgecolor=MODEL_COLORS[m], facecolor=facecolor, marker='*', s=150, zorder=5)
                    if mean_val > y_max:
                        y_max = mean_val
                    if mean_val < y_min:
                        y_min = mean_val
            ax.set_ylim(bottom=y_min*0.95, top=y_max*1.05)

            if r == 0:
                ax.set_title(PRETTY_NAME[ds_name.lower()])
            if r == 1:
                ax.set_xlabel("Training Points ($log_{10}$ scale)")
            if c == 0:
                ax.set_ylabel(metric.upper() if metric == 'rmse' else "Spearman Rho")

            if r == 0 and c == 0:
                handles, labels = ax.get_legend_handles_labels()
                for m in models:
                    if "greedy" not in m:
                        ax.scatter([], [], color=MODEL_COLORS[m], label=PRETTY_NAME[m])
                ax.legend(fontsize=8, loc='upper right')

            # x ticks and labels
            ax.set_xlim(np.log10(sizes[0])*0.95, np.log10(full_size)*1.05)
            ticks = [10, 25, 50, 100, 200, 500, 1000, 3000]
            ax.set_xticks(np.log10(ticks))
            ax.set_xticklabels(list(map(str, ticks)))
                
    plt.tight_layout()
    plt.savefig('../results/performance_trajectory.pdf', dpi=300)
    plt.close()
