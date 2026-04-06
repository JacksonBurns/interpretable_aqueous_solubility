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
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.gridspec as gridspec
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr, spearmanr

def bootstrap_model_comparison_plot(datasets_dict, target_col='logS', n_bootstraps=2000, outname="model_comparison_tiled.pdf"):
    
    all_metrics = []
    
    # Fisher transform helpers for correlation CIs
    def fisher_z(r):
        r = np.clip(r, -0.999999, 0.999999)
        return 0.5 * np.log((1 + r) / (1 - r))

    def inv_fisher(z):
        return (np.exp(2*z) - 1) / (np.exp(2*z) + 1)

    n_plots = len(datasets_dict)
    
    # --- Setup Figure Layout ---
    if n_plots == 2:
        fig = plt.figure(figsize=(12, 4))
        gs = gridspec.GridSpec(1, 3, width_ratios=[1, 0.45, 1])
        axes = [fig.add_subplot(gs[0]), fig.add_subplot(gs[2])]
        ax_leg = fig.add_subplot(gs[1])
        ax_leg.axis('off')
    else:
        fig, axes = plt.subplots(1, n_plots, figsize=(5.5 * n_plots, 5.5))
        if n_plots == 1: axes = [axes]
        ax_leg = None

    model_colors_dict = {}
    color_palette = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # --- Process Each Dataset ---
    for i, (ds_name, df) in enumerate(datasets_dict.items()):
        ax = axes[i]
        print(f"\n{'='*50}\nEvaluating Dataset: {ds_name.upper()}\n{'='*50}")

        models = [c for c in df.columns if c.endswith('_pred')]
        results = {}
        y_true = df[target_col].values
        indices = np.arange(len(df))

        for m in models:
            if m not in model_colors_dict:
                model_colors_dict[m] = color_palette[len(model_colors_dict) % len(color_palette)]

        # 1. Base metrics
        print(f"{'Model':<20} | {'RMSE':<8} | {'MAE':<8} | {'Pearson':<8} | {'Spearman':<8}")
        print("-" * 65)
        for model in models:
            y_pred = df[model].values
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            mae = mean_absolute_error(y_true, y_pred)
            p_r, _ = pearsonr(y_true, y_pred)
            s_r, _ = spearmanr(y_true, y_pred)
            
            results[model] = {
                'rmse': rmse, 'mae': mae, 'pearson': p_r, 'spearman': s_r, 
                'boot_rmse': [], 'boot_mae': [], 'boot_p': [], 'boot_s': []
            }
            print(f"{model:<20} | {rmse:.4f}   | {mae:.4f}   | {p_r:.4f}    | {s_r:.4f}")

        # 2. Bootstrap
        print(f"\nRunning {n_bootstraps} bootstraps for {ds_name}...")
        for _ in range(n_bootstraps):
            boot_idx = np.random.choice(indices, size=len(indices), replace=True)
            y_boot = y_true[boot_idx]
            for model in models:
                y_pred_boot = df[model].values[boot_idx]
                
                results[model]['boot_rmse'].append(np.sqrt(mean_squared_error(y_boot, y_pred_boot)))
                results[model]['boot_mae'].append(mean_absolute_error(y_boot, y_pred_boot))
                
                p_boot, _ = pearsonr(y_boot, y_pred_boot)
                s_boot, _ = spearmanr(y_boot, y_pred_boot)
                results[model]['boot_p'].append(fisher_z(p_boot))
                results[model]['boot_s'].append(fisher_z(s_boot))

        # 3. Compute CIs
        for model in models:
            rmse_arr = np.array(results[model]['boot_rmse'])
            mae_arr = np.array(results[model]['boot_mae'])
            p_arr = np.array(results[model]['boot_p'])
            s_arr = np.array(results[model]['boot_s'])
            
            results[model]['rmse_ci'] = np.percentile(rmse_arr, [2.5, 97.5])
            results[model]['mae_ci'] = np.percentile(mae_arr, [2.5, 97.5])
            
            p_ci_z = np.percentile(p_arr, [2.5, 97.5])
            s_ci_z = np.percentile(s_arr, [2.5, 97.5])
            
            results[model]['pearson_ci'] = [inv_fisher(p_ci_z[0]), inv_fisher(p_ci_z[1])]
            results[model]['spearman_ci'] = [inv_fisher(s_ci_z[0]), inv_fisher(s_ci_z[1])]

        # 4. Pairwise Significance (for plotting lines)
        significance = {}
        for idx_m1, m1 in enumerate(models):
            for m2 in models[idx_m1+1:]:
                rmse_diff = np.array(results[m1]['boot_rmse']) - np.array(results[m2]['boot_rmse'])
                s_diff = np.array(results[m1]['boot_s']) - np.array(results[m2]['boot_s'])

                rmse_ci = np.percentile(rmse_diff, [2.5, 97.5])
                rmse_sig = (rmse_ci[0] > 0) or (rmse_ci[1] < 0)

                s_ci = np.percentile(s_diff, [2.5, 97.5])
                s_sig = (s_ci[0] > 0) or (s_ci[1] < 0)

                significance[(m1, m2)] = (rmse_sig, s_sig)

        def get_sig(m_a, m_b):
            idx_a, idx_b = models.index(m_a), models.index(m_b)
            return significance[(m_a, m_b)] if idx_a < idx_b else significance[(m_b, m_a)]

        # 5. Pareto Calculation & Axis Setup (RMSE vs Spearman)
        points = [{'model': m, 'rmse': results[m]['rmse'], 'spearman': results[m]['spearman']} for m in models]
        
        all_rmse = [pt['rmse'] for pt in points]
        all_s = [pt['spearman'] for pt in points]
        rmse_pad = max((max(all_rmse) - min(all_rmse)) * 0.15, 0.05)
        s_pad = max((max(all_s) - min(all_s)) * 0.15, 0.05)

        min_rmse_plot, max_rmse_plot = min(all_rmse) - rmse_pad, max(all_rmse) + rmse_pad
        min_s_plot, max_s_plot = min(all_s) - s_pad, max(all_s) + s_pad

        # Inverted X axis (RMSE), Standard Y axis (Spearman)
        ax.set_xlim(max_rmse_plot, min_rmse_plot) 
        ax.set_ylim(min_s_plot, max_s_plot)

        # Sort ascending by RMSE (lower is better), descending by Spearman (higher is better)
        points.sort(key=lambda x: (x['rmse'], -x['spearman']))
        
        pareto_front = [points[0]]
        best_s = points[0]['spearman']

        for pt in points[1:]:
            if pt['spearman'] > best_s:
                pareto_front.append(pt)
                best_s = pt['spearman']

        pareto_rmse = [pt['rmse'] for pt in pareto_front]
        pareto_s = [pt['spearman'] for pt in pareto_front]

        # Draw Pareto line and shadow region
        ax.plot(pareto_rmse, pareto_s, '--', color='gray', alpha=0.8, linewidth=2, zorder=1)
        
        pareto_rmse_ext = [pareto_rmse[0]] + pareto_rmse + [max_rmse_plot]
        pareto_s_ext = [min_s_plot] + pareto_s + [pareto_s[-1]]
        ax.fill_between(pareto_rmse_ext, pareto_s_ext, y2=min_s_plot, color='#e5e5e5', alpha=0.6, zorder=0)

        # 6. Plot Models with Split Error Bar Styling
        for m_idx, model in enumerate(models):
            rmse = results[model]['rmse']
            s = results[model]['spearman']
            rmse_ci = results[model]['rmse_ci']
            s_ci = results[model]['spearman_ci']

            is_pareto = any(pt['model'] == model for pt in pareto_front)
            rmse_is_indist = False
            s_is_indist = False
            
            if is_pareto:
                rmse_is_indist, s_is_indist = True, True
            else:
                for p_mod in pareto_front:
                    rmse_sig, s_sig = get_sig(model, p_mod['model'])
                    if not rmse_sig: rmse_is_indist = True
                    if not s_sig: s_is_indist = True

            color = model_colors_dict[model]
            mfc = color if is_pareto else 'none'
            alpha_val = 1.0 if is_pareto else 0.8
            
            eb = ax.errorbar(
                rmse, s,
                xerr=[[rmse - rmse_ci[0]], [rmse_ci[1] - rmse]],
                yerr=[[s - s_ci[0]], [s_ci[1] - s]],
                fmt='o', color=color, capsize=4, zorder=3, alpha=alpha_val, 
                markersize=8, markerfacecolor=mfc, markeredgewidth=2
            )

            ls_x = '-' if rmse_is_indist else '--'
            ls_y = '-' if s_is_indist else '--'

            if len(eb[2]) == 2:
                eb[2][0].set_linestyle(ls_x)
                eb[2][1].set_linestyle(ls_y)
            if len(eb[1]) == 4:
                eb[1][0].set_linestyle(ls_x)
                eb[1][1].set_linestyle(ls_x)
                eb[1][2].set_linestyle(ls_y)
                eb[1][3].set_linestyle(ls_y)

            # --- SAVE ALL METRICS TO DATAFRAME ---
            all_metrics.append({
                'Dataset': ds_name,
                'Model': model.replace('_pred', ''),
                'RMSE': rmse,
                'RMSE_CI_lower': rmse_ci[0],
                'RMSE_CI_upper': rmse_ci[1],
                'MAE': results[model]['mae'],
                'MAE_CI_lower': results[model]['mae_ci'][0],
                'MAE_CI_upper': results[model]['mae_ci'][1],
                'Pearson_r': results[model]['pearson'],
                'Pearson_CI_lower': results[model]['pearson_ci'][0],
                'Pearson_CI_upper': results[model]['pearson_ci'][1],
                'Spearman_rho': s,
                'Spearman_CI_lower': s_ci[0],
                'Spearman_CI_upper': s_ci[1],
                'Is_Pareto': is_pareto
            })

        # 7. Subplot Formatting
        ax.set_xlabel("RMSE (→ lower is better)", fontsize=10, weight='bold')
        ax.set_ylabel("Spearman rho (→ higher is better)", fontsize=10, weight='bold')
        ax.set_title(f"{ds_name.upper()} Dataset", fontsize=11)
        ax.grid(alpha=0.3, linestyle=':')

    # --- Global Formatting & Central Legend ---
    if ax_leg is not None:
        model_handles = [
            mlines.Line2D([], [], color=color, marker='s', linestyle='none', markersize=8, label=m.replace('_pred', ''))
            for m, color in model_colors_dict.items()
        ]
        
        style_handles = [
            mlines.Line2D([], [], color='gray', marker='o', markerfacecolor='gray', linestyle='none', label='Pareto Optimal'),
            mlines.Line2D([], [], color='gray', marker='o', markerfacecolor='none', linestyle='none', label='Dominated'),
            mlines.Line2D([], [], color='gray', linestyle='-', label='Indistinguishable metric'),
            mlines.Line2D([], [], color='gray', linestyle='--', label='Significantly Worse metric'),
            mlines.Line2D([], [], color='gray', linestyle=':', label='Pareto Boundary')
        ]
        
        leg_models = ax_leg.legend(handles=model_handles, loc='upper center', title="Models", bbox_to_anchor=(0.35, 1.0))
        ax_leg.add_artist(leg_models) 
        ax_leg.legend(handles=style_handles, loc='lower center', title="Statistical Status Guides", bbox_to_anchor=(0.35, 0.0), fontsize=9)

    plt.tight_layout()
    plt.savefig(outname, dpi=300)
    print(f"\nPlot saved to {outname}")

    # --- Compile DataFrame ---
    df_results = pd.DataFrame(all_metrics)
    csv_outname = outname.replace('.pdf', '.csv') if outname.endswith('.pdf') else f"{outname}.csv"
    df_results.to_csv(csv_outname, index=False)
    print(f"Metrics saved to {csv_outname}")
    
    return df_results


if __name__ == "__main__":
    biogen_df = pd.read_csv("biogen_pred.csv")
    ochem_df = pd.read_csv("ochem_pred.csv")
    datasets_dict = {"biogen": biogen_df, "ochem": ochem_df}
    bootstrap_model_comparison_plot(datasets_dict, outname="../results/simultaneous_comparison.pdf")

    for model in ["pysr", "symantic", "symanticgp", "esol", "rf", "chemeleon"]:
        parity_plot(ochem_df["logS"], ochem_df[f"{model}_pred"], f"../results/{model}_ochem_parity.pdf")
        parity_plot(biogen_df["logS"], biogen_df[f"{model}_pred"], f"../results/{model}_biogen_parity.pdf")
