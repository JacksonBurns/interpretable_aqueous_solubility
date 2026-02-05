import pandas as pd

from main import parity_plot

if __name__ == "__main__":
    for f in ("biogen", "ochem"):
        true_arr = pd.read_csv(f"{f}.csv")["logS"].to_numpy()
        pred_arr = pd.read_csv(f"chemeleon_results/{f}_pred.csv")["logS"].to_numpy()
        parity_plot(true_arr, pred_arr, f"results/chemeleon_{f}_full_parity")
        if f == "biogen":
            continue
        mask = (true_arr > -7) & (true_arr < -3)
        parity_plot(true_arr[mask], pred_arr[mask], f"results/chemeleon_{f}_subset_parity")
