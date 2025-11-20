# validate_seeds.py
import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from main import load_series_from_csv, make_valid_savgol_window
from weights import compute_weights_precise
from append_tuples import build_synthetic_path


def generate_paths_for_seeds(
    csv_path: str,
    n_runs: int,
    base_seed: int,
    window: int,
    polyorder: int,
    gamma: float,
    alpha: float,
    beta: float,
    rho: float,
    use_price_only: bool = True,
    start_index: int = 0,
):
    """
    Generate n_runs synthetic price paths for a single CSV, all starting from P[start_index].
    """
    # 1) Load price series once
    P, df_sorted, price_col, date_col = load_series_from_csv(
        csv_path, date_col=None, price_col=None
    )
    P = np.asarray(P, dtype=float)
    N = len(P)
    if N < 5:
        raise ValueError(f"Series too short: N={N}")

    # 2) Build grid and log-price
    y = np.arange(N, dtype=float)
    f = np.log(P)

    # 3) Valid Savitzky–Golay window
    window_valid = make_valid_savgol_window(N, window, polyorder)

    # 4) Weights and probabilities (same for all seeds)
    s, p, neg, pos = compute_weights_precise(
        y,
        f,
        window=window_valid,
        polyorder=polyorder,
        gamma=gamma,
        alpha=alpha,
        beta=beta,
        rho=rho,
        eps=1e-12,
    )

    # 5) Generate synthetic price paths for different seeds
    all_paths = np.empty((n_runs, N), dtype=float)
    for k in range(n_runs):
        seed = base_seed + k
        payload = {"price": P}
        synth = build_synthetic_path(
            payload,
            p,
            n_periodi=N,
            start_index=start_index,
            use_price_only=use_price_only,
            seed=seed,
        )
        all_paths[k, :] = np.asarray(synth["price"], dtype=float)

    return P, all_paths, (p, s)


def main():
    parser = argparse.ArgumentParser(
        description="Validate Synthetic Path Maker over multiple seeds on one CSV."
    )
    parser.add_argument("--csv", required=True, help="Input CSV file (single asset).")
    parser.add_argument(
        "--n-runs",
        type=int,
        default=100,
        help="Number of synthetic paths (different seeds).",
    )
    parser.add_argument(
        "--base-seed",
        type=int,
        default=0,
        help="Base seed; run k uses base_seed + k.",
    )
    parser.add_argument("--window", type=int, default=31)
    parser.add_argument("--polyorder", type=int, default=3)
    parser.add_argument("--gamma", type=float, default=1.5)
    parser.add_argument("--alpha", type=float, default=2.0)
    parser.add_argument("--beta", type=float, default=0.2)
    parser.add_argument("--rho", type=float, default=0.1)
    parser.add_argument(
        "--use-price-only",
        action="store_true",
        help="If set, use price-only returns for all features.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Index in the historical series from which all synthetic paths start.",
    )
    parser.add_argument(
        "--out-prefix",
        default="validation_price",
        help="Prefix for output files.",
    )

    args = parser.parse_args()

    # ------------------- GENERAZIONE PATHS -------------------
    print("Loading series and generating synthetic paths...")
    P, all_paths, (p, s) = generate_paths_for_seeds(
        csv_path=args.csv,
        n_runs=args.n_runs,
        base_seed=args.base_seed,
        window=args.window,
        polyorder=args.polyorder,
        gamma=args.gamma,
        alpha=args.alpha,
        beta=args.beta,
        rho=args.rho,
        use_price_only=args.use_price_only,
        start_index=args.start_index,
    )

    N = all_paths.shape[1]
    t = np.arange(N, dtype=int)

    print(f"Generated {args.n_runs} paths of length {N}.")
    # ---------------------------------------------------------

    # ========== PLOT DELLE STRADE SIMULATE ==========
    fig, ax = plt.subplots(figsize=(8, 4))

    max_plot = min(all_paths.shape[0], 200)  # non plottare più di 200 traiettorie
    for k in range(max_plot):
        ax.plot(t, all_paths[k, :], alpha=0.08, linewidth=0.7)

    ax.plot(t, P, linewidth=1.5, color="black", label="original")

    ax.set_xlabel("t")
    ax.set_ylabel("price")
    ax.set_title("Simulated paths (raw prices)")
    ax.legend(loc="best")
    fig.tight_layout()

    # salva nella working directory
    out_dir = os.getcwd()
    out_path = os.path.join(out_dir, f"{args.out_prefix}_paths.png")

    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    print(f"Saved paths plot to: {out_path}")
    # =================================================

    # ---------- ORIGIN ALIGNMENT (shift only) ----------
    P0 = float(P[args.start_index])
    P_shift = P - P0
    all_paths_shift = all_paths - P0
    # ---------------------------------------------------

    mean_path = all_paths_shift.mean(axis=0)
    var_path = all_paths_shift.var(axis=0, ddof=0)
    std_path = np.sqrt(var_path)

    df_stats = pd.DataFrame(
        {
            "t": t,
            "price_original": P,
            "price_shifted": P_shift,
            "E_price_shifted": mean_path,
            "Var_price_shifted": var_path,
            "Std_price_shifted": std_path,
        }
    )
    stats_path = os.path.join(out_dir, f"{args.out_prefix}_moments.csv")
    df_stats.to_csv(stats_path, index=False)

    final_values = all_paths_shift[:, -1]
    E_final = final_values.mean()
    Std_final = final_values.std(ddof=0)

    df_final = pd.DataFrame({"X_T": final_values})
    final_path = os.path.join(out_dir, f"{args.out_prefix}_final_samples.csv")
    df_final.to_csv(final_path, index=False)

    print(f"Saved moments to: {stats_path}")
    print(f"Saved terminal samples to: {final_path}")
    print(f"E[X_T]   ≈ {E_final:.6f}")
    print(f"Std[X_T] ≈ {Std_final:.6f}")


if __name__ == "__main__":
    main()
