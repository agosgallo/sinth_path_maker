# validate_seeds.py
import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from main import load_series_from_csv, make_valid_savgol_window
from weights import compute_weights_precise
from append_tuples import build_synthetic_path
from metrics import compute_returns, realised_volatility


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
    Generate n_runs synthetic price paths for a single CSV,
    all starting from P[start_index].
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
            start_index=start_index,     # all paths start from same historical index
            use_price_only=use_price_only,
            seed=seed,
        )
        all_paths[k, :] = np.asarray(synth["price"], dtype=float)

    return P, all_paths, (p, s)


def _moment_stats(x: np.ndarray):
    """
    Basic moment + quantile summary for a 1D array:
    mean, std, skewness, kurtosis, and a few quantiles.
    """
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return {
            "mean": np.nan,
            "std": np.nan,
            "skew": np.nan,
            "kurt": np.nan,
            "q01": np.nan,
            "q05": np.nan,
            "q50": np.nan,
            "q95": np.nan,
            "q99": np.nan,
        }

    mean = x.mean()
    std = x.std(ddof=0)
    if std > 0:
        skew = ((x - mean) ** 3).mean() / (std ** 3)
        kurt = ((x - mean) ** 4).mean() / (std ** 4)
    else:
        skew = np.nan
        kurt = np.nan

    q01, q05, q50, q95, q99 = np.quantile(x, [0.01, 0.05, 0.5, 0.95, 0.99])

    return {
        "mean": mean,
        "std": std,
        "skew": skew,
        "kurt": kurt,
        "q01": q01,
        "q05": q05,
        "q50": q50,
        "q95": q95,
        "q99": q99,
    }


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

    # ------------------- GENERATE PATHS -------------------
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
    out_dir = os.getcwd()
    # -----------------------------------------------------

    # ========== PLOT OF SIMULATED PATHS ==========
    fig, ax = plt.subplots(figsize=(8, 4))

    max_plot = min(all_paths.shape[0], 200)  # limit visual clutter
    for k in range(max_plot):
        ax.plot(t, all_paths[k, :], alpha=0.08, linewidth=0.7)

    ax.plot(t, P, linewidth=1.5, color="black", label="original")

    ax.set_xlabel("t")
    ax.set_ylabel("price")
    ax.set_title("Simulated paths (raw prices)")
    ax.legend(loc="best")
    fig.tight_layout()

    paths_png = os.path.join(out_dir, f"{args.out_prefix}_paths.png")
    fig.savefig(paths_png, dpi=150)
    plt.close(fig)

    print(f"Saved paths plot to: {paths_png}")
    # ============================================

    # ---------- ORIGIN ALIGNMENT (shift only) ----------
    P0 = float(P[args.start_index])
    P_shift = P - P0
    all_paths_shift = all_paths - P0
    # ---------------------------------------------------

    # ========= TIME-WISE METRICS =========
    mean_path = all_paths_shift.mean(axis=0)
    var_path = all_paths_shift.var(axis=0, ddof=0)
    std_path = np.sqrt(var_path)

    # fraction of synthetic paths above original at each t
    coverage = (all_paths_shift >= P_shift).mean(axis=0)

    # z-score of the original path vs ensemble
    z_score = np.full_like(mean_path, np.nan, dtype=float)
    mask = std_path > 0
    z_score[mask] = (P_shift[mask] - mean_path[mask]) / std_path[mask]
    # =====================================

    df_stats = pd.DataFrame(
        {
            "t": t,
            "price_original": P,
            "price_shifted": P_shift,
            "E_price_shifted": mean_path,
            "Std_price_shifted": std_path,
            "coverage_above_original": coverage,
            "z_original": z_score,
        }
    )
    stats_path = os.path.join(out_dir, f"{args.out_prefix}_moments.csv")
    df_stats.to_csv(stats_path, index=False)

    # ========= PER-PATH METRICS (X_T, realised vol) =========
    K = all_paths.shape[0]
    per_path_metrics = {
        "run_id": [],
        "X_T": [],
        "realised_vol": [],
    }

    for k in range(K):
        Pk = all_paths[k, :]             # raw prices
        Xk_T = all_paths_shift[k, -1]    # shifted terminal value

        rk = compute_returns(Pk)
        vol_k = realised_volatility(rk)

        per_path_metrics["run_id"].append(k)
        per_path_metrics["X_T"].append(Xk_T)
        per_path_metrics["realised_vol"].append(vol_k)

    df_paths = pd.DataFrame(per_path_metrics)
    paths_metrics_path = os.path.join(out_dir, f"{args.out_prefix}_per_path_metrics.csv")
    df_paths.to_csv(paths_metrics_path, index=False)
    # =====================================

    # ========= TERMINAL VALUE SUMMARY =========
    final_values = all_paths_shift[:, -1]
    E_final = final_values.mean()
    Std_final = final_values.std(ddof=0)

    df_final = pd.DataFrame({"X_T": final_values})
    final_path = os.path.join(out_dir, f"{args.out_prefix}_final_samples.csv")
    df_final.to_csv(final_path, index=False)
    # ==========================================

    # ========= RETURN-LEVEL STATS (ORIGINAL vs SYNTHETIC) =========
    # Original returns
    r_orig = compute_returns(P)
    orig_stats = _moment_stats(r_orig)

    # Pooled synthetic returns
    all_returns = []
    for k in range(K):
        rk = compute_returns(all_paths[k, :])
        all_returns.append(rk)
    if all_returns:
        all_returns = np.concatenate(all_returns, axis=0)
    syn_stats = _moment_stats(all_returns)

    df_ret = pd.DataFrame(
        {
            "series": ["original", "synthetic_pooled"],
            "mean": [orig_stats["mean"], syn_stats["mean"]],
            "std": [orig_stats["std"], syn_stats["std"]],
            "skew": [orig_stats["skew"], syn_stats["skew"]],
            "kurt": [orig_stats["kurt"], syn_stats["kurt"]],
            "q01": [orig_stats["q01"], syn_stats["q01"]],
            "q05": [orig_stats["q05"], syn_stats["q05"]],
            "q50": [orig_stats["q50"], syn_stats["q50"]],
            "q95": [orig_stats["q95"], syn_stats["q95"]],
            "q99": [orig_stats["q99"], syn_stats["q99"]],
        }
    )
    ret_stats_path = os.path.join(out_dir, f"{args.out_prefix}_return_stats.csv")
    df_ret.to_csv(ret_stats_path, index=False)
    # =============================================================

    # console summary
    print(f"Saved time-wise moments to: {stats_path}")
    print(f"Saved per-path metrics to: {paths_metrics_path}")
    print(f"Saved terminal samples to: {final_path}")
    print(f"Saved return stats to: {ret_stats_path}")
    print(f"E[X_T]   ≈ {E_final:.6f}")
    print(f"Std[X_T] ≈ {Std_final:.6f}")
    print(
        "Original returns: "
        f"mean={orig_stats['mean']:.3e}, std={orig_stats['std']:.3e}, "
        f"skew={orig_stats['skew']:.3f}, kurt={orig_stats['kurt']:.3f}"
    )
    print(
        "Synthetic pooled returns: "
        f"mean={syn_stats['mean']:.3e}, std={syn_stats['std']:.3e}, "
        f"skew={syn_stats['skew']:.3f}, kurt={syn_stats['kurt']:.3f}"
    )


if __name__ == "__main__":
    main()

