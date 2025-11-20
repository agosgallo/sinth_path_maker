# sint_path_maker/main.py
"""
Main script for Synthetic Path Maker.

Workflow:
1) Load a CSV (or pick one at random from a folder)
2) Compute weights from smoothed log-price derivatives
3) Build the empirical CDF (and save the plot)
4) Generate a synthetic path guided by the price weights
5) Save comparison plots and export the synthetic path to CSV

Each run creates a new subfolder under `plot/` (plot1, plot2, ...)
to avoid overwriting previous experiments.
"""

from __future__ import annotations
import os
import glob
import random
import argparse
import numpy as np
import pandas as pd

# ---- package-relative imports (work with: python -m src.main) ----
from weights import compute_weights_precise
from CDF_and_inverse import EquispacedPiecewiseConstantCDF
from append_tuples import build_synthetic_path, save_plots_orig_vs_synth
from utils import io_utils
# ---------------------------------------------------------------------------------------------


# =========================
# Helpers
# =========================
def autodetect_price_col(cols) -> str | None:
    """Heuristically detect a price column."""
    candidates = ["Close", "Adj Close", "close", "adj_close", "Price", "price"]
    for c in candidates:
        if c in cols:
            return c
    return None


def load_series_from_csv(
    csv_path: str,
    *,
    date_col: str | None = None,
    price_col: str | None = None,
) -> tuple[np.ndarray, pd.DataFrame, str, str | None]:
    """
    Load and preprocess a CSV file.
    - Parses/sorts by date if date column is present (or auto-detected).
    - Detects/validates price column.
    - Cleans non-finite and non-positive values (shifts up a tiny epsilon if needed).
    Returns: (price_array, cleaned_dataframe, price_col, date_col)
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    original_cols = list(df.columns)

    # 1) Date handling (sort if available)
    if date_col and date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.sort_values(by=date_col).reset_index(drop=True)
    else:
        for dc in ("Date", "date", "Datetime", "datetime", "Timestamp", "timestamp"):
            if dc in df.columns:
                date_col = dc
                df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
                df = df.sort_values(by=date_col).reset_index(drop=True)
                break

    # 2) Price column detection/validation
    if price_col is None:
        price_col = autodetect_price_col(df.columns)
    if (price_col is None) or (price_col not in df.columns):
        raise ValueError(
            "Could not detect a price column. "
            f"Please pass --price-col. Available columns: {original_cols}"
        )

    # 3) Extract and clean price series
    P = pd.to_numeric(df[price_col], errors="coerce").to_numpy()
    mask = np.isfinite(P)
    if not mask.all():
        df = df.loc[mask].reset_index(drop=True)
        P = P[mask]

    # Ensure strictly positive values (log)
    if np.any(P <= 0):
        positive = P[P > 0]
        if positive.size == 0:
            raise ValueError("Invalid price series (all values <= 0).")
        eps = 0.01 * float(np.nanmin(positive))
        P = P + (eps - np.minimum(P, 0.0))
        df[price_col] = P

    return P.astype(float), df, price_col, date_col


def pick_csv_from_dir(csv_dir: str, seed: int | None = None) -> str:
    """Pick one CSV file at random from a directory."""
    files = sorted(glob.glob(os.path.join(csv_dir, "*.csv")))
    if not files:
        raise FileNotFoundError(f"No CSV files found in: {csv_dir}")
    rng = random.Random(seed)
    return rng.choice(files)


def make_valid_savgol_window(N: int, requested: int, polyorder: int) -> int:
    """
    Make a Savitzky–Golay window valid:
    - odd
    - > polyorder
    - <= N-1
    """
    if N <= 3:
        return max(3, polyorder + 1) | 1  # ensure odd
    w = requested
    # clamp
    w = min(max(w, polyorder + 1), max(3, N - 1))
    # enforce odd
    if w % 2 == 0:
        w += 1 if (w + 1) <= (N - 1) else -1
    # final guard
    if w <= polyorder:
        w = polyorder + 1
        if w % 2 == 0:
            w += 1
    if w >= N:
        w = (N - 1) if ((N - 1) % 2 == 1) else (N - 2)
    return max(w, 3)


# =========================
# Main
# =========================
def main():
    # ---- CLI ----
    ap = argparse.ArgumentParser(description="Run synthetic path generation from CSV input.")
    ap.add_argument("--csv", type=str, default=None,
                    help="Path to a CSV file (overrides --csv-dir).")
    ap.add_argument("--csv-dir", type=str, default=None,
                    help="Directory with CSV files; one will be chosen at random.")
    ap.add_argument("--date-col", type=str, default=None, help="Date column name (e.g., Date).")
    ap.add_argument("--price-col", type=str, default=None, help="Price column name (e.g., Close).")
    ap.add_argument("--seed", type=int, default=112, help="Random seed.")
    ap.add_argument("--window", type=int, default=31, help="Savitzky–Golay window size.")
    ap.add_argument("--polyorder", type=int, default=3, help="Savitzky–Golay polynomial order.")
    ap.add_argument("--gamma", type=float, default=1.5)
    ap.add_argument("--alpha", type=float, default=2.0)
    ap.add_argument("--beta", type=float, default=0.2)
    ap.add_argument("--rho", type=float, default=0.1)
    args = ap.parse_args()

    # ---- 1) Run directory ----
    run_dir = io_utils.make_incremental_subdir(base="plot", prefix="plot")
    print("Run directory created at:", os.path.abspath(run_dir))

    # ---- 2) Select CSV ----
    if args.csv:
        csv_path = args.csv
    elif args.csv_dir:
        csv_path = pick_csv_from_dir(args.csv_dir, seed=args.seed)
    else:
        raise SystemExit("Please provide either --csv <file.csv> or --csv-dir <folder>.")
    print(f"Using CSV file: {csv_path}")

    # ---- 3) Load series ----
    P, df_sorted, price_col, date_col = load_series_from_csv(
        csv_path, date_col=args.date_col, price_col=args.price_col
    )
    N = len(P)
    if N < 5:
        raise ValueError(f"Price series too short ({N}). Need at least ~10 points.")

    y = np.arange(N, dtype=float)
    f = np.log(P)

    # Valid SG window
    window = make_valid_savgol_window(N, args.window, args.polyorder)

    # ---- 4) Weights (and smoothed derivative inside) ----
    s, _  , neg , pos= compute_weights_precise(
        y, f,
        window=args.window, polyorder=args.polyorder,
        gamma=args.gamma, alpha=args.alpha, beta=args.beta,
        rho=args.rho, eps=1e-12
    )
    if not np.isfinite(s).all() or np.allclose(s.sum(), 0.0):
        raise ValueError("Invalid weights (non-finite or all-zero). Check data/params.")

    p = s / s.sum()
    # ---- 5) Empirical CDF (and plot) ----
    _ = EquispacedPiecewiseConstantCDF(y, s)
    io_utils.save_cdf_plot(p, run_dir, fname="01_cdf.png")

    # ---- 6) Save initial price plot ----
    io_utils.save_price_plot(P, run_dir, fname="02_price_initial.png")

    # ---- 7) Synthetic path ----
    payload = {"price": P}  # keep the structure ready for multiple features later
    synth = build_synthetic_path(
        payload, p,
        n_periodi=N,
        start_index=None,     # None -> start from last real observation
        use_price_only=True,  # True -> use price returns for all features
        seed=args.seed
    )

    # ---- 8) Plots: original vs synthetic ----
    title = f"Original vs Synthetic | price={price_col}"
    save_plots_orig_vs_synth(payload, synth, outdir=run_dir, title=title)

    # ---- 9) Export CSV ----
    io_utils.save_synth_csv(synth, run_dir, fname="synth_path.csv")

    print("\nAll results saved in:", os.path.abspath(run_dir))


if __name__ == "__main__":
    main()
