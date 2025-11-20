import os
import re
import csv
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# Utilities for plotting and saving
# -------------------------------

def make_incremental_subdir(base="plot", prefix="plot"):
    """
    Create a new subdirectory under `base` with incremental numbering.
    Example: plot/plot1, plot/plot2, ...
    """
    os.makedirs(base, exist_ok=True)
    entries = [
        d for d in os.listdir(base)
        if os.path.isdir(os.path.join(base, d)) and re.fullmatch(rf"{re.escape(prefix)}\d+", d)
    ]
    nums = [int(re.search(r"(\d+)$", d).group(1)) for d in entries]
    next_n = (max(nums) + 1) if nums else 1
    run_dir = os.path.join(base, f"{prefix}{next_n}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir

def save_price_plot(P, outdir, fname="02_price_initial.png", title="Initial Price"):
    plt.figure()
    plt.plot(np.arange(len(P)), P, marker='o')
    plt.title(title); plt.xlabel("Index"); plt.ylabel("Price"); plt.grid(True)
    path = os.path.join(outdir, fname)
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Saved: {path}")

def save_volume_plot(V, outdir, fname="03_volume_initial.png", title="Initial Volume"):
    plt.figure()
    plt.plot(np.arange(len(V)), V, marker='o')
    plt.title(title); plt.xlabel("Index"); plt.ylabel("Volume"); plt.grid(True)
    path = os.path.join(outdir, fname)
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Saved: {path}")

def save_cdf_plot(p, outdir, fname="01_cdf.png", title="Empirical CDF from Weights"):
    p = np.asarray(p, float); p = p / p.sum()
    c = np.cumsum(p)
    x = np.arange(p.size + 1) - 0.5
    y = np.concatenate([[0.0], c])
    plt.figure()
    plt.step(x, y, where="post")
    plt.ylim(-0.02, 1.02)
    plt.title(title); plt.xlabel("Index"); plt.ylabel("F(i)"); plt.grid(True)
    path = os.path.join(outdir, fname)
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Saved: {path}")

def save_synth_csv(synth_dict, outdir, fname="synth_path.csv"):
    """
    Export the synthetic path as CSV.
    Columns: t, feature1, feature2, ...
    """
    keys = list(synth_dict.keys())
    T = len(next(iter(synth_dict.values())))
    path = os.path.join(outdir, fname)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t"] + keys)
        for t in range(T):
            row = [t] + [float(synth_dict[k][t]) for k in keys]
            w.writerow(row)
    print(f"Saved synthetic CSV: {path}")
