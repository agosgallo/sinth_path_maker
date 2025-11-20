import os
import numpy as np
import matplotlib.pyplot as plt

# =============== 1) INDEX SAMPLING (with replacement) ===============
def _draw_indices_weighted(p, m, *, rng=None):
    """Draw m indices with replacement according to probabilities p (sum=1)."""
    if rng is None:
        rng = np.random.default_rng()
    p = np.asarray(p, float)
    p = p / p.sum()
    n = p.size
    return rng.choice(n, size=m, replace=True, p=p)

# =============== 2) SYNTHETIC PATH (PRICE RETURNS) ===============
def build_synthetic_path(payload, p, n_periodi, *, start_index=None,
                         use_price_only=True, seed=None):
    """
    payload : dict {feature_name: array (N,)}  MUST include 'prezzo'
    p       : probability for each index (len N), derived from weights (only PRICE)
    n_periodi : length of synthetic path (>=1)
    start_index:
        - None  -> start from the LAST real value of each feature
        - j int -> start from the value at row j (base case)
    use_price_only:
        - True  -> apply PRICE returns to ALL features
        - False -> each feature uses its own historical return (same index i)
    """
    assert "price" in payload, "payload must contain 'prezzo'"
    rng = np.random.default_rng(seed)

    # original series
    arrays = {k: np.asarray(v, float) for k, v in payload.items()}
    P = arrays["price"]; N = P.size
    assert N >= 2, "At least 2 prices are required to compute returns."

    # price returns: r[i] = (P[i]-P[i-1])/P[i-1], for i>=1
    r_price = np.zeros(N, float)
    denom = np.maximum(P[:-1], 1e-12)
    r_price[1:] = (P[1:] - P[:-1]) / denom

    # valid indices for returns: i>=1
    p = np.asarray(p, float); p = p / p.sum()
    p_valid = p.copy(); p_valid[0] = 0.0
    if p_valid.sum() == 0.0:
        # fallback: uniform on 1..N-1
        p_valid = np.zeros_like(p); p_valid[1:] = 1.0 / (N - 1)

    # starting index/base values
    if start_index is None:
        # start from LAST real value
        start_vals = {k: arrays[k][-1] for k in arrays}
    else:
        j = int(start_index)
        start_vals = {k: arrays[k][j] for k in arrays}

    # initialize synthetic path
    synth = {k: np.empty(n_periodi, float) for k in arrays}
    for k in synth:
        synth[k][0] = start_vals[k]

    if n_periodi == 1:
        return synth

    # sample indices i1..i_{T-1} (all >=1)
    idx = _draw_indices_weighted(p_valid, n_periodi - 1, rng=rng)

    # iterative update
    for t in range(1, n_periodi):
        i = int(idx[t - 1])             # index used at this step (>=1)
        r_t = r_price[i]                # PRICE return
        if use_price_only:
            for k in synth:
                synth[k][t] = synth[k][t - 1] * (1.0 + r_t)
        else:
            # feature-specific return from the same index i
            for k, base in arrays.items():
                if i == 0 or base[i - 1] == 0.0:
                    rk = 0.0
                else:
                    rk = (base[i] - base[i - 1]) / base[i - 1]
                synth[k][t] = synth[k][t - 1] * (1.0 + rk)

    return synth

# =============== 3) SAVE PLOTS (synthetic starts at 0) ===============
def save_plots_orig_vs_synth(payload, synth, outdir="plot_montecarlo",
                             title="Original vs Synthetic (index from 0)"):
    os.makedirs(outdir, exist_ok=True)
    for k in payload.keys():
        orig = np.asarray(payload[k], float)
        syn  = np.asarray(synth[k], float)

        plt.figure()
        # original: axis 0..N-1
        plt.plot(np.arange(orig.size), orig, marker='o', label=f"{k} original")
        # synthetic: axis 0..T-1 (STARTS AT 0)
        plt.plot(np.arange(syn.size), syn,  marker='o', label=f"{k} synthetic")
        plt.title(f"{title} — {k}")
        plt.xlabel("Time (index from 0)")
        plt.ylabel(k)
        plt.grid(True)
        plt.legend()

        path = os.path.join(outdir, f"{k}_synthetic.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {path}")


## --- Example data ---
#rng = np.random.default_rng(43)
#P = 100 + rng.normal(0, 1, size=100).cumsum()
#V = 100 + rng.normal(0, 1, size=100).cumsum()
#payload = {"price": P, "volume": V}

# --- probabilities from weights (here just uniform for demo) ---
#p = np.ones_like(P) / len(P)

# --- generate synthetic path of 101 steps, starting from day j=4 ---
#synth = build_synthetic_path(payload, p, n_periodi=101, start_index=4,
                            # use_price_only=True, seed=None)

# --- save plots in plot_tuples/ (synthetic with axis starting from 0) ---
#save_plots_orig_vs_synth(payload, synth, outdir="plot_tuples")

