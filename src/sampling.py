import numpy as np

def sample_indices_with_replacement(s, M, seed=None):
    """
    Draw M indices with replacement according to weights s (non-negative).
    Returns:
      - samples: array of M indices in [0, N-1]
      - counts : counts per index (multinomial)
      - p      : normalized probabilities used
    """
    rng = np.random.default_rng(seed)
    s = np.asarray(s, dtype=float)
    if not np.any(s > 0):
        # fallback: uniform distribution
        p = np.full_like(s, 1.0 / s.size)
    else:
        p = s / s.sum()

    # Option A: explicit list of M draws
    samples = rng.choice(len(s), size=M, replace=True, p=p)

    # Option B: counts directly via multinomial (equivalent)
    counts = rng.multinomial(M, pvals=p)

    return samples, counts, p

# Example usage
#if __name__ == "__main__":
    s = np.array([2, 1, 3, 0, 4])  # weights (e.g. from derivative or arbitrary)
    M = 10                         # number of draws
    samples, counts, p = sample_indices_with_replacement(s, M, seed=42)

    print("Probabilities p:", np.round(p, 3))
    print("Samples (indices):", samples)
    print("Counts per index:", counts)
