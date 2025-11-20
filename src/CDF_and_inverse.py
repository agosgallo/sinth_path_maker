# ===== STEP 2 — CDF + INVERSE (equispaced nodes) =====
# Requirements: NumPy

import numpy as np

class EquispacedPiecewiseConstantCDF:
    """
    Piecewise-constant density p(x) on an equispaced grid:
      - nodes: y_i = y0 + i*dy, i=0..n-1
      - cell i: [ y_i - dy/2, y_i + dy/2 )
      - density on cell i: p(x) = s[i] / Z, with Z = dy * sum(s)

    Provides:
      - cdf(x)   vectorized CDF
      - finv(u)  vectorized inverse (u in (0,1))
      - helper for stratified sampling (optional)
    """

    def __init__(self, y, s):
        y = np.asarray(y, dtype=float)
        s = np.asarray(s, dtype=float)
        assert y.ndim == 1 and s.ndim == 1 and y.size == s.size and y.size >= 2
        self.y0 = float(y[0])
        self.n  = y.size
        self.dy = float(y[1] - y[0])

        # fallback if all weights are non-positive
        if not np.any(s > 0):
            s = np.ones_like(s)

        self.s = s
        # cumulative weights: S[i] = sum_{k<=i} s[k]
        self.S = np.cumsum(s)
        # cumulative masses M[i] = dy * S[i]
        self.M = self.dy * self.S
        self.Z = float(self.M[-1])   # total normalizer

        # left boundary of cell 0
        self.b0 = self.y0 - 0.5 * self.dy

        # Precompute: for handling s[i]==0 in finv, jump to next positive index
        self._next_pos = self._build_next_positive_indices(s)

    @staticmethod
    def _build_next_positive_indices(s):
        n = s.size
        out = np.full(n, n - 1, dtype=int)
        nxt = -1
        # scan right to left, keep the last positive index seen
        for i in range(n - 1, -1, -1):
            if s[i] > 0:
                nxt = i
            out[i] = nxt if nxt != -1 else n - 1
        return out

    def cdf(self, x):
        """
        Vectorized CDF. x can be scalar or array.
        F(x) = (M[i-1] + s[i]*(x - b_i)) / Z for x in cell i.
        """
        x = np.asarray(x, dtype=float)
        # cell index: floor((x - b0)/dy)
        i = np.floor((x - self.b0) / self.dy).astype(int)
        i = np.clip(i, 0, self.n - 1)

        # left boundary of cell i
        b_i = self.b0 + i * self.dy

        # M_{i-1}
        M_prev = np.where(i > 0, self.M[i - 1], 0.0)

        # F(x) = (M_prev + s[i]*(x - b_i)) / Z ; if s[i]=0, the linear term is 0
        Fx_num = M_prev + self.s[i] * (x - b_i)
        return Fx_num / self.Z

    def finv(self, u):
        """
        Inverse CDF (vectorized). u in (0,1). Returns x of same shape.
        Handles zero-mass cells by jumping to next positive index.
        """
        u = np.asarray(u, dtype=float)
        # clamp for numerical safety
        u = np.clip(u, np.nextafter(0.0, 1.0), np.nextafter(1.0, 0.0))

        T = u * self.Z  # target mass
        # Find i with M[i-1] <= T < M[i] using searchsorted on M = dy*S
        idx = np.searchsorted(self.M, T, side="right")

        # correct any idx out of range
        idx = np.clip(idx, 0, self.n - 1)

        # if there's a plateau in M due to s==0, jump to next index with s>0
        idx = self._next_pos[idx]

        # M_{i-1} and left boundary b_i
        M_prev = np.where(idx > 0, self.M[idx - 1], 0.0)
        b_i = self.b0 + idx * self.dy
        s_i = self.s[idx]

        # x = b_i + (T - M_prev) / s_i
        # (s_i > 0 guaranteed by correction idx = next_pos[idx])
        x = b_i + (T - M_prev) / s_i

        # clamp numerically within the cell
        b_i1 = b_i + self.dy
        x = np.minimum(np.maximum(x, b_i), np.nextafter(b_i1, -np.inf))
        return x

    # --- optional: stratified sampling using finv ---
    def sample_stratified(self, N, seed=None):
        rng = np.random.default_rng(seed)
        u = (np.arange(N) + rng.random(N)) / N   # stratified in [0,1)
        return self.finv(u)


# =========================
# ESEMPIO D’USO MINIMALE
#if __name__ == "__main__":
    # griglia equispaziata
    #y = np.arange(5.0)        # [0,1,2,3,4], dy=1
    #s = np.array([2, 1, 3, 0, 4], dtype=float)

    #cdf = EquispacedPiecewiseConstantCDF(y, s)

    # CDF in alcuni punti
    #xs = np.array([-0.5, 0.0, 1.5, 2.2, 4.49])
    #print("F(xs):", np.round(cdf.cdf(xs), 4))

    # Inversa in alcuni quantili
    #us = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    #print("F^{-1}(us):", np.round(cdf.finv(us), 4))

    # 10 campioni stratificati
    #X = cdf.sample_stratified(10, seed=123)
    #print("Campioni stratificati:", np.round(X, 4))
