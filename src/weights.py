import numpy as np

try:
    from scipy.signal import savgol_filter
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False


def _moving_average(x, k):
    k = int(k)
    if k % 2 == 0:
        k += 1
    pad = k // 2
    xpad = np.pad(x, (pad, pad), mode="edge")
    kernel = np.ones(k, dtype=float) / k
    return np.convolve(xpad, kernel, mode="valid")


def _triangular_smooth(s, rho):
    if rho <= 0.0:
        return s
    s = np.asarray(s, dtype=float)
    out = s.copy()
    if s.size == 1:
        return s
    out[1:-1] = (1 - rho) * s[1:-1] + 0.5 * rho * (s[:-2] + s[2:])
    out[0] = (1 - rho) * s[0] + rho * s[1]
    out[-1] = (1 - rho) * s[-1] + rho * s[-2]
    return out


def compute_weights_precise(
    y, f,
    *,
    window=9,          # SG window (dispari; ≥ polyorder+2)
    polyorder=3,      # SG polyorder (2..4 tipico)
    gamma=1.0,        # >=1
    alpha=2.0,        # peso salite
    beta=0.2,         # peso discese
    rho=0.1,          # smoothing pesi
    eps=1e-12         # offset per evitare tutti-zero
):
    """
    Algoritmo preciso per i pesi su nodi equispaziati.

    Ritorna: s (array di pesi non negativi), e p (probabilità discrete) come secondario.
    """
    y = np.asarray(y, dtype=float)
    f = np.asarray(f, dtype=float)
    assert y.ndim == 1 and f.ndim == 1 and y.size == f.size and y.size >= 3

    dy = float(y[1] - y[0])

    # 1) Derivata liscia
    if window % 2 == 0:
        window += 1
    if window <= polyorder:
        window = polyorder + 3 if (polyorder + 3) % 2 == 1 else polyorder + 4

    if _HAVE_SCIPY:
        d = savgol_filter(f, window_length=window, polyorder=polyorder,
                          deriv=1, delta=dy, mode="interp")
    else:
        fs = _moving_average(f, window)
        # differenze con bordi 1-lato
        d = np.empty_like(fs)
        d[1:-1] = (fs[2:] - fs[:-2]) / (2 * dy)
        d[0] = (fs[1] - fs[0]) / dy
        d[-1] = (fs[-1] - fs[-2]) / dy


    # 2) Pesi direzionali non negativi
    u = np.maximum(d, 0.0)
    v = np.maximum(-d, 0.0)
    s = alpha * np.power(u, gamma) + beta * np.power(v, gamma) + eps

    # Se tutti s≈eps (caso piatto), rendi uniforme
    if not np.any(s > eps):
        s = np.ones_like(s)

    # 3) Smoothing leggero dei pesi
    s = _triangular_smooth(s, rho=rho)

    # 4) (opz.) Probabilità discrete
    S = s.sum()
    if S <= 0.0:
        p = np.full_like(s, 1.0 / s.size)
    else:
        p = s / S
    """
    Return the sum of positive/negative derivatives.

    """
    # 1.1) Count of derivatives for evaluation
    pos = []
    neg = []
    for element in d:

        if element >0: 

            pos.append(d)

        else:
            neg.append(d)


    pos = np.asarray(pos, dtype = float)
    neg = np.asarray(neg, dtype = float)
    
    return s, p , neg , pos

    #1.1) Count of derivatives for evaluation



#Qui devo prendere questo risultato una sola volta quando runna il main ###IDEA: c'è una sola coppia per CSV
# --- Esempio d'uso ---
#if __name__ == "__main__":
    y = np.arange(0, 100, dtype=float)
    f = 0.05 * y + 5.0 * np.sin(y / 12.0)

    s, p , neg , pos = compute_weights_precise(
    y, f,
    
    window=9,          # SG window (dispari; ≥ polyorder+2)
    polyorder=3,      # SG polyorder (2..4 tipico)
    gamma=1.0,        # >=1
    alpha=2.0,        # peso salite
    beta=0.2,         # peso discese
    rho=0.1,          # smoothing pesi
    eps=1e-12         # offset per evitare tutti-zero
)

    bullish_thr = float(input())
    beta = float(input())
    print("Pesi (prime 6):", np.round(s[:], 4))
    print("Probabilità sommano a:", p.sum())
    print('Le derivate negative sommano a:', neg.sum() )
    print('Le derivate positive sommano a:', pos.sum())
    print('seleziona_bullish_trh:' , bullish_thr )
    print('selezione beta di referenza per il calcolo di alpha: ' )
    alpha_ottimo = (bullish_thr * beta *neg) / (pos - bullish_thr*pos)
    print(f'L\' ottimale per alpha dato beta = {beta}  è: { alpha_ottimo}' )