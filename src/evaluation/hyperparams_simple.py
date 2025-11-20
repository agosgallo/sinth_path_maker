# src/evaluation/hyperparams_simple.py

# ---- robust import of count_positive_negative ----
try:
    # 1) when running as package: python -m src.main
    from src.weights import count_positive_negative
except ModuleNotFoundError:
    try:
        # 2) relative import when __package__ is set: from inside src/evaluation
        from ..weights import count_positive_negative  # requires -m run and __init__.py
    except Exception:
        # 3) fallback: add parent-of-evaluation to sys.path at runtime
        import os, sys
        THIS_DIR = os.path.dirname(__file__)
        PARENT = os.path.abspath(os.path.join(THIS_DIR, ".."))
        if PARENT not in sys.path:
            sys.path.insert(0, PARENT)
        from weights import count_positive_negative
# --------------------------------------------------

def upward_probability_from_derivatives(d, alpha: float, beta: float, eps: float = 1e-6) -> float:
    pos, neg = count_positive_negative(d)  # usa i conteggi
    if pos <= eps and neg <= eps:
        return 0.5
    return (alpha * pos + eps) / (alpha * pos + beta * neg + 2 * eps)

