import numpy as np

def match_samples_to_nodes(y, X):
    """
    Monotone 1–to–1 matching:
    - y: array of nodes (sorted, e.g. equispaced)
    - X: samples drawn from p(x)
    Returns a list of pairs (y_j, x_j).
    """
    y = np.asarray(y, dtype=float)
    X = np.sort(np.asarray(X, dtype=float))
    return list(zip(y, X))

# Example
# y = np.arange(5)  # [0,1,2,3,4]
# X = [0.0, 1.5, 2.17, 3.75, 4.25]  # samples from step 3
# pairs = match_samples_to_nodes(y, X)
# print(pairs)
# -> [(0.0, 0.0), (1.0, 1.5), (2.0, 2.17), (3.0, 3.75), (4.0, 4.25)]
