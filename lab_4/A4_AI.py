# =============================================================================
# Lab 04 - A1 (portion for Lab03-A4: Generalized Minkowski Distance)
# AI Tool Used: Claude (Anthropic, Claude Sonnet 5) via claude.ai
# Prompt summary: "Regenerate A4.py's minkowski_distance function using
#   numpy vectorization instead of an explicit Python for-loop, with input
#   validation on p."
# Human review: verified numerically identical results to the Lab03 loop
#   version for several (vector, p) test cases (see unit tests).
# =============================================================================
import numpy as np


def minkowski_distance(vector1, vector2, p):
    """
    Compute the Minkowski distance of order p between two equal-length
    numeric vectors: (sum(|x_i - y_i|^p))^(1/p)

    Vectorized with numpy instead of an explicit accumulation loop.

    Parameters
    ----------
    vector1, vector2 : sequence of float
    p : int or float, must be >= 1

    Returns
    -------
    float
    """
    if p < 1:
        raise ValueError("p must be >= 1 for a valid Minkowski distance")

    a = np.asarray(vector1, dtype=float)
    b = np.asarray(vector2, dtype=float)
    if a.shape != b.shape:
        raise ValueError("vector1 and vector2 must be the same length")

    return float(np.sum(np.abs(a - b) ** p) ** (1 / p))


if __name__ == "__main__":
    A = [2, 4, 6, 8]
    B = [1, 3, 5, 7]
    p = int(input("Enter value of p: "))

    distance = minkowski_distance(A, B, p)
    print("Minkowski Distance =", distance)

    if p == 1:
        print("This is Manhattan Distance.")
    elif p == 2:
        print("This is Euclidean Distance.")
