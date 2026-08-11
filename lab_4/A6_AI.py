# =============================================================================
# Lab 04 - A1 (portion for Lab03-A6: Compare vs scipy.spatial.distance.minkowski)
# AI Tool Used: Claude (Anthropic, Claude Sonnet 5) via claude.ai
# Prompt summary: "Regenerate A6.py to reuse the A4_AI distance function and
#   compare against scipy, using numpy.isclose for a robust tolerance check."
# Human review: none needed, logic is a direct, transparent comparison.
# =============================================================================
import pandas as pd
import numpy as np
from scipy.spatial.distance import minkowski
from A4_AI import minkowski_distance


def compare_with_scipy(vector1, vector2, p, tolerance=1e-6):
    """
    Compute Minkowski distance using both the custom function and scipy,
    and report whether they agree within `tolerance`.

    Returns
    -------
    (float, float, bool) : (own_distance, scipy_distance, are_close)
    """
    own_distance = minkowski_distance(vector1, vector2, p)
    scipy_distance = float(minkowski(vector1, vector2, p))
    are_close = bool(np.isclose(own_distance, scipy_distance, atol=tolerance))
    return own_distance, scipy_distance, are_close


if __name__ == "__main__":
    df = pd.read_excel(
        "Lab Session Data (1).xlsx",
        sheet_name="marketing_campaign",
    )
    numeric_df = df.select_dtypes(include=["int64", "float64"])
    numeric_df = numeric_df.fillna(numeric_df.mean())

    vector1 = numeric_df.iloc[0].tolist()
    vector2 = numeric_df.iloc[1].tolist()
    p = int(input("Enter value of p: "))

    my_distance, scipy_distance, are_close = compare_with_scipy(vector1, vector2, p)

    print("Distance using my function      :", my_distance)
    print("Distance using scipy function   :", scipy_distance)
    if are_close:
        print("Both results are the same.")
    else:
        print("Results are different.")
