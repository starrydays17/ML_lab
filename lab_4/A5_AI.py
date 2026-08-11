# =============================================================================
# Lab 04 - A1 (portion for Lab03-A5: Minkowski distance p=1..10 plot)
# AI Tool Used: Claude (Anthropic, Claude Sonnet 5) via claude.ai
# Prompt summary: "Regenerate A5.py to reuse the vectorized minkowski
#   distance from A4_AI, and plot p vs distance."
# Human review: kept fillna(mean) imputation strategy from the original
#   so the compared vectors are the same as in the manual version.
# =============================================================================
import pandas as pd
import matplotlib.pyplot as plt
from A4_AI import minkowski_distance


def compute_distance_curve(vector1, vector2, p_values):
    """
    Compute Minkowski distance between two vectors for a range of p values.

    Parameters
    ----------
    vector1, vector2 : sequence of float
    p_values : iterable of int/float

    Returns
    -------
    list[float] of distances, aligned with p_values
    """
    return [minkowski_distance(vector1, vector2, p) for p in p_values]


if __name__ == "__main__":
    df = pd.read_excel(
        "Lab Session Data (1).xlsx",
        sheet_name="marketing_campaign",
    )
    numeric_df = df.select_dtypes(include=["int64", "float64"])
    numeric_df = numeric_df.fillna(numeric_df.mean())

    vector1 = numeric_df.iloc[0].tolist()
    vector2 = numeric_df.iloc[1].tolist()

    p_values = list(range(1, 11))
    distances = compute_distance_curve(vector1, vector2, p_values)

    for p, d in zip(p_values, distances):
        print(f"p = {p} --> Distance = {d}")

    plt.plot(p_values, distances, marker="o")
    plt.title("Minkowski Distance for p = 1 to 10")
    plt.xlabel("Value of p")
    plt.ylabel("Distance")
    plt.grid(True)
    plt.show()
