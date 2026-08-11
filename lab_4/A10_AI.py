# =============================================================================
# Lab 04 - A1 (portion for Lab03-A10: Histogram of a feature)
# AI Tool Used: Claude (Anthropic, Claude Sonnet 5) via claude.ai
# Prompt summary: "Regenerate A10.py to use numpy.histogram() to compute
#   bucketed frequency data explicitly (as suggested in the lab sheet),
#   then plot it with matplotlib."
# Human review: none needed.
# =============================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from A8_AI import calculate_mean, calculate_variance


def compute_histogram_buckets(data, bins=10):
    """
    Compute histogram bucket counts and bin edges for a numeric sequence,
    using numpy.histogram as suggested in the lab sheet.

    Returns
    -------
    (numpy.ndarray, numpy.ndarray) : (counts per bucket, bin edges)
    """
    counts, bin_edges = np.histogram(data, bins=bins)
    return counts, bin_edges


if __name__ == "__main__":
    df = pd.read_excel(
        "Lab Session Data (1).xlsx",
        sheet_name="marketing_campaign",
    )
    feature = "Income"
    data = df[feature].dropna().tolist()

    mean = calculate_mean(data)
    variance = calculate_variance(data)
    counts, bin_edges = compute_histogram_buckets(data, bins=10)

    print("Feature :", feature)
    print("Mean =", mean)
    print("Variance =", variance)
    print("\nBucket counts:", counts.tolist())
    print("Bucket edges :", [round(float(edge), 2) for edge in bin_edges])

    plt.hist(data, bins=10, edgecolor="black")
    plt.title("Histogram of " + feature)
    plt.xlabel(feature)
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()
