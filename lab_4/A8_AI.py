# =============================================================================
# Lab 04 - A1 (portion for Lab03-A8: Mean / Variance / Std from scratch)
# AI Tool Used: Claude (Anthropic, Claude Sonnet 5) via claude.ai
# Prompt summary: "Regenerate A8.py's mean/variance/std functions using
#   numpy vectorization instead of manual accumulation loops, and apply
#   them column-wise over the dataset via a single dataset_statistics
#   function."
# Human review: confirmed population variance/std (ddof=0) convention
#   matches the original Lab03 implementation.
# =============================================================================
import numpy as np
import pandas as pd


def calculate_mean(data):
    """Population mean of a 1-D numeric sequence."""
    values = np.asarray(data, dtype=float)
    return float(np.sum(values) / len(values))


def calculate_variance(data):
    """Population variance (denominator = N, not N-1) of a numeric sequence."""
    values = np.asarray(data, dtype=float)
    mean = calculate_mean(values)
    return float(np.sum((values - mean) ** 2) / len(values))


def calculate_std(data):
    """Population standard deviation of a numeric sequence."""
    return float(np.sqrt(calculate_variance(data)))


def dataset_statistics(df):
    """
    Compute mean, variance and standard deviation for every column of a
    numeric DataFrame.

    Returns
    -------
    (dict, dict, dict) : means, variances, std_devs, each keyed by column name
    """
    means, variances, std_devs = {}, {}, {}
    for column in df.columns:
        values = df[column].to_numpy(dtype=float)
        means[column] = calculate_mean(values)
        variances[column] = calculate_variance(values)
        std_devs[column] = calculate_std(values)
    return means, variances, std_devs


if __name__ == "__main__":
    df = pd.read_excel(
        "Lab Session Data (1).xlsx",
        sheet_name="marketing_campaign",
    )
    numeric_df = df.select_dtypes(include=["int64", "float64"])
    numeric_df = numeric_df.fillna(numeric_df.mean())

    means, variances, std_devs = dataset_statistics(numeric_df)

    print("Mean of Each Feature")
    for key, value in means.items():
        print(f"{key}: {value}")

    print("\nVariance of Each Feature")
    for key, value in variances.items():
        print(f"{key}: {value}")

    print("\nStandard Deviation of Each Feature")
    for key, value in std_devs.items():
        print(f"{key}: {value}")
