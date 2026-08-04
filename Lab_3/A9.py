import pandas as pd
import numpy as np
import math

def calculate_mean(data):
    total = 0
    for value in data:
        total += value
    return total / len(data)

def calculate_variance(data):
    mean = calculate_mean(data)
    total = 0
    for value in data:
        total += (value - mean) ** 2
    return total / len(data)

def calculate_std(data):
    return math.sqrt(calculate_variance(data))

def dataset_statistics(df):
    mean_list = []
    std_list = []
    for column in df.columns:
        values = df[column].tolist()
        mean_list.append(calculate_mean(values))
        std_list.append(calculate_std(values))
    return mean_list, std_list

df = pd.read_excel(
    "Lab Session Data (1).xlsx",
    sheet_name="marketing_campaign"
)
numeric_df = df.select_dtypes(include=["int64", "float64"])
numeric_df = numeric_df.fillna(numeric_df.mean())
my_mean, my_std = dataset_statistics(numeric_df)
numpy_mean = numeric_df.mean().values
numpy_std = numeric_df.std(ddof=0).values
print("Mean Comparison")
print("-" * 60)
print(f"{'Feature':<25}{'Own Mean':<15}{'NumPy Mean'}")
for i, column in enumerate(numeric_df.columns):
    print(f"{column:<25}{my_mean[i]:<15.2f}{numpy_mean[i]:.2f}")
print("\n")
print("Standard Deviation Comparison")
print("-" * 60)
print(f"{'Feature':<25}{'Own Std':<15}{'NumPy Std'}")
for i, column in enumerate(numeric_df.columns):
    print(f"{column:<25}{my_std[i]:<15.2f}{numpy_std[i]:.2f}")