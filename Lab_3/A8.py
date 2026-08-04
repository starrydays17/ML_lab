import pandas as pd
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
    variance = calculate_variance(data)
    return math.sqrt(variance)

def dataset_statistics(df):
    means = {}
    variances = {}
    std_devs = {}
    for column in df.columns:
        values = df[column].tolist()
        means[column] = calculate_mean(values)
        variances[column] = calculate_variance(values)
        std_devs[column] = calculate_std(values)
    return means, variances, std_devs

df = pd.read_excel(
    "Lab Session Data (1).xlsx",
    sheet_name="marketing_campaign"
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