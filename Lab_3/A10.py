import pandas as pd
import matplotlib.pyplot as plt
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

df = pd.read_excel(
    "Lab Session Data (1).xlsx",
    sheet_name="marketing_campaign"
)
feature = "Income"
data = df[feature].dropna().tolist()
mean = calculate_mean(data)
variance = calculate_variance(data)
print("Feature :", feature)
print("Mean =", mean)
print("Variance =", variance)
plt.hist(data, bins=10, edgecolor="black")
plt.title("Histogram of " + feature)
plt.xlabel(feature)
plt.ylabel("Frequency")
plt.grid(True)
plt.show()