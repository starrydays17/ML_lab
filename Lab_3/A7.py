import pandas as pd
import numpy as np
import math

def dot_product(A, B):
    result = 0
    for i in range(len(A)):
        result += A[i] * B[i]
    return result

def euclidean_norm(vector):
    total = 0
    for value in vector:
        total += value ** 2
    return math.sqrt(total)

df = pd.read_excel(
    "Lab Session Data (1).xlsx",
    sheet_name="marketing_campaign"
)
numeric_df = df.select_dtypes(include=["int64", "float64"])
numeric_df = numeric_df.fillna(numeric_df.mean())
A = numeric_df.iloc[0].tolist()
B = numeric_df.iloc[1].tolist()
my_dot = dot_product(A, B)
my_norm_A = euclidean_norm(A)
my_norm_B = euclidean_norm(B)
numpy_dot = np.dot(A, B)
numpy_norm_A = np.linalg.norm(A)
numpy_norm_B = np.linalg.norm(B)
print("Dot Product (Own Function):", my_dot)
print("Dot Product (NumPy):", numpy_dot)
print("\nEuclidean Norm of A (Own Function):", my_norm_A)
print("Euclidean Norm of A (NumPy):", numpy_norm_A)
print("\nEuclidean Norm of B (Own Function):", my_norm_B)
print("Euclidean Norm of B (NumPy):", numpy_norm_B)