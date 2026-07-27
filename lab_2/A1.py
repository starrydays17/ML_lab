import pandas as pd
import numpy as np

def load_data():
    df = pd.read_excel("Lab Session Data (1).xlsx", sheet_name="Purchase data")
    return df

def get_matrices(df):
    X = df[["Candies (#)", "Mangoes (Kg)", "Milk Packets (#)"]].values
    y = df["Payment (Rs)"].values
    return X, y

def feature_rank(X):
    return np.linalg.matrix_rank(X)

def product_cost(X, y):
    X_pinv = np.linalg.pinv(X)
    return X_pinv @ y

df = load_data()
X, y = get_matrices(df)
print("Feature Matrix:")
print(X)
print("\nOutput Vector:")
print(y)
print("\nDimension of Vector Space:",3)
print("Number of Vectors:", 10)
print("\nRank of Matrix:", feature_rank(X))
print("\nEstimated Cost of Products:")
print(product_cost(X, y))