import pandas as pd
from scipy.spatial.distance import minkowski

def minkowski_distance(vector1, vector2, p):
    distance = 0
    for i in range(len(vector1)):
        distance += abs(vector1[i] - vector2[i]) ** p
    return distance ** (1 / p)

df = pd.read_excel(
    "Lab Session Data (1).xlsx",
    sheet_name="marketing_campaign"
)
numeric_df = df.select_dtypes(include=["int64", "float64"])
numeric_df = numeric_df.fillna(numeric_df.mean())
vector1 = numeric_df.iloc[0].tolist()
vector2 = numeric_df.iloc[1].tolist()
p = int(input("Enter value of p: "))
my_distance = minkowski_distance(vector1, vector2, p)
scipy_distance = minkowski(vector1, vector2, p)
print("Distance using my function      :", my_distance)
print("Distance using scipy function   :", scipy_distance)
if abs(my_distance - scipy_distance) < 0.000001:
    print("Both results are the same.")
else:
    print("Results are different.")