import pandas as pd
import matplotlib.pyplot as plt

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
p_values = []
distances = []
for p in range(1, 11):
    d = minkowski_distance(vector1, vector2, p)
    p_values.append(p)
    distances.append(d)
    print(f"p = {p} --> Distance = {d}")
plt.plot(p_values, distances, marker="o")
plt.title("Minkowski Distance for p = 1 to 10")
plt.xlabel("Value of p")
plt.ylabel("Distance")
plt.grid(True)
plt.show()