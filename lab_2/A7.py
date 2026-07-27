import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df=pd.read_excel("Lab Session Data (1).xlsx",sheet_name="thyroid0387_UCI")

df=df.replace("?",np.nan)
df=df.replace({"t":1,"f":0,"M":1,"F":0})

for col in df.columns:
    df[col]=pd.to_numeric(df[col],errors="coerce")

df=df.fillna(0)

first20=df.iloc[:20]

n=len(first20)

cos=np.zeros((n,n))

for i in range(n):
    for j in range(n):
        a=first20.iloc[i].values
        b=first20.iloc[j].values

        dot=np.dot(a,b)
        norm1=np.linalg.norm(a)
        norm2=np.linalg.norm(b)

        if norm1==0 or norm2==0:
            cos[i][j]=0
        else:
            cos[i][j]=dot/(norm1*norm2)

sns.heatmap(cos,annot=True)

plt.title("Cosine Similarity Heatmap")
plt.show()