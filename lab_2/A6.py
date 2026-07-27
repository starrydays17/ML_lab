import pandas as pd
import numpy as np

df=pd.read_excel("Lab Session Data (1).xlsx",sheet_name="thyroid0387_UCI")

df=df.replace("?",np.nan)
df=df.replace({"t":1,"f":0,"M":1,"F":0})

for col in df.columns:
    df[col]=pd.to_numeric(df[col],errors="coerce")

df=df.fillna(0)

v1=df.iloc[0].values
v2=df.iloc[1].values

dot=np.dot(v1,v2)
norm1=np.linalg.norm(v1)
norm2=np.linalg.norm(v2)

cosine=dot/(norm1*norm2)

print("Cosine Similarity =",cosine)