import pandas as pd
import numpy as np

df=pd.read_excel("Lab Session Data (1).xlsx",sheet_name="thyroid0387_UCI")

df=df.replace("?",np.nan)

numeric=[]

for col in df.columns:
    temp=pd.to_numeric(df[col],errors="coerce")
    if temp.notna().sum()>0:
        df[col]=temp.fillna(temp.mean())
        numeric.append(col)

for col in numeric:
    minimum=df[col].min()
    maximum=df[col].max()

    if maximum!=minimum:
        df[col]=(df[col]-minimum)/(maximum-minimum)

print(df.head())