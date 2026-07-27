import pandas as pd
import numpy as np

df=pd.read_excel("Lab Session Data (1).xlsx",sheet_name="thyroid0387_UCI")

df=df.replace("?",np.nan)

for col in df.columns:

    temp=pd.to_numeric(df[col],errors="coerce")

    if temp.notna().sum()>0:
        df[col]=temp.fillna(temp.mean())
    else:
        df[col]=df[col].fillna(df[col].mode()[0])

print(df.isnull().sum())