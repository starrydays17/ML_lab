import pandas as pd
import numpy as np

df=pd.read_excel("Lab Session Data (1).xlsx",sheet_name="thyroid0387_UCI")

print("Data Types")
print(df.dtypes)

print("\nMissing Values")
print(df.replace("?",np.nan).isnull().sum())

print("\nNumeric Columns")

for col in df.columns:
    temp=pd.to_numeric(df[col],errors="coerce")
    if temp.notna().sum()>0:
        print(col)
        print("Min:",temp.min())
        print("Max:",temp.max())
        print("Mean:",temp.mean())
        print("Variance:",temp.var())

        Q1=temp.quantile(0.25)
        Q3=temp.quantile(0.75)
        IQR=Q3-Q1

        out=temp[(temp<(Q1-1.5*IQR)) | (temp>(Q3+1.5*IQR))]
        print("Outliers:",len(out))
        print()