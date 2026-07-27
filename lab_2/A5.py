import pandas as pd

df=pd.read_excel("Lab Session Data (1).xlsx",sheet_name="thyroid0387_UCI")

binary_columns=[
"on thyroxine",
"query on thyroxine",
"on antithyroid medication",
"sick",
"pregnant",
"thyroid surgery",
"I131 treatment",
"query hypothyroid",
"query hyperthyroid",
"lithium",
"goitre",
"tumor",
"hypopituitary",
"psych",
"TSH measured",
"T3 measured",
"TT4 measured",
"T4U measured",
"FTI measured",
"TBG measured"
]

binary=df[binary_columns].replace({"t":1,"f":0})

v1=binary.iloc[0]
v2=binary.iloc[1]

f11=0
f10=0
f01=0
f00=0

for a,b in zip(v1,v2):

    if a==1 and b==1:
        f11+=1

    elif a==1 and b==0:
        f10+=1

    elif a==0 and b==1:
        f01+=1

    else:
        f00+=1

JC=f11/(f11+f10+f01)

SMC=(f11+f00)/(f11+f10+f01+f00)

print("Jaccard Coefficient =",JC)
print("Simple Matching Coefficient =",SMC)