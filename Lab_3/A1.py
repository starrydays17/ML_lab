import pandas as pd

def identify_datatypes(df):
    datatypes = {}
    datatypes["Education"] = "Nominal"
    datatypes["Marital_Status"] = "Nominal"
    datatypes["Year_Birth"] = "Interval"
    datatypes["Dt_Customer"] = "Interval"
    ratio_columns = [
        "ID",
        "Income",
        "Kidhome",
        "Teenhome",
        "Recency",
        "MntWines",
        "MntFruits",
        "MntMeatProducts",
        "MntFishProducts",
        "MntSweetProducts",
        "MntGoldProds",
        "NumDealsPurchases",
        "NumWebPurchases",
        "NumCatalogPurchases",
        "NumStorePurchases",
        "NumWebVisitsMonth",
        "AcceptedCmp3",
        "AcceptedCmp4",
        "AcceptedCmp5",
        "AcceptedCmp1",
        "AcceptedCmp2",
        "Complain",
        "Z_CostContact",
        "Z_Revenue",
        "Response"
    ]
    for column in ratio_columns:
        datatypes[column] = "Ratio"
    return datatypes

df = pd.read_excel(
    "Lab Session Data (1).xlsx",
    sheet_name="marketing_campaign"
)
result = identify_datatypes(df)
print("Feature\t\t\tDatatype")
print("-" * 35)
for feature, datatype in result.items():
    print(f"{feature:<20} {datatype}")