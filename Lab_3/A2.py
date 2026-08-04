import pandas as pd

def label_encode(df, column):
    unique_values = df[column].unique()
    mapping = {}
    for i, value in enumerate(unique_values):
        mapping[value] = i
    encoded = []
    for value in df[column]:
        encoded.append(mapping[value])
    return encoded, mapping
def one_hot_encode(df, column):
    unique_values = df[column].unique()
    one_hot_df = pd.DataFrame()
    for value in unique_values:
        new_column = []
        for item in df[column]:
            if item == value:
                new_column.append(1)
            else:
                new_column.append(0)

        one_hot_df[column + "_" + str(value)] = new_column
    return one_hot_df

df = pd.read_excel(
    "Lab Session Data (1).xlsx",
    sheet_name="marketing_campaign"
)
encoded_values, mapping = label_encode(df, "Education")
df["Education_Label"] = encoded_values
print("Label Encoding Mapping:")
print(mapping)
print("\nFirst 5 Rows:")
print(df[["Education", "Education_Label"]].head())
one_hot = one_hot_encode(df, "Marital_Status")
print("\nOne Hot Encoded Data:")
print(one_hot.head())