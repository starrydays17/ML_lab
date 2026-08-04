import pandas as pd

def label_encode(df, column):
    unique_values = df[column].unique()
    mapping = {}
    for i, value in enumerate(unique_values):
        mapping[value] = i
    encoded = []
    for value in df[column]:
        encoded.append(mapping[value])
    return encoded
def one_hot_encode(df, column):
    unique_values = df[column].unique()
    one_hot_df = pd.DataFrame()
    for value in unique_values:
        temp = []
        for item in df[column]:
            if item == value:
                temp.append(1)
            else:
                temp.append(0)
        one_hot_df[column + "_" + str(value)] = temp
    return one_hot_df
df = pd.read_excel(
    "Lab Session Data (1)(2).xlsx",
    sheet_name="marketing_campaign"
)
print("Original Dataset Shape:", df.shape)
label_df = df.copy()
label_df["Education"] = label_encode(label_df, "Education")
label_df["Marital_Status"] = label_encode(label_df, "Marital_Status")
print("After Label Encoding:", label_df.shape)
onehot_df = df.copy()
education_onehot = one_hot_encode(onehot_df, "Education")
marital_onehot = one_hot_encode(onehot_df, "Marital_Status")
onehot_df = onehot_df.drop(["Education", "Marital_Status"], axis=1)
onehot_df = pd.concat(
    [onehot_df, education_onehot, marital_onehot],
    axis=1
)
print("After One-Hot Encoding:", onehot_df.shape)
print("\nFirst 5 rows of Label Encoded Dataset:")
print(label_df.head())
print("\nFirst 5 rows of One-Hot Encoded Dataset:")
print(onehot_df.head())