# =============================================================================
# Lab 04 - A1 (portion for Lab03-A2: Label Encoding & One-Hot Encoding)
# AI Tool Used: Claude (Anthropic, Claude Sonnet 5) via claude.ai
# Prompt summary: "Regenerate A2.py (label encoding + one-hot encoding
#   helper functions) using library-backed implementations (sklearn /
#   pandas) instead of manual Python loops, keep the same function
#   signatures / return shape as the original so results are comparable."
# Human review: confirmed output mapping and column naming matches the
#   manual Lab03 version's behaviour.
# =============================================================================
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def label_encode(df, column):
    """
    Label-encode a single categorical column.

    Uses sklearn's LabelEncoder instead of a hand-rolled dictionary loop.

    Parameters
    ----------
    df : pandas.DataFrame
    column : str
        Name of the column to encode.

    Returns
    -------
    (list[int], dict) : encoded values (as a list, to match the Lab03
        function's return type) and the {original_value: code} mapping.
    """
    encoder = LabelEncoder()
    encoded_array = encoder.fit_transform(df[column])
    mapping = {
        original: int(code)
        for code, original in enumerate(encoder.classes_)
    }
    return encoded_array.tolist(), mapping


def one_hot_encode(df, column):
    """
    One-hot encode a single categorical column.

    Uses pandas.get_dummies instead of manually building each indicator
    column with a Python loop. Column names are kept in the same
    "<column>_<value>" format as the Lab03 version for comparability.

    Parameters
    ----------
    df : pandas.DataFrame
    column : str

    Returns
    -------
    pandas.DataFrame with one indicator column per unique value.
    """
    one_hot_df = pd.get_dummies(df[column], prefix=column).astype(int)
    return one_hot_df


if __name__ == "__main__":
    df = pd.read_excel(
        "Lab Session Data (1).xlsx",
        sheet_name="marketing_campaign",
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
