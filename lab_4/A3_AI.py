# =============================================================================
# Lab 04 - A1 (portion for Lab03-A3: Encoding the full dataset)
# AI Tool Used: Claude (Anthropic, Claude Sonnet 5) via claude.ai
# Prompt summary: "Regenerate A3.py: apply the A2 encoding functions to the
#   full marketing_campaign dataset and report the dimensionality change."
# Human review: fixed the filename typo present in the original A3.py
#   (it pointed at a file that doesn't exist) so the script actually runs.
# =============================================================================
import pandas as pd
from A2_AI import label_encode, one_hot_encode


def apply_label_encoding(df, columns):
    """Return a copy of df with each column in `columns` label-encoded."""
    encoded_df = df.copy()
    for column in columns:
        encoded_values, _ = label_encode(encoded_df, column)
        encoded_df[column] = encoded_values
    return encoded_df


def apply_one_hot_encoding(df, columns):
    """Return a copy of df with each column in `columns` one-hot encoded
    (original columns dropped, indicator columns appended)."""
    encoded_df = df.copy()
    one_hot_frames = [one_hot_encode(encoded_df, column) for column in columns]
    encoded_df = encoded_df.drop(columns=columns)
    encoded_df = pd.concat([encoded_df] + one_hot_frames, axis=1)
    return encoded_df


if __name__ == "__main__":
    df = pd.read_excel(
        "Lab Session Data (1).xlsx",
        sheet_name="marketing_campaign",
    )
    categorical_columns = ["Education", "Marital_Status"]

    print("Original Dataset Shape:", df.shape)

    label_df = apply_label_encoding(df, categorical_columns)
    print("After Label Encoding:", label_df.shape)

    onehot_df = apply_one_hot_encoding(df, categorical_columns)
    print("After One-Hot Encoding:", onehot_df.shape)

    print("\nFirst 5 rows of Label Encoded Dataset:")
    print(label_df.head())
    print("\nFirst 5 rows of One-Hot Encoded Dataset:")
    print(onehot_df.head())
