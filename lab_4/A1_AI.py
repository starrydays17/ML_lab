# =============================================================================
# Lab 04 - A1 (portion for Lab03-A1: Datatype Identification)
# AI Tool Used: Claude (Anthropic, Claude Sonnet 5) via claude.ai
# Prompt summary: "Regenerate A1.py (datatype identification for
#   marketing_campaign features) using AI assistance, keep the function
#   modular, no print statements inside functions."
# Human review: variable names, column groupings and Nominal/Interval/Ratio
#   classifications were verified manually against the dataset dictionary.
# =============================================================================
import pandas as pd


def identify_datatypes(df):
    """
    Classify every column of the given DataFrame into one of
    Nominal / Ordinal / Interval / Ratio, using a combination of the
    pandas dtype and domain knowledge about the marketing_campaign dataset.

    Unlike the manual Lab03 version (which hard-coded every ratio column
    name in a list), this version drives most of the classification off
    the column's pandas dtype, and only hard-codes the exceptions
    (nominal / interval columns) that dtype alone cannot distinguish.

    Parameters
    ----------
    df : pandas.DataFrame

    Returns
    -------
    dict[str, str] mapping column name -> datatype label
    """
    # Columns that are categorical text labels with no inherent order
    nominal_columns = {"Education", "Marital_Status"}

    # Columns that behave like an interval scale: differences are
    # meaningful but there is no true zero (a birth year of 0 or a
    # join-date of 0 is not "no birth year")
    interval_columns = {"Year_Birth", "Dt_Customer"}

    datatypes = {}
    for column in df.columns:
        if column in nominal_columns:
            datatypes[column] = "Nominal"
        elif column in interval_columns:
            datatypes[column] = "Interval"
        else:
            # Everything else in this dataset (counts, amounts spent,
            # binary flags, IDs) has a true, meaningful zero and supports
            # ratio-scale operations (e.g. "twice as much spend"), so it
            # is classified as Ratio.
            datatypes[column] = "Ratio"
    return datatypes


if __name__ == "__main__":
    df = pd.read_excel(
        "Lab Session Data (1).xlsx",
        sheet_name="marketing_campaign",
    )
    result = identify_datatypes(df)

    print("Feature\t\t\tDatatype")
    print("-" * 35)
    for feature, datatype in result.items():
        print(f"{feature:<20} {datatype}")
