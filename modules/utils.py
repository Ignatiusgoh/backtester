import pandas as pd
from dateutil.relativedelta import relativedelta

def walk_forward_split(df: pd.DataFrame, train_months: int, test_months: int):
    """
    Splits a DataFrame into multiple (train, test) sets using a walk-forward method.

    Parameters:
        df (pd.DataFrame): Indexed by datetime, sorted.
        train_months (int): Number of months for training period.
        test_months (int): Number of months for testing period.

    Returns:
        List of tuples: [(train_df, test_df), ...]
    """
    splits = []
    start = df.index.min()
    end = df.index.max()

    while start + relativedelta(months=train_months + test_months) <= end:
        train_end = start + relativedelta(months=train_months)
        test_end = train_end + relativedelta(months=test_months)

        train_df = df[(df.index >= start) & (df.index < train_end)].copy()
        test_df = df[(df.index >= train_end) & (df.index < test_end)].copy()

        if not train_df.empty and not test_df.empty:
            splits.append((train_df, test_df))

        # Advance by the test period for the next split
        start += relativedelta(months=test_months)

    return splits
