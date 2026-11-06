import pandas as pd


def engineer_features(data: pd.DataFrame) -> pd.DataFrame:
    """Create additional demo features from transaction inputs."""

    df = data.copy()

    df["amount_deviation"] = (
        df["amount"] - df["avg_amount"]
    )

    df["amount_ratio"] = (
        df["amount"] / df["avg_amount"].clip(lower=1)
    )

    df["is_night"] = (
        (df["hour"] >= 0) & (df["hour"] < 6)
    ).astype(int)

    df["velocity_ratio"] = (
        df["transactions_1h"] /
        df["transactions_24h"].clip(lower=1)
    )

    return df