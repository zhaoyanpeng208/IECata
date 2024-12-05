import pandas as pd


def keep_max(data: pd.DataFrame):
    if data["Values"].min() == 0 or data["Values"].max() / data["Values"].min() > 100:
        return None
    _out = data.iloc[0, :].copy()
    _out["Values"] = data["Values"].values.max()
    return _out


def keep_min(data: pd.DataFrame):
    if data["Value"].min() == 0 or data["Value"].max() / data["Value"].min() > 100:
        return None
    _out = data.iloc[0, :].copy()
    _out["Value"] = data["Value"].values.min()
    return _out


def keep_mean(data: pd.DataFrame):
    if data["Value"].min() == 0 or data["Value"].max() / data["Value"].min() > 100:
        return None
    _out = data.iloc[0, :].copy()
    _out["Value"] = data["Value"].values.mean()
    return _out
