import pandas as pd

def read_csv_with_datetime_index(file_path):
    df = pd.read_csv(file_path)
    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index("Date", inplace=True)
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")

    daily_index = pd.date_range(start=df.index.min(), end=df.index.max(), freq='D')
    df = df.reindex(daily_index)

    # Fill missing values with the last valid price using forward-fill
    df["Price"].ffill(inplace=True)

    price_series = df["Price"]

    return price_series

COAL_SERIES = read_csv_with_datetime_index(r"data\Coal (API2) CIF ARA (ARGUS-McCloskey) Futures Historical Data.csv")

GAS_SERIES = read_csv_with_datetime_index(r"data\Dutch TTF Natural Gas Futures Historical Data.csv")

EUA_SERIES = read_csv_with_datetime_index(r"data\European Union Allowance (EUA) Yearly Futures Historical Data.csv")

ES_GAS_SERIES = read_csv_with_datetime_index(r"data\Dutch TTF Natural Gas Futures Historical Data.csv")

def _gas_price(date: pd.Timestamp, area:str):
    if area == 'ES':
        return ES_GAS_SERIES[date.replace(tzinfo=None).floor('D')]
    else:
        return GAS_SERIES[date.replace(tzinfo=None).floor('D')]

def _coal_price(date: pd.Timestamp):
    return COAL_SERIES[date.replace(tzinfo=None).floor('D')]

def _eua_price(date: pd.Timestamp):
    return EUA_SERIES[date.replace(tzinfo=None).floor('D')]