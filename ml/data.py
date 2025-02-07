# ml/data.py
import os
import pandas as pd
import numpy as np


def load_csv(filename: str) -> pd.DataFrame:
    return pd.read_csv(filename)

def process_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Quantitative Metrics:
    - PRICE:
        - % change in ADJUSTED closing
        - % change in 3 days
        - Ma5 & Ma20
        - Close : Ma5 & Close : Ma20
        - Ma5 - Ma20
        - Range Ratio (High - Low) / Closing
    - VOLUME:
        - % change in volume
        - Va5 & Va20 [ volume moving avg ]
        - Volume : Va5 & Volume : Va20
        - Lagged Volume(3 days)
    - VOLATILITY:
        - 5 day Standard Deviation
        - 14 day ATR
        - 14 day RSI
        - 10 day momentum
    - TARGET:
        - 1 if Close tmrw > Close today (profit)
        - 0 if Close tmrw = Close today (Risk Tolerance)
        - -1 if Close tmrw < Close today
    """
    df = df.sort_values('Date').reset_index(drop=True)

    # % change in closing price
    df['Pct_Change_Close'] = df['Adj Close'].pct_change(fill_method=None) * 100

    # % change in closing price over 3 days
    df['Pct_Change_3d'] = df['Adj Close'].pct_change(periods=3, fill_method=None) * 100

    # Moving averages for Close: 5-day and 20-day
    df['Ma5'] = df['Adj Close'].rolling(window=5).mean()
    df['Ma20'] = df['Adj Close'].rolling(window=20).mean()

    # Ratios: Close divided by moving averages
    df['Close_to_Ma5'] = df['Adj Close'] / df['Ma5']
    df['Close_to_Ma20'] = df['Adj Close'] / df['Ma20']

    # Difference between Ma5 and Ma20
    df['Ma5_minus_Ma20'] = df['Ma5'] - df['Ma20']

    # Range Ratio: (High - Low) / Close
    df['Range_Ratio'] = (df['High'] - df['Low']) / df['Adj Close']

    # -----------------------
    # VOLUME METRICS
    # -----------------------
    # % change in Volume
    df['Pct_Change_Volume'] = df['Volume'].pct_change(fill_method=None) * 100

    # Moving averages for Volume: 5-day and 20-day
    df['Va5'] = df['Volume'].rolling(window=5).mean()
    df['Va20'] = df['Volume'].rolling(window=20).mean()

    # Ratios: Volume divided by its moving averages
    df['Volume_to_Va5'] = df['Volume'] / df['Va5']
    df['Volume_to_Va20'] = df['Volume'] / df['Va20']

    # Lagged Volume (3 days ago)
    df['Lagged_Volume_3d'] = df['Volume'].shift(3)

    # -----------------------
    # VOLATILITY METRICS
    # -----------------------
    # 5-day Standard Deviation of closing price
    df['Std_5d'] = df['Adj Close'].rolling(window=5).std()

    # ATR (Average True Range) over 14 days
    # True Range (TR) calculation:
    df['Previous_Close'] = df['Adj Close'].shift(1)
    df['TR'] = df.apply(
        lambda row: max(
            row['High'] - row['Low'],
            abs(row['High'] - row['Previous_Close']) if pd.notnull(row['Previous_Close']) else 0,
            abs(row['Low'] - row['Previous_Close']) if pd.notnull(row['Previous_Close']) else 0
        ),
        axis=1
    )
    df['ATR_14'] = df['TR'].rolling(window=14).mean()

    # RSI (Relative Strength Index) over 14 days
    delta = df['Adj Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss
    df['RSI_14'] = 100 - (100 / (1 + rs))

    # 10-day momentum: difference between today's close and close 10 days ago
    df['Momentum_10'] = df['Adj Close'] - df['Adj Close'].shift(10)

    # --------
    # TARGET VARIABLE(y function)
    # --------
    df['Target'] = np.sign(df['Adj Close'].shift(-1) - df['Adj Close'])
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    return df

def load_and_process(filename: str) -> pd.DataFrame:
    df = load_csv(filename)
    return process_data(df)
