import os
import sys
import datetime
import joblib
import pandas as pd
import yfinance as yf

import ml
import ml.data as data  # for process_data()
import ml.trainer as trainer
import datetime

# The list of sectors (each should have a trained model in Models/)
STOCK_SECTORS = [
    "Basic Materials",
    "Communication Services",
    "Consumer Cyclical",
    "Consumer Defensive",
    "Energy",
    "Financial Services",
    "Healthcare",
    "Industrials",
    "Real Estate",
    "Technology",
    "Utilities",
    "Unknown"  # fallback for missing sector info
]
FEATURE_COLS = [
    'Pct_Change_Close',
    'Pct_Change_3d',
    'Ma5',
    'Ma20',
    'Close_to_Ma5',
    'Close_to_Ma20',
    'Ma5_minus_Ma20',
    'Range_Ratio',
    'Pct_Change_Volume',
    'Va5',
    'Va20',
    'Lagged_Volume_3d',
    'Std_5d',
    'Previous_Close',
    'TR',
    'ATR_14',
    'RSI_14',
    'Momentum_10'
]




# -------------------
# REAL TIME FUNCTIONS
# -------------------
def fetch_adjusted_df(ticker, start, end):
    df = yf.download(ticker, start=start, end=end, auto_adjust=True)
    if df.empty:
        print("[WARNING] No data found for ", ticker)
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)

    df.reset_index(inplace=True)
    # Rename close to adj close
    df.rename(columns={"Close": "Adj Close"}, inplace=True)
    return df

def realSignal(ticker, date):
    """
    Given a ticker and a date (string in 'YYYY-MM-DD' format), this function:
      1. Downloads ~60 days of historical data ending on the given date.
      2. Processes the data (computing moving averages, % changes, etc.) using ml.data.process_data().
      3. Selects the row corresponding to the given date (or uses the most recent row if not found).
      4. Determines the ticker’s sector
      5. Loads the trained model and scaler for that sector from the Models/ folder.
      6. Scales the features and returns the model’s prediction.
    """
    return -1
# ---------------------
# BACKTESTING FUNCTIONS
# ----------------------

def signal(ticker, sector, df: pd.DataFrame, date_str):
    row = df[df["Date"] == date_str]
    if row.empty: # If empty row, use yesterday (Shouldn't happen though)
        row = df.iloc[-1]
    X = row[FEATURE_COLS]
    if sector not in STOCK_SECTORS:
        sector = "Unknown"
    model_path = os.path.join("Models", f"{sector}.joblib")
    scaler_path = os.path.join("Models", f"{sector}_scaler.joblib")
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        print(f"[ERROR] Model or scaler for sector '{sector}' not found.")
        return 0
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    X_scaled = scaler.transform(X)
    pred = model.predict(X_scaled)[0]
    return pred




def simulate(ticker, start, end):
    """
    Simulates day-by-day trades for a given ticker from a given interval

    Trading Rules:
      1) If already holding shares:
         a) If the model prediction is +1 (positive), buy additional shares equal to:
                current_shares * (daily_pct_gain / 100)
            (e.g. a 20% gain means buying 20% more shares)
         b) If the prediction is -1, sell all shares.
      2) If not holding shares:
         a) If the prediction is +1, buy $1000 worth of shares.
         b) Otherwise, do nothing.
      3) If there isn’t enough cash to cover a purchase, “inject” the extra cash and add it to expenses.

    At the end of the simulation any remaining shares are sold.
    The function returns the profit percentage calculated as:
         ((final cash - total expenses) / total expenses) * 100.
    """
    # Download historical daily data from the start date.
    df = fetch_adjusted_df(ticker, start, end)
    if df.empty:
        return None
    df = data.process_data(df)
    sector = yf.Ticker(ticker).info["sector"]

    df['Date'] = df['Date'].dt.strftime("%Y-%m-%d")

    # Portfolio variables:
    holding = 0.0  # number of shares currently held
    cash = 0.0  # cash balance (from sales)
    expenses = 0.0  # total money injected (i.e. cost basis)

    last_price = None  # will hold the previous day's price for computing daily change

    for i, row in df.iterrows():
        date = row['Date']
        price = row['Adj Close']
        if last_price is None:
            last_price = price

        # Compute daily percent gain
        daily_pct_gain = ((price - last_price) / last_price) * 100 if last_price != 0 else 0

        # Get the model’s prediction signal for this day.
        pred = signal(ticker, sector, df, date)

        if holding > 0:
            if pred >= 0:
                # When holding and the prediction is positive, buy additional shares
                if daily_pct_gain > 0:
                    additional_shares = holding * (daily_pct_gain / 100)
                    cost = additional_shares * price
                    # If available cash is insufficient, inject additional funds.
                    if cash < cost:
                        needed = cost - cash
                        cash += needed
                        expenses += needed
                    cash -= cost  # pay for the additional shares
                    holding += additional_shares
            else:
                # If prediction is neutral or negative, sell all shares.
                proceeds = holding * price
                cash += proceeds
                holding = 0
        else:
            if pred >= 0:
                # If not holding any shares and the prediction is positive, buy $1000 worth.
                cost = 1000
                if cash < cost:
                    needed = cost - cash
                    cash += needed
                    expenses += needed
                cash -= cost
                shares_bought = cost / price
                holding += shares_bought

        last_price = price

    # At the simulation’s end, sell any remaining shares.
    if holding > 0:
        final_price = df.iloc[-1]['Adj Close']
        proceeds = holding * final_price
        cash += proceeds
        holding = 0

    # Compute the profit percentage.
    profit_pct = ((cash - expenses) / expenses * 100) if expenses > 0 else 0
    return profit_pct

def backTest(tickers, start,end):
    profit_pcts = {}
    for ticker in tickers:
        profit_pcts[ticker] = simulate(ticker, start, end)
    return profit_pcts

if __name__ == "__main__":
    tickers = [
        "MMM",  # 3M
        "AXP",  # American Express
        "AMGN",  # Amgen
        "AMZN",  # Amazon
        "AAPL",  # Apple
        "BA",  # Boeing
        "CAT",  # Caterpillar
        "CVX",  # Chevron
        "CSCO",  # Cisco Systems
        "KO",  # Coca‑Cola
        "DIS",  # Disney
        "GS",  # Goldman Sachs
        "HD",  # Home Depot
        "HON",  # Honeywell
        "IBM",  # IBM
        "JNJ",  # Johnson & Johnson
        "JPM",  # JPMorgan Chase
        "MCD",  # McDonald's
        "MRK",  # Merck
        "MSFT",  # Microsoft
        "NKE",  # Nike
        "NVDA",  # Nvidia
        "PG",  # Procter & Gamble
        "CRM",  # Salesforce
        "SHW",  # Sherwin-Williams
        "TRV",  # Travelers Companies
        "UNH",  # UnitedHealth Group
        "VZ",  # Verizon
        "V",  # Visa
        "WMT",  # Walmart
    ]
    results = backTest(tickers, datetime.datetime(2023, 1, 1), datetime.datetime(2023, 12, 31))
    for ticker, profit in results.items():
        print(f"{ticker}: {profit:.2f}")