import os
import joblib
import pandas as pd
import yfinance as yf

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
sectors = joblib.load('data/ticker_info_cache.joblib')


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
    start = date - datetime.timedelta(days=60)
    df = fetch_adjusted_df(ticker, start, start)
    df = data.process_data(df)
    sector = trainer.get_ticker_sector(ticker, sectors)
    df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
    return signal(ticker, sector, df, date)


# ---------------------
# BACKTESTING FUNCTIONS
# ----------------------

def signal(ticker, sector, df: pd.DataFrame, date_str):
    row = df[df["Date"] == date_str]
    if row.empty:  # If empty row, use yesterday (Shouldn't happen though)
        row = df.iloc[[-1]]
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
         a) If the model prediction is positive, buy additional shares equal to:
                current_shares * (daily_pct_gain / 100)
            (e.g. a 20% gain means buying 20% more shares)
         b) If the prediction is negative, sell all shares.
      2) If not holding shares:
         a) If the prediction is positve, buy $1000 worth of shares.
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
    sector = trainer.get_ticker_sector(ticker, sectors)

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
    return [expenses, cash, profit_pct]


def backTest(tickers, start, end):
    profit_pcts = {}
    for ticker in tickers:
        profit_pcts[ticker] = simulate(ticker, start, end)
    return profit_pcts


if __name__ == "__main__":
    tickers = [
        "MMM",
        "AOS",
        "ABT",
        "ABBV",
        "ACN",
        "ADBE",
        "AMD",
        "AES",
        "AFL",
        "A",
        "APD",
        "ABNB",
        "AKAM",
        "ALB",
        "ARE",
        "ALGN",
        "ALLE",
        "LNT",
        "ALL",
        "GOOGL",
        "GOOG",
        "MO",
        "AMZN",
        "AMCR",
        "AEE",
        "AEP",
        "AXP",
        "AIG",
        "AMT",
        "AWK",
        "AMP",
        "AME",
        "AMGN",
        "APH",
        "ADI",
        "ANSS",
        "AON",
        "APA",
        "APO",
        "AAPL",
        "AMAT",
        "APTV",
        "ACGL",
        "ADM",
        "ANET",
        "AJG",
        "AIZ",
        "T",
        "ATO",
        "ADSK",
        "ADP",
        "AZO",
        "AVB",
        "AVY",
        "AXON",
        "BKR",
        "BALL",
        "BAC",
        "BAX",
        "BDX",
        "BRKB",
        "BBY",
        "TECH",
        "BIIB",
        "BLK",
        "BX",
        "BK",
        "BA",
        "BKNG",
        "BWA",
        "BSX",
        "BMY",
        "AVGO",
        "BR",
        "BRO",
        "BFB",
        "BLDR",
        "BG",
        "BXP",
        "CHRW",
        "CDNS",
        "CZR",
        "CPT",
        "CPB",
        "COF",
        "CAH",
        "KMX",
        "CCL",
        "CARR",
        "CAT",
        "CBOE",
        "CBRE",
        "CDW",
        "CE",
        "COR",
        "CNC",
        "CNP",
        "CF",
        "CRL",
        "SCHW",
        "CHTR",
        "CVX",
        "CMG",
        "CB",
        "CHD",
        "CI",
        "CINF",
        "CTAS",
        "CSCO",
        "C",
        "CFG",
        "CLX",
        "CME",
        "CMS",
        "KO",
        "CTSH",
        "CL",
        "CMCSA",
        "CAG",
        "COP",
        "ED",
        "STZ",
        "CEG",
        "COO",
        "CPRT",
        "GLW",
        "CPAY",
        "CTVA",
        "CSGP",
        "COST",
        "CTRA",
        "CRWD",
        "CCI",
        "CSX",
        "CMI",
        "CVS",
        "DHR",
        "DRI",
        "DVA",
        "DAY",
        "DECK",
        "DE",
        "DELL",
        "DAL",
        "DVN",
        "DXCM",
        "FANG",
        "DLR",
        "DFS",
        "DG",
        "DLTR",
        "D",
        "DPZ",
        "DOV",
        "DOW",
        "DHI",
        "DTE",
        "DUK",
        "DD",
        "EMN",
        "ETN",
        "EBAY",
        "ECL",
        "EIX",
        "EW",
        "EA",
        "ELV",
        "EMR",
        "ENPH",
        "ETR",
        "EOG",
        "EPAM",
        "EQT",
        "EFX",
        "EQIX",
        "EQR",
        "ERIE",
        "ESS",
        "EL",
        "EG",
        "EVRG",
        "ES",
        "EXC",
        "EXPE",
        "EXPD",
        "EXR",
        "XOM",
        "FFIV",
        "FDS",
        "FICO",
        "FAST",
        "FRT",
        "FDX",
        "FIS",
        "FITB",
        "FSLR",
        "FE",
        "FI",
        "FMC",
        "F",
        "FTNT",
        "FTV",
        "FOXA",
        "FOX",
        "BEN",
        "FCX",
        "GRMN",
        "IT",
        "GE",
        "GEHC",
        "GEV",
        "GEN",
        "GNRC",
        "GD",
        "GIS",
        "GM",
        "GPC",
        "GILD",
        "GPN",
        "GL",
        "GDDY",
        "GS",
        "HAL",
        "HIG",
        "HAS",
        "HCA",
        "DOC",
        "HSIC",
        "HSY",
        "HES",
        "HPE",
        "HLT",
        "HOLX",
        "HD",
        "HON",
        "HRL",
        "HST",
        "HWM",
        "HPQ",
        "HUBB",
        "HUM",
        "HBAN",
        "HII",
        "IBM",
        "IEX",
        "IDXX",
        "ITW",
        "INCY",
        "IR",
        "PODD",
        "INTC",
        "ICE",
        "IFF",
        "IP",
        "IPG",
        "INTU",
        "ISRG",
        "IVZ",
        "INVH",
        "IQV",
        "IRM",
        "JBHT",
        "JBL",
        "JKHY",
        "J",
        "JNJ",
        "JCI",
        "JPM",
        "JNPR",
        "K",
        "KVUE",
        "KDP",
        "KEY",
        "KEYS",
        "KMB",
        "KIM",
        "KMI",
        "KKR",
        "KLAC",
        "KHC",
        "KR",
        "LHX",
        "LH",
        "LRCX",
        "LW",
        "LVS",
        "LDOS",
        "LEN",
        "LII",
        "LLY",
        "LIN",
        "LYV",
        "LKQ",
        "LMT",
        "L",
        "LOW",
        "LULU",
        "LYB",
        "MTB",
        "MPC",
        "MKTX",
        "MAR",
        "MMC",
        "MLM",
        "MAS",
        "MA",
        "MTCH",
        "MKC",
        "MCD",
        "MCK",
        "MDT",
        "MRK",
        "META",
        "MET",
        "MTD",
        "MGM",
        "MCHP",
        "MU",
        "MSFT",
        "MAA",
        "MRNA",
        "MHK",
        "MOH",
        "TAP",
        "MDLZ",
        "MPWR",
        "MNST",
        "MCO",
        "MS",
        "MOS",
        "MSI",
        "MSCI",
        "NDAQ",
        "NTAP",
        "NFLX",
        "NEM",
        "NWSA",
        "NWS",
        "NEE",
        "NKE",
        "NI",
        "NDSN",
        "NSC",
        "NTRS",
        "NOC",
        "NCLH",
        "NRG",
        "NUE",
        "NVDA",
        "NVR",
        "NXPI",
        "ORLY",
        "OXY",
        "ODFL",
        "OMC",
        "ON",
        "OKE",
        "ORCL",
        "OTIS",
        "PCAR",
        "PKG",
        "PLTR",
        "PANW",
        "PARA",
        "PH",
        "PAYX",
        "PAYC",
        "PYPL",
        "PNR",
        "PEP",
        "PFE",
        "PCG",
        "PM",
        "PSX",
        "PNW",
        "PNC",
        "POOL",
        "PPG",
        "PPL",
        "PFG",
        "PG",
        "PGR",
        "PLD",
        "PRU",
        "PEG",
        "PTC",
        "PSA",
        "PHM",
        "PWR",
        "QCOM",
        "DGX",
        "RL",
        "RJF",
        "RTX",
        "O",
        "REG",
        "REGN",
        "RF",
        "RSG",
        "RMD",
        "RVTY",
        "ROK",
        "ROL",
        "ROP",
        "ROST",
        "RCL",
        "SPGI",
        "CRM",
        "SBAC",
        "SLB",
        "STX",
        "SRE",
        "NOW",
        "SHW",
        "SPG",
        "SWKS",
        "SJM",
        "SW",
        "SNA",
        "SOLV",
        "SO",
        "LUV",
        "SWK",
        "SBUX",
        "STT",
        "STLD",
        "STE",
        "SYK",
        "SMCI",
        "SYF",
        "SNPS",
        "SYY",
        "TMUS",
        "TROW",
        "TTWO",
        "TPR",
        "TRGP",
        "TGT",
        "TEL",
        "TDY",
        "TFX",
        "TER",
        "TSLA",
        "TXN",
        "TPL",
        "TXT",
        "TMO",
        "TJX",
        "TSCO",
        "TT",
        "TDG",
        "TRV",
        "TRMB",
        "TFC",
        "TYL",
        "TSN",
        "USB",
        "UBER",
        "UDR",
        "ULTA",
        "UNP",
        "UAL",
        "UPS",
        "URI",
        "UNH",
        "UHS",
        "VLO",
        "VTR",
        "VLTO",
        "VRSN",
        "VRSK",
        "VZ",
        "VRTX",
        "VTRS",
        "VICI",
        "V",
        "VST",
        "VMC",
        "WRB",
        "GWW",
        "WAB",
        "WBA",
        "WMT",
        "DIS",
        "WBD",
        "WM",
        "WAT",
        "WEC",
        "WFC",
        "WELL",
        "WST",
        "WDC",
        "WY",
        "WMB",
        "WTW",
        "WDAY",
        "WYNN",
        "XEL",
        "XYL",
        "YUM",
        "ZBRA",
        "ZBH",
        "ZTS"
    ]

    results = backTest(tickers, datetime.datetime(2024, 1, 1), datetime.datetime(2024, 12, 31))
    # Print header of the Markdown table
    print("| **Ticker** | **Profit pct** | Cash | Expenses |")
    print("| --- | --- | --- | --- |")
    expense = 0
    cash = 0
    # Loop over each ticker and print its data in a table row.
    for ticker, result in results.items():
        if result is None:
            continue
        expenses, cash, profit_pct = result
        print("|", ticker, "|", round(profit_pct, 2), "% |", round(cash, 2), "|", round(expenses, 2), "|")
        expense += round(expenses, 2)
        cash += round(cash, 2)
    print("**Profit:**", round(cash / expense, 2), "%")
