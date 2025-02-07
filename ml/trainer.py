# ml/trainer.py
import joblib
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import yfinance as yf
from yfinance.domain import sector

from ml.data import load_and_process

# ------------------
# Global Definitions
# ------------------

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

RAW_DIR = "data/raw"
DF_DIR = "data/DF"
MODELS_DIR = "Models"
TICKER_INFO_CACHE_PATH = "data/ticker_info_cache.joblib"


# ------------------
# Helper: Ticker Caching [ So we don't overload YFinance everytime we retrain the model
# ------------------
def load_ticker_info_cache():
    """
    Loads a dictionary from ticker -> sector from joblib cache if it exists,
    otherwise returns an empty dict.
    """
    if os.path.exists(TICKER_INFO_CACHE_PATH):
        return joblib.load(TICKER_INFO_CACHE_PATH)
    else:
        return {}


def save_ticker_info_cache(cache_dict):
    """
    Saves the ticker -> sector dictionary to joblib.
    """
    joblib.dump(cache_dict, TICKER_INFO_CACHE_PATH)


def get_ticker_sector(ticker, ticker_cache):
    """
       Fetches the sector for a given ticker.
       If the ticker is in the cache, return immediately.
       Otherwise, attempt a yfinance call, handle exceptions,
       and store the result (or "Unknown") in the cache.
       """
    if ticker in ticker_cache:
        return ticker_cache[ticker]
    sector = "Unknown"
    try:
        # Attempt yfinance call
        sector_candidate = yf.Ticker(ticker).info.get("sector")
        if sector_candidate is not None:
            sector = sector_candidate
    except Exception as e:
        # If there's a 404 or any other error, fallback to "Unknown"
        if "404" in str(e):
            print(f"[WARNING] 404 error for ticker '{ticker}'. Marking sector as 'Unknown'.")
        else:
            print(f"[WARNING] Failed to fetch sector for ticker '{ticker}': {e}")
            quit(1)
    ticker_cache[ticker] = sector
    save_ticker_info_cache(ticker_cache)
    return sector


# ------------------
# Function 1: Process DataFrames
# ------------------

def process_dataframes():
    """
    Reads all CSV files from RAW_DIR, determines each ticker's sector,
    loads/cleans the data
    then groups data by sector and saves each sector to DF_DIR.
    """
    print("[INFO] Starting data processing step...")
    ticker_cache = load_ticker_info_cache()
    sector_frames = {sector: [] for sector in STOCK_SECTORS}

    for file in os.listdir(RAW_DIR):
        if file.endswith(".csv"):
            ticker = os.path.splitext(file)[0]
            filepath = os.path.join(RAW_DIR, file)
            print(f"[INFO] Processing {ticker} from {file}...")
            sector = get_ticker_sector(ticker, ticker_cache)
            if sector not in sector_frames:
                sector = "Unknown"
            tickerFrame = load_and_process(filepath)
            tickerFrame['Ticker'] = ticker
            sector_frames[sector].append(tickerFrame)

    # combine & save sectors:
    for sector, frames in sector_frames.items():
        df_sector = pd.concat(frames, ignore_index=True)
        df_sector.sort_values(by=['Ticker', 'Date'], inplace=True)
        filename = f"{sector}.joblib"
        save_path = os.path.join(DF_DIR, filename)
        joblib.dump(df_sector, save_path)
        print(f"[INFO] Saved {filename} to {save_path}.")
    print("[INFO] Finished data processing step.")


# ------------------
# Function 2: Train Model
# ------------------

def train_model(sector):
    df_path = os.path.join(DF_DIR, f"{sector}.joblib")
    print(f"[INFO] Loading {df_path}...")
    df_sector = joblib.load(df_path)
    print(f"[INFO] Finished loading {df_path}.")

    # Train/Test Split + Scaling(Note: Consider Log scaling)
    X = df_sector[FEATURE_COLS]
    y = df_sector['Target']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    '''
    My Little Pony: Classifier is magic!
    [ Define the MLPClassifier model ]
    [ Adjust Layer Sizes & Other Hyperparameters ] 
    '''
    myLittlePony = MLPClassifier(
        # 100 neurons for a broad set of features, 50 to refine them into abstract representation.
        hidden_layer_sizes=(100, 50),
        # Computationally simple + mitigates the vanishing gradient problem
        activation="relu",
        solver="adam",
        # Can't be too restrictive with something as volatile as stocks but still need to prevent overfitting
        alpha=0.0001,
        learning_rate='adaptive',
        max_iter=500,
        random_state=42
    )

    # Train the model:
    myLittlePony.fit(X_train_scaled, y_train)
    print(f"[INFO]My Little {sector} has graduated from Celestia High in {myLittlePony.n_iter_} iterations!")
    
    # Test eval:
    y_pred = myLittlePony.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"[RESULT] {sector}: Accuracy: {accuracy:.3f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # Save the trained model and the scaler for this sector.
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, f"{sector}.joblib")
    scaler_path = os.path.join(MODELS_DIR, f"{sector}_scaler.joblib")
    joblib.dump(myLittlePony, model_path)
    joblib.dump(scaler, scaler_path)

    print(f"[INFO] Saved MLP model for '{sector}' to '{model_path}'")
    print(f"[INFO] Saved scaler for '{sector}' to '{scaler_path}'")


def main():
    processed_sectors = []
    trained_sectors = []
    already_reprocessed = False  # Until I eventually store the data better

    for sector in STOCK_SECTORS:
        model_path = os.path.join(MODELS_DIR, f"{sector}.joblib")
        df_path = os.path.join(DF_DIR, f"{sector}.joblib")

        # Skip if the model already exists
        if os.path.exists(model_path):
            continue

        # If the processed dataframe does not exist, reprocess the data once.
        if not os.path.exists(df_path):
            if not already_reprocessed:
                process_dataframes()
                already_reprocessed = True
            else:
                print(f"[WARNING] Dataframe for sector '{sector}' is missing, even after reprocessing.")
                continue  # Optionally, skip training if the data is missing

        # Train the model for this sector
        train_model(sector)
        trained_sectors.append(sector)

    output = []
    if processed_sectors:
        output.append(processed_sectors)
    if trained_sectors:
        output.append(trained_sectors)
    print(f"[INFO] {output}")


if __name__ == "__main__":
    main()
