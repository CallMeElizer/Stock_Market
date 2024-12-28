import os
import pandas as pd
import logging

# Thiết lập ghi nhật ký
logging.basicConfig(
    filename='data_processing.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def load_data(data_dir):
    """
    Load all CSV files in the data directory and combine into a single DataFrame.
    Args:
        data_dir (str): Path to the directory containing CSV files.
    Returns:
        pd.DataFrame: Combined DataFrame with all stock data.
    """
    all_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.csv')]
    data_frames = []
    
    for file in all_files:
        try:
            df = pd.read_csv(file)
            # Kiểm tra cột bắt buộc
            required_columns = {'Ticker', 'DTYYYYMMDD', 'Open', 'High', 'Low', 'Close', 'Volume'}
            if not required_columns.issubset(df.columns):
                logging.warning(f"Skipping file {file}: Missing required columns {required_columns - set(df.columns)}.")
                continue
            if df.empty:
                logging.warning(f"Skipping file {file}: File is empty.")
                continue
            data_frames.append(df)
            logging.info(f"Loaded file: {file}")
        except Exception as e:
            logging.error(f"Error loading file {file}: {e}")
    
    if not data_frames:
        raise ValueError("No valid CSV files found in the directory.")
    
    combined_df = pd.concat(data_frames, ignore_index=True)
    logging.info(f"Combined {len(data_frames)} files into a single DataFrame.")
    return combined_df

def calculate_rsi(series, window=14):
    """
    Calculate Relative Strength Index (RSI) for a given series.
    Args:
        series (pd.Series): Series of prices.
        window (int): Lookback period for RSI.
    Returns:
        pd.Series: RSI values.
    """
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(series, span_short=12, span_long=26, signal=9):
    """
    Calculate MACD (Moving Average Convergence Divergence) and Signal Line.
    Args:
        series (pd.Series): Series of prices.
        span_short (int): Short-term EMA span.
        span_long (int): Long-term EMA span.
        signal (int): Signal line EMA span.
    Returns:
        pd.DataFrame: DataFrame containing MACD and Signal Line.
    """
    ema_short = series.ewm(span=span_short, adjust=False).mean()
    ema_long = series.ewm(span=span_long, adjust=False).mean()
    macd = ema_short - ema_long
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

def calculate_stochastic_oscillator(df, k_period=14, d_period=3):
    """
    Calculate Stochastic Oscillator (K% and D%).
    Args:
        df (pd.DataFrame): DataFrame with High, Low, Close columns.
        k_period (int): Lookback period for %K.
        d_period (int): Moving average period for %D.
    Returns:
        pd.DataFrame: DataFrame containing %K and %D.
    """
    low_min = df['Low'].rolling(window=k_period).min()
    high_max = df['High'].rolling(window=k_period).max()
    k_percent = 100 * (df['Close'] - low_min) / (high_max - low_min)
    d_percent = k_percent.rolling(window=d_period).mean()
    return k_percent, d_percent

def preprocess_data(df):
    """
    Preprocess the stock data.
    - Convert date to datetime.
    - Sort by Ticker and Date.
    - Add technical indicators (e.g., SMA, EMA, Bollinger Bands, RSI, MACD, Stochastic Oscillator).
    Args:
        df (pd.DataFrame): Raw stock data.
    Returns:
        pd.DataFrame: Preprocessed data with additional features.
    """
    try:
        # Convert date column to datetime
        df['Date'] = pd.to_datetime(df['DTYYYYMMDD'], format='%Y%m%d')
        df = df.drop(columns=['DTYYYYMMDD'])
        logging.info("Converted 'DTYYYYMMDD' to datetime and dropped the column.")
        
        # Sort data by Ticker and Date
        df = df.sort_values(by=['Ticker', 'Date'])
        logging.info("Sorted data by 'Ticker' and 'Date'.")
        
        # Add moving averages
        df['SMA_10'] = df.groupby('Ticker')['Close'].transform(lambda x: x.rolling(window=10).mean())
        df['EMA_10'] = df.groupby('Ticker')['Close'].transform(lambda x: x.ewm(span=10, adjust=False).mean())
        
        # Add Bollinger Bands
        df['BB_Middle'] = df['SMA_10']
        df['BB_Upper'] = df['BB_Middle'] + 2 * df.groupby('Ticker')['Close'].transform(lambda x: x.rolling(window=10).std())
        df['BB_Lower'] = df['BB_Middle'] - 2 * df.groupby('Ticker')['Close'].transform(lambda x: x.rolling(window=10).std())
        
        # Add daily return
        df['Daily_Return'] = df.groupby('Ticker')['Close'].pct_change()
        
        # Add RSI
        df['RSI_14'] = df.groupby('Ticker')['Close'].transform(lambda x: calculate_rsi(x, window=14))
        
        # Add MACD
        df['MACD'], df['Signal_Line'] = zip(*df.groupby('Ticker')['Close'].apply(lambda x: calculate_macd(x)))
        
        # Add Stochastic Oscillator
        df['%K'], df['%D'] = zip(*df.groupby('Ticker').apply(lambda group: calculate_stochastic_oscillator(group)))
        
        logging.info("Added technical indicators: SMA, EMA, Bollinger Bands, Daily Return, RSI, MACD, Stochastic Oscillator.")
        
    except Exception as e:
        logging.error(f"Error during preprocessing: {e}")
        raise
    
    return df

if __name__ == "__main__":
    # Path to data directory
    data_dir = "C:/Users/Admin/source/repos/STOCK-MARKET.prediction/backend/data"
    try:
        # Load and preprocess data
        raw_data = load_data(data_dir)
        processed_data = preprocess_data(raw_data)
        
        # Save the processed data
        output_file = os.path.join(data_dir, "processed_stock_data.csv")
        processed_data.to_csv(output_file, index=False)
        logging.info(f"Data processing complete. Processed file saved as '{output_file}'.")
        print("Data processing complete. Check 'processed_stock_data.csv' and logs for details.")
    except Exception as e:
        logging.critical(f"Failed to process data: {e}")
        print(f"Error: {e}")
