import os
import pandas as pd
import logging

# Set up logging
logging.basicConfig(
    filename='data_cleaning.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Function to clean column names
def clean_column_names(df):
    """
    Clean column names by removing '<' and '>' characters.
    Args:
        df (pd.DataFrame): DataFrame with columns to be cleaned.
    Returns:
        pd.DataFrame: DataFrame with cleaned column names.
    """
    df.columns = df.columns.str.replace('<', '').str.replace('>', '')
    logging.info(f"Cleaned column names: {df.columns}")
    return df

def load_data(data_dir):
    """
    Load all CSV files in the data directory and combine them into a single DataFrame.
    Args:
        data_dir (str): Path to the directory containing CSV files.
    Returns:
        pd.DataFrame: Combined DataFrame with all stock data.
    """
    try:
        # Get all CSV files in the directory
        all_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.csv')]
        data_frames = []

        if not all_files:
            raise ValueError("No CSV files found in the directory.")

        for file in all_files:
            # Read data from CSV with comma delimiter
            df = pd.read_csv(file, delimiter=',')  # You can change the delimiter if needed (e.g., ';')
            
            # Remove extra whitespace from column names and clean column names
            df = clean_column_names(df)

            # Log the columns in the CSV file
            logging.info(f"Columns in file {file}: {df.columns}")
            
            # Check if necessary columns are in the data
            critical_columns = ['Ticker', 'DTYYYYMMDD', 'Open', 'High', 'Low', 'Close', 'Volume']
            missing_columns = [col for col in critical_columns if col not in df.columns]
            if missing_columns:
                logging.error(f"Missing columns: {missing_columns} in file {file}")
                raise ValueError(f"Missing columns in file: {file}. Missing columns: {missing_columns}")

            data_frames.append(df)
            logging.info(f"File loaded: {file}")

        # Combine all DataFrames
        combined_df = pd.concat(data_frames, ignore_index=True)
        logging.info(f"Combined {len(data_frames)} files into a single DataFrame.")
        return combined_df

    except Exception as e:
        logging.error(f"Error while loading data: {e}")
        raise


def clean_data(df):
    """
    Clean the raw stock data.
    - Drop rows with missing critical values.
    - Fill missing numerical values with the mean.
    - Remove duplicates.
    Args:
        df (pd.DataFrame): Raw stock data.
    Returns:
        pd.DataFrame: Cleaned data.
    """
    try:
        # Check for missing values in critical columns
        critical_columns = ['Ticker', 'DTYYYYMMDD', 'Open', 'High', 'Low', 'Close', 'Volume']
        missing_critical = df[critical_columns].isnull().sum()
        logging.info(f"Missing values in critical columns: {missing_critical}")

        # Drop rows with missing values in critical columns
        df = df.dropna(subset=critical_columns)
        logging.info("Dropped rows with missing critical values.")
        
        # Fill missing numerical values with the mean
        numerical_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in numerical_cols:
            if col in df.columns:
                missing_values = df[col].isnull().sum()
                if missing_values > 0:
                    mean_value = df[col].mean()
                    df[col].fillna(mean_value, inplace=True)
                    logging.info(f"Filled {missing_values} missing values in column '{col}' with mean value: {mean_value}")
        
        # Remove duplicate rows
        initial_row_count = len(df)
        df = df.drop_duplicates()
        final_row_count = len(df)
        logging.info(f"Dropped {initial_row_count - final_row_count} duplicate rows.")

    except Exception as e:
        logging.error(f"Error during cleaning: {e}")
        raise
    
    return df

if __name__ == "__main__":
    # Path to the directory containing data
    data_dir = "C:/Users/Admin/source/repos/STOCK-MARKET.prediction/backend/data"
    
    try:
        # Load and clean data
        raw_data = load_data(data_dir)
        cleaned_data = clean_data(raw_data)
        
        # Save cleaned data
        output_file = os.path.join(data_dir, "cleaned_stock_data.csv")
        cleaned_data.to_csv(output_file, index=False)
        logging.info(f"Data cleaning completed. File saved: '{output_file}'.")
        print("Data cleaning completed. Check 'cleaned_stock_data.csv' and the log file for details.")
    
    except Exception as e:
        logging.critical(f"Unable to clean data: {e}")
        print(f"Error: {e}")
