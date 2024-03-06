import ta
import yfinance as yf
import pandas as pd
from ta.trend import SMAIndicator, MACD, EMAIndicator
from ta.momentum import rsi
from ta.volatility import bollinger_mavg, bollinger_hband, bollinger_wband
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def fetch_historical_data(symbol, start_date, end_date):
    """
    Fetches historical market data for a given stock symbol within a specified date range.

    Parameters:
    - symbol (str): The ticker symbol for the stock (e.g., 'TSLA' for Tesla, '^GSPC' for S&P 500).
    - start_date (str): The start date for the data in 'YYYY-MM-DD' format.
    - end_date (str): The end date for the data in 'YYYY-MM-DD' format.

    Returns:
    - DataFrame: Historical market data for the specified symbol.
    """
    data = yf.download(tickers=[symbol], start=start_date, end=end_date)
    if data.empty:
        raise ValueError(f"No data found for {symbol} between {start_date} and {end_date}.")
    return data


def clean_data(data):
    """
    'Cleans' the input DataFrame by handling missing values and removing outliers.

    Parameters:
    - data (DataFrame): The input data to clean.

    Returns:
    - DataFrame: The cleaned data.
    """

    # Remove rows with missing values
    cleaned_data = data.dropna()

    q_low = cleaned_data['Close'].quantile(0.01)
    q_hi = cleaned_data['Close'].quantile(0.99)

    cleaned_data = cleaned_data[(cleaned_data['Close'] > q_low) & (cleaned_data['Close'] < q_hi)]

    return cleaned_data


def calculate_features(data):
    """
    Calculates technical indicators and adds them as new columns to the DataFrame.

    Parameters:
    - data (DataFrame): The input data, expected to have columns like 'Open', 'High', 'Low', 'Close', and 'Volume'.

    Returns:
    - DataFrame: The input data enhanced with additional technical indicator features.
    """

    data['Close'] = pd.to_numeric(data['Close'], errors='coerce')
    data['Open'] = pd.to_numeric(data['Open'], errors='coerce')
    data['High'] = pd.to_numeric(data['High'], errors='coerce')
    data['Low'] = pd.to_numeric(data['Low'], errors='coerce')
    data['Volume'] = pd.to_numeric(data['Volume'], errors='coerce')

    # Calculate moving averages
    data['SMA'] = data['Close'].rolling(window=5).mean()
    data['EMA'] = data['Close'].ewm(alpha=0.2).mean()
    data['SMA'] = data['Close'].rolling(window=20).mean()

    data['Upper_Band'] = data['SMA'] + (2 * data['Close'].rolling(window=20).std())
    data['Lower_Band'] = data['SMA'] - (2 * data['Close'].rolling(window=20).std())

    # Calculating RSI
    data['RSI'] = ta.momentum.rsi(data['Close'], window=14)

    # Calculating MACD
    macd = ta.trend.MACD(data['Close'])
    data['MACD_diff'] = macd.macd_diff()

    # Calculate Bollinger Bands
    bollinger = ta.volatility.BollingerBands(data['Close'])
    data['bollinger_mavg'] = bollinger.bollinger_mavg()
    data['bollinger_hband'] = bollinger.bollinger_hband()
    data['bollinger_lband'] = bollinger.bollinger_lband()

    # Calculating logarithmic returns
    data['Log_Returns'] = np.log(data['Close'] / data['Close'].shift(1))

    # Calculating historical volatility
    data['Historical_Volatility'] = data['Log_Returns'].rolling(window=20).std() * np.sqrt(252)

    # Add temporal features to help capture any potential periodic trends
    data['Day_of_Week'] = data.index.dayofweek
    data['Month'] = data.index.month
    data['Year'] = data.index.year

    # Apply one-hot encoding to avoid artificial ordinality (e.g., treating Thursday as "greater" than Monday)
    data = pd.concat([
        data,
        pd.get_dummies(data['Day_of_Week'], prefix='Day'),
        pd.get_dummies(data['Month'], prefix='Month')
    ], axis=1)

    data.drop(['Day_of_Week', 'Month'], axis=1, inplace=True)

    return data


def scale_data(data):
    scaler = MinMaxScaler(feature_range=(0, 1))

    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

    return scaled_data, scaler




