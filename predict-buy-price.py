# %%
pip install xgboost binance pandas numpy


# %%
pip install tqdm

# %%
pip install python-binance

# %%

from binance.client import Client
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time
import os
from textblob import TextBlob
import praw  # For Reddit API
import tweepy  # For Twitter API
from bs4 import BeautifulSoup
import requests
from ta import add_all_ta_features
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam

from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, Flatten
import matplotlib.pyplot as plt



API_KEY = 'xxxxxxx'
API_SECRET = 'xxxxxxxxx'


client = Client(API_KEY, API_SECRET)

def calculate_order_book_imbalance(order_book):
    bids = sum([float(bid[1]) for bid in order_book['bids']])
    asks = sum([float(ask[1]) for ask in order_book['asks']])
    return (bids - asks) / (bids + asks) if (bids + asks) != 0 else 0

def get_recent_trade_direction(trades):
    # Calculate the net direction of recent trades
    directions = [1 if trade['isBuyerMaker'] else -1 for trade in trades]
    return sum(directions) / len(directions) if directions else 0

def get_quote_update_frequency(order_book, previous_order_book, interval_seconds):
    # Compare current and previous order books to count changes
    if not previous_order_book:
        return 0
    changes = 0
    for side in ['bids', 'asks']:
        current_prices = {price: qty for price, qty in order_book[side]}
        previous_prices = {price: qty for price, qty in previous_order_book[side]}
        changes += sum(1 for price in current_prices if current_prices[price] != previous_prices.get(price, None))
    return changes / interval_seconds

# Calculate quote update frequency
def calculate_quote_update_frequency(order_book, previous_order_book):
    """
    Calculate the quote update frequency between two order books.
    
    Parameters:
    - order_book (dict): Current order book from Binance.
    - previous_order_book (dict): Previous order book from Binance.
    
    Returns:
    - float: Quote update frequency.
    """
    if not previous_order_book:
        return 0
    changes = 0
    for side in ['bids', 'asks']:
        current_prices = {price: qty for price, qty in order_book[side]}
        previous_prices = {price: qty for price, qty in previous_order_book[side]}
        changes += sum(1 for price in current_prices if current_prices[price] != previous_prices.get(price, None))
    return changes

# Add quote frequency to DataFrame
def add_quote_frequency(df, symbol):
    """
    Add quote update frequency to a DataFrame by fetching the order book for each timestamp.
    
    Parameters:
    - df (pd.DataFrame): Historical data with timestamps.
    - symbol (str): Trading symbol (e.g., "BTCUSDT").
    
    Returns:
    - pd.DataFrame: DataFrame with a new column for quote frequency.
    """
    quote_frequencies = []
    previous_order_book = None
    for _, row in df.iterrows():
        order_book = client.get_order_book(symbol=symbol)
        quote_frequency = calculate_quote_update_frequency(order_book, previous_order_book)
        quote_frequencies.append(quote_frequency)
        previous_order_book = order_book
        time.sleep(1)  # Avoid hitting API rate limits
    df['quote_frequency'] = quote_frequencies
    return df


def fetch_historical_data(symbol, interval='1s', limit=100000):
    """
    Fetch historical kline data for a given symbol and interval, handling large limits.

    Parameters:
    - symbol (str): Trading pair symbol (e.g., "BTCUSDT").
    - interval (str): Kline interval (e.g., "1s", "1m").
    - limit (int): Total number of rows to fetch.

    Returns:
    - pd.DataFrame: DataFrame containing historical data.
    """
    # Binance API maximum limit per request
    max_limit_per_request = 1000

    # List to hold chunks of data
    all_data = []
    end_time = None  # End time for fetching data (None means "current time")

    while limit > 0:
        # Determine the batch size for this request
        request_limit = min(limit, max_limit_per_request)

        # Fetch data from Binance
        klines = client.get_klines(
            symbol=symbol, interval=interval, limit=request_limit, endTime=end_time
        )

        if not klines:
            break  # Exit if no data is returned

        # Append fetched data to the list
        all_data.extend(klines)

        # Update end_time to the earliest timestamp from the last batch
        end_time = klines[0][0]  # Update to the first timestamp of the last batch
        limit -= request_limit  # Decrease remaining limit

        # Avoid hitting API rate limits
        time.sleep(0.1)

    # Convert data to DataFrame
    data = [
        {
            "timestamp": kline[0],
            "open": float(kline[1]),
            "high": float(kline[2]),
            "low": float(kline[3]),
            "close": float(kline[4]),
            "volume": float(kline[5]),
        }
        for kline in all_data
    ]
    df = pd.DataFrame(data)

    # Convert timestamp to datetime for readability
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

    return df

symbol = "BTCUSDT"
interval = "1s"
limit = 100000  # Fetch 100,000 rows of data

# Fetch data
df = fetch_historical_data(symbol, interval=interval, limit=limit)



def add_order_book_features(df, symbol, order_book_data=None):
    """
    Add current buy price and amount to the historical dataset.
    
    Parameters:
    - df: Historical DataFrame with timestamps.
    - symbol: Trading symbol (e.g., 'BTCUSDT').
    - order_book_data: Optional, simulated or fetched order book data.
    
    Returns:
    - Updated DataFrame with 'current_buy_price' and 'current_buy_amount'.
    """
    # Fetch order book data (if not provided)
    if order_book_data is None:
        order_book = client.get_order_book(symbol=symbol)
        highest_bid = float(order_book['bids'][0][0])
        bid_amount = float(order_book['bids'][0][1])
    else:
        highest_bid = order_book_data['bids'][0][0]
        bid_amount = order_book_data['bids'][0][1]

    # Add to DataFrame
    df['current_buy_price'] = highest_bid
    df['current_buy_amount'] = bid_amount
    return df


def add_future_targets(df, shift_period=10):
    """
    Add future buy price and amount as target variables.
    
    Parameters:
    - df: DataFrame with historical data and buy price/amount features.
    - shift_period: Number of rows to shift for future values.
    
    Returns:
    - Updated DataFrame with 'future_buy_price' and 'future_buy_amount'.
    """
    df['future_buy_price'] = df['current_buy_price'].shift(-shift_period)
    df['future_buy_amount'] = df['current_buy_amount'].shift(-shift_period)
    return df




def calculate_technical_features(df, symbol, order_book=None, trades=None,text_data=None):
    """
    Calculate common technical indicators for a given financial dataset.

    Parameters:
    - df (pd.DataFrame): DataFrame containing at least ['open', 'high', 'low', 'close', 'volume'].
    - order_book (dict, optional): Order book data (for custom imbalance calculations).
    - trades (list, optional): Recent trades data (for custom trade direction calculations).

    Returns:
    - pd.DataFrame: DataFrame with additional technical features, including future buy/sell prices.
    """
    # Statistical and custom features
    custom_features = {
        "sma_10": df['close'].rolling(window=10).mean(),
        "ema_10": df['close'].ewm(span=10, adjust=False).mean(),
        "price_change": df['close'].diff(),
        "rolling_mean": df['close'].rolling(window=10).mean(),
        "rolling_std": df['close'].rolling(window=10).std(),
        "rate_of_change": df['close'].pct_change(periods=10),
        "bollinger_upper": df['close'].rolling(window=10).mean() + (2 * df['close'].rolling(window=10).std()),
        "bollinger_lower": df['close'].rolling(window=10).mean() - (2 * df['close'].rolling(window=10).std())
    }

    df = add_order_book_features(df, symbol='BTCUSDT')

# Add future buy price and amount as target variables
    #df = add_future_targets(df, shift_period=10)

    # Custom features
    if order_book:
        df['order_book_imbalance'] = calculate_order_book_imbalance(order_book)
    else:
        df['order_book_imbalance'] = np.nan  # Placeholder if no live data
    
    if trades:
        df['trade_direction'] = get_recent_trade_direction(trades)
    else:
        df['trade_direction'] = np.nan 

    # df['quote_frequency'] = get_quote_update_frequency(order_book, None, 60) 


    # VWAP
    custom_features["vwap"] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()

   

    # Donchian Channels
    custom_features["donchian_upper"] = df['high'].rolling(window=20).max()
    custom_features["donchian_lower"] = df['low'].rolling(window=20).min()
    custom_features["donchian_mid"] = (custom_features["donchian_upper"] + custom_features["donchian_lower"]) / 2

    # Williams %R
    high_14 = df['high'].rolling(window=14).max()
    low_14 = df['low'].rolling(window=14).min()
    custom_features["williams_r"] = -100 * (high_14 - df['close']) / (high_14 - low_14)


    # Future buy/sell prices
    custom_features["future_buy_price"] = df['low'].shift(-10)  # Future lowest price
    custom_features["future_sell_price"] = df['high'].shift(-10)  # Future highest price
    custom_features["future_price"] = df["close"].shift(-10)  # Future close price

    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    custom_features["rsi"] = 100 - (100 / (1 + rs))

    # MACD
    ema_12 = df['close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['close'].ewm(span=26, adjust=False).mean()
    custom_features["macd"] = ema_12 - ema_26
    custom_features["macd_signal"] = custom_features["macd"].ewm(span=9, adjust=False).mean()

    # Stochastic Oscillator
    low_14 = df['low'].rolling(window=14).min()
    high_14 = df['high'].rolling(window=14).max()
    custom_features["stoch_k"] = 100 * (df['close'] - low_14) / (high_14 - low_14)
    custom_features["stoch_d"] = pd.Series(custom_features["stoch_k"]).rolling(window=3).mean()

    # ATR
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    custom_features["atr"] = true_range.rolling(window=14).mean()

    # Convert custom features into a DataFrame
    custom_features_df = pd.DataFrame(custom_features)

    # Add TA library features
    ta_features_df = add_all_ta_features(
        df, open="open", high="high", low="low", close="close", volume="volume", fillna=True
    )

    # Combine all features
    result_df = pd.concat([df, custom_features_df, ta_features_df], axis=1)
    result_df = result_df.loc[:, ~result_df.columns.duplicated()]
    # print(result_df.isnull().sum())
    # Handle NaN values
    result_df.loc[:, 'rsi'] = result_df['rsi'].fillna(50)
    result_df.loc[:, 'price_change'] = result_df['price_change'].fillna(0)
    result_df.loc[:, 'rolling_mean'] = result_df['rolling_mean'].fillna(result_df['close'])
    result_df.loc[:, 'rolling_std'] = result_df['rolling_std'].fillna(0)
    result_df.loc[:, 'rate_of_change'] = result_df['rate_of_change'].fillna(0)
    result_df.loc[:, 'bollinger_upper'] = result_df['bollinger_upper'].fillna(result_df['rolling_mean'])
    result_df.loc[:, 'bollinger_lower'] = result_df['bollinger_lower'].fillna(result_df['rolling_mean'])
    result_df.loc[:, 'stoch_k'] = result_df['stoch_k'].fillna(50)
    result_df.loc[:, 'stoch_d'] = result_df['stoch_d'].fillna(50)
    result_df.loc[:, 'atr'] = result_df['atr'].fillna(0)
    result_df.loc[:, 'order_book_imbalance'] = result_df['order_book_imbalance'].fillna(0)
    result_df.loc[:, 'sma_10'] = result_df['sma_10'].fillna(0)
    result_df = result_df.ffill()
    result_df = result_df.bfill()

    print(result_df.shape)
   


    # Drop rows with remaining NaNs
    result_df.dropna(inplace=True)
    result_df.reset_index(drop=True, inplace=True)
    print(result_df.shape)
    return result_df


order_book = client.get_order_book(symbol=symbol)
trades = client.get_recent_trades(symbol=symbol)

df_with_features = calculate_technical_features(df, symbol, order_book, trades)





# Separate features and target
X = df_with_features.drop(columns=['future_price', 'future_buy_price', 'future_sell_price'])
  # Drop the target column
y = df_with_features['future_buy_price']

# Ensure no NaN values
X = X.dropna()
y = y[X.index]  # Align target with feature indices

# Ensure X contains only numeric columns
X = X.select_dtypes(include=[np.number])

# Scale the features
scaler_X = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X)

# Scale the target (y)
scaler_y = MinMaxScaler()
y_scaled = scaler_y.fit_transform(y.to_numpy().reshape(-1, 1))

# Create LSTM-ready data
def create_lstm_data(data, labels, lookback=60):
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i-lookback:i])  # Use previous `lookback` steps
        y.append(labels[i])          # Predict the current step
    return np.array(X), np.array(y)

lookback = 60
X_lstm, y_lstm = create_lstm_data(X_scaled, y_scaled)

# Reshape X for LSTM input
X_lstm = X_lstm.reshape((X_lstm.shape[0], X_lstm.shape[1], X_lstm.shape[2]))

# # Define LSTM model
# model = Sequential([
#     LSTM(50, return_sequences=True, input_shape=(lookback, X_lstm.shape[2])),
#     LSTM(50),
#     Dropout(0.2),
#     Dense(1)  # Regression output
# ])

X_train, X_test, y_train, y_test = train_test_split(X_lstm, y_lstm, test_size=0.2, random_state=42)

model = Sequential([
    Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])),
    MaxPooling1D(pool_size=2),
    Dropout(0.2),


    LSTM(50, return_sequences=True, input_shape=(lookback, X_lstm.shape[2]), 
         kernel_regularizer=l2(0.01)),  # L2 regularization
    Dropout(0.2),
    LSTM(50, kernel_regularizer=l2(0.01)),
    Dropout(0.2),
    Dense(1, kernel_regularizer=l2(0.01))  # L2 regularization in the final layer
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Split data into training and testing sets


# # Train the model
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2)




# Evaluate the model
loss, mae = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss:.4f}, Mean Absolute Error: {mae:.4f}")


# Make predictions
y_pred_scaled = model.predict(X_test)

# Inverse transform predictions and actual values
y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_test = scaler_y.inverse_transform(y_test)

# Calculate additional metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

# Directional Accuracy (DA)
directional_accuracy = np.mean(np.sign(y_pred[1:] - y_pred[:-1]) == np.sign(y_test[1:] - y_test[:-1])) * 100

# Print all metrics
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"R² Score: {r2:.4f}")
print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
print(f"Directional Accuracy (DA): {directional_accuracy:.2f}%")


plt.figure(figsize=(10, 6))
plt.plot(y_test, label='Actual Prices', alpha=0.75)
plt.plot(y_pred, label='Predicted Prices', alpha=0.75)
plt.title("Actual vs Predicted Prices")
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
plt.show()

import pickle

# Save model and scalers together
with open('model_and_scalers.pkl', 'wb') as f:
    pickle.dump({'model': model, 'scaler_X': scaler_X, 'scaler_y': scaler_y}, f)


# Load model and scalers
with open('model_and_scalers.pkl', 'rb') as f:
    saved_data = pickle.load(f)

model = saved_data['model']
scaler_X = saved_data['scaler_X']
scaler_y = saved_data['scaler_y']    

def fetch_live_features(symbol, scaler,  lookback=60):
    """
    Fetch live features and scale them for prediction, handling NaN values.

    Parameters:
    - symbol (str): The trading symbol (e.g., 'BTCUSDT').
    - scaler (MinMaxScaler): The scaler used for training.
    - feature_columns (list): List of features used in the model.
    - lookback (int): Number of timesteps for LSTM input.

    Returns:
    - np.ndarray: Scaled live features ready for prediction.
    """
    # Calculate required rows to avoid NaNs
    required_rows = lookback + max(14, 10, 15)  # Add the maximum window size

    # Fetch live data
    live_data = fetch_historical_data(symbol, interval='1s', limit=required_rows)
    print(live_data.shape)
    # Compute technical features
    order_book = client.get_order_book(symbol=symbol)
    trades = client.get_recent_trades(symbol=symbol)
    live_data = calculate_technical_features(live_data, symbol, order_book, trades)
    print(live_data.shape)
    live_data=live_data.drop(columns=['future_price', 'future_buy_price', 'future_sell_price'])
    print(live_data.shape)
    live_data = live_data.select_dtypes(include=[np.number])
    print(live_data.shape)
    # Handle NaN values
    live_data = live_data.ffill()  # Forward fill
    live_data = live_data.bfill() 
    

    # Ensure alignment with feature_columns
    

    # Check for sufficient rows
    if len(live_data) < lookback:
        raise ValueError(f"Insufficient live data: Expected at least {lookback} rows, got {len(live_data)}.")

    # Scale live data
    live_data_scaled = scaler.transform(live_data.values[-lookback:])



    # Reshape for LSTM input
    return live_data_scaled.reshape(1, lookback, len(live_data.columns))





# Prepare features for live prediction
live_features = fetch_live_features(symbol, scaler_X, lookback=60)

# Make prediction
predicted_price_scaled = model.predict(live_features)
predicted_price = scaler_y.inverse_transform(predicted_price_scaled)

print(f"Predicted Buy Price (10s ahead): {predicted_price[0][0]:.2f}")



def backtest(model, data, lookback, scaler_X, scaler_y):
    """
    Perform backtesting using the trained model and historical data.

    Parameters:
    - model: Trained LSTM model.
    - data: Historical data with features and target columns.
    - lookback: Number of timesteps to use for predictions.
    - scaler_X: Scaler used to normalize input features.
    - scaler_y: Scaler used to normalize the target variable.

    Returns:
    - metrics: Dictionary of backtesting metrics.
    """
    # Extract features and target
    X = data.drop(columns=['future_buy_price', 'future_sell_price', 'future_price'])
    X =X.select_dtypes(include=[np.number])
    y_buy = data['future_buy_price']
    y_sell = data['future_sell_price']

    # Scale features
    X_scaled = scaler_X.transform(X)
    
    # Prepare data for backtesting
    predictions = []
    actual_prices = []

    for i in range(lookback, len(X_scaled)):
        # Prepare input for LSTM
        input_data = X_scaled[i-lookback:i].reshape(1, lookback, X_scaled.shape[1])
        
        # Predict future prices
        pred_scaled = model.predict(input_data)
        pred = scaler_y.inverse_transform(pred_scaled)
        
        predictions.append(pred[0][0])  # Predicted price
        actual_prices.append(y_sell.iloc[i])  # Actual price

   
    

    return predictions, actual_prices

predictions, actual_prices = backtest(model, df_with_features[:5000], lookback=60, scaler_X=scaler_X, scaler_y=scaler_y)



# %%
plt.figure(figsize=(12, 6))
plt.plot(actual_prices, label='Actual Prices', color='blue')
plt.plot(predictions, label='Predicted Prices', color='orange')
plt.title('Backtesting: Actual vs Predicted Prices')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()


# %%
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

# Evaluate performance
mse = mean_squared_error(actual_prices, predictions)
rmse = np.sqrt(mse)
mae = mean_absolute_error(actual_prices, predictions)
r2 = r2_score(actual_prices, predictions)

print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"R² Score: {r2:.4f}")



# %%
import time
import numpy as np
import pandas as pd
from tqdm import tqdm

def fetch_live_data(symbol, lookback, scaler_X):
    """
    Fetch live data and preprocess it for prediction.
    """
    # Fetch recent historical data for lookback period
    live_data = fetch_historical_data(symbol, interval='1s', limit=lookback)
    #print(f"Initial live data shape: {live_data.shape}")
    
    # Compute technical features
    order_book = client.get_order_book(symbol=symbol)
    trades = client.get_recent_trades(symbol=symbol)
    live_data = calculate_technical_features(live_data, symbol, order_book, trades)
    #print(f"After technical features: {live_data.shape}")

    # Drop unused columns if they exist
    live_data = live_data.drop(columns=['future_price', 'future_buy_price', 'future_sell_price'], errors='ignore')
    # print(f"After dropping target columns: {live_data.shape}")
    
    # Ensure data is numeric
    live_data = live_data.select_dtypes(include=[np.number])
    # print("Live Data Columns:", live_data.columns)

    # print(f"After selecting numeric data: {live_data.shape}")


    # feature_columns = scaler_X.feature_names_in_
    # live_data = live_data[feature_columns]
    

    # Ensure no missing values
    live_data = live_data.ffill()  # Forward fill
    live_data = live_data.bfill() 

    # Check data length
    if live_data.empty or len(live_data) < lookback:
        raise ValueError("Insufficient data for prediction.")

    # Scale features
    live_features_scaled = scaler_X.transform(live_data.values[-lookback:])
    return live_features_scaled.reshape(1, lookback, live_features_scaled.shape[1])

def live_backtest(symbol, model, scaler_X, scaler_y, iterations=1000, lookback=60):
    """
    Perform live backtesting by fetching real-time data and predicting future prices.
    """
    results = []

    for _ in tqdm(range(iterations)):
        try:
            # Step 1: Fetch live features and predict
            live_features = fetch_live_data(symbol, lookback, scaler_X)
            predicted_price_scaled = model.predict(live_features)
            predicted_price = scaler_y.inverse_transform(predicted_price_scaled)[0][0]

            # Step 2: Wait for 10 seconds
            time.sleep(10)

            # Step 3: Fetch actual future price
            order_book = client.get_order_book(symbol=symbol)
            actual_buy_price = float(order_book['bids'][0][0])
            # actual_price = fetch_historical_data(symbol, interval='1s', limit=1)['close'].iloc[-1]
            print({
                'timestamp': pd.Timestamp.now(), 
                'predicted_price': predicted_price, 
                'actual_price': actual_buy_price
            })
            # Log results
            results.append({
                'timestamp': pd.Timestamp.now(), 
                'predicted_price': predicted_price, 
                'actual_price': actual_buy_price
            })
        
        except Exception as e:
            print(f"Error fetching or predicting live data: {e}")
            continue

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    return results_df

# Example usage
symbol = 'BTCUSDT'
results_df = live_backtest(symbol, model, scaler_X, scaler_y, iterations=1000)


# Save results to CSV
results_df.to_csv('live_backtesting_results.csv', index=False)

# Plot actual vs predicted prices
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.plot(results_df['timestamp'], results_df['actual_price'], label='Actual Prices', color='blue')
plt.plot(results_df['timestamp'], results_df['predicted_price'], label='Predicted Prices', color='orange')
plt.xlabel('Timestamp')
plt.ylabel('Price')
plt.title('Live Backtesting: Predicted vs Actual Prices')
plt.legend()
plt.show()



# Extract actual and predicted prices
actual_prices = results_df['actual_price'].values
predicted_prices = results_df['predicted_price'].values

# Calculate evaluation metrics
mae = mean_absolute_error(actual_prices, predicted_prices)
mse = mean_squared_error(actual_prices, predicted_prices)
rmse = np.sqrt(mse)  # RMSE calculation
r2 = r2_score(actual_prices, predicted_prices)

# Print results
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R² Score: {r2:.4f}")





