## ReadMe: Cryptocurrency Price Prediction and Backtesting Using Machine Learning and Deep Learning Models
Overview
This project aims to predict cryptocurrency prices (specifically for Bitcoin trading on Binance) using advanced machine learning and deep learning models. It incorporates historical and live data from the Binance API, calculates technical indicators, and builds predictive models to forecast price movements. Additionally, the code includes mechanisms for backtesting strategies using both historical and live data.

Features
1. Data Fetching
Fetch historical price and volume data using the Binance API.
Retrieve live order book data and recent trades.
2. Feature Engineering
Calculate a variety of technical indicators (e.g., SMA, EMA, RSI, MACD, Bollinger Bands, VWAP, etc.).
Generate custom features such as order book imbalance, trade direction, and quote update frequency.
3. Model Development
Use LSTM and Convolutional Neural Networks (CNN) for deep learning predictions.
Incorporate MinMax scaling for feature normalization.
4. Backtesting
Evaluate model performance on historical data.
Perform live backtesting by fetching real-time data and comparing predictions with actual prices.
5. Metrics
Evaluate predictions using metrics such as:
Mean Absolute Error (MAE)
Mean Squared Error (MSE)
Root Mean Squared Error (RMSE)
R² Score
Directional Accuracy (DA)
Requirements
Dependencies


Install the required libraries:

pip install xgboost binance pandas numpy tensorflow ta tqdm matplotlib
pip install python-binance

API Keys
Generate Binance API keys and replace placeholders in the code:


API_KEY = '<Your Binance API Key>'
API_SECRET = '<Your Binance API Secret>'



Code Workflow
1. Historical Data Preparation
Use fetch_historical_data to retrieve data from Binance.
Add technical indicators and features using calculate_technical_features.
2. Feature Scaling
Normalize features using MinMaxScaler for LSTM readiness.
3. Model Training
Build a sequential neural network combining CNN and LSTM layers.
Train the model on historical data, splitting it into training and testing sets.
4. Evaluation
Use backtesting on both historical and live data.
Plot actual vs. predicted prices for visualization.
5. Live Predictions
Fetch live data in real-time.
Generate predictions for the next price movement.
Compare predicted and actual prices.
Key Functions
fetch_historical_data
Fetches historical price data for a given trading symbol.

calculate_technical_features
Adds advanced technical indicators and features to the dataset.

add_order_book_features & add_future_targets
Enhances the dataset with order book features and future price targets.

fetch_live_features
Fetches live data, processes it, and prepares it for LSTM prediction.

backtest & live_backtest
Performs backtesting on historical and live data to evaluate model performance.

Backtesting Results
Historical Backtesting Metrics:
Mean Squared Error (MSE): 7625.2237
Root Mean Squared Error (RMSE): 87.3225
R² Score: 0.9918
Mean Absolute Percentage Error (MAPE): 0.07%
Directional Accuracy (DA): 98.08%
Live Backtesting Metrics:

Outputs
Visualizations
Training Loss Curve
Actual vs Predicted Prices (Backtesting and Live Predictions)
Saved Models
The trained model and scalers are saved as model_and_scalers.pkl.

Logs
Results from live predictions are saved in live_backtesting_results.csv.

Future Improvements
Incorporate sentiment analysis from Twitter or Reddit for enhanced feature engineering.
Optimize hyperparameters using techniques like grid search.
Add support for multiple cryptocurrencies.

Disclaimer
This project is for educational purposes only. Use with caution when trading cryptocurrencies. Always perform thorough research and consult financial advisors before making investment decisions.
