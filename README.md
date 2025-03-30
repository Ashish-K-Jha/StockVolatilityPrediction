Stock Market Volatility Predictor
A machine learning-powered web application that predicts stock market volatility for the next 5 business days. Built with Flask, scikit-learn, and interactive visualizations.

Multi-feature Volatility Prediction: Uses 12 technical indicators and market features

Interactive Data Visualization: Charts for historical and predicted volatility

Business Day Handling: Accounts for weekends in predictions

Responsive Web Interface: Dark mode UI that works on various devices

Real-time Data: Fetches the latest stock data from Alpha Vantage API

Run the application:

How It Works
Machine Learning Model
The volatility prediction model uses a Random Forest Regressor algorithm with the following features:

Closing price

Daily returns and logarithmic returns

High-Low price range and Open-Close price range

Simple Moving Averages (5-day and 20-day)

Relative Strength Index (RSI)

Volume indicators

Historical volatility measures

Prediction Process
Data Collection: Fetches historical stock data from Alpha Vantage API

Feature Engineering: Calculates technical indicators and volatility measures

Model Training: Uses Random Forest to learn patterns in historical volatility

Prediction: Forecasts future volatility based on recent market behavior

Visualization: Displays results through interactive charts and tables
