import numpy as np
import pandas as pd
import pickle
from datetime import timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from alpha_vantage.timeseries import TimeSeries
from pandas.tseries.offsets import BDay

class VolatilityModel:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.api_key = 'HOH8KC0N7O83EDQD'  # Replace with your API key
        self.symbol = None
        self.feature_columns = None

    def fetch_stock_data(self, symbol):
        self.symbol = symbol
        ts = TimeSeries(key=self.api_key, output_format='pandas')
        data, _ = ts.get_daily(symbol=symbol, outputsize='full')
        data = data.sort_index()  # ensure data is oldest to newest
        return data

    def engineer_features(self, data):
        """
        Calculate various technical indicators and features for volatility prediction.
        """
        # Basic returns and volatility
        data['Returns'] = data['4. close'].pct_change()
        data['Log_Returns'] = np.log(data['4. close'] / data['4. close'].shift(1))
        data['Volatility'] = data['Returns'].rolling(window=5).std()
        
        # Price-based features
        data['High_Low_Range'] = (data['2. high'] - data['3. low']) / data['4. close']
        data['Open_Close_Range'] = np.abs(data['1. open'] - data['4. close']) / data['4. close']
        
        # Moving averages
        data['SMA_5'] = data['4. close'].rolling(window=5).mean()
        data['SMA_20'] = data['4. close'].rolling(window=20).mean()
        
        # RSI calculation
        delta = data['4. close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))
        
        # Volume features
        data['Volume_Change'] = data['5. volume'].pct_change()
        data['Volume_MA_5'] = data['5. volume'].rolling(window=5).mean()
        
        # Previous volatilities
        data['Volatility_10'] = data['Returns'].rolling(window=10).std()
        data['Volatility_20'] = data['Returns'].rolling(window=20).std()
        
        # Drop NaN values
        data.dropna(inplace=True)
        
        return data

    def train_model(self, data):
        """
        Train the model using multiple features.
        """
        data = self.engineer_features(data)
        
        # If data is too small, handle gracefully
        if len(data) < 20:
            raise ValueError("Not enough data to compute features. Try a different symbol or output size.")
        
        # Select features for prediction
        self.feature_columns = [
            '4. close', 'Returns', 'Log_Returns', 'High_Low_Range', 
            'Open_Close_Range', 'SMA_5', 'SMA_20', 'RSI',
            'Volume_Change', 'Volume_MA_5', 'Volatility_10', 'Volatility_20'
        ]
        
        X = data[self.feature_columns].values
        y = data['Volatility'].values
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        self.model.fit(X_train_scaled, y_train)
        
        # Save the trained model, scaler, and feature columns
        with open('volatility_model.pkl', 'wb') as f:
            pickle.dump((self.model, self.scaler, self.feature_columns), f)

    def load_model(self):
        with open('volatility_model.pkl', 'rb') as f:
            self.model, self.scaler, self.feature_columns = pickle.load(f)

    def predict_volatility(self, data, days=5):
        if self.model is None:
            raise ValueError("Model has not been trained or loaded yet")
        
        data = self.engineer_features(data)
        
        # Get the latest data point with all features
        latest_data = data[self.feature_columns].iloc[-1].values
        
        predictions = []
        future_dates = []
        last_date = pd.to_datetime(data.index[-1])
        current_data = latest_data.copy()
        
        # Get historical volatility for reference
        historical_vol = data['Volatility'][-30:].values
        avg_vol = np.mean(historical_vol)
        std_vol = np.std(historical_vol)
        
        for i in range(days):
            # Scale the current data
            scaled_data = self.scaler.transform(current_data.reshape(1, -1))
            
            # Make prediction
            base_pred = self.model.predict(scaled_data)[0]
            
            # Add some randomness based on historical patterns
            random_factor = np.random.normal(0, std_vol * 0.3)
            pred = max(0.001, base_pred + random_factor)
            
            # Ensure prediction doesn't deviate too far
            if pred > avg_vol * 3:
                pred = avg_vol * 3
                
            predictions.append(pred)
            
            # Create future date using business days
            future_date = last_date + BDay(i+1)
            future_dates.append(future_date.strftime('%Y-%m-%d'))
            
            # Update features for next prediction (simplified)
            # Update close price based on volatility
            price_change = np.random.normal(0, pred) * current_data[0]
            current_data[0] += price_change
            
            # Update returns
            current_data[1] = price_change / (current_data[0] - price_change)
            current_data[2] = np.log(1 + current_data[1])
            
            # Update other features (simplified)
            current_data[3] = current_data[3] * (1 + np.random.normal(0, 0.01))  # High_Low_Range
            current_data[4] = current_data[4] * (1 + np.random.normal(0, 0.01))  # Open_Close_Range
            current_data[5] = (current_data[5] * 4 + current_data[0]) / 5  # SMA_5
            current_data[6] = (current_data[6] * 19 + current_data[0]) / 20  # SMA_20
            # RSI and other features remain relatively stable for simplicity
        
        return predictions, future_dates

def train_and_save_model(symbol='RELIANCE.BSE'):
    model_instance = VolatilityModel()
    data = model_instance.fetch_stock_data(symbol)
    model_instance.train_model(data)
    return model_instance

if __name__ == "__main__":
    model_instance = train_and_save_model()
    print("Model training complete!")
