from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime, timedelta
import plotly
import plotly.graph_objs as go
from model import VolatilityModel, train_and_save_model
from pandas.tseries.offsets import BDay

app = Flask(__name__)

# Initialize model
model = VolatilityModel()
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)

# Check if model exists, if not, train it
if not os.path.exists('volatility_model.pkl'):
    print("Training new model...")
    model = train_and_save_model()
else:
    print("Loading existing model...")
    model.load_model()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get stock symbol from request
        symbol = request.form.get('symbol', 'RELIANCE.BSE')
        
        # Fetch and process data
        data = model.fetch_stock_data(symbol)
        
        # Sort data to ensure chronological order
        data = data.sort_index(ascending=True)
        processed_data = model.engineer_features(data)
        
        # Ensure we're using recent data
        current_date = datetime.now()
        cutoff_date = current_date - timedelta(days=365)
        recent_data = processed_data[processed_data.index >= pd.Timestamp(cutoff_date)]
        
        if len(recent_data) < 30:
            recent_data = processed_data.iloc[-100:]
        
        # Make predictions using the enhanced model
        predictions, future_dates = model.predict_volatility(recent_data)
        
        # Create historical data for plotting
        historical_dates = recent_data.index[-30:].strftime('%Y-%m-%d').tolist()
        historical_values = recent_data['Volatility'][-30:].tolist()
        historical_prices = recent_data['4. close'][-30:].tolist()
        
        # Create price chart
        price_fig = go.Figure()
        price_fig.add_trace(go.Scatter(
            x=historical_dates,
            y=historical_prices,
            mode='lines',
            name='Close Price'
        ))
        price_fig.update_layout(
            title='Recent Stock Price',
            xaxis_title='Date',
            yaxis_title='Price'
        )
        
        # Create volatility chart with proper scaling
        vol_fig = go.Figure()
        vol_fig.add_trace(go.Scatter(
            x=historical_dates,
            y=historical_values,
            mode='lines',
            name='Historical Volatility'
        ))
        vol_fig.add_trace(go.Scatter(
            x=future_dates,
            y=predictions,
            mode='lines+markers',
            name='Predicted Volatility'
        ))
        
        # Calculate y-axis range to prevent extreme scaling
        max_historical = max(historical_values) if historical_values else 0.05
        max_predicted = max(predictions) if predictions else 0.05
        y_max = max(max_historical, max_predicted) * 1.2  # Add 20% margin
        
        vol_fig.update_layout(
            title='Volatility Prediction',
            xaxis_title='Date',
            yaxis_title='Volatility',
            yaxis=dict(
                range=[0, y_max]
            )
        )
        
        # Convert plots to JSON
        price_graph = json.dumps(price_fig, cls=plotly.utils.PlotlyJSONEncoder)
        vol_graph = json.dumps(vol_fig, cls=plotly.utils.PlotlyJSONEncoder)
        
        # Get latest stock data
        latest_data = {
            'date': recent_data.index[-1].strftime('%Y-%m-%d'),
            'open': float(recent_data['1. open'][-1]),
            'high': float(recent_data['2. high'][-1]),
            'low': float(recent_data['3. low'][-1]),
            'close': float(recent_data['4. close'][-1]),
            'volume': int(recent_data['5. volume'][-1]),
            'current_volatility': float(recent_data['Volatility'][-1])
        }
        
        return jsonify({
            'success': True,
            'symbol': symbol,
            'latest_data': latest_data,
            'predictions': {
                'dates': future_dates,
                'values': [float(val) for val in predictions]
            },
            'price_graph': price_graph,
            'vol_graph': vol_graph
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

if __name__ == '__main__':
    app.run(debug=True)
