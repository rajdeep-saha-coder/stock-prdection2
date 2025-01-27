from flask import Flask, request, render_template
import numpy as np
import joblib
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
import matplotlib.pyplot as plt
import io
import base64

# Initialize the Flask application
app = Flask(__name__)

# Load the trained models
lr_model = joblib.load('linear_regression_model.pkl')  # Load the Linear Regression model
lstm_model = load_model('lstm_model.h5')  # Load the LSTM model

# Initialize the scaler (should match the one used in training)
scaler = MinMaxScaler(feature_range=(0, 1))

@app.route('/')
def home():
    return render_template('index.html')  # Serve the HTML page

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the stock ticker from the form input
        stock_ticker = request.form['ticker'].strip()

        # Fetch historical stock data for the last 60 days
        stock_data = yf.download(stock_ticker, period='60d', interval='1d')

        if len(stock_data) < 60:
            return render_template('index.html', error=f'Not enough data for {stock_ticker}. Please try another ticker.')

        # Extract the closing prices for the last 60 days
        features = stock_data['Close'].values[-60:]

        # Reshape the features for prediction
        features = features.reshape(-1, 1)  # Shape: (60, 1)

        # Scale the features
        scaled_features = scaler.fit_transform(features)  # Scale the data

        # Prepare input for Linear Regression and LSTM
        lr_input = scaled_features.flatten().reshape(1, -1)  # Shape: (1, 60)
        lstm_input = scaled_features.reshape(1, 60, 1)  # Shape: (1, 60, 1)

        # Predict using Linear Regression
        lr_prediction = lr_model.predict(lr_input)

        # Predict using LSTM
        lstm_prediction = lstm_model.predict(lstm_input)

        # Inverse transform predictions
        lr_prediction_inverse = scaler.inverse_transform(lr_prediction.reshape(-1, 1))
        lstm_prediction_inverse = scaler.inverse_transform(lstm_prediction.reshape(-1, 1))

        # Prepare data for plotting
        original_prices = stock_data['Close'].values[-60:]
        lr_predictions = np.append(original_prices[:-1], lr_prediction_inverse)
        lstm_predictions = np.append(original_prices[:-1], lstm_prediction_inverse)

        # Plotting the graphs
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        ax1.plot(original_prices, label='Original Prices', color='blue')
        ax1.set_title('Original Stock Prices')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Price (USD)')
        ax1.legend()
        fig1.tight_layout()

        fig2, ax2 = plt.subplots(figsize=(10, 6))
        ax2.plot(original_prices, label='Original Prices', color='blue')
        ax2.plot(lr_predictions, label='Linear Regression Prediction', color='red', linestyle='--')
        ax2.set_title('Stock Price Prediction - Linear Regression')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Price (USD)')
        ax2.legend()
        fig2.tight_layout()

        fig3, ax3 = plt.subplots(figsize=(10, 6))
        ax3.plot(original_prices, label='Original Prices', color='blue')
        ax3.plot(lstm_predictions, label='LSTM Prediction', color='green', linestyle='--')
        ax3.set_title('Stock Price Prediction - LSTM')
        ax3.set_xlabel('Date')
        ax3.set_ylabel('Price (USD)')
        ax3.legend()
        fig3.tight_layout()

        # Convert plots to base64 strings
        img1 = io.BytesIO()
        fig1.savefig(img1, format='png')
        img1.seek(0)
        plot_url1 = base64.b64encode(img1.getvalue()).decode()

        img2 = io.BytesIO()
        fig2.savefig(img2, format='png')
        img2.seek(0)
        plot_url2 = base64.b64encode(img2.getvalue()).decode()

        img3 = io.BytesIO()
        fig3.savefig(img3, format='png')
        img3.seek(0)
        plot_url3 = base64.b64encode(img3.getvalue()).decode()

        # Calculate accuracy (Mean Absolute Error for comparison)
        from sklearn.metrics import mean_absolute_error

        original_prices_for_comparison = original_prices[1:]  # Exclude the last value for comparison
        lr_mae = mean_absolute_error(original_prices_for_comparison, lr_predictions[1:])
        lstm_mae = mean_absolute_error(original_prices_for_comparison, lstm_predictions[1:])
        
        # Render predictions, graphs, and accuracy on the page
        return render_template(
            'index.html',
            lr_prediction=f'{lr_prediction_inverse[0][0]:.2f}',
            lstm_prediction=f'{lstm_prediction_inverse[0][0]:.2f}',
            plot_url1=plot_url1,
            plot_url2=plot_url2,
            plot_url3=plot_url3,
            stock_ticker=stock_ticker,
            lr_mae=f'{lr_mae:.4f}',
            lstm_mae=f'{lstm_mae:.4f}'
        )

    except Exception as e:
        return render_template('index.html', error=f'Error: {str(e)}')

if __name__ == '__main__':
    app.run(debug=True)
