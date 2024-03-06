from data_preprocessing import fetch_historical_data, clean_data, calculate_features, scale_data
from splitting_dataset import split_data
from model import create_lstm_model, train_model, make_prediction
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import datetime as dt
import yfinance as yf
import numpy as np


def main():
    symbol = '^GSPC'
    start = dt.datetime(2012, 1, 1)
    end = dt.datetime(2023, 1, 1)
    raw_data = fetch_historical_data(symbol, start, end)

    cleaned_data = clean_data(raw_data)

    data = calculate_features(cleaned_data)

    scaled_data, scaler = scale_data(data)

    X_train, t_train = split_data(scaled_data)

    model = create_lstm_model((X_train.shape[1], 1))

    # If needed, the history variable stores the training loss and metrics values.
    history = train_model(model, X_train, t_train)

    """Testing model accuracy on existing data"""

    # Load test data
    test_start = dt.datetime(2022, 1, 1)
    test_end = dt.datetime.now()

    pred_days = 14

    test_data = yf.download(tickers=['^GSPC'], start=test_start, end=test_end)
    # actual_prices = test_data['Close'].values

    total_dataset = pd.concat((data['Close'], test_data['Close']), axis=0)

    model_inputs = total_dataset[len(total_dataset) - len(test_data) - pred_days:].values

    model_inputs = model_inputs.reshape(-1, 1)

    model_inputs = scaler.transform(model_inputs)

    # Make predictions on test data
    # x_test = []

    # for x in range(60, len(model_inputs)):
        # x_test.append(model_inputs[x - 60:x, 0])

    # x_test = np.array(x_test)
    # x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    # predicted_prices = model.predict(x_test)
    # predicted_prices = scaler.inverse_transform(predicted_prices)

    # Predicting the price for the next day:
    real_data = model_inputs[-pred_days:]  # Ensure this selects exactly 60 elements
    real_data = np.array(real_data)
    real_data = np.array(real_data).reshape(1, pred_days, 1)  # Explicitly reshape to (1, 60, 1)

    real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))
    prediction = make_prediction(model, real_data)
    prediction = scaler.inverse_transform(prediction)
    print(f"Closing price for tomorrow: {prediction}")


if __name__ == "__main__":
    main()



