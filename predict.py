import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('./models/bitcoin_price_model.h5')

# Load the new dataset
new_data = pd.read_csv('./dataset/test.csv')


# Data preprocessing
features = ['open', 'high', 'low', 'close', 'Volume BTC', 'Volume USD', 'MA', 'UpperBand', 'LowerBand', 'obv', 'RSI']

# Normalize the data
scaler_features = MinMaxScaler()
scaler_target = MinMaxScaler()

scaled_features = scaler_features.fit_transform(new_data[features].values)
scaled_target = scaler_target.fit_transform(new_data[['close']].values)

# Reshape the data for LSTM
scaled_features = np.reshape(scaled_features, (scaled_features.shape[0], scaled_features.shape[1], 1))

# Make predictions
predictions = model.predict(scaled_features)

# Inverse transform the predictions to get the actual values
predictions = scaler_target.inverse_transform(predictions)

print("Predictions : ", predictions)