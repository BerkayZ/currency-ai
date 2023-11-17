import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from datetime import datetime, timedelta

# Load the dataset
data = pd.read_csv('./dataset/prepared_data/btc-min-nonan.csv') 

# Data preprocessing
features = ['open', 'high', 'low', 'close', 'Volume BTC', 'Volume USD', 'MA', 'UpperBand', 'LowerBand', 'obv', 'RSI']
target = 'close'

# Use the last 30 minutes' data to predict the next 30 minutes
prediction_window = 30

# Create a new column for the target variable shifted by the prediction window
data['target'] = data[target].shift(-prediction_window)

# Drop rows with NaN values (resulting from shifting)
data = data.dropna()

# Normalize the data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data[features + ['target']].values)

# Split the data into features and target
X = scaled_data[:, :-1]
y = scaled_data[:, -1]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Reshape the data for LSTM
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Train the model
model.fit(X_train, y_train, batch_size=64, epochs=10, validation_data=(X_test, y_test))

# Save the trained model
model.save('./models/bitcoin_price_model.h5')