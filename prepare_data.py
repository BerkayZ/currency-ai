import glob
import pandas as pd
import numpy as np

file_pattern = './dataset/btc-min/*.csv'
files = glob.glob(file_pattern)

df = pd.DataFrame()

for file in files:
    df_temp = pd.read_csv(file)
    df = pd.concat([df, df_temp])

df.sort_values(by='date', inplace=True)
df.reset_index(drop=True, inplace=True)

df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)

# Calculate the True Range (TR)
df['high-low'] = df['high'] - df['low']
df['high-prev_close'] = np.abs(df['high'] - df['close'].shift(1))
df['low-prev_close'] = np.abs(df['low'] - df['close'].shift(1))
df['true_range'] = df[['high-low', 'high-prev_close', 'low-prev_close']].max(axis=1)

# Calculate +DM and -DM
df['plus_dm'] = np.where(df['high'].diff() > df['low'].diff(), df['high'].diff(), 0)
df['minus_dm'] = np.where(df['low'].diff() > df['high'].diff(), df['low'].diff(), 0)

# Calculate the 14-day Smoothed True Range (ATR)
window_size_atr = 14 
df['atr'] = df['true_range'].rolling(window=window_size_atr).mean()

# Calculate the 14-day +DM and -DM
df['plus_dm'] = df['plus_dm'].rolling(window=window_size_atr).mean()
df['minus_dm'] = df['minus_dm'].rolling(window=window_size_atr).mean()

# Calculate the Positive Directional Index (+DI) and Negative Directional Index (-DI)
df['plus_di'] = (df['plus_dm'] / df['atr']) * 100
df['minus_di'] = (df['minus_dm'] / df['atr']) * 100

# Calculate the Directional Movement Index (DX)
df['dx'] = np.abs((df['plus_di'] - df['minus_di']) / (df['plus_di'] + df['minus_di'])) * 100

# Calculate the 14-day Smoothed DX
window_size_adx = 14
df['adx'] = df['dx'].rolling(window=window_size_adx).mean()


# Calculate OBV
df['obv'] = np.where(df['close'] > df['close'].shift(1), df['Volume BTC'], np.where(df['close'] < df['close'].shift(1), -df['Volume BTC'], 0)).cumsum()

# Calculate MA
window_size_ma = 10
df['MA'] = df['close'].rolling(window=window_size_ma).mean()

# Calculate the standard deviation over the same window for Bollinger Bands
df['std'] = df['close'].rolling(window=window_size_ma).std()

# Define the multiplier for Bollinger Bands (typically K=2)
k = 2

# Calculate the upper and lower bands
df['UpperBand'] = df['MA'] + k * df['std']
df['LowerBand'] = df['MA'] - k * df['std']



df['delta'] = df['close'] - df['close'].shift(1)
window_size_rsi = 14  # Adjust the window size for RSI based on your preferences

# Calculate gains and losses
gains = np.where(df['delta'] > 0, df['delta'], 0)
losses = np.where(df['delta'] < 0, -df['delta'], 0)

# Calculate the average of all up moves (AvgU) and down moves (AvgD)
avg_gains = pd.Series(gains).rolling(window=window_size_rsi, min_periods=1).mean().fillna(0)
avg_losses = pd.Series(losses).rolling(window=window_size_rsi, min_periods=1).mean().fillna(0)

# Avoid division by zero
avg_losses[avg_losses == 0] = np.nan

# Calculate relative strength (RS) and convert to NumPy array
rs = avg_gains / avg_losses
rs_values = rs.values


df['RSI'] = 100 - (100 / (1 + rs_values))

# Select columns and save

selected_columns = ['open', 'high', 'low', 'close', 'Volume BTC', 'Volume USD', 'MA', 'UpperBand', 'LowerBand', 'obv', 'RSI']

print(df[selected_columns])

# Create a new DataFrame with only the selected columns
selected_df = df[selected_columns]
selected_df.fillna(0, inplace=True)

# Save the DataFrame to a CSV file
selected_df.to_csv('./dataset/prepared_data/btc-min-nonan.csv', index=True)