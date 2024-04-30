import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import joblib
import pickle
import os
from tensorflow.keras.models import load_model
import random
import time

# Specify the path to the CSV file in Google Drive
dataset_file_path = '/content/drive/MyDrive/Datasets/ETC/'

# Specify model file path in Google Drive
model_file_path = '/content/drive/MyDrive/Projects/Crypto/lstm_model.h5'

def list_files_in_directory(directory_path):
    try:
        # Get the list of files in the directory
        files = [f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))]
        return files
    except OSError as e:
        print(f"Error reading directory {directory_path}: {e}")
        return None

def read_data(file_path):
    data = pd.read_csv(file_path)
    print(data.columns)
    return data

def preprocess_data(data, sequence_length=10):
    # Normalize 'close' column values between 0 and 1
    data['close'] = (data['close'] - data['close'].min()) / (data['close'].max() - data['close'].min())

    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data['close'].values[i:i + sequence_length])
        y.append(data['close'].values[i + sequence_length])

    X, y = np.array(X), np.array(y)

    train_size = int(0.8 * len(X))
    X_train, y_train = X[:train_size], y[:train_size]
    X_test, y_test = X[train_size:], y[train_size:]

    return X_train, y_train, X_test, y_test


def build_model(sequence_length=10):
    model = Sequential([
        LSTM(50, activation='relu', input_shape=(sequence_length, 1), return_sequences=True),
        Dropout(0.2),
        LSTM(50, activation='relu', return_sequences=True),
        Dropout(0.2),
        LSTM(50, activation='relu'),
        Dropout(0.2),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse')
    return model

def train_model(model, X_train, y_train, epochs=10, batch_size=32, validation_split=0.1, save_interval=1,
                checkpoint_filepath=None):
    # Define the ModelCheckpoint callback
    # checkpoint_filepath = 'checkpoint_model.h5'
    model_checkpoint_callback = ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=False,
        save_best_only=True,
        monitor='val_loss',
        mode='min',
        period=save_interval  # Save every 'save_interval' epochs
    )

    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
        validation_split=validation_split,
        callbacks=[early_stopping, model_checkpoint_callback]
    )

    return history


def save_model(model, model_file='lstm_model.h5'):
    model.save(model_file)
    print(f"Model saved to {model_file}")

def load_saved_model(model_file='lstm_model.h5'):
    model = load_model(model_file)
    print(f"Model loaded from {model_file}")
    return model

def evaluate_model(model, X_test, y_test):
    loss = model.evaluate(X_test, y_test, verbose=0)
    print(f'Test Loss: {loss}')
    return loss

def make_predictions(model, X_test):
    predicted_prices = model.predict(X_test)
    return predicted_prices

def visualize_results(actual_prices, predicted_prices, index, train_size, sequence_length, filename):
    plt.figure(figsize=(12, 6))
    plt.plot(index[train_size + sequence_length:], actual_prices, label='Actual Prices', color='blue')
    plt.plot(index[train_size + sequence_length:], predicted_prices, label='Predicted Prices', color='red')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Cryptocurrency Price Prediction with LSTM')
    plt.legend()
    plt.savefig(filename)

def plot_training_history(history, filename):
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training History')
    plt.legend()
    plt.savefig(filename)

def main():
    df = pd.DataFrame()
    for i in range(1):
        # Specify the path to the CSV file
        files = list_files_in_directory(dataset_file_path)
        files = sorted(files)
        for file in files:
            # rand_filename = files[int(random.random() * len(files))]
            csv_file_path = os.path.join(dataset_file_path, file)

            # Read data
            _df = read_data(csv_file_path)
            df = pd.concat([df, _df], ignore_index=True)

    # Check if model file exists
    if os.path.exists(model_file_path):
        # Load the existing model
        model = load_saved_model(model_file_path)
        X_train, y_train, X_test, y_test = preprocess_data(df)
        print('[*] Loading Model')
    else:
        # Data Preprocessing
        X_train, y_train, X_test, y_test = preprocess_data(df)

        # Build and train the model
        model = build_model()

        # Save the model
        save_model(model, model_file=model_file_path)

        # Load the model
        model = load_saved_model(model_file=model_file_path)
        print('[*] Building Model')

    # Train the model
    history = train_model(model, X_train, y_train, checkpoint_filepath=model_file_path)

    # Evaluate the loaded model
    evaluate_model(model, X_test, y_test)

    # Make predictions using the loaded model
    loaded_predicted_prices = make_predictions(model, X_test)

    # Visualize the results
    visualize_results(y_test, loaded_predicted_prices, df.index, len(X_train), 10, f'w_loaded_{time.time()}.png')
    visualize_results(y_test, loaded_predicted_prices, df.index, len(X_train), 10, f'/content/drive/MyDrive/Projects/Crypto/w_loaded_{time.time()}.png')

    # Plot training history
    plot_training_history(history, f'training_history_{time.time()}.png')
    plot_training_history(history, f'/content/drive/MyDrive/Projects/Crypto/training_history_{time.time()}.png')


if __name__ == "__main__":
    while True:
        main()