import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import joblib
import pickle
import os
from tensorflow.keras.models import load_model
import random


# Specify the path to the CSV file in Google Drive
dataset_file_path = '/content/drive/MyDrive/Datasets/ETC/'

# Specify model and scaler file paths in Google Drive
model_file_path = '/content/drive/MyDrive/Projects/Crypto/lstm_model.h5'
scaler_file_path = '/content/drive/MyDrive/Projects/Crypto/scaler.pkl'


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

def preprocess_data(data, sequence_length=10, scaler=None):
    if scaler is None:
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data[['close']].values)
    else:
        scaled_data = scaler.transform(data[['close']].values)
    X, y = [], []
    for i in range(len(scaled_data) - sequence_length):
        X.append(scaled_data[i:i+sequence_length])
        y.append(scaled_data[i+sequence_length])

    X, y = np.array(X), np.array(y)

    train_size = int(0.8 * len(X))
    X_train, y_train = X[:train_size], y[:train_size]
    X_test, y_test = X[train_size:], y[train_size:]

    return X_train, y_train, X_test, y_test, scaler

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

def train_model(model, X_train, y_train, epochs=100, batch_size=32, validation_split=0.1, save_interval=1,
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


def save_model(model, scaler, model_file='lstm_model.h5', scaler_file='scaler.pkl'):
    model.save(model_file)
    joblib.dump(scaler, scaler_file)
    print(f"Model and scaler saved to {model_file} and {scaler_file}")

def load_saved_model(model_file='lstm_model.h5', scaler_file='scaler.pkl'):
    model = load_model(model_file)
    scaler = joblib.load(scaler_file)
    print(f"Model and scaler loaded from {model_file} and {scaler_file}")
    return model, scaler

def evaluate_model(model, X_test, y_test):
    loss = model.evaluate(X_test, y_test, verbose=0)
    print(f'Test Loss: {loss}')
    return loss

def make_predictions(model, X_test, scaler):
    predicted_prices = model.predict(X_test)
    predicted_prices = scaler.inverse_transform(predicted_prices)
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
    for i in range(10):
        # Specify the path to the CSV file
        files = list_files_in_directory(dataset_file_path)
        rand_filename = files[int(random.random() * len(files))]
        csv_file_path = os.path.join(dataset_file_path, rand_filename)

        # Read data
        _df = read_data(csv_file_path)
        df = pd.concat([df, _df], ignore_index=True)

    # Check if model file exists
    if os.path.exists(model_file_path) and os.path.exists(scaler_file_path):
        # Load the existing model and scaler
        model, scaler = load_saved_model(model_file_path, scaler_file_path)
        X_train, y_train, X_test, y_test, scaler = preprocess_data(df)
        print('[*] Loading Model')
    else:
        # Data Preprocessing
        X_train, y_train, X_test, y_test, scaler = preprocess_data(df)

        # Build and train the model
        model = build_model()

        # Save the model and scaler
        save_model(model, scaler, model_file=model_file_path, scaler_file=scaler_file_path)

        # Load the model and scaler
        loaded_model, loaded_scaler = load_saved_model(model_file=model_file_path, scaler_file=scaler_file_path)
        print('[*] Building Model')

    # Train the model
    history = train_model(model, X_train, y_train, checkpoint_filepath=model_file_path)

    # Evaluate the loaded model
    evaluate_model(model, X_test, y_test)

    # Make predictions using the loaded model and scaler
    loaded_predicted_prices = make_predictions(model, X_test, scaler)

    # Visualize the results
    visualize_results(y_test, loaded_predicted_prices, df.index, len(X_train), 10, 'w_loaded.png')

    # Plot training history
    plot_training_history(history, 'training_history.png')


if __name__ == "__main__":
    while True:
        main()