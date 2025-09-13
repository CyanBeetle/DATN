import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model, save_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, GRU, Conv1D, MaxPooling1D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import pickle
import os
from datetime import datetime, timedelta
import random

# Import from the new utility file
# Assuming prediction_utils.py is in the same directory or accessible via PYTHONPATH
from prediction_utils import prepare_input_data_for_prediction, generate_predictions_from_input, load_all_models_and_scalers


# Define paths
DATA_PATH = 'Input/synthetic_traffic_dataset.csv'
OUTPUT_DIR = 'processed_data/speed_prediction'
MODEL_DIR = 'processed_data/saved_models'

# Create directories if they don't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

def load_and_prepare_data():
    """
    Load and prepare the data for model training.
    This function is primarily for TRAINING data preparation.
    The utility `prepare_input_data_for_prediction` is for INFERENCE data prep.
    This function will now focus on creating the full feature set for training,
    including one-hot encoding of day_of_week as strings for get_dummies.
    """
    print("Loading and preparing data for TRAINING...")
    
    df = pd.read_csv(DATA_PATH)

    # Ensure essential numerical columns are numeric, coercing errors to NaN
    cols_to_numeric = ['speed_kmh', 'vehicle_count', 'hour', 'minute']
    for col in cols_to_numeric:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Fill NaNs - consider a more robust strategy if needed
    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)
    df.fillna(0, inplace=True) # Fill any remaining NaNs with 0
    
    # Convert timestamp and set as index
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')
    df.set_index('timestamp', inplace=True)

    # Feature Engineering
    # Ensure 'hour' and 'minute' are present from the index if not columns
    if 'hour' not in df.columns: df['hour'] = df.index.hour
    if 'minute' not in df.columns: df['minute'] = df.index.minute
    
    df['day_of_week'] = df.index.dayofweek # 0=Monday, 6=Sunday
    
    # Ensure speed_kmh and vehicle_count are numeric before calculations
    df['speed_kmh'] = pd.to_numeric(df['speed_kmh'], errors='coerce').fillna(0)
    df['vehicle_count'] = pd.to_numeric(df['vehicle_count'], errors='coerce').fillna(0)

    df['traffic_density'] = df['vehicle_count'] / (df['speed_kmh'] + 1e-6)
    df['speed_diff'] = df['speed_kmh'].diff() # NaNs will be handled later
    df['vehicle_diff'] = df['vehicle_count'].diff() # NaNs will be handled later

    # Cyclical time features (using the utility's logic for consistency if desired, or keep local)
    df = add_cyclical_features(df) # Assuming add_cyclical_features is defined below or imported
    
    # One-hot encode day_of_week (as string type for get_dummies)
    # This should match the preparation in `prepare_input_data_for_prediction`
    df['day_of_week_str'] = df['day_of_week'].astype(str)
    df = pd.get_dummies(df, columns=['day_of_week_str'], prefix='day', dtype=int)

    # Handle NaNs created by .diff() or other operations
    df.fillna(method='bfill', inplace=True) # Backfill first to handle initial NaNs from diff
    df.fillna(method='ffill', inplace=True) # Then forward fill
    df.fillna(0, inplace=True) # Fill any remaining NaNs with 0

    print(f"Data prepared for training. Shape: {df.shape}, Columns: {df.columns.tolist()}")
    return df

def add_cyclical_features(df):
    """
    Add cyclical features for time components
    """
    # Ensure source columns exist
    if 'hour' not in df.columns or 'minute' not in df.columns or 'day_of_week' not in df.columns:
        print("Warning: 'hour', 'minute', or 'day_of_week' column missing for cyclical feature generation.")
        # Attempt to create them if index is datetime and they are missing
        if isinstance(df.index, pd.DatetimeIndex):
            if 'hour' not in df.columns: df['hour'] = df.index.hour
            if 'minute' not in df.columns: df['minute'] = df.index.minute
            if 'day_of_week' not in df.columns: df['day_of_week'] = df.index.dayofweek
        else:
            return df # Cannot proceed if columns are missing and index is not datetime
            
    df['hour_sin'] = np.sin(df['hour'] * (2 * np.pi / 24))
    df['hour_cos'] = np.cos(df['hour'] * (2 * np.pi / 24))
    
    df['minute_sin'] = np.sin(df['minute'] * (2 * np.pi / 60))
    df['minute_cos'] = np.cos(df['minute'] * (2 * np.pi / 60))
    
    df['day_of_week_sin'] = np.sin(df['day_of_week'] * (2 * np.pi / 7))
    df['day_of_week_cos'] = np.cos(df['day_of_week'] * (2 * np.pi / 7))
    
    return df

def scale_features(df, features_to_scale, scaler_type='minmax'):
    """
    Scale numerical features
    """
    if scaler_type == 'minmax':
        scaler = MinMaxScaler()
    else:
        scaler = StandardScaler()
    
    # Fit and transform
    df[features_to_scale] = scaler.fit_transform(df[features_to_scale])
    
    return df, scaler

def create_sequences(data, n_steps_in, n_steps_out, granularity=1):
    """
    Create sequences for time series prediction with appropriate granularity
    
    Parameters:
    - data: Input data array with features
    - n_steps_in: Number of lookback steps
    - n_steps_out: Number of prediction steps
    - granularity: Step size for output sequence (1=every step, 5=every 5th step, etc.)
    
    Returns:
    - X: Input sequences
    - y: Target sequences with specified granularity
    """
    X, y = [], []
    
    for i in range(len(data) - n_steps_in - n_steps_out * granularity + 1):
        # Input sequence (look back)
        X.append(data[i:(i + n_steps_in)])
        
        # Target sequence (forecast horizon) with specified granularity
        target_indices = [i + n_steps_in + j * granularity for j in range(n_steps_out)]
        target_sequence = data[target_indices, 0]  # 0 index is speed_kmh
        y.append(target_sequence)
    
    return np.array(X), np.array(y)

def build_lstm_model(input_shape, n_steps_out, model_type='lstm'):
    """
    Build an LSTM model for time series prediction
    """
    model = Sequential()
    model.add(tf.keras.Input(shape=input_shape)) # Add Input layer

    if model_type == 'lstm':
        model.add(LSTM(64, activation='relu', return_sequences=True)) # Removed input_shape
        model.add(Dropout(0.2))
        model.add(LSTM(32, activation='relu'))
        model.add(Dropout(0.2))
    elif model_type == 'bilstm':
        # Simplified architecture for BiLSTM
        model.add(Bidirectional(LSTM(32, activation='relu', return_sequences=True))) # Reduced units
        model.add(Dropout(0.3)) # Slightly increased dropout
        model.add(Bidirectional(LSTM(16, activation='relu')))
        model.add(Dropout(0.3)) # Slightly increased dropout
    elif model_type == 'gru':
        model.add(GRU(64, activation='relu', return_sequences=True)) # Removed input_shape
        model.add(Dropout(0.2))
        model.add(GRU(32, activation='relu'))
        model.add(Dropout(0.2))
    elif model_type == 'cnn-lstm':
        model.add(Conv1D(filters=64, kernel_size=3, activation='relu')) # Removed input_shape
        model.add(MaxPooling1D(pool_size=2))
        model.add(LSTM(64, activation='relu', return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(32, activation='relu'))
        model.add(Dropout(0.2))

    model.add(Dense(n_steps_out))

    # Apply clipnorm for bilstm if it's the selected type for long_term, otherwise standard Adam
    if model_type == 'bilstm': 
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, clipnorm=1.0) # Reduced LR, kept clipnorm
        print(f"Applied Adam optimizer with learning_rate=0.0001 and clipnorm=1.0 for {model_type}")
    else:
        optimizer = tf.keras.optimizers.Adam()
        print(f"Applied standard Adam optimizer for {model_type}")

    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

    return model

def train_model(X_train, y_train, X_val, y_val, model, epochs=100, batch_size=32, model_path=None):
    """
    Train the model with early stopping
    """
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    ]
    
    if model_path:
        callbacks.append(ModelCheckpoint(model_path, save_best_only=True))
    
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )
    
    return model, history

def evaluate_model(model, X_test, y_test, speed_scaler):
    """
    Evaluate the model on test data
    """
    # Get predictions
    y_pred = model.predict(X_test)
    
    # If we're predicting a sequence, we need to invert the scaling for each prediction
    # Create a dummy array with the right shape for inverse transformation
    dummy = np.zeros((y_pred.shape[0], y_pred.shape[1], 1))
    dummy[:, :, 0] = y_pred
    
    # Inverse transform to get back to original scale
    y_pred_orig = speed_scaler.inverse_transform(dummy[:, :, 0])
    
    # Do the same for actual values
    dummy = np.zeros((y_test.shape[0], y_test.shape[1], 1))
    dummy[:, :, 0] = y_test
    y_test_orig = speed_scaler.inverse_transform(dummy[:, :, 0])
    
    # Calculate metrics
    mae = np.mean(np.abs(y_pred_orig - y_test_orig))
    rmse = np.sqrt(np.mean((y_pred_orig - y_test_orig)**2))
    r2 = r2_score(y_test_orig.reshape(-1), y_pred_orig.reshape(-1))
    
    print(f"Test MAE: {mae:.2f} km/h")
    print(f"Test RMSE: {rmse:.2f} km/h")
    print(f"Test R2 Score: {r2:.3f}")
    
    # Calculate metrics for each step in the prediction horizon
    step_maes, step_rmses, step_r2s = [], [], []
    for i in range(y_pred_orig.shape[1]):
        step_mae = np.mean(np.abs(y_pred_orig[:, i] - y_test_orig[:, i]))
        step_rmse = np.sqrt(np.mean((y_pred_orig[:, i] - y_test_orig[:, i])**2))
        step_r2 = r2_score(y_test_orig[:, i], y_pred_orig[:, i])
        print(f"Step {i+1} - MAE: {step_mae:.2f} km/h, RMSE: {step_rmse:.2f} km/h, R2: {step_r2:.3f}")
        step_maes.append(step_mae)
        step_rmses.append(step_rmse)
        step_r2s.append(step_r2)
    
    return y_pred_orig, y_test_orig, mae, rmse, r2, step_maes, step_rmses, step_r2s

def plot_predictions(y_test, y_pred, n_samples=5, title="Predictions vs Actual"):
    """
    Plot predictions vs actual values
    """
    plt.figure(figsize=(15, 8))
    
    for i in range(min(n_samples, len(y_test))):
        plt.subplot(n_samples, 1, i+1)
        plt.plot(y_test[i], 'b-', label='Actual')
        plt.plot(y_pred[i], 'r-', label='Predicted')
        plt.title(f'Sample {i+1}')
        plt.ylabel('Speed (km/h)')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/{title.replace(' ', '_')}.png")
    plt.close()

def define_prediction_horizons():
    """
    Define prediction horizons for the time series model
    """
    horizons = {
        'short_term': {
            'description': 'Short-term speed prediction for immediate traffic management',
            'n_steps_in': 60,  # Last 60 minutes of historical data (1 sample per minute)
            'n_steps_out': 15,  # Next 15 minutes with 1-minute granularity
            'granularity': '1 minute',
            'use_case': 'Immediate route adjustments, fine-tuning very short-term ETAs, alerting to sudden congestion build-up',
            'expected_accuracy': 'High (relies heavily on recent patterns)',
            'model_type': 'cnn-lstm'  # RUN 4: Changed from gru (Run 3)
        },
        'medium_term': {
            'description': 'Medium-term speed prediction for proactive traffic management',
            'n_steps_in': 180,  # Last 3 hours of historical data (1 sample per minute)
            'n_steps_out': 12,  # Next 1 hour with 5-minute granularity
            'granularity': '5 minutes',
            'use_case': 'Tactical route planning for city-level trips, estimating ETAs for trips up to an hour',
            'expected_accuracy': 'Moderate (balances recent patterns with cyclical trends)',
            'model_type': 'lstm'  # RUN 4: Changed from cnn-lstm (Run 3)
        },
        'long_term': {
            'description': 'Long-term speed prediction for strategic traffic planning',
            'n_steps_in': 360,  # Last 6 hours of historical data (1 sample per minute)
            'n_steps_out': 12,  # Next 6 hours with 30-minute granularity
            'granularity': '30 minutes',
            'use_case': 'Strategic departure time planning for longer journeys, general traffic outlook',
            'expected_accuracy': 'Lower (relies more on cyclical patterns, weather, and historical trends)',
            'model_type': 'bilstm'  # RUN 4: Changed from lstm (Run 3)
        },
        'cnn_lstm': { # This key refers to the "CNN_LSTM PREDICTION HORIZON"
            'description': 'Hybrid CNN-LSTM model for feature extraction and sequence prediction',
            'n_steps_in': 120,  # Last 2 hours of historical data
            'n_steps_out': 24,  # Next 2 hours with 5-minute granularity
            'granularity': '5 minutes',
            'use_case': 'Enhanced feature extraction for dense traffic patterns',
            'expected_accuracy': 'High (CNN extracts features that LSTM can use for prediction)',
            'model_type': 'gru'  # RUN 4: Changed from bilstm (Run 3)
        }
    }
    
    return horizons

def main():
    # Step 0: Define prediction horizons
    horizons = define_prediction_horizons()
    for horizon, details in horizons.items():
        print(f"\n{horizon.upper()} PREDICTION HORIZON:")
        for key, value in details.items():
            print(f"  {key}: {value}")
    
    # Load and prepare data
    df = load_and_prepare_data()
    print(f"Loaded data with shape: {df.shape}")
    
    # Add cyclical features
    df = add_cyclical_features(df)
    
    # Store model objects metadata
    trained_models_metadata = {} # Changed from trained_models

    # Phase 2: Build models for each horizon
    for horizon_name, horizon_config in horizons.items():
        print(f"\n\nTraining model for {horizon_name} horizon...")
        
        n_steps_in = horizon_config['n_steps_in']
        n_steps_out = horizon_config['n_steps_out']
        model_type = horizon_config['model_type']
        
        # Determine granularity for this horizon
        if horizon_name == 'short_term':
            granularity = 1  # 1-minute intervals
        elif horizon_name == 'medium_term' or horizon_name == 'cnn_lstm':
            granularity = 5  # 5-minute intervals
        else:  # long_term
            granularity = 30  # 30-minute intervals
            
        # Scale features
        numerical_features = ['speed_kmh', 'vehicle_count']
        
        # Add traffic density if available
        if 'traffic_density' in df.columns:
            numerical_features.append('traffic_density')
        
        # Add speed_diff and vehicle_diff if available
        if 'speed_diff' in df.columns:
            numerical_features.append('speed_diff')
        if 'vehicle_diff' in df.columns:
            numerical_features.append('vehicle_diff')
            
        df_scaled = df.copy()
        
        # Create separate scalers for each feature
        horizon_scalers = {}
        for feature in numerical_features:
            df_scaled, scaler = scale_features(df_scaled, [feature])
            horizon_scalers[feature] = scaler
        
        # Select features for model input
        features = [
            'speed_kmh', 'vehicle_count',
            'hour_sin', 'hour_cos', 'minute_sin', 'minute_cos',
            'day_of_week_sin', 'day_of_week_cos'
        ]
        
        # Add additional features if available
        if 'traffic_density' in df_scaled.columns:
            features.append('traffic_density')
        if 'speed_diff' in df_scaled.columns:
            features.append('speed_diff')
        if 'vehicle_diff' in df_scaled.columns:
            features.append('vehicle_diff')
        
        # Remove one-hot encoded congestion_code features
        # cong_code_cols = [col for col in df_scaled.columns if col.startswith('cong_code_')]
        # features.extend(cong_code_cols) # REMOVED
        
        # If day_of_week was one-hot encoded, it would be added here.
        # Currently using cyclical, so no 'day_' columns are expected unless explicitly created in load_and_prepare_data
        day_cols = [col for col in df_scaled.columns if col.startswith('day_') and 'day_of_week' not in col] # Avoid cyclical
        features.extend(day_cols)
        
        # Prepare data for sequences
        data = df_scaled[features].values
        X, y = create_sequences(data, n_steps_in, n_steps_out, granularity)
        
        print(f"Created sequences with shape X: {X.shape}, y: {y.shape}")
        
        # Split data chronologically
        train_size = int(len(X) * 0.7)
        val_size = int(len(X) * 0.15)
        
        X_train, y_train = X[:train_size], y[:train_size]
        X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
        X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]
        
        # Build and train model
        input_shape = (X_train.shape[1], X_train.shape[2])
        model = build_lstm_model(input_shape, n_steps_out, model_type=model_type)
        
        model_path = f"{MODEL_DIR}/{horizon_name}_model.keras"
        trained_model, history = train_model(X_train, y_train, X_val, y_val, model, model_path=model_path)
        
        # Evaluate model
        y_pred, y_test_actual, mae, rmse, r2, step_maes, step_rmses, step_r2s = evaluate_model(trained_model, X_test, y_test, horizon_scalers['speed_kmh'])
        
        # Plot predictions
        plot_predictions(y_test_actual, y_pred, n_samples=5, title=f"{horizon_name}_predictions")
        
        # ---- Start: New artifact saving logic ----
        horizon_artifact_dir = os.path.join(MODEL_DIR, horizon_name)
        os.makedirs(horizon_artifact_dir, exist_ok=True)

        # Explicit save to new structure
        keras_model_path_structured = os.path.join(horizon_artifact_dir, "model.keras")
        trained_model.save(keras_model_path_structured)
        print(f"Saved Keras model for {horizon_name} to {keras_model_path_structured}")

        # Save scalers for this horizon
        horizon_scalers_path = os.path.join(horizon_artifact_dir, "scalers.pkl")
        save_objects(horizon_scalers, horizon_scalers_path) # horizon_scalers is {feature_name: scaler_obj}
        print(f"Saved scalers for {horizon_name} to {horizon_scalers_path}")

        # Save feature order for this horizon
        features_order_path = os.path.join(horizon_artifact_dir, "features_order.pkl")
        save_objects(features, features_order_path) # features is the list of feature names
        print(f"Saved feature order for {horizon_name} to {features_order_path}")
        
        # Store metadata, referencing the saved artifact paths
        trained_models_metadata[horizon_name] = {
            'n_steps_in': n_steps_in,
            'n_steps_out': n_steps_out,
            'features_order_path': features_order_path,
            'scalers_path': horizon_scalers_path,
            'keras_model_path': keras_model_path_structured,
            'granularity': horizon_config['granularity'],
            'mae': mae,
            'rmse': rmse,
            'r2_score': r2,
            'step_maes': step_maes,
            'step_rmses': step_rmses,
            'step_r2_scores': step_r2s
        }
        # ---- End: New artifact saving logic ----
    
    # Save the global metadata file
    save_objects(trained_models_metadata, f"{MODEL_DIR}/all_models_metadata.pkl")
    print(f"Saved all models metadata to {MODEL_DIR}/all_models_metadata.pkl")
        
    # Test prediction with user input (using the new utility functions)
    print("\n\nTesting prediction with user input using utility functions...")
    
    # Load models and scalers using the utility
    # This will need to be adapted to the new loading mechanism in prediction_utils.py
    loaded_models_for_test, loaded_scalers_for_test = load_all_models_and_scalers(MODEL_DIR)

    if not loaded_models_for_test or not loaded_scalers_for_test:
        print("Failed to load models/scalers for test prediction. Skipping.")
        return # Exit main if loading fails

    # Sample user input
    user_input_test = {
        'day_of_week': 1,  # Tuesday (0=Mon, 1=Tue, ...)
        'time_of_day': '08:30',  # 8:30 AM
        'weather_harsh': True 
    }
    
    # Get recent data for prediction (raw slice from the original dataset)
    # Determine max lookback needed from all loaded models
    max_n_steps_in_for_test = 0
    if loaded_models_for_test:
        max_n_steps_in_for_test = max(model_info.get('n_steps_in', 0) 
                                      for model_info in loaded_models_for_test.values() if model_info)
    
    if max_n_steps_in_for_test == 0:
        print("Could not determine max_n_steps_in for test prediction. Using a default of 60.")
        max_n_steps_in_for_test = 60 # Default if no models or info found

    # Load a small slice of the original data to simulate "recent_data"
    # df_full_for_test_slicing = pd.read_csv(DATA_PATH) # df is already loaded and prepared for training
    
    # We need a raw-like slice that prepare_input_data_for_prediction can process.
    # The `df` variable at this stage in `main()` is already heavily processed for training.
    # For a realistic test of `prepare_input_data_for_prediction`, we should load a fresh raw slice.
    df_raw_for_test_slice = pd.read_csv(DATA_PATH)
    
    # Take a tail slice. Add a small buffer for diff calculations within prepare_input_data_for_prediction.
    # The number of rows should be at least max_n_steps_in_for_test + a few for diffs.
    buffer_for_diffs = 5 
    num_rows_to_slice = max_n_steps_in_for_test + buffer_for_diffs
    
    if len(df_raw_for_test_slice) < num_rows_to_slice:
        print(f"Warning: Not enough raw data for test slice. Required {num_rows_to_slice}, got {len(df_raw_for_test_slice)}. Using all available.")
        recent_raw_data_for_test = df_raw_for_test_slice.copy()
    else:
        recent_raw_data_for_test = df_raw_for_test_slice.tail(num_rows_to_slice).copy()

    # Prepare this raw slice using the utility function (this is crucial for testing consistency)
    recent_prepared_data_for_test = prepare_input_data_for_prediction(recent_raw_data_for_test)

    if recent_prepared_data_for_test is None:
        print("Failed to prepare recent data for test prediction. Skipping test.")
        return

    # Make predictions using the utility function
    test_predictions = generate_predictions_from_input(
        models_dict=loaded_models_for_test,
        scalers_dict=loaded_scalers_for_test,
        recent_data_unscaled_with_features=recent_prepared_data_for_test, # Pass the prepared data
        user_input_conditions=user_input_test
    )
    
    # Display predictions
    print("\n--- Test Predictions from speed_prediction_system.py using prediction_utils ---")
    for horizon, pred_info in test_predictions.items():
        print(f"\n{horizon.upper()} PREDICTIONS (Weather: {'Harsh' if pred_info['weather_effect'] else 'Normal'}):")
        if 'timestamps' in pred_info and 'speeds' in pred_info:
            for ts, speed in zip(pred_info['timestamps'], pred_info['speeds']):
                print(f"  {ts}: {speed:.2f} km/h")
        else:
            print(f"  No prediction data available for {horizon}.")

def save_objects(obj, filename):
    """
    Save objects using pickle
    """
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)

def load_objects(filename):
    """
    Load objects using pickle
    """
    with open(filename, 'rb') as f:
        return pickle.load(f)

if __name__ == "__main__":
    main()