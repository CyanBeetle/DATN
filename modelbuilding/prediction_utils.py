import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime, timedelta
import random
from tensorflow.keras.models import load_model

# Default model directory, assuming script execution from 'modelbuilding' parent
# It's often better to pass this path as an argument to functions.
DEFAULT_MODEL_DIR = 'processed_data/saved_models'

def load_all_models_and_scalers(model_base_dir=DEFAULT_MODEL_DIR):
    """
    Load all trained Keras model objects, their specific scalers, and feature orders
    by scanning subdirectories in model_base_dir.
    Each subdirectory is assumed to be a horizon_name and contain:
    - model.keras
    - scalers.pkl (dictionary of feature_name -> scaler_object)
    - features_order.pkl (list of feature names)
    """
    loaded_models_info = {}  # This will be models_dict for generate_predictions_from_input
    loaded_scalers_for_horizons = {}  # This will be scalers_dict for generate_predictions_from_input

    if not os.path.isdir(model_base_dir):
        print(f"Error: Model base directory not found: {model_base_dir}")
        return None, None

    print(f"Scanning for model artifacts in: {model_base_dir}")
    for horizon_name in os.listdir(model_base_dir):
        horizon_artifact_dir = os.path.join(model_base_dir, horizon_name)
        if os.path.isdir(horizon_artifact_dir):
            # Skip files like 'all_models_metadata.pkl' at the root of MODEL_DIR
            if not (os.path.exists(os.path.join(horizon_artifact_dir, "model.keras")) and 
                    os.path.exists(os.path.join(horizon_artifact_dir, "scalers.pkl")) and 
                    os.path.exists(os.path.join(horizon_artifact_dir, "features_order.pkl"))):
                if horizon_name != "all_models_metadata.pkl": # Common to log if it's an unexpected dir struct
                    print(f"Skipping directory {horizon_artifact_dir} as it does not appear to be a complete model artifact directory (missing model.keras, scalers.pkl, or features_order.pkl).")
                continue # Skip if it's not a directory containing all expected artifacts
            
            print(f"Processing artifact directory: {horizon_artifact_dir}")
            try:
                keras_model_path = os.path.join(horizon_artifact_dir, "model.keras")
                scalers_path = os.path.join(horizon_artifact_dir, "scalers.pkl")
                features_order_path = os.path.join(horizon_artifact_dir, "features_order.pkl")

                keras_model = load_model(keras_model_path)
                
                with open(scalers_path, 'rb') as f:
                    horizon_scalers = pickle.load(f)  # {feature_name: scaler_obj}
                
                with open(features_order_path, 'rb') as f:
                    feature_order_list = pickle.load(f)  # list of feature names

                n_steps_in = keras_model.input_shape[1]
                n_steps_out = keras_model.output_shape[1]
                
                # Infer granularity (this part is heuristic; ideally, save/load explicitly from metadata if available)
                granularity_str = '1 minute'  # Default
                if "short_term" in horizon_name.lower(): granularity_str = '1 minute'
                elif "15" in horizon_name.lower() and ("min" in horizon_name.lower() or "minute" in horizon_name.lower()): granularity_str = '1 minute' # For 15 min output steps
                elif "medium_term" in horizon_name.lower(): granularity_str = '5 minutes'
                elif "1hour" in horizon_name.lower() or ("60" in horizon_name.lower() and ("min" in horizon_name.lower() or "minute" in horizon_name.lower())): granularity_str = '5 minutes' # For 1-hour horizon, 5-min steps
                elif "long_term" in horizon_name.lower(): granularity_str = '30 minutes'
                elif "6hour" in horizon_name.lower(): granularity_str = '30 minutes' # For 6-hour horizon, 30-min steps
                elif "cnn_lstm" in horizon_name.lower(): granularity_str = '5 minutes'
                # More specific granularity parsing if model names are structured e.g. short_term_1min_15steps

                loaded_models_info[horizon_name] = {
                    'model': keras_model,
                    'n_steps_in': n_steps_in,
                    'n_steps_out': n_steps_out,
                    'features_order': feature_order_list,
                    'granularity': granularity_str
                }
                loaded_scalers_for_horizons[horizon_name] = horizon_scalers

                print(f"Successfully loaded artifacts for horizon: {horizon_name}")

            except Exception as e:
                print(f"Error loading artifacts for horizon {horizon_name} from {horizon_artifact_dir}: {e}")
    
    if not loaded_models_info:
        print(f"No model artifacts successfully loaded from {model_base_dir}. Check directory structure and file integrity.")
        return None, None
        
    return loaded_models_info, loaded_scalers_for_horizons

def prepare_input_data_for_prediction(df_recent_raw):
    """
    Prepares a raw DataFrame slice (e.g., recent data) for prediction.
    This includes deriving time features, feature engineering (density, diffs),
    cyclical features, and one-hot encoding for day of the week.
    The output DataFrame is unscaled but has all structural features.
    """
    df = df_recent_raw.copy()

    # Ensure timestamp is datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        df.set_index('timestamp', inplace=True)
    else: # If index is already datetime, ensure it's sorted
        df.sort_index(inplace=True)

    # Derive hour, minute if not present
    if 'hour' not in df.columns:
        df['hour'] = df.index.hour
    if 'minute' not in df.columns:
        df['minute'] = df.index.minute
    
    # Derive day_of_week
    df['day_of_week'] = df.index.dayofweek # 0=Monday, 6=Sunday

    # Ensure numeric types for key features and fill NaNs from coercion
    df['speed_kmh'] = pd.to_numeric(df['speed_kmh'], errors='coerce').fillna(0)
    df['vehicle_count'] = pd.to_numeric(df['vehicle_count'], errors='coerce').fillna(0)

    # Feature Engineering (must match training feature engineering)
    df['traffic_density'] = df['vehicle_count'] / (df['speed_kmh'] + 1e-6) # Add epsilon to avoid division by zero
    df['speed_diff'] = df['speed_kmh'].diff() 
    df['vehicle_diff'] = df['vehicle_count'].diff()

    # Cyclical time features
    df['hour_sin'] = np.sin(df['hour'] * (2 * np.pi / 24))
    df['hour_cos'] = np.cos(df['hour'] * (2 * np.pi / 24))
    df['minute_sin'] = np.sin(df['minute'] * (2 * np.pi / 60))
    df['minute_cos'] = np.cos(df['minute'] * (2 * np.pi / 60))
    df['day_of_week_sin'] = np.sin(df['day_of_week'] * (2 * np.pi / 7))
    df['day_of_week_cos'] = np.cos(df['day_of_week'] * (2 * np.pi / 7))
    
    # One-hot encode day_of_week (consistent with training: day_0, day_1, ...)
    df['day_of_week_str'] = df['day_of_week'].astype(str) 
    
    # Ensure all day categories (0-6) are known for pd.get_dummies
    all_day_categories = [str(i) for i in range(7)]
    df['day_of_week_str'] = pd.Categorical(df['day_of_week_str'], categories=all_day_categories)
    
    df = pd.get_dummies(df, columns=['day_of_week_str'], prefix='day', dtype=int)

    # Handle NaNs created by .diff() on the first row(s)
    # Fill NaNs: bfill for initial diff NaNs, then ffill, then 0 for any stubborn ones.
    df.fillna(method='bfill', inplace=True)
    df.fillna(method='ffill', inplace=True)
    df.fillna(0, inplace=True)
    
    return df

def generate_predictions_from_input(models_dict, scalers_dict, recent_data_unscaled_with_features, user_input_conditions):
    """
    Generates speed predictions for multiple horizons using loaded models and scalers.
    'recent_data_unscaled_with_features' must be prepared by 'prepare_input_data_for_prediction'.
    'user_input_conditions' is a dict: {'day_of_week', 'time_of_day', 'weather_harsh'}.
    """
    predictions_output = {}
    
    day_of_week_input = user_input_conditions['day_of_week'] # Numeric 0-6
    hour_input, minute_input = map(int, user_input_conditions['time_of_day'].split(':'))
    weather_harsh_input = user_input_conditions['weather_harsh'] # Boolean
    
    # Work on a copy of the recent data
    recent_data_for_processing = recent_data_unscaled_with_features.copy()
    
    # Apply weather effect if harsh (modifies the unscaled copy)
    if weather_harsh_input:
        reduction_factor = 1.0 - (random.uniform(15, 25) / 100)
        if 'speed_kmh' in recent_data_for_processing.columns:
            recent_data_for_processing['speed_kmh'] *= reduction_factor
        if 'vehicle_count' in recent_data_for_processing.columns:
            recent_data_for_processing['vehicle_count'] *= reduction_factor
        # Recalculate dependent features if they exist and are affected
        if ('traffic_density' in recent_data_for_processing.columns and
            'speed_kmh' in recent_data_for_processing.columns and
            'vehicle_count' in recent_data_for_processing.columns):
            recent_data_for_processing['traffic_density'] = (
                recent_data_for_processing['vehicle_count'] /
                (recent_data_for_processing['speed_kmh'] + 1e-6)
            )
        # Note: Diffs are historical; weather effect is applied to current state representation.

    for horizon_name, model_config in models_dict.items():
        if model_config.get('model') is None:
            print(f"Skipping prediction for {horizon_name} as Keras model is not loaded.")
            continue

        model_object = model_config['model']
        n_steps_in_model = model_config['n_steps_in']
        n_steps_out_model = model_config['n_steps_out']
        features_order_model = model_config['features_order']
        
        current_horizon_scalers = scalers_dict.get(horizon_name)
        if not current_horizon_scalers:
            print(f"Warning: Scalers not found for horizon {horizon_name}. Skipping prediction.")
            continue

        # Ensure enough data points for the lookback window
        if len(recent_data_for_processing) < n_steps_in_model:
            print(f"Warning: Not enough recent data for {horizon_name} prediction. Need {n_steps_in_model} points, got {len(recent_data_for_processing)}.")
            continue
            
        # Get the most recent n_steps_in data points (unscaled but with all structural features)
        sequence_for_input_unscaled = recent_data_for_processing.tail(n_steps_in_model).copy()
        
        # Scale the features in this sequence that were scaled during training
        X_input_scaled_sequence = sequence_for_input_unscaled.copy()
        for feature_name_to_scale, scaler_obj in current_horizon_scalers.items():
            if feature_name_to_scale in X_input_scaled_sequence.columns:
                # Scaler expects a 2D array
                X_input_scaled_sequence[[feature_name_to_scale]] = scaler_obj.transform(X_input_scaled_sequence[[feature_name_to_scale]])
            else:
                # This warning is important if a feature expected by a scaler is missing
                print(f"Warning: Feature '{feature_name_to_scale}' expected by scaler not found in input sequence for horizon '{horizon_name}'.")

        # Override time features for the entire input window with the user's current time context
        # This is a common approach if models are trained to expect current time context repeated over the lookback.
        X_input_scaled_sequence['hour_sin'] = np.sin(hour_input * (2 * np.pi / 24))
        X_input_scaled_sequence['hour_cos'] = np.cos(hour_input * (2 * np.pi / 24))
        X_input_scaled_sequence['minute_sin'] = np.sin(minute_input * (2 * np.pi / 60))
        X_input_scaled_sequence['minute_cos'] = np.cos(minute_input * (2 * np.pi / 60))
        X_input_scaled_sequence['day_of_week_sin'] = np.sin(day_of_week_input * (2 * np.pi / 7))
        X_input_scaled_sequence['day_of_week_cos'] = np.cos(day_of_week_input * (2 * np.pi / 7))

        # One-hot encode day_of_week for the current prediction time, matching training (e.g., day_0, day_1)
        for i in range(7):
            day_col_name = f'day_{i}' # Assumes prefix 'day' and original was 0-6 string
            if day_col_name in features_order_model: # Only set if model expects this feature
                 X_input_scaled_sequence[day_col_name] = 1 if day_of_week_input == i else 0
        
        # Ensure all features expected by the model are present, fill with 0 if not (e.g., a day_X not set by loop)
        for feature_in_model_order in features_order_model:
            if feature_in_model_order not in X_input_scaled_sequence.columns:
                X_input_scaled_sequence[feature_in_model_order] = 0 
                
        # Select features in the exact order the model was trained on
        X_input_final_features = X_input_scaled_sequence[features_order_model]
        
        # Reshape for model input (samples, timesteps, features)
        X_input_array = X_input_final_features.values.reshape(1, n_steps_in_model, len(features_order_model))
        
        # Make prediction
        y_pred_scaled = model_object.predict(X_input_array, verbose=0) # verbose=0 for cleaner output
        
        # Inverse transform the prediction (target is 'speed_kmh')
        speed_scaler_for_horizon = current_horizon_scalers.get('speed_kmh')
        if not speed_scaler_for_horizon:
            print(f"Warning: Speed scaler not found for horizon {horizon_name}. Cannot unscale predictions.")
            continue

        # Model might output (1, n_steps_out) or (n_steps_out,). Ensure 2D for scaler.
        if y_pred_scaled.ndim == 1: 
            y_pred_scaled = y_pred_scaled.reshape(-1, 1) # Make it (n_steps_out, 1)
        elif y_pred_scaled.shape[0] == 1 : # if (1, n_steps_out)
            y_pred_scaled = y_pred_scaled.reshape(-1,1) # Make it (n_steps_out, 1)
        
        y_pred_unscaled_flat = speed_scaler_for_horizon.inverse_transform(y_pred_scaled)
        y_pred_unscaled_list = y_pred_unscaled_flat.flatten().tolist() # Get 1D list of speeds

        # Create timestamps for predictions based on horizon granularity
        start_time = datetime.strptime(f"{hour_input:02d}:{minute_input:02d}:00", "%H:%M:%S")
        time_granularity_str = model_config.get('granularity', '1 minute') # Default if not specified

        if time_granularity_str == '1 minute':
            timestamps = [start_time + timedelta(minutes=i+1) for i in range(n_steps_out_model)]
        elif time_granularity_str == '5 minutes':
            timestamps = [start_time + timedelta(minutes=(i+1)*5) for i in range(n_steps_out_model)]
        elif time_granularity_str == '30 minutes':
            timestamps = [start_time + timedelta(minutes=(i+1)*30) for i in range(n_steps_out_model)]
        else: # Default to 1-minute if granularity string is unknown
            print(f"Warning: Unknown granularity '{time_granularity_str}' for {horizon_name}. Defaulting to 1 minute.")
            timestamps = [start_time + timedelta(minutes=i+1) for i in range(n_steps_out_model)]
        
        timestamps_str_list = [t.strftime("%H:%M:%S") for t in timestamps]
        
        # Store predictions
        predictions_output[horizon_name] = {
            'timestamps': timestamps_str_list,
            'speeds': y_pred_unscaled_list,
            'granularity': time_granularity_str,
            'weather_effect': weather_harsh_input # Record if weather effect was applied
        }
    
    return predictions_output
