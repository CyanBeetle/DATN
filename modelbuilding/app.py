from flask import Flask, render_template, request, jsonify, redirect, url_for
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import pickle
import matplotlib
matplotlib.use('Agg') # Use Agg backend for Matplotlib to avoid GUI errors in headless environments
import matplotlib.pyplot as plt
import io
import base64
import sys
import random
import tensorflow as tf

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from prediction_utils import prepare_input_data_for_prediction

app = Flask(__name__)

MODEL_DIR = '../backendv3/ModelStorage/Set1/saved_models/' # Updated to target Set1 specifically
OUTPUT_DIR = 'processed_data/predictions' # Kept for potential future use
DATA_PATH = 'Input/synthetic_traffic_dataset.csv'
# SCALER_FILENAME = 'speed_scaler.pkl' # Removed - scalers are model-specific

# STANDARD_FEATURES_ORDER is removed as feature order will be loaded per model.

os.makedirs(OUTPUT_DIR, exist_ok=True)
# MODEL_DIR is now expected to exist and be populated by the reorganize script.
# os.makedirs(MODEL_DIR, exist_ok=True) 
os.makedirs('templates', exist_ok=True)

def create_templates():
    """
    Create templates directory and HTML files if they don't exist.
    Simplified for Keras file selection.
    """
    # Create templates directory
    os.makedirs('templates', exist_ok=True)
    
    # Create index.html (Simplified)
    index_html = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Traffic Speed Prediction</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.2.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body { padding-top: 20px; padding-bottom: 40px; background-color: #f5f5f5; }
        .container { max-width: 700px; }
        .heading-container { text-align: center; margin-bottom: 30px; }
        .prediction-form { background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }
    </style>
</head>
<body>
    <div class="container">
        <div class="heading-container">
            <h1>Traffic Speed Prediction</h1>
            <p class="lead">Select a model and provide input to generate predictions.</p>
        </div>
        
        {% if no_models_found %}
            <div class="alert alert-warning">
                No compatible models found in <strong>{{ model_dir_path }}</strong>. 
                Please ensure models are reorganized with model.keras, scalers.pkl (dictionary), and features_order.pkl in per-horizon subdirectories.
            </div>
        {% else %}
            {% if error %}
                <div class="alert alert-danger">
                    Error: {{ error }}
                </div>
            {% endif %}
            
            <div class="prediction-form">
                <h3>Make a Prediction</h3>
                <form action="/predict" method="post">
                    <div class="mb-3">
                        <label for="keras_model_file" class="form-label">Select Model (Horizon):</label>
                        <select class="form-select" id="keras_model_file" name="keras_model_file" required>
                            {% for model_info in available_models %}
                                <option value="{{ model_info.id }}">{{ model_info.display_name }}</option>
                            {% endfor %}
                        </select>
                    </div>
                    <div class="mb-3">
                        <label for="day_of_week" class="form-label">Day of Week:</label>
                        <select class="form-select" id="day_of_week" name="day_of_week" required>
                            <option value="0">Monday</option>
                            <option value="1">Tuesday</option>
                            <option value="2">Wednesday</option>
                            <option value="3">Thursday</option>
                            <option value="4">Friday</option>
                            <option value="5">Saturday</option>
                            <option value="6">Sunday</option>
                        </select>
                    </div>
                    <div class="mb-3">
                        <label for="time_of_day" class="form-label">Time of Day (HH:MM):</label>
                        <input type="text" class="form-control" id="time_of_day" name="time_of_day" 
                               placeholder="e.g. 08:30" required pattern="^([01]?[0-9]|2[0-3]):[0-5][0-9]$">
                    </div>
                    <div class="mb-3 form-check">
                        <input type="checkbox" class="form-check-input" id="weather_harsh" name="weather_harsh">
                        <label class="form-check-label" for="weather_harsh">Harsh Weather Conditions</label>
                    </div>
                    <button type="submit" class="btn btn-primary w-100">Predict</button>
                </form>
            </div>
        {% endif %}
    </div>
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.2.3/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
    '''
    with open('templates/index.html', 'w', encoding='utf-8') as f:
        f.write(index_html)
    
    # Create prediction.html (Simplified for single model output)
    prediction_html = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Prediction Results</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.2.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body { padding-top: 20px; padding-bottom: 40px; background-color: #f5f5f5; }
        .container { max-width: 800px; }
        .heading-container { text-align: center; margin-bottom: 30px; }
        .prediction-card { background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 0 10px rgba(0,0,0,0.1); margin-bottom: 30px; }
        .plot-container img { max-width: 100%; height: auto; border-radius: 5px; margin-bottom: 20px; }
    </style>
</head>
<body>
    <div class="container">
        <div class="heading-container">
            <h1>Prediction Results</h1>
        </div>

        {% if error %}
            <div class="alert alert-danger">
                Error: {{ error }}
            </div>
            <div class="text-center mt-3">
                <a href="/" class="btn btn-primary">Try Again</a>
            </div>
        {% else %}
            <div class="prediction-card">
                <h3>Model: {{ model_file_name }}</h3>
                <p><strong>Input Conditions:</strong> Day: {{ day_name }}, Time: {{ user_input.time_of_day }}, Weather: {{ 'Harsh' if user_input.weather_harsh else 'Normal' }}</p>
                
                {% if plot_image %}
                <div class="plot-container text-center">
                    <img src="data:image/png;base64,{{ plot_image }}" alt="Prediction Plot">
                </div>
                {% endif %}
                
                <h4>Predicted Speeds ({{ granularity_display }}):</h4>
                <div class="table-responsive">
                    <table class="table table-sm table-striped table-bordered">
                        <thead><tr><th>Time</th><th>Speed (km/h)</th></tr></thead>
                        <tbody>
                            {% for time, speed in zip(predicted_timestamps, predicted_speeds) %}
                                <tr><td>{{ time }}</td><td>{{ "%.2f"|format(speed) }}</td></tr>
                            {% endfor %}
                        </tbody>
                    </table>
                </div>
            </div>
            <div class="text-center mt-3">
                <a href="/" class="btn btn-primary">Make Another Prediction</a>
            </div>
        {% endif %}
    </div>
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.2.3/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
    '''
    with open('templates/prediction.html', 'w', encoding='utf-8') as f:
        f.write(prediction_html)
    
    # models.html is removed, so no need to create it here.

# Call create_templates directly at script startup
create_templates()

def get_available_models(): # Renamed and refactored
    """
    Get the list of available models by scanning subdirectories in MODEL_DIR.
    Each subdirectory is a horizon and should contain model.keras, scalers.pkl, features_order.pkl.
    """
    available_models_list = []
    if not os.path.exists(MODEL_DIR) or not os.path.isdir(MODEL_DIR):
        print(f"Warning: Model directory {MODEL_DIR} not found or is not a directory.")
        return []

    for horizon_id in os.listdir(MODEL_DIR):
        horizon_dir_path = os.path.join(MODEL_DIR, horizon_id)
        if os.path.isdir(horizon_dir_path):
            model_path = os.path.join(horizon_dir_path, 'model.keras')
            scalers_path = os.path.join(horizon_dir_path, 'scalers.pkl') # Dict of scalers
            features_order_path = os.path.join(horizon_dir_path, 'features_order.pkl')

            if os.path.exists(model_path) and os.path.exists(scalers_path) and os.path.exists(features_order_path):
                # Heuristic for display name and inferred horizon
                display_name_parts = horizon_id.replace('_', ' ').split('-')
                display_name = ' '.join(s.capitalize() for s in display_name_parts)
                
                inferred_horizon_display = "Unknown Horizon"
                if "short_term" in horizon_id.lower() or "15min" in horizon_id.lower() or "10_min" in horizon_id.lower() or "30_min" in horizon_id.lower() :
                    inferred_horizon_display = f"{display_name} (Short-Term)"
                elif "medium_term" in horizon_id.lower() or "1hour" in horizon_id.lower() or "60min" in horizon_id.lower():
                    inferred_horizon_display = f"{display_name} (Medium-Term)"
                elif "long_term" in horizon_id.lower() or "2hour" in horizon_id.lower() or "6hour" in horizon_id.lower() or "12hour" in horizon_id.lower():
                    inferred_horizon_display = f"{display_name} (Long-Term)"
                elif "cnn_lstm" in horizon_id.lower(): # Specific case for cnn_lstm model name
                     inferred_horizon_display = f"{display_name} (CNN-LSTM)"
                else:
                    inferred_horizon_display = display_name

                available_models_list.append({
                    'id': horizon_id, # The subdirectory name, e.g., "short_term"
                    'display_name': inferred_horizon_display,
                    'model_path': model_path,
                    'scalers_path': scalers_path,
                    'features_order_path': features_order_path
                })
            else:
                print(f"Warning: Skipping directory {horizon_dir_path} as it's missing one or more required files (model.keras, scalers.pkl, features_order.pkl).")
                
    return sorted(available_models_list, key=lambda x: x['display_name'])

@app.route('/')
def index():
    """
    Render the home page with a list of available models.
    """
    available_models = get_available_models()
    if not available_models:
        full_model_dir_path = os.path.abspath(MODEL_DIR)
        return render_template('index.html', no_models_found=True, model_dir_path=full_model_dir_path)
    return render_template('index.html', available_models=available_models, error=request.args.get('error'))

def generate_future_timestamps(start_time_str, n_steps_out, granularity_minutes):
    start_time = datetime.strptime(start_time_str, "%H:%M")
    return [(start_time + timedelta(minutes=i * granularity_minutes)).strftime("%H:%M") for i in range(n_steps_out)]

@app.route('/predict', methods=['POST'])
def predict():
    try:
        day_of_week = int(request.form['day_of_week'])
        time_of_day_str = request.form['time_of_day']
        weather_harsh = 'weather_harsh' in request.form
        selected_model_id = request.form['keras_model_file'] # This is now the horizon_id (directory name)
        
        all_available_models = get_available_models()
        selected_model_info = next((m for m in all_available_models if m['id'] == selected_model_id), None)

        if not selected_model_info:
            return render_template('index.html', error=f"Selected model '{selected_model_id}' not found or misconfigured.", available_models=all_available_models)

        model_path = selected_model_info['model_path']
        scalers_path = selected_model_info['scalers_path'] # Path to the dict of scalers
        features_order_path = selected_model_info['features_order_path']
        model_display_name = selected_model_info['display_name']

        # Basic file existence checks (already done by get_available_models, but good for robustness)
        if not os.path.exists(model_path) or not os.path.exists(scalers_path) or not os.path.exists(features_order_path):
             return render_template('index.html', error=f"One or more artifact files are missing for model {model_display_name}.", available_models=all_available_models)

        # Load Keras Model
        model = tf.keras.models.load_model(model_path, compile=False) 
        # It's good practice to compile if you need to evaluate or if the model wasn't saved compiled
        # However, for prediction only, compile=False is fine and sometimes necessary if custom objects aren't handled.
        # model.compile(optimizer='adam', loss='mse', metrics=['mae']) # Optional: re-compile if needed

        # Load the dictionary of scalers for this specific model
        with open(scalers_path, 'rb') as f:
            horizon_scalers_dict = pickle.load(f) # e.g., {'speed_kmh': scaler_obj, 'vehicle_count': scaler_obj}
            if not isinstance(horizon_scalers_dict, dict):
                return render_template('index.html', error=f"Scaler file {scalers_path} for {model_display_name} did not contain a dictionary.", available_models=all_available_models)
            if 'speed_kmh' not in horizon_scalers_dict or not hasattr(horizon_scalers_dict['speed_kmh'], 'inverse_transform'):
                 return render_template('index.html', error=f"Speed scaler ('speed_kmh') missing or invalid in {scalers_path} for {model_display_name}.", available_models=all_available_models)

        # Load the feature order list for this specific model
        with open(features_order_path, 'rb') as f:
            model_features_order = pickle.load(f) # List of feature names
            if not isinstance(model_features_order, list):
                return render_template('index.html', error=f"Feature order file {features_order_path} for {model_display_name} did not contain a list.", available_models=all_available_models)


        n_steps_in = model.input_shape[1]
        n_steps_out = model.output_shape[1] # Assuming model output shape is (batch, n_steps_out) or (batch, n_steps_out, 1)
        num_model_expected_features = model.input_shape[2]
        
        raw_data_df = pd.read_csv(DATA_PATH)
        # Ensure enough data for lookback plus buffer for feature engineering (e.g., diffs)
        required_raw_rows = n_steps_in + 20 # Buffer for robust feature gen by prepare_input_data...
        if len(raw_data_df) < required_raw_rows: 
             return render_template('index.html', error=f"Not enough data in {DATA_PATH} (need {required_raw_rows}, have {len(raw_data_df)}) for lookback {n_steps_in}.", available_models=all_available_models)
        
        recent_raw_data = raw_data_df.tail(required_raw_rows).copy()
        
        # This utility adds cyclical time, day OHE, density, diffs etc.
        prepared_data_all_features = prepare_input_data_for_prediction(recent_raw_data) 
        
        # ---- START DEBUG PRINT ----
        if prepared_data_all_features is not None:
            print(f"DEBUG: Columns from prepare_input_data_for_prediction in app.py: {prepared_data_all_features.columns.tolist()}")
        else:
            print("DEBUG: prepared_data_all_features is None in app.py")
        # ---- END DEBUG PRINT ----

        if prepared_data_all_features is None or prepared_data_all_features.empty:
            return render_template('index.html', error="Failed to prepare recent data using utility.", available_models=all_available_models)

        # Take the last n_steps_in of the fully prepared data
        input_df_unscaled = prepared_data_all_features.tail(n_steps_in).copy()
        
        if len(input_df_unscaled) < n_steps_in:
             return render_template('index.html', error=f"Not enough processed data rows ({len(input_df_unscaled)}) for lookback {n_steps_in} after preparing recent data.", available_models=all_available_models)

        # Align features to the specific order the model expects
        # and ensure all expected features are present, filling with 0 if missing (though ideally they should all be there from prepare_input_data...)
        aligned_input_df_unscaled = pd.DataFrame(columns=model_features_order)
        for feature_name in model_features_order:
            if feature_name in input_df_unscaled.columns:
                aligned_input_df_unscaled[feature_name] = input_df_unscaled[feature_name]
            else:
                # This case should ideally not happen if prepare_input_data_for_prediction is comprehensive
                # and model_features_order is correct.
                print(f"Warning: Feature '{feature_name}' expected by model {model_display_name} not found in prepared data. Filling with 0.")
                aligned_input_df_unscaled[feature_name] = 0 
        
        # Critical Check: Number of features for the model must match
        if aligned_input_df_unscaled.shape[1] != num_model_expected_features:
            error_message = (f"Feature count mismatch for model {model_display_name}: "
                             f"Model expects {num_model_expected_features} features, but input has {aligned_input_df_unscaled.shape[1]}. "
                             f"Expected order: {model_features_order}. "
                             f"Columns in aligned input: {list(aligned_input_df_unscaled.columns)}."
                             f"Columns in prepared data from utility: {list(prepared_data_all_features.columns)}.")
            print(error_message) # Log for debugging
            return render_template('index.html', error=error_message, available_models=all_available_models)
            
        # Scale the features using the loaded dictionary of scalers
        input_df_scaled = aligned_input_df_unscaled.copy()
        for feature_name, scaler_obj in horizon_scalers_dict.items():
            if feature_name in input_df_scaled.columns:
                try:
                    # Scaler expects a 2D array, so select column as DataFrame
                    input_df_scaled[[feature_name]] = scaler_obj.transform(input_df_scaled[[feature_name]])
                except Exception as e:
                    # This might happen if a feature was in scalers_dict but not in model_features_order or vice-versa
                    # or if scaler was for a feature not present after alignment.
                    print(f"Warning: Could not scale feature '{feature_name}' for model {model_display_name}. Error: {e}")
            # else: # This feature was scaled during training, but is not an input to this model.
                  # print(f"Debug: Feature '{feature_name}' in scalers_dict but not in model's final input columns for {model_display_name}.")


        model_input_array = input_df_scaled.values.reshape((1, n_steps_in, num_model_expected_features))

        raw_predictions = model.predict(model_input_array)[0] # Assuming first dim is batch_size=1
        
        # Output might be (n_steps_out,) or (n_steps_out, 1). Ensure it's 2D for inverse_transform.
        if raw_predictions.ndim == 1:
            predicted_speeds_scaled = raw_predictions.reshape(-1, 1)
        else: # Assumed (n_steps_out, 1) already
            predicted_speeds_scaled = raw_predictions

        speed_scaler_for_output = horizon_scalers_dict['speed_kmh'] # We checked this exists
        predicted_speeds_actual = speed_scaler_for_output.inverse_transform(predicted_speeds_scaled).flatten()


        if weather_harsh:
            reduction = random.uniform(0.15, 0.25)
            predicted_speeds_actual = np.maximum(0, [s * (1 - reduction) for s in predicted_speeds_actual])

        granularity_minutes = 5 # Default
        # Try to infer granularity from model name for display/logic
        # This could also be loaded from metadata if available in the future.
        if "10_min" in model_display_name.lower(): 
            granularity_minutes = 10
        elif "15_min" in model_display_name.lower() or "15min" in model_display_name.lower(): 
            granularity_minutes = 1 # If n_steps_out implies 1-min steps over 15 min
            if n_steps_out == 15: 
                granularity_minutes = 1 
            else: 
                granularity_minutes = 15 # Or assume it's a single 15-min block if n_steps_out is 1
        elif "short_term" in model_display_name.lower(): 
            granularity_minutes = 1 # Default for generic short_term to 1-min steps if n_steps_out matches
            if n_steps_out != 15 and n_steps_out != 30 and n_steps_out !=10: # Heuristic
                 granularity_minutes = 5
        elif "medium_term" in model_display_name.lower() or "1hour" in model_display_name.lower() or "60min" in model_display_name.lower(): 
            granularity_minutes = 5 # Default for generic medium_term often 5-min intervals for an hour
        elif "long_term" in model_display_name.lower() or "2hour" in model_display_name.lower() or "6hour" in model_display_name.lower() or "12hour" in model_display_name.lower(): 
            granularity_minutes = 30 # Default for generic long_term
        elif "cnn_lstm" in model_display_name.lower(): 
            granularity_minutes = 5 # Default for cnn_lstm based on training
        
        granularity_display = f"{granularity_minutes}-minute intervals"
            
        predicted_timestamps = generate_future_timestamps(time_of_day_str, n_steps_out, granularity_minutes)

        plt.figure(figsize=(10, 5))
        plt.plot(predicted_timestamps, predicted_speeds_actual, 'b-o', label=f'Predicted Speed ({granularity_display})')
        plt.title(f"Traffic Speed Prediction - {model_display_name}")
        plt.ylabel('Speed (km/h)'); plt.xlabel('Time'); plt.xticks(rotation=45)
        plt.legend(); plt.grid(True, linestyle='--', alpha=0.7); plt.tight_layout()

        buffer = io.BytesIO()
        plt.savefig(buffer, format='png'); buffer.seek(0)
        plot_image = base64.b64encode(buffer.getvalue()).decode('utf-8'); plt.close()
        
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        
        # Zip the lists for the template
        predictions_for_template = zip(predicted_timestamps, predicted_speeds_actual)
        
        return render_template('prediction.html', model_file_name=model_display_name,
                              user_input={'day_of_week': day_of_week, 'time_of_day': time_of_day_str, 'weather_harsh': weather_harsh},
                              day_name=day_names[day_of_week], 
                              # Pass the zipped list directly
                              predictions_for_template=predictions_for_template, 
                              # Keep individual lists if needed elsewhere, or remove if only zip is used
                              # predicted_speeds=predicted_speeds_actual, 
                              # predicted_timestamps=predicted_timestamps, 
                              granularity_display=granularity_display, plot_image=plot_image)

    except ValueError as ve:
        app.logger.error(f"ValueError in prediction: {ve}", exc_info=True)
        return render_template('index.html', error=f"Input Error: {str(ve)}", available_models=get_available_models())
    except Exception as e:
        app.logger.error(f"Prediction error: {e}", exc_info=True)
        return render_template('index.html', error=f"An unexpected error occurred during prediction: {str(e)}", available_models=get_available_models())

if __name__ == '__main__':
    app.run(debug=True)