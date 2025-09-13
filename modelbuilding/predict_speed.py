import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import argparse
import sys
from datetime import datetime, timedelta
import json

# Import from the new utility file
from prediction_utils import load_all_models_and_scalers, prepare_input_data_for_prediction, generate_predictions_from_input


# Define paths
MODEL_DIR = 'processed_data/saved_models'
OUTPUT_DIR = 'processed_data/predictions'
DATA_PATH = 'Input/synthetic_traffic_dataset.csv'

# Create directories if they don't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

def load_models_and_scalers():
    """
    Load the trained models and scalers using the utility function.
    """
    models_with_keras, scalers = load_all_models_and_scalers(MODEL_DIR)
    if models_with_keras and scalers:
        print("Successfully loaded all models and scalers via utility in predict_speed.py.")
    else:
        print("Failed to load models or scalers via utility in predict_speed.py.")
    return models_with_keras, scalers

def load_recent_data(n_steps_in, sample_data=False): # sample_data arg might be less relevant now
    """
    Load and prepare recent traffic data for prediction using the utility function.
    If sample_data is True, it uses DATA_PATH. Otherwise, it should ideally fetch live data.
    For now, it will always use DATA_PATH as live fetching isn't implemented.
    """
    try:
        # Load raw data from DATA_PATH
        df_raw = pd.read_csv(DATA_PATH)

        if 'timestamp' not in df_raw.columns and df_raw.index.name == 'timestamp':
            df_raw.reset_index(inplace=True)
        elif 'timestamp' not in df_raw.columns:
            print("Error: 'timestamp' column is missing in predict_speed.py.")
            return None
            
        # Take a slice for recent data - ensure enough for lookback + diffs
        if len(df_raw) < n_steps_in:
            print(f"Warning: Not enough raw data for lookback in predict_speed.py. Need {n_steps_in}, got {len(df_raw)}.")
            recent_raw_data_slice = df_raw.copy()
        else:
            # Take more than n_steps_in to allow prepare_input_data_for_prediction to calculate diffs correctly
            # The actual sequence length is handled by generate_predictions_from_input
            recent_raw_data_slice = df_raw.tail(n_steps_in + 5).copy() 

        # Prepare this recent data slice using the utility function
        prepared_data = prepare_input_data_for_prediction(recent_raw_data_slice)
        
        if prepared_data is None:
            print("Error: Failed to prepare recent data using utility function in predict_speed.py.")
            return None
        
        print(f"Successfully loaded and prepared recent data of shape: {prepared_data.shape} in predict_speed.py")
        return prepared_data

    except Exception as e:
        print(f"Error loading or preparing recent data in predict_speed.py: {e}")
        return None

def predict_with_user_input(models, scalers, recent_prepared_data, user_input):
    """
    Make predictions using user-provided current conditions via the utility function.
    """
    predictions = generate_predictions_from_input(
        models_dict=models,
        scalers_dict=scalers,
        recent_data_unscaled_with_features=recent_prepared_data, # This is the prepared data
        user_input_conditions=user_input
    )
    return predictions

def plot_predictions(predictions, save_path=None):
    """
    Plot predictions for all horizons
    """
    # Number of horizons to plot
    n_horizons = len(predictions)
    
    plt.figure(figsize=(15, 5 * n_horizons))
    
    for i, (horizon, pred) in enumerate(predictions.items()):
        plt.subplot(n_horizons, 1, i+1)
        
        # Plot speed predictions
        timestamps = pred['timestamps']
        speeds = pred['speeds']
        
        plt.plot(timestamps, speeds, 'b-o', label=f'Predicted Speed (km/h)')
        
        # Add title and labels
        plt.title(f"{horizon.replace('_', ' ').title()} Prediction" + 
                 (f" - Weather Effect Applied" if pred['weather_effect'] else ""))
        plt.ylabel('Speed (km/h)')
        plt.xlabel('Time')
        
        # Add granularity in the legend
        plt.legend([f'Predicted Speed - {pred["granularity"]} intervals'])
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45)
        
        # Add grid
        plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    
    plt.close()

def export_predictions_to_json(predictions, output_file):
    """
    Export predictions to a JSON file
    """
    with open(output_file, 'w') as f:
        json.dump(predictions, f, indent=2)
    
    print(f"Predictions exported to {output_file}")

def parse_arguments():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(description='Traffic Speed Prediction')
    
    # Required arguments
    parser.add_argument('--day', type=int, required=True, choices=range(7),
                        help='Day of week (0=Monday, 6=Sunday)')
    parser.add_argument('--time', type=str, required=True,
                        help='Time of day in HH:MM format')
    
    # Optional arguments
    parser.add_argument('--weather_harsh', action='store_true',
                        help='Flag to indicate harsh weather conditions')
    parser.add_argument('--sample', action='store_true',
                        help='Use sample data instead of live data')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file for JSON predictions')
    parser.add_argument('--plot', type=str, default=None,
                        help='Output file for prediction plot')
    
    return parser.parse_args()

def validate_time_format(time_str):
    """
    Validate time format (HH:MM)
    """
    try:
        hour, minute = map(int, time_str.split(':'))
        if hour < 0 or hour > 23 or minute < 0 or minute > 59:
            return False
        return True
    except:
        return False

def main():
    """
    Main function
    """
    # Parse arguments
    args = parse_arguments()
    
    # Validate time format
    if not validate_time_format(args.time):
        print("Error: Time must be in HH:MM format")
        sys.exit(1)
    
    # Load models and scalers
    models, scalers = load_models_and_scalers()
    if models is None or scalers is None:
        print("Error: Failed to load models or scalers")
        sys.exit(1)
    
    # Determine maximum lookback needed
    max_steps_in = max([model_info['n_steps_in'] for model_info in models.values()])
    
    # Load recent data
    recent_data = load_recent_data(max_steps_in, sample_data=args.sample)
    
    # Create user input
    user_input = {
        'day_of_week': args.day,
        'time_of_day': args.time,
        'weather_harsh': args.weather_harsh
    }
    
    # Make predictions
    predictions = predict_with_user_input(models, scalers, recent_data, user_input)
    
    # Display predictions
    print("\nPREDICTIONS:")
    print(f"Day: {['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'][args.day]}")
    print(f"Time: {args.time}")
    print(f"Weather conditions: {'Harsh' if args.weather_harsh else 'Normal'}")
    print("\n")
    
    for horizon, pred in predictions.items():
        print(f"{horizon.replace('_', ' ').upper()} PREDICTIONS (Granularity: {pred['granularity']}):")
        for i, (ts, speed) in enumerate(zip(pred['timestamps'], pred['speeds'])):
            print(f"  {ts}: {speed:.2f} km/h")
        print("\n")
    
    # Export predictions to JSON if specified
    if args.output:
        export_predictions_to_json(predictions, args.output)
    
    # Plot predictions if specified
    if args.plot:
        plot_predictions(predictions, args.plot)
    else:
        # Always show the plot in interactive mode
        plot_predictions(predictions)
        plt.show()

if __name__ == "__main__":
    main()