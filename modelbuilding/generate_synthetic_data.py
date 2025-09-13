import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os

# --- Configuration ---
START_DATE = datetime(2021, 10, 27, 7, 0, 0)  # Align with original dataset start
NUM_DAYS = 7
TIME_INTERVAL_SECONDS = 20 # Align with original dataset interval

ORIGINAL_DATA_PATH = 'modelbuilding/Input/traffic_dataset.csv' # This will now be the modified one
SYNTHETIC_DATA_PATH = 'modelbuilding/Input/synthetic_traffic_dataset.csv' # Output for this script

# Define the target directory path for inputs and outputs of this script
TARGET_DATA_DIR = 'modelbuilding/Input'

# Ensure target Input directory exists
if not os.path.exists(TARGET_DATA_DIR):
    os.makedirs(TARGET_DATA_DIR)

# --- Profile Definitions (Simplified - needs significant tuning) ---

def get_time_of_day_segment(hour):
    if 5 <= hour < 10: return "morning_rush"  # 5 AM - 9 AM
    if 10 <= hour < 12: return "mid_morning"
    if 12 <= hour < 14: return "lunch_time"
    if 14 <= hour < 16: return "afternoon"
    if 16 <= hour < 19: return "evening_rush" # 4 PM - 7 PM
    if 19 <= hour < 23: return "evening"
    return "night" # 11 PM - 5 AM

# Base multipliers [vehicle_count_multiplier, speed_kmh_base_factor]
# speed_kmh_base_factor: Multiplier for a typical free-flow speed.
# Actual speed will be derived also based on vehicle count.
DAILY_PROFILES = {
    "weekday": {
        "morning_rush": [1.0, 0.9],
        "mid_morning":  [0.7, 1.0],
        "lunch_time":   [0.6, 0.95],
        "afternoon":    [0.8, 1.0],
        "evening_rush": [0.9, 0.85],
        "evening":      [0.5, 1.1],
        "night":        [0.2, 1.2]
    },
    "weekend": {
        "morning_rush": [0.5, 1.0], # Later, less intense rush
        "mid_morning":  [0.6, 1.0],
        "lunch_time":   [0.7, 0.9],
        "afternoon":    [0.7, 0.95],
        "evening_rush": [0.4, 1.0], # Less of an evening rush
        "evening":      [0.6, 1.1],
        "night":        [0.15, 1.2]
    }
}

# Base characteristics (can be derived from original_df if available and representative)
DEFAULT_MAX_SPEED = 80  # km/h (e.g., speed limit or typical free-flow)
DEFAULT_MIN_SPEED = 10  # km/h
DEFAULT_MAX_VEHICLES_PEAK = 100 # Adjusted default, will be overridden by original data if available

# --- Helper Functions ---

def add_noise(value, percentage):
    noise = random.uniform(-percentage, percentage)
    return value * (1 + noise)

# --- Main Generation Logic ---
def generate_synthetic_data():
    print(f"Starting synthetic data generation for {NUM_DAYS} days...")
    
    original_df = None
    original_speeds = []
    original_counts = []
    original_data_length = 0
    original_hours = []
    original_day_of_week = [] # This will store values from the identified day column
    original_day_of_week_column_name = None # To store the actual column name found

    try:
        if os.path.exists(ORIGINAL_DATA_PATH):
            original_df = pd.read_csv(ORIGINAL_DATA_PATH)
            
            # Determine the actual column name for day of the week
            if 'day_of_week' in original_df.columns:
                original_day_of_week_column_name = 'day_of_week'
            elif 'day' in original_df.columns: # As per user's dataset structure
                original_day_of_week_column_name = 'day'
                print(f"Info: Using column '{original_day_of_week_column_name}' from '{ORIGINAL_DATA_PATH}' as the source for day-of-week information.")
            
            # Define base essential columns for checking
            base_essential_columns = ['speed_kmh', 'vehicle_count', 'hour']
            
            # Check if all base essential columns are present
            missing_base_cols = [col for col in base_essential_columns if col not in original_df.columns]
            if missing_base_cols:
                print(f"Error: Original data ({ORIGINAL_DATA_PATH}) is missing essential columns: {missing_base_cols}. Using defaults.")
                original_data_length = 0 # Ensure fallback to defaults
            else:
                # Add the identified day column to the list for dropna if found
                columns_for_dropna = base_essential_columns.copy()
                if original_day_of_week_column_name:
                    columns_for_dropna.append(original_day_of_week_column_name)
                else:
                    print(f"Warning: Neither 'day_of_week' nor 'day' column found in {ORIGINAL_DATA_PATH}. Day-specific patterns from original data cannot be fully utilized.")

                original_df_cleaned = original_df.dropna(subset=columns_for_dropna)
                
                if not original_df_cleaned.empty:
                    original_speeds = original_df_cleaned['speed_kmh'].tolist()
                    original_counts = original_df_cleaned['vehicle_count'].tolist()
                    original_hours = original_df_cleaned['hour'].tolist()
                    
                    if original_day_of_week_column_name and original_day_of_week_column_name in original_df_cleaned.columns:
                        original_day_of_week = original_df_cleaned[original_day_of_week_column_name].tolist()
                    # If original_day_of_week_column_name was None, original_day_of_week remains an empty list as initialized.
                    
                    original_data_length = len(original_speeds)
                    if original_data_length > 0:
                        print(f"Loaded original data ({ORIGINAL_DATA_PATH}) with {original_data_length} samples to inform generation.")
                    else:
                        print(f"Original data ({ORIGINAL_DATA_PATH}) is empty after dropping NaNs from essential columns ({columns_for_dropna}). Using defaults.")
                else:
                    print(f"Original data ({ORIGINAL_DATA_PATH}) is empty after attempting to drop NaNs based on columns: {columns_for_dropna}. Using defaults.")
                    original_data_length = 0
        else:
            print(f"Original data file not found at {ORIGINAL_DATA_PATH}. Using defaults.")
            original_data_length = 0
    except Exception as e:
        print(f"Error loading or processing original data from {ORIGINAL_DATA_PATH}: {e}. Using defaults.")
        original_data_length = 0

    max_observed_speed = max(original_speeds) if original_speeds else DEFAULT_MAX_SPEED
    min_observed_speed = min(original_speeds) if original_speeds else DEFAULT_MIN_SPEED
    max_observed_vehicles = max(original_counts) if original_counts else DEFAULT_MAX_VEHICLES_PEAK
    
    # Estimate a general max vehicles for segment for density calculation
    # This is a rough estimate; specific segment max might be more nuanced.
    # Using max_observed_vehicles directly as a general upper bound for density scaling.
    estimated_max_vehicles_overall = max_observed_vehicles 

    all_data = []
    current_timestamp = START_DATE
    original_data_idx = 0
    elapsed_seconds = 0 # Initialize elapsed seconds counter

    for day_num in range(NUM_DAYS):
        day_type = "weekend" if current_timestamp.weekday() >= 5 else "weekday"
        print(f"Generating Day {day_num + 1} ({current_timestamp.strftime('%Y-%m-%d')}, {day_type})...")
        
        day_start_timestamp = current_timestamp # Save start for the day
        for _ in range(int(24 * 60 * 60 / TIME_INTERVAL_SECONDS)): # Iterate for a full day in 20-sec intervals
            hour_num = current_timestamp.hour
            segment = get_time_of_day_segment(hour_num)
            profile_multipliers = DAILY_PROFILES[day_type][segment]
            
            vehicle_count_multiplier = profile_multipliers[0]
            speed_base_factor = profile_multipliers[1]
            
            # --- Generate Vehicle Count ---
            # Try to pick a more contextually relevant base_vehicle_count from original data
            base_vehicle_count = 0
            if original_data_length > 0:
                # Simple approach: cycle through original_counts
                # A more advanced approach could try to match by hour/day_of_week from original data
                # For now, stick to cycling to ensure it runs, but acknowledge this could be improved.
                base_vehicle_count = original_counts[original_data_idx % original_data_length]
                original_data_idx += 1
            else:
                # Fallback to random if no original data
                base_vehicle_count = random.uniform(0.05, 0.3) * estimated_max_vehicles_overall
            
            vehicle_count = base_vehicle_count * vehicle_count_multiplier
            vehicle_count = add_noise(vehicle_count, 0.15) # Increased noise slightly
            vehicle_count = max(0, int(round(vehicle_count)))

            # --- Generate Speed ---
            current_base_speed = max_observed_speed * speed_base_factor 
            
            # Simplified density effect: higher vehicle count reduces speed towards min_observed_speed
            # Normalize vehicle count against a general max for this effect
            density_factor = (vehicle_count / (estimated_max_vehicles_overall + 1e-6))
            # Make the effect more pronounced as density increases
            speed_reduction_factor = density_factor ** 1.2 
            
            # Speed reduction is proportional to the range between base speed and min speed
            speed_reduction = speed_reduction_factor * (current_base_speed - min_observed_speed)
            
            speed_kmh = current_base_speed - speed_reduction
            speed_kmh = add_noise(speed_kmh, 0.10) # Increased noise slightly
            speed_kmh = np.clip(speed_kmh, min_observed_speed, max_observed_speed)
            speed_kmh = round(speed_kmh, 2)

            # Removed congestion_category and congestion_code from here
            all_data.append({
                "timestamp_seconds": elapsed_seconds, # Use the elapsed_seconds counter
                "timestamp": current_timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                "speed_kmh": speed_kmh,
                "vehicle_count": vehicle_count,
                "hour": current_timestamp.hour,
                "minute": current_timestamp.minute,
                "day": current_timestamp.weekday(), # Day of the week (0=Monday, 6=Sunday)
                "week": current_timestamp.isocalendar()[1],
                "month": current_timestamp.month,
                "year": current_timestamp.year
                # "day_of_week" column is removed as its value is now in "day"
            })
            
            current_timestamp += timedelta(seconds=TIME_INTERVAL_SECONDS)
            elapsed_seconds += TIME_INTERVAL_SECONDS # Increment elapsed seconds
        # Ensure the next day starts correctly if any rounding issues with intervals
        current_timestamp = day_start_timestamp + timedelta(days=1)

    synthetic_df = pd.DataFrame(all_data)
    # Ensure correct dtypes for columns that will be used in training
    synthetic_df['speed_kmh'] = pd.to_numeric(synthetic_df['speed_kmh'], errors='coerce')
    synthetic_df['vehicle_count'] = pd.to_numeric(synthetic_df['vehicle_count'], errors='coerce')
    # Ensure time-related integer columns are integers
    time_int_cols = ['hour', 'minute', 'day', 'week', 'month', 'year'] # 'day_of_week' removed
    for col in time_int_cols:
        if col in synthetic_df.columns:
            synthetic_df[col] = pd.to_numeric(synthetic_df[col], errors='coerce').fillna(0).astype(int)


    synthetic_df.to_csv(SYNTHETIC_DATA_PATH, index=False)
    print(f"Successfully generated {len(synthetic_df)} records.")
    print(f"Synthetic data saved to {SYNTHETIC_DATA_PATH}")
    print("First 5 rows of synthetic data:")
    print(synthetic_df.head())
    print("Synthetic data info:")
    synthetic_df.info()
    print("Synthetic data description:")
    print(synthetic_df.describe())


if __name__ == "__main__":
    generate_synthetic_data()
    # Example of how to check the generated data
    # df_check = pd.read_csv(SYNTHETIC_DATA_PATH)
    # print(df_check.head())
    # print(df_check.describe())
    # print(df_check.info())