import pandas as pd
import numpy as np
import os
from datetime import datetime, time # Ensure datetime and time are imported
from typing import List, Dict, Any, Optional, Union # Add Optional and List here
from sklearn.preprocessing import StandardScaler # Or specific scaler type used
import logging # Make sure logging is imported at the top

# Add logger for this util file
logger = logging.getLogger(__name__)

# Define the standard order of features expected by the models (USER PROVIDED 18 FEATURES)
STANDARD_FEATURES_ORDER = [
    'speed_kmh', 'vehicle_count', 
    'hour_sin', 'hour_cos', 
    'minute_sin', 'minute_cos', 
    'day_of_week_sin', 'day_of_week_cos', 
    'traffic_density', 
    'speed_diff', 'vehicle_diff', 
    'day_0', 'day_1', 'day_2', 'day_3', 'day_4', 'day_5', 'day_6'
]

# Path to the primary dataset
# Relative to this file (backendv3/ml/utils/prediction_preparation_utils.py)
# So, ../../ goes up from utils, then ml, then into ModelStorage
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'ModelStorage', 'Data')
ACTUAL_DATA_PATH = os.path.join(DATA_DIR, 'inputdataset_MaiChiTho.csv')


def _calculate_cyclical_features(df, col_name_val_from_df, max_val, output_col_prefix):
    # col_name_val_from_df is the column in df that holds the raw values (e.g., 'hour', 'minute', 'day_of_week_num')
    df[output_col_prefix + '_sin'] = np.sin(2 * np.pi * df[col_name_val_from_df]/max_val)
    df[output_col_prefix + '_cos'] = np.cos(2 * np.pi * df[col_name_val_from_df]/max_val)
    return df

def prepare_recent_data_for_prediction(
    raw_data_df_slice: pd.DataFrame, # This is the slice of n_steps_in from the main CSV
    n_steps_in: int,
    target_features_order: Optional[List[str]] = None
) -> Optional[pd.DataFrame]:
    """
    Prepares a given slice of 'n_steps_in' data points for prediction.
    raw_data_df_slice should contain 'timestamp', 'speed_kmh', 'vehicle_count', 'hour', 'minute'.
    It adds time-based features, cyclical features, and basic diffs.
    It aligns features to 'target_features_order' (the 18 features).
    """
    if target_features_order is None:
        target_features_order = STANDARD_FEATURES_ORDER

    required_raw_cols = ['timestamp', 'speed_kmh', 'vehicle_count', 'hour', 'minute']
    if not all(col in raw_data_df_slice.columns for col in required_raw_cols):
        logger.error(f"[PREP_SLICE] raw_data_df_slice missing required columns. Has: {raw_data_df_slice.columns.tolist()}, Needs: {required_raw_cols}")
        return None

    if len(raw_data_df_slice) != n_steps_in:
        logger.error(f"[PREP_SLICE] raw_data_df_slice length ({len(raw_data_df_slice)}) != n_steps_in ({n_steps_in}).")
        return None

    df = raw_data_df_slice.copy()

    # Ensure timestamp is datetime
    try:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    except Exception as e:
        logger.error(f"[PREP_SLICE] Error converting 'timestamp' to datetime: {e}.")
        return None

    # Derive day_of_week (Monday=0, Sunday=6) from the timestamp column for cyclical and one-hot
    df['day_of_week_num'] = df['timestamp'].dt.dayofweek 

    # Cyclical features using 'hour', 'minute', 'day_of_week_num'
    # 'hour' and 'minute' columns are expected to be present in raw_data_df_slice from inputdataset_MaiChiTho.csv
    df = _calculate_cyclical_features(df, 'hour', 23.0, 'hour') # Uses df['hour']
    df = _calculate_cyclical_features(df, 'minute', 59.0, 'minute') # Uses df['minute']
    df = _calculate_cyclical_features(df, 'day_of_week_num', 6.0, 'day_of_week') # Uses df['day_of_week_num']

    # One-hot encode day_of_week_num
    try:
        day_dummies = pd.get_dummies(df['day_of_week_num'], prefix='day', drop_first=False).astype(int)
        for i in range(7): # Ensure all day_0 to day_6 columns
            if f'day_{i}' not in day_dummies.columns:
                day_dummies[f'day_{i}'] = 0
        df = pd.concat([df, day_dummies[[f'day_{i}' for i in range(7)]]], axis=1)
    except Exception as e:
        logger.error(f"[PREP_SLICE] Error during one-hot encoding: {e}")
        return None
        
    # Traffic density: vehicle_count is used as a proxy as per original logic.
    df['traffic_density'] = df['vehicle_count'] * 0.1 # Arbitrary scaling, ensure consistency with training

    # Difference features
    # Need at least 2 rows in the original df passed to this func if we calculate diff here.
    # Since n_steps_in could be 1, it's safer if diffs are calculated on a slightly larger slice
    # OR ensure the input `raw_data_df_slice` is already `n_steps_in` and diffs are pre-calculated or handled carefully.
    # For now, standard .diff().fillna(0) on the slice itself.
    df['speed_diff'] = df['speed_kmh'].diff().fillna(0)
    df['vehicle_diff'] = df['vehicle_count'].diff().fillna(0)
    
    # Select and order features according to target_features_order (18 features)
    final_df_ordered = pd.DataFrame(index=df.index) # Preserve index if needed
    
    missing_cols_in_source = []
    for col in target_features_order:
        if col in df.columns:
            final_df_ordered[col] = df[col]
        else:
            # This should not happen if feature engineering is correct for the 18 features
            logger.warning(f"[PREP_SLICE] Feature '{col}' missing in engineered df. Filling with 0.")
            final_df_ordered[col] = 0.0 
            missing_cols_in_source.append(col)
            
    if missing_cols_in_source:
        logger.warning(f"[PREP_SLICE] Missing source features (filled with 0): {missing_cols_in_source}. Available cols: {df.columns.tolist()}")


    # Final check on feature count and order
    if list(final_df_ordered.columns) != target_features_order:
        logger.error(f"[PREP_SLICE] Final column order/count mismatch. Expected: {target_features_order}, Got: {list(final_df_ordered.columns)}")
        # Attempt to reorder/select one last time
        try:
            final_df_ordered = final_df_ordered[target_features_order]
        except KeyError as e:
            logger.error(f"[PREP_SLICE] Failed reorder: {e}"); return None

    if len(final_df_ordered) != n_steps_in:
        logger.error(f"[PREP_SLICE] Final row count {len(final_df_ordered)} != {n_steps_in}")
        return None
        
    return final_df_ordered

def load_and_prepare_feature_engineered_input(
    n_steps_in: int,
    target_day_of_week: int,
    target_time_str: str,
    data_path: str = ACTUAL_DATA_PATH 
) -> Optional[pd.DataFrame]:
    """
    Load and prepare the most recent feature-engineered input data for a given number of steps,
    targeting a specific day of the week and time.
    
    n_steps_in         : Desired number of input time steps (e.g., 12 for hourly data)
    target_day_of_week : Day of the week to target (0=Monday, 6=Sunday)
    target_time_str    : Target time as a string in "HH:MM" 24-hour format
    data_path          : Path to the CSV data file (default: ACTUAL_DATA_PATH)
    
    Returns a DataFrame with the latest n_steps_in rows for the specified day and time,
    containing all required features in the correct order. Returns None if there is an error.
    """
    # MINIMAL LOGGING START
    logger.info(f"[LOAD_PREP_MINIMAL] Called with: n_steps={n_steps_in}, day={target_day_of_week}, time='{target_time_str}'")
    # MINIMAL LOGGING END
    if not os.path.exists(data_path):
        logger.error(f"[LOAD_PREP] Data file NOT FOUND: {data_path}")
        return None
    try:
        main_df = pd.read_csv(data_path)
        if 'timestamp' not in main_df.columns: 
            logger.error("[LOAD_PREP] 'timestamp' column missing."); return None
        main_df['timestamp'] = pd.to_datetime(main_df['timestamp'])
    except Exception as e_read:
        logger.error(f"[LOAD_PREP] Error reading/processing CSV {data_path}: {e_read}")
        return None
    if len(main_df) < n_steps_in:
        logger.error(f"[LOAD_PREP] Insufficient data in CSV ({len(main_df)} rows) for n_steps_in={n_steps_in}.")
        return None
    try:
        target_time_obj = datetime.strptime(target_time_str, "%H:%M").time()
    except ValueError:
        logger.error(f"[LOAD_PREP] Invalid target_time_str format: {target_time_str}. Expected HH:MM.")
        return None

    target_end_index = -1
    for i in range(len(main_df) - 1, -1, -1):
        row_timestamp = main_df.iloc[i]['timestamp']
        if pd.isna(row_timestamp): continue 
        row_day_of_week = row_timestamp.dayofweek 
        row_time = row_timestamp.time()
        if row_day_of_week == target_day_of_week and row_time == target_time_obj:
            target_end_index = i
            # MINIMAL LOGGING START
            logger.info(f"[LOAD_PREP_MINIMAL] Match found: target_end_index={target_end_index}, matched_timestamp={row_timestamp}")
            # MINIMAL LOGGING END
            break
    
    if target_end_index == -1:
        logger.warning(f"[LOAD_PREP] No exact match for day={target_day_of_week}, time={target_time_str}. Total rows: {len(main_df)}.")
        return None 

    if target_end_index < n_steps_in -1:
        logger.error(f"[LOAD_PREP] Insufficient history. Matched index={target_end_index}, n_steps_in={n_steps_in}")
        return None

    start_index = target_end_index - n_steps_in + 1
    # MINIMAL LOGGING START
    logger.info(f"[LOAD_PREP_MINIMAL] Calculated slice: start_index={start_index}, end_index={target_end_index}")
    # MINIMAL LOGGING END
    raw_slice_for_model = main_df.iloc[start_index : target_end_index + 1]
        
    if raw_slice_for_model.empty:
        logger.error("[LOAD_PREP] Extracted raw_slice_for_model is EMPTY!")
        return None
    if len(raw_slice_for_model) != n_steps_in:
        logger.error(f"[LOAD_PREP] Extracted slice length {len(raw_slice_for_model)} != n_steps_in {n_steps_in}")
        return None
        
    # MINIMAL LOGGING START
    logger.info(f"[LOAD_PREP_MINIMAL] Raw slice extracted. First timestamp: {raw_slice_for_model.iloc[0]['timestamp']}, Last timestamp: {raw_slice_for_model.iloc[-1]['timestamp']}")
    # MINIMAL LOGGING END
    
    feature_engineered_df = prepare_recent_data_for_prediction(
        raw_data_df_slice=raw_slice_for_model,
        n_steps_in=n_steps_in
    )
    if feature_engineered_df is None:
        logger.error("[LOAD_PREP] Feature engineering (prepare_recent_data_for_prediction) returned None.")
    else:
        logger.info(f"[LOAD_PREP] Feature engineering successful. Shape: {feature_engineered_df.shape}")
    return feature_engineered_df


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    logger.info("Testing NEW prediction preparation utils with enhanced logging...")
    logger.info(f"Looking for data at: {ACTUAL_DATA_PATH}")
    n_steps = 12
    test_target_day = 0 
    test_target_time = "08:40" 
    logger.info(f"\nAttempting to load for n_steps={n_steps}, day={test_target_day}, time='{test_target_time}'")
    data_dir_for_test = os.path.dirname(ACTUAL_DATA_PATH)
    if not os.path.exists(data_dir_for_test):
        try: os.makedirs(data_dir_for_test); logger.info(f"Created test dir: {data_dir_for_test}")
        except FileExistsError: pass
    if not os.path.exists(ACTUAL_DATA_PATH):
        logger.info(f"Dummy data file {ACTUAL_DATA_PATH} not found. Creating for testing.")
        num_dummy_rows = 2000 
        start_dt = datetime(2023, 1, 1, 0, 0, 0)
        dummy_timestamps = [start_dt + pd.Timedelta(minutes=i*20) for i in range(num_dummy_rows)]
        dummy_data = {
            'timestamp_seconds': [i*20*60 for i in range(num_dummy_rows)],
            'timestamp': dummy_timestamps,
            'speed_kmh': np.random.uniform(20, 70, num_dummy_rows),
            'vehicle_count': np.random.randint(5, 60, num_dummy_rows),
            'hour': [dt.hour for dt in dummy_timestamps],
            'minute': [dt.minute for dt in dummy_timestamps],
            'day': [dt.day for dt in dummy_timestamps],
            'week': [dt.isocalendar().week for dt in dummy_timestamps],
            'month': [dt.month for dt in dummy_timestamps],
            'year': [dt.year for dt in dummy_timestamps]
        }
        dummy_df = pd.DataFrame(dummy_data)
        try: dummy_df.to_csv(ACTUAL_DATA_PATH, index=False); logger.info(f"Created dummy file: {ACTUAL_DATA_PATH}")
        except Exception as e: logger.error(f"Failed to create dummy file: {e}")
    test_input_df = load_and_prepare_feature_engineered_input(
        n_steps_in=n_steps, target_day_of_week=test_target_day, target_time_str=test_target_time)
    if test_input_df is not None:
        logger.info(f"Successfully prepared data: Shape={test_input_df.shape}, Columns({len(test_input_df.columns)})={test_input_df.columns.tolist()}")
        logger.info(f"Head:\n{test_input_df.head()}")
        if all(f in test_input_df.columns for f in STANDARD_FEATURES_ORDER) and len(test_input_df.columns) == len(STANDARD_FEATURES_ORDER):
            logger.info("Feature order/count OK.")
        else: logger.error("Feature order/count MISMATCH.")
    else: logger.error(f"Failed to prepare data for day={test_target_day}, time='{test_target_time}'.")