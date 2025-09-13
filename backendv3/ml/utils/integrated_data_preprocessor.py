"""
Integrated Data Preprocessor utilities for UC11.
Replicates and adapts logic from modelbuildingv3/preprocess.py.
"""
import pandas as pd
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import logging
from typing import Tuple, Optional, List, Dict, Any

logger = logging.getLogger(__name__)

class IntegratedDataPreprocessor:
    def __init__(self, input_csv_path: str, output_ml_data_dir: str):
        """
        Args:
            input_csv_path: Path to the interval-based CSV dataset (e.g., from IntegratedDatasetCreator).
            output_ml_data_dir: Directory where the processed ML-ready .npy files will be saved.
        """
        self.input_csv_path = Path(input_csv_path)
        self.output_ml_data_dir = Path(output_ml_data_dir)
        self.output_ml_data_dir.mkdir(parents=True, exist_ok=True)
        self.data: Optional[pd.DataFrame] = None
        self.scaler: Optional[StandardScaler] = None # Or MinMaxScaler

    def load_data(self) -> bool:
        if not self.input_csv_path.exists():
            logger.error(f"Input CSV file not found: {self.input_csv_path}")
            return False
        try:
            self.data = pd.read_csv(self.input_csv_path)
            logger.info(f"Successfully loaded data from {self.input_csv_path}. Shape: {self.data.shape}")
            # Ensure 'interval_start_time' is parsed as datetime
            if 'interval_start_time' in self.data.columns:
                self.data['interval_start_time'] = pd.to_datetime(self.data['interval_start_time'])
            else:
                logger.error("'interval_start_time' column missing from CSV.")
                return False
            return True
        except Exception as e:
            logger.error(f"Error loading CSV data: {e}")
            return False

    def clean_data(self) -> bool:
        if self.data is None:
            logger.error("Data not loaded. Call load_data() first.")
            return False
        
        # Handle missing values (example: forward fill for time series, or mean/median imputation)
        # For vehicle_count, 0 is a valid value, but NaNs might appear if intervals had no source data.
        self.data['vehicle_count'] = self.data['vehicle_count'].fillna(0) # Or other strategy
        # Example: Drop rows if critical data like time is missing (already handled by load_data check)

        # Remove duplicates based on timestamp if any (unlikely with resample but good practice)
        self.data.drop_duplicates(subset=['interval_start_time'], keep='first', inplace=True)
        self.data.sort_values('interval_start_time', inplace=True)
        self.data.reset_index(drop=True, inplace=True)
        logger.info(f"Data cleaned. Shape after cleaning: {self.data.shape}")
        return True

    def extract_temporal_features(self) -> bool:
        if self.data is None or 'interval_start_time' not in self.data.columns:
            logger.error("Data or 'interval_start_time' not available for feature extraction.")
            return False
        
        dt_series = self.data['interval_start_time']
        self.data['hour'] = dt_series.dt.hour
        self.data['day_of_week'] = dt_series.dt.dayofweek # Monday=0, Sunday=6
        self.data['day_of_month'] = dt_series.dt.day
        self.data['month'] = dt_series.dt.month
        self.data['year'] = dt_series.dt.year
        self.data['week_of_year'] = dt_series.dt.isocalendar().week.astype(int)
        # self.data['time_of_day_sin'] = np.sin(2 * np.pi * self.data['hour']/24.0)
        # self.data['time_of_day_cos'] = np.cos(2 * np.pi * self.data['hour']/24.0)
        logger.info(f"Temporal features extracted. Columns: {self.data.columns.tolist()}")
        return True

    def scale_features(self, features_to_scale: List[str], target_feature: Optional[str] = None) -> bool:
        if self.data is None:
            logger.error("Data not loaded for scaling.")
            return False

        self.scaler = StandardScaler() # Or MinMaxScaler()
        
        # Scale specified features
        if features_to_scale:
            self.data[features_to_scale] = self.scaler.fit_transform(self.data[features_to_scale])
            logger.info(f"Scaled features: {features_to_scale}")
        
        # Optionally scale the target variable separately if needed (often done, but depends on model)
        # if target_feature and target_feature in self.data.columns:
        #     self.target_scaler = StandardScaler()
        #     self.data[target_feature] = self.target_scaler.fit_transform(self.data[[target_feature]])
        #     logger.info(f"Scaled target feature: {target_feature}")
        return True

    def create_sequences_for_lstm(self, sequence_length: int, target_column: str = 'vehicle_count') -> Optional[Tuple[np.ndarray, np.ndarray]]:
        if self.data is None:
            logger.error("Data not available for sequence creation.")
            return None
        
        # Example for LSTM: assuming 'vehicle_count' is the target and other numerical columns are features
        # This is a simplified sequence creation. Real implementation might involve more features.
        df_for_sequences = self.data[[target_column] + [col for col in ['hour', 'day_of_week'] if col in self.data.columns]]
        if df_for_sequences.empty:
            logger.error("No data to create sequences from.")
            return None
            
        data_np = df_for_sequences.values
        X, y = [], []
        for i in range(len(data_np) - sequence_length):
            X.append(data_np[i:(i + sequence_length)])
            y.append(data_np[i + sequence_length, df_for_sequences.columns.get_loc(target_column)]) # Target is vehicle_count
        
        if not X or not y:
            logger.warning("Could not create any sequences. Data might be too short for the sequence length.")
            return None
            
        return np.array(X), np.array(y)

    def split_and_save_data(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        test_size: float = 0.2, 
        val_size: Optional[float] = 0.1, # Proportion of training set for validation
        random_state: int = 42,
        file_prefix: str = "ml_dataset"
    ) -> Dict[str, Path]:
        """
        Splits data into train, validation (optional), and test sets, then saves them.
        """
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, shuffle=False) # Time series often not shuffled
        
        paths = {}
        np.save(self.output_ml_data_dir / f"{file_prefix}_X_train.npy", X_train)
        paths['X_train'] = self.output_ml_data_dir / f"{file_prefix}_X_train.npy"
        np.save(self.output_ml_data_dir / f"{file_prefix}_y_train.npy", y_train)
        paths['y_train'] = self.output_ml_data_dir / f"{file_prefix}_y_train.npy"
        
        if val_size and val_size > 0:
            # Create validation set from the training set
            X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_size, random_state=random_state, shuffle=False)
            np.save(self.output_ml_data_dir / f"{file_prefix}_X_val.npy", X_val)
            paths['X_val'] = self.output_ml_data_dir / f"{file_prefix}_X_val.npy"
            np.save(self.output_ml_data_dir / f"{file_prefix}_y_val.npy", y_val)
            paths['y_val'] = self.output_ml_data_dir / f"{file_prefix}_y_val.npy"
            # Update original X_train, y_train after splitting off validation set
            np.save(self.output_ml_data_dir / f"{file_prefix}_X_train.npy", X_train) # Overwrite
            np.save(self.output_ml_data_dir / f"{file_prefix}_y_train.npy", y_train) # Overwrite

        np.save(self.output_ml_data_dir / f"{file_prefix}_X_test.npy", X_test)
        paths['X_test'] = self.output_ml_data_dir / f"{file_prefix}_X_test.npy"
        np.save(self.output_ml_data_dir / f"{file_prefix}_y_test.npy", y_test)
        paths['y_test'] = self.output_ml_data_dir / f"{file_prefix}_y_test.npy"

        logger.info(f"ML data splits saved to {self.output_ml_data_dir}")
        logger.info(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
        if val_size and 'X_val' in paths:
             logger.info(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")
        logger.info(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
        return paths

    def run_full_preprocessing_pipeline(
        self, 
        features_to_scale: List[str] = ['vehicle_count', 'hour', 'day_of_week', 'month'],
        target_column: str = 'vehicle_count',
        sequence_length: Optional[int] = 12, # e.g., 12 steps (could be 12 * 5-min intervals = 1 hour)
        use_sequences: bool = True # Set to False for non-sequential models like ARIMA on raw counts
    ) -> Optional[Dict[str, Path]]:
        """Runs the complete preprocessing pipeline."""
        if not self.load_data(): return None
        if not self.clean_data(): return None
        if not self.extract_temporal_features(): return None
        
        # Select features for the model (target will be handled by sequence creation or as y)
        # Example: only use vehicle_count and some temporal features
        model_features = [col for col in [target_column, 'hour', 'day_of_week', 'month'] if col in self.data.columns]
        if not model_features: 
            logger.error("No features selected for model training.")
            return None
        
        df_model_data = self.data[model_features].copy()

        # Scaling - apply to the selected features for modeling
        self.scaler = StandardScaler()
        df_model_data[model_features] = self.scaler.fit_transform(df_model_data[model_features])
        logger.info(f"Scaled model features: {model_features}")

        if use_sequences and sequence_length is not None:
            # Create sequences from the scaled data
            X_seq, y_seq = [], []
            # The target_column index needs to be found within the df_model_data
            target_idx_in_model_data = df_model_data.columns.get_loc(target_column)

            for i in range(len(df_model_data) - sequence_length):
                X_seq.append(df_model_data.iloc[i:(i + sequence_length)].values)
                y_seq.append(df_model_data.iloc[i + sequence_length, target_idx_in_model_data])
            
            if not X_seq or not y_seq:
                logger.warning("Could not create any sequences. Data might be too short.")
                return None
            X, y = np.array(X_seq), np.array(y_seq)
        elif not use_sequences:
            # For non-sequential models (e.g. ARIMA on vehicle_count directly, or regression)
            # X would be lagged features, y would be current vehicle_count.
            # This part needs specific implementation based on the non-sequential model.
            # For now, let's assume we want to predict vehicle_count using other features in a non-sequential way.
            y = df_model_data[target_column].values
            X_cols = [col for col in model_features if col != target_column]
            if not X_cols: 
                logger.error("No exogenous features for non-sequential model.")
                return None
            X = df_model_data[X_cols].values
            logger.info(f"Prepared data for non-sequential model. X shape: {X.shape}, y shape: {y.shape}")
        else:
            logger.error("Configuration error: use_sequences is True but sequence_length is None.")
            return None

        return self.split_and_save_data(X, y)

# Example usage:
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    mock_input_csv_dir = Path("__test_mock_interval_csvs__") # From IntegratedDatasetCreator test
    mock_input_csv_dir.mkdir(exist_ok=True)
    mock_output_ml_dir = Path("__test_mock_ml_ready_data__")

    # Create a dummy CSV if not present from previous step
    dummy_csv_path = mock_input_csv_dir / "traffic_counts_60sec_intervals.csv"
    if not dummy_csv_path.exists():
        logger.info(f"Creating dummy CSV for preprocessor test: {dummy_csv_path}")
        timestamps = pd.date_range(start='2023-01-01', periods=200, freq='1min')
        vehicle_counts = np.random.randint(0, 50, size=200)
        dummy_df = pd.DataFrame({'interval_start_time': timestamps, 'vehicle_count': vehicle_counts})
        dummy_df.to_csv(dummy_csv_path, index=False)

    preprocessor = IntegratedDataPreprocessor(
        input_csv_path=str(dummy_csv_path),
        output_ml_data_dir=str(mock_output_ml_dir)
    )
    
    saved_file_paths = preprocessor.run_full_preprocessing_pipeline(use_sequences=True, sequence_length=12)
    if saved_file_paths:
        logger.info(f"Preprocessing pipeline successful. ML data saved: {saved_file_paths}")
    else:
        logger.error("Preprocessing pipeline failed.")

    # Test non-sequential path
    # saved_file_paths_non_seq = preprocessor.run_full_preprocessing_pipeline(use_sequences=False)
    # if saved_file_paths_non_seq:
    #     logger.info(f"Non-sequential preprocessing pipeline successful. ML data saved: {saved_file_paths_non_seq}")
    # else:
    #     logger.error("Non-sequential preprocessing pipeline failed.")

    # Clean up
    # import shutil
    # shutil.rmtree(mock_output_ml_dir, ignore_errors=True)
    # If dummy_csv_path was created by this test only: os.remove(dummy_csv_path) 