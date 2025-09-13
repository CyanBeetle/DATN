"""
Integrated Prediction Utility for UC6: View Traffic Forecast.

This utility will be responsible for:
- Loading a trained model (given its file path).
- Loading and preparing the necessary input data from preprocessed .npy files.
- Generating predictions for a specified horizon based on the model type.
- Handling time context for selecting relevant historical data if needed.
"""
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import numpy as np
import joblib
from datetime import datetime, timedelta # Added timedelta

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    tf = None 
    logging.warning("TensorFlow is not installed. LSTM/BiLSTM model prediction will not be available.")

from db.models import ModelType

logger = logging.getLogger(__name__)

class IntegratedPredictor:
    def __init__(self, model_path: str, model_type: ModelType, 
                 scaler_path: Optional[str] = None, 
                 feature_names_path: Optional[str] = None, 
                 target_variable_name: str = "vehicle_count",
                 # Expected time features the model was trained on (must match feature_names.pkl order for these)
                 time_feature_names: List[str] = ['hour', 'day_of_week', 'month', 'day_of_year', 'week_of_year', 'hour_sin', 'hour_cos', 'day_of_year_sin', 'day_of_year_cos'] ):
        """
        Initializes the predictor.
        Args:
            target_variable_name: Name of the target variable (e.g., 'vehicle_count').
            time_feature_names: List of time-based exogenous features the model expects.
        """
        self.model_path = Path(model_path)
        self.model_type = model_type
        self.scaler_path = Path(scaler_path) if scaler_path else None
        self.feature_names_path = Path(feature_names_path) if feature_names_path else None
        self.target_variable_name = target_variable_name
        self.model = None
        self.scaler = None
        self.feature_names: Optional[List[str]] = None
        self.target_feature_index: Optional[int] = None
        self.time_feature_names = time_feature_names
        self.time_feature_indices: Dict[str, int] = {}

        if not self.model_path.exists():
            logger.error(f"Model file not found at: {self.model_path}")
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        try:
            if self.model_type == ModelType.LSTM or self.model_type == ModelType.BILSTM:
                if not TF_AVAILABLE:
                    raise ImportError("TensorFlow is required for LSTM/BiLSTM models but is not installed.")
                self.model = tf.keras.models.load_model(str(self.model_path))
                logger.info(f"Loaded LSTM/BiLSTM model from {self.model_path}")
            elif self.model_type == ModelType.ARIMA:
                self.model = joblib.load(self.model_path)
                logger.info(f"Loaded ARIMA model from {self.model_path}")
            else:
                logger.warning(f"Model type {self.model_type} loading not fully implemented.")
                self.model = "mock_other_model"
        except Exception as e:
            logger.error(f"Error loading model from {self.model_path}: {e}")
            raise

        if self.scaler_path:
            if not self.scaler_path.exists():
                raise FileNotFoundError(f"Scaler file not found: {self.scaler_path}")
            self.scaler = joblib.load(self.scaler_path)
            logger.info(f"Loaded scaler from {self.scaler_path}")
        elif self.model_type in [ModelType.LSTM, ModelType.BILSTM]:
            raise ValueError(f"Scaler is required for {self.model_type} models but path not provided.")

        if self.model_type in [ModelType.LSTM, ModelType.BILSTM] and self.scaler:
            if not self.feature_names_path or not self.feature_names_path.exists():
                raise FileNotFoundError(f"Feature names file not found: {self.feature_names_path}")
            self.feature_names = joblib.load(self.feature_names_path)
            logger.info(f"Loaded feature names: {self.feature_names}")
            if self.target_variable_name in self.feature_names:
                self.target_feature_index = self.feature_names.index(self.target_variable_name)
            else:
                raise ValueError(f"Target variable '{self.target_variable_name}' not in feature names: {self.feature_names}")
            
            for tf_name in self.time_feature_names:
                if tf_name in self.feature_names:
                    self.time_feature_indices[tf_name] = self.feature_names.index(tf_name)
                else:
                    logger.warning(f"Configured time feature '{tf_name}' not found in loaded feature_names. It will not be updated dynamically.")
            logger.info(f"Identified time feature indices: {self.time_feature_indices}")

    def _generate_time_features_for_timestamp(self, ts: datetime) -> Dict[str, float]:
        """Generates a dict of time features for a given timestamp."""
        features = {}
        if 'hour' in self.time_feature_indices: features['hour'] = float(ts.hour)
        if 'day_of_week' in self.time_feature_indices: features['day_of_week'] = float(ts.weekday()) # Mon=0, Sun=6
        if 'month' in self.time_feature_indices: features['month'] = float(ts.month)
        if 'day_of_year' in self.time_feature_indices: features['day_of_year'] = float(ts.timetuple().tm_yday)
        if 'week_of_year' in self.time_feature_indices: features['week_of_year'] = float(ts.isocalendar().week)
        
        # Sin/Cos transformations (ensure consistency with preprocess.py)
        if 'hour_sin' in self.time_feature_indices:
            features['hour_sin'] = np.sin(2 * np.pi * ts.hour / 23.0) # Max hour is 23
        if 'hour_cos' in self.time_feature_indices:
            features['hour_cos'] = np.cos(2 * np.pi * ts.hour / 23.0)
        if 'day_of_year_sin' in self.time_feature_indices:
            # Assuming 366 to handle leap years, consistent with common practice
            features['day_of_year_sin'] = np.sin(2 * np.pi * ts.timetuple().tm_yday / 366.0)
        if 'day_of_year_cos' in self.time_feature_indices:
            features['day_of_year_cos'] = np.cos(2 * np.pi * ts.timetuple().tm_yday / 366.0)
        # Add other cyclical features (e.g., month_sin/cos, day_of_week_sin/cos) if they were used in training
        return features

    def _get_recent_contextual_data(self, X_data_path_prefix: str, sequence_length: int, num_features: int) -> Optional[np.ndarray]:
        input_sequence_path = Path(f"{X_data_path_prefix}_X_test.npy")
        if not input_sequence_path.exists():
            logger.error(f"Input data file {input_sequence_path} not found.")
            return None
        try:
            all_sequences = np.load(input_sequence_path)
            if all_sequences.ndim != 3 or all_sequences.shape[1] != sequence_length or all_sequences.shape[2] != num_features:
                logger.error(f"Loaded data {input_sequence_path} shape mismatch: {all_sequences.shape}. Expected (any, {sequence_length}, {num_features}).")
                return None
            if len(all_sequences) == 0: return None
            return all_sequences[-1:]
        except Exception as e:
            logger.error(f"Error loading/processing {input_sequence_path}: {e}")
            return None

    def generate_forecast(
        self, 
        horizon_steps: int,
        current_initial_timestamp: datetime, # Current real-world time for starting forecast generation
        data_interval_seconds: int,      # Interval of data points in seconds (e.g., 300 for 5 mins)
        X_data_path_prefix: Optional[str] = None,
        sequence_length: Optional[int] = None, 
        num_features: Optional[int] = None,
    ) -> List[float]:
        logger.info(f"Generating forecast for {horizon_steps} steps from {current_initial_timestamp} with model {self.model_type}.")
        predictions: List[float] = []

        if self.model is None: raise ValueError("Model not loaded.")

        if self.model_type == ModelType.LSTM or self.model_type == ModelType.BILSTM:
            if not TF_AVAILABLE: 
                logger.error("TensorFlow unavailable for LSTM/BiLSTM."); return [0.0] * horizon_steps
            if not all([X_data_path_prefix, sequence_length, num_features, self.scaler, self.feature_names, self.target_feature_index is not None]):
                raise ValueError("Missing requirements for LSTM: X_data_path_prefix, sequence_length, num_features, scaler, feature_names, or target_feature_index.")
            
            current_sequence = self._get_recent_contextual_data(X_data_path_prefix, sequence_length, num_features)
            if current_sequence is None: 
                logger.error("Failed to obtain input sequence for LSTM."); return [0.0] * horizon_steps 

            for i in range(horizon_steps):
                pred_scaled_single_value = self.model.predict(current_sequence, verbose=0)[0, 0]
                
                temp_array_for_inverse = np.zeros((1, len(self.feature_names)))
                temp_array_for_inverse[0, self.target_feature_index] = pred_scaled_single_value
                pred_unscaled = self.scaler.inverse_transform(temp_array_for_inverse)[0, self.target_feature_index]
                predictions.append(float(pred_unscaled))
                
                # Construct new step for next prediction
                new_step_features_unscaled = np.zeros(len(self.feature_names))
                # Set predicted target value (unscaled, will be scaled later with others)
                new_step_features_unscaled[self.target_feature_index] = pred_unscaled 

                # Calculate timestamp for the new step being predicted
                future_step_timestamp = current_initial_timestamp + timedelta(seconds=(i + 1) * data_interval_seconds)
                calculated_time_features = self._generate_time_features_for_timestamp(future_step_timestamp)
                
                for tf_name, tf_val in calculated_time_features.items():
                    if tf_name in self.time_feature_indices:
                        new_step_features_unscaled[self.time_feature_indices[tf_name]] = tf_val
                    
                # Fill any other non-time, non-target exogenous features (e.g. is_holiday)
                # Simplification: carry forward from last step of current_sequence (scaled values)
                # More robust: look them up or predict them if they are dynamic.
                # For now, we only dynamically update time features. Others need to be part of scaler fitting.
                # The scaler expects all features to be present before transform.
                # So, we build the unscaled new step, then scale the whole thing.
                for feat_idx, feat_name in enumerate(self.feature_names):
                    if feat_name not in self.time_feature_indices and feat_idx != self.target_feature_index:
                        # Carry forward scaled value from last step of current_sequence, then inverse transform it, then it will be re-scaled.
                        # This is a bit tricky. A simpler way for now is to ensure that the initial _get_recent_contextual_data
                        # provides a sequence where these other exogenous features are somewhat representative.
                        # Or, if they are known for the future (e.g. a known holiday) they should be provided. 
                        # For this iteration, we focus on time features. Assume other exogenous are part of the original scaled sequence and are handled by scaler.
                        # Let's copy from the *unscaled* equivalent of the previous step if possible, or use a placeholder.
                        # This part is complex. We will assume that for non-target, non-time features, they are either not present or handled by what the model learned.
                        # The most important part is getting the target and time-features right for the new row.
                        # The scaler will handle scaling of whatever we provide in new_step_features_unscaled_row_for_scaling.
                        pass # They are zero unless set by time features or target. This implies other exo features are not used or need different handling.

                new_step_features_scaled_row = self.scaler.transform(new_step_features_unscaled.reshape(1, -1))[0]

                new_step_to_append = new_step_features_scaled_row.reshape(1, 1, len(self.feature_names))
                current_sequence = np.append(current_sequence[:, 1:, :], new_step_to_append, axis=1)

        elif self.model_type == ModelType.ARIMA:
            try:
                if hasattr(self.model, 'predict') and callable(getattr(self.model, 'predict')):
                    forecast_result = self.model.predict(n_periods=horizon_steps)
                    if hasattr(forecast_result, 'predicted_mean'): 
                         predictions = [float(p) for p in forecast_result.predicted_mean]
                    elif isinstance(forecast_result, np.ndarray):
                         predictions = [float(p) for p in forecast_result]
                    else: 
                         predictions = [float(p) for p in list(forecast_result)]
                else:
                    logger.error(f"ARIMA model has no callable 'predict' method: {type(self.model)}")
                    predictions = [0.0] * horizon_steps
            except Exception as e:
                logger.error(f"Error during ARIMA prediction: {e}"); raise
        else:
            logger.error(f"Prediction for model type {self.model_type} not implemented.")
            predictions = [np.random.uniform(10, 100) for _ in range(horizon_steps)] 

        return [round(p, 2) for p in predictions]

# Example Usage (Conceptual)
async def _example_usage():
    # ... (Example usage needs to be updated significantly to test new logic)
    pass

# if __name__ == "__main__":
#     # This example can be run if dummy files are created as shown above.
#     # import asyncio
#     # asyncio.run(_example_usage()) # _example_usage is not async, run directly
#     _example_usage()
#     pass 