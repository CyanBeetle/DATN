from fastapi import APIRouter, HTTPException, Body
from typing import List, Dict, Any, Optional
import logging
import os
import pickle # For loading scalers
import tensorflow as tf
import pandas as pd
import numpy as np
import io
import base64
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import random # For weather harshness adjustment
from pydantic import BaseModel

# Assuming services and utils are correctly importable based on python path
from ml.services.file_based_model_service import list_filesystem_models, get_filesystem_model_details
from ml.utils.prediction_preparation_utils import (
    load_and_prepare_feature_engineered_input,
    STANDARD_FEATURES_ORDER,
)

router = APIRouter()
logger = logging.getLogger(__name__)

# Configure Matplotlib to use a non-GUI backend
plt.switch_backend('Agg')

class PredictionInput(BaseModel):
    model_identifier: str # e.g., "Set1/my_model.keras"
    day_of_week: int # 0 for Monday, 6 for Sunday
    time_of_day: str # HH:MM format, e.g., "08:30"
    weather_harsh: bool = False

def generate_future_timestamps(start_time_str: str, n_steps_out: int, granularity_minutes: int) -> List[str]:
    start_time = datetime.strptime(start_time_str, "%H:%M")
    timestamps = []
    current_time = start_time
    for _ in range(n_steps_out):
        timestamps.append(current_time.strftime("%H:%M"))
        current_time += timedelta(minutes=granularity_minutes)
    return timestamps

@router.get("/models", response_model=List[Dict[str, Any]])
async def get_available_models_for_user():
    """Lists available models for user forecast selection (UC06)."""
    try:
        models = list_filesystem_models()
        # Filter for models that are 'Available' (model + scaler found)
        available_models = [
            {
                "id": m["id"],
                "display_name": f"{m['set_name']} - {m['model_name']}",
                "inferred_horizon": m['model_name'] # Use model_name as the horizon identifier
            }
            for m in models if m["status"] == "Available"
        ]
        if not available_models:
            logger.warning("No 'Available' models found by file_based_model_service for user endpoint.")
            # Return empty list, frontend should handle this
        return available_models
    except Exception as e:
        logger.error(f"Error in /api/ml/models: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error retrieving available models.")

@router.post("/predict")
async def perform_prediction(input_data: PredictionInput = Body(...)):
    """Performs traffic forecast using a selected model (UC06)."""
    logger.info(f"Received prediction request for model: {input_data.model_identifier}, day: {input_data.day_of_week}, time: {input_data.time_of_day}")

    model_details = get_filesystem_model_details(input_data.model_identifier)
    if not model_details or model_details["status"] != "Available":
        logger.error(f"Model {input_data.model_identifier} not found or not available for prediction.")
        raise HTTPException(status_code=404, detail=f"Model '{input_data.model_identifier}' not found or is misconfigured.")

    model_path = model_details["model_path"]
    scaler_path = model_details["scaler_path"]
    model_display_name = model_details["id"]
    model_file_name_only = model_details["model_name"]

    try:
        model = tf.keras.models.load_model(model_path, compile=False)

        with open(scaler_path, 'rb') as f:
            all_scalers_data = pickle.load(f)
            # horizon_key = model_details.get("horizon_name") # No longer needed to index all_scalers_data
            
            # if not horizon_key:
            #     logger.error(f"CRITICAL: horizon_name not found in model_details for model ID {input_data.model_identifier}.")
            #     raise HTTPException(status_code=500, detail="Scaler configuration error: Model details incomplete (missing horizon_name).")

            speed_scaler = None
            feature_specific_scalers = None 

            # logger.info(f"Attempting to find scalers for horizon_key: '{horizon_key}' in {scaler_path}")
            logger.info(f"Loading scalers from: {scaler_path}")

            # all_scalers_data directly contains the feature-to-scaler mapping
            if isinstance(all_scalers_data, dict):
                feature_specific_scalers = all_scalers_data
                logger.info(f"Successfully loaded feature_specific_scalers dict. Keys: {list(feature_specific_scalers.keys())}")
                
                if 'speed_kmh' in feature_specific_scalers:
                    speed_scaler = feature_specific_scalers['speed_kmh']
                    if not (hasattr(speed_scaler, 'transform') and hasattr(speed_scaler, 'inverse_transform')):
                        logger.error(f"Object for 'speed_kmh' is not a valid scaler in {scaler_path}.")
                        speed_scaler = None # Invalidate it
                else:
                    logger.warning(f"'speed_kmh' (target) scaler key not found directly in {scaler_path}.")
            else:
                logger.error(f"Content of {scaler_path} is not a dictionary as expected for feature_specific_scalers.")

            if not speed_scaler:
                logger.error(f"CRITICAL: Target variable ('speed_kmh') scaler not found or invalid for model {model_display_name}.")
                raise HTTPException(status_code=500, detail="Scaler config error: Could not find/validate target variable scaler.")
            
            if not feature_specific_scalers:
                logger.error(f"CRITICAL: Feature-specific scalers dictionary not loaded for model {model_display_name}.")
                raise HTTPException(status_code=500, detail="Scaler config error: Could not load feature-specific scalers dict.")

        n_steps_in = model.input_shape[1]
        n_steps_out = model.output_shape[1]
        num_model_features = model.input_shape[2]

        if num_model_features != len(STANDARD_FEATURES_ORDER):
            msg = (
                f"Model {model_display_name} expects {num_model_features} features, "
                f"but STANDARD_FEATURES_ORDER has {len(STANDARD_FEATURES_ORDER)} features. This is a mismatch."
            )
            logger.error(msg)
            raise HTTPException(status_code=500, detail=msg)

        # Load and prepare input data using the MODIFIED utility, passing user's day and time
        input_df_unscaled_ordered = load_and_prepare_feature_engineered_input(
            n_steps_in=n_steps_in,
            target_day_of_week=input_data.day_of_week,
            target_time_str=input_data.time_of_day
        )

        if input_df_unscaled_ordered is None:
            logger.error(f"Failed to prepare input data for day={input_data.day_of_week}, time={input_data.time_of_day}.")
            raise HTTPException(status_code=500, detail=f"Error preparing input data. No data found for selected day/time or insufficient history.")
        
        if len(input_df_unscaled_ordered) != n_steps_in:
            logger.error(f"Prepared data has {len(input_df_unscaled_ordered)} rows, but model expects {n_steps_in}.")
            raise HTTPException(status_code=500, detail="Input data row count mismatch after preparation.")

        if input_df_unscaled_ordered.shape[1] != num_model_features:
            msg = (
                f"Feature count mismatch for model {model_display_name} after preparation: "
                f"Model expects {num_model_features} features, "
                f"prepared input has {input_df_unscaled_ordered.shape[1]} features. "
                f"Expected order: {STANDARD_FEATURES_ORDER}."
            )
            logger.error(msg)
            logger.error(f"Columns in prepared data: {input_df_unscaled_ordered.columns.tolist()}")
            raise HTTPException(status_code=500, detail=msg)
            
        # Ensure columns are in STANDARD_FEATURES_ORDER before scaling
        input_df_unscaled_ordered = input_df_unscaled_ordered[STANDARD_FEATURES_ORDER]
        
        # Perform per-feature scaling
        input_df_scaled = input_df_unscaled_ordered.copy()
        scaled_feature_count = 0
        for feature_name in STANDARD_FEATURES_ORDER:
            if feature_name in feature_specific_scalers:
                scaler = feature_specific_scalers[feature_name]
                if hasattr(scaler, 'transform'):
                    try:
                        # Scaler expects 2D array: df[[feature_name]] gives a DataFrame (2D)
                        scaled_column = scaler.transform(input_df_unscaled_ordered[[feature_name]])
                        input_df_scaled[feature_name] = scaled_column.flatten() # Flatten if transform returns 2D column vector
                        scaled_feature_count += 1
                    except Exception as e_scale_feat:
                        logger.error(f"Error scaling feature '{feature_name}' for model {model_display_name}: {e_scale_feat}")
                        raise HTTPException(status_code=500, detail=f"Error during scaling of feature '{feature_name}'.")
                else:
                    logger.warning(f"Scaler found for feature '{feature_name}' but it lacks a 'transform' method. Using original data.")
            else:
                # This is expected for features not in `numerical_features` during training (e.g. cyclical, one-hot)
                logger.debug(f"No specific scaler found for feature '{feature_name}'. Using original data for this feature.")
        
        logger.info(f"Performed per-feature scaling. {scaled_feature_count} features were scaled out of {len(STANDARD_FEATURES_ORDER)}.")
        
        model_input_array = input_df_scaled.values.reshape((1, n_steps_in, num_model_features))

        raw_predictions = model.predict(model_input_array)[0]
        predicted_speeds_scaled = raw_predictions.reshape(-1, 1) 
        predicted_speeds_actual = speed_scaler.inverse_transform(predicted_speeds_scaled).flatten()

        if input_data.weather_harsh:
            reduction = random.uniform(0.15, 0.25)
            predicted_speeds_actual = np.maximum(0, predicted_speeds_actual * (1 - reduction))

        granularity_minutes = 5 # Default
        model_metadata = model_details.get("model_specific_metadata", {})
        granularity_from_meta_raw = model_metadata.get("granularity")

        if granularity_from_meta_raw is not None:
            if isinstance(granularity_from_meta_raw, int):
                granularity_minutes = granularity_from_meta_raw
                logger.info(f"Using integer granularity {granularity_minutes} from metadata.")
            elif isinstance(granularity_from_meta_raw, str):
                try:
                    # Attempt to extract just the number part if it's a string like "5 minutes"
                    numeric_part = "".join(filter(str.isdigit, granularity_from_meta_raw))
                    if numeric_part:
                        granularity_minutes = int(numeric_part)
                        logger.info(f"Extracted numeric granularity {granularity_minutes} from metadata string '{granularity_from_meta_raw}'.")
                    else:
                        logger.warning(f"Could not extract numeric part from granularity string '{granularity_from_meta_raw}'. Falling back to filename inference.")
                        # Fallback will be handled by the else block after this if-elif-else
                except ValueError:
                    logger.warning(f"ValueError converting extracted numeric part from '{granularity_from_meta_raw}' to int. Falling back.")
                    # Fallback
            else:
                 logger.warning(f"Granularity '{granularity_from_meta_raw}' from metadata is not an int or string. Falling back.")
                 # Fallback
        
        # If granularity_minutes is still the default (5) AND granularity_from_meta_raw was None (meaning it wasn't in metadata)
        # OR if parsing from meta failed and it needs to fallback from string/other types
        # then apply filename inference.
        # This logic ensures filename inference is a true fallback.
        if granularity_from_meta_raw is None or \
           (isinstance(granularity_from_meta_raw, str) and not "".join(filter(str.isdigit, granularity_from_meta_raw))) or \
           (not isinstance(granularity_from_meta_raw, int) and not isinstance(granularity_from_meta_raw, str)):
            logger.info(f"Granularity not successfully parsed from metadata (raw value: '{granularity_from_meta_raw}'). Attempting filename inference.")
            filename_lower = model_file_name_only.lower()
            if any(x in filename_lower for x in ["10_min"]): granularity_minutes = 10
            elif any(x in filename_lower for x in ["15_min", "15min"]): granularity_minutes = 15
            elif any(x in filename_lower for x in ["30_min"]): granularity_minutes = 30
            elif "short_term" in filename_lower: granularity_minutes = 5 
            elif any(x in filename_lower for x in ["medium_term", "1hour", "60min", "cnn_lstm"]): granularity_minutes = 5
            elif "long_term" in filename_lower : granularity_minutes = 30 
            elif any(x in filename_lower for x in ["2hour", "6hour", "12hour"]): granularity_minutes = 30 
            
        granularity_display = f"{granularity_minutes}-minute intervals"
        # The future timestamps should start from the user's selected time + 1 granularity step
        # Example: user selects 08:30, granularity 5min. Predictions are for 08:35, 08:40, ...
        start_time_for_future = datetime.strptime(input_data.time_of_day, "%H:%M") # + timedelta(minutes=granularity_minutes) 
        # No, the prediction starts AT input_data.time_of_day for the first step out if model predicts t+1, t+2 from t.
        # Or, if model predicts from t+granularity, then input_data.time_of_day is the last point of input sequence.
        # The generate_future_timestamps will create labels from input_data.time_of_day onwards.
        predicted_timestamps = generate_future_timestamps(input_data.time_of_day, n_steps_out, granularity_minutes)


        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(predicted_timestamps, predicted_speeds_actual, 'b-o', label=f'Predicted Speed ({granularity_display})')
        ax.set_title(f"Traffic Speed Prediction - {model_display_name}")
        ax.set_ylabel('Speed (km/h)')
        ax.set_xlabel(f'Time (starting {input_data.time_of_day})')
        ax.tick_params(axis='x', rotation=45)
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
        fig.tight_layout()

        buffer = io.BytesIO()
        fig.savefig(buffer, format='png'); buffer.seek(0)
        plot_image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plt.close(fig)
        
        day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

        return {
            "model_file_name": model_display_name,
            "user_input": input_data.dict(),
            "day_name": day_names[input_data.day_of_week],
            "predicted_speeds": predicted_speeds_actual.tolist(),
            "predicted_timestamps": predicted_timestamps,
            "granularity_display": granularity_display,
            "plot_image_base64": plot_image_base64,
        }
    except HTTPException as http_exc: # Re-raise HTTPExceptions
        raise http_exc
    except Exception as e:
        logger.error(f"Error during prediction for model {input_data.model_identifier}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred during prediction: {str(e)}")