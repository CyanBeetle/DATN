"""
Service for UC6: View Traffic Forecast.

This service orchestrates fetching the default prediction model, 
its associated data, and using the IntegratedPredictor to generate a forecast.
It also handles the logic for potentially using a baseline ARIMA model for comparison on short horizons.
"""
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from pathlib import Path

from fastapi import HTTPException, status
from motor.motor_asyncio import AsyncIOMotorDatabase
from pydantic import BaseModel
import joblib

from db.models import PredictionModelInDB, TrainingDatasetInDB, ModelType
from db.crud import prediction_model_crud, training_dataset_crud
from ml.utils.integrated_predictor import IntegratedPredictor, TF_AVAILABLE
from app.config import settings

logger = logging.getLogger(__name__)

class ForecastPoint(BaseModel):
    timestamp: datetime
    predicted_value: float

class SingleModelForecast(BaseModel):
    model_name_type: str # e.g., "Main LSTM Model (LSTM)"
    model_version: str
    points: List[ForecastPoint]
    confidence: Optional[float] = None # Static confidence for now

class TrafficForecastResponse(BaseModel):
    primary_forecast: SingleModelForecast
    comparison_forecast: Optional[SingleModelForecast] = None # For ARIMA on short horizons
    forecast_generated_at: datetime
    forecast_horizon_minutes: int
    data_interval_minutes: int
    caveats: Optional[str] = None

async def _generate_single_model_forecast(
    db: AsyncIOMotorDatabase,
    model_doc: PredictionModelInDB,
    current_time_utc: datetime,
    horizon_steps: int,
    data_interval_seconds: int,
    is_comparison_model: bool = False # To slightly alter logging/naming if needed
) -> SingleModelForecast:
    """Helper to generate forecast from a single specified model."""
    if not model_doc.file_path or not Path(model_doc.file_path).exists():
        logger.error(f"Model file missing for {model_doc.name} ({model_doc.id})")
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=f"Model file missing for {model_doc.name}.")

    if not model_doc.source_dataset_ids:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=f"Model {model_doc.name} has no source_dataset_ids.")

    primary_source_bundle_id_str = str(model_doc.source_dataset_ids[0])
    source_bundle_doc = await training_dataset_crud.get_training_dataset_by_id(db, primary_source_bundle_id_str)
    if not (source_bundle_doc and source_bundle_doc.data_type == "ML_BUNDLE" and source_bundle_doc.ml_dataset_path_prefix):
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=f"Training data for model {model_doc.name} misconfigured.")

    ml_data_path_prefix = source_bundle_doc.ml_dataset_path_prefix
    scaler_path_str = f"{ml_data_path_prefix}_scaler.pkl"
    feature_names_path_str = f"{ml_data_path_prefix}_feature_names.pkl"
    scaler_path_for_predictor = None
    feature_names_path_for_predictor = None

    if model_doc.model_type in [ModelType.LSTM, ModelType.BILSTM]:
        if Path(scaler_path_str).exists(): scaler_path_for_predictor = scaler_path_str
        else: raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=f"Scaler file missing for {model_doc.model_type.value} model {model_doc.name}.")
        if Path(feature_names_path_str).exists(): feature_names_path_for_predictor = feature_names_path_str
        else: raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=f"Feature names file missing for {model_doc.model_type.value} model {model_doc.name}.")
    
    training_params = model_doc.training_parameters or {}
    # data_interval_seconds is passed in, but it should match what the model was trained with.
    # sequence_length, num_features are critical for LSTM based models.
    sequence_length = training_params.get("sequence_length")
    num_features = training_params.get("num_features_for_model")
    target_variable_name = training_params.get("target_variable_name", "vehicle_count")
    model_specific_time_features = training_params.get("time_feature_names", IntegratedPredictor.__init__.__defaults__[3]) 

    if model_doc.model_type in [ModelType.LSTM, ModelType.BILSTM] and not num_features and feature_names_path_for_predictor:
        try:
            loaded_feature_names = joblib.load(feature_names_path_for_predictor)
            num_features = len(loaded_feature_names)
        except Exception as e:
            logger.warning(f"Could not load feature_names for {model_doc.name} to infer num_features: {e}")

    predictor = IntegratedPredictor(
        model_path=model_doc.file_path,
        model_type=model_doc.model_type,
        scaler_path=scaler_path_for_predictor,
        feature_names_path=feature_names_path_for_predictor,
        target_variable_name=target_variable_name,
        time_feature_names=model_specific_time_features
    )
    predicted_values = predictor.generate_forecast(
        horizon_steps=horizon_steps,
        current_initial_timestamp=current_time_utc,
        data_interval_seconds=data_interval_seconds,
        X_data_path_prefix=ml_data_path_prefix,
        sequence_length=sequence_length,
        num_features=num_features
    )

    forecast_points: List[ForecastPoint] = []
    next_timestamp = current_time_utc 
    for val in predicted_values:
        next_timestamp += timedelta(seconds=data_interval_seconds)
        forecast_points.append(ForecastPoint(timestamp=next_timestamp, predicted_value=val))
    
    # Static confidence for now
    confidence = 0.85 if model_doc.model_type in [ModelType.LSTM, ModelType.BILSTM] else 0.75
    if is_comparison_model: confidence -= 0.1 # Slightly lower for comparison

    return SingleModelForecast(
        model_name_type=f"{model_doc.name} ({model_doc.model_type.value})",
        model_version=model_doc.version,
        points=forecast_points,
        confidence=confidence
    )

async def get_traffic_forecast(
    db: AsyncIOMotorDatabase,
    current_time_utc: datetime,
    horizon_minutes: int, 
) -> TrafficForecastResponse:
    if not (5 <= horizon_minutes <= 1440):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Forecast horizon must be between 5 and 1440 minutes.")

    # 1. Get the default model (this will be LSTM/BiLSTM or any other type set as default)
    default_model = await prediction_model_crud.get_default_prediction_model(db)
    if not default_model:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No default active prediction model configured.")
    logger.info(f"Primary model for forecast: {default_model.name} (ID: {default_model.id}, Type: {default_model.model_type.value})")

    # Determine data interval from the default model's training parameters or its source bundle
    # This assumes all models used (default and baseline ARIMA) are compatible with this interval.
    default_model_training_params = default_model.training_parameters or {}
    data_interval_seconds = default_model_training_params.get("interval_seconds")
    if not data_interval_seconds:
        if default_model.source_dataset_ids:
            source_bundle_doc = await training_dataset_crud.get_training_dataset_by_id(db, str(default_model.source_dataset_ids[0]))
            if source_bundle_doc and source_bundle_doc.training_parameters:
                 data_interval_seconds = source_bundle_doc.training_parameters.get("interval_seconds")
    if not data_interval_seconds or data_interval_seconds <=0:
        data_interval_seconds = settings.DATA_AGGREGATION_INTERVAL_SECONDS # Fallback
        logger.warning(f"Could not determine data_interval_seconds from model/bundle, falling back to default: {data_interval_seconds}s")

    data_interval_minutes = data_interval_seconds / 60.0
    horizon_steps = int(round(horizon_minutes / data_interval_minutes))
    if horizon_steps <= 0: raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Horizon too short for data interval.")

    primary_forecast_result: Optional[SingleModelForecast] = None
    comparison_forecast_result: Optional[SingleModelForecast] = None
    caveats_list = []

    try:
        primary_forecast_result = await _generate_single_model_forecast(
            db, default_model, current_time_utc, horizon_steps, data_interval_seconds
        )
        if not TF_AVAILABLE and default_model.model_type in [ModelType.LSTM, ModelType.BILSTM]:
            caveats_list.append("TensorFlow not installed; Primary model (LSTM/BiLSTM) predictions may be unreliable or mock.")
        elif isinstance(default_model.file_path, str) and "mock" in default_model.file_path.lower(): # A bit of a hack to check mock
             caveats_list.append("Primary forecast may use a MOCK/PLACEHOLDER model.")

    except Exception as e:
        logger.exception(f"Error generating primary forecast with model {default_model.id}: {e}")
        # Decide if we should still try ARIMA or just fail.
        # For now, let primary model failure be critical.
        if isinstance(e, HTTPException): raise e
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error generating primary forecast: {str(e)}")

    # 2. For horizons <= 30 minutes, also try to get baseline ARIMA forecast
    if horizon_minutes <= 30:
        baseline_arima_model = await prediction_model_crud.get_baseline_arima_model(db)
        if baseline_arima_model:
            logger.info(f"Generating comparison forecast with baseline ARIMA: {baseline_arima_model.name} (ID: {baseline_arima_model.id})")
            try:
                comparison_forecast_result = await _generate_single_model_forecast(
                    db, baseline_arima_model, current_time_utc, horizon_steps, data_interval_seconds, is_comparison_model=True
                )
                if isinstance(baseline_arima_model.file_path, str) and "mock" in baseline_arima_model.file_path.lower():
                    caveats_list.append("Comparison ARIMA forecast may use a MOCK/PLACEHOLDER model.")
            except Exception as e:
                logger.warning(f"Could not generate comparison forecast with ARIMA model {baseline_arima_model.id}: {e}")
                caveats_list.append(f"Failed to generate comparison ARIMA forecast: {str(e)[:100]}")
        else:
            logger.info("No baseline ARIMA model configured or active for comparison.")
            caveats_list.append("No baseline ARIMA model available for short-term comparison.")

    final_caveats = "; ".join(caveats_list) if caveats_list else None

    return TrafficForecastResponse(
        primary_forecast=primary_forecast_result,
        comparison_forecast=comparison_forecast_result,
        forecast_generated_at=datetime.utcnow(),
        forecast_horizon_minutes=horizon_minutes,
        data_interval_minutes=round(data_interval_minutes, 2),
        caveats=final_caveats
    )

# Need to import Path from pathlib and BaseModel from Pydantic for the service file to be self-contained initially
from pathlib import Path
from pydantic import BaseModel

# Need to ensure joblib is imported if used for loading feature_names for num_features inference
import joblib 