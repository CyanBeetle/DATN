from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any
import logging
import pandas as pd
import tensorflow as tf # Added for model loading
import pickle # Added for scaler loading
import os # Added for path normalization
import numpy as np

from ml.services.file_based_model_service import list_filesystem_models, get_filesystem_model_details
# Updated to reflect the renamed function in prediction_preparation_utils.py
from ml.utils.prediction_preparation_utils import load_and_prepare_feature_engineered_input, STANDARD_FEATURES_ORDER # Removed load_raw_data_for_background_shap as it's replaced by new logic below
from app.config import settings # For model/data paths if needed, though service might handle it

router = APIRouter()
logger = logging.getLogger(__name__)

@router.get("/models-overview", response_model=List[Dict[str, Any]])
async def get_admin_models_overview():
    """Provides a comprehensive overview of all discovered filesystem models for the admin dashboard (UC11, UC12)."""
    logger.info("Admin request for models overview.")
    try:
        models_base_info = list_filesystem_models()
        detailed_models_overview = []

        if not models_base_info:
            logger.warning("No models found by file_based_model_service for admin overview.")
            return []

        for model_base in models_base_info:
            # Only try to get full details if the model and scaler were found
            if model_base["status"] == "Available":
                details = get_filesystem_model_details(model_base["id"])
                if details: # Should contain input_shape, output_shape etc.
                    detailed_models_overview.append(details)
                else:
                    # If details fetching failed (e.g. model load error), append base info with error status
                    error_info = model_base.copy()
                    error_info["input_shape"] = "Error retrieving details"
                    error_info["output_shape"] = "Error retrieving details"
                    error_info["status"] = details.get("status", "Error Retrieving Details") if details else "Error Retrieving Details"
                    detailed_models_overview.append(error_info)
            else:
                # If model status is not 'Available' (e.g. 'Missing Scaler'), just append base info
                detailed_models_overview.append(model_base)
        
        return detailed_models_overview
    except Exception as e:
        logger.error(f"Error in /api/admin/ml/models-overview: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error retrieving models overview for admin.")