import os
import tensorflow as tf
from typing import List, Dict, Optional, Any
import numpy as np # Added for numpy type conversion
import pickle # For loading .pkl files

# Determine path to ModelStorage based on this script's location
_CURRENT_FILE_PATH = os.path.abspath(__file__)
_SERVICES_DIR = os.path.dirname(_CURRENT_FILE_PATH)
_ML_DIR = os.path.dirname(_SERVICES_DIR)
_BACKENDV3_DIR = os.path.dirname(_ML_DIR)
MODEL_STORAGE_DIR = os.path.join(_BACKENDV3_DIR, "ModelStorage")

# Verify the path and print a warning if not found
if not (os.path.exists(MODEL_STORAGE_DIR) and os.path.isdir(MODEL_STORAGE_DIR)):
    print(f"CRITICAL WARNING: Model storage directory NOT FOUND at calculated path: {MODEL_STORAGE_DIR}")
    print(f"Calculations based on __file__: {_CURRENT_FILE_PATH}")
    print(f"Please ensure the 'ModelStorage' directory exists at the root of the 'backendv3' directory.")
    # To prevent further errors, we can set MODEL_STORAGE_DIR to a state that list_filesystem_models will handle as an error
    # Or raise an exception, but for now, the existing error handling in list_filesystem_models should take over.

MODEL_SETS = ["Set1", "Set2", "Set3"]

def _get_display_horizon_from_dirname(dirname: str) -> str:
    """Generates a display-friendly horizon name from a directory name."""
    name_lower = dirname.lower()
    if "cnn_lstm" in name_lower:
        return "Medium-term (CNN-LSTM)"
    elif "short_term" in name_lower:
        return "Short-term"
    elif "medium_term" in name_lower:
        return "Medium-term"
    elif "long_term" in name_lower:
        return "Long-term"
    return dirname.replace("_", " ").capitalize()

def list_filesystem_models() -> List[Dict[str, Any]]:
    """Scans the filesystem for Keras models (model.keras), scalers (scalers.pkl),
    feature order (features_order.pkl), and all_models_metadata.pkl within
    MODEL_STORAGE_DIR/SetX/saved_models/.
    """
    discovered_models = []

    if not (os.path.exists(MODEL_STORAGE_DIR) and os.path.isdir(MODEL_STORAGE_DIR)):
        print(f"Error: Model storage directory not found or not a directory at '{MODEL_STORAGE_DIR}'. Cannot list models.")
        return []

    for set_name in MODEL_SETS:
        models_base_dir = os.path.join(MODEL_STORAGE_DIR, set_name, "saved_models")

        if not os.path.isdir(models_base_dir):
            # print(f"Info: Models base directory '{models_base_dir}' not found for set '{set_name}'. Skipping this set.")
            continue

        all_models_metadata_path_for_set = os.path.join(models_base_dir, "all_models_metadata.pkl")
        if not os.path.exists(all_models_metadata_path_for_set):
            # print(f"Warning: all_models_metadata.pkl not found in {models_base_dir}. Models from this set will not have this metadata linked.")
            all_models_metadata_path_for_set = None # Set to None if not found

        try:
            for horizon_dir_name in os.listdir(models_base_dir):
                horizon_full_path = os.path.join(models_base_dir, horizon_dir_name)
                if os.path.isdir(horizon_full_path):
                    model_file_path = os.path.join(horizon_full_path, "model.keras")
                    scaler_file_path = os.path.join(horizon_full_path, "scalers.pkl")
                    features_order_file_path = os.path.join(horizon_full_path, "features_order.pkl")

                    if os.path.exists(model_file_path):
                        model_id = os.path.join(set_name, "saved_models", horizon_dir_name)
                        display_model_name = _get_display_horizon_from_dirname(horizon_dir_name)
                        
                        status_parts = []
                        actual_scaler_path = None
                        scaler_actual_name = None
                        actual_features_order_path = None

                        if os.path.exists(scaler_file_path):
                            actual_scaler_path = scaler_file_path
                            scaler_actual_name = "scalers.pkl"
                        else:
                            status_parts.append("Missing Scaler")
                        
                        if os.path.exists(features_order_file_path):
                            actual_features_order_path = features_order_file_path
                        else:
                            status_parts.append("Missing Features Order")

                        status = "Available" if not status_parts else ", ".join(status_parts)

                        discovered_models.append({
                            "id": model_id,
                            "model_name": display_model_name, # User-friendly name for display
                            "horizon_name": horizon_dir_name, # Raw directory name
                            "set_name": set_name,
                            "model_filename": "model.keras", # Standardized Keras model filename
                            "model_path": model_file_path,
                            "scaler_path": actual_scaler_path,
                            "scaler_name": scaler_actual_name,
                            "features_order_path": actual_features_order_path,
                            "all_models_metadata_path": all_models_metadata_path_for_set, # Added path to metadata
                            "status": status,
                        })
        except Exception as e:
            print(f"Error scanning horizon directories in {models_base_dir}: {e}")
            
    return sorted(discovered_models, key=lambda x: x["id"])

def get_filesystem_model_details(model_id: str) -> Optional[Dict[str, Any]]:
    """Retrieves details for a specific model_id, including its input/output shapes,
    feature order, and relevant metadata.
    model_id is expected to be like 'Set1/saved_models/short_term'.
    """
    all_models = list_filesystem_models()
    model_info = next((m for m in all_models if m["id"] == model_id), None)

    if not model_info:
        print(f"Model with ID '{model_id}' not found.")
        return None
    
    # We can attempt to load metadata even if model status is not 'Available'
    # as metadata files might exist independently.
    # However, Keras model details (summary, shapes) require the model file.

    details = model_info.copy() # Start with existing info

    # Load Keras model details if available
    if model_info["status"] == "Available" and model_info["model_path"] and os.path.exists(model_info["model_path"]):
        model_path = model_info["model_path"]
        try:
            model = tf.keras.models.load_model(model_path, compile=False)
            details["input_shape"] = str(model.input_shape)
            details["output_shape"] = str(model.output_shape)

            summary_list = []
            model.summary(print_fn=lambda x: summary_list.append(x))
            details["summary"] = "\\n".join(summary_list)

            try:
                config = model.get_config()
                def convert_numpy_types(obj):
                    if isinstance(obj, dict):
                        return {k: convert_numpy_types(v) for k, v in obj.items()}
                    elif isinstance(obj, list):
                        return [convert_numpy_types(i) for i in obj]
                    elif hasattr(obj, 'tolist'): # For numpy arrays
                        return obj.tolist()
                    elif isinstance(obj, (np.integer, np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
                        return int(obj)
                    elif isinstance(obj, (np.floating, np.float_, np.float16, np.float32, np.float64)):
                        return float(obj)
                    elif isinstance(obj, np.bool_):
                        return bool(obj)
                    return obj
                details["parameters"] = convert_numpy_types(config)
            except Exception as e_config:
                print(f"Note: Could not retrieve model config/parameters for {model_path}: {e_config}")
                details["parameters"] = {"error": "Could not retrieve parameters"}
            
            del model
            tf.keras.backend.clear_session()

        except Exception as e:
            print(f"Error loading Keras model {model_path} to get shapes: {e}")
            details["input_shape"] = "Error loading model"
            details["output_shape"] = "Error loading model"
            details["summary"] = "Error loading model summary"
            details["parameters"] = {"error": "Error loading model parameters"}
            # Update status in the details if loading failed
            details["status_load_error"] = "Error Loading Keras Model"
    elif model_info["status"] != "Available":
         details["input_shape"] = "Model file or scaler missing"
         details["output_shape"] = "Model file or scaler missing"
         details["summary"] = "Model file or scaler missing"
         details["parameters"] = {"error": "Model file or scaler missing"}


    # Load features_order.pkl
    if model_info.get("features_order_path") and os.path.exists(model_info["features_order_path"]):
        try:
            with open(model_info["features_order_path"], 'rb') as f:
                features_order = pickle.load(f)
            details["features_order"] = features_order
        except Exception as e:
            print(f"Error loading features_order.pkl from {model_info['features_order_path']}: {e}")
            details["features_order"] = {"error": f"Could not load features_order.pkl: {e}"}
    else:
        details["features_order"] = "Not found or path missing"

    # Load all_models_metadata.pkl
    if model_info.get("all_models_metadata_path") and os.path.exists(model_info["all_models_metadata_path"]):
        try:
            with open(model_info["all_models_metadata_path"], 'rb') as f:
                all_metadata = pickle.load(f)
            
            # Attempt to extract metadata specific to this model's horizon_name
            # This assumes all_metadata is a dict keyed by horizon_name
            horizon_key = model_info.get("horizon_name") 
            if horizon_key and isinstance(all_metadata, dict) and horizon_key in all_metadata:
                details["model_specific_metadata"] = all_metadata[horizon_key]
            else:
                # If not keyed by horizon, or key not found, store all metadata or a note
                details["model_specific_metadata"] = all_metadata # Or a more specific note
                # print(f"Note: Could not find metadata for horizon '{horizon_key}' in all_models_metadata.pkl. Attaching all metadata.")

        except Exception as e:
            print(f"Error loading all_models_metadata.pkl from {model_info['all_models_metadata_path']}: {e}")
            details["model_specific_metadata"] = {"error": f"Could not load all_models_metadata.pkl: {e}"}
    else:
        details["model_specific_metadata"] = "Not found or path missing"
        
    return details

if __name__ == '__main__':
    # Example Usage (for testing this service directly)
    print(f"Attempting to list models from: {os.path.abspath(MODEL_STORAGE_DIR)}")
    
    models = list_filesystem_models()
    if models:
        print("\nDiscovered model entries:") # Removed len(models) for cleaner output with potential warnings
        for model_entry in models:
            print(f"  ID: {model_entry['id']}")
            print(f"    Set Name: {model_entry['set_name']}")
            print(f"    Horizon Name (Dir): {model_entry['horizon_name']}")
            print(f"    Model Display Name: {model_entry['model_name']}")
            print(f"    Model Filename: {model_entry['model_filename']}")
            print(f"    Model Path: {model_entry['model_path']}")
            print(f"    Scaler Name: {model_entry.get('scaler_name', 'N/A')}")
            print(f"    Scaler Path: {model_entry.get('scaler_path', 'N/A')}")
            print(f"    Features Order Path: {model_entry.get('features_order_path', 'N/A')}")
            print(f"    All Models Metadata Path: {model_entry.get('all_models_metadata_path', 'N/A')}") # Added for testing
            print(f"    Status: {model_entry['status']}")

        print("\nFetching details for the first available model...")
        first_available_model = next((m for m in models if m['status'] == 'Available'), None)
        
        if first_available_model:
            details = get_filesystem_model_details(first_available_model['id'])
            if details:
                print(f"\nDetails for {details['id']}:")
                print(f"  Input Shape: {details.get('input_shape')}")
                print(f"  Output Shape: {details.get('output_shape')}")
                # print(f"  Full Info: {details}") # Potentially very verbose
            else:
                print(f"Could not fetch details for {first_available_model['id']}.")
        else:
            print("No 'Available' models found to fetch details for.")
            
    else:
        print("No models discovered. Check paths and ModelStorage contents.")