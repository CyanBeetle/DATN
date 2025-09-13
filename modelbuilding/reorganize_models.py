import os
import pickle
import shutil

# Configuration
OLD_MODEL_DIR_BASE = r'C:\Users\admin\Desktop\CapstoneApp\HoangPhi\backendv3\ModelStorage\Set3\saved_models' # Updated to specific path
OLD_TRAINED_MODELS_METADATA_FILE = os.path.join(OLD_MODEL_DIR_BASE, 'trained_models.pkl')
OLD_SCALERS_FILE = os.path.join(OLD_MODEL_DIR_BASE, 'scalers.pkl')

# The base directory will be the same, but we are creating subdirectories within it.
NEW_MODEL_DIR_BASE = OLD_MODEL_DIR_BASE 
NEW_ALL_MODELS_METADATA_FILE = os.path.join(NEW_MODEL_DIR_BASE, 'all_models_metadata.pkl')

def save_objects(obj, filename):
    """Save objects using pickle"""
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)

def load_objects(filename):
    """Load objects using pickle"""
    if not os.path.exists(filename):
        print(f"Error: File not found - {filename}")
        return None
    try:
        with open(filename, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        return None

def main():
    print("Starting model reorganization script.")
    print(f"Target directory for reorganization: {OLD_MODEL_DIR_BASE}")
    print(f"IMPORTANT: Ensure you have backed up this directory before proceeding.")
    # input("Press Enter to continue if you have a backup, or Ctrl+C to cancel...")

    old_trained_models_metadata = load_objects(OLD_TRAINED_MODELS_METADATA_FILE)
    old_all_scalers = load_objects(OLD_SCALERS_FILE) # This is the single scalers.pkl

    if not old_trained_models_metadata:
        print(f"Could not load the metadata file: {OLD_TRAINED_MODELS_METADATA_FILE}. This file is essential. Exiting.")
        return
    if not old_all_scalers:
        print(f"Could not load the main scalers file: {OLD_SCALERS_FILE}. This file is essential. Exiting.")
        return

    new_all_models_metadata_output = {}
    horizons_processed = 0

    # Expected model filenames based on common convention. This might need adjustment if names differ.
    # The script will iterate through horizons found in trained_models.pkl and construct the expected old model name.
    expected_model_names = {
        "short_term": "short_term_model.keras",
        "medium_term": "medium_term_model.keras",
        "long_term": "long_term_model.keras",
        "cnn_lstm": "cnn_lstm_model.keras"  # Assuming 'cnn_lstm' is the horizon key in trained_models.pkl
    }
    # If you have different horizon names in your trained_models.pkl, adjust the keys above.

    print(f"Found horizons in old metadata file ({OLD_TRAINED_MODELS_METADATA_FILE}): {list(old_trained_models_metadata.keys())}")

    for horizon_name, model_meta in old_trained_models_metadata.items():
        print(f"\nProcessing horizon: {horizon_name}...")

        # Determine the old Keras model filename for this horizon
        # Default to constructing the name if not found in expected_model_names, or use a specific mapping
        old_keras_model_filename = expected_model_names.get(horizon_name, f"{horizon_name}_model.keras")
        old_keras_model_path = os.path.join(OLD_MODEL_DIR_BASE, old_keras_model_filename)

        if not os.path.exists(old_keras_model_path):
            print(f"  Warning: Old Keras model file not found at {old_keras_model_path} (expected name: {old_keras_model_filename}). Skipping this horizon.")
            continue

        # Create new artifact directory for this horizon (subdirectory within OLD_MODEL_DIR_BASE)
        new_horizon_artifact_dir = os.path.join(NEW_MODEL_DIR_BASE, horizon_name)
        os.makedirs(new_horizon_artifact_dir, exist_ok=True)
        print(f"  Created/Ensured directory: {new_horizon_artifact_dir}")

        # 1. Copy and rename Keras model
        new_keras_model_path = os.path.join(new_horizon_artifact_dir, 'model.keras')
        try:
            shutil.copy2(old_keras_model_path, new_keras_model_path)
            print(f"  Copied Keras model from {old_keras_model_path} to: {new_keras_model_path}")
        except Exception as e:
            print(f"  Error copying Keras model for {horizon_name}: {e}. Skipping.")
            continue
            
        # 2. Extract and save scalers for this horizon from the single old_all_scalers file
        # It's assumed old_all_scalers is a dict like: {'short_term': {scalers_for_short_term}, 'medium_term': ...}
        horizon_specific_scalers = old_all_scalers.get(horizon_name)
        if not horizon_specific_scalers:
            print(f"  Warning: Scalers not found for horizon '{horizon_name}' within the main {OLD_SCALERS_FILE}. Cannot save scalers.pkl for this horizon.")
            new_scalers_path = None
        else:
            new_scalers_path = os.path.join(new_horizon_artifact_dir, 'scalers.pkl')
            try:
                save_objects(horizon_specific_scalers, new_scalers_path)
                print(f"  Saved specific scalers for {horizon_name} to: {new_scalers_path}")
            except Exception as e:
                print(f"  Error saving scalers for {horizon_name}: {e}. Scalers path will be None.")
                new_scalers_path = None

        # 3. Extract and save feature order for this horizon from the old metadata
        feature_order_list = model_meta.get('features_order')
        if not feature_order_list:
            print(f"  Warning: 'features_order' not found for horizon '{horizon_name}' in {OLD_TRAINED_MODELS_METADATA_FILE}. Cannot save features_order.pkl.")
            new_features_order_path = None
        else:
            new_features_order_path = os.path.join(new_horizon_artifact_dir, 'features_order.pkl')
            try:
                save_objects(feature_order_list, new_features_order_path)
                print(f"  Saved feature order for {horizon_name} to: {new_features_order_path}")
            except Exception as e:
                print(f"  Error saving feature order for {horizon_name}: {e}. Feature order path will be None.")
                new_features_order_path = None

        # 4. Prepare metadata for the new all_models_metadata.pkl
        # Carry over relevant information. The 'model' object itself is not stored.
        new_single_model_meta = {
            key: val for key, val in model_meta.items() if key != 'model' and key != 'features_order'
        }
        new_single_model_meta['keras_model_path'] = new_keras_model_path
        new_single_model_meta['scalers_path'] = new_scalers_path
        new_single_model_meta['features_order_path'] = new_features_order_path
        
        new_all_models_metadata_output[horizon_name] = new_single_model_meta
        horizons_processed += 1

    if horizons_processed > 0:
        try:
            save_objects(new_all_models_metadata_output, NEW_ALL_MODELS_METADATA_FILE)
            print(f"\nSuccessfully created new global metadata file: {NEW_ALL_MODELS_METADATA_FILE}")
        except Exception as e:
            print(f"\nError saving new global metadata file {NEW_ALL_MODELS_METADATA_FILE}: {e}")
    else:
        print("\nNo horizons were processed. New global metadata file not created.")

    print("\nReorganization script finished.")
    print(f"Please verify the contents of subdirectories within {NEW_MODEL_DIR_BASE} and the new {NEW_ALL_MODELS_METADATA_FILE} file.")

if __name__ == '__main__':
    main() 