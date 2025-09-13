'''
Script to inspect Keras model files and scaler pickle files.

Usage:
python inspect_model_and_scalers.py <path_to_keras_model> <path_to_scalers_pickle_file>
Example:
python .\\\\backendv3\\\\utils\\\\inspect_model_and_scalers.py r".\\backendv3\\ModelStorage\\Set1\\saved_models\\cnn_lstm" r".\\backendv3\\ModelStorage\\Set1\\saved_models\\cnn_lstm\\scalers.pkl"
'''
import argparse
import pickle
import tensorflow as tf
import os

def inspect_keras_model(model_path):
    print(f"--- Inspecting Keras Model: {model_path} ---")
    if not os.path.exists(model_path):
        print(f"Error: Model path {model_path} does not exist.")
        return

    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        print("\nModel Summary:")
        model.summary()

        print("\nModel Input Shape(s):")
        if isinstance(model.input_shape, list):
            for i, shape in enumerate(model.input_shape):
                print(f"  Input {i}: {shape}")
        else:
            print(f"  Input: {model.input_shape}")

        print("\nModel Output Shape(s):")
        if isinstance(model.output_shape, list):
            for i, shape in enumerate(model.output_shape):
                print(f"  Output {i}: {shape}")
        else:
            print(f"  Output: {model.output_shape}")
        
        # Clean up Keras session to prevent resource leaks if script is run multiple times
        tf.keras.backend.clear_session()

    except Exception as e:
        print(f"Error loading or inspecting Keras model: {e}")

def inspect_scaler_file(scaler_path):
    print(f"\n--- Inspecting Scaler File: {scaler_path} ---")
    if not os.path.exists(scaler_path):
        print(f"Error: Scaler path {scaler_path} does not exist.")
        return

    try:
        with open(scaler_path, 'rb') as f:
            scalers = pickle.load(f)
        
        print(f"Type of loaded scaler data: {type(scalers)}")

        if isinstance(scalers, dict):
            print("\nScaler Dictionary Keys and Value Types:")
            if not scalers:
                print("  The dictionary is empty.")
            else:
                for key, value in scalers.items():
                    print(f"  Key: '{key}', Value Type: {type(value)}")
                    # If value is a scikit-learn scaler, you might want to print more details
                    if hasattr(value, 'get_params'): # Check if it looks like a scikit-learn estimator
                        # print(f"    Scaler params: {value.get_params()}")
                        if hasattr(value, 'n_features_in_'):
                            print(f"    Scaler n_features_in_: {value.n_features_in_}")
                        if hasattr(value, 'feature_names_in_'):
                            print(f"    Scaler feature_names_in_: {value.feature_names_in_}")
        elif isinstance(scalers, list):
            print(f"\nScaler List Length: {len(scalers)}")
            if scalers:
                print(f"  Type of first element: {type(scalers[0])}")
            else:
                print("  The list is empty.")
        else:
            print("  Scaler data is not a dictionary or list. Further inspection might be needed manually.")
            
    except Exception as e:
        print(f"Error loading or inspecting scaler file: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect Keras model and scaler files.")
    parser.add_argument("model_path", help="Path to the Keras model file (e.g., .h5 or SavedModel directory)")
    parser.add_argument("scaler_path", help="Path to the scalers.pkl file")

    args = parser.parse_args()

    print(f"TensorFlow version: {tf.__version__}")
    print(f"Keras version: {tf.keras.__version__}")

    inspect_keras_model(args.model_path)
    inspect_scaler_file(args.scaler_path)

    print("\n--- Inspection Complete ---")
