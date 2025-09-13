import pickle
import sys
import pandas as pd # Added for DataFrame inspection

def inspect_pickle_file(file_path): # Renamed for clarity
    """
    Loads a pickle file and inspects its contents.
    """
    print(f"Attempting to load pickle file: {file_path}")
    try:
        with open(file_path, 'rb') as f:
            loaded_object = pickle.load(f)
    except Exception as e:
        print(f"Error loading pickle file: {e}")
        return

    print(f"Successfully loaded. Type of loaded object: {type(loaded_object)}")

    if isinstance(loaded_object, dict):
        print("Object is a dictionary. Keys found:")
        keys = list(loaded_object.keys())
        print(keys)

        for key, value in loaded_object.items():
            print(f"  Key: '{key}'")
            print(f"    Type of value: {type(value)}")
            if hasattr(value, '__dict__') and not isinstance(value, (list, tuple, pd.DataFrame)): # Avoid trying to __dict__ common data structures
                try:
                    # Attempt to access attributes common to scikit-learn scalers
                    if hasattr(value, 'mean_'):
                        print(f"    Scaler has attribute 'mean_'. Length: {len(value.mean_) if hasattr(value.mean_, '__len__') else 'N/A'}")
                    if hasattr(value, 'scale_'):
                        print(f"    Scaler has attribute 'scale_'. Length: {len(value.scale_) if hasattr(value.scale_, '__len__') else 'N/A'}")
                    if hasattr(value, 'n_features_in_'):
                        print(f"    Scaler has attribute 'n_features_in_': {value.n_features_in_}")
                    if hasattr(value, 'feature_names_in_'):
                        print(f"    Scaler has attribute 'feature_names_in_'. Count: {len(value.feature_names_in_) if hasattr(value.feature_names_in_, '__len__') else 'N/A'}")
                    # Add more attributes if needed
                except Exception as e_attr:
                    print(f"    Could not inspect attributes for key '{key}': {e_attr}")
            elif isinstance(value, (list, tuple, set)):
                print(f"    Value is a sequence of length: {len(value)}")
                if len(value) < 20: # Print content if not too long
                    print(f"    Content: {value}")
                else:
                    print(f"    First 10 elements: {list(value)[:10]}")

            elif isinstance(value, pd.DataFrame):
                print(f"    Value is a pandas DataFrame. Shape: {value.shape}")
                print(f"    Columns: {value.columns.tolist()}")
            elif isinstance(value, (int, float, str, bool)):
                print(f"    Value: {value}")
            # Add other specific type checks if needed
    elif isinstance(loaded_object, (list, tuple, set)):
        print(f"Object is a {type(loaded_object).__name__} of length: {len(loaded_object)}")
        if len(loaded_object) < 50: # Print content if not too long
            print(f"Content: {loaded_object}")
        else:
            print(f"First 20 elements: {list(loaded_object)[:20]}")
            # Potentially print last few elements too if very long
            print(f"Last 20 elements: {list(loaded_object)[-20:]}")
    elif isinstance(loaded_object, pd.DataFrame):
        print(f"Object is a pandas DataFrame. Shape: {loaded_object.shape}")
        print(f"Columns: {loaded_object.columns.tolist()}")
        print("First 5 rows:")
        print(loaded_object.head())
    else:
        print("Loaded object is not a dictionary, list, or DataFrame. Further inspection might be needed based on its type.")
        # You could add more specific type handling here if you expect other types

if __name__ == "__main__":
    if len(sys.argv) > 1:
        pickle_file_path = sys.argv[1]
    else:
        # Default path, PLEASE CHANGE THIS to the specific file you want to inspect
        pickle_file_path = "C:\\\\Users\\\\admin\\\\Desktop\\\\CapstoneApp\\\\HoangPhi\\\\backendv3\\\\ModelStorage\\\\Set1\\\\saved_models\\\\cnn_lstm\\\\features_order.pkl"
        print(f"No path provided, using default: {pickle_file_path}")

    if not pickle_file_path:
        print("Error: Please provide the path to the .pkl file as a command-line argument or edit the script.")
    else:
        inspect_pickle_file(pickle_file_path) # Ensure this calls the renamed function
