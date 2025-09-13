import os
import argparse
import subprocess
import sys
import time

def check_requirements():
    """
    Check if all required packages are installed
    """
    try:
        import tensorflow as tf
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        import flask
        print("All required packages are installed.")
        return True
    except ImportError as e:
        print(f"Missing required package: {e}")
        print("Please install the required packages using: pip install -r requirements.txt")
        return False

def create_directories():
    """
    Create necessary directories
    """
    directories = [
        'processed_data',
        'processed_data/saved_models',
        'processed_data/speed_prediction',
        'processed_data/predictions',
        'templates',
        'static'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    print("Created necessary directories.")

def train_models():
    """
    Train the prediction models
    """
    print("\n" + "="*80)
    print("TRAINING PREDICTION MODELS")
    print("="*80)
    
    try:
        # Import and run the speed_prediction_system
        from speed_prediction_system import main as train_main
        train_main()
        print("\nModel training completed successfully!")
        return True
    except Exception as e:
        print(f"Error during model training: {e}")
        return False

def run_prediction_example():
    """
    Run a prediction example with the trained models
    """
    print("\n" + "="*80)
    print("RUNNING PREDICTION EXAMPLE")
    print("="*80)
    
    try:
        # Run prediction script with example parameters
        cmd = [
            sys.executable, 'predict_speed.py',
            '--day', '1',
            '--time', '08:00',
            '--weather_harsh',
            '--sample',
            '--output', 'processed_data/predictions/example_prediction.json',
            '--plot', 'static/example_prediction.png'
        ]
        
        subprocess.run(cmd, check=True)
        print("\nPrediction example completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running prediction example: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error: {e}")
        return False

def run_web_app():
    """
    Start the Flask web application
    """
    print("\n" + "="*80)
    print("STARTING WEB APPLICATION")
    print("="*80)
    
    try:
        print("Starting Flask web application...")
        print("Access the application at http://localhost:5000")
        print("Press Ctrl+C to stop the server")
        
        # Run the Flask app
        from app import app
        app.run(debug=True)
        
        return True
    except Exception as e:
        print(f"Error starting web application: {e}")
        return False

def verify_data_file():
    """
    Verify that the input data file exists
    """
    data_path = 'Input/synthetic_traffic_dataset.csv'
    
    if not os.path.exists(data_path):
        print(f"ERROR: Input data file not found at {data_path}")
        return False
    
    print(f"Found input data file: {data_path}")
    return True

def parse_arguments():
    """
    Parse command-line arguments
    """
    parser = argparse.ArgumentParser(description='Run the traffic speed prediction pipeline')
    
    parser.add_argument('--train-only', action='store_true',
                        help='Train models only, do not start web application')
    parser.add_argument('--web-only', action='store_true',
                        help='Start web application only, do not train models')
    parser.add_argument('--skip-examples', action='store_true',
                        help='Skip running prediction examples')
    
    return parser.parse_args()

def main():
    """
    Main function to run the pipeline
    """
    print("Traffic Speed Prediction Pipeline")
    print("="*80)
    
    # Parse arguments
    args = parse_arguments()
    
    # Check requirements
    if not check_requirements():
        return
    
    # Create necessary directories
    create_directories()
    
    # Verify data file exists
    if not verify_data_file():
        return
    
    # Run the pipeline based on arguments
    if not args.web_only:
        # Train models
        if not train_models():
            print("Model training failed. Exiting.")
            return
        
        # Run prediction example
        if not args.skip_examples:
            if not run_prediction_example():
                print("Warning: Prediction example failed, but continuing.")
    
    # Start web application
    if not args.train_only:
        run_web_app()

if __name__ == "__main__":
    main() 