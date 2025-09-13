# Traffic Speed Prediction System

A sophisticated time series prediction system for forecasting traffic speeds based on historical data and current conditions. This system utilizes deep learning models to predict future traffic speeds at multiple time horizons.

## System Overview

The Traffic Speed Prediction System consists of the following components:

1. **Data Processing Pipeline**: Processes historical traffic data and prepares it for model training.
2. **Model Training Pipeline**: Trains multiple deep learning models for different prediction horizons.
3. **Prediction API**: Provides a programmatic interface for making speed predictions.
4. **Web Interface**: Offers a user-friendly interface for visualizing predictions.

## Prediction Horizons

The system defines three primary prediction horizons:

1. **Short-term Prediction (15 minutes)**
   - Uses 60 minutes of historical data at 1-minute intervals
   - Predicts next 15 minutes with 1-minute granularity (15 output values)
   - Primarily relies on recent traffic patterns and immediate conditions
   - Ideal for immediate route adjustments, fine-tuning short-term ETAs, alerting to sudden congestion
   - Uses LSTM architecture for high accuracy in short-term forecasting

2. **Medium-term Prediction (1 hour)**
   - Uses 180 minutes of historical data at 1-minute intervals
   - Predicts next 1 hour with 5-minute granularity (12 output values)
   - Balances recent trends with established intra-day patterns
   - Useful for tactical route planning for city-level trips, estimating ETAs for trips up to an hour
   - Uses Bidirectional LSTM architecture to capture complex patterns

3. **Long-term Prediction (6 hours)**
   - Uses 360 minutes of historical data at 1-minute intervals
   - Predicts next 6 hours with 30-minute granularity (12 output values)
   - Relies more on cyclical patterns, weather, and historical trends
   - Valuable for strategic departure time planning for longer journeys
   - Uses GRU architecture for long sequence modeling

4. **CNN-LSTM Hybrid Model (2 hours)**
   - Uses 120 minutes of historical data
   - Predicts next 2 hours with 5-minute granularity (24 output values)
   - CNN component extracts spatial features that LSTM can use for prediction
   - Well-suited for dense traffic patterns with complex interactions

## Installation

### Prerequisites

- Python 3.8 or higher
- Required packages listed in `requirements.txt`

### Setup

1. Clone this repository:
   ```
   git clone <repository-url>
   cd traffic-speed-prediction
   ```

2. Install required packages:
   ```
   pip install -r requirements.txt
   ```

3. Prepare your dataset:
   - Place your traffic dataset in the `Input/` directory
   - The dataset should be named `traffic_dataset.csv`
   - Format: `timestamp_seconds,timestamp,speed_kmh,vehicle_count,congestion_category,congestion_code,hour,minute,day,week,month,year`

## Usage

### Training Models

To train the prediction models, run:

```
python run_pipeline.py --train-only
```

This will:
1. Preprocess the input data
2. Train models for all prediction horizons
3. Save the trained models in `processed_data/saved_models/`
4. Generate evaluation plots in `processed_data/speed_prediction/`

## Input Data (`Input/traffic_dataset.csv`)

The primary input for training the models is `Input/traffic_dataset.csv` with `\Input\synthetic_traffic_dataset.csv`. is the extension with same format and longer observation period. This file should contain historical traffic data with the following columns:

*   **`timestamp_seconds`**: (Integer) The timestamp as the total number of seconds elapsed from a reference point (e.g., start of the dataset).
*   **`timestamp`**: (String) A human-readable timestamp string, typically in `YYYY-MM-DD HH:MM:SS` format. This is used for reference and can be used to derive other time-based features if they are not already present.
*   **`speed_kmh`**: (Float/Numeric) The average speed of traffic in kilometers per hour for the given interval. This is the primary target variable for prediction.
*   **`vehicle_count`**: (Integer) The number of vehicles detected during the given interval. This is an important feature for predicting speed.
*   **`hour`**: (Integer) The hour of the day (0-23) for the record.
*   **`minute`**: (Integer) The minute of the hour (0-59) for the record.
*   **`day`**: (Integer) The day of the week (0 for Monday through 6 for Sunday). 
*   **`week`**: (Integer) The week number of the year.
*   **`month`**: (Integer) The month of the year (1-12).
*   **`year`**: (Integer) The year of the record.

**Note on `day_of_week` vs `day`:**
The system scripts (like `generate_synthetic_data.py` and `speed_prediction_system.py`) will primarily look for a column named `day_of_week` for day-of-the-week information. If this column is not present, `generate_synthetic_data.py` will attempt to use the `day` column as a fallback, assuming it represents the day of the week (e.g., 0-6 or 1-7). Ensure this column accurately reflects the day of the week for proper pattern learning. The `speed_prediction_system.py` and `prediction_utils.py` typically derive `day_of_week` from the `timestamp` index directly.

### Making Predictions via Command Line

For quick predictions without the web interface:

```
python predict_speed.py --day 1 --time 08:30 --weather_harsh
```

Parameters:
- `--day`: Day of week (0=Monday, 6=Sunday)
- `--time`: Time of day in HH:MM format
- `--weather_harsh`: Flag to indicate harsh weather conditions
- `--sample`: Use sample data instead of live data
- `--output`: Path to save predictions as JSON
- `--plot`: Path to save prediction plot

### Running the Web Application

To start the web interface:

```
python run_pipeline.py --web-only
```

This will start a Flask web server at http://localhost:5000 where you can:
- Choose prediction parameters (day, time, weather conditions)
- Select specific models or use all models
- View prediction results with interactive visualizations
- Access detailed model information

### Running the Complete Pipeline

To run both training and web interface:

```
python run_pipeline.py
```

This will train models (if not skipped and `processed_data/saved_models` is empty or `--force-train` is used),
run a prediction example, and then start the web application.

## Evaluation Metrics

The system uses the following metrics to evaluate model performance on the original scale of `speed_kmh`:

- **Mean Absolute Error (MAE):** Average absolute difference between predicted and actual speeds. Interpretable in km/h.
- **Root Mean Squared Error (RMSE):** Square root of the average of squared differences. Penalizes larger errors more. Interpretable in km/h.
- **R-squared (R²):** Proportion of the variance in the actual speeds that is predictable from the model. Ranges from -infinity to 1 (higher is better).

These metrics are calculated for the overall test set and for each individual future time step within the prediction horizon.

## API Reference

The system provides a REST API for programmatic access:

### Prediction Endpoint

**POST** `/api/predict`

Request Body:
```json
{
  "day_of_week": 1,
  "time_of_day": "08:30",
  "weather_harsh": true,
  "selected_model": "short_term"
}
```

Response:
```json
{
  "predictions": {
    "short_term": {
      "timestamps": ["08:31:00", "08:32:00", "..."],
      "speeds": [45.2, 44.8, "..."],
      "granularity": "1 minute",
      "weather_effect": true
    }
  },
  "input": {
    "day_of_week": 1,
    "time_of_day": "08:30",
    "weather_harsh": true
  },
  "day_name": "Tuesday"
}
```

## Model Architecture

Each prediction horizon uses a tailored neural network architecture:

1. **LSTM (Short-term)**
   - Two LSTM layers (64 and 32 units)
   - Dropout layers (0.2) for regularization
   - Dense output layer

2. **Bidirectional LSTM (Medium-term)**
   - Two bidirectional LSTM layers
   - Better captures patterns flowing in both time directions
   - Dropout for regularization

3. **GRU (Long-term)**
   - More efficient for longer sequences
   - Two GRU layers with regularization

4. **CNN-LSTM (Hybrid)**
   - 1D convolutional layer for feature extraction
   - Max pooling to reduce dimensionality
   - LSTM layers for sequence prediction

## Data Processing Features

The system employs several techniques to improve prediction quality:

- **Feature Engineering**
  - Traffic density calculation (vehicle_count / speed_kmh)
  - Trend features (speed_diff, vehicle_diff)
  - Cyclical time encoding (sine/cosine transformations)

- **Categorical Features**
  - One-hot encoding for day of week
  - One-hot encoding for congestion codes (note: `congestion_category` is assumed to be redundant and is not used)

- **Preprocessing**
  - MinMax scaling for numerical features
  - Sequence creation with varying granularity

## References

- TensorFlow and Keras for deep learning models
- Flask for web application development
- Pandas and NumPy for data processing
- Matplotlib for visualization