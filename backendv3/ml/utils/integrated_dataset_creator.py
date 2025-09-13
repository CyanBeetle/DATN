"""
Integrated Dataset Creator utilities for UC11.
Replicates and adapts logic from modelbuildingv3/datasetcreate.py.
"""
import json
import os
from pathlib import Path
import pandas as pd
from datetime import datetime, timedelta
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

class IntegratedDatasetCreator:
    def __init__(self, input_json_paths: List[str], output_csv_path: str, interval_seconds: int):
        """
        Args:
            input_json_paths: List of full paths to _analysis.json files from video processing.
            output_csv_path: Full path where the final interval-based CSV dataset will be saved.
            interval_seconds: The interval in seconds for aggregating data.
        """
        self.input_json_paths: List[Path] = [Path(p) for p in input_json_paths]
        self.output_csv_path: Path = Path(output_csv_path)
        self.interval_seconds: int = interval_seconds
        
        # Ensure output directory for the CSV exists
        self.output_csv_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.already_processed_json_files: List[Path] = [] # To track processed files within a single run

    def _load_single_json_analysis(self, json_file_path: Path) -> Optional[Dict[str, Any]]:
        try:
            with open(json_file_path, 'r') as f:
                data = json.load(f)
            # More specific check for required fields for traffic counting
            if "detections" in data and isinstance(data["detections"], list):
                # Store video_path with analysis_data if needed later, but not strictly for current detections list
                # analysis_data["source_video_path"] = data.get("video_path", str(json_file_path.name))
                return data
            else:
                logger.warning(f"JSON file {json_file_path} is missing 'detections' list or it's invalid. Skipping.")
                return None
        except json.JSONDecodeError:
            logger.error(f"Error decoding JSON from {json_file_path}. Skipping.")
            return None
        except Exception as e:
            logger.error(f"Error loading JSON file {json_file_path}: {e}. Skipping.")
            return None

    def _collect_all_detections(self) -> List[Dict[str, Any]]:
        """
        Loads specified JSON files and extracts detection data.
        Skips files already processed in the current instance.
        """
        all_detections = []
        
        for json_file_path in self.input_json_paths:
            if not json_file_path.exists():
                logger.warning(f"Input JSON file does not exist: {json_file_path}. Skipping.")
                continue

            if json_file_path in self.already_processed_json_files:
                logger.info(f"Skipping already processed JSON file in this run: {json_file_path.name}")
                continue
            
            logger.info(f"Processing JSON analysis file: {json_file_path.name}")
            analysis_data = self._load_single_json_analysis(json_file_path)
            
            if analysis_data and analysis_data.get("detections"):
                # Extract relevant info for each detection
                # Crucially, we need timestamp_ms. Original video's FPS and start time might be needed
                # if timestamp_ms is relative to the start of THAT video and we need to align across videos.
                # For now, assuming timestamp_ms can be globally sorted or is already absolute.
                # This is a critical assumption.
                video_start_time_str = analysis_data.get("processed_datetime_utc") # Preferable
                video_path_from_json = analysis_data.get("video_path", str(json_file_path.stem))

                # If video_start_time_str is available, parse it to create absolute timestamps for detections
                video_epoch_start_dt = None
                if video_start_time_str:
                    try:
                        video_epoch_start_dt = datetime.fromisoformat(video_start_time_str.replace('Z', '+00:00'))
                    except ValueError:
                        logger.warning(f"Could not parse 'processed_datetime_utc' ({video_start_time_str}) from {json_file_path.name}. Timestamps may not be absolute.")


                for det in analysis_data["detections"]:
                    detection_data = {"source_json": str(json_file_path.name)}
                    if 'timestamp_ms' in det:
                        if video_epoch_start_dt: # Create absolute datetime for the detection
                            detection_data['datetime'] = video_epoch_start_dt + timedelta(milliseconds=det["timestamp_ms"])
                        else: # Fallback to using raw ms if no start time (less ideal for multi-video alignment)
                            detection_data['timestamp_ms_relative'] = det["timestamp_ms"]
                        
                        # Add other fields if useful for later, e.g., class_id
                        detection_data['class_id'] = det.get('class_id', -1) # -1 for unknown
                        all_detections.append(detection_data)
                    else:
                        logger.warning(f"Detection in {json_file_path.name} missing 'timestamp_ms'.")

                self.already_processed_json_files.append(json_file_path)
            else:
                logger.warning(f"No valid detections found or failed to load: {json_file_path.name}")
        
        logger.info(f"Collected {len(all_detections)} detections from {len(self.already_processed_json_files)} JSON files specified.")
        return all_detections

    def _aggregate_and_save_csv(self, all_detections: List[Dict[str, Any]]) -> bool:
        """
        Aggregates detections into time intervals and saves as a CSV file to self.output_csv_path.
        Returns True on success, False on failure.
        """
        if not all_detections:
            logger.warning("No detections provided to create interval dataset.")
            return False

        df_detections = pd.DataFrame(all_detections)
        
        # Prioritize 'datetime' if available (absolute time), else 'timestamp_ms_relative' (less robust)
        if 'datetime' in df_detections.columns:
            if df_detections['datetime'].isnull().all():
                 logger.error("All 'datetime' values are null. Cannot proceed with absolute time aggregation.")
                 if 'timestamp_ms_relative' not in df_detections.columns:
                     logger.error("Fallback 'timestamp_ms_relative' also missing.")
                     return False
                 else: # Fallback if datetime failed for all
                    logger.warning("Falling back to 'timestamp_ms_relative'. Timestamps may not be aligned across different source videos if they had different start times.")
                    df_detections['datetime_from_relative_ms'] = pd.to_datetime(df_detections['timestamp_ms_relative'], unit='ms', origin='unix') # Arbitrary origin if no global ref
                    time_col_for_resample = 'datetime_from_relative_ms'
            else: # Use absolute datetime
                df_detections.dropna(subset=['datetime'], inplace=True) # Drop rows where absolute datetime couldn't be parsed
                time_col_for_resample = 'datetime'

        elif 'timestamp_ms_relative' in df_detections.columns:
            logger.warning("Using 'timestamp_ms_relative' for aggregation. Assumes a common reference point or single video source for correct time alignment.")
            # Create a datetime column from ms, assuming a common arbitrary start if not globally aligned
            df_detections['datetime_from_relative_ms'] = pd.to_datetime(df_detections['timestamp_ms_relative'], unit='ms', origin='unix') # Example origin
            time_col_for_resample = 'datetime_from_relative_ms'
        else:
            logger.error("No suitable timestamp column ('datetime' or 'timestamp_ms_relative') found in detections for aggregation.")
            return False

        if df_detections.empty or time_col_for_resample not in df_detections.columns:
            logger.warning("Detection data is empty or missing the chosen time column for resampling. Cannot create interval dataset.")
            return False
            
        df_detections = df_detections.set_index(time_col_for_resample)
        
        df_interval_counts = df_detections.resample(f'{self.interval_seconds}S').size().reset_index(name='vehicle_count')
        df_interval_counts.rename(columns={time_col_for_resample: 'interval_start_time'}, inplace=True)

        if df_interval_counts.empty:
            logger.warning(f"No data after resampling into {self.interval_seconds}s intervals.")
            return False
        
        try:
            df_interval_counts.to_csv(self.output_csv_path, index=False)
            logger.info(f"Interval dataset saved to: {self.output_csv_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save interval dataset CSV to {self.output_csv_path}: {e}")
            return False

    def create_dataset(self) -> bool:
        """
        Orchestrates the collection of detections and creation of the interval-based CSV.
        Returns:
            bool: True if the dataset CSV was successfully created, False otherwise.
        """
        logger.info(f"Starting dataset creation. Output target: {self.output_csv_path}, Interval: {self.interval_seconds}s.")
        all_detections = self._collect_all_detections()
        if not all_detections:
            logger.error("No detections were collected. Cannot create dataset.")
            return False
        
        success = self._aggregate_and_save_csv(all_detections)
        if success:
            logger.info("Dataset CSV created successfully.")
        else:
            logger.error("Failed to create dataset CSV.")
        return success

# Example usage for testing this module
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    
    mock_json_dir = Path("__test_mock_processed_json_for_creator__")
    mock_json_dir.mkdir(exist_ok=True)
    mock_csv_output_file = Path("__test_mock_interval_csvs_output__") / "test_traffic_counts_60sec.csv"

    # Create mock JSON analysis files
    utc_now_iso = datetime.utcnow().isoformat()
    sample_detection_data1 = {
        "video_path": "video1.mp4", "fps": 30, "roi_points_normalized": [[0,0],[1,0],[1,1],[0,1]],
        "processed_datetime_utc": utc_now_iso,
        "detections": [
            {"timestamp_ms": 1000, "class_id": 0}, {"timestamp_ms": 1500, "class_id": 1},
            {"timestamp_ms": 60000, "class_id": 0} # 1 minute
        ]
    }
    # Simulate another video processed 10 minutes later
    utc_later_iso = (datetime.utcnow() + timedelta(minutes=10)).isoformat()
    sample_detection_data2 = {
        "video_path": "video2.mp4", "fps": 25, "roi_points_normalized": [[0,0],[1,0],[1,1],[0,1]],
        "processed_datetime_utc": utc_later_iso,
        "detections": [
            {"timestamp_ms": 3000, "class_id": 0}, {"timestamp_ms": 5000, "class_id": 2}, 
            {"timestamp_ms": 70000, "class_id": 0} # relative 70s into this video
        ]
    }
    mock_json_file1_path = mock_json_dir / "video1_analysis.json"
    mock_json_file2_path = mock_json_dir / "video2_analysis.json"

    with open(mock_json_file1_path, "w") as f:
        json.dump(sample_detection_data1, f)
    with open(mock_json_file2_path, "w") as f:
        json.dump(sample_detection_data2, f)

    input_files = [str(mock_json_file1_path), str(mock_json_file2_path)]

    creator = IntegratedDatasetCreator(
        input_json_paths=input_files,
        output_csv_path=str(mock_csv_output_file),
        interval_seconds=60 # 1-minute intervals
    )
    
    if creator.create_dataset():
        logger.info(f"Test CSV created successfully: {mock_csv_output_file}")
        # Verify content (optional)
        if mock_csv_output_file.exists():
            df_check = pd.read_csv(mock_csv_output_file)
            logger.info(f"Generated CSV content head:\\n{df_check.head()}")
            logger.info(f"Generated CSV info:\\n{df_check.info()}")

    else:
        logger.error("Test CSV creation failed.")

    # Clean up mock directories and files
    # import shutil
    # shutil.rmtree(mock_json_dir, ignore_errors=True)
    # shutil.rmtree(mock_csv_output_file.parent, ignore_errors=True) 