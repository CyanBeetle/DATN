"""Batch video processing for traffic monitoring."""
import os
import sys
import cv2
import time
import json
import numpy as np
from pathlib import Path
import logging
import tkinter as tk
from tkinter import messagebox, ttk
from datetime import datetime

# Add parent directory to path to import application modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import application modules (excluding config)
from video.processor import VideoProcessor
from utils.roi import ROIHandler

# Hardcoded paths
INPUT_DIR = r"F:\Capstone\InputVideo"
OUTPUT_DIR = r"C:\Users\admin\Desktop\CapstoneApp\HoangPhi\backendv3\results"
LOGS_DIR = os.path.join(OUTPUT_DIR, "logs")

# Ensure logs directory exists
os.makedirs(LOGS_DIR, exist_ok=True)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(LOGS_DIR, 'batch_processing.log'))
    ]
)
logger = logging.getLogger(__name__)


def show_videos_properties(video_files):
    """
    Display a popup showing properties of all videos and let user select which videos to process
    
    Args:
        video_files (list): List of video file paths
        
    Returns:
        tuple: (video_properties, selected_videos, action)
               - video_properties: List of video properties dictionaries
               - selected_videos: List of selected video indices or None for all
               - action: "selected", "all", or "cancel"
    """
    # Extract properties for all videos
    video_properties = []
    for video_path in video_files:
        props = get_video_properties(video_path)
        if props:
            video_properties.append(props)
    
    # Create tkinter window
    root = tk.Tk()
    root.title("Video Selection")
    root.geometry("700x550")
    
    # Add title and instructions
    tk.Label(
        root, 
        text="Traffic Monitoring System", 
        font=("Arial", 16, "bold")
    ).pack(pady=10)
    
    tk.Label(
        root, 
        text="Select videos to process or choose 'Process All'", 
        font=("Arial", 10)
    ).pack(pady=5)
    
    # Create frame for treeview
    frame = tk.Frame(root)
    frame.pack(pady=10, fill="both", expand=True)
    
    # Create scrollbar
    scrollbar = ttk.Scrollbar(frame)
    scrollbar.pack(side="right", fill="y")
    
    # Create treeview with scrollbar and selection
    columns = ("Filename", "Resolution", "FPS", "Duration", "Total Frames")
    tree = ttk.Treeview(frame, columns=columns, show="headings", yscrollcommand=scrollbar.set, selectmode="extended")
    
    # Configure scrollbar
    scrollbar.config(command=tree.yview)
    
    # Set column headings
    for col in columns:
        tree.heading(col, text=col)
        tree.column(col, width=100)
    
    # Set column widths
    tree.column("Filename", width=200)
    
    # Add data to treeview
    for i, props in enumerate(video_properties):
        tree.insert("", "end", iid=str(i), values=(
            props["filename"],
            f"{props['width']}x{props['height']}",
            f"{props['fps']:.2f}",
            f"{props['duration']} sec",
            props["total_frames"]
        ))
    
    # Pack treeview
    tree.pack(side="left", fill="both", expand=True)
    
    # Variables to store result
    result = {"selected": [], "action": "cancel"}
    
    # Add buttons
    button_frame = tk.Frame(root)
    button_frame.pack(pady=10, fill="x")
    
    # Process Selected button
    def process_selected():
        selected_items = tree.selection()
        if not selected_items:
            messagebox.showwarning("No Selection", "Please select at least one video to process")
            return
        result["selected"] = [int(item) for item in selected_items]
        result["action"] = "selected"
        root.destroy()
    
    process_selected_btn = tk.Button(
        button_frame, 
        text="Process Selected", 
        command=process_selected, 
        width=15, 
        height=1, 
        bg="#4CAF50", 
        fg="white",
        font=("Arial", 10, "bold")
    )
    process_selected_btn.pack(side="left", padx=10, expand=True)
    
    # Process All button
    def process_all():
        result["selected"] = None  # None means all
        result["action"] = "all"
        root.destroy()
    
    process_all_btn = tk.Button(
        button_frame, 
        text="Process All", 
        command=process_all, 
        width=10, 
        height=1,
        bg="#2196F3", 
        fg="white",
        font=("Arial", 10)
    )
    process_all_btn.pack(side="left", padx=10, expand=True)
    
    # Cancel button
    def cancel():
        result["action"] = "cancel"
        root.destroy()
    
    cancel_btn = tk.Button(
        button_frame, 
        text="Cancel", 
        command=cancel, 
        width=10, 
        height=1,
        bg="#f44336", 
        fg="white",
        font=("Arial", 10)
    )
    cancel_btn.pack(side="left", padx=10, expand=True)
    
    # Center the window
    root.update_idletasks()
    width = root.winfo_width()
    height = root.winfo_height()
    x = (root.winfo_screenwidth() // 2) - (width // 2)
    y = (root.winfo_screenheight() // 2) - (height // 2)
    root.geometry('{}x{}+{}+{}'.format(width, height, x, y))
    
    # Wait for window to be closed
    root.mainloop()
    
    return video_properties, result["selected"], result["action"]


def get_video_properties(video_path):
    """Extract and return video properties"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Could not open video: {video_path}")
        return None
        
    properties = {
        "filename": os.path.basename(video_path),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "duration": int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)),
        "path": video_path
    }
    
    cap.release()
    return properties


def select_roi_for_video(video_path):
    """
    Allow user to select ROI on the first frame of the video
    
    Args:
        video_path (str): Path to the video file
        
    Returns:
        list: ROI points as a list of 4 points [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
    """
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Could not open video: {video_path}")
        return None
    
    # Read first frame
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        logger.error("Could not read frame from video")
        return None
    
    # Resize frame if too large for display
    display_width = 1280
    scale = min(1.0, display_width / frame.shape[1])
    if scale < 1.0:
        frame = cv2.resize(frame, None, fx=scale, fy=scale)
    
    # Instructions
    print("\nSelect 4 points to define the Region of Interest (ROI)")
    print("Click on 4 corners of the region in this order:")
    print("  1. Top-left (Point A)")
    print("  2. Top-right (Point B)")
    print("  3. Bottom-right (Point C)")
    print("  4. Bottom-left (Point D)")
    
    # Use OpenCV ROI selection
    points = []
    point_labels = ['A', 'B', 'C', 'D']
    
    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # Adjust coordinates if frame was resized
            orig_x, orig_y = int(x / scale), int(y / scale)
            if len(points) < 4:
                points.append([orig_x, orig_y])
            
            # Draw points and lines
            frame_copy = frame.copy()
            
            # Draw existing points and labels
            for i, pt in enumerate(points):
                # Convert to display coordinates
                disp_x, disp_y = int(pt[0] * scale), int(pt[1] * scale)
                
                # Draw point
                cv2.circle(frame_copy, (disp_x, disp_y), 5, (0, 255, 0), -1)
                
                # Draw label next to point
                cv2.putText(
                    frame_copy,
                    point_labels[i],
                    (disp_x + 10, disp_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )
                
                # Draw lines connecting points
                if i > 0:
                    prev_x, prev_y = int(points[i-1][0] * scale), int(points[i-1][1] * scale)
                    cv2.line(frame_copy, (prev_x, prev_y), (disp_x, disp_y), (0, 255, 0), 2)
            
            # Connect last point to first if we have 4 points
            if len(points) == 4:
                first_x, first_y = int(points[0][0] * scale), int(points[0][1] * scale)
                last_x, last_y = int(points[3][0] * scale), int(points[3][1] * scale)
                cv2.line(frame_copy, (last_x, last_y), (first_x, first_y), (0, 255, 0), 2)
                
                # Add confirmation text
                cv2.putText(
                    frame_copy,
                    "Selection complete! Press Enter to confirm.",
                    (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )
            
            cv2.imshow("Select ROI (Press Enter to confirm, 'r' to reset, ESC to cancel)", frame_copy)
    
    # Create window and set callback
    cv2.namedWindow("Select ROI (Press Enter to confirm, 'r' to reset, ESC to cancel)")
    cv2.setMouseCallback("Select ROI (Press Enter to confirm, 'r' to reset, ESC to cancel)", mouse_callback)
    cv2.imshow("Select ROI (Press Enter to confirm, 'r' to reset, ESC to cancel)", frame)
    
    # Wait for user input
    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == 13:  # Enter key
            if len(points) == 4:
                break
            else:
                print(f"Please select 4 points. Currently selected: {len(points)}")
        elif key == ord('r'):
            points = []
            frame_copy = frame.copy()
            cv2.imshow("Select ROI (Press Enter to confirm, 'r' to reset, ESC to cancel)", frame_copy)
            print("ROI selection reset. Please select 4 points.")
        elif key == 27:  # ESC key
            points = []
            break
    
    cv2.destroyAllWindows()
    
    if len(points) != 4:
        logger.warning("ROI selection was canceled or incomplete")
        return None
    
    # Validate ROI before returning
    roi_handler = ROIHandler()
    valid, message = roi_handler.validate_roi(points)
    
    if not valid:
        logger.error(f"Invalid ROI: {message}")
        messagebox.showerror("Invalid ROI", f"{message}\nPlease try again.")
        return None
    
    # Log and return points as list (not numpy array)
    points_list = [point for point in points]
    logger.info(f"Selected ROI points: {points_list}")
    print(f"Selected ROI points: {points_list}")
    return points_list


def get_roi_dimensions():
    """
    Show a UI for user to input ROI dimensions
    
    Returns:
        tuple: (width, height) in meters
    """
    # Default values
    default_width = 24.0  # Width in meters (A to B)
    default_height = 30.0  # Height in meters (B to C)
    
    # Create tkinter root
    root = tk.Tk()
    root.title("ROI Dimensions")
    root.geometry("450x300")
    
    # Variables for input
    width_var = tk.StringVar(value=str(default_width))
    height_var = tk.StringVar(value=str(default_height))
    result = {"width": default_width, "height": default_height}
    
    # Title
    tk.Label(
        root, 
        text="Enter ROI Dimensions", 
        font=("Arial", 16, "bold")
    ).pack(pady=10)
    
    # Description
    tk.Label(
        root, 
        text="Enter the real-world dimensions of the selected region.\n"
             "These values are used for accurate speed calculation.",
        justify="left"
    ).pack(pady=10)
    
    # Width input frame
    width_frame = tk.Frame(root)
    width_frame.pack(fill="x", padx=20, pady=5)
    
    tk.Label(
        width_frame, 
        text="Width (A→B, top-left to top-right):", 
        width=30, 
        anchor="w"
    ).pack(side="left")
    
    width_entry = tk.Entry(width_frame, textvariable=width_var, width=10)
    width_entry.pack(side="left", padx=5)
    
    tk.Label(width_frame, text="meters").pack(side="left")
    
    # Height input frame
    height_frame = tk.Frame(root)
    height_frame.pack(fill="x", padx=20, pady=5)
    
    tk.Label(
        height_frame, 
        text="Height (B→C, top-right to bottom-right):", 
        width=30, 
        anchor="w"
    ).pack(side="left")
    
    height_entry = tk.Entry(height_frame, textvariable=height_var, width=10)
    height_entry.pack(side="left", padx=5)
    
    tk.Label(height_frame, text="meters").pack(side="left")
    
    # Buttons
    button_frame = tk.Frame(root)
    button_frame.pack(pady=20)
    
    def on_confirm():
        try:
            # Parse input values
            result["width"] = float(width_var.get())
            result["height"] = float(height_var.get())
            
            # Validate input
            if result["width"] <= 0 or result["height"] <= 0:
                messagebox.showerror("Invalid Input", "Dimensions must be positive values")
                return
            root.destroy()
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter valid numbers for width and height")
    
    def on_use_default():
        result["width"] = default_width
        result["height"] = default_height
        root.destroy()
    
    # Confirm button
    tk.Button(
        button_frame,
        text="Confirm",
        command=on_confirm,
        width=10,
        bg="#4CAF50",
        fg="white",
        font=("Arial", 10, "bold")
    ).pack(side="left", padx=10)
    
    # Use defaults button
    tk.Button(
        button_frame,
        text="Use Default Values",
        command=on_use_default,
        width=15
    ).pack(side="left", padx=10)
    
    # Set focus on width entry
    width_entry.focus_set()
    
    # Handle window close as use default
    root.protocol("WM_DELETE_WINDOW", on_use_default)
    
    # Wait for user input
    root.mainloop()
    
    # Log and return result
    logger.info(f"Using ROI dimensions: width={result['width']}m, height={result['height']}m")
    print(f"\nUsing ROI dimensions: width={result['width']}m, height={result['height']}m")
    
    return result["width"], result["height"]


def select_processing_fps(video_fps):
    """
    Show a UI for the user to select processing FPS
    
    Args:
        video_fps (float): Original FPS of the video
        
    Returns:
        float: Processing FPS to use (frames to process per second)
    """
    # Default values based on video FPS
    suggested_fps = min(10.0, video_fps / 2)  # Suggest half the video FPS but max 10 FPS
    
    # Preset options
    presets = {
        "Very Low (1 FPS)": 1.0,
        "Low (2 FPS)": 2.0, 
        "Medium (5 FPS)": 5.0,
        "High (10 FPS)": 10.0,
        "Very High (15 FPS)": 15.0,
        "Original Video FPS": video_fps
    }
    
    # Create tkinter window
    root = tk.Tk()
    root.title("Processing Frame Rate Selection")
    root.geometry("550x420")
    
    # Selected FPS
    selected_fps = tk.DoubleVar(value=suggested_fps)
    selected_preset = tk.StringVar()
    
    # Title
    tk.Label(
        root,
        text="Select Processing Frame Rate",
        font=("Arial", 16, "bold")
    ).pack(pady=10)
    
    # Description
    tk.Label(
        root,
        text="Higher FPS provides better tracking accuracy but increases processing time.\n"
             "Lower FPS is faster but may miss vehicles moving at high speeds.",
        justify="left",
        font=("Arial", 10)
    ).pack(pady=10)
    
    # Video info
    tk.Label(
        root,
        text=f"Original video: {video_fps:.2f} FPS",
        font=("Arial", 10, "italic")
    ).pack(pady=5)
    
    # Frame for preset buttons
    presets_frame = tk.Frame(root)
    presets_frame.pack(pady=15, fill="x")
    
    tk.Label(
        presets_frame,
        text="Preset Options:",
        font=("Arial", 10, "bold")
    ).pack(anchor="w", padx=20)
    
    # Function to update selected FPS from preset
    def select_preset(preset_name):
        selected_preset.set(preset_name)
        selected_fps.set(presets[preset_name])
        fps_slider.set(presets[preset_name])
        update_frame_skip_preview()
        
    # Create preset buttons in a grid
    preset_buttons_frame = tk.Frame(presets_frame)
    preset_buttons_frame.pack(pady=5, fill="x", padx=20)
    
    col = 0
    row = 0
    for preset_name, fps_value in presets.items():
        btn = tk.Button(
            preset_buttons_frame,
            text=preset_name,
            command=lambda name=preset_name: select_preset(name),
            width=20,
            height=2
        )
        btn.grid(row=row, column=col, padx=5, pady=5)
        col += 1
        if col > 1:
            col = 0
            row += 1
    
    # Custom FPS slider
    custom_frame = tk.Frame(root)
    custom_frame.pack(pady=10, fill="x", padx=20)
    
    tk.Label(
        custom_frame,
        text="Custom FPS:",
        font=("Arial", 10, "bold")
    ).pack(anchor="w")
    
    # Frame for slider and value display
    slider_frame = tk.Frame(custom_frame)
    slider_frame.pack(fill="x", pady=5)
    
    min_fps = max(0.5, video_fps / 30)  # Minimum reasonable FPS
    max_fps = min(video_fps, 30)  # Maximum reasonable FPS
    
    # Create the slider
    fps_slider = tk.Scale(
        slider_frame, 
        from_=min_fps,
        to=max_fps,
        orient=tk.HORIZONTAL,
        length=300,
        resolution=0.5,
        variable=selected_fps,
        command=lambda _: update_frame_skip_preview()
    )
    fps_slider.pack(side="left", fill="x", expand=True)
    
    # Entry for manual FPS input
    fps_entry_var = tk.StringVar(value=str(suggested_fps))
    
    def update_slider_from_entry():
        try:
            value = float(fps_entry_var.get())
            if min_fps <= value <= max_fps:
                selected_fps.set(value)
                fps_slider.set(value)
                update_frame_skip_preview()
            else:
                messagebox.showwarning("Invalid Value", 
                                      f"Please enter a value between {min_fps:.1f} and {max_fps:.1f}")
                fps_entry_var.set(str(selected_fps.get()))
        except ValueError:
            messagebox.showwarning("Invalid Input", "Please enter a valid number")
            fps_entry_var.set(str(selected_fps.get()))
    
    entry_frame = tk.Frame(slider_frame)
    entry_frame.pack(side="right", padx=10)
    
    fps_entry = tk.Entry(entry_frame, textvariable=fps_entry_var, width=6)
    fps_entry.pack(side="left")
    
    tk.Label(entry_frame, text="FPS").pack(side="left", padx=2)
    
    tk.Button(
        entry_frame,
        text="Set",
        command=update_slider_from_entry,
        width=4
    ).pack(side="left", padx=5)
    
    # Preview frame
    preview_frame = tk.Frame(root)
    preview_frame.pack(pady=15, fill="x", padx=20)
    
    tk.Label(
        preview_frame,
        text="Processing Preview:",
        font=("Arial", 10, "bold")
    ).pack(anchor="w")
    
    frame_skip_var = tk.StringVar()
    processing_fps_var = tk.StringVar()
    expected_time_var = tk.StringVar()
    
    tk.Label(preview_frame, textvariable=frame_skip_var).pack(anchor="w", pady=2)
    tk.Label(preview_frame, textvariable=processing_fps_var).pack(anchor="w", pady=2)
    tk.Label(preview_frame, textvariable=expected_time_var).pack(anchor="w", pady=2)
    
    # Function to update preview
    def update_frame_skip_preview():
        fps_value = selected_fps.get()
        frame_skip = max(1, int(video_fps / fps_value))
        actual_fps = video_fps / frame_skip
        
        # Estimate processing time - assuming 30 seconds per 100 frames at 2 FPS
        one_minute_video_frames = 60 * actual_fps
        estimated_minutes = one_minute_video_frames / 100 * (30 / 60)
        
        frame_skip_var.set(f"Every {frame_skip} frames will be processed")
        processing_fps_var.set(f"Actual processing rate: {actual_fps:.2f} FPS")
        expected_time_var.set(f"Estimated processing time: {estimated_minutes:.1f} minutes per minute of video")
        
        # Update preset selection if it matches a preset
        for name, value in presets.items():
            if abs(value - fps_value) < 0.01:
                selected_preset.set(name)
                break
        else:
            selected_preset.set("Custom")
    
    # Call once to initialize
    update_frame_skip_preview()
    
    # Buttons
    button_frame = tk.Frame(root)
    button_frame.pack(pady=15)
    
    # Result to return
    result = {"fps": suggested_fps}
    
    def on_confirm():
        result["fps"] = selected_fps.get()
        root.destroy()
    
    # Confirm button with improved styling
    confirm_btn = tk.Button(
        button_frame,
        text="Confirm",
        command=on_confirm,
        width=15,
        height=2,
        bg="#4CAF50",
        fg="white",
        font=("Arial", 12, "bold")
    )
    confirm_btn.pack(side="left", padx=10)
    
    # Handle window close - use the current selected value
    def on_window_close():
        result["fps"] = selected_fps.get()
        root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_window_close)
    
    # Center the window
    root.update_idletasks()
    width = root.winfo_width()
    height = root.winfo_height()
    x = (root.winfo_screenwidth() // 2) - (width // 2)
    y = (root.winfo_screenheight() // 2) - (height // 2)
    root.geometry('{}x{}+{}+{}'.format(width, height, x, y))
    
    # Bind Enter key to confirm button
    root.bind('<Return>', lambda event: on_confirm())
    
    # Set focus on the confirm button
    confirm_btn.focus_set()
    
    # Wait for user input
    root.mainloop()
    
    # Log the selection
    logger.info(f"Selected processing FPS: {result['fps']}")
    print(f"\nSelected processing FPS: {result['fps']}")
    
    return result["fps"]


def process_videos(input_folder, output_folder, roi_points, target_width, target_height, selected_videos, processing_fps):
    """
    Process all videos in the input folder
    
    Args:
        input_folder (str): Path to input folder containing videos
        output_folder (str): Path to output folder for results
        roi_points (list): ROI points for perspective transform
        target_width (float): Width of ROI in meters (left to right)
        target_height (float): Height of ROI in meters (top to bottom)
        selected_videos (list): List of selected video paths
        processing_fps (float): User-selected FPS for processing
        
    Returns:
        list: Processing results for each video
    """
    try:
        # Initialize progress tracking
        print("\nPreparing to process videos...")
        start_time = time.time()
        
        # Initialize processor with GPU acceleration
        processor = VideoProcessor(
            model_name="yolov8x.pt",
            confidence=0.3,
            use_gpu=True
        )
        
        # Process each video
        results = []
        total_vehicles = 0
        success_count = 0
        error_count = 0
        
        for i, video_path in enumerate(selected_videos):
            filename = os.path.basename(video_path)
            output_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_analyzed.mp4")
            
            # Check if output files already exist and remove them if they do
            if os.path.exists(output_path):
                print(f"  Previous output file found. Replacing: {os.path.basename(output_path)}")
                try:
                    os.remove(output_path)
                except Exception as e:
                    print(f"  Warning: Could not remove existing file: {e}")
            
            # Also check for existing JSON result files with the same base name
            base_name = os.path.splitext(filename)[0]
            results_dir = OUTPUT_DIR
            existing_results = [f for f in os.listdir(results_dir) 
                               if f.startswith(base_name) and f.endswith('.json')]
            
            for result_file in existing_results:
                result_path = os.path.join(results_dir, result_file)
                print(f"  Previous result file found. Replacing: {result_file}")
                try:
                    os.remove(result_path)
                except Exception as e:
                    print(f"  Warning: Could not remove existing result file: {e}")
            
            print(f"\nProcessing video {i+1}/{len(selected_videos)}: {filename}")
            print(f"  Input: {video_path}")
            print(f"  Output: {output_path}")
            
            try:
                # Determine frame skip value based on user-selected FPS
                cap = cv2.VideoCapture(video_path)
                fps = cap.get(cv2.CAP_PROP_FPS)
                cap.release()
                
                frame_skip = max(1, int(fps / processing_fps))
                print(f"  Frame skip: {frame_skip} (processing at approximately {processing_fps:.2f} FPS)")
                
                # Process the video
                print("  Processing video...")
                stats = processor.process_video(
                    input_path=video_path,
                    output_path=output_path,
                    roi_points=roi_points,
                    target_width=target_width,
                    target_height=target_height,
                    frame_skip=frame_skip
                )
                
                # Update stats
                stats["input_file"] = filename
                stats["output_file"] = os.path.basename(output_path)
                total_vehicles += stats["unique_vehicles"]
                success_count += 1
                results.append(stats)
                
                print(f"  ✓ Success! Processed {stats['processed_frames']} frames")
                print(f"  ✓ Detected {stats['unique_vehicles']} unique vehicles")
                print(f"  ✓ Processing time: {stats['processing_time']:.1f} seconds")
                
            except Exception as e:
                logger.exception(f"Error processing video {filename}: {e}")
                error_count += 1
                
                results.append({
                    "input_file": filename,
                    "error": str(e),
                    "timestamp": str(datetime.now())
                })
                
                print(f"  ❌ Error processing video: {e}")
        
        # Create summary report
        print("\nGenerating summary report...")
        
        # Save detailed results
        timestamp = int(time.time())
        report_path = os.path.join(output_folder, f"batch_analysis_report_{timestamp}.json")
        
        report_data = {
            "summary": {
                "total_videos": len(selected_videos),
                "successful": success_count,
                "failed": error_count,
                "total_vehicles_detected": total_vehicles,
                "total_processing_time": time.time() - start_time,
                "timestamp": str(datetime.now()),
                "processing_fps": processing_fps  # Add the selected FPS to report
            },
            "roi_config": {
                "points": roi_points,
                "width_meters": target_width,
                "height_meters": target_height
            },
            "video_results": results
        }
        
        # Save report
        with open(report_path, "w") as f:
            json.dump(report_data, f, indent=2)
        
        # Save ROI configuration
        roi_path = os.path.join(output_folder, "roi_points.json")
        roi_data = {
            "roi_points": roi_points,
            "width_meters": target_width,
            "height_meters": target_height,
            "timestamp": str(datetime.now())
        }
        
        with open(roi_path, "w") as f:
            json.dump(roi_data, f, indent=2)
        
        # Print summary
        processing_time = time.time() - start_time
        print("\n" + "=" * 60)
        print(f"Batch processing completed in {processing_time:.1f} seconds")
        print(f"Videos processed: {success_count} successful, {error_count} failed")
        print(f"Total vehicles detected: {total_vehicles}")
        print(f"Processed at: {processing_fps:.2f} FPS")
        print(f"Results saved to: {report_path}")
        print("=" * 60)
        
        if error_count > 0:
            print(f"\n⚠️ Completed with {error_count} errors")
        else:
            print("\n✅ All videos processed successfully")
        
        return results
        
    except Exception as e:
        logger.exception(f"Error in batch processing: {e}")
        print(f"\n❌ Fatal error in batch processing: {e}")
        return []


def main():
    """Main function to process traffic videos"""
    try:
        print("=" * 80)
        print(" Traffic Monitoring System - Batch Video Processing ")
        print("=" * 80)
        
        # Use hardcoded directories
        input_folder = INPUT_DIR
        output_folder = OUTPUT_DIR
        
        # Ensure folders exist
        Path(input_folder).mkdir(parents=True, exist_ok=True)
        Path(output_folder).mkdir(parents=True, exist_ok=True)
        
        print(f"\nInput folder: {input_folder}")
        print(f"Output folder: {output_folder}")
        
        # Check if input folder has video files
        video_extensions = ['.mp4', '.mov', '.avi']
        video_files = [f for f in os.listdir(input_folder) 
                      if any(f.lower().endswith(ext) for ext in video_extensions)]
        
        if not video_files:
            print(f"\nNo video files found in {input_folder}")
            print("Please add .mp4, .mov, or .avi files to this folder and run the script again.")
            return
        
        # Step 1: Show properties of all videos and let user select videos
        print(f"\nFound {len(video_files)} videos in the input folder.")
        print("Displaying video properties...")
        video_paths = [os.path.join(input_folder, f) for f in video_files]
        video_properties, selected_indices, action = show_videos_properties(video_paths)
        
        if action == "cancel":
            print("Operation canceled by user.")
            return
        
        # Filter videos based on selection
        if action == "selected":
            selected_video_paths = [video_paths[i] for i in selected_indices]
            print(f"\nSelected {len(selected_video_paths)} videos for processing.")
        else:  # action == "all"
            selected_video_paths = video_paths
            print(f"\nProcessing all {len(selected_video_paths)} videos.")
        
        if not video_properties:
            print("Could not read video properties. Please check the input files.")
            return
        
        # Select reference video for ROI selection (use first selected video)
        reference_video = selected_video_paths[0]
        print(f"\nUsing {os.path.basename(reference_video)} as reference for ROI selection")
        
        # Step 2: Get ROI from user
        roi_points = None
        roi_attempts = 0
        max_attempts = 3
        
        while roi_points is None and roi_attempts < max_attempts:
            roi_attempts += 1
            roi_points = select_roi_for_video(reference_video)
            
            if roi_points is None and roi_attempts < max_attempts:
                print(f"\nROI selection failed. Please try again ({roi_attempts}/{max_attempts})")
        
        if roi_points is None:
            print("\nROI selection failed multiple times. Using default ROI.")
            # Generate default ROI
            roi_handler = ROIHandler()
            roi_points = roi_handler.generate_default_roi(reference_video)
            # Convert to list if it's numpy array
            if isinstance(roi_points, np.ndarray):
                roi_points = roi_points.tolist()
            print(f"Generated default ROI: {roi_points}")
        
        # Step 3: Get ROI dimensions from user
        width_meters, height_meters = get_roi_dimensions()
        
        # Step 4: Select processing FPS
        cap = cv2.VideoCapture(reference_video)
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        processing_fps = select_processing_fps(video_fps)
        
        # Step 5: Process videos
        print("\nStarting batch processing...")
        process_videos(
            input_folder=input_folder,
            output_folder=output_folder,
            roi_points=roi_points,
            target_width=width_meters,
            target_height=height_meters,
            selected_videos=selected_video_paths,
            processing_fps=processing_fps  # Pass the user-selected FPS
        )
        
    except Exception as e:
        logger.exception(f"An unexpected error occurred: {e}")
        print(f"\n❌ An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()