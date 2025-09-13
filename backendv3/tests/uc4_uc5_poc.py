import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import torch
from ultralytics import YOLO
import supervision as sv
from pathlib import Path
import logging
import datetime
import os
import sys

# Basic logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Constants and Paths ---
POC_SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = POC_SCRIPT_DIR.parent
DEFAULT_MODEL_NAME = "yolov8x.pt" # Ensure this model is accessible
MODEL_PATH_PRIMARY = BACKEND_DIR / DEFAULT_MODEL_NAME
MODEL_PATH_SECONDARY = BACKEND_DIR.parent / DEFAULT_MODEL_NAME # Workspace root
ASSETS_DIR = BACKEND_DIR / "assets"
DEBUG_OUTPUT_DIR = ASSETS_DIR / "debug"

# Ensure debug output directory exists
DEBUG_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Copied and Adapted VehicleDetector class from video/detector.py ---
class VehicleDetectorPOC:
    def __init__(self, model_path_str="yolov8x.pt", confidence=0.3, use_gpu=True, imgsz=1280, augment=False): # Added imgsz and augment
        self.device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        logger.info(f"VehicleDetectorPOC: Using device: {self.device} for detection")
        
        resolved_model_path = Path(model_path_str)
        
        # Check if the provided path is absolute and valid
        if resolved_model_path.is_file():
            pass # Use as is
        # Check relative to backend dir
        elif (BACKEND_DIR / resolved_model_path.name).is_file():
            resolved_model_path = BACKEND_DIR / resolved_model_path.name
        # Check relative to workspace root
        elif (BACKEND_DIR.parent / resolved_model_path.name).is_file():
            resolved_model_path = BACKEND_DIR.parent / resolved_model_path.name
        else:
            logger.error(f"Model file '{model_path_str}' not found at specified path, backend dir, or workspace root.")
            raise FileNotFoundError(f"Model file not found: {model_path_str}")
        
        logger.info(f"VehicleDetectorPOC: Loading model from: {resolved_model_path}")
        self.model = YOLO(str(resolved_model_path))
        self.model.to(self.device)
        
        self.confidence = confidence
        self.imgsz = imgsz  # Store imgsz
        self.augment = augment # Store augment
        self.class_names = self.model.names
        
    def detect(self, frame): # Removed resolution parameter, uses self.imgsz
        # Sharpen the image
        sharpening_kernel = np.array([[-1, -1, -1],
                                      [-1,  9, -1],
                                      [-1, -1, -1]])
        sharpened_frame = cv2.filter2D(frame, -1, sharpening_kernel)

        # The 'augment' parameter enables Test-Time Augmentation
        # The 'imgsz' parameter sets the input image size for the model
        result = self.model(sharpened_frame, imgsz=self.imgsz, verbose=False, device=self.device, augment=self.augment)[0]
        detections = sv.Detections.from_ultralytics(result)
        
        detections = detections[detections.confidence > self.confidence]
        vehicle_class_ids = [2, 3, 5, 7] # COCO: car, motorcycle, bus, truck
        vehicle_mask = np.isin(detections.class_id, vehicle_class_ids)
        detections = detections[vehicle_mask]
        
        return detections

# --- Copied and Adapted Congestion Calculation Logic ---
CONGESTION_THRESHOLDS_POC = [1.0, 2.0, 3.0, 5.0]

def _get_congestion_level_poc(density: float) -> int:
    for level, threshold in enumerate(CONGESTION_THRESHOLDS_POC, 1):
        if density < threshold:
            return level
    return len(CONGESTION_THRESHOLDS_POC) + 1

def _get_congestion_text_poc(level: int) -> str:
    congestion_texts = {
        1: "Free flowing", 2: "Light traffic", 3: "Moderate traffic",
        4: "Heavy traffic", 5: "Very heavy / Jammed"
    }
    return congestion_texts.get(level, "Unknown")

def _count_vehicles_in_roi_poc(detections, mask, frame_shape) -> int:
    if not hasattr(detections, 'xyxy') or not hasattr(detections.xyxy, '__len__') or len(detections.xyxy) == 0:
        return 0
    
    vehicle_count = 0
    for i in range(len(detections.xyxy)):
        bbox = detections.xyxy[i]
        x1, y1, x2, y2 = map(int, bbox[:4])
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        
        if 0 <= center_x < frame_shape[1] and 0 <= center_y < frame_shape[0]:
            if mask[center_y, center_x] > 0:
                vehicle_count += 1
    return vehicle_count

def calculate_congestion_poc(frame_path: str, roi_points_normalized: list, 
                             roi_width_meters: float, roi_height_meters: float,
                             detector: VehicleDetectorPOC) -> dict:
    try:
        frame = cv2.imread(frame_path)
        if frame is None:
            raise ValueError(f"Could not load frame from {frame_path}")

        h, w = frame.shape[:2]
        roi_polygon_pixels = np.array([
            (int(p["x"] * w), int(p["y"] * h)) for p in roi_points_normalized
        ], dtype=np.int32)

        detections = detector.detect(frame)
        
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [roi_polygon_pixels], 255)
        
        vehicles_in_roi = _count_vehicles_in_roi_poc(detections, mask, frame.shape)
        
        roi_area_m2 = roi_width_meters * roi_height_meters
        density = (vehicles_in_roi / roi_area_m2) * 100 if roi_area_m2 > 0.01 else 0
        congestion_level = _get_congestion_level_poc(density)
        
        return {
            "congestion_level": congestion_level,
            "congestion_text": _get_congestion_text_poc(congestion_level),
            "vehicle_count": vehicles_in_roi,
            "roi_area_m2": round(roi_area_m2, 2),
            "vehicle_density": round(density, 2),
            "calculation_timestamp": datetime.datetime.now().isoformat(),
            "detections_for_drawing": detections,
            "roi_polygon_pixels_for_drawing": roi_polygon_pixels
        }
    except Exception as e:
        logger.error(f"Error in POC congestion calculation: {str(e)}", exc_info=True)
        return {"error": str(e), "congestion_level": 0, "congestion_text": "Error", "vehicle_count": 0}

def debug_draw_roi_and_detections_poc(frame_path, roi_polygon_pixels, detections, output_dir, original_filename):
    frame = cv2.imread(frame_path)
    if frame is None: return None
    
    cv2.polylines(frame, [roi_polygon_pixels], True, (0, 255, 0), 2)
    
    if detections and hasattr(detections, 'xyxy') and hasattr(detections.xyxy, '__len__') and len(detections.xyxy) > 0:
        for i in range(len(detections.xyxy)):
            bbox = detections.xyxy[i]
            x1, y1, x2, y2 = map(int, bbox[:4])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
            cv2.circle(frame, (center_x, center_y), 4, (255, 0, 0), -1)

    output_filename = f"debug_poc_{original_filename}"
    output_path = Path(output_dir) / output_filename
    cv2.imwrite(str(output_path), frame)
    logger.info(f"POC Debug image saved to: {output_path}")
    return str(output_path)

# --- Tkinter UI Application ---
class CongestionPOCApp:
    def __init__(self, root, model_file_path):
        self.root = root
        self.root.title("UC4/UC5 Congestion POC - 4-Point ROI")
        self.root.geometry("1200x900") # Adjusted height for new button

        self.image_path = None
        self.tk_image_input = None
        self.tk_image_output = None
        self.current_image_pil = None
        
        # ROI state for 4-point selection
        self.roi_click_points_canvas = [] # Stores (x,y) canvas coordinates of clicks
        self.drawn_roi_element_ids = [] # Stores IDs of points and lines on canvas
        self.drawn_roi_normalized = None # List of 4 normalized point dicts [{'x': nx, 'y': ny}, ...]

        self.canvas_display_info = {
            "image_id": None, "original_w": 0, "original_h": 0,
            "scale": 1.0, "offset_x": 0, "offset_y": 0,
            "scaled_w": 0, "scaled_h": 0
        }
        
        try:
            self.detector = VehicleDetectorPOC(
                model_path_str=str(model_file_path),
                confidence=0.25, # Reduced confidence
                imgsz=640,       # Set input size
                augment=False    # TTA disabled by default
            )
        except Exception as e:
            messagebox.showerror("Initialization Error", f"Failed to initialize Vehicle Detector: {e}")
            self.detector = None
            raise 

        main_frame = ttk.Frame(root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        controls_frame = ttk.Frame(main_frame)
        controls_frame.pack(fill=tk.X, pady=5)

        btn_select_image = ttk.Button(controls_frame, text="Select Image", command=self.select_image)
        btn_select_image.pack(side=tk.LEFT, padx=5)
        self.lbl_image_path = ttk.Label(controls_frame, text="No image selected")
        self.lbl_image_path.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)

        roi_frame = ttk.LabelFrame(main_frame, text="ROI Definition (Click 4 points on Input Image)", padding="10")
        roi_frame.pack(fill=tk.X, pady=10)
        
        self.roi_entries = {}
        roi_config = [
            ("actual_w_m", "Actual ROI Width (m):", "10"), 
            ("actual_h_m", "Actual ROI Height (m):", "10")
        ]
        for i, (key, label_text, default_val) in enumerate(roi_config):
            lbl = ttk.Label(roi_frame, text=label_text)
            lbl.grid(row=i, column=0, padx=5, pady=2, sticky=tk.W)
            entry = ttk.Entry(roi_frame, width=10)
            entry.grid(row=i, column=1, padx=5, pady=2, sticky=tk.EW)
            entry.insert(0, default_val)
            self.roi_entries[key] = entry
        
        self.lbl_drawn_roi_info = ttk.Label(roi_frame, text="ROI Points: Click 4 points on the image.")
        self.lbl_drawn_roi_info.grid(row=len(roi_config), column=0, columnspan=2, padx=5, pady=5, sticky=tk.W)

        btn_reset_roi = ttk.Button(roi_frame, text="Reset ROI Points", command=self._clear_drawn_roi)
        btn_reset_roi.grid(row=len(roi_config) + 1, column=0, columnspan=2, pady=5)


        btn_process = ttk.Button(main_frame, text="Process Image", command=self.process_image)
        btn_process.pack(pady=10)

        results_frame = ttk.LabelFrame(main_frame, text="Results", padding="10")
        results_frame.pack(fill=tk.X, pady=10)
        self.lbl_results = {}
        result_keys_display = ["Vehicle Count:", "Density (veh/100m²):", "Congestion Level:", "Congestion Text:"]
        for i, key_display in enumerate(result_keys_display):
            lbl_key = ttk.Label(results_frame, text=key_display)
            lbl_key.grid(row=i, column=0, sticky=tk.W, padx=5)
            lbl_val = ttk.Label(results_frame, text="N/A")
            lbl_val.grid(row=i, column=1, sticky=tk.W, padx=5)
            self.lbl_results[key_display] = lbl_val
        
        image_display_frame = ttk.Frame(main_frame)
        image_display_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        self.canvas_input_image = tk.Canvas(image_display_frame, bg="lightgrey", relief=tk.SUNKEN, bd=2)
        self.canvas_input_image.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        self.canvas_input_image.bind("<ButtonPress-1>", self._on_canvas_press)
        # Removed <B1-Motion> and <ButtonRelease-1> bindings for rectangle drawing
        
        self.lbl_image_output_display = ttk.Label(image_display_frame, text="Output Image Preview", relief=tk.SUNKEN)
        self.lbl_image_output_display.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)

    def _clear_drawn_roi(self, from_select_image=False):
        for item_id in self.drawn_roi_element_ids:
            self.canvas_input_image.delete(item_id)
        self.drawn_roi_element_ids = []
        self.roi_click_points_canvas = []
        self.drawn_roi_normalized = None
        if not from_select_image: # Avoid double update if called from select_image
            self.lbl_drawn_roi_info.config(text="ROI Points: Click 4 points on the image.")
        logger.info("Cleared drawn ROI points and shapes.")

    def _on_canvas_press(self, event):
        if not self.current_image_pil:
            messagebox.showwarning("No Image", "Please select an image first.")
            return

        if len(self.roi_click_points_canvas) >= 4:
            messagebox.showinfo("ROI Complete", "4 points already selected. Reset ROI to select new points.")
            return

        # Get click coordinates relative to the canvas widget
        cx = self.canvas_input_image.canvasx(event.x)
        cy = self.canvas_input_image.canvasy(event.y)

        # Check if click is within the displayed image area on canvas
        img_offset_x = self.canvas_display_info["offset_x"]
        img_offset_y = self.canvas_display_info["offset_y"]
        scaled_img_w = self.canvas_display_info["scaled_w"]
        scaled_img_h = self.canvas_display_info["scaled_h"]

        if not (img_offset_x <= cx < img_offset_x + scaled_img_w and \
                img_offset_y <= cy < img_offset_y + scaled_img_h):
            messagebox.showwarning("Out of Bounds", "Please click within the image area.")
            return

        self.roi_click_points_canvas.append((cx, cy))
        
        # Draw a small circle for the point
        radius = 3
        point_id = self.canvas_input_image.create_oval(
            cx - radius, cy - radius, cx + radius, cy + radius, 
            fill="red", outline="red"
        )
        self.drawn_roi_element_ids.append(point_id)

        num_points = len(self.roi_click_points_canvas)
        self.lbl_drawn_roi_info.config(text=f"ROI Points: {num_points}/4 selected. Click {4-num_points} more.")

        # Draw lines between points
        if num_points > 1:
            p_prev = self.roi_click_points_canvas[-2]
            p_curr = self.roi_click_points_canvas[-1]
            line_id = self.canvas_input_image.create_line(p_prev[0], p_prev[1], p_curr[0], p_curr[1], fill="red", width=2)
            self.drawn_roi_element_ids.append(line_id)

        if num_points == 4:
            # Close the polygon (4th point to 1st point)
            p_last = self.roi_click_points_canvas[-1]
            p_first = self.roi_click_points_canvas[0]
            closing_line_id = self.canvas_input_image.create_line(
                p_last[0], p_last[1], p_first[0], p_first[1], 
                fill="lime green", width=2, tags="final_roi_line"
            )
            self.drawn_roi_element_ids.append(closing_line_id)
            
            # Change color of existing lines to green
            for item_id in self.drawn_roi_element_ids:
                if self.canvas_input_image.type(item_id) == "line":
                    self.canvas_input_image.itemconfig(item_id, fill="lime green")
                elif self.canvas_input_image.type(item_id) == "oval": # Points
                     self.canvas_input_image.itemconfig(item_id, fill="lime green", outline="lime green")


            self._finalize_roi_points()
            self.lbl_drawn_roi_info.config(text=f"ROI Defined (4 points). Normalized: " + 
                                           ", ".join([f"({p['x']:.2f},{p['y']:.2f})" for p in self.drawn_roi_normalized]))
            logger.info(f"4 ROI points selected and finalized: {self.drawn_roi_normalized}")


    def _finalize_roi_points(self):
        if len(self.roi_click_points_canvas) != 4:
            logger.warning("Attempted to finalize ROI without 4 points.")
            return

        normalized_points = []
        
        offset_x = self.canvas_display_info["offset_x"]
        offset_y = self.canvas_display_info["offset_y"]
        scaled_img_w = self.canvas_display_info["scaled_w"]
        scaled_img_h = self.canvas_display_info["scaled_h"]
        original_w = self.canvas_display_info["original_w"]
        original_h = self.canvas_display_info["original_h"]

        if scaled_img_w == 0 or scaled_img_h == 0 or original_w == 0 or original_h == 0:
            logger.error("Image dimensions are zero, cannot normalize ROI points.")
            messagebox.showerror("Error", "Image dimension error, cannot normalize ROI.")
            self._clear_drawn_roi()
            return

        for (cx, cy) in self.roi_click_points_canvas:
            # Point relative to scaled image's top-left on canvas
            roi_on_scaled_img_x = cx - offset_x
            roi_on_scaled_img_y = cy - offset_y

            # Clamp to scaled image boundaries (though click check should prevent this)
            roi_on_scaled_img_x = max(0, min(roi_on_scaled_img_x, scaled_img_w))
            roi_on_scaled_img_y = max(0, min(roi_on_scaled_img_y, scaled_img_h))
            
            # Normalize
            norm_x = roi_on_scaled_img_x / scaled_img_w
            norm_y = roi_on_scaled_img_y / scaled_img_h
            
            normalized_points.append({"x": norm_x, "y": norm_y})
        
        self.drawn_roi_normalized = normalized_points
        # Visual feedback is updated in _on_canvas_press after this call

    # Removed _on_canvas_drag and _on_canvas_release methods

    def select_image(self):
        self.image_path = filedialog.askopenfilename(
            initialdir=str(ASSETS_DIR), title="Select Camera Image",
            filetypes=(("JPEG files", "*.jpg"), ("PNG files", "*.png"), ("All files", "*.*"))
        )
        if self.image_path:
            self.lbl_image_path.config(text=self.image_path)
            self.display_image(self.image_path, self.canvas_input_image, "input") 
            
            self.lbl_image_output_display.config(image=''); 
            if hasattr(self.lbl_image_output_display, 'image'): self.lbl_image_output_display.image = None 
            for lbl in self.lbl_results.values(): lbl.config(text="N/A")
            
            self._clear_drawn_roi(from_select_image=True) # Clear ROI for new image
            self.lbl_drawn_roi_info.config(text="ROI Points: Click 4 points on the image.")


    def display_image(self, path, widget, type_str):
        try:
            img_pil = Image.open(path)
            img_w, img_h = img_pil.size

            self.root.update_idletasks() 
            widget_w = widget.winfo_width() if widget.winfo_width() > 20 else 550 
            widget_h = widget.winfo_height() if widget.winfo_height() > 20 else 550

            scale_w_ratio = widget_w / img_w
            scale_h_ratio = widget_h / img_h
            scale = min(scale_w_ratio, scale_h_ratio, 1.0)

            new_w = int(img_w * scale)
            new_h = int(img_h * scale)
            
            img_resized_pil = img_pil.resize((new_w, new_h), Image.Resampling.LANCZOS)
            tk_image = ImageTk.PhotoImage(img_resized_pil)

            if type_str == "input" and isinstance(widget, tk.Canvas):
                self.current_image_pil = img_pil
                self.tk_image_input = tk_image
                
                offset_x = (widget_w - new_w) / 2
                offset_y = (widget_h - new_h) / 2

                # Preserve existing image_id if any, to delete it before drawing new one
                old_image_id = self.canvas_display_info.get("image_id")

                self.canvas_display_info = {
                    "original_w": img_w, "original_h": img_h,
                    "scale": scale, "offset_x": offset_x, "offset_y": offset_y,
                    "scaled_w": new_w, "scaled_h": new_h,
                    "image_id": old_image_id 
                }
                
                if old_image_id: # Delete previous canvas image
                    widget.delete(old_image_id)
                # Note: _clear_drawn_roi is called separately in select_image for ROI elements

                new_image_id = widget.create_image(offset_x, offset_y, anchor=tk.NW, image=tk_image)
                self.canvas_display_info["image_id"] = new_image_id # Update with new image_id
                widget.image = tk_image
            
            elif type_str == "output" and isinstance(widget, ttk.Label):
                self.tk_image_output = tk_image
                widget.config(image=tk_image)
                widget.image = tk_image
            
            else:
                logger.warning(f"Display_image called with unhandled widget type or type_str: {type_str}, {type(widget)}")

        except Exception as e:
            messagebox.showerror("Image Display Error", f"Failed to display image {path}: {e}")
            logger.error(f"Error displaying image {path}: {e}", exc_info=True)

    def process_image(self):
        if not self.image_path:
            messagebox.showwarning("No Image", "Please select an image first.")
            return
        if not self.detector:
            messagebox.showerror("Error", "Vehicle detector not initialized. Cannot process.")
            return
        if not self.drawn_roi_normalized or len(self.drawn_roi_normalized) != 4:
            messagebox.showwarning("No ROI", "Please define a 4-point ROI on the input image first.")
            return

        try:
            actual_w_m = float(self.roi_entries['actual_w_m'].get())
            actual_h_m = float(self.roi_entries['actual_h_m'].get())

            if actual_w_m <= 0 or actual_h_m <= 0:
                messagebox.showerror("Invalid ROI", "Actual ROI dimensions in meters must be positive.")
                return

            # self.drawn_roi_normalized is already the list of 4 points
            roi_points_normalized = self.drawn_roi_normalized

            results = calculate_congestion_poc(
                self.image_path, roi_points_normalized, actual_w_m, actual_h_m, self.detector
            )

            if "error" in results:
                messagebox.showerror("Processing Error", results["error"])
                for key_display in self.lbl_results: self.lbl_results[key_display].config(text="Error")
                return

            self.lbl_results["Vehicle Count:"].config(text=str(results["vehicle_count"]))
            self.lbl_results["Density (veh/100m²):"].config(text=f"{results['vehicle_density']:.2f}")
            self.lbl_results["Congestion Level:"].config(text=str(results["congestion_level"]))
            self.lbl_results["Congestion Text:"].config(text=results["congestion_text"])

            original_filename = Path(self.image_path).name
            output_image_path = debug_draw_roi_and_detections_poc(
                self.image_path, results["roi_polygon_pixels_for_drawing"],
                results["detections_for_drawing"], DEBUG_OUTPUT_DIR, original_filename
            )

            if output_image_path:
                self.root.update_idletasks() 
                self.display_image(output_image_path, self.lbl_image_output_display, "output")
            else:
                messagebox.showerror("Output Error", "Failed to generate or save debug image.")

        except ValueError as ve:
            messagebox.showerror("Input Error", f"Invalid ROI input. Please enter numbers. Details: {ve}")
            logger.error(f"ROI Input Error: {ve}", exc_info=True)
        except Exception as e:
            messagebox.showerror("Processing Error", f"An unexpected error occurred: {e}")
            logger.error(f"Error during processing: {e}", exc_info=True)

if __name__ == "__main__":
    final_model_path = None
    if MODEL_PATH_PRIMARY.is_file():
        final_model_path = MODEL_PATH_PRIMARY
    elif MODEL_PATH_SECONDARY.is_file():
        final_model_path = MODEL_PATH_SECONDARY
    
    root = tk.Tk()
    if not final_model_path:
        logger.error(f"CRITICAL: YOLO model '{DEFAULT_MODEL_NAME}' not found at {MODEL_PATH_PRIMARY} or {MODEL_PATH_SECONDARY}")
        root.withdraw() # Hide main window before showing error and exiting
        messagebox.showerror("Startup Error", f"YOLO Model file '{DEFAULT_MODEL_NAME}' not found. POC cannot start.")
        root.destroy()
        sys.exit(1)
    
    logger.info(f"Using model for POC: {final_model_path}")
    try:
        app = CongestionPOCApp(root, final_model_path)
        root.mainloop()
    except Exception as e: # Catch init errors from CongestionPOCApp
        logger.error(f"Application failed to start: {e}", exc_info=True)
        if not root.winfo_exists(): # If root was destroyed or never fully made
            root_err = tk.Tk()
            root_err.withdraw()
            messagebox.showerror("Application Startup Error", f"Failed to initialize: {e}")
            root_err.destroy()
        else: # if root still exists, show error there
            messagebox.showerror("Application Startup Error", f"Failed to initialize: {e}")
            root.destroy()
        sys.exit(1)
