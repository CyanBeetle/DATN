"""
GPU Diagnostics Tool for Traffic Monitoring System

This standalone script performs comprehensive diagnostics on GPU availability,
CUDA support, and PyTorch configuration to help troubleshoot GPU usage issues.
"""

import os
import sys
import platform
import subprocess
from pathlib import Path
import importlib.util
from typing import Dict, Any, List, Optional

# Add parent directory to path to import application modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Initialize dictionaries to store diagnostic results
results = {
    "system": {},
    "cuda": {},
    "pytorch": {},
    "ultralytics": {},
    "yolo_model": {},
    "issues": []
}

def print_header(message: str) -> None:
    """Print a formatted header message."""
    print("\n" + "=" * 80)
    print(f" {message} ".center(80, "="))
    print("=" * 80)

def print_section(message: str) -> None:
    """Print a section header."""
    print(f"\n--- {message} ---")

def check_system() -> None:
    """Check system information."""
    print_section("System Information")
    
    results["system"]["os"] = platform.system()
    results["system"]["os_version"] = platform.version()
    results["system"]["python_version"] = platform.python_version()
    results["system"]["architecture"] = platform.architecture()[0]
    results["system"]["processor"] = platform.processor()
    
    print(f"OS: {results['system']['os']} {results['system']['os_version']}")
    print(f"Python: {results['system']['python_version']}")
    print(f"Architecture: {results['system']['architecture']}")
    print(f"Processor: {results['system']['processor']}")

def check_cuda_installation() -> None:
    """Check CUDA toolkit installation on the system."""
    print_section("CUDA Installation Check")
    
    # Check if NVIDIA drivers are installed
    try:
        nvidia_smi_output = subprocess.check_output("nvidia-smi", shell=True, stderr=subprocess.STDOUT).decode("utf-8")
        results["cuda"]["nvidia_driver"] = True
        results["cuda"]["nvidia_smi_output"] = nvidia_smi_output.split("\n")[0]
        print(f"✓ NVIDIA Driver: {results['cuda']['nvidia_smi_output']}")
        
        # Extract driver version
        for line in nvidia_smi_output.split("\n"):
            if "Driver Version:" in line:
                results["cuda"]["driver_version"] = line.split("Driver Version:")[1].strip().split()[0]
                break
    except (subprocess.CalledProcessError, FileNotFoundError):
        results["cuda"]["nvidia_driver"] = False
        results["issues"].append("NVIDIA drivers not found or not properly installed")
        print("✗ NVIDIA Driver: Not found or not accessible")
    
    # Check if CUDA toolkit is installed
    try:
        nvcc_output = subprocess.check_output("nvcc --version", shell=True, stderr=subprocess.STDOUT).decode("utf-8")
        results["cuda"]["nvcc"] = True
        
        # Extract CUDA version
        for line in nvcc_output.split("\n"):
            if "release" in line:
                results["cuda"]["version"] = line.split("release")[1].strip().split()[0].rstrip(",")
                break
                
        print(f"✓ CUDA Toolkit: {results['cuda'].get('version', 'Version not detected')}")
    except (subprocess.CalledProcessError, FileNotFoundError):
        results["cuda"]["nvcc"] = False
        results["issues"].append("CUDA toolkit not found or not properly installed")
        print("✗ CUDA Toolkit: Not found or not accessible")

def check_pytorch() -> None:
    """Check PyTorch installation and CUDA support."""
    print_section("PyTorch CUDA Support")
    
    try:
        import torch
        results["pytorch"]["installed"] = True
        results["pytorch"]["version"] = torch.__version__
        print(f"✓ PyTorch: v{results['pytorch']['version']}")
        
        # Check CUDA availability in PyTorch
        results["pytorch"]["cuda_available"] = torch.cuda.is_available()
        if results["pytorch"]["cuda_available"]:
            results["pytorch"]["cuda_version"] = torch.version.cuda
            results["pytorch"]["device_count"] = torch.cuda.device_count()
            results["pytorch"]["current_device"] = torch.cuda.current_device()
            
            print(f"✓ PyTorch CUDA: Available (v{results['pytorch']['cuda_version']})")
            print(f"✓ GPU Devices: {results['pytorch']['device_count']}")
            
            # Print each GPU device info
            for i in range(results["pytorch"]["device_count"]):
                device_name = torch.cuda.get_device_name(i)
                device_cap = torch.cuda.get_device_capability(i)
                print(f"  - GPU {i}: {device_name} (Compute Capability {device_cap[0]}.{device_cap[1]})")
                
            # Simple PyTorch CUDA test
            try:
                x = torch.tensor([1, 2, 3], device='cuda')
                print(f"✓ CUDA Test: Successfully created tensor on GPU: {x.device}")
            except Exception as e:
                results["issues"].append(f"PyTorch CUDA test failed: {str(e)}")
                print(f"✗ CUDA Test: Failed with error: {str(e)}")
        else:
            results["issues"].append("PyTorch CUDA support is not available")
            print("✗ PyTorch CUDA: Not available")
            
            # Check CUDA_VISIBLE_DEVICES environment variable
            cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", None)
            if cuda_visible:
                print(f"  - CUDA_VISIBLE_DEVICES: {cuda_visible}")
                if cuda_visible == "-1":
                    results["issues"].append("CUDA_VISIBLE_DEVICES is set to -1, making GPUs invisible")
            
    except ImportError as e:
        results["pytorch"]["installed"] = False
        results["issues"].append(f"PyTorch import error: {str(e)}")
        print(f"✗ PyTorch: Not installed or import error: {str(e)}")

def check_ultralytics() -> None:
    """Check Ultralytics YOLOv8 installation."""
    print_section("Ultralytics YOLOv8 Check")
    
    try:
        import ultralytics
        results["ultralytics"]["installed"] = True
        results["ultralytics"]["version"] = ultralytics.__version__
        print(f"✓ Ultralytics: v{results['ultralytics']['version']}")
        
        # Check if YOLO uses CUDA
        try:
            from ultralytics import YOLO
            import torch
            
            # Check if the model file exists
            model_paths = [
                os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "yolov8x.pt"),
                "yolov8n.pt"  # Will download if not exists
            ]
            
            model_path = None
            for path in model_paths:
                if os.path.exists(path):
                    model_path = path
                    break
            
            if not model_path:
                model_path = model_paths[1]  # Use default small model
                
            print(f"Loading YOLO model from: {model_path}")
            model = YOLO(model_path)
            
            # Check device
            results["yolo_model"]["device"] = str(model.device)
            results["yolo_model"]["is_cuda"] = "cuda" in str(model.device).lower()
            
            print(f"✓ YOLO Model Device: {results['yolo_model']['device']}")
            
            if not results["yolo_model"]["is_cuda"]:
                results["issues"].append(f"YOLO model using {results['yolo_model']['device']} instead of CUDA")
                print("✗ YOLO is not using CUDA")
            
            # Try to explicitly move model to CUDA
            if torch.cuda.is_available():
                try:
                    print("Attempting to explicitly move model to CUDA...")
                    model.to('cuda')
                    print(f"✓ Model after explicit move: {model.device}")
                    
                    # Check if model can perform inference on GPU
                    print("Testing inference on GPU...")
                    test_img = torch.zeros((3, 640, 640), device='cuda')
                    _ = model(test_img)
                    print("✓ Successfully ran inference on GPU")
                except Exception as e:
                    results["issues"].append(f"Failed to move model to CUDA: {str(e)}")
                    print(f"✗ Could not move model to CUDA: {str(e)}")
                
        except Exception as e:
            results["issues"].append(f"Failed to load YOLO model: {str(e)}")
            print(f"✗ YOLO Model: Error loading model: {str(e)}")
        
    except ImportError as e:
        results["ultralytics"]["installed"] = False
        results["issues"].append(f"Ultralytics import error: {str(e)}")
        print(f"✗ Ultralytics: Not installed or import error: {str(e)}")

def check_torch_backends() -> None:
    """Check PyTorch backends and CUDA settings."""
    print_section("PyTorch Backend Configuration")
    
    try:
        import torch
        
        # Check CUDA flags
        print("CUDA Settings:")
        if hasattr(torch, '_C'):
            results["pytorch"]["cudnn_enabled"] = torch.backends.cudnn.enabled
            results["pytorch"]["cudnn_version"] = torch.backends.cudnn.version()
            print(f"✓ cuDNN: {'Enabled' if torch.backends.cudnn.enabled else 'Disabled'} (v{torch.backends.cudnn.version()})")
            
            results["pytorch"]["cudnn_benchmark"] = torch.backends.cudnn.benchmark
            print(f"✓ cuDNN Benchmark: {'Enabled' if torch.backends.cudnn.benchmark else 'Disabled'}")
            
            results["pytorch"]["cudnn_deterministic"] = torch.backends.cudnn.deterministic
            print(f"✓ cuDNN Deterministic: {'Enabled' if torch.backends.cudnn.deterministic else 'Disabled'}")
            
            if not torch.backends.cudnn.enabled:
                results["issues"].append("cuDNN is not enabled which can significantly slow down neural network operations")
    except:
        pass

def check_yolo_model_usage() -> None:
    """Test YOLO detection with GPU usage."""
    print_section("YOLO Model Usage Test")
    
    try:
        from video.detector import VehicleDetector
        import numpy as np
        import time
        import cv2
        
        # Create test image
        test_img = np.zeros((720, 1280, 3), dtype=np.uint8)
        cv2.putText(test_img, "Test Image", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Test with CPU
        print("Testing detection on CPU...")
        detector_cpu = VehicleDetector(use_gpu=False)
        start_time = time.time()
        _ = detector_cpu.detect(test_img)
        cpu_time = time.time() - start_time
        print(f"CPU Detection time: {cpu_time:.4f} seconds")
        print(f"Device used: {detector_cpu.device}")
        
        # Test with GPU (if available)
        try:
            print("\nTesting detection on GPU...")
            detector_gpu = VehicleDetector(use_gpu=True)
            start_time = time.time()
            _ = detector_gpu.detect(test_img)
            gpu_time = time.time() - start_time
            print(f"GPU Detection time: {gpu_time:.4f} seconds")
            print(f"Device used: {detector_gpu.device}")
            
            if "cuda" in detector_gpu.device.lower():
                print(f"✓ YOLO using GPU: {detector_gpu.device}")
                if cpu_time > gpu_time:
                    speedup = cpu_time / gpu_time
                    print(f"✓ GPU is {speedup:.2f}x faster than CPU")
                else:
                    results["issues"].append("GPU detection is not faster than CPU")
                    print(f"✗ GPU is not faster than CPU (GPU: {gpu_time:.4f}s, CPU: {cpu_time:.4f}s)")
            else:
                results["issues"].append("VehicleDetector not using GPU despite use_gpu=True")
                print(f"✗ VehicleDetector still using CPU ({detector_gpu.device}) despite use_gpu=True")
                
        except Exception as e:
            results["issues"].append(f"Error during GPU detection test: {str(e)}")
            print(f"✗ GPU detection test failed: {str(e)}")
            
    except Exception as e:
        results["issues"].append(f"Could not import VehicleDetector: {str(e)}")
        print(f"✗ Failed to test VehicleDetector: {str(e)}")

def fix_video_detector():
    """Try to fix VehicleDetector to use GPU."""
    print_section("Attempting to Fix VehicleDetector")
    
    try:
        import torch
        if not torch.cuda.is_available():
            print("❌ Cannot fix: CUDA not available in PyTorch")
            return False
            
        from video.detector import VehicleDetector
        
        # First check if it's already working
        detector = VehicleDetector(use_gpu=True)
        if "cuda" in detector.device.lower():
            print("✓ VehicleDetector already correctly using GPU")
            return True
            
        print("Looking for issues in the VehicleDetector class...")
        
        # Check the file content
        file_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "video", "detector.py")
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                content = f.read()
                
            # Look for common issues
            issues = []
            
            if "self.device = \"cuda\"" in content and "self.model.to(self.device)" not in content:
                issues.append("Model is not being moved to CUDA device")
                
            if "self.device = \"cuda\"" in content and ".to(self.device)" not in content:
                issues.append("Model outputs are not being moved to CUDA device")
                
            # Print found issues
            if issues:
                print("Found potential issues:")
                for issue in issues:
                    print(f"- {issue}")
                print("\nSuggested fix: Ensure the model is moved to CUDA device with .to(device)")
            else:
                print("No obvious issues found in the code. The problem might be with environment or dependencies.")
                
        else:
            print(f"❌ Cannot analyze: File not found at {file_path}")
            
    except Exception as e:
        print(f"❌ Error during fix attempt: {str(e)}")
        return False
    
    return False

def check_recommended_fixes():
    """Provide recommended fixes based on diagnostics."""
    print_section("Recommended Fixes")
    
    if not results["issues"]:
        print("✓ No issues detected with GPU setup!")
        return
    
    # Check if PyTorch has CUDA support
    if not results.get("pytorch", {}).get("cuda_available", False):
        print("⚠️ PyTorch doesn't have CUDA support. Reinstall PyTorch with CUDA:")
        print("  pip uninstall torch torchvision torchaudio -y")
        print("  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    
    # Check if model is being explicitly moved to GPU
    if "YOLO model using" in str(results["issues"]):
        print("\n⚠️ YOLO model not using GPU. Fix VehicleDetector class:")
        print("  1. Open backendv3/video/detector.py")
        print("  2. Find the __init__ method")
        print("  3. Make sure these lines are present and correct:")
        print("     - self.device = \"cuda\" if use_gpu and torch.cuda.is_available() else \"cpu\"")
        print("     - self.model.to(self.device)")
        print("  4. In detect method, ensure tensors are moved to the device:")
        print("     - results = self.model(frame, device=self.device)[0]")
    
    # Check if environment variables are blocking CUDA
    if os.environ.get("CUDA_VISIBLE_DEVICES", None) == "-1":
        print("\n⚠️ CUDA_VISIBLE_DEVICES is set to -1, blocking GPU access:")
        print("  Set CUDA_VISIBLE_DEVICES environment variable to your GPU id (usually 0):")
        print("  Temporary solution: set CUDA_VISIBLE_DEVICES=0")
        print("  Permanent solution: Add to system environment variables")

def run_all_checks() -> Dict[str, Any]:
    """Run all diagnostic checks."""
    print_header("GPU/CUDA Diagnostic Tool for Traffic Monitoring System")
    
    check_system()
    check_cuda_installation()
    check_pytorch()
    check_torch_backends()
    check_ultralytics()
    check_yolo_model_usage()
    
    # Attempt to fix if there are issues
    if results["issues"]:
        fix_video_detector()
    
    # Summarize findings
    print_header("Diagnostic Summary")
    
    if not results.get("issues"):
        print("\n✓ No issues detected with GPU setup!")
        print("If you're still experiencing problems with GPU acceleration:")
        print("  1. Check your code is correctly using the CUDA device")
        print("  2. Ensure CUDA memory is not exhausted by other processes")
        print("  3. Try restarting your computer to clear GPU memory")
    else:
        print("\n⚠️ The following issues were found:")
        for i, issue in enumerate(results.get("issues", [])):
            print(f"  {i+1}. {issue}")
            
        # Print recommended fixes
        check_recommended_fixes()
            
    return results

if __name__ == "__main__":
    try:
        results = run_all_checks()
        print("\nRun this tool again after making changes to verify the fixes.")
    except Exception as e:
        print(f"\n❌ Diagnostics failed with error: {str(e)}")
        import traceback
        traceback.print_exc()