"""GPU resource management utilities."""
import time
import gc
import psutil
import logging

logger = logging.getLogger(__name__)

# Try to import GPU monitoring tools
try:
    import torch
    import pynvml as nvidia_smi
    GPU_AVAILABLE = torch.cuda.is_available()
except ImportError:
    GPU_AVAILABLE = False
    logger.info("GPU monitoring libraries not available. GPU features will be disabled.")


class ResourceMonitor:
    """Monitor and manage system resources."""
    
    def __init__(self, cpu_threshold=90, memory_threshold=90, gpu_threshold=80):
        """
        Initialize resource monitor with thresholds.
        
        Args:
            cpu_threshold (int): CPU usage percentage threshold for cooling down
            memory_threshold (int): Memory usage percentage threshold for cooling down
            gpu_threshold (int): GPU memory usage percentage threshold for cooling down
        """
        self.cpu_threshold = cpu_threshold
        self.memory_threshold = memory_threshold
        self.gpu_threshold = gpu_threshold
        
        # Initialize NVIDIA management interface if available
        if GPU_AVAILABLE:
            try:
                nvidia_smi.nvmlInit()
                self.device_count = nvidia_smi.nvmlDeviceGetCount()
                logger.info(f"Found {self.device_count} NVIDIA GPU devices")
            except:
                logger.warning("Failed to initialize NVIDIA management interface")
                self.device_count = 0
        else:
            self.device_count = 0
    
    def check_resources(self):
        """
        Check current resource usage levels.
        
        Returns:
            dict: Resource usage information
        """
        resources = {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "gpu_available": GPU_AVAILABLE,
            "gpu_memory_used": None,
            "gpu_memory_total": None,
            "gpu_memory_percent": 0
        }
        
        # Get GPU memory info if available
        if GPU_AVAILABLE:
            try:
                if hasattr(torch.cuda, 'memory_allocated') and hasattr(torch.cuda, 'memory_reserved'):
                    # PyTorch method
                    resources["gpu_memory_used"] = torch.cuda.memory_allocated() / (1024 ** 3)  # GB
                    resources["gpu_memory_total"] = torch.cuda.memory_reserved() / (1024 ** 3)  # GB
                    resources["gpu_memory_percent"] = (resources["gpu_memory_used"] / resources["gpu_memory_total"] * 100) if resources["gpu_memory_total"] > 0 else 0
                
                # Alternative using nvidia-smi
                elif hasattr(self, 'device_count') and self.device_count > 0:
                    handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
                    info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
                    resources["gpu_memory_used"] = info.used / (1024 ** 3)  # GB
                    resources["gpu_memory_total"] = info.total / (1024 ** 3)  # GB
                    resources["gpu_memory_percent"] = (info.used / info.total) * 100
            except Exception as e:
                logger.warning(f"Error getting GPU information: {str(e)}")
                
        return resources
    
    def cool_down_if_needed(self):
        """
        Free up resources if thresholds are exceeded.
        
        Returns:
            bool: True if cool-down occurred, False otherwise
        """
        resources = self.check_resources()
        needs_cooldown = False
        
        if resources['cpu_percent'] > self.cpu_threshold:
            logger.warning(f"⚠️ High CPU usage detected ({resources['cpu_percent']}%). Cooling down...")
            needs_cooldown = True
            
        if resources['memory_percent'] > self.memory_threshold:
            logger.warning(f"⚠️ High memory usage detected ({resources['memory_percent']}%). Cooling down...")
            needs_cooldown = True
            
        if GPU_AVAILABLE and resources.get('gpu_memory_percent', 0) > self.gpu_threshold:
            logger.warning(f"⚠️ High GPU memory usage detected: {resources.get('gpu_memory_percent', 0):.1f}%. Cooling down...")
            needs_cooldown = True
        
        if needs_cooldown:
            # Force garbage collection
            gc.collect()
            if GPU_AVAILABLE and hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()
                
            # Wait for resources to free up
            logger.info("Waiting 5 seconds to let system resources free up...")
            time.sleep(5)
            return True
            
        return False
    
    def get_optimal_device(self):
        """
        Determine the optimal device (CPU/GPU) to use based on current resource state.
        
        Returns:
            str: 'cuda' if GPU is available and not overloaded, otherwise 'cpu'
        """
        if not GPU_AVAILABLE:
            return "cpu"
            
        resources = self.check_resources()
        
        # If GPU memory usage is too high, use CPU instead
        if resources.get('gpu_memory_percent', 0) > self.gpu_threshold:
            logger.warning(f"GPU memory usage is high ({resources.get('gpu_memory_percent', 0):.1f}%). Using CPU instead.")
            return "cpu"
        
        return "cuda"