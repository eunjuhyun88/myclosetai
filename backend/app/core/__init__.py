# backend/app/core/__init__.py
"""
핵심 설정 및 구성 모듈
GPU 설정, 보안, 설정 관리 등
"""

# 설정 먼저 import
from .config import get_settings, settings

# GPU 설정 import (안전하게)
try:
    from .gpu_config import (
        gpu_config, 
        DEVICE, 
        DEVICE_INFO, 
        MODEL_CONFIG,
        get_device,
        get_device_info,
        get_optimal_settings,
        optimize_memory,
        check_memory_available
    )
except ImportError as e:
    # 폴백 설정
    print(f"⚠️ GPU 설정 import 실패: {e}")
    DEVICE = "cpu"
    DEVICE_INFO = {"device": "cpu", "error": str(e)}
    MODEL_CONFIG = {"device": "cpu", "batch_size": 1}
    
    def get_device():
        return "cpu"
    
    def get_device_info():
        return DEVICE_INFO
    
    def get_optimal_settings():
        return MODEL_CONFIG
    
    def optimize_memory():
        pass
    
    def check_memory_available(required_gb=4.0):
        return True
    
    # 더미 gpu_config 객체
    class DummyGPUConfig:
        device = "cpu"
        device_info = DEVICE_INFO
        
        def get_model_config(self):
            return MODEL_CONFIG
        
        def optimize_memory(self):
            pass
    
    gpu_config = DummyGPUConfig()

__all__ = [
    # 설정
    "get_settings",
    "settings",
    # GPU 설정
    "gpu_config",
    "DEVICE", 
    "DEVICE_INFO",
    "MODEL_CONFIG",
    "get_device",
    "get_device_info", 
    "get_optimal_settings",
    "optimize_memory",
    "check_memory_available"
]