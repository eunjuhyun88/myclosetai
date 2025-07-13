"""메모리 관리 유틸리티"""
import psutil
import torch
import logging

logger = logging.getLogger(__name__)

class GPUMemoryManager:
    def __init__(self, device="mps", memory_limit_gb=16.0):
        self.device = device
        self.memory_limit_gb = memory_limit_gb
    
    def clear_cache(self):
        """메모리 정리"""
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def check_memory_usage(self):
        """메모리 사용량 확인"""
        memory = psutil.virtual_memory()
        used_gb = memory.used / (1024**3)
        if used_gb > self.memory_limit_gb * 0.9:
            logger.warning(f"메모리 사용량 높음: {used_gb:.1f}GB")
            self.clear_cache()
