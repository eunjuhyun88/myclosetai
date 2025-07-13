"""
GPU 메모리 매니저 - M3 Max 최적화
"""

import torch
import psutil
import logging
from typing import Dict, Optional

class GPUMemoryManager:
    """GPU 메모리 관리 클래스"""
    
    def __init__(self, device: str = "mps", memory_limit_gb: float = 16.0):
        self.device = device
        self.memory_limit_gb = memory_limit_gb
        self.logger = logging.getLogger(__name__)
        
    def clear_cache(self):
        """메모리 캐시 정리"""
        if self.device == "mps":
            try:
                torch.mps.empty_cache()
            except:
                pass
        elif self.device == "cuda":
            torch.cuda.empty_cache()
    
    def check_memory_usage(self) -> Dict[str, float]:
        """메모리 사용량 확인"""
        memory_info = {}
        
        # 시스템 메모리
        memory_info["system_memory_gb"] = psutil.virtual_memory().used / (1024**3)
        
        # GPU 메모리 (가능한 경우)
        if self.device == "mps":
            try:
                memory_info["mps_memory_gb"] = torch.mps.current_allocated_memory() / (1024**3)
            except:
                memory_info["mps_memory_gb"] = 0.0
        
        return memory_info
