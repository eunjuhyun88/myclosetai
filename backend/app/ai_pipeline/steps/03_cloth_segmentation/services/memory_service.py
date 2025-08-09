#!/usr/bin/env python3
"""
🔥 MyCloset AI - Step 03: 의류 세그멘테이션 - Memory Service
=====================================================================

메모리 관리를 위한 전용 서비스

Author: MyCloset AI Team  
Date: 2025-08-01
Version: 1.0
"""

import logging
import gc
from typing import Dict, Any, Optional
import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)

class MemoryService:
    """메모리 관리 서비스"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """초기화"""
        self.config = config or {}
        self.logger = logging.getLogger(f"{__name__}.MemoryService")
        self.memory_stats = {}
        
    def cleanup_memory(self) -> bool:
        """메모리 정리"""
        try:
            # Python GC 정리
            gc.collect()
            
            # PyTorch 메모리 정리
            if TORCH_AVAILABLE:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    torch.mps.empty_cache()
            
            self.logger.info("✅ 메모리 정리 완료")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ 메모리 정리 실패: {e}")
            return False
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """메모리 사용량 가져오기"""
        try:
            memory_info = {}
            
            # Python 메모리 정보
            import psutil
            process = psutil.Process()
            memory_info['python_memory'] = {
                'rss': process.memory_info().rss / 1024 / 1024,  # MB
                'vms': process.memory_info().vms / 1024 / 1024,  # MB
            }
            
            # PyTorch 메모리 정보
            if TORCH_AVAILABLE:
                if torch.cuda.is_available():
                    memory_info['cuda_memory'] = {
                        'allocated': torch.cuda.memory_allocated() / 1024 / 1024,  # MB
                        'cached': torch.cuda.memory_reserved() / 1024 / 1024,  # MB
                    }
                elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    memory_info['mps_memory'] = {
                        'available': 'MPS memory info not available'
                    }
            
            return memory_info
            
        except Exception as e:
            self.logger.error(f"❌ 메모리 사용량 조회 실패: {e}")
            return {}
    
    def optimize_memory(self) -> bool:
        """메모리 최적화"""
        try:
            # 1. 불필요한 객체 정리
            gc.collect()
            
            # 2. PyTorch 메모리 최적화
            if TORCH_AVAILABLE:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    torch.mps.empty_cache()
            
            # 3. NumPy 메모리 최적화
            if hasattr(np, 'get_include'):
                # NumPy 메모리 최적화 (가능한 경우)
                pass
            
            self.logger.info("✅ 메모리 최적화 완료")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ 메모리 최적화 실패: {e}")
            return False
    
    def monitor_memory(self) -> Dict[str, Any]:
        """메모리 모니터링"""
        try:
            memory_stats = self.get_memory_usage()
            
            # 메모리 사용량 분석
            if 'python_memory' in memory_stats:
                rss_mb = memory_stats['python_memory']['rss']
                if rss_mb > 1000:  # 1GB 이상
                    self.logger.warning(f"⚠️ 높은 메모리 사용량: {rss_mb:.1f}MB")
            
            return memory_stats
            
        except Exception as e:
            self.logger.error(f"❌ 메모리 모니터링 실패: {e}")
            return {}
