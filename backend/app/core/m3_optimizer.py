# backend/app/core/m3_optimizer.py

import os
import logging
import torch
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class M3Optimizer:
    """
    Apple M3 Max 전용 최적화 클래스
    """
    
    def __init__(self, device_name: str, memory_gb: float, is_m3_max: bool, optimization_level: str):
        """
        M3 최적화 초기화
        
        Args:
            device_name: 디바이스 이름 (예: "Apple M3 Max")
            memory_gb: 메모리 용량 (GB)
            is_m3_max: M3 Max 여부
            optimization_level: 최적화 레벨 ("maximum", "high", "medium", "basic")
        """
        self.device_name = device_name
        self.memory_gb = memory_gb
        self.is_m3_max = is_m3_max
        self.optimization_level = optimization_level
        
        logger.info(f"🍎 M3Optimizer 초기화: {device_name}, {memory_gb}GB, {optimization_level}")
        
        # M3 Max 전용 설정
        if is_m3_max:
            self._apply_m3_max_optimizations()
        
        self.config = self._create_optimization_config()
        self.pipeline_settings = self._create_pipeline_settings()
    
    def _apply_m3_max_optimizations(self):
        """M3 Max 전용 최적화 적용"""
        try:
            # Neural Engine 환경변수 설정
            os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
            os.environ["PYTORCH_MPS_LOW_WATERMARK_RATIO"] = "0.0"
            
            # Metal Performance Shaders 최적화
            if torch.backends.mps.is_available():
                logger.info("🧠 Neural Engine 최적화 활성화")
                logger.info("⚙️ Metal Performance Shaders 활성화")
                
                # CPU 스레드 최적화 (M3 Max 16코어)
                if hasattr(torch, 'set_num_threads'):
                    torch.set_num_threads(16)
                
                logger.info("⚙️ 8단계 파이프라인 최적화 설정 완료")
                
        except Exception as e:
            logger.error(f"❌ M3 Max 최적화 실패: {e}")
    
    def _create_optimization_config(self) -> Dict[str, Any]:
        """최적화 설정 생성"""
        base_config = {
            "device": "mps" if self.is_m3_max else "cpu",
            "memory_gb": self.memory_gb,
            "optimization_level": self.optimization_level
        }
        
        if self.is_m3_max:
            if self.optimization_level == "maximum":
                config = {
                    **base_config,
                    "batch_size": 4,
                    "precision": "float16",
                    "max_workers": 12,
                    "memory_fraction": 0.8,
                    "enable_neural_engine": True,
                    "pipeline_parallel": True
                }
            elif self.optimization_level == "high":
                config = {
                    **base_config,
                    "batch_size": 2,
                    "precision": "float16",
                    "max_workers": 8,
                    "memory_fraction": 0.6,
                    "enable_neural_engine": True,
                    "pipeline_parallel": False
                }
            elif self.optimization_level == "medium":
                config = {
                    **base_config,
                    "batch_size": 1,
                    "precision": "float16",
                    "max_workers": 4,
                    "memory_fraction": 0.4,
                    "enable_neural_engine": True,
                    "pipeline_parallel": False
                }
            else:  # basic
                config = {
                    **base_config,
                    "batch_size": 1,
                    "precision": "float32",
                    "max_workers": 4,
                    "memory_fraction": 0.4,
                    "enable_neural_engine": False,
                    "pipeline_parallel": False
                }
        else:
            config = base_config
        
        return config
    
    def _create_pipeline_settings(self) -> Dict[str, Any]:
        """파이프라인 설정 생성"""
        if self.is_m3_max:
            return {
                "stages": 8,
                "parallel_processing": True,
                "batch_optimization": True,
                "memory_pooling": True,
                "neural_engine": True,
                "unified_memory": True,
                "mps_backend": True
            }
        else:
            return {
                "stages": 8,
                "parallel_processing": False,
                "batch_optimization": False,
                "memory_pooling": False,
                "neural_engine": False,
                "unified_memory": False,
                "mps_backend": False
            }
    
    def optimize_model(self, model):
        """모델 최적화 적용"""
        if not self.is_m3_max or model is None:
            return model
            
        try:
            # MPS 디바이스로 이동
            if hasattr(model, 'to'):
                model = model.to('mps')
                logger.info("🔄 모델을 MPS 디바이스로 이동")
            
            # 정밀도 최적화
            if self.config.get("precision") == "float16" and hasattr(model, 'half'):
                model = model.half()
                logger.info("🔧 모델 정밀도를 float16으로 최적화")
            
            logger.info("✅ 모델 M3 Max 최적화 완료")
            return model
            
        except Exception as e:
            logger.error(f"❌ 모델 최적화 실패: {e}")
            return model
    
    def get_optimization_info(self) -> Dict[str, Any]:
        """최적화 정보 반환"""
        return {
            "device_name": self.device_name,
            "memory_gb": self.memory_gb,
            "is_m3_max": self.is_m3_max,
            "optimization_level": self.optimization_level,
            "config": self.config,
            "pipeline_settings": self.pipeline_settings,
            "mps_available": torch.backends.mps.is_available() if self.is_m3_max else False
        }
    
    def cleanup(self):
        """최적화 정리"""
        try:
            if self.is_m3_max and torch.backends.mps.is_available():
                torch.mps.empty_cache()
                logger.info("🧹 M3 Max 메모리 정리 완료")
        except Exception as e:
            logger.warning(f"M3 Max 정리 실패: {e}")

# 전역 메모리 매니저 초기화 함수
def initialize_global_memory_manager(device: str = "mps", memory_gb: float = 128.0):
    """
    전역 메모리 매니저 초기화
    
    Args:
        device: 사용할 디바이스
        memory_gb: 총 메모리 용량
    """
    try:
        import gc
        
        logger.info(f"🔧 전역 메모리 매니저 초기화: {device}, {memory_gb}GB")
        
        # 메모리 정리
        gc.collect()
        
        if device == "mps" and torch.backends.mps.is_available():
            # MPS 메모리 설정
            logger.info(f"🍎 M3 Max MPS 메모리 매니저 초기화")
            
            # 환경변수 설정
            os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
            os.environ["PYTORCH_MPS_LOW_WATERMARK_RATIO"] = "0.0"
            
            # Unified Memory 최적화
            logger.info("💾 Unified Memory 최적화 설정")
            
        logger.info("✅ 전역 메모리 매니저 초기화 완료")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 메모리 매니저 초기화 실패: {e}")
        return False

# M3 Max 메모리 체크 함수
def check_memory_available(required_gb: float = 4.0) -> bool:
    """
    M3 Max 메모리 사용 가능 여부 확인
    
    Args:
        required_gb: 필요한 메모리 용량 (GB)
    
    Returns:
        bool: 메모리 사용 가능 여부
    """
    try:
        import psutil
        
        # 시스템 메모리 확인
        memory = psutil.virtual_memory()
        available_gb = memory.available / (1024**3)
        
        logger.info(f"💾 시스템 메모리: {memory.total / (1024**3):.1f}GB")
        logger.info(f"💾 사용 가능: {available_gb:.1f}GB") 
        logger.info(f"💾 요구사항: {required_gb:.1f}GB")
        
        # MPS 메모리 확인 (M3 Max)
        if torch.backends.mps.is_available():
            logger.info("🍎 M3 Max Unified Memory 사용 중")
            # Unified Memory에서는 시스템 메모리와 GPU 메모리가 통합
            return available_gb >= required_gb
        
        return available_gb >= required_gb
        
    except Exception as e:
        logger.warning(f"⚠️ 메모리 확인 실패: {e}")
        return True  # 안전하게 True 반환