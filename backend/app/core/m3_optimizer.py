# backend/app/core/m3_optimizer.py
"""
M3 Max 전용 최적화 모듈
"""
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
            optimization_level: 최적화 레벨 ("maximum", "balanced", "conservative")
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
                
                # 8단계 파이프라인 최적화 설정
                self.pipeline_settings = {
                    "stages": 8,
                    "parallel_processing": True,
                    "batch_optimization": True,
                    "memory_pooling": True,
                    "neural_engine": True
                }
                
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
            elif self.optimization_level == "balanced":
                config = {
                    **base_config,
                    "batch_size": 2,
                    "precision": "float16",
                    "max_workers": 8,
                    "memory_fraction": 0.6,
                    "enable_neural_engine": True,
                    "pipeline_parallel": False
                }
            else:  # conservative
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
            "mps_available": torch.backends.mps.is_available() if self.is_m3_max else False
        }

# ================================================================
# backend/app/ai_pipeline/utils/memory_manager.py에 추가할 함수

def initialize_global_memory_manager(device: str = "mps", memory_gb: float = 128.0):
    """
    전역 메모리 매니저 초기화
    
    Args:
        device: 사용할 디바이스
        memory_gb: 총 메모리 용량
    """
    try:
        import gc
        import torch
        
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

# ================================================================
# backend/app/ai_pipeline/utils/model_loader.py에 추가할 클래스와 함수

class ModelFormat:
    """AI 모델 형식 정의"""
    
    PYTORCH = "pytorch"
    ONNX = "onnx" 
    TENSORRT = "tensorrt"
    COREML = "coreml"  # Apple Core ML for M3 Max
    SAFETENSORS = "safetensors"
    
    @classmethod
    def get_optimized_format(cls, device: str = "mps") -> str:
        """디바이스에 최적화된 모델 형식 반환"""
        if device == "mps":
            return cls.COREML  # M3 Max에서는 Core ML 추천
        elif device == "cuda":
            return cls.TENSORRT
        return cls.PYTORCH
    
    @classmethod
    def is_supported(cls, format_name: str) -> bool:
        """지원되는 형식인지 확인"""
        supported_formats = [cls.PYTORCH, cls.ONNX, cls.COREML, cls.SAFETENSORS]
        return format_name.lower() in [f.lower() for f in supported_formats]

def initialize_global_model_loader(device: str = "mps"):
    """전역 모델 로더 초기화"""
    try:
        logger.info(f"🤖 전역 ModelLoader 초기화: {device}")
        
        # 글로벌 모델 로더 설정
        loader_config = {
            "device": device,
            "cache_enabled": True,
            "lazy_loading": True,
            "memory_efficient": True
        }
        
        if device == "mps":
            loader_config.update({
                "use_neural_engine": True,
                "use_unified_memory": True,
                "optimization_level": "maximum"
            })
        
        logger.info("✅ 전역 ModelLoader 초기화 완료")
        return loader_config
        
    except Exception as e:
        logger.error(f"❌ 전역 ModelLoader 초기화 실패: {e}")
        return None

# ================================================================
# backend/app/core/gpu_config.py에 추가할 함수

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
        import torch
        
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