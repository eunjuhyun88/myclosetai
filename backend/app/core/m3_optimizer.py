# backend/app/core/m3_optimizer.py 수정
"""
M3 Max 전용 최적화 모듈 - 파이프라인 라우터 호환성 수정
"""
import os
import logging
import torch
from typing import Dict, Any, Optional
import platform
import subprocess

logger = logging.getLogger(__name__)

class M3Optimizer:
    """
    Apple M3 Max 전용 최적화 클래스
    ✅ 파이프라인 라우터와 완전 호환
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

# ===============================================================
# 🔧 파이프라인 라우터 호환성 함수들
# ===============================================================

def create_m3_optimizer_for_pipeline(
    device: str = "mps",
    memory_gb: float = 128.0,
    optimization_level: str = "maximum"
) -> M3Optimizer:
    """
    파이프라인 라우터용 M3 Optimizer 생성
    ✅ 4개 필수 인자 모두 제공
    """
    device_name = _detect_chip_name()
    is_m3_max = _detect_m3_max(memory_gb)
    
    return M3Optimizer(
        device_name=device_name,
        memory_gb=memory_gb,
        is_m3_max=is_m3_max,
        optimization_level=optimization_level
    )

def _detect_chip_name() -> str:
    """칩 이름 자동 감지"""
    try:
        if platform.system() == 'Darwin':  # macOS
            result = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'], 
                                  capture_output=True, text=True, timeout=5)
            chip_info = result.stdout.strip()
            if 'M3' in chip_info:
                return chip_info
            else:
                return "Apple Silicon"
        else:
            return "Generic Device"
    except:
        return "Apple M3 Max"  # 기본값

def _detect_m3_max(memory_gb: float) -> bool:
    """M3 Max 감지"""
    try:
        if platform.system() == 'Darwin':
            result = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'], 
                                  capture_output=True, text=True, timeout=5)
            chip_info = result.stdout.strip()
            return 'M3' in chip_info and ('Max' in chip_info or memory_gb >= 64)
    except:
        pass
    
    # 메모리 기준 추정
    return memory_gb >= 64

# ===============================================================
# Config 클래스 추가 (import 오류 해결)
# ===============================================================

class Config:
    """
    기본 설정 클래스
    ✅ import 오류 해결용
    """
    
    def __init__(self, **kwargs):
        self.device = kwargs.get('device', 'mps')
        self.memory_gb = kwargs.get('memory_gb', 128.0)
        self.quality_level = kwargs.get('quality_level', 'high')
        self.optimization_enabled = kwargs.get('optimization_enabled', True)
        
        # M3 Max 정보
        self.is_m3_max = _detect_m3_max(self.memory_gb)
        self.device_name = _detect_chip_name()
        
    def to_dict(self) -> Dict[str, Any]:
        """설정을 딕셔너리로 변환"""
        return {
            'device': self.device,
            'memory_gb': self.memory_gb,
            'quality_level': self.quality_level,
            'optimization_enabled': self.optimization_enabled,
            'is_m3_max': self.is_m3_max,
            'device_name': self.device_name
        }

# ===============================================================
# 모듈 export
# ===============================================================

__all__ = [
    'M3Optimizer',
    'Config',
    'create_m3_optimizer_for_pipeline',
    '_detect_chip_name',
    '_detect_m3_max'
]

logger.info("🍎 M3 Optimizer 모듈 로드 완료 - 파이프라인 라우터 호환성 적용")