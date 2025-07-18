# backend/app/ai_pipeline/steps/base_step_mixin.py
"""
🔥 MyCloset AI - 완전 수정된 BaseStepMixin v2.0
✅ 모든 Step 클래스의 logger 속성 누락 문제 완전 해결
✅ ModelLoader 인터페이스 완벽 연동  
✅ 표준화된 초기화 패턴
✅ M3 Max 128GB 최적화
✅ 비동기 처리 완벽 지원
"""

import os
import gc
import time
import asyncio
import logging
import threading
import traceback
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Type
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor

# PyTorch import (안전)
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

# 이미지 처리 라이브러리
try:
    import cv2
    import numpy as np
    from PIL import Image
    CV_AVAILABLE = True
except ImportError:
    CV_AVAILABLE = False

# ==============================================
# 🔥 완전 수정된 BaseStepMixin
# ==============================================

class BaseStepMixin:
    """
    🔥 모든 Step 클래스가 상속받는 기본 Mixin
    ✅ logger 속성 누락 문제 완전 해결
    ✅ ModelLoader 인터페이스 안전한 연동
    ✅ 표준화된 초기화 패턴
    ✅ M3 Max 최적화 지원
    """
    
    def __init__(self, *args, **kwargs):
        """기본 Mixin 초기화 - 모든 Step 클래스에서 호출되어야 함"""
        # 🔥 logger 속성 누락 문제 해결 - 반드시 먼저 설정
        if not hasattr(self, 'logger'):
            class_name = self.__class__.__name__
            self.logger = logging.getLogger(f"pipeline.{class_name}")
            self.logger.info(f"🔧 {class_name} logger 초기화 완료")
        
        # 기본 속성들 초기화
        self.step_name = getattr(self, 'step_name', self.__class__.__name__)
        self.device = getattr(self, 'device', self._auto_detect_device())
        self.is_initialized = False
        self.model_interface = None
        self.performance_metrics = {}
        self.error_count = 0
        self.last_error = None
        
        # M3 Max 최적화 설정
        self._setup_m3_max_optimization()
        
        # ModelLoader 인터페이스 설정 (안전)
        self._setup_model_interface_safe()
        
        self.logger.info(f"✅ {self.step_name} BaseStepMixin 초기화 완료")
    
    def _auto_detect_device(self) -> str:
        """디바이스 자동 탐지 (M3 Max 최적화)"""
        if not TORCH_AVAILABLE:
            return "cpu"
            
        # M3 Max MPS 지원 확인
        if torch.backends.mps.is_available():
            return "mps"
        elif torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"
    
    def _setup_m3_max_optimization(self):
        """M3 Max 최적화 설정"""
        try:
            if self.device == "mps" and TORCH_AVAILABLE:
                # M3 Max 특화 설정
                os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
                os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
                
                # 메모리 최적화
                if hasattr(torch.backends.mps, 'empty_cache'):
                    torch.backends.mps.empty_cache()
                
                self.logger.info("🍎 M3 Max MPS 최적화 설정 완료")
        except Exception as e:
            self.logger.warning(f"⚠️ M3 Max 최적화 설정 실패: {e}")
    
    def _setup_model_interface_safe(self):
        """ModelLoader 인터페이스 안전한 설정"""
        try:
            # 순환 import 방지를 위한 늦은 import
            from ..utils.model_loader import get_global_model_loader
            
            model_loader = get_global_model_loader()
            if model_loader:
                self.model_interface = model_loader.create_step_interface(self.step_name)
                self.logger.info(f"🔗 {self.step_name} ModelLoader 인터페이스 연결 완료")
            else:
                self.logger.warning(f"⚠️ {self.step_name} 전역 ModelLoader를 찾을 수 없음")
                self.model_interface = None
                
        except Exception as e:
            self.logger.error(f"❌ {self.step_name} ModelLoader 인터페이스 설정 실패: {e}")
            self.model_interface = None
    
    async def initialize_step(self) -> bool:
        """Step 완전 초기화"""
        try:
            # 기본 초기화 확인
            if not hasattr(self, 'logger'):
                self.logger = logging.getLogger(f"pipeline.{self.__class__.__name__}")
            
            if not hasattr(self, 'device'):
                self.device = self._auto_detect_device()
            
            # ModelLoader 인터페이스 재설정 (필요시)
            if not hasattr(self, 'model_interface') or self.model_interface is None:
                self._setup_model_interface_safe()
            
            self.is_initialized = True
            self.logger.info(f"✅ {self.step_name} 완전 초기화 완료")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ {self.step_name} 초기화 실패: {e}")
            self.last_error = str(e)
            self.error_count += 1
            return False
    
    async def get_model(self, model_name: Optional[str] = None) -> Optional[Any]:
        """모델 로드 (Step에서 사용)"""
        try:
            if not self.model_interface:
                self.logger.warning(f"⚠️ {self.step_name} ModelLoader 인터페이스가 없습니다")
                return None
            
            if model_name:
                return await self.model_interface.get_model(model_name)
            else:
                # 권장 모델 자동 로드
                return await self.model_interface.get_recommended_model()
                
        except Exception as e:
            self.logger.error(f"❌ {self.step_name} 모델 로드 실패: {e}")
            self.last_error = str(e)
            self.error_count += 1
            return None
    
    def cleanup_models(self):
        """모델 정리"""
        try:
            if hasattr(self, 'model_interface') and self.model_interface:
                self.model_interface.unload_models()
                self.logger.info(f"🧹 {self.step_name} 모델 정리 완료")
        except Exception as e:
            self.logger.error(f"❌ {self.step_name} 모델 정리 실패: {e}")
    
    def get_step_info(self) -> Dict[str, Any]:
        """Step 상태 정보 반환"""
        return {
            "step_name": self.step_name,
            "device": self.device,
            "is_initialized": self.is_initialized,
            "has_model_interface": self.model_interface is not None,
            "error_count": self.error_count,
            "last_error": self.last_error,
            "performance_metrics": self.performance_metrics
        }
    
    def record_performance(self, operation: str, duration: float, success: bool = True):
        """성능 메트릭 기록"""
        if operation not in self.performance_metrics:
            self.performance_metrics[operation] = {
                "total_calls": 0,
                "success_calls": 0,
                "total_duration": 0.0,
                "avg_duration": 0.0,
                "last_duration": 0.0
            }
        
        metrics = self.performance_metrics[operation]
        metrics["total_calls"] += 1
        metrics["total_duration"] += duration
        metrics["last_duration"] = duration
        metrics["avg_duration"] = metrics["total_duration"] / metrics["total_calls"]
        
        if success:
            metrics["success_calls"] += 1
    
    async def process(self, *args, **kwargs) -> Dict[str, Any]:
        """🔧 기본 처리 메서드 (하위 클래스에서 오버라이드)"""
        try:
            start_time = time.time()
            self.logger.info(f"🔄 {self.step_name} 기본 처리 실행")
            
            # 초기화 확인
            if not self.is_initialized:
                await self.initialize_step()
            
            # 기본 처리 결과
            result = {
                'success': True,
                'step_name': self.step_name,
                'result': f'{self.step_name} 기본 처리 완료',
                'confidence': 0.5,
                'processing_time': time.time() - start_time,
                'metadata': {
                    'device': self.device,
                    'fallback': True,
                    'model_interface_available': self.model_interface is not None
                }
            }
            
            # 성능 기록
            self.record_performance("process", result['processing_time'], True)
            return result
            
        except Exception as e:
            duration = time.time() - start_time if 'start_time' in locals() else 0.0
            self.logger.error(f"❌ {self.step_name} 처리 실패: {e}")
            self.last_error = str(e)
            self.error_count += 1
            
            # 성능 기록 (실패)
            self.record_performance("process", duration, False)
            
            return {
                'success': False,
                'step_name': self.step_name,
                'error': str(e),
                'confidence': 0.0,
                'processing_time': duration
            }
    
    def __del__(self):
        """소멸자"""
        try:
            self.cleanup_models()
        except:
            pass

# ==============================================
# 🔥 Step별 특화 Mixin들
# ==============================================

class HumanParsingMixin(BaseStepMixin):
    """Human Parsing Step 특화 Mixin"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.step_number = 1
        self.step_type = "human_parsing"
        self.num_classes = 20
        self.output_format = "segmentation_mask"

class PoseEstimationMixin(BaseStepMixin):
    """Pose Estimation Step 특화 Mixin"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.step_number = 2
        self.step_type = "pose_estimation"
        self.num_keypoints = 18
        self.output_format = "keypoints_heatmap"

class ClothSegmentationMixin(BaseStepMixin):
    """Cloth Segmentation Step 특화 Mixin"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.step_number = 3
        self.step_type = "cloth_segmentation"
        self.binary_output = True
        self.output_format = "binary_mask"

class GeometricMatchingMixin(BaseStepMixin):
    """Geometric Matching Step 특화 Mixin"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.step_number = 4
        self.step_type = "geometric_matching"
        self.num_control_points = 25
        self.output_format = "transformation_matrix"

class ClothWarpingMixin(BaseStepMixin):
    """Cloth Warping Step 특화 Mixin"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.step_number = 5
        self.step_type = "cloth_warping"
        self.enable_physics = True
        self.output_format = "warped_cloth"

class VirtualFittingMixin(BaseStepMixin):
    """Virtual Fitting Step 특화 Mixin"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.step_number = 6
        self.step_type = "virtual_fitting"
        self.diffusion_steps = 50
        self.output_format = "rgb_image"

class PostProcessingMixin(BaseStepMixin):
    """Post Processing Step 특화 Mixin"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.step_number = 7
        self.step_type = "post_processing"
        self.upscale_factor = 2
        self.output_format = "enhanced_image"

class QualityAssessmentMixin(BaseStepMixin):
    """Quality Assessment Step 특화 Mixin"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.step_number = 8
        self.step_type = "quality_assessment"
        self.quality_threshold = 0.7
        self.output_format = "quality_scores"

# ==============================================
# 🔧 유틸리티 함수들 및 데코레이터
# ==============================================

def ensure_step_initialization(func):
    """Step 클래스 초기화 보장 데코레이터"""
    async def wrapper(self, *args, **kwargs):
        # logger 속성 확인 및 설정
        if not hasattr(self, 'logger'):
            self.logger = logging.getLogger(f"pipeline.{self.__class__.__name__}")
        
        # BaseStepMixin 초기화 확인
        if not hasattr(self, 'is_initialized') or not self.is_initialized:
            await self.initialize_step()
        
        return await func(self, *args, **kwargs)
    return wrapper

def safe_step_method(func):
    """Step 메서드 안전 실행 데코레이터"""
    async def wrapper(self, *args, **kwargs):
        try:
            # logger 확인
            if not hasattr(self, 'logger'):
                self.logger = logging.getLogger(f"pipeline.{self.__class__.__name__}")
            
            return await func(self, *args, **kwargs)
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.error(f"❌ {func.__name__} 실행 실패: {e}")
                if hasattr(self, 'error_count'):
                    self.error_count += 1
                if hasattr(self, 'last_error'):
                    self.last_error = str(e)
            
            # 기본 에러 응답 반환
            return {
                'success': False,
                'error': str(e),
                'step_name': getattr(self, 'step_name', self.__class__.__name__),
                'method_name': func.__name__
            }
    return wrapper

def performance_monitor(operation_name: str):
    """성능 모니터링 데코레이터"""
    def decorator(func):
        async def wrapper(self, *args, **kwargs):
            start_time = time.time()
            success = True
            try:
                result = await func(self, *args, **kwargs)
                return result
            except Exception as e:
                success = False
                raise e
            finally:
                duration = time.time() - start_time
                if hasattr(self, 'record_performance'):
                    self.record_performance(operation_name, duration, success)
        return wrapper
    return decorator

# ==============================================
# 🔥 모듈 익스포트
# ==============================================

__all__ = [
    # 기본 Mixin
    'BaseStepMixin',
    
    # Step별 특화 Mixin들
    'HumanParsingMixin',
    'PoseEstimationMixin', 
    'ClothSegmentationMixin',
    'GeometricMatchingMixin',
    'ClothWarpingMixin',
    'VirtualFittingMixin',
    'PostProcessingMixin',
    'QualityAssessmentMixin',
    
    # 유틸리티 데코레이터
    'ensure_step_initialization',
    'safe_step_method',
    'performance_monitor'
]

# 로거 설정
logger = logging.getLogger(__name__)
logger.info("✅ BaseStepMixin v2.0 로드 완료 - 모든 Step 클래스 logger 속성 누락 문제 해결")
logger.info("🔗 ModelLoader 인터페이스 완벽 연동")
logger.info("🍎 M3 Max 128GB 최적화 지원")