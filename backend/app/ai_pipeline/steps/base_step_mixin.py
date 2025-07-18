# app/ai_pipeline/steps/base_step_mixin.py
"""
🔥 MyCloset AI - BaseStepMixin v2.2 - 초기화 문제 완전 해결
✅ object.__init__() 파라미터 문제 해결
✅ 다중 상속 안전한 처리 
✅ logger 속성 누락 문제 완전 해결
✅ _auto_detect_device() device 인자 오류 완전 해결
✅ ModelLoader 인터페이스 완벽 연동
✅ M3 Max 128GB 최적화
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
# 🔥 완전 수정된 BaseStepMixin v2.2 - 초기화 문제 해결
# ==============================================

class BaseStepMixin:
    """
    🔥 모든 Step 클래스가 상속받는 기본 Mixin - 초기화 문제 완전 해결
    ✅ object.__init__() 파라미터 문제 완전 해결
    ✅ logger 속성 누락 문제 완전 해결
    ✅ _auto_detect_device() 인자 없이 호출 가능
    ✅ ModelLoader 인터페이스 안전한 연동
    ✅ 표준화된 초기화 패턴
    ✅ M3 Max 최적화 지원
    """
    
    def __init__(self, *args, **kwargs):
        """
        🔥 기본 Mixin 초기화 - 다중 상속 안전 처리
        
        이 메서드는 모든 Step 클래스에서 호출되어야 함
        """
        # 🔥 다중 상속 시 안전한 super() 호출
        try:
            # kwargs에서 BaseStepMixin이 모르는 파라미터들 필터링
            base_kwargs = {}
            
            # 알려진 파라미터들만 전달
            known_params = {
                'device', 'quality_level', 'device_type', 'memory_gb', 
                'is_m3_max', 'optimization_enabled', 'batch_size'
            }
            
            # 다른 클래스들이 사용할 수 있는 파라미터들은 유지
            for key, value in kwargs.items():
                if key not in known_params:
                    base_kwargs[key] = value
            
            # 안전한 super() 호출 - 파라미터 없이
            super().__init__()
            
        except TypeError as e:
            # super().__init__()이 파라미터를 받지 않는 경우 (object 클래스)
            # 이 경우는 정상이므로 무시
            pass
        
        # 🔥 logger 속성 누락 문제 해결 - 반드시 먼저 설정
        if not hasattr(self, 'logger'):
            class_name = self.__class__.__name__
            self.logger = logging.getLogger(f"pipeline.{class_name}")
            self.logger.info(f"🔧 {class_name} logger 초기화 완료")
        
        # 기본 속성들 초기화 - device 인자 없이 안전하게 호출
        self.step_name = getattr(self, 'step_name', self.__class__.__name__)
        
        # 🔥 device 설정 - Step 클래스별 안전한 처리
        if not hasattr(self, 'device'):
            # kwargs에서 device 파라미터 확인
            device_from_kwargs = kwargs.get('device', kwargs.get('preferred_device', 'auto'))
            try:
                self.device = self._auto_detect_device(device_from_kwargs)
            except TypeError:
                # Step 클래스에서 인자를 요구하는 경우 기본값 전달
                self.device = "mps" if TORCH_AVAILABLE and torch.backends.mps.is_available() else "cpu"
                self.logger.info(f"🔧 기본 디바이스 설정: {self.device}")
        
        # 🔥 기본 속성들 안전하게 초기화
        if not hasattr(self, 'is_initialized'):
            self.is_initialized = False
        if not hasattr(self, 'model_interface'):
            self.model_interface = None
        if not hasattr(self, 'performance_metrics'):
            self.performance_metrics = {}
        if not hasattr(self, 'error_count'):
            self.error_count = 0
        if not hasattr(self, 'last_error'):
            self.last_error = None
        
        # M3 Max 최적화 설정
        self._setup_m3_max_optimization()
        
        # ModelLoader 인터페이스 설정 (안전)
        self._setup_model_interface_safe()
        
        self.logger.info(f"✅ {self.step_name} BaseStepMixin v2.2 초기화 완료")
    
    def _auto_detect_device(self, preferred_device: Optional[str] = None, device: Optional[str] = None) -> str:
        """🔥 디바이스 자동 탐지 - 모든 Step 클래스와 호환 (M3 Max 최적화)"""
        try:
            # device 또는 preferred_device 중 하나라도 주어진 경우 우선 사용
            target_device = device or preferred_device
            if target_device and target_device != "auto":
                if hasattr(self, 'logger'):
                    self.logger.info(f"🎯 지정된 디바이스 사용: {target_device}")
                return target_device
            
            if not TORCH_AVAILABLE:
                if hasattr(self, 'logger'):
                    self.logger.warning("⚠️ PyTorch 없음, CPU 사용")
                return "cpu"
                
            # M3 Max MPS 지원 확인 (최우선)
            if torch.backends.mps.is_available():
                if hasattr(self, 'logger'):
                    self.logger.info("🍎 M3 Max MPS 디바이스 감지")
                return "mps"
            elif torch.cuda.is_available():
                if hasattr(self, 'logger'):
                    self.logger.info("🔥 CUDA 디바이스 감지")
                return "cuda"
            else:
                if hasattr(self, 'logger'):
                    self.logger.info("💻 CPU 디바이스 사용")
                return "cpu"
                
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.error(f"❌ 디바이스 탐지 실패: {e}, CPU 사용")
            return "cpu"
    
    def _setup_m3_max_optimization(self):
        """M3 Max 최적화 설정"""
        try:
            if getattr(self, 'device', 'cpu') == "mps" and TORCH_AVAILABLE:
                # M3 Max 특화 설정
                os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
                os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
                
                # 메모리 최적화
                if hasattr(torch.backends.mps, 'empty_cache'):
                    torch.backends.mps.empty_cache()
                
                if hasattr(self, 'logger'):
                    self.logger.info("🍎 M3 Max MPS 최적화 설정 완료")
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.warning(f"⚠️ M3 Max 최적화 설정 실패: {e}")
    
    def _setup_model_interface_safe(self):
        """ModelLoader 인터페이스 안전한 설정"""
        try:
            # 순환 import 방지를 위한 늦은 import
            from ..utils.model_loader import get_global_model_loader
            
            model_loader = get_global_model_loader()
            if model_loader:
                self.model_interface = model_loader.create_step_interface(self.step_name)
                if hasattr(self, 'logger'):
                    self.logger.info(f"🔗 {self.step_name} ModelLoader 인터페이스 연결 완료")
            else:
                if hasattr(self, 'logger'):
                    self.logger.warning(f"⚠️ {self.step_name} 전역 ModelLoader를 찾을 수 없음")
                self.model_interface = None
                
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.error(f"❌ {self.step_name} ModelLoader 인터페이스 설정 실패: {e}")
            self.model_interface = None
    
    async def initialize_step(self) -> bool:
        """Step 완전 초기화"""
        try:
            # 기본 초기화 확인
            if not hasattr(self, 'logger'):
                self.logger = logging.getLogger(f"pipeline.{self.__class__.__name__}")
            
            if not hasattr(self, 'device'):
                self.device = self._auto_detect_device()  # 🔥 인자 없이 호출
            
            # ModelLoader 인터페이스 재설정 (필요시)
            if not hasattr(self, 'model_interface') or self.model_interface is None:
                self._setup_model_interface_safe()
            
            self.is_initialized = True
            self.logger.info(f"✅ {self.step_name} 완전 초기화 완료")
            return True
            
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.error(f"❌ {self.step_name} 초기화 실패: {e}")
            self.last_error = str(e)
            self.error_count += 1
            return False
    
    async def get_model(self, model_name: Optional[str] = None) -> Optional[Any]:
        """모델 로드 (Step에서 사용)"""
        try:
            if not self.model_interface:
                if hasattr(self, 'logger'):
                    self.logger.warning(f"⚠️ {self.step_name} ModelLoader 인터페이스가 없습니다")
                return None
            
            if model_name:
                return await self.model_interface.get_model(model_name)
            else:
                # 권장 모델 자동 로드
                return await self.model_interface.get_recommended_model()
                
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.error(f"❌ {self.step_name} 모델 로드 실패: {e}")
            self.last_error = str(e)
            self.error_count += 1
            return None
    
    def cleanup_models(self):
        """모델 정리"""
        try:
            if hasattr(self, 'model_interface') and self.model_interface:
                self.model_interface.unload_models()
                if hasattr(self, 'logger'):
                    self.logger.info(f"🧹 {self.step_name} 모델 정리 완료")
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.error(f"❌ {self.step_name} 모델 정리 실패: {e}")
    
    def get_step_info(self) -> Dict[str, Any]:
        """Step 상태 정보 반환"""
        return {
            "step_name": getattr(self, 'step_name', self.__class__.__name__),
            "device": getattr(self, 'device', 'unknown'),
            "is_initialized": getattr(self, 'is_initialized', False),
            "has_model_interface": getattr(self, 'model_interface', None) is not None,
            "error_count": getattr(self, 'error_count', 0),
            "last_error": getattr(self, 'last_error', None),
            "performance_metrics": getattr(self, 'performance_metrics', {})
        }
    
    def record_performance(self, operation: str, duration: float, success: bool = True):
        """성능 메트릭 기록"""
        if not hasattr(self, 'performance_metrics'):
            self.performance_metrics = {}
            
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
            if hasattr(self, 'logger'):
                self.logger.info(f"🔄 {self.step_name} 기본 처리 실행")
            
            # 초기화 확인
            if not getattr(self, 'is_initialized', False):
                await self.initialize_step()
            
            # 기본 처리 결과
            result = {
                'success': True,
                'step_name': getattr(self, 'step_name', self.__class__.__name__),
                'result': f'{getattr(self, "step_name", self.__class__.__name__)} 기본 처리 완료',
                'confidence': 0.5,
                'processing_time': time.time() - start_time,
                'metadata': {
                    'device': getattr(self, 'device', 'unknown'),
                    'fallback': True,
                    'model_interface_available': getattr(self, 'model_interface', None) is not None
                }
            }
            
            # 성능 기록
            self.record_performance("process", result['processing_time'], True)
            return result
            
        except Exception as e:
            duration = time.time() - start_time if 'start_time' in locals() else 0.0
            if hasattr(self, 'logger'):
                self.logger.error(f"❌ {getattr(self, 'step_name', self.__class__.__name__)} 처리 실패: {e}")
            
            if hasattr(self, 'last_error'):
                self.last_error = str(e)
            if hasattr(self, 'error_count'):
                self.error_count += 1
            
            # 성능 기록 (실패)
            self.record_performance("process", duration, False)
            
            return {
                'success': False,
                'step_name': getattr(self, 'step_name', self.__class__.__name__),
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
# 🔥 Step별 특화 Mixin들 - 안전한 초기화
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
logger.info("✅ BaseStepMixin v2.2 로드 완료 - 초기화 문제 완전 해결")
logger.info("🔗 ModelLoader 인터페이스 완벽 연동")
logger.info("🍎 M3 Max 128GB 최적화 지원")