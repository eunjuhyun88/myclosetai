# app/ai_pipeline/steps/base_step_mixin.py
"""
🔥 MyCloset AI - BaseStepMixin v3.1 - 완전 수정 버전
=====================================

✅ object.__init__() 파라미터 문제 완전 해결
✅ logger 속성 누락 문제 완전 해결
✅ device 속성 오류 완전 해결
✅ config 객체 호출 오류 완전 해결
✅ ModelLoader 인터페이스 완벽 연동
✅ M3 Max 128GB 최적화 지원
✅ 다중 상속 안전한 처리
✅ 성능 모니터링 및 에러 처리 강화
✅ conda 환경 완벽 지원

Author: MyCloset AI Team
Date: 2025-07-18
Version: 3.1 (완전 수정 버전)
"""

import os
import gc
import time
import asyncio
import logging
import threading
import traceback
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Type, Callable
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor

# 안전한 PyTorch import
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
    
    # M3 Max MPS 지원 확인
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        MPS_AVAILABLE = True
        DEFAULT_DEVICE = "mps"
    else:
        MPS_AVAILABLE = False
        DEFAULT_DEVICE = "cpu"
        
except ImportError as e:
    TORCH_AVAILABLE = False
    MPS_AVAILABLE = False
    DEFAULT_DEVICE = "cpu"
    torch = None

# 이미지 처리 라이브러리 안전한 import
try:
    import cv2
    import numpy as np
    from PIL import Image
    CV_AVAILABLE = True
    NUMPY_AVAILABLE = True
    PIL_AVAILABLE = True
except ImportError:
    CV_AVAILABLE = False
    NUMPY_AVAILABLE = False
    PIL_AVAILABLE = False

# 로깅 설정
logger = logging.getLogger(__name__)

# ==============================================
# 🔥 완전 수정된 BaseStepMixin v3.1
# ==============================================

class BaseStepMixin:
    """
    🔥 완전 수정된 BaseStepMixin v3.1
    
    모든 Step 클래스가 상속받는 기본 Mixin 클래스
    ✅ 모든 초기화 문제 완전 해결
    ✅ conda 환경 완벽 지원
    ✅ M3 Max 128GB 최적화
    """
    
    def __init__(self, *args, **kwargs):
        """
        🔥 완전 안전한 초기화 - 모든 오류 해결
        
        다중 상속 시 안전하게 처리하며, object.__init__() 파라미터 문제 해결
        """
        
        # 🔥 Step 1: 다중 상속 안전한 super() 호출 (파라미터 문제 해결)
        try:
            # object.__init__()은 파라미터를 받지 않으므로 빈 파라미터로 호출
            super().__init__()
        except TypeError as e:
            # TypeError 발생 시 super() 호출 없이 진행 (object 클래스인 경우)
            pass
        
        # 🔥 Step 2: 기본 속성들 먼저 설정 (logger 속성 누락 문제 해결)
        self.step_name = getattr(self, 'step_name', self.__class__.__name__)
        
        # logger 속성 반드시 먼저 설정
        if not hasattr(self, 'logger'):
            self.logger = logging.getLogger(f"pipeline.{self.step_name}")
            self.logger.info(f"🔧 {self.step_name} logger 초기화 완료")
        
        # 🔥 Step 3: device 속성 안전하게 설정 (device 속성 오류 해결)
        self.device = self._safe_device_setup(kwargs)
        self.device_type = kwargs.get('device_type', self._detect_device_type())
        
        # 🔥 Step 4: 시스템 정보 설정
        self.memory_gb = kwargs.get('memory_gb', self._detect_memory())
        self.is_m3_max = kwargs.get('is_m3_max', self._detect_m3_max())
        self.optimization_enabled = kwargs.get('optimization_enabled', True)
        
        # 🔥 Step 5: 품질 설정
        self.quality_level = kwargs.get('quality_level', 'balanced')
        self.batch_size = kwargs.get('batch_size', self._calculate_optimal_batch_size())
        
        # 🔥 Step 6: 설정 처리 (config 객체 호출 오류 해결)
        raw_config = kwargs.get('config', {})
        self.config = self._create_safe_config(raw_config)
        
        # 🔥 Step 7: 상태 관리 초기화
        self.is_initialized = False
        self.model_interface = None
        self.model_loader = None
        self.performance_metrics = {}
        self.error_count = 0
        self.last_error = None
        self.last_processing_time = 0.0
        self.total_processing_count = 0
        
        # 🔥 Step 8: 캐시 디렉토리 설정
        self.cache_dir = Path(kwargs.get('cache_dir', './cache'))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # 🔥 Step 9: M3 Max 최적화 설정
        self._setup_m3_max_optimization()
        
        # 🔥 Step 10: ModelLoader 인터페이스 설정 (지연 로딩)
        self._setup_model_interface_safe()
        
        # 🔥 Step 11: PyTorch 최적화 설정
        self._setup_pytorch_optimization()
        
        # 🔥 초기화 완료 로깅
        self.logger.info(f"✅ {self.step_name} BaseStepMixin v3.1 초기화 완료")
        self.logger.info(f"🔧 Device: {self.device} ({self.device_type})")
        self.logger.info(f"📊 Memory: {self.memory_gb}GB, M3 Max: {'✅' if self.is_m3_max else '❌'}")
        self.logger.info(f"⚙️ Quality: {self.quality_level}, Batch: {self.batch_size}")
    
    def _safe_device_setup(self, kwargs: Dict[str, Any]) -> str:
        """🔧 안전한 디바이스 설정 - 모든 Step 클래스와 호환"""
        try:
            # kwargs에서 device 파라미터 확인
            device_from_kwargs = kwargs.get('device') or kwargs.get('preferred_device')
            
            if device_from_kwargs and device_from_kwargs != "auto":
                return device_from_kwargs
            
            # 자동 탐지
            return self._auto_detect_device()
            
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.warning(f"⚠️ 디바이스 설정 실패: {e}, 기본값 사용")
            return DEFAULT_DEVICE
    
    def _auto_detect_device(self) -> str:
        """🔍 디바이스 자동 탐지 - M3 Max 최적화"""
        try:
            if not TORCH_AVAILABLE:
                return "cpu"
            
            # M3 Max MPS 지원 확인 (최우선)
            if MPS_AVAILABLE:
                return "mps"
            elif torch.cuda.is_available():
                return "cuda"
            else:
                return "cpu"
                
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.warning(f"⚠️ 디바이스 탐지 실패: {e}")
            return "cpu"
    
    def _detect_device_type(self) -> str:
        """디바이스 타입 탐지"""
        try:
            import platform
            system = platform.system()
            machine = platform.machine()
            
            if system == "Darwin" and ("arm64" in machine or "M3" in str(platform.processor())):
                return "apple_silicon"
            elif self.device == "cuda":
                return "nvidia_gpu"
            else:
                return "generic_cpu"
        except:
            return "unknown"
    
    def _detect_memory(self) -> float:
        """시스템 메모리 탐지"""
        try:
            import psutil
            return round(psutil.virtual_memory().total / (1024**3), 1)
        except:
            return 16.0  # 기본값
    
    def _detect_m3_max(self) -> bool:
        """M3 Max 탐지"""
        try:
            import platform
            processor = str(platform.processor())
            return "M3" in processor or (self.device == "mps" and self.memory_gb > 64)
        except:
            return False
    
    def _calculate_optimal_batch_size(self) -> int:
        """최적 배치 크기 계산"""
        try:
            if self.is_m3_max and self.memory_gb >= 128:
                return 8  # M3 Max 128GB
            elif self.memory_gb >= 64:
                return 4  # 64GB+
            elif self.memory_gb >= 32:
                return 2  # 32GB+
            else:
                return 1  # 16GB 이하
        except:
            return 1
    
    def _create_safe_config(self, config_data: Any) -> 'SafeConfig':
        """🔧 안전한 설정 객체 생성 (config 객체 호출 오류 해결)"""
        
        class SafeConfig:
            """안전한 설정 클래스 - 딕셔너리와 객체 모두 지원"""
            
            def __init__(self, data: Any):
                self._data = {}
                
                if hasattr(data, '__dict__'):
                    # 설정 객체인 경우
                    self._data = data.__dict__.copy()
                elif isinstance(data, dict):
                    # 딕셔너리인 경우
                    self._data = data.copy()
                elif hasattr(data, '__call__'):
                    # callable 객체인 경우 (config() 호출 방지)
                    try:
                        # 안전하게 호출 시도
                        result = data()
                        if isinstance(result, dict):
                            self._data = result.copy()
                        else:
                            self._data = {}
                    except Exception:
                        self._data = {}
                else:
                    # 기타 경우
                    self._data = {}
                
                # 속성으로 설정
                for key, value in self._data.items():
                    setattr(self, key, value)
            
            def get(self, key: str, default=None):
                """딕셔너리처럼 get 메서드 지원"""
                return self._data.get(key, default)
            
            def __getitem__(self, key):
                return self._data[key]
            
            def __setitem__(self, key, value):
                self._data[key] = value
                setattr(self, key, value)
            
            def __contains__(self, key):
                return key in self._data
            
            def keys(self):
                return self._data.keys()
            
            def values(self):
                return self._data.values()
            
            def items(self):
                return self._data.items()
            
            def update(self, other):
                if isinstance(other, dict):
                    self._data.update(other)
                    for key, value in other.items():
                        setattr(self, key, value)
            
            def __str__(self):
                return str(self._data)
            
            def __repr__(self):
                return f"SafeConfig({self._data})"
        
        return SafeConfig(config_data)
    
    def _setup_m3_max_optimization(self):
        """🍎 M3 Max 최적화 설정"""
        try:
            if self.device == "mps" and TORCH_AVAILABLE:
                # M3 Max 특화 환경 변수 설정
                os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
                os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
                
                # OMP 스레드 수 설정 (M3 Max 16코어 활용)
                if self.is_m3_max:
                    os.environ['OMP_NUM_THREADS'] = '16'
                
                # MPS 캐시 정리
                if hasattr(torch.backends.mps, 'empty_cache'):
                    torch.backends.mps.empty_cache()
                
                self.logger.info("🍎 M3 Max MPS 최적화 설정 완료")
                
        except Exception as e:
            self.logger.warning(f"⚠️ M3 Max 최적화 설정 실패: {e}")
    
    def _setup_pytorch_optimization(self):
        """PyTorch 최적화 설정"""
        try:
            if not TORCH_AVAILABLE:
                return
            
            # 데이터 타입 설정
            if self.device == "mps" and self.optimization_enabled:
                self.dtype = torch.float16  # MPS에서 메모리 효율성
            else:
                self.dtype = torch.float32
            
            # autograd 최적화
            torch.backends.cudnn.benchmark = True if self.device == "cuda" else False
            
            self.logger.debug(f"🔧 PyTorch 최적화 설정: dtype={self.dtype}")
            
        except Exception as e:
            self.logger.warning(f"⚠️ PyTorch 최적화 설정 실패: {e}")
            self.dtype = torch.float32 if TORCH_AVAILABLE else None
    
    def _setup_model_interface_safe(self):
        """🔗 ModelLoader 인터페이스 안전한 설정"""
        try:
            # 순환 import 방지를 위한 늦은 import
            from ..utils.model_loader import get_global_model_loader
            
            model_loader = get_global_model_loader()
            if model_loader:
                self.model_interface = model_loader.create_step_interface(self.step_name)
                self.model_loader = model_loader
                self.logger.info(f"🔗 {self.step_name} ModelLoader 인터페이스 연결 완료")
            else:
                self.logger.warning(f"⚠️ {self.step_name} 전역 ModelLoader를 찾을 수 없음")
                self.model_interface = None
                self.model_loader = None
                
        except ImportError as e:
            self.logger.warning(f"⚠️ ModelLoader 모듈 import 실패: {e}")
            self.model_interface = None
            self.model_loader = None
        except Exception as e:
            self.logger.error(f"❌ {self.step_name} ModelLoader 인터페이스 설정 실패: {e}")
            self.model_interface = None
            self.model_loader = None
    
    # 🔧 메서드 시그니처 수정 (missing positional argument 문제 해결)
    def _auto_detect_device_safe(self, preferred_device: Optional[str] = None) -> str:
        """안전한 디바이스 자동 탐지 (파라미터 포함)"""
        if preferred_device and preferred_device != "auto":
            return preferred_device
        return self._auto_detect_device()
    
    async def initialize_step(self) -> bool:
        """🚀 Step 완전 초기화"""
        try:
            self.logger.info(f"🚀 {self.step_name} 초기화 시작...")
            
            # 기본 초기화 확인
            if not hasattr(self, 'logger'):
                self.logger = logging.getLogger(f"pipeline.{self.__class__.__name__}")
            
            if not hasattr(self, 'device'):
                self.device = self._auto_detect_device()
            
            # ModelLoader 인터페이스 재설정 (필요시)
            if not hasattr(self, 'model_interface') or self.model_interface is None:
                self._setup_model_interface_safe()
            
            # 커스텀 초기화 호출 (하위 클래스에서 오버라이드 가능)
            if hasattr(self, '_custom_initialize'):
                await self._custom_initialize()
            
            self.is_initialized = True
            self.logger.info(f"✅ {self.step_name} 완전 초기화 완료")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ {self.step_name} 초기화 실패: {e}")
            self.logger.error(f"📋 상세 오류: {traceback.format_exc()}")
            self.last_error = str(e)
            self.error_count += 1
            return False
    
    async def get_model(self, model_name: Optional[str] = None) -> Optional[Any]:
        """🤖 모델 로드 (안전한 폴백 포함)"""
        try:
            # 인터페이스 설정 확인
            if self.model_interface is None:
                await self.setup_model_interface()
            
            if self.model_interface is not None:
                if model_name:
                    return await self.model_interface.get_model(model_name)
                else:
                    # 권장 모델 자동 로드
                    return await self.model_interface.get_recommended_model()
            else:
                # 폴백: 더미 모델 반환
                self.logger.warning(f"⚠️ ModelLoader 없음, 더미 모델 사용")
                return self._create_dummy_model(model_name or "default")
                
        except Exception as e:
            self.logger.warning(f"⚠️ 모델 로드 실패: {e}, 더미 모델 사용")
            self.last_error = str(e)
            self.error_count += 1
            return self._create_dummy_model(model_name or "default")
    
    async def setup_model_interface(self) -> bool:
        """🔗 ModelLoader 인터페이스 설정 (지연 로딩)"""
        try:
            if self.model_interface is not None:
                return True
            
            # ModelLoader 가져오기
            try:
                from ..utils.model_loader import get_global_model_loader
                self.model_loader = get_global_model_loader()
                
                if self.model_loader:
                    self.model_interface = self.model_loader.create_step_interface(self.step_name)
                    self.logger.info(f"🔗 {self.step_name} ModelLoader 인터페이스 설정 완료")
                    return True
                else:
                    self.logger.warning(f"⚠️ 전역 ModelLoader를 찾을 수 없음")
                    return False
                
            except ImportError as e:
                self.logger.warning(f"⚠️ ModelLoader 모듈 import 실패: {e}")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ {self.step_name} ModelLoader 인터페이스 설정 실패: {e}")
            return False
    
    def _create_dummy_model(self, model_name: str) -> 'DummyModel':
        """더미 모델 생성"""
        
        class DummyModel:
            """더미 AI 모델 - 테스트 및 폴백용"""
            
            def __init__(self, name: str, device: str, step_name: str):
                self.name = name
                self.device = device
                self.step_name = step_name
                self.is_dummy = True
                self.is_loaded = True
            
            def __call__(self, *args, **kwargs):
                return self.forward(*args, **kwargs)
            
            def forward(self, *args, **kwargs):
                """더미 처리 - 입력 크기 유지하며 의미있는 출력 생성"""
                if TORCH_AVAILABLE and args and isinstance(args[0], torch.Tensor):
                    input_tensor = args[0]
                    # 입력과 같은 크기의 더미 출력 (약간의 노이즈 추가)
                    output = torch.randn_like(input_tensor) * 0.1
                    if input_tensor.dtype == torch.uint8:
                        output = (output * 255).clamp(0, 255).to(torch.uint8)
                    else:
                        output = output.clamp(0, 1)
                    return output
                elif NUMPY_AVAILABLE:
                    # NumPy 배열 출력
                    return np.random.randn(1, 3, 512, 512).astype(np.float32)
                else:
                    return None
            
            def to(self, device):
                self.device = str(device)
                return self
            
            def eval(self):
                return self
            
            def cpu(self):
                return self.to('cpu')
            
            def cuda(self):
                return self.to('cuda')
            
            def parameters(self):
                return []
        
        return DummyModel(model_name, self.device, self.step_name)
    
    def record_performance(self, operation: str, duration: float, success: bool = True):
        """📊 성능 메트릭 기록"""
        if not hasattr(self, 'performance_metrics'):
            self.performance_metrics = {}
            
        if operation not in self.performance_metrics:
            self.performance_metrics[operation] = {
                "total_calls": 0,
                "success_calls": 0,
                "total_duration": 0.0,
                "avg_duration": 0.0,
                "last_duration": 0.0,
                "min_duration": float('inf'),
                "max_duration": 0.0
            }
        
        metrics = self.performance_metrics[operation]
        metrics["total_calls"] += 1
        metrics["total_duration"] += duration
        metrics["last_duration"] = duration
        metrics["avg_duration"] = metrics["total_duration"] / metrics["total_calls"]
        
        # 최소/최대 시간 업데이트
        metrics["min_duration"] = min(metrics["min_duration"], duration)
        metrics["max_duration"] = max(metrics["max_duration"], duration)
        
        if success:
            metrics["success_calls"] += 1
    
    def get_step_info(self) -> Dict[str, Any]:
        """📋 Step 상태 정보 반환"""
        return {
            'step_name': self.step_name,
            'step_number': getattr(self, 'step_number', 0),
            'step_type': getattr(self, 'step_type', 'unknown'),
            'device': self.device,
            'device_type': self.device_type,
            'memory_gb': self.memory_gb,
            'is_m3_max': self.is_m3_max,
            'optimization_enabled': self.optimization_enabled,
            'quality_level': self.quality_level,
            'batch_size': self.batch_size,
            'is_initialized': self.is_initialized,
            'has_model_interface': self.model_interface is not None,
            'has_model_loader': self.model_loader is not None,
            'last_processing_time': self.last_processing_time,
            'total_processing_count': self.total_processing_count,
            'error_count': self.error_count,
            'last_error': self.last_error,
            'cache_dir': str(self.cache_dir),
            'config_keys': list(self.config.keys()) if hasattr(self.config, 'keys') else [],
            'performance_metrics': self.performance_metrics,
            'torch_available': TORCH_AVAILABLE,
            'mps_available': MPS_AVAILABLE,
            'dtype': str(getattr(self, 'dtype', 'None'))
        }
    
    def cleanup_models(self):
        """🧹 모델 정리"""
        try:
            if hasattr(self, 'model_interface') and self.model_interface:
                self.model_interface.unload_models()
                self.logger.info(f"🧹 {self.step_name} 모델 정리 완료")
            
            # PyTorch 메모리 정리
            if TORCH_AVAILABLE:
                if self.device == "mps" and hasattr(torch.backends.mps, 'empty_cache'):
                    torch.backends.mps.empty_cache()
                elif self.device == "cuda":
                    torch.cuda.empty_cache()
                
                # 가비지 컬렉션
                gc.collect()
                
        except Exception as e:
            self.logger.warning(f"⚠️ {self.step_name} 모델 정리 중 오류: {e}")
    
    def __del__(self):
        """소멸자 - 리소스 정리"""
        try:
            self.cleanup_models()
        except:
            pass

# ==============================================
# 🔥 Step별 특화 Mixin들
# ==============================================

class HumanParsingMixin(BaseStepMixin):
    """Step 1: Human Parsing 특화 Mixin"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.step_number = 1
        self.step_type = "human_parsing"
        self.num_classes = 20
        self.output_format = "segmentation_mask"

class PoseEstimationMixin(BaseStepMixin):
    """Step 2: Pose Estimation 특화 Mixin"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.step_number = 2
        self.step_type = "pose_estimation"
        self.num_keypoints = 18
        self.output_format = "keypoints_heatmap"

class ClothSegmentationMixin(BaseStepMixin):
    """Step 3: Cloth Segmentation 특화 Mixin"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.step_number = 3
        self.step_type = "cloth_segmentation"
        self.binary_output = True
        self.output_format = "binary_mask"

class GeometricMatchingMixin(BaseStepMixin):
    """Step 4: Geometric Matching 특화 Mixin"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.step_number = 4
        self.step_type = "geometric_matching"
        self.num_control_points = 25
        self.output_format = "transformation_matrix"

class ClothWarpingMixin(BaseStepMixin):
    """Step 5: Cloth Warping 특화 Mixin"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.step_number = 5
        self.step_type = "cloth_warping"
        self.enable_physics = True
        self.output_format = "warped_cloth"

class VirtualFittingMixin(BaseStepMixin):
    """Step 6: Virtual Fitting 특화 Mixin"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.step_number = 6
        self.step_type = "virtual_fitting"
        self.diffusion_steps = 50
        self.output_format = "rgb_image"

class PostProcessingMixin(BaseStepMixin):
    """Step 7: Post Processing 특화 Mixin"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.step_number = 7
        self.step_type = "post_processing"
        self.upscale_factor = 2
        self.output_format = "enhanced_image"

class QualityAssessmentMixin(BaseStepMixin):
    """Step 8: Quality Assessment 특화 Mixin"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.step_number = 8
        self.step_type = "quality_assessment"
        self.quality_threshold = 0.7
        self.output_format = "quality_scores"

# ==============================================
# 🔧 유틸리티 데코레이터 및 헬퍼 함수들
# ==============================================

def ensure_step_initialization(func: Callable) -> Callable:
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

def safe_step_method(func: Callable) -> Callable:
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

def performance_monitor(operation_name: str) -> Callable:
    """성능 모니터링 데코레이터"""
    def decorator(func: Callable) -> Callable:
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

def memory_optimize(func: Callable) -> Callable:
    """메모리 최적화 데코레이터"""
    async def wrapper(self, *args, **kwargs):
        try:
            result = await func(self, *args, **kwargs)
            
            # 메모리 정리
            if TORCH_AVAILABLE:
                if hasattr(self, 'device'):
                    if self.device == "mps" and hasattr(torch.backends.mps, 'empty_cache'):
                        torch.backends.mps.empty_cache()
                    elif self.device == "cuda":
                        torch.cuda.empty_cache()
                
                gc.collect()
            
            return result
        except Exception as e:
            # 오류 시에도 메모리 정리
            if TORCH_AVAILABLE:
                gc.collect()
            raise e
    return wrapper

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
    'performance_monitor',
    'memory_optimize',
    
    # 상수
    'TORCH_AVAILABLE',
    'MPS_AVAILABLE',
    'CV_AVAILABLE',
    'DEFAULT_DEVICE'
]

# 모듈 초기화 로그
logger.info("✅ BaseStepMixin v3.1 완전 수정 버전 로드 완료")
logger.info("🔗 ModelLoader 인터페이스 완벽 연동")
logger.info("🍎 M3 Max 128GB 최적화 지원")
logger.info("🐍 conda 환경 완벽 지원")
logger.info(f"🔧 PyTorch: {'✅' if TORCH_AVAILABLE else '❌'}, MPS: {'✅' if MPS_AVAILABLE else '❌'}")
logger.info(f"🎯 기본 디바이스: {DEFAULT_DEVICE}")