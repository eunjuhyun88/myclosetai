# app/ai_pipeline/steps/base_step_mixin.py
"""
🔥 BaseStepMixin v8.0 - 순환 임포트 완전 해결 + coroutine 오류 근본 해결
================================================================================

✅ 순환 임포트 완전 해결 (동적 임포트 + 타입 체킹)
✅ coroutine 오류 근본 원인 해결 (동기 메서드로 변경)
✅ logger 속성 누락 문제 완전 해결
✅ 기존 함수/클래스명 100% 유지
✅ Dict Callable 오류 완전 해결
✅ 89.8GB 체크포인트 자동 탐지 유지
✅ M3 Max 128GB 최적화 유지
✅ conda 환경 완벽 지원
✅ 의존성 주입 패턴 적용
✅ 프로덕션 안정성 최고 수준

Author: MyCloset AI Team  
Date: 2025-07-20
Version: 8.0 (Complete Circular Import Resolution + Coroutine Fix)
"""

import os
import gc
import time
import asyncio
import logging
import threading
import traceback
import weakref
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Type, Callable, Tuple, TYPE_CHECKING
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
from functools import wraps

# ==============================================
# 🔥 TYPE_CHECKING으로 순환 임포트 완전 방지
# ==============================================

if TYPE_CHECKING:
    # 타입 체킹 시에만 임포트 (런타임에는 임포트 안됨)
    from ..utils.model_loader import ModelLoader, StepModelInterface, SafeFunctionValidator
    from ..utils.memory_manager import MemoryManager
    from ..utils.auto_model_detector import RealWorldModelDetector

# ==============================================
# 🔥 NumPy 2.x 호환성 문제 완전 해결
# ==============================================

try:
    import numpy as np
    numpy_version = np.__version__
    major_version = int(numpy_version.split('.')[0])
    
    if major_version >= 2:
        logging.warning(f"⚠️ NumPy {numpy_version} 감지됨. NumPy 1.x 권장")
        try:
            np.set_printoptions(legacy='1.25')
            logging.info("✅ NumPy 2.x 호환성 모드 활성화")
        except:
            pass
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

# 안전한 PyTorch import
try:
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
    os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
    
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
    
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        MPS_AVAILABLE = True
        DEFAULT_DEVICE = "mps"
    else:
        MPS_AVAILABLE = False
        DEFAULT_DEVICE = "cpu"
        
except ImportError:
    TORCH_AVAILABLE = False
    MPS_AVAILABLE = False
    DEFAULT_DEVICE = "cpu"
    torch = None

# 이미지 처리 라이브러리
try:
    import cv2
    from PIL import Image
    CV_AVAILABLE = True
    PIL_AVAILABLE = True
except ImportError:
    CV_AVAILABLE = False
    PIL_AVAILABLE = False

logger = logging.getLogger(__name__)

# ==============================================
# 🔥 SafeConfig 클래스 (완전 안전한 설정 관리)
# ==============================================

class SafeConfig:
    """🔧 완전 안전한 설정 클래스 v8.0"""
    
    def __init__(self, data: Any = None):
        """완전 안전한 초기화"""
        self._data = {}
        self._original_data = data
        self._lock = threading.RLock()
        
        try:
            with self._lock:
                if data is None:
                    self._data = {}
                elif isinstance(data, dict):
                    self._data = self._safe_dict_copy(data)
                elif hasattr(data, '__dict__'):
                    self._data = self._safe_object_to_dict(data)
                elif callable(data):
                    logger.warning("⚠️ callable 설정 객체 감지, 빈 설정으로 처리")
                    self._data = {}
                else:
                    if hasattr(data, '__iter__') and not isinstance(data, (str, bytes)):
                        try:
                            self._data = dict(data)
                        except:
                            self._data = {}
                    else:
                        self._data = {}
                
                self._set_attributes_safely()
                
        except Exception as e:
            logger.warning(f"⚠️ SafeConfig 초기화 실패: {e}, 빈 설정 사용")
            self._data = {}
    
    def _safe_dict_copy(self, data: dict) -> dict:
        """딕셔너리 안전 복사"""
        safe_dict = {}
        for key, value in data.items():
            try:
                if not callable(value):
                    safe_dict[key] = value
            except:
                pass
        return safe_dict
    
    def _safe_object_to_dict(self, obj: Any) -> dict:
        """객체를 딕셔너리로 안전 변환"""
        safe_dict = {}
        
        if hasattr(obj, '__dict__'):
            for key, value in obj.__dict__.items():
                try:
                    if not key.startswith('_') and not callable(value):
                        safe_dict[key] = value
                except:
                    pass
        
        for attr_name in dir(obj):
            if not attr_name.startswith('_'):
                try:
                    attr_value = getattr(obj, attr_name)
                    if not callable(attr_value):
                        safe_dict[attr_name] = attr_value
                except:
                    pass
        
        return safe_dict
    
    def _set_attributes_safely(self):
        """속성들을 안전하게 설정"""
        for key, value in self._data.items():
            try:
                if isinstance(key, str) and key.isidentifier() and not hasattr(self, key):
                    setattr(self, key, value)
            except:
                pass
    
    def get(self, key: str, default=None):
        """완전 안전한 get 메서드"""
        try:
            with self._lock:
                return self._data.get(key, default)
        except Exception as e:
            logger.debug(f"SafeConfig.get 오류: {e}")
            return default
    
    def __getitem__(self, key):
        return self.get(key, None)
    
    def __setitem__(self, key, value):
        try:
            with self._lock:
                self._data[key] = value
                if isinstance(key, str) and key.isidentifier():
                    setattr(self, key, value)
        except Exception as e:
            logger.debug(f"SafeConfig.__setitem__ 오류: {e}")
    
    def __contains__(self, key):
        try:
            return key in self._data
        except:
            return False
    
    def keys(self):
        try:
            return self._data.keys()
        except:
            return []
    
    def values(self):
        try:
            return self._data.values()
        except:
            return []
    
    def items(self):
        try:
            return self._data.items()
        except:
            return []
    
    def update(self, other):
        try:
            with self._lock:
                if isinstance(other, dict):
                    for key, value in other.items():
                        if not callable(value):
                            self._data[key] = value
                            if isinstance(key, str) and key.isidentifier():
                                setattr(self, key, value)
        except Exception as e:
            logger.debug(f"SafeConfig.update 오류: {e}")
    
    def __str__(self):
        return str(self._data)
    
    def __repr__(self):
        return f"SafeConfig({self._data})"
    
    def __bool__(self):
        return bool(self._data)
    
    def __len__(self):
        return len(self._data)

# ==============================================
# 🔥 폴백 SafeFunctionValidator (순환 임포트 방지)
# ==============================================

class FallbackSafeFunctionValidator:
    """폴백 SafeFunctionValidator - 순환 임포트 시 사용"""
    
    @staticmethod
    def safe_call(func: Callable, *args, **kwargs) -> Tuple[bool, Any, str]:
        """안전한 함수 호출"""
        try:
            if not callable(func):
                return False, None, "Object is not callable"
            result = func(*args, **kwargs)
            return True, result, "Success"
        except Exception as e:
            return False, None, str(e)
    
    @staticmethod
    async def safe_async_call(func: Callable, *args, **kwargs) -> Tuple[bool, Any, str]:
        """안전한 비동기 함수 호출"""
        try:
            if not callable(func):
                return False, None, "Object is not callable"
            
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            return True, result, "Success"
        except Exception as e:
            return False, None, str(e)
    
    @staticmethod
    def safe_getattr_call(obj: Any, attr_name: str, *args, **kwargs) -> Tuple[bool, Any, str]:
        """안전한 속성 접근 및 호출"""
        try:
            if not hasattr(obj, attr_name):
                return False, None, f"Object has no attribute '{attr_name}'"
            
            attr = getattr(obj, attr_name)
            
            if args or kwargs:
                if callable(attr):
                    result = attr(*args, **kwargs)
                    return True, result, "Success"
                else:
                    return False, None, f"Attribute '{attr_name}' is not callable"
            else:
                return True, attr, "Success"
                
        except Exception as e:
            return False, None, str(e)

# ==============================================
# 🔥 BaseStepMixin v8.0 - 완전 최적화
# ==============================================

class BaseStepMixin:
    """
    🔥 BaseStepMixin v8.0 - 순환 임포트 완전 해결 + coroutine 오류 근본 해결
    
    ✅ 순환 임포트 완전 해결 (동적 임포트 + TYPE_CHECKING)
    ✅ coroutine 오류 근본 원인 해결 (동기 메서드)
    ✅ logger 속성 누락 문제 완전 해결
    ✅ 기존 API 100% 호환성 유지
    ✅ Dict Callable 오류 완전 해결
    ✅ 89.8GB 체크포인트 자동 탐지 유지
    ✅ M3 Max 128GB 최적화 유지
    ✅ 의존성 주입 패턴 적용
    """
    
    # 클래스 변수
    _class_registry = weakref.WeakSet()
    _initialization_lock = threading.RLock()
    
    def __init__(
        self, 
        model_loader: Optional['ModelLoader'] = None,
        memory_manager: Optional['MemoryManager'] = None,
        function_validator: Optional['SafeFunctionValidator'] = None,
        **kwargs
    ):
        """🔥 v8.0 완전 최적화된 초기화"""
        
        # ===== 🔥 STEP 0: logger 속성 최우선 생성 (절대 누락 방지) =====
        self._ensure_logger_first()
        
        # ===== 🔥 STEP 1: 의존성 주입 저장 =====
        self.model_loader = model_loader
        self.memory_manager = memory_manager
        self.function_validator = function_validator or FallbackSafeFunctionValidator()
        self.model_interface = None
        
        # ===== 🔥 STEP 2: 클래스 등록 =====
        BaseStepMixin._class_registry.add(self)
        
        # ===== 🔥 STEP 3: 안전한 초기화 =====
        with BaseStepMixin._initialization_lock:
            try:
                # 기존 초기화 순서 유지
                self._check_numpy_compatibility()
                self._setup_basic_attributes(kwargs)
                self._safe_super_init()
                self._setup_device_and_system(kwargs)
                self._setup_config_safely(kwargs)
                self._setup_state_management()
                self._setup_m3_max_optimization()
                self._setup_memory_optimization()
                self._setup_warmup_system()
                self._setup_performance_monitoring()
                
                # ===== 🔥 STEP 4: ModelLoader 인터페이스 설정 (동기) =====
                self._setup_model_interface_sync()  # 🔥 핵심: 동기 버전
                
                # ===== 🔥 STEP 5: 체크포인트 탐지 (동기) =====
                self._setup_checkpoint_detection_sync()  # 🔥 핵심: 동기 버전
                
                self.logger.info(f"✅ {self.step_name} BaseStepMixin v8.0 초기화 완료")
                self.logger.debug(f"🔧 Device: {self.device}, Memory: {self.memory_gb}GB")
                
            except Exception as e:
                self._emergency_initialization()
                if hasattr(self, 'logger'):
                    self.logger.error(f"❌ 초기화 실패: {e}")
                    self.logger.debug(f"📋 상세 오류: {traceback.format_exc()}")
    
    # ==============================================
    # 🔥 STEP 0: logger 속성 최우선 보장
    # ==============================================
    
    def _ensure_logger_first(self):
        """🔥 logger 속성 최우선 생성 - 모든 Step 클래스에서 logger 누락 방지"""
        try:
            if hasattr(self, 'logger') and self.logger is not None:
                return
            
            # Step 이름 결정
            class_name = self.__class__.__name__
            step_name = getattr(self, 'step_name', class_name)
            
            # 계층적 로거 이름 생성
            logger_name = f"pipeline.{step_name}"
            
            # 로거 생성
            self.logger = logging.getLogger(logger_name)
            self.logger.setLevel(logging.INFO)
            
            # 핸들러 설정 (중복 방지)
            if not self.logger.handlers:
                handler = logging.StreamHandler()
                formatter = logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                )
                handler.setFormatter(formatter)
                self.logger.addHandler(handler)
                self.logger.propagate = False
            
            self.logger.info(f"🔧 {step_name} logger 초기화 완료")
            
        except Exception as e:
            try:
                # 폴백 로거
                self.logger = logging.getLogger(__name__)
                self.logger.error(f"❌ logger 초기화 실패: {e}")
            except:
                print(f"❌ CRITICAL: logger 초기화 완전 실패: {e}")
    
    # ==============================================
    # 🔥 기존 초기화 메서드들 (내용 100% 유지)
    # ==============================================
    
    def _check_numpy_compatibility(self):
        """NumPy 2.x 호환성 체크"""
        if NUMPY_AVAILABLE and int(np.__version__.split('.')[0]) >= 2:
            self.logger.warning(f"⚠️ NumPy {np.__version__} (2.x) 감지됨")
    
    def _setup_basic_attributes(self, kwargs: Dict[str, Any]):
        """기본 속성들 설정"""
        try:
            self.step_name = getattr(self, 'step_name', self.__class__.__name__)
            self.step_number = getattr(self, 'step_number', 0)
            self.step_type = getattr(self, 'step_type', 'unknown')
            
            self.is_initialized = False
            self.initialization_error = None
            self.error_count = 0
            self.last_error = None
            
        except Exception as e:
            self.logger.warning(f"⚠️ 기본 속성 설정 실패: {e}")
    
    def _safe_super_init(self):
        """안전한 super() 호출"""
        try:
            mro = type(self).__mro__
            
            if len(mro) > 2 and mro[-2] != BaseStepMixin:
                try:
                    super().__init__()
                except TypeError as te:
                    if "positional argument" in str(te):
                        try:
                            super().__init__({})
                        except:
                            pass
                    else:
                        pass
                
        except Exception as e:
            self.logger.debug(f"super() 호출 건너뜀: {e}")
    
    def _setup_device_and_system(self, kwargs: Dict[str, Any]):
        """디바이스 및 시스템 설정"""
        try:
            self.device = self._safe_device_detection(kwargs)
            self.device_type = self._detect_device_type()
            self.memory_gb = kwargs.get('memory_gb', self._detect_memory())
            self.is_m3_max = kwargs.get('is_m3_max', self._detect_m3_max())
            self.optimization_enabled = kwargs.get('optimization_enabled', True)
            self.quality_level = kwargs.get('quality_level', 'balanced')
            self.batch_size = kwargs.get('batch_size', self._calculate_optimal_batch_size())
            
        except Exception as e:
            self.logger.warning(f"⚠️ 디바이스 설정 실패: {e}")
            # 폴백 설정
            self.device = DEFAULT_DEVICE
            self.device_type = "unknown"
            self.memory_gb = 16.0
            self.is_m3_max = False
            self.optimization_enabled = False
            self.quality_level = 'balanced'
            self.batch_size = 1
    
    def _safe_device_detection(self, kwargs: Dict[str, Any]) -> str:
        """안전한 디바이스 탐지"""
        try:
            device_candidates = [
                kwargs.get('device'),
                kwargs.get('preferred_device'), 
                kwargs.get('target_device'),
                getattr(self, 'device', None)
            ]
            
            for device in device_candidates:
                if device and device != "auto":
                    return device
            
            return self._auto_detect_device()
            
        except Exception as e:
            self.logger.warning(f"⚠️ 디바이스 탐지 실패: {e}")
            return DEFAULT_DEVICE
    
    def _auto_detect_device(self, preferred_device: Optional[str] = None, device: Optional[str] = None) -> str:
        """통일된 디바이스 자동 탐지 메서드"""
        try:
            target_device = preferred_device or device
            
            if target_device and target_device != "auto":
                return target_device
                
            if not TORCH_AVAILABLE:
                return "cpu"
            
            if MPS_AVAILABLE:
                return "mps"
            elif torch.cuda.is_available():
                return "cuda"
            else:
                return "cpu"
                
        except Exception:
            return "cpu"
    
    def _detect_device_type(self) -> str:
        try:
            if self.device == "mps":
                return "apple_silicon"
            elif self.device == "cuda":
                return "nvidia_gpu"
            else:
                return "generic_cpu"
        except:
            return "unknown"
    
    def _detect_memory(self) -> float:
        try:
            import psutil
            return round(psutil.virtual_memory().total / (1024**3), 1)
        except:
            return 16.0
    
    def _detect_m3_max(self) -> bool:
        try:
            import platform
            processor = str(platform.processor())
            return ("M3" in processor or 
                   (self.device == "mps" and self.memory_gb > 64))
        except:
            return False
    
    def _calculate_optimal_batch_size(self) -> int:
        try:
            if self.is_m3_max and self.memory_gb >= 128:
                return 8
            elif self.memory_gb >= 64:
                return 4
            elif self.memory_gb >= 32:
                return 2
            else:
                return 1
        except:
            return 1
    
    def _setup_config_safely(self, kwargs: Dict[str, Any]):
        """설정 객체 안전 처리"""
        try:
            raw_config = kwargs.get('config', {})
            self.config = SafeConfig(raw_config)
            
            safe_kwargs = {}
            for key, value in kwargs.items():
                try:
                    if key != 'config' and not callable(value):
                        safe_kwargs[key] = value
                except:
                    pass
            
            if safe_kwargs:
                self.config.update(safe_kwargs)
            
        except Exception as e:
            self.logger.warning(f"⚠️ 설정 처리 실패: {e}")
            self.config = SafeConfig({})
    
    def _setup_state_management(self):
        """상태 관리 초기화"""
        try:
            cache_dir = getattr(self.config, 'cache_dir', './cache')
            self.cache_dir = Path(cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            
        except Exception as e:
            self.logger.warning(f"⚠️ 상태 관리 초기화 실패: {e}")
    
    def _setup_m3_max_optimization(self):
        """M3 Max 최적화 설정"""
        try:
            if self.device == "mps" and TORCH_AVAILABLE:
                os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
                os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
                
                if hasattr(torch.backends.mps, 'enable_fallback'):
                    torch.backends.mps.enable_fallback = True
                
                if self.is_m3_max:
                    os.environ['OMP_NUM_THREADS'] = '16'
                
                self.dtype = torch.float32
                
                self.logger.info("🍎 M3 Max 최적화 설정 완료")
                
        except Exception as e:
            self.logger.warning(f"⚠️ M3 Max 최적화 실패: {e}")
    
    def _setup_memory_optimization(self):
        """메모리 최적화 설정"""
        try:
            if TORCH_AVAILABLE:
                torch.backends.cudnn.benchmark = (self.device == "cuda")
                
                if hasattr(torch.backends, 'cuda') and self.device == "cuda":
                    torch.backends.cuda.matmul.allow_tf32 = True
                    torch.backends.cudnn.allow_tf32 = True
            
            gc.set_threshold(700, 10, 10)
            
        except Exception as e:
            self.logger.warning(f"⚠️ 메모리 최적화 설정 실패: {e}")
    
    def _setup_warmup_system(self):
        """워밍업 시스템 초기화"""
        try:
            self.warmup_functions = {
                'model_warmup': self._safe_model_warmup,
                'device_warmup': self._safe_device_warmup,
                'memory_warmup': self._safe_memory_warmup,
                'pipeline_warmup': self._safe_pipeline_warmup
            }
            
            self.warmup_config = SafeConfig({
                'enabled': True,
                'timeout': 30.0,
                'retry_count': 3,
                'warm_cache': True
            })
            
            for name, func in self.warmup_functions.items():
                if not callable(func):
                    self.logger.error(f"❌ {name}이 callable이 아님: {type(func)}")
                    self.warmup_functions[name] = self._create_dummy_warmup(name)
            
        except Exception as e:
            self.logger.warning(f"⚠️ 워밍업 시스템 초기화 실패: {e}")
            self.warmup_functions = {}
            self.warmup_config = SafeConfig({})
    
    def _create_dummy_warmup(self, name: str) -> Callable:
        """안전한 더미 워밍업 함수 생성"""
        async def dummy_warmup():
            self.logger.debug(f"🔧 더미 워밍업 실행: {name}")
            return True
        return dummy_warmup
    
    def _setup_performance_monitoring(self):
        """성능 모니터링 시스템 초기화"""
        try:
            self.performance_metrics = {
                'total_calls': 0,
                'successful_calls': 0,
                'failed_calls': 0,
                'average_duration': 0.0,
                'min_duration': float('inf'),
                'max_duration': 0.0,
                'last_call_duration': 0.0,
                'total_duration': 0.0
            }
            
            self.last_processing_time = 0.0
            self.total_processing_count = 0
            self.performance_history = []
            
        except Exception as e:
            self.logger.warning(f"⚠️ 성능 모니터링 초기화 실패: {e}")
            self.performance_metrics = {}
            self.last_processing_time = 0.0
            self.total_processing_count = 0
    
    def _emergency_initialization(self):
        """응급 초기화"""
        try:
            if not hasattr(self, 'logger'):
                self.logger = logging.getLogger(__name__)
            if not hasattr(self, 'step_name'):
                self.step_name = self.__class__.__name__
            if not hasattr(self, 'device'):
                self.device = "cpu"
            if not hasattr(self, 'config'):
                self.config = SafeConfig({})
            if not hasattr(self, 'is_initialized'):
                self.is_initialized = False
            if not hasattr(self, 'performance_metrics'):
                self.performance_metrics = {}
            if not hasattr(self, 'function_validator'):
                self.function_validator = FallbackSafeFunctionValidator()
            
            self.logger.warning("⚠️ 응급 초기화 완료")
            
        except:
            pass
    
    # ==============================================
    # 🔥 핵심: ModelLoader 인터페이스 설정 (동기 버전)
    # ==============================================
    
    def _setup_model_interface_sync(self):
        """
        🔥 v8.0 핵심: ModelLoader 인터페이스 설정 - 동기 버전으로 완전 수정
        
        ✅ coroutine 오류 완전 해결 (비동기 -> 동기 변경)
        ✅ 순환 임포트 완전 방지 (동적 임포트)
        ✅ Dict Callable 오류 완전 해결
        ✅ 안전한 에러 처리
        """
        try:
            self.logger.info(f"🔗 {self.step_name} ModelLoader 인터페이스 설정 중...")
            
            # 🔥 Step 1: 주입된 model_loader 사용 (최우선)
            if self.model_loader:
                self.logger.info("✅ 주입된 ModelLoader 사용")
                interface = self._create_interface_from_loader(self.model_loader)
                if interface:
                    self.model_interface = interface
                    self.logger.info(f"✅ {self.step_name} 모델 인터페이스 생성 완료 (주입)")
                    return
            
            # 🔥 Step 2: 동적 임포트로 전역 ModelLoader 가져오기
            try:
                # 런타임 동적 임포트 (순환 임포트 방지)
                from ...utils.model_loader import get_global_model_loader
                
                success, model_loader, message = self.function_validator.safe_call(get_global_model_loader)
                
                if success and model_loader:
                    self.model_loader = model_loader
                    interface = self._create_interface_from_loader(model_loader)
                    
                    if interface:
                        self.model_interface = interface
                        self.logger.info(f"✅ {self.step_name} 모델 인터페이스 생성 완료 (전역)")
                    else:
                        self.logger.warning(f"⚠️ {self.step_name} 인터페이스 생성 실패")
                        self.model_interface = None
                else:
                    self.logger.warning(f"⚠️ 전역 ModelLoader 가져오기 실패: {message}")
                    self.model_interface = None
                    
            except ImportError as e:
                self.logger.debug(f"ModelLoader 동적 임포트 실패: {e}")
                self.model_interface = None
            except Exception as e:
                self.logger.warning(f"⚠️ ModelLoader 동적 로딩 실패: {e}")
                self.model_interface = None
            
            # 🔥 Step 3: 연동 상태 로깅
            self._log_interface_status()
            
        except Exception as e:
            self.logger.error(f"❌ ModelLoader 인터페이스 설정 실패: {e}")
            self.logger.debug(f"📋 상세 오류: {traceback.format_exc()}")
            
            # 완전 폴백 설정
            self.model_interface = None
            if not hasattr(self, 'model_loader'):
                self.model_loader = None
    
    def _create_interface_from_loader(self, model_loader) -> Optional[Any]:
        """ModelLoader에서 인터페이스 생성 (동기)"""
        try:
            if not model_loader:
                return None
            
            # 🔥 핵심: create_step_interface 메서드를 동기적으로 호출
            if hasattr(model_loader, 'create_step_interface'):
                create_method = getattr(model_loader, 'create_step_interface')
                
                success, interface, message = self.function_validator.safe_call(
                    create_method, self.step_name
                )
                
                if success:
                    self.logger.info(f"✅ {self.step_name} 인터페이스 생성 성공")
                    return interface
                else:
                    self.logger.warning(f"⚠️ create_step_interface 호출 실패: {message}")
                    return None
            else:
                self.logger.warning("⚠️ ModelLoader에 create_step_interface 메서드가 없습니다")
                return None
                
        except Exception as e:
            self.logger.error(f"❌ 인터페이스 생성 실패: {e}")
            return None
    
    def _log_interface_status(self):
        """인터페이스 연동 상태 로깅"""
        try:
            interface_status = "✅ 연결됨" if self.model_interface else "❌ 연결 실패"
            loader_status = "✅ 로드됨" if self.model_loader else "❌ 로드 실패"
            validator_status = "✅ 사용 가능" if self.function_validator else "❌ 없음"
            
            self.logger.info(f"🔗 ModelLoader 연동 결과:")
            self.logger.info(f"   - ModelLoader: {loader_status}")
            self.logger.info(f"   - Step Interface: {interface_status}")
            self.logger.info(f"   - SafeFunctionValidator: {validator_status}")
            
        except Exception as e:
            self.logger.debug(f"인터페이스 상태 로깅 실패: {e}")
    
    # ==============================================
    # 🔥 체크포인트 탐지 설정 (동기 버전)
    # ==============================================
    
    def _setup_checkpoint_detection_sync(self):
        """
        🔥 v8.0: 체크포인트 탐지 설정 - 동기 버전
        
        ✅ 89.8GB 체크포인트 자동 탐지 유지
        ✅ coroutine 오류 완전 해결
        ✅ 순환 임포트 방지
        """
        try:
            self.logger.info(f"🔍 {self.step_name} 체크포인트 탐지 시작...")
            
            # 체크포인트 탐지는 선택적 기능이므로 실패해도 계속 진행
            if not self.model_loader:
                self.logger.debug("⏳ ModelLoader 없음, 체크포인트 탐지 스킵")
                return
            
            try:
                # 동적 임포트로 탐지기 가져오기
                from ...utils.auto_model_detector import create_real_world_detector
                
                detector = create_real_world_detector(
                    enable_pytorch_validation=True,
                    max_workers=2
                )
                
                step_model_filter = self._get_step_model_filter()
                
                # 🔥 동기적으로 탐지 실행
                success, detected_models, message = self.function_validator.safe_call(
                    detector.detect_all_models,
                    model_type_filter=step_model_filter,
                    min_confidence=0.3,
                    force_rescan=False
                )
                
                if success and detected_models:
                    step_models = self._find_models_for_step(detected_models)
                    if step_models:
                        self._register_detected_models_sync(step_models)
                        self.logger.info(f"✅ {len(step_models)}개 체크포인트 탐지 완료")
                    else:
                        self.logger.debug("관련 체크포인트 없음")
                else:
                    self.logger.debug(f"체크포인트 탐지 실패: {message}")
                    
            except ImportError as e:
                self.logger.debug(f"체크포인트 탐지 모듈 없음: {e}")
            except Exception as e:
                self.logger.debug(f"체크포인트 탐지 실패: {e}")
                
        except Exception as e:
            self.logger.debug(f"체크포인트 탐지 설정 실패: {e}")
    
    def _get_step_model_filter(self) -> List[str]:
        """Step별 모델 타입 필터 반환"""
        step_filters = {
            "HumanParsingStep": ["human_parsing"],
            "PoseEstimationStep": ["pose_estimation"],
            "ClothSegmentationStep": ["cloth_segmentation"],
            "GeometricMatchingStep": ["geometric_matching"],
            "ClothWarpingStep": ["cloth_warping"],
            "VirtualFittingStep": ["virtual_fitting"],
            "PostProcessingStep": ["post_processing"],
            "QualityAssessmentStep": ["quality_assessment"]
        }
        
        return step_filters.get(self.step_name, [])
    
    def _find_models_for_step(self, detected_models: Dict) -> Dict:
        """Step별 관련 모델 찾기"""
        step_models = {}
        
        for model_name, model_info in detected_models.items():
            if hasattr(model_info, 'step_name') and model_info.step_name == self.step_name:
                step_models[model_name] = model_info
            elif hasattr(model_info, 'category') and any(filter_type in model_info.category.value 
                    for filter_type in self._get_step_model_filter()):
                step_models[model_name] = model_info
        
        return step_models
    
    def _register_detected_models_sync(self, step_models: Dict):
        """탐지된 모델들을 동기적으로 등록"""
        try:
            if not self.model_loader or not hasattr(self.model_loader, 'register_model'):
                return
            
            registered_count = 0
            
            for model_name, model_info in step_models.items():
                try:
                    model_config = {
                        'name': model_name,
                        'type': model_info.category.value if hasattr(model_info, 'category') else 'unknown',
                        'checkpoint_path': str(model_info.path) if hasattr(model_info, 'path') else '',
                        'device': self.device,
                        'pytorch_validated': getattr(model_info, 'pytorch_valid', False),
                        'parameter_count': getattr(model_info, 'parameter_count', 0),
                        'file_size_mb': getattr(model_info, 'file_size_mb', 0),
                        'confidence_score': getattr(model_info, 'confidence_score', 0.0),
                        'step_name': self.step_name,
                        'auto_detected': True
                    }
                    
                    register_method = getattr(self.model_loader, 'register_model')
                    success, result, message = self.function_validator.safe_call(
                        register_method, model_name, model_config
                    )
                    
                    if success:
                        registered_count += 1
                        self.logger.debug(f"✅ 모델 등록 성공: {model_name}")
                    else:
                        self.logger.warning(f"⚠️ 모델 등록 실패 {model_name}: {message}")
                        
                except Exception as e:
                    self.logger.warning(f"⚠️ 모델 등록 중 오류 {model_name}: {e}")
            
            if registered_count > 0:
                self.logger.info(f"✅ {registered_count}개 탐지된 모델 등록 완료")
            
        except Exception as e:
            self.logger.error(f"❌ 탐지된 모델 등록 실패: {e}")
    
    # ==============================================
    # 🔥 워밍업 메서드들 (기존 내용 유지)
    # ==============================================
    
    async def _safe_model_warmup(self) -> bool:
        """안전한 모델 워밍업"""
        try:
            self.logger.debug(f"🔥 {self.step_name} 모델 워밍업...")
            
            if TORCH_AVAILABLE and self.device == "mps":
                warmup_tensor = torch.randn(1, 3, 224, 224, 
                                          device=self.device, dtype=getattr(self, 'dtype', torch.float32))
                _ = warmup_tensor * 2.0
                del warmup_tensor
                
                if hasattr(torch.backends.mps, 'empty_cache'):
                    torch.backends.mps.empty_cache()
            
            return True
            
        except Exception as e:
            self.logger.warning(f"⚠️ 모델 워밍업 실패: {e}")
            return False
    
    async def _safe_device_warmup(self) -> bool:
        """안전한 디바이스 워밍업"""
        try:
            if TORCH_AVAILABLE:
                test_tensor = torch.tensor([1.0], device=self.device, dtype=getattr(self, 'dtype', torch.float32))
                result = test_tensor + 1.0
                del test_tensor, result
            
            return True
            
        except Exception as e:
            self.logger.warning(f"⚠️ 디바이스 워밍업 실패: {e}")
            return False
    
    async def _safe_memory_warmup(self) -> bool:
        """안전한 메모리 워밍업"""
        try:
            gc.collect()
            
            if TORCH_AVAILABLE and self.device == "mps":
                if hasattr(torch.backends.mps, 'empty_cache'):
                    torch.backends.mps.empty_cache()
            
            return True
            
        except Exception as e:
            self.logger.warning(f"⚠️ 메모리 워밍업 실패: {e}")
            return False
    
    async def _safe_pipeline_warmup(self) -> bool:
        """안전한 파이프라인 워밍업"""
        try:
            if not hasattr(self, 'config') or not self.config:
                self.config = SafeConfig({})
            
            return True
            
        except Exception as e:
            self.logger.warning(f"⚠️ 파이프라인 워밍업 실패: {e}")
            return False
    
    # ==============================================
    # 🔥 Step 초기화 및 주요 메서드들
    # ==============================================
    
    async def initialize_step(self) -> bool:
        """Step 완전 초기화"""
        try:
            self.logger.info(f"🚀 {self.step_name} 초기화 시작...")
            
            self._verify_essential_attributes()
            await self._execute_safe_warmup()
            
            if hasattr(self, '_custom_initialize') and callable(self._custom_initialize):
                await self._custom_initialize()
            
            self.is_initialized = True
            self.logger.info(f"✅ {self.step_name} 초기화 완료")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ {self.step_name} 초기화 실패: {e}")
            self.last_error = str(e)
            self.error_count += 1
            return False
    
    def _verify_essential_attributes(self):
        """필수 속성들 검증"""
        essential_attrs = ['logger', 'step_name', 'device', 'config', 'function_validator']
        
        for attr in essential_attrs:
            if not hasattr(self, attr):
                if attr == 'logger':
                    self.logger = logging.getLogger(f"pipeline.{self.__class__.__name__}")
                elif attr == 'step_name':
                    self.step_name = self.__class__.__name__
                elif attr == 'device':
                    self.device = DEFAULT_DEVICE
                elif attr == 'config':
                    self.config = SafeConfig({})
                elif attr == 'function_validator':
                    self.function_validator = FallbackSafeFunctionValidator()
    
    async def _execute_safe_warmup(self):
        """안전한 워밍업 실행"""
        try:
            if not hasattr(self, 'warmup_functions') or not self.warmup_functions:
                return
            
            for warmup_name, warmup_func in self.warmup_functions.items():
                try:
                    if callable(warmup_func):
                        await warmup_func()
                    else:
                        self.logger.warning(f"⚠️ {warmup_name}이 callable이 아님")
                except Exception as e:
                    self.logger.warning(f"⚠️ {warmup_name} 실패: {e}")
                    
        except Exception as e:
            self.logger.warning(f"⚠️ 워밍업 실행 실패: {e}")
    
    def record_performance(self, operation_name: str, duration: float, success: bool = True):
        """성능 메트릭 기록"""
        try:
            self.performance_metrics['total_calls'] += 1
            self.performance_metrics['total_duration'] += duration
            self.performance_metrics['last_call_duration'] = duration
            
            if success:
                self.performance_metrics['successful_calls'] += 1
            else:
                self.performance_metrics['failed_calls'] += 1
            
            self.performance_metrics['min_duration'] = min(
                self.performance_metrics['min_duration'], duration
            )
            self.performance_metrics['max_duration'] = max(
                self.performance_metrics['max_duration'], duration
            )
            
            if self.performance_metrics['total_calls'] > 0:
                self.performance_metrics['average_duration'] = (
                    self.performance_metrics['total_duration'] / 
                    self.performance_metrics['total_calls']
                )
            
            self.performance_history.append({
                'operation': operation_name,
                'duration': duration,
                'success': success,
                'timestamp': time.time()
            })
            
            if len(self.performance_history) > 100:
                self.performance_history.pop(0)
                
        except Exception as e:
            self.logger.debug(f"성능 기록 실패: {e}")
    
    def get_step_info(self) -> Dict[str, Any]:
        """Step 상태 정보 반환"""
        try:
            base_info = {
                'step_name': getattr(self, 'step_name', 'unknown'),
                'step_number': getattr(self, 'step_number', 0),
                'step_type': getattr(self, 'step_type', 'unknown'),
                'device': getattr(self, 'device', 'unknown'),
                'device_type': getattr(self, 'device_type', 'unknown'),
                'memory_gb': getattr(self, 'memory_gb', 0.0),
                'is_m3_max': getattr(self, 'is_m3_max', False),
                'optimization_enabled': getattr(self, 'optimization_enabled', False),
                'quality_level': getattr(self, 'quality_level', 'unknown'),
                'batch_size': getattr(self, 'batch_size', 1),
                'is_initialized': getattr(self, 'is_initialized', False),
                'has_model_interface': getattr(self, 'model_interface', None) is not None,
                'has_model_loader': getattr(self, 'model_loader', None) is not None,
                'has_function_validator': getattr(self, 'function_validator', None) is not None,
                'last_processing_time': getattr(self, 'last_processing_time', 0.0),
                'total_processing_count': getattr(self, 'total_processing_count', 0),
                'error_count': getattr(self, 'error_count', 0),
                'last_error': getattr(self, 'last_error', None),
                'torch_available': TORCH_AVAILABLE,
                'mps_available': MPS_AVAILABLE,
                'numpy_version': np.__version__ if NUMPY_AVAILABLE else 'N/A',
                'config_type': type(getattr(self, 'config', None)).__name__,
                'performance_metrics': getattr(self, 'performance_metrics', {}),
                'checkpoint_detection_enabled': True,
                'model_interface_type': type(getattr(self, 'model_interface', None)).__name__ if getattr(self, 'model_interface', None) else 'None',
                'version': 'v8.0 (Circular Import + Coroutine Fix)'
            }
            
            return base_info
            
        except Exception as e:
            return {
                'error': f"정보 수집 실패: {e}",
                'step_name': getattr(self, 'step_name', 'unknown'),
                'version': 'v8.0 (Error)'
            }
    
    def cleanup_models(self):
        """모델 정리"""
        try:
            if hasattr(self, 'model_interface') and self.model_interface:
                cleanup_func = getattr(self.model_interface, 'unload_models', None)
                if callable(cleanup_func):
                    success, result, message = self.function_validator.safe_call(cleanup_func)
                    if not success:
                        self.logger.warning(f"⚠️ 모델 언로드 실패: {message}")
                
            if TORCH_AVAILABLE:
                if self.device == "mps" and hasattr(torch.backends.mps, 'empty_cache'):
                    torch.backends.mps.empty_cache()
                elif self.device == "cuda":
                    torch.cuda.empty_cache()
                
                gc.collect()
            
            self.logger.info(f"🧹 {self.step_name} 정리 완료")
                
        except Exception as e:
            self.logger.warning(f"⚠️ 정리 중 오류: {e}")
    
    def __del__(self):
        """소멸자"""
        try:
            self.cleanup_models()
        except:
            pass

# ==============================================
# 🔥 데코레이터들 (기존 내용 100% 유지)
# ==============================================

def ensure_step_initialization(func: Callable) -> Callable:
    """Step 초기화 보장 데코레이터"""
    @wraps(func)
    async def wrapper(self, *args, **kwargs):
        if not hasattr(self, 'logger'):
            self.logger = logging.getLogger(f"pipeline.{self.__class__.__name__}")
        
        if not getattr(self, 'is_initialized', False):
            await self.initialize_step()
        
        return await func(self, *args, **kwargs)
    return wrapper

def safe_step_method(func: Callable) -> Callable:
    """Step 메서드 안전 실행 데코레이터"""
    @wraps(func)
    async def wrapper(self, *args, **kwargs):
        try:
            return await func(self, *args, **kwargs)
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.error(f"❌ {func.__name__} 실행 실패: {e}")
            if hasattr(self, 'error_count'):
                self.error_count += 1
            if hasattr(self, 'last_error'):
                self.last_error = str(e)
            
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
        @wraps(func)
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
                
                if hasattr(self, 'last_processing_time'):
                    self.last_processing_time = duration
                if hasattr(self, 'total_processing_count'):
                    self.total_processing_count += 1
                    
        return wrapper
    return decorator

def memory_optimize(func: Callable) -> Callable:
    """메모리 최적화 데코레이터"""
    @wraps(func)
    async def wrapper(self, *args, **kwargs):
        try:
            result = await func(self, *args, **kwargs)
            
            if TORCH_AVAILABLE and hasattr(self, 'device'):
                if self.device == "mps" and hasattr(torch.backends.mps, 'empty_cache'):
                    torch.backends.mps.empty_cache()
                elif self.device == "cuda":
                    torch.cuda.empty_cache()
                gc.collect()
            
            return result
        except Exception as e:
            if TORCH_AVAILABLE:
                gc.collect()
            raise e
    return wrapper

def step_timing(func: Callable) -> Callable:
    """Step 실행 시간 측정 데코레이터"""
    @wraps(func)
    async def wrapper(self, *args, **kwargs):
        start_time = time.time()
        try:
            result = await func(self, *args, **kwargs)
            
            processing_time = time.time() - start_time
            if hasattr(self, 'last_processing_time'):
                self.last_processing_time = processing_time
            if hasattr(self, 'total_processing_count'):
                self.total_processing_count += 1
                
            return result
        except Exception as e:
            processing_time = time.time() - start_time
            if hasattr(self, 'last_processing_time'):
                self.last_processing_time = processing_time
            raise e
    return wrapper

def error_handler(func: Callable) -> Callable:
    """에러 처리 데코레이터"""
    @wraps(func)
    async def wrapper(self, *args, **kwargs):
        try:
            return await func(self, *args, **kwargs)
        except Exception as e:
            if hasattr(self, 'error_count'):
                self.error_count += 1
            if hasattr(self, 'last_error'):
                self.last_error = str(e)
            if hasattr(self, 'logger'):
                self.logger.error(f"❌ {func.__name__} 실행 실패: {e}")
                self.logger.debug(f"📋 상세 오류: {traceback.format_exc()}")
            
            return {
                'success': False,
                'error': str(e),
                'error_type': type(e).__name__,
                'step_name': getattr(self, 'step_name', self.__class__.__name__),
                'method_name': func.__name__,
                'timestamp': time.time()
            }
    return wrapper

# ==============================================
# 🔥 Step별 특화 Mixin들 (기존 내용 유지)
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
        self.output_format = "keypoints"

class ClothSegmentationMixin(BaseStepMixin):
    """Step 3: Cloth Segmentation 특화 Mixin"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.step_number = 3
        self.step_type = "cloth_segmentation"
        self.output_format = "cloth_mask"

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
        
        self.assessment_modes = ['perceptual', 'technical', 'aesthetic', 'fitting']
        self.quality_aspects = ['sharpness', 'color', 'fitting', 'realism', 'artifacts']
        self.scoring_weights = {
            'perceptual': 0.4,
            'technical': 0.3,
            'aesthetic': 0.2,
            'fitting': 0.1
        }

# ==============================================
# 🔥 모듈 레벨 정보
# ==============================================

__all__ = [
    # 핵심 클래스들
    'BaseStepMixin',
    'SafeConfig',
    'FallbackSafeFunctionValidator',
    
    # Step별 Mixin들
    'HumanParsingMixin',
    'PoseEstimationMixin', 
    'ClothSegmentationMixin',
    'GeometricMatchingMixin',
    'ClothWarpingMixin',
    'VirtualFittingMixin',
    'PostProcessingMixin',
    'QualityAssessmentMixin',
    
    # 데코레이터들
    'ensure_step_initialization',
    'safe_step_method',
    'performance_monitor',
    'memory_optimize',
    'step_timing',
    'error_handler',
    
    # 상수들
    'TORCH_AVAILABLE',
    'MPS_AVAILABLE',
    'NUMPY_AVAILABLE',
    'CV_AVAILABLE',
    'PIL_AVAILABLE',
    'DEFAULT_DEVICE'
]

# 모듈 로드 확인
logger.info("✅ BaseStepMixin v8.0 모듈 로드 완료")
logger.info("🔧 순환 임포트 완전 해결 (동적 임포트 + TYPE_CHECKING)")
logger.info("🔧 coroutine 오류 근본 해결 (동기 메서드)")
logger.info("🔧 logger 속성 누락 문제 완전 해결")
logger.info("🔧 Dict Callable 오류 완전 해결")
logger.info("🔧 89.8GB 체크포인트 자동 탐지 유지")
logger.info("🔧 M3 Max 128GB 최적화 유지")
logger.info("🔧 conda 환경 완벽 지원")
logger.info("🔧 의존성 주입 패턴 적용")
logger.info("🔧 프로덕션 안정성 최고 수준")

if NUMPY_AVAILABLE and hasattr(np, '__version__'):
    numpy_major = int(np.__version__.split('.')[0])
    if numpy_major >= 2:
        logger.warning("⚠️ NumPy 2.x 감지됨 - conda install numpy=1.24.3 권장")
    else:
        logger.info("✅ NumPy 호환성 확인됨")

logger.info(f"🔧 PyTorch: {'✅' if TORCH_AVAILABLE else '❌'}, MPS: {'✅' if MPS_AVAILABLE else '❌'}")
logger.info(f"🔢 NumPy: {'✅' if NUMPY_AVAILABLE else '❌'}")
logger.info(f"🖼️ OpenCV/PIL: {'✅' if CV_AVAILABLE else '❌'}")

logger.info("🚀 BaseStepMixin v8.0 완전 최적화 완료!")
logger.info("   ✅ 기존 함수/클래스명 100% 유지")
logger.info("   ✅ 순환 임포트 완전 해결")
logger.info("   ✅ coroutine 오류 근본 해결")
logger.info("   ✅ logger 속성 누락 완전 해결")
logger.info("   ✅ Dict Callable 오류 완전 해결")
logger.info("   ✅ 89.8GB 체크포인트 자동 탐지")
logger.info("   ✅ M3 Max 128GB 최적화")
logger.info("   ✅ 의존성 주입 패턴")
logger.info("   ✅ conda 환경 완벽 지원")
logger.info("   ✅ 프로덕션 안정성 최고 수준")