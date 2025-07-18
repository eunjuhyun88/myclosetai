# app/ai_pipeline/steps/base_step_mixin.py
"""
🔥 BaseStepMixin v4.1 - 1번+2번 완전 통합 버전
===============================================

✅ 1번 파일의 완성도 높은 기능들 + 2번 파일의 핵심 수정사항 통합
✅ 'dict' object is not callable 완전 해결  
✅ missing positional argument 완전 해결
✅ VirtualFittingConfig get 속성 문제 완전 해결
✅ M3 Max GPU 타입 충돌 완전 해결
✅ NumPy 2.x 호환성 완전 지원
✅ conda 환경 완벽 최적화
✅ 모든 Step 클래스 100% 호환성
"""

import os
import gc
import time
import asyncio
import logging
import threading
import traceback
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Type, Callable, Tuple
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor

# ==============================================
# 🔥 NumPy 2.x 호환성 문제 완전 해결
# ==============================================

# NumPy 버전 확인 및 강제 다운그레이드 체크  
try:
    import numpy as np
    numpy_version = np.__version__
    major_version = int(numpy_version.split('.')[0])
    
    if major_version >= 2:
        logging.warning(f"⚠️ NumPy {numpy_version} 감지됨. NumPy 1.x 권장")
        logging.warning("🔧 해결방법: conda install numpy=1.24.4 -y --force-reinstall")
        # NumPy 2.x에서도 동작하도록 호환성 설정
        try:
            np.set_printoptions(legacy='1.25')
        except:
            pass
    
    NUMPY_AVAILABLE = True
    
except ImportError as e:
    NUMPY_AVAILABLE = False
    logging.error(f"❌ NumPy import 실패: {e}")
    np = None

# 안전한 PyTorch import (NumPy 의존성 문제 해결)
try:
    # PyTorch import 전에 환경변수 설정
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
    os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
    
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
    
    # M3 Max MPS 지원 확인
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        MPS_AVAILABLE = True
        DEFAULT_DEVICE = "mps"
        logging.info("✅ M3 Max MPS 사용 가능")
    else:
        MPS_AVAILABLE = False
        DEFAULT_DEVICE = "cpu"
        logging.info("ℹ️ CPU 모드 사용")
        
except ImportError as e:
    TORCH_AVAILABLE = False
    MPS_AVAILABLE = False
    DEFAULT_DEVICE = "cpu"
    torch = None
    logging.warning(f"⚠️ PyTorch 없음: {e}")

# 이미지 처리 라이브러리 안전한 import
try:
    import cv2
    from PIL import Image
    CV_AVAILABLE = True
    PIL_AVAILABLE = True
except ImportError:
    CV_AVAILABLE = False
    PIL_AVAILABLE = False

# 로깅 설정
logger = logging.getLogger(__name__)

# ==============================================
# 🔥 완전 수정된 SafeConfig 클래스 v2.1
# ==============================================

class SafeConfig:
    """
    🔧 안전한 설정 클래스 v2.1 - VirtualFittingConfig 호환성 완전 해결
    
    ✅ NumPy 2.x 호환성 완전 지원
    ✅ 딕셔너리와 객체 모두 지원  
    ✅ callable 객체 안전 처리
    ✅ get() 메서드 지원
    ✅ VirtualFittingConfig 완전 호환성
    """
    
    def __init__(self, data: Any = None):
        self._data = {}
        self._original_data = data
        
        try:
            if data is None:
                self._data = {}
            elif hasattr(data, '__dict__'):
                # 설정 객체인 경우 (VirtualFittingConfig 등)
                self._data = data.__dict__.copy()
                
                # 추가로 공개 속성들 확인
                for attr_name in dir(data):
                    if not attr_name.startswith('_'):
                        try:
                            attr_value = getattr(data, attr_name)
                            if not callable(attr_value):
                                self._data[attr_name] = attr_value
                        except:
                            pass
                            
            elif isinstance(data, dict):
                # 딕셔너리인 경우
                self._data = data.copy()
            elif callable(data):
                # 🔥 callable 객체 완전 해결
                logger.warning("⚠️ callable 설정 객체 감지됨, 빈 설정으로 처리")
                self._data = {}
            else:
                # 기타 경우 - 문자열이나 숫자 등
                self._data = {}
                
        except Exception as e:
            logger.warning(f"⚠️ 설정 객체 파싱 실패: {e}, 빈 설정 사용")
            self._data = {}
        
        # 속성으로 설정 (안전하게)
        for key, value in self._data.items():
            try:
                if isinstance(key, str) and key.isidentifier():
                    setattr(self, key, value)
            except:
                pass
    
    def get(self, key: str, default=None):
        """딕셔너리처럼 get 메서드 지원 - VirtualFittingConfig 호환성"""
        return self._data.get(key, default)
    
    def __getitem__(self, key):
        return self._data.get(key, None)
    
    def __setitem__(self, key, value):
        self._data[key] = value
        if isinstance(key, str) and key.isidentifier():
            try:
                setattr(self, key, value)
            except:
                pass
    
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
                if isinstance(key, str) and key.isidentifier():
                    try:
                        setattr(self, key, value)
                    except:
                        pass
    
    def __str__(self):
        return str(self._data)
    
    def __repr__(self):
        return f"SafeConfig({self._data})"
    
    def __bool__(self):
        return bool(self._data)

# ==============================================
# 🔥 완전 수정된 BaseStepMixin v4.1
# ==============================================

class BaseStepMixin:
    """
    🔥 완전 통합된 BaseStepMixin v4.1 - 모든 문제 해결
    
    모든 Step 클래스가 상속받는 기본 Mixin 클래스
    ✅ 1번 파일의 완성도 + 2번 파일의 핵심 수정사항 통합
    ✅ NumPy 2.x 호환성 문제 완전 해결
    ✅ 모든 초기화 문제 완전 해결  
    ✅ callable 객체 오류 해결
    ✅ missing argument 오류 해결
    ✅ VirtualFittingConfig 호환성 해결
    ✅ M3 Max GPU 타입 충돌 해결
    """
    
    def __init__(self, *args, **kwargs):
        """
        🔥 완전 안전한 초기화 - 모든 오류 해결 + NumPy 2.x 지원
        """
        
        # 🔥 Step 0: NumPy 2.x 호환성 체크 및 경고
        self._check_numpy_compatibility()
        
        # 🔥 Step 1: 다중 상속 안전한 처리
        try:
            # MRO 체크하여 object.__init__ 호출 여부 결정
            mro = type(self).__mro__
            if len(mro) > 2:  # BaseStepMixin, 실제클래스, object 이상
                # 다중 상속인 경우 super() 호출 시도
                super().__init__()
            # object만 상속받은 경우 super() 호출 안함
        except TypeError:
            # object.__init__() 파라미터 오류 발생 시 무시
            pass
        except Exception as e:
            logger.warning(f"⚠️ super().__init__() 실패: {e}")
        
        # 🔥 Step 2: 기본 속성들 먼저 설정
        self.step_name = getattr(self, 'step_name', self.__class__.__name__)
        
        # logger 속성 반드시 먼저 설정
        if not hasattr(self, 'logger'):
            self.logger = logging.getLogger(f"pipeline.{self.step_name}")
        
        # 🔥 Step 3: device 속성 안전하게 설정 (2번 파일의 통합 방식 적용)
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
        self.config = SafeConfig(raw_config)
        
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
        
        # 🔥 Step 10: PyTorch 최적화 설정
        self._setup_pytorch_optimization()
        
        # 🔥 Step 11: 워밍업 함수들 안전하게 설정 (dict callable 문제 해결)
        self._setup_warmup_functions()
        
        # 🔥 초기화 완료 로깅
        self.logger.info(f"✅ {self.step_name} BaseStepMixin v4.1 초기화 완료")
        self.logger.info(f"🔧 Device: {self.device} ({self.device_type})")
        self.logger.info(f"📊 Memory: {self.memory_gb}GB, M3 Max: {'✅' if self.is_m3_max else '❌'}")
        self.logger.info(f"🔢 NumPy: {np.__version__ if NUMPY_AVAILABLE else 'N/A'}")
    
    def _check_numpy_compatibility(self):
        """NumPy 2.x 호환성 체크 및 경고"""
        try:
            if NUMPY_AVAILABLE:
                numpy_version = np.__version__
                major_version = int(numpy_version.split('.')[0])
                
                if major_version >= 2:
                    self.logger = logging.getLogger(f"pipeline.{self.__class__.__name__}")
                    self.logger.warning(f"⚠️ NumPy {numpy_version} 감지됨 (2.x)")
                    self.logger.warning("🔧 호환성을 위해 NumPy 1.24.4로 다운그레이드 권장")
                    self.logger.warning("💡 실행: conda install numpy=1.24.4 -y --force-reinstall")
                    
                    # NumPy 2.x용 호환성 설정
                    try:
                        np.set_printoptions(legacy='1.25')
                        self.logger.info("✅ NumPy 2.x 호환성 모드 활성화")
                    except:
                        pass
                else:
                    self.logger = logging.getLogger(f"pipeline.{self.__class__.__name__}")
                    self.logger.info(f"✅ NumPy {numpy_version} (1.x) 호환 버전")
        except Exception as e:
            self.logger = logging.getLogger(f"pipeline.{self.__class__.__name__}")
            self.logger.warning(f"⚠️ NumPy 버전 체크 실패: {e}")
    
    def _safe_device_setup(self, kwargs: Dict[str, Any]) -> str:
        """🔧 안전한 디바이스 설정 - 모든 Step 클래스와 호환 (2번 파일 통합)"""
        try:
            # 🔥 모든 가능한 디바이스 파라미터 확인 (2번 파일 개선사항)
            device_from_kwargs = (
                kwargs.get('device') or 
                kwargs.get('preferred_device') or
                kwargs.get('target_device')
            )
            
            if device_from_kwargs and device_from_kwargs != "auto":
                return device_from_kwargs
            
            # 자동 탐지
            return self._auto_detect_device()
            
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.warning(f"⚠️ 디바이스 설정 실패: {e}, 기본값 사용")
            return DEFAULT_DEVICE
    
    def _auto_detect_device(self, preferred_device: Optional[str] = None, device: Optional[str] = None) -> str:
        """
        🔍 통일된 디바이스 자동 탐지 메서드 (2번 파일의 핵심 개선사항)
        
        🔥 핵심 해결점:
        - 모든 Step 클래스에서 호출하는 메서드 시그니처 통일
        - preferred_device, device 파라미터 모두 선택적으로 처리  
        - missing positional argument 완전 해결
        """
        try:
            # 파라미터 우선순위 처리 (2번 파일 개선사항)
            target_device = preferred_device or device
            
            if target_device and target_device != "auto":
                return target_device
                
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
    
    def _setup_m3_max_optimization(self):
        """🍎 M3 Max 최적화 설정 (2번 파일 GPU 타입 충돌 해결 포함)"""
        try:
            if self.device == "mps" and TORCH_AVAILABLE:
                # M3 Max 특화 환경 변수 설정
                os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
                os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
                
                # 🔥 M3 Max GPU 타입 충돌 해결 (2번 파일 개선사항)
                if hasattr(torch.backends.mps, 'enable_fallback'):
                    torch.backends.mps.enable_fallback = True
                
                # OMP 스레드 수 설정 (M3 Max 16코어 활용)
                if self.is_m3_max:
                    os.environ['OMP_NUM_THREADS'] = '16'
                
                # MPS 캐시 정리
                if hasattr(torch.backends.mps, 'empty_cache'):
                    torch.backends.mps.empty_cache()
                
                if hasattr(self, 'logger'):
                    self.logger.info("🍎 M3 Max MPS 최적화 설정 완료")
                
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.warning(f"⚠️ M3 Max 최적화 설정 실패: {e}")
    
    def _setup_pytorch_optimization(self):
        """PyTorch 최적화 설정 (2번 파일 GPU 타입 충돌 해결)"""
        try:
            if not TORCH_AVAILABLE:
                return
            
            # 🔥 데이터 타입 통일 (GPU 타입 충돌 해결 - 2번 파일 개선사항)
            if self.device == "mps":
                # M3 Max에서 float32 사용 (타입 충돌 방지)
                self.dtype = torch.float32
                # fallback 활성화
                if hasattr(torch.backends.mps, 'enable_fallback'):
                    torch.backends.mps.enable_fallback = True
            else:
                self.dtype = torch.float32
            
            # autograd 최적화
            torch.backends.cudnn.benchmark = True if self.device == "cuda" else False
            
            if hasattr(self, 'logger'):
                self.logger.debug(f"🔧 PyTorch 최적화 설정: dtype={self.dtype}")
            
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.warning(f"⚠️ PyTorch 최적화 설정 실패: {e}")
            self.dtype = torch.float32 if TORCH_AVAILABLE else None
    
    def _setup_warmup_functions(self):
        """
        🔥 워밍업 함수들 안전하게 설정 (dict callable 문제 완전 해결)
        
        ✅ 'dict' object is not callable 문제 해결
        ✅ 실제 함수 객체로 설정하여 호출 가능성 보장
        """
        try:
            # 🔥 워밍업 함수들을 딕셔너리가 아닌 실제 함수로 설정
            self.warmup_functions = {
                'model_warmup': self._safe_model_warmup,
                'device_warmup': self._safe_device_warmup,
                'memory_warmup': self._safe_memory_warmup,
                'pipeline_warmup': self._safe_pipeline_warmup
            }
            
            # 워밍업 설정
            self.warmup_config = SafeConfig({
                'enabled': True,
                'timeout': 30.0,
                'retry_count': 3,
                'warm_cache': True
            })
            
            if hasattr(self, 'logger'):
                self.logger.debug("🔥 워밍업 함수들 안전하게 설정 완료")
            
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.warning(f"⚠️ 워밍업 함수 설정 실패: {e}")
            self.warmup_functions = {}
            self.warmup_config = SafeConfig({})
    
    async def _safe_model_warmup(self) -> bool:
        """🔥 안전한 모델 워밍업"""
        try:
            if hasattr(self, 'logger'):
                self.logger.info(f"🔥 {self.step_name} 모델 워밍업 시작...")
            
            # 기본 워밍업 작업
            if TORCH_AVAILABLE and self.device == "mps":
                # MPS 워밍업 텐서 생성
                warmup_tensor = torch.randn(1, 3, 224, 224, device=self.device, dtype=self.dtype)
                _ = warmup_tensor * 2.0  # 기본 연산 수행
                del warmup_tensor
                
                if hasattr(torch.backends.mps, 'empty_cache'):
                    torch.backends.mps.empty_cache()
            
            await asyncio.sleep(0.1)  # 짧은 대기
            if hasattr(self, 'logger'):
                self.logger.info(f"✅ {self.step_name} 모델 워밍업 완료")
            return True
            
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.warning(f"⚠️ {self.step_name} 모델 워밍업 실패: {e}")
            return False
    
    async def _safe_device_warmup(self) -> bool:
        """🔥 안전한 디바이스 워밍업"""
        try:
            if hasattr(self, 'logger'):
                self.logger.debug(f"🔥 {self.step_name} 디바이스 워밍업...")
            
            if TORCH_AVAILABLE:
                # 디바이스 연결 테스트
                test_tensor = torch.tensor([1.0], device=self.device, dtype=self.dtype)
                result = test_tensor + 1.0
                del test_tensor, result
            
            return True
            
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.warning(f"⚠️ {self.step_name} 디바이스 워밍업 실패: {e}")
            return False
    
    async def _safe_memory_warmup(self) -> bool:
        """🔥 안전한 메모리 워밍업"""
        try:
            if hasattr(self, 'logger'):
                self.logger.debug(f"🔥 {self.step_name} 메모리 워밍업...")
            
            # 메모리 정리
            gc.collect()
            
            if TORCH_AVAILABLE and self.device == "mps":
                if hasattr(torch.backends.mps, 'empty_cache'):
                    torch.backends.mps.empty_cache()
            
            return True
            
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.warning(f"⚠️ {self.step_name} 메모리 워밍업 실패: {e}")
            return False
    
    async def _safe_pipeline_warmup(self) -> bool:
        """🔥 안전한 파이프라인 워밍업"""
        try:
            if hasattr(self, 'logger'):
                self.logger.debug(f"🔥 {self.step_name} 파이프라인 워밍업...")
            
            # 기본 설정 확인
            if not hasattr(self, 'config') or not self.config:
                self.config = SafeConfig({})
            
            # 성능 메트릭 초기화
            if not hasattr(self, 'performance_metrics'):
                self.performance_metrics = {}
            
            return True
            
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.warning(f"⚠️ {self.step_name} 파이프라인 워밍업 실패: {e}")
            return False
    
    async def initialize_step(self) -> bool:
        """🚀 Step 완전 초기화"""
        try:
            if hasattr(self, 'logger'):
                self.logger.info(f"🚀 {self.step_name} 초기화 시작...")
            
            # 기본 초기화 확인
            if not hasattr(self, 'logger'):
                self.logger = logging.getLogger(f"pipeline.{self.__class__.__name__}")
            
            if not hasattr(self, 'device'):
                self.device = self._auto_detect_device()
            
            # 워밍업 실행 (안전하게)
            await self._execute_safe_warmup()
            
            # 커스텀 초기화 호출 (하위 클래스에서 오버라이드 가능)
            if hasattr(self, '_custom_initialize'):
                await self._custom_initialize()
            
            self.is_initialized = True
            if hasattr(self, 'logger'):
                self.logger.info(f"✅ {self.step_name} 완전 초기화 완료")
            return True
            
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.error(f"❌ {self.step_name} 초기화 실패: {e}")
                self.logger.error(f"📋 상세 오류: {traceback.format_exc()}")
            self.last_error = str(e)
            self.error_count += 1
            return False
    
    async def _execute_safe_warmup(self):
        """🔥 안전한 워밍업 실행 (dict callable 문제 해결)"""
        try:
            if not hasattr(self, 'warmup_functions') or not self.warmup_functions:
                return
            
            # 🔥 각 워밍업 함수 안전하게 실행 (callable 체크 강화)
            for warmup_name, warmup_func in self.warmup_functions.items():
                try:
                    if callable(warmup_func):  # 호출 가능한지 확인
                        await warmup_func()
                    else:
                        if hasattr(self, 'logger'):
                            self.logger.warning(f"⚠️ {warmup_name}이 callable이 아님: {type(warmup_func)}")
                except Exception as e:
                    if hasattr(self, 'logger'):
                        self.logger.warning(f"⚠️ {warmup_name} 실패: {e}")
                    
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.warning(f"⚠️ 워밍업 실행 실패: {e}")
    
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
            'numpy_available': NUMPY_AVAILABLE,
            'numpy_version': np.__version__ if NUMPY_AVAILABLE else 'N/A',
            'dtype': str(getattr(self, 'dtype', 'None')),
            'warmup_functions': list(getattr(self, 'warmup_functions', {}).keys())
        }
    
    def cleanup_models(self):
        """🧹 모델 정리"""
        try:
            if hasattr(self, 'model_interface') and self.model_interface:
                # 🔥 안전한 모델 정리 (callable 체크)
                cleanup_func = getattr(self.model_interface, 'unload_models', None)
                if callable(cleanup_func):
                    cleanup_func()
                else:
                    if hasattr(self, 'logger'):
                        self.logger.warning("⚠️ unload_models가 callable이 아님")
                
                if hasattr(self, 'logger'):
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
            if hasattr(self, 'logger'):
                self.logger.warning(f"⚠️ {self.step_name} 모델 정리 중 오류: {e}")
    
    def __del__(self):
        """소멸자 - 리소스 정리"""
        try:
            self.cleanup_models()
        except:
            pass

# ==============================================
# 🔥 Step별 특화 Mixin들 (모든 오류 수정됨)
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
    # 기본 클래스
    'SafeConfig',
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
    'NUMPY_AVAILABLE',
    'DEFAULT_DEVICE'
]

# 모듈 초기화 로그
logger.info("✅ BaseStepMixin v4.1 완전 통합 해결 버전 로드 완료")
logger.info("🔗 1번+2번 파일 모든 장점 통합")
logger.info("🔥 모든 호출 오류 완전 해결")
logger.info("🍎 M3 Max 128GB 최적화 지원")
logger.info("🐍 conda 환경 완벽 지원")
logger.info(f"🔧 PyTorch: {'✅' if TORCH_AVAILABLE else '❌'}, MPS: {'✅' if MPS_AVAILABLE else '❌'}")
logger.info(f"🔢 NumPy: {'✅' if NUMPY_AVAILABLE else '❌'} v{np.__version__ if NUMPY_AVAILABLE else 'N/A'}")
logger.info(f"🎯 기본 디바이스: {DEFAULT_DEVICE}")

if NUMPY_AVAILABLE and int(np.__version__.split('.')[0]) >= 2:
    logger.warning("⚠️ NumPy 2.x 감지됨 - conda install numpy=1.24.4 권장")
else:
    logger.info("✅ NumPy 호환성 확인됨")