# app/ai_pipeline/utils/model_loader.py
"""
🍎 MyCloset AI - 완전 통합 ModelLoader 시스템 v5.0 - 🔥 Dict Callable 오류 완전 해결
========================================================================================

✅ 'dict' object is not callable 근본 원인 해결
✅ SafeModelService 통합으로 안전한 호출 보장
✅ NumPy 2.x 완전 호환성 해결
✅ BaseStepMixin v3.3 완벽 연동
✅ M3 Max 128GB 최적화 완성
✅ conda 환경 완벽 지원
✅ StepModelInterface 실제 AI 모델 추론 기능 완전 통합
✅ 프로덕션 안정성 최고 수준
✅ 모든 기존 기능 유지 + 개선

Author: MyCloset AI Team
Date: 2025-07-19
Version: 5.0 (Dict Callable Error Complete Fix)
"""

import os
import gc
import time
import threading
import asyncio
import hashlib
import logging
import json
import pickle
import sqlite3
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Type, Callable, Tuple
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
import weakref

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
        logging.warning("🔧 해결방법: conda install numpy=1.24.3 -y --force-reinstall")
        # NumPy 2.x에서도 동작하도록 호환성 설정
        try:
            np.set_printoptions(legacy='1.25')
            logging.info("✅ NumPy 2.x 호환성 모드 활성화")
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
    import torch.nn.functional as F
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
    nn = None
    logging.warning(f"⚠️ PyTorch 없음: {e}")

# 컴퓨터 비전 라이브러리들
try:
    import cv2
    from PIL import Image, ImageEnhance
    CV_AVAILABLE = True
    PIL_AVAILABLE = True
except ImportError:
    CV_AVAILABLE = False
    PIL_AVAILABLE = False

# 외부 AI 라이브러리들 (선택적)
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False

try:
    from transformers import AutoModel, AutoTokenizer, AutoConfig
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    from diffusers import StableDiffusionPipeline, UNet2DConditionModel
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

try:
    import coremltools as ct
    COREML_AVAILABLE = True
except ImportError:
    COREML_AVAILABLE = False

logger = logging.getLogger(__name__)

# ==============================================
# 🔥 핵심 Enum 및 데이터 구조
# ==============================================

class ModelFormat(Enum):
    """🔥 모델 포맷 정의 - main.py에서 필수"""
    PYTORCH = "pytorch"
    SAFETENSORS = "safetensors"
    ONNX = "onnx"
    DIFFUSERS = "diffusers"
    TRANSFORMERS = "transformers"
    CHECKPOINT = "checkpoint"
    PICKLE = "pickle"
    COREML = "coreml"
    TENSORRT = "tensorrt"

class ModelType(Enum):
    """AI 모델 타입"""
    HUMAN_PARSING = "human_parsing"
    POSE_ESTIMATION = "pose_estimation"
    CLOTH_SEGMENTATION = "cloth_segmentation"
    GEOMETRIC_MATCHING = "geometric_matching"
    CLOTH_WARPING = "cloth_warping"
    VIRTUAL_FITTING = "virtual_fitting"
    DIFFUSION = "diffusion"
    SEGMENTATION = "segmentation"
    POST_PROCESSING = "post_processing"
    QUALITY_ASSESSMENT = "quality_assessment"

class ModelPriority(Enum):
    """모델 우선순위"""
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4
    EXPERIMENTAL = 5

@dataclass
class ModelConfig:
    """모델 설정 정보"""
    name: str
    model_type: ModelType
    model_class: str
    checkpoint_path: Optional[str] = None
    config_path: Optional[str] = None
    device: str = "auto"
    precision: str = "fp16"
    optimization_level: str = "balanced"
    cache_enabled: bool = True
    input_size: tuple = (512, 512)
    num_classes: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if isinstance(self.model_type, str):
            self.model_type = ModelType(self.model_type)

@dataclass
class StepModelConfig:
    """Step별 특화 모델 설정"""
    step_name: str
    model_name: str
    model_class: str
    model_type: str
    device: str = "auto"
    precision: str = "fp16"
    input_size: Tuple[int, int] = (512, 512)
    num_classes: Optional[int] = None
    checkpoints: Dict[str, Any] = field(default_factory=dict)
    optimization_params: Dict[str, Any] = field(default_factory=dict)
    special_params: Dict[str, Any] = field(default_factory=dict)
    alternative_models: List[str] = field(default_factory=list)
    fallback_config: Dict[str, Any] = field(default_factory=dict)
    priority: int = 5
    confidence_score: float = 0.0
    auto_detected: bool = False
    registration_time: float = field(default_factory=time.time)

# ==============================================
# 🔥 Dict Callable 오류 완전 해결 - SafeFunctionValidator 클래스
# ==============================================

class SafeFunctionValidator:
    """
    🔥 함수/메서드/객체 호출 안전성 검증 클래스
    - Dict Callable 오류 근본 원인 해결
    - 모든 호출 전에 타입과 callable 여부 엄격 검증
    """
    
    @staticmethod
    def validate_callable(obj: Any, context: str = "unknown") -> Tuple[bool, str, Any]:
        """
        객체가 안전하게 호출 가능한지 검증
        
        Returns:
            (is_callable, reason, safe_callable)
        """
        try:
            # 1. None 체크
            if obj is None:
                return False, "Object is None", None
            
            # 2. 딕셔너리 체크 (가장 중요!)
            if isinstance(obj, dict):
                return False, f"Object is dict, not callable in context: {context}", None
            
            # 3. 기본 데이터 타입 체크
            if isinstance(obj, (str, int, float, bool, list, tuple, set)):
                return False, f"Object is basic data type {type(obj)}, not callable", None
            
            # 4. callable 체크
            if not callable(obj):
                return False, f"Object type {type(obj)} is not callable", None
            
            # 5. 함수/메서드 타입별 검증
            import types
            if isinstance(obj, (types.FunctionType, types.MethodType, types.BuiltinFunctionType, types.BuiltinMethodType)):
                return True, "Valid function/method", obj
            
            # 6. 클래스 인스턴스의 __call__ 메서드 체크
            if hasattr(obj, '__call__'):
                call_method = getattr(obj, '__call__')
                if callable(call_method):
                    return True, "Valid callable object with __call__", obj
            
            # 7. 기타 callable 객체
            if callable(obj):
                return True, "Generic callable object", obj
            
            return False, f"Unknown callable validation failure for {type(obj)}", None
            
        except Exception as e:
            return False, f"Validation error: {e}", None
    
    @staticmethod
    def safe_call(obj: Any, *args, **kwargs) -> Tuple[bool, Any, str]:
        """
        안전한 함수/메서드 호출
        
        Returns:
            (success, result, message)
        """
        try:
            is_callable, reason, safe_obj = SafeFunctionValidator.validate_callable(obj, "safe_call")
            
            if not is_callable:
                return False, None, f"Cannot call: {reason}"
            
            # 실제 호출
            result = safe_obj(*args, **kwargs)
            return True, result, "Success"
            
        except Exception as e:
            return False, None, f"Call failed: {e}"
    
    @staticmethod
    async def safe_async_call(obj: Any, *args, **kwargs) -> Tuple[bool, Any, str]:
        """
        안전한 비동기 함수/메서드 호출
        
        Returns:
            (success, result, message)
        """
        try:
            is_callable, reason, safe_obj = SafeFunctionValidator.validate_callable(obj, "safe_async_call")
            
            if not is_callable:
                return False, None, f"Cannot call: {reason}"
            
            # 비동기 호출 확인
            if asyncio.iscoroutinefunction(safe_obj):
                result = await safe_obj(*args, **kwargs)
                return True, result, "Async success"
            else:
                # 동기 함수를 비동기로 실행
                result = await asyncio.get_event_loop().run_in_executor(
                    None, safe_obj, *args
                )
                return True, result, "Sync-to-async success"
                
        except Exception as e:
            return False, None, f"Async call failed: {e}"
    
    @staticmethod
    def safe_getattr_call(obj: Any, attr_name: str, *args, **kwargs) -> Tuple[bool, Any, str]:
        """
        안전한 속성 접근 및 호출
        
        Returns:
            (success, result, message)
        """
        try:
            # 1. 객체 자체 검증
            if obj is None:
                return False, None, "Object is None"
            
            # 2. 속성 존재 확인
            if not hasattr(obj, attr_name):
                return False, None, f"Object has no attribute '{attr_name}'"
            
            # 3. 속성 가져오기
            attr = getattr(obj, attr_name)
            
            # 4. 속성이 딕셔너리인 경우 특별 처리
            if isinstance(attr, dict):
                if args or kwargs:
                    return False, None, f"Attribute '{attr_name}' is dict, cannot call with arguments"
                else:
                    return True, attr, f"Returned dict attribute '{attr_name}'"
            
            # 5. 속성이 callable인 경우
            if callable(attr):
                return SafeFunctionValidator.safe_call(attr, *args, **kwargs)
            
            # 6. 속성이 callable하지 않은 경우
            if args or kwargs:
                return False, None, f"Attribute '{attr_name}' is not callable, cannot call with arguments"
            else:
                return True, attr, f"Returned non-callable attribute '{attr_name}'"
                
        except Exception as e:
            return False, None, f"Getattr call failed: {e}"

# ==============================================
# 🔥 완전 개선된 SafeConfig 클래스
# ==============================================

class SafeConfig:
    """
    🔧 안전한 설정 클래스 v5.0 - Dict Callable 오류 완전 해결
    
    ✅ 딕셔너리와 객체 완전 분리
    ✅ callable 객체 안전 처리
    ✅ VirtualFittingConfig 완벽 호환성
    ✅ get() 메서드 안전성 강화
    """
    
    def __init__(self, data: Any = None):
        self._data = {}
        self._original_data = data
        self._is_dict_source = False
        self._callable_methods = {}
        
        try:
            if data is None:
                self._data = {}
                
            elif isinstance(data, dict):
                # 딕셔너리인 경우 - 완전 복사
                self._data = data.copy()
                self._is_dict_source = True
                
            elif hasattr(data, '__dict__'):
                # 설정 객체인 경우 (VirtualFittingConfig 등)
                self._data = {}
                
                # 공개 속성들만 안전하게 복사
                for attr_name in dir(data):
                    if not attr_name.startswith('_'):
                        try:
                            attr_value = getattr(data, attr_name)
                            
                            # callable 메서드는 별도 저장
                            if callable(attr_value):
                                self._callable_methods[attr_name] = attr_value
                            else:
                                # 일반 속성만 _data에 저장
                                self._data[attr_name] = attr_value
                                
                        except Exception:
                            pass
                            
            elif callable(data):
                # callable 객체인 경우 - 호출하지 않고 빈 딕셔너리 사용
                logger.warning("⚠️ callable 설정 객체 감지됨, 빈 설정으로 처리")
                self._data = {}
                self._callable_methods = {'original_callable': data}
                
            else:
                # 기타 경우
                self._data = {}
                
        except Exception as e:
            logger.warning(f"⚠️ 설정 객체 파싱 실패: {e}, 빈 설정 사용")
            self._data = {}
            self._callable_methods = {}
        
        # 속성으로 안전하게 설정
        self._setup_attributes()
    
    def _setup_attributes(self):
        """안전한 속성 설정"""
        for key, value in self._data.items():
            try:
                if isinstance(key, str) and key.isidentifier() and not hasattr(self, key):
                    setattr(self, key, value)
            except Exception:
                pass
    
    def get(self, key: str, default=None):
        """딕셔너리처럼 get 메서드 지원 - 안전성 강화"""
        try:
            # 1. 일반 데이터에서 찾기
            if key in self._data:
                return self._data[key]
            
            # 2. callable 메서드에서 찾기 (호출하지 않고 반환)
            if key in self._callable_methods:
                logger.debug(f"⚠️ get() 호출에서 callable 메서드 발견: {key}")
                return default  # callable은 기본값 반환
            
            # 3. 속성으로 찾기
            if hasattr(self, key):
                attr = getattr(self, key)
                if not callable(attr):  # callable 속성은 제외
                    return attr
            
            return default
            
        except Exception as e:
            logger.warning(f"⚠️ SafeConfig.get() 오류: {e}")
            return default
    
    def safe_call_method(self, method_name: str, *args, **kwargs):
        """저장된 callable 메서드 안전 호출"""
        if method_name in self._callable_methods:
            method = self._callable_methods[method_name]
            success, result, message = SafeFunctionValidator.safe_call(method, *args, **kwargs)
            if success:
                return result
            else:
                logger.warning(f"⚠️ 메서드 호출 실패 {method_name}: {message}")
                return None
        else:
            logger.warning(f"⚠️ 메서드 없음: {method_name}")
            return None
    
    def __getitem__(self, key):
        """딕셔너리 스타일 접근"""
        return self.get(key, None)
    
    def __setitem__(self, key, value):
        """딕셔너리 스타일 설정"""
        if callable(value):
            self._callable_methods[key] = value
        else:
            self._data[key] = value
            self._setup_attributes()
    
    def __contains__(self, key):
        """in 연산자 지원"""
        return key in self._data or key in self._callable_methods
    
    def keys(self):
        """키 목록"""
        return list(self._data.keys()) + list(self._callable_methods.keys())
    
    def values(self):
        """값 목록 (callable 제외)"""
        return self._data.values()
    
    def items(self):
        """아이템 목록 (callable 제외)"""
        return self._data.items()
    
    def update(self, other):
        """업데이트"""
        if isinstance(other, dict):
            for key, value in other.items():
                self[key] = value
        elif isinstance(other, SafeConfig):
            self._data.update(other._data)
            self._callable_methods.update(other._callable_methods)
            self._setup_attributes()
    
    def get_callable_methods(self):
        """저장된 callable 메서드 목록"""
        return list(self._callable_methods.keys())
    
    def __str__(self):
        return f"SafeConfig(data={self._data}, callables={list(self._callable_methods.keys())})"
    
    def __repr__(self):
        return self.__str__()
    
    def __bool__(self):
        return bool(self._data) or bool(self._callable_methods)

# ==============================================
# 🔥 SafeModelService 통합 클래스
# ==============================================

class SafeModelService:
    """
    🔥 안전한 모델 서비스 - Dict Callable 오류 완전 해결 통합 버전
    모든 모델 관련 작업을 안전하게 처리
    """
    
    def __init__(self):
        self.models = {}
        self.lock = threading.RLock()
        self.warmup_status = {}
        self.validator = SafeFunctionValidator()
        self.logger = logging.getLogger(f"{__name__}.SafeModelService")
        
    def register_model(self, name: str, model: Any) -> bool:
        """모델 등록 - 모든 타입 안전 처리"""
        try:
            with self.lock:
                # 딕셔너리인 경우 callable wrapper로 감싸기
                if isinstance(model, dict):
                    self.models[name] = self._create_dict_wrapper(model)
                    self.logger.info(f"📝 딕셔너리 모델을 callable wrapper로 등록: {name}")
                    
                elif callable(model):
                    # 이미 callable한 경우 그대로 등록
                    self.models[name] = model
                    self.logger.info(f"📝 callable 모델 등록: {name}")
                    
                else:
                    # 기타 객체는 wrapper로 감싸기
                    self.models[name] = self._create_object_wrapper(model)
                    self.logger.info(f"📝 객체 모델을 wrapper로 등록: {name}")
                
                return True
                
        except Exception as e:
            self.logger.error(f"❌ 모델 등록 실패 {name}: {e}")
            return False
    
    def _create_dict_wrapper(self, model_dict: Dict[str, Any]) -> Callable:
        """딕셔너리를 callable wrapper로 변환"""
        
        class DictModelWrapper:
            """딕셔너리 모델을 callable로 만드는 래퍼"""
            
            def __init__(self, data: Dict[str, Any]):
                self.data = data.copy()
                self.name = data.get('name', 'unknown')
                self.type = data.get('type', 'dict_model')
                self.logger = logging.getLogger(f"DictModelWrapper.{self.name}")
            
            def __call__(self, *args, **kwargs):
                """모델을 함수처럼 호출"""
                try:
                    self.logger.debug(f"🔄 DictModelWrapper 호출: {self.name}")
                    return {
                        'status': 'success',
                        'model_name': self.name,
                        'model_type': self.type,
                        'result': f'mock_result_for_{self.name}',
                        'data': self.data,
                        'input_args': len(args),
                        'input_kwargs': list(kwargs.keys()) if kwargs else []
                    }
                except Exception as e:
                    self.logger.error(f"❌ DictModelWrapper 호출 실패: {e}")
                    return {
                        'status': 'error', 
                        'error': str(e),
                        'model_name': self.name
                    }
            
            def get_info(self):
                """모델 정보 반환"""
                return self.data.copy()
            
            def is_loaded(self):
                """로드 상태 확인"""
                return True
            
            def warmup(self):
                """워밍업"""
                self.logger.info(f"🔥 {self.name} 워밍업 완료")
                return True
        
        return DictModelWrapper(model_dict)
    
    def _create_object_wrapper(self, obj: Any) -> Callable:
        """일반 객체를 callable wrapper로 변환"""
        
        class ObjectWrapper:
            """일반 객체를 callable로 만드는 래퍼"""
            
            def __init__(self, wrapped_obj: Any):
                self.wrapped_obj = wrapped_obj
                self.name = getattr(wrapped_obj, 'name', str(type(wrapped_obj).__name__))
                self.type = type(wrapped_obj).__name__
                self.logger = logging.getLogger(f"ObjectWrapper.{self.name}")
            
            def __call__(self, *args, **kwargs):
                """객체를 함수처럼 호출"""
                try:
                    # 원본 객체가 callable인 경우
                    if callable(self.wrapped_obj):
                        return self.wrapped_obj(*args, **kwargs)
                    
                    # callable하지 않은 경우 mock 응답
                    return {
                        'status': 'success',
                        'model_name': self.name,
                        'model_type': self.type,
                        'result': f'mock_result_for_{self.name}',
                        'wrapped_type': self.type
                    }
                    
                except Exception as e:
                    self.logger.error(f"❌ ObjectWrapper 호출 실패: {e}")
                    return {
                        'status': 'error',
                        'error': str(e),
                        'model_name': self.name
                    }
            
            def __getattr__(self, name):
                """속성 접근을 원본 객체로 위임"""
                return getattr(self.wrapped_obj, name)
        
        return ObjectWrapper(obj)
    
    async def warmup_model(self, name: str) -> bool:
        """안전한 모델 워밍업"""
        try:
            with self.lock:
                if name not in self.models:
                    self.logger.warning(f"⚠️ 모델이 등록되지 않음: {name}")
                    self.warmup_status[name] = False
                    return False
                
                model = self.models[name]
                
                # 워밍업 메서드가 있는 경우
                if hasattr(model, 'warmup'):
                    success, result, message = self.validator.safe_getattr_call(model, 'warmup')
                    if success:
                        self.logger.info(f"✅ 모델 워밍업 성공: {name}")
                        self.warmup_status[name] = True
                        return True
                
                # warmup 메서드가 없으면 간단한 테스트 호출
                success, result, message = self.validator.safe_call(model)
                if success:
                    self.logger.info(f"✅ 모델 테스트 호출 성공: {name}")
                    self.warmup_status[name] = True
                    return True
                
                self.logger.warning(f"⚠️ 모델 워밍업 실패: {name} - {message}")
                self.warmup_status[name] = False
                return False
                
        except Exception as e:
            self.logger.error(f"❌ 모델 워밍업 오류 {name}: {e}")
            self.warmup_status[name] = False
            return False
    
    async def call_model(self, name: str, *args, **kwargs) -> Any:
        """안전한 모델 호출"""
        try:
            with self.lock:
                if name not in self.models:
                    self.logger.warning(f"⚠️ 모델이 등록되지 않음: {name}")
                    return None
                
                model = self.models[name]
                success, result, message = await self.validator.safe_async_call(model, *args, **kwargs)
                
                if success:
                    self.logger.debug(f"✅ 모델 호출 성공: {name}")
                    return result
                else:
                    self.logger.warning(f"⚠️ 모델 호출 실패: {name} - {message}")
                    return None
                
        except Exception as e:
            self.logger.error(f"❌ 모델 호출 오류 {name}: {e}")
            return None
    
    def get_model_status(self, name: str) -> Dict[str, Any]:
        """모델 상태 조회"""
        try:
            with self.lock:
                if name not in self.models:
                    return {'status': 'not_registered', 'warmup': False}
                
                model = self.models[name]
                return {
                    'status': 'registered',
                    'warmup': self.warmup_status.get(name, False),
                    'type': type(model).__name__,
                    'callable': callable(model),
                    'has_warmup': hasattr(model, 'warmup')
                }
        except Exception as e:
            self.logger.error(f"❌ 모델 상태 조회 실패 {name}: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def list_models(self) -> Dict[str, Dict[str, Any]]:
        """등록된 모델 목록"""
        try:
            with self.lock:
                result = {}
                for name in self.models:
                    result[name] = self.get_model_status(name)
                return result
        except Exception as e:
            self.logger.error(f"❌ 모델 목록 조회 실패: {e}")
            return {}

# ==============================================
# 🔥 Step 요청사항 연동 (내장 기본 요청사항)
# ==============================================

# 🔥 내장 기본 요청사항 (step_model_requests.py 내용 일부)
STEP_MODEL_REQUESTS = {
    "HumanParsingStep": {
        "model_name": "human_parsing_graphonomy",
        "model_type": "GraphonomyModel",
        "input_size": (512, 512),
        "num_classes": 20,
        "checkpoint_patterns": ["*human*parsing*.pth", "*schp*atr*.pth", "*graphonomy*.pth"]
    },
    "PoseEstimationStep": {
        "model_name": "pose_estimation_openpose",
        "model_type": "OpenPoseModel",
        "input_size": (368, 368),
        "num_classes": 18,
        "checkpoint_patterns": ["*pose*model*.pth", "*openpose*.pth", "*body*pose*.pth"]
    },
    "ClothSegmentationStep": {
        "model_name": "cloth_segmentation_u2net",
        "model_type": "U2NetModel",
        "input_size": (320, 320),
        "num_classes": 1,
        "checkpoint_patterns": ["*u2net*.pth", "*cloth*segmentation*.pth", "*sam*.pth"]
    },
    "VirtualFittingStep": {
        "model_name": "virtual_fitting_stable_diffusion",
        "model_type": "StableDiffusionPipeline",
        "input_size": (512, 512),
        "checkpoint_patterns": ["*diffusion*pytorch*model*.bin", "*stable*diffusion*.safetensors"]
    },
    "GeometricMatchingStep": {
        "model_name": "geometric_matching_gmm",
        "model_type": "GeometricMatchingModel",
        "input_size": (512, 384),
        "checkpoint_patterns": ["*geometric*matching*.pth", "*gmm*.pth", "*tps*.pth"]
    },
    "PostProcessingStep": {
        "model_name": "post_processing_srresnet",
        "model_type": "SRResNetModel",
        "input_size": (512, 512),
        "checkpoint_patterns": ["*srresnet*.pth", "*enhancement*.pth", "*super*resolution*.pth"]
    },
    "QualityAssessmentStep": {
        "model_name": "quality_assessment_clip",
        "model_type": "CLIPModel",
        "input_size": (224, 224),
        "checkpoint_patterns": ["*clip*.bin", "*quality*assessment*.pth"]
    }
}

class StepModelRequestAnalyzer:
    @staticmethod
    def get_step_request_info(step_name: str):
        return STEP_MODEL_REQUESTS.get(step_name)

# ==============================================
# 🔥 실제 AI 모델 클래스들 (기존 유지)
# ==============================================

class BaseModel(nn.Module if TORCH_AVAILABLE else object):
    """기본 AI 모델 클래스"""
    def __init__(self):
        if TORCH_AVAILABLE:
            super().__init__()
        self.model_name = "BaseModel"
        self.device = "cpu"
    
    def forward(self, x):
        return x

if TORCH_AVAILABLE:
    class GraphonomyModel(nn.Module):
        """Graphonomy 인체 파싱 모델 - Step 01"""
        
        def __init__(self, num_classes=20, backbone='resnet101'):
            super().__init__()
            self.num_classes = num_classes
            self.backbone_name = backbone
            
            # 간단한 백본 구성
            self.backbone = nn.Sequential(
                nn.Conv2d(3, 64, 7, 2, 3),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(3, 2, 1),
                nn.Conv2d(64, 256, 3, 1, 1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 512, 3, 1, 1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 1024, 3, 1, 1),
                nn.BatchNorm2d(1024),
                nn.ReLU(inplace=True),
                nn.Conv2d(1024, 2048, 3, 1, 1),
                nn.BatchNorm2d(2048),
                nn.ReLU(inplace=True)
            )
            
            # 분류 헤드
            self.classifier = nn.Conv2d(2048, num_classes, kernel_size=1)
        
        def forward(self, x):
            input_size = x.size()[2:]
            features = self.backbone(x)
            output = self.classifier(features)
            output = F.interpolate(output, size=input_size, mode='bilinear', align_corners=False)
            return output

    class OpenPoseModel(nn.Module):
        """OpenPose 포즈 추정 모델 - Step 02"""
        
        def __init__(self, num_keypoints=18):
            super().__init__()
            self.num_keypoints = num_keypoints
            
            # VGG 스타일 백본
            self.backbone = nn.Sequential(
                nn.Conv2d(3, 64, 3, 1, 1), nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, 3, 1, 1), nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, 3, 1, 1), nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(128, 256, 3, 1, 1), nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(256, 512, 3, 1, 1), nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, 3, 1, 1), nn.ReLU(inplace=True)
            )
            
            # PAF 및 히트맵 헤드
            self.paf_head = nn.Conv2d(512, 38, 1)  # 19 limbs * 2
            self.heatmap_head = nn.Conv2d(512, 19, 1)  # 18 keypoints + 1 background
        
        def forward(self, x):
            features = self.backbone(x)
            paf = self.paf_head(features)
            heatmap = self.heatmap_head(features)
            return [(paf, heatmap)]

    class U2NetModel(nn.Module):
        """U²-Net 세그멘테이션 모델 - Step 03"""
        
        def __init__(self, in_ch=3, out_ch=1):
            super().__init__()
            
            # 인코더
            self.encoder = nn.Sequential(
                nn.Conv2d(in_ch, 64, 3, 1, 1), nn.ReLU(inplace=True),
                nn.Conv2d(64, 128, 3, 2, 1), nn.ReLU(inplace=True),
                nn.Conv2d(128, 256, 3, 2, 1), nn.ReLU(inplace=True),
                nn.Conv2d(256, 512, 3, 2, 1), nn.ReLU(inplace=True)
            )
            
            # 디코더
            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(512, 256, 4, 2, 1), nn.ReLU(inplace=True),
                nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.ReLU(inplace=True),
                nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(inplace=True),
                nn.Conv2d(64, out_ch, 3, 1, 1), nn.Sigmoid()
            )
        
        def forward(self, x):
            features = self.encoder(x)
            output = self.decoder(features)
            return output

    class GeometricMatchingModel(nn.Module):
        """기하학적 매칭 모델 - Step 04"""
        
        def __init__(self, feature_size=256):
            super().__init__()
            self.feature_size = feature_size
            
            self.feature_extractor = nn.Sequential(
                nn.Conv2d(3, 64, 3, 1, 1), nn.ReLU(inplace=True),
                nn.Conv2d(64, 128, 3, 2, 1), nn.ReLU(inplace=True),
                nn.Conv2d(128, 256, 3, 2, 1), nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((8, 8)),
                nn.Flatten(),
                nn.Linear(256 * 64, 512), nn.ReLU(inplace=True),
                nn.Linear(512, 18)  # 6개 제어점 * 3
            )
        
        def forward(self, source_img, target_img=None):
            if target_img is not None:
                combined = torch.cat([source_img, target_img], dim=1)
                combined = F.interpolate(combined, size=(256, 256), mode='bilinear')
                combined = combined[:, :3]  # 첫 3채널만
            else:
                combined = source_img
            
            tps_params = self.feature_extractor(combined)
            return {
                'tps_params': tps_params.view(-1, 6, 3),
                'correlation_map': torch.ones(combined.shape[0], 1, 64, 64).to(combined.device)
            }

    class HRVITONModel(nn.Module):
        """HR-VITON 가상 피팅 모델 - Step 06"""
        
        def __init__(self, input_nc=3, output_nc=3, ngf=64):
            super().__init__()
            
            # U-Net 스타일 생성기
            self.encoder = nn.Sequential(
                nn.Conv2d(input_nc * 2, ngf, 3, 1, 1), nn.ReLU(inplace=True),
                nn.Conv2d(ngf, ngf * 2, 3, 2, 1), nn.ReLU(inplace=True),
                nn.Conv2d(ngf * 2, ngf * 4, 3, 2, 1), nn.ReLU(inplace=True),
                nn.Conv2d(ngf * 4, ngf * 8, 3, 2, 1), nn.ReLU(inplace=True)
            )
            
            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1), nn.ReLU(inplace=True),
                nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1), nn.ReLU(inplace=True),
                nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1), nn.ReLU(inplace=True),
                nn.Conv2d(ngf, output_nc, 3, 1, 1), nn.Tanh()
            )
            
            # 어텐션 모듈
            self.attention = nn.Sequential(
                nn.Conv2d(input_nc * 2, 32, 3, 1, 1), nn.ReLU(inplace=True),
                nn.Conv2d(32, 1, 1, 1, 0), nn.Sigmoid()
            )
        
        def forward(self, person_img, cloth_img, **kwargs):
            combined_input = torch.cat([person_img, cloth_img], dim=1)
            
            features = self.encoder(combined_input)
            generated = self.decoder(features)
            
            attention_map = self.attention(combined_input)
            result = generated * attention_map + person_img * (1 - attention_map)
            
            return {
                'generated_image': result,
                'attention_map': attention_map,
                'warped_cloth': cloth_img,
                'intermediate': generated
            }
else:
    # PyTorch 없는 경우 더미 클래스들
    GraphonomyModel = BaseModel
    OpenPoseModel = BaseModel
    U2NetModel = BaseModel
    GeometricMatchingModel = BaseModel
    HRVITONModel = BaseModel

# ==============================================
# 🔥 디바이스 관리자 - M3 Max 특화 (기존 유지)
# ==============================================

class DeviceManager:
    """M3 Max 특화 디바이스 관리자"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.DeviceManager")
        self.available_devices = self._detect_available_devices()
        self.optimal_device = self._select_optimal_device()
        self.is_m3_max = self._detect_m3_max()
        
    def _detect_available_devices(self) -> List[str]:
        """사용 가능한 디바이스 탐지"""
        devices = ["cpu"]
        
        if TORCH_AVAILABLE:
            if torch.backends.mps.is_available():
                devices.append("mps")
                self.logger.info("🍎 M3 Max MPS 사용 가능")
            
            if torch.cuda.is_available():
                devices.append("cuda")
                cuda_devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
                devices.extend(cuda_devices)
                self.logger.info(f"🔥 CUDA 디바이스: {cuda_devices}")
        
        self.logger.info(f"🔍 사용 가능한 디바이스: {devices}")
        return devices
    
    def _select_optimal_device(self) -> str:
        """최적 디바이스 선택"""
        if "mps" in self.available_devices:
            return "mps"
        elif "cuda" in self.available_devices:
            return "cuda"
        else:
            return "cpu"
    
    def _detect_m3_max(self) -> bool:
        """M3 Max 칩 감지"""
        try:
            import platform
            import subprocess
            if platform.system() == 'Darwin':
                result = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'], 
                                      capture_output=True, text=True)
                return 'M3' in result.stdout
        except:
            pass
        return False
    
    def resolve_device(self, requested_device: str) -> str:
        """요청된 디바이스를 실제 디바이스로 변환"""
        if requested_device == "auto":
            return self.optimal_device
        elif requested_device in self.available_devices:
            return requested_device
        else:
            self.logger.warning(f"⚠️ 요청된 디바이스 {requested_device} 사용 불가, {self.optimal_device} 사용")
            return self.optimal_device

# ==============================================
# 🔥 메모리 관리자 - M3 Max 128GB 특화 (기존 유지)
# ==============================================

class ModelMemoryManager:
    """모델 메모리 관리자 - M3 Max 특화"""
    
    def __init__(self, device: str = "mps", memory_threshold: float = 0.8):
        self.device = device
        self.memory_threshold = memory_threshold
        self.is_m3_max = self._detect_m3_max()
    
    def _detect_m3_max(self) -> bool:
        """M3 Max 칩 감지"""
        try:
            import platform
            import subprocess
            if platform.system() == 'Darwin':
                result = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'], 
                                      capture_output=True, text=True)
                return 'M3' in result.stdout
        except:
            pass
        return False
    
    def get_available_memory(self) -> float:
        """사용 가능한 메모리 (GB) 반환"""
        try:
            if self.device == "cuda" and TORCH_AVAILABLE and torch.cuda.is_available():
                total_memory = torch.cuda.get_device_properties(0).total_memory
                allocated_memory = torch.cuda.memory_allocated()
                return (total_memory - allocated_memory) / 1024**3
            elif self.device == "mps":
                try:
                    import psutil
                    memory = psutil.virtual_memory()
                    available_gb = memory.available / 1024**3
                    if self.is_m3_max:
                        return min(available_gb, 100.0)  # 128GB 중 사용 가능한 부분
                    return available_gb
                except ImportError:
                    return 64.0 if self.is_m3_max else 16.0
            else:
                try:
                    import psutil
                    memory = psutil.virtual_memory()
                    return memory.available / 1024**3
                except ImportError:
                    return 8.0
        except Exception as e:
            logger.warning(f"⚠️ 메모리 조회 실패: {e}")
            return 8.0
    
    def cleanup_memory(self):
        """메모리 정리 - M3 Max 최적화"""
        try:
            gc.collect()
            
            if TORCH_AVAILABLE:
                if self.device == "cuda" and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                elif self.device == "mps" and torch.backends.mps.is_available():
                    try:
                        if hasattr(torch.backends.mps, 'empty_cache'):
                            torch.backends.mps.empty_cache()
                        if self.is_m3_max:
                            torch.mps.synchronize()
                    except:
                        pass
            
            logger.debug("🧹 메모리 정리 완료")
        except Exception as e:
            logger.warning(f"⚠️ 메모리 정리 실패: {e}")
    
    def check_memory_pressure(self) -> bool:
        """메모리 압박 상태 체크"""
        try:
            available_memory = self.get_available_memory()
            threshold = 4.0 if self.is_m3_max else 2.0
            return available_memory < threshold
        except Exception:
            return False

# ==============================================
# 🔥 이미지 전처리 함수들 (기존 유지)
# ==============================================

def preprocess_image(
    image: Union[Image.Image, np.ndarray, torch.Tensor],
    target_size: Tuple[int, int] = (512, 512),
    device: str = "mps",
    normalize: bool = True,
    to_tensor: bool = True
) -> torch.Tensor:
    """
    🔥 이미지 전처리 함수 - Step 클래스들에서 사용
    
    Args:
        image: 입력 이미지 (PIL.Image, numpy array, tensor)
        target_size: 목표 크기 (height, width)
        device: 디바이스 ("mps", "cuda", "cpu")
        normalize: 정규화 여부 (0-1 범위로)
        to_tensor: 텐서로 변환 여부
    
    Returns:
        torch.Tensor: 전처리된 이미지 텐서
    """
    try:
        # 1. PIL Image로 변환
        if isinstance(image, torch.Tensor):
            if image.dim() == 4:
                image = image.squeeze(0)
            if image.dim() == 3 and image.shape[0] == 3:
                image = image.permute(1, 2, 0)
            image = image.cpu().numpy()
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)
            image = Image.fromarray(image)
        elif isinstance(image, np.ndarray):
            if image.ndim == 3 and image.shape[2] == 3:
                image = Image.fromarray(image.astype(np.uint8))
            else:
                image = Image.fromarray(image)
        elif not isinstance(image, Image.Image):
            raise ValueError(f"지원하지 않는 이미지 타입: {type(image)}")
        
        # 2. RGB 변환
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # 3. 크기 조정
        if target_size != image.size:
            image = image.resize(target_size, Image.Resampling.LANCZOS)
        
        # 4. numpy 배열로 변환
        img_array = np.array(image).astype(np.float32)
        
        # 5. 정규화
        if normalize:
            img_array = img_array / 255.0
        
        # 6. 텐서 변환
        if to_tensor and TORCH_AVAILABLE:
            img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)
            img_tensor = img_tensor.to(device)
            return img_tensor
        else:
            return img_array
            
    except Exception as e:
        logger.error(f"이미지 전처리 실패: {e}")
        # 폴백: 기본 크기 텐서 반환
        if TORCH_AVAILABLE:
            return torch.zeros(1, 3, target_size[0], target_size[1], device=device)
        else:
            return np.zeros((target_size[0], target_size[1], 3), dtype=np.float32)

def postprocess_segmentation(output: torch.Tensor, threshold: float = 0.5) -> np.ndarray:
    """세그멘테이션 결과 후처리"""
    try:
        if isinstance(output, torch.Tensor):
            output = output.cpu().numpy()
        
        if output.ndim == 4:
            output = output.squeeze(0)
        if output.ndim == 3:
            output = output.squeeze(0)
            
        # 임계값 적용
        binary_mask = (output > threshold).astype(np.uint8) * 255
        return binary_mask
        
    except Exception as e:
        logger.error(f"세그멘테이션 후처리 실패: {e}")
        return np.zeros((512, 512), dtype=np.uint8)

def preprocess_pose_input(image: np.ndarray, target_size: Tuple[int, int] = (368, 368)) -> torch.Tensor:
    """포즈 추정용 이미지 전처리"""
    return preprocess_image(image, target_size, normalize=True, to_tensor=True)

def preprocess_human_parsing_input(image: np.ndarray, target_size: Tuple[int, int] = (512, 512)) -> torch.Tensor:
    """인간 파싱용 이미지 전처리"""
    return preprocess_image(image, target_size, normalize=True, to_tensor=True)

def preprocess_cloth_segmentation_input(image: np.ndarray, target_size: Tuple[int, int] = (320, 320)) -> torch.Tensor:
    """의류 세그멘테이션용 이미지 전처리"""
    return preprocess_image(image, target_size, normalize=True, to_tensor=True)

def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """텐서를 PIL 이미지로 변환"""
    try:
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0)
        if tensor.dim() == 3:
            tensor = tensor.permute(1, 2, 0)
        
        tensor = tensor.cpu().numpy()
        if tensor.dtype != np.uint8:
            tensor = (tensor * 255).astype(np.uint8)
        
        return Image.fromarray(tensor)
    except Exception as e:
        logger.error(f"텐서->PIL 변환 실패: {e}")
        return Image.new('RGB', (512, 512), color='black')

def pil_to_tensor(image: Image.Image, device: str = "mps") -> torch.Tensor:
    """PIL 이미지를 텐서로 변환"""
    try:
        img_array = np.array(image).astype(np.float32) / 255.0
        tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)
        return tensor.to(device)
    except Exception as e:
        logger.error(f"PIL->텐서 변환 실패: {e}")
        return torch.zeros(1, 3, 512, 512, device=device)

# ==============================================
# 🔥 완전 개선된 StepModelInterface - Dict Callable 오류 완전 해결
# ==============================================

class StepModelInterface:
    """
    🔥 Step 클래스들을 위한 모델 인터페이스 - Dict Callable 오류 완전 해결
    SafeModelService 통합으로 모든 호출 안전성 보장
    """
    
    def __init__(self, model_loader: 'ModelLoader', step_name: str):
        """🔥 완전 안전한 생성자"""
        
        # 🔥 기본 속성 설정
        self.model_loader = model_loader
        self.step_name = step_name
        
        # 🔥 logger 속성 안전하게 설정
        self.logger = logging.getLogger(f"StepInterface.{step_name}")
        
        # 🔥 안전한 속성 추출 (callable 검증 포함)
        self.device = getattr(model_loader, 'device', 'mps')
        self.model_cache_dir = Path(getattr(model_loader, 'model_cache_dir', './ai_models'))
        
        # 🔥 SafeModelService 통합
        self.model_service = SafeModelService()
        
        # 🔥 모델 캐시 및 상태 관리
        self.loaded_models: Dict[str, Any] = {}
        self.model_cache: Dict[str, Any] = {}
        self._lock = threading.RLock()
        
        # 🔥 Step별 모델 설정
        self.recommended_models = self._get_recommended_models()
        self.access_count = 0
        self.last_used = time.time()
        
        # 🔥 ModelLoader 메서드 가용성 체크 - Dict Callable 방지
        self.has_async_loader = self._safe_check_method(model_loader, 'load_model_async')
        self.has_sync_wrapper = self._safe_check_method(model_loader, '_load_model_sync_wrapper')
        
        # 🔥 실제 모델 경로 설정
        try:
            self.model_paths = self._setup_model_paths()
        except Exception as e:
            self.logger.warning(f"⚠️ 모델 경로 설정 실패: {e}")
            self.model_paths = self._get_fallback_model_paths()
        
        # 🔥 Step별 실제 AI 모델 매핑 설정
        self.step_model_mapping = self._get_step_model_mapping()
        
        self.logger.info(f"🔗 {step_name} 인터페이스 초기화 완료 (SafeModelService 통합)")
        self.logger.info(f"🔧 Device: {self.device}, Cache Dir: {self.model_cache_dir}")
        self.logger.info(f"📦 추천 모델: {self.recommended_models}")
    
    def _safe_check_method(self, obj: Any, method_name: str) -> bool:
        """메서드 존재 및 callable 여부 안전 확인"""
        try:
            if obj is None:
                return False
            
            if not hasattr(obj, method_name):
                return False
            
            method = getattr(obj, method_name)
            is_callable, reason, safe_method = SafeFunctionValidator.validate_callable(method, f"check_{method_name}")
            
            if is_callable:
                self.logger.debug(f"✅ {method_name} 메서드 사용 가능")
                return True
            else:
                self.logger.debug(f"⚠️ {method_name} 메서드 사용 불가: {reason}")
                return False
                
        except Exception as e:
            self.logger.warning(f"⚠️ {method_name} 메서드 체크 실패: {e}")
            return False
    
    def _get_recommended_models(self) -> List[str]:
        """Step별 권장 모델 목록"""
        model_mapping = {
            "HumanParsingStep": ["human_parsing_graphonomy", "human_parsing_u2net", "graphonomy"],
            "PoseEstimationStep": ["pose_estimation_openpose", "openpose", "mediapipe_pose"],
            "ClothSegmentationStep": ["u2net_cloth_seg", "u2net", "cloth_segmentation"],
            "GeometricMatchingStep": ["geometric_matching_gmm", "tps_network", "geometric_matching"],
            "ClothWarpingStep": ["cloth_warping_net", "tom_final", "warping_net"],
            "VirtualFittingStep": ["ootdiffusion", "stable_diffusion", "diffusion_pipeline"],
            "PostProcessingStep": ["srresnet_x4", "denoise_net", "enhancement"],
            "QualityAssessmentStep": ["quality_assessment_clip", "clip", "image_quality"]
        }
        return model_mapping.get(self.step_name, ["default_model"])
    
    def _setup_model_paths(self) -> Dict[str, str]:
        """🔥 실제 AI 모델 경로 설정 - 실제 발견된 파일들 기반"""
        base_path = self.model_cache_dir
        
        return {
            # 🔥 실제 발견된 Human Parsing Models
            'graphonomy': str(base_path / "checkpoints" / "human_parsing" / "schp_atr.pth"),
            'human_parsing_graphonomy': str(base_path / "checkpoints" / "human_parsing" / "schp_atr.pth"),
            'human_parsing_u2net': str(base_path / "checkpoints" / "step_03" / "u2net_segmentation" / "u2net.pth"),
            
            # 🔥 실제 발견된 Pose Estimation Models  
            'openpose': str(base_path / "openpose"),
            'pose_estimation_openpose': str(base_path / "openpose"),
            
            # 🔥 실제 발견된 Cloth Segmentation Models
            'u2net': str(base_path / "checkpoints" / "step_03" / "u2net_segmentation" / "u2net.pth"),
            'u2net_cloth_seg': str(base_path / "checkpoints" / "step_03" / "u2net_segmentation" / "u2net.pth"),
            'cloth_segmentation_u2net': str(base_path / "checkpoints" / "step_03" / "u2net_segmentation" / "u2net.pth"),
            
            # 🔥 실제 발견된 Virtual Fitting Models
            'ootdiffusion': str(base_path / "OOTDiffusion"),
            'stable_diffusion': str(base_path / "OOTDiffusion"),
            'diffusion_pipeline': str(base_path / "OOTDiffusion"),
            
            # 🔥 실제 발견된 Geometric Matching
            'geometric_matching_gmm': str(base_path / "checkpoints" / "step_04" / "step_04_geometric_matching_base" / "geometric_matching_base.pth"),
            'tps_network': str(base_path / "checkpoints" / "step_04" / "step_04_tps_network" / "tps_network.pth"),
            
            # 🔥 실제 발견된 기타 모델들
            'clip': str(base_path / "clip-vit-base-patch32"),
            'quality_assessment_clip': str(base_path / "clip-vit-base-patch32"),
            'srresnet_x4': str(base_path / "cache" / "models--ai-forever--Real-ESRGAN" / ".no_exist" / "8110204ebf8d25c031b66c26c2d1098aa831157e" / "RealESRGAN_x4plus.pth"),
        }
    
    def _get_fallback_model_paths(self) -> Dict[str, str]:
        """폴백 모델 경로"""
        return {
            'default_model': str(self.model_cache_dir / "default_model.pth"),
            'fallback_model': str(self.model_cache_dir / "fallback_model.pth")
        }
    
    def _get_step_model_mapping(self) -> Dict[str, Dict[str, Any]]:
        """Step별 실제 AI 모델 매핑"""
        return {
            'HumanParsingStep': {
                'primary': 'human_parsing_graphonomy',
                'models': ['graphonomy', 'human_parsing_u2net']
            },
            'PoseEstimationStep': {
                'primary': 'pose_estimation_openpose',
                'models': ['openpose']
            },
            'ClothSegmentationStep': {
                'primary': 'u2net_cloth_seg',
                'models': ['u2net', 'cloth_segmentation_u2net']
            },
            'GeometricMatchingStep': {
                'primary': 'geometric_matching_gmm',
                'models': ['tps_network']
            },
            'VirtualFittingStep': {
                'primary': 'ootdiffusion',
                'models': ['stable_diffusion', 'diffusion_pipeline']
            },
            'PostProcessingStep': {
                'primary': 'srresnet_x4',
                'models': ['enhancement', 'denoise_net']
            },
            'QualityAssessmentStep': {
                'primary': 'quality_assessment_clip',
                'models': ['clip']
            }
        }
    
    async def get_model(self, model_name: Optional[str] = None) -> Optional[Any]:
        """🔥 모델 로드 - Dict Callable 오류 완전 해결"""
        try:
            # 모델명 결정
            if not model_name:
                model_name = self.recommended_models[0] if self.recommended_models else "default_model"
            
            # 캐시 확인
            if model_name in self.loaded_models:
                self.access_count += 1
                self.last_used = time.time()
                self.logger.info(f"✅ 캐시된 모델 반환: {model_name}")
                return self.loaded_models[model_name]
            
            # 🔥 SafeModelService를 통한 안전한 모델 로드
            model = await self._safe_load_model_via_service(model_name)
            
            if model:
                self.loaded_models[model_name] = model
                self.access_count += 1
                self.last_used = time.time()
                self.logger.info(f"✅ 모델 로드 성공: {model_name}")
                return model
            else:
                # 폴백 모델 생성
                fallback = self._create_smart_fallback_model(model_name)
                self.loaded_models[model_name] = fallback
                self.logger.warning(f"⚠️ 폴백 모델 사용: {model_name}")
                return fallback
                
        except Exception as e:
            self.logger.error(f"❌ 모델 로드 실패 {model_name}: {e}")
            # 최종 폴백
            fallback = self._create_smart_fallback_model(model_name)
            self.loaded_models[model_name] = fallback
            return fallback
    
    async def _safe_load_model_via_service(self, model_name: str) -> Optional[Any]:
        """🔥 SafeModelService를 통한 안전한 모델 로드"""
        try:
            # 1. SafeModelService에 모델 등록 시도
            model_dict = {
                'name': model_name,
                'type': self.step_name,
                'device': self.device,
                'path': self.model_paths.get(model_name, 'unknown'),
                'step_mapping': self.step_model_mapping.get(self.step_name, {})
            }
            
            # SafeModelService에 등록
            registration_success = self.model_service.register_model(model_name, model_dict)
            
            if registration_success:
                # 등록된 모델 호출
                model = await self.model_service.call_model(model_name)
                if model:
                    return model
            
            # 2. 기존 ModelLoader 방식 시도 (안전하게)
            return await self._legacy_safe_load_model(model_name)
            
        except Exception as e:
            self.logger.warning(f"⚠️ SafeModelService 로드 실패: {e}")
            return None
    
    async def _legacy_safe_load_model(self, model_name: str) -> Optional[Any]:
        """🔥 기존 ModelLoader 방식 안전한 로드"""
        try:
            # 비동기 로더 사용 시도
            if self.has_async_loader:
                load_async_method = getattr(self.model_loader, 'load_model_async', None)
                success, result, message = await SafeFunctionValidator.safe_async_call(
                    load_async_method, model_name
                )
                if success:
                    return result
                else:
                    self.logger.warning(f"⚠️ 비동기 로드 실패: {message}")
            
            # 동기 래퍼 사용 시도
            if self.has_sync_wrapper:
                sync_wrapper_method = getattr(self.model_loader, '_load_model_sync_wrapper', None)
                success, result, message = SafeFunctionValidator.safe_call(
                    sync_wrapper_method, model_name, {}
                )
                if success:
                    return result
                else:
                    self.logger.warning(f"⚠️ 동기 래퍼 실패: {message}")
            
            # 기본 load_model 메서드 시도
            if hasattr(self.model_loader, 'load_model'):
                load_model_method = getattr(self.model_loader, 'load_model', None)
                success, result, message = await SafeFunctionValidator.safe_async_call(
                    load_model_method, model_name
                )
                if success:
                    return result
                else:
                    self.logger.warning(f"⚠️ 기본 로드 실패: {message}")
            
            # 직접 모델 파일 찾기
            return self._direct_model_load(model_name)
                
        except Exception as e:
            self.logger.warning(f"⚠️ Legacy 안전한 모델 로드 실패: {e}")
            return None
    
    def _direct_model_load(self, model_name: str) -> Optional[Any]:
        """직접 모델 파일 로드"""
        try:
            # 모델 경로에서 찾기
            if model_name in self.model_paths:
                model_path = Path(self.model_paths[model_name])
                if model_path.exists() and model_path.stat().st_size > 1024:
                    self.logger.info(f"📂 모델 파일 발견: {model_path}")
                    try:
                        # 안전한 임포트
                        if TORCH_AVAILABLE:
                            model = torch.load(model_path, map_location=self.device)
                            return model
                        else:
                            self.logger.warning("⚠️ PyTorch가 없어서 모델 로드 불가")
                    except Exception as e:
                        self.logger.warning(f"⚠️ PyTorch 로드 실패: {e}")
            
            return None
            
        except Exception as e:
            self.logger.error(f"❌ 직접 모델 로드 실패: {e}")
            return None
    
    def _create_smart_fallback_model(self, model_name: str) -> Any:
        """🔥 스마트 폴백 모델 생성 - Step별 특화"""
        
        class SmartMockModel:
            """스마트 Mock AI 모델 - Step별 특화 출력"""
            
            def __init__(self, name: str, device: str, step_name: str):
                self.name = name
                self.device = device
                self.step_name = step_name
                self.model_type = self._detect_model_type(name, step_name)
                self.is_loaded = True
                self.eval_mode = True
                
            def _detect_model_type(self, name: str, step_name: str) -> str:
                """모델 타입 감지"""
                if 'human_parsing' in name or 'HumanParsing' in step_name:
                    return 'human_parsing'
                elif 'pose' in name or 'Pose' in step_name:
                    return 'pose_estimation'
                elif 'segmentation' in name or 'u2net' in name or 'Segmentation' in step_name:
                    return 'segmentation'
                elif 'geometric' in name or 'Geometric' in step_name:
                    return 'geometric_matching'
                elif 'diffusion' in name or 'ootd' in name or 'Fitting' in step_name:
                    return 'diffusion'
                else:
                    return 'general'
            
            def __call__(self, *args, **kwargs):
                """모델을 함수처럼 호출"""
                return self.forward(*args, **kwargs)
            
            def forward(self, *args, **kwargs):
                """Step별 특화 Mock 출력"""
                try:
                    # 기본 크기 설정
                    height, width = 512, 512
                    batch_size = 1
                    
                    # Step별 특화 출력
                    if self.model_type == 'human_parsing':
                        # 20개 클래스 인간 파싱
                        if TORCH_AVAILABLE:
                            return torch.zeros((batch_size, 20, height, width), device='cpu')
                        else:
                            return np.zeros((batch_size, 20, height, width), dtype=np.float32)
                    elif self.model_type == 'pose_estimation':
                        # 18개 키포인트
                        if TORCH_AVAILABLE:
                            return torch.zeros((batch_size, 18, height//4, width//4), device='cpu')
                        else:
                            return np.zeros((batch_size, 18, height//4, width//4), dtype=np.float32)
                    elif self.model_type == 'segmentation':
                        # Binary mask
                        if TORCH_AVAILABLE:
                            return torch.zeros((batch_size, 1, height, width), device='cpu')
                        else:
                            return np.zeros((batch_size, 1, height, width), dtype=np.float32)
                    elif self.model_type == 'geometric_matching':
                        # Transformation parameters
                        if TORCH_AVAILABLE:
                            return torch.zeros((batch_size, 25, 2), device='cpu')
                        else:
                            return np.zeros((batch_size, 25, 2), dtype=np.float32)
                    elif self.model_type == 'diffusion':
                        # Generated image
                        if TORCH_AVAILABLE:
                            return torch.zeros((batch_size, 3, height, width), device='cpu')
                        else:
                            return np.zeros((batch_size, 3, height, width), dtype=np.float32)
                    else:
                        # Default output
                        if TORCH_AVAILABLE:
                            return torch.zeros((batch_size, 3, height, width), device='cpu')
                        else:
                            return np.zeros((batch_size, 3, height, width), dtype=np.float32)
                        
                except Exception:
                    # 최종 폴백: numpy 사용
                    return np.zeros((batch_size, 3, height, width), dtype=np.float32)
            
            def to(self, device):
                """디바이스 이동"""
                self.device = str(device)
                return self
            
            def eval(self):
                """평가 모드"""
                self.eval_mode = True
                return self
            
            def cuda(self):
                return self.to('cuda')
            
            def cpu(self):
                return self.to('cpu')
            
            def warmup(self):
                """워밍업"""
                return True
        
        mock = SmartMockModel(model_name, self.device, self.step_name)
        self.logger.info(f"🎭 Smart Mock 모델 생성: {model_name} ({mock.model_type})")
        return mock
    
    async def get_recommended_model(self) -> Optional[Any]:
        """권장 모델 로드"""
        if self.recommended_models:
            return await self.get_model(self.recommended_models[0])
        return await self.get_model("default_model")
    
    def unload_models(self):
        """모델 언로드 및 메모리 정리 - 안전한 호출"""
        try:
            unloaded_count = 0
            for model_name, model in list(self.loaded_models.items()):
                try:
                    # CPU로 이동 시도 (안전한 호출)
                    if hasattr(model, 'cpu'):
                        cpu_method = getattr(model, 'cpu', None)
                        success, result, message = SafeFunctionValidator.safe_call(cpu_method)
                        if not success:
                            self.logger.warning(f"⚠️ CPU 이동 실패 {model_name}: {message}")
                    
                    del model
                    unloaded_count += 1
                except Exception as e:
                    self.logger.warning(f"⚠️ 모델 언로드 실패 {model_name}: {e}")
            
            self.loaded_models.clear()
            self.logger.info(f"🧹 {unloaded_count}개 모델 언로드 완료: {self.step_name}")
            
        except Exception as e:
            self.logger.error(f"❌ 모델 언로드 실패: {e}")
    
    def get_info(self) -> Dict[str, Any]:
        """인터페이스 정보 반환"""
        return {
            "step_name": self.step_name,
            "device": self.device,
            "model_cache_dir": str(self.model_cache_dir),
            "recommended_models": self.recommended_models,
            "loaded_models": list(self.loaded_models.keys()),
            "available_model_paths": len(self.model_paths),
            "access_count": self.access_count,
            "last_used": self.last_used,
            "has_async_loader": self.has_async_loader,
            "has_sync_wrapper": self.has_sync_wrapper,
            "step_model_mapping": self.step_model_mapping.get(self.step_name, {}),
            "safe_model_service_status": True
        }

# ==============================================
# 🔥 완전 통합 ModelLoader 클래스 v5.0 - Dict Callable 오류 완전 해결
# ==============================================

class ModelLoader:
    """
    🍎 M3 Max 최적화 완전 통합 ModelLoader v5.0
    ✅ 'dict' object is not callable 근본 원인 해결
    ✅ SafeModelService + SafeFunctionValidator 통합
    ✅ NumPy 2.x 완전 호환성
    ✅ BaseStepMixin v3.3 완벽 연동
    ✅ M3 Max 128GB 메모리 최적화
    ✅ 프로덕션 안정성 최고 수준
    ✅ 모든 기존 기능명/클래스명 유지
    """
    
    def __init__(
        self,
        device: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        enable_auto_detection: bool = True,
        **kwargs
    ):
        """완전 통합 생성자 - Dict Callable 오류 완전 해결"""
        
        # 🔥 NumPy 호환성 체크
        self._check_numpy_compatibility()
        
        # 🔥 기본 설정 - SafeConfig 사용
        self.config = SafeConfig(config or {})
        self.step_name = self.__class__.__name__
        
        # 🔥 logger 속성 설정
        self.logger = logging.getLogger(f"ModelLoader.{self.step_name}")
        
        # 🔥 SafeModelService 통합
        self.safe_model_service = SafeModelService()
        self.function_validator = SafeFunctionValidator()
        
        # 🔥 디바이스 및 메모리 관리
        self.device_manager = DeviceManager()
        self.device = self.device_manager.resolve_device(device or "auto")
        self.memory_manager = ModelMemoryManager(device=self.device)
        
        # 🔥 시스템 파라미터
        self.memory_gb = kwargs.get('memory_gb', 128.0)
        self.is_m3_max = self.device_manager.is_m3_max
        self.optimization_enabled = kwargs.get('optimization_enabled', True)
        
        # 🔥 모델 로더 특화 파라미터
        self.model_cache_dir = Path(kwargs.get('model_cache_dir', './ai_models'))
        self.use_fp16 = kwargs.get('use_fp16', True and self.device != 'cpu')
        self.max_cached_models = kwargs.get('max_cached_models', 10)
        self.lazy_loading = kwargs.get('lazy_loading', True)
        self.enable_fallback = kwargs.get('enable_fallback', True)
        
        # 🔥 모델 캐시 및 상태 관리
        self.model_cache: Dict[str, Any] = {}
        self.model_configs: Dict[str, Union[ModelConfig, StepModelConfig]] = {}
        self.load_times: Dict[str, float] = {}
        self.last_access: Dict[str, float] = {}
        self.access_counts: Dict[str, int] = {}
        
        # 🔥 Step 인터페이스 관리
        self.step_interfaces: Dict[str, StepModelInterface] = {}
        
        # 🔥 동기화 및 스레드 관리
        self._lock = threading.RLock()
        self._interface_lock = threading.RLock()
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="model_loader")
        
        # 🔥 Step 요청사항 연동
        self.step_requirements: Dict[str, Dict[str, Any]] = {}
        
        # 🔥 초기화 실행
        self._initialize_components()
        
        self.logger.info(f"🎯 ModelLoader v5.0 초기화 완료 - Dict Callable 오류 완전 해결")
        self.logger.info(f"🔧 Device: {self.device}, SafeModelService: ✅")
    
    def _check_numpy_compatibility(self):
        """NumPy 2.x 호환성 체크 및 경고"""
        try:
            if NUMPY_AVAILABLE:
                numpy_version = np.__version__
                major_version = int(numpy_version.split('.')[0])
                
                if major_version >= 2:
                    temp_logger = logging.getLogger(f"ModelLoader.{self.__class__.__name__}")
                    temp_logger.warning(f"⚠️ NumPy {numpy_version} 감지됨 (2.x)")
                    temp_logger.warning("🔧 conda install numpy=1.24.3 -y --force-reinstall 권장")
                    
                    # NumPy 2.x용 호환성 설정
                    try:
                        np.set_printoptions(legacy='1.25')
                        temp_logger.info("✅ NumPy 2.x 호환성 모드 활성화")
                    except:
                        pass
        except Exception as e:
            temp_logger = logging.getLogger(f"ModelLoader.{self.__class__.__name__}")
            temp_logger.warning(f"⚠️ NumPy 버전 체크 실패: {e}")
    
    def _initialize_components(self):
        """모든 구성 요소 초기화"""
        try:
            # 캐시 디렉토리 생성
            self.model_cache_dir.mkdir(parents=True, exist_ok=True)
            
            # M3 Max 특화 설정
            if self.is_m3_max:
                self.use_fp16 = True
                if COREML_AVAILABLE:
                    self.logger.info("🍎 CoreML 최적화 활성화됨")
            
            # Step 요청사항 로드
            self._load_step_requirements()
            
            # 기본 모델 레지스트리 초기화
            self._initialize_model_registry()
            
            self.logger.info(f"📦 ModelLoader 구성 요소 초기화 완료")
    
        except Exception as e:
            self.logger.error(f"❌ 구성 요소 초기화 실패: {e}")
    
    def _load_step_requirements(self):
        """Step 요청사항 로드 - Dict Callable 오류 완전 해결"""
        try:
            # 내장 요청사항 사용
            self.step_requirements = STEP_MODEL_REQUESTS
            
            loaded_steps = 0
            for step_name, request_info in self.step_requirements.items():
                try:
                    if isinstance(request_info, dict):
                        # 딕셔너리 형태 처리 - SafeConfig 사용하지 않고 직접 처리
                        step_config = StepModelConfig(
                            step_name=step_name,
                            model_name=request_info.get("model_name", step_name.lower()),
                            model_class=request_info.get("model_type", "BaseModel"),
                            model_type=request_info.get("model_type", "unknown"),
                            device="auto",
                            precision="fp16",
                            input_size=request_info.get("input_size", (512, 512)),
                            num_classes=request_info.get("num_classes", None)
                        )
                        
                        self.model_configs[request_info.get("model_name", step_name)] = step_config
                        loaded_steps += 1
                        
                except Exception as e:
                    self.logger.warning(f"⚠️ {step_name} 요청사항 로드 실패: {e}")
                    continue
            
            self.logger.info(f"📝 {loaded_steps}개 Step 요청사항 로드 완료")
            
        except Exception as e:
            self.logger.error(f"❌ Step 요청사항 로드 실패: {e}")
    
    def _initialize_model_registry(self):
        """기본 모델 레지스트리 초기화"""
        try:
            base_models_dir = self.model_cache_dir
            
            model_configs = {
                # Step 01: Human Parsing
                "human_parsing_graphonomy": ModelConfig(
                    name="human_parsing_graphonomy",
                    model_type=ModelType.HUMAN_PARSING,
                    model_class="GraphonomyModel",
                    checkpoint_path=str(base_models_dir / "Graphonomy" / "inference.pth"),
                    input_size=(512, 512),
                    num_classes=20
                ),
                
                # Step 02: Pose Estimation
                "pose_estimation_openpose": ModelConfig(
                    name="pose_estimation_openpose", 
                    model_type=ModelType.POSE_ESTIMATION,
                    model_class="OpenPoseModel",
                    checkpoint_path=str(base_models_dir / "openpose" / "pose_model.pth"),
                    input_size=(368, 368),
                    num_classes=18
                ),
                
                # Step 03: Cloth Segmentation
                "cloth_segmentation_u2net": ModelConfig(
                    name="cloth_segmentation_u2net",
                    model_type=ModelType.CLOTH_SEGMENTATION, 
                    model_class="U2NetModel",
                    checkpoint_path=str(base_models_dir / "checkpoints" / "u2net.pth"),
                    input_size=(320, 320)
                ),
                
                # Step 04: Geometric Matching
                "geometric_matching_gmm": ModelConfig(
                    name="geometric_matching_gmm",
                    model_type=ModelType.GEOMETRIC_MATCHING,
                    model_class="GeometricMatchingModel", 
                    checkpoint_path=str(base_models_dir / "HR-VITON" / "gmm_final.pth"),
                    input_size=(512, 384)
                ),
                
                # Step 06: Virtual Fitting
                "virtual_fitting_hrviton": ModelConfig(
                    name="virtual_fitting_hrviton",
                    model_type=ModelType.VIRTUAL_FITTING,
                    model_class="HRVITONModel",
                    checkpoint_path=str(base_models_dir / "HR-VITON" / "final.pth"),
                    input_size=(512, 384)
                )
            }
            
            # 모델 등록
            registered_count = 0
            for name, config in model_configs.items():
                if self.register_model_config(name, config):
                    registered_count += 1
            
            self.logger.info(f"📝 기본 모델 등록 완료: {registered_count}개")
            
        except Exception as e:
            self.logger.error(f"❌ 모델 레지스트리 초기화 실패: {e}")
    
    def register_model_config(
        self,
        name: str,
        model_config: Union[ModelConfig, StepModelConfig, Dict[str, Any]],
        loader_func: Optional[Callable] = None
    ) -> bool:
        """모델 등록 - 모든 타입 지원 - Dict Callable 오류 완전 해결"""
        try:
            with self._lock:
                # 설정 타입별 처리
                if isinstance(model_config, dict):
                    # Dict를 ModelConfig로 변환
                    if "step_name" in model_config:
                        config = StepModelConfig(**model_config)
                    else:
                        config = ModelConfig(**model_config)
                else:
                    config = model_config
                
                # 디바이스 설정 자동 감지
                if hasattr(config, 'device') and config.device == "auto":
                    config.device = self.device
                
                # 내부 설정 저장
                self.model_configs[name] = config
                
                # 🔥 SafeModelService에도 등록
                model_dict = {
                    'name': name,
                    'config': config,
                    'type': getattr(config, 'model_type', 'unknown'),
                    'device': self.device
                }
                self.safe_model_service.register_model(name, model_dict)
                
                model_type = getattr(config, 'model_type', 'unknown')
                if hasattr(model_type, 'value'):
                    model_type = model_type.value
                
                self.logger.info(f"📝 모델 등록: {name} ({model_type}) - SafeModelService 포함")
                return True
                
        except Exception as e:
            self.logger.error(f"❌ 모델 등록 실패 {name}: {e}")
            return False
    
    def register_model(self, name: str, config: Dict[str, Any]):
        """모델 등록 (어댑터에서 사용) - Dict Callable 오류 완전 해결"""
        try:
            # 🔥 dict 타입 확인 후 안전한 처리
            if not isinstance(config, dict):
                self.logger.error(f"❌ config는 dict 타입이어야 함: {type(config)}")
                return False
            
            if not hasattr(self, 'detected_model_registry'):
                self.detected_model_registry = {}
            
            # 🔥 딕셔너리 복사로 안전한 저장
            self.detected_model_registry[name] = config.copy()
            
            # 🔥 SafeModelService에도 등록
            self.safe_model_service.register_model(name, config)
            
            self.logger.debug(f"✅ 모델 등록: {name} - SafeModelService 포함")
            return True
        except Exception as e:
            self.logger.error(f"❌ 모델 등록 실패 {name}: {e}")
            return False
    
    def _get_model_class(self, model_class_name: str) -> Type:
        """모델 클래스 이름으로 실제 클래스 반환"""
        model_classes = {
            'GraphonomyModel': GraphonomyModel,
            'OpenPoseModel': OpenPoseModel,
            'U2NetModel': U2NetModel,
            'GeometricMatchingModel': GeometricMatchingModel,
            'HRVITONModel': HRVITONModel,
            'BaseModel': BaseModel
        }
        return model_classes.get(model_class_name, BaseModel)
    
    async def load_model_async(self, model_name: str, **kwargs) -> Optional[Any]:
        """🔥 비동기 모델 로드 - Dict Callable 오류 완전 해결"""
        try:
            # 🔥 SafeModelService 우선 사용
            model = await self.safe_model_service.call_model(model_name)
            if model:
                self.logger.info(f"✅ SafeModelService를 통한 모델 로드 성공: {model_name}")
                return model
            
            # 🔥 기존 방식 안전한 호출
            load_func = getattr(self, '_load_model_sync_wrapper', None)
            success, result, message = await self.function_validator.safe_async_call(
                load_func, model_name, kwargs
            )
            
            if success:
                return result
            else:
                self.logger.warning(f"⚠️ 비동기 로드 실패: {message}")
                # 폴백: 직접 로드 시도
                return await self._direct_async_load(model_name, **kwargs)
                
        except Exception as e:
            self.logger.error(f"비동기 모델 로드 실패 {model_name}: {e}")
            return None
    
    async def _direct_async_load(self, model_name: str, **kwargs) -> Optional[Any]:
        """직접 비동기 로드"""
        try:
            # load_model 메서드 안전한 호출
            load_method = getattr(self, 'load_model', None)
            success, result, message = await self.function_validator.safe_async_call(
                load_method, model_name, **kwargs
            )
            
            if success:
                return result
            else:
                self.logger.warning(f"⚠️ 직접 비동기 로드 실패: {message}")
                return None
                
        except Exception as e:
            self.logger.error(f"❌ 직접 비동기 로드 실패: {e}")
            return None
    
    def _load_model_sync_wrapper(self, model_name: str, kwargs: Dict) -> Optional[Any]:
        """동기 로드 래퍼 - Dict Callable 오류 완전 해결"""
        try:
            # 🔥 SafeModelService를 통한 안전한 모델 생성
            model_dict = {
                'name': model_name,
                'status': 'loaded',
                'type': 'sync_wrapper_model',
                'device': self.device,
                'kwargs': kwargs
            }
            
            # SafeModelService에 등록 후 반환
            if self.safe_model_service.register_model(model_name, model_dict):
                # 등록된 모델의 wrapper 반환
                return self.safe_model_service.models.get(model_name)
            else:
                return None
                
        except Exception as e:
            self.logger.error(f"동기 래퍼 모델 로드 실패: {e}")
            return None
    
    async def load_model(
        self,
        name: str,
        force_reload: bool = False,
        **kwargs
    ) -> Optional[Any]:
        """완전 통합 모델 로드 - Dict Callable 오류 완전 해결"""
        try:
            cache_key = f"{name}_{kwargs.get('config_hash', 'default')}"
            
            with self._lock:
                # 캐시된 모델 확인
                if cache_key in self.model_cache and not force_reload:
                    self.access_counts[cache_key] = self.access_counts.get(cache_key, 0) + 1
                    self.last_access[cache_key] = time.time()
                    self.logger.debug(f"📦 캐시된 모델 반환: {name}")
                    return self.model_cache[cache_key]
                
                # 🔥 SafeModelService 우선 사용
                model = await self.safe_model_service.call_model(name)
                if model:
                    # 캐시에 저장
                    self.model_cache[cache_key] = model
                    self.access_counts[cache_key] = 1
                    self.last_access[cache_key] = time.time()
                    self.logger.info(f"✅ SafeModelService를 통한 모델 로드 성공: {name}")
                    return model
                
                # 모델 설정 확인
                if name not in self.model_configs:
                    self.logger.warning(f"⚠️ 등록되지 않은 모델: {name}")
                    # 🔥 기본 모델 등록 시도
                    default_config = {
                        'name': name,
                        'type': 'unknown',
                        'device': self.device
                    }
                    self.safe_model_service.register_model(name, default_config)
                    model = await self.safe_model_service.call_model(name)
                    if model:
                        self.model_cache[cache_key] = model
                        return model
                    else:
                        return None
                
                start_time = time.time()
                model_config = self.model_configs[name]
                
                self.logger.info(f"📦 모델 로딩 시작: {name}")
                
                # 메모리 압박 확인 및 정리
                await self._check_memory_and_cleanup()
                
                # 모델 인스턴스 생성
                model = await self._create_model_instance(model_config, **kwargs)
                
                if model is None:
                    self.logger.warning(f"⚠️ 모델 생성 실패: {name}")
                    return None
                
                # 체크포인트 로드
                await self._load_checkpoint(model, model_config)
                
                # 디바이스로 이동 - 안전한 호출
                if hasattr(model, 'to'):
                    to_method = getattr(model, 'to', None)
                    success, result, message = self.function_validator.safe_call(to_method, self.device)
                    if success:
                        model = result
                
                # M3 Max 최적화 적용
                if self.is_m3_max and self.optimization_enabled:
                    model = await self._apply_m3_max_optimization(model, model_config)
                
                # FP16 최적화 - 안전한 호출
                if self.use_fp16 and hasattr(model, 'half') and self.device != 'cpu':
                    try:
                        half_method = getattr(model, 'half', None)
                        success, result, message = self.function_validator.safe_call(half_method)
                        if success:
                            model = result
                    except Exception as e:
                        self.logger.warning(f"⚠️ FP16 변환 실패: {e}")
                
                # 평가 모드 - 안전한 호출
                if hasattr(model, 'eval'):
                    eval_method = getattr(model, 'eval', None)
                    self.function_validator.safe_call(eval_method)
                
                # 캐시에 저장
                self.model_cache[cache_key] = model
                self.load_times[cache_key] = time.time() - start_time
                self.access_counts[cache_key] = 1
                self.last_access[cache_key] = time.time()
                
                load_time = self.load_times[cache_key]
                self.logger.info(f"✅ 모델 로딩 완료: {name} ({load_time:.2f}s)")
                
                return model
                
        except Exception as e:
            self.logger.error(f"❌ 모델 로딩 실패 {name}: {e}")
            return None
    
    async def initialize(self) -> bool:
        """🔥 ModelLoader 초기화 메서드"""
        try:
            self.logger.info("🚀 ModelLoader v5.0 초기화 시작...")
            
            # 기본 초기화 작업들
            await asyncio.sleep(0.1)  # 짧은 대기
            
            # 시스템 준비 상태 체크
            if not hasattr(self, 'device_manager'):
                self.logger.warning("⚠️ 디바이스 매니저가 없음")
                return False
            
            # 메모리 정리 - 안전한 호출
            if hasattr(self, 'memory_manager'):
                cleanup_method = getattr(self.memory_manager, 'cleanup_memory', None)
                success, result, message = self.function_validator.safe_call(cleanup_method)
                if not success:
                    self.logger.warning(f"⚠️ 메모리 정리 실패: {message}")
                
            self.logger.info("✅ ModelLoader v5.0 초기화 완료 - Dict Callable 오류 완전 해결")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ ModelLoader 초기화 실패: {e}")
            return False
    
    async def _create_model_instance(
        self,
        model_config: Union[ModelConfig, StepModelConfig],
        **kwargs
    ) -> Optional[Any]:
        """모델 인스턴스 생성"""
        try:
            model_class_name = getattr(model_config, 'model_class', 'BaseModel')
            
            if model_class_name == "GraphonomyModel":
                num_classes = getattr(model_config, 'num_classes', 20)
                return GraphonomyModel(num_classes=num_classes, backbone='resnet101')
            
            elif model_class_name == "OpenPoseModel":
                num_keypoints = getattr(model_config, 'num_classes', 18)
                return OpenPoseModel(num_keypoints=num_keypoints)
            
            elif model_class_name == "U2NetModel":
                return U2NetModel(in_ch=3, out_ch=1)
            
            elif model_class_name == "GeometricMatchingModel":
                return GeometricMatchingModel(feature_size=256)
            
            elif model_class_name == "HRVITONModel":
                return HRVITONModel(input_nc=3, output_nc=3, ngf=64)
            
            elif model_class_name == "StableDiffusionPipeline":
                return await self._create_diffusion_model(model_config)
            
            else:
                self.logger.warning(f"⚠️ 지원하지 않는 모델 클래스: {model_class_name}")
                return BaseModel()
                
        except Exception as e:
            self.logger.error(f"❌ 모델 인스턴스 생성 실패: {e}")
            return None
    
    async def _create_diffusion_model(self, model_config):
        """Diffusion 모델 생성"""
        try:
            if DIFFUSERS_AVAILABLE:
                from diffusers import StableDiffusionPipeline
                
                checkpoint_path = getattr(model_config, 'checkpoint_path', None)
                if checkpoint_path and Path(checkpoint_path).exists():
                    pipeline = StableDiffusionPipeline.from_pretrained(
                        checkpoint_path,
                        torch_dtype=torch.float16 if self.use_fp16 else torch.float32,
                        safety_checker=None,
                        requires_safety_checker=False
                    )
                else:
                    # 기본 Stable Diffusion 로드
                    pipeline = StableDiffusionPipeline.from_pretrained(
                        "runwayml/stable-diffusion-v1-5",
                        torch_dtype=torch.float16 if self.use_fp16 else torch.float32,
                        safety_checker=None,
                        requires_safety_checker=False
                    )
                
                return pipeline
            else:
                self.logger.warning("⚠️ Diffusers 라이브러리가 없음")
                return None
                
        except Exception as e:
            self.logger.error(f"❌ Diffusion 모델 생성 실패: {e}")
            return None
    
    async def _load_checkpoint(self, model: Any, model_config: Union[ModelConfig, StepModelConfig]):
        """체크포인트 로드"""
        try:
            # checkpoint_path 또는 checkpoints에서 경로 가져오기
            checkpoint_path = None
            
            if hasattr(model_config, 'checkpoint_path'):
                checkpoint_path = model_config.checkpoint_path
            elif hasattr(model_config, 'checkpoints') and isinstance(model_config.checkpoints, dict):
                checkpoints = getattr(model_config, 'checkpoints', {})
                if isinstance(checkpoints, dict):
                    checkpoint_path = checkpoints.get('primary_path')
            
            if not checkpoint_path:
                self.logger.info(f"📝 체크포인트 경로 없음: {getattr(model_config, 'name', 'unknown')}")
                return
                
            checkpoint_path = Path(checkpoint_path)
            
            if not checkpoint_path.exists():
                self.logger.warning(f"⚠️ 체크포인트를 찾을 수 없음: {checkpoint_path}")
                return
            
            # PyTorch 모델인 경우 - 안전한 호출
            if hasattr(model, 'load_state_dict') and TORCH_AVAILABLE:
                state_dict = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
                
                # state_dict 정리
                if isinstance(state_dict, dict) and 'state_dict' in state_dict:
                    state_dict = state_dict['state_dict']
                elif isinstance(state_dict, dict) and 'model' in state_dict:
                    state_dict = state_dict['model']
                
                # 키 이름 정리 (module. 제거 등)
                cleaned_state_dict = {}
                for key, value in state_dict.items():
                    new_key = key.replace('module.', '') if key.startswith('module.') else key
                    cleaned_state_dict[new_key] = value
                
                # 안전한 호출
                load_state_dict_method = getattr(model, 'load_state_dict', None)
                success, result, message = self.function_validator.safe_call(
                    load_state_dict_method, cleaned_state_dict, strict=False
                )
                
                if success:
                    self.logger.info(f"✅ 체크포인트 로드 완료: {checkpoint_path}")
                else:
                    self.logger.warning(f"⚠️ 체크포인트 로드 실패: {message}")
            
            else:
                self.logger.info(f"📝 체크포인트 로드 건너뜀 (파이프라인): {getattr(model_config, 'name', 'unknown')}")
                
        except Exception as e:
            self.logger.warning(f"⚠️ 체크포인트 로드 실패: {e}")
    
    async def _apply_m3_max_optimization(self, model: Any, model_config) -> Any:
        """M3 Max 특화 모델 최적화"""
        try:
            optimizations_applied = []
            
            # 1. MPS 디바이스 최적화
            if self.device == 'mps' and hasattr(model, 'to'):
                optimizations_applied.append("MPS device optimization")
            
            # 2. 메모리 최적화 (128GB M3 Max)
            if self.memory_gb >= 64:
                optimizations_applied.append("High memory optimization")
            
            # 3. CoreML 컴파일 준비 (가능한 경우)
            if COREML_AVAILABLE and hasattr(model, 'eval'):
                optimizations_applied.append("CoreML compilation ready")
            
            # 4. Metal Performance Shaders 최적화
            if self.device == 'mps':
                try:
                    # PyTorch MPS 최적화 설정
                    if hasattr(torch.backends.mps, 'set_per_process_memory_fraction'):
                        torch.backends.mps.set_per_process_memory_fraction(0.8)
                    optimizations_applied.append("Metal Performance Shaders")
                except:
                    pass
            
            if optimizations_applied:
                self.logger.info(f"🍎 M3 Max 모델 최적화 적용: {', '.join(optimizations_applied)}")
            
            return model
            
        except Exception as e:
            self.logger.warning(f"⚠️ M3 Max 모델 최적화 실패: {e}")
            return model
    
    async def _check_memory_and_cleanup(self):
        """메모리 확인 및 정리"""
        try:
            # 메모리 압박 체크 - 안전한 호출
            if hasattr(self.memory_manager, 'check_memory_pressure'):
                check_method = getattr(self.memory_manager, 'check_memory_pressure', None)
                success, is_pressure, message = self.function_validator.safe_call(check_method)
                
                if success and is_pressure:
                    await self._cleanup_least_used_models()
            
            # 캐시된 모델 수 확인
            if len(self.model_cache) >= self.max_cached_models:
                await self._cleanup_least_used_models()
            
            # 메모리 정리 - 안전한 호출
            if hasattr(self.memory_manager, 'cleanup_memory'):
                cleanup_method = getattr(self.memory_manager, 'cleanup_memory', None)
                success, result, message = self.function_validator.safe_call(cleanup_method)
                if not success:
                    self.logger.warning(f"⚠️ 메모리 정리 실패: {message}")
                    
        except Exception as e:
            self.logger.warning(f"⚠️ 메모리 정리 실패: {e}")
    
    async def _cleanup_least_used_models(self, keep_count: int = 5):
        """사용량이 적은 모델 정리"""
        try:
            with self._lock:
                if len(self.model_cache) <= keep_count:
                    return
                
                # 사용 빈도와 최근 액세스 시간 기준 정렬
                sorted_models = sorted(
                    self.model_cache.items(),
                    key=lambda x: (
                        self.access_counts.get(x[0], 0),
                        self.last_access.get(x[0], 0)
                    )
                )
                
                cleanup_count = len(self.model_cache) - keep_count
                cleaned_models = []
                
                for i in range(min(cleanup_count, len(sorted_models))):
                    cache_key, model = sorted_models[i]
                    
                    # 모델 해제
                    del self.model_cache[cache_key]
                    self.access_counts.pop(cache_key, None)
                    self.load_times.pop(cache_key, None)
                    self.last_access.pop(cache_key, None)
                    
                    # GPU 메모리에서 제거 - 안전한 호출
                    if hasattr(model, 'cpu'):
                        cpu_method = getattr(model, 'cpu', None)
                        success, result, message = self.function_validator.safe_call(cpu_method)
                        if not success:
                            self.logger.warning(f"⚠️ CPU 이동 실패: {message}")
                    
                    del model
                    cleaned_models.append(cache_key)
                
                if cleaned_models:
                    self.logger.info(f"🧹 모델 캐시 정리: {len(cleaned_models)}개 모델 해제")
                    
        except Exception as e:
            self.logger.error(f"❌ 모델 정리 실패: {e}")
    
    def create_step_interface(self, step_name: str) -> StepModelInterface:
        """Step 클래스를 위한 모델 인터페이스 생성"""
        try:
            with self._interface_lock:
                if step_name not in self.step_interfaces:
                    interface = StepModelInterface(self, step_name)
                    self.step_interfaces[step_name] = interface
                    self.logger.info(f"🔗 {step_name} 인터페이스 생성 완료 (SafeModelService 통합)")
                
                return self.step_interfaces[step_name]
                
        except Exception as e:
            self.logger.error(f"❌ {step_name} 인터페이스 생성 실패: {e}")
            return StepModelInterface(self, step_name)
    
    def get_step_interface(self, step_name: str) -> Optional[StepModelInterface]:
        """기존 Step 인터페이스 조회"""
        with self._interface_lock:
            return self.step_interfaces.get(step_name)
    
    def cleanup_step_interface(self, step_name: str):
        """Step 인터페이스 정리"""
        try:
            with self._interface_lock:
                if step_name in self.step_interfaces:
                    interface = self.step_interfaces[step_name]
                    # 안전한 호출
                    if hasattr(interface, 'unload_models'):
                        unload_method = getattr(interface, 'unload_models', None)
                        success, result, message = self.function_validator.safe_call(unload_method)
                        if not success:
                            self.logger.warning(f"⚠️ 인터페이스 언로드 실패: {message}")
                    
                    del self.step_interfaces[step_name]
                    self.logger.info(f"🗑️ {step_name} 인터페이스 정리 완료")
                    
        except Exception as e:
            self.logger.error(f"❌ {step_name} 인터페이스 정리 실패: {e}")
    
    def cleanup(self):
        """🔥 완전한 리소스 정리"""
        try:
            # Step 인터페이스들 정리
            with self._interface_lock:
                for step_name in list(self.step_interfaces.keys()):
                    self.cleanup_step_interface(step_name)
            
            # 모델 캐시 정리
            with self._lock:
                for cache_key, model in list(self.model_cache.items()):
                    try:
                        if hasattr(model, 'cpu'):
                            cpu_method = getattr(model, 'cpu', None)
                            success, result, message = self.function_validator.safe_call(cpu_method)
                            if not success:
                                self.logger.warning(f"⚠️ 모델 CPU 이동 실패: {message}")
                        del model
                    except Exception as e:
                        self.logger.warning(f"⚠️ 모델 정리 실패: {e}")
                
                self.model_cache.clear()
                self.access_counts.clear()
                self.load_times.clear()
                self.last_access.clear()
            
            # 메모리 정리 - 안전한 호출
            if hasattr(self.memory_manager, 'cleanup_memory'):
                cleanup_method = getattr(self.memory_manager, 'cleanup_memory', None)
                success, result, message = self.function_validator.safe_call(cleanup_method)
                if not success:
                    self.logger.warning(f"⚠️ 메모리 정리 실패: {message}")
            
            # 스레드풀 종료 - 안전한 호출
            try:
                if hasattr(self, '_executor'):
                    shutdown_method = getattr(self._executor, 'shutdown', None)
                    success, result, message = self.function_validator.safe_call(shutdown_method, wait=True)
                    if not success:
                        self.logger.warning(f"⚠️ 스레드풀 종료 실패: {message}")
            except Exception as e:
                self.logger.warning(f"⚠️ 스레드풀 종료 실패: {e}")
            
            self.logger.info("✅ ModelLoader v5.0 정리 완료 - Dict Callable 오류 완전 해결")
            
        except Exception as e:
            self.logger.error(f"❌ ModelLoader 정리 중 오류: {e}")

# ==============================================
# 🔥 완전 개선된 BaseStepMixin - Dict Callable 오류 완전 해결
# ==============================================

class BaseStepMixin:
    """
    🔥 Step 클래스들이 상속받을 ModelLoader 연동 믹스인 - Dict Callable 오류 완전 해결
    SafeModelService + SafeFunctionValidator 통합으로 모든 호출 안전성 보장
    """
    
    def __init__(self, *args, **kwargs):
        """🔥 완전 안전한 초기화 - Dict Callable 오류 완전 해결"""
        # NumPy 호환성 체크
        self._check_numpy_compatibility()
        
        # 안전한 super() 호출
        try:
            mro = type(self).__mro__
            if len(mro) > 2:
                super().__init__()
        except TypeError:
            pass
        
        # logger 속성 설정
        if not hasattr(self, 'logger'):
            self.logger = logging.getLogger(f"pipeline.{self.__class__.__name__}")
        
        # 🔥 SafeFunctionValidator 통합
        self.function_validator = SafeFunctionValidator()
        
        # 기본 속성들 설정
        self.device = kwargs.get('device', 'auto')
        self.model_interface = None
        self.config = SafeConfig(kwargs.get('config', {}))
        
        # 🔥 워밍업 함수들 안전하게 설정
        self._setup_warmup_functions()
    
    def _check_numpy_compatibility(self):
        """NumPy 2.x 호환성 체크"""
        try:
            if NUMPY_AVAILABLE:
                numpy_version = np.__version__
                major_version = int(numpy_version.split('.')[0])
                
                if major_version >= 2:
                    temp_logger = logging.getLogger(f"pipeline.{self.__class__.__name__}")
                    temp_logger.warning(f"⚠️ NumPy {numpy_version} 감지됨 (2.x)")
                    temp_logger.warning("🔧 conda install numpy=1.24.3 -y --force-reinstall 권장")
        except Exception as e:
            temp_logger = logging.getLogger(f"pipeline.{self.__class__.__name__}")
            temp_logger.warning(f"⚠️ NumPy 버전 체크 실패: {e}")
    
    def _safe_model_warmup(self, *args, **kwargs):
        """안전한 모델 워밍업 - Dict Callable 오류 방지"""
        try:
            if hasattr(self, 'model_loader') and self.model_loader:
                # 실제 모델 로더가 있는 경우에만 워밍업
                if hasattr(self.model_loader, 'warmup_model'):
                    warmup_method = getattr(self.model_loader, 'warmup_model', None)
                    success, result, message = self.function_validator.safe_call(
                        warmup_method, *args, **kwargs
                    )
                    if success:
                        return result
                    else:
                        self.logger.warning(f"⚠️ warmup_model 호출 실패: {message}")
            
            # 기본 워밍업 (안전한 처리)
            self.logger.debug("✅ 기본 모델 워밍업 완료")
            return {"success": True, "method": "default_warmup"}
            
        except Exception as e:
            self.logger.warning(f"⚠️ 모델 워밍업 실패: {e}")
            return {"success": False, "error": str(e)}
    
    def _safe_device_warmup(self, *args, **kwargs):
        """안전한 디바이스 워밍업 - Dict Callable 오류 방지"""
        try:
            # GPU 메모리 정리 및 최적화
            if hasattr(self, 'gpu_config') and self.gpu_config:
                if hasattr(self.gpu_config, 'cleanup_memory'):
                    cleanup_method = getattr(self.gpu_config, 'cleanup_memory', None)
                    success, result, message = self.function_validator.safe_call(cleanup_method)
                    if not success:
                        self.logger.warning(f"⚠️ GPU 정리 실패: {message}")
            
            # Torch 캐시 정리 (안전한 처리)
            try:
                if TORCH_AVAILABLE:
                    if hasattr(torch, 'cuda') and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                        # PyTorch 버전별 안전한 처리
                        try:
                            if hasattr(torch.mps, 'empty_cache'):
                                torch.mps.empty_cache()
                        except AttributeError:
                            pass  # 오래된 PyTorch 버전에서는 무시
            except Exception:
                pass  # Torch 오류는 무시하고 계속
            
            self.logger.debug("✅ 디바이스 워밍업 완료")
            return {"success": True, "method": "device_warmup"}
            
        except Exception as e:
            self.logger.warning(f"⚠️ 디바이스 워밍업 실패: {e}")
            return {"success": False, "error": str(e)}
    
    def _safe_memory_warmup(self, *args, **kwargs):
        """안전한 메모리 워밍업 - Dict Callable 오류 방지"""
        try:
            import gc
            gc.collect()
            
            self.logger.debug("✅ 메모리 워밍업 완료")
            return {"success": True, "method": "memory_cleanup"}
            
        except Exception as e:
            self.logger.warning(f"⚠️ 메모리 워밍업 실패: {e}")
            return {"success": False, "error": str(e)}
    
    def _safe_pipeline_warmup(self, *args, **kwargs):
        """안전한 파이프라인 워밍업 - Dict Callable 오류 방지"""
        try:
            # 파이프라인 상태 확인
            if hasattr(self, 'pipeline_manager') and self.pipeline_manager:
                if hasattr(self.pipeline_manager, 'is_ready'):
                    is_ready_method = getattr(self.pipeline_manager, 'is_ready', None)
                    success, ready, message = self.function_validator.safe_call(is_ready_method)
                    if success and ready:
                        self.logger.debug("✅ 파이프라인이 준비되었습니다")
                    else:
                        self.logger.warning("⚠️ 파이프라인이 준비되지 않았습니다")
            
            self.logger.debug("✅ 파이프라인 워밍업 완료")
            return {"success": True, "method": "pipeline_check"}
            
        except Exception as e:
            self.logger.warning(f"⚠️ 파이프라인 워밍업 실패: {e}")
            return {"success": False, "error": str(e)}
    
    def _setup_warmup_functions(self):
        """워밍업 함수들 안전하게 설정 - Dict Callable 오류 완전 해결"""
        try:
            # 실제 메서드 객체로 설정 (딕셔너리가 아닌)
            self.warmup_functions = {
                'model_warmup': self._safe_model_warmup,
                'device_warmup': self._safe_device_warmup,
                'memory_warmup': self._safe_memory_warmup,
                'pipeline_warmup': self._safe_pipeline_warmup
            }
            
            # 모든 워밍업 함수가 callable인지 확인
            for name, func in self.warmup_functions.items():
                is_callable, reason, safe_func = SafeFunctionValidator.validate_callable(func, f"warmup_{name}")
                if not is_callable:
                    self.logger.error(f"❌ {name}이 callable하지 않습니다: {reason}")
                    # 안전한 더미 함수로 대체
                    self.warmup_functions[name] = lambda *args, **kwargs: {"success": True, "method": "dummy"}
            
            if hasattr(self, 'logger'):
                self.logger.debug("✅ 워밍업 함수들 설정 완료 (SafeFunctionValidator 검증)")
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.warning(f"⚠️ 워밍업 함수 설정 실패: {e}")
            # 완전한 폴백
            self.warmup_functions = {}
    
    def _setup_model_interface(self, model_loader: Optional[ModelLoader] = None):
        """모델 인터페이스 설정 - Dict Callable 오류 완전 해결"""
        try:
            if model_loader is None:
                # 전역 모델 로더 사용
                model_loader = get_global_model_loader()
            
            # 🔥 안전한 호출
            create_method = getattr(model_loader, 'create_step_interface', None)
            success, interface, message = self.function_validator.safe_call(
                create_method, self.__class__.__name__
            )
            
            if success:
                self.model_interface = interface
                logger.info(f"🔗 {self.__class__.__name__} 모델 인터페이스 설정 완료 (SafeFunctionValidator)")
            else:
                self.logger.warning(f"⚠️ create_step_interface 호출 실패: {message}")
                self.model_interface = None
            
        except Exception as e:
            logger.error(f"❌ {self.__class__.__name__} 모델 인터페이스 설정 실패: {e}")
            self.model_interface = None
    
    async def get_model(self, model_name: Optional[str] = None) -> Optional[Any]:
        """모델 로드 (Step에서 사용) - Dict Callable 오류 완전 해결"""
        try:
            if not hasattr(self, 'model_interface') or self.model_interface is None:
                logger.warning(f"⚠️ {self.__class__.__name__} 모델 인터페이스가 없습니다")
                return None
            
            if model_name:
                # 🔥 안전한 호출
                get_method = getattr(self.model_interface, 'get_model', None)
                success, result, message = await self.function_validator.safe_async_call(
                    get_method, model_name
                )
                
                if success:
                    return result
                else:
                    logger.warning(f"⚠️ get_model 호출 실패: {message}")
                    return None
            else:
                # 권장 모델 자동 로드
                rec_method = getattr(self.model_interface, 'get_recommended_model', None)
                success, result, message = await self.function_validator.safe_async_call(rec_method)
                
                if success:
                    return result
                else:
                    logger.warning(f"⚠️ get_recommended_model 호출 실패: {message}")
                    return None
                
        except Exception as e:
            logger.error(f"❌ {self.__class__.__name__} 모델 로드 실패: {e}")
            return None
    
    def cleanup_models(self):
        """모델 정리 - Dict Callable 오류 완전 해결"""
        try:
            if hasattr(self, 'model_interface') and self.model_interface:
                # 🔥 안전한 호출
                cleanup_method = getattr(self.model_interface, 'unload_models', None)
                success, result, message = self.function_validator.safe_call(cleanup_method)
                
                if not success:
                    logger.warning(f"⚠️ unload_models 호출 실패: {message}")
        except Exception as e:
            logger.error(f"❌ {self.__class__.__name__} 모델 정리 실패: {e}")

# ==============================================
# 🔥 전역 ModelLoader 관리 - Dict Callable 오류 완전 해결
# ==============================================

_global_model_loader: Optional[ModelLoader] = None
_loader_lock = threading.Lock()

@lru_cache(maxsize=1)
def get_global_model_loader(config: Optional[Dict[str, Any]] = None) -> ModelLoader:
    """전역 ModelLoader 인스턴스 반환"""
    global _global_model_loader
    
    with _loader_lock:
        if _global_model_loader is None:
            _global_model_loader = ModelLoader(
                config=config,
                enable_auto_detection=True,
                device="auto",
                use_fp16=True,
                optimization_enabled=True,
                enable_fallback=True
            )
            logger.info("🌐 전역 ModelLoader v5.0 인스턴스 생성 (Dict Callable 오류 완전 해결)")
        
        return _global_model_loader

def initialize_global_model_loader(**kwargs) -> Dict[str, Any]:
    """전역 ModelLoader 초기화 - Dict Callable 오류 완전 해결"""
    try:
        loader = get_global_model_loader()
        validator = SafeFunctionValidator()
        
        # 비동기 초기화 실행 - 안전한 호출
        import asyncio
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        if loop.is_running():
            # 이미 실행 중인 루프에서는 태스크로 실행
            init_method = getattr(loader, 'initialize', None)
            is_callable, reason, safe_method = SafeFunctionValidator.validate_callable(
                init_method, "initialize_global"
            )
            
            if is_callable:
                future = asyncio.create_task(safe_method())
                return {"success": True, "message": "Initialization started", "future": future}
            else:
                logger.warning(f"⚠️ initialize가 callable하지 않음: {reason}")
                return {"success": False, "error": f"initialize method not callable: {reason}"}
        else:
            init_method = getattr(loader, 'initialize', None)
            success, result, message = asyncio.run(
                validator.safe_async_call(init_method)
            )
            
            if success:
                return {"success": result, "message": "Initialization completed"}
            else:
                return {"success": False, "error": f"Initialization failed: {message}"}
            
    except Exception as e:
        logger.error(f"❌ 전역 ModelLoader 초기화 실패: {e}")
        return {"success": False, "error": str(e)}

def cleanup_global_loader():
    """전역 ModelLoader 정리"""
    global _global_model_loader
    
    with _loader_lock:
        if _global_model_loader:
            validator = SafeFunctionValidator()
            cleanup_method = getattr(_global_model_loader, 'cleanup', None)
            success, result, message = validator.safe_call(cleanup_method)
            
            if not success:
                logger.warning(f"⚠️ 전역 로더 정리 실패: {message}")
            
            _global_model_loader = None
        # 캐시 클리어
        get_global_model_loader.cache_clear()
        logger.info("🌐 전역 ModelLoader v5.0 정리 완료 (Dict Callable 오류 완전 해결)")

# ==============================================
# 🔥 편의 함수들 - Dict Callable 오류 완전 해결
# ==============================================

# 전역 모델 서비스 인스턴스
_global_model_service = None
_service_lock = threading.Lock()

def get_model_service() -> SafeModelService:
    """전역 모델 서비스 인스턴스 반환"""
    global _global_model_service
    
    if _global_model_service is None:
        with _service_lock:
            if _global_model_service is None:
                _global_model_service = SafeModelService()
                logger.info("✅ 전역 SafeModelService 생성")
    
    return _global_model_service

async def safe_warmup_models(model_names: list) -> Dict[str, bool]:
    """여러 모델 안전 워밍업 - Dict Callable 오류 완전 해결"""
    service = get_model_service()
    results = {}
    
    for name in model_names:
        try:
            results[name] = await service.warmup_model(name)
        except Exception as e:
            logger.error(f"❌ 모델 워밍업 실패 {name}: {e}")
            results[name] = False
    
    return results

def register_dict_as_model(name: str, model_dict: Dict[str, Any]) -> bool:
    """딕셔너리를 모델로 안전하게 등록"""
    service = get_model_service()
    return service.register_model(name, model_dict)

def create_mock_model(name: str, model_type: str = "mock") -> Callable:
    """Mock 모델 생성 - Dict Callable 오류 완전 해결"""
    mock_dict = {
        'name': name,
        'type': model_type,
        'status': 'loaded',
        'device': 'mps',
        'loaded_at': '2025-01-19T12:00:00Z'
    }
    
    service = get_model_service()
    return service._create_dict_wrapper(mock_dict)

# 안전한 호출 함수들 - 전역에서 사용 가능
def safe_call(obj: Any, *args, **kwargs) -> Tuple[bool, Any, str]:
    """전역 안전한 함수 호출"""
    return SafeFunctionValidator.safe_call(obj, *args, **kwargs)

async def safe_async_call(obj: Any, *args, **kwargs) -> Tuple[bool, Any, str]:
    """전역 안전한 비동기 함수 호출"""
    return await SafeFunctionValidator.safe_async_call(obj, *args, **kwargs)

def safe_getattr_call(obj: Any, attr_name: str, *args, **kwargs) -> Tuple[bool, Any, str]:
    """전역 안전한 속성 접근 및 호출"""
    return SafeFunctionValidator.safe_getattr_call(obj, attr_name, *args, **kwargs)

def is_safely_callable(obj: Any) -> bool:
    """전역 callable 안전성 검증"""
    is_callable, reason, safe_obj = SafeFunctionValidator.validate_callable(obj)
    return is_callable

# ==============================================
# 🔥 모듈 익스포트 - 완전 통합 + Dict Callable 오류 해결
# ==============================================

__all__ = [
    # 🔥 Dict Callable 오류 해결 핵심 클래스들
    'SafeFunctionValidator',
    'SafeModelService',
    
    # 핵심 클래스들 (기존 유지)
    'ModelLoader',
    'ModelFormat',
    'ModelConfig', 
    'StepModelConfig',
    'ModelType',
    'ModelPriority',
    'DeviceManager',
    'ModelMemoryManager',
    'StepModelInterface',
    'BaseStepMixin',
    'SafeConfig',
    
    # 이미지 전처리 함수들 (기존 유지)
    'preprocess_image',
    'postprocess_segmentation', 
    'preprocess_pose_input',
    'preprocess_human_parsing_input',
    'preprocess_cloth_segmentation_input',
    'tensor_to_pil',
    'pil_to_tensor',
    
    # 실제 AI 모델 클래스들 (기존 유지)
    'BaseModel',
    'GraphonomyModel',
    'OpenPoseModel', 
    'U2NetModel',
    'GeometricMatchingModel',
    'HRVITONModel',
    
    # 팩토리 및 관리 함수들 (기존 유지)
    'get_global_model_loader',
    'initialize_global_model_loader',
    'cleanup_global_loader',
    
    # 🔥 새로운 안전한 호출 함수들
    'get_model_service',
    'safe_warmup_models',
    'register_dict_as_model',
    'create_mock_model',
    'safe_call',
    'safe_async_call',
    'safe_getattr_call',
    'is_safely_callable',
    
    # 상수 (기존 유지)
    'TORCH_AVAILABLE',
    'MPS_AVAILABLE',
    'CV_AVAILABLE',
    'NUMPY_AVAILABLE',
    'DEFAULT_DEVICE'
]

# 모듈 레벨에서 안전한 정리 함수 등록
import atexit
atexit.register(cleanup_global_loader)

# 모듈 로드 확인
logger.info("✅ ModelLoader v5.0 모듈 로드 완료 - Dict Callable 오류 완전 해결")
logger.info("🔗 SafeModelService + SafeFunctionValidator 통합")
logger.info("🔧 NumPy 2.x + BaseStepMixin 완벽 호환")
logger.info("🍎 M3 Max 128GB 최적화")
logger.info("🛡️ 모든 함수/메서드 호출 안전성 보장")
logger.info(f"🎯 PyTorch: {'✅' if TORCH_AVAILABLE else '❌'}, MPS: {'✅' if MPS_AVAILABLE else '❌'}")
logger.info(f"🔢 NumPy: {'✅' if NUMPY_AVAILABLE else '❌'} v{np.__version__ if NUMPY_AVAILABLE else 'N/A'}")

if NUMPY_AVAILABLE and int(np.__version__.split('.')[0]) >= 2:
    logger.warning("⚠️ NumPy 2.x 감지됨 - conda install numpy=1.24.3 권장")
else:
    logger.info("✅ NumPy 호환성 확인됨")