#!/usr/bin/env python3
"""
🔥 MyCloset AI - Unified Common Imports for ALL AI Pipeline Steps
=================================================================

모든 AI pipeline step 파일들에서 공통으로 사용되는 import 블록들을 완전히 통합한 모듈입니다.
이 모듈 하나만 import하면 모든 Step에서 필요한 라이브러리와 유틸리티를 사용할 수 있습니다.

사용법:
    from app.ai_pipeline.utils.common_imports import *
    
    또는
    
    from app.ai_pipeline.utils.common_imports import (
        torch, nn, F, transforms, Image, np, cv2,
        TORCH_AVAILABLE, MPS_AVAILABLE, PIL_AVAILABLE,
        MyClosetAIException, track_exception, ErrorCodes
    )

Author: MyCloset AI Team
Date: 2025-08-03
Version: 2.0 (Unified for All Steps)
"""

# ==============================================
# 🔥 1단계: 표준 라이브러리 imports (Python 모범 사례 순서)
# ==============================================

import os
import sys
import gc
import time
import asyncio
import logging
import threading
import traceback
import hashlib
import json
import base64
import math
import warnings
import weakref
import uuid
import subprocess
import platform
from datetime import datetime, timedelta

from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List, Union, Callable, TYPE_CHECKING, Set
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from io import BytesIO, StringIO
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache, wraps, partial
from contextlib import asynccontextmanager, contextmanager
from collections import defaultdict, deque
from itertools import chain

# ==============================================
# 🔥 2단계: 환경 감지 및 디바이스 설정 (최우선)
# ==============================================

def detect_m3_max() -> bool:
    """M3 Max 칩셋 감지"""
    try:
        if platform.system() != "Darwin":
            return False
        result = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            brand = result.stdout.strip().lower()
            return any(chip in brand for chip in ['m3 max', 'm3 pro', 'm3', 'm2 max', 'm2 pro', 'm1 max'])
        return False
    except Exception:
        return False

def detect_conda_env() -> Dict[str, Any]:
    """Conda 환경 감지"""
    try:
        conda_env = os.environ.get('CONDA_DEFAULT_ENV', 'unknown')
        conda_prefix = os.environ.get('CONDA_PREFIX', '')
        in_conda = 'CONDA_DEFAULT_ENV' in os.environ
        
        # 메모리 감지
        memory_gb = 16.0
        try:
            if platform.system() == "Darwin":
                result = subprocess.run(['sysctl', '-n', 'hw.memsize'], 
                                      capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    memory_bytes = int(result.stdout.strip())
                    memory_gb = memory_bytes / (1024**3)
        except:
            pass
        
        return {
            'in_conda': in_conda,
            'conda_env': conda_env,
            'conda_prefix': conda_prefix,
            'is_target_env': conda_env in ['myclosetlast', 'mycloset-ai-clean'],
            'memory_gb': memory_gb
        }
    except Exception:
        return {
            'in_conda': False,
            'conda_env': 'unknown',
            'conda_prefix': '',
            'is_target_env': False,
            'memory_gb': 16.0
        }

# 환경 정보 전역 변수
IS_M3_MAX = detect_m3_max()
CONDA_INFO = detect_conda_env()
MEMORY_GB = CONDA_INFO['memory_gb']

# ==============================================
# 🔥 3단계: AI/ML 라이브러리 imports (가용성 변수 먼저 정의)
# ==============================================

# ✅ 가용성 변수들을 먼저 초기화 (매우 중요!)
TORCH_AVAILABLE = False
MPS_AVAILABLE = False
CUDA_AVAILABLE = False
PIL_AVAILABLE = False
NUMPY_AVAILABLE = False
CV2_AVAILABLE = False
SCIPY_AVAILABLE = False
SKIMAGE_AVAILABLE = False
DENSECRF_AVAILABLE = False
MEDIAPIPE_AVAILABLE = False
ULTRALYTICS_AVAILABLE = False
TRANSFORMERS_AVAILABLE = False
SAM_AVAILABLE = False
SAFETENSORS_AVAILABLE = False

# 디바이스 및 환경
DEVICE = "cpu"
TORCH_VERSION = "unknown"

# ==============================================
# NumPy (필수 - 모든 Step에서 사용)
# ==============================================
try:
    import numpy as np
    NUMPY_AVAILABLE = True
    print("✅ NumPy 로드 완료")
except ImportError as e:
    print(f"❌ NumPy import 실패: {e}")
    np = None

# ==============================================
# PyTorch (필수 - 모든 AI Step에서 사용)
# ==============================================
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from torch.cuda.amp import autocast
    import torchvision.transforms as transforms
    from torchvision.transforms.functional import resize, to_pil_image, to_tensor
    
    TORCH_AVAILABLE = True
    TORCH_VERSION = torch.__version__
    
    # 디바이스 자동 감지
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        MPS_AVAILABLE = True
        DEVICE = "mps"
        # M3 Max 최적화
        if IS_M3_MAX:
            os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
            os.environ['TORCH_MPS_PREFER_METAL'] = '1'
            try:
                torch.mps.set_per_process_memory_fraction(0.7)
            except:
                pass
    elif torch.cuda.is_available():
        CUDA_AVAILABLE = True
        DEVICE = "cuda"
    else:
        DEVICE = "cpu"
        
    print(f"✅ PyTorch {TORCH_VERSION} 로드 완료, 디바이스: {DEVICE}")
    if MPS_AVAILABLE:
        print("🍎 MPS 사용 가능")
    if CUDA_AVAILABLE:
        print("🔥 CUDA 사용 가능")
        
except ImportError as e:
    print(f"❌ PyTorch import 실패: {e}")
    torch = None
    nn = None
    F = None
    transforms = None
    DataLoader = None
    autocast = None

# ==============================================
# PIL (필수 - 모든 이미지 처리 Step에서 사용)
# ==============================================
try:
    from PIL import Image, ImageEnhance, ImageFilter, ImageDraw, ImageFont, ImageOps
    PIL_AVAILABLE = True
    print("✅ PIL 로드 완료")
except ImportError as e:
    print(f"❌ PIL import 실패: {e}")
    Image = None

# ==============================================
# OpenCV (중요 - 대부분 Step에서 사용)
# ==============================================
try:
    import cv2
    CV2_AVAILABLE = True
    print("✅ OpenCV 로드 완료")
except ImportError:
    print("⚠️ OpenCV 없음 - 일부 기능 제한")
    cv2 = None

# ==============================================
# SciPy (고급 후처리용)
# ==============================================
try:
    import scipy
    from scipy import ndimage
    from scipy.ndimage import gaussian_filter, median_filter
    from scipy.signal import convolve2d
    from scipy.optimize import curve_fit
    SCIPY_AVAILABLE = True
    print("✅ SciPy 로드 완료")
except ImportError:
    print("⚠️ SciPy 없음 - 고급 후처리 제한")
    scipy = None
    ndimage = None

# ==============================================
# Scikit-image (고급 이미지 처리)
# ==============================================
try:
    from skimage import measure, morphology, segmentation, filters, restoration, exposure
    from skimage.metrics import structural_similarity as ssim
    SKIMAGE_AVAILABLE = True
    print("✅ scikit-image 로드 완료")
except ImportError:
    print("⚠️ scikit-image 없음 - 일부 이미지 처리 제한")
    measure = None
    morphology = None
    segmentation = None
    filters = None

# ==============================================
# DenseCRF (CRF 후처리용)
# ==============================================
try:
    import pydensecrf.densecrf as dcrf
    from pydensecrf.utils import unary_from_softmax
    DENSECRF_AVAILABLE = True
    print("✅ DenseCRF 로드 완료")
except ImportError:
    print("⚠️ DenseCRF 없음 - CRF 후처리 제한")
    dcrf = None
    unary_from_softmax = None

# ==============================================
# MediaPipe (Pose Estimation용)
# ==============================================
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
    print("✅ MediaPipe 로드 완료")
except ImportError:
    print("⚠️ MediaPipe 없음 - Pose Estimation 일부 제한")
    mp = None

# ==============================================
# Ultralytics YOLO (Pose Estimation용)
# ==============================================
try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
    print("✅ Ultralytics YOLO 로드 완료")
except ImportError:
    print("⚠️ Ultralytics 없음 - YOLO 기반 기능 제한")
    YOLO = None

# ==============================================
# Transformers (일부 Step에서 사용)
# ==============================================
try:
    from transformers import AutoModel, AutoTokenizer, AutoProcessor
    TRANSFORMERS_AVAILABLE = True
    print("✅ Transformers 로드 완료")
except ImportError:
    print("⚠️ Transformers 없음 - 일부 모델 제한")
    AutoModel = None
    AutoTokenizer = None

# ==============================================
# Segment Anything Model (SAM)
# ==============================================
try:
    import segment_anything as sam
    SAM_AVAILABLE = True
    print("✅ SAM 로드 완료")
except ImportError:
    print("⚠️ SAM 없음 - Segment Anything 기능 제한")
    sam = None

# ==============================================
# SafeTensors (모델 로딩용)
# ==============================================
try:
    import safetensors.torch as st
    SAFETENSORS_AVAILABLE = True
    print("✅ SafeTensors 로드 완료")
except ImportError:
    print("⚠️ SafeTensors 없음 - 일부 모델 로딩 제한")
    st = None

# ==============================================
# 🔥 4단계: 프로젝트 내부 모듈 imports (에러 처리 포함)
# ==============================================

# ✅ 에러 처리 시스템 가용성 변수
EXCEPTIONS_AVAILABLE = False
MOCK_DIAGNOSTIC_AVAILABLE = False

# ==============================================
# 통합된 에러 처리 시스템
# ==============================================
try:
    from app.core.exceptions import (
        MyClosetAIException,
        MockDataDetectionError, 
        DataQualityError, 
        ModelInferenceError,
        ModelLoadingError,
        ImageProcessingError,
        DataValidationError,
        ConfigurationError,
        error_tracker,
        detect_mock_data,
        log_detailed_error,
        create_mock_data_diagnosis_response,
        track_exception,
        get_error_summary,
        create_exception_response,
        convert_to_mycloset_exception,
        ErrorCodes
    )
    EXCEPTIONS_AVAILABLE = True
    print("✅ 통합 에러 처리 시스템 로드 완료")
except ImportError:
    print("⚠️ 통합 에러 처리 시스템 없음 - 기본 에러 처리만 사용")
    # 폴백 클래스들
    class MyClosetAIException(Exception):
        pass
    class MockDataDetectionError(Exception):
        pass
    class DataQualityError(Exception):
        pass
    class ModelInferenceError(Exception):
        pass
    class ModelLoadingError(Exception):
        pass
    class ImageProcessingError(Exception):
        pass
    class DataValidationError(Exception):
        pass
    class ConfigurationError(Exception):
        pass
    
    # 폴백 함수들
    def detect_mock_data(data):
        return {'is_mock': False}
    def track_exception(error, context, step_id):
        pass
    def log_detailed_error(error, context, step_id):
        pass
    def create_exception_response(error, step_name, step_id, session_id):
        return {'success': False, 'error': str(error)}
    def convert_to_mycloset_exception(error, context):
        return MyClosetAIException(str(error))
    
    class ErrorCodes:
        DATA_VALIDATION_FAILED = "DATA_VALIDATION_FAILED"
        MODEL_LOADING_FAILED = "MODEL_LOADING_FAILED"
        AI_INFERENCE_FAILED = "AI_INFERENCE_FAILED"
        CONFIGURATION_ERROR = "CONFIGURATION_ERROR"

# ==============================================
# Mock Data Diagnostic System
# ==============================================
try:
    from app.core.mock_diagnostic import (
        MockDataDiagnostic,
        diagnose_step_data,
        get_diagnostic_summary,
        diagnostic_decorator
    )
    MOCK_DIAGNOSTIC_AVAILABLE = True
    print("✅ Mock 진단 시스템 로드 완료")
except ImportError:
    print("⚠️ Mock 진단 시스템 없음")
    # 폴백 클래스와 함수들
    class MockDataDiagnostic:
        pass
    def diagnose_step_data(data):
        return {'is_mock': False}
    def get_diagnostic_summary():
        return {}
    def diagnostic_decorator(func):
        return func

# ==============================================
# 🔥 5단계: Central Hub DI Container 안전 import (순환참조 방지)
# ==============================================

if TYPE_CHECKING:
    from app.core.di_container import CentralHubDIContainer
    from app.ai_pipeline.utils.model_loader import ModelLoader
    from app.ai_pipeline.steps.base_step_mixin import BaseStepMixin

def _get_central_hub_container():
    """Central Hub DI Container 안전한 동적 해결"""
    try:
        import importlib
        module = importlib.import_module('app.core.di_container')
        get_global_fn = getattr(module, 'get_global_container', None)
        if get_global_fn:
            return get_global_fn()
        return None
    except ImportError:
        return None
    except Exception:
        return None

def get_base_step_mixin_class():
    """BaseStepMixin 클래스를 동적으로 가져오기 (순환참조 방지)"""
    try:
        import importlib
        module = importlib.import_module('.base_step_mixin', package='app.ai_pipeline.steps')
        return getattr(module, 'BaseStepMixin', None)
    except ImportError:
        try:
            # 폴백: 상대 경로
            from .base_step_mixin import BaseStepMixin
            return BaseStepMixin
        except ImportError:
            logging.getLogger(__name__).error("❌ BaseStepMixin 동적 import 실패")
            return None

def _inject_dependencies_safe(step_instance):
    """Central Hub DI Container를 통한 안전한 의존성 주입"""
    try:
        container = _get_central_hub_container()
        if container and hasattr(container, 'inject_to_step'):
            return container.inject_to_step(step_instance)
        return 0
    except Exception:
        return 0

def _get_service_from_central_hub(service_key: str):
    """Central Hub를 통한 안전한 서비스 조회"""
    try:
        container = _get_central_hub_container()
        if container:
            return container.get(service_key)
        return None
    except Exception:
        return None

# ==============================================
# 🔥 6단계: 유틸리티 함수들
# ==============================================

def get_available_libraries() -> Dict[str, bool]:
    """사용 가능한 라이브러리 목록 반환"""
    return {
        'numpy': NUMPY_AVAILABLE,
        'torch': TORCH_AVAILABLE,
        'mps': MPS_AVAILABLE,
        'cuda': CUDA_AVAILABLE,
        'pil': PIL_AVAILABLE,
        'cv2': CV2_AVAILABLE,
        'scipy': SCIPY_AVAILABLE,
        'densecrf': DENSECRF_AVAILABLE,
        'skimage': SKIMAGE_AVAILABLE,
        'mediapipe': MEDIAPIPE_AVAILABLE,
        'ultralytics': ULTRALYTICS_AVAILABLE,
        'transformers': TRANSFORMERS_AVAILABLE,
        'sam': SAM_AVAILABLE,
        'safetensors': SAFETENSORS_AVAILABLE,
        'exceptions': EXCEPTIONS_AVAILABLE,
        'mock_diagnostic': MOCK_DIAGNOSTIC_AVAILABLE
    }

def log_library_status(logger: logging.Logger):
    """라이브러리 상태를 로그에 출력"""
    libraries = get_available_libraries()
    logger.info("📚 사용 가능한 라이브러리 상태:")
    for lib, available in libraries.items():
        status = "✅" if available else "❌"
        logger.info(f"  {status} {lib}")

def safe_copy(obj: Any) -> Any:
    """안전한 복사 함수 - DetailedDataSpec 에러 해결"""
    try:
        # 기본 타입들은 그대로 반환
        if obj is None or isinstance(obj, (bool, int, float, str)):
            return obj
        
        # 리스트나 튜플
        elif isinstance(obj, (list, tuple)):
            return type(obj)(safe_copy(item) for item in obj)
        
        # 딕셔너리
        elif isinstance(obj, dict):
            return {key: safe_copy(value) for key, value in obj.items()}
        
        # 집합
        elif isinstance(obj, set):
            return {safe_copy(item) for item in obj}
        
        # copy 모듈 사용 가능한 경우
        else:
            try:
                import copy
                return copy.deepcopy(obj)
            except:
                try:
                    import copy
                    return copy.copy(obj)
                except:
                    # 복사할 수 없는 경우 원본 반환 (예: 함수, 클래스 등)
                    return obj
                    
    except Exception:
        # 모든 실패 케이스에서 원본 반환
        return obj

def optimize_memory():
    """메모리 최적화 (M3 Max 특화)"""
    try:
        # Python GC
        gc.collect()
        
        # PyTorch 메모리 정리
        if TORCH_AVAILABLE:
            if MPS_AVAILABLE:
                try:
                    torch.mps.empty_cache()
                    if hasattr(torch.mps, 'synchronize'):
                        torch.mps.synchronize()
                except:
                    pass
            elif CUDA_AVAILABLE:
                try:
                    torch.cuda.empty_cache()
                except:
                    pass
                    
    except Exception:
        pass

def get_optimal_device():
    """최적 디바이스 반환"""
    if TORCH_AVAILABLE:
        if MPS_AVAILABLE and IS_M3_MAX:
            return "mps"
        elif CUDA_AVAILABLE:
            return "cuda"
    return "cpu"

# ==============================================
# 🔥 7단계: 공통 상수들
# ==============================================

# 디바이스 상수
DEVICE_CPU = "cpu"
DEVICE_CUDA = "cuda"
DEVICE_MPS = "mps"

# 기본 설정값들
DEFAULT_INPUT_SIZE = (512, 512)
DEFAULT_CONFIDENCE_THRESHOLD = 0.7
DEFAULT_QUALITY_THRESHOLD = 0.8
DEFAULT_BATCH_SIZE = 1

# Step 정보
STEP_NAMES = [
    "HumanParsingStep",
    "PoseEstimationStep", 
    "ClothSegmentationStep",
    "GeometricMatchingStep",
    "ClothWarpingStep",
    "VirtualFittingStep",
    "PostProcessingStep",
    "QualityAssessmentStep"
]

# 에러 메시지 템플릿
ERROR_TEMPLATES = {
    'model_loading': "모델 로딩 실패: {model_name}",
    'inference': "추론 실패: {step_name}",
    'preprocessing': "전처리 실패: {step_name}",
    'postprocessing': "후처리 실패: {step_name}",
    'validation': "데이터 검증 실패: {step_name}",
    'torch_unavailable': "PyTorch가 사용할 수 없습니다: {error}",
    'device_error': "디바이스 설정 오류: {device}"
}

# COCO Keypoints (Pose Estimation용)
COCO_KEYPOINTS = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

# Human Parsing Classes (20 classes)
HUMAN_PARSING_CLASSES = {
    0: 'background', 1: 'hat', 2: 'hair', 3: 'glove', 4: 'sunglasses',
    5: 'upper_clothes', 6: 'dress', 7: 'coat', 8: 'socks', 9: 'pants',
    10: 'torso_skin', 11: 'scarf', 12: 'skirt', 13: 'face',
    14: 'left_arm', 15: 'right_arm', 16: 'left_leg', 17: 'right_leg',
    18: 'left_shoe', 19: 'right_shoe'
}

# ==============================================
# 🔥 8단계: 초기화 및 환경 정보 출력
# ==============================================

# 환경 정보 출력
print(f"🔥 MyCloset AI - Common Imports v2.0 초기화 완료!")
print(f"🖥️ 환경: M3 Max={IS_M3_MAX}, conda={CONDA_INFO['conda_env']}")
print(f"🎯 디바이스: {DEVICE}, PyTorch={TORCH_AVAILABLE}, 메모리={MEMORY_GB:.1f}GB")
print(f"📚 라이브러리: PIL={PIL_AVAILABLE}, CV2={CV2_AVAILABLE}, SciPy={SCIPY_AVAILABLE}")

# 전역으로 사용 가능하도록 설정
globals()['safe_copy'] = safe_copy

# ==============================================
# 🔥 9단계: __all__ 정의 (명시적 export)
# ==============================================

__all__ = [
    # 표준 라이브러리
    'os', 'sys', 'gc', 'time', 'asyncio', 'logging', 'threading', 'traceback',
    'hashlib', 'json', 'base64', 'math', 'warnings', 'weakref', 'uuid',
    'subprocess', 'platform', 'datetime', 'timedelta',
    'Path', 'Dict', 'Any', 'Optional', 'Tuple', 'List', 'Union', 'Callable', 'TYPE_CHECKING', 'Set',
    'dataclass', 'field', 'Enum', 'IntEnum', 'BytesIO', 'StringIO', 'ThreadPoolExecutor',
    'lru_cache', 'wraps', 'partial', 'asynccontextmanager', 'contextmanager',
    'defaultdict', 'deque', 'chain',
    
    # AI/ML 라이브러리
    'np', 'torch', 'nn', 'F', 'DataLoader', 'autocast', 'transforms',
    'resize', 'to_pil_image', 'to_tensor',
    'Image', 'ImageEnhance', 'ImageFilter', 'ImageDraw', 'ImageFont', 'ImageOps',
    'cv2', 'scipy', 'ndimage', 'gaussian_filter', 'median_filter', 'convolve2d',
    'measure', 'morphology', 'segmentation', 'filters', 'restoration', 'exposure', 'ssim',
    'dcrf', 'unary_from_softmax', 'mp', 'YOLO', 'AutoModel', 'AutoTokenizer', 'sam', 'st',
    
    # 가용성 변수들
    'TORCH_AVAILABLE', 'MPS_AVAILABLE', 'CUDA_AVAILABLE', 'PIL_AVAILABLE', 'NUMPY_AVAILABLE',
    'CV2_AVAILABLE', 'SCIPY_AVAILABLE', 'SKIMAGE_AVAILABLE', 'DENSECRF_AVAILABLE',
    'MEDIAPIPE_AVAILABLE', 'ULTRALYTICS_AVAILABLE', 'TRANSFORMERS_AVAILABLE',
    'SAM_AVAILABLE', 'SAFETENSORS_AVAILABLE', 'EXCEPTIONS_AVAILABLE', 'MOCK_DIAGNOSTIC_AVAILABLE',
    
    # 환경 변수
    'IS_M3_MAX', 'CONDA_INFO', 'MEMORY_GB', 'DEVICE', 'TORCH_VERSION',
    
    # 에러 처리 시스템
    'MyClosetAIException', 'MockDataDetectionError', 'DataQualityError', 'ModelInferenceError',
    'ModelLoadingError', 'ImageProcessingError', 'DataValidationError', 'ConfigurationError',
    'error_tracker', 'detect_mock_data', 'log_detailed_error', 'create_mock_data_diagnosis_response',
    'track_exception', 'get_error_summary', 'create_exception_response', 'convert_to_mycloset_exception',
    'ErrorCodes',
    
    # Mock 진단 시스템
    'MockDataDiagnostic', 'diagnose_step_data', 'get_diagnostic_summary', 'diagnostic_decorator',
    
    # Central Hub 함수들
    '_get_central_hub_container', 'get_base_step_mixin_class', '_inject_dependencies_safe',
    '_get_service_from_central_hub',
    
    # 유틸리티 함수들
    'detect_m3_max', 'detect_conda_env', 'get_available_libraries', 'log_library_status',
    'safe_copy', 'optimize_memory', 'get_optimal_device',
    
    # 공통 상수들
    'DEVICE_CPU', 'DEVICE_CUDA', 'DEVICE_MPS',
    'DEFAULT_INPUT_SIZE', 'DEFAULT_CONFIDENCE_THRESHOLD', 'DEFAULT_QUALITY_THRESHOLD', 'DEFAULT_BATCH_SIZE',
    'STEP_NAMES', 'ERROR_TEMPLATES', 'COCO_KEYPOINTS', 'HUMAN_PARSING_CLASSES'
]