#!/usr/bin/env python3
"""
🔥 MyCloset AI - Common Imports for AI Pipeline Steps
=====================================================

AI pipeline step 파일들에서 공통으로 사용되는 import 블록들을 정리한 유틸리티 모듈입니다.
각 step 파일에서 이 모듈을 import하여 중복된 import 코드를 제거할 수 있습니다.

Author: MyCloset AI Team
Date: 2025-07-31
Version: 1.0
"""

# ==============================================
# 🔥 표준 라이브러리 imports
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

from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List, Union, Callable, TYPE_CHECKING
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache, wraps
from contextlib import asynccontextmanager

# ==============================================
# 🔥 통합된 에러 처리 시스템 import
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
except ImportError:
    EXCEPTIONS_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("통합 에러 처리 시스템을 import할 수 없습니다. 기본 에러 처리만 사용합니다.")

# ==============================================
# 🔥 Mock Data Diagnostic System import
# ==============================================
try:
    from app.core.mock_data_diagnostic import (
        MockDataDiagnostic,
        diagnose_step_data,
        get_diagnostic_summary,
        diagnostic_decorator
    )
    MOCK_DIAGNOSTIC_AVAILABLE = True
except ImportError:
    MOCK_DIAGNOSTIC_AVAILABLE = False

# ==============================================
# 🔥 Central Hub DI Container 안전 import (순환참조 방지)
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

# ==============================================
# 🔥 AI/ML 라이브러리 imports (선택적)
# ==============================================

# NumPy 필수
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

# PyTorch 필수 (MPS 지원)
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from torchvision import transforms
    TORCH_AVAILABLE = True
    MPS_AVAILABLE = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
except ImportError:
    TORCH_AVAILABLE = False
    MPS_AVAILABLE = False
    torch = None
    nn = None
    F = None
    DataLoader = None
    transforms = None

# PIL 필수
try:
    from PIL import Image, ImageFilter, ImageEnhance
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    Image = None
    ImageFilter = None
    ImageEnhance = None

# OpenCV 선택사항
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    cv2 = None
    
# SciPy 선택사항
try:
    import scipy
    import scipy.ndimage as ndimage  # 홀 채우기에서 사용
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    scipy = None
    ndimage = None

# DenseCRF 고급 후처리 (선택사항)
try:
    import pydensecrf.densecrf as dcrf
    from pydensecrf.utils import unary_from_softmax
    DENSECRF_AVAILABLE = True
except ImportError:
    DENSECRF_AVAILABLE = False
    dcrf = None
    unary_from_softmax = None

# Scikit-image 고급 이미지 처리 (선택사항)
try:
    from skimage import measure, morphology, segmentation, filters
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    measure = None
    morphology = None
    segmentation = None
    filters = None

# ==============================================
# 🔥 유틸리티 함수들
# ==============================================

def detect_m3_max() -> bool:
    """M3 Max 칩셋 감지"""
    try:
        import platform
        import subprocess
        
        if platform.system() != "Darwin":
            return False
            
        # M3 Max 감지
        result = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            brand = result.stdout.strip().lower()
            return 'm3 max' in brand or 'm3 pro' in brand or 'm3' in brand
            
        return False
    except Exception:
        return False

def get_available_libraries() -> Dict[str, bool]:
    """사용 가능한 라이브러리 목록 반환"""
    return {
        'numpy': NUMPY_AVAILABLE,
        'torch': TORCH_AVAILABLE,
        'mps': MPS_AVAILABLE,
        'pil': PIL_AVAILABLE,
        'cv2': CV2_AVAILABLE,
        'scipy': SCIPY_AVAILABLE,
        'densecrf': DENSECRF_AVAILABLE,
        'skimage': SKIMAGE_AVAILABLE,
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

# ==============================================
# 🔥 공통 상수들
# ==============================================

# 디바이스 상수
DEVICE_CPU = "cpu"
DEVICE_CUDA = "cuda"
DEVICE_MPS = "mps"

# 기본 설정값들
DEFAULT_INPUT_SIZE = (512, 512)
DEFAULT_CONFIDENCE_THRESHOLD = 0.7
DEFAULT_QUALITY_THRESHOLD = 0.8

# 에러 메시지 템플릿
ERROR_TEMPLATES = {
    'model_loading': "모델 로딩 실패: {model_name}",
    'inference': "추론 실패: {step_name}",
    'preprocessing': "전처리 실패: {step_name}",
    'postprocessing': "후처리 실패: {step_name}",
    'validation': "데이터 검증 실패: {step_name}"
} 