#!/usr/bin/env python3
"""
🔥 MyCloset AI - AI Pipeline Utils Package
==========================================

AI pipeline에서 사용되는 공통 유틸리티 모듈들을 포함하는 패키지입니다.

Author: MyCloset AI Team
Date: 2025-07-31
Version: 1.0
"""

from .common_imports import (
    # 표준 라이브러리
    os, sys, gc, time, asyncio, logging, threading, traceback,
    hashlib, json, base64, math, warnings,
    Path, Dict, Any, Optional, Tuple, List, Union, Callable, TYPE_CHECKING,
    dataclass, field, Enum, IntEnum, BytesIO, ThreadPoolExecutor,
    lru_cache, wraps, asynccontextmanager,
    
    # 에러 처리 시스템
    MyClosetAIException, MockDataDetectionError, DataQualityError, ModelInferenceError,
    ModelLoadingError, ImageProcessingError, DataValidationError, ConfigurationError,
    error_tracker, detect_mock_data, log_detailed_error, create_mock_data_diagnosis_response,
    track_exception, get_error_summary, create_exception_response, convert_to_mycloset_exception,
    ErrorCodes, EXCEPTIONS_AVAILABLE,
    
    # Mock Data Diagnostic
    MockDataDiagnostic, diagnose_step_data, get_diagnostic_summary, diagnostic_decorator,
    MOCK_DIAGNOSTIC_AVAILABLE,
    
    # Central Hub DI Container
    _get_central_hub_container, get_base_step_mixin_class,
    
    # AI/ML 라이브러리
    np, torch, nn, F, DataLoader, transforms, Image, cv2, scipy, ndimage,
    dcrf, unary_from_softmax, measure, morphology, segmentation, filters,
    NUMPY_AVAILABLE, TORCH_AVAILABLE, MPS_AVAILABLE, PIL_AVAILABLE,
    CV2_AVAILABLE, SCIPY_AVAILABLE, DENSECRF_AVAILABLE, SKIMAGE_AVAILABLE,
    
    # 유틸리티 함수들
    detect_m3_max, get_available_libraries, log_library_status,
    
    # 공통 상수들
    DEVICE_CPU, DEVICE_CUDA, DEVICE_MPS,
    DEFAULT_INPUT_SIZE, DEFAULT_CONFIDENCE_THRESHOLD, DEFAULT_QUALITY_THRESHOLD,
    ERROR_TEMPLATES
)

__all__ = [
    # 표준 라이브러리
    'os', 'sys', 'gc', 'time', 'asyncio', 'logging', 'threading', 'traceback',
    'hashlib', 'json', 'base64', 'math', 'warnings',
    'Path', 'Dict', 'Any', 'Optional', 'Tuple', 'List', 'Union', 'Callable', 'TYPE_CHECKING',
    'dataclass', 'field', 'Enum', 'IntEnum', 'BytesIO', 'ThreadPoolExecutor',
    'lru_cache', 'wraps', 'asynccontextmanager',
    
    # 에러 처리 시스템
    'MyClosetAIException', 'MockDataDetectionError', 'DataQualityError', 'ModelInferenceError',
    'ModelLoadingError', 'ImageProcessingError', 'DataValidationError', 'ConfigurationError',
    'error_tracker', 'detect_mock_data', 'log_detailed_error', 'create_mock_data_diagnosis_response',
    'track_exception', 'get_error_summary', 'create_exception_response', 'convert_to_mycloset_exception',
    'ErrorCodes', 'EXCEPTIONS_AVAILABLE',
    
    # Mock Data Diagnostic
    'MockDataDiagnostic', 'diagnose_step_data', 'get_diagnostic_summary', 'diagnostic_decorator',
    'MOCK_DIAGNOSTIC_AVAILABLE',
    
    # Central Hub DI Container
    '_get_central_hub_container', 'get_base_step_mixin_class',
    
    # AI/ML 라이브러리
    'np', 'torch', 'nn', 'F', 'DataLoader', 'transforms', 'Image', 'cv2', 'scipy', 'ndimage',
    'dcrf', 'unary_from_softmax', 'measure', 'morphology', 'segmentation', 'filters',
    'NUMPY_AVAILABLE', 'TORCH_AVAILABLE', 'MPS_AVAILABLE', 'PIL_AVAILABLE',
    'CV2_AVAILABLE', 'SCIPY_AVAILABLE', 'DENSECRF_AVAILABLE', 'SKIMAGE_AVAILABLE',
    
    # 유틸리티 함수들
    'detect_m3_max', 'get_available_libraries', 'log_library_status',
    
    # 공통 상수들
    'DEVICE_CPU', 'DEVICE_CUDA', 'DEVICE_MPS',
    'DEFAULT_INPUT_SIZE', 'DEFAULT_CONFIDENCE_THRESHOLD', 'DEFAULT_QUALITY_THRESHOLD',
    'ERROR_TEMPLATES'
]