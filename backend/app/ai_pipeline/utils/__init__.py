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

# 모델 관련 모듈들은 models 패키지에서 import
try:
    from ..models import (
        # 모델 탐지 관련
        OptimizedFileMapper,
        OptimizedDetectedModel,
        OptimizedModelDetector,
        get_global_detector,
        quick_model_detection,
        detect_ultra_large_models,
        detect_available_models,
        validate_model_structure,
        create_global_detector,
        cleanup_global_detector,
        get_model_detection_summary,
        
        # 모델 로딩 관련
        ModelLoader,
        StepModelInterface,
        StepModelFactory,
        get_global_model_loader,
        initialize_global_model_loader,
        get_model_loader_v6,
        get_step_factory,
        load_model_for_step,
        create_step_interface,
        initialize_all_steps,
        analyze_checkpoint,
        cleanup_model_loader,
        
        # 동적 모델 탐지 관련
        DynamicModelDetector,
        ModelCategory,
        DetectedModelFile,
        
        # 신경망 아키텍처 관련
        NeuralArchitectureManager,
        ArchitectureRegistry,
        
        # 체크포인트 관련
        CheckpointModelLoader,
        CheckpointAnalyzer,
        
        # 모델 요청 관련
        StepModelRequest,
        get_step_request,
        get_all_step_requests,
        StepModelAnalyzer,
        
        # 모델 아키텍처 관련
        SAMModel,
        U2NetModel,
        OpenPoseModel,
        GMMModel,
        TOMModel,
        OOTDModel,
        TPSModel,
        RAFTModel,
        RealESRGANModel,
        CLIPModel,
        LPIPSModel,
        DeepLabV3PlusModel,
        MobileSAMModel,
        VITONHDModel,
        GFPGANModel,
        HRNetPoseModel,
        GraphonomyModel
    )
    MODELS_AVAILABLE = True
except ImportError as e:
    MODELS_AVAILABLE = False
    print(f"⚠️ Models 패키지 import 실패: {e}")

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
    'ERROR_TEMPLATES',
    
    # 모델 관련 (models 패키지에서)
    'MODELS_AVAILABLE'
]

# 모델 관련 모듈들이 사용 가능한 경우에만 추가
if MODELS_AVAILABLE:
    __all__.extend([
        # 모델 탐지 관련
        'OptimizedFileMapper',
        'OptimizedDetectedModel',
        'OptimizedModelDetector',
        'get_global_detector',
        'quick_model_detection',
        'detect_ultra_large_models',
        'detect_available_models',
        'validate_model_structure',
        'create_global_detector',
        'cleanup_global_detector',
        'get_model_detection_summary',
        
        # 모델 로딩 관련
        'ModelLoader',
        'StepModelInterface',
        'StepModelFactory',
        'get_global_model_loader',
        'initialize_global_model_loader',
        'get_model_loader_v6',
        'get_step_factory',
        'load_model_for_step',
        'create_step_interface',
        'initialize_all_steps',
        'analyze_checkpoint',
        'cleanup_model_loader',
        
        # 동적 모델 탐지 관련
        'DynamicModelDetector',
        'ModelCategory',
        'DetectedModelFile',
        
        # 신경망 아키텍처 관련
        'NeuralArchitectureManager',
        'ArchitectureRegistry',
        
        # 체크포인트 관련
        'CheckpointModelLoader',
        'CheckpointAnalyzer',
        
        # 모델 요청 관련
        'StepModelRequest',
        'get_step_request',
        'get_all_step_requests',
        'StepModelAnalyzer',
        
        # 모델 아키텍처 관련
        'SAMModel',
        'U2NetModel',
        'OpenPoseModel',
        'GMMModel',
        'TOMModel',
        'OOTDModel',
        'TPSModel',
        'RAFTModel',
        'RealESRGANModel',
        'CLIPModel',
        'LPIPSModel',
        'DeepLabV3PlusModel',
        'MobileSAMModel',
        'VITONHDModel',
        'GFPGANModel',
        'HRNetPoseModel',
        'GraphonomyModel'
    ])