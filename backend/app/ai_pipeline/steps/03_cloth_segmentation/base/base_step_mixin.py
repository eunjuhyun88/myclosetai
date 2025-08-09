#!/usr/bin/env python3
"""
🔥 MyCloset AI - Step 03: 의류 세그멘테이션 - Base Step Mixin
=====================================================================

BaseStepMixin 클래스와 관련 유틸리티 함수들을 분리한 모듈

Author: MyCloset AI Team  
Date: 2025-08-01
Version: 1.0
"""

import logging
import weakref
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple, TYPE_CHECKING

# 공통 imports 시스템 사용
try:
    from app.ai_pipeline.utils.common_imports import (
        os, gc, time, threading, math, hashlib, json, base64, warnings, np,
        Path, Dict, Any, Optional, Union, List, Tuple, TYPE_CHECKING,
        dataclass, field, Enum, BytesIO, ThreadPoolExecutor,
        MyClosetAIException, ModelLoadingError, ImageProcessingError, DataValidationError, ConfigurationError,
        error_tracker, track_exception, get_error_summary, create_exception_response, convert_to_mycloset_exception,
        ErrorCodes, EXCEPTIONS_AVAILABLE,
        detect_mock_data, diagnose_step_data, MOCK_DIAGNOSTIC_AVAILABLE,
        cv2, PIL_AVAILABLE, CV2_AVAILABLE, NUMPY_AVAILABLE, Image, ImageEnhance
    )
except ImportError:
    # 폴백 imports
    import os
    import gc
    import time
    import logging
    import threading
    import math
    import hashlib
    import json
    import base64
    import warnings
    from typing import Dict, Any, Optional, List, Tuple, TYPE_CHECKING
    from dataclasses import dataclass, field
    from enum import Enum
    from io import BytesIO
    from concurrent.futures import ThreadPoolExecutor
    from pathlib import Path

# PIL Image import 추가
if 'PIL_AVAILABLE' in globals() and PIL_AVAILABLE:
    from PIL import Image as PILImage

# 경고 무시 설정
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=ImportWarning)

logger = logging.getLogger(__name__)

def detect_m3_max():
    """M3 Max 감지"""
    try:
        import platform
        import subprocess
        if platform.system() == 'Darwin':
            result = subprocess.run(
                ['sysctl', '-n', 'machdep.cpu.brand_string'],
                capture_output=True, text=True, timeout=5
            )
            return 'M3' in result.stdout
    except:
        pass
    return False

IS_M3_MAX = detect_m3_max()
MEMORY_GB = 16.0

class BaseStepMixin:
    """ClothSegmentationStep용 BaseStepMixin 클래스"""
    
    def __init__(self, **kwargs):
        print(f"🔥 [디버깅] BaseStepMixin __init__ 시작")
        
        # 기본 속성들 (안전한 초기화)
        try:
            print(f"🔥 [디버깅] Logger 초기화 시작")
            self.logger = logging.getLogger(self.__class__.__name__)
            print(f"🔥 [디버깅] Logger 초기화 완료")
        except Exception as e:
            print(f"🔥 [디버깅] Logger 초기화 실패: {e}")
            self.logger = None
        
        try:
            print(f"🔥 [디버깅] 기본 속성 설정 시작")
            self.step_name = kwargs.get('step_name', 'ClothSegmentationStep')
            self.step_id = kwargs.get('step_id', 3)
            self.device = kwargs.get('device', 'cpu')
            print(f"🔥 [디버깅] 기본 속성 설정 완료")
        except Exception as e:
            print(f"🔥 [디버깅] 기본 속성 설정 실패: {e}")
            self.step_name = 'ClothSegmentationStep'
            self.step_id = 3
            self.device = 'cpu'
        
        # AI 모델 관련 속성들 (ClothSegmentation이 필요로 하는)
        try:
            print(f"🔥 [디버깅] AI 모델 속성 초기화 시작")
            self.ai_models = {}
            self.models_loading_status = {
                'deeplabv3plus': False,
                'maskrcnn': False,
                'sam_huge': False,
                'u2net_cloth': False,
                'total_loaded': 0,
                'loading_errors': []
            }
            self.model_interface = None
            self.loaded_models = {}
            print(f"🔥 [디버깅] AI 모델 속성 초기화 완료")
        except Exception as e:
            print(f"🔥 [디버깅] AI 모델 속성 초기화 실패: {e}")
            self.ai_models = {}
            self.models_loading_status = {'loading_errors': []}
            self.model_interface = None
            self.loaded_models = {}
        
        # ClothSegmentation 특화 속성들
        try:
            print(f"🔥 [디버깅] ClothSegmentation 속성 초기화 시작")
            self.segmentation_models = {}
            self.segmentation_ready = False
            self.cloth_cache = {}
            print(f"🔥 [디버깅] ClothSegmentation 속성 초기화 완료")
        except Exception as e:
            print(f"🔥 [디버깅] ClothSegmentation 속성 초기화 실패: {e}")
            self.segmentation_models = {}
            self.segmentation_ready = False
            self.cloth_cache = {}
        
        # 의류 카테고리 정의
        try:
            print(f"🔥 [디버깅] 의류 카테고리 정의 시작")
            self.cloth_categories = {
                0: 'background',
                1: 'shirt', 2: 't_shirt', 3: 'sweater', 4: 'hoodie',
                5: 'jacket', 6: 'coat', 7: 'dress', 8: 'skirt',
                9: 'pants', 10: 'jeans', 11: 'shorts',
                12: 'shoes', 13: 'boots', 14: 'sneakers',
                15: 'bag', 16: 'hat', 17: 'glasses', 18: 'scarf', 19: 'belt'
            }
            print(f"🔥 [디버깅] 의류 카테고리 정의 완료")
        except Exception as e:
            print(f"🔥 [디버깅] 의류 카테고리 정의 실패: {e}")
            self.cloth_categories = {}
        
        # 상태 관련 속성들
        try:
            print(f"🔥 [디버깅] 상태 속성 초기화 시작")
            self.is_initialized = False
            self.is_ready = False
            self.has_model = False
            self.model_loaded = False
            print(f"🔥 [디버깅] 상태 속성 초기화 완료")
        except Exception as e:
            print(f"🔥 [디버깅] 상태 속성 초기화 실패: {e}")
            self.is_initialized = False
            self.is_ready = False
            self.has_model = False
            self.model_loaded = False

    def process(self, **kwargs) -> Dict[str, Any]:
        """기본 처리 메서드"""
        try:
            self.logger.info(f"🔥 {self.step_name} 처리 시작")
            
            # 입력 검증
            if not self._validate_input(kwargs):
                return self._create_error_response("입력 검증 실패")
            
            # AI 추론 실행
            result = self._run_ai_inference(kwargs)
            
            # 결과 검증
            if not self._validate_output(result):
                return self._create_error_response("출력 검증 실패")
            
            self.logger.info(f"🔥 {self.step_name} 처리 완료")
            return result
            
        except Exception as e:
            self.logger.error(f"❌ {self.step_name} 처리 실패: {e}")
            return self._create_error_response(str(e))

    def initialize(self) -> bool:
        """초기화 메서드"""
        try:
            self.logger.info(f"🔥 {self.step_name} 초기화 시작")
            
            # 기본 초기화
            if not self._initialize_basic():
                return False
            
            # 모델 로딩
            if not self._load_models():
                return False
            
            # 상태 설정
            self.is_initialized = True
            self.is_ready = True
            
            self.logger.info(f"🔥 {self.step_name} 초기화 완료")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ {self.step_name} 초기화 실패: {e}")
            return False

    def cleanup(self):
        """정리 메서드"""
        try:
            self.logger.info(f"🔥 {self.step_name} 정리 시작")
            
            # 모델 정리
            self._cleanup_models()
            
            # 캐시 정리
            self._cleanup_cache()
            
            # 상태 초기화
            self.is_initialized = False
            self.is_ready = False
            
            self.logger.info(f"🔥 {self.step_name} 정리 완료")
            
        except Exception as e:
            self.logger.error(f"❌ {self.step_name} 정리 실패: {e}")

    def get_status(self) -> Dict[str, Any]:
        """상태 조회 메서드"""
        return {
            'step_name': self.step_name,
            'step_id': self.step_id,
            'is_initialized': self.is_initialized,
            'is_ready': self.is_ready,
            'has_model': self.has_model,
            'model_loaded': self.model_loaded,
            'device': self.device,
            'models_loading_status': self.models_loading_status
        }

    def set_model_loader(self, model_loader):
        """모델 로더 설정"""
        self.model_loader = model_loader

    def set_memory_manager(self, memory_manager):
        """메모리 매니저 설정"""
        self.memory_manager = memory_manager

    def set_data_converter(self, data_converter):
        """데이터 변환기 설정"""
        self.data_converter = data_converter

    def set_di_container(self, di_container):
        """DI 컨테이너 설정"""
        self.di_container = di_container

    def _get_step_requirements(self) -> Dict[str, Any]:
        """스텝 요구사항 조회"""
        return {
            'step_name': self.step_name,
            'step_id': self.step_id,
            'required_models': ['deeplabv3plus', 'u2net_cloth', 'sam_huge'],
            'required_memory': '8GB',
            'required_device': 'cpu'
        }

    def _validate_input(self, kwargs) -> bool:
        """입력 검증"""
        return True

    def _validate_output(self, result) -> bool:
        """출력 검증"""
        return True

    def _create_error_response(self, message: str) -> Dict[str, Any]:
        """에러 응답 생성"""
        return {
            'success': False,
            'error': message,
            'step_name': self.step_name,
            'step_id': self.step_id
        }

    def _initialize_basic(self) -> bool:
        """기본 초기화"""
        return True

    def _load_models(self) -> bool:
        """모델 로딩"""
        return True

    def _cleanup_models(self):
        """모델 정리"""
        pass

    def _cleanup_cache(self):
        """캐시 정리"""
        pass

    def _run_ai_inference(self, kwargs) -> Dict[str, Any]:
        """AI 추론 실행"""
        return {'success': True, 'message': '기본 추론 완료'}

def _get_central_hub_container():
    """중앙 허브 컨테이너 조회"""
    try:
        from app.ai_pipeline.core.di_container import DIContainer
        return DIContainer.get_instance()
    except ImportError:
        return None

def _inject_dependencies_safe(step_instance):
    """의존성 안전 주입"""
    try:
        container = _get_central_hub_container()
        if container:
            # 의존성 주입 로직
            pass
    except Exception as e:
        logger.warning(f"의존성 주입 실패: {e}")

def _get_service_from_central_hub(service_key: str):
    """중앙 허브에서 서비스 조회"""
    try:
        container = _get_central_hub_container()
        if container:
            return container.get_service(service_key)
    except Exception as e:
        logger.warning(f"서비스 조회 실패: {e}")
    return None
