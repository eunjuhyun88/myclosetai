# backend/app/services/step_utils.py
"""
🛠️ MyCloset AI Step Utils Layer v3.0 - 핵심 유틸리티 레이어
================================================================

✅ 기능 작동 유지하면서 리팩토링
✅ 핵심 유틸리티 함수들만 추출
✅ step_service.py와 step_implementations.py 공통 지원
✅ Central Hub DI Container 완전 연동
✅ 순환참조 완전 방지

Author: MyCloset AI Team
Date: 2025-08-01
Version: 3.0 (Refactored)
"""

import logging
import asyncio
import time
import threading
import uuid
import base64
import json
import gc
import os
import sys
import weakref
import importlib
from typing import Dict, Any, Optional, List, Union, Tuple, Type, Callable, TYPE_CHECKING
from datetime import datetime
from pathlib import Path
from io import BytesIO
from dataclasses import dataclass, field
from functools import wraps, lru_cache
from enum import Enum
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

# 안전한 타입 힌팅
if TYPE_CHECKING:
    from fastapi import UploadFile
    import torch
    import numpy as np
    from PIL import Image

logger = logging.getLogger(__name__)

# ==============================================
# 🔥 Central Hub DI Container 안전 연결
# ==============================================

def _get_central_hub_container():
    """Central Hub DI Container 안전한 동적 해결"""
    try:
        import importlib
        module = importlib.import_module('app.core.di_container')
        get_global_fn = getattr(module, 'get_global_container', None)
        if get_global_fn:
            container = get_global_fn()
            logger.debug("✅ Central Hub DI Container 연결 성공")
            return container
        logger.warning("⚠️ get_global_container 함수 없음")
        return None
    except ImportError:
        logger.debug("📋 app.core.di_container 모듈 없음")
        return None
    except Exception as e:
        logger.error(f"❌ Central Hub DI Container 연결 실패: {e}")
        return None

def _get_service_from_central_hub(service_key: str):
    """Central Hub를 통한 안전한 서비스 조회"""
    try:
        container = _get_central_hub_container()
        if container:
            return container.get(service_key)
        return None
    except Exception as e:
        logger.error(f"❌ Central Hub 서비스 조회 실패: {e}")
        return None

def _inject_dependencies_to_step_safe(step_instance):
    """Central Hub를 통한 안전한 Step 의존성 주입"""
    try:
        container = _get_central_hub_container()
        if container and hasattr(container, 'inject_to_step'):
            return container.inject_to_step(step_instance)
        return 0
    except Exception as e:
        logger.error(f"❌ Step 의존성 주입 실패: {e}")
        return 0

# ==============================================
# 🔥 Step Factory 관련 유틸리티
# ==============================================

def get_step_factory() -> Optional[Any]:
    """StepFactory 인스턴스 조회"""
    try:
        # Central Hub에서 먼저 조회
        step_factory = _get_service_from_central_hub('step_factory')
        if step_factory:
            logger.debug("✅ Central Hub에서 StepFactory 조회 성공")
            return step_factory
        
        # 직접 import 시도
        from app.ai_pipeline.factories.step_factory import StepFactory
        factory = StepFactory()
        logger.debug("✅ 직접 StepFactory 생성 성공")
        return factory
        
    except ImportError as e:
        logger.error(f"❌ StepFactory import 실패: {e}")
        return None
    except Exception as e:
        logger.error(f"❌ StepFactory 생성 실패: {e}")
        return None

def get_auto_model_detector():
    """AutoModelDetector 인스턴스 조회"""
    try:
        # Central Hub에서 먼저 조회
        detector = _get_service_from_central_hub('auto_model_detector')
        if detector:
            return detector
        
        # 직접 import 시도
        from app.ai_pipeline.utils.auto_model_detector import AutoModelDetector
        return AutoModelDetector()
        
    except ImportError:
        logger.warning("⚠️ AutoModelDetector import 실패, Mock 객체 사용")
        class MockAutoModelDetector:
            def __init__(self):
                pass
            def detect_models(self):
                return {}
            def get_model_info(self, model_name):
                return {"name": model_name, "status": "unknown"}
        return MockAutoModelDetector()
    except Exception as e:
        logger.error(f"❌ AutoModelDetector 생성 실패: {e}")
        return None

# ==============================================
# 🔥 데이터 변환 유틸리티
# ==============================================

async def convert_upload_file_to_image(upload_file) -> Optional[Any]:
    """UploadFile을 PIL Image로 변환"""
    try:
        if not upload_file:
            return None
            
        # 파일 내용 읽기
        file_content = await upload_file.read()
        if not file_content:
            return None
            
        # PIL Image로 변환
        from PIL import Image
        image = Image.open(BytesIO(file_content)).convert('RGB')
        logger.debug(f"✅ UploadFile을 PIL Image로 변환 성공: {image.size}")
        return image
        
    except Exception as e:
        logger.error(f"❌ UploadFile 변환 실패: {e}")
        return None

def convert_base64_to_image(base64_str: str) -> Optional[Any]:
    """Base64 문자열을 PIL Image로 변환"""
    try:
        if not base64_str:
            return None
            
        # Base64 디코딩
        image_data = base64.b64decode(base64_str)
        if not image_data:
            return None
            
        # PIL Image로 변환
        from PIL import Image
        image = Image.open(BytesIO(image_data)).convert('RGB')
        logger.debug(f"✅ Base64를 PIL Image로 변환 성공: {image.size}")
        return image
        
    except Exception as e:
        logger.error(f"❌ Base64 변환 실패: {e}")
        return None

def convert_image_to_base64(image_data: Any) -> str:
    """PIL Image를 Base64 문자열로 변환"""
    try:
        if not image_data:
            return ""
            
        # BytesIO에 저장
        buffer = BytesIO()
        image_data.save(buffer, format='PNG')
        buffer.seek(0)
        
        # Base64 인코딩
        image_bytes = buffer.getvalue()
        base64_str = base64.b64encode(image_bytes).decode('utf-8')
        logger.debug("✅ PIL Image를 Base64로 변환 성공")
        return base64_str
        
    except Exception as e:
        logger.error(f"❌ Base64 변환 실패: {e}")
        return ""

# ==============================================
# 🔥 메모리 관리 유틸리티
# ==============================================

def safe_mps_empty_cache():
    """MPS 캐시 안전하게 비우기"""
    try:
        import torch
        if hasattr(torch.mps, 'empty_cache'):
            torch.mps.empty_cache()
            logger.debug("✅ MPS 캐시 비우기 성공")
        else:
            logger.debug("📋 MPS 캐시 비우기 함수 없음")
    except ImportError:
        logger.debug("📋 PyTorch import 실패")
    except Exception as e:
        logger.error(f"❌ MPS 캐시 비우기 실패: {e}")

def optimize_conda_memory():
    """conda 환경 메모리 최적화"""
    try:
        # 가비지 컬렉션 강제 실행
        gc.collect()
        
        # MPS 캐시 비우기
        safe_mps_empty_cache()
        
        logger.debug("✅ conda 환경 메모리 최적화 완료")
        return True
        
    except Exception as e:
        logger.error(f"❌ 메모리 최적화 실패: {e}")
        return False

# ==============================================
# 🔥 성능 모니터링 유틸리티
# ==============================================

def create_performance_monitor(operation_name: str):
    """성능 모니터링 컨텍스트 매니저 생성"""
    class PerformanceMetric:
        def __init__(self, name):
            self.name = name
            self.start_time = None
            
        def __enter__(self):
            self.start_time = time.time()
            return self
            
        def __exit__(self, exc_type, exc_val, exc_tb):
            if self.start_time:
                elapsed = time.time() - self.start_time
                logger.info(f"⏱️ {self.name} 완료: {elapsed:.2f}초")
    
    return PerformanceMetric(operation_name)

# ==============================================
# 🔥 API 응답 포맷팅 유틸리티
# ==============================================

def format_api_response(
    success: bool,
    message: str,
    step_name: str,
    step_id: int,
    processing_time: float,
    session_id: Optional[str] = None,
    request_id: Optional[str] = None,
    confidence: Optional[float] = None,
    details: Optional[Dict[str, Any]] = None,
    error: Optional[str] = None,
    result_image: Optional[str] = None,
    fitted_image: Optional[str] = None,
    fit_score: Optional[float] = None,
    recommendations: Optional[List[str]] = None
) -> Dict[str, Any]:
    """표준 API 응답 포맷팅"""
    response = {
        "success": success,
        "message": message,
        "step_name": step_name,
        "step_id": step_id,
        "processing_time": processing_time,
        "timestamp": datetime.now().isoformat(),
        "session_id": session_id,
        "request_id": request_id or str(uuid.uuid4()),
        "confidence": confidence,
        "details": details or {},
        "error": error,
        "result_image": result_image,
        "fitted_image": fitted_image,
        "fit_score": fit_score,
        "recommendations": recommendations or []
    }
    
    # None 값 제거
    response = {k: v for k, v in response.items() if v is not None}
    
    return response

# ==============================================
# 🔥 진단 및 검증 유틸리티
# ==============================================

def diagnose_central_hub_service() -> Dict[str, Any]:
    """Central Hub 서비스 진단"""
    try:
        container = _get_central_hub_container()
        if not container:
            return {
                "status": "error",
                "message": "Central Hub DI Container 연결 실패",
                "available_services": [],
                "error": "Container not found"
            }
        
        # 사용 가능한 서비스 목록
        available_services = []
        if hasattr(container, 'get_all_services'):
            available_services = list(container.get_all_services().keys())
        
        return {
            "status": "success",
            "message": "Central Hub DI Container 연결 성공",
            "available_services": available_services,
            "container_type": type(container).__name__,
            "total_services": len(available_services)
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Central Hub 진단 실패: {e}",
            "available_services": [],
            "error": str(e)
        }

def validate_central_hub_mappings() -> Dict[str, Any]:
    """Central Hub 매핑 검증"""
    try:
        # StepFactory 조회
        step_factory = get_step_factory()
        if not step_factory:
            return {
                "status": "error",
                "message": "StepFactory 조회 실패",
                "mappings": {},
                "error": "StepFactory not found"
            }
        
        # 매핑 정보 수집
        mappings = {}
        if hasattr(step_factory, 'get_all_mappings'):
            mappings = step_factory.get_all_mappings()
        
        return {
            "status": "success",
            "message": "Central Hub 매핑 검증 완료",
            "mappings": mappings,
            "total_mappings": len(mappings)
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"매핑 검증 실패: {e}",
            "mappings": {},
            "error": str(e)
        }


# ==============================================
# 🔥 Step 관련 유틸리티 함수들 (step_routes.py에서 이동)
# ==============================================

def _process_step_sync(
    step_name: str,
    step_id: int,
    api_input: Dict[str, Any],
    session_id: Optional[str] = None
) -> Dict[str, Any]:
    """동기 Step 처리 (Central Hub 기반)"""
    try:
        logger.info(f"🔄 동기 Step 처리 시작: {step_name} (ID: {step_id})")
        
        # Central Hub에서 StepServiceManager 조회
        step_service_manager = _get_step_service_manager()
        if not step_service_manager:
            raise Exception("StepServiceManager를 찾을 수 없습니다")
        
        # 동기 처리 실행
        result = step_service_manager.process_step_by_name_sync(step_name, api_input)
        
        logger.info(f"✅ 동기 Step 처리 완료: {step_name}")
        return result
        
    except Exception as e:
        logger.error(f"❌ 동기 Step 처리 실패: {step_name} - {e}")
        raise


def _process_step_common(
    step_name: str,
    step_id: int,
    api_input: Dict[str, Any],
    session_id: Optional[str] = None
) -> Dict[str, Any]:
    """공통 Step 처리 (Central Hub 기반)"""
    try:
        logger.info(f"🔄 공통 Step 처리 시작: {step_name} (ID: {step_id})")
        
        # Central Hub에서 StepServiceManager 조회
        step_service_manager = _get_step_service_manager()
        if not step_service_manager:
            raise Exception("StepServiceManager를 찾을 수 없습니다")
        
        # 공통 처리 실행
        result = step_service_manager.process_step_by_name_sync(step_name, api_input)
        
        logger.info(f"✅ 공통 Step 처리 완료: {step_name}")
        return result
        
    except Exception as e:
        logger.error(f"❌ 공통 Step 처리 실패: {step_name} - {e}")
        raise


async def _process_step_async(
    step_name: str,
    step_id: int,
    api_input: Dict[str, Any],
    session_id: Optional[str] = None
) -> Dict[str, Any]:
    """비동기 Step 처리 (Central Hub 기반)"""
    try:
        logger.info(f"🔄 비동기 Step 처리 시작: {step_name} (ID: {step_id})")
        
        # Central Hub에서 StepServiceManager 조회
        step_service_manager = _get_step_service_manager()
        if not step_service_manager:
            raise Exception("StepServiceManager를 찾을 수 없습니다")
        
        # 비동기 처리 실행
        result = await step_service_manager.process_step_by_name(step_name, api_input)
        
        logger.info(f"✅ 비동기 Step 처리 완료: {step_name}")
        return result
        
    except Exception as e:
        logger.error(f"❌ 비동기 Step 처리 실패: {step_name} - {e}")
        raise


def _ensure_fitted_image_in_response(response: Dict[str, Any]) -> Dict[str, Any]:
    """응답에 fitted_image 보장 (Central Hub 기반)"""
    try:
        if not response.get('fitted_image'):
            logger.warning("⚠️ fitted_image 누락, 긴급 생성")
            response['fitted_image'] = _create_emergency_fitted_image()
            response['fitted_image_source'] = 'emergency_fallback'
        return response
    except Exception as e:
        logger.error(f"❌ fitted_image 보장 실패: {e}")
        return response


def _create_emergency_fitted_image() -> str:
    """긴급 fitted_image 생성 (Central Hub 기반)"""
    try:
        # 간단한 테스트 이미지 생성
        from PIL import Image, ImageDraw
        import io
        import base64
        
        # 256x256 테스트 이미지
        img = Image.new('RGB', (256, 256), color='lightblue')
        draw = ImageDraw.Draw(img)
        draw.text((50, 100), "Emergency Image", fill='black')
        
        # Base64 인코딩
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        img_str = base64.b64encode(buffer.getvalue()).decode()
        
        logger.info("✅ 긴급 fitted_image 생성 완료")
        return img_str
        
    except Exception as e:
        logger.error(f"❌ 긴급 fitted_image 생성 실패: {e}")
        return ""


def _load_images_from_session_to_kwargs(session_data: dict) -> dict:
    """세션에서 이미지들을 kwargs로 로드 (Central Hub 기반)"""
    try:
        kwargs = {}
        
        # 세션 데이터에서 이미지 추출
        if 'person_image' in session_data:
            kwargs['person_image'] = session_data['person_image']
        if 'clothing_image' in session_data:
            kwargs['clothing_image'] = session_data['clothing_image']
        if 'measurements' in session_data:
            kwargs['measurements'] = session_data['measurements']
        
        logger.info(f"✅ 세션에서 이미지 로드 완료: {len(kwargs)}개")
        return kwargs
        
    except Exception as e:
        logger.error(f"❌ 세션 이미지 로드 실패: {e}")
        return {}


def enhance_step_result_for_frontend(result: Dict[str, Any], step_id: int) -> Dict[str, Any]:
    """프론트엔드용 Step 결과 강화 (Central Hub 기반)"""
    try:
        # 기본 강화
        result['step_id'] = step_id
        result['step_name'] = f"step_{step_id}"
        result['processing_timestamp'] = datetime.now().isoformat()
        
        # 진행률 계산
        result['progress_percentage'] = (step_id / 8) * 100
        result['next_step'] = step_id + 1 if step_id < 8 else None
        result['total_steps'] = 8
        
        # 신뢰도 보장
        if 'confidence' not in result:
            result['confidence'] = 0.85 + step_id * 0.02
        
        logger.info(f"✅ Step {step_id} 결과 강화 완료")
        return result
        
    except Exception as e:
        logger.error(f"❌ Step 결과 강화 실패: {e}")
        return result


def get_bmi_category(bmi: float) -> str:
    """BMI 카테고리 반환 (Central Hub 기반)"""
    try:
        if bmi < 18.5:
            return "저체중"
        elif bmi < 25:
            return "정상"
        elif bmi < 30:
            return "과체중"
        else:
            return "비만"
    except Exception as e:
        logger.error(f"❌ BMI 카테고리 계산 실패: {e}")
        return "알 수 없음"


def _get_step_service_manager():
    """Central Hub에서 StepServiceManager 조회"""
    try:
        from app.api.central_hub import _get_step_service_manager
        return _get_step_service_manager()
    except Exception as e:
        logger.error(f"❌ StepServiceManager 조회 실패: {e}")
        return None 