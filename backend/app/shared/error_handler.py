# backend/app/shared/error_handler.py
"""
🔥 MyCloset AI Error Handler
================================================================================

에러 처리를 위한 공통 모듈입니다.

- handle_api_error: API 에러 처리
- create_error_response: 에러 응답 생성

Author: MyCloset AI Team
Date: 2025-08-01
Version: 1.0
"""

import logging
import traceback
from typing import Dict, Any, Optional
from datetime import datetime
from fastapi import HTTPException

logger = logging.getLogger(__name__)


def handle_api_error(
    error: Exception,
    step_name: str = "unknown",
    step_id: int = 0,
    session_id: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """API 에러 처리 및 로깅"""
    
    # 에러 정보 수집
    error_type = type(error).__name__
    error_message = str(error)
    error_traceback = traceback.format_exc()
    
    # 로깅
    logger.error(f"❌ API 에러 발생 - Step {step_id} ({step_name}): {error_type}: {error_message}")
    logger.error(f"❌ 세션 ID: {session_id}")
    logger.error(f"❌ 컨텍스트: {context}")
    logger.error(f"❌ 스택 트레이스: {error_traceback}")
    
    # 에러 응답 생성
    error_response = {
        "success": False,
        "error": {
            "type": error_type,
            "message": error_message,
            "step_name": step_name,
            "step_id": step_id,
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(),
            "context": context or {}
        },
        "step_name": step_name,
        "step_id": step_id,
        "session_id": session_id,
        "processing_time": 0.0,
        "confidence": 0.0,
        "device": "mps",
        "timestamp": datetime.now().isoformat(),
        
        # 프론트엔드 호환성
        "progress_percentage": (step_id / 8) * 100 if step_id > 0 else 0,
        "current_step": step_id,
        "total_steps": 8,
        "remaining_steps": max(0, 8 - step_id),
        
        # Central Hub 정보
        "central_hub_di_container_v70": True,
        "circular_reference_free": True,
        "single_source_of_truth": True,
        "dependency_inversion": True,
        "conda_environment": "myclosetlast",
        "mycloset_optimized": True,
        "m3_max_optimized": True,
        "memory_gb": 128,
        "central_hub_used": True,
        "di_container_integration": True
    }
    
    return error_response


def create_error_response(
    error_message: str,
    error_type: str = "ValidationError",
    status_code: int = 400,
    step_name: str = "unknown",
    step_id: int = 0,
    session_id: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """에러 응답 생성"""
    
    logger.warning(f"⚠️ 에러 응답 생성 - {error_type}: {error_message}")
    
    return {
        "success": False,
        "message": error_message,
        "error": {
            "type": error_type,
            "message": error_message,
            "step_name": step_name,
            "step_id": step_id,
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(),
            "context": context or {}
        },
        "step_name": step_name,
        "step_id": step_id,
        "session_id": session_id,
        "processing_time": 0.0,
        "confidence": 0.0,
        "device": "mps",
        "timestamp": datetime.now().isoformat(),
        
        # 프론트엔드 호환성
        "progress_percentage": (step_id / 8) * 100 if step_id > 0 else 0,
        "current_step": step_id,
        "total_steps": 8,
        "remaining_steps": max(0, 8 - step_id),
        
        # Central Hub 정보
        "central_hub_di_container_v70": True,
        "circular_reference_free": True,
        "single_source_of_truth": True,
        "dependency_inversion": True,
        "conda_environment": "myclosetlast",
        "mycloset_optimized": True,
        "m3_max_optimized": True,
        "memory_gb": 128,
        "central_hub_used": True,
        "di_container_integration": True
    }


def handle_validation_error(
    field: str,
    value: Any,
    expected_type: str,
    step_name: str = "unknown",
    step_id: int = 0,
    session_id: Optional[str] = None
) -> Dict[str, Any]:
    """검증 에러 처리"""
    
    error_message = f"필드 '{field}'의 값 '{value}'이(가) {expected_type} 타입이 아닙니다."
    
    return create_error_response(
        error_message=error_message,
        error_type="ValidationError",
        status_code=400,
        step_name=step_name,
        step_id=step_id,
        session_id=session_id,
        context={"field": field, "value": value, "expected_type": expected_type}
    )


def handle_session_error(
    session_id: str,
    error_message: str,
    step_name: str = "unknown",
    step_id: int = 0
) -> Dict[str, Any]:
    """세션 에러 처리"""
    
    return create_error_response(
        error_message=error_message,
        error_type="SessionError",
        status_code=404,
        step_name=step_name,
        step_id=step_id,
        session_id=session_id,
        context={"session_id": session_id}
    )


def handle_processing_error(
    step_name: str,
    step_id: int,
    error_message: str,
    session_id: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """처리 에러 처리"""
    
    return create_error_response(
        error_message=error_message,
        error_type="ProcessingError",
        status_code=500,
        step_name=step_name,
        step_id=step_id,
        session_id=session_id,
        context=context
    ) 