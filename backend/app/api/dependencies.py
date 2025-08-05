"""
의존성 주입 함수들
"""

import logging
import traceback
from fastapi import HTTPException, Depends
from typing import Optional

logger = logging.getLogger(__name__)


def get_session_manager_dependency():
    """SessionManager Dependency 함수 (Central Hub 기반)"""
    try:
        logger.info("🔄 SessionManager 의존성 주입 시작...")
        
        from app.core.session_manager import get_session_manager
        session_manager = get_session_manager()
        
        if not session_manager:
            logger.error("❌ SessionManager 생성 실패")
            raise HTTPException(
                status_code=503,
                detail="SessionManager not available from Central Hub"
            )
        
        logger.info("✅ SessionManager 의존성 주입 성공")
        logger.info(f"🔍 SessionManager 타입: {type(session_manager).__name__}")
        logger.info(f"🔍 SessionManager 세션 수: {len(session_manager.sessions) if hasattr(session_manager, 'sessions') else 'N/A'}")
        if hasattr(session_manager, 'sessions'):
            logger.info(f"🔍 SessionManager 세션 키들: {list(session_manager.sessions.keys())}")
        if hasattr(session_manager, 'db_path'):
            logger.info(f"🔍 SessionManager DB 경로: {session_manager.db_path}")
        return session_manager
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ SessionManager 의존성 주입 실패: {e}")
        logger.error(f"❌ SessionManager 의존성 주입 오류 상세: {traceback.format_exc()}")
        raise HTTPException(
            status_code=503,
            detail=f"세션 관리자 초기화 실패: {str(e)}"
        )


def get_step_service_manager_dependency():
    """StepServiceManager Dependency 함수 (동기, Central Hub 기반)"""
    try:
        from app.api.central_hub import _get_step_service_manager
        step_service_manager = _get_step_service_manager()
        if not step_service_manager:
            raise HTTPException(
                status_code=503,
                detail="StepServiceManager not available from Central Hub"
            )
        return step_service_manager
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ StepServiceManager 조회 실패: {e}")
        raise HTTPException(
            status_code=503,
            detail=f"Central Hub AI 서비스 초기화 실패: {str(e)}"
        ) 