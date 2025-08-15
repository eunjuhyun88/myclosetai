#!/usr/bin/env python3
"""
🔥 MyCloset AI - DI Container 표준화된 접근 유틸리티
====================================================

모든 Step에서 일관된 방식으로 DI Container에 접근할 수 있도록 하는 표준화된 인터페이스
폴백 시스템 없이 명확하고 예측 가능한 동작 보장

Author: MyCloset AI Team
Date: 2025-08-14
Version: 1.0 (표준화된 접근 패턴)
"""

import logging
from typing import Any, Optional, Dict, Type, TypeVar
from functools import wraps

logger = logging.getLogger(__name__)

# ==============================================
# 🔥 표준화된 DI Container 접근 함수들
# ==============================================

def get_di_container():
    """표준화된 DI Container 인스턴스 반환"""
    try:
        from app.core.di_container import get_global_container
        return get_global_container()
    except ImportError:
        raise ImportError("DI Container를 import할 수 없습니다. app.core.di_container를 확인해주세요.")

def get_service(service_key: str) -> Optional[Any]:
    """표준화된 서비스 조회"""
    try:
        container = get_di_container()
        return container.get_service(service_key)
    except Exception as e:
        logger.error(f"서비스 조회 실패 ({service_key}): {e}")
        return None

def register_service(service_key: str, service_instance: Any, singleton: bool = True) -> bool:
    """표준화된 서비스 등록"""
    try:
        container = get_di_container()
        container.register_service(service_key, service_instance, singleton)
        logger.info(f"✅ 서비스 등록 성공: {service_key}")
        return True
    except Exception as e:
        logger.error(f"❌ 서비스 등록 실패 ({service_key}): {e}")
        return False

def has_service(service_key: str) -> bool:
    """표준화된 서비스 존재 여부 확인"""
    try:
        container = get_di_container()
        return container.has_service(service_key)
    except Exception as e:
        logger.error(f"서비스 존재 여부 확인 실패 ({service_key}): {e}")
        return False

def list_services() -> list:
    """표준화된 서비스 목록 반환"""
    try:
        container = get_di_container()
        return container.list_services()
    except Exception as e:
        logger.error(f"서비스 목록 조회 실패: {e}")
        return []

# ==============================================
# 🔥 타입 안전한 서비스 접근
# ==============================================

T = TypeVar('T')

def get_service_typed(service_key: str, service_type: Type[T]) -> Optional[T]:
    """타입 안전한 서비스 조회"""
    service = get_service(service_key)
    if service is not None and isinstance(service, service_type):
        return service
    elif service is not None:
        logger.warning(f"서비스 타입 불일치 ({service_key}): 예상 {service_type}, 실제 {type(service)}")
    return None

# ==============================================
# 🔥 데코레이터를 통한 서비스 주입
# ==============================================

def inject_service(service_key: str, attr_name: str = None):
    """서비스 주입 데코레이터"""
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            if not hasattr(self, attr_name or service_key):
                service = get_service(service_key)
                if service is not None:
                    setattr(self, attr_name or service_key, service)
                else:
                    logger.warning(f"서비스 주입 실패: {service_key}")
            return func(self, *args, **kwargs)
        return wrapper
    return decorator

def require_service(service_key: str):
    """필수 서비스 검증 데코레이터"""
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            if not hasattr(self, service_key):
                service = get_service(service_key)
                if service is None:
                    raise RuntimeError(f"필수 서비스를 찾을 수 없습니다: {service_key}")
                setattr(self, service_key, service)
            return func(self, *args, **kwargs)
        return wrapper
    return decorator

# ==============================================
# 🔥 서비스 상태 모니터링
# ==============================================

def get_service_status() -> Dict[str, Any]:
    """모든 서비스의 상태 정보 반환"""
    try:
        container = get_di_container()
        return {
            'total_services': len(container.list_services()),
            'available_services': container.list_services(),
            'container_status': 'active'
        }
    except Exception as e:
        logger.error(f"서비스 상태 조회 실패: {e}")
        return {
            'total_services': 0,
            'available_services': [],
            'container_status': 'error',
            'error': str(e)
        }

def validate_service_dependencies(required_services: list) -> Dict[str, bool]:
    """필수 서비스들의 가용성 검증"""
    results = {}
    for service_key in required_services:
        results[service_key] = has_service(service_key)
    return results

# ==============================================
# 🔥 에러 처리 및 로깅
# ==============================================

class DIContainerError(Exception):
    """DI Container 관련 에러"""
    pass

def safe_service_access(service_key: str, default_value: Any = None):
    """안전한 서비스 접근 (에러 발생 시 기본값 반환)"""
    try:
        return get_service(service_key)
    except Exception as e:
        logger.warning(f"서비스 접근 실패 ({service_key}): {e}, 기본값 사용")
        return default_value

# ==============================================
# 🔥 초기화 및 검증
# ==============================================

def initialize_di_system():
    """DI 시스템 초기화 및 검증"""
    try:
        container = get_di_container()
        if container is None:
            raise DIContainerError("DI Container를 생성할 수 없습니다")
        
        logger.info("✅ DI Container 시스템 초기화 완료")
        return True
    except Exception as e:
        logger.error(f"❌ DI Container 시스템 초기화 실패: {e}")
        raise DIContainerError(f"DI 시스템 초기화 실패: {e}")

def verify_di_system():
    """DI 시스템 상태 검증"""
    try:
        container = get_di_container()
        services = container.list_services()
        logger.info(f"✅ DI 시스템 검증 완료 - {len(services)}개 서비스 등록됨")
        return True
    except Exception as e:
        logger.error(f"❌ DI 시스템 검증 실패: {e}")
        return False

# ==============================================
# 🔥 공개 API
# ==============================================

__all__ = [
    'get_di_container',
    'get_service',
    'register_service',
    'has_service',
    'list_services',
    'get_service_typed',
    'inject_service',
    'require_service',
    'get_service_status',
    'validate_service_dependencies',
    'DIContainerError',
    'safe_service_access',
    'initialize_di_system',
    'verify_di_system'
]
