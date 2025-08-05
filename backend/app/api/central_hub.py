"""
Central Hub 유틸리티
"""

import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


def _get_central_hub_container():
    """Central Hub DI Container 안전한 동적 해결"""
    try:
        import importlib
        module = importlib.import_module('app.core.di_container')
        return module.get_global_container()
    except ImportError:
        return None
    except Exception:
        return None


def _get_step_service_manager():
    """Central Hub를 통한 StepServiceManager 조회"""
    try:
        container = _get_central_hub_container()
        if container:
            return container.get('step_service_manager')
        
        # 폴백: 직접 생성
        from app.services.step_service import StepServiceManager
        return StepServiceManager()
    except Exception:
        return None


def _get_websocket_manager():
    """Central Hub를 통한 WebSocketManager 조회"""
    try:
        container = _get_central_hub_container()
        if container:
            return container.get('websocket_manager')
        return None
    except Exception:
        return None


def _get_memory_manager():
    """Central Hub를 통한 MemoryManager 조회"""
    try:
        container = _get_central_hub_container()
        if container:
            return container.get('memory_manager')
        
        # 폴백: 기본 메모리 매니저 생성
        class BasicMemoryManager:
            def __init__(self):
                self.memory_usage = {}
                self.cache = {}
            
            def optimize(self):
                """기본 메모리 최적화"""
                import gc
                gc.collect()
                self.cache.clear()
                logger.info("✅ 기본 메모리 최적화 완료")
            
            def get_memory_usage(self):
                """메모리 사용량 조회"""
                try:
                    import psutil
                    process = psutil.Process()
                    return {
                        'rss': process.memory_info().rss,
                        'vms': process.memory_info().vms,
                        'percent': process.memory_percent()
                    }
                except Exception:
                    return {'error': '메모리 정보 조회 실패'}
            
            def clear_cache(self):
                """캐시 정리"""
                self.cache.clear()
                logger.info("✅ 기본 캐시 정리 완료")
        
        return BasicMemoryManager()
        
    except Exception as e:
        logger.error(f"❌ MemoryManager 조회 실패: {e}")
        return None 