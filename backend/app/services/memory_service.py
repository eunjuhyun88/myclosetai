"""
메모리 관리 서비스
"""

import gc
import psutil
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


def safe_mps_empty_cache():
    """MPS 캐시 안전하게 비우기 (Central Hub 기반)"""
    try:
        logger.info("🔄 MPS 캐시 정리 시작...")
        
        # PyTorch MPS 캐시 정리
        try:
            import torch
            if hasattr(torch.mps, 'empty_cache'):
                torch.mps.empty_cache()
                logger.info("✅ PyTorch MPS 캐시 정리 완료")
            else:
                logger.info("ℹ️ PyTorch MPS 캐시 정리 함수 없음")
        except ImportError:
            logger.info("ℹ️ PyTorch MPS 없음")
        except Exception as e:
            logger.warning(f"⚠️ PyTorch MPS 캐시 정리 실패: {e}")
        
        # Python 가비지 컬렉션
        try:
            collected = gc.collect()
            logger.info(f"✅ Python 가비지 컬렉션 완료: {collected}개 객체 정리")
        except Exception as e:
            logger.warning(f"⚠️ Python 가비지 컬렉션 실패: {e}")
        
        # 메모리 사용량 로깅
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            logger.info(f"📊 메모리 사용량: {memory_info.rss / 1024 / 1024:.1f}MB")
        except Exception as e:
            logger.warning(f"⚠️ 메모리 사용량 확인 실패: {e}")
        
        logger.info("✅ MPS 캐시 정리 완료")
        
    except Exception as e:
        logger.error(f"❌ MPS 캐시 정리 실패: {e}")
        logger.error(f"❌ 상세 오류: {traceback.format_exc()}")


def optimize_central_hub_memory():
    """Central Hub 메모리 최적화 (Central Hub 기반)"""
    try:
        logger.info("🔄 Central Hub 메모리 최적화 시작...")
        
        # Central Hub Container 조회
        container = _get_central_hub_container()
        if not container:
            logger.warning("⚠️ Central Hub Container 없음")
            return
        
        # 메모리 사용량 확인
        process = psutil.Process()
        memory_before = process.memory_info().rss / 1024 / 1024
        logger.info(f"📊 최적화 전 메모리: {memory_before:.1f}MB")
        
        # 각 서비스별 메모리 정리
        services_to_clean = [
            'session_manager', 'step_service_manager', 
            'websocket_manager', 'memory_manager'
        ]
        
        for service_name in services_to_clean:
            try:
                service = container.get(service_name)
                if service and hasattr(service, 'cleanup_memory'):
                    service.cleanup_memory()
                    logger.info(f"✅ {service_name} 메모리 정리 완료")
                elif service and hasattr(service, 'clear_cache'):
                    service.clear_cache()
                    logger.info(f"✅ {service_name} 캐시 정리 완료")
                else:
                    logger.info(f"ℹ️ {service_name} 메모리 정리 함수 없음")
            except Exception as e:
                logger.warning(f"⚠️ {service_name} 메모리 정리 실패: {e}")
        
        # MPS 캐시 정리
        safe_mps_empty_cache()
        
        # 가비지 컬렉션 강제 실행
        collected = gc.collect()
        logger.info(f"✅ 강제 가비지 컬렉션: {collected}개 객체 정리")
        
        # 메모리 사용량 재확인
        memory_after = process.memory_info().rss / 1024 / 1024
        memory_saved = memory_before - memory_after
        logger.info(f"📊 최적화 후 메모리: {memory_after:.1f}MB")
        logger.info(f"📊 절약된 메모리: {memory_saved:.1f}MB")
        
        logger.info("✅ Central Hub 메모리 최적화 완료")
        
    except Exception as e:
        logger.error(f"❌ Central Hub 메모리 최적화 실패: {e}")
        logger.error(f"❌ 상세 오류: {traceback.format_exc()}")


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