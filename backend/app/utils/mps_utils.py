"""
MPS 유틸리티 모듈 - PyTorch 2.1.2 호환
M3 Max 환경에서 안전한 MPS 메모리 관리
"""

import logging
import gc
from typing import Dict, Any, Optional
import torch

logger = logging.getLogger(__name__)

class MPSMemoryManager:
    """M3 Max MPS 메모리 관리자"""
    
    def __init__(self):
        self.is_available = torch.backends.mps.is_available()
        self.supports_empty_cache = hasattr(torch.mps, 'empty_cache')
        self.supports_synchronize = hasattr(torch.mps, 'synchronize')
        
        logger.info(f"🍎 MPS 메모리 관리자 초기화")
        logger.info(f"   - MPS 사용 가능: {self.is_available}")
        logger.info(f"   - empty_cache 지원: {self.supports_empty_cache}")
        logger.info(f"   - synchronize 지원: {self.supports_synchronize}")
    
    def safe_empty_cache(self) -> Dict[str, Any]:
        """안전한 MPS 메모리 정리"""
        result = {
            "success": False,
            "method": "none",
            "message": "MPS 사용 불가"
        }
        
        if not self.is_available:
            return result
        
        try:
            if self.supports_empty_cache:
                torch.mps.empty_cache()
                result.update({
                    "success": True,
                    "method": "mps_empty_cache",
                    "message": "torch.mps.empty_cache() 실행 완료"
                })
                logger.info("✅ torch.mps.empty_cache() 실행 완료")
                
            elif self.supports_synchronize:
                torch.mps.synchronize()
                result.update({
                    "success": True,
                    "method": "mps_synchronize",
                    "message": "torch.mps.synchronize() 실행 완료"
                })
                logger.info("✅ torch.mps.synchronize() 실행 완료")
                
            else:
                gc.collect()
                result.update({
                    "success": True,
                    "method": "gc_collect",
                    "message": "가비지 컬렉션으로 대체"
                })
                logger.info("✅ 가비지 컬렉션으로 메모리 정리")
            
            return result
            
        except Exception as e:
            result.update({
                "success": False,
                "method": "error",
                "message": f"MPS 메모리 정리 실패: {e}"
            })
            logger.error(f"❌ MPS 메모리 정리 실패: {e}")
            return result
    
    def get_compatibility_info(self) -> Dict[str, Any]:
        """MPS 호환성 정보 조회"""
        return {
            "pytorch_version": torch.__version__,
            "mps_available": self.is_available,
            "mps_built": torch.backends.mps.is_built(),
            "empty_cache_support": self.supports_empty_cache,
            "synchronize_support": self.supports_synchronize,
            "recommended_method": (
                "mps_empty_cache" if self.supports_empty_cache 
                else "mps_synchronize" if self.supports_synchronize 
                else "gc_collect"
            )
        }

# 전역 인스턴스
_mps_manager = None

def get_mps_manager() -> MPSMemoryManager:
    """전역 MPS 관리자 반환"""
    global _mps_manager
    if _mps_manager is None:
        _mps_manager = MPSMemoryManager()
    return _mps_manager

def safe_mps_empty_cache() -> Dict[str, Any]:
    """안전한 MPS 메모리 정리 (함수형 인터페이스)"""
    return get_mps_manager().safe_empty_cache()

def get_mps_compatibility_info() -> Dict[str, Any]:
    """MPS 호환성 정보 조회 (함수형 인터페이스)"""
    return get_mps_manager().get_compatibility_info()
