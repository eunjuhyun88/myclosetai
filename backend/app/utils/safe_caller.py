
"""
🔧 안전한 함수 호출 유틸리티 - dict object is not callable 방지
"""

import logging
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

class SafeFunctionCaller:
    """안전한 함수 호출 래퍼"""
    
    @staticmethod
    def safe_call(obj: Any, *args, **kwargs) -> Any:
        """
        안전한 함수 호출 - dict object is not callable 방지
        
        Args:
            obj: 호출할 객체 (함수 또는 딕셔너리)
            *args: 위치 인수
            **kwargs: 키워드 인수
            
        Returns:
            호출 결과 또는 안전한 기본값
        """
        try:
            # 1차 확인: 실제 callable인지 확인
            if callable(obj):
                return obj(*args, **kwargs)
            
            # 2차 확인: 딕셔너리인 경우
            elif isinstance(obj, dict):
                logger.warning(f"⚠️ 딕셔너리를 함수로 호출 시도: {type(obj)}")
                
                # 딕셔너리에서 callable 찾기
                for key, value in obj.items():
                    if callable(value):
                        logger.info(f"🔍 딕셔너리에서 함수 발견: {key}")
                        return value(*args, **kwargs)
                
                # 특별한 키들 확인
                special_keys = ['function', 'callable', 'method', 'process', 'execute']
                for key in special_keys:
                    if key in obj and callable(obj[key]):
                        logger.info(f"🔍 특별 키에서 함수 발견: {key}")
                        return obj[key](*args, **kwargs)
                
                # callable이 없으면 딕셔너리 자체 반환
                logger.warning("⚠️ 딕셔너리에서 callable을 찾을 수 없음, 딕셔너리 반환")
                return obj
            
            # 3차 확인: None인 경우
            elif obj is None:
                logger.warning("⚠️ None 객체 호출 시도")
                return None
            
            # 4차 확인: 다른 객체인 경우
            else:
                logger.warning(f"⚠️ callable이 아닌 객체 호출 시도: {type(obj)}")
                return obj
                
        except Exception as e:
            logger.error(f"❌ 안전한 함수 호출 실패: {e}")
            return None
    
    @staticmethod
    def safe_get_method(obj: Any, method_name: str, default_func: Optional[Callable] = None) -> Callable:
        """안전한 메서드 가져오기"""
        try:
            if hasattr(obj, method_name):
                method = getattr(obj, method_name)
                if callable(method):
                    return method
                else:
                    logger.warning(f"⚠️ {method_name}이 callable이 아님: {type(method)}")
                    return default_func or (lambda *a, **k: None)
            else:
                logger.warning(f"⚠️ {method_name} 메서드 없음")
                return default_func or (lambda *a, **k: None)
                
        except Exception as e:
            logger.error(f"❌ 메서드 가져오기 실패: {e}")
            return default_func or (lambda *a, **k: None)
    
    @staticmethod
    def safe_warmup(obj: Any, *args, **kwargs) -> bool:
        """안전한 워밍업 실행"""
        try:
            # warmup 메서드 찾기
            warmup_candidates = ['warmup', 'warm_up', 'initialize', 'init', 'prepare']
            
            for method_name in warmup_candidates:
                if hasattr(obj, method_name):
                    method = getattr(obj, method_name)
                    if callable(method):
                        logger.info(f"🔥 {method_name} 메서드로 워밍업 실행")
                        result = method(*args, **kwargs)
                        return result if result is not None else True
            
            # 메서드가 없으면 객체 자체가 callable인지 확인
            if callable(obj):
                logger.info("🔥 객체 자체를 워밍업 함수로 실행")
                result = obj(*args, **kwargs)
                return result if result is not None else True
            
            logger.warning("⚠️ 워밍업 메서드를 찾을 수 없음")
            return False
            
        except Exception as e:
            logger.error(f"❌ 안전한 워밍업 실패: {e}")
            return False

# 전역 함수들
safe_call = SafeFunctionCaller.safe_call
safe_get_method = SafeFunctionCaller.safe_get_method
safe_warmup = SafeFunctionCaller.safe_warmup

__all__ = ['SafeFunctionCaller', 'safe_call', 'safe_get_method', 'safe_warmup']
