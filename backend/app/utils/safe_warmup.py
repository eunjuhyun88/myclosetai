# backend/app/utils/safe_warmup.py
"""
🔧 안전한 워밍업 래퍼 - dict object is not callable 완전 해결
"""

import logging
from typing import Any, Optional, Dict, Union
import asyncio

logger = logging.getLogger(__name__)

class SafeWarmupManager:
    """안전한 워밍업 매니저"""
    
    def __init__(self):
        self.warmup_status: Dict[str, bool] = {}
        self.warmup_errors: Dict[str, str] = {}
    
    def safe_warmup(self, obj: Any, name: str = "unknown", *args, **kwargs) -> bool:
        """
        안전한 워밍업 실행
        
        Args:
            obj: 워밍업할 객체
            name: 객체 이름 (로깅용)
            *args: 추가 인수
            **kwargs: 추가 키워드 인수
            
        Returns:
            bool: 워밍업 성공 여부
        """
        try:
            logger.info(f"🔥 {name} 워밍업 시작...")
            
            # 1. None 체크
            if obj is None:
                logger.warning(f"⚠️ {name}: 워밍업 객체가 None")
                self.warmup_status[name] = False
                self.warmup_errors[name] = "객체가 None"
                return False
            
            # 2. 딕셔너리인 경우 처리
            if isinstance(obj, dict):
                logger.info(f"🔍 {name}: 딕셔너리 객체 감지, 워밍업 함수 검색...")
                
                # 일반적인 워밍업 키들
                warmup_keys = ['warmup', 'warm_up', 'initialize', 'init', 'setup', 'prepare']
                
                for key in warmup_keys:
                    if key in obj:
                        warmup_func = obj[key]
                        if callable(warmup_func):
                            logger.info(f"🎯 {name}: {key} 함수 발견, 실행 중...")
                            result = warmup_func(*args, **kwargs)
                            self.warmup_status[name] = True
                            logger.info(f"✅ {name}: 워밍업 성공 ({key})")
                            return True
                        else:
                            logger.warning(f"⚠️ {name}: {key}가 callable이 아님")
                
                # 워밍업 함수가 없는 경우 스킵
                logger.info(f"ℹ️ {name}: 딕셔너리에 워밍업 함수 없음, 스킵")
                self.warmup_status[name] = True  # 스킵도 성공으로 처리
                return True
            
            # 3. 일반 객체인 경우
            elif hasattr(obj, 'warmup') and callable(obj.warmup):
                logger.info(f"🎯 {name}: warmup 메서드 실행 중...")
                result = obj.warmup(*args, **kwargs)
                self.warmup_status[name] = True
                logger.info(f"✅ {name}: 워밍업 성공")
                return True
            
            # 4. 다른 워밍업 메서드들 확인
            else:
                warmup_methods = ['warm_up', 'initialize', 'init', 'setup', 'prepare']
                
                for method_name in warmup_methods:
                    if hasattr(obj, method_name):
                        method = getattr(obj, method_name)
                        if callable(method):
                            logger.info(f"🎯 {name}: {method_name} 메서드 실행 중...")
                            result = method(*args, **kwargs)
                            self.warmup_status[name] = True
                            logger.info(f"✅ {name}: 워밍업 성공 ({method_name})")
                            return True
                
                # 5. callable 객체인 경우
                if callable(obj):
                    logger.info(f"🎯 {name}: 객체 자체가 callable, 실행 중...")
                    result = obj(*args, **kwargs)
                    self.warmup_status[name] = True
                    logger.info(f"✅ {name}: 워밍업 성공 (callable)")
                    return True
                
                # 6. 워밍업이 필요 없는 경우
                logger.info(f"ℹ️ {name}: 워밍업 메서드 없음, 스킵")
                self.warmup_status[name] = True
                return True
        
        except Exception as e:
            error_msg = f"워밍업 실행 중 오류: {str(e)}"
            logger.error(f"❌ {name}: {error_msg}")
            self.warmup_status[name] = False
            self.warmup_errors[name] = error_msg
            return False
    
    async def async_safe_warmup(self, obj: Any, name: str = "unknown", *args, **kwargs) -> bool:
        """비동기 안전한 워밍업"""
        try:
            # 비동기 워밍업 메서드 확인
            if hasattr(obj, 'async_warmup') and callable(obj.async_warmup):
                logger.info(f"🎯 {name}: async_warmup 메서드 실행 중...")
                result = await obj.async_warmup(*args, **kwargs)
                self.warmup_status[name] = True
                logger.info(f"✅ {name}: 비동기 워밍업 성공")
                return True
            else:
                # 동기 워밍업으로 폴백
                return self.safe_warmup(obj, name, *args, **kwargs)
        
        except Exception as e:
            error_msg = f"비동기 워밍업 실행 중 오류: {str(e)}"
            logger.error(f"❌ {name}: {error_msg}")
            self.warmup_status[name] = False
            self.warmup_errors[name] = error_msg
            return False
    
    def get_warmup_status(self) -> Dict[str, Dict[str, Union[bool, str]]]:
        """워밍업 상태 조회"""
        status = {}
        
        for name in set(list(self.warmup_status.keys()) + list(self.warmup_errors.keys())):
            status[name] = {
                'success': self.warmup_status.get(name, False),
                'error': self.warmup_errors.get(name, None)
            }
        
        return status
    
    def clear_status(self):
        """상태 초기화"""
        self.warmup_status.clear()
        self.warmup_errors.clear()

# 전역 인스턴스
_warmup_manager = SafeWarmupManager()

# 편의 함수들
def safe_warmup(obj: Any, name: str = "unknown", *args, **kwargs) -> bool:
    """안전한 워밍업 (전역 함수)"""
    return _warmup_manager.safe_warmup(obj, name, *args, **kwargs)

async def async_safe_warmup(obj: Any, name: str = "unknown", *args, **kwargs) -> bool:
    """비동기 안전한 워밍업 (전역 함수)"""
    return await _warmup_manager.async_safe_warmup(obj, name, *args, **kwargs)

def get_warmup_status() -> Dict[str, Dict[str, Union[bool, str]]]:
    """워밍업 상태 조회 (전역 함수)"""
    return _warmup_manager.get_warmup_status()

def clear_warmup_status():
    """워밍업 상태 초기화 (전역 함수)"""
    _warmup_manager.clear_status()

__all__ = [
    'SafeWarmupManager', 
    'safe_warmup', 
    'async_safe_warmup', 
    'get_warmup_status', 
    'clear_warmup_status'
]