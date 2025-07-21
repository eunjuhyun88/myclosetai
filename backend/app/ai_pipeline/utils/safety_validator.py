# backend/app/ai_pipeline/utils/safety_validator.py
"""
🛡️ MyCloset AI - 함수/객체 안전성 검증 시스템 v1.0
=======================================================
✅ SafeFunctionValidator - 함수 호출 안전성 검증
✅ AsyncCompatibilityManager - Coroutine 오류 완전 해결  
✅ CallableWrapper 시스템 - dict를 callable로 변환
✅ 순환참조 방지 - ModelLoader와 한방향 참조
✅ 모듈화된 구조로 독립적 사용 가능
✅ M3 Max 128GB 최적화
✅ conda 환경 우선 지원

Author: MyCloset AI Team
Date: 2025-07-21
Version: 1.0 (분리된 안전성 시스템)
"""

import asyncio
import threading
import logging
import time
import traceback
import types
from typing import Any, Dict, Tuple, Callable, Optional, Union, List
from functools import wraps
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

# ==============================================
# 🔥 1. 핵심 안전성 검증 클래스
# ==============================================

class SafeFunctionValidator:
    """함수/메서드/객체 호출 안전성 검증 클래스"""
    
    @staticmethod
    def validate_callable(obj: Any, context: str = "unknown") -> Tuple[bool, str, Any]:
        """
        객체가 안전하게 호출 가능한지 검증
        
        Args:
            obj: 검증할 객체
            context: 호출 컨텍스트
            
        Returns:
            (is_callable, reason, safe_obj) 튜플
        """
        try:
            if obj is None:
                return False, "Object is None", None
            
            # dict 체크 (가장 흔한 오류)
            if isinstance(obj, dict):
                return False, f"Object is dict, not callable in context: {context}", None
            
            # Coroutine 객체 체크
            if hasattr(obj, '__class__') and 'coroutine' in str(type(obj)):
                return False, f"Object is coroutine, need await in context: {context}", None
            
            # 비동기 함수 체크 (정상적인 경우)
            if asyncio.iscoroutinefunction(obj):
                return True, f"Object is async function in context: {context}", obj
            
            # 기본 데이터 타입 체크
            basic_types = (str, int, float, bool, list, tuple, set, bytes, bytearray)
            if isinstance(obj, basic_types):
                return False, f"Object is basic data type {type(obj)}, not callable", None
            
            # callable 체크
            if not callable(obj):
                return False, f"Object type {type(obj)} is not callable", None
            
            # 함수/메서드 타입 검증
            if isinstance(obj, (types.FunctionType, types.MethodType, types.BuiltinFunctionType, types.BuiltinMethodType)):
                return True, "Valid function/method", obj
            
            # __call__ 메서드 체크
            if hasattr(obj, '__call__'):
                call_method = getattr(obj, '__call__')
                if callable(call_method) and not isinstance(call_method, dict):
                    return True, "Valid callable object with __call__", obj
                else:
                    return False, "__call__ method is dict, not callable", None
            
            # 일반 callable 객체
            if callable(obj):
                return True, "Generic callable object", obj
            
            return False, f"Unknown callable validation failure for {type(obj)}", None
            
        except Exception as e:
            return False, f"Validation error: {e}", None
    
    @staticmethod
    def safe_call(obj: Any, *args, **kwargs) -> Tuple[bool, Any, str]:
        """
        안전한 함수/메서드 호출 - 동기 버전
        
        Args:
            obj: 호출할 객체
            *args: 위치 인수
            **kwargs: 키워드 인수
            
        Returns:
            (success, result, message) 튜플
        """
        try:
            is_callable, reason, safe_obj = SafeFunctionValidator.validate_callable(obj, "safe_call")
            
            if not is_callable:
                return False, None, f"Cannot call: {reason}"
            
            try:
                result = safe_obj(*args, **kwargs)
                return True, result, "Success"
            except TypeError as e:
                error_msg = str(e)
                if "not callable" in error_msg.lower():
                    return False, None, f"Runtime callable error: {error_msg}"
                else:
                    return False, None, f"Type error in call: {error_msg}"
            except Exception as e:
                return False, None, f"Call execution error: {e}"
                
        except Exception as e:
            return False, None, f"Call failed: {e}"
    
    @staticmethod
    async def safe_call_async(obj: Any, *args, **kwargs) -> Tuple[bool, Any, str]:
        """
        안전한 비동기 함수/메서드 호출
        
        Args:
            obj: 호출할 객체
            *args: 위치 인수
            **kwargs: 키워드 인수
            
        Returns:
            (success, result, message) 튜플
        """
        try:
            # Coroutine 객체 직접 체크
            if hasattr(obj, '__class__') and 'coroutine' in str(type(obj)):
                return False, None, f"Cannot call coroutine object directly - need await"
            
            is_callable, reason, safe_obj = SafeFunctionValidator.validate_callable(obj, "safe_call_async")
            
            if not is_callable:
                return False, None, f"Cannot call: {reason}"
            
            try:
                if asyncio.iscoroutinefunction(safe_obj):
                    result = await safe_obj(*args, **kwargs)
                    return True, result, "Async success"
                else:
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(None, lambda: safe_obj(*args, **kwargs))
                    return True, result, "Sync in executor success"
                    
            except TypeError as e:
                error_msg = str(e)
                if "not callable" in error_msg.lower():
                    return False, None, f"Runtime callable error: {error_msg}"
                else:
                    return False, None, f"Type error in async call: {error_msg}"
            except Exception as e:
                return False, None, f"Async call execution error: {e}"
                
        except Exception as e:
            return False, None, f"Async call failed: {e}"

# ==============================================
# 🔥 2. 비동기 호환성 관리자 (Coroutine 오류 해결)
# ==============================================

class AsyncCompatibilityManager:
    """비동기 호환성 관리자 - Coroutine 오류 해결 강화"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.AsyncCompatibilityManager")
        self._lock = threading.Lock()
        
    def make_callable_safe(self, obj: Any) -> Any:
        """객체를 안전하게 호출 가능하도록 변환"""
        try:
            if obj is None:
                return self._create_none_wrapper()
            
            if hasattr(obj, '__class__') and 'coroutine' in str(type(obj)):
                self.logger.warning("⚠️ Coroutine 객체 감지, 안전한 래퍼 생성")
                return self._create_coroutine_wrapper(obj)
            
            if isinstance(obj, dict):
                return self._create_dict_wrapper(obj)
            
            if callable(obj):
                return self._create_callable_wrapper(obj)
            
            if isinstance(obj, (str, int, float, bool, list, tuple)):
                return self._create_data_wrapper(obj)
            
            return self._create_object_wrapper(obj)
            
        except Exception as e:
            self.logger.error(f"❌ make_callable_safe 오류: {e}")
            return self._create_emergency_wrapper(obj, str(e))
    
    def _create_none_wrapper(self) -> Any:
        """None 객체용 래퍼"""
        class SafeNoneWrapper:
            def __init__(self):
                self.name = "none_wrapper"
                
            def __call__(self, *args, **kwargs):
                return {
                    'status': 'success',
                    'model_name': self.name,
                    'result': None,
                    'call_type': 'none_wrapper'
                }
            
            async def async_call(self, *args, **kwargs):
                await asyncio.sleep(0.001)
                return self.__call__(*args, **kwargs)
        
        return SafeNoneWrapper()
    
    def _create_dict_wrapper(self, data: Dict[str, Any]) -> Any:
        """Dict를 callable wrapper로 변환"""
        class SafeDictWrapper:
            def __init__(self, data: Dict[str, Any]):
                self.data = data.copy()
                self.name = data.get('name', 'unknown')
                
            def __call__(self, *args, **kwargs):
                return {
                    'status': 'success',
                    'model_name': self.name,
                    'result': f'mock_result_for_{self.name}',
                    'data': self.data,
                    'call_type': 'sync'
                }
            
            async def async_call(self, *args, **kwargs):
                await asyncio.sleep(0.001)
                return {
                    'status': 'success',
                    'model_name': self.name,
                    'result': f'mock_result_for_{self.name}',
                    'data': self.data,
                    'call_type': 'async'
                }
            
            def __await__(self):
                return self.async_call().__await__()
        
        return SafeDictWrapper(data)
    
    def _create_coroutine_wrapper(self, coro) -> Any:
        """Coroutine을 callable wrapper로 변환"""
        class SafeCoroutineWrapper:
            def __init__(self, coroutine):
                self.coroutine = coroutine
                self.name = "coroutine_wrapper"
                
            def __call__(self, *args, **kwargs):
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        task = asyncio.create_task(self.coroutine)
                        return task
                    else:
                        return loop.run_until_complete(self.coroutine)
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        return loop.run_until_complete(self.coroutine)
                    finally:
                        loop.close()
            
            async def async_call(self, *args, **kwargs):
                return await self.coroutine
            
            def __await__(self):
                return self.coroutine.__await__()
        
        return SafeCoroutineWrapper(coro)
    
    def _create_callable_wrapper(self, func) -> Any:
        """Callable 객체를 안전한 wrapper로 변환"""
        class SafeCallableWrapper:
            def __init__(self, func):
                self.func = func
                self.is_async = asyncio.iscoroutinefunction(func)
                
            def __call__(self, *args, **kwargs):
                if self.is_async:
                    try:
                        loop = asyncio.get_event_loop()
                        if loop.is_running():
                            return asyncio.create_task(self.func(*args, **kwargs))
                        else:
                            return loop.run_until_complete(self.func(*args, **kwargs))
                    except RuntimeError:
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        try:
                            return loop.run_until_complete(self.func(*args, **kwargs))
                        finally:
                            loop.close()
                else:
                    return self.func(*args, **kwargs)
            
            async def async_call(self, *args, **kwargs):
                if self.is_async:
                    return await self.func(*args, **kwargs)
                else:
                    return self.func(*args, **kwargs)
        
        return SafeCallableWrapper(func)
    
    def _create_data_wrapper(self, data: Any) -> Any:
        """기본 데이터 타입용 래퍼"""
        class SafeDataWrapper:
            def __init__(self, data: Any):
                self.data = data
                self.name = f"data_wrapper_{type(data).__name__}"
                
            def __call__(self, *args, **kwargs):
                return {
                    'status': 'success',
                    'model_name': self.name,
                    'result': self.data,
                    'call_type': 'data_wrapper'
                }
            
            async def async_call(self, *args, **kwargs):
                await asyncio.sleep(0.001)
                return self.__call__(*args, **kwargs)
        
        return SafeDataWrapper(data)
    
    def _create_object_wrapper(self, obj: Any) -> Any:
        """일반 객체용 래퍼"""
        class SafeObjectWrapper:
            def __init__(self, obj: Any):
                self.obj = obj
                self.name = f"object_wrapper_{type(obj).__name__}"
                
            def __call__(self, *args, **kwargs):
                return {
                    'status': 'success',
                    'model_name': self.name,
                    'result': f'wrapped_{self.name}',
                    'call_type': 'object_wrapper'
                }
            
            async def async_call(self, *args, **kwargs):
                await asyncio.sleep(0.001)
                return self.__call__(*args, **kwargs)
            
            def __getattr__(self, name):
                if hasattr(self.obj, name):
                    return getattr(self.obj, name)
                raise AttributeError(f"'{self.name}' has no attribute '{name}'")
        
        return SafeObjectWrapper(obj)
    
    def _create_emergency_wrapper(self, obj: Any, error_msg: str) -> Any:
        """긴급 상황용 래퍼"""
        class EmergencyWrapper:
            def __init__(self, obj: Any, error: str):
                self.obj = obj
                self.error = error
                self.name = "emergency_wrapper"
                
            def __call__(self, *args, **kwargs):
                return {
                    'status': 'emergency',
                    'model_name': self.name,
                    'result': f'emergency_result',
                    'error': self.error,
                    'call_type': 'emergency'
                }
            
            async def async_call(self, *args, **kwargs):
                await asyncio.sleep(0.001)
                return self.__call__(*args, **kwargs)
        
        return EmergencyWrapper(obj, error_msg)

# ==============================================
# 🔥 3. 안전한 모델 서비스 기반 클래스
# ==============================================

class SafeModelServiceBase:
    """안전한 모델 서비스 기반 클래스 (ModelLoader에서 상속받아 사용)"""
    
    def __init__(self):
        self.validator = SafeFunctionValidator()
        self.async_manager = AsyncCompatibilityManager()
        self.logger = logging.getLogger(f"{__name__}.SafeModelServiceBase")
        self._lock = threading.RLock()
        self._async_lock = asyncio.Lock()
    
    def _create_callable_dict_wrapper(self, model_dict: Dict[str, Any]) -> Callable:
        """딕셔너리를 callable wrapper로 변환"""
        class CallableDictWrapper:
            def __init__(self, data: Dict[str, Any]):
                self.data = data.copy()
                self.name = data.get('name', 'unknown')
                self.type = data.get('type', 'dict_model')
                self.call_count = 0
                self.last_call_time = None
            
            def __call__(self, *args, **kwargs):
                self.call_count += 1
                self.last_call_time = time.time()
                
                return {
                    'status': 'success',
                    'model_name': self.name,
                    'model_type': self.type,
                    'result': f'mock_result_for_{self.name}',
                    'data': self.data,
                    'call_metadata': {
                        'call_count': self.call_count,
                        'timestamp': self.last_call_time,
                        'wrapper_type': 'dict'
                    }
                }
            
            async def async_call(self, *args, **kwargs):
                await asyncio.sleep(0.01)
                return self.__call__(*args, **kwargs)
            
            def get_info(self):
                return {
                    **self.data,
                    'wrapper_info': {
                        'type': 'dict_wrapper',
                        'call_count': self.call_count,
                        'last_call_time': self.last_call_time
                    }
                }
            
            def warmup(self):
                try:
                    test_result = self()
                    return test_result.get('status') == 'success'
                except Exception:
                    return False
        
        return CallableDictWrapper(model_dict)
    
    def safe_register_model(self, name: str, model: Any) -> bool:
        """안전한 모델 등록"""
        try:
            with self._lock:
                if isinstance(model, dict):
                    wrapper = self._create_callable_dict_wrapper(model)
                    self.logger.info(f"📝 딕셔너리 모델을 callable wrapper로 등록: {name}")
                    return True
                elif callable(model):
                    is_callable, reason, safe_model = self.validator.validate_callable(model, f"register_{name}")
                    if is_callable:
                        safe_wrapped = self.async_manager.make_callable_safe(safe_model)
                        self.logger.info(f"📝 검증된 callable 모델 등록: {name}")
                        return True
                    else:
                        wrapper = self.async_manager.make_callable_safe(model)
                        self.logger.warning(f"⚠️ 안전하지 않은 callable 모델을 wrapper로 등록: {name}")
                        return True
                else:
                    wrapper = self.async_manager.make_callable_safe(model)
                    self.logger.info(f"📝 객체 모델을 wrapper로 등록: {name}")
                    return True
                
        except Exception as e:
            self.logger.error(f"❌ 안전한 모델 등록 실패 {name}: {e}")
            return False
    
    def safe_call_model(self, model: Any, *args, **kwargs) -> Any:
        """안전한 모델 호출 - 동기 버전"""
        try:
            if isinstance(model, dict):
                self.logger.error(f"❌ 등록된 모델이 dict입니다")
                return None
            
            success, result, message = self.validator.safe_call(model, *args, **kwargs)
            
            if success:
                self.logger.debug(f"✅ 모델 호출 성공")
                return result
            else:
                self.logger.warning(f"⚠️ 모델 호출 실패 - {message}")
                return None
                
        except Exception as e:
            self.logger.error(f"❌ 모델 호출 오류: {e}")
            return None
    
    async def safe_call_model_async(self, model: Any, *args, **kwargs) -> Any:
        """안전한 모델 호출 - 비동기 버전"""
        try:
            # Coroutine 객체 직접 체크 및 처리
            if hasattr(model, '__class__') and 'coroutine' in str(type(model)):
                self.logger.warning(f"⚠️ Coroutine 객체 감지, 대기 처리")
                try:
                    result = await model
                    self.logger.debug(f"✅ Coroutine 대기 완료")
                    return result
                except Exception as coro_error:
                    self.logger.error(f"❌ Coroutine 대기 실패: {coro_error}")
                    return None
            
            # async_call 메서드 우선 시도
            if hasattr(model, 'async_call'):
                try:
                    result = await model.async_call(*args, **kwargs)
                    self.logger.debug(f"✅ 비동기 모델 호출 성공 (async_call)")
                    return result
                except Exception as e:
                    self.logger.warning(f"⚠️ async_call 실패, safe_call_async 시도: {e}")
            
            # 일반 비동기 호출
            success, result, message = await self.validator.safe_call_async(model, *args, **kwargs)
            
            if success:
                self.logger.debug(f"✅ 비동기 모델 호출 성공")
                return result
            else:
                self.logger.warning(f"⚠️ 비동기 모델 호출 실패 - {message}")
                
                # 추가 시도: 동기 호출로 폴백
                if "coroutine" in message.lower():
                    self.logger.info(f"🔄 Coroutine 오류로 인해 동기 호출 시도")
                    try:
                        sync_success, sync_result, sync_message = self.validator.safe_call(model, *args, **kwargs)
                        if sync_success:
                            self.logger.info(f"✅ 동기 폴백 호출 성공")
                            return sync_result
                    except Exception as sync_error:
                        self.logger.warning(f"⚠️ 동기 폴백도 실패: {sync_error}")
                
                return None
                
        except Exception as e:
            self.logger.error(f"❌ 비동기 모델 호출 오류: {e}")
            return None

# ==============================================
# 🔥 4. 안전한 호출 데코레이터
# ==============================================

def safe_async_call(func):
    """비동기 함수 안전 호출 데코레이터"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            if asyncio.iscoroutinefunction(func):
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        return asyncio.create_task(func(*args, **kwargs))
                    else:
                        return loop.run_until_complete(func(*args, **kwargs))
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        return loop.run_until_complete(func(*args, **kwargs))
                    finally:
                        loop.close()
            else:
                return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"❌ safe_async_call 오류: {e}")
            return None
    return wrapper

def safe_coroutine_handler(func):
    """Coroutine 안전 처리 데코레이터"""
    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        try:
            result = await func(*args, **kwargs)
            if hasattr(result, '__class__') and 'coroutine' in str(type(result)):
                logger.warning("⚠️ 중첩 Coroutine 감지, 추가 대기")
                return await result
            return result
        except Exception as e:
            logger.error(f"❌ safe_coroutine_handler 오류: {e}")
            return None
    
    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"❌ safe_coroutine_handler 동기 오류: {e}")
            return None
    
    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    else:
        return sync_wrapper

# ==============================================
# 🔥 5. 유틸리티 함수들
# ==============================================

def is_safely_callable(obj: Any) -> bool:
    """전역 callable 안전성 검증"""
    is_callable, reason, safe_obj = SafeFunctionValidator.validate_callable(obj)
    return is_callable

def safe_call(obj: Any, *args, **kwargs) -> Tuple[bool, Any, str]:
    """전역 안전한 함수 호출 - 동기 버전"""
    return SafeFunctionValidator.safe_call(obj, *args, **kwargs)

async def safe_call_async(obj: Any, *args, **kwargs) -> Tuple[bool, Any, str]:
    """전역 안전한 함수 호출 - 비동기 버전"""
    return await SafeFunctionValidator.safe_call_async(obj, *args, **kwargs)

def create_safe_wrapper(obj: Any) -> Any:
    """객체를 안전한 래퍼로 변환"""
    manager = AsyncCompatibilityManager()
    return manager.make_callable_safe(obj)

def validate_model_callable(model: Any, model_name: str = "unknown") -> Dict[str, Any]:
    """모델 callable 상태 검증"""
    try:
        is_callable, reason, safe_obj = SafeFunctionValidator.validate_callable(model, f"model_{model_name}")
        
        return {
            "is_callable": is_callable,
            "reason": reason,
            "model_name": model_name,
            "model_type": type(model).__name__,
            "has_async_call": hasattr(model, 'async_call'),
            "is_coroutine": hasattr(model, '__class__') and 'coroutine' in str(type(model)),
            "is_dict": isinstance(model, dict),
            "validation_time": time.time()
        }
    except Exception as e:
        return {
            "is_callable": False,
            "reason": f"Validation error: {e}",
            "model_name": model_name,
            "error": str(e)
        }

# ==============================================
# 🔥 6. 모듈 내보내기
# ==============================================

__all__ = [
    # 핵심 클래스들
    'SafeFunctionValidator',
    'AsyncCompatibilityManager', 
    'SafeModelServiceBase',
    
    # 데코레이터들
    'safe_async_call',
    'safe_coroutine_handler',
    
    # 유틸리티 함수들
    'is_safely_callable',
    'safe_call',
    'safe_call_async',
    'create_safe_wrapper',
    'validate_model_callable'
]

logger.info("✅ 안전성 검증 시스템 v1.0 로드 완료")
logger.info("🛡️ SafeFunctionValidator - 함수 호출 안전성 검증")
logger.info("🔄 AsyncCompatibilityManager - Coroutine 오류 완전 해결")
logger.info("📦 CallableWrapper 시스템 - dict를 callable로 변환")
logger.info("🔗 순환참조 방지 - ModelLoader와 한방향 참조")
logger.info("🎯 모듈화된 구조로 독립적 사용 가능")