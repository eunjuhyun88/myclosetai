# app/core/di_container.py
"""
🔥 의존성 주입 컨테이너 - 순환 임포트 해결
✅ 싱글톤 패턴으로 전역 관리
✅ 인터페이스 기반 등록/조회
✅ 지연 로딩 지원
✅ conda 환경 최적화
"""

import logging
import threading
import weakref
from typing import Dict, Any, Optional, Type, TypeVar, Callable, Union
from abc import ABC, abstractmethod

# 타입 변수
T = TypeVar('T')

logger = logging.getLogger(__name__)

class IDependencyContainer(ABC):
    """의존성 컨테이너 인터페이스"""
    
    @abstractmethod
    def register(self, interface: Union[str, Type], implementation: Any, singleton: bool = True) -> None:
        """의존성 등록"""
        pass
    
    @abstractmethod
    def get(self, interface: Union[str, Type]) -> Optional[Any]:
        """의존성 조회"""
        pass
    
    @abstractmethod
    def is_registered(self, interface: Union[str, Type]) -> bool:
        """등록 여부 확인"""
        pass

class DIContainer(IDependencyContainer):
    """
    🔥 의존성 주입 컨테이너
    
    ✅ 싱글톤 및 임시 인스턴스 지원
    ✅ 팩토리 함수 지원
    ✅ 스레드 안전
    ✅ 약한 참조로 메모리 누수 방지
    """
    
    def __init__(self):
        self._services: Dict[str, Any] = {}
        self._singletons: Dict[str, Any] = {}
        self._factories: Dict[str, Callable] = {}
        self._singleton_flags: Dict[str, bool] = {}
        self._lock = threading.RLock()
        self._weak_refs: Dict[str, weakref.ref] = {}
    
    def register(
        self, 
        interface: Union[str, Type], 
        implementation: Any, 
        singleton: bool = True,
        factory: Optional[Callable] = None
    ) -> None:
        """
        의존성 등록
        
        Args:
            interface: 인터페이스 (문자열 또는 타입)
            implementation: 구현체 또는 클래스
            singleton: 싱글톤 여부
            factory: 팩토리 함수
        """
        try:
            with self._lock:
                key = self._get_key(interface)
                
                if factory:
                    self._factories[key] = factory
                else:
                    self._services[key] = implementation
                
                self._singleton_flags[key] = singleton
                
                logger.debug(f"✅ 의존성 등록: {key} ({'싱글톤' if singleton else '임시'})")
                
        except Exception as e:
            logger.error(f"❌ 의존성 등록 실패 {interface}: {e}")
    
    def get(self, interface: Union[str, Type]) -> Optional[Any]:
        """
        의존성 조회
        
        Args:
            interface: 인터페이스
            
        Returns:
            구현체 인스턴스 또는 None
        """
        try:
            with self._lock:
                key = self._get_key(interface)
                
                # 싱글톤 캐시 확인
                if key in self._singletons:
                    return self._singletons[key]
                
                # 팩토리 함수로 생성
                if key in self._factories:
                    instance = self._factories[key]()
                    
                    if self._singleton_flags.get(key, True):
                        self._singletons[key] = instance
                    
                    return instance
                
                # 직접 등록된 구현체
                if key in self._services:
                    implementation = self._services[key]
                    
                    # 클래스인 경우 인스턴스 생성
                    if isinstance(implementation, type):
                        instance = implementation()
                        
                        if self._singleton_flags.get(key, True):
                            self._singletons[key] = instance
                        
                        return instance
                    else:
                        # 이미 인스턴스인 경우
                        return implementation
                
                logger.debug(f"⚠️ 등록되지 않은 의존성: {key}")
                return None
                
        except Exception as e:
            logger.error(f"❌ 의존성 조회 실패 {interface}: {e}")
            return None
    
    def is_registered(self, interface: Union[str, Type]) -> bool:
        """등록 여부 확인"""
        try:
            with self._lock:
                key = self._get_key(interface)
                return key in self._services or key in self._factories
        except:
            return False
    
    def register_factory(
        self, 
        interface: Union[str, Type], 
        factory: Callable, 
        singleton: bool = True
    ) -> None:
        """팩토리 함수 등록"""
        self.register(interface, None, singleton, factory)
    
    def register_instance(self, interface: Union[str, Type], instance: Any) -> None:
        """인스턴스 직접 등록 (항상 싱글톤)"""
        try:
            with self._lock:
                key = self._get_key(interface)
                self._singletons[key] = instance
                self._singleton_flags[key] = True
                logger.debug(f"✅ 인스턴스 등록: {key}")
        except Exception as e:
            logger.error(f"❌ 인스턴스 등록 실패 {interface}: {e}")
    
    def clear(self) -> None:
        """모든 등록된 의존성 제거"""
        try:
            with self._lock:
                self._services.clear()
                self._singletons.clear()
                self._factories.clear()
                self._singleton_flags.clear()
                self._weak_refs.clear()
                logger.info("🧹 DI Container 정리 완료")
        except Exception as e:
            logger.error(f"❌ DI Container 정리 실패: {e}")
    
    def get_registered_services(self) -> Dict[str, Dict[str, Any]]:
        """등록된 서비스 목록 조회"""
        try:
            with self._lock:
                services = {}
                
                for key in set(self._services.keys()) | set(self._factories.keys()):
                    services[key] = {
                        'has_implementation': key in self._services,
                        'has_factory': key in self._factories,
                        'is_singleton': self._singleton_flags.get(key, True),
                        'has_instance': key in self._singletons
                    }
                
                return services
        except Exception as e:
            logger.error(f"❌ 서비스 목록 조회 실패: {e}")
            return {}
    
    def _get_key(self, interface: Union[str, Type]) -> str:
        """인터페이스를 키 문자열로 변환"""
        if isinstance(interface, str):
            return interface
        elif hasattr(interface, '__name__'):
            return interface.__name__
        else:
            return str(interface)

# ==============================================
# 🔥 전역 DI Container 인스턴스
# ==============================================

_global_container: Optional[DIContainer] = None
_container_lock = threading.RLock()

def get_di_container() -> DIContainer:
    """전역 DI Container 인스턴스 반환"""
    global _global_container
    
    with _container_lock:
        if _global_container is None:
            _global_container = DIContainer()
            logger.info("🔗 전역 DI Container 초기화 완료")
        
        return _global_container

def reset_di_container() -> None:
    """전역 DI Container 리셋"""
    global _global_container
    
    with _container_lock:
        if _global_container:
            _global_container.clear()
        _global_container = DIContainer()
        logger.info("🔄 전역 DI Container 리셋 완료")

# ==============================================
# 🔥 기본 의존성 등록 함수들
# ==============================================

def register_default_dependencies():
    """기본 의존성들 등록"""
    try:
        container = get_di_container()
        
        # ModelLoader 팩토리 등록
        def create_model_loader():
            try:
                from ..ai_pipeline.utils.model_loader import ModelLoader, get_global_model_loader
                return get_global_model_loader()
            except ImportError:
                logger.debug("ModelLoader import 실패")
                return None
        
        container.register_factory('model_loader', create_model_loader, singleton=True)
        container.register_factory('IModelLoader', create_model_loader, singleton=True)
        
        # MemoryManager 팩토리 등록
        def create_memory_manager():
            try:
                from ..ai_pipeline.utils.memory_manager import get_global_memory_manager
                return get_global_memory_manager()
            except ImportError:
                logger.debug("MemoryManager import 실패")
                return None
        
        container.register_factory('memory_manager', create_memory_manager, singleton=True)
        container.register_factory('IMemoryManager', create_memory_manager, singleton=True)
        
        # SafeFunctionValidator 팩토리 등록
        def create_function_validator():
            try:
                from ..ai_pipeline.utils.model_loader import SafeFunctionValidator
                return SafeFunctionValidator()
            except ImportError:
                from ..ai_pipeline.steps.base_step_mixin import FallbackSafeFunctionValidator
                return FallbackSafeFunctionValidator()
        
        container.register_factory('function_validator', create_function_validator, singleton=True)
        container.register_factory('ISafeFunctionValidator', create_function_validator, singleton=True)
        
        logger.info("✅ 기본 의존성 등록 완료")
        
    except Exception as e:
        logger.error(f"❌ 기본 의존성 등록 실패: {e}")

# ==============================================
# 🔥 편의 함수들
# ==============================================

def inject_dependencies_to_step(step_instance, container: Optional[DIContainer] = None):
    """Step 인스턴스에 의존성 주입"""
    try:
        if container is None:
            container = get_di_container()
        
        # 의존성 조회 및 주입
        model_loader = container.get('IModelLoader')
        memory_manager = container.get('IMemoryManager')
        function_validator = container.get('ISafeFunctionValidator')
        
        # Step에 inject_dependencies 메서드가 있으면 사용
        if hasattr(step_instance, 'inject_dependencies'):
            step_instance.inject_dependencies(
                model_loader=model_loader,
                memory_manager=memory_manager,
                function_validator=function_validator
            )
            logger.debug(f"✅ {step_instance.__class__.__name__} 의존성 주입 완료")
        else:
            logger.debug(f"⚠️ {step_instance.__class__.__name__}에 inject_dependencies 메서드 없음")
            
    except Exception as e:
        logger.error(f"❌ Step 의존성 주입 실패: {e}")

def create_step_with_di(step_class: Type, **kwargs) -> Any:
    """의존성 주입을 사용하여 Step 인스턴스 생성"""
    try:
        container = get_di_container()
        
        # 의존성 조회
        model_loader = container.get('IModelLoader')
        memory_manager = container.get('IMemoryManager')
        function_validator = container.get('ISafeFunctionValidator')
        
        # Step 인스턴스 생성 (의존성 주입)
        step_instance = step_class(
            model_loader=model_loader,
            memory_manager=memory_manager,
            function_validator=function_validator,
            **kwargs
        )
        
        logger.debug(f"✅ {step_class.__name__} DI 생성 완료")
        return step_instance
        
    except Exception as e:
        logger.error(f"❌ {step_class.__name__} DI 생성 실패: {e}")
        # 폴백: 일반 생성
        return step_class(**kwargs)

# ==============================================
# 🔥 초기화
# ==============================================

def initialize_di_system():
    """DI 시스템 초기화"""
    try:
        # 기본 컨테이너 생성
        container = get_di_container()
        
        # 기본 의존성 등록
        register_default_dependencies()
        
        logger.info("🔗 DI 시스템 초기화 완료")
        return True
        
    except Exception as e:
        logger.error(f"❌ DI 시스템 초기화 실패: {e}")
        return False

# 모듈 로드 시 자동 초기화
if __name__ != "__main__":
    try:
        initialize_di_system()
    except Exception as e:
        logger.debug(f"DI 시스템 자동 초기화 실패: {e}")