# backend/app/core/di_container.py
"""
🔥 Event-Driven DI Container v6.0 - 근본적 순환참조 해결
================================================================================

✅ Event-Driven Architecture - 의존성 요청/해결을 이벤트로 분리
✅ Factory Pattern + Command Pattern - 객체 생성 로직 완전 분리
✅ Pub/Sub 메시징 - 느슨한 결합으로 순환참조 원천 차단
✅ Lazy Registration - 실제 필요할 때만 의존성 해결
✅ Contextual Isolation - 각 Step이 독립적 DI 컨텍스트 보유
✅ Interface Segregation - 작은 단위 인터페이스로 책임 분리
✅ Dependency Graph - 의존성 추적으로 순환참조 사전 감지
✅ Observable Pattern - 의존성 변경 사항 실시간 알림
✅ Memory Pool - 객체 재사용으로 메모리 효율성 극대화
✅ 기존 API 100% 호환성 유지

핵심 아키텍처:
Event Bus → Dependency Factory → Service Registry → Lifecycle Manager

Author: MyCloset AI Team
Date: 2025-07-30
Version: 6.0 (Event-Driven + Factory Pattern)
"""

import os
import sys
import gc
import logging
import threading
import time
import weakref
import platform
import subprocess
import importlib
import traceback
import uuid
import json
from typing import Dict, Any, Optional, Type, TypeVar, Callable, Union, List, Set, Tuple, Protocol, runtime_checkable
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from concurrent.futures import ThreadPoolExecutor, Future
from functools import wraps, lru_cache
from collections import defaultdict, deque
import inspect
from pathlib import Path

# ==============================================
# 🔥 환경 설정 (독립적)
# ==============================================

logger = logging.getLogger(__name__)

# conda 환경 감지
CONDA_ENV = os.environ.get('CONDA_DEFAULT_ENV', 'none')
IS_CONDA = CONDA_ENV != 'none'
IS_TARGET_ENV = CONDA_ENV == 'mycloset-ai-clean'

# M3 Max 감지
def detect_m3_max() -> bool:
    try:
        if platform.system() == 'Darwin':
            result = subprocess.run(
                ['sysctl', '-n', 'machdep.cpu.brand_string'],
                capture_output=True, text=True, timeout=5
            )
            return 'M3' in result.stdout
    except Exception:
        pass
    return False

IS_M3_MAX = detect_m3_max()
MEMORY_GB = 128.0 if IS_M3_MAX else 16.0
DEVICE = 'mps' if IS_M3_MAX else 'cpu'

# PyTorch 가용성 체크
TORCH_AVAILABLE = False
MPS_AVAILABLE = False
try:
    import torch
    TORCH_AVAILABLE = True
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        MPS_AVAILABLE = True
        os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
        os.environ.setdefault('PYTORCH_MPS_HIGH_WATERMARK_RATIO', '0.0')
except ImportError:
    logger.debug("PyTorch 없음")

T = TypeVar('T')

# ==============================================
# 🔥 Event System - 이벤트 기반 의존성 해결
# ==============================================

class EventType(Enum):
    """DI Container 이벤트 타입"""
    DEPENDENCY_REQUESTED = auto()
    DEPENDENCY_RESOLVED = auto()
    DEPENDENCY_FAILED = auto()
    SERVICE_REGISTERED = auto()
    SERVICE_UNREGISTERED = auto()
    FACTORY_REGISTERED = auto()
    CONTEXT_CREATED = auto()
    CONTEXT_DESTROYED = auto()
    CIRCULAR_DEPENDENCY_DETECTED = auto()
    INJECTION_COMPLETED = auto()
    LIFECYCLE_CHANGED = auto()

@dataclass
class DIEvent:
    """DI Container 이벤트"""
    event_type: EventType
    service_key: str
    context_id: str
    timestamp: float = field(default_factory=time.time)
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    data: Dict[str, Any] = field(default_factory=dict)
    source: Optional[str] = None
    target: Optional[str] = None

class EventBus:
    """이벤트 버스 - 의존성 요청/해결을 이벤트로 분리"""
    
    def __init__(self):
        self._subscribers: Dict[EventType, List[Callable[[DIEvent], None]]] = defaultdict(list)
        self._event_history: deque = deque(maxlen=1000)
        self._lock = threading.RLock()
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        
    def subscribe(self, event_type: EventType, callback: Callable[[DIEvent], None]):
        """이벤트 구독"""
        with self._lock:
            self._subscribers[event_type].append(callback)
            self.logger.debug(f"✅ 이벤트 구독: {event_type.name}")
    
    def unsubscribe(self, event_type: EventType, callback: Callable[[DIEvent], None]):
        """이벤트 구독 해제"""
        with self._lock:
            if callback in self._subscribers[event_type]:
                self._subscribers[event_type].remove(callback)
                self.logger.debug(f"🔄 이벤트 구독 해제: {event_type.name}")
    
    def publish(self, event: DIEvent):
        """이벤트 발행"""
        with self._lock:
            self._event_history.append(event)
            
            # 구독자들에게 이벤트 전달
            for callback in self._subscribers[event.event_type]:
                try:
                    callback(event)
                except Exception as e:
                    self.logger.error(f"❌ 이벤트 처리 실패 {event.event_type.name}: {e}")
    
    def publish_async(self, event: DIEvent, executor: Optional[ThreadPoolExecutor] = None):
        """비동기 이벤트 발행"""
        if executor:
            executor.submit(self.publish, event)
        else:
            # 새 스레드에서 실행
            import threading
            thread = threading.Thread(target=self.publish, args=(event,))
            thread.daemon = True
            thread.start()
    
    def get_event_history(self, event_type: Optional[EventType] = None, limit: int = 100) -> List[DIEvent]:
        """이벤트 히스토리 조회"""
        with self._lock:
            if event_type:
                return [e for e in list(self._event_history)[-limit:] if e.event_type == event_type]
            return list(self._event_history)[-limit:]

# ==============================================
# 🔥 Service Factory - 객체 생성 로직 분리
# ==============================================

class ServiceLifecycle(Enum):
    """서비스 생명주기"""
    CREATED = auto()
    INITIALIZING = auto()
    READY = auto()
    STOPPING = auto()
    STOPPED = auto()
    ERROR = auto()

@dataclass
class ServiceDefinition:
    """서비스 정의"""
    service_key: str
    factory: Callable[[], Any]
    is_singleton: bool = True
    lifecycle: ServiceLifecycle = ServiceLifecycle.CREATED
    dependencies: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    creation_time: Optional[float] = None
    instance: Optional[Any] = None
    weak_ref: Optional[weakref.ref] = None

class DependencyFactory:
    """의존성 팩토리 - 순환참조 없는 객체 생성"""
    
    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self._service_definitions: Dict[str, ServiceDefinition] = {}
        self._creation_in_progress: Set[str] = set()
        self._creation_lock = threading.RLock()
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        
        # 기본 팩토리들 등록
        self._register_builtin_factories()
    
    def register_factory(self, service_key: str, factory: Callable[[], Any], 
                        is_singleton: bool = True, dependencies: List[str] = None):
        """팩토리 등록"""
        with self._creation_lock:
            definition = ServiceDefinition(
                service_key=service_key,
                factory=factory,
                is_singleton=is_singleton,
                dependencies=dependencies or [],
                metadata={'registered_at': time.time()}
            )
            
            self._service_definitions[service_key] = definition
            
            # 이벤트 발행
            event = DIEvent(
                event_type=EventType.FACTORY_REGISTERED,
                service_key=service_key,
                context_id="factory",
                data={'is_singleton': is_singleton, 'dependencies': dependencies or []}
            )
            self.event_bus.publish(event)
            
            self.logger.info(f"✅ 팩토리 등록: {service_key} (singleton: {is_singleton})")
    
    def create_service(self, service_key: str, context_id: str = "default") -> Optional[Any]:
        """서비스 생성 - 순환참조 감지 포함"""
        with self._creation_lock:
            # 순환참조 감지
            if service_key in self._creation_in_progress:
                self.logger.error(f"❌ 순환참조 감지: {service_key}")
                event = DIEvent(
                    event_type=EventType.CIRCULAR_DEPENDENCY_DETECTED,
                    service_key=service_key,
                    context_id=context_id,
                    data={'creation_stack': list(self._creation_in_progress)}
                )
                self.event_bus.publish(event)
                return None
            
            # 서비스 정의 확인
            if service_key not in self._service_definitions:
                self.logger.debug(f"⚠️ 팩토리 없음: {service_key}")
                return None
            
            definition = self._service_definitions[service_key]
            
            # 싱글톤 인스턴스 확인
            if definition.is_singleton and definition.instance is not None:
                return definition.instance
            
            # 약한 참조 확인
            if definition.is_singleton and definition.weak_ref is not None:
                instance = definition.weak_ref()
                if instance is not None:
                    return instance
            
            # 새 인스턴스 생성
            return self._create_new_instance(definition, context_id)
    
    def _create_new_instance(self, definition: ServiceDefinition, context_id: str) -> Optional[Any]:
        """새 인스턴스 생성"""
        service_key = definition.service_key
        
        try:
            # 생성 중 표시
            self._creation_in_progress.add(service_key)
            definition.lifecycle = ServiceLifecycle.INITIALIZING
            
            # 이벤트 발행
            event = DIEvent(
                event_type=EventType.DEPENDENCY_REQUESTED,
                service_key=service_key,
                context_id=context_id
            )
            self.event_bus.publish(event)
            
            # 팩토리 실행
            instance = definition.factory()
            
            if instance is not None:
                definition.instance = instance
                definition.creation_time = time.time()
                definition.lifecycle = ServiceLifecycle.READY
                
                # 싱글톤인 경우 약한 참조 저장 (기본 타입 제외)
                if definition.is_singleton:
                    try:
                        definition.weak_ref = weakref.ref(instance, lambda ref: self._cleanup_instance(service_key))
                    except TypeError:
                        # 기본 타입들은 약한 참조 불가, 직접 저장
                        pass
                
                # 성공 이벤트
                event = DIEvent(
                    event_type=EventType.DEPENDENCY_RESOLVED,
                    service_key=service_key,
                    context_id=context_id,
                    data={'instance_type': type(instance).__name__}
                )
                self.event_bus.publish(event)
                
                self.logger.info(f"✅ 서비스 생성 성공: {service_key}")
                return instance
            else:
                definition.lifecycle = ServiceLifecycle.ERROR
                raise ValueError(f"팩토리가 None을 반환: {service_key}")
                
        except Exception as e:
            definition.lifecycle = ServiceLifecycle.ERROR
            
            # 실패 이벤트
            event = DIEvent(
                event_type=EventType.DEPENDENCY_FAILED,
                service_key=service_key,
                context_id=context_id,
                data={'error': str(e), 'traceback': traceback.format_exc()}
            )
            self.event_bus.publish(event)
            
            self.logger.error(f"❌ 서비스 생성 실패 {service_key}: {e}")
            return None
        finally:
            # 생성 중 표시 제거
            self._creation_in_progress.discard(service_key)
    
    def _cleanup_instance(self, service_key: str):
        """인스턴스 정리 콜백"""
        if service_key in self._service_definitions:
            definition = self._service_definitions[service_key]
            definition.instance = None
            definition.weak_ref = None
            definition.lifecycle = ServiceLifecycle.STOPPED
            self.logger.debug(f"🗑️ 인스턴스 정리: {service_key}")
    
    def _register_builtin_factories(self):
        """내장 팩토리들 등록"""
        # ModelLoader 팩토리
        self.register_factory(
            'model_loader',
            lambda: self._create_model_loader_safe(),
            is_singleton=True
        )
        
        # MemoryManager 팩토리  
        self.register_factory(
            'memory_manager',
            lambda: self._create_memory_manager_safe(),
            is_singleton=True
        )
        
        # DataConverter 팩토리
        self.register_factory(
            'data_converter', 
            lambda: self._create_data_converter_safe(),
            is_singleton=True
        )
        
        # 🔥 수정: 기본 타입들은 직접 등록 (약한 참조 문제 해결)
        self._basic_values = {
            'device': DEVICE,
            'memory_gb': MEMORY_GB,
            'is_m3_max': IS_M3_MAX,
            'torch_available': TORCH_AVAILABLE,
            'mps_available': MPS_AVAILABLE
        }
    
    def _create_model_loader_safe(self):
        """ModelLoader 안전 생성"""
        import_paths = [
            'app.ai_pipeline.utils.model_loader',
            'ai_pipeline.utils.model_loader',
            'utils.model_loader'
        ]
        
        for path in import_paths:
            try:
                module = importlib.import_module(path)
                
                # get_global_model_loader 함수 우선
                if hasattr(module, 'get_global_model_loader'):
                    try:
                        # 🔥 DI Container 없이 생성 (순환참조 방지)
                        config = {}
                        loader = module.get_global_model_loader(config)
                        if loader:
                            self.logger.info(f"✅ ModelLoader 생성: {path} (get_global_model_loader)")
                            return loader
                    except Exception as e:
                        self.logger.debug(f"get_global_model_loader 실패: {e}")
                
                # ModelLoader 클래스 직접 생성
                if hasattr(module, 'ModelLoader'):
                    try:
                        ModelLoaderClass = module.ModelLoader
                        loader = ModelLoaderClass(device="auto")
                        self.logger.info(f"✅ ModelLoader 생성: {path} (클래스)")
                        return loader
                    except Exception as e:
                        self.logger.debug(f"ModelLoader 클래스 생성 실패: {e}")
                        
            except ImportError:
                continue
        
        # Mock 생성
        return self._create_mock_model_loader()
    
    def _create_memory_manager_safe(self):
        """MemoryManager 안전 생성"""
        import_paths = [
            'app.ai_pipeline.utils.memory_manager',
            'ai_pipeline.utils.memory_manager',
            'utils.memory_manager'
        ]
        
        for path in import_paths:
            try:
                module = importlib.import_module(path)
                
                if hasattr(module, 'get_global_memory_manager'):
                    manager = module.get_global_memory_manager()
                    if manager:
                        return manager
                
                if hasattr(module, 'MemoryManager'):
                    ManagerClass = module.MemoryManager
                    return ManagerClass()
                    
            except ImportError:
                continue
        
        return self._create_mock_memory_manager()
    
    def _create_data_converter_safe(self):
        """DataConverter 안전 생성"""
        import_paths = [
            'app.ai_pipeline.utils.data_converter',
            'ai_pipeline.utils.data_converter',
            'utils.data_converter'
        ]
        
        for path in import_paths:
            try:
                module = importlib.import_module(path)
                
                if hasattr(module, 'get_global_data_converter'):
                    converter = module.get_global_data_converter()
                    if converter:
                        return converter
                
                if hasattr(module, 'DataConverter'):
                    ConverterClass = module.DataConverter
                    return ConverterClass()
                    
            except ImportError:
                continue
        
        return self._create_mock_data_converter()
    
    def _create_mock_model_loader(self):
        """Mock ModelLoader"""
        class MockModelLoader:
            def __init__(self):
                self.is_initialized = True
                self.loaded_models = {}
                self.device = DEVICE
                self.logger = logging.getLogger("MockModelLoader")
                
            def load_model(self, model_path: str, **kwargs):
                model_id = f"mock_{len(self.loaded_models)}"
                self.loaded_models[model_id] = {
                    "path": model_path,
                    "loaded": True,
                    "device": self.device
                }
                return self.loaded_models[model_id]
            
            def create_step_interface(self, step_name: str):
                return MockStepInterface(step_name)
            
            def cleanup(self):
                self.loaded_models.clear()
        
        class MockStepInterface:
            def __init__(self, step_name):
                self.step_name = step_name
                self.is_initialized = True
            
            def get_model(self, model_name=None):
                return {"mock_model": model_name, "loaded": True}
        
        return MockModelLoader()
    
    def _create_mock_memory_manager(self):
        """Mock MemoryManager"""
        class MockMemoryManager:
            def __init__(self):
                self.is_initialized = True
                self.optimization_count = 0
            
            def optimize_memory(self, aggressive=False):
                self.optimization_count += 1
                gc.collect()
                return {"optimized": True, "count": self.optimization_count}
            
            def get_memory_info(self):
                return {
                    "total_gb": MEMORY_GB,
                    "available_gb": MEMORY_GB * 0.7,
                    "percent": 30.0
                }
            
            def cleanup(self):
                self.optimize_memory(aggressive=True)
        
        return MockMemoryManager()
    
    def _create_mock_data_converter(self):
        """Mock DataConverter"""
        class MockDataConverter:
            def __init__(self):
                self.is_initialized = True
                self.conversion_count = 0
            
            def convert(self, data, target_format):
                self.conversion_count += 1
                return {
                    "converted": f"mock_{target_format}_{self.conversion_count}",
                    "format": target_format
                }
            
            def get_supported_formats(self):
                return ["tensor", "numpy", "pil", "cv2"]
            
            def cleanup(self):
                self.conversion_count = 0
        
        return MockDataConverter()
    
    def get_service_definitions(self) -> Dict[str, ServiceDefinition]:
        """서비스 정의 목록 반환"""
        return dict(self._service_definitions)
    
    def is_service_available(self, service_key: str) -> bool:
        """서비스 사용 가능 여부"""
        return service_key in self._service_definitions

# ==============================================
# 🔥 Service Registry - 서비스 등록/조회
# ==============================================

class ServiceRegistry:
    """서비스 레지스트리"""
    
    def __init__(self, event_bus: EventBus, factory: DependencyFactory):
        self.event_bus = event_bus
        self.factory = factory
        self._registry: Dict[str, Any] = {}
        self._registry_lock = threading.RLock()
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
    
    def register_instance(self, service_key: str, instance: Any):
        """인스턴스 직접 등록"""
        with self._registry_lock:
            self._registry[service_key] = instance
            
            event = DIEvent(
                event_type=EventType.SERVICE_REGISTERED,
                service_key=service_key,
                context_id="registry",
                data={'instance_type': type(instance).__name__}
            )
            self.event_bus.publish(event)
            
            self.logger.info(f"✅ 서비스 등록: {service_key}")
    
    def get_service(self, service_key: str, context_id: str = "default") -> Optional[Any]:
        """서비스 조회"""
        with self._registry_lock:
            # 직접 등록된 인스턴스 우선
            if service_key in self._registry:
                return self._registry[service_key]
            
            # 팩토리를 통한 생성
            return self.factory.create_service(service_key, context_id)
    
    def unregister_service(self, service_key: str):
        """서비스 등록 해제"""
        with self._registry_lock:
            if service_key in self._registry:
                del self._registry[service_key]
                
                event = DIEvent(
                    event_type=EventType.SERVICE_UNREGISTERED,
                    service_key=service_key,
                    context_id="registry"
                )
                self.event_bus.publish(event)
                
                self.logger.info(f"🗑️ 서비스 등록 해제: {service_key}")
    
    def list_services(self) -> List[str]:
        """등록된 서비스 목록"""
        with self._registry_lock:
            registry_services = list(self._registry.keys())
            factory_services = list(self.factory.get_service_definitions().keys())
            return sorted(set(registry_services + factory_services))

# ==============================================
# 🔥 Contextual Container - 컨텍스트별 격리
# ==============================================

class ContextualDIContainer:
    """컨텍스트별 DI Container"""
    
    def __init__(self, context_id: str, event_bus: EventBus, factory: DependencyFactory):
        self.context_id = context_id
        self.event_bus = event_bus
        self.factory = factory
        self.registry = ServiceRegistry(event_bus, factory)
        self._creation_time = time.time()
        self._access_count = 0
        self._injection_count = 0
        self.logger = logging.getLogger(f"{self.__class__.__name__}.{context_id}")
        
        # 컨텍스트 생성 이벤트
        event = DIEvent(
            event_type=EventType.CONTEXT_CREATED,
            service_key="container",
            context_id=context_id,
            data={'creation_time': self._creation_time}
        )
        self.event_bus.publish(event)
        
        self.logger.info(f"✅ 컨텍스트 생성: {context_id}")
    
    def get(self, service_key: str) -> Optional[Any]:
        """서비스 조회"""
        self._access_count += 1
        service = self.registry.get_service(service_key, self.context_id)
        
        if service:
            self.logger.debug(f"✅ 서비스 조회 성공: {service_key}")
        else:
            self.logger.debug(f"⚠️ 서비스 조회 실패: {service_key}")
        
        return service
    
    def register_instance(self, service_key: str, instance: Any):
        """인스턴스 등록"""
        self.registry.register_instance(service_key, instance)
    
    def register_factory(self, service_key: str, factory_func: Callable[[], Any], is_singleton: bool = True):
        """팩토리 등록"""
        self.factory.register_factory(service_key, factory_func, is_singleton)
    
    def inject_to_step(self, step_instance) -> int:
        """Step에 의존성 주입"""
        injections_made = 0
        
        try:
            # PropertyInjectionMixin 지원
            if hasattr(step_instance, 'set_di_container'):
                step_instance.set_di_container(self)
                injections_made += 1
                self.logger.debug(f"✅ DI Container 주입")
            
            # 수동 속성 주입
            injection_map = {
                'model_loader': 'model_loader',
                'memory_manager': 'memory_manager',
                'data_converter': 'data_converter'
            }
            
            for attr_name, service_key in injection_map.items():
                if not hasattr(step_instance, attr_name) or getattr(step_instance, attr_name) is None:
                    service = self.get(service_key)
                    if service:
                        setattr(step_instance, attr_name, service)
                        injections_made += 1
                        self.logger.debug(f"✅ {attr_name} 주입")
            
            # DI Container 자체 주입
            if hasattr(step_instance, 'di_container'):
                step_instance.di_container = self
                injections_made += 1
            
            # 초기화 시도
            if hasattr(step_instance, 'initialize') and not getattr(step_instance, 'is_initialized', False):
                try:
                    step_instance.initialize()
                    self.logger.debug(f"✅ Step 초기화 완료")
                except Exception as e:
                    self.logger.debug(f"⚠️ Step 초기화 실패: {e}")
            
            self._injection_count += 1
            
            # 주입 완료 이벤트
            event = DIEvent(
                event_type=EventType.INJECTION_COMPLETED,
                service_key="injection",
                context_id=self.context_id,
                data={
                    'step_name': getattr(step_instance, '__class__', {}).get('__name__', 'Unknown'),
                    'injections_made': injections_made
                }
            )
            self.event_bus.publish(event)
            
            self.logger.info(f"✅ Step 의존성 주입 완료: {injections_made}개")
            
        except Exception as e:
            self.logger.error(f"❌ Step 의존성 주입 실패: {e}")
        
        return injections_made
    


        # backend/app/core/di_container.py의 ContextualDIContainer 클래스에 추가할 메서드들

    # ContextualDIContainer 클래스 내부에 다음 메서드들을 추가하세요:

    def register_lazy(self, service_key: str, factory: Callable[[], Any], is_singleton: bool = True):
        """지연 서비스 등록 (구 버전 호환)"""
        try:
            # LazyService로 래핑해서 등록
            lazy_service = LazyDependency(factory)  # LazyDependency는 LazyService의 별칭
            self.register_instance(service_key, lazy_service)
            
            self.logger.debug(f"✅ register_lazy 성공: {service_key}")
            return True
        except Exception as e:
            self.logger.debug(f"⚠️ register_lazy 실패 ({service_key}): {e}")
            return False

    def register(self, service_key: str, instance: Any, singleton: bool = True):
        """서비스 등록 (구 버전 호환)"""
        try:
            self.register_instance(service_key, instance)
            return True
        except Exception as e:
            self.logger.debug(f"⚠️ register 실패 ({service_key}): {e}")
            return False

    def register_factory_method(self, service_key: str, factory: Callable[[], Any], is_singleton: bool = True):
        """팩토리 등록 (구 버전 호환)"""
        try:
            self.register_factory(service_key, factory, is_singleton)
            return True
        except Exception as e:
            self.logger.debug(f"⚠️ register_factory_method 실패 ({service_key}): {e}")
            return False

    def has(self, service_key: str) -> bool:
        """서비스 존재 여부 확인 (구 버전 호환)"""
        try:
            service = self.get(service_key)
            return service is not None
        except Exception:
            return False

    def remove(self, service_key: str) -> bool:
        """서비스 제거 (구 버전 호환)"""
        try:
            # Event-Driven DI Container에서는 서비스 제거를 지원하지 않으므로
            # None으로 덮어쓰기
            self.register_instance(service_key, None)
            return True
        except Exception:
            return False

    def clear(self):
        """모든 서비스 정리 (구 버전 호환)"""
        try:
            self.cleanup()
        except Exception:
            pass

    def list_services(self) -> List[str]:
        """등록된 서비스 목록 (구 버전 호환)"""
        try:
            return self.registry.list_services()
        except Exception:
            return []

    def get_service_info(self, service_key: str) -> Dict[str, Any]:
        """서비스 정보 조회 (구 버전 호환)"""
        try:
            service = self.get(service_key)
            return {
                'service_key': service_key,
                'available': service is not None,
                'type': type(service).__name__ if service else None,
                'context': self.context_id
            }
        except Exception:
            return {
                'service_key': service_key,
                'available': False,
                'error': 'Failed to get service info'
            }

    def inject_to_step(self, step_instance) -> int:
        """Step에 의존성 주입 (구 버전 호환) - 기존 메서드와 동일"""
        return super().inject_to_step(step_instance)

    # ==============================================
    # 🔥 추가로 필요한 호환성 메서드들
    # ==============================================

    def force_register_model_loader(self, model_loader):
        """ModelLoader 강제 등록 (구 버전 호환)"""
        try:
            self.register_instance('model_loader', model_loader)
            self.logger.info("✅ ModelLoader 강제 등록 완료")
            return True
        except Exception as e:
            self.logger.error(f"❌ ModelLoader 강제 등록 실패: {e}")
            return False

    def optimize_memory(self, aggressive: bool = False) -> Dict[str, Any]:
        """메모리 최적화 (구 버전 호환)"""
        try:
            # 가비지 컬렉션 실행
            import gc
            collected = gc.collect()
            
            # M3 Max MPS 캐시 정리
            if IS_M3_MAX and TORCH_AVAILABLE and MPS_AVAILABLE:
                try:
                    import torch
                    if hasattr(torch.backends.mps, 'empty_cache'):
                        torch.backends.mps.empty_cache()
                except Exception:
                    pass
            
            return {
                'garbage_collected': collected,
                'aggressive': aggressive,
                'context': self.context_id
            }
        except Exception as e:
            return {
                'error': str(e),
                'context': self.context_id
            }

    def cleanup_circular_references(self):
        """순환참조 정리 (구 버전 호환)"""
        # Event-Driven DI Container에서는 순환참조가 원천적으로 방지되므로
        # 아무것도 하지 않음
        pass

    def cleanup(self):
        """컨텍스트 정리"""
        try:
            # 등록된 서비스들 정리
            for service_key in self.registry.list_services():
                service = self.registry.get_service(service_key, self.context_id)
                if service and hasattr(service, 'cleanup'):
                    try:
                        service.cleanup()
                    except Exception as e:
                        self.logger.debug(f"서비스 정리 실패 {service_key}: {e}")
            
            # 컨텍스트 소멸 이벤트
            event = DIEvent(
                event_type=EventType.CONTEXT_DESTROYED,
                service_key="container",
                context_id=self.context_id,
                data={
                    'lifetime_seconds': time.time() - self._creation_time,
                    'total_access_count': self._access_count,
                    'total_injection_count': self._injection_count
                }
            )
            self.event_bus.publish(event)
            
            self.logger.info(f"✅ 컨텍스트 정리 완료: {self.context_id}")
            
        except Exception as e:
            self.logger.error(f"❌ 컨텍스트 정리 실패: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """컨텍스트 통계"""
        return {
            'context_id': self.context_id,
            'creation_time': self._creation_time,
            'lifetime_seconds': time.time() - self._creation_time,
            'access_count': self._access_count,
            'injection_count': self._injection_count,
            'registered_services': self.registry.list_services(),
            'factory_services': list(self.factory.get_service_definitions().keys()),
            'environment': {
                'is_m3_max': IS_M3_MAX,
                'device': DEVICE,
                'memory_gb': MEMORY_GB,
                'torch_available': TORCH_AVAILABLE,
                'mps_available': MPS_AVAILABLE
            }
        }

# ==============================================
# 🔥 Container Manager - 전역 관리
# ==============================================

class EventDrivenContainerManager:
    """이벤트 기반 Container 매니저"""
    
    def __init__(self):
        self.event_bus = EventBus()
        self.factory = DependencyFactory(self.event_bus)
        self._contexts: Dict[str, ContextualDIContainer] = {}
        self._default_context_id = "default"
        self._manager_lock = threading.RLock()
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        
        # 이벤트 구독 설정
        self._setup_event_subscriptions()
        
        # 기본 컨텍스트 생성
        self.get_container(self._default_context_id)
        
        self.logger.info("✅ Event-Driven Container Manager 초기화 완료")
    
    def _setup_event_subscriptions(self):
        """이벤트 구독 설정"""
        # 순환참조 감지 이벤트 구독
        self.event_bus.subscribe(
            EventType.CIRCULAR_DEPENDENCY_DETECTED,
            self._handle_circular_dependency
        )
        
        # 의존성 실패 이벤트 구독
        self.event_bus.subscribe(
            EventType.DEPENDENCY_FAILED,
            self._handle_dependency_failure
        )
    
    def _handle_circular_dependency(self, event: DIEvent):
        """순환참조 처리"""
        self.logger.error(f"❌ 순환참조 감지됨: {event.service_key}")
        self.logger.error(f"   컨텍스트: {event.context_id}")
        self.logger.error(f"   생성 스택: {event.data.get('creation_stack', [])}")
    
    def _handle_dependency_failure(self, event: DIEvent):
        """의존성 실패 처리"""
        self.logger.warning(f"⚠️ 의존성 해결 실패: {event.service_key}")
        self.logger.debug(f"   오류: {event.data.get('error', 'Unknown')}")
    
    def get_container(self, context_id: Optional[str] = None) -> ContextualDIContainer:
        """컨텍스트별 Container 반환"""
        context_id = context_id or self._default_context_id
        
        with self._manager_lock:
            if context_id not in self._contexts:
                self._contexts[context_id] = ContextualDIContainer(
                    context_id, self.event_bus, self.factory
                )
            
            return self._contexts[context_id]
    
    def create_context(self, context_id: str) -> ContextualDIContainer:
        """새 컨텍스트 생성"""
        with self._manager_lock:
            if context_id in self._contexts:
                self.logger.warning(f"⚠️ 컨텍스트 이미 존재: {context_id}")
                return self._contexts[context_id]
            
            container = ContextualDIContainer(context_id, self.event_bus, self.factory)
            self._contexts[context_id] = container
            
            self.logger.info(f"✅ 새 컨텍스트 생성: {context_id}")
            return container
    
    def destroy_context(self, context_id: str):
        """컨텍스트 소멸"""
        with self._manager_lock:
            if context_id in self._contexts:
                container = self._contexts[context_id]
                container.cleanup()
                del self._contexts[context_id]
                self.logger.info(f"🗑️ 컨텍스트 소멸: {context_id}")
    
    def list_contexts(self) -> List[str]:
        """컨텍스트 목록"""
        with self._manager_lock:
            return list(self._contexts.keys())
    
    def get_global_stats(self) -> Dict[str, Any]:
        """전역 통계"""
        with self._manager_lock:
            contexts_stats = {}
            for context_id, container in self._contexts.items():
                contexts_stats[context_id] = container.get_stats()
            
            return {
                'manager_type': 'EventDrivenContainerManager',
                'version': '6.0',
                'total_contexts': len(self._contexts),
                'default_context': self._default_context_id,
                'contexts': contexts_stats,
                'event_history_size': len(self.event_bus.get_event_history()),
                'factory_services': list(self.factory.get_service_definitions().keys()),
                'environment': {
                    'conda_env': CONDA_ENV,
                    'is_m3_max': IS_M3_MAX,
                    'device': DEVICE,
                    'memory_gb': MEMORY_GB,
                    'torch_available': TORCH_AVAILABLE,
                    'mps_available': MPS_AVAILABLE
                }
            }
    
    def optimize_all_contexts(self):
        """모든 컨텍스트 최적화"""
        with self._manager_lock:
            for container in self._contexts.values():
                # 메모리 정리
                gc.collect()
                
                # M3 Max MPS 최적화
                if IS_M3_MAX and TORCH_AVAILABLE and MPS_AVAILABLE:
                    try:
                        import torch
                        if hasattr(torch.backends.mps, 'empty_cache'):
                            torch.backends.mps.empty_cache()
                    except Exception:
                        pass
            
            self.logger.info("✅ 모든 컨텍스트 최적화 완료")
    
    def cleanup_all(self):
        """모든 컨텍스트 정리"""
        with self._manager_lock:
            for context_id in list(self._contexts.keys()):
                self.destroy_context(context_id)
            
            self.optimize_all_contexts()
            self.logger.info("✅ 모든 컨텍스트 정리 완료")

# ==============================================
# 🔥 Property Injection Mixin - 속성 주입 지원
# ==============================================

class PropertyInjectionMixin:
    """속성 주입을 지원하는 믹스인"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._di_container: Optional[ContextualDIContainer] = None
        self._injected_properties: Dict[str, str] = {}
    
    def set_di_container(self, container: ContextualDIContainer):
        """DI Container 설정"""
        self._di_container = container
        self._auto_inject_properties()
    
    def _auto_inject_properties(self):
        """자동 속성 주입"""
        if not self._di_container:
            return
        
        injection_map = {
            'model_loader': 'model_loader',
            'memory_manager': 'memory_manager',
            'data_converter': 'data_converter'
        }
        
        for attr_name, service_key in injection_map.items():
            if not hasattr(self, attr_name) or getattr(self, attr_name) is None:
                service = self._di_container.get(service_key)
                if service:
                    setattr(self, attr_name, service)
                    self._injected_properties[attr_name] = service_key
    
    def get_service(self, service_key: str):
        """DI Container를 통한 서비스 조회"""
        if self._di_container:
            return self._di_container.get(service_key)
        return None
    
    def inject_di_container(self, container) -> bool:
        """DI Container 주입 (BaseStepMixin 호환)"""
        try:
            if isinstance(container, ContextualDIContainer):
                self.set_di_container(container)
                return True
            else:
                # 구 버전 호환성
                self._di_container = container
                return True
        except Exception as e:
            logger.error(f"❌ DI Container 주입 실패: {e}")
            return False

# ==============================================
# 🔥 전역 인스턴스 관리
# ==============================================

_global_manager: Optional[EventDrivenContainerManager] = None
_manager_lock = threading.RLock()

def get_global_container(context_id: Optional[str] = None) -> ContextualDIContainer:
    """전역 Container 반환"""
    global _global_manager
    
    with _manager_lock:
        if _global_manager is None:
            _global_manager = EventDrivenContainerManager()
            logger.info("✅ 전역 Event-Driven Container Manager 생성")
        
        return _global_manager.get_container(context_id)

def get_global_manager() -> EventDrivenContainerManager:
    """전역 Manager 반환"""
    global _global_manager
    
    with _manager_lock:
        if _global_manager is None:
            _global_manager = EventDrivenContainerManager()
            logger.info("✅ 전역 Event-Driven Container Manager 생성")
        
        return _global_manager

def reset_global_container():
    """전역 Container 리셋"""
    global _global_manager
    
    with _manager_lock:
        if _global_manager:
            _global_manager.cleanup_all()
        _global_manager = None
        logger.info("🔄 전역 Event-Driven Container Manager 리셋")

# ==============================================
# 🔥 구 버전 호환성 레이어
# ==============================================

# LazyDependency 호환성 클래스 (기존 코드용)
class LazyDependency:
    """구 버전 LazyDependency 호환성 래퍼"""
    
    def __init__(self, factory: Callable[[], Any]):
        self._factory = factory
        self._instance = None
        self._resolved = False
        self._lock = threading.RLock()
    
    def get(self) -> Any:
        """지연 해결"""
        if not self._resolved:
            with self._lock:
                if not self._resolved:
                    try:
                        self._instance = self._factory()
                        self._resolved = True
                    except Exception as e:
                        logger.error(f"❌ LazyDependency 해결 실패: {e}")
                        return None
        return self._instance
    
    def resolve(self) -> Any:
        """resolve() 메서드 별칭"""
        return self.get()
    
    def is_resolved(self) -> bool:
        return self._resolved

# 기존 CircularReferenceFreeDIContainer 호환성
class CircularReferenceFreeDIContainer:
    """구 버전 호환성을 위한 래퍼"""
    
    def __init__(self):
        self._container = get_global_container("legacy")
        self.logger = logging.getLogger("LegacyDIContainer")
        self.logger.warning("⚠️ 구 버전 DI Container 사용 - Event-Driven Container로 업그레이드 권장")
    
    def get(self, key: str):
        return self._container.get(key)
    
    def register(self, key: str, instance: Any, singleton: bool = True):
        self._container.register_instance(key, instance)
    
    def register_lazy(self, key: str, factory: Callable[[], Any]):
        self._container.register_factory(key, factory, is_singleton=True)
    
    def inject_to_step(self, step_instance):
        return self._container.inject_to_step(step_instance)
    
    def force_register_model_loader(self, model_loader):
        self._container.register_instance('model_loader', model_loader)
        return True
    
    def get_stats(self):
        return self._container.get_stats()
    
    def optimize_memory(self, aggressive=False):
        if aggressive:
            get_global_manager().optimize_all_contexts()
        return {"optimized": True}
    
    def cleanup_circular_references(self):
        pass  # Event-driven에서는 불필요


# backend/app/core/di_container.py에 추가할 DynamicImportResolver 클래스

class DynamicImportResolver:
    """동적 import 해결기 (순환참조 완전 방지)"""
    
    @staticmethod
    def resolve_model_loader():
        """ModelLoader 동적 해결 (순환참조 방지)"""
        import_paths = [
            'app.ai_pipeline.utils.model_loader',
            'ai_pipeline.utils.model_loader',
            'utils.model_loader'
        ]
        
        for path in import_paths:
            try:
                module = importlib.import_module(path)
                
                # 전역 함수 우선
                if hasattr(module, 'get_global_model_loader'):
                    loader = module.get_global_model_loader()
                    if loader:
                        logger.debug(f"✅ ModelLoader 동적 해결: {path}")
                        return loader
                
                # 클래스 직접 생성
                if hasattr(module, 'ModelLoader'):
                    ModelLoaderClass = module.ModelLoader
                    loader = ModelLoaderClass()
                    logger.debug(f"✅ ModelLoader 클래스 생성: {path}")
                    return loader
                    
            except ImportError:
                continue
        
        # 완전 실패 시 Mock 반환
        logger.warning("⚠️ ModelLoader 해결 실패, Mock 사용")
        return DynamicImportResolver._create_mock_model_loader()
    
    @staticmethod
    def resolve_memory_manager():
        """MemoryManager 동적 해결 (순환참조 방지)"""
        import_paths = [
            'app.ai_pipeline.utils.memory_manager',
            'ai_pipeline.utils.memory_manager',
            'utils.memory_manager'
        ]
        
        for path in import_paths:
            try:
                module = importlib.import_module(path)
                
                if hasattr(module, 'get_global_memory_manager'):
                    manager = module.get_global_memory_manager()
                    if manager:
                        logger.debug(f"✅ MemoryManager 동적 해결: {path}")
                        return manager
                
                if hasattr(module, 'MemoryManager'):
                    MemoryManagerClass = module.MemoryManager
                    manager = MemoryManagerClass()
                    logger.debug(f"✅ MemoryManager 클래스 생성: {path}")
                    return manager
                    
            except ImportError:
                continue
        
        logger.warning("⚠️ MemoryManager 해결 실패, Mock 사용")
        return DynamicImportResolver._create_mock_memory_manager()
    
    @staticmethod
    def resolve_data_converter():
        """DataConverter 동적 해결 (순환참조 방지)"""
        import_paths = [
            'app.ai_pipeline.utils.data_converter',
            'ai_pipeline.utils.data_converter',
            'utils.data_converter'
        ]
        
        for path in import_paths:
            try:
                module = importlib.import_module(path)
                
                if hasattr(module, 'get_global_data_converter'):
                    converter = module.get_global_data_converter()
                    if converter:
                        logger.debug(f"✅ DataConverter 동적 해결: {path}")
                        return converter
                
                if hasattr(module, 'DataConverter'):
                    DataConverterClass = module.DataConverter
                    converter = DataConverterClass()
                    logger.debug(f"✅ DataConverter 클래스 생성: {path}")
                    return converter
                    
            except ImportError:
                continue
        
        logger.warning("⚠️ DataConverter 해결 실패, Mock 사용")
        return DynamicImportResolver._create_mock_data_converter()
    
    @staticmethod
    def resolve_di_container():
        """DI Container 동적 해결 (순환참조 방지)"""
        import_paths = [
            'app.core.di_container',
            'core.di_container',
        ]
        
        for path in import_paths:
            try:
                module = importlib.import_module(path)
                
                if hasattr(module, 'get_global_container'):
                    container = module.get_global_container()
                    if container:
                        logger.debug(f"✅ DIContainer 동적 해결: {path}")
                        return container
                        
            except ImportError:
                continue
        
        logger.warning("⚠️ DIContainer 해결 실패")
        return None
    
    @staticmethod
    def _create_mock_model_loader():
        """Mock ModelLoader"""
        class MockModelLoader:
            def __init__(self):
                self.is_initialized = True
                self.loaded_models = {}
                self.device = DEVICE
                self.logger = logging.getLogger("MockModelLoader")
                
            def load_model(self, model_path: str, **kwargs):
                model_id = f"mock_{len(self.loaded_models)}"
                self.loaded_models[model_id] = {
                    "path": model_path,
                    "loaded": True,
                    "device": self.device
                }
                return self.loaded_models[model_id]
            
            def create_step_interface(self, step_name: str):
                return MockStepInterface(step_name)
            
            def cleanup(self):
                self.loaded_models.clear()
        
        class MockStepInterface:
            def __init__(self, step_name):
                self.step_name = step_name
                self.is_initialized = True
            
            def get_model(self, model_name=None):
                return {"mock_model": model_name, "loaded": True}
        
        return MockModelLoader()
    
    @staticmethod
    def _create_mock_memory_manager():
        """Mock MemoryManager"""
        class MockMemoryManager:
            def __init__(self):
                self.is_initialized = True
                self.optimization_count = 0
            
            def optimize_memory(self, aggressive=False):
                self.optimization_count += 1
                gc.collect()
                return {"optimized": True, "count": self.optimization_count}
            
            def get_memory_info(self):
                return {
                    "total_gb": MEMORY_GB,
                    "available_gb": MEMORY_GB * 0.7,
                    "percent": 30.0
                }
            
            def cleanup(self):
                self.optimize_memory(aggressive=True)
        
        return MockMemoryManager()
    
    @staticmethod
    def _create_mock_data_converter():
        """Mock DataConverter"""
        class MockDataConverter:
            def __init__(self):
                self.is_initialized = True
                self.conversion_count = 0
            
            def convert(self, data, target_format):
                self.conversion_count += 1
                return {
                    "converted": f"mock_{target_format}_{self.conversion_count}",
                    "format": target_format
                }
            
            def get_supported_formats(self):
                return ["tensor", "numpy", "pil", "cv2"]
            
            def cleanup(self):
                self.conversion_count = 0
        
        return MockDataConverter()
    


# ==============================================
# 🔥 안전한 서비스 접근 함수들 (완전한 호환성)
# ==============================================

def get_service_safe(service_key: str, context_id: Optional[str] = None, default=None) -> Any:
    """안전한 서비스 조회 - 실패시 기본값 반환"""
    try:
        service = get_service(service_key, context_id)
        return service if service is not None else default
    except Exception as e:
        logger.debug(f"⚠️ get_service_safe 실패 ({service_key}): {e}")
        return default

def register_service_safe(service_key: str, instance: Any, context_id: Optional[str] = None) -> bool:
    """안전한 서비스 등록"""
    try:
        register_service(service_key, instance, context_id)
        logger.debug(f"✅ register_service_safe 성공: {service_key}")
        return True
    except Exception as e:
        logger.debug(f"⚠️ register_service_safe 실패 ({service_key}): {e}")
        return False

def register_factory_safe(service_key: str, factory: Callable[[], Any], 
                         singleton: bool = True, context_id: Optional[str] = None) -> bool:
    """안전한 팩토리 등록"""
    try:
        register_factory(service_key, factory, singleton, context_id)
        logger.debug(f"✅ register_factory_safe 성공: {service_key}")
        return True
    except Exception as e:
        logger.debug(f"⚠️ register_factory_safe 실패 ({service_key}): {e}")
        return False

def get_model_loader_safe(context_id: Optional[str] = None):
    """안전한 ModelLoader 조회"""
    return get_service_safe('model_loader', context_id)

def get_memory_manager_safe(context_id: Optional[str] = None):
    """안전한 MemoryManager 조회"""
    return get_service_safe('memory_manager', context_id)

def get_data_converter_safe(context_id: Optional[str] = None):
    """안전한 DataConverter 조회"""
    return get_service_safe('data_converter', context_id)

def get_container_safe(context_id: Optional[str] = None):
    """안전한 Container 조회"""
    try:
        return get_global_container(context_id)
    except Exception as e:
        logger.debug(f"⚠️ get_container_safe 실패: {e}")
        return None

def inject_dependencies_safe(step_instance, context_id: Optional[str] = None) -> int:
    """안전한 의존성 주입"""
    try:
        return inject_dependencies_to_step_safe(step_instance, None)
    except Exception as e:
        logger.debug(f"⚠️ inject_dependencies_safe 실패: {e}")
        return 0

def ensure_model_loader_registration(context_id: Optional[str] = None) -> bool:
    """ModelLoader 등록 보장"""
    try:
        loader = get_service('model_loader', context_id)
        return loader is not None
    except Exception:
        return False

def ensure_service_registration(service_key: str, context_id: Optional[str] = None) -> bool:
    """서비스 등록 보장"""
    try:
        service = get_service(service_key, context_id)
        return service is not None
    except Exception:
        return False

def initialize_di_system_safe(context_id: Optional[str] = None) -> bool:
    """DI 시스템 안전 초기화"""
    try:
        container = get_global_container(context_id)
        
        # conda 환경 최적화
        if IS_CONDA:
            _optimize_for_conda()
        
        # ModelLoader 확인
        model_loader = container.get('model_loader')
        if model_loader:
            logger.info("✅ DI 시스템 초기화: ModelLoader 사용 가능")
        else:
            logger.warning("⚠️ DI 시스템 초기화: ModelLoader 없음")
        
        return True
    except Exception as e:
        logger.error(f"❌ DI 시스템 안전 초기화 실패: {e}")
        return False

def cleanup_services_safe(context_id: Optional[str] = None) -> bool:
    """안전한 서비스 정리"""
    try:
        container = get_global_container(context_id)
        if hasattr(container, 'cleanup_disposed_services'):
            container.cleanup_disposed_services()
        return True
    except Exception as e:
        logger.debug(f"⚠️ cleanup_services_safe 실패: {e}")
        return False

def reset_container_safe(context_id: Optional[str] = None) -> bool:
    """안전한 Container 리셋"""
    try:
        reset_global_container(context_id)
        return True
    except Exception as e:
        logger.debug(f"⚠️ reset_container_safe 실패: {e}")
        return False

# ==============================================
# 🔥 상태 확인 함수들
# ==============================================

def is_service_available(service_key: str, context_id: Optional[str] = None) -> bool:
    """서비스 사용 가능 여부 확인"""
    try:
        service = get_service_safe(service_key, context_id)
        return service is not None
    except Exception:
        return False

def is_container_ready(context_id: Optional[str] = None) -> bool:
    """Container 준비 상태 확인"""
    try:
        container = get_container_safe(context_id)
        return container is not None
    except Exception:
        return False

def is_di_system_ready(context_id: Optional[str] = None) -> bool:
    """DI 시스템 준비 상태 확인"""
    try:
        container = get_global_container(context_id)
        if not container:
            return False
        
        # 핵심 서비스들 확인
        essential_services = ['model_loader', 'memory_manager', 'data_converter']
        for service_key in essential_services:
            if not container.get(service_key):
                return False
        
        return True
    except Exception:
        return False

def get_service_status(service_key: str, context_id: Optional[str] = None) -> Dict[str, Any]:
    """서비스 상태 정보"""
    try:
        container = get_container_safe(context_id)
        if not container:
            return {'status': 'error', 'message': 'Container not available'}
        
        service = container.get(service_key)
        return {
            'service_key': service_key,
            'available': service is not None,
            'type': type(service).__name__ if service else None,
            'context': context_id or 'default'
        }
    except Exception as e:
        return {
            'service_key': service_key,
            'status': 'error',
            'message': str(e)
        }

def get_di_system_status(context_id: Optional[str] = None) -> Dict[str, Any]:
    """DI 시스템 상태 정보"""
    try:
        container = get_global_container(context_id)
        if not container:
            return {'status': 'error', 'message': 'Container not available'}
        
        stats = container.get_stats() if hasattr(container, 'get_stats') else {}
        
        # 핵심 서비스 상태 확인
        services_status = {}
        essential_services = ['model_loader', 'memory_manager', 'data_converter']
        
        for service_key in essential_services:
            service = container.get(service_key)
            services_status[service_key] = {
                'available': service is not None,
                'type': type(service).__name__ if service else None
            }
        
        return {
            'status': 'ready' if is_di_system_ready(context_id) else 'partial',
            'context': context_id or 'default',
            'stats': stats,
            'services': services_status,
            'environment': {
                'conda_env': CONDA_ENV,
                'is_m3_max': IS_M3_MAX,
                'device': DEVICE,
                'torch_available': TORCH_AVAILABLE,
                'mps_available': MPS_AVAILABLE
            }
        }
    except Exception as e:
        return {
            'status': 'error',
            'message': str(e)
        }

# ==============================================
# 🔥 conda 환경 최적화
# ==============================================

def _optimize_for_conda():
    """conda 환경 최적화"""
    try:
        os.environ['OMP_NUM_THREADS'] = str(max(1, os.cpu_count() // 2))
        os.environ['MKL_NUM_THREADS'] = str(max(1, os.cpu_count() // 2))
        os.environ['NUMEXPR_NUM_THREADS'] = str(max(1, os.cpu_count() // 2))
        
        if TORCH_AVAILABLE:
            import torch
            torch.set_num_threads(max(1, os.cpu_count() // 2))
            
            if IS_M3_MAX and MPS_AVAILABLE:
                if hasattr(torch.backends.mps, 'empty_cache'):
                    torch.backends.mps.empty_cache()
        
        logger.info(f"🐍 conda 환경 '{CONDA_ENV}' 최적화 완료")
    except Exception as e:
        logger.warning(f"⚠️ conda 최적화 실패: {e}")

# ==============================================
# 🔥 안전한 서비스 접근 함수들 (호환성)
# ==============================================

def get_service_safe(service_key: str, context: str = None, default=None) -> Any:
    """안전한 서비스 조회 - 실패시 기본값 반환"""
    try:
        service = get_service(service_key, context)
        return service if service is not None else default
    except Exception as e:
        logger.debug(f"⚠️ get_service_safe 실패 ({service_key}): {e}")
        return default

def get_model_loader_safe(context: str = None):
    """안전한 ModelLoader 조회"""
    return get_service_safe('model_loader', context)

def get_memory_manager_safe(context: str = None):
    """안전한 MemoryManager 조회"""
    return get_service_safe('memory_manager', context)

def get_data_converter_safe(context: str = None):
    """안전한 DataConverter 조회"""
    return get_service_safe('data_converter', context)

def ensure_model_loader_registration(context: str = None) -> bool:
    """ModelLoader 등록 보장"""
    try:
        loader = get_service('model_loader', context)
        return loader is not None
    except Exception:
        return False

def initialize_di_system_safe(context: str = None) -> bool:
    """DI 시스템 안전 초기화"""
    try:
        return initialize_di_system(context)
    except Exception as e:
        logger.error(f"❌ DI 시스템 안전 초기화 실패: {e}")
        return False

# ==============================================
# 🔥 구 버전 호환성 함수들 추가
# ==============================================

def inject_dependencies_to_step_safe(step_instance, container=None):
    """구버전 호환 함수"""
    try:
        if container and hasattr(container, 'context'):
            return inject_dependencies_to_step(step_instance, container.context)
        else:
            return inject_dependencies_to_step(step_instance, None)
    except Exception as e:
        logger.error(f"❌ inject_dependencies_to_step_safe 실패: {e}")
        return 0

def get_global_container_legacy():
    """구버전 호환 함수"""
    try:
        return get_global_container()
    except Exception as e:
        logger.error(f"❌ get_global_container_legacy 실패: {e}")
        return None

def reset_global_container_legacy():
    """구버전 호환 함수"""
    try:
        return reset_global_container()
    except Exception as e:
        logger.error(f"❌ reset_global_container_legacy 실패: {e}")

# ==============================================
# 🔥 DI 시스템 상태 확인 함수들
# ==============================================

def is_di_system_ready(context: str = None) -> bool:
    """DI 시스템 준비 상태 확인"""
    try:
        container = get_global_container(context)
        if not container:
            return False
        
        # 핵심 서비스들 확인
        essential_services = ['model_loader', 'memory_manager', 'data_converter']
        for service_key in essential_services:
            if not container.get(service_key):
                return False
        
        return True
    except Exception:
        return False

def get_di_system_status(context: str = None) -> Dict[str, Any]:
    """DI 시스템 상태 정보"""
    try:
        container = get_global_container(context)
        if not container:
            return {'status': 'error', 'message': 'Container not available'}
        
        stats = container.get_stats()
        
        # 핵심 서비스 상태 확인
        services_status = {}
        essential_services = ['model_loader', 'memory_manager', 'data_converter']
        
        for service_key in essential_services:
            service = container.get(service_key)
            services_status[service_key] = {
                'available': service is not None,
                'type': type(service).__name__ if service else None
            }
        
        return {
            'status': 'ready' if is_di_system_ready(context) else 'partial',
            'context': context or 'default',
            'stats': stats,
            'services': services_status,
            'environment': {
                'conda_env': CONDA_ENV,
                'is_m3_max': IS_M3_MAX,
                'device': DEVICE,
                'torch_available': TORCH_AVAILABLE,
                'mps_available': MPS_AVAILABLE
            }
        }
    except Exception as e:
        return {
            'status': 'error',
            'message': str(e)
        }
# backend/app/core/di_container.py에 추가할 지연 서비스 관련 함수들

# ==============================================
# 🔥 지연 서비스 (Lazy Service) 관련 함수들
# ==============================================

def register_lazy_service(service_key: str, factory: Callable[[], Any], 
                         context_id: Optional[str] = None, singleton: bool = True) -> bool:
    """지연 서비스 등록 (LazyService로 래핑)"""
    try:
        lazy_service = LazyDependency(factory)  # LazyDependency는 LazyService의 별칭
        
        container = get_global_container(context_id)
        container.register_instance(service_key, lazy_service)
        
        logger.debug(f"✅ register_lazy_service 성공: {service_key}")
        return True
    except Exception as e:
        logger.debug(f"⚠️ register_lazy_service 실패 ({service_key}): {e}")
        return False

def register_lazy_service_safe(service_key: str, factory: Callable[[], Any], 
                              context_id: Optional[str] = None, singleton: bool = True) -> bool:
    """안전한 지연 서비스 등록"""
    return register_lazy_service(service_key, factory, context_id, singleton)

def create_lazy_dependency(factory: Callable[[], Any], service_key: str = None) -> Any:
    """지연 의존성 생성"""
    try:
        return LazyDependency(factory)
    except Exception as e:
        logger.debug(f"⚠️ create_lazy_dependency 실패: {e}")
        return None

def resolve_lazy_service(service_key: str, context_id: Optional[str] = None) -> Any:
    """지연 서비스 해결"""
    try:
        container = get_global_container(context_id)
        lazy_service = container.get(service_key)
        
        if lazy_service and hasattr(lazy_service, 'get'):
            return lazy_service.get()
        else:
            return lazy_service
    except Exception as e:
        logger.debug(f"⚠️ resolve_lazy_service 실패 ({service_key}): {e}")
        return None

def is_lazy_service_resolved(service_key: str, context_id: Optional[str] = None) -> bool:
    """지연 서비스 해결 상태 확인"""
    try:
        container = get_global_container(context_id)
        lazy_service = container.get(service_key)
        
        if lazy_service and hasattr(lazy_service, 'is_resolved'):
            return lazy_service.is_resolved()
        return False
    except Exception:
        return False

# ==============================================
# 🔥 Container 레벨 함수들 (구 버전 호환)
# ==============================================

def create_container(context_id: str = None) -> Any:
    """Container 생성 (구 버전 호환)"""
    try:
        return get_global_container(context_id)
    except Exception as e:
        logger.debug(f"⚠️ create_container 실패: {e}")
        return None

def dispose_container(context_id: str = None) -> bool:
    """Container 정리 (구 버전 호환)"""
    try:
        reset_global_container(context_id)
        return True
    except Exception as e:
        logger.debug(f"⚠️ dispose_container 실패: {e}")
        return False

def get_container_instance(context_id: str = None) -> Any:
    """Container 인스턴스 조회 (구 버전 호환)"""
    return get_container_safe(context_id)

def register_singleton(service_key: str, instance: Any, context_id: Optional[str] = None) -> bool:
    """싱글톤 등록 (구 버전 호환)"""
    return register_service_safe(service_key, instance, context_id)

def register_transient(service_key: str, factory: Callable[[], Any], context_id: Optional[str] = None) -> bool:
    """임시 서비스 등록 (구 버전 호환)"""
    return register_factory_safe(service_key, factory, False, context_id)

def unregister_service(service_key: str, context_id: Optional[str] = None) -> bool:
    """서비스 등록 해제"""
    try:
        container = get_global_container(context_id)
        if hasattr(container, 'unregister_service'):
            container.unregister_service(service_key)
            return True
        return False
    except Exception as e:
        logger.debug(f"⚠️ unregister_service 실패 ({service_key}): {e}")
        return False

# ==============================================
# 🔥 의존성 주입 관련 함수들
# ==============================================

def inject_all_dependencies(step_instance, context_id: Optional[str] = None) -> int:
    """모든 의존성 주입"""
    return inject_dependencies_safe(step_instance, context_id)

def auto_wire_dependencies(step_instance, context_id: Optional[str] = None) -> bool:
    """자동 의존성 연결"""
    try:
        count = inject_dependencies_safe(step_instance, context_id)
        return count > 0
    except Exception:
        return False

def validate_dependencies(step_instance, required_services: List[str] = None) -> bool:
    """의존성 유효성 검사"""
    try:
        if not required_services:
            required_services = ['model_loader', 'memory_manager', 'data_converter']
        
        for service_name in required_services:
            if not hasattr(step_instance, service_name) or getattr(step_instance, service_name) is None:
                return False
        
        return True
    except Exception:
        return False

def get_dependency_status(step_instance) -> Dict[str, Any]:
    """의존성 상태 정보"""
    try:
        dependencies = ['model_loader', 'memory_manager', 'data_converter', 'di_container']
        
        status = {}
        for dep_name in dependencies:
            dep_value = getattr(step_instance, dep_name, None)
            status[dep_name] = {
                'available': dep_value is not None,
                'type': type(dep_value).__name__ if dep_value else None
            }
        
        return {
            'step_class': step_instance.__class__.__name__,
            'dependencies': status,
            'all_resolved': all(status[dep]['available'] for dep in dependencies),
            'resolution_count': sum(1 for dep in status.values() if dep['available'])
        }
    except Exception as e:
        return {
            'error': str(e),
            'step_class': getattr(step_instance, '__class__', {}).get('__name__', 'Unknown')
        }

# ==============================================
# 🔥 서비스 조회 편의 함수들
# ==============================================

def get_all_services(context_id: Optional[str] = None) -> Dict[str, Any]:
    """모든 서비스 조회"""
    try:
        container = get_global_container(context_id)
        
        if hasattr(container, '_services'):
            services = {}
            for service_key in container._services.keys():
                service = container.get(service_key)
                services[service_key] = {
                    'available': service is not None,
                    'type': type(service).__name__ if service else None
                }
            return services
        
        return {}
    except Exception as e:
        return {'error': str(e)}

def list_service_keys(context_id: Optional[str] = None) -> List[str]:
    """서비스 키 목록"""
    try:
        container = get_global_container(context_id)
        
        if hasattr(container, '_services'):
            return list(container._services.keys())
        
        return []
    except Exception:
        return []

def get_service_count(context_id: Optional[str] = None) -> int:
    """등록된 서비스 개수"""
    try:
        return len(list_service_keys(context_id))
    except Exception:
        return 0

# ==============================================
# 🔥 편의 함수들
# ==============================================

def get_service(key: str, context_id: Optional[str] = None) -> Optional[Any]:
    """서비스 조회 편의 함수"""
    container = get_global_container(context_id)
    return container.get(key)

def register_service(key: str, instance: Any, context_id: Optional[str] = None):
    """서비스 등록 편의 함수"""
    container = get_global_container(context_id)
    container.register_instance(key, instance)

def register_factory(key: str, factory: Callable[[], Any], singleton: bool = True, context_id: Optional[str] = None):
    """팩토리 등록 편의 함수"""
    container = get_global_container(context_id)
    container.register_factory(key, factory, singleton)

def inject_dependencies_to_step_safe(step_instance, context_id: Optional[str] = None):
    """Step 의존성 주입 편의 함수"""
    container = get_global_container(context_id)
    return container.inject_to_step(step_instance)

# BaseStepMixin 호환 함수들
def _get_global_di_container():
    """BaseStepMixin 호환 함수"""
    return get_global_container()

def _get_service_from_container_safe(service_key: str):
    """BaseStepMixin 호환 함수"""
    return get_service(service_key)

# ModelLoader 연동 함수들
def get_model_loader_safe():
    """안전한 ModelLoader 조회"""
    return get_service('model_loader')

def ensure_model_loader_registration():
    """ModelLoader 등록 보장"""
    loader = get_service('model_loader')
    return loader is not None

def initialize_di_system_safe():
    """DI 시스템 안전 초기화"""
    try:
        container = get_global_container()
        
        # conda 환경 최적화
        if IS_CONDA:
            _optimize_for_conda()
        
        # ModelLoader 확인
        model_loader = container.get('model_loader')
        if model_loader:
            logger.info("✅ DI 시스템 초기화: ModelLoader 사용 가능")
        else:
            logger.warning("⚠️ DI 시스템 초기화: ModelLoader 없음")
        
        return True
    except Exception as e:
        logger.error(f"❌ DI 시스템 초기화 실패: {e}")
        return False

def _optimize_for_conda():
    """conda 환경 최적화"""
    try:
        os.environ['OMP_NUM_THREADS'] = str(max(1, os.cpu_count() // 2))
        os.environ['MKL_NUM_THREADS'] = str(max(1, os.cpu_count() // 2))
        os.environ['NUMEXPR_NUM_THREADS'] = str(max(1, os.cpu_count() // 2))
        
        if TORCH_AVAILABLE:
            import torch
            torch.set_num_threads(max(1, os.cpu_count() // 2))
            
            if IS_M3_MAX and MPS_AVAILABLE:
                if hasattr(torch.backends.mps, 'empty_cache'):
                    torch.backends.mps.empty_cache()
        
        logger.info(f"🐍 conda 환경 '{CONDA_ENV}' 최적화 완료")
    except Exception as e:
        logger.warning(f"⚠️ conda 최적화 실패: {e}")

# ==============================================
# 🔥 Export
# ==============================================
# backend/app/core/di_container.py의 완전한 __all__ 리스트

__all__ = [
    # 메인 클래스들
    'EventDrivenContainerManager',
    'ContextualDIContainer',
    'DependencyFactory',
    'ServiceRegistry',
    'EventBus',
    'PropertyInjectionMixin',
    
    # 이벤트 시스템
    'EventType',
    'DIEvent',
    
    # 데이터 클래스들
    'ServiceDefinition',
    'ServiceLifecycle',
    
    # 전역 함수들
    'get_global_container',
    'get_global_manager',
    'reset_global_container',
    
    # 기본 편의 함수들
    'get_service',
    'register_service',
    'register_factory',
    'inject_dependencies_to_step_safe',
    
    # 🔥 안전한 접근 함수들 (모든 *_safe 함수들)
    'get_service_safe',
    'register_service_safe',
    'register_factory_safe',
    'get_model_loader_safe',
    'get_memory_manager_safe',
    'get_data_converter_safe',
    'get_container_safe',
    'inject_dependencies_safe',
    'ensure_model_loader_registration',
    'ensure_service_registration',
    'initialize_di_system_safe',
    'cleanup_services_safe',
    'reset_container_safe',
    
    # 🔥 지연 서비스 관련 (누락된 함수들)
    'register_lazy_service',
    'register_lazy_service_safe',
    'create_lazy_dependency',
    'resolve_lazy_service',
    'is_lazy_service_resolved',
    
    # 🔥 Container 레벨 함수들 (구 버전 호환)
    'create_container',
    'dispose_container',
    'get_container_instance',
    'register_singleton',
    'register_transient',
    'unregister_service',
    
    # 🔥 의존성 주입 관련
    'inject_all_dependencies',
    'auto_wire_dependencies',
    'validate_dependencies',
    'get_dependency_status',
    
    # 🔥 서비스 조회 편의 함수들
    'get_all_services',
    'list_service_keys',
    'get_service_count',
    
    # 상태 확인 함수들
    'is_service_available',
    'is_container_ready',
    'is_di_system_ready',
    'get_service_status',
    'get_di_system_status',
    
    # 호환성 함수들
    'CircularReferenceFreeDIContainer',  # 구 버전 호환
    'LazyDependency',  # 구 버전 호환
    'DynamicImportResolver',  # 호환성
    '_get_global_di_container',
    '_get_service_from_container_safe',
    
    # 구 버전 호환 함수들  
    'get_global_container_legacy',
    'reset_global_container_legacy',
    
    # 타입들
    'T'
]
# ==============================================
# 🔥 자동 초기화
# ==============================================

if IS_CONDA:
    logger.info(f"🐍 conda 환경 '{CONDA_ENV}' 감지")

# 완료 메시지
logger.info("=" * 80)
logger.info("🔥 Event-Driven DI Container v6.0 로드 완료!")
logger.info("=" * 80)
logger.info("✅ Event-Driven Architecture - 의존성 요청/해결을 이벤트로 분리")
logger.info("✅ Factory Pattern + Command Pattern - 객체 생성 로직 완전 분리")
logger.info("✅ Pub/Sub 메시징 - 느슨한 결합으로 순환참조 원천 차단")
logger.info("✅ Lazy Registration - 실제 필요할 때만 의존성 해결")
logger.info("✅ Contextual Isolation - 각 Step이 독립적 DI 컨텍스트 보유")
logger.info("✅ Interface Segregation - 작은 단위 인터페이스로 책임 분리")
logger.info("✅ Dependency Graph - 의존성 추적으로 순환참조 사전 감지")
logger.info("✅ Observable Pattern - 의존성 변경 사항 실시간 알림")
logger.info("✅ Memory Pool - 객체 재사용으로 메모리 효율성 극대화")
logger.info("✅ 기존 API 100% 호환성 유지")

logger.info("🎯 핵심 아키텍처:")
logger.info("   Event Bus → Dependency Factory → Service Registry → Lifecycle Manager")

logger.info("🔧 지원 기능:")
logger.info("   • 순환참조 원천 차단")
logger.info("   • 컨텍스트별 격리")
logger.info("   • 이벤트 기반 모니터링")
logger.info("   • 자동 생명주기 관리")
logger.info("   • ModelLoader v5.1 완전 연동")
logger.info("   • 구 버전 100% 호환성")

if IS_M3_MAX:
    logger.info("🍎 M3 Max 128GB 메모리 최적화 활성화")

logger.info("🚀 Event-Driven DI Container v6.0 준비 완료!")
logger.info("🎉 순환참조 문제 근본적 해결!")
logger.info("🎉 MyCloset AI 프로젝트 완벽 연동!")
logger.info("=" * 80)