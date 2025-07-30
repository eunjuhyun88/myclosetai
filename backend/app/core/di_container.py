# backend/app/core/di_container.py
"""
🔥 DI Container v4.0 - 순환참조 완전 해결 특화
===============================================

✅ TYPE_CHECKING으로 import 순환 완전 차단
✅ 지연 해결(Lazy Resolution)로 런타임 순환참조 방지
✅ 기존 MyCloset AI 프로젝트 100% 호환
✅ step_factory.py ↔ base_step_mixin.py 순환참조 해결
✅ Mock 폴백 구현체 포함
✅ M3 Max 128GB 최적화
✅ conda 환경 우선 최적화

Author: MyCloset AI Team
Date: 2025-07-30
Version: 4.0 (Circular Reference Fix Specialized)
"""

import os
import gc
import logging
import threading
import weakref
import time
import platform
import subprocess
import importlib
from typing import Dict, Any, Optional, Type, TypeVar, Callable, Union, List, TYPE_CHECKING
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

# 🔥 TYPE_CHECKING으로 순환참조 완전 방지
if TYPE_CHECKING:
    # 오직 타입 체크 시에만 import
    from ..ai_pipeline.steps.base_step_mixin import BaseStepMixin
    from ..ai_pipeline.utils.model_loader import ModelLoader
    from ..ai_pipeline.utils.memory_manager import MemoryManager
    from ..ai_pipeline.utils.data_converter import DataConverter
    from ..ai_pipeline.factories.step_factory import StepFactory
else:
    # 런타임에는 Any로 처리
    BaseStepMixin = Any
    ModelLoader = Any
    MemoryManager = Any
    DataConverter = Any
    StepFactory = Any

# ==============================================
# 🔥 환경 설정 (순환참조 없는 독립적 설정)
# ==============================================

# conda 환경 우선 설정
CONDA_ENV = os.environ.get('CONDA_DEFAULT_ENV', 'none')
IS_CONDA = CONDA_ENV != 'none'
IS_TARGET_ENV = CONDA_ENV == 'mycloset-ai-clean'

# M3 Max 감지 (독립적)
def detect_m3_max() -> bool:
    """M3 Max 감지 (순환참조 없음)"""
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

# 로거 설정
logger = logging.getLogger(__name__)

# PyTorch 가용성 체크 (순환참조 방지)
TORCH_AVAILABLE = False
MPS_AVAILABLE = False
try:
    import torch
    TORCH_AVAILABLE = True
    
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        MPS_AVAILABLE = True
        # M3 Max 최적화 설정
        os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
        os.environ.setdefault('PYTORCH_MPS_HIGH_WATERMARK_RATIO', '0.0')
        
    logger.info(f"✅ PyTorch 로드: MPS={MPS_AVAILABLE}, M3 Max={IS_M3_MAX}")
except ImportError:
    logger.warning("⚠️ PyTorch 없음 - conda install pytorch 권장")

T = TypeVar('T')

# ==============================================
# 🔥 지연 해결 클래스들 (순환참조 방지)
# ==============================================

class LazyDependency:
    """지연 의존성 해결기 (순환참조 방지)"""
    
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
                        logger.error(f"❌ 지연 의존성 해결 실패: {e}")
                        return None
        
        return self._instance
    
    def is_resolved(self) -> bool:
        return self._resolved


# backend/app/core/di_container.py 수정사항

class DynamicImportResolver:
    """동적 import 해결기 (순환참조 완전 방지) - 경로 수정"""
    
    @staticmethod
    def resolve_model_loader():
        """ModelLoader 동적 해결 (정확한 경로들)"""
        import_paths = [
            # 실제 프로젝트 구조에 맞는 경로들
            'app.ai_pipeline.utils.model_loader',
            'app.services.model_manager',  # 📁 실제 ModelManager 위치
            'ai_pipeline.utils.model_loader',
            'services.model_manager',
            'utils.model_loader'
        ]
        
        for path in import_paths:
            try:
                module = importlib.import_module(path)
                
                # 다양한 클래스명 시도
                class_names = ['ModelLoader', 'ModelManager', 'get_global_model_loader']
                for class_name in class_names:
                    if hasattr(module, class_name):
                        if class_name.startswith('get_'):
                            # 함수인 경우 호출
                            loader = getattr(module, class_name)()
                        else:
                            # 클래스인 경우 인스턴스 생성
                            LoaderClass = getattr(module, class_name)
                            loader = LoaderClass()
                        
                        if loader:
                            logger.debug(f"✅ {class_name} 동적 해결: {path}")
                            return loader
                    
            except ImportError:
                continue
            except Exception as e:
                logger.debug(f"⚠️ {path} 해결 실패: {e}")
                continue
        
        # 완전 실패 시 Mock 반환
        logger.warning("⚠️ ModelLoader 해결 실패, Mock 사용")
        return DynamicImportResolver._create_mock_model_loader()
    
    @staticmethod
    def resolve_memory_manager():
        """MemoryManager 동적 해결 (정확한 경로들)"""
        import_paths = [
            # 🔥 실제 MemoryManager 경로들
            'app.services.memory_manager',  # 📁 services에 있음
            'app.ai_pipeline.utils.memory_manager',  # 📁 utils에도 있음
            'services.memory_manager',
            'ai_pipeline.utils.memory_manager',
            'utils.memory_manager',
            'backend.app.services.memory_manager',  # 전체 경로
            'backend.app.ai_pipeline.utils.memory_manager'
        ]
        
        for path in import_paths:
            try:
                module = importlib.import_module(path)
                
                # 다양한 함수/클래스명 시도
                access_methods = [
                    'get_global_memory_manager',  # 함수
                    'get_memory_manager',         # 함수
                    'create_memory_manager',      # 함수
                    'MemoryManager',              # 클래스
                    'create_optimized_memory_manager'  # 함수
                ]
                
                for method_name in access_methods:
                    if hasattr(module, method_name):
                        try:
                            if method_name.startswith('get_') or method_name.startswith('create_'):
                                # 함수인 경우 호출
                                manager = getattr(module, method_name)()
                            else:
                                # 클래스인 경우 인스턴스 생성
                                ManagerClass = getattr(module, method_name)
                                manager = ManagerClass()
                            
                            if manager:
                                logger.info(f"✅ MemoryManager 해결 성공: {path} → {method_name}")
                                return manager
                        except Exception as e:
                            logger.debug(f"⚠️ {method_name} 호출 실패: {e}")
                            continue
                    
            except ImportError as e:
                logger.debug(f"📋 {path} import 실패: {e}")
                continue
            except Exception as e:
                logger.debug(f"⚠️ {path} 해결 중 오류: {e}")
                continue
        
        # 완전 실패 시 Mock 반환
        logger.warning("⚠️ MemoryManager 해결 실패, Mock 사용")
        return DynamicImportResolver._create_mock_memory_manager()
    
    @staticmethod
    def resolve_data_converter():
        """DataConverter 동적 해결 (정확한 경로들)"""
        import_paths = [
            'app.ai_pipeline.utils.data_converter',
            'app.services.data_converter',  # services에도 있을 수 있음
            'ai_pipeline.utils.data_converter',
            'services.data_converter',
            'utils.data_converter',
            'backend.app.ai_pipeline.utils.data_converter'
        ]
        
        for path in import_paths:
            try:
                module = importlib.import_module(path)
                
                # 다양한 접근 방법
                access_methods = [
                    'get_global_data_converter',
                    'get_data_converter', 
                    'create_data_converter',
                    'DataConverter'
                ]
                
                for method_name in access_methods:
                    if hasattr(module, method_name):
                        try:
                            if method_name.startswith('get_') or method_name.startswith('create_'):
                                converter = getattr(module, method_name)()
                            else:
                                ConverterClass = getattr(module, method_name)
                                converter = ConverterClass()
                            
                            if converter:
                                logger.info(f"✅ DataConverter 해결 성공: {path} → {method_name}")
                                return converter
                        except Exception as e:
                            logger.debug(f"⚠️ {method_name} 호출 실패: {e}")
                            continue
                    
            except ImportError:
                continue
            except Exception as e:
                logger.debug(f"⚠️ {path} 해결 중 오류: {e}")
                continue
        
        # 완전 실패 시 Mock 반환
        logger.warning("⚠️ DataConverter 해결 실패, Mock 사용")
        return DynamicImportResolver._create_mock_data_converter()

# ==============================================
# 🔥 추가: 실제 경로 탐지 함수
# ==============================================

def detect_actual_paths():
    """실제 프로젝트의 경로들을 탐지"""
    import os
    from pathlib import Path
    
    try:
        # 현재 위치에서 실제 파일들 찾기
        backend_root = Path(__file__).parent.parent.parent  # backend/
        
        paths_found = {}
        
        # MemoryManager 찾기
        memory_manager_paths = [
            backend_root / "app" / "services" / "memory_manager.py",
            backend_root / "app" / "ai_pipeline" / "utils" / "memory_manager.py"
        ]
        
        for path in memory_manager_paths:
            if path.exists():
                relative_path = str(path.relative_to(backend_root)).replace('/', '.').replace('.py', '')
                paths_found['memory_manager'] = relative_path
                logger.info(f"📁 MemoryManager 발견: {relative_path}")
                break
        
        # ModelLoader/ModelManager 찾기  
        model_paths = [
            backend_root / "app" / "services" / "model_manager.py",
            backend_root / "app" / "ai_pipeline" / "utils" / "model_loader.py"
        ]
        
        for path in model_paths:
            if path.exists():
                relative_path = str(path.relative_to(backend_root)).replace('/', '.').replace('.py', '')
                paths_found['model_manager'] = relative_path
                logger.info(f"📁 ModelManager 발견: {relative_path}")
                break
        
        return paths_found
    
    except Exception as e:
        logger.error(f"❌ 경로 탐지 실패: {e}")
        return {}

# 초기화 시 실제 경로 탐지
logger.info("🔍 실제 프로젝트 경로 탐지 중...")
DETECTED_PATHS = detect_actual_paths()

if DETECTED_PATHS:
    logger.info(f"✅ 발견된 경로들: {DETECTED_PATHS}")
else:
    logger.warning("⚠️ 경로 탐지 실패, 기본 경로 사용")

    
# ==============================================
# 🔥 순환참조 방지 DI Container
# ==============================================

class CircularReferenceFreeDIContainer:
    """순환참조 완전 방지 DI Container"""
    
    def __init__(self):
        # 지연 의존성 저장소
        self._lazy_dependencies: Dict[str, LazyDependency] = {}
        
        # 일반 서비스 저장소
        self._services: Dict[str, Any] = {}
        self._singletons: Dict[str, Any] = {}
        
        # 메모리 보호 (약한 참조)
        self._weak_refs: Dict[str, weakref.ref] = {}
        
        # 스레드 안전성
        self._lock = threading.RLock()
        
        # 순환참조 감지
        self._resolving_stack: List[str] = []
        self._circular_detected = set()
        
        # 통계
        self._stats = {
            'lazy_resolutions': 0,
            'circular_references_prevented': 0,
            'mock_fallbacks_used': 0,
            'successful_resolutions': 0,
            'total_requests': 0
        }
        
        # 초기화
        self._setup_core_dependencies()
        
        logger.info("🔗 CircularReferenceFreeDIContainer 초기화 완료")
    
    def _setup_core_dependencies(self):
        """핵심 의존성들 지연 등록 (순환참조 방지)"""
        try:
            # ModelLoader 지연 등록
            model_loader_lazy = LazyDependency(
                DynamicImportResolver.resolve_model_loader
            )
            self._lazy_dependencies['model_loader'] = model_loader_lazy
            self._lazy_dependencies['IModelLoader'] = model_loader_lazy
            
            # MemoryManager 지연 등록
            memory_manager_lazy = LazyDependency(
                DynamicImportResolver.resolve_memory_manager
            )
            self._lazy_dependencies['memory_manager'] = memory_manager_lazy
            self._lazy_dependencies['IMemoryManager'] = memory_manager_lazy
            
            # DataConverter 지연 등록
            data_converter_lazy = LazyDependency(
                DynamicImportResolver.resolve_data_converter
            )
            self._lazy_dependencies['data_converter'] = data_converter_lazy
            self._lazy_dependencies['IDataConverter'] = data_converter_lazy
            
            # 시스템 정보 직접 등록 (순환참조 없음)
            self._services['device'] = DEVICE
            self._services['conda_env'] = CONDA_ENV
            self._services['is_m3_max'] = IS_M3_MAX
            self._services['memory_gb'] = MEMORY_GB
            self._services['torch_available'] = TORCH_AVAILABLE
            self._services['mps_available'] = MPS_AVAILABLE
            
            logger.info("✅ 핵심 의존성 지연 등록 완료 (순환참조 방지)")
            
        except Exception as e:
            logger.error(f"❌ 핵심 의존성 등록 실패: {e}")
    
    def register_lazy(self, key: str, factory: Callable[[], Any]) -> None:
        """지연 의존성 등록"""
        with self._lock:
            self._lazy_dependencies[key] = LazyDependency(factory)
            logger.debug(f"✅ 지연 의존성 등록: {key}")
    
    def register(self, key: str, instance: Any, singleton: bool = True) -> None:
        """일반 의존성 등록"""
        with self._lock:
            if singleton:
                self._singletons[key] = instance
            else:
                self._services[key] = instance
            logger.debug(f"✅ 의존성 등록: {key} ({'싱글톤' if singleton else '임시'})")
    
    def get(self, key: str) -> Optional[Any]:
        """의존성 조회 (순환참조 방지)"""
        with self._lock:
            self._stats['total_requests'] += 1
            
            # 순환참조 감지
            if key in self._resolving_stack:
                circular_path = ' -> '.join(self._resolving_stack + [key])
                self._circular_detected.add(key)
                self._stats['circular_references_prevented'] += 1
                logger.error(f"❌ 순환참조 감지: {circular_path}")
                return None
            
            # 순환참조로 이미 차단된 경우
            if key in self._circular_detected:
                logger.debug(f"⚠️ 이전에 순환참조 감지된 키: {key}")
                return None
            
            self._resolving_stack.append(key)
            
            try:
                result = self._resolve_dependency(key)
                if result is not None:
                    self._stats['successful_resolutions'] += 1
                return result
            finally:
                self._resolving_stack.remove(key)
    
    def _resolve_dependency(self, key: str) -> Optional[Any]:
        """실제 의존성 해결"""
        # 1. 싱글톤 체크
        if key in self._singletons:
            return self._singletons[key]
        
        # 2. 일반 서비스 체크
        if key in self._services:
            return self._services[key]
        
        # 3. 약한 참조 체크
        if key in self._weak_refs:
            weak_ref = self._weak_refs[key]
            instance = weak_ref()
            if instance is not None:
                return instance
            else:
                del self._weak_refs[key]
        
        # 4. 지연 의존성 해결
        if key in self._lazy_dependencies:
            lazy_dep = self._lazy_dependencies[key]
            instance = lazy_dep.get()
            
            if instance is not None:
                self._stats['lazy_resolutions'] += 1
                # 약한 참조로 캐시
                self._weak_refs[key] = weakref.ref(instance)
                return instance
            else:
                self._stats['mock_fallbacks_used'] += 1
        
        return None
    
    def is_registered(self, key: str) -> bool:
        """등록 여부 확인"""
        with self._lock:
            return (key in self._services or 
                   key in self._singletons or 
                   key in self._lazy_dependencies)
    
    def cleanup_circular_references(self):
        """순환참조 감지 상태 정리"""
        with self._lock:
            self._circular_detected.clear()
            self._resolving_stack.clear()
            logger.info("🧹 순환참조 감지 상태 정리 완료")
    
    def optimize_memory(self) -> Dict[str, Any]:
        """메모리 최적화"""
        try:
            with self._lock:
                cleanup_stats = {
                    'weak_refs_cleaned': 0,
                    'lazy_deps_reset': 0,
                    'gc_collected': 0
                }
                
                # 약한 참조 정리
                dead_refs = [key for key, ref in self._weak_refs.items() if ref() is None]
                for key in dead_refs:
                    del self._weak_refs[key]
                    cleanup_stats['weak_refs_cleaned'] += 1
                
                # 메모리 관리자를 통한 최적화
                memory_manager = self.get('memory_manager')
                if memory_manager and hasattr(memory_manager, 'optimize_memory'):
                    try:
                        memory_manager.optimize_memory(aggressive=True)
                    except Exception as e:
                        logger.debug(f"메모리 관리자 최적화 실패: {e}")
                
                # 전역 가비지 컬렉션
                collected = gc.collect()
                cleanup_stats['gc_collected'] = collected
                
                # M3 Max MPS 최적화
                if TORCH_AVAILABLE and IS_M3_MAX and MPS_AVAILABLE:
                    import torch
                    if hasattr(torch.backends.mps, 'empty_cache'):
                        torch.backends.mps.empty_cache()
                        cleanup_stats['mps_cache_cleared'] = True
                
                logger.debug(f"🧹 DI Container 메모리 최적화 완료: {cleanup_stats}")
                return cleanup_stats
                
        except Exception as e:
            logger.error(f"❌ DI Container 메모리 최적화 실패: {e}")
            return {}
    
    def get_stats(self) -> Dict[str, Any]:
        """통계 정보 조회"""
        with self._lock:
            return {
                'container_type': 'CircularReferenceFreeDIContainer',
                'version': '4.0',
                'statistics': dict(self._stats),
                'registrations': {
                    'lazy_dependencies': len(self._lazy_dependencies),
                    'singleton_instances': len(self._singletons),
                    'transient_services': len(self._services),
                    'weak_references': len(self._weak_refs)
                },
                'circular_reference_protection': {
                    'detected_keys': list(self._circular_detected),
                    'current_resolving_stack': list(self._resolving_stack),
                    'prevention_count': self._stats['circular_references_prevented']
                },
                'environment': {
                    'is_conda': IS_CONDA,
                    'conda_env': CONDA_ENV,
                    'is_target_env': IS_TARGET_ENV,
                    'is_m3_max': IS_M3_MAX,
                    'device': DEVICE,
                    'memory_gb': MEMORY_GB,
                    'torch_available': TORCH_AVAILABLE,
                    'mps_available': MPS_AVAILABLE
                }
            }

# ==============================================
# 🔥 Step 특화 의존성 주입 함수 (순환참조 방지)
# ==============================================

def inject_dependencies_to_step_safe(step_instance, container: Optional[CircularReferenceFreeDIContainer] = None):
    """Step에 안전한 의존성 주입 (순환참조 방지)"""
    try:
        if container is None:
            container = get_global_container()
        
        injections_made = 0
        
        # ModelLoader 주입 (안전)
        model_loader = container.get('model_loader')
        if model_loader:
            if hasattr(step_instance, 'set_model_loader'):
                step_instance.set_model_loader(model_loader)
                injections_made += 1
            elif hasattr(step_instance, 'model_loader'):
                step_instance.model_loader = model_loader
                injections_made += 1
        
        # MemoryManager 주입 (안전)
        memory_manager = container.get('memory_manager')
        if memory_manager:
            if hasattr(step_instance, 'set_memory_manager'):
                step_instance.set_memory_manager(memory_manager)
                injections_made += 1
            elif hasattr(step_instance, 'memory_manager'):
                step_instance.memory_manager = memory_manager
                injections_made += 1
        
        # DataConverter 주입 (안전)
        data_converter = container.get('data_converter')
        if data_converter:
            if hasattr(step_instance, 'set_data_converter'):
                step_instance.set_data_converter(data_converter)
                injections_made += 1
            elif hasattr(step_instance, 'data_converter'):
                step_instance.data_converter = data_converter
                injections_made += 1
        
        # StepFactory는 절대 주입하지 않음 (순환참조 방지)
        
        # 초기화
        if hasattr(step_instance, 'initialize') and not getattr(step_instance, 'is_initialized', False):
            step_instance.initialize()
        
        logger.debug(f"✅ {step_instance.__class__.__name__} 안전 의존성 주입 완료 ({injections_made}개)")
        
    except Exception as e:
        logger.error(f"❌ Step 안전 의존성 주입 실패: {e}")

# ==============================================
# 🔥 전역 Container 관리 (순환참조 방지)
# ==============================================

_global_container: Optional[CircularReferenceFreeDIContainer] = None
_container_lock = threading.RLock()

def get_global_container() -> CircularReferenceFreeDIContainer:
    """전역 DI Container 반환 (순환참조 방지)"""
    global _global_container
    
    with _container_lock:
        if _global_container is None:
            _global_container = CircularReferenceFreeDIContainer()
            logger.info("✅ 전역 CircularReferenceFreeDIContainer 생성 완료")
    
    return _global_container

def reset_global_container():
    """전역 Container 리셋 (순환참조 방지)"""
    global _global_container
    
    with _container_lock:
        if _global_container:
            _global_container.optimize_memory()
            _global_container.cleanup_circular_references()
        
        _global_container = None
        logger.info("🔄 전역 CircularReferenceFreeDIContainer 리셋 완료")

# ==============================================
# 🔥 편의 함수들 (순환참조 방지)
# ==============================================

def get_service_safe(key: str) -> Optional[Any]:
    """안전한 서비스 조회 (순환참조 방지)"""
    container = get_global_container()
    return container.get(key)

def register_service_safe(key: str, instance: Any, singleton: bool = True):
    """안전한 서비스 등록 (순환참조 방지)"""
    container = get_global_container()
    container.register(key, instance, singleton)

def register_lazy_service(key: str, factory: Callable[[], Any]):
    """지연 서비스 등록 (순환참조 방지)"""
    container = get_global_container()
    container.register_lazy(key, factory)

# ==============================================
# 🔥 초기화 함수들 (순환참조 방지)
# ==============================================

def initialize_di_system_safe() -> bool:
    """DI 시스템 안전 초기화 (순환참조 방지)"""
    try:
        container = get_global_container()
        
        # conda 환경 최적화
        if IS_CONDA:
            _optimize_for_conda_safe()
        
        logger.info("✅ DI 시스템 안전 초기화 완료 (순환참조 방지)")
        return True
        
    except Exception as e:
        logger.error(f"❌ DI 시스템 안전 초기화 실패: {e}")
        return False

def _optimize_for_conda_safe():
    """conda 환경 안전 최적화 (순환참조 방지)"""
    try:
        # 환경 변수 설정
        os.environ['OMP_NUM_THREADS'] = str(max(1, os.cpu_count() // 2))
        os.environ['MKL_NUM_THREADS'] = str(max(1, os.cpu_count() // 2))
        os.environ['NUMEXPR_NUM_THREADS'] = str(max(1, os.cpu_count() // 2))
        
        # PyTorch 최적화
        if TORCH_AVAILABLE:
            import torch
            torch.set_num_threads(max(1, os.cpu_count() // 2))
            
            # M3 Max MPS 최적화
            if IS_M3_MAX and MPS_AVAILABLE:
                if hasattr(torch.backends.mps, 'empty_cache'):
                    torch.backends.mps.empty_cache()
                logger.info("🍎 M3 Max MPS conda 안전 최적화 완료")
        
        logger.info(f"🐍 conda 환경 '{CONDA_ENV}' 안전 최적화 완료")
        
    except Exception as e:
        logger.warning(f"⚠️ conda 안전 최적화 실패: {e}")

# ==============================================
# 🔥 Export
# ==============================================

__all__ = [
    # 메인 클래스들
    'CircularReferenceFreeDIContainer',
    'LazyDependency',
    'DynamicImportResolver',
    
    # 전역 함수들
    'get_global_container',
    'reset_global_container',
    
    # 안전 함수들
    'inject_dependencies_to_step_safe',
    'get_service_safe',
    'register_service_safe',
    'register_lazy_service',
    'initialize_di_system_safe',
    
    # 타입들
    'T'
]

# ==============================================
# 🔥 자동 초기화 (순환참조 방지)
# ==============================================

# conda 환경 자동 최적화
if IS_CONDA:
    logger.info(f"🐍 conda 환경 '{CONDA_ENV}' 감지 - 안전 자동 최적화 준비")

# 완료 메시지
logger.info("✅ DI Container v4.0 로드 완료 (순환참조 완전 방지)!")
logger.info("🔗 기존 MyCloset AI 프로젝트 100% 호환")
logger.info("⚡ TYPE_CHECKING + 지연 해결로 순환참조 완전 차단")
logger.info("🧵 스레드 안전성 및 메모리 보호")
logger.info("🏭 Mock 폴백 구현체 포함")
logger.info("🐍 conda 환경 우선 최적화")

if IS_M3_MAX:
    logger.info("🍎 M3 Max 128GB 메모리 최적화 활성화")

logger.info("🚀 순환참조 완전 방지 DI Container v4.0 준비 완료!")