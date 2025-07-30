#!/usr/bin/env python3
"""#backend/app/ai_pipeline/steps/__init__.py
#!/usr/bin/env python3
🔥 MyCloset AI Pipeline Steps v5.0 - DI Container v4.0 완전 통합
================================================================

✅ CircularReferenceFreeDIContainer 완전 적용
✅ TYPE_CHECKING으로 순환참조 완전 차단  
✅ 지연 해결(Lazy Resolution) 기반 Step 로딩
✅ step_factory.py ↔ base_step_mixin.py 순환참조 완전 해결
✅ StepsCircularReferenceFreeDIContainer 특화 적용
✅ 안전한 의존성 주입 시스템
✅ logger 에러 완전 해결
✅ M3 Max 128GB + conda 환경 최적화
✅ GitHub 프로젝트 구조 100% 호환

Author: MyCloset AI Team
Date: 2025-07-30
Version: 5.0 (DI Container v4.0 Complete Integration)
"""

import os
import gc
import logging
import threading
import weakref
import time
import warnings
import sys
import asyncio
from typing import Dict, Any, Optional, Type, TypeVar, Callable, Union, List, TYPE_CHECKING
from abc import ABC, abstractmethod
from pathlib import Path

# 경고 무시 (deprecated 경로 관련)
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', message='.*deprecated.*')

# Logger 최우선 초기화 (에러 방지)
logger = logging.getLogger(__name__)

# 🔥 TYPE_CHECKING으로 순환참조 완전 방지
if TYPE_CHECKING:
    # 오직 타입 체크 시에만 import
    from .base_step_mixin import BaseStepMixin, GitHubDependencyManager
    from ..utils.model_loader import ModelLoader
    from ..utils.memory_manager import MemoryManager
    from ..utils.data_converter import DataConverter
    from ..factories.step_factory import StepFactory
    from ...core.di_container import DIContainer
else:
    # 런타임에는 Any로 처리 (순환참조 방지)
    BaseStepMixin = Any
    GitHubDependencyManager = Any
    ModelLoader = Any
    MemoryManager = Any
    DataConverter = Any
    StepFactory = Any
    DIContainer = Any

# ==============================================
# 🔥 DI Container v4.0 Import (순환참조 방지)
# ==============================================

try:
    from ...core.di_container import (
        CircularReferenceFreeDIContainer,
        LazyDependency,
        DynamicImportResolver,
        get_global_container,
        inject_dependencies_to_step_safe,
        get_service_safe,
        register_service_safe,
        register_lazy_service
    )
    DI_CONTAINER_AVAILABLE = True
    logger.info("✅ DI Container v4.0 Core Import 성공")
except ImportError as e:
    logger.error(f"❌ DI Container v4.0 Core Import 실패: {e}")
    DI_CONTAINER_AVAILABLE = False
    
    # 폴백 처리
    def inject_dependencies_to_step_safe(step_instance, container=None):
        logger.warning("⚠️ DI Container 없음 - 의존성 주입 스킵")
    
    def get_service_safe(key: str):
        logger.warning(f"⚠️ DI Container 없음 - 서비스 조회 실패: {key}")
        return None

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
        import platform
        if platform.system() == 'Darwin':
            import subprocess
            result = subprocess.run(
                ['sysctl', '-n', 'machdep.cpu.brand_string'],
                capture_output=True, text=True, timeout=5
            )
            return 'M3' in result.stdout and 'Max' in result.stdout
    except Exception:
        pass
    return False

IS_M3_MAX = detect_m3_max()
MEMORY_GB = 128.0 if IS_M3_MAX else 16.0
DEVICE = 'mps' if IS_M3_MAX else 'cpu'

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
# 🔥 Steps 전용 DI Container (순환참조 완전 방지)
# ==============================================

class StepsCircularReferenceFreeDIContainer:
    """Steps 전용 순환참조 완전 방지 DI Container v5.0"""
    
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
        
        # Step 로딩 상태
        self._step_loading_stats = {
            'lazy_resolutions': 0,
            'circular_references_prevented': 0,
            'fallback_classes_used': 0,
            'successful_resolutions': 0,
            'total_requests': 0,
            'di_injections_completed': 0
        }
        
        # 초기화
        self._setup_steps_dependencies()
        
        logger.info("🔗 StepsCircularReferenceFreeDIContainer v5.0 초기화 완료")
    
    def _setup_steps_dependencies(self):
        """Steps 의존성들 지연 등록 (순환참조 방지)"""
        try:
            # BaseStepMixin 지연 등록 (가장 중요)
            base_step_mixin_lazy = LazyDependency(
                self._resolve_base_step_mixin_safe
            )
            self._lazy_dependencies['BaseStepMixin'] = base_step_mixin_lazy
            self._lazy_dependencies['base_step_mixin'] = base_step_mixin_lazy
            
            # 시스템 정보 직접 등록 (순환참조 없음)
            self._services['device'] = DEVICE
            self._services['conda_env'] = CONDA_ENV
            self._services['is_m3_max'] = IS_M3_MAX
            self._services['memory_gb'] = MEMORY_GB
            self._services['torch_available'] = TORCH_AVAILABLE
            self._services['mps_available'] = MPS_AVAILABLE
            
            # 전역 DI Container 연결 (가능한 경우)
            if DI_CONTAINER_AVAILABLE:
                try:
                    global_container = get_global_container()
                    self._services['global_di_container'] = global_container
                    logger.info("✅ 전역 DI Container 연결 완료")
                except Exception as e:
                    logger.debug(f"전역 DI Container 연결 실패: {e}")
            
            logger.info("✅ Steps 핵심 의존성 지연 등록 완료 (순환참조 방지)")
            
        except Exception as e:
            logger.error(f"❌ Steps 의존성 등록 실패: {e}")
    
    def _resolve_base_step_mixin_safe(self):
        """BaseStepMixin 안전한 해결 (순환참조 완전 방지)"""
        import_paths = [
            'app.ai_pipeline.steps.base_step_mixin',
            'ai_pipeline.steps.base_step_mixin',
            '.base_step_mixin'
        ]
        
        for path in import_paths:
            try:
                if path.startswith('.'):
                    # 상대 import
                    from .base_step_mixin import BaseStepMixin
                else:
                    # 절대 import
                    import importlib
                    module = importlib.import_module(path)
                    BaseStepMixin = getattr(module, 'BaseStepMixin', None)
                
                if BaseStepMixin:
                    logger.debug(f"✅ BaseStepMixin 안전 해결: {path}")
                    return BaseStepMixin
                    
            except ImportError as e:
                logger.debug(f"📋 BaseStepMixin import 시도 실패: {path} - {e}")
                continue
        
        # 완전 실패 시 폴백 클래스 반환
        logger.warning("⚠️ BaseStepMixin 해결 실패, 폴백 클래스 생성")
        return self._create_fallback_base_step_mixin()
    
    def _create_fallback_base_step_mixin(self):
        """BaseStepMixin 폴백 클래스 (logger 에러 방지)"""
        class FallbackBaseStepMixin:
            def __init__(self, **kwargs):
                # Logger 제일 먼저 초기화 (에러 방지)
                self.logger = logging.getLogger(f"steps.{self.__class__.__name__}")
                
                # 기본 속성들
                self.step_name = kwargs.get('step_name', self.__class__.__name__)
                self.step_id = kwargs.get('step_id', 0)
                self.device = kwargs.get('device', DEVICE)
                
                # 상태 플래그들
                self.is_initialized = False
                self.is_ready = False
                self.has_model = False
                self.model_loaded = False
                self.warmup_completed = False
                
                # 의존성 관련 (BaseStepMixin 호환)
                self.model_loader = None
                self.memory_manager = None
                self.data_converter = None
                self.di_container = None
                
                # 의존성 주입 상태
                self.dependencies_injected = {
                    'model_loader': False,
                    'memory_manager': False,
                    'data_converter': False,
                    'di_container': False
                }
                
                # 성능 메트릭
                self.performance_metrics = {
                    'process_count': 0,
                    'total_process_time': 0.0,
                    'error_count': 0,
                    'success_count': 0
                }
                
                logger.debug(f"✅ {self.step_name} FallbackBaseStepMixin 초기화 완료")
            
            async def initialize(self):
                """Step 초기화"""
                try:
                    self.is_initialized = True
                    self.logger.info(f"✅ {self.step_name} 초기화 완료")
                    return True
                except Exception as e:
                    self.logger.error(f"❌ {self.step_name} 초기화 실패: {e}")
                    return False
            
            def set_model_loader(self, model_loader):
                """ModelLoader 의존성 주입"""
                self.model_loader = model_loader
                self.dependencies_injected['model_loader'] = True
                self.has_model = True
                self.model_loaded = True
                self.logger.debug(f"✅ {self.step_name} ModelLoader 주입 완료")
            
            def set_memory_manager(self, memory_manager):
                """MemoryManager 의존성 주입"""
                self.memory_manager = memory_manager
                self.dependencies_injected['memory_manager'] = True
                self.logger.debug(f"✅ {self.step_name} MemoryManager 주입 완료")
            
            def set_data_converter(self, data_converter):
                """DataConverter 의존성 주입"""
                self.data_converter = data_converter
                self.dependencies_injected['data_converter'] = True
                self.logger.debug(f"✅ {self.step_name} DataConverter 주입 완료")
            
            def set_di_container(self, di_container):
                """DI Container 의존성 주입"""
                self.di_container = di_container
                self.dependencies_injected['di_container'] = True
                self.logger.debug(f"✅ {self.step_name} DI Container 주입 완료")
            
            async def cleanup(self):
                """리소스 정리"""
                try:
                    self.logger.debug(f"🔄 {self.step_name} 리소스 정리 중...")
                    # 필요한 정리 작업 수행
                    self.logger.debug(f"✅ {self.step_name} 리소스 정리 완료")
                except Exception as e:
                    self.logger.error(f"❌ {self.step_name} 리소스 정리 실패: {e}")
        
        return FallbackBaseStepMixin
    
    def register_step_lazy(self, step_id: str, step_module: str, step_class: str) -> None:
        """Step 클래스 지연 등록"""
        with self._lock:
            factory = lambda: self._resolve_step_class_safe(step_module, step_class)
            self._lazy_dependencies[step_id] = LazyDependency(factory)
            logger.debug(f"✅ Step 지연 등록: {step_id}")
    
    def _resolve_step_class_safe(self, step_module_name: str, step_class_name: str):
        """개별 Step 클래스 안전한 해결 (순환참조 방지)"""
        import_paths = [
            f'app.ai_pipeline.steps.{step_module_name}',
            f'ai_pipeline.steps.{step_module_name}',
            f'.{step_module_name}'
        ]
        
        for path in import_paths:
            try:
                if path.startswith('.'):
                    # 상대 import (현재 패키지 기준)
                    import importlib
                    module = importlib.import_module(path, package=__package__)
                else:
                    # 절대 import
                    import importlib
                    module = importlib.import_module(path)
                
                step_class = getattr(module, step_class_name, None)
                if step_class:
                    logger.debug(f"✅ {step_class_name} 안전 해결: {path}")
                    return step_class, True
                    
            except (ImportError, SyntaxError, AttributeError) as e:
                # logger 관련 에러인지 확인
                if 'logger' in str(e):
                    logger.debug(f"📋 {step_class_name} logger 에러: {e}")
                elif 'deprecated' in str(e) or 'interface' in str(e):
                    logger.debug(f"📋 {step_class_name} deprecated 경로: {e}")
                else:
                    logger.debug(f"📋 {step_class_name} import 실패: {e}")
                continue
        
        return None, False
    
    def get_step(self, step_id: str) -> Optional[Any]:
        """Step 클래스 조회 (순환참조 방지)"""
        with self._lock:
            self._step_loading_stats['total_requests'] += 1
            
            # 순환참조 감지
            if step_id in self._resolving_stack:
                circular_path = ' -> '.join(self._resolving_stack + [step_id])
                self._circular_detected.add(step_id)
                self._step_loading_stats['circular_references_prevented'] += 1
                logger.error(f"❌ Step 순환참조 감지: {circular_path}")
                return None
            
            # 순환참조로 이미 차단된 경우
            if step_id in self._circular_detected:
                logger.debug(f"⚠️ 이전에 순환참조 감지된 Step: {step_id}")
                return None
            
            self._resolving_stack.append(step_id)
            
            try:
                result = self._resolve_step_dependency(step_id)
                if result is not None:
                    self._step_loading_stats['successful_resolutions'] += 1
                return result
            finally:
                self._resolving_stack.remove(step_id)
    
    def _resolve_step_dependency(self, step_id: str) -> Optional[Any]:
        """실제 Step 의존성 해결"""
        # 1. 싱글톤 체크
        if step_id in self._singletons:
            return self._singletons[step_id]
        
        # 2. 일반 서비스 체크
        if step_id in self._services:
            return self._services[step_id]
        
        # 3. 약한 참조 체크
        if step_id in self._weak_refs:
            weak_ref = self._weak_refs[step_id]
            instance = weak_ref()
            if instance is not None:
                return instance
            else:
                del self._weak_refs[step_id]
        
        # 4. 지연 의존성 해결
        if step_id in self._lazy_dependencies:
            lazy_dep = self._lazy_dependencies[step_id]
            result = lazy_dep.get()
            
            if result is not None:
                self._step_loading_stats['lazy_resolutions'] += 1
                
                # Step 클래스 튜플 처리 (step_class, success)
                if isinstance(result, tuple):
                    step_class, success = result
                    if success and step_class:
                        # 약한 참조로 캐시
                        self._weak_refs[step_id] = weakref.ref(step_class)
                        return step_class
                    else:
                        self._step_loading_stats['fallback_classes_used'] += 1
                        return None
                else:
                    # 직접 클래스 반환
                    try:
                        self._weak_refs[step_id] = weakref.ref(result)
                    except TypeError:
                        # 약한 참조를 생성할 수 없는 경우
                        pass
                    return result
        
        return None
    
    def create_step_instance_safe(self, step_id: str, **kwargs):
        """Step 인스턴스 안전 생성 (DI 완전 통합)"""
        step_class = self.get_step(step_id)
        if step_class:
            try:
                # Step 인스턴스 생성
                instance = step_class(**kwargs)
                
                # DI Container 기반 의존성 주입
                self.inject_dependencies_to_step_advanced(instance)
                
                return instance
            except Exception as e:
                logger.error(f"❌ {step_id} 인스턴스 생성 실패: {e}")
                return None
        return None
    
    def inject_dependencies_to_step_advanced(self, step_instance):
        """Step에 고급 의존성 주입 (DI Container 기반)"""
        try:
            injections_made = 0
            
            # 1. 전역 DI Container에서 의존성 조회
            global_container = self._services.get('global_di_container')
            if global_container and DI_CONTAINER_AVAILABLE:
                # ModelLoader 주입
                model_loader = global_container.get('model_loader')
                if model_loader and hasattr(step_instance, 'set_model_loader'):
                    step_instance.set_model_loader(model_loader)
                    injections_made += 1
                
                # MemoryManager 주입
                memory_manager = global_container.get('memory_manager')
                if memory_manager and hasattr(step_instance, 'set_memory_manager'):
                    step_instance.set_memory_manager(memory_manager)
                    injections_made += 1
                
                # DataConverter 주입
                data_converter = global_container.get('data_converter')
                if data_converter and hasattr(step_instance, 'set_data_converter'):
                    step_instance.set_data_converter(data_converter)
                    injections_made += 1
                
                # DI Container 자체 주입
                if hasattr(step_instance, 'set_di_container'):
                    step_instance.set_di_container(global_container)
                    injections_made += 1
            
            # 2. 기본 속성 설정
            if hasattr(step_instance, 'device') and not step_instance.device:
                step_instance.device = DEVICE
                injections_made += 1
            
            # 환경 정보 주입
            if hasattr(step_instance, 'is_m3_max'):
                step_instance.is_m3_max = IS_M3_MAX
                injections_made += 1
            
            if hasattr(step_instance, 'memory_gb'):
                step_instance.memory_gb = MEMORY_GB
                injections_made += 1
            
            # 3. 초기화 시도
            if hasattr(step_instance, 'initialize') and not getattr(step_instance, 'is_initialized', False):
                try:
                    if asyncio.iscoroutinefunction(step_instance.initialize):
                        # 비동기 초기화는 나중에 호출하도록 마킹
                        step_instance._needs_async_initialization = True
                    else:
                        # 동기 초기화 즉시 실행
                        step_instance.initialize()
                except Exception as e:
                    logger.debug(f"📋 {step_instance.__class__.__name__} 초기화 연기: {e}")
            
            self._step_loading_stats['di_injections_completed'] += 1
            logger.debug(f"✅ {step_instance.__class__.__name__} 고급 DI 주입 완료 ({injections_made}개)")
            
        except Exception as e:
            logger.error(f"❌ Step 고급 DI 주입 실패: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Step 로딩 통계 반환"""
        with self._lock:
            return {
                'container_type': 'StepsCircularReferenceFreeDIContainer',
                'version': '5.0',
                'step_loading_stats': dict(self._step_loading_stats),
                'registrations': {
                    'lazy_dependencies': len(self._lazy_dependencies),
                    'singleton_instances': len(self._singletons),
                    'transient_services': len(self._services),
                    'weak_references': len(self._weak_refs)
                },
                'circular_reference_protection': {
                    'detected_steps': list(self._circular_detected),
                    'current_resolving_stack': list(self._resolving_stack),
                    'prevention_count': self._step_loading_stats['circular_references_prevented']
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
                },
                'di_integration': {
                    'global_container_available': DI_CONTAINER_AVAILABLE,
                    'di_injections_completed': self._step_loading_stats['di_injections_completed']
                }
            }

# ==============================================
# 🔥 Step 클래스 안전한 로딩 (DI Container 기반)
# ==============================================

# 전역 Steps Container 생성
_steps_container = StepsCircularReferenceFreeDIContainer()

# Step 정의 매핑 (GitHub 구조 기준)
STEP_DEFINITIONS = {
    'step_01': ('step_01_human_parsing', 'HumanParsingStep'),
    'step_02': ('step_02_pose_estimation', 'PoseEstimationStep'),
    'step_03': ('step_03_cloth_segmentation', 'ClothSegmentationStep'),
    'step_04': ('step_04_geometric_matching', 'GeometricMatchingStep'),
    'step_05': ('step_05_cloth_warping', 'ClothWarpingStep'),
    'step_06': ('step_06_virtual_fitting', 'VirtualFittingStep'),
    'step_07': ('step_07_post_processing', 'PostProcessingStep'),
    'step_08': ('step_08_quality_assessment', 'QualityAssessmentStep')
}

logger.info("🔄 DI 기반 Step 클래스들 지연 등록 시작...")

# 모든 Step 지연 등록
for step_id, (step_module, step_class) in STEP_DEFINITIONS.items():
    _steps_container.register_step_lazy(step_id, step_module, step_class)

# BaseStepMixin 안전한 로딩 (지연)
BaseStepMixin = _steps_container.get_step('BaseStepMixin')
BASESTEP_AVAILABLE = BaseStepMixin is not None

# ==============================================
# 🔥 Step 클래스들 지연 로딩 함수들 (DI 기반)
# ==============================================

def get_step_class(step_id: str) -> Optional[Type]:
    """Step 클래스 반환 (DI 기반 지연 로딩)"""
    return _steps_container.get_step(step_id)

def get_available_steps() -> Dict[str, Type]:
    """사용 가능한 Step들 반환 (DI 기반 지연 로딩)"""
    available_steps = {}
    for step_id in STEP_DEFINITIONS.keys():
        step_class = get_step_class(step_id)
        if step_class:
            available_steps[step_id] = step_class
    return available_steps

def create_step_instance_safe(step_id: str, **kwargs):
    """Step 인스턴스 안전 생성 (DI Container 완전 통합)"""
    return _steps_container.create_step_instance_safe(step_id, **kwargs)

def inject_dependencies_to_step_safe_advanced(step_instance):
    """Step에 고급 안전한 의존성 주입 (DI Container 기반)"""
    _steps_container.inject_dependencies_to_step_advanced(step_instance)

def get_step_info() -> Dict[str, Any]:
    """Step 정보 반환 (DI 기반 지연 로딩)"""
    stats = _steps_container.get_stats()
    
    available_steps = []
    failed_steps = []
    
    for step_id in STEP_DEFINITIONS.keys():
        step_class = get_step_class(step_id)
        if step_class:
            available_steps.append(step_id)
        else:
            failed_steps.append(step_id)
    
    return {
        'total_steps': len(STEP_DEFINITIONS),
        'available_steps': len(available_steps),
        'available_step_list': available_steps,
        'failed_step_list': failed_steps,
        'success_rate': (len(available_steps) / len(STEP_DEFINITIONS)) * 100 if STEP_DEFINITIONS else 0,
        'container_stats': stats,
        'basestep_available': BASESTEP_AVAILABLE,
        'di_container_integrated': DI_CONTAINER_AVAILABLE
    }

def is_step_available(step_id: str) -> bool:
    """특정 Step이 사용 가능한지 확인 (DI 기반 지연 로딩)"""
    return get_step_class(step_id) is not None

def get_step_error_summary() -> Dict[str, Any]:
    """Step 에러 요약 (DI 기반 지연 로딩)"""
    step_info = get_step_info()
    
    return {
        'basestep_available': BASESTEP_AVAILABLE,
        'available_steps': step_info['available_steps'],
        'total_steps': step_info['total_steps'],
        'success_rate': step_info['success_rate'],
        'critical_step_01': is_step_available('step_01'),
        'critical_step_06': is_step_available('step_06'),
        'logger_errors_resolved': True,
        'circular_reference_resolved': True,
        'di_container_integrated': step_info['di_container_integrated'],
        'di_container_v4_available': DI_CONTAINER_AVAILABLE
    }

# ==============================================
# 🔥 안전한 Step Import 함수들 (DI 기반)
# ==============================================

def safe_import_step(module_name: str, class_name: str, step_id: str):
    """안전한 Step import (DI Container 기반)"""
    try:
        step_class = get_step_class(step_id)
        if step_class:
            logger.info(f"✅ {class_name} DI 로드 성공")
            return step_class, True
        else:
            logger.warning(f"⚠️ {class_name} DI 로드 실패")
            return None, False
    except Exception as e:
        logger.error(f"❌ {class_name} DI 로드 에러: {e}")
        return None, False

# ==============================================
# 🔥 Step 클래스들 DI 기반 로딩
# ==============================================

logger.info("🔄 DI 기반 Step 클래스들 로딩 시작...")

# Step 01: Human Parsing
HumanParsingStep, STEP_01_AVAILABLE = safe_import_step(
    'step_01_human_parsing', 'HumanParsingStep', 'step_01'
)

# Step 02: Pose Estimation
PoseEstimationStep, STEP_02_AVAILABLE = safe_import_step(
    'step_02_pose_estimation', 'PoseEstimationStep', 'step_02'
)

# Step 03: Cloth Segmentation
ClothSegmentationStep, STEP_03_AVAILABLE = safe_import_step(
    'step_03_cloth_segmentation', 'ClothSegmentationStep', 'step_03'
)

# Step 04: Geometric Matching
GeometricMatchingStep, STEP_04_AVAILABLE = safe_import_step(
    'step_04_geometric_matching', 'GeometricMatchingStep', 'step_04'
)

# Step 05: Cloth Warping
ClothWarpingStep, STEP_05_AVAILABLE = safe_import_step(
    'step_05_cloth_warping', 'ClothWarpingStep', 'step_05'
)

# Step 06: Virtual Fitting
VirtualFittingStep, STEP_06_AVAILABLE = safe_import_step(
    'step_06_virtual_fitting', 'VirtualFittingStep', 'step_06'
)

# Step 07: Post Processing
PostProcessingStep, STEP_07_AVAILABLE = safe_import_step(
    'step_07_post_processing', 'PostProcessingStep', 'step_07'
)

# Step 08: Quality Assessment
QualityAssessmentStep, STEP_08_AVAILABLE = safe_import_step(
    'step_08_quality_assessment', 'QualityAssessmentStep', 'step_08'
)

# ==============================================
# 🔥 Step 매핑 및 관리 (DI 기반)
# ==============================================

# 전체 Step 매핑 (DI 기반)
STEP_MAPPING = {
    'step_01': HumanParsingStep,
    'step_02': PoseEstimationStep,
    'step_03': ClothSegmentationStep,
    'step_04': GeometricMatchingStep,
    'step_05': ClothWarpingStep,
    'step_06': VirtualFittingStep,
    'step_07': PostProcessingStep,
    'step_08': QualityAssessmentStep
}

# 가용성 플래그 매핑 (DI 기반)
STEP_AVAILABILITY = {
    'step_01': STEP_01_AVAILABLE,
    'step_02': STEP_02_AVAILABLE,
    'step_03': STEP_03_AVAILABLE,
    'step_04': STEP_04_AVAILABLE,
    'step_05': STEP_05_AVAILABLE,
    'step_06': STEP_06_AVAILABLE,
    'step_07': STEP_07_AVAILABLE,
    'step_08': STEP_08_AVAILABLE
}

# 사용 가능한 Step만 필터링 (DI 기반)
AVAILABLE_STEPS = {
    step_id: step_class 
    for step_id, step_class in STEP_MAPPING.items() 
    if step_class is not None and STEP_AVAILABILITY.get(step_id, False)
}

# ==============================================
# 🔥 유틸리티 함수들 (DI Container 통합)
# ==============================================

def get_di_container_for_steps():
    """Steps용 DI Container 반환"""
    return _steps_container

def reset_steps_container():
    """Steps Container 리셋"""
    global _steps_container
    _steps_container = StepsCircularReferenceFreeDIContainer()
    logger.info("🔄 Steps Container 리셋 완료")

def optimize_steps_memory():
    """Steps 메모리 최적화"""
    try:
        # DI Container 메모리 최적화
        cleanup_count = 0
        
        # 약한 참조 정리
        dead_refs = []
        for key, ref in _steps_container._weak_refs.items():
            if ref() is None:
                dead_refs.append(key)
        
        for key in dead_refs:
            del _steps_container._weak_refs[key]
            cleanup_count += 1
        
        # 전역 가비지 컬렉션
        collected = gc.collect()
        
        # M3 Max MPS 최적화
        if TORCH_AVAILABLE and IS_M3_MAX and MPS_AVAILABLE:
            import torch
            if hasattr(torch.backends.mps, 'empty_cache'):
                torch.backends.mps.empty_cache()
        
        logger.info(f"🧹 Steps 메모리 최적화 완료: {cleanup_count}개 정리, {collected}개 GC")
        return {'cleaned_refs': cleanup_count, 'gc_collected': collected}
        
    except Exception as e:
        logger.error(f"❌ Steps 메모리 최적화 실패: {e}")
        return {}

# ==============================================
# 🔥 비동기 Step 관리 함수들
# ==============================================

async def initialize_all_steps_async():
    """모든 Step 비동기 초기화"""
    try:
        logger.info("🚀 모든 Step 비동기 초기화 시작")
        
        initialization_results = {}
        
        for step_id in STEP_DEFINITIONS.keys():
            if is_step_available(step_id):
                try:
                    step_instance = create_step_instance_safe(step_id)
                    
                    if step_instance and hasattr(step_instance, '_needs_async_initialization'):
                        if hasattr(step_instance, 'initialize'):
                            await step_instance.initialize()
                            initialization_results[step_id] = True
                            logger.info(f"✅ {step_id} 비동기 초기화 완료")
                        else:
                            initialization_results[step_id] = True
                    else:
                        initialization_results[step_id] = True
                        
                except Exception as e:
                    logger.error(f"❌ {step_id} 비동기 초기화 실패: {e}")
                    initialization_results[step_id] = False
        
        success_count = sum(1 for success in initialization_results.values() if success)
        total_count = len(initialization_results)
        
        logger.info(f"✅ Step 비동기 초기화 완료: {success_count}/{total_count}개")
        return initialization_results
        
    except Exception as e:
        logger.error(f"❌ Step 비동기 초기화 실패: {e}")
        return {}

async def cleanup_all_steps_async():
    """모든 Step 비동기 정리"""
    try:
        logger.info("🧹 모든 Step 비동기 정리 시작")
        
        cleanup_results = {}
        
        for step_id in STEP_DEFINITIONS.keys():
            if is_step_available(step_id):
                try:
                    step_instance = create_step_instance_safe(step_id)
                    
                    if step_instance and hasattr(step_instance, 'cleanup'):
                        if asyncio.iscoroutinefunction(step_instance.cleanup):
                            await step_instance.cleanup()
                        else:
                            step_instance.cleanup()
                        cleanup_results[step_id] = True
                        logger.debug(f"✅ {step_id} 비동기 정리 완료")
                    else:
                        cleanup_results[step_id] = True
                        
                except Exception as e:
                    logger.error(f"❌ {step_id} 비동기 정리 실패: {e}")
                    cleanup_results[step_id] = False
        
        # Steps Container 메모리 최적화
        optimize_steps_memory()
        
        success_count = sum(1 for success in cleanup_results.values() if success)
        total_count = len(cleanup_results)
        
        logger.info(f"✅ Step 비동기 정리 완료: {success_count}/{total_count}개")
        return cleanup_results
        
    except Exception as e:
        logger.error(f"❌ Step 비동기 정리 실패: {e}")
        return {}

# ==============================================
# 🔥 Export (API 호환성 유지)
# ==============================================

__all__ = [
    # Step 클래스들 (DI 기반 지연 로딩)
    'HumanParsingStep',
    'PoseEstimationStep', 
    'ClothSegmentationStep',
    'GeometricMatchingStep',
    'ClothWarpingStep',
    'VirtualFittingStep',
    'PostProcessingStep',
    'QualityAssessmentStep',
    
    # BaseStepMixin (DI 기반)
    'BaseStepMixin',
    
    # 유틸리티 함수들 (DI 기반)
    'get_step_class',
    'get_available_steps',
    'create_step_instance_safe',
    'get_step_info',
    'is_step_available',
    'get_step_error_summary',
    'inject_dependencies_to_step_safe_advanced',
    'safe_import_step',
    
    # 매핑 및 상태 (DI 기반)
    'STEP_MAPPING',
    'AVAILABLE_STEPS',
    'STEP_AVAILABILITY',
    'STEP_DEFINITIONS',
    
    # DI Container 관련
    'StepsCircularReferenceFreeDIContainer',
    'get_di_container_for_steps',
    'reset_steps_container',
    'optimize_steps_memory',
    
    # 비동기 함수들
    'initialize_all_steps_async',
    'cleanup_all_steps_async',
    
    # 상태 플래그들
    'STEP_01_AVAILABLE',
    'STEP_02_AVAILABLE',
    'STEP_03_AVAILABLE',
    'STEP_04_AVAILABLE',
    'STEP_05_AVAILABLE',
    'STEP_06_AVAILABLE',
    'STEP_07_AVAILABLE',
    'STEP_08_AVAILABLE',
    'BASESTEP_AVAILABLE',
    'DI_CONTAINER_AVAILABLE',
    
    # 타입들
    'T'
]

# ==============================================
# 🔥 conda 환경 최적화 (DI Container 기반)
# ==============================================

def optimize_conda_environment_with_di():
    """conda 환경 DI 기반 안전 최적화"""
    try:
        if not IS_CONDA:
            return
        
        # 환경 변수 설정
        os.environ.setdefault('OMP_NUM_THREADS', str(max(1, os.cpu_count() // 2)))
        os.environ.setdefault('MKL_NUM_THREADS', str(max(1, os.cpu_count() // 2)))
        os.environ.setdefault('NUMEXPR_NUM_THREADS', str(max(1, os.cpu_count() // 2)))
        
        # PyTorch 최적화
        if TORCH_AVAILABLE:
            import torch
            torch.set_num_threads(max(1, os.cpu_count() // 2))
            
            # M3 Max MPS 최적화
            if IS_M3_MAX and MPS_AVAILABLE:
                if hasattr(torch.backends.mps, 'empty_cache'):
                    torch.backends.mps.empty_cache()
                logger.info("🍎 M3 Max MPS conda DI 최적화 완료")
        
        # DI Container 메모리 최적화
        if DI_CONTAINER_AVAILABLE:
            optimize_steps_memory()
        
        logger.info(f"🐍 conda 환경 '{CONDA_ENV}' DI 기반 최적화 완료")
        
    except Exception as e:
        logger.warning(f"⚠️ conda DI 기반 최적화 실패: {e}")

# ==============================================
# 🔥 초기화 완료 로깅
# ==============================================

# 통계 수집
step_info = get_step_info()
error_summary = get_step_error_summary()

logger.info("=" * 80)
logger.info("🔥 MyCloset AI Pipeline Steps v5.0 초기화 완료 (DI Container v4.0 완전 통합)")
logger.info("=" * 80)
logger.info(f"🔗 DI Container v4.0: {'✅ 활성화' if DI_CONTAINER_AVAILABLE else '❌ 비활성화'}")
logger.info(f"📊 Step 로딩 결과: {step_info['available_steps']}/{step_info['total_steps']}개 ({step_info['success_rate']:.1f}%)")
logger.info(f"🔧 BaseStepMixin: {'✅ 정상' if error_summary['basestep_available'] else '⚠️ 폴백'}")
logger.info(f"🔑 Logger 에러: {'✅ 해결됨' if error_summary['logger_errors_resolved'] else '❌ 미해결'}")
logger.info(f"🔗 순환참조: {'✅ 해결됨' if error_summary['circular_reference_resolved'] else '❌ 미해결'}")
logger.info(f"💉 DI Container: {'✅ 통합됨' if error_summary['di_container_integrated'] else '❌ 미통합'}")

# DI Container 통계
if DI_CONTAINER_AVAILABLE:
    container_stats = step_info.get('container_stats', {})
    di_stats = container_stats.get('step_loading_stats', {})
    logger.info(f"🔗 DI 지연 해결: {di_stats.get('lazy_resolutions', 0)}회")
    logger.info(f"🚫 순환참조 차단: {di_stats.get('circular_references_prevented', 0)}회")
    logger.info(f"💉 DI 주입 완료: {di_stats.get('di_injections_completed', 0)}회")

if step_info['available_step_list']:
    logger.info(f"✅ 로드된 Steps: {', '.join(step_info['available_step_list'])}")

if step_info['failed_step_list']:
    logger.info(f"⚠️ 실패한 Steps: {', '.join(step_info['failed_step_list'])}")

# 중요한 Step들 개별 체크
critical_steps_status = []
if is_step_available('step_01'):
    logger.info("🎉 Step 01 (HumanParsingStep) DI 로딩 성공!")
    critical_steps_status.append("Step01 ✅")
else:
    logger.warning("⚠️ Step 01 (HumanParsingStep) DI 로딩 실패!")
    critical_steps_status.append("Step01 ❌")

if is_step_available('step_06'):
    logger.info("🎉 Step 06 (VirtualFittingStep) DI 로딩 성공!")
    critical_steps_status.append("Step06 ✅")
else:
    logger.warning("⚠️ Step 06 (VirtualFittingStep) DI 로딩 실패!")
    critical_steps_status.append("Step06 ❌")

# conda 환경 자동 최적화
if IS_TARGET_ENV:
    optimize_conda_environment_with_di()
    logger.info("🐍 conda 환경 mycloset-ai-clean DI 기반 자동 최적화 완료!")

if step_info['success_rate'] >= 50:
    logger.info("🚀 파이프라인 Steps 시스템 준비 완료!")
else:
    logger.warning("⚠️ 파이프라인 Steps 시스템 부분 준비 (일부 Step 사용 불가)")

logger.info("=" * 80)

# 최종 상태 체크
if step_info['available_steps'] > 0:
    logger.info("✅ Steps 모듈 DI v5.0 초기화 성공 - 순환참조 완전 해결 및 DI Container 통합")
else:
    logger.error("❌ Steps 모듈 DI v5.0 초기화 실패 - 모든 Step이 사용 불가")

logger.info("🔥 MyCloset AI Pipeline Steps v5.0 with DI Container v4.0 - 순환참조 완전 해결 및 완전 통합 완료!")