# backend/app/ai_pipeline/factories/step_factory.py
"""
🔥 StepFactory v5.0 - 의존성 주입 패턴 + 순환참조 완전 해결
==============================================================

✅ TYPE_CHECKING 패턴으로 순환참조 완전 방지
✅ 통합된 의존성 주입 시스템
✅ 표준화된 Step 생성 패턴
✅ 모든 Step 클래스 호환성 보장
✅ 향상된 에러 처리

Author: MyCloset AI Team
Date: 2025-07-24
Version: 5.0 (Dependency Injection Pattern)
"""

import os
import logging
import asyncio
import threading
import time
import weakref
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Type, Callable, TYPE_CHECKING
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from abc import ABC, abstractmethod

# 🔥 TYPE_CHECKING으로 순환참조 완전 방지
if TYPE_CHECKING:
    from ..steps.base_step_mixin import BaseStepMixin
    from ..utils.model_loader import ModelLoader
    from ..utils.memory_manager import MemoryManager
    from ..utils.data_converter import DataConverter
    from ..core.di_container import DIContainer

# ==============================================
# 🔥 기본 설정 및 로깅
# ==============================================

logger = logging.getLogger(__name__)

# 환경 정보
CONDA_ENV = os.environ.get('CONDA_DEFAULT_ENV', 'none')
IS_M3_MAX = False

try:
    import platform
    import subprocess
    if platform.system() == 'Darwin':
        result = subprocess.run(
            ['sysctl', '-n', 'machdep.cpu.brand_string'],
            capture_output=True, text=True, timeout=5
        )
        IS_M3_MAX = 'M3' in result.stdout
except:
    pass

# ==============================================
# 🔥 데이터 구조 정의
# ==============================================

class StepType(Enum):
    """Step 타입 정의"""
    HUMAN_PARSING = "human_parsing"
    POSE_ESTIMATION = "pose_estimation"
    CLOTH_SEGMENTATION = "cloth_segmentation"
    GEOMETRIC_MATCHING = "geometric_matching"
    CLOTH_WARPING = "cloth_warping"
    VIRTUAL_FITTING = "virtual_fitting"
    POST_PROCESSING = "post_processing"
    QUALITY_ASSESSMENT = "quality_assessment"

class StepPriority(IntEnum):
    """Step 우선순위"""
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4

@dataclass
class StepConfig:
    """Step 설정"""
    step_name: str
    step_id: int
    step_type: StepType
    device: str = "auto"
    use_fp16: bool = True
    batch_size: int = 1
    confidence_threshold: float = 0.8
    auto_memory_cleanup: bool = True
    auto_warmup: bool = True
    optimization_enabled: bool = True
    strict_mode: bool = False
    priority: StepPriority = StepPriority.MEDIUM
    
    # 의존성 설정
    require_model_loader: bool = True
    require_memory_manager: bool = False
    require_data_converter: bool = False
    auto_inject_dependencies: bool = True

@dataclass
class StepCreationResult:
    """Step 생성 결과"""
    success: bool
    step_instance: Optional['BaseStepMixin'] = None
    step_name: str = ""
    step_type: Optional[StepType] = None
    dependencies_injected: Dict[str, bool] = field(default_factory=dict)
    initialization_time: float = 0.0
    error_message: Optional[str] = None
    warnings: List[str] = field(default_factory=list)

@dataclass
class DependencyBundle:
    """의존성 번들"""
    model_loader: Optional['ModelLoader'] = None
    memory_manager: Optional['MemoryManager'] = None
    data_converter: Optional['DataConverter'] = None
    di_container: Optional['DIContainer'] = None

# ==============================================
# 🔥 의존성 해결기
# ==============================================

class DependencyResolver:
    """의존성 해결기 (순환참조 방지)"""
    
    def __init__(self):
        self.logger = logging.getLogger("DependencyResolver")
        self._resolved_cache: Dict[str, Any] = {}
        self._resolution_lock = threading.Lock()
    
    def resolve_model_loader(self, config: Optional[Dict[str, Any]] = None) -> Optional['ModelLoader']:
        """ModelLoader 해결 (동적 import)"""
        try:
            with self._resolution_lock:
                cache_key = "model_loader"
                if cache_key in self._resolved_cache:
                    return self._resolved_cache[cache_key]
                
                # 동적 import로 순환참조 방지
                import importlib
                model_loader_module = importlib.import_module('app.ai_pipeline.utils.model_loader')
                get_global_loader = getattr(model_loader_module, 'get_global_model_loader', None)
                
                if get_global_loader:
                    model_loader = get_global_loader(config)
                    # 초기화 확인
                    if hasattr(model_loader, 'initialize'):
                        if not model_loader.is_initialized():
                            success = model_loader.initialize()
                            if not success:
                                self.logger.error("ModelLoader 초기화 실패")
                                return None
                    
                    self._resolved_cache[cache_key] = model_loader
                    self.logger.info("✅ ModelLoader 해결 완료")
                    return model_loader
                else:
                    self.logger.error("get_global_model_loader 함수를 찾을 수 없음")
                    return None
                    
        except Exception as e:
            self.logger.error(f"❌ ModelLoader 해결 실패: {e}")
            return None
    
    def resolve_memory_manager(self) -> Optional['MemoryManager']:
        """MemoryManager 해결 (동적 import)"""
        try:
            with self._resolution_lock:
                cache_key = "memory_manager"
                if cache_key in self._resolved_cache:
                    return self._resolved_cache[cache_key]
                
                import importlib
                memory_module = importlib.import_module('app.ai_pipeline.utils.memory_manager')
                get_global_manager = getattr(memory_module, 'get_global_memory_manager', None)
                
                if get_global_manager:
                    memory_manager = get_global_manager()
                    self._resolved_cache[cache_key] = memory_manager
                    self.logger.info("✅ MemoryManager 해결 완료")
                    return memory_manager
                    
        except Exception as e:
            self.logger.debug(f"MemoryManager 해결 실패: {e}")
            return None
    
    def resolve_data_converter(self) -> Optional['DataConverter']:
        """DataConverter 해결 (동적 import)"""
        try:
            with self._resolution_lock:
                cache_key = "data_converter"
                if cache_key in self._resolved_cache:
                    return self._resolved_cache[cache_key]
                
                import importlib
                converter_module = importlib.import_module('app.ai_pipeline.utils.data_converter')
                get_global_converter = getattr(converter_module, 'get_global_data_converter', None)
                
                if get_global_converter:
                    data_converter = get_global_converter()
                    self._resolved_cache[cache_key] = data_converter
                    self.logger.info("✅ DataConverter 해결 완료")
                    return data_converter
                    
        except Exception as e:
            self.logger.debug(f"DataConverter 해결 실패: {e}")
            return None
    
    def resolve_di_container(self) -> Optional['DIContainer']:
        """DI Container 해결 (동적 import)"""
        try:
            with self._resolution_lock:
                cache_key = "di_container"
                if cache_key in self._resolved_cache:
                    return self._resolved_cache[cache_key]
                
                import importlib
                di_module = importlib.import_module('app.core.di_container')
                get_global_container = getattr(di_module, 'get_global_di_container', None)
                
                if get_global_container:
                    di_container = get_global_container()
                    self._resolved_cache[cache_key] = di_container
                    self.logger.info("✅ DI Container 해결 완료")
                    return di_container
                    
        except Exception as e:
            self.logger.debug(f"DI Container 해결 실패: {e}")
            return None
    
    def resolve_step_mixin_class(self, step_type: StepType) -> Optional[Type]:
        """BaseStepMixin 클래스 해결 (동적 import)"""
        try:
            cache_key = f"step_mixin_{step_type.value}"
            if cache_key in self._resolved_cache:
                return self._resolved_cache[cache_key]
            
            import importlib
            mixin_module = importlib.import_module('app.ai_pipeline.steps.base_step_mixin')
            
            # Step 타입별 클래스 매핑
            class_mapping = {
                StepType.HUMAN_PARSING: 'HumanParsingMixin',
                StepType.POSE_ESTIMATION: 'PoseEstimationMixin',
                StepType.CLOTH_SEGMENTATION: 'ClothSegmentationMixin',
                StepType.GEOMETRIC_MATCHING: 'GeometricMatchingMixin',
                StepType.CLOTH_WARPING: 'ClothWarpingMixin',
                StepType.VIRTUAL_FITTING: 'VirtualFittingMixin',
                StepType.POST_PROCESSING: 'PostProcessingMixin',
                StepType.QUALITY_ASSESSMENT: 'QualityAssessmentMixin'
            }
            
            class_name = class_mapping.get(step_type, 'BaseStepMixin')
            StepClass = getattr(mixin_module, class_name, None)
            
            if not StepClass:
                self.logger.error(f"{class_name} 클래스를 찾을 수 없음")
                return None
            
            self._resolved_cache[cache_key] = StepClass
            self.logger.info(f"✅ {class_name} 클래스 해결 완료")
            return StepClass
            
        except Exception as e:
            self.logger.error(f"❌ Step 클래스 해결 실패: {e}")
            return None
    
    def create_dependency_bundle(self, config: StepConfig) -> DependencyBundle:
        """의존성 번들 생성"""
        try:
            bundle = DependencyBundle()
            
            # 필수 의존성부터 해결
            if config.require_model_loader:
                bundle.model_loader = self.resolve_model_loader()
            
            if config.require_memory_manager:
                bundle.memory_manager = self.resolve_memory_manager()
            
            if config.require_data_converter:
                bundle.data_converter = self.resolve_data_converter()
            
            # DI Container는 항상 시도
            bundle.di_container = self.resolve_di_container()
            
            return bundle
            
        except Exception as e:
            self.logger.error(f"❌ 의존성 번들 생성 실패: {e}")
            return DependencyBundle()
    
    def clear_cache(self):
        """캐시 정리"""
        with self._resolution_lock:
            self._resolved_cache.clear()
            self.logger.info("🧹 의존성 해결기 캐시 정리 완료")

# ==============================================
# 🔥 메인 StepFactory 클래스
# ==============================================

class StepFactory:
    """의존성 주입 패턴 StepFactory v5.0"""
    
    def __init__(self):
        self.logger = logging.getLogger("StepFactory")
        
        # 의존성 해결기
        self.dependency_resolver = DependencyResolver()
        
        # 캐시 및 상태 관리
        self._step_cache: Dict[str, weakref.ref] = {}
        self._creation_stats = {
            'total_created': 0,
            'successful_creations': 0,
            'failed_creations': 0,
            'cache_hits': 0,
            'dependencies_resolved': 0
        }
        
        # 동기화
        self._lock = threading.RLock()
        
        self.logger.info("🏭 StepFactory v5.0 초기화 완료 (의존성 주입 패턴)")
    
    def create_step(
        self, 
        step_type: Union[StepType, str], 
        config: Optional[StepConfig] = None,
        use_cache: bool = True,
        **kwargs
    ) -> StepCreationResult:
        """통합 Step 생성 메서드"""
        start_time = time.time()
        
        try:
            # Step 타입 정규화
            if isinstance(step_type, str):
                try:
                    step_type = StepType(step_type.lower())
                except ValueError:
                    return StepCreationResult(
                        success=False,
                        error_message=f"지원하지 않는 Step 타입: {step_type}"
                    )
            
            # 설정 생성
            if config is None:
                config = self._create_default_config(step_type, **kwargs)
            
            # 캐시 확인
            if use_cache:
                cached_step = self._get_cached_step(config.step_name)
                if cached_step:
                    self._creation_stats['cache_hits'] += 1
                    return StepCreationResult(
                        success=True,
                        step_instance=cached_step,
                        step_name=config.step_name,
                        step_type=step_type,
                        initialization_time=time.time() - start_time
                    )
            
            # Step 생성 실행
            result = self._create_step_instance(step_type, config)
            
            # 성공 시 캐시에 저장
            if result.success and result.step_instance and use_cache:
                self._cache_step(config.step_name, result.step_instance)
            
            # 통계 업데이트
            self._creation_stats['total_created'] += 1
            if result.success:
                self._creation_stats['successful_creations'] += 1
            else:
                self._creation_stats['failed_creations'] += 1
            
            result.initialization_time = time.time() - start_time
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Step 생성 실패: {e}")
            return StepCreationResult(
                success=False,
                error_message=f"Step 생성 중 예외 발생: {str(e)}",
                initialization_time=time.time() - start_time
            )
    
    def _create_default_config(self, step_type: StepType, **kwargs) -> StepConfig:
        """기본 설정 생성"""
        # Step별 기본 설정
        step_configs = {
            StepType.HUMAN_PARSING: {
                'step_name': 'HumanParsingStep',
                'step_id': 1,
                'priority': StepPriority.CRITICAL
            },
            StepType.POSE_ESTIMATION: {
                'step_name': 'PoseEstimationStep',
                'step_id': 2,
                'priority': StepPriority.HIGH
            },
            StepType.CLOTH_SEGMENTATION: {
                'step_name': 'ClothSegmentationStep',
                'step_id': 3,
                'priority': StepPriority.HIGH
            },
            StepType.GEOMETRIC_MATCHING: {
                'step_name': 'GeometricMatchingStep',
                'step_id': 4,
                'priority': StepPriority.MEDIUM
            },
            StepType.CLOTH_WARPING: {
                'step_name': 'ClothWarpingStep',
                'step_id': 5,
                'priority': StepPriority.MEDIUM
            },
            StepType.VIRTUAL_FITTING: {
                'step_name': 'VirtualFittingStep',
                'step_id': 6,
                'priority': StepPriority.CRITICAL
            },
            StepType.POST_PROCESSING: {
                'step_name': 'PostProcessingStep',
                'step_id': 7,
                'priority': StepPriority.LOW
            },
            StepType.QUALITY_ASSESSMENT: {
                'step_name': 'QualityAssessmentStep',
                'step_id': 8,
                'priority': StepPriority.LOW
            }
        }
        
        default_config = step_configs.get(step_type, {
            'step_name': f'{step_type.value.title()}Step',
            'step_id': 0,
            'priority': StepPriority.MEDIUM
        })
        
        # kwargs로 덮어쓰기
        default_config.update(kwargs)
        
        return StepConfig(
            step_type=step_type,
            **default_config
        )
    
    def _create_step_instance(self, step_type: StepType, config: StepConfig) -> StepCreationResult:
        """Step 인스턴스 생성"""
        try:
            self.logger.info(f"🔄 {config.step_name} 생성 시작...")
            
            # 1. Step 클래스 해결
            StepClass = self.dependency_resolver.resolve_step_mixin_class(step_type)
            if not StepClass:
                return StepCreationResult(
                    success=False,
                    step_name=config.step_name,
                    step_type=step_type,
                    error_message=f"{step_type.value} Step 클래스를 찾을 수 없음"
                )
            
            # 2. 의존성 번들 생성
            dependency_bundle = self.dependency_resolver.create_dependency_bundle(config)
            
            # 3. Step 인스턴스 생성
            step_kwargs = {
                'step_name': config.step_name,
                'step_id': config.step_id,
                'device': config.device,
                'use_fp16': config.use_fp16,
                'batch_size': config.batch_size,
                'confidence_threshold': config.confidence_threshold,
                'auto_memory_cleanup': config.auto_memory_cleanup,
                'auto_warmup': config.auto_warmup,
                'optimization_enabled': config.optimization_enabled,
                'strict_mode': config.strict_mode,
                'auto_inject_dependencies': config.auto_inject_dependencies,
                'require_model_loader': config.require_model_loader,
                'require_memory_manager': config.require_memory_manager,
                'require_data_converter': config.require_data_converter
            }
            
            step_instance = StepClass(**step_kwargs)
            
            # 4. 의존성 주입
            dependencies_injected = self._inject_dependencies(step_instance, dependency_bundle, config)
            
            # 5. 초기화 실행
            initialization_success = self._initialize_step(step_instance, config)
            
            if not initialization_success and config.strict_mode:
                return StepCreationResult(
                    success=False,
                    step_name=config.step_name,
                    step_type=step_type,
                    error_message="Step 초기화 실패 (Strict Mode)",
                    dependencies_injected=dependencies_injected
                )
            
            self.logger.info(f"✅ {config.step_name} 생성 완료")
            
            return StepCreationResult(
                success=True,
                step_instance=step_instance,
                step_name=config.step_name,
                step_type=step_type,
                dependencies_injected=dependencies_injected
            )
            
        except Exception as e:
            self.logger.error(f"❌ {config.step_name} 생성 실패: {e}")
            return StepCreationResult(
                success=False,
                step_name=config.step_name,
                step_type=step_type,
                error_message=f"Step 인스턴스 생성 실패: {str(e)}"
            )
    
    def _inject_dependencies(
        self, 
        step_instance: 'BaseStepMixin', 
        dependency_bundle: DependencyBundle,
        config: StepConfig
    ) -> Dict[str, bool]:
        """의존성 주입 실행"""
        injection_results = {}
        
        try:
            # ModelLoader 주입
            if dependency_bundle.model_loader and hasattr(step_instance, 'set_model_loader'):
                try:
                    step_instance.set_model_loader(dependency_bundle.model_loader)
                    injection_results['model_loader'] = True
                    self._creation_stats['dependencies_resolved'] += 1
                except Exception as e:
                    self.logger.warning(f"⚠️ ModelLoader 주입 실패: {e}")
                    injection_results['model_loader'] = False
            else:
                injection_results['model_loader'] = False
            
            # MemoryManager 주입
            if dependency_bundle.memory_manager and hasattr(step_instance, 'set_memory_manager'):
                try:
                    step_instance.set_memory_manager(dependency_bundle.memory_manager)
                    injection_results['memory_manager'] = True
                    self._creation_stats['dependencies_resolved'] += 1
                except Exception as e:
                    self.logger.warning(f"⚠️ MemoryManager 주입 실패: {e}")
                    injection_results['memory_manager'] = False
            else:
                injection_results['memory_manager'] = False
            
            # DataConverter 주입
            if dependency_bundle.data_converter and hasattr(step_instance, 'set_data_converter'):
                try:
                    step_instance.set_data_converter(dependency_bundle.data_converter)
                    injection_results['data_converter'] = True
                    self._creation_stats['dependencies_resolved'] += 1
                except Exception as e:
                    self.logger.warning(f"⚠️ DataConverter 주입 실패: {e}")
                    injection_results['data_converter'] = False
            else:
                injection_results['data_converter'] = False
            
            # DI Container 주입
            if dependency_bundle.di_container and hasattr(step_instance, 'set_di_container'):
                try:
                    step_instance.set_di_container(dependency_bundle.di_container)
                    injection_results['di_container'] = True
                    self._creation_stats['dependencies_resolved'] += 1
                except Exception as e:
                    self.logger.warning(f"⚠️ DI Container 주입 실패: {e}")
                    injection_results['di_container'] = False
            else:
                injection_results['di_container'] = False
            
            # 필수 의존성 검증
            required_dependencies = []
            if config.require_model_loader:
                required_dependencies.append('model_loader')
            if config.require_memory_manager:
                required_dependencies.append('memory_manager')
            if config.require_data_converter:
                required_dependencies.append('data_converter')
            
            missing_dependencies = [
                dep for dep in required_dependencies 
                if not injection_results.get(dep, False)
            ]
            
            if missing_dependencies and config.strict_mode:
                self.logger.error(f"❌ 필수 의존성 누락: {missing_dependencies}")
                raise RuntimeError(f"필수 의존성이 주입되지 않음: {missing_dependencies}")
            
            success_count = sum(1 for success in injection_results.values() if success)
            self.logger.info(f"💉 의존성 주입 완료: {success_count}/{len(injection_results)} 성공")
            
            return injection_results
            
        except Exception as e:
            self.logger.error(f"❌ 의존성 주입 실패: {e}")
            return injection_results
    
    def _initialize_step(self, step_instance: 'BaseStepMixin', config: StepConfig) -> bool:
        """Step 초기화 실행"""
        try:
            # BaseStepMixin 초기화
            if hasattr(step_instance, 'initialize'):
                success = step_instance.initialize()
                if not success:
                    self.logger.error(f"❌ {config.step_name} 초기화 실패")
                    return False
            
            # 워밍업 실행 (설정된 경우)
            if config.auto_warmup and hasattr(step_instance, 'warmup'):
                try:
                    warmup_result = step_instance.warmup()
                    if warmup_result.get('success', False):
                        self.logger.info(f"🔥 {config.step_name} 워밍업 완료")
                    else:
                        self.logger.warning(f"⚠️ {config.step_name} 워밍업 실패")
                except Exception as warmup_error:
                    self.logger.warning(f"⚠️ {config.step_name} 워밍업 오류: {warmup_error}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ {config.step_name} 초기화 실패: {e}")
            return False
    
    def _get_cached_step(self, step_name: str) -> Optional['BaseStepMixin']:
        """캐시된 Step 조회"""
        try:
            with self._lock:
                if step_name in self._step_cache:
                    weak_ref = self._step_cache[step_name]
                    step_instance = weak_ref()
                    if step_instance is not None:
                        self.logger.debug(f"♻️ 캐시된 Step 반환: {step_name}")
                        return step_instance
                    else:
                        # 약한 참조가 해제됨
                        del self._step_cache[step_name]
                return None
        except Exception as e:
            self.logger.debug(f"캐시 조회 실패: {e}")
            return None
    
    def _cache_step(self, step_name: str, step_instance: 'BaseStepMixin'):
        """Step 캐시에 저장"""
        try:
            with self._lock:
                self._step_cache[step_name] = weakref.ref(step_instance)
                self.logger.debug(f"💾 Step 캐시에 저장: {step_name}")
        except Exception as e:
            self.logger.debug(f"캐시 저장 실패: {e}")
    
    # ==============================================
    # 🔥 비동기 Step 생성
    # ==============================================
    
    async def create_step_async(
        self, 
        step_type: Union[StepType, str], 
        config: Optional[StepConfig] = None,
        use_cache: bool = True,
        **kwargs
    ) -> StepCreationResult:
        """비동기 Step 생성"""
        try:
            # 동기 생성을 executor에서 실행
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                self.create_step,
                step_type,
                config,
                use_cache,
                **kwargs
            )
            
            # 비동기 초기화 (가능한 경우)
            if result.success and result.step_instance:
                if hasattr(result.step_instance, 'initialize_async'):
                    try:
                        await result.step_instance.initialize_async()
                        self.logger.info(f"✅ {result.step_name} 비동기 초기화 완료")
                    except Exception as async_init_error:
                        self.logger.warning(f"⚠️ {result.step_name} 비동기 초기화 실패: {async_init_error}")
                        if not result.warnings:
                            result.warnings = []
                        result.warnings.append(f"비동기 초기화 실패: {async_init_error}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ 비동기 Step 생성 실패: {e}")
            return StepCreationResult(
                success=False,
                error_message=f"비동기 Step 생성 실패: {str(e)}"
            )
    
    # ==============================================
    # 🔥 편의 메서드들
    # ==============================================
    
    def create_human_parsing_step(self, **kwargs) -> StepCreationResult:
        """Human Parsing Step 생성"""
        return self.create_step(StepType.HUMAN_PARSING, **kwargs)
    
    def create_pose_estimation_step(self, **kwargs) -> StepCreationResult:
        """Pose Estimation Step 생성"""
        return self.create_step(StepType.POSE_ESTIMATION, **kwargs)
    
    def create_cloth_segmentation_step(self, **kwargs) -> StepCreationResult:
        """Cloth Segmentation Step 생성"""
        return self.create_step(StepType.CLOTH_SEGMENTATION, **kwargs)
    
    def create_geometric_matching_step(self, **kwargs) -> StepCreationResult:
        """Geometric Matching Step 생성"""
        return self.create_step(StepType.GEOMETRIC_MATCHING, **kwargs)
    
    def create_cloth_warping_step(self, **kwargs) -> StepCreationResult:
        """Cloth Warping Step 생성"""
        return self.create_step(StepType.CLOTH_WARPING, **kwargs)
    
    def create_virtual_fitting_step(self, **kwargs) -> StepCreationResult:
        """Virtual Fitting Step 생성"""
        return self.create_step(StepType.VIRTUAL_FITTING, **kwargs)
    
    def create_post_processing_step(self, **kwargs) -> StepCreationResult:
        """Post Processing Step 생성"""
        return self.create_step(StepType.POST_PROCESSING, **kwargs)
    
    def create_quality_assessment_step(self, **kwargs) -> StepCreationResult:
        """Quality Assessment Step 생성"""
        return self.create_step(StepType.QUALITY_ASSESSMENT, **kwargs)
    
    # 비동기 편의 메서드들
    async def create_human_parsing_step_async(self, **kwargs) -> StepCreationResult:
        """비동기 Human Parsing Step 생성"""
        return await self.create_step_async(StepType.HUMAN_PARSING, **kwargs)
    
    async def create_pose_estimation_step_async(self, **kwargs) -> StepCreationResult:
        """비동기 Pose Estimation Step 생성"""
        return await self.create_step_async(StepType.POSE_ESTIMATION, **kwargs)
    
    async def create_cloth_segmentation_step_async(self, **kwargs) -> StepCreationResult:
        """비동기 Cloth Segmentation Step 생성"""
        return await self.create_step_async(StepType.CLOTH_SEGMENTATION, **kwargs)
    
    async def create_virtual_fitting_step_async(self, **kwargs) -> StepCreationResult:
        """비동기 Virtual Fitting Step 생성"""
        return await self.create_step_async(StepType.VIRTUAL_FITTING, **kwargs)
    
    # ==============================================
    # 🔥 전체 파이프라인 생성
    # ==============================================
    
    def create_full_pipeline(self, device: str = "auto", **kwargs) -> Dict[str, StepCreationResult]:
        """전체 AI 파이프라인 생성"""
        try:
            self.logger.info("🚀 전체 AI 파이프라인 생성 시작...")
            
            pipeline_results = {}
            
            # 모든 Step 타입에 대해 순차적으로 생성
            for step_type in StepType:
                try:
                    config_kwargs = {
                        'device': device,
                        **kwargs
                    }
                    
                    result = self.create_step(step_type, **config_kwargs)
                    pipeline_results[step_type.value] = result
                    
                    if result.success:
                        self.logger.info(f"✅ {result.step_name} 파이프라인 생성 성공")
                    else:
                        self.logger.error(f"❌ {step_type.value} 파이프라인 생성 실패: {result.error_message}")
                        
                except Exception as step_error:
                    self.logger.error(f"❌ {step_type.value} Step 생성 중 예외: {step_error}")
                    pipeline_results[step_type.value] = StepCreationResult(
                        success=False,
                        step_name=f"{step_type.value}Step",
                        step_type=step_type,
                        error_message=str(step_error)
                    )
            
            success_count = sum(1 for result in pipeline_results.values() if result.success)
            total_count = len(pipeline_results)
            
            self.logger.info(f"🏁 전체 파이프라인 생성 완료: {success_count}/{total_count} 성공")
            
            return pipeline_results
            
        except Exception as e:
            self.logger.error(f"❌ 전체 파이프라인 생성 실패: {e}")
            return {}
    
    async def create_full_pipeline_async(self, device: str = "auto", **kwargs) -> Dict[str, StepCreationResult]:
        """비동기 전체 AI 파이프라인 생성"""
        try:
            self.logger.info("🚀 비동기 전체 AI 파이프라인 생성 시작...")
            
            # 모든 Step을 동시에 생성
            tasks = []
            for step_type in StepType:
                config_kwargs = {
                    'device': device,
                    **kwargs
                }
                task = asyncio.create_task(
                    self.create_step_async(step_type, **config_kwargs)
                )
                tasks.append((step_type, task))
            
            # 모든 Task 완료 대기
            pipeline_results = {}
            for step_type, task in tasks:
                try:
                    result = await task
                    pipeline_results[step_type.value] = result
                    
                    if result.success:
                        self.logger.info(f"✅ {result.step_name} 비동기 파이프라인 생성 성공")
                    else:
                        self.logger.error(f"❌ {step_type.value} 비동기 파이프라인 생성 실패")
                        
                except Exception as step_error:
                    self.logger.error(f"❌ {step_type.value} 비동기 Step 생성 중 예외: {step_error}")
                    pipeline_results[step_type.value] = StepCreationResult(
                        success=False,
                        step_name=f"{step_type.value}Step",
                        step_type=step_type,
                        error_message=str(step_error)
                    )
            
            success_count = sum(1 for result in pipeline_results.values() if result.success)
            total_count = len(pipeline_results)
            
            self.logger.info(f"🏁 비동기 전체 파이프라인 생성 완료: {success_count}/{total_count} 성공")
            
            return pipeline_results
            
        except Exception as e:
            self.logger.error(f"❌ 비동기 전체 파이프라인 생성 실패: {e}")
            return {}
    
    # ==============================================
    # 🔥 상태 및 통계 메서드들
    # ==============================================
    
    def get_creation_statistics(self) -> Dict[str, Any]:
        """생성 통계 조회"""
        try:
            with self._lock:
                total = self._creation_stats['total_created']
                success_rate = (
                    self._creation_stats['successful_creations'] / max(1, total) * 100
                )
                
                return {
                    'total_created': total,
                    'successful_creations': self._creation_stats['successful_creations'],
                    'failed_creations': self._creation_stats['failed_creations'],
                    'success_rate': round(success_rate, 2),
                    'cache_hits': self._creation_stats['cache_hits'],
                    'dependencies_resolved': self._creation_stats['dependencies_resolved'],
                    'cached_steps': len(self._step_cache),
                    'active_cache_entries': len([
                        ref for ref in self._step_cache.values() 
                        if ref() is not None
                    ])
                }
        except Exception as e:
            self.logger.error(f"❌ 통계 조회 실패: {e}")
            return {}
    
    def clear_cache(self):
        """캐시 정리"""
        try:
            with self._lock:
                self._step_cache.clear()
                self.dependency_resolver.clear_cache()
                self.logger.info("🧹 StepFactory 캐시 정리 완료")
        except Exception as e:
            self.logger.error(f"❌ 캐시 정리 실패: {e}")
    
    def validate_dependencies(self) -> Dict[str, bool]:
        """의존성 검증"""
        try:
            validation_results = {}
            
            # ModelLoader 검증
            model_loader = self.dependency_resolver.resolve_model_loader()
            validation_results['model_loader'] = model_loader is not None
            
            # MemoryManager 검증
            memory_manager = self.dependency_resolver.resolve_memory_manager()
            validation_results['memory_manager'] = memory_manager is not None
            
            # DataConverter 검증
            data_converter = self.dependency_resolver.resolve_data_converter()
            validation_results['data_converter'] = data_converter is not None
            
            # DI Container 검증
            di_container = self.dependency_resolver.resolve_di_container()
            validation_results['di_container'] = di_container is not None
            
            # BaseStepMixin 검증
            for step_type in StepType:
                step_class = self.dependency_resolver.resolve_step_mixin_class(step_type)
                validation_results[f'step_class_{step_type.value}'] = step_class is not None
            
            return validation_results
            
        except Exception as e:
            self.logger.error(f"❌ 의존성 검증 실패: {e}")
            return {}

# ==============================================
# 🔥 전역 StepFactory 관리
# ==============================================

_global_step_factory: Optional[StepFactory] = None
_factory_lock = threading.Lock()

def get_global_step_factory() -> StepFactory:
    """전역 StepFactory 인스턴스 반환"""
    global _global_step_factory
    
    with _factory_lock:
        if _global_step_factory is None:
            _global_step_factory = StepFactory()
            logger.info("✅ 전역 StepFactory 생성 완료")
        
        return _global_step_factory

# ==============================================
# 🔥 편의 함수들
# ==============================================

def create_step(
    step_type: Union[StepType, str], 
    config: Optional[StepConfig] = None,
    **kwargs
) -> StepCreationResult:
    """전역 Step 생성 함수"""
    factory = get_global_step_factory()
    return factory.create_step(step_type, config, **kwargs)

async def create_step_async(
    step_type: Union[StepType, str], 
    config: Optional[StepConfig] = None,
    **kwargs
) -> StepCreationResult:
    """전역 비동기 Step 생성 함수"""
    factory = get_global_step_factory()
    return await factory.create_step_async(step_type, config, **kwargs)

def create_human_parsing_step(**kwargs) -> StepCreationResult:
    """Human Parsing Step 생성"""
    return create_step(StepType.HUMAN_PARSING, **kwargs)

def create_pose_estimation_step(**kwargs) -> StepCreationResult:
    """Pose Estimation Step 생성"""
    return create_step(StepType.POSE_ESTIMATION, **kwargs)

def create_cloth_segmentation_step(**kwargs) -> StepCreationResult:
    """Cloth Segmentation Step 생성"""
    return create_step(StepType.CLOTH_SEGMENTATION, **kwargs)

def create_virtual_fitting_step(**kwargs) -> StepCreationResult:
    """Virtual Fitting Step 생성"""
    return create_step(StepType.VIRTUAL_FITTING, **kwargs)

def create_full_pipeline(device: str = "auto", **kwargs) -> Dict[str, StepCreationResult]:
    """전체 파이프라인 생성"""
    factory = get_global_step_factory()
    return factory.create_full_pipeline(device, **kwargs)

async def create_full_pipeline_async(device: str = "auto", **kwargs) -> Dict[str, StepCreationResult]:
    """비동기 전체 파이프라인 생성"""
    factory = get_global_step_factory()
    return await factory.create_full_pipeline_async(device, **kwargs)

def validate_step_dependencies() -> Dict[str, bool]:
    """Step 의존성 검증"""
    factory = get_global_step_factory()
    return factory.validate_dependencies()

def get_step_factory_statistics() -> Dict[str, Any]:
    """StepFactory 통계 조회"""
    factory = get_global_step_factory()
    return factory.get_creation_statistics()

def clear_step_factory_cache():
    """StepFactory 캐시 정리"""
    factory = get_global_step_factory()
    factory.clear_cache()

# ==============================================
# 🔥 Export
# ==============================================

__all__ = [
    # 메인 클래스들
    'StepFactory',
    'DependencyResolver',
    
    # 데이터 구조들
    'StepType',
    'StepPriority',
    'StepConfig',
    'StepCreationResult',
    'DependencyBundle',
    
    # 전역 함수들
    'get_global_step_factory',
    
    # 편의 함수들
    'create_step',
    'create_step_async',
    'create_human_parsing_step',
    'create_pose_estimation_step',
    'create_cloth_segmentation_step',
    'create_virtual_fitting_step',
    'create_full_pipeline',
    'create_full_pipeline_async',
    
    # 유틸리티 함수들
    'validate_step_dependencies',
    'get_step_factory_statistics',
    'clear_step_factory_cache'
]

# 모듈 로드 완료
logger.info("=" * 80)
logger.info("🏭 StepFactory v5.0 - 의존성 주입 패턴 + 순환참조 완전 해결")
logger.info("=" * 80)
logger.info("✅ TYPE_CHECKING 패턴으로 순환참조 완전 방지")
logger.info("✅ 통합된 의존성 주입 시스템")
logger.info("✅ 표준화된 Step 생성 패턴")
logger.info("✅ 모든 Step 클래스 호환성 보장")
logger.info("✅ 향상된 에러 처리")
logger.info("✅ 비동기 Step 생성 지원")
logger.info("✅ 전체 파이프라인 생성 기능")
logger.info("✅ 약한 참조 기반 캐싱")
logger.info("✅ 프로덕션 레벨 안정성")
logger.info("=" * 80)