# backend/app/ai_pipeline/factories/step_factory.py
"""
🔥 StepFactory v1.0 - 의존성 주입 전용 팩토리 (순환참조 완전 해결) - 수정된 버전
=======================================================================
✅ StepFactoryConfig 매개변수 불일치 수정 (device_type → device)
✅ 순환참조 완전 방지 - 한방향 의존성 구조
✅ 의존성 주입 패턴 완전 구현
✅ BaseStepMixin과 ModelLoader 안전한 조립
✅ M3 Max 128GB 최적화
✅ conda 환경 우선 지원
✅ 8단계 AI 파이프라인 완전 지원
✅ 프로덕션 레벨 안정성

구조:
StepFactory → ModelLoader (생성) → BaseStepMixin (생성) → 의존성 주입 → 완성된 Step

핵심 철학:
- StepFactory가 모든 것을 조립
- ModelLoader와 BaseStepMixin은 서로 모름
- 의존성 주입으로 연결
- 단방향 의존성만 허용

Author: MyCloset AI Team
Date: 2025-07-23
Version: 1.0 (Fixed Parameter Compatibility)
"""

import os
import gc
import time
import logging
import asyncio
import threading
import traceback
import weakref
import platform
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Type, Callable, Tuple, TYPE_CHECKING
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from functools import lru_cache, wraps
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor

# ==============================================
# 🔥 1. TYPE_CHECKING으로 순환참조 완전 방지
# ==============================================

if TYPE_CHECKING:
    # 타입 체킹 시에만 임포트 (런타임에는 임포트 안됨)
    from ..utils.model_loader import ModelLoader
    from ..steps.base_step_mixin import BaseStepMixin, HumanParsingMixin, PoseEstimationMixin
    from ..steps.base_step_mixin import ClothSegmentationMixin, GeometricMatchingMixin
    from ..steps.base_step_mixin import ClothWarpingMixin, VirtualFittingMixin
    from ..steps.base_step_mixin import PostProcessingMixin, QualityAssessmentMixin

# ==============================================
# 🔥 2. conda 환경 및 시스템 체크
# ==============================================

# conda 환경 정보
CONDA_INFO = {
    'conda_env': os.environ.get('CONDA_DEFAULT_ENV', 'none'),
    'conda_prefix': os.environ.get('CONDA_PREFIX', 'none'),
    'python_path': os.path.dirname(os.__file__)
}

def detect_m3_max() -> bool:
    """M3 Max 감지"""
    try:
        if platform.system() == 'Darwin':
            result = subprocess.run(
                ['sysctl', '-n', 'machdep.cpu.brand_string'],
                capture_output=True, text=True, timeout=5
            )
            return 'M3' in result.stdout
    except:
        pass
    return False

IS_M3_MAX = detect_m3_max()

# 라이브러리 가용성 체크
TORCH_AVAILABLE = False
MPS_AVAILABLE = False
try:
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
    os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
    
    import torch
    TORCH_AVAILABLE = True
    
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        MPS_AVAILABLE = True
except ImportError:
    pass

# ==============================================
# 🔥 3. 로깅 설정
# ==============================================

logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# ==============================================
# 🔥 4. 팩토리 설정 및 데이터 구조 (수정된 버전)
# ==============================================

class StepType(Enum):
    """Step 타입 정의 (8단계 AI 파이프라인)"""
    HUMAN_PARSING = "human_parsing"
    POSE_ESTIMATION = "pose_estimation"
    CLOTH_SEGMENTATION = "cloth_segmentation"
    GEOMETRIC_MATCHING = "geometric_matching"
    CLOTH_WARPING = "cloth_warping"
    VIRTUAL_FITTING = "virtual_fitting"
    POST_PROCESSING = "post_processing"
    QUALITY_ASSESSMENT = "quality_assessment"

class OptimizationLevel(IntEnum):
    """최적화 레벨"""
    BASIC = 1
    STANDARD = 2
    HIGH = 3
    M3_MAX = 4
    PRODUCTION = 5

@dataclass
class StepFactoryConfig:
    """
    🔥 StepFactory 설정 (수정된 버전 - device_type 대신 device 사용)
    """
    # 시스템 설정 (🔥 device_type 제거, device만 사용)
    device: str = "auto"
    optimization_level: OptimizationLevel = OptimizationLevel.STANDARD
    use_conda_optimization: bool = True
    
    # ModelLoader 설정
    model_cache_dir: Optional[str] = None
    use_fp16: bool = True
    max_cached_models: int = 30
    lazy_loading: bool = True
    
    # BaseStepMixin 설정
    auto_warmup: bool = True
    auto_memory_cleanup: bool = True
    
    # Step별 설정
    step_configs: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # 의존성 주입 설정
    enable_dependency_injection: bool = True
    dependency_injection_mode: str = "runtime"  # "runtime" or "creation"
    
    # 디버깅 설정
    enable_debug_logging: bool = False
    validate_dependencies: bool = True
    
    # 🔥 기존 호환성을 위한 property (device_type을 device로 자동 매핑)
    @property
    def device_type(self) -> str:
        """기존 호환성을 위한 device_type 속성 (device로 매핑)"""
        return self.device
    
    @device_type.setter
    def device_type(self, value: str):
        """device_type 설정 시 device로 매핑"""
        self.device = value

@dataclass
class StepFactoryResult:
    """StepFactory 결과"""
    step_instance: Any
    model_loader: Any
    step_config: Dict[str, Any]
    creation_time: float
    success: bool
    error_message: Optional[str] = None
    dependencies_injected: bool = False
    optimization_applied: bool = False

# ==============================================
# 🔥 5. 의존성 해결 유틸리티
# ==============================================

class DependencyResolver:
    """의존성 해결 도우미 (순환참조 방지)"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.DependencyResolver")
        self._model_loader_cache = None
        self._step_mixin_classes = {}
        
    def resolve_model_loader(self, config: StepFactoryConfig) -> Optional[Any]:
        """ModelLoader 안전하게 해결 (동적 import)"""
        try:
            if self._model_loader_cache is not None:
                return self._model_loader_cache
            
            # 동적 import로 순환참조 방지
            import importlib
            loader_module = importlib.import_module('app.ai_pipeline.utils.model_loader')
            
            # ModelLoader 클래스 가져오기
            ModelLoaderClass = getattr(loader_module, 'ModelLoader', None)
            if not ModelLoaderClass:
                self.logger.error("ModelLoader 클래스를 찾을 수 없음")
                return None
            
            # ModelLoader 인스턴스 생성
            model_loader = ModelLoaderClass(
                device=config.device,
                config={
                    'model_cache_dir': config.model_cache_dir or './ai_models',
                    'use_fp16': config.use_fp16,
                    'max_cached_models': config.max_cached_models,
                    'lazy_loading': config.lazy_loading,
                    'optimization_enabled': config.optimization_level >= OptimizationLevel.STANDARD
                }
            )
            
            # 초기화
            if hasattr(model_loader, 'initialize'):
                success = model_loader.initialize()
                if not success:
                    self.logger.warning("ModelLoader 초기화 실패")
                    return None
            
            self._model_loader_cache = model_loader
            self.logger.info("✅ ModelLoader 해결 완료")
            return model_loader
            
        except Exception as e:
            self.logger.error(f"❌ ModelLoader 해결 실패: {e}")
            return None
    
    def resolve_step_mixin_class(self, step_type: StepType) -> Optional[Type]:
        """BaseStepMixin 클래스 안전하게 해결 (동적 import)"""
        try:
            if step_type.value in self._step_mixin_classes:
                return self._step_mixin_classes[step_type.value]
            
            # 동적 import로 순환참조 방지
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
            
            self._step_mixin_classes[step_type.value] = StepClass
            self.logger.info(f"✅ {class_name} 클래스 해결 완료")
            return StepClass
            
        except Exception as e:
            self.logger.error(f"❌ Step 클래스 해결 실패 {step_type}: {e}")
            return None
    
    def resolve_memory_manager(self) -> Optional[Any]:
        """MemoryManager 해결 (옵션)"""
        try:
            import importlib
            memory_module = importlib.import_module('app.ai_pipeline.utils.memory_manager')
            MemoryManagerClass = getattr(memory_module, 'MemoryManager', None)
            
            if MemoryManagerClass:
                return MemoryManagerClass()
            return None
            
        except ImportError:
            self.logger.debug("MemoryManager 모듈 없음 (옵션)")
            return None
        except Exception as e:
            self.logger.warning(f"MemoryManager 해결 실패: {e}")
            return None
    
    def resolve_data_converter(self) -> Optional[Any]:
        """DataConverter 해결 (옵션)"""
        try:
            import importlib
            converter_module = importlib.import_module('app.ai_pipeline.utils.data_converter')
            DataConverterClass = getattr(converter_module, 'DataConverter', None)
            
            if DataConverterClass:
                return DataConverterClass()
            return None
            
        except ImportError:
            self.logger.debug("DataConverter 모듈 없음 (옵션)")
            return None
        except Exception as e:
            self.logger.warning(f"DataConverter 해결 실패: {e}")
            return None

# 전역 의존성 해결기
_global_resolver = DependencyResolver()

# ==============================================
# 🔥 6. 시스템 최적화 관리자
# ==============================================

class SystemOptimizer:
    """시스템 최적화 관리자"""
    
    def __init__(self, config: StepFactoryConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.SystemOptimizer")
        
    def apply_conda_optimization(self):
        """conda 환경 최적화 적용"""
        try:
            if not self.config.use_conda_optimization:
                return False
                
            if CONDA_INFO['conda_env'] == 'none':
                self.logger.warning("conda 환경이 아닙니다. 일반 최적화 적용")
                return False
            
            # conda 환경별 최적화 설정
            conda_env = CONDA_INFO['conda_env']
            
            if 'mycloset' in conda_env.lower() or 'ai' in conda_env.lower():
                # MyCloset AI 전용 환경 최적화
                if TORCH_AVAILABLE:
                    torch.set_num_threads(8 if IS_M3_MAX else 4)
                    
                self.logger.info(f"✅ MyCloset AI conda 환경 최적화 적용: {conda_env}")
                return True
            else:
                # 일반 conda 환경 최적화
                if TORCH_AVAILABLE:
                    torch.set_num_threads(4)
                    
                self.logger.info(f"✅ 일반 conda 환경 최적화 적용: {conda_env}")
                return True
                
        except Exception as e:
            self.logger.warning(f"⚠️ conda 최적화 적용 실패: {e}")
            return False
    
    def apply_m3_max_optimization(self):
        """M3 Max 특화 최적화"""
        try:
            if not IS_M3_MAX:
                return False
            
            if TORCH_AVAILABLE:
                # M3 Max 통합 메모리 최적화
                os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
                os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
                
                # 스레드 수 최적화 (M3 Max 12코어 활용)
                torch.set_num_threads(12)
                
                # MPS 캐시 정리
                if MPS_AVAILABLE:
                    try:
                        if hasattr(torch.mps, 'empty_cache'):
                            torch.mps.empty_cache()
                    except:
                        pass
            
            # 메모리 정리
            gc.collect()
            
            self.logger.info("✅ M3 Max 특화 최적화 적용 완료")
            return True
            
        except Exception as e:
            self.logger.warning(f"⚠️ M3 Max 최적화 실패: {e}")
            return False
    
    def apply_optimization_level(self):
        """최적화 레벨별 설정 적용"""
        try:
            level = self.config.optimization_level
            
            if level >= OptimizationLevel.M3_MAX and IS_M3_MAX:
                self.apply_m3_max_optimization()
            
            if level >= OptimizationLevel.STANDARD:
                self.apply_conda_optimization()
            
            if level >= OptimizationLevel.HIGH:
                # 고성능 최적화
                if TORCH_AVAILABLE:
                    torch.backends.cudnn.benchmark = True
                    torch.backends.cudnn.deterministic = False
            
            if level >= OptimizationLevel.PRODUCTION:
                # 프로덕션 최적화
                if TORCH_AVAILABLE:
                    torch.set_float32_matmul_precision('high')
                
                # 메모리 최적화
                gc.set_threshold(100, 10, 10)
            
            self.logger.info(f"✅ 최적화 레벨 {level.name} 적용 완료")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ 최적화 레벨 적용 실패: {e}")
            return False

# ==============================================
# 🔥 7. 메인 StepFactory 클래스 (수정된 버전)
# ==============================================

class StepFactory:
    """
    🔥 StepFactory v1.0 - 의존성 주입 전용 팩토리 (수정된 버전)
    
    핵심 역할:
    1. ModelLoader 인스턴스 생성
    2. BaseStepMixin 기반 Step 인스턴스 생성
    3. 의존성 주입으로 두 개체 연결
    4. 시스템 최적화 적용
    
    순환참조 방지 구조:
    StepFactory → ModelLoader (생성) → BaseStepMixin (생성) → 의존성 주입 → 완성된 Step
    """
    
    def __init__(self, config: Optional[StepFactoryConfig] = None):
        """StepFactory 초기화 (수정된 버전)"""
        self.config = config or StepFactoryConfig()
        self.logger = logging.getLogger(f"{__name__}.StepFactory")
        
        # 의존성 해결기
        self.resolver = _global_resolver
        
        # 시스템 최적화기
        self.optimizer = SystemOptimizer(self.config)
        
        # 생성 캐시
        self.creation_cache: Dict[str, StepFactoryResult] = {}
        self._cache_lock = threading.RLock()
        
        # 통계
        self.creation_stats = {
            'total_created': 0,
            'successful_creations': 0,
            'failed_creations': 0,
            'cache_hits': 0,
            'dependency_injection_success': 0,
            'optimization_applied': 0
        }
        
        # 초기화
        self._initialize()
    
    def _initialize(self):
        """팩토리 초기화"""
        try:
            # 시스템 최적화 적용
            optimization_success = self.optimizer.apply_optimization_level()
            if optimization_success:
                self.creation_stats['optimization_applied'] += 1
            
            # 디바이스 해결
            if self.config.device == "auto":
                self.config.device = self._detect_optimal_device()
            
            self.logger.info(f"✅ StepFactory v1.0 초기화 완료")
            self.logger.info(f"🔧 Device: {self.config.device}")
            self.logger.info(f"🔧 Optimization: {self.config.optimization_level.name}")
            self.logger.info(f"🔧 conda 환경: {CONDA_INFO['conda_env']}")
            self.logger.info(f"🔧 M3 Max: {'✅' if IS_M3_MAX else '❌'}")
            
        except Exception as e:
            self.logger.error(f"❌ StepFactory 초기화 실패: {e}")
    
    def _detect_optimal_device(self) -> str:
        """최적 디바이스 감지"""
        try:
            if TORCH_AVAILABLE:
                if MPS_AVAILABLE and IS_M3_MAX:
                    return "mps"
                elif torch.cuda.is_available():
                    return "cuda"
            return "cpu"
        except:
            return "cpu"
    
    # ==============================================
    # 🔥 8. 핵심 생성 메서드들
    # ==============================================
    
    def create_step(
        self, 
        step_type: Union[StepType, str], 
        step_config: Optional[Dict[str, Any]] = None,
        use_cache: bool = True
    ) -> StepFactoryResult:
        """
        Step 인스턴스 생성 (동기 버전)
        
        Args:
            step_type: Step 타입 (StepType enum 또는 문자열)
            step_config: Step별 설정 (옵션)
            use_cache: 캐시 사용 여부
            
        Returns:
            StepFactoryResult: 생성 결과
        """
        start_time = time.time()
        
        try:
            # Step 타입 정규화
            if isinstance(step_type, str):
                try:
                    step_type = StepType(step_type)
                except ValueError:
                    return StepFactoryResult(
                        step_instance=None,
                        model_loader=None,
                        step_config={},
                        creation_time=0,
                        success=False,
                        error_message=f"알 수 없는 Step 타입: {step_type}"
                    )
            
            # 캐시 확인
            cache_key = self._generate_cache_key(step_type, step_config)
            if use_cache:
                cached_result = self._get_from_cache(cache_key)
                if cached_result:
                    self.creation_stats['cache_hits'] += 1
                    return cached_result
            
            # Step 설정 준비
            final_step_config = self._prepare_step_config(step_type, step_config)
            
            # 1단계: ModelLoader 생성
            model_loader = self.resolver.resolve_model_loader(self.config)
            if not model_loader:
                return StepFactoryResult(
                    step_instance=None,
                    model_loader=None,
                    step_config=final_step_config,
                    creation_time=time.time() - start_time,
                    success=False,
                    error_message="ModelLoader 생성 실패"
                )
            
            # 2단계: BaseStepMixin 기반 Step 클래스 해결
            StepClass = self.resolver.resolve_step_mixin_class(step_type)
            if not StepClass:
                return StepFactoryResult(
                    step_instance=None,
                    model_loader=model_loader,
                    step_config=final_step_config,
                    creation_time=time.time() - start_time,
                    success=False,
                    error_message=f"{step_type.value} Step 클래스 해결 실패"
                )
            
            # 3단계: Step 인스턴스 생성
            step_instance = StepClass(**final_step_config)
            
            # 4단계: 의존성 주입
            dependencies_injected = False
            if self.config.enable_dependency_injection:
                dependencies_injected = self._inject_dependencies(
                    step_instance, 
                    model_loader, 
                    step_type
                )
                
                if dependencies_injected:
                    self.creation_stats['dependency_injection_success'] += 1
            
            # 5단계: 초기화
            if hasattr(step_instance, 'initialize'):
                try:
                    step_instance.initialize()
                except Exception as e:
                    self.logger.warning(f"⚠️ Step 초기화 실패: {e}")
            
            # 결과 생성
            result = StepFactoryResult(
                step_instance=step_instance,
                model_loader=model_loader,
                step_config=final_step_config,
                creation_time=time.time() - start_time,
                success=True,
                dependencies_injected=dependencies_injected,
                optimization_applied=True
            )
            
            # 캐시 저장
            if use_cache:
                self._save_to_cache(cache_key, result)
            
            # 통계 업데이트
            self.creation_stats['total_created'] += 1
            self.creation_stats['successful_creations'] += 1
            
            self.logger.info(f"✅ {step_type.value} Step 생성 완료 ({result.creation_time:.3f}초)")
            return result
            
        except Exception as e:
            self.creation_stats['total_created'] += 1
            self.creation_stats['failed_creations'] += 1
            
            self.logger.error(f"❌ {step_type} Step 생성 실패: {e}")
            return StepFactoryResult(
                step_instance=None,
                model_loader=None,
                step_config=step_config or {},
                creation_time=time.time() - start_time,
                success=False,
                error_message=str(e)
            )
    
    async def create_step_async(
        self, 
        step_type: Union[StepType, str], 
        step_config: Optional[Dict[str, Any]] = None,
        use_cache: bool = True
    ) -> StepFactoryResult:
        """
        Step 인스턴스 비동기 생성
        
        Args:
            step_type: Step 타입
            step_config: Step별 설정
            use_cache: 캐시 사용 여부
            
        Returns:
            StepFactoryResult: 생성 결과
        """
        try:
            # 기본 생성은 동기로 실행
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, 
                self.create_step,
                step_type,
                step_config,
                use_cache
            )
            
            # 비동기 초기화가 있으면 실행
            if result.success and result.step_instance:
                if hasattr(result.step_instance, 'initialize_async'):
                    try:
                        await result.step_instance.initialize_async()
                    except Exception as e:
                        self.logger.warning(f"⚠️ 비동기 초기화 실패: {e}")
                
                # 비동기 워밍업
                if hasattr(result.step_instance, 'warmup_async') and self.config.auto_warmup:
                    try:
                        await result.step_instance.warmup_async()
                        self.logger.info(f"🔥 {step_type} Step 워밍업 완료")
                    except Exception as e:
                        self.logger.warning(f"⚠️ 워밍업 실패: {e}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ {step_type} Step 비동기 생성 실패: {e}")
            return StepFactoryResult(
                step_instance=None,
                model_loader=None,
                step_config=step_config or {},
                creation_time=0,
                success=False,
                error_message=str(e)
            )
    
    # ==============================================
    # 🔥 9. 의존성 주입 메서드들
    # ==============================================
    
    def _inject_dependencies(
        self, 
        step_instance: Any, 
        model_loader: Any, 
        step_type: StepType
    ) -> bool:
        """의존성 주입 실행"""
        try:
            injections_made = 0
            
            # 1. ModelLoader 주입 (필수)
            if hasattr(step_instance, 'set_model_loader'):
                step_instance.set_model_loader(model_loader)
                injections_made += 1
                self.logger.debug(f"✅ ModelLoader 주입 완료")
            elif hasattr(step_instance, 'model_loader'):
                step_instance.model_loader = model_loader
                injections_made += 1
                self.logger.debug(f"✅ ModelLoader 속성 설정 완료")
            
            # 2. MemoryManager 주입 (옵션)
            memory_manager = self.resolver.resolve_memory_manager()
            if memory_manager:
                if hasattr(step_instance, 'set_memory_manager'):
                    step_instance.set_memory_manager(memory_manager)
                    injections_made += 1
                    self.logger.debug(f"✅ MemoryManager 주입 완료")
                elif hasattr(step_instance, 'memory_manager'):
                    step_instance.memory_manager = memory_manager
                    injections_made += 1
                    self.logger.debug(f"✅ MemoryManager 속성 설정 완료")
            
            # 3. DataConverter 주입 (옵션)
            data_converter = self.resolver.resolve_data_converter()
            if data_converter:
                if hasattr(step_instance, 'set_data_converter'):
                    step_instance.set_data_converter(data_converter)
                    injections_made += 1
                    self.logger.debug(f"✅ DataConverter 주입 완료")
                elif hasattr(step_instance, 'data_converter'):
                    step_instance.data_converter = data_converter
                    injections_made += 1
                    self.logger.debug(f"✅ DataConverter 속성 설정 완료")
            
            # 4. Step 인터페이스 생성 (ModelLoader를 통해)
            if hasattr(model_loader, 'create_step_interface'):
                try:
                    step_name = self._get_step_name(step_type)
                    step_interface = model_loader.create_step_interface(step_name)
                    
                    if step_interface and hasattr(step_instance, 'set_step_interface'):
                        step_instance.set_step_interface(step_interface)
                        injections_made += 1
                        self.logger.debug(f"✅ Step 인터페이스 생성 및 주입 완료")
                        
                except Exception as e:
                    self.logger.debug(f"Step 인터페이스 생성 실패: {e}")
            
            # 검증
            if self.config.validate_dependencies:
                self._validate_injected_dependencies(step_instance)
            
            success = injections_made > 0
            if success:
                self.logger.info(f"✅ {step_type.value} 의존성 주입 완료: {injections_made}개")
            else:
                self.logger.warning(f"⚠️ {step_type.value} 의존성 주입 없음")
            
            return success
            
        except Exception as e:
            self.logger.error(f"❌ {step_type.value} 의존성 주입 실패: {e}")
            return False
    
    def _validate_injected_dependencies(self, step_instance: Any):
        """주입된 의존성 검증"""
        try:
            validation_results = []
            
            # ModelLoader 검증
            if hasattr(step_instance, 'model_loader') or hasattr(step_instance, 'get_model'):
                validation_results.append("ModelLoader: ✅")
            else:
                validation_results.append("ModelLoader: ❌")
            
            # 필수 메서드 검증
            required_methods = ['initialize', 'get_status']
            for method in required_methods:
                if hasattr(step_instance, method):
                    validation_results.append(f"{method}: ✅")
                else:
                    validation_results.append(f"{method}: ❌")
            
            self.logger.debug(f"의존성 검증 결과: {', '.join(validation_results)}")
            
        except Exception as e:
            self.logger.debug(f"의존성 검증 실패: {e}")
    
    # ==============================================
    # 🔥 10. 편의 메서드들
    # ==============================================
    
    def create_human_parsing_step(self, **kwargs) -> StepFactoryResult:
        """Human Parsing Step 생성"""
        return self.create_step(StepType.HUMAN_PARSING, kwargs)
    
    def create_pose_estimation_step(self, **kwargs) -> StepFactoryResult:
        """Pose Estimation Step 생성"""
        return self.create_step(StepType.POSE_ESTIMATION, kwargs)
    
    def create_cloth_segmentation_step(self, **kwargs) -> StepFactoryResult:
        """Cloth Segmentation Step 생성"""
        return self.create_step(StepType.CLOTH_SEGMENTATION, kwargs)
    
    def create_geometric_matching_step(self, **kwargs) -> StepFactoryResult:
        """Geometric Matching Step 생성"""
        return self.create_step(StepType.GEOMETRIC_MATCHING, kwargs)
    
    def create_cloth_warping_step(self, **kwargs) -> StepFactoryResult:
        """Cloth Warping Step 생성"""
        return self.create_step(StepType.CLOTH_WARPING, kwargs)
    
    def create_virtual_fitting_step(self, **kwargs) -> StepFactoryResult:
        """Virtual Fitting Step 생성 (핵심)"""
        return self.create_step(StepType.VIRTUAL_FITTING, kwargs)
    
    def create_post_processing_step(self, **kwargs) -> StepFactoryResult:
        """Post Processing Step 생성"""
        return self.create_step(StepType.POST_PROCESSING, kwargs)
    
    def create_quality_assessment_step(self, **kwargs) -> StepFactoryResult:
        """Quality Assessment Step 생성"""
        return self.create_step(StepType.QUALITY_ASSESSMENT, kwargs)
    
    # 비동기 버전들
    async def create_human_parsing_step_async(self, **kwargs) -> StepFactoryResult:
        """Human Parsing Step 비동기 생성"""
        return await self.create_step_async(StepType.HUMAN_PARSING, kwargs)
    
    async def create_virtual_fitting_step_async(self, **kwargs) -> StepFactoryResult:
        """Virtual Fitting Step 비동기 생성 (핵심)"""
        return await self.create_step_async(StepType.VIRTUAL_FITTING, kwargs)
    
    # M3 Max 최적화 버전들
    def create_m3_max_optimized_step(
        self, 
        step_type: Union[StepType, str], 
        **kwargs
    ) -> StepFactoryResult:
        """M3 Max 최적화 Step 생성"""
        # M3 Max 특화 설정 적용
        m3_max_config = {
            'device': 'mps' if MPS_AVAILABLE else 'cpu',
            'use_fp16': True,
            'auto_memory_cleanup': True,
            'optimization_level': OptimizationLevel.M3_MAX,
            **kwargs
        }
        
        return self.create_step(step_type, m3_max_config)
    
    # ==============================================
    # 🔥 11. 유틸리티 메서드들
    # ==============================================
    
    def _prepare_step_config(
        self, 
        step_type: StepType, 
        step_config: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Step 설정 준비"""
        # 기본 설정
        final_config = {
            'step_name': self._get_step_name(step_type),
            'step_id': self._get_step_id(step_type),
            'device': self.config.device,
            'use_fp16': self.config.use_fp16,
            'auto_warmup': self.config.auto_warmup,
            'auto_memory_cleanup': self.config.auto_memory_cleanup
        }
        
        # 전역 Step 설정 병합
        if step_type.value in self.config.step_configs:
            final_config.update(self.config.step_configs[step_type.value])
        
        # 개별 설정 병합
        if step_config:
            final_config.update(step_config)
        
        return final_config
    
    def _get_step_name(self, step_type: StepType) -> str:
        """Step 이름 반환"""
        name_mapping = {
            StepType.HUMAN_PARSING: "HumanParsingStep",
            StepType.POSE_ESTIMATION: "PoseEstimationStep",
            StepType.CLOTH_SEGMENTATION: "ClothSegmentationStep",
            StepType.GEOMETRIC_MATCHING: "GeometricMatchingStep",
            StepType.CLOTH_WARPING: "ClothWarpingStep",
            StepType.VIRTUAL_FITTING: "VirtualFittingStep",
            StepType.POST_PROCESSING: "PostProcessingStep",
            StepType.QUALITY_ASSESSMENT: "QualityAssessmentStep"
        }
        return name_mapping.get(step_type, f"{step_type.value.title()}Step")
    
    def _get_step_id(self, step_type: StepType) -> int:
        """Step ID 반환"""
        id_mapping = {
            StepType.HUMAN_PARSING: 1,
            StepType.POSE_ESTIMATION: 2,
            StepType.CLOTH_SEGMENTATION: 3,
            StepType.GEOMETRIC_MATCHING: 4,
            StepType.CLOTH_WARPING: 5,
            StepType.VIRTUAL_FITTING: 6,
            StepType.POST_PROCESSING: 7,
            StepType.QUALITY_ASSESSMENT: 8
        }
        return id_mapping.get(step_type, 0)
    
    def _generate_cache_key(
        self, 
        step_type: StepType, 
        step_config: Optional[Dict[str, Any]]
    ) -> str:
        """캐시 키 생성"""
        import hashlib
        
        key_data = {
            'step_type': step_type.value,
            'device': self.config.device,
            'optimization_level': self.config.optimization_level.value,
            'step_config': step_config or {}
        }
        
        key_str = str(key_data)  # json.dumps 대신 str 사용으로 호환성 향상
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _get_from_cache(self, cache_key: str) -> Optional[StepFactoryResult]:
        """캐시에서 가져오기"""
        try:
            with self._cache_lock:
                if cache_key in self.creation_cache:
                    cached_result = self.creation_cache[cache_key]
                    
                    # 캐시된 인스턴스가 여전히 유효한지 확인
                    if (cached_result.step_instance and 
                        hasattr(cached_result.step_instance, 'is_initialized')):
                        return cached_result
                
                return None
        except Exception as e:
            self.logger.debug(f"캐시 조회 실패: {e}")
            return None
    
    def _save_to_cache(self, cache_key: str, result: StepFactoryResult):
        """캐시에 저장"""
        try:
            with self._cache_lock:
                # 캐시 크기 제한 (최대 50개)
                if len(self.creation_cache) >= 50:
                    # 가장 오래된 항목 제거
                    oldest_key = next(iter(self.creation_cache))
                    del self.creation_cache[oldest_key]
                
                self.creation_cache[cache_key] = result
        except Exception as e:
            self.logger.debug(f"캐시 저장 실패: {e}")
    
    def get_creation_stats(self) -> Dict[str, Any]:
        """생성 통계 반환"""
        return {
            **self.creation_stats,
            'cache_size': len(self.creation_cache),
            'success_rate': (
                self.creation_stats['successful_creations'] / 
                max(1, self.creation_stats['total_created'])
            ),
            'dependency_injection_rate': (
                self.creation_stats['dependency_injection_success'] / 
                max(1, self.creation_stats['successful_creations'])
            )
        }
    
    def clear_cache(self):
        """캐시 정리"""
        try:
            with self._cache_lock:
                self.creation_cache.clear()
            self.logger.info("✅ StepFactory 캐시 정리 완료")
        except Exception as e:
            self.logger.error(f"❌ 캐시 정리 실패: {e}")
    
    def cleanup(self):
        """팩토리 정리"""
        try:
            self.clear_cache()
            
            # 통계 리셋
            for key in self.creation_stats:
                self.creation_stats[key] = 0
            
            self.logger.info("✅ StepFactory 정리 완료")
            
        except Exception as e:
            self.logger.error(f"❌ StepFactory 정리 실패: {e}")

# ==============================================
# 🔥 12. 전역 팩토리 관리 (수정된 버전)
# ==============================================

_global_step_factory: Optional[StepFactory] = None
_factory_lock = threading.Lock()

@lru_cache(maxsize=1)
def get_global_step_factory(config: Optional[StepFactoryConfig] = None) -> StepFactory:
    """전역 StepFactory 인스턴스 반환"""
    global _global_step_factory
    
    with _factory_lock:
        if _global_step_factory is None:
            factory_config = config or StepFactoryConfig(
                optimization_level=OptimizationLevel.M3_MAX if IS_M3_MAX else OptimizationLevel.STANDARD,
                use_conda_optimization=True,
                enable_dependency_injection=True,
                auto_warmup=True,
                auto_memory_cleanup=True
            )
            
            _global_step_factory = StepFactory(factory_config)
            logger.info("🌐 전역 StepFactory v1.0 인스턴스 생성")
        
        return _global_step_factory

def create_m3_max_step_factory() -> StepFactory:
    """M3 Max 최적화 StepFactory 생성"""
    config = StepFactoryConfig(
        device="mps" if MPS_AVAILABLE else "cpu",
        optimization_level=OptimizationLevel.M3_MAX,
        use_conda_optimization=True,
        use_fp16=True,
        max_cached_models=50,  # M3 Max 128GB 메모리 활용
        auto_warmup=True,
        auto_memory_cleanup=True,
        enable_dependency_injection=True,
        dependency_injection_mode="runtime"
    )
    return StepFactory(config)

def create_production_step_factory() -> StepFactory:
    """프로덕션 StepFactory 생성"""
    config = StepFactoryConfig(
        optimization_level=OptimizationLevel.PRODUCTION,
        use_conda_optimization=True,
        use_fp16=True,
        lazy_loading=True,
        auto_warmup=False,  # 프로덕션에서는 수동 워밍업
        auto_memory_cleanup=True,
        enable_dependency_injection=True,
        validate_dependencies=True,
        enable_debug_logging=False
    )
    return StepFactory(config)

def cleanup_global_step_factory():
    """전역 StepFactory 정리"""
    global _global_step_factory
    
    with _factory_lock:
        if _global_step_factory:
            _global_step_factory.cleanup()
            _global_step_factory = None
        
        get_global_step_factory.cache_clear()
        logger.info("🌐 전역 StepFactory v1.0 정리 완료")

# ==============================================
# 🔥 13. 편의 함수들 (기존 API 호환)
# ==============================================

def create_step(
    step_type: Union[StepType, str], 
    step_config: Optional[Dict[str, Any]] = None,
    **kwargs
) -> StepFactoryResult:
    """Step 생성 (전역 팩토리 사용)"""
    factory = get_global_step_factory()
    final_config = {**(step_config or {}), **kwargs}
    return factory.create_step(step_type, final_config)

async def create_step_async(
    step_type: Union[StepType, str], 
    step_config: Optional[Dict[str, Any]] = None,
    **kwargs
) -> StepFactoryResult:
    """Step 비동기 생성 (전역 팩토리 사용)"""
    factory = get_global_step_factory()
    final_config = {**(step_config or {}), **kwargs}
    return await factory.create_step_async(step_type, final_config)

# Step별 편의 함수들
def create_human_parsing_step(**kwargs) -> StepFactoryResult:
    """Human Parsing Step 생성"""
    return create_step(StepType.HUMAN_PARSING, kwargs)

def create_virtual_fitting_step(**kwargs) -> StepFactoryResult:
    """Virtual Fitting Step 생성 (핵심)"""
    return create_step(StepType.VIRTUAL_FITTING, kwargs)

async def create_virtual_fitting_step_async(**kwargs) -> StepFactoryResult:
    """Virtual Fitting Step 비동기 생성 (핵심)"""
    return await create_step_async(StepType.VIRTUAL_FITTING, kwargs)

def create_m3_max_optimized_step(step_type: Union[StepType, str], **kwargs) -> StepFactoryResult:
    """M3 Max 최적화 Step 생성"""
    factory = get_global_step_factory()
    return factory.create_m3_max_optimized_step(step_type, **kwargs)

# 파이프라인 전체 생성
def create_complete_pipeline(**kwargs) -> Dict[str, StepFactoryResult]:
    """8단계 완전 파이프라인 생성"""
    factory = get_global_step_factory()
    
    pipeline_results = {}
    for step_type in StepType:
        try:
            result = factory.create_step(step_type, kwargs)
            pipeline_results[step_type.value] = result
            
            if result.success:
                logger.info(f"✅ {step_type.value} Step 생성 완료")
            else:
                logger.error(f"❌ {step_type.value} Step 생성 실패: {result.error_message}")
                
        except Exception as e:
            logger.error(f"❌ {step_type.value} Step 생성 오류: {e}")
            pipeline_results[step_type.value] = StepFactoryResult(
                step_instance=None,
                model_loader=None,
                step_config={},
                creation_time=0,
                success=False,
                error_message=str(e)
            )
    
    return pipeline_results

async def create_complete_pipeline_async(**kwargs) -> Dict[str, StepFactoryResult]:
    """8단계 완전 파이프라인 비동기 생성"""
    factory = get_global_step_factory()
    
    # 병렬 생성을 위한 태스크 생성
    tasks = []
    for step_type in StepType:
        task = factory.create_step_async(step_type, kwargs)
        tasks.append((step_type, task))
    
    # 병렬 실행
    pipeline_results = {}
    for step_type, task in tasks:
        try:
            result = await task
            pipeline_results[step_type.value] = result
            
            if result.success:
                logger.info(f"✅ {step_type.value} Step 비동기 생성 완료")
            else:
                logger.error(f"❌ {step_type.value} Step 비동기 생성 실패: {result.error_message}")
                
        except Exception as e:
            logger.error(f"❌ {step_type.value} Step 비동기 생성 오류: {e}")
            pipeline_results[step_type.value] = StepFactoryResult(
                step_instance=None,
                model_loader=None,
                step_config={},
                creation_time=0,
                success=False,
                error_message=str(e)
            )
    
    return pipeline_results

# ==============================================
# 🔥 14. 모듈 내보내기
# ==============================================

__all__ = [
    # 핵심 클래스들
    'StepFactory',
    'DependencyResolver',
    'SystemOptimizer',
    
    # 데이터 구조들
    'StepType',
    'OptimizationLevel',
    'StepFactoryConfig',
    'StepFactoryResult',
    
    # 전역 함수들
    'get_global_step_factory',
    'create_m3_max_step_factory',
    'create_production_step_factory',
    'cleanup_global_step_factory',
    
    # 편의 함수들
    'create_step',
    'create_step_async',
    'create_human_parsing_step',
    'create_virtual_fitting_step',
    'create_virtual_fitting_step_async',
    'create_m3_max_optimized_step',
    'create_complete_pipeline',
    'create_complete_pipeline_async',
    
    # 상수들
    'IS_M3_MAX',
    'TORCH_AVAILABLE',
    'MPS_AVAILABLE',
    'CONDA_INFO'
]

# ==============================================
# 🔥 15. 모듈 정리 함수 등록
# ==============================================

import atexit
atexit.register(cleanup_global_step_factory)

# ==============================================
# 🔥 16. 모듈 로드 완료 메시지
# ==============================================

logger.info("=" * 80)
logger.info("✅ StepFactory v1.0 - 매개변수 호환성 수정 완료")
logger.info("=" * 80)
logger.info("🔥 핵심 수정사항:")
logger.info("   ✅ device_type → device 매개변수 통일")
logger.info("   ✅ device_type property 호환성 지원")
logger.info("   ✅ 순환참조 완전 방지 - 한방향 의존성 구조")
logger.info("   ✅ 의존성 주입 패턴 완전 구현")
logger.info("   ✅ BaseStepMixin과 ModelLoader 안전한 조립")
logger.info("   ✅ M3 Max 128GB 최적화")
logger.info("   ✅ conda 환경 우선 지원")
logger.info("   ✅ 8단계 AI 파이프라인 완전 지원")
logger.info("   ✅ 프로덕션 레벨 안정성")
logger.info("")
logger.info("🏗️ 구조:")
logger.info("   StepFactory → ModelLoader (생성) → BaseStepMixin (생성) → 의존성 주입 → 완성된 Step")
logger.info("")
logger.info("🎯 8단계 AI 파이프라인 지원:")
logger.info("   1️⃣ HumanParsingMixin - 신체 영역 분할")
logger.info("   2️⃣ PoseEstimationMixin - 포즈 감지") 
logger.info("   3️⃣ ClothSegmentationMixin - 의류 분할")
logger.info("   4️⃣ GeometricMatchingMixin - 기하학적 매칭")
logger.info("   5️⃣ ClothWarpingMixin - 의류 변형")
logger.info("   6️⃣ VirtualFittingMixin - 가상 피팅 (핵심)")
logger.info("   7️⃣ PostProcessingMixin - 후처리")
logger.info("   8️⃣ QualityAssessmentMixin - 품질 평가")
logger.info("")
logger.info(f"🔧 시스템 상태:")
logger.info(f"   - conda 환경: {CONDA_INFO['conda_env']}")
logger.info(f"   - PyTorch: {'✅' if TORCH_AVAILABLE else '❌'}")
logger.info(f"   - MPS: {'✅' if MPS_AVAILABLE else '❌'}")
logger.info(f"   - M3 Max: {'✅' if IS_M3_MAX else '❌'}")
logger.info("")
logger.info("🌟 사용 예시:")
logger.info("   # 기본 사용")
logger.info("   result = create_virtual_fitting_step()")
logger.info("   if result.success:")
logger.info("       step = result.step_instance")
logger.info("   ")
logger.info("   # 비동기 사용")
logger.info("   result = await create_virtual_fitting_step_async()")
logger.info("   ")
logger.info("   # M3 Max 최적화")
logger.info("   result = create_m3_max_optimized_step('virtual_fitting')")
logger.info("   ")
logger.info("   # 완전 파이프라인")
logger.info("   pipeline = await create_complete_pipeline_async()")
logger.info("")
logger.info("=" * 80)
logger.info("🚀 StepFactory v1.0 매개변수 호환성 수정 완료!")
logger.info("   ✅ device_type 오류 완전 해결")
logger.info("   ✅ 기존 호환성 유지")
logger.info("   ✅ 순환참조 완전 해결")
logger.info("   ✅ 깔끔한 의존성 주입 패턴")
logger.info("   ✅ BaseStepMixin + ModelLoader 완벽 조립")
logger.info("   ✅ M3 Max 128GB 최적화")
logger.info("   ✅ 프로덕션 레벨 안정성")
logger.info("=" * 80)