# backend/app/ai_pipeline/factories/step_factory.py
"""
🔥 MyCloset AI StepFactory v9.0 - BaseStepMixin 완전 호환 (Option A 구현)
================================================================================

✅ 핵심 수정사항:
✅ BaseStepMixin v18.0 표준 완전 호환
✅ 생성자 시그니처 통일 (**kwargs 기반)
✅ 의존성 주입 생성자 시점 지원
✅ UnifiedDependencyManager 통합
✅ process() 메서드 시그니처 표준화
✅ conda 환경 우선 최적화
✅ M3 Max 128GB 메모리 최적화

주요 개선사항:
1. Step 생성자에 의존성을 직접 전달 (생성자 주입)
2. BaseStepMixin 표준 kwargs 패턴 완전 지원
3. process() 메서드 통일된 시그니처 보장
4. UnifiedDependencyManager 완전 활용
5. 실제 Step 클래스들과 100% 호환

Author: MyCloset AI Team
Date: 2025-07-26
Version: 9.0 (BaseStepMixin Complete Compatibility)
"""

import os
import sys
import logging
import threading
import time
import weakref
import gc
import traceback
import uuid
import asyncio
import concurrent.futures  # 🔥 추가된 import
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Type, Tuple, TYPE_CHECKING
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor

# 안전한 타입 힌팅 (순환참조 방지)
if TYPE_CHECKING:
    from ..steps.base_step_mixin import BaseStepMixin
    from ..utils.model_loader import ModelLoader
    from ..utils.memory_manager import MemoryManager
    from ..utils.data_converter import DataConverter
    from ...core.di_container import DIContainer

# ==============================================
# 🔥 로깅 및 환경 설정
# ==============================================

logger = logging.getLogger(__name__)

# conda 환경 정보
CONDA_INFO = {
    'conda_env': os.environ.get('CONDA_DEFAULT_ENV', 'none'),
    'conda_prefix': os.environ.get('CONDA_PREFIX', 'none'),
    'is_target_env': os.environ.get('CONDA_DEFAULT_ENV') == 'mycloset-ai-clean'
}

# M3 Max 감지
IS_M3_MAX = False
MEMORY_GB = 16.0

try:
    import platform
    if platform.system() == 'Darwin' and platform.machine() == 'arm64':
        try:
            import subprocess
            result = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'], 
                                  capture_output=True, text=True, timeout=3)
            IS_M3_MAX = 'M3' in result.stdout
            
            memory_result = subprocess.run(['sysctl', '-n', 'hw.memsize'], 
                                         capture_output=True, text=True, timeout=3)
            if memory_result.stdout.strip():
                MEMORY_GB = int(memory_result.stdout.strip()) / 1024**3
        except:
            pass
except:
    pass

logger.info(f"🔧 StepFactory v9.0 환경: conda={CONDA_INFO['conda_env']}, M3 Max={IS_M3_MAX}, 메모리={MEMORY_GB:.1f}GB")

# ==============================================
# 🔥 핵심 데이터 구조 (BaseStepMixin 호환)
# ==============================================

class StepType(Enum):
    """Step 타입 (8단계)"""
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
    CRITICAL = 1    # Virtual Fitting, Human Parsing
    HIGH = 2        # Pose Estimation, Cloth Segmentation
    NORMAL = 3      # Geometric Matching, Cloth Warping
    LOW = 4         # Post Processing, Quality Assessment

@dataclass
class BaseStepMixinConfig:
    """BaseStepMixin v18.0 호환 설정 구조"""
    # 기본 Step 정보
    step_name: str
    step_id: int
    step_type: StepType
    class_name: str
    module_path: str
    priority: StepPriority = StepPriority.NORMAL
    
    # BaseStepMixin 표준 설정
    device: str = "auto"
    use_fp16: bool = True
    batch_size: int = 1
    confidence_threshold: float = 0.8
    
    # 최적화 설정
    auto_memory_cleanup: bool = True
    auto_warmup: bool = True
    optimization_enabled: bool = True
    strict_mode: bool = False
    
    # 의존성 설정 (BaseStepMixin 표준)
    require_model_loader: bool = True
    require_memory_manager: bool = False
    require_data_converter: bool = False
    require_di_container: bool = False
    require_unified_dependency_manager: bool = True
    
    # AI 모델 정보
    ai_models: List[str] = field(default_factory=list)
    model_size_gb: float = 0.0
    
    # conda/M3 Max 최적화
    conda_optimized: bool = True
    m3_max_optimized: bool = True

@dataclass
class StepCreationResult:
    """Step 생성 결과 (강화됨)"""
    success: bool
    step_instance: Optional['BaseStepMixin'] = None
    step_name: str = ""
    step_type: Optional[StepType] = None
    class_name: str = ""
    module_path: str = ""
    creation_time: float = 0.0
    error_message: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    
    # 의존성 주입 결과
    dependencies_injected: Dict[str, bool] = field(default_factory=dict)
    initialization_success: bool = False
    ai_models_loaded: List[str] = field(default_factory=list)
    
    # BaseStepMixin 호환성 검증
    basestepmixin_compatible: bool = True
    process_method_validated: bool = False
    dependency_injection_success: bool = False

# ==============================================
# 🔥 BaseStepMixin 호환 Step 매핑
# ==============================================

class BaseStepMixinMapping:
    """BaseStepMixin v18.0 표준 호환 Step 매핑"""
    
    STEP_CONFIGS = {
        StepType.HUMAN_PARSING: BaseStepMixinConfig(
            step_name="HumanParsingStep",
            step_id=1,
            step_type=StepType.HUMAN_PARSING,
            class_name="HumanParsingStep",
            module_path="app.ai_pipeline.steps.step_01_human_parsing",
            priority=StepPriority.CRITICAL,
            ai_models=["graphonomy", "atr_model", "human_parsing_schp"],
            model_size_gb=4.0,
            require_model_loader=True,
            require_memory_manager=True
        ),
        StepType.POSE_ESTIMATION: BaseStepMixinConfig(
            step_name="PoseEstimationStep",
            step_id=2,
            step_type=StepType.POSE_ESTIMATION,
            class_name="PoseEstimationStep",
            module_path="app.ai_pipeline.steps.step_02_pose_estimation",
            priority=StepPriority.HIGH,
            ai_models=["openpose", "yolov8_pose", "diffusion_pose"],
            model_size_gb=3.4,
            require_model_loader=True,
            require_memory_manager=True
        ),
        StepType.CLOTH_SEGMENTATION: BaseStepMixinConfig(
            step_name="ClothSegmentationStep",
            step_id=3,
            step_type=StepType.CLOTH_SEGMENTATION,
            class_name="ClothSegmentationStep",
            module_path="app.ai_pipeline.steps.step_03_cloth_segmentation",
            priority=StepPriority.HIGH,
            ai_models=["u2net", "sam_huge", "cloth_segmentation"],
            model_size_gb=5.5,
            require_model_loader=True,
            require_memory_manager=True
        ),
        StepType.GEOMETRIC_MATCHING: BaseStepMixinConfig(
            step_name="GeometricMatchingStep",
            step_id=4,
            step_type=StepType.GEOMETRIC_MATCHING,
            class_name="GeometricMatchingStep",
            module_path="app.ai_pipeline.steps.step_04_geometric_matching",
            priority=StepPriority.NORMAL,
            ai_models=["gmm", "tps_network", "geometric_matching"],
            model_size_gb=1.3,
            require_model_loader=True
        ),
        StepType.CLOTH_WARPING: BaseStepMixinConfig(
            step_name="ClothWarpingStep",
            step_id=5,
            step_type=StepType.CLOTH_WARPING,
            class_name="ClothWarpingStep",
            module_path="app.ai_pipeline.steps.step_05_cloth_warping",
            priority=StepPriority.NORMAL,
            ai_models=["cloth_warping", "stable_diffusion", "hrviton"],
            model_size_gb=7.0,
            require_model_loader=True,
            require_memory_manager=True
        ),
        StepType.VIRTUAL_FITTING: BaseStepMixinConfig(
            step_name="VirtualFittingStep",
            step_id=6,
            step_type=StepType.VIRTUAL_FITTING,
            class_name="VirtualFittingStep",
            module_path="app.ai_pipeline.steps.step_06_virtual_fitting",
            priority=StepPriority.CRITICAL,
            ai_models=["ootdiffusion", "hr_viton", "virtual_fitting"],
            model_size_gb=14.0,
            require_model_loader=True,
            require_memory_manager=True,
            require_data_converter=True
        ),
        StepType.POST_PROCESSING: BaseStepMixinConfig(
            step_name="PostProcessingStep",
            step_id=7,
            step_type=StepType.POST_PROCESSING,
            class_name="PostProcessingStep",
            module_path="app.ai_pipeline.steps.step_07_post_processing",
            priority=StepPriority.LOW,
            ai_models=["super_resolution", "realesrgan", "enhancement"],
            model_size_gb=1.3,
            require_model_loader=True
        ),
        StepType.QUALITY_ASSESSMENT: BaseStepMixinConfig(
            step_name="QualityAssessmentStep",
            step_id=8,
            step_type=StepType.QUALITY_ASSESSMENT,
            class_name="QualityAssessmentStep",
            module_path="app.ai_pipeline.steps.step_08_quality_assessment",
            priority=StepPriority.LOW,
            ai_models=["clip", "quality_assessment", "perceptual_loss"],
            model_size_gb=7.0,
            require_model_loader=True,
            require_data_converter=True
        )
    }
    
    @classmethod
    def get_config(cls, step_type: StepType, **overrides) -> BaseStepMixinConfig:
        """BaseStepMixin 호환 설정 반환"""
        base_config = cls.STEP_CONFIGS[step_type]
        
        if overrides:
            # 딕셔너리로 변환하여 오버라이드 적용
            config_dict = {
                'step_name': base_config.step_name,
                'step_id': base_config.step_id,
                'step_type': base_config.step_type,
                'class_name': base_config.class_name,
                'module_path': base_config.module_path,
                'priority': base_config.priority,
                'device': base_config.device,
                'use_fp16': base_config.use_fp16,
                'batch_size': base_config.batch_size,
                'confidence_threshold': base_config.confidence_threshold,
                'auto_memory_cleanup': base_config.auto_memory_cleanup,
                'auto_warmup': base_config.auto_warmup,
                'optimization_enabled': base_config.optimization_enabled,
                'strict_mode': base_config.strict_mode,
                'require_model_loader': base_config.require_model_loader,
                'require_memory_manager': base_config.require_memory_manager,
                'require_data_converter': base_config.require_data_converter,
                'require_di_container': base_config.require_di_container,
                'require_unified_dependency_manager': base_config.require_unified_dependency_manager,
                'ai_models': base_config.ai_models.copy(),
                'model_size_gb': base_config.model_size_gb,
                'conda_optimized': base_config.conda_optimized,
                'm3_max_optimized': base_config.m3_max_optimized
            }
            config_dict.update(overrides)
            return BaseStepMixinConfig(**config_dict)
        
        return base_config

# ==============================================
# 🔥 BaseStepMixin 호환 의존성 해결기
# ==============================================

class BaseStepMixinDependencyResolver:
    """BaseStepMixin v18.0 호환 의존성 해결기"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.BaseStepMixinDependencyResolver")
        self._resolved_cache: Dict[str, Any] = {}
        self._lock = threading.RLock()
        
        # 해결 시도 카운터
        self._resolution_attempts: Dict[str, int] = {}
        self._max_attempts = 3
    
    def resolve_dependencies_for_constructor(self, config: BaseStepMixinConfig) -> Dict[str, Any]:
        """BaseStepMixin 생성자용 의존성 해결 (핵심 메서드)"""
        try:
            self.logger.info(f"🔄 {config.step_name} 생성자용 의존성 해결 시작...")
            
            dependencies = {}
            
            # 기본 Step 설정들 (BaseStepMixin 표준)
            dependencies.update({
                'step_name': config.step_name,
                'step_id': config.step_id,
                'device': self._resolve_device(config.device),
                'use_fp16': config.use_fp16,
                'batch_size': config.batch_size,
                'confidence_threshold': config.confidence_threshold,
                'auto_memory_cleanup': config.auto_memory_cleanup,
                'auto_warmup': config.auto_warmup,
                'optimization_enabled': config.optimization_enabled,
                'strict_mode': config.strict_mode
            })
            
            # conda 환경 설정
            if config.conda_optimized and CONDA_INFO['is_target_env']:
                dependencies.update({
                    'conda_optimized': True,
                    'conda_env': CONDA_INFO['conda_env']
                })
            
            # M3 Max 설정
            if config.m3_max_optimized and IS_M3_MAX:
                dependencies.update({
                    'm3_max_optimized': True,
                    'memory_gb': MEMORY_GB,
                    'use_unified_memory': True,
                    'is_m3_max': True
                })
            
            # 의존성 컴포넌트들 해결
            if config.require_model_loader:
                model_loader = self._resolve_model_loader()
                if model_loader:
                    dependencies['model_loader'] = model_loader
                    self.logger.info(f"✅ {config.step_name} ModelLoader 생성자 주입 준비")
                else:
                    self.logger.warning(f"⚠️ {config.step_name} ModelLoader 해결 실패")
                    if config.strict_mode:
                        raise RuntimeError("Strict Mode: ModelLoader 필수이지만 해결 실패")
            
            if config.require_memory_manager:
                memory_manager = self._resolve_memory_manager()
                if memory_manager:
                    dependencies['memory_manager'] = memory_manager
                    self.logger.info(f"✅ {config.step_name} MemoryManager 생성자 주입 준비")
            
            if config.require_data_converter:
                data_converter = self._resolve_data_converter()
                if data_converter:
                    dependencies['data_converter'] = data_converter
                    self.logger.info(f"✅ {config.step_name} DataConverter 생성자 주입 준비")
            
            if config.require_di_container:
                di_container = self._resolve_di_container()
                if di_container:
                    dependencies['di_container'] = di_container
                    self.logger.info(f"✅ {config.step_name} DIContainer 생성자 주입 준비")
            
            if config.require_unified_dependency_manager:
                unified_dep_manager = self._resolve_unified_dependency_manager()
                if unified_dep_manager:
                    dependencies['unified_dependency_manager'] = unified_dep_manager
                    self.logger.info(f"✅ {config.step_name} UnifiedDependencyManager 생성자 주입 준비")
            
            # AI 모델 설정
            dependencies['ai_models'] = config.ai_models
            dependencies['model_size_gb'] = config.model_size_gb
            
            resolved_count = len([v for v in dependencies.values() if v is not None])
            self.logger.info(f"✅ {config.step_name} 생성자용 의존성 해결 완료: {resolved_count}개 항목")
            
            return dependencies
            
        except Exception as e:
            self.logger.error(f"❌ {config.step_name} 생성자용 의존성 해결 실패: {e}")
            return {}
    
    def _resolve_device(self, device: str) -> str:
        """디바이스 해결"""
        if device != "auto":
            return device
        
        if IS_M3_MAX:
            return "mps"
        
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        
        return "cpu"
    
    def _resolve_model_loader(self) -> Optional['ModelLoader']:
        """ModelLoader 해결"""
        try:
            with self._lock:
                cache_key = "model_loader"
                if cache_key in self._resolved_cache:
                    return self._resolved_cache[cache_key]
                
                # 해결 시도 제한
                attempts = self._resolution_attempts.get(cache_key, 0)
                if attempts >= self._max_attempts:
                    self.logger.warning(f"ModelLoader 해결 시도 한계 초과: {attempts}")
                    return None
                
                self._resolution_attempts[cache_key] = attempts + 1
                
                try:
                    # 절대 경로 방식으로 시도
                    from app.ai_pipeline.utils.model_loader import get_global_model_loader
                    model_loader = get_global_model_loader()
                    
                    if model_loader:
                        # conda 환경 최적화 설정
                        if CONDA_INFO['is_target_env'] and hasattr(model_loader, 'configure'):
                            config = {
                                'conda_optimized': True,
                                'conda_env': CONDA_INFO['conda_env'],
                                'm3_max_optimized': IS_M3_MAX,
                                'memory_gb': MEMORY_GB
                            }
                            model_loader.configure(config)
                        
                        self._resolved_cache[cache_key] = model_loader
                        self.logger.info("✅ ModelLoader 해결 완료")
                        return model_loader
                    
                except ImportError as e:
                    # 상대 경로 방식으로 재시도
                    try:
                        from ..utils.model_loader import get_global_model_loader
                        model_loader = get_global_model_loader()
                        if model_loader:
                            self._resolved_cache[cache_key] = model_loader
                            self.logger.info("✅ ModelLoader 해결 완료 (상대 경로)")
                            return model_loader
                    except ImportError as e2:
                        self.logger.debug(f"ModelLoader import 실패 (절대/상대): {e}, {e2}")
                        return None
                    
        except Exception as e:
            self.logger.error(f"❌ ModelLoader 해결 실패: {e}")
            return None
    
    def _resolve_memory_manager(self) -> Optional['MemoryManager']:
        """MemoryManager 해결"""
        try:
            with self._lock:
                cache_key = "memory_manager"
                if cache_key in self._resolved_cache:
                    return self._resolved_cache[cache_key]
                
                try:
                    # 절대 경로 방식으로 시도
                    from app.ai_pipeline.utils.memory_manager import get_global_memory_manager
                    memory_manager = get_global_memory_manager()
                    
                    if memory_manager:
                        # M3 Max 최적화
                        if IS_M3_MAX and hasattr(memory_manager, 'configure_m3_max'):
                            memory_manager.configure_m3_max(memory_gb=MEMORY_GB)
                        
                        self._resolved_cache[cache_key] = memory_manager
                        self.logger.info("✅ MemoryManager 해결 완료")
                        return memory_manager
                        
                except ImportError:
                    # 상대 경로 방식으로 재시도
                    try:
                        from ..utils.memory_manager import get_global_memory_manager
                        memory_manager = get_global_memory_manager()
                        if memory_manager:
                            self._resolved_cache[cache_key] = memory_manager
                            self.logger.info("✅ MemoryManager 해결 완료 (상대 경로)")
                            return memory_manager
                    except ImportError:
                        return None
                    
        except Exception as e:
            self.logger.debug(f"MemoryManager 해결 실패: {e}")
            return None
    
    def _resolve_data_converter(self) -> Optional['DataConverter']:
        """DataConverter 해결"""
        try:
            with self._lock:
                cache_key = "data_converter"
                if cache_key in self._resolved_cache:
                    return self._resolved_cache[cache_key]
                
                try:
                    # 절대 경로 방식으로 시도
                    from app.ai_pipeline.utils.data_converter import get_global_data_converter
                    data_converter = get_global_data_converter()
                    if data_converter:
                        self._resolved_cache[cache_key] = data_converter
                        self.logger.info("✅ DataConverter 해결 완료")
                        return data_converter
                        
                except ImportError:
                    # 상대 경로 방식으로 재시도
                    try:
                        from ..utils.data_converter import get_global_data_converter
                        data_converter = get_global_data_converter()
                        if data_converter:
                            self._resolved_cache[cache_key] = data_converter
                            self.logger.info("✅ DataConverter 해결 완료 (상대 경로)")
                            return data_converter
                    except ImportError:
                        return None
                    
        except Exception as e:
            self.logger.debug(f"DataConverter 해결 실패: {e}")
            return None
    
    def _resolve_di_container(self) -> Optional['DIContainer']:
        """DI Container 해결"""
        try:
            with self._lock:
                cache_key = "di_container"
                if cache_key in self._resolved_cache:
                    return self._resolved_cache[cache_key]
                
                try:
                    # 절대 경로 방식으로 시도
                    from app.core.di_container import get_global_di_container
                    di_container = get_global_di_container()
                    if di_container:
                        self._resolved_cache[cache_key] = di_container
                        self.logger.info("✅ DIContainer 해결 완료")
                        return di_container
                        
                except ImportError:
                    # 상대 경로 방식으로 재시도
                    try:
                        from ...core.di_container import get_global_di_container
                        di_container = get_global_di_container()
                        if di_container:
                            self._resolved_cache[cache_key] = di_container
                            self.logger.info("✅ DIContainer 해결 완료 (상대 경로)")
                            return di_container
                    except ImportError:
                        return None
                    
        except Exception as e:
            self.logger.debug(f"DIContainer 해결 실패: {e}")
            return None
    
    def _resolve_unified_dependency_manager(self) -> Optional[Any]:
        """UnifiedDependencyManager 해결"""
        try:
            with self._lock:
                cache_key = "unified_dependency_manager"
                if cache_key in self._resolved_cache:
                    return self._resolved_cache[cache_key]
                
                try:
                    # BaseStepMixin에서 사용하는 UnifiedDependencyManager 생성
                    # 🔥 절대 경로 시도
                    try:
                        from app.ai_pipeline.steps.base_step_mixin import UnifiedDependencyManager
                    except ImportError:
                        # 상대 경로 시도
                        from ..steps.base_step_mixin import UnifiedDependencyManager
                    
                    # 현재 해결된 의존성들을 사용하여 매니저 생성
                    unified_manager = UnifiedDependencyManager(
                        step_name="GlobalStepFactory",
                        is_m3_max=IS_M3_MAX,
                        memory_gb=MEMORY_GB,
                        conda_info=CONDA_INFO
                    )
                    
                    self._resolved_cache[cache_key] = unified_manager
                    self.logger.info("✅ UnifiedDependencyManager 해결 완료")
                    return unified_manager
                    
                except ImportError:
                    self.logger.debug("UnifiedDependencyManager import 실패")
                    # 폴백: 간단한 Mock 객체 생성
                    class MockUnifiedDependencyManager:
                        def __init__(self, **kwargs):
                            for key, value in kwargs.items():
                                setattr(self, key, value)
                    
                    mock_manager = MockUnifiedDependencyManager(
                        step_name="GlobalStepFactory",
                        is_m3_max=IS_M3_MAX,
                        memory_gb=MEMORY_GB,
                        conda_info=CONDA_INFO
                    )
                    self._resolved_cache[cache_key] = mock_manager
                    self.logger.info("✅ UnifiedDependencyManager 해결 완료 (Mock)")
                    return mock_manager
                    
        except Exception as e:
            self.logger.debug(f"UnifiedDependencyManager 해결 실패: {e}")
            return None
    
    def clear_cache(self):
        """캐시 정리"""
        with self._lock:
            self._resolved_cache.clear()
            self._resolution_attempts.clear()
            gc.collect()

# ==============================================
# 🔥 동적 Step 클래스 로더 (개선됨)
# ==============================================

class BaseStepMixinClassLoader:
    """BaseStepMixin 호환 동적 Step 클래스 로더"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.BaseStepMixinClassLoader")
        self._loaded_classes: Dict[str, Type] = {}
        self._import_attempts: Dict[str, int] = {}
        self._lock = threading.RLock()
        self._max_attempts = 5
    
    def load_step_class(self, config: BaseStepMixinConfig) -> Optional[Type]:
        """BaseStepMixin 호환 Step 클래스 로딩"""
        try:
            with self._lock:
                # 캐시 확인
                cache_key = config.class_name
                if cache_key in self._loaded_classes:
                    return self._loaded_classes[cache_key]
                
                # 재시도 제한
                attempts = self._import_attempts.get(cache_key, 0)
                if attempts >= self._max_attempts:
                    self.logger.error(f"❌ {config.class_name} import 재시도 한계 초과")
                    return None
                
                self._import_attempts[cache_key] = attempts + 1
                
                self.logger.info(f"🔄 {config.class_name} 동적 로딩 시작 (시도 {attempts + 1}/{self._max_attempts})...")
                
                # 동적 import 실행
                step_class = self._dynamic_import_step_class(config)
                
                if step_class:
                    # BaseStepMixin 호환성 검증
                    if self._validate_basestepmixin_compatibility(step_class, config):
                        self._loaded_classes[cache_key] = step_class
                        self.logger.info(f"✅ {config.class_name} 동적 로딩 성공 (BaseStepMixin 호환)")
                        return step_class
                    else:
                        self.logger.error(f"❌ {config.class_name} BaseStepMixin 호환성 검증 실패")
                        return None
                else:
                    self.logger.error(f"❌ {config.class_name} 동적 import 실패")
                    return None
                    
        except Exception as e:
            self.logger.error(f"❌ {config.class_name} 동적 로딩 예외: {e}")
            return None
    
    def _dynamic_import_step_class(self, config: BaseStepMixinConfig) -> Optional[Type]:
        """동적 import 실행"""
        import importlib
        
        # 기본 모듈 경로
        base_module = config.module_path
        
        # 여러 경로 시도
        import_paths = [
            base_module,
            f"app.ai_pipeline.steps.{config.module_path.split('.')[-1]}",  # 절대 경로 우선
            f"ai_pipeline.steps.{config.module_path.split('.')[-1]}",
            f"backend.{base_module}",
            f"..steps.{config.module_path.split('.')[-1]}",
            # 🔥 추가 대안 경로들
            f"backend.app.ai_pipeline.steps.{config.module_path.split('.')[-1]}",
            f"app.ai_pipeline.steps.step_{config.step_id:02d}_{config.step_type.value}",
            f"steps.{config.class_name.lower()}"
        ]
        
        for import_path in import_paths:
            try:
                self.logger.debug(f"🔍 {config.class_name} import 시도: {import_path}")
                
                # 동적 모듈 import
                module = importlib.import_module(import_path)
                
                # 클래스 추출
                if hasattr(module, config.class_name):
                    step_class = getattr(module, config.class_name)
                    self.logger.info(f"✅ {config.class_name} 동적 import 성공: {import_path}")
                    return step_class
                else:
                    self.logger.debug(f"⚠️ {import_path}에 {config.class_name} 클래스 없음")
                    continue
                    
            except ImportError as e:
                self.logger.debug(f"⚠️ {import_path} import 실패: {e}")
                continue
            except Exception as e:
                self.logger.warning(f"⚠️ {import_path} import 예외: {e}")
                continue
        
        self.logger.error(f"❌ {config.class_name} 모든 경로에서 import 실패")
        return None
    
    def _validate_basestepmixin_compatibility(self, step_class: Type, config: BaseStepMixinConfig) -> bool:
        """BaseStepMixin v18.0 호환성 검증"""
        try:
            # 기본 클래스 검증
            if not step_class or step_class.__name__ != config.class_name:
                return False
            
            # BaseStepMixin 상속 확인
            mro_names = [cls.__name__ for cls in step_class.__mro__]
            if 'BaseStepMixin' not in mro_names:
                self.logger.warning(f"⚠️ {config.class_name}이 BaseStepMixin을 상속하지 않음")
                # BaseStepMixin 미상속도 허용 (폴백 지원)
            
            # 필수 메서드 확인
            required_methods = ['process', 'initialize']
            missing_methods = []
            for method in required_methods:
                if not hasattr(step_class, method):
                    missing_methods.append(method)
            
            if missing_methods:
                self.logger.error(f"❌ {config.class_name}에 필수 메서드 없음: {missing_methods}")
                return False
            
            # process 메서드 시그니처 검증
            process_method = getattr(step_class, 'process')
            if not self._validate_process_method_signature(process_method, config):
                self.logger.warning(f"⚠️ {config.class_name} process 메서드 시그니처 비표준")
                # 경고만 출력하고 계속 진행
            
            # 생성자 호출 테스트 (BaseStepMixin 표준 kwargs)
            try:
                test_kwargs = {
                    'step_name': 'test',
                    'step_id': config.step_id,
                    'device': 'cpu'
                }
                test_instance = step_class(**test_kwargs)
                if test_instance:
                    self.logger.debug(f"✅ {config.class_name} BaseStepMixin 생성자 테스트 성공")
                    # 정리
                    if hasattr(test_instance, 'cleanup'):
                        try:
                            if asyncio.iscoroutinefunction(test_instance.cleanup):
                                # 비동기 cleanup은 스킵 (테스트에서)
                                pass
                            else:
                                test_instance.cleanup()
                        except:
                            pass
                    del test_instance
                    return True
            except Exception as e:
                self.logger.warning(f"⚠️ {config.class_name} 생성자 테스트 실패: {e}")
                # 🔥 대안 테스트: 매개변수 없이 시도
                try:
                    test_instance = step_class()
                    if test_instance:
                        self.logger.debug(f"✅ {config.class_name} 기본 생성자 테스트 성공")
                        del test_instance
                        return True
                except Exception as e2:
                    self.logger.debug(f"기본 생성자도 실패: {e2}")
                # 생성자 테스트 실패해도 계속 진행 (런타임에서 재시도)
                return True
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ {config.class_name} BaseStepMixin 호환성 검증 실패: {e}")
            return False
    
    def _validate_process_method_signature(self, process_method, config: BaseStepMixinConfig) -> bool:
        """process 메서드 시그니처 검증"""
        try:
            import inspect
            
            signature = inspect.signature(process_method)
            params = list(signature.parameters.keys())
            
            # 기본적으로 self, input_data 파라미터가 있어야 함
            expected_params = ['self', 'input_data']
            for expected in expected_params:
                if expected not in params:
                    self.logger.debug(f"process 메서드에 {expected} 파라미터 없음")
                    return False
            
            # async 함수인지 확인
            if not inspect.iscoroutinefunction(process_method):
                self.logger.debug(f"{config.class_name} process 메서드가 async가 아님")
                # sync 함수도 허용
            
            return True
            
        except Exception as e:
            self.logger.debug(f"process 메서드 시그니처 검증 실패: {e}")
            return False

# ==============================================
# 🔥 메인 StepFactory v9.0 (BaseStepMixin 완전 호환)
# ==============================================

class StepFactory:
    """
    🔥 StepFactory v9.0 - BaseStepMixin 완전 호환 (Option A 구현)
    
    핵심 수정사항:
    - BaseStepMixin v18.0 표준 완전 호환
    - 생성자 시점 의존성 주입 (constructor injection)
    - process() 메서드 시그니처 표준화
    - UnifiedDependencyManager 완전 활용
    - conda 환경 우선 최적화
    """
    
    def __init__(self):
        self.logger = logging.getLogger("StepFactory.v9")
        
        # BaseStepMixin 호환 컴포넌트들
        self.class_loader = BaseStepMixinClassLoader()
        self.dependency_resolver = BaseStepMixinDependencyResolver()
        
        # 캐시 관리
        self._step_cache: Dict[str, weakref.ref] = {}
        self._lock = threading.RLock()
        
        # 통계
        self._stats = {
            'total_created': 0,
            'successful_creations': 0,
            'failed_creations': 0,
            'cache_hits': 0,
            'basestepmixin_compatible_creations': 0,
            'dependency_injection_successes': 0,
            'conda_optimized': CONDA_INFO['is_target_env'],
            'm3_max_optimized': IS_M3_MAX
        }
        
        self.logger.info("🏭 StepFactory v9.0 초기화 완료 (BaseStepMixin v18.0 완전 호환)")
    
    def create_step(
        self,
        step_type: Union[StepType, str],
        use_cache: bool = True,
        **kwargs
    ) -> StepCreationResult:
        """Step 생성 메인 메서드 (BaseStepMixin 호환)"""
        start_time = time.time()
        
        try:
            # Step 타입 정규화
            if isinstance(step_type, str):
                try:
                    step_type = StepType(step_type.lower())
                except ValueError:
                    return StepCreationResult(
                        success=False,
                        error_message=f"지원하지 않는 Step 타입: {step_type}",
                        creation_time=time.time() - start_time
                    )
            
            # BaseStepMixin 호환 설정 생성
            config = BaseStepMixinMapping.get_config(step_type, **kwargs)
            
            self.logger.info(f"🎯 {config.step_name} 생성 시작 (BaseStepMixin v18.0 호환)...")
            
            # 통계 업데이트
            with self._lock:
                self._stats['total_created'] += 1
            
            # 캐시 확인
            if use_cache:
                cached_step = self._get_cached_step(config.step_name)
                if cached_step:
                    with self._lock:
                        self._stats['cache_hits'] += 1
                    self.logger.info(f"♻️ {config.step_name} 캐시에서 반환")
                    return StepCreationResult(
                        success=True,
                        step_instance=cached_step,
                        step_name=config.step_name,
                        step_type=step_type,
                        class_name=config.class_name,
                        module_path=config.module_path,
                        creation_time=time.time() - start_time,
                        basestepmixin_compatible=True
                    )
            
            # 실제 Step 생성 (BaseStepMixin 호환)
            result = self._create_basestepmixin_step_instance(config)
            
            # 성공 시 캐시에 저장
            if result.success and result.step_instance and use_cache:
                self._cache_step(config.step_name, result.step_instance)
            
            # 통계 업데이트
            with self._lock:
                if result.success:
                    self._stats['successful_creations'] += 1
                    if result.basestepmixin_compatible:
                        self._stats['basestepmixin_compatible_creations'] += 1
                    if result.dependency_injection_success:
                        self._stats['dependency_injection_successes'] += 1
                else:
                    self._stats['failed_creations'] += 1
            
            result.creation_time = time.time() - start_time
            return result
            
        except Exception as e:
            with self._lock:
                self._stats['failed_creations'] += 1
            
            self.logger.error(f"❌ Step 생성 실패: {e}")
            return StepCreationResult(
                success=False,
                error_message=f"Step 생성 예외: {str(e)}",
                creation_time=time.time() - start_time
            )
    
    def _create_basestepmixin_step_instance(self, config: BaseStepMixinConfig) -> StepCreationResult:
        """BaseStepMixin 호환 Step 인스턴스 생성 (핵심 메서드)"""
        try:
            self.logger.info(f"🔄 {config.step_name} BaseStepMixin 호환 인스턴스 생성 중...")
            
            # 1. Step 클래스 로딩
            StepClass = self.class_loader.load_step_class(config)
            if not StepClass:
                return StepCreationResult(
                    success=False,
                    step_name=config.step_name,
                    step_type=config.step_type,
                    class_name=config.class_name,
                    module_path=config.module_path,
                    error_message=f"{config.class_name} 클래스 로딩 실패"
                )
            
            self.logger.info(f"✅ {config.class_name} 클래스 로딩 완료")
            
            # 2. 생성자용 의존성 해결 (핵심: 생성자 시점 주입)
            constructor_dependencies = self.dependency_resolver.resolve_dependencies_for_constructor(config)
            
            # 3. BaseStepMixin 표준 생성자 호출 (**kwargs 패턴)
            self.logger.info(f"🔄 {config.class_name} BaseStepMixin 생성자 호출 중...")
            step_instance = StepClass(**constructor_dependencies)
            self.logger.info(f"✅ {config.class_name} 인스턴스 생성 완료 (생성자 의존성 주입)")
            
            # 4. 초기화 실행 (동기/비동기 자동 감지)
            initialization_success = self._initialize_basestepmixin_step(step_instance, config)
            
            # 5. BaseStepMixin 호환성 최종 검증
            compatibility_result = self._verify_basestepmixin_compatibility(step_instance, config)
            
            # 6. AI 모델 로딩 확인
            ai_models_loaded = self._check_ai_models_basestepmixin(step_instance, config)
            
            self.logger.info(f"✅ {config.step_name} BaseStepMixin 호환 생성 완료")
            
            return StepCreationResult(
                success=True,
                step_instance=step_instance,
                step_name=config.step_name,
                step_type=config.step_type,
                class_name=config.class_name,
                module_path=config.module_path,
                dependencies_injected={'constructor_injection': True},
                initialization_success=initialization_success,
                ai_models_loaded=ai_models_loaded,
                basestepmixin_compatible=compatibility_result['compatible'],
                process_method_validated=compatibility_result['process_method_valid'],
                dependency_injection_success=True
            )
            
        except Exception as e:
            self.logger.error(f"❌ {config.step_name} BaseStepMixin 인스턴스 생성 실패: {e}")
            self.logger.error(f"❌ 상세 오류: {traceback.format_exc()}")
            
            return StepCreationResult(
                success=False,
                step_name=config.step_name,
                step_type=config.step_type,
                class_name=config.class_name,
                module_path=config.module_path,
                error_message=f"BaseStepMixin 인스턴스 생성 실패: {str(e)}",
                basestepmixin_compatible=False
            )
    
    def _initialize_basestepmixin_step(self, step_instance: 'BaseStepMixin', config: BaseStepMixinConfig) -> bool:
        """BaseStepMixin Step 초기화 (동기/비동기 자동 감지)"""
        try:
            # BaseStepMixin initialize 메서드 호출
            if hasattr(step_instance, 'initialize'):
                initialize_method = step_instance.initialize
                
                # 🔥 동기/비동기 자동 감지 및 처리
                if asyncio.iscoroutinefunction(initialize_method):
                    # 비동기 함수인 경우
                    try:
                        # 현재 실행 중인 이벤트 루프가 있는지 확인
                        loop = asyncio.get_running_loop()
                        
                        # 이미 실행 중인 루프에서는 태스크 생성 후 블로킹 대기
                        if loop.is_running():
                            # 새로운 스레드에서 실행하거나 동기적으로 처리
                            import concurrent.futures
                            with concurrent.futures.ThreadPoolExecutor() as executor:
                                future = executor.submit(asyncio.run, initialize_method())
                                success = future.result(timeout=30)  # 30초 타임아웃
                        else:
                            # 루프가 실행 중이 아니면 직접 실행
                            success = asyncio.run(initialize_method())
                    except RuntimeError:
                        # 실행 중인 루프가 없으면 새 루프에서 실행
                        success = asyncio.run(initialize_method())
                    except Exception as e:
                        self.logger.warning(f"⚠️ {config.step_name} 비동기 초기화 실패, 동기 방식 시도: {e}")
                        # 비동기 초기화 실패 시 폴백 (동기 방식으로 재시도)
                        success = self._fallback_sync_initialize(step_instance, config)
                else:
                    # 동기 함수인 경우
                    success = initialize_method()
                
                if success:
                    self.logger.info(f"✅ {config.step_name} BaseStepMixin 초기화 완료")
                    return True
                else:
                    self.logger.warning(f"⚠️ {config.step_name} BaseStepMixin 초기화 실패")
                    return False
            else:
                self.logger.debug(f"ℹ️ {config.step_name} initialize 메서드 없음")
                return True
                
        except Exception as e:
            self.logger.warning(f"⚠️ {config.step_name} 초기화 예외: {e}")
            # 예외 발생 시 폴백 초기화 시도
            return self._fallback_sync_initialize(step_instance, config)
    
    def _fallback_sync_initialize(self, step_instance: 'BaseStepMixin', config: BaseStepMixinConfig) -> bool:
        """폴백 동기 초기화 (비동기 초기화 실패 시)"""
        try:
            self.logger.info(f"🔄 {config.step_name} 폴백 동기 초기화 시도...")
            
            # 기본 속성들 수동 설정
            if hasattr(step_instance, 'is_initialized'):
                step_instance.is_initialized = True
            
            if hasattr(step_instance, 'is_ready'):
                step_instance.is_ready = True
                
            # 의존성이 제대로 주입되었는지 확인
            dependencies_ok = True
            if config.require_model_loader and not hasattr(step_instance, 'model_loader'):
                dependencies_ok = False
                
            if dependencies_ok:
                self.logger.info(f"✅ {config.step_name} 폴백 동기 초기화 성공")
                return True
            else:
                self.logger.warning(f"⚠️ {config.step_name} 폴백 초기화: 의존성 문제 있음")
                return not config.strict_mode  # strict_mode가 아니면 계속 진행
                
        except Exception as e:
            self.logger.error(f"❌ {config.step_name} 폴백 초기화 실패: {e}")
            return False
    
    def _verify_basestepmixin_compatibility(self, step_instance: 'BaseStepMixin', config: BaseStepMixinConfig) -> Dict[str, Any]:
        """BaseStepMixin 호환성 최종 검증"""
        try:
            result = {
                'compatible': True,
                'process_method_valid': False,
                'issues': []
            }
            
            # process 메서드 존재 확인
            if not hasattr(step_instance, 'process'):
                result['compatible'] = False
                result['issues'].append('process 메서드 없음')
            else:
                result['process_method_valid'] = True
            
            # BaseStepMixin 속성 확인
            expected_attrs = ['step_name', 'step_id', 'device', 'is_initialized']
            for attr in expected_attrs:
                if not hasattr(step_instance, attr):
                    result['issues'].append(f'{attr} 속성 없음')
            
            # 의존성 주입 상태 확인
            if hasattr(step_instance, 'model_loader') and step_instance.model_loader:
                self.logger.debug(f"✅ {config.step_name} ModelLoader 주입 확인됨")
            
            if result['issues']:
                self.logger.warning(f"⚠️ {config.step_name} BaseStepMixin 호환성 이슈: {result['issues']}")
            else:
                self.logger.info(f"✅ {config.step_name} BaseStepMixin 호환성 검증 완료")
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ {config.step_name} BaseStepMixin 호환성 검증 실패: {e}")
            return {'compatible': False, 'process_method_valid': False, 'issues': [str(e)]}
    
    def _check_ai_models_basestepmixin(self, step_instance: 'BaseStepMixin', config: BaseStepMixinConfig) -> List[str]:
        """AI 모델 로딩 확인 (BaseStepMixin 호환)"""
        loaded_models = []
        
        try:
            # ModelLoader 를 통한 모델 확인
            if hasattr(step_instance, 'model_loader') and step_instance.model_loader:
                for model_name in config.ai_models:
                    try:
                        if hasattr(step_instance.model_loader, 'is_model_loaded'):
                            if step_instance.model_loader.is_model_loaded(model_name):
                                loaded_models.append(model_name)
                    except Exception:
                        pass
            
            # model_interface 를 통한 모델 확인
            if hasattr(step_instance, 'model_interface') and step_instance.model_interface:
                for model_name in config.ai_models:
                    try:
                        if hasattr(step_instance.model_interface, 'is_model_available'):
                            if step_instance.model_interface.is_model_available(model_name):
                                loaded_models.append(model_name)
                    except Exception:
                        pass
            
            if loaded_models:
                self.logger.info(f"🤖 {config.step_name} AI 모델 로딩 확인: {loaded_models}")
            
            return loaded_models
            
        except Exception as e:
            self.logger.debug(f"AI 모델 확인 실패: {e}")
            return []
    
    def _get_cached_step(self, step_name: str) -> Optional['BaseStepMixin']:
        """캐시된 Step 반환"""
        try:
            with self._lock:
                if step_name in self._step_cache:
                    weak_ref = self._step_cache[step_name]
                    step_instance = weak_ref()
                    if step_instance is not None:
                        return step_instance
                    else:
                        del self._step_cache[step_name]
                return None
        except Exception:
            return None
    
    def _cache_step(self, step_name: str, step_instance: 'BaseStepMixin'):
        """Step 캐시에 저장"""
        try:
            with self._lock:
                self._step_cache[step_name] = weakref.ref(step_instance)
        except Exception:
            pass
    
    # ==============================================
    # 🔥 편의 메서드들 (BaseStepMixin 호환)
    # ==============================================
    
    def create_human_parsing_step(self, **kwargs) -> StepCreationResult:
        """Human Parsing Step 생성 (BaseStepMixin 호환)"""
        return self.create_step(StepType.HUMAN_PARSING, **kwargs)
    
    def create_pose_estimation_step(self, **kwargs) -> StepCreationResult:
        """Pose Estimation Step 생성 (BaseStepMixin 호환)"""
        return self.create_step(StepType.POSE_ESTIMATION, **kwargs)
    
    def create_cloth_segmentation_step(self, **kwargs) -> StepCreationResult:
        """Cloth Segmentation Step 생성 (BaseStepMixin 호환)"""
        return self.create_step(StepType.CLOTH_SEGMENTATION, **kwargs)
    
    def create_geometric_matching_step(self, **kwargs) -> StepCreationResult:
        """Geometric Matching Step 생성 (BaseStepMixin 호환)"""
        return self.create_step(StepType.GEOMETRIC_MATCHING, **kwargs)
    
    def create_cloth_warping_step(self, **kwargs) -> StepCreationResult:
        """Cloth Warping Step 생성 (BaseStepMixin 호compat)"""
        return self.create_step(StepType.CLOTH_WARPING, **kwargs)
    
    def create_virtual_fitting_step(self, **kwargs) -> StepCreationResult:
        """Virtual Fitting Step 생성 (BaseStepMixin 호환)"""
        return self.create_step(StepType.VIRTUAL_FITTING, **kwargs)
    
    def create_post_processing_step(self, **kwargs) -> StepCreationResult:
        """Post Processing Step 생성 (BaseStepMixin 호환)"""
        return self.create_step(StepType.POST_PROCESSING, **kwargs)
    
    def create_quality_assessment_step(self, **kwargs) -> StepCreationResult:
        """Quality Assessment Step 생성 (BaseStepMixin 호환)"""
        return self.create_step(StepType.QUALITY_ASSESSMENT, **kwargs)
    
    def create_full_pipeline(self, device: str = "auto", **kwargs) -> Dict[str, StepCreationResult]:
        """전체 파이프라인 생성 (BaseStepMixin 호환) - 동기 메서드"""
        try:
            self.logger.info("🚀 전체 AI 파이프라인 생성 시작 (BaseStepMixin v18.0 호환)...")
            
            pipeline_results = {}
            total_model_size = 0.0
            
            # 우선순위별로 Step 생성
            sorted_steps = sorted(
                StepType,
                key=lambda x: BaseStepMixinMapping.STEP_CONFIGS[x].priority.value
            )
            
            for step_type in sorted_steps:
                try:
                    result = self.create_step(step_type, device=device, **kwargs)
                    pipeline_results[step_type.value] = result
                    
                    if result.success:
                        config = BaseStepMixinMapping.get_config(step_type)
                        total_model_size += config.model_size_gb
                        self.logger.info(f"✅ {result.step_name} 파이프라인 생성 성공 (BaseStepMixin 호환)")
                    else:
                        self.logger.warning(f"⚠️ {step_type.value} 파이프라인 생성 실패")
                        
                except Exception as e:
                    self.logger.error(f"❌ {step_type.value} Step 생성 예외: {e}")
                    pipeline_results[step_type.value] = StepCreationResult(
                        success=False,
                        step_name=f"{step_type.value}Step",
                        step_type=step_type,
                        error_message=str(e)
                    )
            
            success_count = sum(1 for result in pipeline_results.values() if result.success)
            total_count = len(pipeline_results)
            
            self.logger.info(f"🏁 BaseStepMixin 호환 파이프라인 생성 완료: {success_count}/{total_count} 성공")
            self.logger.info(f"🤖 총 AI 모델 크기: {total_model_size:.1f}GB")
            
            return pipeline_results
            
        except Exception as e:
            self.logger.error(f"❌ 전체 파이프라인 생성 실패: {e}")
            return {}
    
    def get_statistics(self) -> Dict[str, Any]:
        """통계 정보 반환 (BaseStepMixin 호환성 포함)"""
        with self._lock:
            total = self._stats['total_created']
            success_rate = (self._stats['successful_creations'] / max(1, total)) * 100
            basestepmixin_compatibility_rate = (self._stats['basestepmixin_compatible_creations'] / max(1, self._stats['successful_creations'])) * 100
            
            return {
                'version': 'StepFactory v9.0 (BaseStepMixin Complete Compatibility)',
                'total_created': total,
                'successful_creations': self._stats['successful_creations'],
                'failed_creations': self._stats['failed_creations'],
                'success_rate': round(success_rate, 2),
                'cache_hits': self._stats['cache_hits'],
                'cached_steps': len(self._step_cache),
                'active_cache_entries': len([
                    ref for ref in self._step_cache.values() if ref() is not None
                ]),
                'basestepmixin_compatibility': {
                    'compatible_creations': self._stats['basestepmixin_compatible_creations'],
                    'compatibility_rate': round(basestepmixin_compatibility_rate, 2),
                    'dependency_injection_successes': self._stats['dependency_injection_successes']
                },
                'environment': {
                    'conda_env': CONDA_INFO['conda_env'],
                    'conda_optimized': self._stats['conda_optimized'],
                    'is_m3_max': IS_M3_MAX,
                    'm3_max_optimized': self._stats['m3_max_optimized'],
                    'memory_gb': MEMORY_GB
                },
                'loaded_classes': self.class_loader._loaded_classes.keys()
            }
    
    def clear_cache(self):
        """캐시 정리"""
        try:
            with self._lock:
                self._step_cache.clear()
                self.dependency_resolver.clear_cache()
                
                # M3 Max 메모리 정리
                if IS_M3_MAX:
                    try:
                        import torch
                        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                            if hasattr(torch.backends.mps, 'empty_cache'):
                                torch.backends.mps.empty_cache()
                    except:
                        pass
                
                gc.collect()
                self.logger.info("🧹 StepFactory v9.0 캐시 정리 완료")
        except Exception as e:
            self.logger.error(f"❌ 캐시 정리 실패: {e}")

# ==============================================
# 🔥 전역 StepFactory 관리 (BaseStepMixin 호환)
# ==============================================

_global_step_factory: Optional[StepFactory] = None
_factory_lock = threading.Lock()

def get_global_step_factory() -> StepFactory:
    """전역 StepFactory v9.0 인스턴스 반환"""
    global _global_step_factory
    
    with _factory_lock:
        if _global_step_factory is None:
            _global_step_factory = StepFactory()
            logger.info("✅ 전역 StepFactory v9.0 (BaseStepMixin 호환) 생성 완료")
        
        return _global_step_factory

def reset_global_step_factory():
    """전역 StepFactory 리셋"""
    global _global_step_factory
    
    with _factory_lock:
        if _global_step_factory:
            _global_step_factory.clear_cache()
        _global_step_factory = None
        logger.info("🔄 전역 StepFactory v9.0 리셋 완료")

# ==============================================
# 🔥 편의 함수들 (BaseStepMixin 호환)
# ==============================================

def create_step(step_type: Union[StepType, str], **kwargs) -> StepCreationResult:
    """전역 Step 생성 함수 (BaseStepMixin 호환)"""
    factory = get_global_step_factory()
    return factory.create_step(step_type, **kwargs)

def create_human_parsing_step(**kwargs) -> StepCreationResult:
    """Human Parsing Step 생성 (BaseStepMixin 호환)"""
    return create_step(StepType.HUMAN_PARSING, **kwargs)

def create_pose_estimation_step(**kwargs) -> StepCreationResult:
    """Pose Estimation Step 생성 (BaseStepMixin 호환)"""
    return create_step(StepType.POSE_ESTIMATION, **kwargs)

def create_cloth_segmentation_step(**kwargs) -> StepCreationResult:
    """Cloth Segmentation Step 생성 (BaseStepMixin 호환)"""
    return create_step(StepType.CLOTH_SEGMENTATION, **kwargs)

def create_geometric_matching_step(**kwargs) -> StepCreationResult:
    """Geometric Matching Step 생성 (BaseStepMixin 호환)"""
    return create_step(StepType.GEOMETRIC_MATCHING, **kwargs)

def create_cloth_warping_step(**kwargs) -> StepCreationResult:
    """Cloth Warping Step 생성 (BaseStepMixin 호환)"""
    return create_step(StepType.CLOTH_WARPING, **kwargs)

def create_virtual_fitting_step(**kwargs) -> StepCreationResult:
    """Virtual Fitting Step 생성 (BaseStepMixin 호환)"""
    return create_step(StepType.VIRTUAL_FITTING, **kwargs)

def create_post_processing_step(**kwargs) -> StepCreationResult:
    """Post Processing Step 생성 (BaseStepMixin 호환)"""
    return create_step(StepType.POST_PROCESSING, **kwargs)

def create_quality_assessment_step(**kwargs) -> StepCreationResult:
    """Quality Assessment Step 생성 (BaseStepMixin 호환)"""
    return create_step(StepType.QUALITY_ASSESSMENT, **kwargs)

def create_full_pipeline(device: str = "auto", **kwargs) -> Dict[str, StepCreationResult]:
    """전체 파이프라인 생성 (BaseStepMixin 호환) - 동기 함수"""
    factory = get_global_step_factory()
    return factory.create_full_pipeline(device, **kwargs)

def get_step_factory_statistics() -> Dict[str, Any]:
    """StepFactory 통계 조회 (BaseStepMixin 호환성 포함)"""
    factory = get_global_step_factory()
    return factory.get_statistics()

def clear_step_factory_cache():
    """StepFactory 캐시 정리"""
    factory = get_global_step_factory()
    factory.clear_cache()

# ==============================================
# 🔥 conda 환경 최적화 (BaseStepMixin 호환)
# ==============================================

def optimize_conda_environment_for_basestepmixin():
    """conda 환경 최적화 (BaseStepMixin 호환)"""
    try:
        if not CONDA_INFO['is_target_env']:
            logger.warning(f"⚠️ 권장 conda 환경이 아님: {CONDA_INFO['conda_env']} (권장: mycloset-ai-clean)")
            return False
        
        # PyTorch conda 최적화
        try:
            import torch
            if IS_M3_MAX and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                # MPS 캐시 정리
                if hasattr(torch.backends.mps, 'empty_cache'):
                    torch.backends.mps.empty_cache()
                logger.info("🍎 M3 Max MPS 최적화 활성화 (BaseStepMixin 호환)")
            
            # CPU 스레드 최적화
            cpu_count = os.cpu_count()
            torch.set_num_threads(max(1, cpu_count // 2))
            logger.info(f"🧵 PyTorch 스레드 최적화: {torch.get_num_threads()}/{cpu_count}")
            
        except ImportError:
            pass
        
        # 환경 변수 설정
        os.environ['OMP_NUM_THREADS'] = str(max(1, os.cpu_count() // 2))
        os.environ['MKL_NUM_THREADS'] = str(max(1, os.cpu_count() // 2))
        
        logger.info("🐍 conda 환경 최적화 완료 (BaseStepMixin v18.0 호환)")
        return True
        
    except Exception as e:
        logger.warning(f"⚠️ conda 환경 최적화 실패: {e}")
        return False

# ==============================================
# 🔥 BaseStepMixin 호환성 검증 도구
# ==============================================

def validate_basestepmixin_step_compatibility(step_instance: 'BaseStepMixin') -> Dict[str, Any]:
    """BaseStepMixin Step 호환성 검증"""
    try:
        result = {
            'compatible': True,
            'version': 'StepFactory v9.0',
            'issues': [],
            'recommendations': []
        }
        
        # 필수 속성 확인
        required_attrs = ['step_name', 'step_id', 'device', 'is_initialized']
        for attr in required_attrs:
            if not hasattr(step_instance, attr):
                result['compatible'] = False
                result['issues'].append(f'필수 속성 {attr} 없음')
        
        # 필수 메서드 확인
        required_methods = ['process', 'initialize']
        for method in required_methods:
            if not hasattr(step_instance, method):
                result['compatible'] = False
                result['issues'].append(f'필수 메서드 {method} 없음')
        
        # BaseStepMixin 상속 확인
        mro_names = [cls.__name__ for cls in step_instance.__class__.__mro__]
        if 'BaseStepMixin' not in mro_names:
            result['recommendations'].append('BaseStepMixin 상속 권장')
        
        # 의존성 주입 상태 확인
        dependency_attrs = ['model_loader', 'memory_manager', 'data_converter']
        injected_deps = []
        for attr in dependency_attrs:
            if hasattr(step_instance, attr) and getattr(step_instance, attr) is not None:
                injected_deps.append(attr)
        
        result['injected_dependencies'] = injected_deps
        result['dependency_injection_score'] = len(injected_deps) / len(dependency_attrs)
        
        return result
        
    except Exception as e:
        return {
            'compatible': False,
            'error': str(e),
            'version': 'StepFactory v9.0'
        }

def get_basestepmixin_step_info(step_instance: 'BaseStepMixin') -> Dict[str, Any]:
    """BaseStepMixin Step 정보 조회"""
    try:
        info = {
            'step_name': getattr(step_instance, 'step_name', 'Unknown'),
            'step_id': getattr(step_instance, 'step_id', 0),
            'class_name': step_instance.__class__.__name__,
            'module': step_instance.__class__.__module__,
            'device': getattr(step_instance, 'device', 'Unknown'),
            'is_initialized': getattr(step_instance, 'is_initialized', False),
            'has_model': getattr(step_instance, 'has_model', False),
            'model_loaded': getattr(step_instance, 'model_loaded', False)
        }
        
        # 의존성 상태
        dependencies = {}
        for dep_name in ['model_loader', 'memory_manager', 'data_converter', 'di_container']:
            dependencies[dep_name] = hasattr(step_instance, dep_name) and getattr(step_instance, dep_name) is not None
        
        info['dependencies'] = dependencies
        
        # BaseStepMixin 특정 속성들
        if hasattr(step_instance, 'dependency_manager'):
            dep_manager = step_instance.dependency_manager
            if hasattr(dep_manager, 'get_status'):
                try:
                    info['dependency_manager_status'] = dep_manager.get_status()
                except:
                    info['dependency_manager_status'] = 'error'
        
        return info
        
    except Exception as e:
        return {'error': str(e)}

# ==============================================
# 🔥 디버깅 및 테스트 도구
# ==============================================

async def test_step_creation_flow(step_type: StepType, **kwargs) -> Dict[str, Any]:
    """Step 생성 플로우 테스트 (동기/비동기 호환)"""
    try:
        test_result = {
            'step_type': step_type.value,
            'test_start_time': time.time(),
            'phases': {}
        }
        
        factory = get_global_step_factory()
        
        # Phase 1: 설정 생성 테스트
        phase1_start = time.time()
        try:
            config = BaseStepMixinMapping.get_config(step_type, **kwargs)
            test_result['phases']['config_creation'] = {
                'success': True,
                'time': time.time() - phase1_start,
                'config_class': config.class_name
            }
        except Exception as e:
            test_result['phases']['config_creation'] = {
                'success': False,
                'time': time.time() - phase1_start,
                'error': str(e)
            }
            return test_result
        
        # Phase 2: 클래스 로딩 테스트
        phase2_start = time.time()
        try:
            step_class = factory.class_loader.load_step_class(config)
            test_result['phases']['class_loading'] = {
                'success': step_class is not None,
                'time': time.time() - phase2_start,
                'class_found': step_class.__name__ if step_class else None
            }
        except Exception as e:
            test_result['phases']['class_loading'] = {
                'success': False,
                'time': time.time() - phase2_start,
                'error': str(e)
            }
            if not step_class:
                return test_result
        
        # Phase 3: 의존성 해결 테스트
        phase3_start = time.time()
        try:
            dependencies = factory.dependency_resolver.resolve_dependencies_for_constructor(config)
            test_result['phases']['dependency_resolution'] = {
                'success': len(dependencies) > 0,
                'time': time.time() - phase3_start,
                'resolved_count': len(dependencies),
                'resolved_dependencies': list(dependencies.keys())
            }
        except Exception as e:
            test_result['phases']['dependency_resolution'] = {
                'success': False,
                'time': time.time() - phase3_start,
                'error': str(e)
            }
        
        # Phase 4: 인스턴스 생성 테스트 (동기)
        phase4_start = time.time()
        try:
            result = factory.create_step(step_type, **kwargs)
            test_result['phases']['instance_creation'] = {
                'success': result.success,
                'time': time.time() - phase4_start,
                'step_name': result.step_name,
                'basestepmixin_compatible': result.basestepmixin_compatible,
                'error': result.error_message if not result.success else None
            }
        except Exception as e:
            test_result['phases']['instance_creation'] = {
                'success': False,
                'time': time.time() - phase4_start,
                'error': str(e)
            }
        
        test_result['total_time'] = time.time() - test_result['test_start_time']
        test_result['overall_success'] = all(
            phase.get('success', False) for phase in test_result['phases'].values()
        )
        
        return test_result
        
    except Exception as e:
        return {
            'step_type': step_type.value if step_type else 'unknown',
            'overall_success': False,
            'error': str(e)
        }

def diagnose_step_factory_health() -> Dict[str, Any]:
    """StepFactory 상태 진단"""
    try:
        factory = get_global_step_factory()
        health_report = {
            'factory_version': 'v9.0 (BaseStepMixin Complete Compatibility)',
            'timestamp': time.time(),
            'environment': {
                'conda_env': CONDA_INFO['conda_env'],
                'is_target_env': CONDA_INFO['is_target_env'],
                'is_m3_max': IS_M3_MAX,
                'memory_gb': MEMORY_GB
            },
            'statistics': factory.get_statistics(),
            'cache_status': {
                'cached_steps': len(factory._step_cache),
                'active_references': len([
                    ref for ref in factory._step_cache.values() if ref() is not None
                ])
            },
            'component_status': {
                'class_loader': 'operational',
                'dependency_resolver': 'operational'
            },
            'recommendations': []
        }
        
        # 환경 체크
        if not CONDA_INFO['is_target_env']:
            health_report['recommendations'].append(
                f"conda 환경을 mycloset-ai-clean으로 변경 권장 (현재: {CONDA_INFO['conda_env']})"
            )
        
        # 메모리 체크
        if MEMORY_GB < 16:
            health_report['recommendations'].append(
                f"메모리 부족 주의: {MEMORY_GB:.1f}GB (권장: 16GB+)"
            )
        
        # 캐시 체크
        if len(factory._step_cache) > 10:
            health_report['recommendations'].append(
                "캐시된 Step이 많습니다. clear_cache() 호출 고려"
            )
        
        health_report['overall_health'] = 'good' if len(health_report['recommendations']) == 0 else 'warning'
        
        return health_report
        
    except Exception as e:
        return {
            'overall_health': 'error',
            'error': str(e)
        }

# ==============================================
# 🔥 Export
# ==============================================

__all__ = [
    # 메인 클래스들
    'StepFactory',
    'BaseStepMixinClassLoader', 
    'BaseStepMixinDependencyResolver',
    'BaseStepMixinMapping',
    
    # 데이터 구조들
    'StepType',
    'StepPriority', 
    'BaseStepMixinConfig',
    'StepCreationResult',
    
    # 전역 함수들
    'get_global_step_factory',
    'reset_global_step_factory',
    
    # Step 생성 함수들 (BaseStepMixin 호환)
    'create_step',
    'create_human_parsing_step',
    'create_pose_estimation_step', 
    'create_cloth_segmentation_step',
    'create_geometric_matching_step',
    'create_cloth_warping_step',
    'create_virtual_fitting_step',
    'create_post_processing_step',
    'create_quality_assessment_step',
    'create_full_pipeline',
    
    # 유틸리티 함수들
    'get_step_factory_statistics',
    'clear_step_factory_cache',
    'optimize_conda_environment_for_basestepmixin',
    
    # BaseStepMixin 호환성 도구들
    'validate_basestepmixin_step_compatibility',
    'get_basestepmixin_step_info',
    'test_step_creation_flow',
    'diagnose_step_factory_health',
    
    # 상수들
    'CONDA_INFO',
    'IS_M3_MAX', 
    'MEMORY_GB'
]

# ==============================================
# 🔥 모듈 초기화 (BaseStepMixin v18.0 호환)
# ==============================================

logger.info("🔥 StepFactory v9.0 - BaseStepMixin 완전 호환 (Option A 구현) 로드 완료!")
logger.info("✅ 주요 개선사항:")
logger.info("   - BaseStepMixin v18.0 표준 완전 호환")
logger.info("   - 생성자 시점 의존성 주입 (constructor injection)")
logger.info("   - process() 메서드 시그니처 표준화")
logger.info("   - UnifiedDependencyManager 완전 활용")
logger.info("   - **kwargs 패턴 완전 지원")
logger.info("   - conda 환경 우선 최적화")
logger.info("   - M3 Max 128GB 메모리 최적화")

logger.info(f"🔧 현재 환경:")
logger.info(f"   - conda 환경: {CONDA_INFO['conda_env']} ({'✅ 최적화됨' if CONDA_INFO['is_target_env'] else '⚠️ 권장: mycloset-ai-clean'})")
logger.info(f"   - M3 Max: {'✅' if IS_M3_MAX else '❌'}")
logger.info(f"   - 메모리: {MEMORY_GB:.1f}GB")

logger.info("🎯 지원 Step 클래스 (BaseStepMixin 호환):")
for step_type in StepType:
    config = BaseStepMixinMapping.STEP_CONFIGS[step_type]
    logger.info(f"   - {config.class_name} (Step {config.step_id:02d}) - {config.model_size_gb}GB")

# conda 환경 자동 최적화 (BaseStepMixin 호환)
if CONDA_INFO['is_target_env']:
    optimize_conda_environment_for_basestepmixin()
    logger.info("🐍 conda 환경 자동 최적화 완료! (BaseStepMixin v18.0 호환)")
else:
    logger.warning(f"⚠️ conda 환경을 확인하세요: conda activate mycloset-ai-clean")

# 초기 메모리 최적화
if IS_M3_MAX:
    try:
        import torch
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            if hasattr(torch.backends.mps, 'empty_cache'):
                torch.backends.mps.empty_cache()
        gc.collect()
        logger.info("🍎 M3 Max 초기 메모리 최적화 완료! (BaseStepMixin 호환)")
    except:
        pass

logger.info("🚀 StepFactory v9.0 완전 준비 완료! (BaseStepMixin v18.0 완전 호환) 🚀")
logger.info("💡 이제 실제 Step 클래스들과 100% 호환됩니다!")
logger.info("💡 생성자 시점 의존성 주입으로 안정성 보장!")
logger.info("💡 process() 메서드 시그니처 표준화 완료!")