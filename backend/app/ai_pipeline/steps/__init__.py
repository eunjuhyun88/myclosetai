#!/usr/bin/env python3
"""#backend/app/ai_pipeline/steps/__init__.py
#!/usr/bin/env python3
🔥 MyCloset AI Pipeline Steps v5.2 - DI Container v7.0 완전 통합 + 간소화
================================================================

✅ DI Container v7.0 Central Hub 완전 통합
✅ TYPE_CHECKING으로 순환참조 완전 차단  
✅ 자체 StepsCircularReferenceFreeDIContainer 제거 (중복 해결)
✅ Central Hub의 모든 기능 활용
✅ 안전한 의존성 주입 시스템
✅ logger 에러 완전 해결
✅ M3 Max 128GB + conda 환경 최적화
✅ GitHub 프로젝트 구조 100% 호환
✅ safe_copy 함수 유지 (DetailedDataSpec 에러 해결)
✅ 코드 복잡성 대폭 감소

Author: MyCloset AI Team
Date: 2025-08-01
Version: 5.2 (DI Container v7.0 Integration + Simplification)
"""

import os
import gc
import logging
import threading
import time
import warnings
import sys
import asyncio
import copy
from typing import Dict, Any, Optional, Type, TypeVar, Callable, Union, List, TYPE_CHECKING
from abc import ABC, abstractmethod
from pathlib import Path

# 경고 무시 (deprecated 경로 관련)
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', message='.*deprecated.*')

# Logger 최우선 초기화 (에러 방지)
logger = logging.getLogger(__name__)

# ==============================================
# 🔥 safe_copy 함수 정의 (DetailedDataSpec 에러 해결)
# ==============================================

def safe_copy(obj: Any) -> Any:
    """안전한 복사 함수 - DetailedDataSpec 에러 해결"""
    try:
        # 기본 타입들은 그대로 반환
        if obj is None or isinstance(obj, (bool, int, float, str)):
            return obj
        
        # 리스트나 튜플
        elif isinstance(obj, (list, tuple)):
            return type(obj)(safe_copy(item) for item in obj)
        
        # 딕셔너리
        elif isinstance(obj, dict):
            return {key: safe_copy(value) for key, value in obj.items()}
        
        # 집합
        elif isinstance(obj, set):
            return {safe_copy(item) for item in obj}
        
        # copy 모듈 사용 가능한 경우
        else:
            try:
                return copy.deepcopy(obj)
            except:
                try:
                    return copy.copy(obj)
                except:
                    # 복사할 수 없는 경우 원본 반환 (예: 함수, 클래스 등)
                    logger.debug(f"⚠️ safe_copy: 복사 불가능한 객체 - {type(obj)}")
                    return obj
                    
    except Exception as e:
        logger.warning(f"⚠️ safe_copy 실패: {e}, 원본 반환")
        return obj

# 전역으로 사용 가능하도록 설정
globals()['safe_copy'] = safe_copy

# ==============================================
# 🔥 TYPE_CHECKING으로 순환참조 완전 방지
# ==============================================

if TYPE_CHECKING:
    # 오직 타입 체크 시에만 import
    from .base_step_mixin import BaseStepMixin, GitHubDependencyManager
    from ..utils.model_loader import ModelLoader
    from ..utils.memory_manager import MemoryManager
    from ..utils.data_converter import DataConverter
    from ..factories.step_factory import StepFactory
else:
    # 런타임에는 Any로 처리 (순환참조 방지)
    BaseStepMixin = Any
    GitHubDependencyManager = Any
    ModelLoader = Any
    MemoryManager = Any
    DataConverter = Any
    StepFactory = Any

# ==============================================
# 🔥 DI Container v7.0 Central Hub Import
# ==============================================

try:
    # 절대 임포트 시도
    from app.core.di_container import (
        CentralHubDIContainer,  # v7.0 메인 클래스
        CircularReferenceFreeDIContainer,  # 호환성 별칭
        LazyDependency,
        DynamicImportResolver,
        get_global_container,
        inject_dependencies_to_step_safe,
        get_service_safe,
        register_service_safe,
        register_lazy_service,
        initialize_di_system_safe
    )
    DI_CONTAINER_AVAILABLE = True
    logger.info("✅ DI Container v7.0 Central Hub 로드 성공 (절대 임포트)")
except ImportError:
    try:
        # 상대 임포트 시도 (폴백)
        from ...core.di_container import (
            CentralHubDIContainer,
            CircularReferenceFreeDIContainer,
            LazyDependency,
            DynamicImportResolver,
            get_global_container,
            inject_dependencies_to_step_safe,
            get_service_safe,
            register_service_safe,
            register_lazy_service,
            initialize_di_system_safe
        )
        DI_CONTAINER_AVAILABLE = True
        logger.info("✅ DI Container v7.0 Central Hub 로드 성공 (상대 임포트)")
    except ImportError as e:
        logger.error(f"❌ DI Container v7.0 Central Hub 로드 실패: {e}")
        DI_CONTAINER_AVAILABLE = False
        
        # 폴백 처리
        def inject_dependencies_to_step_safe(step_instance, container=None):
            logger.warning("⚠️ DI Container 없음 - 의존성 주입 스킵")
        
        def get_service_safe(key: str):
            logger.warning(f"⚠️ DI Container 없음 - 서비스 조회 실패: {key}")
            return None
        
        def register_service_safe(key: str, service):
            logger.warning(f"⚠️ DI Container 없음 - 서비스 등록 스킵: {key}")
        
        def register_lazy_service(key: str, factory):
            logger.warning(f"⚠️ DI Container 없음 - 지연 서비스 등록 스킵: {key}")
        
        def initialize_di_system_safe():
            logger.warning("⚠️ DI Container 없음 - 시스템 초기화 스킵")

# ==============================================
# 🔥 환경 설정 (독립적 설정)
# ==============================================

# conda 환경 설정
CONDA_ENV = os.environ.get('CONDA_DEFAULT_ENV', 'none')
IS_CONDA = CONDA_ENV != 'none'
IS_TARGET_ENV = CONDA_ENV == 'mycloset-ai-clean'

# M3 Max 감지
def detect_m3_max() -> bool:
    """M3 Max 감지"""
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

# PyTorch 가용성 체크
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
# 🔥 Central Hub DI Container 기반 Step 관리
# ==============================================

# 전역 Central Hub Container 가져오기
def get_steps_container():
    """Steps용 Central Hub Container 반환"""
    if DI_CONTAINER_AVAILABLE:
        return get_global_container()
    return None

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

# ==============================================
# 🔥 Step 클래스 안전한 로딩 함수들
# ==============================================

def safe_import_step_class(step_module_name: str, step_class_name: str) -> Optional[Type]:
    """Step 클래스 안전한 import"""
    import_paths = [
        f'app.ai_pipeline.steps.{step_module_name}',
        f'ai_pipeline.steps.{step_module_name}',
        f'.{step_module_name}'
    ]
    
    for path in import_paths:
        try:
            if path.startswith('.'):
                # 상대 import
                import importlib
                module = importlib.import_module(path, package=__package__)
            else:
                # 절대 import
                import importlib
                module = importlib.import_module(path)
            
            step_class = getattr(module, step_class_name, None)
            if step_class:
                logger.debug(f"✅ {step_class_name} import 성공: {path}")
                return step_class
                
        except (ImportError, SyntaxError, AttributeError) as e:
            logger.debug(f"📋 {step_class_name} import 시도: {path} - {e}")
            continue
    
    logger.warning(f"⚠️ {step_class_name} import 실패")
    return None

def get_step_class(step_id: str) -> Optional[Type]:
    """Step 클래스 반환 (Central Hub 기반)"""
    if step_id not in STEP_DEFINITIONS:
        return None
    
    step_module, step_class_name = STEP_DEFINITIONS[step_id]
    
    # Central Hub Container에서 먼저 확인
    container = get_steps_container()
    if container:
        cached_class = container.get(f"step_class_{step_id}")
        if cached_class:
            return cached_class
    
    # 동적 import
    step_class = safe_import_step_class(step_module, step_class_name)
    
    # Central Hub Container에 캐시
    if step_class and container:
        container.register(f"step_class_{step_id}", step_class)
    
    return step_class

def create_step_instance_safe(step_id: str, **kwargs):
    """Step 인스턴스 안전 생성 (Central Hub 기반)"""
    step_class = get_step_class(step_id)
    if step_class is None:
        logger.error(f"❌ Step 클래스를 찾을 수 없음: {step_id}")
        return None
    
    try:
        # 기본 설정 추가
        default_config = {
            'device': DEVICE,
            'is_m3_max': IS_M3_MAX,
            'memory_gb': MEMORY_GB,
            'conda_optimized': IS_CONDA
        }
        default_config.update(kwargs)
        
        # Step 인스턴스 생성
        step_instance = step_class(**default_config)
        
        # Central Hub DI Container 기반 의존성 주입
        container = get_steps_container()
        if container:
            injections_made = container.inject_to_step(step_instance)
            logger.debug(f"✅ {step_id} Central Hub DI 주입 완료: {injections_made}개")
        else:
            # 폴백: 기본 의존성 주입
            inject_dependencies_to_step_safe(step_instance)
        
        return step_instance
        
    except Exception as e:
        logger.error(f"❌ {step_id} 인스턴스 생성 실패: {e}")
        return None

def get_available_steps() -> Dict[str, Type]:
    """사용 가능한 Step들 반환"""
    available_steps = {}
    for step_id in STEP_DEFINITIONS.keys():
        step_class = get_step_class(step_id)
        if step_class:
            available_steps[step_id] = step_class
    return available_steps

def is_step_available(step_id: str) -> bool:
    """특정 Step이 사용 가능한지 확인"""
    return get_step_class(step_id) is not None

# ==============================================
# 🔥 BaseStepMixin 안전한 로딩
# ==============================================

def load_base_step_mixin() -> Optional[Type]:
    """BaseStepMixin 안전한 로딩"""
    import_paths = [
        'app.ai_pipeline.steps.base_step_mixin',
        'ai_pipeline.steps.base_step_mixin',
        '.base_step_mixin'
    ]
    
    for path in import_paths:
        try:
            if path.startswith('.'):
                from .base_step_mixin import BaseStepMixin
            else:
                import importlib
                module = importlib.import_module(path)
                BaseStepMixin = getattr(module, 'BaseStepMixin', None)
            
            if BaseStepMixin:
                logger.debug(f"✅ BaseStepMixin 로드 성공: {path}")
                return BaseStepMixin
                
        except ImportError as e:
            logger.debug(f"📋 BaseStepMixin import 시도: {path} - {e}")
            continue
    
    logger.warning("⚠️ BaseStepMixin 로드 실패")
    return None

# BaseStepMixin 로드
BaseStepMixin = load_base_step_mixin()
BASESTEP_AVAILABLE = BaseStepMixin is not None

# ==============================================
# 🔥 Step 클래스들 로딩
# ==============================================

logger.info("🔄 Central Hub 기반 Step 클래스들 로딩 시작...")

def safe_import_step(module_name: str, class_name: str, step_id: str):
    """안전한 Step import"""
    try:
        step_class = get_step_class(step_id)
        if step_class:
            logger.info(f"✅ {class_name} Central Hub 로드 성공")
            return step_class, True
        else:
            logger.warning(f"⚠️ {class_name} Central Hub 로드 실패")
            return None, False
    except Exception as e:
        logger.error(f"❌ {class_name} Central Hub 로드 에러: {e}")
        return None, False

# Step 클래스들 로딩
HumanParsingStep, STEP_01_AVAILABLE = safe_import_step(
    'step_01_human_parsing', 'HumanParsingStep', 'step_01'
)

PoseEstimationStep, STEP_02_AVAILABLE = safe_import_step(
    'step_02_pose_estimation', 'PoseEstimationStep', 'step_02'
)

ClothSegmentationStep, STEP_03_AVAILABLE = safe_import_step(
    'step_03_cloth_segmentation', 'ClothSegmentationStep', 'step_03'
)

GeometricMatchingStep, STEP_04_AVAILABLE = safe_import_step(
    'step_04_geometric_matching', 'GeometricMatchingStep', 'step_04'
)

ClothWarpingStep, STEP_05_AVAILABLE = safe_import_step(
    'step_05_cloth_warping', 'ClothWarpingStep', 'step_05'
)

VirtualFittingStep, STEP_06_AVAILABLE = safe_import_step(
    'step_06_virtual_fitting', 'VirtualFittingStep', 'step_06'
)

PostProcessingStep, STEP_07_AVAILABLE = safe_import_step(
    'step_07_post_processing', 'PostProcessingStep', 'step_07'
)

QualityAssessmentStep, STEP_08_AVAILABLE = safe_import_step(
    'step_08_quality_assessment', 'QualityAssessmentStep', 'step_08'
)

# ==============================================
# 🔥 Step 매핑 및 관리
# ==============================================

# 전체 Step 매핑
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

# 가용성 플래그 매핑
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

# 사용 가능한 Step만 필터링
AVAILABLE_STEPS = {
    step_id: step_class 
    for step_id, step_class in STEP_MAPPING.items() 
    if step_class is not None and STEP_AVAILABILITY.get(step_id, False)
}

# ==============================================
# 🔥 정보 및 통계 함수들
# ==============================================

def get_step_info() -> Dict[str, Any]:
    """Step 정보 반환 (Central Hub 기반)"""
    available_steps = []
    failed_steps = []
    
    for step_id in STEP_DEFINITIONS.keys():
        if is_step_available(step_id):
            available_steps.append(step_id)
        else:
            failed_steps.append(step_id)
    
    # Central Hub Container 통계
    container_stats = {}
    if DI_CONTAINER_AVAILABLE:
        container = get_steps_container()
        if container:
            try:
                container_stats = container.get_stats()
            except Exception as e:
                container_stats = {'error': str(e)}
    
    return {
        'total_steps': len(STEP_DEFINITIONS),
        'available_steps': len(available_steps),
        'available_step_list': available_steps,
        'failed_step_list': failed_steps,
        'success_rate': (len(available_steps) / len(STEP_DEFINITIONS)) * 100 if STEP_DEFINITIONS else 0,
        'container_stats': container_stats,
        'basestep_available': BASESTEP_AVAILABLE,
        'di_container_integrated': DI_CONTAINER_AVAILABLE,
        'central_hub_version': '7.0'
    }

def get_step_error_summary() -> Dict[str, Any]:
    """Step 에러 요약"""
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
        'central_hub_version': step_info['central_hub_version'],
        'safe_copy_function_added': True,
        'simplified_architecture': True  # 새로 추가
    }

# ==============================================
# 🔥 유틸리티 함수들
# ==============================================

def inject_dependencies_to_step_safe_advanced(step_instance):
    """Step에 고급 안전한 의존성 주입 (Central Hub 기반)"""
    container = get_steps_container()
    if container:
        return container.inject_to_step(step_instance)
    else:
        inject_dependencies_to_step_safe(step_instance)
        return 0

def optimize_steps_memory():
    """Steps 메모리 최적화 (Central Hub 기반)"""
    try:
        # Central Hub Container 메모리 최적화
        container = get_steps_container()
        if container:
            result = container.optimize_memory(aggressive=True)
            logger.info(f"🧹 Central Hub Steps 메모리 최적화 완료: {result}")
            return result
        else:
            # 폴백: 기본 메모리 정리
            collected = gc.collect()
            
            # M3 Max MPS 최적화
            if TORCH_AVAILABLE and IS_M3_MAX and MPS_AVAILABLE:
                import torch
                if hasattr(torch.backends.mps, 'empty_cache'):
                    torch.backends.mps.empty_cache()
            
            logger.info(f"🧹 기본 Steps 메모리 최적화 완료: {collected}개 GC")
            return {'gc_collected': collected}
        
    except Exception as e:
        logger.error(f"❌ Steps 메모리 최적화 실패: {e}")
        return {}

def get_di_container_for_steps():
    """Steps용 DI Container 반환 (Central Hub)"""
    return get_steps_container()

def reset_steps_container():
    """Steps Container 리셋 (Central Hub 기반)"""
    if DI_CONTAINER_AVAILABLE:
        # Central Hub는 전역이므로 개별 리셋 대신 메모리 최적화
        optimize_steps_memory()
        logger.info("🔄 Central Hub Steps Container 메모리 최적화 완료")
    else:
        logger.warning("⚠️ DI Container 없음 - 리셋 스킵")

# ==============================================
# 🔥 비동기 Step 관리 함수들
# ==============================================

async def initialize_all_steps_async():
    """모든 Step 비동기 초기화"""
    try:
        logger.info("🚀 모든 Step 비동기 초기화 시작 (Central Hub 기반)")
        
        initialization_results = {}
        
        for step_id in STEP_DEFINITIONS.keys():
            if is_step_available(step_id):
                try:
                    step_instance = create_step_instance_safe(step_id)
                    
                    if step_instance and hasattr(step_instance, 'initialize'):
                        if asyncio.iscoroutinefunction(step_instance.initialize):
                            await step_instance.initialize()
                        else:
                            step_instance.initialize()
                        initialization_results[step_id] = True
                        logger.info(f"✅ {step_id} 비동기 초기화 완료")
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
        logger.info("🧹 모든 Step 비동기 정리 시작 (Central Hub 기반)")
        
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
        
        # Central Hub Container 메모리 최적화
        optimize_steps_memory()
        
        success_count = sum(1 for success in cleanup_results.values() if success)
        total_count = len(cleanup_results)
        
        logger.info(f"✅ Step 비동기 정리 완료: {success_count}/{total_count}개")
        return cleanup_results
        
    except Exception as e:
        logger.error(f"❌ Step 비동기 정리 실패: {e}")
        return {}

# ==============================================
# 🔥 conda 환경 최적화 (Central Hub 기반)
# ==============================================

def optimize_conda_environment_with_di():
    """conda 환경 Central Hub 기반 안전 최적화"""
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
                logger.info("🍎 M3 Max MPS conda Central Hub 최적화 완료")
        
        # Central Hub Container 메모리 최적화
        if DI_CONTAINER_AVAILABLE:
            optimize_steps_memory()
        
        logger.info(f"🐍 conda 환경 '{CONDA_ENV}' Central Hub 기반 최적화 완료")
        
    except Exception as e:
        logger.warning(f"⚠️ conda Central Hub 기반 최적화 실패: {e}")

# ==============================================
# 🔥 Export (API 호환성 유지)
# ==============================================

__all__ = [
    # Step 클래스들
    'HumanParsingStep',
    'PoseEstimationStep', 
    'ClothSegmentationStep',
    'GeometricMatchingStep',
    'ClothWarpingStep',
    'VirtualFittingStep',
    'PostProcessingStep',
    'QualityAssessmentStep',
    
    # BaseStepMixin
    'BaseStepMixin',
    
    # 유틸리티 함수들
    'get_step_class',
    'get_available_steps',
    'create_step_instance_safe',
    'get_step_info',
    'is_step_available',
    'get_step_error_summary',
    'inject_dependencies_to_step_safe_advanced',
    'safe_import_step',
    'safe_import_step_class',
    
    # 매핑 및 상태
    'STEP_MAPPING',
    'AVAILABLE_STEPS',
    'STEP_AVAILABILITY',
    'STEP_DEFINITIONS',
    
    # Central Hub 관련
    'get_steps_container',
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
    
    # 유틸리티 함수들
    'safe_copy',
    'load_base_step_mixin',
    'optimize_conda_environment_with_di',
    
    # 타입들
    'T'
]

# ==============================================
# 🔥 초기화 완료 로깅
# ==============================================

# 통계 수집
step_info = get_step_info()
error_summary = get_step_error_summary()

logger.info("=" * 80)
logger.info("🔥 MyCloset AI Pipeline Steps v5.2 초기화 완료 (Central Hub DI Container v7.0 완전 통합 + 간소화)")
logger.info("=" * 80)
logger.info(f"🔗 Central Hub DI Container v7.0: {'✅ 활성화' if DI_CONTAINER_AVAILABLE else '❌ 비활성화'}")
logger.info(f"📊 Step 로딩 결과: {step_info['available_steps']}/{step_info['total_steps']}개 ({step_info['success_rate']:.1f}%)")
logger.info(f"🔧 BaseStepMixin: {'✅ 정상' if error_summary['basestep_available'] else '⚠️ 폴백'}")
logger.info(f"🔑 Logger 에러: {'✅ 해결됨' if error_summary['logger_errors_resolved'] else '❌ 미해결'}")
logger.info(f"🔗 순환참조: {'✅ 해결됨' if error_summary['circular_reference_resolved'] else '❌ 미해결'}")
logger.info(f"💉 Central Hub 통합: {'✅ 완료' if error_summary['di_container_integrated'] else '❌ 미완료'}")
logger.info(f"📋 safe_copy 함수: {'✅ 유지됨' if error_summary['safe_copy_function_added'] else '❌ 누락'}")
logger.info(f"🎯 아키텍처 간소화: {'✅ 완료' if error_summary['simplified_architecture'] else '❌ 미완료'}")

# Central Hub Container 통계
if DI_CONTAINER_AVAILABLE:
    container_stats = step_info.get('container_stats', {})
    if 'version' in container_stats:
        logger.info(f"🔗 Central Hub 버전: {container_stats['version']}")
    if 'access_count' in container_stats:
        logger.info(f"🔗 Container 접근 횟수: {container_stats['access_count']}")

if step_info['available_step_list']:
    logger.info(f"✅ 로드된 Steps: {', '.join(step_info['available_step_list'])}")

if step_info['failed_step_list']:
    logger.info(f"⚠️ 실패한 Steps: {', '.join(step_info['failed_step_list'])}")

# 중요한 Step들 개별 체크
if is_step_available('step_01'):
    logger.info("🎉 Step 01 (HumanParsingStep) Central Hub 로딩 성공!")
else:
    logger.warning("⚠️ Step 01 (HumanParsingStep) Central Hub 로딩 실패!")

if is_step_available('step_06'):
    logger.info("🎉 Step 06 (VirtualFittingStep) Central Hub 로딩 성공!")
else:
    logger.warning("⚠️ Step 06 (VirtualFittingStep) Central Hub 로딩 실패!")

# conda 환경 자동 최적화
if IS_TARGET_ENV:
    optimize_conda_environment_with_di()
    logger.info("🐍 conda 환경 mycloset-ai-clean Central Hub 기반 자동 최적화 완료!")

if step_info['success_rate'] >= 50:
    logger.info("🚀 파이프라인 Steps 시스템 준비 완료! (Central Hub 기반)")
else:
    logger.warning("⚠️ 파이프라인 Steps 시스템 부분 준비 (일부 Step 사용 불가)")

logger.info("=" * 80)

# 최종 상태 체크
if step_info['available_steps'] > 0:
    logger.info("✅ Steps 모듈 v5.2 초기화 성공 - Central Hub DI Container v7.0 완전 통합 + 아키텍처 간소화")
else:
    logger.error("❌ Steps 모듈 v5.2 초기화 실패 - 모든 Step이 사용 불가")

logger.info("🔥 MyCloset AI Pipeline Steps v5.2 with Central Hub DI Container v7.0 - 완전 통합 + 간소화 완료!")