
# ============================================================================
# 📁 backend/app/ai_pipeline/__init__.py - AI 파이프라인 모듈
# ============================================================================

"""
🤖 MyCloset AI Pipeline 모듈 - conda 환경 우선 AI 파이프라인
===========================================================

✅ conda 환경 우선 최적화
✅ 순환참조 완전 방지 (지연 로딩 패턴) 
✅ 8단계 AI 파이프라인 통합 관리
✅ Step 클래스들 안전한 로딩
✅ ModelLoader, MemoryManager 통합
✅ M3 Max 128GB 메모리 최적화
✅ 동적 AI 모델 로딩

역할: AI 파이프라인의 전체 라이프사이클과 Step 관리를 담당
"""

import os
import sys
import logging
import threading
import time
from pathlib import Path
from typing import Dict, Any, Optional, List, Type, Union

# 상위 패키지에서 시스템 정보 가져오기
try:
    from .. import SYSTEM_INFO, AI_MODEL_PATHS, IS_CONDA, CONDA_ENV, _lazy_loader
except ImportError:
    SYSTEM_INFO = {'device': 'cpu', 'is_m3_max': False, 'memory_gb': 16.0}
    AI_MODEL_PATHS = {'ai_models_root': Path(__file__).parent.parent.parent / 'ai_models'}
    IS_CONDA = 'CONDA_DEFAULT_ENV' in os.environ
    CONDA_ENV = os.environ.get('CONDA_DEFAULT_ENV', 'none')
    _lazy_loader = None

# 로거 설정
logger = logging.getLogger(__name__)

# =============================================================================
# 🔥 AI Pipeline 정보
# =============================================================================

__version__ = "4.0.0"
__author__ = "MyCloset AI Team"
__description__ = "AI Pipeline System with Conda Priority and Lazy Loading"

# Step 정보
STEP_MODULES = {
    'step_01': 'step_01_human_parsing',
    'step_02': 'step_02_pose_estimation',
    'step_03': 'step_03_cloth_segmentation',
    'step_04': 'step_04_geometric_matching',
    'step_05': 'step_05_cloth_warping',
    'step_06': 'step_06_virtual_fitting',
    'step_07': 'step_07_post_processing',
    'step_08': 'step_08_quality_assessment'
}

STEP_CLASSES = {
    'step_01': 'HumanParsingStep',
    'step_02': 'PoseEstimationStep',
    'step_03': 'ClothSegmentationStep',
    'step_04': 'GeometricMatchingStep',
    'step_05': 'ClothWarpingStep',
    'step_06': 'VirtualFittingStep',
    'step_07': 'PostProcessingStep',
    'step_08': 'QualityAssessmentStep'
}

# =============================================================================
# 🔥 지연 로딩 함수들 (순환참조 방지)
# =============================================================================

def get_pipeline_manager_class():
    """PipelineManager 클래스 지연 로딩"""
    if _lazy_loader:
        return _lazy_loader.get_class('pipeline_manager', 'PipelineManager', 'app.ai_pipeline')
    
    try:
        from .pipeline_manager import PipelineManager
        return PipelineManager
    except ImportError as e:
        logger.warning(f"PipelineManager 클래스 로딩 실패: {e}")
        return None

def get_model_loader_class():
    """ModelLoader 클래스 지연 로딩"""
    if _lazy_loader:
        return _lazy_loader.get_class('model_loader', 'ModelLoader', 'app.ai_pipeline.utils')
    
    try:
        from .utils.model_loader import ModelLoader
        return ModelLoader
    except ImportError as e:
        logger.warning(f"ModelLoader 클래스 로딩 실패: {e}")
        return None

def get_memory_manager_class():
    """MemoryManager 클래스 지연 로딩"""
    if _lazy_loader:
        return _lazy_loader.get_class('memory_manager', 'MemoryManager', 'app.ai_pipeline.utils')
    
    try:
        from .utils.memory_manager import MemoryManager
        return MemoryManager
    except ImportError as e:
        logger.warning(f"MemoryManager 클래스 로딩 실패: {e}")
        return None

def get_step_factory_class():
    """StepFactory 클래스 지연 로딩"""
    if _lazy_loader:
        return _lazy_loader.get_class('step_factory', 'StepFactory', 'app.ai_pipeline.factories')
    
    try:
        from .factories.step_factory import StepFactory
        return StepFactory
    except ImportError as e:
        logger.warning(f"StepFactory 클래스 로딩 실패: {e}")
        return None

# =============================================================================
# 🔥 Step 클래스 지연 로딩 (순환참조 방지)
# =============================================================================

def safe_import_step(step_id: str) -> Optional[Type[Any]]:
    """안전한 Step 클래스 import (지연 로딩)"""
    try:
        module_name = STEP_MODULES.get(step_id)
        class_name = STEP_CLASSES.get(step_id)
        
        if not module_name or not class_name:
            logger.error(f"❌ 알 수 없는 Step ID: {step_id}")
            return None
        
        if _lazy_loader:
            return _lazy_loader.get_class(module_name, class_name, 'app.ai_pipeline.steps')
        
        # 직접 import (폴백)
        try:
            import importlib
            full_module_name = f"app.ai_pipeline.steps.{module_name}"
            module = importlib.import_module(full_module_name)
            step_class = getattr(module, class_name, None)
            
            if step_class:
                logger.debug(f"✅ {step_id} ({class_name}) import 성공")
                return step_class
            else:
                logger.error(f"❌ {class_name} 클래스를 {module_name}에서 찾을 수 없음")
                return None
                
        except ImportError as e:
            logger.warning(f"❌ {step_id} import 실패: {e}")
            return None
        
    except Exception as e:
        logger.error(f"❌ {step_id} 예상치 못한 오류: {e}")
        return None

def load_all_steps() -> Dict[str, Optional[Type[Any]]]:
    """모든 Step 클래스 지연 로딩"""
    loaded_steps = {}
    
    for step_id in STEP_MODULES.keys():
        step_class = safe_import_step(step_id)
        loaded_steps[step_id] = step_class
    
    available_count = sum(1 for step in loaded_steps.values() if step is not None)
    logger.info(f"✅ Step 로딩 완료: {available_count}/8개")
    
    return loaded_steps

# =============================================================================
# 🔥 팩토리 함수들 (conda 환경 최적화)
# =============================================================================

def create_pipeline_manager(**kwargs) -> Optional[Any]:
    """PipelineManager 인스턴스 생성 (conda 환경 최적화)"""
    PipelineManager = get_pipeline_manager_class()
    if PipelineManager:
        # conda 환경 설정 추가
        pipeline_config = {
            'device': SYSTEM_INFO.get('device', 'cpu'),
            'is_m3_max': SYSTEM_INFO.get('is_m3_max', False),
            'memory_gb': SYSTEM_INFO.get('memory_gb', 16.0),
            'conda_optimized': IS_CONDA,
            'conda_env': CONDA_ENV
        }
        pipeline_config.update(kwargs)
        
        try:
            return PipelineManager(**pipeline_config)
        except Exception as e:
            logger.error(f"PipelineManager 생성 실패: {e}")
            return None
    return None

def create_model_loader(**kwargs) -> Optional[Any]:
    """ModelLoader 인스턴스 생성 (conda 환경 최적화)"""
    ModelLoader = get_model_loader_class()
    if ModelLoader:
        # conda 환경 모델 로딩 설정
        loader_config = {
            'device': SYSTEM_INFO.get('device', 'cpu'),
            'models_path': str(AI_MODEL_PATHS.get('ai_models_root', '.')),
            'conda_optimized': IS_CONDA,
            'memory_efficient': SYSTEM_INFO.get('is_m3_max', False)
        }
        loader_config.update(kwargs)
        
        try:
            return ModelLoader(**loader_config)
        except Exception as e:
            logger.error(f"ModelLoader 생성 실패: {e}")
            return None
    return None

def create_step_instance(step_name: Union[str, int], **kwargs) -> Optional[Any]:
    """Step 인스턴스 생성 (conda 환경 최적화)"""
    try:
        if isinstance(step_name, int):
            step_key = f"step_{step_name:02d}"
        else:
            step_key = step_name
        
        step_class = safe_import_step(step_key)
        if step_class is None:
            logger.error(f"Step 클래스를 찾을 수 없음: {step_name}")
            return None
        
        # conda 환경 Step 설정
        step_config = {
            'device': SYSTEM_INFO.get('device', 'cpu'),
            'is_m3_max': SYSTEM_INFO.get('is_m3_max', False),
            'memory_gb': SYSTEM_INFO.get('memory_gb', 16.0),
            'conda_optimized': IS_CONDA
        }
        step_config.update(kwargs)
        
        return step_class(**step_config)
        
    except Exception as e:
        logger.error(f"Step 인스턴스 생성 실패 {step_name}: {e}")
        return None

# =============================================================================
# 🔥 전역 인스턴스 관리 (싱글톤 패턴)
# =============================================================================

_global_instances = {}
_instance_lock = threading.RLock()

def get_global_pipeline_manager():
    """전역 PipelineManager 인스턴스 반환"""
    with _instance_lock:
        if 'pipeline_manager' not in _global_instances:
            _global_instances['pipeline_manager'] = create_pipeline_manager()
        return _global_instances['pipeline_manager']

def get_global_model_loader():
    """전역 ModelLoader 인스턴스 반환"""
    with _instance_lock:
        if 'model_loader' not in _global_instances:
            _global_instances['model_loader'] = create_model_loader()
        return _global_instances['model_loader']

# =============================================================================
# 🔥 파이프라인 상태 관리
# =============================================================================

def get_pipeline_status() -> Dict[str, Any]:
    """파이프라인 시스템 상태 반환"""
    loaded_steps = load_all_steps()
    available_steps = [k for k, v in loaded_steps.items() if v is not None]
    
    return {
        'version': __version__,
        'system_info': SYSTEM_INFO,
        'conda_environment': IS_CONDA,
        'conda_env_name': CONDA_ENV,
        'availability': {
            'pipeline_manager': get_pipeline_manager_class() is not None,
            'model_loader': get_model_loader_class() is not None,
            'memory_manager': get_memory_manager_class() is not None,
            'step_factory': get_step_factory_class() is not None,
        },
        'steps': {
            'total_steps': len(STEP_MODULES),
            'available_steps': len(available_steps),
            'loaded_steps': available_steps,
            'step_classes': {k: v is not None for k, v in loaded_steps.items()}
        },
        'ai_models': {
            'models_path': str(AI_MODEL_PATHS.get('ai_models_root', '')),
            'models_exist': AI_MODEL_PATHS.get('ai_models_root', Path('.')).exists()
        }
    }

def list_available_steps() -> List[str]:
    """사용 가능한 Step 목록 반환"""
    loaded_steps = load_all_steps()
    return [k for k, v in loaded_steps.items() if v is not None]

# =============================================================================
# 🔥 초기화 함수들
# =============================================================================

async def initialize_pipeline_system(**kwargs) -> Dict[str, Any]:
    """전체 파이프라인 시스템 초기화 (conda 환경 최적화)"""
    try:
        start_time = time.time()
        results = {}
        
        logger.info("🚀 AI 파이프라인 시스템 초기화 시작...")
        
        # 1. ModelLoader 초기화
        try:
            model_loader = create_model_loader(**kwargs)
            results['model_loader'] = {
                'success': model_loader is not None,
                'instance': model_loader
            }
            if model_loader:
                logger.info("✅ ModelLoader 초기화 완료")
        except Exception as e:
            logger.error(f"❌ ModelLoader 초기화 실패: {e}")
            results['model_loader'] = {'success': False, 'error': str(e)}
        
        # 2. PipelineManager 초기화
        try:
            pipeline_manager = create_pipeline_manager(**kwargs)
            results['pipeline_manager'] = {
                'success': pipeline_manager is not None,
                'instance': pipeline_manager
            }
            if pipeline_manager:
                logger.info("✅ PipelineManager 초기화 완료")
        except Exception as e:
            logger.error(f"❌ PipelineManager 초기화 실패: {e}")
            results['pipeline_manager'] = {'success': False, 'error': str(e)}
        
        # 3. Step 클래스들 로딩
        try:
            loaded_steps = load_all_steps()
            results['steps'] = {
                'success': len(loaded_steps) > 0,
                'loaded_count': sum(1 for step in loaded_steps.values() if step is not None),
                'total_count': len(STEP_MODULES),
                'steps': loaded_steps
            }
            logger.info(f"✅ Step 클래스 로딩 완료: {results['steps']['loaded_count']}/8개")
        except Exception as e:
            logger.error(f"❌ Step 클래스 로딩 실패: {e}")
            results['steps'] = {'success': False, 'error': str(e)}
        
        # 초기화 완료
        initialization_time = time.time() - start_time
        results['overall'] = {
            'success': any(result.get('success', False) for result in results.values()),
            'initialization_time': initialization_time,
            'conda_optimized': IS_CONDA,
            'device': SYSTEM_INFO.get('device', 'cpu')
        }
        
        logger.info(f"🎉 AI 파이프라인 시스템 초기화 완료 ({initialization_time:.2f}초)")
        return results
        
    except Exception as e:
        logger.error(f"❌ AI 파이프라인 시스템 초기화 실패: {e}")
        return {'overall': {'success': False, 'error': str(e)}}

def cleanup_pipeline_system():
    """파이프라인 시스템 정리"""
    try:
        logger.info("🧹 AI 파이프라인 시스템 정리 시작...")
        
        # 전역 인스턴스 정리
        with _instance_lock:
            for name, instance in _global_instances.items():
                try:
                    if hasattr(instance, 'cleanup'):
                        instance.cleanup()
                except Exception as e:
                    logger.warning(f"인스턴스 정리 실패 {name}: {e}")
            
            _global_instances.clear()
        
        # 메모리 정리
        import gc
        gc.collect()
        
        # MPS 캐시 정리 (M3 Max)
        if SYSTEM_INFO.get('device') == 'mps':
            try:
                import torch
                if hasattr(torch.mps, 'empty_cache'):
                    torch.mps.empty_cache()
            except Exception as e:
                logger.warning(f"MPS 캐시 정리 실패: {e}")
        
        logger.info("✅ AI 파이프라인 시스템 정리 완료")
        
    except Exception as e:
        logger.error(f"❌ AI 파이프라인 시스템 정리 실패: {e}")

# =============================================================================
# 🔥 AI Pipeline 모듈 Export
# =============================================================================

__all__ = [
    # 🔥 버전 정보
    '__version__',
    '__author__',
    '__description__',
    
    # 📊 Step 정보
    'STEP_MODULES',
    'STEP_CLASSES',
    
    # 🔗 지연 로딩 함수들
    'get_pipeline_manager_class',
    'get_model_loader_class',
    'get_memory_manager_class',
    'get_step_factory_class',
    
    # 🔧 Step 관리 함수들
    'safe_import_step',
    'load_all_steps',
    'list_available_steps',
    
    # 🏭 팩토리 함수들
    'create_pipeline_manager',
    'create_model_loader',
    'create_step_instance',
    
    # 🌍 전역 인스턴스 함수들
    'get_global_pipeline_manager',
    'get_global_model_loader',
    
    # 🔧 상태 관리 함수들
    'get_pipeline_status',
    
    # 🚀 초기화 함수들
    'initialize_pipeline_system',
    'cleanup_pipeline_system',
]

# 초기화 정보 출력
logger.info("🤖 MyCloset AI Pipeline 모듈 초기화 완료")
logger.info(f"🐍 conda 최적화: {IS_CONDA}")
logger.info(f"🍎 M3 Max: {SYSTEM_INFO.get('is_m3_max', False)}")
logger.info(f"📊 총 Step 수: {len(STEP_MODULES)}")
logger.info(f"🔗 지연 로딩: 활성화")
