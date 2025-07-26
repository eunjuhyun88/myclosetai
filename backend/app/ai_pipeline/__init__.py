#!/usr/bin/env python3
"""
🔥 MyCloset AI 파이프라인 시스템 v7.1 - Step 01 경로 문제 완전 해결
================================================================

✅ Step 01 모듈 경로 문제 해결 (human_body_parsing → human_parsing)
✅ 정확한 파일명 매핑 시스템
✅ 복잡한 동적 생성 제거
✅ 직접 매핑으로 단순화

문제 해결:
- 기존: step_01_human_body_parsing (잘못된 경로)
- 수정: step_01_human_parsing (올바른 경로)

Author: MyCloset AI Team
Date: 2025-07-25
Version: v7.1 (Step 01 Path Fix)
"""

import logging
import sys
import warnings
from typing import Dict, Any, Optional, List, Type
from pathlib import Path

# 경고 무시
warnings.filterwarnings('ignore')

# =============================================================================
# 🔥 기본 설정 및 로깅
# =============================================================================

logger = logging.getLogger(__name__)

# 상위 패키지에서 시스템 정보 가져오기
try:
    from .. import get_system_info, is_conda_environment, is_m3_max, get_device
    SYSTEM_INFO = get_system_info()
    IS_CONDA = is_conda_environment()
    IS_M3_MAX = is_m3_max()
    DEVICE = get_device()
    logger.info("✅ 상위 패키지에서 시스템 정보 로드 성공")
except ImportError as e:
    logger.warning(f"⚠️ 상위 패키지 로드 실패, 기본값 사용: {e}")
    SYSTEM_INFO = {'device': 'cpu', 'is_m3_max': False, 'memory_gb': 16.0}
    IS_CONDA = False
    IS_M3_MAX = False
    DEVICE = 'cpu'

# =============================================================================
# 🔥 정확한 Step 매핑 (파일명 기반)
# =============================================================================

# Step별 정확한 모듈명과 클래스명 매핑
STEP_MAPPING = {
    'step_01': {
        'module': 'app.ai_pipeline.steps.step_01_human_parsing',  # 🔥 올바른 경로
        'class': 'HumanParsingStep',
        'description': '인체 파싱 - Human Body Parsing',
        'models': ['SCHP', 'Graphonomy'],
        'priority': 2
    },
    'step_02': {
        'module': 'app.ai_pipeline.steps.step_02_pose_estimation',
        'class': 'PoseEstimationStep',
        'description': '포즈 추정 - Pose Estimation',
        'models': ['OpenPose', 'YOLO-Pose'],
        'priority': 4
    },
    'step_03': {
        'module': 'app.ai_pipeline.steps.step_03_cloth_segmentation',
        'class': 'ClothSegmentationStep',
        'description': '의류 분할 - Cloth Segmentation',
        'models': ['U2Net', 'SAM'],
        'priority': 3
    },
    'step_04': {
        'module': 'app.ai_pipeline.steps.step_04_geometric_matching',
        'class': 'GeometricMatchingStep',
        'description': '기하학적 매칭 - Geometric Matching',
        'models': ['TPS', 'GMM'],
        'priority': 7
    },
    'step_05': {
        'module': 'app.ai_pipeline.steps.step_05_cloth_warping',
        'class': 'ClothWarpingStep',
        'description': '의류 변형 - Cloth Warping',
        'models': ['Advanced Warping'],
        'priority': 8
    },
    'step_06': {
        'module': 'app.ai_pipeline.steps.step_06_virtual_fitting',
        'class': 'VirtualFittingStep',
        'description': '가상 피팅 - Virtual Fitting',
        'models': ['OOTDiffusion', 'IDM-VTON'],
        'priority': 1  # 가장 중요
    },
    'step_07': {
        'module': 'app.ai_pipeline.steps.step_07_post_processing',
        'class': 'PostProcessingStep',
        'description': '후처리 - Post Processing',
        'models': ['RealESRGAN', 'Enhancement'],
        'priority': 5
    },
    'step_08': {
        'module': 'app.ai_pipeline.steps.step_08_quality_assessment',
        'class': 'QualityAssessmentStep',
        'description': '품질 평가 - Quality Assessment',
        'models': ['CLIP', 'Quality Metrics'],
        'priority': 6
    }
}

# conda 환경에서 로딩 우선순위
LOADING_PRIORITY = sorted(STEP_MAPPING.keys(), 
                         key=lambda x: STEP_MAPPING[x]['priority'])

# =============================================================================
# 🔥 단순화된 파이프라인 로더 (정확한 경로 사용)
# =============================================================================

class FixedPipelineLoader:
    """수정된 파이프라인 로더 - 정확한 경로 사용"""
    
    def __init__(self):
        self._loaded_classes = {}
        self._failed_loads = set()
        self.logger = logging.getLogger(f"{__name__}.FixedPipelineLoader")
        
    def safe_import_step(self, step_id: str) -> Optional[Type]:
        """안전한 Step 클래스 import (정확한 경로 사용)"""
        if step_id in self._loaded_classes:
            return self._loaded_classes[step_id]
            
        if step_id in self._failed_loads:
            return None
            
        try:
            step_info = STEP_MAPPING.get(step_id)
            if not step_info:
                self.logger.warning(f"⚠️ 알 수 없는 Step ID: {step_id}")
                return None
                
            # 🔥 정확한 모듈명과 클래스명 사용
            module_name = step_info['module']
            class_name = step_info['class']
            
            # 동적 import 시도
            import importlib
            try:
                self.logger.debug(f"🔄 {step_id} 로딩 시도: {module_name}")
                module = importlib.import_module(module_name)
                step_class = getattr(module, class_name, None)
                
                if step_class:
                    self._loaded_classes[step_id] = step_class
                    self.logger.info(f"✅ {step_id} ({class_name}) 로드 성공")
                    return step_class
                else:
                    self.logger.warning(f"⚠️ {class_name} 클래스를 찾을 수 없음")
                    
            except ImportError as e:
                self.logger.debug(f"📋 {step_id} 모듈 없음 (정상): {e}")
                
        except Exception as e:
            self.logger.error(f"❌ {step_id} 로드 실패: {e}")
            
        # 실패 기록
        self._failed_loads.add(step_id)
        self._loaded_classes[step_id] = None
        return None
        
    def load_all_available_steps(self) -> Dict[str, Optional[Type]]:
        """사용 가능한 모든 Step 로드"""
        loaded_steps = {}
        
        # conda 환경이면 우선순위 순으로 로드
        load_order = LOADING_PRIORITY if IS_CONDA else STEP_MAPPING.keys()
        
        for step_id in load_order:
            step_class = self.safe_import_step(step_id)
            loaded_steps[step_id] = step_class
            
        available_count = sum(1 for step in loaded_steps.values() if step is not None)
        total_count = len(STEP_MAPPING)
        
        self.logger.info(f"📊 Step 로딩 완료: {available_count}/{total_count}개")
        if IS_CONDA:
            self.logger.info("🐍 conda 환경: 우선순위 기반 로딩 적용")
            
        return loaded_steps

# 전역 로더 인스턴스
_pipeline_loader = FixedPipelineLoader()

# =============================================================================
# 🔥 유틸리티 모듈 안전한 로딩
# =============================================================================

def _safe_import_utils():
    """유틸리티 모듈들 안전하게 import"""
    utils_status = {
        'model_loader': False,
        'memory_manager': False,
        'data_converter': False,
        'model_interface': False
    }
    
    try:
        from .utils import (
            get_step_model_interface,
            get_step_memory_manager, 
            get_step_data_converter,
            preprocess_image_for_step
        )
        utils_status.update({
            'model_loader': True,
            'memory_manager': True,
            'data_converter': True,
            'model_interface': True
        })
        logger.info("✅ 파이프라인 유틸리티 모듈 로드 성공")
        
        # 전역에 추가
        globals().update({
            'get_step_model_interface': get_step_model_interface,
            'get_step_memory_manager': get_step_memory_manager,
            'get_step_data_converter': get_step_data_converter,
            'preprocess_image_for_step': preprocess_image_for_step
        })
        
    except ImportError as e:
        logger.warning(f"⚠️ 유틸리티 모듈 로드 실패: {e}")
        
        # 폴백 함수들
        def _fallback_function(name: str):
            def fallback(*args, **kwargs):
                logger.warning(f"⚠️ {name} 함수 사용 불가 (모듈 로드 실패)")
                return None
            return fallback
            
        globals().update({
            'get_step_model_interface': _fallback_function('get_step_model_interface'),
            'get_step_memory_manager': _fallback_function('get_step_memory_manager'),
            'get_step_data_converter': _fallback_function('get_step_data_converter'),
            'preprocess_image_for_step': _fallback_function('preprocess_image_for_step')
        })
    
    return utils_status

# 유틸리티 로딩
UTILS_STATUS = _safe_import_utils()

# =============================================================================
# 🔥 파이프라인 관리 함수들
# =============================================================================

def get_pipeline_status() -> Dict[str, Any]:
    """파이프라인 전체 상태 반환"""
    loaded_steps = _pipeline_loader.load_all_available_steps()
    available_steps = [k for k, v in loaded_steps.items() if v is not None]
    
    return {
        'system_info': SYSTEM_INFO,
        'conda_optimized': IS_CONDA,
        'm3_max_optimized': IS_M3_MAX,
        'device': DEVICE,
        'total_steps': len(STEP_MAPPING),
        'available_steps': len(available_steps),
        'loaded_steps': available_steps,
        'failed_steps': [k for k, v in loaded_steps.items() if v is None],
        'success_rate': (len(available_steps) / len(STEP_MAPPING)) * 100,
        'utils_status': UTILS_STATUS,
        'loading_priority': LOADING_PRIORITY if IS_CONDA else None
    }

def get_step_class(step_name: str) -> Optional[Type]:
    """Step 클래스 반환"""
    if step_name.startswith('step_'):
        return _pipeline_loader.safe_import_step(step_name)
    else:
        # 클래스명으로 검색
        for step_id, step_info in STEP_MAPPING.items():
            if step_info['class'] == step_name:
                return _pipeline_loader.safe_import_step(step_id)
    return None

def create_step_instance(step_name: str, **kwargs) -> Optional[Any]:
    """Step 인스턴스 생성"""
    step_class = get_step_class(step_name)
    if step_class is None:
        logger.error(f"❌ Step 클래스를 찾을 수 없음: {step_name}")
        return None
        
    try:
        # 기본 설정 추가
        default_config = {
            'device': DEVICE,
            'is_m3_max': IS_M3_MAX,
            'memory_gb': SYSTEM_INFO.get('memory_gb', 16.0),
            'conda_optimized': IS_CONDA
        }
        default_config.update(kwargs)
        
        return step_class(**default_config)
        
    except Exception as e:
        logger.error(f"❌ Step 인스턴스 생성 실패 {step_name}: {e}")
        return None

def list_available_steps() -> List[str]:
    """사용 가능한 Step 목록 반환"""
    loaded_steps = _pipeline_loader.load_all_available_steps()
    return [step_id for step_id, step_class in loaded_steps.items() if step_class is not None]

def get_step_info(step_id: str) -> Dict[str, Any]:
    """Step 정보 반환"""
    step_config = STEP_MAPPING.get(step_id, {})
    step_class = _pipeline_loader._loaded_classes.get(step_id)
    
    return {
        'step_id': step_id,
        'module': step_config.get('module', ''),
        'class': step_config.get('class', 'Unknown'),
        'description': step_config.get('description', ''),
        'models': step_config.get('models', []),
        'priority': step_config.get('priority', 10),
        'available': step_class is not None,
        'loaded': step_class is not None,
        'failed': step_id in _pipeline_loader._failed_loads
    }

async def initialize_pipeline_system() -> bool:
    """파이프라인 시스템 초기화"""
    try:
        logger.info("🚀 파이프라인 시스템 초기화 시작")
        
        # Step 클래스들 로드
        loaded_steps = _pipeline_loader.load_all_available_steps()
        available_count = sum(1 for step in loaded_steps.values() if step is not None)
        
        logger.info(f"✅ 파이프라인 시스템 초기화 완료: {available_count}/{len(STEP_MAPPING)}개 Step")
        return available_count > 0
        
    except Exception as e:
        logger.error(f"❌ 파이프라인 시스템 초기화 실패: {e}")
        return False

async def cleanup_pipeline_system() -> None:
    """파이프라인 시스템 정리"""
    try:
        logger.info("🧹 파이프라인 시스템 정리 시작")
        
        # 캐시 정리
        _pipeline_loader._loaded_classes.clear()
        _pipeline_loader._failed_loads.clear()
        
        # GPU 메모리 정리 (가능한 경우)
        if DEVICE in ['cuda', 'mps']:
            try:
                import torch
                if DEVICE == 'cuda' and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                elif DEVICE == 'mps' and torch.backends.mps.is_available():
                    # M3 Max 메모리 정리 (안전하게)
                    import gc
                    gc.collect()
            except:
                pass
                
        logger.info("✅ 파이프라인 시스템 정리 완료")
        
    except Exception as e:
        logger.warning(f"⚠️ 파이프라인 시스템 정리 실패: {e}")

# =============================================================================
# 🔥 자동 Step 클래스 로딩 (전역 변수)
# =============================================================================

# 사용 가능한 Step 클래스들을 전역 변수로 설정
try:
    _loaded_steps = _pipeline_loader.load_all_available_steps()
    
    # 개별 Step 클래스들을 전역에 추가
    for step_id, step_class in _loaded_steps.items():
        if step_class:
            step_info = STEP_MAPPING[step_id]
            class_name = step_info['class']
            globals()[class_name] = step_class
            
    logger.info("✅ Step 클래스들 전역 설정 완료")
    
except Exception as e:
    logger.warning(f"⚠️ Step 클래스 전역 설정 실패: {e}")

# =============================================================================
# 🔥 Export 목록
# =============================================================================

__all__ = [
    # 🎯 파이프라인 상수
    'STEP_MAPPING',
    'LOADING_PRIORITY',
    'SYSTEM_INFO',
    
    # 🔧 파이프라인 관리 함수들
    'get_pipeline_status',
    'get_step_class',
    'create_step_instance',
    'list_available_steps',
    'get_step_info',
    'initialize_pipeline_system',
    'cleanup_pipeline_system',
    
    # 🛠️ 유틸리티 함수들 (조건부)
    'get_step_model_interface',
    'get_step_memory_manager',
    'get_step_data_converter', 
    'preprocess_image_for_step',
    
    # 📊 상태 정보
    'UTILS_STATUS',
    'IS_CONDA',
    'IS_M3_MAX',
    'DEVICE'
]

# Step 클래스들도 동적으로 추가
for step_info in STEP_MAPPING.values():
    class_name = step_info['class']
    if class_name in globals():
        __all__.append(class_name)

# =============================================================================
# 🔥 초기화 완료 메시지
# =============================================================================

def _print_initialization_summary():
    """초기화 요약 출력"""
    status = get_pipeline_status()
    available_count = status['available_steps']
    total_count = status['total_steps']
    success_rate = status['success_rate']
    
    print(f"\n🍎 MyCloset AI 파이프라인 시스템 v7.0 초기화 완료!")
    print(f"📊 사용 가능한 Step: {available_count}/{total_count}개 ({success_rate:.1f}%)")
    print(f"🐍 conda 환경: {'✅' if IS_CONDA else '❌'}")
    print(f"🍎 M3 Max: {'✅' if IS_M3_MAX else '❌'}")
    print(f"🖥️ 디바이스: {DEVICE}")
    print(f"🛠️ 유틸리티: {sum(UTILS_STATUS.values())}/4개 사용 가능")
    
    if available_count > 0:
        print(f"✅ 로드된 Steps: {', '.join(status['loaded_steps'])}")
    
    if status['failed_steps']:
        print(f"⚠️ 실패한 Steps: {', '.join(status['failed_steps'])}")
        
    print("🚀 파이프라인 시스템 준비 완료!\n")

# 초기화 상태 출력 (한 번만)
if not hasattr(sys, '_mycloset_pipeline_initialized'):
    _print_initialization_summary()
    sys._mycloset_pipeline_initialized = True

logger.info("🍎 MyCloset AI 파이프라인 시스템 초기화 완료")