# backend/app/ai_pipeline/__init__.py
"""
🍎 MyCloset AI 파이프라인 시스템 v7.0 - 단순화된 초기화
================================================================

✅ 단순하고 안정적인 초기화
✅ 순환참조 완전 방지
✅ conda 환경 우선 최적화
✅ M3 Max 128GB 메모리 활용
✅ 8단계 AI 파이프라인 지원
✅ 지연 로딩으로 성능 최적화
✅ 실패 허용적 설계 (Fault Tolerant)

8단계 AI 파이프라인:
Step 1: HumanParsingStep (SCHP/Graphonomy)
Step 2: PoseEstimationStep (OpenPose/YOLO)
Step 3: ClothSegmentationStep (U2Net/SAM)
Step 4: GeometricMatchingStep (TPS/GMM)
Step 5: ClothWarpingStep (Advanced Warping)
Step 6: VirtualFittingStep (OOTDiffusion/IDM-VTON)
Step 7: PostProcessingStep (Enhancement/SR)
Step 8: QualityAssessmentStep (CLIP/Quality)

작성자: MyCloset AI Team
날짜: 2025-07-23
버전: v7.0.0 (Simplified Pipeline Initialization)
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
# 🔥 파이프라인 상수 정의
# =============================================================================

# 8단계 파이프라인 정의
PIPELINE_STEPS = {
    'step_01': {
        'name': 'HumanParsingStep',
        'description': '인체 파싱 - Human Body Parsing',
        'models': ['SCHP', 'Graphonomy'],
        'priority': 2
    },
    'step_02': {
        'name': 'PoseEstimationStep', 
        'description': '포즈 추정 - Pose Estimation',
        'models': ['OpenPose', 'YOLO-Pose'],
        'priority': 4
    },
    'step_03': {
        'name': 'ClothSegmentationStep',
        'description': '의류 분할 - Cloth Segmentation', 
        'models': ['U2Net', 'SAM'],
        'priority': 3
    },
    'step_04': {
        'name': 'GeometricMatchingStep',
        'description': '기하학적 매칭 - Geometric Matching',
        'models': ['TPS', 'GMM'],
        'priority': 7
    },
    'step_05': {
        'name': 'ClothWarpingStep',
        'description': '의류 변형 - Cloth Warping',
        'models': ['Advanced Warping'],
        'priority': 8
    },
    'step_06': {
        'name': 'VirtualFittingStep',
        'description': '가상 피팅 - Virtual Fitting',
        'models': ['OOTDiffusion', 'IDM-VTON'],
        'priority': 1  # 가장 중요
    },
    'step_07': {
        'name': 'PostProcessingStep',
        'description': '후처리 - Post Processing',
        'models': ['RealESRGAN', 'Enhancement'],
        'priority': 5
    },
    'step_08': {
        'name': 'QualityAssessmentStep',
        'description': '품질 평가 - Quality Assessment',
        'models': ['CLIP', 'Quality Metrics'],
        'priority': 6
    }
}

# conda 환경에서 로딩 우선순위
LOADING_PRIORITY = sorted(PIPELINE_STEPS.keys(), 
                         key=lambda x: PIPELINE_STEPS[x]['priority'])

# =============================================================================
# 🔥 지연 로딩 매니저 (단순화)
# =============================================================================

class SimplePipelineLoader:
    """단순화된 파이프라인 로더"""
    
    def __init__(self):
        self._loaded_modules = {}
        self._loaded_classes = {}
        self._failed_loads = set()
        self.logger = logging.getLogger(f"{__name__}.SimplePipelineLoader")
        
    def safe_import_step(self, step_id: str) -> Optional[Type]:
        """안전한 Step 클래스 import"""
        if step_id in self._loaded_classes:
            return self._loaded_classes[step_id]
            
        if step_id in self._failed_loads:
            return None
            
        try:
            step_info = PIPELINE_STEPS.get(step_id)
            if not step_info:
                self.logger.warning(f"⚠️ 알 수 없는 Step ID: {step_id}")
                return None
                
            # 모듈 이름 생성
            module_name = f"app.ai_pipeline.steps.{step_id}_{step_info['description'].split(' - ')[1].lower().replace(' ', '_')}"
            class_name = step_info['name']
            
            # 동적 import 시도
            import importlib
            try:
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
        load_order = LOADING_PRIORITY if IS_CONDA else PIPELINE_STEPS.keys()
        
        for step_id in load_order:
            step_class = self.safe_import_step(step_id)
            loaded_steps[step_id] = step_class
            
        available_count = sum(1 for step in loaded_steps.values() if step is not None)
        total_count = len(PIPELINE_STEPS)
        
        self.logger.info(f"📊 Step 로딩 완료: {available_count}/{total_count}개")
        if IS_CONDA:
            self.logger.info("🐍 conda 환경: 우선순위 기반 로딩 적용")
            
        return loaded_steps
        
    def get_step_info(self, step_id: str) -> Dict[str, Any]:
        """Step 정보 반환"""
        step_config = PIPELINE_STEPS.get(step_id, {})
        step_class = self._loaded_classes.get(step_id)
        
        return {
            'step_id': step_id,
            'name': step_config.get('name', 'Unknown'),
            'description': step_config.get('description', ''),
            'models': step_config.get('models', []),
            'priority': step_config.get('priority', 10),
            'available': step_class is not None,
            'loaded': step_class is not None,
            'failed': step_id in self._failed_loads
        }

# 전역 로더 인스턴스
_pipeline_loader = SimplePipelineLoader()

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
        'total_steps': len(PIPELINE_STEPS),
        'available_steps': len(available_steps),
        'loaded_steps': available_steps,
        'failed_steps': [k for k, v in loaded_steps.items() if v is None],
        'success_rate': (len(available_steps) / len(PIPELINE_STEPS)) * 100,
        'utils_status': UTILS_STATUS,
        'loading_priority': LOADING_PRIORITY if IS_CONDA else None
    }

def get_step_class(step_name: str) -> Optional[Type]:
    """Step 클래스 반환"""
    if step_name.startswith('step_'):
        return _pipeline_loader.safe_import_step(step_name)
    else:
        # 클래스명으로 검색
        for step_id, step_info in PIPELINE_STEPS.items():
            if step_info['name'] == step_name:
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
    return _pipeline_loader.get_step_info(step_id)

async def initialize_pipeline_system() -> bool:
    """파이프라인 시스템 초기화"""
    try:
        logger.info("🚀 파이프라인 시스템 초기화 시작")
        
        # Step 클래스들 로드
        loaded_steps = _pipeline_loader.load_all_available_steps()
        available_count = sum(1 for step in loaded_steps.values() if step is not None)
        
        logger.info(f"✅ 파이프라인 시스템 초기화 완료: {available_count}/{len(PIPELINE_STEPS)}개 Step")
        return available_count > 0
        
    except Exception as e:
        logger.error(f"❌ 파이프라인 시스템 초기화 실패: {e}")
        return False

async def cleanup_pipeline_system() -> None:
    """파이프라인 시스템 정리"""
    try:
        logger.info("🧹 파이프라인 시스템 정리 시작")
        
        # 캐시 정리
        _pipeline_loader._loaded_modules.clear()
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
            step_info = PIPELINE_STEPS[step_id]
            class_name = step_info['name']
            globals()[class_name] = step_class
            
    logger.info("✅ Step 클래스들 전역 설정 완료")
    
except Exception as e:
    logger.warning(f"⚠️ Step 클래스 전역 설정 실패: {e}")

# =============================================================================
# 🔥 Export 목록
# =============================================================================

__all__ = [
    # 🎯 파이프라인 상수
    'PIPELINE_STEPS',
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
for step_info in PIPELINE_STEPS.values():
    class_name = step_info['name']
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