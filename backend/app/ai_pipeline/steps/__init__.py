
# ============================================================================
# 📁 backend/app/ai_pipeline/steps/__init__.py - Step 클래스 관리
# ============================================================================

"""
🎯 MyCloset AI Pipeline Steps 모듈 - conda 환경 우선 Step 관리
==========================================================

✅ conda 환경 우선 최적화
✅ 순환참조 완전 방지 (지연 로딩 패턴)
✅ 8단계 Step 클래스 안전한 로딩
✅ 동적 import 및 오류 처리
✅ Step 간 의존성 관리
✅ 성능 최적화된 로딩

역할: 8단계 AI 파이프라인 Step 클래스들의 로딩과 관리를 담당
"""

import sys
import importlib
import logging
import threading
from pathlib import Path
from typing import Dict, Type, Any, Optional, List

# 상위 패키지에서 시스템 정보 가져오기
try:
    from ... import SYSTEM_INFO, IS_CONDA, CONDA_ENV, _lazy_loader
except ImportError:
    SYSTEM_INFO = {'device': 'cpu', 'is_m3_max': False}
    IS_CONDA = 'CONDA_DEFAULT_ENV' in os.environ
    CONDA_ENV = os.environ.get('CONDA_DEFAULT_ENV', 'none')
    _lazy_loader = None

# 로거 설정
logger = logging.getLogger(__name__)

# =============================================================================
# 🔥 Step 모듈 정보 (8단계)
# =============================================================================

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

# Step 설명
STEP_DESCRIPTIONS = {
    'step_01': '인체 파싱 - 신체 부위 분할',
    'step_02': '포즈 추정 - 신체 포즈 감지',
    'step_03': '의류 분할 - 의류 영역 분할',
    'step_04': '기하학적 매칭 - 의류-신체 매칭',
    'step_05': '의류 변형 - 의류 워핑',
    'step_06': '가상 피팅 - 최종 합성',
    'step_07': '후처리 - 품질 향상',
    'step_08': '품질 평가 - 결과 분석'
}

# =============================================================================
# 🔥 지연 로딩 시스템 (순환참조 방지)
# =============================================================================

class StepLazyLoader:
    """Step 전용 지연 로더 - conda 환경 최적화"""
    
    def __init__(self):
        self._step_cache = {}
        self._loading = set()
        self._lock = threading.RLock()
        self._conda_optimized = IS_CONDA
    
    def safe_import_step(self, step_id: str) -> Optional[Type[Any]]:
        """안전한 Step 클래스 import (지연 로딩)"""
        with self._lock:
            # 캐시 확인
            if step_id in self._step_cache:
                return self._step_cache[step_id]
            
            # 순환 로딩 방지
            if step_id in self._loading:
                logger.warning(f"순환참조 감지: {step_id}")
                return None
            
            self._loading.add(step_id)
            
            try:
                module_name = STEP_MODULES.get(step_id)
                class_name = STEP_CLASSES.get(step_id)
                
                if not module_name or not class_name:
                    logger.error(f"❌ 알 수 없는 Step ID: {step_id}")
                    self._step_cache[step_id] = None
                    return None
                
                # importlib 사용 (conda 환경 안정성)
                full_module_name = f"app.ai_pipeline.steps.{module_name}"
                
                try:
                    module = importlib.import_module(full_module_name)
                    step_class = getattr(module, class_name, None)
                    
                    if step_class is None:
                        logger.error(f"❌ {class_name} 클래스를 {module_name}에서 찾을 수 없음")
                        self._step_cache[step_id] = None
                        return None
                    
                    # conda 환경에서 추가 검증
                    if self._conda_optimized and hasattr(step_class, '__init__'):
                        logger.debug(f"✅ conda 환경에서 {step_id} ({class_name}) 검증 완료")
                    
                    self._step_cache[step_id] = step_class
                    logger.info(f"✅ {step_id} ({class_name}) import 성공")
                    return step_class
                    
                except ImportError as e:
                    logger.warning(f"❌ {step_id} import 실패: {e}")
                    self._step_cache[step_id] = None
                    return None
                
            except Exception as e:
                logger.error(f"❌ {step_id} 예상치 못한 오류: {e}")
                self._step_cache[step_id] = None
                return None
            
            finally:
                self._loading.discard(step_id)
    
    def load_all_steps(self) -> Dict[str, Optional[Type[Any]]]:
        """모든 Step 클래스 로드"""
        loaded_steps = {}
        
        for step_id in STEP_MODULES.keys():
            step_class = self.safe_import_step(step_id)
            loaded_steps[step_id] = step_class
        
        available_count = sum(1 for step in loaded_steps.values() if step is not None)
        logger.info(f"✅ Step 로딩 완료: {available_count}/8개")
        
        return loaded_steps
    
    def get_step_info(self, step_id: str) -> Dict[str, Any]:
        """Step 정보 반환"""
        step_class = self.safe_import_step(step_id)
        
        return {
            'step_id': step_id,
            'module_name': STEP_MODULES.get(step_id),
            'class_name': STEP_CLASSES.get(step_id),
            'description': STEP_DESCRIPTIONS.get(step_id),
            'available': step_class is not None,
            'class_type': str(type(step_class)) if step_class else None,
            'conda_optimized': self._conda_optimized
        }
    
    def clear_cache(self):
        """캐시 초기화"""
        with self._lock:
            self._step_cache.clear()
            logger.info("🧹 Step 캐시 초기화 완료")

# 전역 Step 로더
_step_loader = StepLazyLoader()

# =============================================================================
# 🔥 Step 관리 함수들 (외부 인터페이스)
# =============================================================================

def safe_import_step(step_id: str) -> Optional[Type[Any]]:
    """안전한 Step 클래스 import"""
    return _step_loader.safe_import_step(step_id)

def load_all_steps() -> Dict[str, Optional[Type[Any]]]:
    """모든 Step 클래스 로드"""
    return _step_loader.load_all_steps()

def get_step_class(step_name: Union[str, int]) -> Optional[Type[Any]]:
    """Step 클래스 반환"""
    try:
        if isinstance(step_name, int):
            step_key = f"step_{step_name:02d}"
        else:
            step_key = step_name
        
        return safe_import_step(step_key)
    except Exception as e:
        logger.error(f"Step 클래스 조회 실패 {step_name}: {e}")
        return None

def create_step_instance(step_name: Union[str, int], **kwargs) -> Optional[Any]:
    """Step 인스턴스 생성 (conda 환경 최적화)"""
    try:
        step_class = get_step_class(step_name)
        if step_class is None:
            logger.error(f"Step 클래스를 찾을 수 없음: {step_name}")
            return None
        
        # conda 환경 기본 설정 추가
        default_config = {
            "device": SYSTEM_INFO.get('device', 'cpu'),
            "is_m3_max": SYSTEM_INFO.get('is_m3_max', False),
            "memory_gb": SYSTEM_INFO.get('memory_gb', 16.0),
            "conda_optimized": IS_CONDA,
            "conda_env": CONDA_ENV
        }
        default_config.update(kwargs)
        
        return step_class(**default_config)
        
    except Exception as e:
        logger.error(f"Step 인스턴스 생성 실패 {step_name}: {e}")
        return None

def list_available_steps() -> List[str]:
    """사용 가능한 Step 목록 반환"""
    loaded_steps = load_all_steps()
    return [step_id for step_id, step_class in loaded_steps.items() if step_class is not None]

def get_step_info(step_id: str) -> Dict[str, Any]:
    """Step 정보 반환"""
    return _step_loader.get_step_info(step_id)

def get_steps_status() -> Dict[str, Any]:
    """전체 Step 상태 반환"""
    loaded_steps = load_all_steps()
    available_steps = [k for k, v in loaded_steps.items() if v is not None]
    
    return {
        'total_steps': len(STEP_MODULES),
        'available_steps': len(available_steps),
        'loaded_steps': available_steps,
        'failed_steps': [k for k, v in loaded_steps.items() if v is None],
        'conda_optimized': IS_CONDA,
        'conda_env': CONDA_ENV,
        'device': SYSTEM_INFO.get('device', 'cpu'),
        'step_details': {step_id: get_step_info(step_id) for step_id in STEP_MODULES.keys()}
    }

# =============================================================================
# 🔥 자동 로딩 및 글로벌 변수 설정
# =============================================================================

# Step 클래스들 자동 로딩 시도
try:
    ALL_STEPS = load_all_steps()
    
    # 개별 클래스 글로벌 변수로 내보내기 (하위 호환성)
    for step_id, step_class in ALL_STEPS.items():
        if step_class:
            class_name = STEP_CLASSES[step_id]
            globals()[class_name] = step_class
    
    logger.info("🎉 모든 Step 클래스 글로벌 설정 완료")
    
except Exception as e:
    logger.error(f"❌ Step 클래스 자동 로딩 실패: {e}")
    ALL_STEPS = {}

# =============================================================================
# 🔥 Steps 모듈 Export
# =============================================================================

__all__ = [
    # 🔥 Step 정보
    'STEP_MODULES',
    'STEP_CLASSES', 
    'STEP_DESCRIPTIONS',
    
    # 🔗 Step 관리 함수들
    'safe_import_step',
    'load_all_steps',
    'get_step_class',
    'create_step_instance',
    'list_available_steps',
    'get_step_info',
    'get_steps_status',
    
    # 📊 로딩 결과
    'ALL_STEPS',
    
    # 🔧 개별 Step 클래스들 (동적 추가)
] + list(STEP_CLASSES.values())

# 초기화 정보 출력
available_count = len([k for k, v in ALL_STEPS.items() if v is not None])
logger.info("🎯 MyCloset AI Pipeline Steps 모듈 초기화 완료")
logger.info(f"📊 로딩된 Step: {available_count}/8개")
logger.info(f"🐍 conda 최적화: {IS_CONDA}")
logger.info(f"🔗 지연 로딩: 활성화")