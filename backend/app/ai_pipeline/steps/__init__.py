# backend/app/ai_pipeline/steps/__init__.py
"""
AI Pipeline Steps - 순환 참조 방지 및 안전한 import 시스템
✅ 기존 step 클래스들 활용
✅ 안전한 지연 로딩
✅ 모델 클래스 역할도 겸함
🔥 별도 모델 클래스 불필요 - step 클래스가 AI 모델 처리까지 담당
"""

import logging
import threading
import weakref
from typing import Dict, Any, Optional, Type, List, Callable
import importlib
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# ==============================================
# 🔧 Step 클래스 레지스트리 시스템
# ==============================================

class StepRegistry:
    """
    🎯 Step 클래스 중앙 관리 시스템
    - 순환 참조 방지
    - 지연 로딩
    - 에러 안전성
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self.logger = logging.getLogger(f"{__name__}.StepRegistry")
        
        # Step 정의 (파일명과 클래스명 매핑)
        self.step_definitions = {
            'step_01_human_parsing': {
                'module': 'step_01_human_parsing',
                'class_name': 'HumanParsingStep',
                'ai_model_type': 'human_parsing',
                'description': '인간 파싱 및 신체 영역 분할'
            },
            'step_02_pose_estimation': {
                'module': 'step_02_pose_estimation', 
                'class_name': 'PoseEstimationStep',
                'ai_model_type': 'pose_estimation',
                'description': '포즈 추정 및 키포인트 검출'
            },
            'step_03_cloth_segmentation': {
                'module': 'step_03_cloth_segmentation',
                'class_name': 'ClothSegmentationStep', 
                'ai_model_type': 'cloth_segmentation',
                'description': '의류 분할 및 마스킹'
            },
            'step_04_geometric_matching': {
                'module': 'step_04_geometric_matching',
                'class_name': 'GeometricMatchingStep',
                'ai_model_type': 'geometric_matching', 
                'description': '기하학적 매칭 및 변형'
            },
            'step_05_cloth_warping': {
                'module': 'step_05_cloth_warping',
                'class_name': 'ClothWarpingStep',
                'ai_model_type': 'cloth_warping',
                'description': '의류 워핑 및 변형'
            },
            'step_06_virtual_fitting': {
                'module': 'step_06_virtual_fitting',
                'class_name': 'VirtualFittingStep', 
                'ai_model_type': 'virtual_fitting',
                'description': '가상 피팅 및 합성'
            },
            'step_07_post_processing': {
                'module': 'step_07_post_processing',
                'class_name': 'PostProcessingStep',
                'ai_model_type': 'post_processing',
                'description': '후처리 및 이미지 향상'
            },
            'step_08_quality_assessment': {
                'module': 'step_08_quality_assessment',
                'class_name': 'QualityAssessmentStep',
                'ai_model_type': 'quality_assessment', 
                'description': '품질 평가 및 검증'
            }
        }
        
        # 로드된 클래스 캐시 (약한 참조 사용)
        self.loaded_classes: Dict[str, weakref.ref] = {}
        self.import_errors: Dict[str, str] = {}
        self.import_attempts: Dict[str, int] = {}
        
        self._initialized = True
        self.logger.info("🎯 StepRegistry 초기화 완료")
    
    def get_step_class(self, step_key: str) -> Optional[Type]:
        """
        🔍 Step 클래스 안전하게 가져오기
        
        Args:
            step_key: step 식별자 (예: 'step_01_human_parsing')
        
        Returns:
            Step 클래스 또는 None
        """
        if step_key not in self.step_definitions:
            self.logger.error(f"❌ 알 수 없는 step: {step_key}")
            return None
        
        # 캐시에서 확인 (약한 참조)
        if step_key in self.loaded_classes:
            cached_ref = self.loaded_classes[step_key]
            cached_class = cached_ref()
            if cached_class is not None:
                return cached_class
            else:
                # 약한 참조가 제거됨, 캐시에서 삭제
                del self.loaded_classes[step_key]
        
        # 동적 import 시도
        return self._import_step_class(step_key)
    
    def _import_step_class(self, step_key: str) -> Optional[Type]:
        """동적으로 step 클래스 import"""
        step_def = self.step_definitions[step_key]
        module_name = step_def['module']
        class_name = step_def['class_name']
        
        try:
            # import 시도 횟수 체크
            attempts = self.import_attempts.get(step_key, 0)
            if attempts >= 3:
                self.logger.warning(f"⚠️ {step_key} import 시도 한계 초과 (3회)")
                return None
            
            self.import_attempts[step_key] = attempts + 1
            
            # 모듈 import
            full_module_name = f".{module_name}"
            module = importlib.import_module(full_module_name, package=__name__)
            
            # 클래스 가져오기
            if not hasattr(module, class_name):
                raise AttributeError(f"모듈 {module_name}에 {class_name} 클래스가 없습니다")
            
            step_class = getattr(module, class_name)
            
            # 약한 참조로 캐시
            self.loaded_classes[step_key] = weakref.ref(step_class)
            
            # 에러 기록 초기화
            if step_key in self.import_errors:
                del self.import_errors[step_key]
            
            self.logger.info(f"✅ {step_key} ({class_name}) import 성공")
            return step_class
            
        except Exception as e:
            error_msg = f"❌ {step_key} import 실패: {e}"
            self.logger.error(error_msg)
            self.import_errors[step_key] = str(e)
            return None
    
    def get_all_available_steps(self) -> Dict[str, Type]:
        """사용 가능한 모든 step 클래스들 반환"""
        available_steps = {}
        
        for step_key in self.step_definitions.keys():
            step_class = self.get_step_class(step_key)
            if step_class is not None:
                available_steps[step_key] = step_class
        
        self.logger.info(f"📊 사용 가능한 Steps: {len(available_steps)}/{len(self.step_definitions)}")
        return available_steps
    
    def check_step_health(self) -> Dict[str, Any]:
        """Step들의 상태 체크"""
        health_info = {
            'total_steps': len(self.step_definitions),
            'loaded_steps': 0,
            'failed_steps': len(self.import_errors),
            'step_status': {},
            'import_errors': self.import_errors.copy(),
            'import_attempts': self.import_attempts.copy()
        }
        
        for step_key, step_def in self.step_definitions.items():
            step_class = self.get_step_class(step_key)
            status = {
                'available': step_class is not None,
                'class_name': step_def['class_name'],
                'ai_model_type': step_def['ai_model_type'],
                'description': step_def['description']
            }
            
            if step_class is not None:
                health_info['loaded_steps'] += 1
                status['class_object'] = step_class.__name__
            
            health_info['step_status'][step_key] = status
        
        return health_info
    
    def reload_step(self, step_key: str) -> bool:
        """특정 step 강제 리로드"""
        if step_key not in self.step_definitions:
            return False
        
        try:
            # 캐시 클리어
            if step_key in self.loaded_classes:
                del self.loaded_classes[step_key]
            if step_key in self.import_errors:
                del self.import_errors[step_key]
            
            self.import_attempts[step_key] = 0
            
            # 모듈 리로드
            step_def = self.step_definitions[step_key]
            module_name = f".{step_def['module']}"
            
            if module_name in sys.modules:
                importlib.reload(sys.modules[module_name])
            
            # 다시 import
            step_class = self._import_step_class(step_key)
            return step_class is not None
            
        except Exception as e:
            self.logger.error(f"❌ {step_key} 리로드 실패: {e}")
            return False

# ==============================================
# 🌟 전역 레지스트리 인스턴스
# ==============================================

# 전역 레지스트리 (싱글톤)
_step_registry = StepRegistry()

# ==============================================
# 🔗 Public API 함수들 
# ==============================================

def get_step_class(step_name: str) -> Optional[Type]:
    """
    🎯 Step 클래스 가져오기 (Main API)
    
    Args:
        step_name: step 이름 (예: 'human_parsing', 'step_01_human_parsing')
    
    Returns:
        Step 클래스 또는 None
    """
    # step_name 정규화
    if not step_name.startswith('step_'):
        # 'human_parsing' -> 'step_01_human_parsing' 형태로 변환
        step_mapping = {
            'human_parsing': 'step_01_human_parsing',
            'pose_estimation': 'step_02_pose_estimation', 
            'cloth_segmentation': 'step_03_cloth_segmentation',
            'geometric_matching': 'step_04_geometric_matching',
            'cloth_warping': 'step_05_cloth_warping',
            'virtual_fitting': 'step_06_virtual_fitting',
            'post_processing': 'step_07_post_processing',
            'quality_assessment': 'step_08_quality_assessment'
        }
        step_name = step_mapping.get(step_name, step_name)
    
    return _step_registry.get_step_class(step_name)

def get_all_step_classes() -> Dict[str, Type]:
    """모든 사용 가능한 step 클래스들 반환"""
    return _step_registry.get_all_available_steps()

def check_steps_health() -> Dict[str, Any]:
    """Steps 상태 체크"""
    return _step_registry.check_step_health()

def reload_step(step_name: str) -> bool:
    """Step 리로드"""
    return _step_registry.reload_step(step_name)

def get_step_info(step_name: str) -> Optional[Dict[str, Any]]:
    """Step 정보 반환"""
    step_key = step_name if step_name.startswith('step_') else f"step_0{list(_step_registry.step_definitions.keys()).index(step_name)+1}_{step_name}"
    
    if step_key in _step_registry.step_definitions:
        step_def = _step_registry.step_definitions[step_key].copy()
        step_class = get_step_class(step_key)
        step_def['available'] = step_class is not None
        step_def['step_key'] = step_key
        return step_def
    
    return None

# ==============================================
# 🔄 Step 클래스 팩토리 함수들
# ==============================================

def create_step_instance(step_name: str, **kwargs) -> Optional[Any]:
    """
    🏭 Step 인스턴스 생성 팩토리
    
    Args:
        step_name: step 이름
        **kwargs: 생성자 파라미터들
    
    Returns:
        Step 인스턴스 또는 None
    """
    step_class = get_step_class(step_name)
    if step_class is None:
        logger.error(f"❌ {step_name} 클래스를 찾을 수 없습니다")
        return None
    
    try:
        # Step 인스턴스 생성
        instance = step_class(**kwargs)
        logger.info(f"✅ {step_name} 인스턴스 생성 완료")
        return instance
        
    except Exception as e:
        logger.error(f"❌ {step_name} 인스턴스 생성 실패: {e}")
        return None

def create_all_step_instances(**common_kwargs) -> Dict[str, Any]:
    """모든 Step 인스턴스들 생성"""
    instances = {}
    
    for step_key in _step_registry.step_definitions.keys():
        instance = create_step_instance(step_key, **common_kwargs)
        if instance is not None:
            instances[step_key] = instance
    
    logger.info(f"🏭 Step 인스턴스 생성 완료: {len(instances)}/{len(_step_registry.step_definitions)}")
    return instances

# ==============================================
# 🎯 기존 호환성을 위한 직접 import (안전)
# ==============================================

# 기존 코드 호환성을 위해 클래스들을 직접 노출 (지연 로딩)
def __getattr__(name: str):
    """동적 attribute 접근 (Python 3.7+)"""
    class_mapping = {
        'HumanParsingStep': 'step_01_human_parsing',
        'PoseEstimationStep': 'step_02_pose_estimation',
        'ClothSegmentationStep': 'step_03_cloth_segmentation', 
        'GeometricMatchingStep': 'step_04_geometric_matching',
        'ClothWarpingStep': 'step_05_cloth_warping',
        'VirtualFittingStep': 'step_06_virtual_fitting',
        'PostProcessingStep': 'step_07_post_processing',
        'QualityAssessmentStep': 'step_08_quality_assessment'
    }
    
    if name in class_mapping:
        step_key = class_mapping[name]
        step_class = get_step_class(step_key)
        if step_class is not None:
            return step_class
        else:
            raise ImportError(f"❌ {name} 클래스를 로드할 수 없습니다")
    
    raise AttributeError(f"❌ '{name}' 속성을 찾을 수 없습니다")

# ==============================================
# 🔧 모듈 메타데이터
# ==============================================

__all__ = [
    # Step 클래스들 (동적 로딩)
    'HumanParsingStep',
    'PoseEstimationStep', 
    'ClothSegmentationStep',
    'GeometricMatchingStep',
    'ClothWarpingStep',
    'VirtualFittingStep',
    'PostProcessingStep',
    'QualityAssessmentStep',
    
    # API 함수들
    'get_step_class',
    'get_all_step_classes',
    'check_steps_health',
    'reload_step',
    'get_step_info',
    'create_step_instance',
    'create_all_step_instances',
    
    # 레지스트리
    'StepRegistry'
]

# ==============================================
# 🎉 모듈 로드 완료 메시지
# ==============================================

logger.info("🎉 AI Pipeline Steps 모듈 로드 완료!")
logger.info("✅ 순환 참조 방지 시스템 적용")
logger.info("✅ 안전한 지연 로딩 구현")  
logger.info("✅ Step 클래스들이 AI 모델 역할까지 겸함")
logger.info("🔥 별도 모델 클래스 불필요 - 통합 설계 완성!")