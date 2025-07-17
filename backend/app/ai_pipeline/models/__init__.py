# backend/app/ai_pipeline/models/__init__.py
"""
AI Pipeline Models - Step 클래스 연동 및 모델 관리
✅ 별도 모델 클래스 대신 Steps 클래스 활용
✅ Step 클래스들이 AI 모델 처리까지 담당
🔥 통합 설계 - Step = AI Model + Processing Logic
"""

import logging
from typing import Dict, Any, Optional, Type, List
from pathlib import Path

logger = logging.getLogger(__name__)

# ==============================================
# 🎯 Step-Model 통합 관리
# ==============================================

class StepModelManager:
    """
    🤖 Step 클래스 = AI 모델 통합 관리자
    Step 클래스들이 AI 모델의 역할까지 겸하는 구조 관리
    """
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.StepModelManager")
        
        # Step-Model 매핑 정보
        self.step_model_mapping = {
            'step_01_human_parsing': {
                'model_type': 'graphonomy',
                'model_file': 'graphonomy.pth',
                'input_size': (512, 512),
                'num_classes': 20,
                'description': '인간 파싱 모델 (Graphonomy 기반)'
            },
            'step_02_pose_estimation': {
                'model_type': 'openpose',
                'model_file': 'openpose_body.pth', 
                'input_size': (368, 368),
                'num_keypoints': 18,
                'description': '포즈 추정 모델 (OpenPose 기반)'
            },
            'step_03_cloth_segmentation': {
                'model_type': 'u2net',
                'model_file': 'u2net.pth',
                'input_size': (320, 320),
                'num_classes': 2,
                'description': '의류 분할 모델 (U2-Net 기반)'
            },
            'step_04_geometric_matching': {
                'model_type': 'gmm',
                'model_file': 'gmm_final.pth',
                'input_size': (256, 192),
                'description': '기하학적 매칭 모델 (GMM)'
            },
            'step_05_cloth_warping': {
                'model_type': 'tom',
                'model_file': 'tom_final.pth',
                'input_size': (256, 192),
                'description': '의류 워핑 모델 (TOM)'
            },
            'step_06_virtual_fitting': {
                'model_type': 'hrviton',
                'model_file': 'hrviton_final.pth',
                'input_size': (512, 384),
                'description': '가상 피팅 모델 (HR-VITON)'
            },
            'step_07_post_processing': {
                'model_type': 'enhancer',
                'model_file': 'enhancer.pth',
                'input_size': (512, 512),
                'description': '후처리 향상 모델'
            },
            'step_08_quality_assessment': {
                'model_type': 'quality_scorer',
                'model_file': 'quality.pth',
                'input_size': (224, 224),
                'description': '품질 평가 모델'
            }
        }
        
        self.logger.info("🤖 StepModelManager 초기화 완료")
    
    def get_step_model_info(self, step_name: str) -> Optional[Dict[str, Any]]:
        """Step의 모델 정보 반환"""
        return self.step_model_mapping.get(step_name)
    
    def get_all_step_models(self) -> Dict[str, Dict[str, Any]]:
        """모든 Step-Model 정보 반환"""
        return self.step_model_mapping.copy()
    
    def validate_step_model(self, step_name: str, model_dir: Path) -> bool:
        """Step 모델 파일 존재 여부 확인"""
        model_info = self.get_step_model_info(step_name)
        if not model_info:
            return False
        
        model_file = model_info['model_file']
        model_path = model_dir / "checkpoints" / step_name / model_file
        
        exists = model_path.exists()
        if not exists:
            self.logger.warning(f"⚠️ {step_name} 모델 파일 없음: {model_path}")
        
        return exists

# ==============================================
# 🔗 Step 클래스 연동 함수들
# ==============================================

def get_step_as_model(step_name: str) -> Optional[Type]:
    """
    🎯 Step 클래스를 AI 모델로 사용
    Step 클래스 = AI 모델 + 처리 로직
    """
    try:
        # steps 모듈에서 동적 import
        from ..steps import get_step_class
        
        step_class = get_step_class(step_name)
        if step_class is None:
            logger.error(f"❌ {step_name} Step 클래스를 찾을 수 없습니다")
            return None
        
        logger.info(f"✅ {step_name} Step을 AI 모델로 반환")
        return step_class
        
    except Exception as e:
        logger.error(f"❌ {step_name} Step-Model 로드 실패: {e}")
        return None

def create_step_model_instance(step_name: str, **kwargs) -> Optional[Any]:
    """Step 클래스 인스턴스를 AI 모델로 생성"""
    try:
        from ..steps import create_step_instance
        
        instance = create_step_instance(step_name, **kwargs)
        if instance is None:
            logger.error(f"❌ {step_name} Step-Model 인스턴스 생성 실패")
            return None
        
        logger.info(f"✅ {step_name} Step-Model 인스턴스 생성 완료")
        return instance
        
    except Exception as e:
        logger.error(f"❌ {step_name} Step-Model 인스턴스 생성 오류: {e}")
        return None

def get_available_step_models() -> List[str]:
    """사용 가능한 Step-Model 목록 반환"""
    try:
        from ..steps import check_steps_health
        
        health_info = check_steps_health()
        available_steps = []
        
        for step_key, status in health_info['step_status'].items():
            if status['available']:
                available_steps.append(step_key)
        
        logger.info(f"📊 사용 가능한 Step-Models: {len(available_steps)}")
        return available_steps
        
    except Exception as e:
        logger.error(f"❌ Step-Model 목록 조회 실패: {e}")
        return []

def validate_all_step_models(model_dir: Optional[Path] = None) -> Dict[str, bool]:
    """모든 Step-Model 유효성 검사"""
    if model_dir is None:
        # 기본 모델 디렉토리
        model_dir = Path(__file__).parent / "ai_models"
    
    manager = StepModelManager()
    validation_results = {}
    
    for step_name in manager.step_model_mapping.keys():
        validation_results[step_name] = manager.validate_step_model(step_name, model_dir)
    
    valid_count = sum(validation_results.values())
    total_count = len(validation_results)
    
    logger.info(f"🔍 Step-Model 검증 완료: {valid_count}/{total_count}")
    return validation_results

# ==============================================
# 🎯 하위 호환성 지원
# ==============================================

# 기존 모델 클래스 이름들을 Step 클래스로 매핑 (하위 호환성)
class ModelClassAdapter:
    """기존 모델 클래스명을 Step 클래스로 연결하는 어댑터"""
    
    @staticmethod
    def GraphonomyModel(**kwargs):
        """Graphonomy 모델 → HumanParsingStep"""
        return create_step_model_instance('step_01_human_parsing', **kwargs)
    
    @staticmethod
    def OpenPoseModel(**kwargs):
        """OpenPose 모델 → PoseEstimationStep"""
        return create_step_model_instance('step_02_pose_estimation', **kwargs)
    
    @staticmethod
    def U2NetModel(**kwargs):
        """U2-Net 모델 → ClothSegmentationStep"""
        return create_step_model_instance('step_03_cloth_segmentation', **kwargs)
    
    @staticmethod
    def GeometricMatchingModel(**kwargs):
        """GMM 모델 → GeometricMatchingStep"""
        return create_step_model_instance('step_04_geometric_matching', **kwargs)
    
    @staticmethod
    def HRVITONModel(**kwargs):
        """HR-VITON 모델 → VirtualFittingStep"""
        return create_step_model_instance('step_06_virtual_fitting', **kwargs)

# 전역 어댑터 인스턴스
_adapter = ModelClassAdapter()

# 하위 호환성을 위한 모델 클래스들 (실제로는 Step 클래스들)
GraphonomyModel = _adapter.GraphonomyModel
OpenPoseModel = _adapter.OpenPoseModel
U2NetModel = _adapter.U2NetModel
GeometricMatchingModel = _adapter.GeometricMatchingModel
HRVITONModel = _adapter.HRVITONModel

# ==============================================
# 🔧 전역 관리자 인스턴스
# ==============================================

# 전역 Step-Model 관리자
_step_model_manager = StepModelManager()

def get_step_model_manager() -> StepModelManager:
    """전역 Step-Model 관리자 반환"""
    return _step_model_manager

# ==============================================
# 🎯 모듈 exports
# ==============================================

__all__ = [
    # Step-Model 통합 관리
    'StepModelManager',
    'get_step_model_manager',
    
    # Step을 모델로 사용하는 함수들
    'get_step_as_model',
    'create_step_model_instance', 
    'get_available_step_models',
    'validate_all_step_models',
    
    # 하위 호환성 모델 클래스들 (실제로는 Step 클래스들)
    'GraphonomyModel',
    'OpenPoseModel', 
    'U2NetModel',
    'GeometricMatchingModel',
    'HRVITONModel',
    
    # 어댑터
    'ModelClassAdapter'
]

# ==============================================
# 🎉 모듈 로드 완료
# ==============================================

logger.info("🎉 AI Pipeline Models 모듈 로드 완료!")
logger.info("✅ Step 클래스 = AI 모델 통합 설계")
logger.info("✅ 별도 모델 클래스 불필요") 
logger.info("✅ Step 클래스가 모델 처리까지 담당")
logger.info("🔥 통합 아키텍처 완성!")