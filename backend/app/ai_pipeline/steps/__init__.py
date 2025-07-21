# backend/app/ai_pipeline/steps/__init__.py
"""
MyCloset AI - Step 클래스들 통합 import 모듈
'package' 키워드 오류 해결 버전
"""

import sys
import importlib
import logging
from pathlib import Path
from typing import Dict, Type, Any

logger = logging.getLogger(__name__)

# 기존 문제가 있던 __import__ 방식 대신 importlib 사용
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

def safe_import_step(step_id: str) -> Type[Any]:
    """안전한 Step 클래스 import"""
    try:
        module_name = STEP_MODULES.get(step_id)
        class_name = STEP_CLASSES.get(step_id)
        
        if not module_name or not class_name:
            logger.error(f"❌ 알 수 없는 Step ID: {step_id}")
            return None
            
        # importlib.import_module 사용 (package 키워드 오류 해결)
        full_module_name = f"app.ai_pipeline.steps.{module_name}"
        module = importlib.import_module(full_module_name)
        
        step_class = getattr(module, class_name, None)
        if step_class is None:
            logger.error(f"❌ {class_name} 클래스를 {module_name}에서 찾을 수 없음")
            return None
            
        logger.info(f"✅ {step_id} ({class_name}) import 성공")
        return step_class
        
    except ImportError as e:
        logger.error(f"❌ {step_id} import 실패: {e}")
        return None
    except Exception as e:
        logger.error(f"❌ {step_id} 예상치 못한 오류: {e}")
        return None

def load_all_steps() -> Dict[str, Type[Any]]:
    """모든 Step 클래스 로드"""
    loaded_steps = {}
    
    for step_id in STEP_MODULES.keys():
        step_class = safe_import_step(step_id)
        if step_class:
            loaded_steps[step_id] = step_class
            
    logger.info(f"✅ Step 로딩 완료: {len(loaded_steps)}/8개")
    return loaded_steps

# 자동 로딩 및 내보내기
try:
    ALL_STEPS = load_all_steps()
    
    # 개별 클래스 내보내기 (하위 호환성)
    for step_id, step_class in ALL_STEPS.items():
        globals()[STEP_CLASSES[step_id]] = step_class
        
    logger.info("🎉 모든 Step 클래스 import 완료")
    
except Exception as e:
    logger.error(f"❌ Step 클래스 로딩 중 오류: {e}")
    ALL_STEPS = {}

__all__ = list(STEP_CLASSES.values()) + ['ALL_STEPS', 'safe_import_step', 'load_all_steps']