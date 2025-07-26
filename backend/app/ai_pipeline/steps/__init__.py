#!/usr/bin/env python3
"""
🔥 MyCloset AI Pipeline Steps - 단순한 __init__.py (Step 01 문제 해결)
================================================================

✅ 복잡한 로더 시스템 완전 제거
✅ 직접 import로 단순화
✅ 개별 테스트 성공한 것 그대로 활용
✅ 725줄 → 50줄로 단순화

문제 해결:
- 개별 import는 성공: ✅ BaseStepMixin, ✅ HumanParsingStep  
- 복잡한 로더가 문제: ❌ Step01SpecialLoader, SimpleStepLoader
- 해결책: 직접 import 사용

Author: MyCloset AI Team  
Date: 2025-07-25
Version: Simple v1.0 (Problem Solved)
"""

import logging
from typing import Dict, Any, Optional, Type

logger = logging.getLogger(__name__)

# =============================================================================
# 🔥 직접 import (복잡한 로더 없이)
# =============================================================================

# 단계별 직접 import
try:
    from .step_01_human_parsing import HumanParsingStep
    STEP_01_AVAILABLE = True
    logger.info("✅ Step 01 (HumanParsingStep) import 성공")
except ImportError as e:
    STEP_01_AVAILABLE = False
    HumanParsingStep = None
    logger.error(f"❌ Step 01 import 실패: {e}")

try:
    from .step_02_pose_estimation import PoseEstimationStep
    STEP_02_AVAILABLE = True
    logger.info("✅ Step 02 (PoseEstimationStep) import 성공")
except ImportError:
    STEP_02_AVAILABLE = False
    PoseEstimationStep = None
    logger.debug("📋 Step 02 import 실패 (정상)")

try:
    from .step_03_cloth_segmentation import ClothSegmentationStep
    STEP_03_AVAILABLE = True
    logger.info("✅ Step 03 (ClothSegmentationStep) import 성공")
except ImportError:
    STEP_03_AVAILABLE = False
    ClothSegmentationStep = None
    logger.debug("📋 Step 03 import 실패 (정상)")

try:
    from .step_04_geometric_matching import GeometricMatchingStep
    STEP_04_AVAILABLE = True
    logger.info("✅ Step 04 (GeometricMatchingStep) import 성공")
except ImportError:
    STEP_04_AVAILABLE = False
    GeometricMatchingStep = None
    logger.debug("📋 Step 04 import 실패 (정상)")

try:
    from .step_05_cloth_warping import ClothWarpingStep
    STEP_05_AVAILABLE = True
    logger.info("✅ Step 05 (ClothWarpingStep) import 성공")
except ImportError:
    STEP_05_AVAILABLE = False
    ClothWarpingStep = None
    logger.debug("📋 Step 05 import 실패 (정상)")

try:
    from .step_06_virtual_fitting import VirtualFittingStep
    STEP_06_AVAILABLE = True
    logger.info("✅ Step 06 (VirtualFittingStep) import 성공")
except ImportError:
    STEP_06_AVAILABLE = False
    VirtualFittingStep = None
    logger.debug("📋 Step 06 import 실패 (정상)")

try:
    from .step_07_post_processing import PostProcessingStep
    STEP_07_AVAILABLE = True
    logger.info("✅ Step 07 (PostProcessingStep) import 성공")
except ImportError:
    STEP_07_AVAILABLE = False
    PostProcessingStep = None
    logger.debug("📋 Step 07 import 실패 (정상)")

try:
    from .step_08_quality_assessment import QualityAssessmentStep
    STEP_08_AVAILABLE = True
    logger.info("✅ Step 08 (QualityAssessmentStep) import 성공")
except ImportError:
    STEP_08_AVAILABLE = False
    QualityAssessmentStep = None
    logger.debug("📋 Step 08 import 실패 (정상)")

# =============================================================================
# 🔥 Step 매핑 (단순화)
# =============================================================================

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

# 사용 가능한 Step만 필터링
AVAILABLE_STEPS = {
    step_id: step_class 
    for step_id, step_class in STEP_MAPPING.items() 
    if step_class is not None
}

# =============================================================================
# 🔥 단순한 인터페이스 함수들
# =============================================================================

def get_step_class(step_id: str) -> Optional[Type]:
    """Step 클래스 반환 (단순화)"""
    return AVAILABLE_STEPS.get(step_id)

def get_available_steps() -> Dict[str, Type]:
    """사용 가능한 Step들 반환"""
    return AVAILABLE_STEPS.copy()

def create_step_instance(step_id: str, **kwargs):
    """Step 인스턴스 생성 (단순화)"""
    step_class = get_step_class(step_id)
    if step_class:
        return step_class(**kwargs)
    return None

def get_step_info() -> Dict[str, Any]:
    """Step 정보 반환 (단순화)"""
    available_list = list(AVAILABLE_STEPS.keys())
    failed_list = [
        step_id for step_id, step_class in STEP_MAPPING.items() 
        if step_class is None
    ]
    
    return {
        'total_steps': len(STEP_MAPPING),
        'available_steps': len(available_list),
        'available_step_list': available_list,
        'failed_step_list': failed_list,
        'success_rate': (len(available_list) / len(STEP_MAPPING)) * 100
    }

# =============================================================================
# 🔥 Export (기존 API 호환)
# =============================================================================

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
    
    # 인터페이스 함수들
    'get_step_class',
    'get_available_steps',
    'create_step_instance',
    'get_step_info',
    
    # 매핑
    'STEP_MAPPING',
    'AVAILABLE_STEPS'
]

# =============================================================================
# 🔥 초기화 로그 (단순화)
# =============================================================================

step_info = get_step_info()

logger.info("=" * 60)
logger.info("🔥 Step 로딩 완료 (단순화된 방식)")
logger.info("=" * 60)
logger.info(f"📊 사용 가능한 Step: {step_info['available_steps']}/{step_info['total_steps']}개 ({step_info['success_rate']:.1f}%)")

if step_info['available_step_list']:
    logger.info(f"✅ 로드된 Steps: {', '.join(step_info['available_step_list'])}")

if step_info['failed_step_list']:
    logger.info(f"⚠️ 실패한 Steps: {', '.join(step_info['failed_step_list'])}")

# Step 01 특별 체크
if 'step_01' in step_info['available_step_list']:
    logger.info("🎉 Step 01 (HumanParsingStep) 로딩 성공!")
else:
    logger.error("❌ Step 01 (HumanParsingStep) 로딩 실패!")

logger.info("🚀 단순화된 Step 시스템 준비 완료!")
logger.info("=" * 60)