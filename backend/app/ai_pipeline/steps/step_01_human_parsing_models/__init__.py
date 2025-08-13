"""
🔥 Human Parsing 모듈 패키지 - 기존 완전한 BaseStepMixin 활용
========================================================

기존 완전한 BaseStepMixin v20.0 (5120줄)을 활용한 Human Parsing 관련 모듈들을 포함합니다.

구조:
- config.py: 설정 클래스들
- models/: AI 모델들
- ensemble/: 앙상블 시스템
- postprocessing/: 후처리 시스템
- utils/: 유틸리티 함수들

Author: MyCloset AI Team
Date: 2025-08-07
Version: 2.0 (BaseStepMixin 활용)
"""

# 패키지 버전
__version__ = "2.0.0"

# 메인 BaseStepMixin import
try:
    from app.ai_pipeline.steps.base.base_step_mixin import BaseStepMixin
    BASE_STEP_MIXIN_AVAILABLE = True
except ImportError:
    try:
        from ..base.base_step_mixin import BaseStepMixin
        BASE_STEP_MIXIN_AVAILABLE = True
    except ImportError:
        BASE_STEP_MIXIN_AVAILABLE = False
        raise ImportError("BaseStepMixin을 import할 수 없습니다. 메인 BaseStepMixin을 사용하세요.")

# 주요 모듈들
__all__ = [
    "BaseStepMixin",
    "HumanParsingStep",
    "config",
    "models", 
    "ensemble",
    "postprocessing",
    "utils"
]

# HumanParsingStep 클래스 import
try:
    from .step_01_human_parsing import HumanParsingStep
except ImportError:
    # 직접 파일에서 import 시도
    try:
        from app.ai_pipeline.steps.step_01_human_parsing import HumanParsingStep
    except ImportError:
        # 폴백: 빈 클래스
        class HumanParsingStep:
            def __init__(self, **kwargs):
                pass
