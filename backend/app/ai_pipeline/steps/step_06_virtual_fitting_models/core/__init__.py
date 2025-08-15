#!/usr/bin/env python3
"""
🔥 MyCloset AI - Virtual Fitting Core
====================================

🎯 가상 피팅 핵심 기능
✅ 앙상블 코어
✅ 추론 코어
✅ 피팅 코어
✅ 스텝 코어
"""

import logging

# 로거 설정
logger = logging.getLogger(__name__)

try:
    from .ensemble_core import VirtualFittingEnsembleCore
    from .inference_core import VirtualFittingInferenceCore
    from .fitting_core import VirtualFittingCore
    from .step_core import VirtualFittingStepCore
    
    __all__ = [
        "VirtualFittingEnsembleCore",
        "VirtualFittingInferenceCore",
        "VirtualFittingCore",
        "VirtualFittingStepCore"
    ]
    
except ImportError as e:
    logger.error(f"코어 모듈 로드 실패: {e}")
    raise ImportError(f"코어 모듈을 로드할 수 없습니다: {e}")

logger.info("✅ Virtual Fitting 코어 모듈 로드 완료")
