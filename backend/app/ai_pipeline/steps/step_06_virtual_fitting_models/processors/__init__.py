#!/usr/bin/env python3
"""
🔥 MyCloset AI - Virtual Fitting Processors
==========================================

🎯 가상 피팅 처리기들
✅ 가상 피팅 프로세서
✅ 가상 피팅 품질 향상기
✅ 가상 피팅 검증기
✅ 가상 피팅 최적화기
✅ 가상 피팅 특수 케이스 처리기
"""

import logging

# 로거 설정
logger = logging.getLogger(__name__)

try:
    from .virtual_fitting_processor import VirtualFittingProcessor
    from .virtual_fitting_quality_enhancer import VirtualFittingQualityEnhancer
    from .virtual_fitting_validator import VirtualFittingValidator
    from .virtual_fitting_optimizer import VirtualFittingOptimizer
    from .virtual_fitting_special_case_processor import VirtualFittingSpecialCaseProcessor
    
    __all__ = [
        "VirtualFittingProcessor",
        "VirtualFittingQualityEnhancer",
        "VirtualFittingValidator",
        "VirtualFittingOptimizer",
        "VirtualFittingSpecialCaseProcessor"
    ]
    
except ImportError as e:
    logger.error(f"처리기 모듈 로드 실패: {e}")
    raise ImportError(f"처리기 모듈을 로드할 수 없습니다: {e}")

logger.info("✅ Virtual Fitting 처리기 모듈 로드 완료")
