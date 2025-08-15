#!/usr/bin/env python3
"""
🔥 MyCloset AI - Virtual Fitting Services
========================================

🎯 가상 피팅 서비스들
✅ 가상 피팅 메인 서비스
✅ 가상 피팅 품질 관리 서비스
✅ 가상 피팅 검증 서비스
✅ 가상 피팅 최적화 서비스
✅ 가상 피팅 모니터링 서비스
"""

import logging

# 로거 설정
logger = logging.getLogger(__name__)

try:
    from .virtual_fitting_service import VirtualFittingService
    from .virtual_fitting_quality_service import VirtualFittingQualityService
    from .virtual_fitting_validation_service import VirtualFittingValidationService
    from .virtual_fitting_optimization_service import VirtualFittingOptimizationService
    from .virtual_fitting_monitoring_service import VirtualFittingMonitoringService
    
    __all__ = [
        "VirtualFittingService",
        "VirtualFittingQualityService",
        "VirtualFittingValidationService",
        "VirtualFittingOptimizationService",
        "VirtualFittingMonitoringService"
    ]
    
except ImportError as e:
    logger.error(f"서비스 모듈 로드 실패: {e}")
    raise ImportError(f"서비스 모듈을 로드할 수 없습니다: {e}")

logger.info("✅ Virtual Fitting 서비스 모듈 로드 완료")
