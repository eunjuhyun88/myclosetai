#!/usr/bin/env python3
"""
🔥 MyCloset AI - Base Step Package
==================================

AI Pipeline Step들의 기본 기능을 제공하는 패키지
모듈화된 구조로 각 기능을 독립적으로 관리

Author: MyCloset AI Team
Date: 2025-08-14
Version: 2.0 (모듈화 버전)
"""

# 핵심 모듈들 import - 상대 import로 변경
try:
    from .core.base_step_mixin import BaseStepMixin
    from .core.step_interface import StepInterface
    print("✅ 상대 경로로 Core 모듈 import 성공")
except ImportError as e:
    print(f"⚠️ 상대 경로로 Core 모듈 import 실패: {e}")
    BaseStepMixin = None
    StepInterface = None

# 기능별 모듈들 import (오류 방지)
try:
    from .features.dependency_injection import DependencyInjectionMixin
except ImportError:
    DependencyInjectionMixin = None

try:
    from .features.performance_tracking import PerformanceTrackingMixin
except ImportError:
    PerformanceTrackingMixin = None

try:
    from .features.data_conversion import DataConversionMixin
except ImportError:
    DataConversionMixin = None

try:
    from .features.central_hub import CentralHubMixin
except ImportError:
    CentralHubMixin = None

# 유틸리티 모듈들 import (오류 방지)
try:
    from .utils.validation import ValidationMixin
except ImportError:
    ValidationMixin = None

try:
    from .utils.error_handling import ErrorHandlingMixin
except ImportError:
    ErrorHandlingMixin = None

# 주요 클래스들을 직접 노출 (None이 아닌 것만)
__all__ = []
if BaseStepMixin:
    __all__.append('BaseStepMixin')
    print(f"✅ BaseStepMixin 로드 성공: {BaseStepMixin}")
else:
    print("❌ BaseStepMixin 로드 실패")

if StepInterface:
    __all__.append('StepInterface')
if DependencyInjectionMixin:
    __all__.append('DependencyInjectionMixin')
if PerformanceTrackingMixin:
    __all__.append('PerformanceTrackingMixin')
if DataConversionMixin:
    __all__.append('DataConversionMixin')
if CentralHubMixin:
    __all__.append('CentralHubMixin')
if ValidationMixin:
    __all__.append('ValidationMixin')
if ErrorHandlingMixin:
    __all__.append('ErrorHandlingMixin')

# 버전 정보
__version__ = "2.0.0"
__author__ = "MyCloset AI Team"
