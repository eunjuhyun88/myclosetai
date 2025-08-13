#!/usr/bin/env python3
"""
🔥 MyCloset AI - Step 03: 의류 세그멘테이션 - Config Package
=====================================================================

설정 관련 클래스들을 포함하는 패키지

Author: MyCloset AI Team  
Date: 2025-08-01
Version: 1.0
"""

# types.py에서 클래스들 export
try:
    from .types import (
        SegmentationMethod, ClothCategory, QualityLevel,
        SegmentationModel, SegmentationQuality, ClothSegmentationConfig, SegmentationResult
    )
except ImportError:
    SegmentationMethod = None
    ClothCategory = None
    QualityLevel = None
    SegmentationModel = None
    SegmentationQuality = None
    ClothSegmentationConfig = None
    SegmentationResult = None

# config.py에서 클래스들 export
try:
    from .config import *
except ImportError:
    pass

__all__ = [
    'SegmentationMethod',
    'ClothCategory', 
    'QualityLevel',
    'SegmentationModel',
    'SegmentationQuality',
    'ClothSegmentationConfig',
    'SegmentationResult'
]
