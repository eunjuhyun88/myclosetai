#!/usr/bin/env python3
"""
🔥 MyCloset AI - Step 03: 의류 세그멘테이션 - Core Package
=====================================================================

의류 세그멘테이션의 핵심 기능들 (논리적 통합)

Author: MyCloset AI Team  
Date: 2025-08-01
Version: 1.0
"""

from .segmentation_core import SegmentationCore
from .ensemble_core import EnsembleCore

__all__ = [
    'SegmentationCore',
    'EnsembleCore'
]
