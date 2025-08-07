"""
🔥 Human Parsing 앙상블 패키지
============================

Human Parsing 관련 앙상블 시스템을 포함합니다.

모듈:
- hybrid_ensemble.py: 하이브리드 앙상블 모듈
- memory_efficient_ensemble.py: 메모리 효율적 앙상블
- model_ensemble_manager.py: 모델 앙상블 매니저

Author: MyCloset AI Team
Date: 2025-08-07
Version: 1.0
"""

from .model_ensemble_manager import ModelEnsembleManager
from .memory_efficient_ensemble import MemoryEfficientEnsembleSystem

__all__ = [
    "ModelEnsembleManager",
    "MemoryEfficientEnsembleSystem"
]
