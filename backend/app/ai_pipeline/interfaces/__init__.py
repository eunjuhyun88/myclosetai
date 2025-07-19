# app/ai_pipeline/interfaces/__init__.py
"""
🔥 AI Pipeline 인터페이스 모듈
✅ 순환 임포트 해결을 위한 추상 인터페이스들
✅ 의존성 주입 패턴 지원
"""

from .model_interface import (
    IModelLoader,
    IStepInterface,
    IMemoryManager,
    IDataConverter
)

__all__ = [
    'IModelLoader',
    'IStepInterface', 
    'IMemoryManager',
    'IDataConverter'
]