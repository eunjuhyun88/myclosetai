"""
Post Processing Models Configuration Module

이 모듈은 후처리 모델들의 설정을 제공합니다.
"""

from .model_config import PostProcessingModelConfig
from .inference_config import PostProcessingInferenceConfig

__all__ = [
    'PostProcessingModelConfig',
    'PostProcessingInferenceConfig'
]
