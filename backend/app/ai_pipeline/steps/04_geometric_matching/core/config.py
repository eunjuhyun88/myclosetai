"""
Configuration classes for geometric matching step.
"""

import time
from dataclasses import dataclass, field
from typing import Dict, Any, Optional


@dataclass
class GeometricMatchingConfig:
    """기하학적 매칭 설정"""
    input_size: tuple = (256, 192)
    confidence_threshold: float = 0.7
    enable_visualization: bool = True
    device: str = "auto"
    matching_method: str = "advanced_deeplab_aspp_self_attention"


@dataclass
class ProcessingStatus:
    """처리 상태 추적 클래스"""
    models_loaded: bool = False
    advanced_ai_loaded: bool = False
    model_creation_success: bool = False
    requirements_compatible: bool = False
    initialization_complete: bool = False
    last_updated: float = field(default_factory=time.time)
    
    def update_status(self, **kwargs):
        """상태 업데이트"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        self.last_updated = time.time()
    
    def get_status_summary(self) -> Dict[str, Any]:
        """상태 요약 반환"""
        return {
            'models_loaded': self.models_loaded,
            'advanced_ai_loaded': self.advanced_ai_loaded,
            'model_creation_success': self.model_creation_success,
            'requirements_compatible': self.requirements_compatible,
            'initialization_complete': self.initialization_complete,
            'last_updated': self.last_updated
        }
    
    def is_ready(self) -> bool:
        """모든 요구사항이 충족되었는지 확인"""
        return all([
            self.models_loaded,
            self.advanced_ai_loaded,
            self.model_creation_success,
            self.requirements_compatible,
            self.initialization_complete
        ])
