#!/usr/bin/env python3
"""
🔥 MyCloset AI - Step Interface
================================

Step과 외부 시스템 간의 표준화된 인터페이스
API 호출, 데이터 변환, 상태 관리 등을 담당

Author: MyCloset AI Team
Date: 2025-08-14
Version: 2.0
"""

import logging
from typing import Dict, Any, Optional, Union
from abc import ABC, abstractmethod

class StepInterface(ABC):
    """Step과 외부 시스템 간의 표준화된 인터페이스"""
    
    def __init__(self, step_instance):
        self.step = step_instance
        self.logger = logging.getLogger(f"interface.{step_instance.__class__.__name__}")
    
    @abstractmethod
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Step 처리 실행"""
        pass
    
    def get_status(self) -> Dict[str, Any]:
        """Step 상태 반환"""
        return {
            'step_name': getattr(self.step, 'step_name', 'Unknown'),
            'step_id': getattr(self.step, 'step_id', 0),
            'is_initialized': getattr(self.step, 'is_initialized', False),
            'is_ready': getattr(self.step, 'is_ready', False),
            'dependencies_valid': self._check_dependencies()
        }
    
    def _check_dependencies(self) -> bool:
        """의존성 상태 확인"""
        try:
            if hasattr(self.step, 'validate_dependencies'):
                result = self.step.validate_dependencies()
                if isinstance(result, dict):
                    return result.get('all_dependencies_valid', False)
            return False
        except Exception:
            return False
