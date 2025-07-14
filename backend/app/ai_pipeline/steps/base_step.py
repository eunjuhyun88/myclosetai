# app/ai_pipeline/steps/base_step.py
"""
베이스 스텝 클래스 - import 에러 해결용
기존 Step 클래스들이 이미 올바른 생성자 패턴을 사용하고 있음
"""
from typing import Dict, Any, Optional
from abc import ABC, abstractmethod
import logging

class BaseStep(ABC):
    """기본 Step 베이스 클래스"""
    
    def __init__(self, device: str, config: Optional[Dict[str, Any]] = None):
        """
        기본 생성자
        
        Args:
            device: 디바이스 ('cpu', 'cuda', 'mps')
            config: 설정 딕셔너리
        """
        self.device = device
        self.config = config or {}
        self.step_name = self.__class__.__name__
        self.logger = logging.getLogger(f"pipeline.{self.step_name}")
        self.is_initialized = False
    
    @abstractmethod
    async def initialize(self) -> bool:
        """스텝 초기화"""
        pass
    
    @abstractmethod
    async def process(self, *args, **kwargs) -> Dict[str, Any]:
        """메인 처리"""
        pass

# 하위 호환성을 위한 별칭
OptimalStepConstructor = BaseStep