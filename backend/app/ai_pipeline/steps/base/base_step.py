"""
Step 기본 인터페이스
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass
import logging
import time


@dataclass
class StepResult:
    """Step 실행 결과"""
    success: bool
    data: Optional[Dict[str, Any]] = None
    error_message: str = ""
    processing_time: float = 0.0
    step_name: str = ""


class BaseStep(ABC):
    """모든 Step의 기본 인터페이스"""
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None, logger=None):
        self.name = name
        self.config = config or {}
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.models = {}
        self._is_initialized = False
        
    @abstractmethod
    def _load_models(self) -> bool:
        """모델 로딩 - 하위 클래스에서 구현"""
        pass
    
    @abstractmethod
    def _run_inference(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """AI 추론 실행 - 하위 클래스에서 구현"""
        pass
    
    def initialize(self) -> bool:
        """Step 초기화"""
        try:
            self.logger.info(f"🚀 {self.name} Step 초기화 시작")
            start_time = time.time()
            
            if not self._load_models():
                self.logger.error(f"❌ {self.name} 모델 로딩 실패")
                return False
            
            self._is_initialized = True
            processing_time = time.time() - start_time
            self.logger.info(f"✅ {self.name} Step 초기화 완료 ({processing_time:.2f}s)")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ {self.name} Step 초기화 실패: {str(e)}")
            return False
    
    def process(self, input_data: Dict[str, Any]) -> StepResult:
        """Step 처리 - 메인 메서드"""
        try:
            if not self._is_initialized:
                if not self.initialize():
                    return StepResult(
                        success=False,
                        error_message=f"{self.name} Step 초기화 실패",
                        step_name=self.name
                    )
            
            self.logger.info(f"🔄 {self.name} Step 처리 시작")
            start_time = time.time()
            
            # 입력 데이터 검증
            if not self._validate_input(input_data):
                return StepResult(
                    success=False,
                    error_message=f"{self.name} 입력 데이터 검증 실패",
                    step_name=self.name
                )
            
            # AI 추론 실행
            result_data = self._run_inference(input_data)
            
            processing_time = time.time() - start_time
            self.logger.info(f"✅ {self.name} Step 처리 완료 ({processing_time:.2f}s)")
            
            return StepResult(
                success=True,
                data=result_data,
                processing_time=processing_time,
                step_name=self.name
            )
            
        except Exception as e:
            self.logger.error(f"❌ {self.name} Step 처리 실패: {str(e)}")
            return StepResult(
                success=False,
                error_message=str(e),
                step_name=self.name
            )
    
    def _validate_input(self, input_data: Dict[str, Any]) -> bool:
        """입력 데이터 검증 - 기본 구현"""
        return input_data is not None
    
    def cleanup(self):
        """리소스 정리"""
        try:
            self.models.clear()
            self._is_initialized = False
            self.logger.info(f"🧹 {self.name} Step 리소스 정리 완료")
        except Exception as e:
            self.logger.error(f"❌ {self.name} Step 리소스 정리 실패: {str(e)}")
    
    @property
    def is_initialized(self) -> bool:
        """초기화 상태 확인"""
        return self._is_initialized
