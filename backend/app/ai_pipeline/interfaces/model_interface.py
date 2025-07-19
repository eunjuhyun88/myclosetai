# app/ai_pipeline/interfaces/model_interface.py
"""
🔥 모델 로더 인터페이스 - 순환 임포트 해결
✅ 추상 인터페이스로 결합도 낮춤
✅ 의존성 주입 패턴 지원
✅ 기존 기능 100% 호환
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union, Callable
import asyncio

class IModelLoader(ABC):
    """모델 로더 인터페이스"""
    
    @abstractmethod
    def create_step_interface(self, step_name: str) -> 'IStepInterface':
        """Step 인터페이스 생성"""
        pass
    
    @abstractmethod
    async def load_model(self, model_config: Dict[str, Any]) -> Any:
        """모델 로드"""
        pass
    
    @abstractmethod
    def get_model(self, model_name: str) -> Optional[Any]:
        """모델 조회"""
        pass
    
    @abstractmethod
    def register_model(self, model_name: str, model_config: Dict[str, Any]) -> bool:
        """모델 등록"""
        pass
    
    @abstractmethod
    def cleanup(self) -> None:
        """리소스 정리"""
        pass

class IStepInterface(ABC):
    """Step 인터페이스"""
    
    @abstractmethod
    async def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """AI 모델 처리"""
        pass
    
    @abstractmethod
    def list_available_models(self) -> List[str]:
        """사용 가능한 모델 목록"""
        pass
    
    @abstractmethod
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """모델 정보 조회"""
        pass

class IMemoryManager(ABC):
    """메모리 관리자 인터페이스"""
    
    @abstractmethod
    def cleanup_memory(self, aggressive: bool = False) -> Dict[str, Any]:
        """메모리 정리"""
        pass
    
    @abstractmethod
    def get_memory_stats(self) -> Dict[str, Any]:
        """메모리 상태 조회"""
        pass

class IDataConverter(ABC):
    """데이터 변환기 인터페이스"""
    
    @abstractmethod
    def preprocess_image(self, image: Any, **kwargs) -> Any:
        """이미지 전처리"""
        pass
    
    @abstractmethod
    def postprocess_result(self, result: Any, **kwargs) -> Any:
        """결과 후처리"""
        pass

class ISafeFunctionValidator(ABC):
    """안전한 함수 호출 검증기 인터페이스"""
    
    @abstractmethod
    def safe_call(self, func: Callable, *args, **kwargs) -> tuple[bool, Any, str]:
        """안전한 함수 호출
        
        Returns:
            (success: bool, result: Any, message: str)
        """
        pass
    
    @abstractmethod
    async def safe_async_call(self, func: Callable, *args, **kwargs) -> tuple[bool, Any, str]:
        """안전한 비동기 함수 호출"""
        pass