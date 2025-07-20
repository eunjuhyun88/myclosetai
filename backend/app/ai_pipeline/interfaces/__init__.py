# app/ai_pipeline/interfaces/__init__.py
"""인터페이스 패키지"""

from .model_interface import IModelLoader, IStepInterface
from .memory_interface import IMemoryManager  
from .data_interface import IDataConverter

__all__ = [
    'IModelLoader',
    'IStepInterface', 
    'IMemoryManager',
    'IDataConverter'
]

# ==============================================
# app/ai_pipeline/interfaces/model_interface.py
# ==============================================
"""
🔥 모델 관련 인터페이스 정의
============================

✅ 순환참조 방지를 위한 추상 인터페이스
✅ 기존 ModelLoader와 100% 호환
✅ 타입 힌팅 지원
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List, Union
import asyncio

class IStepInterface(ABC):
    """Step 인터페이스 추상화"""
    
    @abstractmethod
    def get_model(self, model_name: Optional[str] = None) -> Optional[Any]:
        """동기 모델 조회"""
        pass
    
    @abstractmethod
    async def get_model_async(self, model_name: Optional[str] = None) -> Optional[Any]:
        """비동기 모델 조회"""
        pass
    
    @abstractmethod
    def list_available_models(self) -> List[str]:
        """사용 가능한 모델 목록"""
        pass

class IModelLoader(ABC):
    """모델 로더 인터페이스 추상화"""
    
    @abstractmethod
    def get_model(self, model_name: str, **kwargs) -> Optional[Any]:
        """동기 모델 로드"""
        pass
    
    @abstractmethod
    async def get_model_async(self, model_name: str, **kwargs) -> Optional[Any]:
        """비동기 모델 로드"""
        pass
    
    @abstractmethod
    def create_step_interface(self, step_name: str, **kwargs) -> IStepInterface:
        """Step 인터페이스 생성"""
        pass
    
    @abstractmethod
    def list_models(self) -> Dict[str, Dict[str, Any]]:
        """모델 목록 조회"""
        pass
    
    @abstractmethod
    def cleanup(self):
        """리소스 정리"""
        pass

# ==============================================
# app/ai_pipeline/interfaces/memory_interface.py
# ==============================================
"""
🔥 메모리 관리 인터페이스 정의
============================
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

class IMemoryManager(ABC):
    """메모리 관리자 인터페이스"""
    
    @abstractmethod
    def optimize_memory(self, **kwargs) -> Dict[str, Any]:
        """동기 메모리 최적화"""
        pass
    
    @abstractmethod
    async def optimize_memory_async(self, **kwargs) -> Dict[str, Any]:
        """비동기 메모리 최적화"""
        pass
    
    @abstractmethod
    def get_memory_status(self) -> Dict[str, Any]:
        """메모리 상태 조회"""
        pass
    
    @abstractmethod
    def cleanup_memory(self, aggressive: bool = False) -> Dict[str, Any]:
        """메모리 정리"""
        pass

# ==============================================
# app/ai_pipeline/interfaces/data_interface.py
# ==============================================
"""
🔥 데이터 변환 인터페이스 정의
============================
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple, Union
from PIL import Image
import numpy as np

class IDataConverter(ABC):
    """데이터 변환기 인터페이스"""
    
    @abstractmethod
    def convert_image(self, image: Any, target_format: str = "tensor", **kwargs) -> Any:
        """이미지 변환"""
        pass
    
    @abstractmethod
    def preprocess_image(self, image: Any, size: Tuple[int, int] = (512, 512), **kwargs) -> Any:
        """이미지 전처리"""
        pass
    
    @abstractmethod
    def postprocess_result(self, result: Any, output_format: str = "image", **kwargs) -> Any:
        """결과 후처리"""
        pass
    
    @abstractmethod
    def tensor_to_image(self, tensor: Any, **kwargs) -> Image.Image:
        """텐서를 이미지로 변환"""
        pass
    
    @abstractmethod
    def image_to_tensor(self, image: Union[Image.Image, np.ndarray], **kwargs) -> Any:
        """이미지를 텐서로 변환"""
        pass

# ==============================================
# app/ai_pipeline/interfaces/step_interface.py
# ==============================================
"""
🔥 Step 관련 인터페이스 정의
===========================
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

class IBaseStep(ABC):
    """베이스 Step 인터페이스"""
    
    @abstractmethod
    def initialize_step(self) -> bool:
        """Step 초기화"""
        pass
    
    @abstractmethod
    async def initialize_step_async(self) -> bool:
        """Step 비동기 초기화"""
        pass
    
    @abstractmethod
    def process(self, data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """동기 처리"""
        pass
    
    @abstractmethod
    async def process_async(self, data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """비동기 처리"""
        pass
    
    @abstractmethod
    def get_status(self) -> Dict[str, Any]:
        """Step 상태 조회"""
        pass
    
    @abstractmethod
    def cleanup(self):
        """Step 정리"""
        pass

class IStepFactory(ABC):
    """Step 팩토리 인터페이스"""
    
    @abstractmethod
    def create_step(self, step_name: str, **kwargs) -> IBaseStep:
        """Step 생성"""
        pass
    
    @abstractmethod
    def get_available_steps(self) -> List[str]:
        """사용 가능한 Step 목록"""
        pass

# ==============================================
# app/ai_pipeline/interfaces/pipeline_interface.py
# ==============================================
"""
🔥 파이프라인 인터페이스 정의
============================
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from enum import Enum

class ProcessingStatus(Enum):
    """처리 상태"""
    IDLE = "idle"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class IPipelineManager(ABC):
    """파이프라인 매니저 인터페이스"""
    
    @abstractmethod
    def initialize(self) -> bool:
        """파이프라인 초기화"""
        pass
    
    @abstractmethod
    async def initialize_async(self) -> bool:
        """파이프라인 비동기 초기화"""
        pass
    
    @abstractmethod
    def process_virtual_fitting(self, session_id: str, user_image: Any, cloth_image: Any, **kwargs) -> Dict[str, Any]:
        """가상 피팅 처리"""
        pass
    
    @abstractmethod
    async def process_virtual_fitting_async(self, session_id: str, user_image: Any, cloth_image: Any, **kwargs) -> Dict[str, Any]:
        """가상 피팅 비동기 처리"""
        pass
    
    @abstractmethod
    def get_status(self) -> Dict[str, Any]:
        """파이프라인 상태 조회"""
        pass
    
    @abstractmethod
    def cleanup(self):
        """파이프라인 정리"""
        pass