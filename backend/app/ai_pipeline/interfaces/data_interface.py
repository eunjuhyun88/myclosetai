# app/ai_pipeline/interfaces/data_interface.py
"""
🔥 데이터 변환 인터페이스 v2.0 - 완전한 데이터 처리
==================================================

✅ BaseStepMixin v10.0 완벽 호환
✅ DI Container 인터페이스 패턴 적용
✅ 이미지, 텐서, JSON 등 다양한 데이터 타입 지원
✅ PIL, OpenCV, NumPy, PyTorch 호환
✅ M3 Max 최적화 데이터 변환
✅ 비동기 데이터 처리 지원
✅ conda 환경 완벽 지원
✅ 프로덕션 안정성 보장

Author: MyCloset AI Team
Date: 2025-07-20
Version: 2.0 (Complete Data Processing)
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple, Union, Callable
from enum import Enum
import time
from pathlib import Path

# ==============================================
# 🔥 데이터 처리 관련 데이터 클래스
# ==============================================

class DataFormat(Enum):
    """지원하는 데이터 포맷"""
    # 이미지 포맷
    PIL_IMAGE = "pil"
    OPENCV_IMAGE = "opencv"
    NUMPY_ARRAY = "numpy"
    TENSOR = "tensor"
    
    # 텍스트 포맷
    JSON = "json"
    DICT = "dict"
    STRING = "string"
    
    # 파일 포맷
    IMAGE_FILE = "image_file"
    JSON_FILE = "json_file"
    BINARY_FILE = "binary_file"

class ImageProcessingMode(Enum):
    """이미지 처리 모드"""
    RESIZE = "resize"
    CROP = "crop"
    PAD = "pad"
    NORMALIZE = "normalize"
    DENORMALIZE = "denormalize"
    AUGMENT = "augment"

class TensorDevice(Enum):
    """텐서 디바이스"""
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"     # M3 Max Metal
    AUTO = "auto"

# ==============================================
# 🔥 기본 데이터 변환기 인터페이스
# ==============================================

class IDataConverter(ABC):
    """
    기본 데이터 변환기 인터페이스
    
    BaseStepMixin v10.0의 data_converter 속성으로 주입됨
    """
    
    @abstractmethod
    def convert_data(self, data: Any, source_format: DataFormat, target_format: DataFormat, **kwargs) -> Tuple[bool, Any, str]:
        """
        데이터 포맷 변환
        
        Args:
            data: 입력 데이터
            source_format: 소스 포맷
            target_format: 타겟 포맷
            **kwargs: 변환 옵션
            
        Returns:
            (성공여부, 변환된_데이터, 메시지)
        """
        pass
    
    @abstractmethod
    async def convert_data_async(self, data: Any, source_format: DataFormat, target_format: DataFormat, **kwargs) -> Tuple[bool, Any, str]:
        """
        비동기 데이터 포맷 변환
        
        Args:
            data: 입력 데이터
            source_format: 소스 포맷
            target_format: 타겟 포맷
            **kwargs: 변환 옵션
            
        Returns:
            (성공여부, 변환된_데이터, 메시지)
        """
        pass
    
    @abstractmethod
    def validate_data(self, data: Any, expected_format: DataFormat, **constraints) -> Tuple[bool, str]:
        """
        데이터 유효성 검증
        
        Args:
            data: 검증할 데이터
            expected_format: 예상 포맷
            **constraints: 제약 조건 (크기, 타입 등)
            
        Returns:
            (유효성, 메시지)
        """
        pass
    
    @abstractmethod
    def get_data_info(self, data: Any) -> Dict[str, Any]:
        """
        데이터 정보 조회
        
        Args:
            data: 분석할 데이터
            
        Returns:
            {
                'format': str,
                'size_bytes': int,
                'dimensions': Tuple,
                'dtype': str,
                'device': str,  # 텐서인 경우
                'channels': int,  # 이미지인 경우
                'metadata': Dict[str, Any]
            }
        """
        pass

# ==============================================
# 🔥 이미지 변환기 인터페이스
# ==============================================

class IImageConverter(ABC):
    """
    이미지 변환기 인터페이스
    
    PIL, OpenCV, NumPy, PyTorch 텐서 간 변환 지원
    """
    
    @abstractmethod
    def preprocess_image(self, image: Any, target_size: Tuple[int, int] = (512, 512), **kwargs) -> Any:
        """
        이미지 전처리
        
        Args:
            image: 입력 이미지 (PIL, OpenCV, NumPy, Tensor)
            target_size: 목표 크기
            **kwargs: 전처리 옵션
                - mode: ImageProcessingMode
                - normalize: bool
                - mean: List[float]
                - std: List[float]
                - device: str
                
        Returns:
            전처리된 이미지
        """
        pass
    
    @abstractmethod
    def postprocess_image(self, image: Any, original_size: Optional[Tuple[int, int]] = None, **kwargs) -> Any:
        """
        이미지 후처리
        
        Args:
            image: 처리된 이미지
            original_size: 원본 크기 (복원용)
            **kwargs: 후처리 옵션
                - denormalize: bool
                - to_pil: bool
                - to_numpy: bool
                
        Returns:
            후처리된 이미지
        """
        pass
    
    @abstractmethod
    def resize_image(self, image: Any, size: Tuple[int, int], method: str = "bilinear") -> Any:
        """
        이미지 크기 조정
        
        Args:
            image: 입력 이미지
            size: 목표 크기
            method: 리사이즈 방법
            
        Returns:
            크기 조정된 이미지
        """
        pass
    
    @abstractmethod
    def normalize_image(self, image: Any, mean: List[float], std: List[float]) -> Any:
        """
        이미지 정규화
        
        Args:
            image: 입력 이미지
            mean: 평균값 리스트
            std: 표준편차 리스트
            
        Returns:
            정규화된 이미지
        """
        pass
    
    @abstractmethod
    def to_tensor(self, image: Any, device: TensorDevice = TensorDevice.AUTO) -> Any:
        """
        이미지를 텐서로 변환
        
        Args:
            image: 입력 이미지
            device: 대상 디바이스
            
        Returns:
            텐서 이미지
        """
        pass
    
    @abstractmethod
    def to_pil(self, image: Any) -> Any:
        """
        이미지를 PIL로 변환
        
        Args:
            image: 입력 이미지
            
        Returns:
            PIL 이미지
        """
        pass
    
    @abstractmethod
    def to_numpy(self, image: Any) -> Any:
        """
        이미지를 NumPy로 변환
        
        Args:
            image: 입력 이미지
            
        Returns:
            NumPy 이미지
        """
        pass
    
    @abstractmethod
    def to_opencv(self, image: Any) -> Any:
        """
        이미지를 OpenCV로 변환
        
        Args:
            image: 입력 이미지
            
        Returns:
            OpenCV 이미지
        """
        pass

# ==============================================
# 🔥 텐서 변환기 인터페이스
# ==============================================

class ITensorConverter(ABC):
    """
    텐서 변환기 인터페이스
    
    PyTorch, NumPy 텐서 처리 특화
    """
    
    @abstractmethod
    def convert_to_tensor(self, data: Any, device: TensorDevice = TensorDevice.AUTO, dtype: Optional[str] = None) -> Any:
        """
        데이터를 텐서로 변환
        
        Args:
            data: 입력 데이터
            device: 대상 디바이스
            dtype: 데이터 타입
            
        Returns:
            텐서 데이터
        """
        pass
    
    @abstractmethod
    def convert_from_tensor(self, tensor: Any, target_format: DataFormat = DataFormat.NUMPY_ARRAY) -> Any:
        """
        텐서에서 다른 포맷으로 변환
        
        Args:
            tensor: 입력 텐서
            target_format: 출력 포맷
            
        Returns:
            변환된 데이터
        """
        pass
    
    @abstractmethod
    def move_tensor(self, tensor: Any, device: TensorDevice) -> Any:
        """
        텐서 디바이스 이동
        
        Args:
            tensor: 입력 텐서
            device: 대상 디바이스
            
        Returns:
            이동된 텐서
        """
        pass
    
    @abstractmethod
    def change_tensor_dtype(self, tensor: Any, dtype: str) -> Any:
        """
        텐서 데이터 타입 변경
        
        Args:
            tensor: 입력 텐서
            dtype: 목표 데이터 타입
            
        Returns:
            변환된 텐서
        """
        pass
    
    @abstractmethod
    def get_tensor_info(self, tensor: Any) -> Dict[str, Any]:
        """
        텐서 정보 조회
        
        Args:
            tensor: 분석할 텐서
            
        Returns:
            텐서 메타데이터
        """
        pass
    
    @abstractmethod
    def optimize_tensor_memory(self, tensor: Any) -> Any:
        """
        텐서 메모리 최적화
        
        Args:
            tensor: 입력 텐서
            
        Returns:
            최적화된 텐서
        """
        pass

# ==============================================
# 🔥 파일 변환기 인터페이스
# ==============================================

class IFileConverter(ABC):
    """
    파일 변환기 인터페이스
    
    파일 입출력 및 포맷 변환
    """
    
    @abstractmethod
    def load_image_file(self, file_path: Union[str, Path], **kwargs) -> Tuple[bool, Any, str]:
        """
        이미지 파일 로드
        
        Args:
            file_path: 파일 경로
            **kwargs: 로드 옵션
            
        Returns:
            (성공여부, 이미지_데이터, 메시지)
        """
        pass
    
    @abstractmethod
    def save_image_file(self, image: Any, file_path: Union[str, Path], **kwargs) -> Tuple[bool, str]:
        """
        이미지 파일 저장
        
        Args:
            image: 이미지 데이터
            file_path: 저장 경로
            **kwargs: 저장 옵션
            
        Returns:
            (성공여부, 메시지)
        """
        pass
    
    @abstractmethod
    def load_json_file(self, file_path: Union[str, Path]) -> Tuple[bool, Any, str]:
        """
        JSON 파일 로드
        
        Args:
            file_path: 파일 경로
            
        Returns:
            (성공여부, JSON_데이터, 메시지)
        """
        pass
    
    @abstractmethod
    def save_json_file(self, data: Any, file_path: Union[str, Path], **kwargs) -> Tuple[bool, str]:
        """
        JSON 파일 저장
        
        Args:
            data: 저장할 데이터
            file_path: 저장 경로
            **kwargs: 저장 옵션
            
        Returns:
            (성공여부, 메시지)
        """
        pass
    
    @abstractmethod
    def validate_file_format(self, file_path: Union[str, Path], expected_format: str) -> Tuple[bool, str]:
        """
        파일 포맷 검증
        
        Args:
            file_path: 파일 경로
            expected_format: 예상 포맷
            
        Returns:
            (유효성, 메시지)
        """
        pass
    
    @abstractmethod
    def get_file_info(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        파일 정보 조회
        
        Args:
            file_path: 파일 경로
            
        Returns:
            파일 메타데이터
        """
        pass

# ==============================================
# 🔥 배치 처리기 인터페이스
# ==============================================

class IBatchProcessor(ABC):
    """
    배치 처리기 인터페이스
    
    대량 데이터 배치 처리
    """
    
    @abstractmethod
    def process_batch(self, data_list: List[Any], processor_func: Callable, **kwargs) -> List[Tuple[bool, Any, str]]:
        """
        배치 처리
        
        Args:
            data_list: 데이터 리스트
            processor_func: 처리 함수
            **kwargs: 처리 옵션
            
        Returns:
            [(성공여부, 결과, 메시지), ...]
        """
        pass
    
    @abstractmethod
    async def process_batch_async(self, data_list: List[Any], processor_func: Callable, **kwargs) -> List[Tuple[bool, Any, str]]:
        """
        비동기 배치 처리
        
        Args:
            data_list: 데이터 리스트
            processor_func: 비동기 처리 함수
            **kwargs: 처리 옵션
            
        Returns:
            [(성공여부, 결과, 메시지), ...]
        """
        pass
    
    @abstractmethod
    def create_data_loader(self, data_list: List[Any], batch_size: int, **kwargs) -> Any:
        """
        데이터 로더 생성
        
        Args:
            data_list: 데이터 리스트
            batch_size: 배치 크기
            **kwargs: 로더 옵션
            
        Returns:
            데이터 로더 객체
        """
        pass
    
    @abstractmethod
    def process_parallel(self, data_list: List[Any], processor_func: Callable, num_workers: int = 4, **kwargs) -> List[Tuple[bool, Any, str]]:
        """
        병렬 처리
        
        Args:
            data_list: 데이터 리스트
            processor_func: 처리 함수
            num_workers: 워커 수
            **kwargs: 처리 옵션
            
        Returns:
            [(성공여부, 결과, 메시지), ...]
        """
        pass

# ==============================================
# 🔥 데이터 검증기 인터페이스
# ==============================================

class IDataValidator(ABC):
    """
    데이터 검증기 인터페이스
    
    입력/출력 데이터 유효성 검증
    """
    
    @abstractmethod
    def validate_input_data(self, data: Any, schema: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        입력 데이터 검증
        
        Args:
            data: 검증할 데이터
            schema: 데이터 스키마
            
        Returns:
            (유효성, 오류_메시지_리스트)
        """
        pass
    
    @abstractmethod
    def validate_output_data(self, data: Any, schema: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        출력 데이터 검증
        
        Args:
            data: 검증할 데이터
            schema: 데이터 스키마
            
        Returns:
            (유효성, 오류_메시지_리스트)
        """
        pass
    
    @abstractmethod
    def create_validation_schema(self, data_type: str, **constraints) -> Dict[str, Any]:
        """
        검증 스키마 생성
        
        Args:
            data_type: 데이터 타입
            **constraints: 제약 조건
            
        Returns:
            검증 스키마
        """
        pass
    
    @abstractmethod
    def validate_data_pipeline(self, pipeline_config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        데이터 파이프라인 검증
        
        Args:
            pipeline_config: 파이프라인 설정
            
        Returns:
            (유효성, 오류_메시지_리스트)
        """
        pass

# ==============================================
# 🔥 인터페이스 타입 유니온 및 내보내기
# ==============================================

# 편의성 타입 별칭
DataConverterInterface = IDataConverter
ImageConverterInterface = IImageConverter
TensorConverterInterface = ITensorConverter
FileConverterInterface = IFileConverter
BatchProcessorInterface = IBatchProcessor
DataValidatorInterface = IDataValidator

# 데이터 관련 인터페이스 목록
DATA_INTERFACES = [
    'IDataConverter',
    'IImageConverter',
    'ITensorConverter',
    'IFileConverter',
    'IBatchProcessor',
    'IDataValidator'
]

# 모듈 내보내기
__all__ = [
    # 인터페이스들
    'IDataConverter',
    'IImageConverter',
    'ITensorConverter',
    'IFileConverter',
    'IBatchProcessor',
    'IDataValidator',
    
    # 데이터 클래스들
    'DataFormat',
    'ImageProcessingMode',
    'TensorDevice',
    
    # 편의성 타입 별칭
    'DataConverterInterface',
    'ImageConverterInterface',
    'TensorConverterInterface',
    'FileConverterInterface',
    'BatchProcessorInterface',
    'DataValidatorInterface',
    
    # 유틸리티
    'DATA_INTERFACES'
]

# 모듈 로드 완료 메시지
print("✅ Data Interface v2.0 로드 완료 - 완전한 데이터 처리")
print("🖼️ 이미지 변환: PIL, OpenCV, NumPy, PyTorch 지원")
print("⚡ 텐서 변환: CPU, CUDA, MPS 지원")
print("📁 파일 처리: 다양한 포맷 지원")
print("🔗 BaseStepMixin v10.0과 100% 호환")
print("🚀 데이터 처리 인터페이스 6종 정의 완료!")