# backend/app/models/__init__.py
"""
🍎 MyCloset AI 데이터 모델 패키지 v7.0 - 단순화된 모델 정의
================================================================

✅ 단순하고 안정적인 모델 초기화
✅ Pydantic v2 완벽 지원
✅ FastAPI 완전 호환
✅ conda 환경 우선 최적화
✅ M3 Max 성능 최적화
✅ 타입 안전성 보장
✅ 실패 허용적 설계

데이터 모델:
- 사용자 모델 (User, UserProfile)
- 이미지 모델 (ImageData, ProcessedImage)
- AI 처리 모델 (PipelineRequest, PipelineResponse)
- 세션 모델 (SessionData, ProcessingStatus)

작성자: MyCloset AI Team
날짜: 2025-07-23
버전: v7.0.0 (Simplified Model Definition)
"""

import logging
import sys
from typing import Dict, Any, Optional, List, Union
from functools import lru_cache
import warnings
from datetime import datetime
from enum import Enum

# 경고 무시
warnings.filterwarnings('ignore')

# =============================================================================
# 🔥 기본 설정 및 시스템 정보
# =============================================================================

logger = logging.getLogger(__name__)

# 상위 패키지에서 시스템 정보 가져오기
try:
    from .. import get_system_info, is_conda_environment, is_m3_max, get_device
    SYSTEM_INFO = get_system_info()
    IS_CONDA = is_conda_environment()
    IS_M3_MAX = is_m3_max()
    DEVICE = get_device()
    logger.info("✅ 상위 패키지에서 시스템 정보 로드 성공")
except ImportError as e:
    logger.warning(f"⚠️ 상위 패키지 로드 실패, 기본값 사용: {e}")
    SYSTEM_INFO = {'device': 'cpu', 'is_m3_max': False, 'memory_gb': 16.0}
    IS_CONDA = False
    IS_M3_MAX = False
    DEVICE = 'cpu'

# =============================================================================
# 🔥 Pydantic 안전한 import
# =============================================================================

try:
    from pydantic import BaseModel, Field, validator, root_validator
    from pydantic import ConfigDict  # Pydantic v2
    PYDANTIC_V2 = True
    PYDANTIC_AVAILABLE = True
    logger.info("✅ Pydantic v2 로드 성공")
except ImportError:
    try:
        from pydantic import BaseModel, Field, validator, root_validator
        from pydantic import Config  # Pydantic v1
        PYDANTIC_V2 = False
        PYDANTIC_AVAILABLE = True
        logger.info("✅ Pydantic v1 로드 성공")
    except ImportError:
        logger.warning("⚠️ Pydantic 없음, 기본 모델 사용")
        PYDANTIC_AVAILABLE = False
        PYDANTIC_V2 = False
        
        # 기본 BaseModel 정의
        class BaseModel:
            def __init__(self, **kwargs):
                for key, value in kwargs.items():
                    setattr(self, key, value)
            
            def dict(self):
                return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
            
            def json(self):
                import json
                return json.dumps(self.dict())

# =============================================================================
# 🔥 기본 Enum 정의
# =============================================================================

class ProcessingStatus(Enum):
    """처리 상태"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class ImageFormat(Enum):
    """이미지 포맷"""
    JPEG = "jpeg"
    PNG = "png"
    WEBP = "webp"
    RGB = "rgb"
    RGBA = "rgba"

class PipelineStep(Enum):
    """파이프라인 단계"""
    HUMAN_PARSING = "human_parsing"
    POSE_ESTIMATION = "pose_estimation"
    CLOTH_SEGMENTATION = "cloth_segmentation"
    GEOMETRIC_MATCHING = "geometric_matching"
    CLOTH_WARPING = "cloth_warping"
    VIRTUAL_FITTING = "virtual_fitting"
    POST_PROCESSING = "post_processing"
    QUALITY_ASSESSMENT = "quality_assessment"

class DeviceType(Enum):
    """디바이스 타입"""
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"
    AUTO = "auto"

# =============================================================================
# 🔥 기본 데이터 모델들
# =============================================================================

if PYDANTIC_AVAILABLE:
    # Pydantic 기반 모델들
    
    class BaseConfig:
        """기본 설정"""
        if PYDANTIC_V2:
            model_config = ConfigDict(
                str_strip_whitespace=True,
                validate_assignment=True,
                use_enum_values=True,
                extra='forbid'
            )
        else:
            class Config:
                str_strip_whitespace = True
                validate_assignment = True
                use_enum_values = True
                extra = 'forbid'
    
    class SystemInfo(BaseModel, BaseConfig):
        """시스템 정보 모델"""
        device: DeviceType = Field(default=DeviceType.CPU, description="사용 디바이스")
        is_conda: bool = Field(default=False, description="conda 환경 여부")
        is_m3_max: bool = Field(default=False, description="M3 Max 여부")
        memory_gb: float = Field(default=16.0, ge=1.0, le=1024.0, description="메모리 크기(GB)")
        cpu_count: int = Field(default=4, ge=1, le=128, description="CPU 코어 수")
        python_version: str = Field(default="3.11.0", description="Python 버전")
        
    class ImageData(BaseModel, BaseConfig):
        """이미지 데이터 모델"""
        filename: str = Field(..., min_length=1, description="파일명")
        format: ImageFormat = Field(default=ImageFormat.JPEG, description="이미지 포맷")
        width: int = Field(..., ge=1, le=8192, description="이미지 너비")
        height: int = Field(..., ge=1, le=8192, description="이미지 높이")
        size_bytes: int = Field(..., ge=1, description="파일 크기(바이트)")
        created_at: datetime = Field(default_factory=datetime.now, description="생성 시간")
        
        if not PYDANTIC_V2:
            @validator('filename')
            def validate_filename(cls, v):
                if not v or len(v.strip()) == 0:
                    raise ValueError('파일명은 필수입니다')
                return v.strip()
    
    class ProcessedImage(BaseModel, BaseConfig):
        """처리된 이미지 모델"""
        original: ImageData = Field(..., description="원본 이미지")
        processed_path: str = Field(..., description="처리된 이미지 경로")
        processing_time: float = Field(..., ge=0, description="처리 시간(초)")
        step: PipelineStep = Field(..., description="처리 단계")
        metadata: Dict[str, Any] = Field(default_factory=dict, description="추가 메타데이터")
        
    class PipelineRequest(BaseModel, BaseConfig):
        """파이프라인 요청 모델"""
        session_id: str = Field(..., min_length=1, description="세션 ID")
        person_image: ImageData = Field(..., description="인물 이미지")
        clothing_image: ImageData = Field(..., description="의류 이미지")
        target_steps: List[PipelineStep] = Field(
            default_factory=lambda: list(PipelineStep),
            description="실행할 단계들"
        )
        config: Dict[str, Any] = Field(default_factory=dict, description="설정")
        priority: int = Field(default=5, ge=1, le=10, description="우선순위")
        
    class PipelineResponse(BaseModel, BaseConfig):
        """파이프라인 응답 모델"""
        session_id: str = Field(..., description="세션 ID")
        status: ProcessingStatus = Field(..., description="처리 상태")
        results: List[ProcessedImage] = Field(default_factory=list, description="처리 결과들")
        error_message: Optional[str] = Field(default=None, description="에러 메시지")
        total_time: float = Field(default=0.0, ge=0, description="전체 처리 시간(초)")
        completed_steps: List[PipelineStep] = Field(default_factory=list, description="완료된 단계들")
        
    class UploadRequest(BaseModel, BaseConfig):
        """파일 업로드 요청 모델"""
        session_id: str = Field(..., min_length=1, description="세션 ID")
        file_type: str = Field(..., description="파일 타입")
        max_size_mb: int = Field(default=10, ge=1, le=100, description="최대 파일 크기(MB)")
        allowed_formats: List[str] = Field(
            default_factory=lambda: ["jpeg", "jpg", "png", "webp"],
            description="허용된 포맷들"
        )
        
    class SessionData(BaseModel, BaseConfig):
        """세션 데이터 모델"""
        session_id: str = Field(..., min_length=1, description="세션 ID")
        created_at: datetime = Field(default_factory=datetime.now, description="생성 시간")
        last_activity: datetime = Field(default_factory=datetime.now, description="마지막 활동")
        status: ProcessingStatus = Field(default=ProcessingStatus.PENDING, description="세션 상태")
        user_agent: Optional[str] = Field(default=None, description="사용자 에이전트")
        ip_address: Optional[str] = Field(default=None, description="IP 주소")
        uploaded_files: List[str] = Field(default_factory=list, description="업로드된 파일들")
        
    class HealthCheckResponse(BaseModel, BaseConfig):
        """헬스 체크 응답 모델"""
        status: str = Field(default="healthy", description="상태")
        timestamp: datetime = Field(default_factory=datetime.now, description="체크 시간")
        system_info: SystemInfo = Field(..., description="시스템 정보")
        version: str = Field(default="v7.0.0", description="API 버전")
        uptime_seconds: float = Field(default=0.0, ge=0, description="업타임(초)")

else:
    # Pydantic 없을 때 기본 모델들
    
    class SystemInfo(BaseModel):
        def __init__(self, **kwargs):
            self.device = kwargs.get('device', 'cpu')
            self.is_conda = kwargs.get('is_conda', False)
            self.is_m3_max = kwargs.get('is_m3_max', False)
            self.memory_gb = kwargs.get('memory_gb', 16.0)
            self.cpu_count = kwargs.get('cpu_count', 4)
            self.python_version = kwargs.get('python_version', '3.11.0')
    
    class ImageData(BaseModel):
        def __init__(self, **kwargs):
            self.filename = kwargs.get('filename', '')
            self.format = kwargs.get('format', 'jpeg')
            self.width = kwargs.get('width', 512)
            self.height = kwargs.get('height', 512)
            self.size_bytes = kwargs.get('size_bytes', 0)
            self.created_at = kwargs.get('created_at', datetime.now())
    
    class ProcessedImage(BaseModel):
        def __init__(self, **kwargs):
            self.original = kwargs.get('original')
            self.processed_path = kwargs.get('processed_path', '')
            self.processing_time = kwargs.get('processing_time', 0.0)
            self.step = kwargs.get('step', 'unknown')
            self.metadata = kwargs.get('metadata', {})
    
    class PipelineRequest(BaseModel):
        def __init__(self, **kwargs):
            self.session_id = kwargs.get('session_id', '')
            self.person_image = kwargs.get('person_image')
            self.clothing_image = kwargs.get('clothing_image')
            self.target_steps = kwargs.get('target_steps', [])
            self.config = kwargs.get('config', {})
            self.priority = kwargs.get('priority', 5)
    
    class PipelineResponse(BaseModel):
        def __init__(self, **kwargs):
            self.session_id = kwargs.get('session_id', '')
            self.status = kwargs.get('status', 'pending')
            self.results = kwargs.get('results', [])
            self.error_message = kwargs.get('error_message')
            self.total_time = kwargs.get('total_time', 0.0)
            self.completed_steps = kwargs.get('completed_steps', [])
    
    class UploadRequest(BaseModel):
        def __init__(self, **kwargs):
            self.session_id = kwargs.get('session_id', '')
            self.file_type = kwargs.get('file_type', 'image')
            self.max_size_mb = kwargs.get('max_size_mb', 10)
            self.allowed_formats = kwargs.get('allowed_formats', ['jpeg', 'jpg', 'png'])
    
    class SessionData(BaseModel):
        def __init__(self, **kwargs):
            self.session_id = kwargs.get('session_id', '')
            self.created_at = kwargs.get('created_at', datetime.now())
            self.last_activity = kwargs.get('last_activity', datetime.now())
            self.status = kwargs.get('status', 'pending')
            self.user_agent = kwargs.get('user_agent')
            self.ip_address = kwargs.get('ip_address')
            self.uploaded_files = kwargs.get('uploaded_files', [])
    
    class HealthCheckResponse(BaseModel):
        def __init__(self, **kwargs):
            self.status = kwargs.get('status', 'healthy')
            self.timestamp = kwargs.get('timestamp', datetime.now())
            self.system_info = kwargs.get('system_info')
            self.version = kwargs.get('version', 'v7.0.0')
            self.uptime_seconds = kwargs.get('uptime_seconds', 0.0)

# =============================================================================
# 🔥 모델 팩토리 함수들
# =============================================================================

@lru_cache(maxsize=1)
def get_system_info_model() -> SystemInfo:
    """시스템 정보 모델 인스턴스 반환"""
    return SystemInfo(
        device=DEVICE,
        is_conda=IS_CONDA,
        is_m3_max=IS_M3_MAX,
        memory_gb=SYSTEM_INFO.get('memory_gb', 16.0),
        cpu_count=SYSTEM_INFO.get('cpu_count', 4),
        python_version=SYSTEM_INFO.get('python_version', '3.11.0')
    )

def create_image_data(filename: str, width: int, height: int, size_bytes: int, **kwargs) -> ImageData:
    """ImageData 모델 인스턴스 생성"""
    return ImageData(
        filename=filename,
        width=width,
        height=height,
        size_bytes=size_bytes,
        format=kwargs.get('format', ImageFormat.JPEG),
        created_at=kwargs.get('created_at', datetime.now())
    )

def create_pipeline_request(session_id: str, person_image: ImageData, clothing_image: ImageData, **kwargs) -> PipelineRequest:
    """PipelineRequest 모델 인스턴스 생성"""
    return PipelineRequest(
        session_id=session_id,
        person_image=person_image,
        clothing_image=clothing_image,
        target_steps=kwargs.get('target_steps', list(PipelineStep)),
        config=kwargs.get('config', {}),
        priority=kwargs.get('priority', 5)
    )

def create_pipeline_response(session_id: str, status: ProcessingStatus, **kwargs) -> PipelineResponse:
    """PipelineResponse 모델 인스턴스 생성"""
    return PipelineResponse(
        session_id=session_id,
        status=status,
        results=kwargs.get('results', []),
        error_message=kwargs.get('error_message'),
        total_time=kwargs.get('total_time', 0.0),
        completed_steps=kwargs.get('completed_steps', [])
    )

def create_health_check_response(**kwargs) -> HealthCheckResponse:
    """HealthCheckResponse 모델 인스턴스 생성"""
    return HealthCheckResponse(
        status=kwargs.get('status', 'healthy'),
        timestamp=datetime.now(),
        system_info=get_system_info_model(),
        version=kwargs.get('version', 'v7.0.0'),
        uptime_seconds=kwargs.get('uptime_seconds', 0.0)
    )

# =============================================================================
# 🔥 모델 검증 함수들
# =============================================================================

def validate_image_data(data: Dict[str, Any]) -> bool:
    """ImageData 검증"""
    try:
        required_fields = ['filename', 'width', 'height', 'size_bytes']
        for field in required_fields:
            if field not in data:
                return False
        
        # 기본 검증
        if not data['filename'] or len(data['filename'].strip()) == 0:
            return False
        if data['width'] <= 0 or data['height'] <= 0:
            return False
        if data['size_bytes'] <= 0:
            return False
        
        return True
    except:
        return False

def validate_pipeline_request(data: Dict[str, Any]) -> bool:
    """PipelineRequest 검증"""
    try:
        required_fields = ['session_id', 'person_image', 'clothing_image']
        for field in required_fields:
            if field not in data:
                return False
        
        # 세션 ID 검증
        if not data['session_id'] or len(data['session_id'].strip()) == 0:
            return False
        
        # 이미지 데이터 검증
        if not validate_image_data(data['person_image']):
            return False
        if not validate_image_data(data['clothing_image']):
            return False
        
        return True
    except:
        return False

# =============================================================================
# 🔥 Export 목록
# =============================================================================

__all__ = [
    # 🎯 Enum들
    'ProcessingStatus',
    'ImageFormat',
    'PipelineStep',
    'DeviceType',
    
    # 🔧 모델 클래스들
    'SystemInfo',
    'ImageData',
    'ProcessedImage',
    'PipelineRequest',
    'PipelineResponse',
    'UploadRequest',
    'SessionData',
    'HealthCheckResponse',
    
    # 🏭 팩토리 함수들
    'get_system_info_model',
    'create_image_data',
    'create_pipeline_request',
    'create_pipeline_response',
    'create_health_check_response',
    
    # 🔍 검증 함수들
    'validate_image_data',
    'validate_pipeline_request',
    
    # 📊 상태 정보
    'PYDANTIC_AVAILABLE',
    'PYDANTIC_V2',
    'SYSTEM_INFO',
    'IS_CONDA',
    'IS_M3_MAX',
    'DEVICE'
]

# =============================================================================
# 🔥 초기화 완료 메시지
# =============================================================================

def _print_initialization_summary():
    """초기화 요약 출력"""
    model_count = len([cls for cls in globals().values() 
                      if isinstance(cls, type) and issubclass(cls, BaseModel)])
    
    print(f"\n🍎 MyCloset AI 데이터 모델 v7.0 초기화 완료!")
    print(f"📋 정의된 모델: {model_count}개")
    print(f"🔧 Pydantic: {'✅' if PYDANTIC_AVAILABLE else '❌'} {'(v2)' if PYDANTIC_V2 else '(v1)' if PYDANTIC_AVAILABLE else ''}")
    print(f"🐍 conda 환경: {'✅' if IS_CONDA else '❌'}")
    print(f"🍎 M3 Max: {'✅' if IS_M3_MAX else '❌'}")
    print(f"🖥️ 디바이스: {DEVICE}")
    
    # 주요 모델들
    main_models = ['SystemInfo', 'ImageData', 'PipelineRequest', 'PipelineResponse']
    available_models = [model for model in main_models if model in globals()]
    print(f"✅ 주요 모델: {', '.join(available_models)}")
    
    print("🚀 데이터 모델 시스템 준비 완료!\n")

# 초기화 상태 출력 (한 번만)
if not hasattr(sys, '_mycloset_models_initialized'):
    _print_initialization_summary()
    sys._mycloset_models_initialized = True

logger.info("🍎 MyCloset AI 데이터 모델 시스템 초기화 완료")