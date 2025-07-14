# app/config.py
"""
MyCloset AI 백엔드 통합 설정 모듈 - M3 Max 128GB 최적화
Pydantic V2 호환, get_settings 함수 포함한 완전한 설정 관리

주요 기능:
- M3 Max 자동 감지 및 최적화
- Pydantic V2 완전 호환
- 환경 변수 지원
- 디렉토리 자동 생성
- 설정 검증
- 싱글톤 패턴
"""

import os
import sys
import time  # 누락된 time 모듈 추가
import platform
import psutil
from pathlib import Path
from typing import List, Optional, Dict, Any, Union
from enum import Enum
from app.config import get_settings, settings  # ✅ 완전한 모듈

# Pydantic V2 import
from pydantic import BaseModel, Field, field_validator, ConfigDict
from pydantic_settings import BaseSettings

# ===============================
# Enum 정의
# ===============================

class DeviceType(str, Enum):
    """디바이스 타입"""
    AUTO = "auto"
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"

class QualityLevel(str, Enum):
    """품질 레벨"""
    FAST = "fast"
    BALANCED = "balanced"
    HIGH = "high"
    ULTRA = "ultra"  # M3 Max 전용

class PrecisionType(str, Enum):
    """정밀도 타입"""
    FP32 = "fp32"
    FP16 = "fp16"
    INT8 = "int8"

class LogLevel(str, Enum):
    """로그 레벨"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

# ===============================
# M3 Max 감지 및 최적화 클래스
# ===============================

class M3MaxDetector:
    """M3 Max 환경 감지 및 최적화 설정"""
    
    def __init__(self):
        self.is_apple_silicon = self._is_apple_silicon()
        self.memory_gb = self._get_memory_gb()
        self.cpu_cores = self._get_cpu_cores()
        self.is_m3_max = self._detect_m3_max()
        
    def _is_apple_silicon(self) -> bool:
        """Apple Silicon 감지"""
        return platform.system() == "Darwin" and platform.machine() == "arm64"
    
    def _get_memory_gb(self) -> float:
        """시스템 메모리 용량(GB) 반환"""
        try:
            return round(psutil.virtual_memory().total / (1024**3), 1)
        except:
            return 8.0  # 기본값
    
    def _get_cpu_cores(self) -> int:
        """CPU 코어 수 반환"""
        try:
            return psutil.cpu_count(logical=False) or 4
        except:
            return 4  # 기본값
    
    def _detect_m3_max(self) -> bool:
        """M3 Max 환경 감지"""
        return (
            self.is_apple_silicon and 
            self.memory_gb >= 120 and  # 128GB 환경
            self.cpu_cores >= 12  # M3 Max는 12코어 이상
        )
    
    def get_optimized_settings(self) -> Dict[str, Any]:
        """M3 Max 최적화된 설정 반환"""
        if not self.is_m3_max:
            return {
                "batch_size": 1,
                "max_workers": 2,
                "concurrent_sessions": 2,
                "memory_pool_gb": 4,
                "cache_size_gb": 2,
                "default_quality": QualityLevel.BALANCED
            }
        
        # M3 Max 128GB 특화 설정
        return {
            "batch_size": 4,
            "max_workers": 6,
            "concurrent_sessions": 8,
            "memory_pool_gb": 32,
            "cache_size_gb": 16,
            "default_quality": QualityLevel.ULTRA,
            "enable_neural_engine": True,
            "enable_mps": True,
            "memory_bandwidth": "400GB/s",
            "high_resolution": True
        }

# ===============================
# 메인 설정 클래스 (Pydantic V2)
# ===============================

class Settings(BaseSettings):
    """M3 Max 최적화된 애플리케이션 설정 클래스"""
    
    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="allow",
        str_strip_whitespace=True,
        validate_assignment=True,
        arbitrary_types_allowed=True
    )
    
    # ===============================
    # 앱 기본 정보
    # ===============================
    APP_NAME: str = Field(
        default="MyCloset AI Backend",
        description="애플리케이션 이름"
    )
    APP_VERSION: str = Field(
        default="3.0.0",
        description="애플리케이션 버전"
    )
    DEBUG: bool = Field(
        default=True,
        description="디버그 모드"
    )
    HOST: str = Field(
        default="0.0.0.0",
        description="호스트 주소"
    )
    PORT: int = Field(
        default=8000,
        ge=1024,
        le=65535,
        description="포트 번호"
    )
    
    # ===============================
    # CORS 설정
    # ===============================
    CORS_ORIGINS: List[str] = Field(
        default=[
            "http://localhost:3000",
            "http://localhost:5173", 
            "http://127.0.0.1:3000",
            "http://127.0.0.1:5173",
            "http://localhost:8080",
            "http://127.0.0.1:8080"
        ],
        description="CORS 허용 도메인"
    )
    CORS_ALLOW_CREDENTIALS: bool = Field(
        default=True,
        description="CORS 인증 허용"
    )
    CORS_ALLOW_METHODS: List[str] = Field(
        default=["*"],
        description="CORS 허용 메서드"
    )
    CORS_ALLOW_HEADERS: List[str] = Field(
        default=["*"],
        description="CORS 허용 헤더"
    )
    
    # ===============================
    # 파일 업로드 설정
    # ===============================
    MAX_UPLOAD_SIZE: int = Field(
        default=100 * 1024 * 1024,  # M3 Max에서 100MB
        gt=0,
        description="최대 업로드 크기 (bytes)"
    )
    ALLOWED_EXTENSIONS: List[str] = Field(
        default=["jpg", "jpeg", "png", "webp", "bmp", "tiff", "heic"],
        description="허용된 이미지 확장자"
    )
    
    # ===============================
    # AI 모델 설정
    # ===============================
    USE_GPU: bool = Field(
        default=True,
        description="GPU 사용 여부"
    )
    DEVICE: DeviceType = Field(
        default=DeviceType.AUTO,
        description="사용할 디바이스"
    )
    MODEL_PRECISION: PrecisionType = Field(
        default=PrecisionType.FP16,
        description="모델 정밀도"
    )
    BATCH_SIZE: int = Field(
        default=1,
        ge=1,
        le=16,
        description="배치 크기"
    )
    MAX_WORKERS: int = Field(
        default=4,
        ge=1,
        le=16,
        description="최대 워커 수"
    )
    
    # ===============================
    # M3 Max 특화 설정
    # ===============================
    M3_MAX_DETECTED: bool = Field(
        default=False,
        description="M3 Max 감지 여부"
    )
    ENABLE_NEURAL_ENGINE: bool = Field(
        default=True,
        description="Neural Engine 사용"
    )
    ENABLE_MPS_OPTIMIZATION: bool = Field(
        default=True,
        description="MPS 최적화 사용"
    )
    MEMORY_POOL_SIZE_GB: int = Field(
        default=16,
        ge=4,
        le=128,
        description="메모리 풀 크기 (GB)"
    )
    CACHE_SIZE_GB: int = Field(
        default=8,
        ge=1,
        le=64,
        description="캐시 크기 (GB)"
    )
    MAX_CONCURRENT_SESSIONS: int = Field(
        default=4,
        ge=1,
        le=32,
        description="최대 동시 세션"
    )
    
    # ===============================
    # 경로 설정 (자동 계산)
    # ===============================
    PROJECT_ROOT: Optional[str] = Field(
        default=None,
        description="프로젝트 루트 경로"
    )
    
    def __init__(self, **kwargs):
        # PROJECT_ROOT가 없으면 자동 설정
        if not kwargs.get('PROJECT_ROOT'):
            kwargs['PROJECT_ROOT'] = str(Path(__file__).parent.parent)
        
        super().__init__(**kwargs)
        
        # M3 Max 감지 및 최적화
        self._detect_and_optimize_m3_max()
        
        # 디렉토리 자동 생성
        self._ensure_directories()
    
    @property
    def STATIC_DIR(self) -> Path:
        """정적 파일 디렉토리"""
        return Path(self.PROJECT_ROOT) / "static"
    
    @property
    def UPLOAD_DIR(self) -> Path:
        """업로드 디렉토리"""
        return self.STATIC_DIR / "uploads"
    
    @property
    def RESULTS_DIR(self) -> Path:
        """결과 디렉토리"""
        return self.STATIC_DIR / "results"
    
    @property
    def MODELS_DIR(self) -> Path:
        """AI 모델 디렉토리"""
        return Path(self.PROJECT_ROOT) / "models" / "ai_models"
    
    @property
    def CACHE_DIR(self) -> Path:
        """캐시 디렉토리"""
        return Path(self.PROJECT_ROOT) / "cache"
    
    @property
    def LOGS_DIR(self) -> Path:
        """로그 디렉토리"""
        return Path(self.PROJECT_ROOT) / "logs"
    
    # ===============================
    # 로깅 설정
    # ===============================
    LOG_LEVEL: LogLevel = Field(
        default=LogLevel.INFO,
        description="로그 레벨"
    )
    LOG_FILE: Optional[str] = Field(
        default=None,
        description="로그 파일 경로"
    )
    LOG_FORMAT: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="로그 포맷"
    )
    LOG_MAX_BYTES: int = Field(
        default=20 * 1024 * 1024,  # 20MB
        description="로그 파일 최대 크기"
    )
    LOG_BACKUP_COUNT: int = Field(
        default=10,
        description="로그 백업 파일 수"
    )
    LOG_TO_CONSOLE: bool = Field(
        default=True,
        description="콘솔 로그 출력"
    )
    LOG_TO_FILE: bool = Field(
        default=True,
        description="파일 로그 출력"
    )
    
    # ===============================
    # 보안 설정
    # ===============================
    SECRET_KEY: str = Field(
        default="mycloset-ai-m3max-secret-key-change-in-production",
        min_length=10,
        description="비밀 키"
    )
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(
        default=60,
        ge=1,
        description="액세스 토큰 만료 시간"
    )
    
    # ===============================
    # API 설정
    # ===============================
    API_V1_STR: str = Field(
        default="/api/v1",
        description="API v1 경로"
    )
    API_TIMEOUT: int = Field(
        default=600,  # M3 Max에서 10분
        ge=60,
        description="API 타임아웃 (초)"
    )
    ENABLE_API_DOCS: bool = Field(
        default=True,
        description="API 문서 활성화"
    )
    
    # ===============================
    # 품질 설정
    # ===============================
    DEFAULT_QUALITY: QualityLevel = Field(
        default=QualityLevel.HIGH,
        description="기본 품질 레벨"
    )
    QUALITY_THRESHOLDS: Dict[str, float] = Field(
        default={
            "fast": 0.6,
            "balanced": 0.7,
            "high": 0.8,
            "ultra": 0.9
        },
        description="품질 임계값"
    )
    
    # ===============================
    # AI 파이프라인 설정
    # ===============================
    PIPELINE_STEPS: List[str] = Field(
        default=[
            "human_parsing",
            "pose_estimation",
            "cloth_segmentation", 
            "geometric_matching",
            "cloth_warping",
            "virtual_fitting",
            "post_processing",
            "quality_assessment"
        ],
        description="파이프라인 단계"
    )
    
    # ===============================
    # WebSocket 설정
    # ===============================
    WEBSOCKET_MAX_CONNECTIONS: int = Field(
        default=100,
        ge=1,
        description="WebSocket 최대 연결 수"
    )
    WEBSOCKET_HEARTBEAT_INTERVAL: int = Field(
        default=30,
        ge=5,
        description="WebSocket 하트비트 간격"
    )
    
    # ===============================
    # 유효성 검증 (Pydantic V2)
    # ===============================
    
    @field_validator('PORT')
    @classmethod
    def validate_port(cls, v: int) -> int:
        """포트 번호 유효성 검증"""
        if not (1024 <= v <= 65535):
            raise ValueError('PORT must be between 1024 and 65535')
        return v
    
    @field_validator('BATCH_SIZE')
    @classmethod
    def validate_batch_size(cls, v: int) -> int:
        """배치 크기 유효성 검증"""
        if v <= 0:
            raise ValueError('BATCH_SIZE must be positive')
        return v
    
    @field_validator('MAX_UPLOAD_SIZE')
    @classmethod
    def validate_upload_size(cls, v: int) -> int:
        """업로드 크기 유효성 검증"""
        if v <= 0:
            raise ValueError('MAX_UPLOAD_SIZE must be positive')
        if v > 500 * 1024 * 1024:  # 500MB 제한
            raise ValueError('MAX_UPLOAD_SIZE cannot exceed 500MB')
        return v
    
    @field_validator('CORS_ORIGINS')
    @classmethod
    def validate_cors_origins(cls, v: List[str]) -> List[str]:
        """CORS 도메인 유효성 검증"""
        if not v:
            raise ValueError('CORS_ORIGINS cannot be empty')
        return v
    
    # ===============================
    # M3 Max 최적화 메서드들
    # ===============================
    
    def _detect_and_optimize_m3_max(self):
        """M3 Max 감지 및 최적화"""
        detector = M3MaxDetector()
        self.M3_MAX_DETECTED = detector.is_m3_max
        
        if self.M3_MAX_DETECTED:
            # M3 Max 최적화 설정 적용
            optimized = detector.get_optimized_settings()
            
            self.BATCH_SIZE = optimized["batch_size"]
            self.MAX_WORKERS = optimized["max_workers"]
            self.MAX_CONCURRENT_SESSIONS = optimized["concurrent_sessions"]
            self.MEMORY_POOL_SIZE_GB = optimized["memory_pool_gb"]
            self.CACHE_SIZE_GB = optimized["cache_size_gb"]
            self.DEFAULT_QUALITY = optimized["default_quality"]
            
            # M3 Max 전용 설정 업데이트
            self.ENABLE_NEURAL_ENGINE = optimized.get("enable_neural_engine", True)
            self.ENABLE_MPS_OPTIMIZATION = optimized.get("enable_mps", True)
            
            # 업로드 크기 증가 (M3 Max에서)
            if self.MAX_UPLOAD_SIZE < 100 * 1024 * 1024:
                self.MAX_UPLOAD_SIZE = 100 * 1024 * 1024
            
            print("🍎 M3 Max 128GB 환경 감지 - 최적화 모드 활성화")
        else:
            print(f"💻 일반 환경 감지: {detector.memory_gb}GB, {detector.cpu_cores}코어")
    
    def _ensure_directories(self):
        """필요한 디렉토리 생성"""
        directories = [
            self.STATIC_DIR,
            self.UPLOAD_DIR,
            self.RESULTS_DIR,
            self.MODELS_DIR,
            self.CACHE_DIR,
            self.LOGS_DIR
        ]
        
        created_count = 0
        for directory in directories:
            if not directory.exists():
                directory.mkdir(parents=True, exist_ok=True)
                created_count += 1
        
        if created_count > 0:
            print(f"📁 필요한 디렉토리 생성 완료: {created_count}개")
    
    # ===============================
    # 설정 정보 메서드들
    # ===============================
    
    def get_m3_max_config(self) -> Dict[str, Any]:
        """M3 Max 최적화 설정 반환"""
        if not self.M3_MAX_DETECTED:
            return {}
        
        return {
            "neural_engine_enabled": self.ENABLE_NEURAL_ENGINE,
            "mps_optimization": self.ENABLE_MPS_OPTIMIZATION,
            "batch_size": self.BATCH_SIZE,
            "memory_pool_gb": self.MEMORY_POOL_SIZE_GB,
            "cache_size_gb": self.CACHE_SIZE_GB,
            "max_concurrent_sessions": self.MAX_CONCURRENT_SESSIONS,
            "quality_level": self.DEFAULT_QUALITY,
            "high_resolution": True,
            "metal_performance_shaders": True
        }
    
    def is_m3_max_optimized(self) -> bool:
        """M3 Max 최적화 여부 반환"""
        return self.M3_MAX_DETECTED
    
    def get_device_info(self) -> Dict[str, Any]:
        """디바이스 정보 반환"""
        detector = M3MaxDetector()
        return {
            "platform": platform.system(),
            "machine": platform.machine(), 
            "is_apple_silicon": detector.is_apple_silicon,
            "is_m3_max": detector.is_m3_max,
            "memory_gb": detector.memory_gb,
            "cpu_cores": detector.cpu_cores,
            "recommended_device": "mps" if detector.is_apple_silicon else "cpu"
        }
    
    def print_config_summary(self):
        """설정 요약 출력"""
        print("\n" + "="*50)
        print("🔧 MyCloset AI 설정 정보")
        print("="*50)
        print(f"앱 이름: {self.APP_NAME}")
        print(f"버전: {self.APP_VERSION}")
        print(f"호스트: {self.HOST}:{self.PORT}")
        print(f"M3 Max 최적화: {'🍎 활성화' if self.M3_MAX_DETECTED else '❌ 비활성화'}")
        print(f"디버그 모드: {self.DEBUG}")
        
        if self.M3_MAX_DETECTED:
            print(f"\n🍎 M3 Max 최적화 상태:")
            print(f"  - 배치 크기: {self.BATCH_SIZE}")
            print(f"  - 최대 워커: {self.MAX_WORKERS}")
            print(f"  - 동시 세션: {self.MAX_CONCURRENT_SESSIONS}")
            print(f"  - 메모리 풀: {self.MEMORY_POOL_SIZE_GB}GB")
            print(f"  - 캐시 크기: {self.CACHE_SIZE_GB}GB")
            print(f"  - 기본 품질: {self.DEFAULT_QUALITY}")
        
        print(f"\n📁 디렉토리:")
        print(f"  - 프로젝트 루트: {self.PROJECT_ROOT}")
        print(f"  - 업로드: {self.UPLOAD_DIR}")
        print(f"  - 결과: {self.RESULTS_DIR}")
        print(f"  - 로그: {self.LOGS_DIR}")
        print("="*50)

# ===============================
# 싱글톤 설정 관리
# ===============================

_settings: Optional[Settings] = None

def get_settings() -> Settings:
    """
    설정 인스턴스를 반환하는 함수 (싱글톤 패턴)
    
    Returns:
        Settings: 애플리케이션 설정 인스턴스
    """
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings

def reload_settings() -> Settings:
    """설정 강제 재로드"""
    global _settings
    _settings = None
    return get_settings()

# ===============================
# 편의 함수들
# ===============================

def is_m3_max() -> bool:
    """M3 Max 환경 여부 확인"""
    return get_settings().M3_MAX_DETECTED

def get_device_type() -> str:
    """추천 디바이스 타입 반환"""
    settings = get_settings()
    if settings.DEVICE != DeviceType.AUTO:
        return settings.DEVICE.value
    
    device_info = settings.get_device_info()
    return device_info["recommended_device"]

def get_optimal_batch_size() -> int:
    """최적 배치 크기 반환"""
    return get_settings().BATCH_SIZE

def get_memory_config() -> Dict[str, int]:
    """메모리 설정 반환"""
    settings = get_settings()
    return {
        "memory_pool_gb": settings.MEMORY_POOL_SIZE_GB,
        "cache_size_gb": settings.CACHE_SIZE_GB,
        "max_upload_mb": settings.MAX_UPLOAD_SIZE // (1024 * 1024)
    }

# ===============================
# WebSocket 설정
# ===============================

WEBSOCKET_CONFIG = {
    "ping_interval": 20,
    "ping_timeout": 10,
    "close_timeout": 10,
    "max_size": 10 * 1024 * 1024,  # 10MB
    "max_queue": 32,
    "read_limit": 2 ** 16,
    "write_limit": 2 ** 16,
}

# ===============================
# 기존 코드 호환성
# ===============================

# 기존 코드와의 호환성을 위한 전역 settings 객체
settings = get_settings()

# ===============================
# 모듈 초기화
# ===============================

def _initialize_module():
    """모듈 초기화 (import 시 자동 실행)"""
    global settings
    settings = get_settings()
    
    # 환경 변수 설정
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")
    
    # 디버그 모드에서만 설정 정보 출력
    if settings.DEBUG:
        print("✅ 설정 모듈 초기화 완료")

# 모듈 로드 시 자동 실행
_initialize_module()

# ===============================
# 개발용 테스트 코드
# ===============================

if __name__ == "__main__":
    # 설정 테스트 및 정보 출력
    test_settings = get_settings()
    test_settings.print_config_summary()
    
    print(f"\n🔧 디바이스 정보:")
    device_info = test_settings.get_device_info()
    for key, value in device_info.items():
        print(f"  - {key}: {value}")
    
    if test_settings.M3_MAX_DETECTED:
        print(f"\n🍎 M3 Max 특화 설정:")
        m3_config = test_settings.get_m3_max_config()
        for key, value in m3_config.items():
            print(f"  - {key}: {value}")
    
    print(f"\n📊 메모리 설정:")
    memory_config = get_memory_config()
    for key, value in memory_config.items():
        print(f"  - {key}: {value}")
    
    print(f"\n✅ 설정 검증 완료 - 총 {len(test_settings.PIPELINE_STEPS)}개 파이프라인 단계")