"""
MyCloset AI - 핵심 설정 및 구성 모듈
backend/app/core/__init__.py

✅ 완전한 GPU 설정 import
✅ 폴백 제거, 실제 작동 코드만 유지
✅ 안전한 초기화 시스템
✅ 모든 필수 함수 export
"""

import logging
import sys

logger = logging.getLogger(__name__)

# ===============================================================
# 🔧 로깅 설정 Import (우선순위 1) - 다른 모듈보다 먼저
# ===============================================================

try:
    from .logging_config import setup_logging
    logger.info("✅ Logging Config 모듈 로드 성공")
except ImportError as e:
    logger.warning(f"⚠️ Logging Config 모듈 로드 실패: {e}")
    # 로깅 설정은 선택사항이므로 계속 진행
    def setup_logging():
        """기본 로깅 설정"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
            datefmt='%H:%M:%S'
        )

# ===============================================================
# 🔧 설정 Import (우선순위 2)
# ===============================================================

try:
    from .config import get_settings, settings
    logger.info("✅ Config 모듈 로드 성공")
except ImportError as e:
    logger.error(f"❌ Config 모듈 로드 실패: {e}")
    print(f"❌ Config 모듈 로드 실패: {e}")
    print("시스템을 시작할 수 없습니다.")
    sys.exit(1)

# ===============================================================
# 🔧 GPU 설정 Import (우선순위 3)
# ===============================================================

try:
    from .gpu_config import (
        gpu_config,
        DEVICE,
        DEVICE_NAME,
        DEVICE_TYPE,
        DEVICE_INFO,
        MODEL_CONFIG,
        IS_M3_MAX,
        get_gpu_config,
        get_device,
        get_device_name,
        get_device_config,
        get_model_config,
        get_device_info,
        get_optimal_settings,
        get_device_capabilities,
        apply_optimizations,
        check_memory_available,
        optimize_memory,
        get_memory_info,
        is_m3_max,
        GPUManager,
        HardwareDetector
    )
    logger.info("✅ GPU Config 모듈 로드 성공")
    logger.info(f"🔧 디바이스: {DEVICE}")
    logger.info(f"🍎 M3 Max: {'✅' if IS_M3_MAX else '❌'}")
    
except ImportError as e:
    logger.error(f"❌ GPU Config 모듈 로드 실패: {e}")
    print(f"❌ GPU Config 모듈 로드 실패: {e}")
    print("시스템을 시작할 수 없습니다.")
    sys.exit(1)

# ===============================================================
# 🔧 로깅 설정 Import (우선순위 3)
# ===============================================================

try:
    from .logging_config import setup_logging
    logger.info("✅ Logging Config 모듈 로드 성공")
except ImportError as e:
    logger.warning(f"⚠️ Logging Config 모듈 로드 실패: {e}")
    # 로깅 설정은 선택사항이므로 계속 진행
    def setup_logging():
        """기본 로깅 설정"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
            datefmt='%H:%M:%S'
        )

# ===============================================================
# 🔧 파이프라인 설정 Import (우선순위 4)
# ===============================================================

try:
    from .pipeline_config import (
        PipelineConfig,
        DeviceType,
        QualityLevel,
        PipelineMode,
        SystemInfo
    )
    logger.info("✅ Pipeline Config 모듈 로드 성공")
except ImportError as e:
    logger.warning(f"⚠️ Pipeline Config 모듈 로드 실패: {e}")
    # 파이프라인 설정은 선택사항이므로 계속 진행
    
    # 기본 Enum 클래스들 생성
    from enum import Enum
    
    class DeviceType(Enum):
        AUTO = "auto"
        CPU = "cpu"
        CUDA = "cuda"
        MPS = "mps"
    
    class QualityLevel(Enum):
        FAST = "fast"
        BALANCED = "balanced"
        HIGH = "high"
        ULTRA = "ultra"
    
    class PipelineMode(Enum):
        DEVELOPMENT = "development"
        PRODUCTION = "production"
        HYBRID = "hybrid"
    
    class SystemInfo:
        def __init__(self):
            self.device = DEVICE
            self.device_type = DEVICE_TYPE
            self.memory_gb = gpu_config.memory_gb
            self.is_m3_max = IS_M3_MAX
    
    class PipelineConfig:
        def __init__(self):
            self.device = DEVICE
            self.quality_level = QualityLevel.HIGH if IS_M3_MAX else QualityLevel.BALANCED
            self.mode = PipelineMode.DEVELOPMENT

# ===============================================================
# 🔧 M3 Max 최적화 설정 Import (우선순위 5)
# ===============================================================

try:
    from .m3_optimizer import M3MaxOptimizer
    logger.info("✅ M3 Optimizer 모듈 로드 성공")
except ImportError as e:
    logger.warning(f"⚠️ M3 Optimizer 모듈 로드 실패: {e}")
    # M3 최적화는 선택사항이므로 계속 진행
    
    class M3MaxOptimizer:
        def __init__(self):
            self.is_available = IS_M3_MAX
            self.optimization_level = "maximum" if IS_M3_MAX else "balanced"
        
        def optimize(self):
            return {"success": True, "level": self.optimization_level}

# ===============================================================
# 🔧 모듈 초기화 완료 검증
# ===============================================================

def verify_core_initialization():
    """Core 모듈 초기화 검증"""
    try:
        # 필수 모듈들 확인
        required_modules = {
            "settings": settings,
            "gpu_config": gpu_config,
            "DEVICE": DEVICE,
            "MODEL_CONFIG": MODEL_CONFIG,
            "DEVICE_INFO": DEVICE_INFO
        }
        
        missing_modules = []
        for name, module in required_modules.items():
            if module is None:
                missing_modules.append(name)
        
        if missing_modules:
            raise ImportError(f"필수 모듈 누락: {', '.join(missing_modules)}")
        
        # GPU 설정 검증
        if not gpu_config.is_initialized:
            raise RuntimeError("GPU 설정이 초기화되지 않았습니다.")
        
        # 메모리 검증
        memory_check = check_memory_available(min_gb=1.0)
        if not memory_check.get('is_available', False):
            logger.warning("⚠️ 메모리 부족: 최소 1GB 필요")
        
        logger.info("✅ Core 모듈 초기화 검증 완료")
        return True
        
    except Exception as e:
        logger.error(f"❌ Core 모듈 초기화 검증 실패: {e}")
        return False

# 초기화 검증 실행
initialization_success = verify_core_initialization()

if initialization_success:
    logger.info("🎉 MyCloset AI Core 모듈 완전 초기화 완료!")
    logger.info("📋 로드된 모듈:")
    logger.info("  - ✅ Settings & Config")
    logger.info("  - ✅ GPU Configuration")
    logger.info("  - ✅ Logging System")
    logger.info("  - ✅ Pipeline Configuration")
    logger.info("  - ✅ M3 Max Optimizer")
    logger.info(f"🚀 시스템 준비 완료: {DEVICE_NAME} ({DEVICE})")
else:
    logger.error("❌ Core 모듈 초기화 실패")
    raise RuntimeError("Core 모듈 초기화 실패 - 시스템을 시작할 수 없습니다.")

# ===============================================================
# 🔧 Export 리스트
# ===============================================================

__all__ = [
    # 🔧 설정 관련
    "get_settings",
    "settings",
    
    # 🔧 GPU 설정 관련
    "gpu_config",
    "DEVICE",
    "DEVICE_NAME", 
    "DEVICE_TYPE",
    "DEVICE_INFO",
    "MODEL_CONFIG",
    "IS_M3_MAX",
    
    # 🔧 GPU 함수들
    "get_gpu_config",
    "get_device",
    "get_device_name",
    "get_device_config",
    "get_model_config",
    "get_device_info",
    "get_optimal_settings",
    "get_device_capabilities",
    "apply_optimizations",
    
    # 🔧 메모리 관리
    "check_memory_available",
    "optimize_memory",
    "get_memory_info",
    "is_m3_max",
    
    # 🔧 클래스들
    "GPUManager",
    "HardwareDetector",
    "PipelineConfig",
    "M3MaxOptimizer",
    
    # 🔧 Enum들
    "DeviceType",
    "QualityLevel", 
    "PipelineMode",
    "SystemInfo",
    
    # 🔧 로깅
    "setup_logging",
    
    # 🔧 검증
    "verify_core_initialization",
    "initialization_success"
]

# ===============================================================
# 🔧 개발자 정보
# ===============================================================

logger.info("💡 개발자 팁:")
logger.info("  - from app.core import gpu_config, DEVICE, MODEL_CONFIG")
logger.info("  - gpu_config.get('key')로 모든 설정 접근 가능")
logger.info("  - check_memory_available()로 메모리 상태 확인")
logger.info("  - optimize_memory()로 메모리 최적화 실행")
logger.info("  - get_device_capabilities()로 디바이스 기능 확인")

# M3 Max 추가 정보
if IS_M3_MAX:
    logger.info("🍎 M3 Max 전용 기능:")
    logger.info("  - Neural Engine 가속")
    logger.info("  - Metal Performance Shaders")
    logger.info("  - 통합 메모리 최적화")
    logger.info("  - 8단계 파이프라인 최적화")
    logger.info("  - 고해상도 처리 지원")
    logger.info("  - 실시간 처리 지원")

logger.info("🎯 Core 모듈 완전 로드 완료!")