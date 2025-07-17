# app/ai_pipeline/utils/__init__.py
"""
🍎 MyCloset AI 파이프라인 유틸리티 v4.0 - 최적화된 통합 시스템
✅ 순환참조 완전 방지 + 단방향 의존성 구조
✅ Step 클래스 완벽 지원 + 간단한 인터페이스
✅ 실제 AI 모델 자동 탐지 + 로딩 + 추론
✅ M3 Max 128GB 최적화 + Neural Engine 활용
✅ 프로덕션 안정성 + 확장성 보장

의존성 흐름:
step_model_requests → auto_model_detector → model_loader → __init__ → Step 클래스들
"""

import os
import sys
import time
import logging
import threading
from typing import Dict, Any, Optional, List, Callable
from pathlib import Path
from functools import lru_cache

# 기본 로깅 설정
logger = logging.getLogger(__name__)

# ==============================================
# 🔥 시스템 정보 감지 (최적화됨)
# ==============================================

@lru_cache(maxsize=1)
def _get_system_info() -> Dict[str, Any]:
    """시스템 정보 캐시 (한번만 실행)"""
    try:
        import platform
        
        system_info = {
            "platform": platform.system(),
            "machine": platform.machine(),
            "cpu_count": os.cpu_count() or 4,
            "python_version": ".".join(map(str, sys.version_info[:3]))
        }
        
        # M3 Max 감지
        is_m3_max = False
        if platform.system() == 'Darwin' and platform.machine() == 'arm64':
            try:
                import subprocess
                result = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'], 
                                      capture_output=True, text=True, timeout=3)
                is_m3_max = 'M3' in result.stdout
            except:
                pass
        
        system_info["is_m3_max"] = is_m3_max
        
        # 메모리 정보
        try:
            import psutil
            system_info["memory_gb"] = round(psutil.virtual_memory().total / (1024**3))
        except ImportError:
            system_info["memory_gb"] = 16
        
        # 디바이스 감지
        device = "cpu"
        try:
            import torch
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
        except ImportError:
            pass
        
        system_info["device"] = device
        
        return system_info
        
    except Exception as e:
        logger.warning(f"시스템 정보 감지 실패: {e}")
        return {
            "platform": "unknown",
            "is_m3_max": False,
            "device": "cpu",
            "cpu_count": 4,
            "memory_gb": 16,
            "python_version": "3.8.0"
        }

# 시스템 정보 전역 변수
SYSTEM_INFO = _get_system_info()
IS_M3_MAX = SYSTEM_INFO["is_m3_max"]
DEVICE = SYSTEM_INFO["device"]
MEMORY_GB = SYSTEM_INFO["memory_gb"]

# ==============================================
# 🔥 핵심 모듈 안전 Import (단방향 의존성)
# ==============================================

# 1. Step 모델 요청 정의 (최하위 - 의존성 없음)
try:
    from .step_model_requests import (
        STEP_MODEL_REQUESTS,
        ModelRequest,
        StepPriority,
        get_step_request,
        get_all_step_requests,
        get_checkpoint_patterns,
        get_model_config_for_step,
        validate_model_for_step,
        get_step_priorities,
        get_steps_by_priority
    )
    STEP_REQUESTS_AVAILABLE = True
    logger.info("✅ Step Model Requests 로드 성공")
except ImportError as e:
    STEP_REQUESTS_AVAILABLE = False
    logger.warning(f"⚠️ Step Model Requests 로드 실패: {e}")

# 2. 자동 모델 탐지 시스템 (step_model_requests 의존)
try:
    from .auto_model_detector import (
        AutoModelDetector,
        DetectedModel,
        DetectionStatus,
        quick_detect_models,
        detect_and_export_for_loader,
        validate_detected_models
    )
    AUTO_DETECTOR_AVAILABLE = True
    logger.info("✅ Auto Model Detector 로드 성공")
except ImportError as e:
    AUTO_DETECTOR_AVAILABLE = False
    logger.warning(f"⚠️ Auto Model Detector 로드 실패: {e}")

# 3. 모델 로더 시스템 (위 두 모듈 의존)
try:
    from .model_loader import (
        ModelLoader,
        ModelConfig,
        ModelType,
        ModelFormat,
        LoadedModel,
        StepModelInterface,
        BaseStepMixin,
        # AI 모델 클래스들
        BaseModel,
        GraphonomyModel,
        OpenPoseModel,
        U2NetModel,
        GeometricMatchingModel,
        HRVITONModel,
        # 유틸리티 함수들
        preprocess_image,
        postprocess_segmentation,
        # 전역 함수들
        get_global_model_loader,
        initialize_global_model_loader,
        cleanup_global_loader
    )
    MODEL_LOADER_AVAILABLE = True
    logger.info("✅ Model Loader 로드 성공")
except ImportError as e:
    MODEL_LOADER_AVAILABLE = False
    logger.warning(f"⚠️ Model Loader 로드 실패: {e}")

# ==============================================
# 🔥 통합 관리 시스템
# ==============================================

class PipelineUtils:
    """
    🍎 파이프라인 유틸리티 통합 관리자
    Step 클래스에서 사용하는 모든 기능을 통합 제공
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self.logger = logging.getLogger(f"{__name__}.PipelineUtils")
        
        # 상태 관리
        self.is_initialized = False
        self.initialization_time = None
        self.model_loader = None
        
        # 통계
        self.stats = {
            "total_models_detected": 0,
            "models_loaded": 0,
            "step_interfaces_created": 0,
            "total_requests": 0
        }
        
        self._initialized = True
        
        self.logger.info("🎯 PipelineUtils 인스턴스 생성")
    
    async def initialize(self, **kwargs) -> Dict[str, Any]:
        """통합 초기화"""
        if self.is_initialized:
            return {"success": True, "message": "Already initialized"}
        
        try:
            start_time = time.time()
            self.logger.info("🚀 PipelineUtils 초기화 시작...")
            
            results = {
                "step_requests": STEP_REQUESTS_AVAILABLE,
                "auto_detector": AUTO_DETECTOR_AVAILABLE,
                "model_loader": False,
                "auto_detection_count": 0,
                "errors": []
            }
            
            # 1. ModelLoader 초기화
            if MODEL_LOADER_AVAILABLE:
                try:
                    init_result = initialize_global_model_loader(**kwargs)
                    if init_result.get("success"):
                        self.model_loader = get_global_model_loader()
                        results["model_loader"] = True
                        self.logger.info("✅ ModelLoader 초기화 성공")
                    else:
                        results["errors"].append(f"ModelLoader: {init_result.get('error', 'Unknown')}")
                except Exception as e:
                    results["errors"].append(f"ModelLoader: {e}")
                    self.logger.error(f"❌ ModelLoader 초기화 실패: {e}")
            
            # 2. 자동 모델 탐지 (선택적)
            if AUTO_DETECTOR_AVAILABLE and kwargs.get("auto_detect", True):
                try:
                    detected_models = quick_detect_models(min_confidence=0.7)
                    results["auto_detection_count"] = len(detected_models)
                    self.stats["total_models_detected"] = len(detected_models)
                    
                    if detected_models:
                        self.logger.info(f"🔍 자동 탐지 완료: {len(detected_models)}개 모델")
                    else:
                        self.logger.info("🔍 자동 탐지 완료: 탐지된 모델 없음")
                        
                except Exception as e:
                    results["errors"].append(f"AutoDetector: {e}")
                    self.logger.warning(f"⚠️ 자동 탐지 실패: {e}")
            
            # 3. 폴백 ModelLoader 생성 (실패 시)
            if not self.model_loader:
                try:
                    self.model_loader = get_global_model_loader()
                    self.logger.info("✅ 폴백 ModelLoader 생성")
                except Exception as e:
                    results["errors"].append(f"Fallback ModelLoader: {e}")
                    self.logger.error(f"❌ 폴백 ModelLoader 생성 실패: {e}")
            
            # 초기화 완료
            self.is_initialized = True
            self.initialization_time = time.time() - start_time
            
            success_count = sum([
                results["step_requests"],
                results["auto_detector"],
                results["model_loader"]
            ])
            
            self.logger.info(f"🎉 PipelineUtils 초기화 완료 ({self.initialization_time:.2f}s)")
            self.logger.info(f"📊 사용 가능한 모듈: {success_count}/3")
            
            return {
                "success": True,
                "initialization_time": self.initialization_time,
                "system_info": SYSTEM_INFO,
                "modules": results,
                "stats": self.stats
            }
            
        except Exception as e:
            self.logger.error(f"❌ PipelineUtils 초기화 실패: {e}")
            return {
                "success": False,
                "error": str(e),
                "system_info": SYSTEM_INFO
            }
    
    def create_step_interface(self, step_name: str) -> Dict[str, Any]:
        """
        🔥 Step 클래스용 통합 인터페이스 생성
        모든 Step 클래스에서 사용하는 핵심 함수
        """
        try:
            self.stats["step_interfaces_created"] += 1
            
            # 기본 인터페이스
            interface = {
                "step_name": step_name,
                "system_info": SYSTEM_INFO,
                "logger": logging.getLogger(f"steps.{step_name}"),
                "initialized": self.is_initialized
            }
            
            # ModelLoader 인터페이스
            if self.model_loader:
                try:
                    model_interface = self.model_loader.create_step_interface(step_name)
                    interface["model_interface"] = model_interface
                    interface["get_model"] = self._create_get_model_func(model_interface)
                    interface["has_model_loader"] = True
                except Exception as e:
                    interface["model_loader_error"] = str(e)
                    interface["has_model_loader"] = False
            else:
                interface["has_model_loader"] = False
            
            # Step 요청 정보
            if STEP_REQUESTS_AVAILABLE:
                step_request = get_step_request(step_name)
                if step_request:
                    interface["step_request"] = step_request
                    interface["recommended_model"] = step_request.model_name
                    interface["input_size"] = step_request.input_size
                    interface["num_classes"] = step_request.num_classes
                    interface["optimization_params"] = step_request.optimization_params
            
            # 유틸리티 함수들
            interface["preprocess_image"] = self._create_preprocess_func()
            interface["postprocess_output"] = self._create_postprocess_func()
            interface["optimize_memory"] = self._create_memory_func()
            
            # 메타데이터
            interface["metadata"] = {
                "creation_time": time.time(),
                "available_modules": {
                    "step_requests": STEP_REQUESTS_AVAILABLE,
                    "auto_detector": AUTO_DETECTOR_AVAILABLE,
                    "model_loader": MODEL_LOADER_AVAILABLE
                }
            }
            
            self.logger.info(f"🔗 {step_name} 인터페이스 생성 완료")
            return interface
            
        except Exception as e:
            self.logger.error(f"❌ {step_name} 인터페이스 생성 실패: {e}")
            return {
                "step_name": step_name,
                "error": str(e),
                "system_info": SYSTEM_INFO,
                "logger": logging.getLogger(f"steps.{step_name}")
            }
    
    def _create_get_model_func(self, model_interface: Any) -> Callable:
        """모델 로드 함수 생성"""
        async def get_model(model_name: Optional[str] = None):
            try:
                self.stats["total_requests"] += 1
                return await model_interface.get_model(model_name)
            except Exception as e:
                self.logger.error(f"모델 로드 실패: {e}")
                return None
        return get_model
    
    def _create_preprocess_func(self) -> Callable:
        """이미지 전처리 함수 생성"""
        def preprocess_func(image, target_size=(512, 512), **kwargs):
            try:
                if MODEL_LOADER_AVAILABLE:
                    return preprocess_image(image, target_size, **kwargs)
                else:
                    self.logger.warning("ModelLoader 없음: 전처리 시뮬레이션")
                    return None
            except Exception as e:
                self.logger.error(f"이미지 전처리 실패: {e}")
                return None
        return preprocess_func
    
    def _create_postprocess_func(self) -> Callable:
        """후처리 함수 생성"""
        def postprocess_func(output, output_type="segmentation", **kwargs):
            try:
                if MODEL_LOADER_AVAILABLE:
                    if output_type == "segmentation":
                        return postprocess_segmentation(output, **kwargs)
                    # 다른 타입들 추가 가능
                    return output
                else:
                    self.logger.warning("ModelLoader 없음: 후처리 시뮬레이션")
                    return None
            except Exception as e:
                self.logger.error(f"후처리 실패: {e}")
                return None
        return postprocess_func
    
    def _create_memory_func(self) -> Callable:
        """메모리 최적화 함수 생성"""
        def optimize_memory():
            try:
                if self.model_loader and hasattr(self.model_loader, 'memory_manager'):
                    self.model_loader.memory_manager.cleanup_memory()
                    return {"success": True}
                else:
                    import gc
                    gc.collect()
                    return {"success": True, "message": "Basic cleanup"}
            except Exception as e:
                self.logger.error(f"메모리 최적화 실패: {e}")
                return {"success": False, "error": str(e)}
        return optimize_memory
    
    def get_status(self) -> Dict[str, Any]:
        """시스템 상태 조회"""
        try:
            status = {
                "initialized": self.is_initialized,
                "initialization_time": self.initialization_time,
                "system_info": SYSTEM_INFO,
                "modules": {
                    "step_requests": STEP_REQUESTS_AVAILABLE,
                    "auto_detector": AUTO_DETECTOR_AVAILABLE,
                    "model_loader": MODEL_LOADER_AVAILABLE
                },
                "stats": self.stats
            }
            
            # ModelLoader 상태
            if self.model_loader:
                try:
                    status["model_loader_info"] = self.model_loader.get_system_info()
                except Exception as e:
                    status["model_loader_error"] = str(e)
            
            return status
            
        except Exception as e:
            return {"error": str(e), "system_info": SYSTEM_INFO}
    
    def cleanup(self):
        """리소스 정리"""
        try:
            if self.model_loader:
                self.model_loader.cleanup()
            
            self.is_initialized = False
            self.logger.info("✅ PipelineUtils 정리 완료")
            
        except Exception as e:
            self.logger.error(f"❌ PipelineUtils 정리 실패: {e}")

# ==============================================
# 🔥 전역 인스턴스 관리
# ==============================================

_global_utils: Optional[PipelineUtils] = None
_utils_lock = threading.Lock()

def get_pipeline_utils() -> PipelineUtils:
    """전역 PipelineUtils 인스턴스 반환"""
    global _global_utils
    
    with _utils_lock:
        if _global_utils is None:
            _global_utils = PipelineUtils()
        return _global_utils

def initialize_pipeline_utils(**kwargs) -> Dict[str, Any]:
    """
    🔥 파이프라인 유틸리티 초기화
    main.py에서 호출하는 진입점
    """
    try:
        utils = get_pipeline_utils()
        
        # 비동기 초기화
        import asyncio
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        if loop.is_running():
            # 실행 중인 루프에서는 태스크 생성
            future = asyncio.create_task(utils.initialize(**kwargs))
            return {"success": True, "message": "Initialization started", "future": future}
        else:
            # 새 루프에서 실행
            result = loop.run_until_complete(utils.initialize(**kwargs))
            return result
            
    except Exception as e:
        logger.error(f"❌ 파이프라인 유틸리티 초기화 실패: {e}")
        return {"success": False, "error": str(e)}

def create_step_interface(step_name: str) -> Dict[str, Any]:
    """
    🔥 Step 클래스용 인터페이스 생성
    모든 Step 클래스에서 사용하는 핵심 함수
    """
    try:
        utils = get_pipeline_utils()
        return utils.create_step_interface(step_name)
    except Exception as e:
        logger.error(f"❌ {step_name} 인터페이스 생성 실패: {e}")
        return {
            "step_name": step_name,
            "error": str(e),
            "system_info": SYSTEM_INFO
        }

def get_system_status() -> Dict[str, Any]:
    """시스템 상태 조회"""
    try:
        utils = get_pipeline_utils()
        return utils.get_status()
    except Exception as e:
        return {"error": str(e), "system_info": SYSTEM_INFO}

def cleanup_pipeline_utils():
    """파이프라인 유틸리티 정리"""
    global _global_utils
    
    try:
        with _utils_lock:
            if _global_utils:
                _global_utils.cleanup()
                _global_utils = None
        
        # 전역 ModelLoader 정리
        if MODEL_LOADER_AVAILABLE:
            cleanup_global_loader()
        
        logger.info("✅ 파이프라인 유틸리티 정리 완료")
        
    except Exception as e:
        logger.warning(f"⚠️ 파이프라인 유틸리티 정리 실패: {e}")

# ==============================================
# 🔥 편의 함수들 (하위 호환성)
# ==============================================

def get_model_loader():
    """ModelLoader 인스턴스 반환 (하위 호환)"""
    try:
        if MODEL_LOADER_AVAILABLE:
            return get_global_model_loader()
        return None
    except Exception as e:
        logger.error(f"ModelLoader 반환 실패: {e}")
        return None

def detect_models(search_paths: Optional[List[Path]] = None):
    """자동 모델 탐지 (하위 호환)"""
    try:
        if AUTO_DETECTOR_AVAILABLE:
            return quick_detect_models(search_paths=search_paths)
        return {}
    except Exception as e:
        logger.error(f"모델 탐지 실패: {e}")
        return {}

def get_step_requirements(step_name: str):
    """Step 요구사항 조회 (하위 호환)"""
    try:
        if STEP_REQUESTS_AVAILABLE:
            return get_step_request(step_name)
        return None
    except Exception as e:
        logger.error(f"Step 요구사항 조회 실패: {e}")
        return None

# ==============================================
# 🔥 __all__ 정의
# ==============================================

__all__ = [
    # 🎯 핵심 함수들 (Step 클래스에서 사용)
    'create_step_interface',
    'initialize_pipeline_utils',
    'get_pipeline_utils',
    'get_system_status',
    'cleanup_pipeline_utils',
    
    # 📊 시스템 정보
    'SYSTEM_INFO',
    'IS_M3_MAX',
    'DEVICE',
    'MEMORY_GB',
    
    # 🔧 편의 함수들
    'get_model_loader',
    'detect_models',
    'get_step_requirements',
    
    # 📦 핵심 클래스 (사용 가능한 경우)
    'PipelineUtils'
]

# Step Model Requests 모듈 export
if STEP_REQUESTS_AVAILABLE:
    __all__.extend([
        'STEP_MODEL_REQUESTS',
        'ModelRequest',
        'StepPriority',
        'get_step_request',
        'get_all_step_requests',
        'get_checkpoint_patterns',
        'get_model_config_for_step',
        'validate_model_for_step',
        'get_step_priorities',
        'get_steps_by_priority'
    ])

# Auto Model Detector 모듈 export
if AUTO_DETECTOR_AVAILABLE:
    __all__.extend([
        'AutoModelDetector',
        'DetectedModel',
        'DetectionStatus',
        'quick_detect_models',
        'detect_and_export_for_loader',
        'validate_detected_models'
    ])

# Model Loader 모듈 export
if MODEL_LOADER_AVAILABLE:
    __all__.extend([
        'ModelLoader',
        'ModelConfig',
        'ModelType',
        'ModelFormat',
        'LoadedModel',
        'StepModelInterface',
        'BaseStepMixin',
        # AI 모델 클래스들
        'BaseModel',
        'GraphonomyModel',
        'OpenPoseModel',
        'U2NetModel',
        'GeometricMatchingModel',
        'HRVITONModel',
        # 유틸리티 함수들
        'preprocess_image',
        'postprocess_segmentation',
        # 전역 함수들
        'get_global_model_loader',
        'initialize_global_model_loader',
        'cleanup_global_loader'
    ])

# ==============================================
# 🔥 모듈 초기화 및 요약
# ==============================================

def _print_initialization_summary():
    """초기화 요약 출력"""
    available_modules = sum([
        STEP_REQUESTS_AVAILABLE,
        AUTO_DETECTOR_AVAILABLE,
        MODEL_LOADER_AVAILABLE
    ])
    
    logger.info("=" * 70)
    logger.info("🍎 MyCloset AI 파이프라인 유틸리티 v4.0")
    logger.info("=" * 70)
    logger.info(f"📊 사용 가능한 모듈: {available_modules}/3")
    logger.info(f"   - Step Model Requests: {'✅' if STEP_REQUESTS_AVAILABLE else '❌'}")
    logger.info(f"   - Auto Model Detector: {'✅' if AUTO_DETECTOR_AVAILABLE else '❌'}")
    logger.info(f"   - Model Loader: {'✅' if MODEL_LOADER_AVAILABLE else '❌'}")
    logger.info(f"🍎 시스템 정보:")
    logger.info(f"   - Platform: {SYSTEM_INFO['platform']}")
    logger.info(f"   - M3 Max: {'✅' if IS_M3_MAX else '❌'}")
    logger.info(f"   - Device: {DEVICE}")
    logger.info(f"   - Memory: {MEMORY_GB}GB")
    
    if available_modules >= 2:
        logger.info("✅ 파이프라인 유틸리티 시스템 준비 완료!")
        logger.info("🔥 Step 클래스에서 create_step_interface() 사용 가능")
    else:
        logger.warning("⚠️ 일부 모듈만 사용 가능 - 제한적 기능 제공")
    
    logger.info("=" * 70)

# 초기화 요약 출력
_print_initialization_summary()

# 종료 시 정리 함수 등록
import atexit
atexit.register(cleanup_pipeline_utils)

# 환경변수 기반 자동 초기화
if os.getenv('AUTO_INIT_PIPELINE_UTILS', 'false').lower() in ('true', '1', 'yes'):
    try:
        result = initialize_pipeline_utils()
        if result.get('success'):
            logger.info("🚀 파이프라인 유틸리티 자동 초기화 완료")
        else:
            logger.warning(f"⚠️ 파이프라인 유틸리티 자동 초기화 실패: {result.get('error')}")
    except Exception as e:
        logger.warning(f"⚠️ 자동 초기화 실패: {e}")

logger.info("🏁 파이프라인 유틸리티 v4.0 로드 완료")