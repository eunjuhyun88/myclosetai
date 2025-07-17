# app/ai_pipeline/utils/__init__.py
"""
🍎 MyCloset AI 파이프라인 유틸리티 모듈 v3.0 - 완전 재구성
✅ 최적 생성자 패턴 적용 + 실제 모델 자동 탐지
✅ M3 Max 128GB 최적화 설계
✅ Step 클래스 완벽 호환 + 프로덕션 안정성
✅ 단순함 + 편의성 + 확장성 + 일관성
🔥 핵심: 모든 Step 클래스에서 바로 사용 가능한 통합 시스템
"""

import os
import gc
import sys
import time
import logging
import asyncio
import threading
from typing import Dict, Any, Optional, Union, List, Callable, Tuple
from pathlib import Path
from functools import lru_cache
import weakref

# 기본 로깅 설정
logger = logging.getLogger(__name__)

# ==============================================
# 🔥 시스템 환경 감지 및 설정
# ==============================================

@lru_cache(maxsize=1)
def _detect_system_info() -> Dict[str, Any]:
    """시스템 정보 감지 (캐시됨)"""
    try:
        import platform
        system_info = {
            "platform": platform.system(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python_version": sys.version_info[:3]
        }
        
        # M3 Max 감지
        is_m3_max = False
        try:
            if platform.system() == 'Darwin':
                import subprocess
                result = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'], 
                                      capture_output=True, text=True, timeout=5)
                is_m3_max = 'M3' in result.stdout
        except:
            pass
        
        system_info["is_m3_max"] = is_m3_max
        
        # GPU 감지
        gpu_type = "없음"
        try:
            import torch
            if torch.backends.mps.is_available():
                gpu_type = "MPS (Apple Silicon)"
            elif torch.cuda.is_available():
                gpu_type = f"CUDA ({torch.cuda.get_device_name(0)})"
        except:
            pass
        
        system_info["gpu_type"] = gpu_type
        
        # 메모리 감지
        try:
            import psutil
            system_info["memory_gb"] = round(psutil.virtual_memory().total / (1024**3))
        except:
            system_info["memory_gb"] = 16  # 기본값
        
        return system_info
        
    except Exception as e:
        logger.warning(f"시스템 정보 감지 실패: {e}")
        return {
            "platform": "Unknown",
            "is_m3_max": False,
            "gpu_type": "없음",
            "memory_gb": 16
        }

# 시스템 정보 전역 변수
SYSTEM_INFO = _detect_system_info()
IS_M3_MAX = SYSTEM_INFO["is_m3_max"]
MEMORY_GB = SYSTEM_INFO["memory_gb"]

# 디바이스 자동 감지
@lru_cache(maxsize=1)
def _detect_default_device() -> str:
    """기본 디바이스 감지"""
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
        elif torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"
    except ImportError:
        return "cpu"

DEFAULT_DEVICE = _detect_default_device()

# PyTorch 가용성 확인
try:
    import torch
    TORCH_AVAILABLE = True
    logger.info(f"✅ PyTorch 사용 가능 - 디바이스: {DEFAULT_DEVICE}")
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("⚠️ PyTorch 사용 불가")

# ==============================================
# 🔥 핵심 유틸리티 모듈들 안전한 Import
# ==============================================

# 1. MemoryManager - 메모리 관리 시스템
try:
    from .memory_manager import (
        MemoryManager,
        MemoryStats,
        create_memory_manager,
        get_memory_manager,
        get_global_memory_manager,
        initialize_global_memory_manager,
        optimize_memory_usage,
        optimize_memory,
        check_memory,
        check_memory_available,
        get_memory_info,
        memory_efficient
    )
    MEMORY_MANAGER_AVAILABLE = True
    logger.info("✅ MemoryManager 모듈 로드 성공")
except ImportError as e:
    MEMORY_MANAGER_AVAILABLE = False
    logger.warning(f"⚠️ MemoryManager import 실패: {e}")
    
    # 🔥 핵심: 폴백 MemoryManager (기본 기능 제공)
    class MemoryManager:
        def __init__(self, device="auto", **kwargs):
            self.device = device if device != "auto" else DEFAULT_DEVICE
            self.logger = logging.getLogger(f"{__name__}.FallbackMemoryManager")
            
        async def initialize(self) -> bool:
            return True
            
        def get_memory_stats(self):
            return {"device": self.device, "status": "fallback"}
            
        def check_memory_pressure(self):
            return {"status": "normal", "message": "fallback mode"}
            
        def clear_cache(self, aggressive=False):
            if TORCH_AVAILABLE:
                import gc
                gc.collect()
                
        async def cleanup(self):
            self.clear_cache()
    
    # 폴백 함수들
    def create_memory_manager(device="auto", **kwargs):
        return MemoryManager(device=device, **kwargs)
    
    def get_memory_manager(**kwargs):
        return create_memory_manager(**kwargs)
    
    def get_global_memory_manager(**kwargs):
        return get_memory_manager(**kwargs)
    
    def initialize_global_memory_manager(device="mps", **kwargs):
        return create_memory_manager(device=device, **kwargs)
    
    def optimize_memory_usage(device=None, aggressive=False):
        manager = create_memory_manager(device=device or DEFAULT_DEVICE)
        manager.clear_cache(aggressive=aggressive)
        return {"success": True, "device": manager.device}
    
    def check_memory_available(min_gb=1.0):
        return True  # 폴백에서는 항상 True
    
    def get_memory_info():
        return {"device": DEFAULT_DEVICE, "fallback": True}

# 2. ModelLoader - AI 모델 로딩 시스템
try:
    from .model_loader import (
        ModelLoader,
        ModelConfig,
        ModelFormat,
        ModelType,
        ModelPriority,
        LoadedModel,
        StepModelInterface,
        BaseStepMixin,
        create_model_loader,
        get_global_model_loader,
        initialize_global_model_loader,
        cleanup_global_loader,
        load_model_async,
        load_model_sync,
        preprocess_image,
        postprocess_segmentation,
        postprocess_pose,
        # AI 모델 클래스들
        BaseModel,
        GraphonomyModel,
        OpenPoseModel,
        U2NetModel,
        GeometricMatchingModel,
        EnhancementModel,
        CLIPModel
    )
    MODEL_LOADER_AVAILABLE = True
    logger.info("✅ ModelLoader 모듈 로드 성공")
except ImportError as e:
    MODEL_LOADER_AVAILABLE = False
    logger.warning(f"⚠️ ModelLoader import 실패: {e}")
    
    # 🔥 핵심: 폴백 ModelLoader (기본 기능 제공)
    from enum import Enum
    from dataclasses import dataclass
    
    class ModelType(Enum):
        HUMAN_PARSING = "human_parsing"
        POSE_ESTIMATION = "pose_estimation"
        CLOTH_SEGMENTATION = "cloth_segmentation"
        GEOMETRIC_MATCHING = "geometric_matching"
        CLOTH_WARPING = "cloth_warping"
        VIRTUAL_FITTING = "virtual_fitting"
        POST_PROCESSING = "post_processing"
        QUALITY_ASSESSMENT = "quality_assessment"
    
    class ModelFormat(Enum):
        PYTORCH = "pytorch"
        SAFETENSORS = "safetensors"
        DIFFUSERS = "diffusers"
        ONNX = "onnx"
    
    class ModelPriority(Enum):
        CRITICAL = 1
        HIGH = 2
        MEDIUM = 3
        LOW = 4
    
    @dataclass
    class ModelConfig:
        name: str
        model_type: ModelType
        model_class: str
        checkpoint_path: Optional[str] = None
        device: str = "auto"
        precision: str = "fp16"
        input_size: Tuple[int, int] = (512, 512)
        priority: ModelPriority = ModelPriority.MEDIUM
        metadata: Dict[str, Any] = None
    
    class ModelLoader:
        def __init__(self, device=None, **kwargs):
            self.device = device if device else DEFAULT_DEVICE
            self.logger = logging.getLogger(f"{__name__}.FallbackModelLoader")
            self.model_configs = {}
            self.loaded_models = {}
            
        async def initialize(self) -> bool:
            return True
            
        def register_model(self, name: str, config: ModelConfig) -> bool:
            self.model_configs[name] = config
            return True
            
        async def load_model(self, name: str, force_reload=False):
            self.logger.warning(f"폴백 모드: {name} 모델 로드 시뮬레이션")
            return None
            
        def list_models(self):
            return list(self.model_configs.keys())
        
        def cleanup(self):
            self.loaded_models.clear()
    
    class StepModelInterface:
        def __init__(self, model_loader, step_name):
            self.model_loader = model_loader
            self.step_name = step_name
            
        async def get_model(self, model_name=None):
            return None
            
        def cleanup(self):
            pass
    
    class BaseStepMixin:
        def _setup_model_interface(self, model_loader=None):
            self.model_interface = StepModelInterface(model_loader or ModelLoader(), self.__class__.__name__)
            
        async def get_model(self, model_name=None):
            if hasattr(self, 'model_interface'):
                return await self.model_interface.get_model(model_name)
            return None
    
    # 폴백 팩토리 함수들
    def create_model_loader(device="auto", **kwargs):
        return ModelLoader(device=device, **kwargs)
    
    def get_global_model_loader():
        return create_model_loader()
    
    def initialize_global_model_loader(**kwargs):
        return {"success": True, "message": "Fallback mode initialized"}
    
    def cleanup_global_loader():
        pass
    
    async def load_model_async(model_name):
        return None
    
    def load_model_sync(model_name):
        return None
    
    def preprocess_image(image, target_size=(512, 512), normalize=True, device="cpu"):
        logger.warning("폴백 모드: 이미지 전처리 시뮬레이션")
        return None
    
    def postprocess_segmentation(output, original_size, threshold=0.5):
        logger.warning("폴백 모드: 세그멘테이션 후처리 시뮬레이션")
        return None
    
    def postprocess_pose(output, original_size, confidence_threshold=0.3):
        logger.warning("폴백 모드: 포즈 후처리 시뮬레이션")
        return {"keypoints": [], "num_keypoints": 0}

# 3. DataConverter - 데이터 변환 시스템
try:
    from .data_converter import (
        DataConverter,
        create_data_converter,
        get_global_data_converter,
        initialize_global_data_converter,
        quick_image_to_tensor,
        quick_tensor_to_image
    )
    DATA_CONVERTER_AVAILABLE = True
    logger.info("✅ DataConverter 모듈 로드 성공")
except ImportError as e:
    DATA_CONVERTER_AVAILABLE = False
    logger.warning(f"⚠️ DataConverter import 실패: {e}")
    
    # 🔥 핵심: 폴백 DataConverter (기본 기능 제공)
    class DataConverter:
        def __init__(self, device=None, **kwargs):
            self.device = device if device else DEFAULT_DEVICE
            self.logger = logging.getLogger(f"{__name__}.FallbackDataConverter")
            self.default_size = kwargs.get('default_size', (512, 512))
            
        async def initialize(self) -> bool:
            return True
            
        def image_to_tensor(self, image, size=None, normalize=False, **kwargs):
            self.logger.warning("폴백 모드: 이미지→텐서 변환 시뮬레이션")
            return None
            
        def tensor_to_image(self, tensor, denormalize=False, format="PIL"):
            self.logger.warning("폴백 모드: 텐서→이미지 변환 시뮬레이션")
            return None
            
        def batch_convert_images(self, images, target_format="tensor", **kwargs):
            return [None] * len(images)
            
        def resize_image(self, image, size, method="bilinear", preserve_aspect_ratio=False):
            self.logger.warning("폴백 모드: 이미지 크기 조정 시뮬레이션")
            return image
    
    # 폴백 팩토리 함수들
    def create_data_converter(default_size=(512, 512), device="auto", **kwargs):
        return DataConverter(device=device, default_size=default_size, **kwargs)
    
    def get_global_data_converter():
        return create_data_converter()
    
    def initialize_global_data_converter(**kwargs):
        return create_data_converter(**kwargs)
    
    def quick_image_to_tensor(image, size=(512, 512)):
        return None
    
    def quick_tensor_to_image(tensor):
        return None

# 4. 자동 모델 탐지 시스템 (선택적)
try:
    from .auto_model_detector import (
        AdvancedModelDetector,
        AdvancedModelLoaderAdapter,
        DetectedModel,
        ModelCategory,
        create_advanced_detector,
        quick_model_detection,
        detect_and_integrate_with_model_loader,
        export_model_registry_code,
        validate_model_paths,
        benchmark_model_loading
    )
    AUTO_MODEL_DETECTOR_AVAILABLE = True
    logger.info("✅ AutoModelDetector 모듈 로드 성공")
except ImportError as e:
    AUTO_MODEL_DETECTOR_AVAILABLE = False
    logger.warning(f"⚠️ AutoModelDetector import 실패: {e}")
    
    # 기본 폴백 (자동 탐지 없이)
    def quick_model_detection(**kwargs):
        return {"total_models": 0, "message": "Auto detection not available"}

# ==============================================
# 🔥 통합 관리 시스템
# ==============================================

class UtilsManager:
    """
    🍎 통합 유틸리티 관리자 - 최적 생성자 패턴 적용
    ✅ 모든 유틸리티를 하나의 인터페이스로 통합
    ✅ Step 클래스에서 바로 사용 가능
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
            
        self.logger = logging.getLogger(f"{__name__}.UtilsManager")
        
        # 컴포넌트 초기화
        self.memory_manager = None
        self.model_loader = None
        self.data_converter = None
        
        # 상태 관리
        self.is_initialized = False
        self.initialization_time = None
        
        self._initialized = True
        
        self.logger.info("🎯 UtilsManager 인스턴스 생성")
    
    async def initialize(
        self,
        device: Optional[str] = None,
        memory_gb: Optional[float] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """통합 초기화"""
        if self.is_initialized:
            return {"success": True, "message": "Already initialized"}
        
        start_time = time.time()
        device = device or DEFAULT_DEVICE
        memory_gb = memory_gb or MEMORY_GB
        
        self.logger.info(f"🚀 UtilsManager 초기화 시작 - 디바이스: {device}")
        
        results = {
            "memory_manager": False,
            "model_loader": False,
            "data_converter": False,
            "errors": []
        }
        
        try:
            # 1. MemoryManager 초기화
            if MEMORY_MANAGER_AVAILABLE:
                try:
                    self.memory_manager = create_memory_manager(
                        device=device,
                        memory_gb=memory_gb,
                        is_m3_max=IS_M3_MAX,
                        optimization_enabled=True,
                        **kwargs
                    )
                    await self.memory_manager.initialize()
                    results["memory_manager"] = True
                    self.logger.info("✅ MemoryManager 초기화 완료")
                except Exception as e:
                    results["errors"].append(f"MemoryManager: {e}")
                    self.logger.error(f"❌ MemoryManager 초기화 실패: {e}")
            
            # 2. ModelLoader 초기화
            if MODEL_LOADER_AVAILABLE:
                try:
                    self.model_loader = create_model_loader(
                        device=device,
                        memory_limit_gb=memory_gb,
                        auto_scan=True,
                        **kwargs
                    )
                    await self.model_loader.initialize()
                    results["model_loader"] = True
                    self.logger.info("✅ ModelLoader 초기화 완료")
                except Exception as e:
                    results["errors"].append(f"ModelLoader: {e}")
                    self.logger.error(f"❌ ModelLoader 초기화 실패: {e}")
            
            # 3. DataConverter 초기화
            if DATA_CONVERTER_AVAILABLE:
                try:
                    self.data_converter = create_data_converter(
                        device=device,
                        default_size=kwargs.get('default_size', (512, 512)),
                        is_m3_max=IS_M3_MAX,
                        **kwargs
                    )
                    await self.data_converter.initialize()
                    results["data_converter"] = True
                    self.logger.info("✅ DataConverter 초기화 완료")
                except Exception as e:
                    results["errors"].append(f"DataConverter: {e}")
                    self.logger.error(f"❌ DataConverter 초기화 실패: {e}")
            
            # 4. 폴백 인스턴스 생성 (필요한 경우)
            if not self.memory_manager:
                self.memory_manager = create_memory_manager(device=device)
            if not self.model_loader:
                self.model_loader = create_model_loader(device=device)
            if not self.data_converter:
                self.data_converter = create_data_converter(device=device)
            
            self.is_initialized = True
            self.initialization_time = time.time() - start_time
            
            success_count = sum(results[key] for key in ["memory_manager", "model_loader", "data_converter"])
            
            self.logger.info(f"🎉 UtilsManager 초기화 완료 ({self.initialization_time:.2f}s)")
            self.logger.info(f"📊 성공한 컴포넌트: {success_count}/3")
            
            return {
                "success": True,
                "initialization_time": self.initialization_time,
                "device": device,
                "components": results,
                "system_info": SYSTEM_INFO
            }
            
        except Exception as e:
            self.logger.error(f"❌ UtilsManager 초기화 실패: {e}")
            return {
                "success": False,
                "error": str(e),
                "components": results
            }
    
    def create_step_interface(self, step_name: str) -> Dict[str, Any]:
        """Step 클래스용 통합 인터페이스 생성"""
        try:
            interface = {
                "step_name": step_name,
                "memory_manager": self.memory_manager,
                "model_loader": self.model_loader,
                "data_converter": self.data_converter,
                "get_model": self._create_get_model_func(step_name),
                "process_image": self._create_process_image_func(),
                "optimize_memory": self._create_optimize_memory_func(),
                "logger": logging.getLogger(f"steps.{step_name}")
            }
            
            self.logger.info(f"🔗 {step_name} 인터페이스 생성 완료")
            return interface
            
        except Exception as e:
            self.logger.error(f"❌ {step_name} 인터페이스 생성 실패: {e}")
            return {"error": str(e)}
    
    def _create_get_model_func(self, step_name: str) -> Callable:
        """모델 로드 함수 생성"""
        async def get_model(model_name: Optional[str] = None):
            try:
                if self.model_loader:
                    if hasattr(self.model_loader, 'create_step_interface'):
                        interface = self.model_loader.create_step_interface(step_name)
                        return await interface.get_model(model_name)
                    else:
                        return await self.model_loader.load_model(model_name)
                return None
            except Exception as e:
                self.logger.error(f"❌ {step_name} 모델 로드 실패: {e}")
                return None
        return get_model
    
    def _create_process_image_func(self) -> Callable:
        """이미지 처리 함수 생성"""
        def process_image(image, operation="to_tensor", **kwargs):
            try:
                if self.data_converter:
                    if operation == "to_tensor":
                        return self.data_converter.image_to_tensor(image, **kwargs)
                    elif operation == "from_tensor":
                        return self.data_converter.tensor_to_image(image, **kwargs)
                    elif operation == "resize":
                        return self.data_converter.resize_image(image, **kwargs)
                return None
            except Exception as e:
                self.logger.error(f"❌ 이미지 처리 실패: {e}")
                return None
        return process_image
    
    def _create_optimize_memory_func(self) -> Callable:
        """메모리 최적화 함수 생성"""
        async def optimize_memory(aggressive: bool = False):
            try:
                if self.memory_manager:
                    if hasattr(self.memory_manager, 'smart_cleanup'):
                        self.memory_manager.smart_cleanup()
                    elif hasattr(self.memory_manager, 'clear_cache'):
                        self.memory_manager.clear_cache(aggressive=aggressive)
                    return {"success": True}
                return {"success": False, "message": "MemoryManager not available"}
            except Exception as e:
                self.logger.error(f"❌ 메모리 최적화 실패: {e}")
                return {"success": False, "error": str(e)}
        return optimize_memory
    
    def get_system_status(self) -> Dict[str, Any]:
        """시스템 상태 조회"""
        try:
            status = {
                "is_initialized": self.is_initialized,
                "initialization_time": self.initialization_time,
                "system_info": SYSTEM_INFO,
                "components": {
                    "memory_manager": {
                        "available": MEMORY_MANAGER_AVAILABLE,
                        "initialized": self.memory_manager is not None
                    },
                    "model_loader": {
                        "available": MODEL_LOADER_AVAILABLE,
                        "initialized": self.model_loader is not None
                    },
                    "data_converter": {
                        "available": DATA_CONVERTER_AVAILABLE,
                        "initialized": self.data_converter is not None
                    }
                }
            }
            
            # 상세 정보 추가
            if self.memory_manager and hasattr(self.memory_manager, 'get_memory_stats'):
                try:
                    status["memory_stats"] = self.memory_manager.get_memory_stats().__dict__
                except:
                    pass
            
            if self.model_loader and hasattr(self.model_loader, 'get_system_info'):
                try:
                    status["model_loader_info"] = self.model_loader.get_system_info()
                except:
                    pass
            
            return status
            
        except Exception as e:
            return {"error": str(e)}
    
    async def cleanup(self):
        """리소스 정리"""
        try:
            if self.model_loader and hasattr(self.model_loader, 'cleanup'):
                self.model_loader.cleanup()
            
            if self.memory_manager and hasattr(self.memory_manager, 'cleanup'):
                await self.memory_manager.cleanup()
            
            self.is_initialized = False
            self.logger.info("✅ UtilsManager 정리 완료")
            
        except Exception as e:
            self.logger.error(f"❌ UtilsManager 정리 실패: {e}")

# ==============================================
# 🔥 전역 함수들 - Step 클래스에서 바로 사용
# ==============================================

# 전역 UtilsManager 인스턴스
_global_utils_manager: Optional[UtilsManager] = None
_utils_lock = threading.Lock()

def get_utils_manager() -> UtilsManager:
    """전역 UtilsManager 인스턴스 반환"""
    global _global_utils_manager
    
    with _utils_lock:
        if _global_utils_manager is None:
            _global_utils_manager = UtilsManager()
        return _global_utils_manager

def initialize_global_utils(device: Optional[str] = None, **kwargs) -> Dict[str, Any]:
    """
    🔥 전역 유틸리티 시스템 초기화 - main.py에서 사용
    
    Args:
        device: 사용할 디바이스 ('auto', 'mps', 'cuda', 'cpu')
        **kwargs: 추가 설정
    
    Returns:
        초기화 결과 정보
    """
    try:
        manager = get_utils_manager()
        
        # 비동기 초기화 실행
        import asyncio
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        if loop.is_running():
            # 이미 실행 중인 루프에서는 태스크로 실행
            future = asyncio.create_task(manager.initialize(device=device, **kwargs))
            return {"success": True, "message": "Initialization started", "future": future}
        else:
            result = loop.run_until_complete(manager.initialize(device=device, **kwargs))
            return result
            
    except Exception as e:
        logger.error(f"❌ 전역 유틸리티 초기화 실패: {e}")
        return {"success": False, "error": str(e)}

def create_step_interface(step_name: str) -> Dict[str, Any]:
    """
    🔥 Step 클래스용 인터페이스 생성 - 모든 Step에서 사용
    
    Args:
        step_name: Step 클래스 이름
    
    Returns:
        Step용 통합 인터페이스
    """
    try:
        manager = get_utils_manager()
        return manager.create_step_interface(step_name)
    except Exception as e:
        logger.error(f"❌ {step_name} 인터페이스 생성 실패: {e}")
        return {"error": str(e)}

def get_system_status() -> Dict[str, Any]:
    """시스템 상태 조회"""
    try:
        manager = get_utils_manager()
        return manager.get_system_status()
    except Exception as e:
        return {"error": str(e)}

def reset_global_utils():
    """전역 유틸리티 리셋"""
    global _global_utils_manager
    
    try:
        with _utils_lock:
            if _global_utils_manager:
                asyncio.create_task(_global_utils_manager.cleanup())
                _global_utils_manager = None
        logger.info("✅ 전역 유틸리티 리셋 완료")
    except Exception as e:
        logger.warning(f"⚠️ 전역 유틸리티 리셋 실패: {e}")

# 편의 함수들 (하위 호환)
def get_memory_manager_instance(**kwargs):
    """메모리 매니저 인스턴스 반환"""
    manager = get_utils_manager()
    return manager.memory_manager or create_memory_manager(**kwargs)

def get_model_loader_instance(**kwargs):
    """모델 로더 인스턴스 반환"""
    manager = get_utils_manager()
    return manager.model_loader or create_model_loader(**kwargs)

def get_data_converter_instance(**kwargs):
    """데이터 변환기 인스턴스 반환"""
    manager = get_utils_manager()
    return manager.data_converter or create_data_converter(**kwargs)

# ==============================================
# 🔥 __all__ 정의 - 모든 export 정리
# ==============================================

__all__ = [
    # 🎯 핵심 클래스들
    'UtilsManager',
    
    # 🔥 주요 전역 함수들 (Step에서 사용)
    'get_utils_manager',
    'initialize_global_utils',
    'create_step_interface',
    'get_system_status',
    'reset_global_utils',
    
    # 📦 컴포넌트 인스턴스 함수들
    'get_memory_manager_instance',
    'get_model_loader_instance', 
    'get_data_converter_instance',
    
    # 🍎 시스템 정보
    'SYSTEM_INFO',
    'IS_M3_MAX',
    'MEMORY_GB',
    'DEFAULT_DEVICE',
    'TORCH_AVAILABLE',
]

# MemoryManager 모듈이 사용 가능한 경우 추가
if MEMORY_MANAGER_AVAILABLE:
    __all__.extend([
        'MemoryManager',
        'MemoryStats',
        'create_memory_manager',
        'get_memory_manager',
        'get_global_memory_manager',
        'initialize_global_memory_manager',
        'optimize_memory_usage',
        'optimize_memory',
        'check_memory',
        'check_memory_available',
        'get_memory_info',
        'memory_efficient'
    ])

# ModelLoader 모듈이 사용 가능한 경우 추가
if MODEL_LOADER_AVAILABLE:
    __all__.extend([
        'ModelLoader',
        'ModelConfig',
        'ModelFormat',
        'ModelType',
        'ModelPriority',
        'LoadedModel',
        'StepModelInterface',
        'BaseStepMixin',
        'create_model_loader',
        'get_global_model_loader',
        'initialize_global_model_loader',
        'cleanup_global_loader',
        'load_model_async',
        'load_model_sync',
        'preprocess_image',
        'postprocess_segmentation',
        'postprocess_pose',
        # AI 모델 클래스들
        'BaseModel',
        'GraphonomyModel',
        'OpenPoseModel',
        'U2NetModel',
        'GeometricMatchingModel',
        'EnhancementModel',
        'CLIPModel'
    ])

# DataConverter 모듈이 사용 가능한 경우 추가
if DATA_CONVERTER_AVAILABLE:
    __all__.extend([
        'DataConverter',
        'create_data_converter',
        'get_global_data_converter',
        'initialize_global_data_converter',
        'quick_image_to_tensor',
        'quick_tensor_to_image'
    ])

# AutoModelDetector 모듈이 사용 가능한 경우 추가
if AUTO_MODEL_DETECTOR_AVAILABLE:
    __all__.extend([
        'AdvancedModelDetector',
        'AdvancedModelLoaderAdapter',
        'DetectedModel',
        'ModelCategory',
        'create_advanced_detector',
        'quick_model_detection',
        'detect_and_integrate_with_model_loader',
        'export_model_registry_code',
        'validate_model_paths',
        'benchmark_model_loading'
    ])

# ==============================================
# 🔥 모듈 초기화 로깅 및 최종 설정
# ==============================================

def _log_initialization_summary():
    """초기화 요약 로깅"""
    available_count = sum([
        MEMORY_MANAGER_AVAILABLE,
        MODEL_LOADER_AVAILABLE, 
        DATA_CONVERTER_AVAILABLE,
        AUTO_MODEL_DETECTOR_AVAILABLE
    ])
    
    total_count = 4
    
    logger.info("=" * 70)
    logger.info("🍎 MyCloset AI 파이프라인 유틸리티 v3.0 - 완전 재구성")
    logger.info("=" * 70)
    logger.info(f"📊 사용 가능한 유틸리티: {available_count}/{total_count}")
    logger.info(f"   - MemoryManager: {'✅' if MEMORY_MANAGER_AVAILABLE else '❌ (폴백 사용)'}")
    logger.info(f"   - ModelLoader: {'✅' if MODEL_LOADER_AVAILABLE else '❌ (폴백 사용)'}")
    logger.info(f"   - DataConverter: {'✅' if DATA_CONVERTER_AVAILABLE else '❌ (폴백 사용)'}")
    logger.info(f"   - AutoModelDetector: {'✅' if AUTO_MODEL_DETECTOR_AVAILABLE else '❌'}")
    logger.info(f"🍎 시스템 정보:")
    logger.info(f"   - Platform: {SYSTEM_INFO['platform']}")
    logger.info(f"   - M3 Max: {'✅' if IS_M3_MAX else '❌'}")
    logger.info(f"   - Memory: {MEMORY_GB}GB")
    logger.info(f"   - GPU: {SYSTEM_INFO['gpu_type']}")
    logger.info(f"   - 기본 디바이스: {DEFAULT_DEVICE}")
    logger.info(f"   - PyTorch: {'✅' if TORCH_AVAILABLE else '❌'}")
    
    if available_count >= 2:  # 최소 2개 이상 사용 가능
        logger.info("✅ 파이프라인 유틸리티 시스템 준비 완료!")
    elif available_count > 0:
        logger.warning(f"⚠️ 일부 유틸리티만 사용 가능 ({available_count}/{total_count}) - 폴백 모드로 동작")
    else:
        logger.error("❌ 모든 유틸리티 사용 불가 - 의존성 설치 필요")
    
    logger.info("=" * 70)

# 초기화 요약 출력
_log_initialization_summary()

# 자동 초기화 (환경변수 기반)
try:
    auto_init = os.getenv('AUTO_INIT_PIPELINE_UTILS', 'false').lower() in ('true', '1', 'yes', 'on')
    if auto_init:
        device = os.getenv('PIPELINE_DEVICE', DEFAULT_DEVICE)
        result = initialize_global_utils(device=device)
        if result.get('success'):
            logger.info("🚀 파이프라인 유틸리티 자동 초기화 완료")
        else:
            logger.warning(f"⚠️ 파이프라인 유틸리티 자동 초기화 실패: {result.get('error', 'Unknown')}")
except Exception as e:
    logger.warning(f"⚠️ 자동 초기화 실패: {e}")

# 종료 시 정리 함수 등록
import atexit

def _cleanup_on_exit():
    """프로그램 종료 시 정리"""
    try:
        reset_global_utils()
        logger.info("🔚 파이프라인 유틸리티 종료 정리 완료")
    except:
        pass

atexit.register(_cleanup_on_exit)

# 최종 확인 메시지
if MEMORY_MANAGER_AVAILABLE or MODEL_LOADER_AVAILABLE or DATA_CONVERTER_AVAILABLE:
    logger.info("🎯 최적 생성자 패턴 AI 파이프라인 유틸리티 v3.0 준비 완료")
    logger.info("🔥 Step 클래스에서 create_step_interface() 함수로 바로 사용 가능")
else:
    logger.error("💥 모든 유틸리티 모듈 사용 불가 - 폴백 모드로만 동작")

logger.info("🏁 AI 파이프라인 유틸리티 모듈 v3.0 로딩 완료")

# 디버그 정보 (개발 환경에서만)
if os.getenv('DEBUG_PIPELINE_UTILS', 'false').lower() in ('true', '1'):
    logger.debug(f"🐛 DEBUG: 전체 export 목록 ({len(__all__)}개)")
    for i, item in enumerate(__all__, 1):
        logger.debug(f"   {i:2d}. {item}")