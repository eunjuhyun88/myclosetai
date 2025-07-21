#app/ai_pipeline/__init__.py
# app/ai_pipeline/__init__.py
"""
🍎 MyCloset AI 파이프라인 메인 모듈 v6.0
✅ 완전한 모듈 구조 및 import 시스템
✅ 순환참조 완전 해결
✅ main.py 완벽 호환
✅ 비동기 처리 완전 지원
✅ M3 Max 128GB 최적화
✅ conda 환경 완벽 지원
✅ 프로덕션 안정성

구조:
- steps/: 8단계 AI 파이프라인 클래스들
- utils/: 유틸리티 및 통합 시스템
- models/: AI 모델 관련 클래스들
- pipeline_manager.py: 파이프라인 통합 관리자
"""

import os
import sys
import logging
import asyncio
import time
import threading
from typing import Dict, Any, Optional, List, Union, Callable, Type
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
import weakref

# 기본 라이브러리들
try:
    import torch
    import numpy as np
    from PIL import Image
    CORE_LIBS_AVAILABLE = True
except ImportError as e:
    logging.warning(f"핵심 라이브러리 import 실패: {e}")
    CORE_LIBS_AVAILABLE = False

# ==============================================
# 🔥 버전 및 기본 정보
# ==============================================

__version__ = "6.0.0"
__author__ = "MyCloset AI Team"
__description__ = "AI 기반 가상 피팅 파이프라인 시스템"

# 로깅 설정
logger = logging.getLogger(__name__)

# ==============================================
# 🔥 시스템 정보 감지
# ==============================================

def _detect_system_info() -> Dict[str, Any]:
    """시스템 정보 자동 감지"""
    try:
        import platform
        
        system_info = {
            "platform": platform.system(),
            "machine": platform.machine(),
            "python_version": ".".join(map(str, sys.version_info[:3])),
            "cpu_count": os.cpu_count() or 4
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
        
        # 디바이스 감지
        device = "cpu"
        if CORE_LIBS_AVAILABLE and torch is not None:
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
        
        system_info["device"] = device
        
        # 메모리 감지
        try:
            import psutil
            system_info["memory_gb"] = round(psutil.virtual_memory().total / (1024**3))
        except ImportError:
            system_info["memory_gb"] = 128 if is_m3_max else 16
        
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

# 전역 시스템 정보
SYSTEM_INFO = _detect_system_info()

# ==============================================
# 🔥 설정 데이터 클래스들
# ==============================================

class PipelineMode(Enum):
    """파이프라인 실행 모드"""
    DEVELOPMENT = "development"
    PRODUCTION = "production"
    TESTING = "testing"
    SIMULATION = "simulation"

class QualityLevel(Enum):
    """품질 레벨"""
    FAST = "fast"
    BALANCED = "balanced"
    HIGH = "high"
    ULTRA = "ultra"

@dataclass
class PipelineConfig:
    """파이프라인 설정"""
    mode: PipelineMode = PipelineMode.PRODUCTION
    quality_level: QualityLevel = QualityLevel.BALANCED
    device: str = "auto"
    batch_size: int = 1
    image_size: int = 512
    use_fp16: bool = True
    enable_caching: bool = True
    memory_limit_gb: float = 16.0
    max_workers: int = 4
    timeout_seconds: int = 300
    save_intermediate: bool = False
    optimization_enabled: bool = True
    
    def __post_init__(self):
        # auto 디바이스 해석
        if self.device == "auto":
            self.device = SYSTEM_INFO["device"]
        
        # M3 Max 최적화
        if SYSTEM_INFO["is_m3_max"]:
            self.memory_limit_gb = min(self.memory_limit_gb, 102.4)  # 128GB의 80%
            self.max_workers = min(self.max_workers, 8)
            self.use_fp16 = True

@dataclass
class ProcessingResult:
    """처리 결과"""
    success: bool
    message: str
    processing_time: float
    confidence: float
    session_id: Optional[str] = None
    step_results: Dict[str, Any] = field(default_factory=dict)
    intermediate_data: Dict[str, Any] = field(default_factory=dict)
    memory_usage: Dict[str, float] = field(default_factory=dict)
    error: Optional[str] = None

# ==============================================
# 🔥 통합 유틸리티 시스템 Import
# ==============================================

# 1. 통합 유틸리티 시스템 (최우선)
try:
    from .utils import (
        # 핵심 관리자들
        get_utils_manager,
        initialize_global_utils,
        get_system_status,
        reset_global_utils,
        
        # 인터페이스 생성 함수들
        get_step_model_interface,
        create_step_interface,
        create_unified_interface,
        
        # 시스템 정보
        SYSTEM_INFO as UTILS_SYSTEM_INFO,
        optimize_system_memory,
        
        # 클래스들
        UnifiedUtilsManager,
        UnifiedStepInterface,
        StepModelInterface,
        SystemConfig,
        StepConfig,
        ModelInfo
    )
    UNIFIED_UTILS_AVAILABLE = True
    logger.info("✅ 통합 유틸리티 시스템 로드 완료")
except ImportError as e:
    UNIFIED_UTILS_AVAILABLE = False
    logger.warning(f"⚠️ 통합 유틸리티 시스템 사용 불가: {e}")
    
    # 폴백 함수들
    def get_step_model_interface(step_name: str, model_loader_instance=None):
        """폴백 함수"""
        logger.warning(f"⚠️ 폴백 모드: {step_name} 인터페이스")
        return {
            "step_name": step_name,
            "error": "통합 유틸리티 시스템 사용 불가",
            "get_model": lambda: None,
            "list_available_models": lambda: []
        }
    
    def initialize_global_utils(**kwargs):
        return {"success": False, "error": "통합 유틸리티 시스템 사용 불가"}

# 2. ModelLoader 시스템
try:
    from .utils.model_loader import (
        ModelLoader,
        get_global_model_loader,
        initialize_global_model_loader,
        SafeModelService,
        StepModelConfig
    )
    MODEL_LOADER_AVAILABLE = True
    logger.info("✅ ModelLoader 시스템 로드 완료")
except ImportError as e:
    MODEL_LOADER_AVAILABLE = False
    logger.warning(f"⚠️ ModelLoader 시스템 사용 불가: {e}")
    
    # 폴백 클래스
    class ModelLoader:
        def __init__(self, **kwargs):
            self.logger = logging.getLogger("fallback.ModelLoader")
        
        def get_model(self, model_name):
            return None

# 3. 메모리 관리 시스템
try:
    from .utils.memory_manager import (
        MemoryManager,
        GPUMemoryManager,
        get_global_memory_manager
    )
    MEMORY_MANAGER_AVAILABLE = True
    logger.info("✅ 메모리 관리 시스템 로드 완료")
except ImportError as e:
    MEMORY_MANAGER_AVAILABLE = False
    logger.warning(f"⚠️ 메모리 관리 시스템 사용 불가: {e}")

# 4. 데이터 변환 시스템
try:
    from .utils.data_converter import (
        DataConverter,
        get_global_data_converter,
        ImageProcessor
    )
    DATA_CONVERTER_AVAILABLE = True
    logger.info("✅ 데이터 변환 시스템 로드 완료")
except ImportError as e:
    DATA_CONVERTER_AVAILABLE = False
    logger.warning(f"⚠️ 데이터 변환 시스템 사용 불가: {e}")

# ==============================================
# 🔥 AI 파이프라인 Steps Import
# ==============================================

# Step 클래스들 Import
AI_STEPS_AVAILABLE = False
_step_classes = {}

try:
    from .steps.step_01_human_parsing import HumanParsingStep, create_human_parsing_step
    from .steps.step_02_pose_estimation import PoseEstimationStep, create_pose_estimation_step
    from .steps.step_03_cloth_segmentation import ClothSegmentationStep, create_cloth_segmentation_step
    from .steps.step_04_geometric_matching import GeometricMatchingStep, create_geometric_matching_step
    from .steps.step_05_cloth_warping import ClothWarpingStep, create_cloth_warping_step
    from .steps.step_06_virtual_fitting import VirtualFittingStep, create_virtual_fitting_step
    from .steps.step_07_post_processing import PostProcessingStep, create_post_processing_step
    from .steps.step_08_quality_assessment import QualityAssessmentStep, create_quality_assessment_step
    
    # Step 클래스 매핑
    _step_classes = {
        "step_01": HumanParsingStep,
        "step_02": PoseEstimationStep,
        "step_03": ClothSegmentationStep,
        "step_04": GeometricMatchingStep,
        "step_05": ClothWarpingStep,
        "step_06": VirtualFittingStep,
        "step_07": PostProcessingStep,
        "step_08": QualityAssessmentStep,
        
        # 이름으로도 접근 가능
        "HumanParsingStep": HumanParsingStep,
        "PoseEstimationStep": PoseEstimationStep,
        "ClothSegmentationStep": ClothSegmentationStep,
        "GeometricMatchingStep": GeometricMatchingStep,
        "ClothWarpingStep": ClothWarpingStep,
        "VirtualFittingStep": VirtualFittingStep,
        "PostProcessingStep": PostProcessingStep,
        "QualityAssessmentStep": QualityAssessmentStep
    }
    
    AI_STEPS_AVAILABLE = True
    logger.info("✅ 8단계 AI Steps 로드 완료")
    
except ImportError as e:
    logger.warning(f"⚠️ AI Steps 일부 로드 실패: {e}")
    
    # 개별 Step들 선택적 로드
    try:
        from .steps.step_01_human_parsing import HumanParsingStep, create_human_parsing_step
        _step_classes["step_01"] = HumanParsingStep
        _step_classes["HumanParsingStep"] = HumanParsingStep
        logger.info("✅ Step 01 Human Parsing 로드 완료")
    except ImportError:
        logger.warning("⚠️ Step 01 Human Parsing 로드 실패")
    
    try:
        from .steps.step_02_pose_estimation import PoseEstimationStep, create_pose_estimation_step
        _step_classes["step_02"] = PoseEstimationStep
        _step_classes["PoseEstimationStep"] = PoseEstimationStep
        logger.info("✅ Step 02 Pose Estimation 로드 완료")
    except ImportError:
        logger.warning("⚠️ Step 02 Pose Estimation 로드 실패")
    
    try:
        from .steps.step_03_cloth_segmentation import ClothSegmentationStep, create_cloth_segmentation_step
        _step_classes["step_03"] = ClothSegmentationStep
        _step_classes["ClothSegmentationStep"] = ClothSegmentationStep
        logger.info("✅ Step 03 Cloth Segmentation 로드 완료")
    except ImportError:
        logger.warning("⚠️ Step 03 Cloth Segmentation 로드 실패")
    
    # 나머지 Step들도 동일하게...
    for step_num in range(4, 9):
        step_names = {
            4: ("geometric_matching", "GeometricMatchingStep"),
            5: ("cloth_warping", "ClothWarpingStep"),
            6: ("virtual_fitting", "VirtualFittingStep"),
            7: ("post_processing", "PostProcessingStep"),
            8: ("quality_assessment", "QualityAssessmentStep")
        }
        
        step_module, step_class = step_names[step_num]
        try:
            module = __import__(f".steps.step_{step_num:02d}_{step_module}", fromlist=[step_class])
            step_cls = getattr(module, step_class)
            _step_classes[f"step_{step_num:02d}"] = step_cls
            _step_classes[step_class] = step_cls
            logger.info(f"✅ Step {step_num:02d} {step_class} 로드 완료")
        except ImportError:
            logger.warning(f"⚠️ Step {step_num:02d} {step_class} 로드 실패")

# ==============================================
# 🔥 PipelineManager Import
# ==============================================

try:
    from .pipeline_manager import (
        PipelineManager,
        create_pipeline,
        create_m3_max_pipeline,
        create_production_pipeline,
        create_development_pipeline,
        create_testing_pipeline,
        get_global_pipeline_manager
    )
    PIPELINE_MANAGER_AVAILABLE = True
    logger.info("✅ PipelineManager 로드 완료")
except ImportError as e:
    PIPELINE_MANAGER_AVAILABLE = False
    logger.warning(f"⚠️ PipelineManager 로드 실패: {e}")
    
    # 폴백 PipelineManager
    class PipelineManager:
        """폴백 PipelineManager"""
        def __init__(self, config: Optional[PipelineConfig] = None, **kwargs):
            self.config = config or PipelineConfig()
            self.logger = logging.getLogger("fallback.PipelineManager")
            self.is_initialized = False
        
        async def initialize(self) -> bool:
            self.is_initialized = True
            return True
        
        async def process_complete_pipeline(self, inputs: Dict[str, Any]) -> ProcessingResult:
            return ProcessingResult(
                success=False,
                message="PipelineManager 폴백 모드",
                processing_time=0.0,
                confidence=0.0,
                error="실제 PipelineManager를 사용할 수 없습니다"
            )
        
        async def cleanup(self):
            pass
    
    def create_m3_max_pipeline(**kwargs) -> PipelineManager:
        return PipelineManager(**kwargs)
    
    def create_production_pipeline(**kwargs) -> PipelineManager:
        return PipelineManager(**kwargs)

# ==============================================
# 🔥 팩토리 함수들
# ==============================================

def get_step_class(step_name: Union[str, int]) -> Optional[Type]:
    """Step 클래스 반환"""
    try:
        if isinstance(step_name, int):
            step_key = f"step_{step_name:02d}"
        else:
            step_key = step_name
        
        return _step_classes.get(step_key)
    except Exception as e:
        logger.error(f"Step 클래스 조회 실패 {step_name}: {e}")
        return None

def create_step_instance(step_name: Union[str, int], **kwargs) -> Optional[Any]:
    """Step 인스턴스 생성"""
    try:
        step_class = get_step_class(step_name)
        if step_class is None:
            logger.error(f"Step 클래스를 찾을 수 없음: {step_name}")
            return None
        
        # 기본 설정 추가
        default_config = {
            "device": SYSTEM_INFO["device"],
            "is_m3_max": SYSTEM_INFO["is_m3_max"],
            "memory_gb": SYSTEM_INFO["memory_gb"]
        }
        default_config.update(kwargs)
        
        return step_class(**default_config)
        
    except Exception as e:
        logger.error(f"Step 인스턴스 생성 실패 {step_name}: {e}")
        return None

def list_available_steps() -> List[str]:
    """사용 가능한 Step 목록 반환"""
    return list(_step_classes.keys())

def get_pipeline_status() -> Dict[str, Any]:
    """파이프라인 시스템 상태 반환"""
    return {
        "version": __version__,
        "system_info": SYSTEM_INFO,
        "availability": {
            "unified_utils": UNIFIED_UTILS_AVAILABLE,
            "model_loader": MODEL_LOADER_AVAILABLE,
            "memory_manager": MEMORY_MANAGER_AVAILABLE,
            "data_converter": DATA_CONVERTER_AVAILABLE,
            "ai_steps": AI_STEPS_AVAILABLE,
            "pipeline_manager": PIPELINE_MANAGER_AVAILABLE,
            "core_libs": CORE_LIBS_AVAILABLE
        },
        "available_steps": list_available_steps(),
        "step_count": len(_step_classes)
    }

# ==============================================
# 🔥 초기화 함수들
# ==============================================

async def initialize_pipeline_system(**kwargs) -> Dict[str, Any]:
    """전체 파이프라인 시스템 초기화"""
    try:
        start_time = time.time()
        results = {}
        
        # 1. 통합 유틸리티 시스템 초기화
        if UNIFIED_UTILS_AVAILABLE:
            try:
                utils_result = initialize_global_utils(**kwargs)
                results["unified_utils"] = utils_result
                logger.info("✅ 통합 유틸리티 시스템 초기화 완료")
            except Exception as e:
                logger.error(f"❌ 통합 유틸리티 시스템 초기화 실패: {e}")
                results["unified_utils"] = {"success": False, "error": str(e)}
        
        # 2. ModelLoader 초기화
        if MODEL_LOADER_AVAILABLE:
            try:
                model_loader_result = initialize_global_model_loader(**kwargs)
                results["model_loader"] = model_loader_result
                logger.info("✅ ModelLoader 시스템 초기화 완료")
            except Exception as e:
                logger.error(f"❌ ModelLoader 시스템 초기화 실패: {e}")
                results["model_loader"] = {"success": False, "error": str(e)}
        
        # 3. 파이프라인 매니저 초기화
        if PIPELINE_MANAGER_AVAILABLE:
            try:
                pipeline = create_m3_max_pipeline(**kwargs)
                await pipeline.initialize()
                results["pipeline_manager"] = {"success": True, "initialized": True}
                logger.info("✅ PipelineManager 초기화 완료")
            except Exception as e:
                logger.error(f"❌ PipelineManager 초기화 실패: {e}")
                results["pipeline_manager"] = {"success": False, "error": str(e)}
        
        initialization_time = time.time() - start_time
        
        return {
            "success": True,
            "initialization_time": initialization_time,
            "results": results,
            "system_status": get_pipeline_status()
        }
        
    except Exception as e:
        logger.error(f"❌ 파이프라인 시스템 초기화 실패: {e}")
        return {
            "success": False,
            "error": str(e),
            "system_status": get_pipeline_status()
        }

async def cleanup_pipeline_system():
    """전체 파이프라인 시스템 정리"""
    try:
        # 통합 유틸리티 시스템 정리
        if UNIFIED_UTILS_AVAILABLE:
            try:
                await reset_global_utils()
                logger.info("✅ 통합 유틸리티 시스템 정리 완료")
            except Exception as e:
                logger.error(f"❌ 통합 유틸리티 시스템 정리 실패: {e}")
        
        # 메모리 정리
        try:
            import gc
            gc.collect()
            
            if CORE_LIBS_AVAILABLE and torch is not None:
                if torch.backends.mps.is_available():
                    try:
                        if hasattr(torch.mps, 'empty_cache'):
                            safe_mps_empty_cache()
                    except:
                        pass
                elif torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            logger.info("✅ 메모리 정리 완료")
        except Exception as e:
            logger.error(f"❌ 메모리 정리 실패: {e}")
        
        logger.info("🎉 파이프라인 시스템 정리 완료")
        
    except Exception as e:
        logger.error(f"❌ 파이프라인 시스템 정리 실패: {e}")

# ==============================================
# 🔥 __all__ 정의
# ==============================================

__all__ = [
    # 🎯 버전 정보
    "__version__",
    "__author__",
    "__description__",
    
    # 📊 시스템 정보
    "SYSTEM_INFO",
    "get_pipeline_status",
    
    # 🔧 설정 클래스들
    "PipelineConfig",
    "ProcessingResult",
    "PipelineMode",
    "QualityLevel",
    
    # 🏗️ 팩토리 함수들
    "get_step_class",
    "create_step_instance",
    "list_available_steps",
    
    # 🚀 초기화 함수들
    "initialize_pipeline_system",
    "cleanup_pipeline_system",
    
    # 📦 Step 클래스들 (사용 가능한 것들만)
    *[class_name for class_name in _step_classes.keys() if not class_name.startswith("step_")]
]

# 조건부 export
if PIPELINE_MANAGER_AVAILABLE:
    __all__.extend([
        "PipelineManager",
        "create_pipeline",
        "create_m3_max_pipeline",
        "create_production_pipeline",
        "create_development_pipeline",
        "create_testing_pipeline",
        "get_global_pipeline_manager"
    ])

if UNIFIED_UTILS_AVAILABLE:
    __all__.extend([
        "get_utils_manager",
        "initialize_global_utils",
        "get_system_status",
        "reset_global_utils",
        "get_step_model_interface",
        "UnifiedUtilsManager",
        "UnifiedStepInterface",
        "StepModelInterface"
    ])

if MODEL_LOADER_AVAILABLE:
    __all__.extend([
        "ModelLoader",
        "get_global_model_loader",
        "initialize_global_model_loader"
    ])

# Step 생성 함수들 (사용 가능한 것들만)
step_creators = []
if "create_human_parsing_step" in globals():
    step_creators.append("create_human_parsing_step")
if "create_pose_estimation_step" in globals():
    step_creators.append("create_pose_estimation_step")
if "create_cloth_segmentation_step" in globals():
    step_creators.append("create_cloth_segmentation_step")
if "create_geometric_matching_step" in globals():
    step_creators.append("create_geometric_matching_step")
if "create_cloth_warping_step" in globals():
    step_creators.append("create_cloth_warping_step")
if "create_virtual_fitting_step" in globals():
    step_creators.append("create_virtual_fitting_step")
if "create_post_processing_step" in globals():
    step_creators.append("create_post_processing_step")
if "create_quality_assessment_step" in globals():
    step_creators.append("create_quality_assessment_step")

__all__.extend(step_creators)

# ==============================================
# 🔥 모듈 초기화 완료 로깅
# ==============================================

logger.info("=" * 80)
logger.info("🍎 MyCloset AI 파이프라인 시스템 v6.0 로드 완료")
logger.info("=" * 80)
logger.info(f"🔧 시스템: {SYSTEM_INFO['platform']} / {SYSTEM_INFO['device']}")
logger.info(f"🍎 M3 Max: {'✅' if SYSTEM_INFO['is_m3_max'] else '❌'}")
logger.info(f"💾 메모리: {SYSTEM_INFO['memory_gb']}GB")
logger.info(f"🧠 CPU 코어: {SYSTEM_INFO['cpu_count']}개")
logger.info(f"🐍 Python: {SYSTEM_INFO['python_version']}")
logger.info("=" * 80)
logger.info("📦 모듈 가용성:")
logger.info(f"   - 통합 유틸리티: {'✅' if UNIFIED_UTILS_AVAILABLE else '❌'}")
logger.info(f"   - ModelLoader: {'✅' if MODEL_LOADER_AVAILABLE else '❌'}")
logger.info(f"   - 메모리 관리: {'✅' if MEMORY_MANAGER_AVAILABLE else '❌'}")
logger.info(f"   - 데이터 변환: {'✅' if DATA_CONVERTER_AVAILABLE else '❌'}")
logger.info(f"   - AI Steps: {'✅' if AI_STEPS_AVAILABLE else '❌'} ({len(_step_classes)}개)")
logger.info(f"   - PipelineManager: {'✅' if PIPELINE_MANAGER_AVAILABLE else '❌'}")
logger.info(f"   - 핵심 라이브러리: {'✅' if CORE_LIBS_AVAILABLE else '❌'}")
logger.info("=" * 80)
logger.info("🎯 사용 가능한 Steps:")
for step_name in sorted([k for k in _step_classes.keys() if not k.startswith("step_")]):
    logger.info(f"   - {step_name}")
logger.info("=" * 80)
logger.info("🚀 초기화 준비 완료! initialize_pipeline_system() 호출하여 시작하세요.")
logger.info("=" * 80)

# 종료 시 정리 등록
import atexit

def _cleanup_on_exit():
    """종료 시 정리"""
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(cleanup_pipeline_system())
        loop.close()
    except Exception as e:
        logger.warning(f"⚠️ 종료 시 정리 실패: {e}")

atexit.register(_cleanup_on_exit)