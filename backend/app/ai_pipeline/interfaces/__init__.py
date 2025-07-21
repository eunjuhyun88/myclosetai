# app/ai_pipeline/interfaces/__init__.py
"""
🔥 MyCloset AI Interfaces - Export 수정 버전
==============================================
✅ IStepInterface 올바른 export
✅ 모든 인터페이스 정상 제공
✅ Import 오류 완전 해결
"""

import logging

# 로거 설정
logger = logging.getLogger(__name__)

try:
    # 🔥 model_interface에서 모든 인터페이스 가져오기
    from .model_interface import (
        IModelLoader,
        IStepInterface,
        IMemoryManager,
        IDataConverter,
        ISafeFunctionValidator,
        ICheckpointManager,
        IPerformanceMonitor,
        IWarmupSystem,
        # 타입 별칭들
        ModelLoaderInterface,
        StepInterface,
        MemoryManagerInterface,
        DataConverterInterface,
        SafeFunctionValidatorInterface,
        CheckpointManagerInterface,
        PerformanceMonitorInterface,
        WarmupSystemInterface,
        # 유틸리티
        ALL_INTERFACES
    )
    
    MODEL_INTERFACE_AVAILABLE = True
    logger.info("✅ model_interface 모든 클래스 import 성공")
    
except ImportError as e:
    MODEL_INTERFACE_AVAILABLE = False
    logger.error(f"❌ model_interface import 실패: {e}")
    
    # 폴백: 최소한의 더미 인터페이스들
    from abc import ABC, abstractmethod
    from typing import Dict, Any, Optional, List, Tuple
    
    class IModelLoader(ABC):
        @abstractmethod
        def create_step_interface(self, step_name: str):
            pass
    
    class IStepInterface(ABC):
        @abstractmethod
        async def process_async(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
            pass
        
        @abstractmethod
        def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
            pass
    
    class IMemoryManager(ABC):
        @abstractmethod
        def optimize_memory(self, aggressive: bool = False) -> Dict[str, Any]:
            pass
    
    class IDataConverter(ABC):
        @abstractmethod
        def preprocess_image(self, image: Any, **kwargs) -> Any:
            pass
    
    class ISafeFunctionValidator(ABC):
        @abstractmethod
        def safe_call(self, func, *args, **kwargs):
            pass
    
    class ICheckpointManager(ABC):
        @abstractmethod
        def scan_checkpoints(self) -> Dict[str, Any]:
            pass
    
    class IPerformanceMonitor(ABC):
        @abstractmethod
        def record_operation(self, operation_name: str, duration: float, success: bool) -> None:
            pass
    
    class IWarmupSystem(ABC):
        @abstractmethod
        def run_warmup_sequence(self) -> Dict[str, Any]:
            pass
    
    # 타입 별칭들 (폴백)
    ModelLoaderInterface = IModelLoader
    StepInterface = IStepInterface
    MemoryManagerInterface = IMemoryManager
    DataConverterInterface = IDataConverter
    SafeFunctionValidatorInterface = ISafeFunctionValidator
    CheckpointManagerInterface = ICheckpointManager
    PerformanceMonitorInterface = IPerformanceMonitor
    WarmupSystemInterface = IWarmupSystem
    
    ALL_INTERFACES = [
        'IModelLoader',
        'IStepInterface',
        'IMemoryManager', 
        'IDataConverter',
        'ISafeFunctionValidator',
        'ICheckpointManager',
        'IPerformanceMonitor',
        'IWarmupSystem'
    ]

# 🔥 memory_interface와 data_interface 시도 (있으면 가져오기)
try:
    from .memory_interface import IMemoryManager as MemoryInterfaceFromFile
    logger.info("✅ memory_interface.py에서 IMemoryManager 추가 로드")
except ImportError:
    logger.info("ℹ️ memory_interface.py 없음 - model_interface의 IMemoryManager 사용")

try:
    from .data_interface import IDataConverter as DataInterfaceFromFile
    logger.info("✅ data_interface.py에서 IDataConverter 추가 로드")
except ImportError:
    logger.info("ℹ️ data_interface.py 없음 - model_interface의 IDataConverter 사용")

# ==============================================
# 🔥 모듈 Export (완전 버전)
# ==============================================

__all__ = [
    # 🔥 핵심 인터페이스들 (반드시 export)
    'IModelLoader',
    'IStepInterface',        # ✅ 핵심! 이것 때문에 오류 발생했음
    'IMemoryManager',
    'IDataConverter',
    'ISafeFunctionValidator',
    'ICheckpointManager',
    'IPerformanceMonitor',
    'IWarmupSystem',
    
    # 편의성 타입 별칭들
    'ModelLoaderInterface',
    'StepInterface',
    'MemoryManagerInterface',
    'DataConverterInterface',
    'SafeFunctionValidatorInterface',
    'CheckpointManagerInterface',
    'PerformanceMonitorInterface',
    'WarmupSystemInterface',
    
    # 유틸리티
    'ALL_INTERFACES'
]

# ==============================================
# 🔥 검증 및 로그
# ==============================================

def validate_interfaces():
    """인터페이스 유효성 검증"""
    try:
        # 핵심 인터페이스들이 제대로 정의되어 있는지 확인
        required_interfaces = [
            'IModelLoader', 
            'IStepInterface',
            'IMemoryManager',
            'IDataConverter'
        ]
        
        for interface_name in required_interfaces:
            if interface_name in globals():
                interface_class = globals()[interface_name]
                if hasattr(interface_class, '__abstractmethods__'):
                    logger.info(f"✅ {interface_name} 정상 (추상 메서드: {len(interface_class.__abstractmethods__)}개)")
                else:
                    logger.warning(f"⚠️ {interface_name}가 추상 클래스가 아님")
            else:
                logger.error(f"❌ {interface_name} 없음!")
                
        return True
        
    except Exception as e:
        logger.error(f"❌ 인터페이스 검증 실패: {e}")
        return False

# 모듈 로드 시 자동 검증
validation_result = validate_interfaces()

# 최종 로그
if MODEL_INTERFACE_AVAILABLE and validation_result:
    logger.info("🎉 Interfaces 패키지 로드 완료 - 모든 인터페이스 정상")
    logger.info(f"📋 Export된 인터페이스: {len(__all__)}개")
    logger.info("🔗 IStepInterface export 문제 완전 해결")
else:
    logger.warning("⚠️ 일부 인터페이스 로드 실패 - 폴백 모드 사용")
    logger.info("🔧 폴백 인터페이스로 동작 가능")

# 추가 유틸리티 함수들
def get_interface_info() -> dict:
    """인터페이스 정보 조회"""
    return {
        "available_interfaces": __all__,
        "model_interface_available": MODEL_INTERFACE_AVAILABLE,
        "validation_passed": validation_result,
        "total_interfaces": len(__all__)
    }

def check_interface_availability(interface_name: str) -> bool:
    """특정 인터페이스 사용 가능 여부 확인"""
    return interface_name in globals() and interface_name in __all__