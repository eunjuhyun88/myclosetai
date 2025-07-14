# app/ai_pipeline/utils/__init__.py
"""
AI 파이프라인 유틸리티 모듈 - 최적 생성자 패턴 적용
단순함 + 편의성 + 확장성 + 일관성
"""

import logging
from typing import Dict, Any, Optional

# 최적 생성자 패턴 기반 유틸리티들 import
try:
    from .memory_manager import (
        MemoryManager,
        create_memory_manager,
        get_global_memory_manager,
        initialize_global_memory_manager
    )
    MEMORY_MANAGER_AVAILABLE = True
except ImportError as e:
    MEMORY_MANAGER_AVAILABLE = False
    logging.warning(f"⚠️ MemoryManager import 실패: {e}")

try:
    from .model_loader import (
        ModelLoader,
        ModelConfig,
        ModelFormat,
        create_model_loader,
        get_global_model_loader,
        initialize_global_model_loader
    )
    MODEL_LOADER_AVAILABLE = True
except ImportError as e:
    MODEL_LOADER_AVAILABLE = False
    logging.warning(f"⚠️ ModelLoader import 실패: {e}")

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
except ImportError as e:
    DATA_CONVERTER_AVAILABLE = False
    logging.warning(f"⚠️ DataConverter import 실패: {e}")

# 로깅 설정
logger = logging.getLogger(__name__)

class OptimalUtilsManager:
    """
    🍎 최적 생성자 패턴 기반 유틸리티 매니저
    모든 유틸리티를 통일된 인터페이스로 관리
    """
    
    def __init__(
        self,
        device: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """
        ✅ 최적 생성자 - 유틸리티 매니저

        Args:
            device: 사용할 디바이스 (None=자동감지, 'cpu', 'cuda', 'mps')
            config: 유틸리티 설정 딕셔너리
            **kwargs: 확장 파라미터들
                - auto_initialize: bool = True  # 자동 초기화
                - memory_gb: float = 16.0  # 메모리 크기
                - is_m3_max: bool = False  # M3 Max 여부
                - optimization_enabled: bool = True  # 최적화 활성화
        """
        self.device = device
        self.config = config or {}
        self.kwargs = kwargs
        
        # 유틸리티 인스턴스들
        self.memory_manager = None
        self.model_loader = None
        self.data_converter = None
        
        # 초기화 상태
        self.initialized = False
        
        # 자동 초기화
        if kwargs.get('auto_initialize', True):
            self.initialize_all()

    def initialize_all(self) -> Dict[str, bool]:
        """모든 유틸리티 초기화"""
        results = {}
        
        try:
            # 1. Memory Manager 초기화
            if MEMORY_MANAGER_AVAILABLE:
                try:
                    self.memory_manager = MemoryManager(
                        device=self.device,
                        config=self.config.get('memory_manager', {}),
                        **self.kwargs
                    )
                    results['memory_manager'] = True
                    logger.info("✅ MemoryManager 초기화 성공")
                except Exception as e:
                    logger.error(f"❌ MemoryManager 초기화 실패: {e}")
                    results['memory_manager'] = False
            else:
                results['memory_manager'] = False
            
            # 2. Model Loader 초기화
            if MODEL_LOADER_AVAILABLE:
                try:
                    self.model_loader = ModelLoader(
                        device=self.device,
                        config=self.config.get('model_loader', {}),
                        **self.kwargs
                    )
                    results['model_loader'] = True
                    logger.info("✅ ModelLoader 초기화 성공")
                except Exception as e:
                    logger.error(f"❌ ModelLoader 초기화 실패: {e}")
                    results['model_loader'] = False
            else:
                results['model_loader'] = False
            
            # 3. Data Converter 초기화
            if DATA_CONVERTER_AVAILABLE:
                try:
                    self.data_converter = DataConverter(
                        device=self.device,
                        config=self.config.get('data_converter', {}),
                        **self.kwargs
                    )
                    results['data_converter'] = True
                    logger.info("✅ DataConverter 초기화 성공")
                except Exception as e:
                    logger.error(f"❌ DataConverter 초기화 실패: {e}")
                    results['data_converter'] = False
            else:
                results['data_converter'] = False
            
            # 초기화 결과 확인
            success_count = sum(results.values())
            total_count = len(results)
            
            self.initialized = success_count > 0
            
            logger.info(f"🔧 유틸리티 초기화 완료: {success_count}/{total_count} 성공")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ 유틸리티 매니저 초기화 실패: {e}")
            return {"error": str(e)}

    def get_memory_manager(self) -> Optional[MemoryManager]:
        """메모리 매니저 반환"""
        return self.memory_manager

    def get_model_loader(self) -> Optional[ModelLoader]:
        """모델 로더 반환"""
        return self.model_loader

    def get_data_converter(self) -> Optional[DataConverter]:
        """데이터 변환기 반환"""
        return self.data_converter

    def get_all_utils(self) -> Dict[str, Any]:
        """모든 유틸리티 인스턴스 반환"""
        return {
            'memory_manager': self.memory_manager,
            'model_loader': self.model_loader,
            'data_converter': self.data_converter
        }

    def get_utils_info(self) -> Dict[str, Any]:
        """유틸리티 정보 조회"""
        info = {
            "manager_initialized": self.initialized,
            "device": self.device,
            "config_keys": list(self.config.keys()),
            "available_utils": {
                "memory_manager": MEMORY_MANAGER_AVAILABLE and self.memory_manager is not None,
                "model_loader": MODEL_LOADER_AVAILABLE and self.model_loader is not None,
                "data_converter": DATA_CONVERTER_AVAILABLE and self.data_converter is not None
            },
            "library_status": {
                "memory_manager": MEMORY_MANAGER_AVAILABLE,
                "model_loader": MODEL_LOADER_AVAILABLE,
                "data_converter": DATA_CONVERTER_AVAILABLE
            }
        }
        
        # 각 유틸리티의 상세 정보 추가
        if self.memory_manager:
            try:
                info["memory_manager_info"] = {
                    "device": self.memory_manager.device,
                    "memory_limit_gb": self.memory_manager.memory_limit_gb,
                    "is_m3_max": self.memory_manager.is_m3_max
                }
            except:
                pass
        
        if self.model_loader:
            try:
                info["model_loader_info"] = {
                    "device": self.model_loader.device,
                    "use_fp16": self.model_loader.use_fp16,
                    "max_cached_models": self.model_loader.max_cached_models
                }
            except:
                pass
        
        if self.data_converter:
            try:
                info["data_converter_info"] = {
                    "device": self.data_converter.device,
                    "default_size": self.data_converter.default_size,
                    "use_gpu_acceleration": self.data_converter.use_gpu_acceleration
                }
            except:
                pass
        
        return info

# 전역 유틸리티 매니저
_global_utils_manager: Optional[OptimalUtilsManager] = None

def get_global_utils_manager() -> Optional[OptimalUtilsManager]:
    """전역 유틸리티 매니저 반환"""
    global _global_utils_manager
    return _global_utils_manager

def initialize_global_utils(**kwargs) -> OptimalUtilsManager:
    """전역 유틸리티 매니저 초기화"""
    global _global_utils_manager
    _global_utils_manager = OptimalUtilsManager(**kwargs)
    return _global_utils_manager

# 편의 함수들 (최적 생성자 패턴 호환)
def create_optimal_utils(
    device: Optional[str] = None,
    memory_gb: float = 16.0,
    is_m3_max: bool = False,
    **kwargs
) -> OptimalUtilsManager:
    """최적 생성자 패턴으로 유틸리티 매니저 생성"""
    return OptimalUtilsManager(
        device=device,
        memory_gb=memory_gb,
        is_m3_max=is_m3_max,
        **kwargs
    )

# 빠른 접근 함수들
def get_memory_manager(**kwargs) -> Optional[MemoryManager]:
    """메모리 매니저 빠른 접근"""
    manager = get_global_utils_manager()
    if manager and manager.memory_manager:
        return manager.memory_manager
    
    # 전역 매니저가 없으면 개별 전역 인스턴스 확인
    if MEMORY_MANAGER_AVAILABLE:
        return get_global_memory_manager()
    
    return None

def get_model_loader(**kwargs) -> Optional[ModelLoader]:
    """모델 로더 빠른 접근"""
    manager = get_global_utils_manager()
    if manager and manager.model_loader:
        return manager.model_loader
    
    # 전역 매니저가 없으면 개별 전역 인스턴스 확인
    if MODEL_LOADER_AVAILABLE:
        return get_global_model_loader()
    
    return None

def get_data_converter(**kwargs) -> Optional[DataConverter]:
    """데이터 변환기 빠른 접근"""
    manager = get_global_utils_manager()
    if manager and manager.data_converter:
        return manager.data_converter
    
    # 전역 매니저가 없으면 개별 전역 인스턴스 확인
    if DATA_CONVERTER_AVAILABLE:
        return get_global_data_converter()
    
    return None

# 하위 호환성 별칭들
if MEMORY_MANAGER_AVAILABLE:
    # 기존 이름으로도 접근 가능
    GPUMemoryManager = MemoryManager

if MODEL_LOADER_AVAILABLE:
    # 기존 이름으로도 접근 가능
    pass

if DATA_CONVERTER_AVAILABLE:
    # 기존 이름으로도 접근 가능
    pass

# Export 목록 구성
__all__ = [
    # 메인 매니저
    'OptimalUtilsManager',
    'get_global_utils_manager',
    'initialize_global_utils',
    'create_optimal_utils',
    
    # 빠른 접근 함수들
    'get_memory_manager',
    'get_model_loader', 
    'get_data_converter',
    
    # 상태 확인
    'MEMORY_MANAGER_AVAILABLE',
    'MODEL_LOADER_AVAILABLE',
    'DATA_CONVERTER_AVAILABLE'
]

# 사용 가능한 유틸리티들을 동적으로 추가
if MEMORY_MANAGER_AVAILABLE:
    __all__.extend([
        'MemoryManager',
        'create_memory_manager',
        'get_global_memory_manager',
        'initialize_global_memory_manager'
    ])

if MODEL_LOADER_AVAILABLE:
    __all__.extend([
        'ModelLoader',
        'ModelConfig',
        'ModelFormat',
        'create_model_loader',
        'get_global_model_loader',
        'initialize_global_model_loader'
    ])

if DATA_CONVERTER_AVAILABLE:
    __all__.extend([
        'DataConverter',
        'create_data_converter',
        'get_global_data_converter',
        'initialize_global_data_converter',
        'quick_image_to_tensor',
        'quick_tensor_to_image'
    ])

# 초기화 로깅
logger.info("🔧 AI 파이프라인 유틸리티 모듈 로드 완료")
logger.info(f"📊 사용 가능한 유틸리티: MemoryManager({MEMORY_MANAGER_AVAILABLE}), ModelLoader({MODEL_LOADER_AVAILABLE}), DataConverter({DATA_CONVERTER_AVAILABLE})")

# 자동 전역 매니저 초기화 (선택적)
try:
    import os
    if os.getenv('AUTO_INIT_UTILS', 'false').lower() == 'true':
        initialize_global_utils()
        logger.info("🚀 전역 유틸리티 매니저 자동 초기화 완료")
except Exception as e:
    logger.warning(f"⚠️ 전역 유틸리티 매니저 자동 초기화 실패: {e}")

# 모듈 로드 완료 확인
if MEMORY_MANAGER_AVAILABLE or MODEL_LOADER_AVAILABLE or DATA_CONVERTER_AVAILABLE:
    logger.info("✅ 최적 생성자 패턴 AI 파이프라인 유틸리티 준비 완료")
else:
    logger.warning("⚠️ 모든 유틸리티 import 실패 - 폴백 모드로 동작")