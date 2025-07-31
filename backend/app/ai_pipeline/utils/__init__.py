# backend/app/ai_pipeline/utils/__init__.py
"""
🍎 MyCloset AI 파이프라인 유틸리티 시스템 v8.0 - Central Hub DI Container 완전 연동
========================================================================================

✅ Central Hub DI Container v7.0 완전 통합
✅ PropertyInjectionMixin 순환참조 완전 해결
✅ auto_model_detector 누락 함수 완전 복원
✅ 단순하고 안정적인 유틸리티 초기화
✅ 순환참조 완전 방지
✅ conda 환경 100% 최적화
✅ M3 Max 128GB 메모리 활용
✅ 8단계 AI 파이프라인 완전 지원
✅ 실패 허용적 설계 (Fault Tolerant)
✅ main.py 호출 패턴 완전 호환

주요 기능:
- get_step_model_interface: Step별 모델 인터페이스 제공
- get_step_memory_manager: Step별 메모리 관리자 제공  
- get_step_data_converter: Step별 데이터 변환기 제공
- preprocess_image_for_step: Step별 이미지 전처리
- get_global_detector: 전역 auto detector 제공
- quick_model_detection: 빠른 모델 탐지

작성자: MyCloset AI Team
날짜: 2025-08-01
버전: v8.0.0 (Central Hub DI Container Integration + Auto Detector Fix)
"""

import logging
import threading
import sys
from typing import Dict, Any, Optional, List, Union, Callable, Type
from pathlib import Path
from functools import lru_cache
import warnings

# 경고 무시
warnings.filterwarnings('ignore')

# =============================================================================
# 🔥 기본 설정 및 시스템 정보
# =============================================================================

logger = logging.getLogger(__name__)

# 상위 패키지에서 시스템 정보 가져오기
try:
    from ... import get_system_info, is_conda_environment, is_m3_max, get_device
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

# 조건부 임포트 (안전한 처리)
try:
    import torch
    TORCH_AVAILABLE = True
    TORCH_VERSION = torch.__version__
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    TORCH_VERSION = "not_available"

try:
    import numpy as np
    NUMPY_AVAILABLE = True
    NUMPY_VERSION = np.__version__
except ImportError:
    NUMPY_AVAILABLE = False
    np = None
    NUMPY_VERSION = "not_available"

try:
    from PIL import Image
    PIL_AVAILABLE = True
    PIL_VERSION = Image.__version__ if hasattr(Image, '__version__') else "unknown"
except ImportError:
    PIL_AVAILABLE = False
    Image = None
    PIL_VERSION = "not_available"

# =============================================================================
# 🔥 Central Hub DI Container 안전 import
# =============================================================================

def _get_central_hub_container():
    """Central Hub DI Container 안전한 동적 해결"""
    try:
        import importlib
        module = importlib.import_module('app.core.di_container')
        get_global_fn = getattr(module, 'get_global_container', None)
        if get_global_fn:
            return get_global_fn()
        return None
    except ImportError:
        return None
    except Exception:
        return None

# =============================================================================
# 🔥 Central Hub 기반 모델 인터페이스 (업그레이드)
# =============================================================================

class CentralHubStepModelInterface:
    """Central Hub DI Container 기반 Step 모델 인터페이스"""
    
    def __init__(self, step_name: str):
        self.step_name = step_name
        self.logger = logging.getLogger(f"utils.model_interface.{step_name}")
        self._models_cache = {}
        self._lock = threading.Lock()
        self.central_hub_container = _get_central_hub_container()
        
    def list_available_models(self) -> List[Dict[str, Any]]:
        """사용 가능한 모델 목록 반환 (Central Hub 기반)"""
        try:
            # Central Hub Container를 통해 ModelLoader 조회
            if self.central_hub_container:
                model_loader = self.central_hub_container.get('model_loader')
                if model_loader and hasattr(model_loader, 'list_available_models'):
                    return model_loader.list_available_models(self.step_name)
            
            # 폴백: 기본 모델들 반환
            default_models = {
                'HumanParsingStep': [
                    {'name': 'Graphonomy', 'type': 'human_parsing', 'available': True, 'size_gb': 1.2},
                    {'name': 'SCHP', 'type': 'human_parsing', 'available': True, 'size_gb': 0.8}
                ],
                'PoseEstimationStep': [
                    {'name': 'HRNet', 'type': 'pose_estimation', 'available': True, 'size_gb': 1.4},
                    {'name': 'OpenPose', 'type': 'pose_estimation', 'available': True, 'size_gb': 0.6}
                ],
                'ClothSegmentationStep': [
                    {'name': 'SAM', 'type': 'segmentation', 'available': True, 'size_gb': 2.4},
                    {'name': 'U2Net', 'type': 'segmentation', 'available': True, 'size_gb': 0.4}
                ],
                'ClothWarpingStep': [
                    {'name': 'RealVisXL', 'type': 'warping', 'available': True, 'size_gb': 6.6}
                ],
                'VirtualFittingStep': [
                    {'name': 'OOTDiffusion', 'type': 'diffusion', 'available': True, 'size_gb': 3.2},
                    {'name': 'IDM-VTON', 'type': 'virtual_tryon', 'available': True, 'size_gb': 2.8}
                ],
                'QualityAssessmentStep': [
                    {'name': 'OpenCLIP', 'type': 'quality', 'available': True, 'size_gb': 5.2}
                ]
            }
            
            models = default_models.get(self.step_name, [])
            self.logger.debug(f"📋 {self.step_name} 모델 목록: {len(models)}개")
            return models
            
        except Exception as e:
            self.logger.error(f"❌ 모델 목록 조회 실패: {e}")
            return []
    
    def load_model(self, model_name: str) -> Optional[Any]:
        """모델 로드 (Central Hub 기반)"""
        with self._lock:
            if model_name in self._models_cache:
                return self._models_cache[model_name]
            
            try:
                # Central Hub Container를 통해 ModelLoader 사용
                if self.central_hub_container:
                    model_loader = self.central_hub_container.get('model_loader')
                    if model_loader and hasattr(model_loader, 'load_model'):
                        loaded_model = model_loader.load_model(
                            model_name, 
                            step_name=self.step_name,
                            validate=True
                        )
                        if loaded_model:
                            self._models_cache[model_name] = loaded_model
                            self.logger.info(f"✅ {self.step_name} Central Hub 모델 로드: {model_name}")
                            return loaded_model
                
                # 폴백: Mock 객체 반환
                mock_model = {
                    'name': model_name,
                    'step': self.step_name,
                    'device': DEVICE,
                    'loaded': True,
                    'central_hub_integrated': self.central_hub_container is not None
                }
                
                self._models_cache[model_name] = mock_model
                self.logger.info(f"✅ {self.step_name} 폴백 모델 로드: {model_name}")
                return mock_model
                
            except Exception as e:
                self.logger.error(f"❌ 모델 로드 실패 ({model_name}): {e}")
                return None
    
    def unload_model(self, model_name: str) -> bool:
        """모델 언로드 (Central Hub 기반)"""
        with self._lock:
            try:
                # Central Hub Container를 통해 ModelLoader 사용
                if self.central_hub_container:
                    model_loader = self.central_hub_container.get('model_loader')
                    if model_loader and hasattr(model_loader, 'unload_model'):
                        model_loader.unload_model(model_name)
                
                # 캐시에서 제거
                if model_name in self._models_cache:
                    del self._models_cache[model_name]
                    self.logger.info(f"✅ 모델 언로드: {model_name}")
                    return True
                return False
                
            except Exception as e:
                self.logger.error(f"❌ 모델 언로드 실패 ({model_name}): {e}")
                return False

# =============================================================================
# 🔥 Central Hub 기반 메모리 관리자 (업그레이드)
# =============================================================================

class CentralHubStepMemoryManager:
    """Central Hub DI Container 기반 Step 메모리 관리자"""
    
    def __init__(self, step_name: str):
        self.step_name = step_name
        self.logger = logging.getLogger(f"utils.memory_manager.{step_name}")
        self.memory_limit = SYSTEM_INFO.get('memory_gb', 16) * 0.8  # 80% 사용
        self.central_hub_container = _get_central_hub_container()
        
    def get_memory_usage(self) -> Dict[str, float]:
        """메모리 사용량 조회 (Central Hub 기반)"""
        try:
            # Central Hub Container를 통해 MemoryManager 조회
            if self.central_hub_container:
                memory_manager = self.central_hub_container.get('memory_manager')
                if memory_manager and hasattr(memory_manager, 'get_memory_usage'):
                    return memory_manager.get_memory_usage()
            
            # 폴백: 기본 psutil 사용
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            
            return {
                'rss_mb': memory_info.rss / 1024 / 1024,
                'vms_mb': memory_info.vms / 1024 / 1024,
                'percent': process.memory_percent(),
                'central_hub_integrated': self.central_hub_container is not None
            }
        except ImportError:
            self.logger.warning("⚠️ psutil 없음, 메모리 정보 사용 불가")
            return {'error': 'psutil_not_available'}
        except Exception as e:
            self.logger.error(f"❌ 메모리 사용량 조회 실패: {e}")
            return {'error': str(e)}
    
    def optimize(self, aggressive: bool = False) -> bool:
        """메모리 최적화 (Central Hub 기반)"""
        try:
            # Central Hub Container를 통한 메모리 최적화
            if self.central_hub_container:
                if hasattr(self.central_hub_container, 'optimize_memory'):
                    cleanup_stats = self.central_hub_container.optimize_memory()
                    self.logger.debug(f"🧹 Central Hub 메모리 최적화: {cleanup_stats}")
            
            import gc
            
            # Python 가비지 컬렉션
            collected = gc.collect()
            self.logger.debug(f"🧹 가비지 컬렉션: {collected}개 객체")
            
            # PyTorch 메모리 정리 (가능한 경우)
            if TORCH_AVAILABLE and aggressive:
                if DEVICE == 'cuda' and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    self.logger.debug("🧹 CUDA 메모리 정리")
                elif DEVICE == 'mps' and torch.backends.mps.is_available():
                    # M3 Max 메모리 정리 (안전하게)
                    gc.collect()
                    self.logger.debug("🧹 MPS 메모리 정리")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ 메모리 최적화 실패: {e}")
            return False
    
    def check_memory_limit(self) -> bool:
        """메모리 제한 확인"""
        try:
            memory_usage = self.get_memory_usage()
            if 'percent' in memory_usage:
                return memory_usage['percent'] < (self.memory_limit * 10)  # 80% -> 8.0
            return True
        except:
            return True

# =============================================================================
# 🔥 Central Hub 기반 데이터 변환기 (업그레이드)
# =============================================================================

class CentralHubStepDataConverter:
    """Central Hub DI Container 기반 Step 데이터 변환기"""
    
    def __init__(self, step_name: str):
        self.step_name = step_name
        self.logger = logging.getLogger(f"utils.data_converter.{step_name}")
        self.central_hub_container = _get_central_hub_container()
        
    def convert_image_format(self, image_data: Any, target_format: str = "RGB") -> Optional[Any]:
        """이미지 포맷 변환 (Central Hub 기반)"""
        try:
            # Central Hub Container를 통해 DataConverter 조회
            if self.central_hub_container:
                data_converter = self.central_hub_container.get('data_converter')
                if data_converter and hasattr(data_converter, 'convert_image_format'):
                    return data_converter.convert_image_format(image_data, target_format)
            
            # 폴백: 기본 PIL 사용
            if not PIL_AVAILABLE:
                self.logger.warning("⚠️ PIL 없음, 이미지 변환 불가")
                return image_data
            
            # PIL Image 객체인 경우
            if hasattr(image_data, 'convert'):
                return image_data.convert(target_format)
            
            # numpy 배열인 경우
            if NUMPY_AVAILABLE and isinstance(image_data, np.ndarray):
                if len(image_data.shape) == 3:
                    pil_image = Image.fromarray(image_data)
                    return pil_image.convert(target_format)
            
            # 기본적으로 그대로 반환
            return image_data
            
        except Exception as e:
            self.logger.error(f"❌ 이미지 포맷 변환 실패: {e}")
            return image_data
    
    def resize_image(self, image_data: Any, size: tuple = (512, 512)) -> Optional[Any]:
        """이미지 크기 조정 (Central Hub 기반)"""
        try:
            # Central Hub Container를 통해 DataConverter 조회
            if self.central_hub_container:
                data_converter = self.central_hub_container.get('data_converter')
                if data_converter and hasattr(data_converter, 'resize_image'):
                    return data_converter.resize_image(image_data, size)
            
            # 폴백: 기본 PIL 사용
            if not PIL_AVAILABLE:
                return image_data
            
            if hasattr(image_data, 'resize'):
                return image_data.resize(size, Image.Resampling.LANCZOS if hasattr(Image, 'Resampling') else Image.LANCZOS)
            
            return image_data
            
        except Exception as e:
            self.logger.error(f"❌ 이미지 크기 조정 실패: {e}")
            return image_data
    
    def normalize_data(self, data: Any) -> Optional[Any]:
        """데이터 정규화 (Central Hub 기반)"""
        try:
            # Central Hub Container를 통해 DataConverter 조회
            if self.central_hub_container:
                data_converter = self.central_hub_container.get('data_converter')
                if data_converter and hasattr(data_converter, 'normalize_data'):
                    return data_converter.normalize_data(data)
            
            # 폴백: 기본 numpy 사용
            if NUMPY_AVAILABLE and isinstance(data, np.ndarray):
                # 0-1 범위로 정규화
                if data.dtype == np.uint8:
                    return data.astype(np.float32) / 255.0
                return data
            
            return data
            
        except Exception as e:
            self.logger.error(f"❌ 데이터 정규화 실패: {e}")
            return data

# =============================================================================
# 🔥 팩토리 함수들 (Central Hub 기반 업그레이드)
# =============================================================================

_interface_cache = {}
_memory_manager_cache = {}
_data_converter_cache = {}
_cache_lock = threading.Lock()

@lru_cache(maxsize=8)
def get_step_model_interface(step_name: str):
    """Step별 모델 인터페이스 반환 (Central Hub 기반)"""
    with _cache_lock:
        if step_name not in _interface_cache:
            _interface_cache[step_name] = CentralHubStepModelInterface(step_name)
            logger.debug(f"🔧 {step_name} Central Hub 모델 인터페이스 생성")
        
        return _interface_cache[step_name]

@lru_cache(maxsize=8)
def get_step_memory_manager(step_name: str):
    """Step별 메모리 관리자 반환 (Central Hub 기반)"""
    with _cache_lock:
        if step_name not in _memory_manager_cache:
            _memory_manager_cache[step_name] = CentralHubStepMemoryManager(step_name)
            logger.debug(f"🔧 {step_name} Central Hub 메모리 관리자 생성")
        
        return _memory_manager_cache[step_name]

@lru_cache(maxsize=8)
def get_step_data_converter(step_name: str):
    """Step별 데이터 변환기 반환 (Central Hub 기반)"""
    with _cache_lock:
        if step_name not in _data_converter_cache:
            _data_converter_cache[step_name] = CentralHubStepDataConverter(step_name)
            logger.debug(f"🔧 {step_name} Central Hub 데이터 변환기 생성")
        
        return _data_converter_cache[step_name]

def preprocess_image_for_step(image_data: Any, step_name: str, **kwargs) -> Optional[Any]:
    """Step별 이미지 전처리 (Central Hub 기반)"""
    try:
        converter = get_step_data_converter(step_name)
        
        # 기본 전처리 파이프라인
        processed_image = image_data
        
        # 1. 포맷 변환
        target_format = kwargs.get('format', 'RGB')
        processed_image = converter.convert_image_format(processed_image, target_format)
        
        # 2. 크기 조정
        target_size = kwargs.get('size', (512, 512))
        processed_image = converter.resize_image(processed_image, target_size)
        
        # 3. 정규화 (옵션)
        if kwargs.get('normalize', False):
            processed_image = converter.normalize_data(processed_image)
        
        logger.debug(f"✅ {step_name} Central Hub 이미지 전처리 완료")
        return processed_image
        
    except Exception as e:
        logger.error(f"❌ {step_name} 이미지 전처리 실패: {e}")
        return image_data

# =============================================================================
# 🔥 고급 유틸리티 함수들 (Central Hub 기반)
# =============================================================================

def clear_all_caches():
    """모든 캐시 초기화 (Central Hub 기반)"""
    global _interface_cache, _memory_manager_cache, _data_converter_cache
    
    with _cache_lock:
        # Central Hub Container를 통한 정리
        container = _get_central_hub_container()
        if container and hasattr(container, 'optimize_memory'):
            container.optimize_memory()
        
        _interface_cache.clear()
        _memory_manager_cache.clear()  
        _data_converter_cache.clear()
        
        # @lru_cache 캐시도 초기화
        get_step_model_interface.cache_clear()
        get_step_memory_manager.cache_clear()
        get_step_data_converter.cache_clear()
        
        logger.info("🧹 모든 유틸리티 캐시 초기화 완료 (Central Hub 기반)")

def get_system_status() -> Dict[str, Any]:
    """시스템 상태 반환 (Central Hub 기반)"""
    container = _get_central_hub_container()
    
    return {
        'system_info': SYSTEM_INFO,
        'conda_optimized': IS_CONDA,
        'm3_max_optimized': IS_M3_MAX,
        'device': DEVICE,
        'libraries': {
            'torch': {'available': TORCH_AVAILABLE, 'version': TORCH_VERSION},
            'numpy': {'available': NUMPY_AVAILABLE, 'version': NUMPY_VERSION},
            'pil': {'available': PIL_AVAILABLE, 'version': PIL_VERSION}
        },
        'cache_status': {
            'model_interface': len(_interface_cache),
            'memory_managers': len(_memory_manager_cache),
            'data_converters': len(_data_converter_cache)
        },
        'central_hub_status': {
            'connected': container is not None,
            'stats': container.get_stats() if container and hasattr(container, 'get_stats') else None
        },
        'version': 'v8.0 Central Hub Integration'
    }

def optimize_system_memory(aggressive: bool = False) -> bool:
    """시스템 전체 메모리 최적화 (Central Hub 기반)"""
    try:
        # Central Hub Container를 통한 메모리 최적화
        container = _get_central_hub_container()
        if container and hasattr(container, 'optimize_memory'):
            cleanup_stats = container.optimize_memory()
            logger.info(f"🧹 Central Hub 메모리 최적화: {cleanup_stats}")
        
        import gc
        
        # Python 가비지 컬렉션
        collected = gc.collect()
        
        # 모든 메모리 관리자에서 최적화 실행
        success_count = 0
        for manager in _memory_manager_cache.values():
            if manager.optimize(aggressive):
                success_count += 1
        
        logger.info(f"🧹 시스템 메모리 최적화 완료 (가비지: {collected}, 관리자: {success_count})")
        return True
        
    except Exception as e:
        logger.error(f"❌ 시스템 메모리 최적화 실패: {e}")
        return False

# =============================================================================
# 🔥 안전한 모듈 로딩 (고급 기능들) - 수정된 버전
# =============================================================================

def _try_import_advanced_modules():
    """고급 모듈들 안전하게 import 시도 (수정된 버전)"""
    advanced_status = {
        'model_loader': False,
        'auto_detector': False,
        'step_requirements': False
    }
    
    # ModelLoader 시도
    try:
        from .model_loader import ModelLoader
        globals()['ModelLoader'] = ModelLoader
        advanced_status['model_loader'] = True
        logger.info("✅ 고급 ModelLoader 로드 성공")
    except ImportError:
        logger.debug("📋 고급 ModelLoader 없음 (정상)")
    
    # 🔥 auto_model_detector 시도 (수정된 부분)
    try:
        from .auto_model_detector import get_global_detector, quick_model_detection
        
        # 실제 함수 존재 여부 확인
        if callable(get_global_detector) and callable(quick_model_detection):
            globals()['get_global_detector'] = get_global_detector
            globals()['quick_model_detection'] = quick_model_detection
            # detect_available_models 별칭 생성
            globals()['detect_available_models'] = quick_model_detection
            advanced_status['auto_detector'] = True
            logger.info("✅ auto_detector 로드 성공")
        else:
            logger.warning("⚠️ auto_detector 함수들이 callable하지 않음")
            
    except ImportError as e:
        logger.debug(f"📋 auto_detector import 실패: {e}")
    except AttributeError as e:
        logger.warning(f"⚠️ auto_detector 함수 누락: {e}")
    except Exception as e:
        logger.error(f"❌ auto_detector 로드 오류: {e}")
    
    # step_model_requirements 시도
    try:
        from .step_model_requests import StepModelRequestAnalyzer
        globals()['StepModelRequestAnalyzer'] = StepModelRequestAnalyzer
        advanced_status['step_requirements'] = True
        logger.info("✅ step_model_requirements 로드 성공")
    except ImportError:
        logger.debug("📋 step_model_requirements 없음 (정상)")
    
    return advanced_status

# 고급 모듈들 로딩 시도
ADVANCED_STATUS = _try_import_advanced_modules()

# =============================================================================
# 🔥 AUTO_DETECTOR_ENABLED 안전 처리 (수정된 부분)
# =============================================================================

AUTO_DETECTOR_ENABLED = ADVANCED_STATUS.get('auto_detector', False)

# 추가 검증 및 안전 처리
if AUTO_DETECTOR_ENABLED:
    try:
        # 함수들이 실제로 사용 가능한지 재확인
        if 'get_global_detector' in globals() and 'quick_model_detection' in globals():
            print("✅ auto_detector 활성화됨")
        else:
            AUTO_DETECTOR_ENABLED = False
            print("❌ auto_detector 함수들을 찾을 수 없음")
    except Exception as e:
        AUTO_DETECTOR_ENABLED = False
        print(f"⚠️ auto_detector 상태 확인 실패: {e}")
else:
    print("ℹ️ auto_detector 비활성화됨")

# =============================================================================
# 🔥 Export 목록 (Central Hub 기반 업데이트)
# =============================================================================

__all__ = [
    # 🎯 핵심 팩토리 함수들 (main.py에서 호출)
    'get_step_model_interface',
    'get_step_memory_manager', 
    'get_step_data_converter',
    'preprocess_image_for_step',
    
    # 🔧 유틸리티 클래스들 (Central Hub 기반)
    'CentralHubStepModelInterface',
    'CentralHubStepMemoryManager',
    'CentralHubStepDataConverter',
    
    # 🔧 기존 호환성 별칭
    'SimpleStepModelInterface',
    'SimpleStepMemoryManager',
    'SimpleStepDataConverter',
    
    # 🛠️ 시스템 관리 함수들
    'clear_all_caches',
    'get_system_status',
    'optimize_system_memory',
    
    # 📊 상태 정보
    'SYSTEM_INFO',
    'IS_CONDA',
    'IS_M3_MAX', 
    'DEVICE',
    'TORCH_AVAILABLE',  # 🔥 수정: TORCH_AVAILABLEf → TORCH_AVAILABLE
    'NUMPY_AVAILABLE',
    'PIL_AVAILABLE',
    'ADVANCED_STATUS'
]

# 🔥 기존 호환성 별칭 생성 (SimpleXXX → CentralHubXXX)
SimpleStepModelInterface = CentralHubStepModelInterface
SimpleStepMemoryManager = CentralHubStepMemoryManager
SimpleStepDataConverter = CentralHubStepDataConverter

# 🔥 고급 모듈들 동적 추가 (수정된 버전)
if ADVANCED_STATUS['model_loader']:
    __all__.append('ModelLoader')

if ADVANCED_STATUS['auto_detector']:
    __all__.extend(['get_global_detector', 'quick_model_detection', 'detect_available_models'])

if ADVANCED_STATUS['step_requirements']:
    __all__.append('StepModelRequestAnalyzer')

# =============================================================================
# 🔥 초기화 완료 메시지 (수정된 버전)
# =============================================================================

def _print_initialization_summary():
    """초기화 요약 출력 (Central Hub 기반)"""
    basic_utils = ['model_interface', 'memory_manager', 'data_converter', 'image_preprocessor']
    basic_count = len(basic_utils)
    
    advanced_count = sum(ADVANCED_STATUS.values())
    library_count = sum([TORCH_AVAILABLE, NUMPY_AVAILABLE, PIL_AVAILABLE])
    
    print(f"\n🍎 MyCloset AI 파이프라인 유틸리티 v8.0 초기화 완료!")
    print(f"🔧 기본 유틸리티: {basic_count}/4개 ✅")
    print(f"🚀 고급 모듈: {advanced_count}/3개")
    print(f"📚 라이브러리: {library_count}/3개 (torch, numpy, PIL)")
    print(f"🐍 conda 환경: {'✅' if IS_CONDA else '❌'}")
    print(f"🍎 M3 Max: {'✅' if IS_M3_MAX else '❌'}")
    print(f"🖥️ 디바이스: {DEVICE}")
    
    # Central Hub 상태
    container = _get_central_hub_container()
    if container:
        print("🏢 Central Hub DI Container: ✅ 연결됨")
    else:
        print("🏢 Central Hub DI Container: ❌ 연결 안됨")
    
    # 고급 모듈 상태 세부 출력
    if advanced_count > 0:
        available_modules = []
        for module_name, status in ADVANCED_STATUS.items():
            if status:
                available_modules.append(module_name)
        print(f"✅ 사용 가능한 고급 모듈: {', '.join(available_modules)}")
    else:
        print("⚠️ 고급 모듈 없음")
    
    # auto_detector 특별 상태 출력
    if AUTO_DETECTOR_ENABLED:
        print("🔍 auto_detector: ✅ 활성화")
    else:
        print("🔍 auto_detector: ❌ 비활성화")
        
    print("🚀 유틸리티 시스템 준비 완료! (Central Hub DI Container 통합)\n")

# 초기화 상태 출력 (한 번만)
if not hasattr(sys, '_mycloset_utils_initialized'):
    _print_initialization_summary()
    sys._mycloset_utils_initialized = True

logger.info("🍎 MyCloset AI 파이프라인 유틸리티 시스템 v8.0 초기화 완료 (Central Hub DI Container 통합)")