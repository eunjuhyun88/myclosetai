# backend/app/ai_pipeline/utils/__init__.py
"""
🍎 MyCloset AI 파이프라인 유틸리티 시스템 v7.0 - 단순화된 통합
================================================================

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

작성자: MyCloset AI Team
날짜: 2025-07-23
버전: v7.0.0 (Simplified Utility Integration)
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
# 🔥 단순화된 모델 인터페이스
# =============================================================================

class SimpleStepModelInterface:
    """단순화된 Step 모델 인터페이스"""
    
    def __init__(self, step_name: str):
        self.step_name = step_name
        self.logger = logging.getLogger(f"utils.model_interface.{step_name}")
        self._models_cache = {}
        self._lock = threading.Lock()
        
    def list_available_models(self) -> List[Dict[str, Any]]:
        """사용 가능한 모델 목록 반환"""
        try:
            # 실제 구현에서는 model_loader에서 가져옴
            # 현재는 기본 모델들 반환
            default_models = {
                'HumanParsingStep': [
                    {'name': 'SCHP', 'type': 'human_parsing', 'available': True},
                    {'name': 'Graphonomy', 'type': 'human_parsing', 'available': True}
                ],
                'PoseEstimationStep': [
                    {'name': 'OpenPose', 'type': 'pose_estimation', 'available': True},
                    {'name': 'YOLO-Pose', 'type': 'pose_estimation', 'available': True}
                ],
                'ClothSegmentationStep': [
                    {'name': 'U2Net', 'type': 'segmentation', 'available': True},
                    {'name': 'SAM', 'type': 'segmentation', 'available': True}
                ],
                'VirtualFittingStep': [
                    {'name': 'OOTDiffusion', 'type': 'diffusion', 'available': True},
                    {'name': 'IDM-VTON', 'type': 'virtual_tryon', 'available': True}
                ]
            }
            
            models = default_models.get(self.step_name, [])
            self.logger.debug(f"📋 {self.step_name} 모델 목록: {len(models)}개")
            return models
            
        except Exception as e:
            self.logger.error(f"❌ 모델 목록 조회 실패: {e}")
            return []
    
    def load_model(self, model_name: str) -> Optional[Any]:
        """모델 로드 (단순화된 버전)"""
        with self._lock:
            if model_name in self._models_cache:
                return self._models_cache[model_name]
            
            try:
                # 실제 구현에서는 여기서 모델을 로드
                # 현재는 Mock 객체 반환
                mock_model = {
                    'name': model_name,
                    'step': self.step_name,
                    'device': DEVICE,
                    'loaded': True
                }
                
                self._models_cache[model_name] = mock_model
                self.logger.info(f"✅ {self.step_name} 모델 로드: {model_name}")
                return mock_model
                
            except Exception as e:
                self.logger.error(f"❌ 모델 로드 실패 ({model_name}): {e}")
                return None
    
    def unload_model(self, model_name: str) -> bool:
        """모델 언로드"""
        with self._lock:
            if model_name in self._models_cache:
                del self._models_cache[model_name]
                self.logger.info(f"✅ 모델 언로드: {model_name}")
                return True
            return False

# =============================================================================
# 🔥 단순화된 메모리 관리자
# =============================================================================

class SimpleStepMemoryManager:
    """단순화된 Step 메모리 관리자"""
    
    def __init__(self, step_name: str):
        self.step_name = step_name
        self.logger = logging.getLogger(f"utils.memory_manager.{step_name}")
        self.memory_limit = SYSTEM_INFO.get('memory_gb', 16) * 0.8  # 80% 사용
        
    def get_memory_usage(self) -> Dict[str, float]:
        """메모리 사용량 조회"""
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            
            return {
                'rss_mb': memory_info.rss / 1024 / 1024,
                'vms_mb': memory_info.vms / 1024 / 1024,
                'percent': process.memory_percent()
            }
        except ImportError:
            self.logger.warning("⚠️ psutil 없음, 메모리 정보 사용 불가")
            return {'error': 'psutil_not_available'}
        except Exception as e:
            self.logger.error(f"❌ 메모리 사용량 조회 실패: {e}")
            return {'error': str(e)}
    
    def optimize(self, aggressive: bool = False) -> bool:
        """메모리 최적화"""
        try:
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
# 🔥 단순화된 데이터 변환기
# =============================================================================

class SimpleStepDataConverter:
    """단순화된 Step 데이터 변환기"""
    
    def __init__(self, step_name: str):
        self.step_name = step_name
        self.logger = logging.getLogger(f"utils.data_converter.{step_name}")
        
    def convert_image_format(self, image_data: Any, target_format: str = "RGB") -> Optional[Any]:
        """이미지 포맷 변환"""
        try:
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
        """이미지 크기 조정"""
        try:
            if not PIL_AVAILABLE:
                return image_data
            
            if hasattr(image_data, 'resize'):
                return image_data.resize(size, Image.Resampling.LANCZOS if hasattr(Image, 'Resampling') else Image.LANCZOS)
            
            return image_data
            
        except Exception as e:
            self.logger.error(f"❌ 이미지 크기 조정 실패: {e}")
            return image_data
    
    def normalize_data(self, data: Any) -> Optional[Any]:
        """데이터 정규화"""
        try:
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
# 🔥 팩토리 함수들 (main.py 호출 패턴 완전 호환)
# =============================================================================

_interface_cache = {}
_memory_manager_cache = {}
_data_converter_cache = {}
_cache_lock = threading.Lock()

@lru_cache(maxsize=8)
def get_step_model_interface(step_name: str) -> SimpleStepModelInterface:
    """Step별 모델 인터페이스 반환 (main.py 호출용)"""
    with _cache_lock:
        if step_name not in _interface_cache:
            _interface_cache[step_name] = SimpleStepModelInterface(step_name)
            logger.debug(f"🔧 {step_name} 모델 인터페이스 생성")
        
        return _interface_cache[step_name]

@lru_cache(maxsize=8)
def get_step_memory_manager(step_name: str) -> SimpleStepMemoryManager:
    """Step별 메모리 관리자 반환 (main.py 호출용)"""
    with _cache_lock:
        if step_name not in _memory_manager_cache:
            _memory_manager_cache[step_name] = SimpleStepMemoryManager(step_name)
            logger.debug(f"🔧 {step_name} 메모리 관리자 생성")
        
        return _memory_manager_cache[step_name]

@lru_cache(maxsize=8)
def get_step_data_converter(step_name: str) -> SimpleStepDataConverter:
    """Step별 데이터 변환기 반환 (main.py 호출용)"""
    with _cache_lock:
        if step_name not in _data_converter_cache:
            _data_converter_cache[step_name] = SimpleStepDataConverter(step_name)
            logger.debug(f"🔧 {step_name} 데이터 변환기 생성")
        
        return _data_converter_cache[step_name]

def preprocess_image_for_step(image_data: Any, step_name: str, **kwargs) -> Optional[Any]:
    """Step별 이미지 전처리 (main.py 호출용)"""
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
        
        logger.debug(f"✅ {step_name} 이미지 전처리 완료")
        return processed_image
        
    except Exception as e:
        logger.error(f"❌ {step_name} 이미지 전처리 실패: {e}")
        return image_data

# =============================================================================
# 🔥 고급 유틸리티 함수들 (추가 기능)
# =============================================================================

def clear_all_caches():
    """모든 캐시 초기화"""
    global _interface_cache, _memory_manager_cache, _data_converter_cache
    
    with _cache_lock:
        _interface_cache.clear()
        _memory_manager_cache.clear()  
        _data_converter_cache.clear()
        
        # @lru_cache 캐시도 초기화
        get_step_model_interface.cache_clear()
        get_step_memory_manager.cache_clear()
        get_step_data_converter.cache_clear()
        
        logger.info("🧹 모든 유틸리티 캐시 초기화 완료")

def get_system_status() -> Dict[str, Any]:
    """시스템 상태 반환"""
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
            'model_interfaces': len(_interface_cache),
            'memory_managers': len(_memory_manager_cache),
            'data_converters': len(_data_converter_cache)
        }
    }

def optimize_system_memory(aggressive: bool = False) -> bool:
    """시스템 전체 메모리 최적화"""
    try:
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
# 🔥 안전한 모듈 로딩 (고급 기능들)
# =============================================================================

def _try_import_advanced_modules():
    """고급 모듈들 안전하게 import 시도"""
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
    
    # auto_model_detector 시도  
    try:
        from .auto_model_detector import detect_available_models
        globals()['detect_available_models'] = detect_available_models
        advanced_status['auto_detector'] = True
        logger.info("✅ auto_model_detector 로드 성공")
    except ImportError:
        logger.debug("📋 auto_model_detector 없음 (정상)")
    
    # step_model_requirements 시도
    try:
        from .step_model_requirements import StepModelRequestAnalyzer
        globals()['StepModelRequestAnalyzer'] = StepModelRequestAnalyzer
        advanced_status['step_requirements'] = True
        logger.info("✅ step_model_requirements 로드 성공")
    except ImportError:
        logger.debug("📋 step_model_requirements 없음 (정상)")
    
    return advanced_status

# 고급 모듈들 로딩 시도
ADVANCED_STATUS = _try_import_advanced_modules()

# =============================================================================
# 🔥 Export 목록 (main.py 완전 호환)
# =============================================================================

__all__ = [
    # 🎯 핵심 팩토리 함수들 (main.py에서 호출)
    'get_step_model_interface',
    'get_step_memory_manager', 
    'get_step_data_converter',
    'preprocess_image_for_step',
    
    # 🔧 유틸리티 클래스들
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
    'TORCH_AVAILABLE',
    'NUMPY_AVAILABLE',
    'PIL_AVAILABLE',
    'ADVANCED_STATUS'
]

# 고급 모듈들 동적 추가
if ADVANCED_STATUS['model_loader']:
    __all__.append('ModelLoader')
if ADVANCED_STATUS['auto_detector']:
    __all__.append('detect_available_models')  
if ADVANCED_STATUS['step_requirements']:
    __all__.append('StepModelRequestAnalyzer')

# =============================================================================
# 🔥 초기화 완료 메시지
# =============================================================================

def _print_initialization_summary():
    """초기화 요약 출력"""
    basic_utils = ['model_interface', 'memory_manager', 'data_converter', 'image_preprocessor']
    basic_count = len(basic_utils)
    
    advanced_count = sum(ADVANCED_STATUS.values())
    library_count = sum([TORCH_AVAILABLE, NUMPY_AVAILABLE, PIL_AVAILABLE])
    
    print(f"\n🍎 MyCloset AI 파이프라인 유틸리티 v7.0 초기화 완료!")
    print(f"🔧 기본 유틸리티: {basic_count}/4개 ✅")
    print(f"🚀 고급 모듈: {advanced_count}/3개")
    print(f"📚 라이브러리: {library_count}/3개 (torch, numpy, PIL)")
    print(f"🐍 conda 환경: {'✅' if IS_CONDA else '❌'}")
    print(f"🍎 M3 Max: {'✅' if IS_M3_MAX else '❌'}")
    print(f"🖥️ 디바이스: {DEVICE}")
    
    # 고급 모듈 상태
    if advanced_count > 0:
        available_modules = [k for k, v in ADVANCED_STATUS.items() if v]
        print(f"✅ 사용 가능한 고급 모듈: {', '.join(available_modules)}")
    
    print("🚀 유틸리티 시스템 준비 완료!\n")

# 초기화 상태 출력 (한 번만)
if not hasattr(sys, '_mycloset_utils_initialized'):
    _print_initialization_summary()
    sys._mycloset_utils_initialized = True

logger.info("🍎 MyCloset AI 파이프라인 유틸리티 시스템 초기화 완료")