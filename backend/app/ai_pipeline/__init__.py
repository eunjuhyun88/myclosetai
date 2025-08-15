#!/usr/bin/env python3
#backend/app/ai_pipeline/__init__.py
#!/usr/bin/env python3
"""
🔥 MyCloset AI Pipeline System v8.1 - DI Container v4.0 완전 적용 + 오류 수정
================================================================

✅ CircularReferenceFreeDIContainer 완전 통합
✅ TYPE_CHECKING으로 순환참조 완전 차단
✅ 지연 해결(Lazy Resolution) 활성화
✅ Step 팩토리 순환참조 완전 해결
✅ 안전한 의존성 주입 시스템
✅ M3 Max 128GB + conda 환경 최적화
✅ GitHub 프로젝트 구조 100% 호환
✅ SCIPY_AVAILABLE 오류 수정
✅ 상대 임포트 오류 수정
✅ DI Container 폴백 시스템 추가

Author: MyCloset AI Team
Date: 2025-08-01
Version: 8.1 (Bug Fixes)
"""
import threading

import os
import gc
import logging
import sys
import time
import warnings
import platform
import asyncio
from typing import Dict, Any, Optional, List, Type, Callable, Union, TYPE_CHECKING
from pathlib import Path

# 경고 무시 (deprecated 경로 관련)
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', message='.*deprecated.*')

# Logger 최우선 초기화 (에러 방지)
logger = logging.getLogger(__name__)

# ==============================================
# 🔥 SCIPY_AVAILABLE 변수 정의 (오류 수정)
# ==============================================

# SciPy 가용성 확인
try:
    import scipy
    SCIPY_AVAILABLE = True
    logger.debug("✅ SciPy 사용 가능")
except ImportError:
    SCIPY_AVAILABLE = False
    logger.debug("⚠️ SciPy 사용 불가 - 기본 기능으로 동작")

# ==============================================
# 🔥 TYPE_CHECKING으로 순환참조 완전 방지
# ==============================================

if TYPE_CHECKING:
    # 오직 타입 체크 시에만 import
    from .steps.base.core.base_step_mixin import BaseStepMixin
    from .models.model_loader import ModelLoader
    from .utils.memory_manager import MemoryManager
    from .utils.data_converter import DataConverter
    from .factories.step_factory import StepFactory
    from .pipeline_manager import PipelineManager
else:
    # 런타임에는 Any로 처리 (순환참조 방지)
    BaseStepMixin = Any
    ModelLoader = Any
    MemoryManager = Any
    DataConverter = Any
    StepFactory = Any
    PipelineManager = Any

# ==============================================
# 🔥 DI Container v4.0 Core 시스템 Import (오류 수정)
# ==============================================

try:
    # 절대 임포트 시도
    from app.core.di_container import (
        CircularReferenceFreeDIContainer,
        LazyDependency,
        DynamicImportResolver,
        get_global_container,
        reset_global_container,
        inject_dependencies_to_step_safe,
        get_service_safe,
        register_service_safe,
        register_lazy_service,
        initialize_di_system_safe
    )
    DI_CONTAINER_AVAILABLE = True
    logger.info("✅ DI Container v4.0 Core 시스템 로드 성공 (절대 임포트)")
except ImportError:
    try:
        # 상대 임포트 시도 (폴백)
        from ..core.di_container import (
            CircularReferenceFreeDIContainer,
            LazyDependency,
            DynamicImportResolver,
            get_global_container,
            reset_global_container,
            inject_dependencies_to_step_safe,
            get_service_safe,
            register_service_safe,
            register_lazy_service,
            initialize_di_system_safe
        )
        DI_CONTAINER_AVAILABLE = True
        logger.info("✅ DI Container v4.0 Core 시스템 로드 성공 (상대 임포트)")
    except ImportError as e:
        logger.error(f"❌ DI Container v4.0 Core 시스템 로드 실패: {e}")
        DI_CONTAINER_AVAILABLE = False
        
        # 폴백 처리
        def inject_dependencies_to_step_safe(step_instance, container=None):
            logger.warning("⚠️ DI Container 없음 - 의존성 주입 스킵")
        
        def get_service_safe(key: str):
            logger.warning(f"⚠️ DI Container 없음 - 서비스 조회 실패: {key}")
            return None
        
        def register_service_safe(key: str, service):
            logger.warning(f"⚠️ DI Container 없음 - 서비스 등록 스킵: {key}")
        
        def register_lazy_service(key: str, factory):
            logger.warning(f"⚠️ DI Container 없음 - 지연 서비스 등록 스킵: {key}")
        
        def initialize_di_system_safe():
            logger.warning("⚠️ DI Container 없음 - 시스템 초기화 스킵")
        
        def get_global_container():
            logger.warning("⚠️ DI Container 없음 - 글로벌 컨테이너 없음")
            return None

# ==============================================
# 🔥 환경 설정 (DI Container 통합) - 오류 수정
# ==============================================

# 시스템 정보 가져오기 (상위 패키지) - 오류 수정
try:
    # 절대 임포트 시도
    from app import get_system_info, is_conda_environment, is_m3_max, get_device
    SYSTEM_INFO = get_system_info()
    IS_CONDA = is_conda_environment()
    IS_M3_MAX = is_m3_max()
    DEVICE = get_device()
    logger.info("✅ 상위 패키지에서 시스템 정보 로드 성공 (절대 임포트)")
except ImportError:
    try:
        # 상대 임포트 시도 (폴백)
        from .. import get_system_info, is_conda_environment, is_m3_max, get_device
        SYSTEM_INFO = get_system_info()
        IS_CONDA = is_conda_environment()
        IS_M3_MAX = is_m3_max()
        DEVICE = get_device()
        logger.info("✅ 상위 패키지에서 시스템 정보 로드 성공 (상대 임포트)")
    except ImportError as e:
        logger.warning(f"⚠️ 상위 패키지 로드 실패, 기본값 사용: {e}")
        
        # 기본값 설정
        CONDA_ENV = os.environ.get('CONDA_DEFAULT_ENV', 'none')
        IS_CONDA = CONDA_ENV != 'none'
        IS_TARGET_ENV = CONDA_ENV == 'mycloset-ai-clean'
        
        # M3 Max 감지
        def _detect_m3_max() -> bool:
            try:
                if platform.system() == 'Darwin':
                    import subprocess
                    result = subprocess.run(
                        ['sysctl', '-n', 'machdep.cpu.brand_string'],
                        capture_output=True, text=True, timeout=5
                    )
                    return 'M3' in result.stdout and 'Max' in result.stdout
            except Exception:
                pass
            return False
        
        IS_M3_MAX = _detect_m3_max()
        MEMORY_GB = 128.0 if IS_M3_MAX else 16.0
        
        # 디바이스 감지
        try:
            import torch
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                DEVICE = 'mps'
            else:
                DEVICE = 'cpu'
        except ImportError:
            DEVICE = 'cpu'
        
        SYSTEM_INFO = {
            'device': DEVICE,
            'is_m3_max': IS_M3_MAX,
            'memory_gb': MEMORY_GB,
            'is_conda': IS_CONDA,
            'conda_env': CONDA_ENV
        }

# PyTorch 최적화 설정 (오류 수정)
TORCH_AVAILABLE = False
MPS_AVAILABLE = False
try:
    import torch
    TORCH_AVAILABLE = True
    
    # PyTorch 2.7 weights_only 호환성 패치
    if hasattr(torch, 'load'):
        original_load = torch.load
        def patched_load(*args, **kwargs):
            if 'weights_only' not in kwargs:
                kwargs['weights_only'] = False
            return original_load(*args, **kwargs)
        torch.load = patched_load
        logger.info("✅ PyTorch 2.7 weights_only 호환성 패치 적용 완료")
    
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        MPS_AVAILABLE = True
        # M3 Max 최적화 설정
        os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
        os.environ.setdefault('PYTORCH_MPS_HIGH_WATERMARK_RATIO', '0.0')
        
    logger.info(f"✅ PyTorch 로드: MPS={MPS_AVAILABLE}, M3 Max={IS_M3_MAX}")
except ImportError:
    logger.warning("⚠️ PyTorch 없음 - conda install pytorch 권장")

# ==============================================
# 🔥 안전한 지연 의존성 해결 클래스 (오류 수정)
# ==============================================

class SafeLazyDependency:
    """안전한 지연 의존성 해결 (LazyDependency 오류 방지)"""
    
    def __init__(self, resolver_func, fallback_value=None):
        self.resolver_func = resolver_func
        self.fallback_value = fallback_value
        self._resolved = False
        self._value = None
        self._lock = threading.Lock()
    
    def resolve(self):
        """의존성 해결"""
        if self._resolved:
            return self._value
        
        with self._lock:
            if self._resolved:
                return self._value
            
            try:
                self._value = self.resolver_func()
                self._resolved = True
                logger.debug(f"✅ SafeLazyDependency 해결 성공")
                return self._value
            except Exception as e:
                logger.warning(f"⚠️ SafeLazyDependency 해결 실패: {e}")
                self._value = self.fallback_value
                self._resolved = True
                return self._value

# ==============================================
# 🔥 DI Container 기반 Step 로딩 시스템
# ==============================================

class DIBasedStepLoader:
    """DI Container 기반 Step 로더 v4.1 (오류 수정)"""
    
    def __init__(self):
        self._container: Optional[CircularReferenceFreeDIContainer] = None
        self._loaded_steps = {}
        self._failed_steps = set()
        self._step_mapping = {}
        self.logger = logging.getLogger(f"{__name__}.DIBasedStepLoader")
        
        # DI Container 초기화
        self._initialize_container()
        
        # Step 매핑 설정
        self._setup_step_mapping()
    
    def _initialize_container(self):
        """DI Container 초기화 (오류 수정)"""
        try:
            if DI_CONTAINER_AVAILABLE:
                self._container = get_global_container()
                
                # 시스템 정보 등록
                if self._container:
                    self._container.register('device', DEVICE)
                    self._container.register('is_m3_max', IS_M3_MAX)
                    self._container.register('memory_gb', SYSTEM_INFO.get('memory_gb', 16.0))
                    self._container.register('is_conda', IS_CONDA)
                    self._container.register('torch_available', TORCH_AVAILABLE)
                    self._container.register('mps_available', MPS_AVAILABLE)
                    self._container.register('scipy_available', SCIPY_AVAILABLE)  # 추가
                    
                    # DI 시스템 초기화
                    initialize_di_system_safe()
                    
                    self.logger.info("✅ DI Container v4.0 초기화 완료")
                else:
                    self.logger.warning("⚠️ DI Container 가져오기 실패")
            else:
                self.logger.warning("⚠️ DI Container 사용 불가 - 폴백 모드")
                
        except Exception as e:
            self.logger.error(f"❌ DI Container 초기화 실패: {e}")
    
    def _setup_step_mapping(self):
        """Step 매핑 설정 (GitHub 구조 기준)"""
        self._step_mapping = {
            'step_01': {
                'module': 'app.ai_pipeline.steps.step_01_human_parsing_models.step_01_human_parsing',
                'class': 'HumanParsingStep',
                'description': '인체 파싱 - Human Body Parsing',
                'models': ['SCHP', 'Graphonomy'],
                'priority': 2
            },
            'step_02': {
                'module': 'app.ai_pipeline.steps.step_02_pose_estimation_models.step_02_pose_estimation',
                'class': 'PoseEstimationStep',
                'description': '포즈 추정 - Pose Estimation',
                'models': ['OpenPose', 'YOLO-Pose'],
                'priority': 4
            },
            'step_03': {
                'module': 'app.ai_pipeline.steps.step_03_cloth_segmentation_models.step_03_cloth_segmentation',
                'class': 'ClothSegmentationStep',
                'description': '의류 분할 - Cloth Segmentation',
                'models': ['U2Net', 'SAM'],
                'priority': 3
            },
            'step_04': {
                'module': 'app.ai_pipeline.steps.step_04_geometric_matching_models.step_04_geometric_matching',
                'class': 'GeometricMatchingStep',
                'description': '기하학적 매칭 - Geometric Matching',
                'models': ['TPS', 'GMM'],
                'priority': 7
            },
            'step_05': {
                'module': 'app.ai_pipeline.steps.step_05_cloth_warping_models.step_05_cloth_warping',
                'class': 'ClothWarpingStep',
                'description': '의류 변형 - Cloth Warping',
                'models': ['Advanced Warping'],
                'priority': 8
            },
            'step_06': {
                'module': 'app.ai_pipeline.steps.step_06_virtual_fitting_models.step_06_virtual_fitting',
                'class': 'VirtualFittingStep',
                'description': '가상 피팅 - Virtual Fitting',
                'models': ['OOTDiffusion', 'IDM-VTON'],
                'priority': 1  # 가장 중요
            },
            'step_07': {
                'module': 'app.ai_pipeline.steps.post_processing.step_07_post_processing',
                'class': 'PostProcessingStep',
                'description': '후처리 - Post Processing',
                'models': ['RealESRGAN', 'Enhancement'],
                'priority': 5
            },
            'step_08': {
                'module': 'app.ai_pipeline.steps.step_08_quality_assessment_models.step_08_quality_assessment',
                'class': 'QualityAssessmentStep',
                'description': '품질 평가 - Quality Assessment',
                'models': ['CLIP', 'Quality Metrics'],
                'priority': 6
            }
        }
        
        # conda 환경에서 우선순위 정렬
        self._loading_priority = sorted(
            self._step_mapping.keys(), 
            key=lambda x: self._step_mapping[x]['priority']
        )
    
    def safe_import_step(self, step_id: str) -> Optional[Type]:
        """DI Container 기반 안전한 Step import (오류 수정)"""
        if step_id in self._loaded_steps:
            return self._loaded_steps[step_id]
        
        if step_id in self._failed_steps:
            return None
        
        try:
            step_info = self._step_mapping.get(step_id)
            if not step_info:
                self.logger.warning(f"⚠️ 알 수 없는 Step ID: {step_id}")
                return None
            
            # DI Container 기반 동적 import
            if self._container:
                # 안전한 지연 로딩으로 Step 클래스 등록
                def step_factory():
                    return self._dynamic_import_step(step_info['module'], step_info['class'])
                
                step_key = f"step_class_{step_id}"
                
                # SafeLazyDependency 사용
                lazy_dep = SafeLazyDependency(step_factory)
                
                try:
                    self._container.register_lazy(step_key, step_factory)
                    step_class = self._container.get(step_key)
                except Exception:
                    # 폴백: SafeLazyDependency 직접 사용
                    step_class = lazy_dep.resolve()
                
                if step_class:
                    self._loaded_steps[step_id] = step_class
                    self.logger.info(f"✅ {step_id} ({step_info['class']}) DI 로드 성공")
                    return step_class
            else:
                # 폴백: 직접 import
                step_class = self._dynamic_import_step(step_info['module'], step_info['class'])
                if step_class:
                    self._loaded_steps[step_id] = step_class
                    self.logger.info(f"✅ {step_id} ({step_info['class']}) 직접 로드 성공")
                    return step_class
        
        except Exception as e:
            self.logger.error(f"❌ {step_id} 로드 실패: {e}")
        
        # 실패 기록
        self._failed_steps.add(step_id)
        self._loaded_steps[step_id] = None
        return None
    
    def _dynamic_import_step(self, module_name: str, class_name: str) -> Optional[Type]:
        """동적 Step import (순환참조 방지, 오류 수정)"""
        import_paths = [
            module_name,
            module_name.replace('app.', ''),
            f".{module_name.split('.')[-1]}"
        ]
        
        for path in import_paths:
            try:
                if path.startswith('.'):
                    # 상대 import
                    import importlib
                    module = importlib.import_module(path, package=__package__)
                else:
                    # 절대 import
                    import importlib
                    module = importlib.import_module(path)
                
                step_class = getattr(module, class_name, None)
                if step_class:
                    self.logger.debug(f"✅ {class_name} 동적 import 성공: {path}")
                    return step_class
                    
            except (ImportError, SyntaxError, AttributeError) as e:
                self.logger.debug(f"📋 {class_name} import 시도: {path} - {e}")
                continue
        
        return None
    
    def load_all_available_steps(self) -> Dict[str, Optional[Type]]:
        """사용 가능한 모든 Step 로드 (DI Container 기반)"""
        loaded_steps = {}
        
        # conda 환경이면 우선순위 순으로 로드
        load_order = self._loading_priority if IS_CONDA else self._step_mapping.keys()
        
        for step_id in load_order:
            step_class = self.safe_import_step(step_id)
            loaded_steps[step_id] = step_class
        
        available_count = sum(1 for step in loaded_steps.values() if step is not None)
        total_count = len(self._step_mapping)
        
        self.logger.info(f"📊 DI 기반 Step 로딩 완료: {available_count}/{total_count}개")
        
        if IS_CONDA:
            self.logger.info("🐍 conda 환경: 우선순위 기반 로딩 적용")
        
        return loaded_steps
    
    def create_step_instance(self, step_id: str, **kwargs) -> Optional[Any]:
        """DI Container 기반 Step 인스턴스 생성"""
        step_class = self.safe_import_step(step_id)
        if step_class is None:
            self.logger.error(f"❌ Step 클래스를 찾을 수 없음: {step_id}")
            return None
        
        try:
            # 기본 설정 추가
            default_config = {
                'device': DEVICE,
                'is_m3_max': IS_M3_MAX,
                'memory_gb': SYSTEM_INFO.get('memory_gb', 16.0),
                'conda_optimized': IS_CONDA,
                'scipy_available': SCIPY_AVAILABLE  # 추가
            }
            default_config.update(kwargs)
            
            # Step 인스턴스 생성
            step_instance = step_class(**default_config)
            
            # DI Container 기반 의존성 주입
            if self._container:
                inject_dependencies_to_step_safe(step_instance, self._container)
            else:
                inject_dependencies_to_step_safe(step_instance)
            
            self.logger.info(f"✅ {step_id} 인스턴스 생성 완료 (DI 주입 포함)")
            return step_instance
            
        except Exception as e:
            self.logger.error(f"❌ {step_id} 인스턴스 생성 실패: {e}")
            return None
    
    def get_container_stats(self) -> Dict[str, Any]:
        """DI Container 통계 반환"""
        if self._container:
            try:
                return self._container.get_stats()
            except Exception as e:
                return {
                    'container_available': True,
                    'stats_error': str(e),
                    'loaded_steps': len(self._loaded_steps),
                    'failed_steps': len(self._failed_steps)
                }
        else:
            return {
                'container_available': False,
                'fallback_mode': True,
                'loaded_steps': len(self._loaded_steps),
                'failed_steps': len(self._failed_steps)
            }

# ==============================================
# 🔥 전역 DI 기반 Step 로더 초기화
# ==============================================

# 전역 Step 로더 생성 (DI Container 기반)
_di_step_loader = DIBasedStepLoader()

# ==============================================
# 🔥 유틸리티 모듈 안전한 로딩 (DI Container 통합)
# ==============================================

def _safe_import_utils_with_di():
    """유틸리티 모듈들 안전하게 import (DI Container 통합)"""
    utils_status = {
        'model_loader': False,
        'memory_manager': False,
        'data_converter': False,
        'model_interface': False
    }
    
    try:
        # DI Container 기반 유틸리티 로딩
        container = get_global_container() if DI_CONTAINER_AVAILABLE else None
        
        if container:
            # DI Container에서 유틸리티 서비스 조회
            model_loader = get_service_safe('model_loader')
            memory_manager = get_service_safe('memory_manager')
            data_converter = get_service_safe('data_converter')
            
            if model_loader:
                utils_status['model_loader'] = True
                globals()['get_step_model_interface'] = lambda: model_loader
            
            if memory_manager:
                utils_status['memory_manager'] = True
                globals()['get_step_memory_manager'] = lambda: memory_manager
            
            if data_converter:
                utils_status['data_converter'] = True
                globals()['get_step_data_converter'] = lambda: data_converter
            
            utils_status['model_interface'] = True
            logger.info("✅ DI Container 기반 유틸리티 로드 성공")
        else:
            # 폴백: 직접 import
            try:
                from .utils import (
                    get_step_model_interface,
                    get_step_memory_manager, 
                    get_step_data_converter,
                    preprocess_image_for_step
                )
                utils_status.update({
                    'model_loader': True,
                    'memory_manager': True,
                    'data_converter': True,
                    'model_interface': True
                })
                logger.info("✅ 직접 유틸리티 모듈 로드 성공")
                
                # 전역에 추가
                globals().update({
                    'get_step_model_interface': get_step_model_interface,
                    'get_step_memory_manager': get_step_memory_manager,
                    'get_step_data_converter': get_step_data_converter,
                    'preprocess_image_for_step': preprocess_image_for_step
                })
                
            except ImportError as e:
                logger.warning(f"⚠️ 직접 유틸리티 모듈 로드 실패: {e}")
                raise
        
    except Exception as e:
        logger.warning(f"⚠️ 유틸리티 모듈 로드 실패: {e}")
        
        # 폴백 함수들
        def _fallback_function(name: str):
            def fallback(*args, **kwargs):
                logger.warning(f"⚠️ {name} 함수 사용 불가 (모듈 로드 실패)")
                return None
            return fallback
        
        globals().update({
            'get_step_model_interface': _fallback_function('get_step_model_interface'),
            'get_step_memory_manager': _fallback_function('get_step_memory_manager'),
            'get_step_data_converter': _fallback_function('get_step_data_converter'),
            'preprocess_image_for_step': _fallback_function('preprocess_image_for_step')
        })
    
    return utils_status

# 유틸리티 로딩 (DI Container 통합)
UTILS_STATUS = _safe_import_utils_with_di()

# ==============================================
# 🔥 파이프라인 관리 함수들 (DI Container 기반)
# ==============================================

def get_pipeline_status() -> Dict[str, Any]:
    """파이프라인 전체 상태 반환 (DI Container 포함)"""
    loaded_steps = _di_step_loader.load_all_available_steps()
    available_steps = [k for k, v in loaded_steps.items() if v is not None]
    container_stats = _di_step_loader.get_container_stats()
    
    return {
        'system_info': SYSTEM_INFO,
        'conda_optimized': IS_CONDA,
        'm3_max_optimized': IS_M3_MAX,
        'device': DEVICE,
        'scipy_available': SCIPY_AVAILABLE,  # 추가
        'total_steps': len(_di_step_loader._step_mapping),
        'available_steps': len(available_steps),
        'loaded_steps': available_steps,
        'failed_steps': [k for k, v in loaded_steps.items() if v is None],
        'success_rate': (len(available_steps) / len(_di_step_loader._step_mapping)) * 100,
        'utils_status': UTILS_STATUS,
        'loading_priority': _di_step_loader._loading_priority if IS_CONDA else None,
        'di_container_available': DI_CONTAINER_AVAILABLE,
        'di_container_stats': container_stats
    }

def get_step_class(step_name: str) -> Optional[Type]:
    """Step 클래스 반환 (DI Container 기반)"""
    if step_name.startswith('step_'):
        return _di_step_loader.safe_import_step(step_name)
    else:
        # 클래스명으로 검색
        for step_id, step_info in _di_step_loader._step_mapping.items():
            if step_info['class'] == step_name:
                return _di_step_loader.safe_import_step(step_id)
    return None

def create_step_instance(step_name: str, **kwargs) -> Optional[Any]:
    """Step 인스턴스 생성 (DI Container 기반)"""
    if step_name.startswith('step_'):
        return _di_step_loader.create_step_instance(step_name, **kwargs)
    else:
        # 클래스명으로 검색
        for step_id, step_info in _di_step_loader._step_mapping.items():
            if step_info['class'] == step_name:
                return _di_step_loader.create_step_instance(step_id, **kwargs)
    
    logger.error(f"❌ Step을 찾을 수 없음: {step_name}")
    return None

def list_available_steps() -> List[str]:
    """사용 가능한 Step 목록 반환"""
    loaded_steps = _di_step_loader.load_all_available_steps()
    return [step_id for step_id, step_class in loaded_steps.items() if step_class is not None]

def get_step_info(step_id: str) -> Dict[str, Any]:
    """Step 정보 반환 (DI Container 기반)"""
    step_config = _di_step_loader._step_mapping.get(step_id, {})
    step_class = _di_step_loader._loaded_steps.get(step_id)
    
    return {
        'step_id': step_id,
        'module': step_config.get('module', ''),
        'class': step_config.get('class', 'Unknown'),
        'description': step_config.get('description', ''),
        'models': step_config.get('models', []),
        'priority': step_config.get('priority', 10),
        'available': step_class is not None,
        'loaded': step_class is not None,
        'failed': step_id in _di_step_loader._failed_steps,
        'di_injected': DI_CONTAINER_AVAILABLE
    }

async def initialize_pipeline_system() -> bool:
    """파이프라인 시스템 초기화 (DI Container 기반)"""
    try:
        logger.info("🚀 파이프라인 시스템 초기화 시작 (DI Container v4.0)")
        
        # DI Container 초기화
        if DI_CONTAINER_AVAILABLE:
            initialize_di_system_safe()
            logger.info("✅ DI Container v4.0 시스템 초기화 완료")
        
        # Step 클래스들 로드
        loaded_steps = _di_step_loader.load_all_available_steps()
        available_count = sum(1 for step in loaded_steps.values() if step is not None)
        
        logger.info(f"✅ 파이프라인 시스템 초기화 완료: {available_count}/{len(_di_step_loader._step_mapping)}개 Step")
        
        # 중요한 Step들 개별 체크
        critical_steps = ['step_06', 'step_01', 'step_04']  # VirtualFitting, HumanParsing, GeometricMatching
        critical_available = 0
        
        for step_id in critical_steps:
            if loaded_steps.get(step_id):
                critical_available += 1
                logger.info(f"🎉 중요 Step {step_id} 로드 성공!")
            else:
                logger.warning(f"⚠️ 중요 Step {step_id} 로드 실패!")
        
        success = available_count > 0 and critical_available >= 1
        
        if success:
            logger.info("🚀 파이프라인 시스템 준비 완료!")
        else:
            logger.warning("⚠️ 파이프라인 시스템 부분 준비")
        
        return success
        
    except Exception as e:
        logger.error(f"❌ 파이프라인 시스템 초기화 실패: {e}")
        return False

async def cleanup_pipeline_system() -> None:
    """파이프라인 시스템 정리 (DI Container 기반)"""
    try:
        logger.info("🧹 파이프라인 시스템 정리 시작 (DI Container v4.0)")
        
        # DI Container 메모리 최적화
        if DI_CONTAINER_AVAILABLE:
            container = get_global_container()
            if container and hasattr(container, 'optimize_memory'):
                try:
                    cleanup_stats = container.optimize_memory()
                    logger.info(f"🧹 DI Container 메모리 최적화: {cleanup_stats}")
                except Exception as e:
                    logger.debug(f"DI Container 메모리 최적화 실패: {e}")
        
        # Step 로더 캐시 정리
        _di_step_loader._loaded_steps.clear()
        _di_step_loader._failed_steps.clear()
        
        # GPU 메모리 정리 (가능한 경우)
        if DEVICE in ['cuda', 'mps']:
            try:
                import torch
                if DEVICE == 'cuda' and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                elif DEVICE == 'mps' and torch.backends.mps.is_available():
                    # M3 Max 메모리 정리 (안전하게)
                    import gc
                    gc.collect()
                    if hasattr(torch.backends.mps, 'empty_cache'):
                        torch.backends.mps.empty_cache()
            except Exception as e:
                logger.debug(f"GPU 메모리 정리 실패: {e}")
        
        logger.info("✅ 파이프라인 시스템 정리 완료")
        
    except Exception as e:
        logger.warning(f"⚠️ 파이프라인 시스템 정리 실패: {e}")

# ==============================================
# 🔥 자동 Step 클래스 로딩 (DI Container 기반)
# ==============================================

# Step 클래스들을 전역 변수로 설정 (지연 로딩)
try:
    _loaded_steps = _di_step_loader.load_all_available_steps()
    
    # 개별 Step 클래스들을 전역에 추가
    for step_id, step_class in _loaded_steps.items():
        if step_class:
            step_info = _di_step_loader._step_mapping[step_id]
            class_name = step_info['class']
            globals()[class_name] = step_class
    
    logger.info("✅ DI 기반 Step 클래스들 전역 설정 완료")
    
except Exception as e:
    logger.warning(f"⚠️ DI 기반 Step 클래스 전역 설정 실패: {e}")

# Step 매핑 호환성 (기존 코드 지원)
STEP_MAPPING = _di_step_loader._step_mapping
LOADING_PRIORITY = _di_step_loader._loading_priority

# 가용성 플래그 매핑 (지연 평가)
def get_step_availability():
    loaded_steps = _di_step_loader._loaded_steps
    return {
        step_id: (loaded_steps.get(step_id) is not None)
        for step_id in _di_step_loader._step_mapping.keys()
    }

STEP_AVAILABILITY = get_step_availability()

# 사용 가능한 Step만 필터링 (지연 평가)
def get_available_steps():
    loaded_steps = _di_step_loader._loaded_steps
    return {
        step_id: step_class 
        for step_id, step_class in loaded_steps.items() 
        if step_class is not None
    }

AVAILABLE_STEPS = get_available_steps()

# ==============================================
# 🔥 Export 목록 (DI Container 기반)
# ==============================================

__all__ = [
    # 🎯 파이프라인 상수
    'STEP_MAPPING',
    'LOADING_PRIORITY',
    'SYSTEM_INFO',
    'STEP_AVAILABILITY',
    'AVAILABLE_STEPS',
    'SCIPY_AVAILABLE',  # 추가
    
    # 🔧 파이프라인 관리 함수들 (DI 기반)
    'get_pipeline_status',
    'get_step_class',
    'create_step_instance',
    'list_available_steps',
    'get_step_info',
    'initialize_pipeline_system',
    'cleanup_pipeline_system',
    
    # 🛠️ 유틸리티 함수들 (조건부)
    'get_step_model_interface',
    'get_step_memory_manager',
    'get_step_data_converter', 
    'preprocess_image_for_step',
    
    # 🔗 DI Container 관련
    'DIBasedStepLoader',
    'SafeLazyDependency',  # 추가
    'inject_dependencies_to_step_safe',
    'get_service_safe',
    'register_service_safe',
    'register_lazy_service',
    'DI_CONTAINER_AVAILABLE',
    
    # 📊 상태 정보
    'UTILS_STATUS',
    'IS_CONDA',
    'IS_M3_MAX',
    'DEVICE',
    'TORCH_AVAILABLE',
    'MPS_AVAILABLE'
]

# Step 클래스들도 동적으로 추가
for step_info in _di_step_loader._step_mapping.values():
    class_name = step_info['class']
    if class_name in globals():
        __all__.append(class_name)

# ==============================================
# 🔥 초기화 완료 메시지 (DI Container 포함)
# ==============================================

def _print_initialization_summary():
    """초기화 요약 출력 (간단 버전)"""
    status = get_pipeline_status()
    available_count = status['available_steps']
    total_count = status['total_steps']
    
    print(f"✅ 파이프라인 시스템 준비 완료 ({available_count}/{total_count}개 Step)")

# 초기화 상태 출력 (한 번만)
if not hasattr(sys, '_mycloset_pipeline_di_initialized'):
    _print_initialization_summary()
    sys._mycloset_pipeline_di_initialized = True

# conda 환경 자동 최적화 (DI Container 기반)
if IS_CONDA and DI_CONTAINER_AVAILABLE:
    try:
        # conda 환경 최적화
        os.environ.setdefault('OMP_NUM_THREADS', str(max(1, os.cpu_count() // 2)))
        os.environ.setdefault('MKL_NUM_THREADS', str(max(1, os.cpu_count() // 2)))
        os.environ.setdefault('NUMEXPR_NUM_THREADS', str(max(1, os.cpu_count() // 2)))
        
        if TORCH_AVAILABLE:
            import torch
            torch.set_num_threads(max(1, os.cpu_count() // 2))
            
            if IS_M3_MAX and MPS_AVAILABLE:
                if hasattr(torch.backends.mps, 'empty_cache'):
                    torch.backends.mps.empty_cache()
                logger.info("🍎 M3 Max MPS conda 최적화 완료")
        
        logger.info(f"🐍 conda 환경 자동 최적화 완료")
        
    except Exception as e:
        logger.warning(f"⚠️ conda 자동 최적화 실패: {e}")

logger.info("🔥 MyCloset AI Pipeline System v8.1 with DI Container v4.0 초기화 완료! (Bug Fixed)")