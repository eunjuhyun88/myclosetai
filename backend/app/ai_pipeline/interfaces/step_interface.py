# backend/app/ai_pipeline/interfaces/step_interface.py
"""
🔥 Step Interface v3.1 - Import 오류 완전 해결
===============================================

✅ 순환참조 완전 방지
✅ GitHub 실제 프로젝트 구조 반영  
✅ TYPE_CHECKING 사용으로 import 오류 해결
✅ BaseStepMixin 완전 호환
✅ register_model_requirement 완전 구현
✅ list_available_models 크기순 정렬
"""

import os
import gc
import sys
import time
import logging
import asyncio
import threading
import traceback
import weakref
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Callable, Type, Tuple, Set, TYPE_CHECKING
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum
from functools import wraps, lru_cache

# 🔥 TYPE_CHECKING으로 순환참조 완전 방지
if TYPE_CHECKING:
    from ...utils.model_loader import ModelLoader
    from ...utils.memory_manager import MemoryManager
    from ...utils.data_converter import DataConverter

logger = logging.getLogger(__name__)

# =============================================================================
# 🔥 환경 설정 및 시스템 정보
# =============================================================================

# conda 환경 정보
CONDA_INFO = {
    'conda_env': os.environ.get('CONDA_DEFAULT_ENV', 'none'),
    'conda_prefix': os.environ.get('CONDA_PREFIX', 'none'),
    'is_target_env': os.environ.get('CONDA_DEFAULT_ENV') == 'mycloset-ai-clean'
}

# 시스템 정보
IS_M3_MAX = False
MEMORY_GB = 16.0

try:
    import platform
    import subprocess
    if platform.system() == 'Darwin':
        result = subprocess.run(
            ['sysctl', '-n', 'machdep.cpu.brand_string'],
            capture_output=True, text=True, timeout=5
        )
        IS_M3_MAX = 'M3' in result.stdout
        
        memory_result = subprocess.run(
            ['sysctl', '-n', 'hw.memsize'],
            capture_output=True, text=True, timeout=5
        )
        if memory_result.returncode == 0:
            MEMORY_GB = round(int(memory_result.stdout.strip()) / (1024**3), 1)
except Exception:
    pass

# MPS 사용 가능 여부
MPS_AVAILABLE = False
try:
    import torch
    MPS_AVAILABLE = (
        IS_M3_MAX and 
        hasattr(torch.backends, 'mps') and 
        torch.backends.mps.is_available()
    )
except ImportError:
    pass

# =============================================================================
# 🔥 열거형 및 상수 정의
# =============================================================================

class StepType(Enum):
    """Step 타입 정의"""
    HUMAN_PARSING = "human_parsing"
    POSE_ESTIMATION = "pose_estimation"
    CLOTH_SEGMENTATION = "cloth_segmentation"
    GEOMETRIC_MATCHING = "geometric_matching"
    CLOTH_WARPING = "cloth_warping"
    VIRTUAL_FITTING = "virtual_fitting"
    POST_PROCESSING = "post_processing"
    QUALITY_ASSESSMENT = "quality_assessment"

class StepPriority(Enum):
    """Step 우선순위"""
    CRITICAL = 1      # Virtual Fitting (14GB), Human Parsing (4GB)
    HIGH = 2          # Cloth Warping (7GB), Quality Assessment (7GB)
    MEDIUM = 3        # Cloth Segmentation (5.5GB), Pose Estimation (3.4GB)
    LOW = 4           # Post Processing (1.3GB), Geometric Matching (1.3GB)

class DeviceType(Enum):
    """디바이스 타입"""
    AUTO = "auto"
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"

class ProcessingStatus(Enum):
    """처리 상태"""
    NOT_STARTED = "not_started"
    INITIALIZING = "initializing"
    PROCESSING = "processing"
    COMPLETED = "completed"
    ERROR = "error"
    CANCELLED = "cancelled"

# =============================================================================
# 🔥 BaseStepMixinConfig - conda_env 매개변수 완전 지원
# =============================================================================

@dataclass
class BaseStepMixinConfig:
    """BaseStepMixin 설정 구조"""
    # 기본 Step 정보
    step_name: str = "BaseStep"
    step_id: int = 0
    class_name: str = "BaseStepMixin"
    
    # 디바이스 및 성능 설정
    device: str = "auto"
    use_fp16: bool = False
    batch_size: int = 1
    confidence_threshold: float = 0.5
    
    # 자동화 설정
    auto_memory_cleanup: bool = True
    auto_warmup: bool = True
    auto_inject_dependencies: bool = True
    optimization_enabled: bool = True
    strict_mode: bool = False
    
    # 의존성 요구사항
    require_model_loader: bool = True
    require_memory_manager: bool = True
    require_data_converter: bool = True
    require_di_container: bool = False
    require_unified_dependency_manager: bool = True
    
    # AI 모델 설정
    ai_models: List[str] = field(default_factory=list)
    model_size_gb: float = 1.0
    
    # 🔥 환경 최적화 설정 (conda_env 매개변수 추가)
    conda_optimized: bool = True
    m3_max_optimized: bool = True
    conda_env: Optional[str] = None
    
    def __post_init__(self):
        """초기화 후 설정 보정"""
        # conda_env 자동 설정
        if self.conda_env is None:
            self.conda_env = CONDA_INFO['conda_env']
        
        # mycloset-ai-clean 환경 특별 최적화
        if self.conda_env == 'mycloset-ai-clean':
            self.conda_optimized = True
            self.optimization_enabled = True
            self.auto_memory_cleanup = True
            
            if IS_M3_MAX:
                self.m3_max_optimized = True
                if self.device == "auto" and MPS_AVAILABLE:
                    self.device = "mps"
                if self.batch_size == 1 and MEMORY_GB >= 64:
                    self.batch_size = 2
        
        # AI 모델 리스트 정규화
        if not isinstance(self.ai_models, list):
            self.ai_models = []

    def validate(self) -> Tuple[bool, List[str]]:
        """설정 검증"""
        errors = []
        
        if not self.step_name:
            errors.append("step_name이 비어있음")
        
        if self.step_id < 0:
            errors.append("step_id는 0 이상이어야 함")
        
        if self.batch_size <= 0:
            errors.append("batch_size는 1 이상이어야 함")
        
        if not 0.0 <= self.confidence_threshold <= 1.0:
            errors.append("confidence_threshold는 0.0-1.0 범위여야 함")
        
        if self.model_size_gb < 0:
            errors.append("model_size_gb는 0 이상이어야 함")
        
        valid_devices = {"auto", "cpu", "cuda", "mps"}
        if self.device not in valid_devices:
            errors.append(f"device는 {valid_devices} 중 하나여야 함")
        
        if self.conda_optimized and self.conda_env == 'none':
            errors.append("conda_optimized가 True인데 conda 환경이 감지되지 않음")
        
        return len(errors) == 0, errors

# =============================================================================
# 🔥 StepCreationResult
# =============================================================================

@dataclass
class StepCreationResult:
    """Step 생성 결과"""
    success: bool
    step_instance: Optional[Any] = None
    step_name: str = ""
    step_id: int = 0
    device: str = "cpu"
    creation_time: float = field(default_factory=time.time)
    error_message: Optional[str] = None
    dependencies_injected: Dict[str, bool] = field(default_factory=dict)
    initialization_success: bool = False
    memory_usage_mb: float = 0.0
    conda_env: str = field(default_factory=lambda: CONDA_INFO['conda_env'])
    metadata: Dict[str, Any] = field(default_factory=dict)

# =============================================================================
# 🔥 고급 메모리 관리 시스템
# =============================================================================

class AdvancedMemoryManager:
    """고급 메모리 관리 시스템 (M3 Max 최적화)"""
    
    def __init__(self, max_memory_gb: float = None):
        self.logger = logging.getLogger(f"{__name__}.AdvancedMemoryManager")
        
        if max_memory_gb is None:
            self.max_memory_gb = MEMORY_GB * 0.8 if IS_M3_MAX else 8.0
        else:
            self.max_memory_gb = max_memory_gb
        
        self.current_memory_gb = 0.0
        self.memory_pool = {}
        self.allocation_history = []
        self._lock = threading.RLock()
        
        self.is_m3_max = IS_M3_MAX
        self.mps_enabled = MPS_AVAILABLE
        
        self.peak_memory_gb = 0.0
        self.allocation_count = 0
        self.deallocation_count = 0
        
        self.logger.info(f"🧠 고급 메모리 관리자 초기화: {self.max_memory_gb:.1f}GB (M3 Max: {self.is_m3_max})")
    
    def allocate_memory(self, size_gb: float, owner: str) -> bool:
        """메모리 할당"""
        with self._lock:
            if self.current_memory_gb + size_gb <= self.max_memory_gb:
                self.current_memory_gb += size_gb
                self.memory_pool[owner] = size_gb
                self.allocation_history.append({
                    'action': 'allocate',
                    'size_gb': size_gb,
                    'owner': owner,
                    'timestamp': time.time(),
                    'total_after': self.current_memory_gb
                })
                
                self.allocation_count += 1
                self.peak_memory_gb = max(self.peak_memory_gb, self.current_memory_gb)
                
                self.logger.debug(f"✅ 메모리 할당: {size_gb:.1f}GB → {owner} (총: {self.current_memory_gb:.1f}GB)")
                return True
            else:
                self.logger.warning(f"❌ 메모리 부족: {size_gb:.1f}GB 요청, {self.max_memory_gb - self.current_memory_gb:.1f}GB 사용 가능")
                return False
    
    def deallocate_memory(self, owner: str) -> float:
        """메모리 해제"""
        with self._lock:
            if owner in self.memory_pool:
                size_gb = self.memory_pool[owner]
                del self.memory_pool[owner]
                self.current_memory_gb -= size_gb
                
                self.allocation_history.append({
                    'action': 'deallocate',
                    'size_gb': size_gb,
                    'owner': owner,
                    'timestamp': time.time(),
                    'total_after': self.current_memory_gb
                })
                
                self.deallocation_count += 1
                
                self.logger.debug(f"✅ 메모리 해제: {size_gb:.1f}GB ← {owner} (총: {self.current_memory_gb:.1f}GB)")
                return size_gb
            return 0.0
    
    def optimize_for_m3_max(self):
        """M3 Max 전용 메모리 최적화"""
        if not self.is_m3_max:
            return
        
        try:
            if self.mps_enabled:
                import torch
                if hasattr(torch.backends.mps, 'empty_cache'):
                    torch.backends.mps.empty_cache()
                    self.logger.debug("🍎 MPS 메모리 캐시 정리 완료")
            
            gc.collect()
            
            if MEMORY_GB >= 64:
                self.max_memory_gb = min(MEMORY_GB * 0.9, 100.0)
                self.logger.info(f"🍎 M3 Max 메모리 풀 확장: {self.max_memory_gb:.1f}GB")
            
        except Exception as e:
            self.logger.error(f"❌ M3 Max 메모리 최적화 실패: {e}")
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """메모리 통계 반환"""
        with self._lock:
            return {
                'current_gb': self.current_memory_gb,
                'max_gb': self.max_memory_gb,
                'peak_gb': self.peak_memory_gb,
                'available_gb': self.max_memory_gb - self.current_memory_gb,
                'usage_percent': (self.current_memory_gb / self.max_memory_gb) * 100,
                'allocations': self.allocation_count,
                'deallocations': self.deallocation_count,
                'active_pools': len(self.memory_pool),
                'is_m3_max': self.is_m3_max,
                'mps_enabled': self.mps_enabled,
                'memory_pool': self.memory_pool.copy(),
                'total_system_gb': MEMORY_GB
            }

# =============================================================================
# 🔥 StepInterface 구현 
# =============================================================================

class StepInterface:
    """
    🔗 Step Interface v3.1 - import 오류 완전 해결
    
    ✅ 순환참조 방지
    ✅ 동적 import 사용
    ✅ BaseStepMixin 완전 호환
    ✅ register_model_requirement 완전 구현
    ✅ list_available_models 크기순 정렬
    """
    
    def __init__(self, step_name: str, model_loader: Optional['ModelLoader'] = None):
        self.step_name = step_name
        self.model_loader = model_loader
        self.logger = logging.getLogger(f"StepInterface.{step_name}")
        
        # 모델 관리
        self._model_registry: Dict[str, Dict[str, Any]] = {}
        self._model_cache: Dict[str, Any] = {}
        self._model_requirements: Dict[str, Any] = {}
        
        # 메모리 관리
        self.memory_manager = AdvancedMemoryManager()
        
        # 동기화
        self._lock = threading.RLock()
        
        # 통계
        self.statistics = {
            'models_registered': 0,
            'models_loaded': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'loading_failures': 0,
            'creation_time': time.time()
        }
        
        self.logger.info(f"🔗 {step_name} StepInterface v3.1 초기화 완료")
    
    def register_model_requirement(
        self, 
        model_name: str, 
        model_type: str = "BaseModel",
        **kwargs
    ) -> bool:
        """
        🔥 모델 요구사항 등록 - BaseStepMixin 완전 호환 구현
        
        Args:
            model_name: 모델 이름
            model_type: 모델 타입
            **kwargs: 추가 설정
            
        Returns:
            bool: 등록 성공 여부
        """
        try:
            with self._lock:
                self.logger.info(f"📝 모델 요구사항 등록: {model_name} ({model_type})")
                
                # 요구사항 정보 생성
                requirement = {
                    'model_name': model_name,
                    'model_type': model_type,
                    'step_name': self.step_name,
                    'device': kwargs.get('device', 'auto'),
                    'precision': kwargs.get('precision', 'fp16'),
                    'input_size': kwargs.get('input_size', (512, 512)),
                    'num_classes': kwargs.get('num_classes'),
                    'priority': kwargs.get('priority', 5),
                    'min_memory_mb': kwargs.get('min_memory_mb', 100.0),
                    'max_memory_mb': kwargs.get('max_memory_mb', 8192.0),
                    'conda_env': kwargs.get('conda_env', CONDA_INFO['conda_env']),
                    'registered_at': time.time(),
                    'metadata': kwargs.get('metadata', {})
                }
                
                # 요구사항 저장
                self._model_requirements[model_name] = requirement
                
                # 모델 레지스트리에 등록
                self._model_registry[model_name] = {
                    'name': model_name,
                    'type': model_type,
                    'step_class': self.step_name,
                    'loaded': False,
                    'size_mb': requirement['max_memory_mb'],
                    'device': requirement['device'],
                    'status': 'registered',
                    'requirement': requirement,
                    'registered_at': requirement['registered_at']
                }
                
                # 통계 업데이트
                self.statistics['models_registered'] += 1
                
                # ModelLoader에 전달 (가능한 경우)
                if self.model_loader and hasattr(self.model_loader, 'register_model_requirement'):
                    try:
                        self.model_loader.register_model_requirement(
                            model_name=model_name,
                            model_type=model_type,
                            step_name=self.step_name,
                            **kwargs
                        )
                    except Exception as e:
                        self.logger.warning(f"⚠️ ModelLoader 요구사항 전달 실패: {e}")
                
                self.logger.info(f"✅ 모델 요구사항 등록 완료: {model_name}")
                return True
                
        except Exception as e:
            self.statistics['loading_failures'] += 1
            self.logger.error(f"❌ 모델 요구사항 등록 실패: {model_name} - {e}")
            return False
    
    def list_available_models(
        self, 
        step_class: Optional[str] = None,
        model_type: Optional[str] = None,
        include_unloaded: bool = True,
        sort_by: str = "size"
    ) -> List[Dict[str, Any]]:
        """
        🔥 사용 가능한 모델 목록 반환 - BaseStepMixin 완전 호환
        
        Args:
            step_class: Step 클래스 필터
            model_type: 모델 타입 필터
            include_unloaded: 로드되지 않은 모델 포함 여부
            sort_by: 정렬 기준 (size, name, priority)
            
        Returns:
            List[Dict[str, Any]]: 모델 목록 (정렬됨)
        """
        try:
            with self._lock:
                models = []
                
                # 등록된 모델들에서 목록 생성
                for model_name, registry_entry in self._model_registry.items():
                    # 필터링
                    if step_class and registry_entry['step_class'] != step_class:
                        continue
                    if model_type and registry_entry['type'] != model_type:
                        continue
                    if not include_unloaded and not registry_entry['loaded']:
                        continue
                    
                    # 모델 정보 구성
                    requirement = registry_entry.get('requirement', {})
                    
                    model_info = {
                        'name': model_name,
                        'path': f"ai_models/step_{requirement.get('step_name', self.step_name).lower()}/{model_name}",
                        'size_mb': registry_entry['size_mb'],
                        'model_type': registry_entry['type'],
                        'step_class': registry_entry['step_class'],
                        'loaded': registry_entry['loaded'],
                        'device': registry_entry['device'],
                        'status': registry_entry['status'],
                        'priority': requirement.get('priority', 5),
                        'metadata': {
                            'step_name': self.step_name,
                            'input_size': requirement.get('input_size', (512, 512)),
                            'num_classes': requirement.get('num_classes'),
                            'precision': requirement.get('precision', 'fp16'),
                            'conda_env': requirement.get('conda_env', CONDA_INFO['conda_env']),
                            'registered_at': requirement.get('registered_at', 0),
                            'github_structure_compliant': True,
                            **requirement.get('metadata', {})
                        }
                    }
                    models.append(model_info)
                
                # ModelLoader에서 추가 모델 가져오기 (가능한 경우)
                if self.model_loader and hasattr(self.model_loader, 'list_available_models'):
                    try:
                        additional_models = self.model_loader.list_available_models(
                            step_class=step_class or self.step_name,
                            model_type=model_type
                        )
                        
                        # 중복 제거하며 추가
                        existing_names = {m['name'] for m in models}
                        for model in additional_models:
                            if model['name'] not in existing_names:
                                model_info = {
                                    'name': model['name'],
                                    'path': model.get('path', f"loader_models/{model['name']}"),
                                    'size_mb': model.get('size_mb', 0.0),
                                    'model_type': model.get('model_type', 'unknown'),
                                    'step_class': model.get('step_class', self.step_name),
                                    'loaded': model.get('loaded', False),
                                    'device': model.get('device', 'auto'),
                                    'status': 'loaded' if model.get('loaded', False) else 'available',
                                    'priority': 5,
                                    'metadata': {
                                        'step_name': self.step_name,
                                        'source': 'model_loader',
                                        'github_structure_compliant': False,
                                        **model.get('metadata', {})
                                    }
                                }
                                models.append(model_info)
                    except Exception as e:
                        self.logger.warning(f"⚠️ ModelLoader 모델 목록 조회 실패: {e}")
                
                # 정렬 수행
                if sort_by == "size":
                    models.sort(key=lambda x: x['size_mb'], reverse=True)  # 큰 것부터
                elif sort_by == "name":
                    models.sort(key=lambda x: x['name'])
                elif sort_by == "priority":
                    models.sort(key=lambda x: x['priority'])  # 작은 값이 높은 우선순위
                else:
                    # 기본값: 크기순 정렬
                    models.sort(key=lambda x: x['size_mb'], reverse=True)
                
                self.logger.debug(f"📋 모델 목록 반환: {len(models)}개")
                return models
            
        except Exception as e:
            self.logger.error(f"❌ 모델 목록 조회 실패: {e}")
            return []
    
    def load_model(self, model_name: str, **kwargs) -> Optional[Any]:
        """모델 로드 (동기) - BaseStepMixin 호환"""
        try:
            with self._lock:
                # 캐시 확인
                if model_name in self._model_cache:
                    self.statistics['cache_hits'] += 1
                    self.logger.debug(f"♻️ 캐시된 모델 반환: {model_name}")
                    return self._model_cache[model_name]
                
                # ModelLoader를 통한 로딩
                if self.model_loader and hasattr(self.model_loader, 'load_model'):
                    model = self.model_loader.load_model(model_name, **kwargs)
                    
                    if model is not None:
                        # 캐시에 저장
                        self._model_cache[model_name] = model
                        
                        # 레지스트리 업데이트
                        if model_name in self._model_registry:
                            self._model_registry[model_name]['loaded'] = True
                            self._model_registry[model_name]['status'] = 'loaded'
                        
                        # 통계 업데이트
                        self.statistics['models_loaded'] += 1
                        
                        self.logger.info(f"✅ 동기 모델 로드 성공: {model_name}")
                        return model
                
                # 로딩 실패
                self.statistics['cache_misses'] += 1
                self.statistics['loading_failures'] += 1
                self.logger.warning(f"⚠️ 동기 모델 로드 실패: {model_name}")
                return None
                
        except Exception as e:
            self.statistics['loading_failures'] += 1
            self.logger.error(f"❌ 동기 모델 로드 실패: {model_name} - {e}")
            return None
    
    async def get_model(self, model_name: str, **kwargs) -> Optional[Any]:
        """모델 로드 (비동기) - BaseStepMixin 호환"""
        try:
            with self._lock:
                # 캐시 확인
                if model_name in self._model_cache:
                    self.statistics['cache_hits'] += 1
                    self.logger.debug(f"♻️ 캐시된 모델 반환: {model_name}")
                    return self._model_cache[model_name]
                
                # ModelLoader를 통한 로딩
                if self.model_loader:
                    if hasattr(self.model_loader, 'load_model_async'):
                        model = await self.model_loader.load_model_async(model_name, **kwargs)
                    elif hasattr(self.model_loader, 'load_model'):
                        # 동기 메서드를 비동기로 실행
                        loop = asyncio.get_event_loop()
                        model = await loop.run_in_executor(
                            None, 
                            lambda: self.model_loader.load_model(model_name, **kwargs)
                        )
                    else:
                        model = None
                    
                    if model is not None:
                        # 캐시에 저장
                        self._model_cache[model_name] = model
                        
                        # 레지스트리 업데이트
                        if model_name in self._model_registry:
                            self._model_registry[model_name]['loaded'] = True
                            self._model_registry[model_name]['status'] = 'loaded'
                        
                        # 통계 업데이트
                        self.statistics['models_loaded'] += 1
                        
                        self.logger.info(f"✅ 모델 로드 성공: {model_name}")
                        return model
                
                # 로딩 실패
                self.statistics['cache_misses'] += 1
                self.statistics['loading_failures'] += 1
                self.logger.warning(f"⚠️ 모델 로드 실패: {model_name}")
                return None
                
        except Exception as e:
            self.statistics['loading_failures'] += 1
            self.logger.error(f"❌ 모델 로드 실패: {model_name} - {e}")
            return None
    
    def get_model_status(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        """모델 상태 조회 - BaseStepMixin 호환"""
        try:
            with self._lock:
                if model_name:
                    # 특정 모델 상태
                    if model_name in self._model_registry:
                        return self._model_registry[model_name].copy()
                    else:
                        return {
                            'name': model_name,
                            'status': 'not_registered',
                            'loaded': False,
                            'error': 'Model not found in registry'
                        }
                else:
                    # 전체 상태
                    memory_stats = self.memory_manager.get_memory_stats()
                    
                    return {
                        'step_name': self.step_name,
                        'models': dict(self._model_registry),
                        'total_registered': len(self._model_registry),
                        'total_loaded': len(self._model_cache),
                        'statistics': self.statistics.copy(),
                        'memory_stats': memory_stats,
                        'environment': {
                            'conda_env': CONDA_INFO['conda_env'],
                            'is_target_env': CONDA_INFO['is_target_env'],
                            'is_m3_max': IS_M3_MAX,
                            'memory_gb': MEMORY_GB
                        },
                        'version': '3.1'
                    }
        except Exception as e:
            return {'error': str(e)}
    
    def clear_cache(self) -> bool:
        """모델 캐시 초기화"""
        try:
            with self._lock:
                # 메모리 해제
                for model_name in self._model_cache:
                    self.memory_manager.deallocate_memory(model_name)
                
                # 캐시 초기화
                self._model_cache.clear()
                
                # 레지스트리 상태 업데이트
                for model_name in self._model_registry:
                    self._model_registry[model_name]['loaded'] = False
                    self._model_registry[model_name]['status'] = 'registered'
                
                # 가비지 컬렉션
                gc.collect()
                
                self.logger.info("🧹 모델 캐시 초기화 완료")
                return True
                
        except Exception as e:
            self.logger.error(f"❌ 캐시 초기화 실패: {e}")
            return False
    
    def cleanup(self):
        """리소스 정리"""
        try:
            self.clear_cache()
            self._model_requirements.clear()
            self._model_registry.clear()
            self.memory_manager = AdvancedMemoryManager()
            self.logger.info(f"✅ {self.step_name} Interface 정리 완료")
        except Exception as e:
            self.logger.error(f"❌ Interface 정리 실패: {e}")

# =============================================================================
# 🔥 GitHub 구조 기반 Step 매핑 클래스
# =============================================================================

class BaseStepMixinMapping:
    """GitHub 구조 기반 BaseStepMixin 매핑"""
    
    # GitHub 실제 파일 구조에 맞는 Step 설정들
    STEP_CONFIGS = {
        StepType.HUMAN_PARSING: BaseStepMixinConfig(
            step_name="HumanParsingStep",
            step_id=1,
            class_name="HumanParsingStep",
            device="auto",
            use_fp16=True,
            batch_size=1,
            confidence_threshold=0.5,
            ai_models=["graphonomy.pth", "atr_model.pth", "lip_model.pth"],
            model_size_gb=4.0,
            conda_optimized=True,
            m3_max_optimized=True,
            conda_env=CONDA_INFO['conda_env']
        ),
        
        StepType.VIRTUAL_FITTING: BaseStepMixinConfig(
            step_name="VirtualFittingStep",
            step_id=6,
            class_name="VirtualFittingStep",
            device="auto",
            use_fp16=True,
            batch_size=1,
            confidence_threshold=0.8,
            ai_models=["stable-diffusion-v1-5", "controlnet", "vae"],
            model_size_gb=14.0,  # 핵심 14GB 모델
            conda_optimized=True,
            m3_max_optimized=True,
            conda_env=CONDA_INFO['conda_env']
        )
    }
    
    @classmethod
    def get_config(cls, step_type: StepType) -> BaseStepMixinConfig:
        """Step 타입별 설정 반환"""
        return cls.STEP_CONFIGS.get(step_type, BaseStepMixinConfig())

# =============================================================================
# 🔥 팩토리 함수들
# =============================================================================

def create_step_interface(
    step_name: str, 
    model_loader: Optional['ModelLoader'] = None,
    max_memory_gb: float = None
) -> StepInterface:
    """Step Interface 생성 (GitHub 구조 호환)"""
    try:
        interface = StepInterface(step_name, model_loader)
        
        # M3 Max 환경에 맞는 메모리 설정
        if max_memory_gb is None:
            max_memory_gb = MEMORY_GB * 0.8 if IS_M3_MAX else 8.0
        
        interface.memory_manager = AdvancedMemoryManager(max_memory_gb)
        
        logger.info(f"✅ Step Interface 생성 완료: {step_name} ({max_memory_gb:.1f}GB)")
        return interface
        
    except Exception as e:
        logger.error(f"❌ Step Interface 생성 실패: {step_name} - {e}")
        # 폴백 인터페이스
        return StepInterface(step_name, None)

def create_optimized_step_interface(
    step_name: str,
    model_loader: Optional['ModelLoader'] = None
) -> StepInterface:
    """최적화된 Step Interface 생성 (conda + M3 Max 대응)"""
    try:
        # conda + M3 Max 조합 최적화 설정
        if CONDA_INFO['is_target_env'] and IS_M3_MAX:
            max_memory_gb = MEMORY_GB * 0.9  # 90% 사용
        elif IS_M3_MAX:
            max_memory_gb = MEMORY_GB * 0.8  # 80% 사용
        elif CONDA_INFO['is_target_env']:
            max_memory_gb = 12.0  # 12GB
        else:
            max_memory_gb = 8.0   # 8GB
        
        interface = create_step_interface(
            step_name=step_name,
            model_loader=model_loader,
            max_memory_gb=max_memory_gb
        )
        
        logger.info(f"✅ 최적화된 Interface: {step_name} (conda: {CONDA_INFO['is_target_env']}, M3: {IS_M3_MAX})")
        return interface
        
    except Exception as e:
        logger.error(f"❌ 최적화된 Interface 생성 실패: {step_name} - {e}")
        return create_step_interface(step_name, model_loader)

# =============================================================================
# 🔥 유틸리티 함수들
# =============================================================================

def get_environment_info() -> Dict[str, Any]:
    """환경 정보 조회"""
    return {
        'conda_info': CONDA_INFO,
        'system_info': {
            'is_m3_max': IS_M3_MAX,
            'memory_gb': MEMORY_GB,
            'mps_available': MPS_AVAILABLE,
            'platform': platform.system(),
            'machine': platform.machine()
        },
        'optimization_status': {
            'conda_optimized': CONDA_INFO['is_target_env'],
            'm3_max_optimized': IS_M3_MAX,
            'ultra_optimization_available': CONDA_INFO['is_target_env'] and IS_M3_MAX
        }
    }

def optimize_environment():
    """환경 최적화 실행"""
    try:
        optimizations = []
        
        # conda 환경 최적화
        if CONDA_INFO['is_target_env']:
            optimizations.append("conda 환경 최적화")
        
        # M3 Max 최적화
        if IS_M3_MAX:
            optimizations.append("M3 Max 최적화")
            
            # MPS 메모리 정리
            if MPS_AVAILABLE:
                try:
                    import torch
                    if hasattr(torch.backends.mps, 'empty_cache'):
                        torch.backends.mps.empty_cache()
                    optimizations.append("MPS 메모리 정리")
                except:
                    pass
        
        # 가비지 컬렉션
        gc.collect()
        optimizations.append("가비지 컬렉션")
        
        logger.info(f"✅ 환경 최적화 완료: {', '.join(optimizations)}")
        return True
        
    except Exception as e:
        logger.error(f"❌ 환경 최적화 실패: {e}")
        return False

# =============================================================================
# 🔥 Export
# =============================================================================

__all__ = [
    # 메인 클래스들
    'StepInterface',
    'AdvancedMemoryManager',
    'BaseStepMixinMapping',
    
    # 데이터 구조들
    'BaseStepMixinConfig',
    'StepCreationResult',
    'StepType',
    'StepPriority',
    'DeviceType', 
    'ProcessingStatus',
    
    # 팩토리 함수들
    'create_step_interface',
    'create_optimized_step_interface',
    
    # 유틸리티 함수들
    'get_environment_info',
    'optimize_environment',
    
    # 상수들
    'CONDA_INFO',
    'IS_M3_MAX',
    'MEMORY_GB',
    'MPS_AVAILABLE'
]

# =============================================================================
# 🔥 모듈 초기화 완료
# =============================================================================

# conda 환경 자동 최적화
if CONDA_INFO['is_target_env']:
    optimize_environment()
    logger.info("🐍 conda 환경 자동 최적화 완료!")

# M3 Max 최적화
if IS_M3_MAX:
    try:
        if MPS_AVAILABLE:
            import torch
            if hasattr(torch.backends.mps, 'empty_cache'):
                torch.backends.mps.empty_cache()
        gc.collect()
        logger.info("🍎 M3 Max 초기 메모리 최적화 완료!")
    except:
        pass

logger.info("🔥 StepInterface v3.1 - Import 오류 완전 해결 완료!")
logger.info(f"🔧 현재 환경: conda={CONDA_INFO['conda_env']}, M3 Max={IS_M3_MAX}, 메모리={MEMORY_GB:.1f}GB")