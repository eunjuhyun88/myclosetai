# backend/app/ai_pipeline/utils/model_loader.py
"""
🔥 ModelLoader v21.0 - 순환참조 완전 해결 + 안정성 강화
========================================================

✅ TYPE_CHECKING 패턴으로 순환참조 완전 방지
✅ StepModelInterface 개선
✅ 안전한 체크포인트 로딩
✅ 향상된 에러 처리
✅ M3 Max 128GB 최적화

Author: MyCloset AI Team
Date: 2025-07-24
Version: 21.0 (Circular Reference Complete Solution)
"""

import os
import gc
import time
import json
import logging
import asyncio
import threading
import traceback
import weakref
import hashlib
import pickle
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Tuple, Type, Set, Callable, TYPE_CHECKING
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache, wraps
from contextlib import asynccontextmanager, contextmanager
from collections import defaultdict
from abc import ABC, abstractmethod

# 🔥 TYPE_CHECKING으로 순환참조 완전 방지
if TYPE_CHECKING:
    from ..steps.base_step_mixin import BaseStepMixin

# ==============================================
# 🔥 안전한 라이브러리 import
# ==============================================

logger = logging.getLogger(__name__)

# PyTorch 안전 import
TORCH_AVAILABLE = False
MPS_AVAILABLE = False
NUMPY_AVAILABLE = False
DEFAULT_DEVICE = "cpu"
IS_M3_MAX = False
CONDA_ENV = "none"

try:
    os.environ.update({
        'PYTORCH_ENABLE_MPS_FALLBACK': '1',
        'PYTORCH_MPS_HIGH_WATERMARK_RATIO': '0.0',
        'MPS_DISABLE_METAL_PERFORMANCE_SHADERS': '0',
        'PYTORCH_MPS_PREFER_DEVICE_PLACEMENT': '1'
    })
    
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    
    TORCH_AVAILABLE = True
    
    if hasattr(torch, 'backends') and hasattr(torch.backends, 'mps'):
        if torch.backends.mps.is_available():
            MPS_AVAILABLE = True
            DEFAULT_DEVICE = "mps"
            
            # M3 Max 감지
            try:
                import platform
                import subprocess
                if platform.system() == 'Darwin':
                    result = subprocess.run(
                        ['sysctl', '-n', 'machdep.cpu.brand_string'],
                        capture_output=True, text=True, timeout=5
                    )
                    IS_M3_MAX = 'M3' in result.stdout
            except:
                pass
                
    elif torch.cuda.is_available():
        DEFAULT_DEVICE = "cuda"
        
except ImportError:
    torch = None

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    np = None

# conda 환경 감지
CONDA_ENV = os.environ.get('CONDA_DEFAULT_ENV', 'none')

# ==============================================
# 🔥 데이터 구조 정의
# ==============================================

class LoadingStatus(Enum):
    """로딩 상태"""
    NOT_LOADED = "not_loaded"
    LOADING = "loading"
    LOADED = "loaded"
    ERROR = "error"
    VALIDATING = "validating"

class ModelFormat(Enum):
    """모델 포맷"""
    PYTORCH = "pth"
    SAFETENSORS = "safetensors"
    TENSORFLOW = "bin"
    ONNX = "onnx"
    PICKLE = "pkl"
    CHECKPOINT = "ckpt"

class ModelType(Enum):
    """AI 모델 타입"""
    HUMAN_PARSING = "human_parsing"
    POSE_ESTIMATION = "pose_estimation"
    CLOTH_SEGMENTATION = "cloth_segmentation"
    GEOMETRIC_MATCHING = "geometric_matching"
    CLOTH_WARPING = "cloth_warping"
    VIRTUAL_FITTING = "virtual_fitting"
    POST_PROCESSING = "post_processing"
    QUALITY_ASSESSMENT = "quality_assessment"

@dataclass
class CheckpointValidation:
    """체크포인트 검증 결과"""
    is_valid: bool
    file_exists: bool
    size_mb: float
    checksum: Optional[str] = None
    error_message: Optional[str] = None
    validation_time: float = 0.0

@dataclass
class ModelConfig:
    """모델 설정 정보"""
    name: str
    model_type: Union[ModelType, str]
    model_class: str
    checkpoint_path: Optional[str] = None
    config_path: Optional[str] = None
    device: str = "auto"
    precision: str = "fp16"
    input_size: tuple = (512, 512)
    num_classes: Optional[int] = None
    file_size_mb: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    validation: Optional[CheckpointValidation] = None
    loading_status: LoadingStatus = LoadingStatus.NOT_LOADED
    last_validated: float = 0.0

@dataclass
class SafeModelCacheEntry:
    """안전한 모델 캐시 엔트리"""
    model: Any
    load_time: float
    last_access: float
    access_count: int
    memory_usage_mb: float
    device: str
    step_name: Optional[str] = None
    validation: Optional[CheckpointValidation] = None
    is_healthy: bool = True
    error_count: int = 0

# ==============================================
# 🔥 안전한 체크포인트 검증기
# ==============================================

class CheckpointValidator:
    """체크포인트 파일 검증기"""
    
    @staticmethod
    def validate_checkpoint_file(checkpoint_path: Union[str, Path]) -> CheckpointValidation:
        """체크포인트 파일 검증"""
        start_time = time.time()
        checkpoint_path = Path(checkpoint_path)
        
        try:
            # 파일 존재 확인
            if not checkpoint_path.exists():
                return CheckpointValidation(
                    is_valid=False,
                    file_exists=False,
                    size_mb=0.0,
                    error_message=f"파일이 존재하지 않음: {checkpoint_path}",
                    validation_time=time.time() - start_time
                )
            
            # 파일 크기 확인
            size_bytes = checkpoint_path.stat().st_size
            size_mb = size_bytes / (1024 * 1024)
            
            if size_bytes == 0:
                return CheckpointValidation(
                    is_valid=False,
                    file_exists=True,
                    size_mb=0.0,
                    error_message="파일 크기가 0바이트",
                    validation_time=time.time() - start_time
                )
            
            # 최소 크기 확인 (10MB 이상)
            if size_mb < 10:
                return CheckpointValidation(
                    is_valid=False,
                    file_exists=True,
                    size_mb=size_mb,
                    error_message=f"파일 크기가 너무 작음: {size_mb:.1f}MB",
                    validation_time=time.time() - start_time
                )
            
            # PyTorch 체크포인트 검증
            if TORCH_AVAILABLE:
                validation_result = CheckpointValidator._validate_pytorch_checkpoint(checkpoint_path)
                if not validation_result.is_valid:
                    return validation_result
            
            # 체크섬 계산 (1GB 미만인 경우만)
            checksum = None
            if size_mb < 1000:
                try:
                    checksum = CheckpointValidator._calculate_checksum(checkpoint_path)
                except:
                    pass
            
            return CheckpointValidation(
                is_valid=True,
                file_exists=True,
                size_mb=size_mb,
                checksum=checksum,
                validation_time=time.time() - start_time
            )
            
        except Exception as e:
            return CheckpointValidation(
                is_valid=False,
                file_exists=checkpoint_path.exists(),
                size_mb=0.0,
                error_message=f"검증 실패: {str(e)}",
                validation_time=time.time() - start_time
            )
    
    @staticmethod
    def _validate_pytorch_checkpoint(checkpoint_path: Path) -> CheckpointValidation:
        """PyTorch 체크포인트 검증"""
        start_time = time.time()
        
        try:
            import torch
            
            # 안전한 로딩 시도
            try:
                checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
                logger.debug(f"✅ 안전한 체크포인트 검증 성공: {checkpoint_path.name}")
            except Exception:
                try:
                    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
                    logger.debug(f"✅ 체크포인트 검증 성공: {checkpoint_path.name}")
                except Exception as load_error:
                    return CheckpointValidation(
                        is_valid=False,
                        file_exists=True,
                        size_mb=checkpoint_path.stat().st_size / (1024**2),
                        error_message=f"PyTorch 로딩 실패: {str(load_error)}",
                        validation_time=time.time() - start_time
                    )
            
            # 체크포인트 구조 검증
            if checkpoint is None:
                return CheckpointValidation(
                    is_valid=False,
                    file_exists=True,
                    size_mb=checkpoint_path.stat().st_size / (1024**2),
                    error_message="체크포인트가 None",
                    validation_time=time.time() - start_time
                )
            
            # 딕셔너리 형태 확인
            if isinstance(checkpoint, dict) and len(checkpoint) == 0:
                return CheckpointValidation(
                    is_valid=False,
                    file_exists=True,
                    size_mb=checkpoint_path.stat().st_size / (1024**2),
                    error_message="빈 체크포인트 딕셔너리",
                    validation_time=time.time() - start_time
                )
            
            return CheckpointValidation(
                is_valid=True,
                file_exists=True,
                size_mb=checkpoint_path.stat().st_size / (1024**2),
                validation_time=time.time() - start_time
            )
            
        except ImportError:
            return CheckpointValidation(
                is_valid=True,
                file_exists=True,
                size_mb=checkpoint_path.stat().st_size / (1024**2),
                error_message="PyTorch 검증 불가",
                validation_time=time.time() - start_time
            )
        except Exception as e:
            return CheckpointValidation(
                is_valid=False,
                file_exists=True,
                size_mb=checkpoint_path.stat().st_size / (1024**2),
                error_message=f"PyTorch 검증 실패: {str(e)}",
                validation_time=time.time() - start_time
            )
    
    @staticmethod
    def _calculate_checksum(file_path: Path) -> str:
        """파일 체크섬 계산"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

# ==============================================
# 🔥 안전한 비동기 컨텍스트 매니저
# ==============================================

class SafeAsyncContextManager:
    """안전한 비동기 컨텍스트 매니저"""
    
    def __init__(self, resource_name: str = "ModelLoader"):
        self.resource_name = resource_name
        self.is_entered = False
        self.logger = logging.getLogger(f"SafeAsyncCM.{resource_name}")
    
    async def __aenter__(self):
        """안전한 비동기 진입"""
        try:
            self.logger.debug(f"🔄 {self.resource_name} 비동기 컨텍스트 진입")
            self.is_entered = True
            return self
        except Exception as e:
            self.logger.error(f"❌ {self.resource_name} 비동기 진입 실패: {e}")
            raise RuntimeError(f"Async context enter failed for {self.resource_name}: {e}")
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """안전한 비동기 종료"""
        try:
            if self.is_entered:
                self.logger.debug(f"🔄 {self.resource_name} 비동기 컨텍스트 종료")
                self.is_entered = False
                
                if exc_type is not None:
                    self.logger.warning(f"⚠️ {self.resource_name} 컨텍스트에서 예외 발생: {exc_type.__name__}: {exc_val}")
                    
            return False
        except Exception as e:
            self.logger.error(f"❌ {self.resource_name} 비동기 종료 실패: {e}")
            return False

# ==============================================
# 🔥 개선된 StepModelInterface
# ==============================================

class StepModelInterface:
    """개선된 Step별 모델 인터페이스"""
    
    def __init__(self, model_loader: 'ModelLoader', step_name: str):
        self.model_loader = model_loader
        self.step_name = step_name
        self.logger = logging.getLogger(f"StepInterface.{step_name}")
        
        # 모델 캐시 및 상태
        self.loaded_models: Dict[str, Any] = {}
        self.model_cache: Dict[str, SafeModelCacheEntry] = {}
        self.model_status: Dict[str, LoadingStatus] = {}
        self._lock = threading.RLock()
        
        # Step 요구사항
        self.step_requirements: Dict[str, Any] = {}
        self.available_models: List[str] = []
        self.creation_time = time.time()
        self.error_count = 0
        self.last_error = None
        
        self.logger.info(f"🔗 {step_name} 인터페이스 초기화 완료")
    
    def register_model_requirement(
        self, 
        model_name: str, 
        model_type: str = "BaseModel",
        **kwargs
    ) -> bool:
        """모델 요구사항 등록"""
        try:
            with self._lock:
                self.logger.info(f"📝 모델 요구사항 등록 시작: {model_name}")
                
                # ModelLoader의 register_model_requirement 호출
                if hasattr(self.model_loader, 'register_model_requirement'):
                    success = self.model_loader.register_model_requirement(
                        model_name=model_name,
                        model_type=model_type,
                        step_name=self.step_name,
                        **kwargs
                    )
                    if success:
                        self.step_requirements[model_name] = {
                            "model_name": model_name,
                            "model_type": model_type,
                            "step_name": self.step_name,
                            "registered_at": time.time(),
                            **kwargs
                        }
                        self.logger.info(f"✅ 모델 요구사항 등록 완료: {model_name}")
                        return True
                else:
                    self.step_requirements[model_name] = {
                        "model_name": model_name,
                        "model_type": model_type,
                        "step_name": self.step_name,
                        "registered_at": time.time(),
                        **kwargs
                    }
                    self.logger.info(f"✅ 로컬 모델 요구사항 등록 완료: {model_name}")
                    return True
               
        except Exception as e:
            self.error_count += 1
            self.last_error = str(e)
            self.logger.error(f"❌ 모델 요구사항 등록 실패: {model_name} - {e}")
            return False
    
    async def get_model(self, model_name: Optional[str] = None) -> Optional[Any]:
        """비동기 모델 로드"""
        async with SafeAsyncContextManager(f"GetModel.{self.step_name}"):
            try:
                if not model_name:
                    model_name = "default_model"
                
                # 캐시 확인
                with self._lock:
                    if model_name in self.model_cache:
                        cache_entry = self.model_cache[model_name]
                        if cache_entry.is_healthy:
                            cache_entry.last_access = time.time()
                            cache_entry.access_count += 1
                            self.logger.debug(f"✅ 캐시된 모델 반환: {model_name}")
                            return cache_entry.model
                        else:
                            del self.model_cache[model_name]
                
                # 로딩 상태 설정
                self.model_status[model_name] = LoadingStatus.LOADING
                
                # ModelLoader를 통한 안전한 체크포인트 로드
                checkpoint = await self._safe_load_checkpoint(model_name)
                
                if checkpoint:
                    # 안전한 캐시 엔트리 생성
                    cache_entry = SafeModelCacheEntry(
                        model=checkpoint,
                        load_time=time.time(),
                        last_access=time.time(),
                        access_count=1,
                        memory_usage_mb=self._estimate_checkpoint_size(checkpoint),
                        device=getattr(checkpoint, 'device', DEFAULT_DEVICE) if hasattr(checkpoint, 'device') else DEFAULT_DEVICE,
                        step_name=self.step_name,
                        is_healthy=True,
                        error_count=0
                    )
                    
                    with self._lock:
                        self.model_cache[model_name] = cache_entry
                        self.loaded_models[model_name] = checkpoint
                        self.model_status[model_name] = LoadingStatus.LOADED
                    
                    self.logger.info(f"✅ 체크포인트 로드 성공: {model_name}")
                    return checkpoint
                
                self.model_status[model_name] = LoadingStatus.ERROR
                return None
                
            except Exception as e:
                self.error_count += 1
                self.last_error = str(e)
                self.model_status[model_name] = LoadingStatus.ERROR
                self.logger.error(f"❌ 모델 로드 실패 {model_name}: {e}")
                return None
    
    async def _safe_load_checkpoint(self, model_name: str) -> Optional[Any]:
        """안전한 체크포인트 로딩"""
        try:
            if hasattr(self.model_loader, 'load_model_async'):
                return await self.model_loader.load_model_async(model_name)
            elif hasattr(self.model_loader, 'load_model'):
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(
                    None, 
                    self.model_loader.load_model, 
                    model_name
                )
            else:
                self.logger.error(f"❌ ModelLoader에 로딩 메서드 없음")
                return None
        except Exception as e:
            self.logger.error(f"❌ 안전한 체크포인트 로딩 실패: {e}")
            return None
    
    def get_model_sync(self, model_name: Optional[str] = None) -> Optional[Any]:
        """동기 모델 로드"""
        try:
            if not model_name:
                model_name = "default_model"
            
            # 캐시 확인
            with self._lock:
                if model_name in self.model_cache:
                    cache_entry = self.model_cache[model_name]
                    if cache_entry.is_healthy:
                        cache_entry.last_access = time.time()
                        cache_entry.access_count += 1
                        return cache_entry.model
                    else:
                        del self.model_cache[model_name]
            
            # ModelLoader를 통한 체크포인트 로드
            checkpoint = None
            if hasattr(self.model_loader, 'load_model'):
                checkpoint = self.model_loader.load_model(model_name)
            
            if checkpoint:
                with self._lock:
                    cache_entry = SafeModelCacheEntry(
                        model=checkpoint,
                        load_time=time.time(),
                        last_access=time.time(),
                        access_count=1,
                        memory_usage_mb=self._estimate_checkpoint_size(checkpoint),
                        device=getattr(checkpoint, 'device', DEFAULT_DEVICE) if hasattr(checkpoint, 'device') else DEFAULT_DEVICE,
                        step_name=self.step_name,
                        is_healthy=True,
                        error_count=0
                    )
                    
                    self.model_cache[model_name] = cache_entry
                    self.loaded_models[model_name] = checkpoint
                    self.model_status[model_name] = LoadingStatus.LOADED
                return checkpoint
            
            self.model_status[model_name] = LoadingStatus.ERROR
            return None
            
        except Exception as e:
            self.error_count += 1
            self.last_error = str(e)
            self.logger.error(f"❌ 동기 모델 로드 실패 {model_name}: {e}")
            return None
    
    def _estimate_checkpoint_size(self, checkpoint) -> float:
        """체크포인트 크기 추정 (MB)"""
        try:
            if TORCH_AVAILABLE and checkpoint is not None:
                if isinstance(checkpoint, dict):
                    total_params = 0
                    for param in checkpoint.values():
                        if hasattr(param, 'numel'):
                            total_params += param.numel()
                    return total_params * 4 / (1024 * 1024)
                elif hasattr(checkpoint, 'parameters'):
                    total_params = sum(p.numel() for p in checkpoint.parameters())
                    return total_params * 4 / (1024 * 1024)
            return 0.0
        except:
            return 0.0
    
    def list_available_models(self) -> List[Dict[str, Any]]:
        """사용 가능한 모델 목록 반환"""
        models = []
        
        # 로컬 모델들 추가
        for model_name in self.step_requirements.keys():
            is_loaded = model_name in self.loaded_models
            cache_entry = self.model_cache.get(model_name)
            
            models.append({
                "name": model_name,
                "path": f"step_models/{model_name}",
                "size_mb": cache_entry.memory_usage_mb if cache_entry else 0.0,
                "model_type": self.step_name.lower(),
                "step_class": self.step_name,
                "loaded": is_loaded,
                "device": cache_entry.device if cache_entry else "auto",
                "metadata": {
                    "step_name": self.step_name,
                    "access_count": cache_entry.access_count if cache_entry else 0
                }
            })
        
        # 크기순 정렬
        models.sort(key=lambda x: x["size_mb"], reverse=True)
        return models

# ==============================================
# 🔥 메인 ModelLoader 클래스
# ==============================================

class ModelLoader:
    """개선된 ModelLoader v21.0 - 순환참조 완전 해결"""
    
    def __init__(
        self,
        device: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """개선된 ModelLoader 생성자"""
        
        # 기본 설정
        self.config = config or {}
        self.step_name = self.__class__.__name__
        self.logger = logging.getLogger(f"ModelLoader.{self.step_name}")
        
        # 디바이스 설정
        self.device = self._resolve_device(device or "auto")
        
        # 시스템 파라미터
        self.memory_gb = self._get_memory_info()
        self.is_m3_max = IS_M3_MAX
        self.conda_env = CONDA_ENV
        
        # 모델 디렉토리
        self.model_cache_dir = self._resolve_model_cache_dir(kwargs.get('model_cache_dir'))
        
        # 설정 파라미터
        self.use_fp16 = kwargs.get('use_fp16', True and self.device != 'cpu')
        self.max_cached_models = kwargs.get('max_cached_models', 30 if self.is_m3_max else 15)
        self.lazy_loading = kwargs.get('lazy_loading', True)
        self.enable_fallback = kwargs.get('enable_fallback', True)
        self.min_model_size_mb = kwargs.get('min_model_size_mb', 10)
        self.optimization_enabled = kwargs.get('optimization_enabled', True)
        
        # 핵심 속성들
        self.loaded_models: Dict[str, Any] = {}
        self.model_configs: Dict[str, Any] = {}
        self.model_cache: Dict[str, Any] = {}
        self.available_models: Dict[str, Any] = {}
        self.step_requirements: Dict[str, Dict[str, Any]] = {}
        self.step_interfaces: Dict[str, Any] = {}
        self._loaded_models = self.loaded_models
        self._is_initialized = False
        
        # 성능 추적
        self.load_times: Dict[str, float] = {}
        self.last_access: Dict[str, float] = {}
        self.access_counts: Dict[str, int] = {}
        self.performance_stats = {
            'models_loaded': 0,
            'cache_hits': 0,
            'load_times': {},
            'memory_usage': {},
            'validation_count': 0,
            'validation_success': 0,
            'checkpoint_loads': 0
        }
        
        # 동기화 및 스레드 관리
        self._lock = threading.RLock()
        self._interface_lock = threading.RLock()
        self._executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="model_loader_v21")
        
        # 체크포인트 검증기
        self.validator = CheckpointValidator()
        
        # 안전한 초기화 실행
        self._safe_initialize()
        
        self.logger.info(f"🎯 개선된 ModelLoader v21.0 초기화 완료")
        self.logger.info(f"🔧 Device: {self.device}, conda: {self.conda_env}, M3 Max: {self.is_m3_max}")
        self.logger.info(f"💾 Memory: {self.memory_gb:.1f}GB")
        self.logger.info(f"📁 모델 캐시 디렉토리: {self.model_cache_dir}")
    
    def _resolve_device(self, device: str) -> str:
        """디바이스 해결"""
        if device == "auto":
            return DEFAULT_DEVICE
        return device
    
    def _resolve_model_cache_dir(self, model_cache_dir_raw) -> Path:
        """모델 캐시 디렉토리 해결"""
        try:
            if model_cache_dir_raw is None:
                # 현재 파일 기준 자동 계산
                current_file = Path(__file__).resolve()
                # backend/app/ai_pipeline/utils/model_loader.py에서 backend/ 찾기
                current_path = current_file.parent
                for i in range(10):
                    if current_path.name == 'backend':
                        ai_models_path = current_path / "ai_models"
                        return ai_models_path
                    if current_path.parent == current_path:
                        break
                    current_path = current_path.parent
                
                # 폴백
                return Path.cwd() / "ai_models"
            else:
                path = Path(model_cache_dir_raw)
                # backend/backend 패턴 제거
                path_str = str(path)
                if "backend/backend" in path_str:
                    path = Path(path_str.replace("backend/backend", "backend"))
                return path.resolve()
                
        except Exception as e:
            self.logger.error(f"❌ 모델 디렉토리 해결 실패: {e}")
            return Path.cwd() / "ai_models"
    
    def _get_memory_info(self) -> float:
        """메모리 정보 조회"""
        try:
            import psutil
            memory = psutil.virtual_memory()
            return memory.total / (1024**3)
        except ImportError:
            return 128.0 if IS_M3_MAX else 16.0
    
    def _safe_initialize(self):
        """안전한 초기화"""
        try:
            # 캐시 디렉토리 확인 및 생성
            if not self.model_cache_dir.exists():
                self.model_cache_dir.mkdir(parents=True, exist_ok=True)
                self.logger.info(f"📁 모델 캐시 디렉토리 생성: {self.model_cache_dir}")
            
            # 기본 모델 레지스트리 초기화
            self._initialize_model_registry()
            
            # 사용 가능한 모델 스캔
            self._scan_available_models()
            
            # 메모리 최적화
            if self.optimization_enabled:
                self._safe_memory_cleanup()
            
            self.logger.info(f"📦 ModelLoader 안전 초기화 완료")
            
        except Exception as e:
            self.logger.error(f"❌ 안전 초기화 실패: {e}")
    
    def _initialize_model_registry(self):
        """기본 모델 레지스트리 초기화"""
        try:
            # 기본 모델 설정들
            default_models = {
                "human_parsing_model": ModelConfig(
                    name="human_parsing_model",
                    model_type=ModelType.HUMAN_PARSING,
                    model_class="HumanParsingModel",
                    device="auto",
                    precision="fp16",
                    input_size=(512, 512),
                    num_classes=20
                ),
                "pose_estimation_model": ModelConfig(
                    name="pose_estimation_model",
                    model_type=ModelType.POSE_ESTIMATION,
                    model_class="PoseEstimationModel",
                    device="auto",
                    precision="fp16",
                    input_size=(368, 368),
                    num_classes=18
                ),
                "cloth_segmentation_model": ModelConfig(
                    name="cloth_segmentation_model",
                    model_type=ModelType.CLOTH_SEGMENTATION,
                    model_class="ClothSegmentationModel",
                    device="auto",
                    precision="fp16",
                    input_size=(320, 320),
                    num_classes=1
                ),
                "virtual_fitting_model": ModelConfig(
                    name="virtual_fitting_model",
                    model_type=ModelType.VIRTUAL_FITTING,
                    model_class="VirtualFittingModel",
                    device="auto",
                    precision="fp16",
                    input_size=(512, 512)
                )
            }
            
            for name, config in default_models.items():
                self.model_configs[name] = config
            
            self.logger.info(f"📝 기본 모델 레지스트리 초기화: {len(default_models)}개")
            
        except Exception as e:
            self.logger.error(f"❌ 모델 레지스트리 초기화 실패: {e}")
    
    def _scan_available_models(self):
        """사용 가능한 모델 스캔"""
        try:
            if not self.model_cache_dir.exists():
                self.logger.warning(f"⚠️ 모델 디렉토리 없음: {self.model_cache_dir}")
                return
            
            # 검색 경로들
            search_paths = [
                self.model_cache_dir,
                self.model_cache_dir / "checkpoints",
                self.model_cache_dir / "models",
                self.model_cache_dir / "step_01_human_parsing",
                self.model_cache_dir / "step_02_pose_estimation",
                self.model_cache_dir / "step_03_cloth_segmentation",
                self.model_cache_dir / "step_06_virtual_fitting",
            ]
            
            # 존재하는 경로만 필터링
            existing_paths = [p for p in search_paths if p.exists()]
            
            scanned_count = 0
            extensions = [".pth", ".pt", ".bin", ".safetensors", ".ckpt"]
            
            for search_path in existing_paths:
                try:
                    for ext in extensions:
                        for model_file in search_path.rglob(f"*{ext}"):
                            try:
                                if not model_file.is_file():
                                    continue
                                
                                size_mb = model_file.stat().st_size / (1024 * 1024)
                                if size_mb < self.min_model_size_mb:
                                    continue
                                
                                # 간단한 검증
                                if not self._quick_validate_file(model_file):
                                    continue
                                
                                relative_path = model_file.relative_to(self.model_cache_dir)
                                model_type, step_class = self._detect_model_info(model_file)
                                
                                model_info = {
                                    "name": model_file.stem,
                                    "path": str(relative_path),
                                    "size_mb": round(size_mb, 2),
                                    "model_type": model_type,
                                    "step_class": step_class,
                                    "loaded": False,
                                    "device": self.device,
                                    "is_valid": True,
                                    "metadata": {
                                        "extension": ext,
                                        "parent_dir": model_file.parent.name,
                                        "full_path": str(model_file),
                                        "is_large": size_mb > 500,
                                        "detected_from": str(search_path.name)
                                    }
                                }
                                
                                self.available_models[model_file.stem] = model_info
                                scanned_count += 1
                                
                            except Exception as e:
                                self.logger.debug(f"⚠️ 파일 처리 실패 {model_file}: {e}")
                                continue
                                
                except Exception as path_error:
                    self.logger.debug(f"⚠️ 경로 스캔 실패 {search_path}: {path_error}")
                    continue
            
            self.logger.info(f"✅ 모델 스캔 완료: {scanned_count}개 등록")
            
        except Exception as e:
            self.logger.error(f"❌ 모델 스캔 실패: {e}")
    
    def _quick_validate_file(self, file_path: Path) -> bool:
        """빠른 파일 검증"""
        try:
            if not file_path.exists():
                return False
            
            size_mb = file_path.stat().st_size / (1024 * 1024)
            if size_mb < 1:
                return False
                
            valid_extensions = {'.pth', '.pt', '.bin', '.safetensors', '.ckpt'}
            if file_path.suffix.lower() not in valid_extensions:
                return False
                
            return True
        except Exception:
            return False
    
    def _detect_model_info(self, model_file: Path) -> tuple:
        """모델 타입 및 Step 클래스 감지"""
        filename = model_file.name.lower()
        path_str = str(model_file).lower()
        
        # 파일명 기반 감지
        if "schp" in filename or "human" in filename or "parsing" in filename:
            return "human_parsing", "HumanParsingStep"
        elif "openpose" in filename or "pose" in filename:
            return "pose_estimation", "PoseEstimationStep"
        elif "u2net" in filename or "segment" in filename or "cloth" in filename:
            return "cloth_segmentation", "ClothSegmentationStep"
        elif "diffusion" in filename or "pytorch_model" in filename:
            return "virtual_fitting", "VirtualFittingStep"
        
        # 경로 기반 감지
        if "step_01" in path_str or "human_parsing" in path_str:
            return "human_parsing", "HumanParsingStep"
        elif "step_02" in path_str or "pose" in path_str:
            return "pose_estimation", "PoseEstimationStep"
        elif "step_03" in path_str or "cloth" in path_str:
            return "cloth_segmentation", "ClothSegmentationStep"
        elif "step_06" in path_str or "virtual" in path_str:
            return "virtual_fitting", "VirtualFittingStep"
        
        return "unknown", "UnknownStep"
    
    def _safe_memory_cleanup(self):
        """안전한 메모리 정리"""
        try:
            gc.collect()
            
            if TORCH_AVAILABLE:
                if self.device == "cuda" and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                elif self.device == "mps" and MPS_AVAILABLE:
                    try:
                        if hasattr(torch.mps, 'empty_cache'):
                            torch.mps.empty_cache()
                    except:
                        pass
        except Exception as e:
            self.logger.debug(f"메모리 정리 실패 (무시): {e}")
    
    # ==============================================
    # 🔥 Step 인터페이스 관리
    # ==============================================
    
    def create_step_interface(self, step_name: str, step_requirements: Optional[Dict[str, Any]] = None) -> StepModelInterface:
        """Step 인터페이스 생성"""
        try:
            with self._interface_lock:
                # Step 요구사항 등록
                if step_requirements:
                    self.register_step_requirements(step_name, step_requirements)
                
                # 기존 인터페이스가 있으면 반환
                if step_name in self.step_interfaces:
                    return self.step_interfaces[step_name]
                
                # 새 인터페이스 생성
                interface = StepModelInterface(self, step_name)
                self.step_interfaces[step_name] = interface
                
                self.logger.info(f"✅ {step_name} 인터페이스 생성 완료")
                return interface
                
        except Exception as e:
            self.logger.error(f"❌ {step_name} 인터페이스 생성 실패: {e}")
            # 폴백 인터페이스 생성
            return StepModelInterface(self, step_name)
    
    def register_step_requirements(
        self, 
        step_name: str, 
        requirements: Union[Dict[str, Any], List[Dict[str, Any]]]
    ) -> bool:
        """Step별 모델 요구사항 등록"""
        try:
            with self._lock:
                self.logger.info(f"📝 {step_name} Step 요구사항 등록 시작...")
                
                if step_name not in self.step_requirements:
                    self.step_requirements[step_name] = {}
                
                # requirements가 리스트인 경우 처리
                if isinstance(requirements, list):
                    processed_requirements = {}
                    for i, req in enumerate(requirements):
                        if isinstance(req, dict):
                            model_name = req.get("model_name", f"{step_name}_model_{i}")
                            processed_requirements[model_name] = req
                    requirements = processed_requirements
                
                # 요구사항 업데이트
                if isinstance(requirements, dict):
                    self.step_requirements[step_name].update(requirements)
                
                self.logger.info(f"✅ {step_name} Step 요구사항 등록 완료")
                return True
                
        except Exception as e:
            self.logger.error(f"❌ {step_name} Step 요구사항 등록 실패: {e}")
            return False
    
    def register_model_requirement(
        self, 
        model_name: str, 
        model_type: str = "BaseModel",
        **kwargs
    ) -> bool:
        """모델 요구사항 등록"""
        try:
            with self._lock:
                model_config = {
                    "name": model_name,
                    "model_type": model_type,
                    "model_class": kwargs.get("model_class", model_type),
                    "device": kwargs.get("device", "auto"),
                    "precision": kwargs.get("precision", "fp16"),
                    "input_size": tuple(kwargs.get("input_size", (512, 512))),
                    "num_classes": kwargs.get("num_classes"),
                    "file_size_mb": kwargs.get("file_size_mb", 0.0),
                    "metadata": kwargs.get("metadata", {})
                }
                
                self.model_configs[model_name] = model_config
                
                self.available_models[model_name] = {
                    "name": model_name,
                    "path": f"requirements/{model_name}",
                    "size_mb": kwargs.get("file_size_mb", 0.0),
                    "model_type": model_type,
                    "step_class": kwargs.get("model_class", model_type),
                    "loaded": False,
                    "device": kwargs.get("device", "auto"),
                    "metadata": {
                        "source": "requirement_registration",
                        "registered_at": time.time(),
                        "step_name": kwargs.get("step_name", "unknown"),
                        **kwargs.get("metadata", {})
                    }
                }
                
                self.logger.info(f"✅ 모델 요구사항 등록 완료: {model_name}")
                return True
                
        except Exception as e:
            self.logger.error(f"❌ 모델 요구사항 등록 실패: {model_name} - {e}")
            return False
    
    def register_model_config(self, name: str, config: Union[ModelConfig, Dict[str, Any]]) -> bool:
        """모델 설정 등록"""
        try:
            with self._lock:
                if isinstance(config, dict):
                    model_config = config
                else:
                    model_config = config.__dict__ if hasattr(config, '__dict__') else config
                
                self.model_configs[name] = model_config
                
                self.available_models[name] = {
                    "name": name,
                    "path": model_config.get("checkpoint_path", f"config/{name}"),
                    "size_mb": model_config.get("file_size_mb", 0.0),
                    "model_type": str(model_config.get("model_type", "unknown")),
                    "step_class": model_config.get("model_class", "BaseModel"),
                    "loaded": False,
                    "device": model_config.get("device", "auto"),
                    "metadata": model_config.get("metadata", {})
                }
                
                self.logger.info(f"✅ 모델 설정 등록 완료: {name}")
                return True
                
        except Exception as e:
            self.logger.error(f"❌ 모델 설정 등록 실패 {name}: {e}")
            return False
    
    def list_available_models(self, step_class: Optional[str] = None, 
                            model_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """사용 가능한 모델 목록 반환"""
        try:
            models = []
            
            for model_name, model_info in self.available_models.items():
                # 필터링
                if step_class and model_info.get("step_class") != step_class:
                    continue
                if model_type and model_info.get("model_type") != model_type:
                    continue
                    
                models.append({
                    "name": model_info["name"],
                    "path": model_info["path"],
                    "size_mb": model_info["size_mb"],
                    "model_type": model_info["model_type"],
                    "step_class": model_info["step_class"],
                    "loaded": model_info["loaded"],
                    "device": model_info["device"],
                    "is_valid": model_info.get("is_valid", True),
                    "metadata": model_info["metadata"]
                })
            
            # 크기순 정렬 (큰 것부터)
            models.sort(key=lambda x: x["size_mb"], reverse=True)
            
            return models
            
        except Exception as e:
            self.logger.error(f"❌ 모델 목록 조회 실패: {e}")
            return []
    
    # ==============================================
    # 🔥 체크포인트 로딩 메서드들
    # ==============================================
    
    async def load_model_async(self, model_name: str, **kwargs) -> Optional[Any]:
        """비동기 체크포인트 로딩"""
        async with SafeAsyncContextManager(f"LoadModel.{model_name}"):
            try:
                # 캐시 확인
                if model_name in self.model_cache:
                    cache_entry = self.model_cache[model_name]
                    if cache_entry.get('is_healthy', True):
                        cache_entry['last_access'] = time.time()
                        cache_entry['access_count'] = cache_entry.get('access_count', 0) + 1
                        self.performance_stats['cache_hits'] += 1
                        self.logger.debug(f"♻️ 캐시된 체크포인트 반환: {model_name}")
                        return cache_entry['model']
                
                if model_name not in self.available_models and model_name not in self.model_configs:
                    self.logger.warning(f"⚠️ 체크포인트 없음: {model_name}")
                    return None
                
                # 비동기로 체크포인트 로딩 실행
                loop = asyncio.get_event_loop()
                checkpoint = await loop.run_in_executor(
                    self._executor, 
                    self._safe_load_checkpoint_sync,
                    model_name,
                    kwargs
                )
                
                if checkpoint is not None:
                    # 캐시 엔트리 생성
                    cache_entry = {
                        'model': checkpoint,
                        'load_time': time.time(),
                        'last_access': time.time(),
                        'access_count': 1,
                        'memory_usage_mb': self._estimate_checkpoint_size(checkpoint),
                        'device': getattr(checkpoint, 'device', self.device) if hasattr(checkpoint, 'device') else self.device,
                        'is_healthy': True,
                        'error_count': 0
                    }
                    
                    self.model_cache[model_name] = cache_entry
                    self.loaded_models[model_name] = checkpoint
                    
                    if model_name in self.available_models:
                        self.available_models[model_name]["loaded"] = True
                    
                    self.performance_stats['models_loaded'] += 1
                    self.performance_stats['checkpoint_loads'] += 1
                    self.logger.info(f"✅ 체크포인트 로딩 완료: {model_name}")
                    
                return checkpoint
                
            except Exception as e:
                self.logger.error(f"❌ 체크포인트 로딩 실패 {model_name}: {e}")
                return None
    
    def load_model(self, model_name: str, **kwargs) -> Optional[Any]:
        """동기 체크포인트 로딩"""
        try:
            # 캐시 확인
            if model_name in self.model_cache:
                cache_entry = self.model_cache[model_name]
                if cache_entry.get('is_healthy', True):
                    cache_entry['last_access'] = time.time()
                    cache_entry['access_count'] = cache_entry.get('access_count', 0) + 1
                    self.performance_stats['cache_hits'] += 1
                    self.logger.debug(f"♻️ 캐시된 체크포인트 반환: {model_name}")
                    return cache_entry['model']
            
            if model_name not in self.available_models and model_name not in self.model_configs:
                self.logger.warning(f"⚠️ 체크포인트 없음: {model_name}")
                return None
            
            return self._safe_load_checkpoint_sync(model_name, kwargs)
            
        except Exception as e:
            self.logger.error(f"❌ 체크포인트 로딩 실패 {model_name}: {e}")
            return None
    
    def _safe_load_checkpoint_sync(self, model_name: str, kwargs: Dict[str, Any]) -> Optional[Any]:
        """안전한 동기 체크포인트 로딩"""
        try:
            start_time = time.time()
            
            # 체크포인트 경로 찾기
            checkpoint_path = self._find_checkpoint_file(model_name)
            if not checkpoint_path or not checkpoint_path.exists():
                self.logger.warning(f"⚠️ 체크포인트 파일 없음: {model_name}")
                return None
            
            # 체크포인트 검증
            validation = self.validator.validate_checkpoint_file(checkpoint_path)
            if not validation.is_valid:
                self.logger.error(f"❌ 체크포인트 검증 실패: {model_name} - {validation.error_message}")
                return None
            
            # PyTorch 체크포인트 로딩
            if TORCH_AVAILABLE:
                try:
                    # GPU 메모리 정리
                    if self.device in ["mps", "cuda"]:
                        self._safe_memory_cleanup()
                    
                    self.logger.info(f"📂 체크포인트 파일 로딩: {checkpoint_path}")
                    
                    # 안전한 체크포인트 로딩
                    checkpoint = None
                    
                    try:
                        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
                        self.logger.debug(f"✅ 안전한 로딩 성공 (weights_only=True): {model_name}")
                    except Exception:
                        try:
                            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
                            self.logger.debug(f"✅ 일반 로딩 성공 (weights_only=False): {model_name}")
                        except Exception as load_error:
                            self.logger.error(f"❌ 모든 로딩 방법 실패: {load_error}")
                            return None
                    
                    if checkpoint is None:
                        self.logger.error(f"❌ 로딩된 체크포인트가 None: {model_name}")
                        return None
                    
                    # 체크포인트 후처리
                    processed_checkpoint = self._post_process_checkpoint(checkpoint, model_name)
                    
                    # 성능 기록
                    load_time = time.time() - start_time
                    self.load_times[model_name] = load_time
                    self.last_access[model_name] = time.time()
                    self.access_counts[model_name] = self.access_counts.get(model_name, 0) + 1
                    
                    self.performance_stats['models_loaded'] += 1
                    self.performance_stats['checkpoint_loads'] += 1
                    self.logger.info(f"✅ 체크포인트 로딩 성공: {model_name} ({load_time:.2f}초)")
                    return processed_checkpoint
                    
                except Exception as e:
                    self.logger.error(f"❌ PyTorch 체크포인트 로딩 실패 {model_name}: {e}")
                    return None
            
            self.logger.warning(f"⚠️ 체크포인트 로딩 불가: {model_name}")
            return None
            
        except Exception as e:
            self.logger.error(f"❌ 안전한 체크포인트 로딩 실패 {model_name}: {e}")
            return None
    
    def _find_checkpoint_file(self, model_name: str) -> Optional[Path]:
        """체크포인트 파일 찾기"""
        try:
            # 모델 설정에서 직접 경로 확인
            if model_name in self.model_configs:
                config = self.model_configs[model_name]
                if config.get('checkpoint_path'):
                    checkpoint_path = Path(config['checkpoint_path'])
                    if checkpoint_path.exists():
                        return checkpoint_path
            
            # available_models에서 찾기
            if model_name in self.available_models:
                model_info = self.available_models[model_name]
                if "full_path" in model_info["metadata"]:
                    full_path = Path(model_info["metadata"]["full_path"])
                    if full_path.exists():
                        return full_path
            
            # 직접 파일명 매칭
            extensions = [".pth", ".pt", ".bin", ".safetensors", ".ckpt"]
            for ext in extensions:
                direct_path = self.model_cache_dir / f"{model_name}{ext}"
                if direct_path.exists():
                    return direct_path
            
            # 패턴 매칭으로 찾기
            return self._find_via_pattern_matching(model_name, extensions)
            
        except Exception as e:
            self.logger.error(f"❌ 체크포인트 파일 찾기 실패 {model_name}: {e}")
            return None
    
    def _find_via_pattern_matching(self, model_name: str, extensions: List[str]) -> Optional[Path]:
        """패턴 매칭으로 찾기"""
        try:
            candidates = []
            for model_file in self.model_cache_dir.rglob("*"):
                if model_file.is_file() and model_file.suffix.lower() in extensions:
                    if model_name.lower() in model_file.name.lower():
                        try:
                            size_mb = model_file.stat().st_size / (1024 * 1024)
                            if size_mb >= self.min_model_size_mb:
                                candidates.append((model_file, size_mb))
                        except:
                            continue
            
            if candidates:
                # 크기순 정렬 (큰 것부터)
                candidates.sort(key=lambda x: x[1], reverse=True)
                best_candidate = candidates[0][0]
                return best_candidate
            
            return None
        except Exception:
            return None
    
    def _post_process_checkpoint(self, checkpoint: Any, model_name: str) -> Any:
        """체크포인트 후처리"""
        try:
            if isinstance(checkpoint, dict):
                if 'model' in checkpoint:
                    return checkpoint['model']
                elif 'state_dict' in checkpoint:
                    return checkpoint['state_dict']
                else:
                    return checkpoint
            return checkpoint
        except Exception as e:
            self.logger.warning(f"⚠️ 체크포인트 후처리 실패: {e}")
            return checkpoint
    
    def _estimate_checkpoint_size(self, checkpoint) -> float:
        """체크포인트 메모리 사용량 추정 (MB)"""
        try:
            if TORCH_AVAILABLE and checkpoint is not None:
                if isinstance(checkpoint, dict):
                    total_params = 0
                    for param in checkpoint.values():
                        if hasattr(param, 'numel'):
                            total_params += param.numel()
                    return total_params * 4 / (1024 * 1024)
                elif hasattr(checkpoint, 'parameters'):
                    total_params = sum(p.numel() for p in checkpoint.parameters())
                    return total_params * 4 / (1024 * 1024)
            return 0.0
        except:
            return 0.0
    
    # ==============================================
    # 🔥 상태 및 성능 메서드들
    # ==============================================
    
    def get_model_status(self, model_name: str) -> Dict[str, Any]:
        """모델 상태 조회"""
        try:
            if model_name in self.model_cache:
                cache_entry = self.model_cache[model_name]
                return {
                    "status": "loaded",
                    "device": cache_entry.get('device', 'unknown'),
                    "memory_usage_mb": cache_entry.get('memory_usage_mb', 0.0),
                    "last_used": cache_entry.get('last_access', 0),
                    "load_time": cache_entry.get('load_time', 0),
                    "access_count": cache_entry.get('access_count', 0),
                    "model_type": type(cache_entry['model']).__name__,
                    "loaded": True,
                    "is_healthy": cache_entry.get('is_healthy', True),
                    "error_count": cache_entry.get('error_count', 0)
                }
            elif model_name in self.model_configs:
                return {
                    "status": "registered",
                    "device": self.device,
                    "memory_usage_mb": 0,
                    "last_used": 0,
                    "load_time": 0,
                    "access_count": 0,
                    "model_type": "Not Loaded",
                    "loaded": False,
                    "is_healthy": True,
                    "error_count": 0
                }
            else:
                return {
                    "status": "not_found",
                    "device": None,
                    "memory_usage_mb": 0,
                    "last_used": 0,
                    "load_time": 0,
                    "access_count": 0,
                    "model_type": None,
                    "loaded": False,
                    "is_healthy": False,
                    "error_count": 0
                }
        except Exception as e:
            self.logger.error(f"❌ 모델 상태 조회 실패 {model_name}: {e}")
            return {"status": "error", "error": str(e)}
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """성능 메트릭 조회"""
        try:
            total_memory = sum(
                entry.get('memory_usage_mb', 0) for entry in self.model_cache.values()
            )
            
            load_times = list(self.load_times.values())
            avg_load_time = sum(load_times) / len(load_times) if load_times else 0
            
            validation_rate = (
                self.performance_stats['validation_success'] / 
                max(1, self.performance_stats['validation_count'])
            )
            
            return {
                "model_counts": {
                    "loaded": len(self.model_cache),
                    "registered": len(self.model_configs),
                    "available": len(self.available_models)
                },
                "memory_usage": {
                    "total_mb": total_memory,
                    "average_per_model_mb": total_memory / len(self.model_cache) if self.model_cache else 0,
                    "device": self.device,
                    "available_memory_gb": self.memory_gb
                },
                "performance_stats": {
                    "cache_hit_rate": self.performance_stats['cache_hits'] / max(1, self.performance_stats['models_loaded']),
                    "average_load_time_sec": avg_load_time,
                    "total_models_loaded": self.performance_stats['models_loaded'],
                    "checkpoint_loads": self.performance_stats.get('checkpoint_loads', 0),
                    "validation_rate": validation_rate,
                    "validation_count": self.performance_stats['validation_count'],
                    "validation_success": self.performance_stats['validation_success']
                },
                "step_interfaces": len(self.step_interfaces),
                "system_info": {
                    "conda_env": self.conda_env,
                    "is_m3_max": self.is_m3_max,
                    "torch_available": TORCH_AVAILABLE,
                    "mps_available": MPS_AVAILABLE,
                    "min_model_size_mb": self.min_model_size_mb
                },
                "version": "21.0"
            }
        except Exception as e:
            self.logger.error(f"❌ 성능 메트릭 조회 실패: {e}")
            return {"error": str(e)}
    
    def unload_model(self, model_name: str) -> bool:
        """모델 언로드"""
        try:
            if model_name in self.model_cache:
                del self.model_cache[model_name]
                
            if model_name in self.loaded_models:
                del self.loaded_models[model_name]
                
            if model_name in self.available_models:
                self.available_models[model_name]["loaded"] = False
                
            self._safe_memory_cleanup()
            
            self.logger.info(f"✅ 모델 언로드 완료: {model_name}")
            return True
            
        except Exception as e:
            self.logger.warning(f"⚠️ 모델 언로드 중 오류: {model_name} - {e}")
            return True  # 오류가 있어도 성공으로 처리
    
    def cleanup(self):
        """리소스 정리"""
        self.logger.info("🧹 ModelLoader 리소스 정리 중...")
        
        try:
            # 모든 모델 언로드
            for model_name in list(self.model_cache.keys()):
                self.unload_model(model_name)
                
            # 캐시 정리
            self.model_cache.clear()
            self.loaded_models.clear()
            self.step_interfaces.clear()
            
            # 스레드풀 종료
            self._executor.shutdown(wait=True)
            
            # 최종 메모리 정리
            self._safe_memory_cleanup()
            
            self.logger.info("✅ ModelLoader 리소스 정리 완료")
            
        except Exception as e:
            self.logger.error(f"❌ 리소스 정리 실패: {e}")
    
    def initialize(self, **kwargs) -> bool:
        """ModelLoader 초기화"""
        try:
            if self._is_initialized:
                return True
            
            # 설정 업데이트
            if kwargs:
                for key, value in kwargs.items():
                    if hasattr(self, key):
                        setattr(self, key, value)
            
            # 재초기화 실행
            self._safe_initialize()
            
            self._is_initialized = True
            self.logger.info(f"✅ ModelLoader 초기화 완료")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ ModelLoader 초기화 실패: {e}")
            return False
    
    async def initialize_async(self, **kwargs) -> bool:
        """비동기 초기화"""
        try:
            result = self.initialize(**kwargs)
            if result:
                self.logger.info("✅ ModelLoader 비동기 초기화 완료")
            return result
        except Exception as e:
            self.logger.error(f"❌ ModelLoader 비동기 초기화 실패: {e}")
            return False
    
    def is_initialized(self) -> bool:
        """초기화 상태 확인"""
        return getattr(self, '_is_initialized', False)
    
    def __del__(self):
        """소멸자"""
        try:
            self.cleanup()
        except:
            pass

# ==============================================
# 🔥 전역 ModelLoader 관리
# ==============================================

_global_model_loader: Optional[ModelLoader] = None
_loader_lock = threading.Lock()

def get_global_model_loader(config: Optional[Dict[str, Any]] = None) -> ModelLoader:
    """전역 ModelLoader 인스턴스 반환"""
    global _global_model_loader
    
    with _loader_lock:
        if _global_model_loader is None:
            # 올바른 AI 모델 경로 계산
            current_file = Path(__file__)
            backend_root = current_file.parents[3]  # backend/
            ai_models_path = backend_root / "ai_models"
            
            try:
                _global_model_loader = ModelLoader(
                    config=config,
                    device="auto",
                    model_cache_dir=str(ai_models_path),
                    use_fp16=True,
                    optimization_enabled=True,
                    enable_fallback=True,
                    min_model_size_mb=10
                )
                logger.info("✅ 전역 ModelLoader 생성 성공")
                
            except Exception as e:
                logger.error(f"❌ 전역 ModelLoader 생성 실패: {e}")
                _global_model_loader = ModelLoader(device="cpu")
                
        return _global_model_loader

def initialize_global_model_loader(**kwargs) -> bool:
    """전역 ModelLoader 초기화"""
    try:
        loader = get_global_model_loader()
        return loader.initialize(**kwargs)
    except Exception as e:
        logger.error(f"❌ 전역 ModelLoader 초기화 실패: {e}")
        return False

async def initialize_global_model_loader_async(**kwargs) -> ModelLoader:
    """전역 ModelLoader 비동기 초기화"""
    try:
        loader = get_global_model_loader()
        success = await loader.initialize_async(**kwargs)
        
        if success:
            logger.info(f"✅ 전역 ModelLoader 비동기 초기화 완료")
        else:
            logger.warning(f"⚠️ 전역 ModelLoader 초기화 일부 실패")
            
        return loader
    except Exception as e:
        logger.error(f"❌ 전역 ModelLoader 비동기 초기화 실패: {e}")
        raise

# ==============================================
# 🔥 유틸리티 함수들
# ==============================================

def create_step_interface(step_name: str, step_requirements: Optional[Dict[str, Any]] = None) -> StepModelInterface:
    """Step 인터페이스 생성"""
    try:
        loader = get_global_model_loader()
        if step_requirements:
            loader.register_step_requirements(step_name, step_requirements)
        return loader.create_step_interface(step_name, step_requirements)
    except Exception as e:
        logger.error(f"❌ Step 인터페이스 생성 실패 {step_name}: {e}")
        return StepModelInterface(get_global_model_loader(), step_name)

def validate_checkpoint_file(checkpoint_path: Union[str, Path]) -> CheckpointValidation:
    """체크포인트 파일 검증 함수"""
    return CheckpointValidator.validate_checkpoint_file(checkpoint_path)

def safe_load_checkpoint(checkpoint_path: Union[str, Path], device: str = "cpu") -> Optional[Any]:
    """안전한 체크포인트 로딩 함수"""
    try:
        validation = validate_checkpoint_file(checkpoint_path)
        if not validation.is_valid:
            logger.error(f"❌ 체크포인트 검증 실패: {validation.error_message}")
            return None
        
        if TORCH_AVAILABLE:
            try:
                checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
                logger.debug(f"✅ 안전한 체크포인트 로딩 성공")
                return checkpoint
            except:
                try:
                    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
                    logger.debug(f"✅ 일반 체크포인트 로딩 성공")
                    return checkpoint
                except Exception as e:
                    logger.error(f"❌ 체크포인트 로딩 실패: {e}")
                    return None
        
        return None
    except Exception as e:
        logger.error(f"❌ 안전한 체크포인트 로딩 실패: {e}")
        return None

# 기존 호환성을 위한 함수들
def get_model(model_name: str) -> Optional[Any]:
    """전역 모델 가져오기"""
    loader = get_global_model_loader()
    return loader.load_model(model_name)

async def get_model_async(model_name: str) -> Optional[Any]:
    """전역 비동기 모델 가져오기"""
    loader = get_global_model_loader()
    return await loader.load_model_async(model_name)

def get_step_model_interface(step_name: str, model_loader_instance=None) -> StepModelInterface:
    """Step 모델 인터페이스 생성"""
    try:
        if model_loader_instance is None:
            model_loader_instance = get_global_model_loader()
        
        return model_loader_instance.create_step_interface(step_name)
    except Exception as e:
        logger.error(f"❌ {step_name} 인터페이스 생성 실패: {e}")
        return StepModelInterface(model_loader_instance or get_global_model_loader(), step_name)

# ==============================================
# 🔥 메모리 관리 함수들
# ==============================================

def safe_mps_empty_cache():
    """안전한 MPS 메모리 정리"""
    try:
        if TORCH_AVAILABLE and MPS_AVAILABLE:
            if hasattr(torch, 'mps'):
                if hasattr(torch.mps, 'empty_cache'):
                    torch.mps.empty_cache()
                if hasattr(torch.mps, 'synchronize'):
                    torch.mps.synchronize()
                return True
        return False
    except Exception:
        return False

def safe_torch_cleanup():
    """안전한 PyTorch 메모리 정리"""
    try:
        gc.collect()
        
        if TORCH_AVAILABLE:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                
            if MPS_AVAILABLE:
                safe_mps_empty_cache()
        
        return True
    except Exception as e:
        logger.warning(f"⚠️ PyTorch 메모리 정리 실패: {e}")
        return False

def get_enhanced_memory_info() -> Dict[str, Any]:
    """향상된 시스템 메모리 정보 조회"""
    try:
        import psutil
        memory = psutil.virtual_memory()
        
        memory_info = {
            "total_gb": memory.total / (1024**3),
            "available_gb": memory.available / (1024**3),
            "used_gb": memory.used / (1024**3),
            "percent": memory.percent,
            "is_m3_max": IS_M3_MAX,
            "conda_env": CONDA_ENV
        }
        
        # GPU 메모리 정보 추가
        if TORCH_AVAILABLE:
            if torch.cuda.is_available():
                memory_info["gpu"] = {
                    "allocated_mb": torch.cuda.memory_allocated() / (1024**2),
                    "reserved_mb": torch.cuda.memory_reserved() / (1024**2),
                    "max_allocated_mb": torch.cuda.max_memory_allocated() / (1024**2)
                }
            elif MPS_AVAILABLE:
                try:
                    if hasattr(torch.mps, 'current_allocated_memory'):
                        memory_info["mps"] = {
                            "allocated_mb": torch.mps.current_allocated_memory() / (1024**2)
                        }
                except:
                    memory_info["mps"] = {"status": "available"}
        
        return memory_info
        
    except ImportError:
        return {
            "total_gb": 128.0 if IS_M3_MAX else 16.0,
            "available_gb": 100.0 if IS_M3_MAX else 12.0,
            "used_gb": 28.0 if IS_M3_MAX else 4.0,
            "percent": 22.0 if IS_M3_MAX else 25.0,
            "is_m3_max": IS_M3_MAX,
            "conda_env": CONDA_ENV
        }

# ==============================================
# 🔥 Export
# ==============================================

__all__ = [
    # 핵심 클래스들
    'ModelLoader',
    'StepModelInterface',
    'CheckpointValidator',
    'SafeAsyncContextManager',
    
    # 데이터 구조들
    'ModelFormat',
    'ModelType', 
    'ModelConfig',
    'SafeModelCacheEntry',
    'CheckpointValidation',
    'LoadingStatus',
    
    # 전역 함수들
    'get_global_model_loader',
    'initialize_global_model_loader',
    'initialize_global_model_loader_async',
    
    # 유틸리티 함수들
    'create_step_interface',
    'validate_checkpoint_file',
    'safe_load_checkpoint',
    'get_step_model_interface',
    
    # 기존 호환성 함수들
    'get_model',
    'get_model_async',
    
    # 안전한 함수들
    'safe_mps_empty_cache',
    'safe_torch_cleanup',
    'get_enhanced_memory_info',
    
    # 상수들
    'TORCH_AVAILABLE',
    'MPS_AVAILABLE',
    'NUMPY_AVAILABLE',
    'DEFAULT_DEVICE',
    'IS_M3_MAX',
    'CONDA_ENV'
]

# 모듈 로드 완료
logger.info("=" * 80)
logger.info("✅ ModelLoader v21.0 - 순환참조 완전 해결 + 안정성 강화")
logger.info("=" * 80)
logger.info("🔥 TYPE_CHECKING 패턴으로 순환참조 완전 방지")
logger.info("✅ StepModelInterface 개선")
logger.info("✅ 안전한 체크포인트 로딩")
logger.info("✅ 향상된 에러 처리")
logger.info("✅ M3 Max 128GB 최적화")
logger.info("✅ 프로덕션 레벨 안정성")
logger.info("=" * 80)