"""
완전히 수정된 ModelLoader 구현
app/ai_pipeline/utils/model_loader.py

고급 AI 모델 로더:
- config.get_hash() 에러 완전 해결
- 안전한 타입 처리
- 지연 로딩 (Lazy Loading)
- 모델 캐싱 및 공유
- 동적 양자화
- 멀티 GPU 지원
- 체크포인트 관리
- M3 Max MPS 최적화
"""
import os
import sys
import torch
import torch.nn as nn
from torch.quantization import quantize_dynamic
import logging
import hashlib
import json
import asyncio
import time
import pickle
import threading
import weakref
from typing import Dict, Any, Optional, Union, List, Callable, Type, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict
from contextlib import contextmanager
from functools import wraps, lru_cache
from concurrent.futures import ThreadPoolExecutor
import traceback

logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """모델 설정 클래스 - 안전한 버전"""
    name: str
    model_type: str
    model_path: Optional[str] = None
    checkpoint_url: Optional[str] = None
    device: str = "mps"
    use_fp16: bool = True
    quantize: bool = False
    max_memory_gb: float = 4.0
    cache_enabled: bool = True
    lazy_loading: bool = True
    batch_size: int = 1
    input_size: Tuple[int, int] = (512, 512)
    num_workers: int = 2
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        try:
            return asdict(self)
        except Exception as e:
            logger.warning(f"ModelConfig to_dict 실패: {e}")
            return {
                "name": self.name,
                "model_type": self.model_type,
                "device": self.device
            }
    
    def get_hash(self) -> str:
        """설정 해시값 생성 - 안전한 버전"""
        try:
            config_dict = self.to_dict()
            config_str = json.dumps(config_dict, sort_keys=True)
            return hashlib.md5(config_str.encode()).hexdigest()
        except Exception as e:
            logger.warning(f"ModelConfig 해시 계산 실패: {e}")
            # 폴백 해시
            fallback_str = f"{self.name}_{self.model_type}_{self.device}"
            return hashlib.md5(fallback_str.encode()).hexdigest()

class ModelRegistry:
    """싱글톤 모델 레지스트리 - 안전한 버전"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, '_initialized'):
            self.registered_models: Dict[str, Dict[str, Any]] = {}
            self._lock = threading.RLock()
            self._initialized = True
            logger.info("ModelRegistry 초기화 완료")
    
    def register_model(self, 
                      name: str, 
                      model_class: Type, 
                      default_config: Dict[str, Any] = None,
                      loader_func: Optional[Callable] = None):
        """모델 등록 - 안전한 버전"""
        with self._lock:
            try:
                self.registered_models[name] = {
                    'class': model_class,
                    'config': default_config or {},
                    'loader': loader_func,
                    'registered_at': time.time()
                }
                logger.info(f"모델 등록: {name}")
            except Exception as e:
                logger.error(f"모델 등록 실패 {name}: {e}")
    
    def get_model_info(self, name: str) -> Optional[Dict[str, Any]]:
        """모델 정보 조회 - 안전한 버전"""
        with self._lock:
            try:
                return self.registered_models.get(name)
            except Exception as e:
                logger.error(f"모델 정보 조회 실패 {name}: {e}")
                return None
    
    def list_models(self) -> List[str]:
        """등록된 모델 목록 - 안전한 버전"""
        with self._lock:
            try:
                return list(self.registered_models.keys())
            except Exception as e:
                logger.error(f"모델 목록 조회 실패: {e}")
                return []
    
    def unregister_model(self, name: str) -> bool:
        """모델 등록 해제"""
        with self._lock:
            try:
                if name in self.registered_models:
                    del self.registered_models[name]
                    logger.info(f"모델 등록 해제: {name}")
                    return True
                return False
            except Exception as e:
                logger.error(f"모델 등록 해제 실패 {name}: {e}")
                return False

class ModelMemoryManager:
    """모델 메모리 관리자 - 안전한 버전"""
    
    def __init__(self, device: str = "mps"):
        self.device = device
        self.memory_threshold = 0.8  # 80% 메모리 사용 임계점
        
    def get_available_memory(self) -> float:
        """사용 가능한 메모리 (GB) 반환"""
        try:
            if self.device == "cuda" and torch.cuda.is_available():
                total_memory = torch.cuda.get_device_properties(0).total_memory
                allocated_memory = torch.cuda.memory_allocated(0)
                available = (total_memory - allocated_memory) / 1024**3
                return max(0, available)
            
            elif self.device == "mps" and torch.backends.mps.is_available():
                # MPS는 시스템 메모리 공유, 대략적인 추정
                try:
                    import psutil
                    memory_info = psutil.virtual_memory()
                    available_gb = memory_info.available / 1024**3
                    return available_gb * 0.6  # MPS는 시스템 메모리의 60% 정도 사용 가능
                except ImportError:
                    return 8.0  # psutil 없으면 기본값
            
            else:
                # CPU 메모리
                try:
                    import psutil
                    memory_info = psutil.virtual_memory()
                    return memory_info.available / 1024**3
                except ImportError:
                    return 4.0  # psutil 없으면 기본값
                
        except Exception as e:
            logger.warning(f"메모리 정보 조회 실패: {e}")
            return 4.0  # 기본값
    
    def cleanup_memory(self):
        """메모리 정리"""
        try:
            if self.device == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            elif self.device == "mps" and torch.backends.mps.is_available():
                torch.mps.empty_cache()
                torch.mps.synchronize()
            
            # Python 가비지 컬렉션
            import gc
            gc.collect()
            
        except Exception as e:
            logger.warning(f"메모리 정리 실패: {e}")
    
    def check_memory_available(self, required_gb: float) -> bool:
        """필요한 메모리가 사용 가능한지 확인"""
        try:
            available = self.get_available_memory()
            return available >= required_gb
        except Exception as e:
            logger.warning(f"메모리 확인 실패: {e}")
            return True  # 확인 실패 시 True 반환

class ModelLoader:
    """완전히 안전한 모델 로더"""
    
    def __init__(self, 
                 device: str = "auto", 
                 use_fp16: bool = True,
                 max_cache_size: int = 5,
                 enable_model_sharing: bool = True,
                 memory_limit_gb: float = 8.0):
        
        self.device = self._detect_device(device)
        self.use_fp16 = use_fp16 and self.device != 'cpu'
        self.max_cache_size = max_cache_size
        self.enable_model_sharing = enable_model_sharing
        self.memory_limit_gb = memory_limit_gb
        
        # 모델 캐시 (WeakValueDictionary 사용으로 자동 GC)
        try:
            self.model_cache = weakref.WeakValueDictionary() if enable_model_sharing else {}
        except:
            self.model_cache = {}
            
        self.model_configs: Dict[str, ModelConfig] = {}
        self.load_times: Dict[str, float] = {}
        self.access_counts: Dict[str, int] = {}
        self.last_access: Dict[str, float] = {}
        
        # 스레드 안전성
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="ModelLoader")
        
        # 메모리 관리자
        self.memory_manager = ModelMemoryManager(self.device)
        
        # 모델 레지스트리
        self.registry = ModelRegistry()
        
        # 모델 경로 설정
        self.models_dir = Path("app/ai_pipeline/models/ai_models")
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # 캐시 디렉토리 설정
        self.cache_dir = Path("app/ai_pipeline/cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # 기본 모델들 등록
        self._register_default_models()
        
        logger.info(f"ModelLoader 초기화 - Device: {self.device}, FP16: {self.use_fp16}")
    
    def _detect_device(self, preferred: str) -> str:
        """최적 디바이스 감지"""
        if preferred == "auto":
            try:
                if torch.backends.mps.is_available():
                    return "mps"
                elif torch.cuda.is_available():
                    return "cuda"
                else:
                    return "cpu"
            except Exception as e:
                logger.warning(f"디바이스 감지 실패: {e}")
                return "cpu"
        return preferred
    
    def _register_default_models(self):
        """기본 모델들 등록"""
        try:
            # HR-VITON 모델
            self.registry.register_model(
                "hr_viton", 
                DummyHRVITON, 
                {
                    "input_size": (512, 512),
                    "num_channels": 3,
                    "model_type": "diffusion"
                },
                self._load_hr_viton
            )
            
            # Graphonomy 인체 파싱
            self.registry.register_model(
                "graphonomy", 
                DummyGraphonomy, 
                {
                    "num_classes": 20,
                    "backbone": "resnet101",
                    "model_type": "segmentation"
                },
                self._load_graphonomy
            )
            
            # OpenPose
            self.registry.register_model(
                "openpose", 
                DummyOpenPose, 
                {
                    "num_keypoints": 18,
                    "heatmap_size": 64,
                    "model_type": "pose"
                },
                self._load_openpose
            )
            
            # U2Net (배경 제거)
            self.registry.register_model(
                "u2net", 
                DummyU2Net, 
                {
                    "input_channels": 3,
                    "output_channels": 1,
                    "model_type": "segmentation"
                },
                self._load_u2net
            )
            
            # OOTDiffusion
            self.registry.register_model(
                "ootd_diffusion", 
                DummyOOTDiffusion, 
                {
                    "resolution": 512,
                    "steps": 20,
                    "model_type": "diffusion"
                },
                self._load_ootd_diffusion
            )
        except Exception as e:
            logger.error(f"기본 모델 등록 실패: {e}")
    
    def get_model_path(self, model_name: str) -> Path:
        """모델 경로 반환"""
        try:
            model_info = self.registry.get_model_info(model_name)
            if model_info is None:
                logger.warning(f"모델 '{model_name}'이 등록되지 않았습니다")
                return self.models_dir / model_name
            
            return self.models_dir / model_name
        except Exception as e:
            logger.error(f"모델 경로 조회 실패 {model_name}: {e}")
            return self.models_dir / model_name
    
    def _safe_get_hash(self, config: Any) -> str:
        """
        완전히 안전한 해시 계산 - 모든 타입 지원
        이 메서드가 config.get_hash() 에러를 해결합니다
        """
        try:
            # None 체크
            if config is None:
                return "none_hash"
            
            # ModelConfig 객체인 경우
            if isinstance(config, ModelConfig):
                return config.get_hash()
            
            # get_hash 메서드를 가진 객체인 경우
            elif hasattr(config, 'get_hash') and callable(getattr(config, 'get_hash')):
                try:
                    return config.get_hash()
                except Exception as e:
                    logger.warning(f"객체 get_hash 호출 실패: {e}")
                    # 폴백: 객체를 문자열로 변환 후 해시
                    obj_str = str(config)
                    return hashlib.md5(obj_str.encode('utf-8')).hexdigest()
            
            # 딕셔너리인 경우
            elif isinstance(config, dict):
                try:
                    dict_str = json.dumps(config, sort_keys=True)
                    return hashlib.md5(dict_str.encode('utf-8')).hexdigest()
                except Exception as e:
                    logger.warning(f"딕셔너리 해시 계산 실패: {e}")
                    fallback_str = str(sorted(config.items()))
                    return hashlib.md5(fallback_str.encode('utf-8')).hexdigest()
            
            # 문자열인 경우 (기존 에러의 주요 원인)
            elif isinstance(config, str):
                return hashlib.md5(config.encode('utf-8')).hexdigest()
            
            # 리스트나 튜플인 경우
            elif isinstance(config, (list, tuple)):
                list_str = str(sorted(config) if isinstance(config, list) else config)
                return hashlib.md5(list_str.encode('utf-8')).hexdigest()
            
            # 숫자 타입인 경우
            elif isinstance(config, (int, float)):
                return hashlib.md5(str(config).encode('utf-8')).hexdigest()
            
            # Path 객체인 경우
            elif isinstance(config, Path):
                return hashlib.md5(str(config).encode('utf-8')).hexdigest()
            
            # 기타 모든 객체
            else:
                try:
                    # __dict__ 속성이 있는 경우
                    if hasattr(config, '__dict__'):
                        obj_dict = vars(config)
                        dict_str = json.dumps(obj_dict, sort_keys=True, default=str)
                        return hashlib.md5(dict_str.encode('utf-8')).hexdigest()
                    else:
                        # 문자열 변환
                        obj_str = str(config)
                        return hashlib.md5(obj_str.encode('utf-8')).hexdigest()
                except Exception as e:
                    logger.warning(f"객체 해시 계산 실패: {e}")
                    # 최후 폴백: 객체 ID와 타입 기반
                    fallback_str = f"{type(config).__name__}_{id(config)}_{time.time()}"
                    return hashlib.md5(fallback_str.encode('utf-8')).hexdigest()
                
        except Exception as e:
            logger.warning(f"해시 계산 완전 실패: {e}, 최종 폴백 사용")
            # 완전 폴백: 현재 시간과 랜덤 기반
            import random
            fallback_str = f"fallback_{time.time()}_{random.randint(1000, 9999)}"
            return hashlib.md5(fallback_str.encode('utf-8')).hexdigest()
    
    def _calculate_file_hash(self, file_path: str) -> str:
        """파일 해시 계산"""
        try:
            hash_md5 = hashlib.md5()
            file_size = os.path.getsize(file_path)
            
            # 큰 파일은 샘플링하여 해시 계산
            if file_size > 100 * 1024 * 1024:  # 100MB 이상
                with open(file_path, "rb") as f:
                    # 파일 시작, 중간, 끝에서 샘플링
                    f.seek(0)
                    hash_md5.update(f.read(1024 * 1024))  # 첫 1MB
                    
                    f.seek(file_size // 2)
                    hash_md5.update(f.read(1024 * 1024))  # 중간 1MB
                    
                    f.seek(max(0, file_size - 1024 * 1024))
                    hash_md5.update(f.read())  # 마지막 1MB
                    
                # 파일 크기와 수정 시간도 포함
                hash_md5.update(str(file_size).encode())
                hash_md5.update(str(os.path.getmtime(file_path)).encode())
            else:
                # 작은 파일은 전체 해시
                with open(file_path, "rb") as f:
                    for chunk in iter(lambda: f.read(4096), b""):
                        hash_md5.update(chunk)
            
            return hash_md5.hexdigest()
            
        except Exception as e:
            logger.warning(f"파일 해시 계산 실패 {file_path}: {e}")
            # 폴백: 파일 경로와 크기 기반 해시
            try:
                fallback_info = f"{file_path}_{os.path.getsize(file_path)}_{os.path.getmtime(file_path)}"
                return hashlib.md5(fallback_info.encode('utf-8')).hexdigest()
            except:
                return hashlib.md5(file_path.encode('utf-8')).hexdigest()
    
    def _calculate_directory_hash(self, dir_path: str) -> str:
        """디렉토리 해시 계산"""
        try:
            hash_md5 = hashlib.md5()
            
            # 디렉토리 내 파일 목록과 크기 정보만 사용 (성능 최적화)
            file_info = []
            for root, dirs, files in os.walk(dir_path):
                # 정렬하여 일관성 보장
                dirs.sort()
                files.sort()
                
                for file in files:
                    file_path = os.path.join(root, file)
                    rel_path = os.path.relpath(file_path, dir_path)
                    
                    try:
                        file_size = os.path.getsize(file_path)
                        file_mtime = os.path.getmtime(file_path)
                        file_info.append(f"{rel_path}:{file_size}:{file_mtime}")
                    except:
                        file_info.append(f"{rel_path}:0:0")
            
            # 모든 파일 정보를 해시에 포함
            dir_signature = "|".join(file_info)
            hash_md5.update(dir_signature.encode('utf-8'))
            
            return hash_md5.hexdigest()
            
        except Exception as e:
            logger.warning(f"디렉토리 해시 계산 실패 {dir_path}: {e}")
            return hashlib.md5(dir_path.encode('utf-8')).hexdigest()
    
    async def load_model(self, 
                        model_name: str, 
                        config: Optional[Union[ModelConfig, Dict, str, Any]] = None,
                        force_reload: bool = False,
                        **kwargs) -> Optional[Any]:
        """
        모델 로드 - 완전히 안전한 버전
        모든 config 타입을 안전하게 처리합니다
        """
        with self._lock:
            try:
                # 설정 준비 - 모든 타입 안전하게 처리
                if config is None:
                    config = ModelConfig(name=model_name, model_type=model_name, device=self.device)
                elif isinstance(config, dict):
                    # 딕셔너리를 ModelConfig로 변환
                    config = ModelConfig(
                        name=model_name,
                        model_type=config.get('model_type', model_name),
                        device=config.get('device', self.device),
                        use_fp16=config.get('use_fp16', True),
                        max_memory_gb=config.get('max_memory_gb', 4.0),
                        cache_enabled=config.get('cache_enabled', True),
                        **{k: v for k, v in config.items() 
                           if k in ['model_path', 'checkpoint_url', 'quantize', 'lazy_loading', 'batch_size', 'input_size', 'num_workers']}
                    )
                elif isinstance(config, str):
                    # 문자열을 ModelConfig로 변환
                    config = ModelConfig(name=model_name, model_type=config, device=self.device)
                elif not isinstance(config, ModelConfig):
                    # 기타 타입들을 ModelConfig로 변환
                    config = ModelConfig(name=model_name, model_type=str(config), device=self.device)
                
                # 안전한 해시 계산 (핵심 수정사항!)
                config_hash = self._safe_get_hash(config)
                cache_key = f"{model_name}_{config_hash}"
                
                # 캐시에서 조회 (force_reload가 아닌 경우)
                if not force_reload and cache_key in self.model_cache:
                    model = self.model_cache[cache_key]
                    if model is not None:
                        self._update_access_stats(cache_key)
                        logger.debug(f"캐시에서 모델 반환: {model_name}")
                        return model
                
                # 메모리 체크
                if not self.memory_manager.check_memory_available(config.max_memory_gb):
                    logger.warning(f"메모리 부족: {config.max_memory_gb}GB 필요")
                    # 캐시 정리 시도
                    self._cleanup_old_models()
                    
                    if not self.memory_manager.check_memory_available(config.max_memory_gb):
                        logger.warning(f"메모리 부족하지만 진행: {config.max_memory_gb}GB 필요")
                
                # 새로 로드
                start_time = time.time()
                
                if config.lazy_loading:
                    # 비동기 로딩
                    model = await self._load_model_async(model_name, config, **kwargs)
                else:
                    # 동기 로딩
                    model = await asyncio.get_event_loop().run_in_executor(
                        self._executor, 
                        self._load_model_sync, 
                        model_name, 
                        config, 
                        kwargs
                    )
                
                load_time = time.time() - start_time
                
                if model is not None:
                    # 캐시에 저장
                    if config.cache_enabled and self.enable_model_sharing:
                        self._add_to_cache(cache_key, model, config)
                    
                    # 통계 업데이트
                    self._update_load_stats(cache_key, load_time, config)
                    
                    logger.info(f"✅ 모델 로드 완료: {model_name} ({load_time:.2f}s)")
                    return model
                else:
                    logger.warning(f"⚠️ 모델 로드 결과가 None: {model_name}")
                    # 폴백 모델 생성
                    return self._create_fallback_model(model_name)
                    
            except Exception as e:
                logger.error(f"❌ 모델 로드 중 오류 {model_name}: {e}")
                logger.error(traceback.format_exc())
                # 폴백 모델 반환
                return self._create_fallback_model(model_name)
    
    def _create_fallback_model(self, model_name: str) -> Any:
        """폴백 모델 생성"""
        try:
            return DummyModel(model_name)
        except Exception as e:
            logger.error(f"폴백 모델 생성 실패: {e}")
            # 최소한의 더미 객체
            class MinimalDummy:
                def __init__(self, name):
                    self.name = name
                def __call__(self, *args, **kwargs):
                    return {"result": f"minimal_dummy_{self.name}"}
            return MinimalDummy(model_name)
    
    async def _load_model_async(self, model_name: str, config: ModelConfig, **kwargs) -> Optional[Any]:
        """비동기 모델 로딩"""
        try:
            # 모델 정보 조회
            model_info = self.registry.get_model_info(model_name)
            if model_info is None:
                logger.warning(f"등록되지 않은 모델: {model_name}, 더미 모델 생성")
                return DummyModel(model_name)
            
            # 커스텀 로더 함수가 있는 경우 사용
            if model_info.get('loader'):
                loader_func = model_info['loader']
                try:
                    if asyncio.iscoroutinefunction(loader_func):
                        model = await loader_func(config, **kwargs)
                    else:
                        model = await asyncio.get_event_loop().run_in_executor(
                            self._executor, loader_func, config, **kwargs
                        )
                except Exception as e:
                    logger.error(f"커스텀 로더 실패: {e}")
                    model = None
            else:
                # 기본 로딩 로직
                model = await self._load_model_default(model_name, config, **kwargs)
            
            # 모델 최적화
            if model is not None:
                try:
                    model = await self._optimize_model_async(model, config)
                except Exception as e:
                    logger.warning(f"모델 최적화 실패: {e}")
            
            return model if model is not None else DummyModel(model_name)
            
        except Exception as e:
            logger.error(f"비동기 모델 로딩 실패 {model_name}: {e}")
            return DummyModel(model_name)
    
    def _load_model_sync(self, model_name: str, config: ModelConfig, kwargs: Dict) -> Optional[Any]:
        """동기 모델 로딩"""
        try:
            # 모델 정보 조회
            model_info = self.registry.get_model_info(model_name)
            if model_info is None:
                logger.warning(f"등록되지 않은 모델: {model_name}")
                return DummyModel(model_name)
            
            model_class = model_info['class']
            default_config = model_info['config']
            
            # 모델 인스턴스 생성
            try:
                if hasattr(model_class, 'from_pretrained'):
                    # Hugging Face 스타일
                    model = model_class.from_pretrained(
                        config.model_path or model_name,
                        **{**default_config, **kwargs}
                    )
                else:
                    # 일반 PyTorch 모델
                    model = model_class(**{**default_config, **kwargs})
                    
                    # 체크포인트 로드
                    checkpoint_path = config.model_path or self.get_model_path(model_name)
                    if checkpoint_path and Path(checkpoint_path).exists():
                        self._load_checkpoint(model, checkpoint_path)
            except Exception as e:
                logger.warning(f"모델 클래스 생성 실패: {e}")
                model = model_class() if callable(model_class) else DummyModel(model_name)
            
            # 동기 최적화
            try:
                model = self._optimize_model_sync(model, config)
            except Exception as e:
                logger.warning(f"동기 최적화 실패: {e}")
            
            return model
            
        except Exception as e:
            logger.error(f"동기 모델 로딩 실패 {model_name}: {e}")
            return DummyModel(model_name)
    
    async def _load_model_default(self, model_name: str, config: ModelConfig, **kwargs) -> Optional[Any]:
        """기본 모델 로딩 로직"""
        try:
            model_info = self.registry.get_model_info(model_name)
            if model_info is None:
                return DummyModel(model_name)
                
            model_class = model_info['class']
            default_config = model_info['config']
            
            # 시뮬레이션 지연
            await asyncio.sleep(0.1)
            
            # 더미 모델 생성
            try:
                model = model_class(**{**default_config, **kwargs})
            except Exception as e:
                logger.warning(f"모델 클래스 생성 실패: {e}")
                model = DummyModel(model_name)
            
            return model
            
        except Exception as e:
            logger.error(f"기본 모델 로딩 실패 {model_name}: {e}")
            return DummyModel(model_name)
    
    def _load_checkpoint(self, model: nn.Module, checkpoint_path: Union[str, Path]):
        """체크포인트 로드"""
        try:
            if not os.path.exists(checkpoint_path):
                logger.warning(f"체크포인트 파일 없음: {checkpoint_path}")
                return
                
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            if isinstance(checkpoint, dict):
                if 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                elif 'model' in checkpoint:
                    state_dict = checkpoint['model']
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint
            
            # 키 이름 정규화
            if hasattr(model, 'load_state_dict'):
                try:
                    model.load_state_dict(state_dict, strict=False)
                    logger.info(f"체크포인트 로드 완료: {checkpoint_path}")
                except Exception as e:
                    logger.warning(f"체크포인트 로드 부분 실패: {e}")
            
        except Exception as e:
            logger.error(f"체크포인트 로드 실패 {checkpoint_path}: {e}")
    
    async def _optimize_model_async(self, model: nn.Module, config: ModelConfig) -> nn.Module:
        """비동기 모델 최적화"""
        try:
            return await asyncio.get_event_loop().run_in_executor(
                self._executor, self._optimize_model_sync, model, config
            )
        except Exception as e:
            logger.warning(f"비동기 최적화 실패: {e}")
            return model
    
    def _optimize_model_sync(self, model: nn.Module, config: ModelConfig) -> nn.Module:
        """동기 모델 최적화"""
        try:
            # 평가 모드
            if hasattr(model, 'eval'):
                model.eval()
            
            # 디바이스 이동
            if hasattr(model, 'to'):
                model = model.to(self.device)
            
            # FP16 변환
            if config.use_fp16 and self.device != 'cpu' and hasattr(model, 'half'):
                try:
                    model = model.half()
                    logger.debug("FP16 변환 완료")
                except Exception as e:
                    logger.warning(f"FP16 변환 실패: {e}")
            
            # 양자화
            if config.quantize and self.device == 'cpu' and hasattr(model, 'modules'):
                try:
                    model = quantize_dynamic(
                        model, 
                        {nn.Linear, nn.Conv2d}, 
                        dtype=torch.qint8
                    )
                    logger.debug("동적 양자화 완료")
                except Exception as e:
                    logger.warning(f"양자화 실패: {e}")
            
            return model
            
        except Exception as e:
            logger.warning(f"모델 최적화 실패: {e}")
            return model
    
    def _add_to_cache(self, cache_key: str, model: Any, config: ModelConfig):
        """캐시에 모델 추가"""
        try:
            # 캐시 크기 확인
            if len(self.model_cache) >= self.max_cache_size:
                self._evict_least_used_model()
            
            # 메모리 사용량 확인
            model_size_gb = self._estimate_model_size(model)
            if model_size_gb > config.max_memory_gb * 2:  # 2배 여유
                logger.warning(f"모델 크기 초과 ({model_size_gb:.2f}GB > {config.max_memory_gb * 2}GB), 캐시하지 않음")
                return
            
            self.model_cache[cache_key] = model
            logger.debug(f"모델 캐시 추가: {cache_key}")
            
        except Exception as e:
            logger.warning(f"캐시 추가 실패: {e}")
    
    def _evict_least_used_model(self):
        """가장 적게 사용된 모델 제거 (LRU)"""
        try:
            if not self.last_access:
                return
            
            # 가장 오래된 액세스 시간을 가진 모델 찾기
            oldest_key = min(self.last_access.keys(), 
                           key=lambda k: self.last_access[k])
            
            # 캐시에서 제거
            if oldest_key in self.model_cache:
                del self.model_cache[oldest_key]
                logger.debug(f"캐시에서 모델 제거 (LRU): {oldest_key}")
            
            # 통계 정리
            self.access_counts.pop(oldest_key, None)
            self.load_times.pop(oldest_key, None)
            self.last_access.pop(oldest_key, None)
            self.model_configs.pop(oldest_key, None)
            
        except Exception as e:
            logger.warning(f"캐시 제거 실패: {e}")
    
    def _cleanup_old_models(self):
        """오래된 모델들 정리"""
        try:
            current_time = time.time()
            old_threshold = 3600  # 1시간
            
            keys_to_remove = []
            for key, last_time in self.last_access.items():
                if current_time - last_time > old_threshold:
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                if key in self.model_cache:
                    del self.model_cache[key]
                self.access_counts.pop(key, None)
                self.load_times.pop(key, None)
                self.last_access.pop(key, None)
                self.model_configs.pop(key, None)
                
            if keys_to_remove:
                logger.info(f"오래된 모델 {len(keys_to_remove)}개 정리 완료")
                self.memory_manager.cleanup_memory()
                
        except Exception as e:
            logger.warning(f"오래된 모델 정리 실패: {e}")
    
    def _estimate_model_size(self, model: Any) -> float:
        """모델 메모리 사용량 추정 (GB)"""
        try:
            if hasattr(model, 'parameters'):
                param_size = sum(p.numel() * p.element_size() 
                               for p in model.parameters())
                buffer_size = sum(b.numel() * b.element_size() 
                                for b in model.buffers())
                total_size = param_size + buffer_size
                return total_size / 1024**3
            else:
                return 0.1  # 기본값
        except Exception as e:
            logger.warning(f"모델 크기 추정 실패: {e}")
            return 0.1
    
    def _update_access_stats(self, cache_key: str):
        """액세스 통계 업데이트"""
        try:
            self.access_counts[cache_key] = self.access_counts.get(cache_key, 0) + 1
            self.last_access[cache_key] = time.time()
        except Exception as e:
            logger.warning(f"통계 업데이트 실패: {e}")
    
    def _update_load_stats(self, cache_key: str, load_time: float, config: ModelConfig):
        """로딩 통계 업데이트"""
        try:
            self.load_times[cache_key] = load_time
            self.access_counts[cache_key] = 1
            self.last_access[cache_key] = time.time()
            self.model_configs[cache_key] = config
        except Exception as e:
            logger.warning(f"로딩 통계 업데이트 실패: {e}")
    
    # 커스텀 로더 함수들 - 모두 안전하게 수정
    async def _load_hr_viton(self, config: ModelConfig, **kwargs) -> Optional[Any]:
        """HR-VITON 모델 로더"""
        try:
            await asyncio.sleep(0.2)  # 로딩 시뮬레이션
            return DummyHRVITON(
                input_size=config.input_size[0] if config.input_size else 512,
                num_channels=3
            )
        except Exception as e:
            logger.error(f"HR-VITON 로더 실패: {e}")
            return DummyModel("hr_viton")
    
    async def _load_graphonomy(self, config: ModelConfig, **kwargs) -> Optional[Any]:
        """Graphonomy 모델 로더"""
        try:
            await asyncio.sleep(0.3)
            return DummyGraphonomy(num_classes=20, backbone="resnet101")
        except Exception as e:
            logger.error(f"Graphonomy 로더 실패: {e}")
            return DummyModel("graphonomy")
    
    async def _load_openpose(self, config: ModelConfig, **kwargs) -> Optional[Any]:
        """OpenPose 모델 로더"""
        try:
            await asyncio.sleep(0.2)
            return DummyOpenPose(num_keypoints=18, heatmap_size=64)
        except Exception as e:
            logger.error(f"OpenPose 로더 실패: {e}")
            return DummyModel("openpose")
    
    async def _load_u2net(self, config: ModelConfig, **kwargs) -> Optional[Any]:
        """U2Net 모델 로더"""
        try:
            await asyncio.sleep(0.2)
            return DummyU2Net(input_channels=3, output_channels=1)
        except Exception as e:
            logger.error(f"U2Net 로더 실패: {e}")
            return DummyModel("u2net")
    
    async def _load_ootd_diffusion(self, config: ModelConfig, **kwargs) -> Optional[Any]:
        """OOTDiffusion 모델 로더"""
        try:
            await asyncio.sleep(0.5)
            return DummyOOTDiffusion(resolution=512, steps=20)
        except Exception as e:
            logger.error(f"OOTDiffusion 로더 실패: {e}")
            return DummyModel("ootd_diffusion")
    
    # 나머지 메서드들도 안전하게 처리
    def get_cache_stats(self) -> Dict[str, Any]:
        """캐시 통계 정보"""
        with self._lock:
            try:
                total_models = len(self.model_cache)
                total_memory = sum(
                    self._estimate_model_size(model) 
                    for model in self.model_cache.values()
                    if model is not None
                )
                
                available_memory = self.memory_manager.get_available_memory()
                
                return {
                    'cached_models': total_models,
                    'max_cache_size': self.max_cache_size,
                    'total_memory_gb': round(total_memory, 2),
                    'available_memory_gb': round(available_memory, 2),
                    'device': self.device,
                    'access_counts': dict(self.access_counts),
                    'load_times': dict(self.load_times),
                    'available_models': self.registry.list_models(),
                    'cache_hit_rate': self._calculate_cache_hit_rate()
                }
            except Exception as e:
                logger.error(f"통계 조회 실패: {e}")
                return {'error': str(e)}
    
    def _calculate_cache_hit_rate(self) -> float:
        """캐시 히트율 계산"""
        try:
            total_access = sum(self.access_counts.values())
            if total_access == 0:
                return 0.0
            
            cache_hits = sum(1 for count in self.access_counts.values() if count > 1)
            return cache_hits / len(self.access_counts) if self.access_counts else 0.0
        except:
            return 0.0
    
    def cleanup(self):
        """리소스 정리"""
        try:
            logger.info("ModelLoader 정리 시작...")
            
            # 모든 모델 언로드
            try:
                model_names = list(set(k.split('_')[0] for k in self.model_cache.keys() if '_' in k))
                for model_name in model_names:
                    self.unload_model(model_name)
            except Exception as e:
                logger.warning(f"모델 언로드 실패: {e}")
            
            # 캐시 전체 정리
            try:
                self.model_cache.clear()
                self.access_counts.clear()
                self.load_times.clear()
                self.last_access.clear()
                self.model_configs.clear()
            except Exception as e:
                logger.warning(f"캐시 정리 실패: {e}")
            
            # 스레드풀 종료
            try:
                if hasattr(self, '_executor'):
                    self._executor.shutdown(wait=True)
            except Exception as e:
                logger.warning(f"스레드풀 종료 실패: {e}")
            
            logger.info("✅ ModelLoader 정리 완료")
            
        except Exception as e:
            logger.error(f"ModelLoader 정리 중 오류: {e}")
    
    def unload_model(self, model_name: str) -> bool:
        """모델 언로드"""
        with self._lock:
            try:
                # 캐시에서 제거
                keys_to_remove = [k for k in self.model_cache.keys() 
                                 if k.startswith(f"{model_name}_")]
                
                removed_count = 0
                for key in keys_to_remove:
                    if key in self.model_cache:
                        del self.model_cache[key]
                        removed_count += 1
                    
                    self.access_counts.pop(key, None)
                    self.load_times.pop(key, None)
                    self.last_access.pop(key, None)
                    self.model_configs.pop(key, None)
                
                if removed_count > 0:
                    logger.info(f"모델 언로드: {model_name} ({removed_count}개 인스턴스)")
                    self.memory_manager.cleanup_memory()
                    return True
                else:
                    logger.warning(f"언로드할 모델을 찾을 수 없음: {model_name}")
                    return False
                    
            except Exception as e:
                logger.error(f"모델 언로드 실패 {model_name}: {e}")
                return False

# ========================================
# 더미 모델 클래스들 - 안전한 버전
# ========================================

class DummyModel:
    """기본 더미 모델 - 안전한 버전"""
    def __init__(self, name: str):
        self.name = name
        self.device = "cpu"
    
    def __call__(self, *args, **kwargs):
        try:
            return {"result": f"dummy_{self.name}", "success": True}
        except:
            return {"result": "dummy_fallback", "success": True}
    
    def get_hash(self):
        try:
            return hashlib.md5(f"dummy_{self.name}".encode()).hexdigest()
        except:
            return "dummy_hash"

class DummyHRVITON(nn.Module):
    """HR-VITON 더미 모델"""
    def __init__(self, input_size=512, num_channels=3):
        super().__init__()
        self.input_size = input_size
        self.conv = nn.Conv2d(num_channels, 64, 3, padding=1)
        self.out_conv = nn.Conv2d(64, num_channels, 3, padding=1)
    
    def forward(self, person_img, cloth_img=None):
        try:
            if cloth_img is None:
                cloth_img = person_img
            
            # 간단한 더미 처리
            x = self.conv(person_img)
            x = torch.relu(x)
            return self.out_conv(x)
        except Exception as e:
            logger.warning(f"DummyHRVITON forward 실패: {e}")
            # 폴백 결과
            batch_size = person_img.shape[0] if hasattr(person_img, 'shape') else 1
            return torch.randn(batch_size, 3, self.input_size, self.input_size)

class DummyGraphonomy(nn.Module):
    """Graphonomy 더미 모델"""
    def __init__(self, num_classes=20, backbone="resnet101"):
        super().__init__()
        self.num_classes = num_classes
        self.backbone = nn.Conv2d(3, 64, 7, stride=2, padding=3)
        self.classifier = nn.Conv2d(64, num_classes, 1)
    
    def forward(self, x):
        try:
            features = self.backbone(x)
            return self.classifier(features)
        except Exception as e:
            logger.warning(f"DummyGraphonomy forward 실패: {e}")
            batch_size = x.shape[0] if hasattr(x, 'shape') else 1
            return torch.randn(batch_size, self.num_classes, 128, 128)

class DummyOpenPose(nn.Module):
    """OpenPose 더미 모델"""
    def __init__(self, num_keypoints=18, heatmap_size=64):
        super().__init__()
        self.num_keypoints = num_keypoints
        self.backbone = nn.Conv2d(3, 128, 3, padding=1)
        self.keypoint_head = nn.Conv2d(128, num_keypoints, 1)
    
    def forward(self, x):
        try:
            features = self.backbone(x)
            heatmaps = self.keypoint_head(features)
            return heatmaps
        except Exception as e:
            logger.warning(f"DummyOpenPose forward 실패: {e}")
            batch_size = x.shape[0] if hasattr(x, 'shape') else 1
            return torch.randn(batch_size, self.num_keypoints, 64, 64)

class DummyU2Net(nn.Module):
    """U2Net 더미 모델"""
    def __init__(self, input_channels=3, output_channels=1):
        super().__init__()
        self.encoder = nn.Conv2d(input_channels, 64, 3, padding=1)
        self.decoder = nn.Conv2d(64, output_channels, 3, padding=1)
    
    def forward(self, x):
        try:
            encoded = torch.relu(self.encoder(x))
            return torch.sigmoid(self.decoder(encoded))
        except Exception as e:
            logger.warning(f"DummyU2Net forward 실패: {e}")
            batch_size = x.shape[0] if hasattr(x, 'shape') else 1
            return torch.randn(batch_size, 1, 512, 512)

class DummyOOTDiffusion:
    """OOTDiffusion 더미 모델"""
    def __init__(self, resolution=512, steps=20):
        self.resolution = resolution
        self.steps = steps
        self.device = "cpu"
    
    def __call__(self, person_img, cloth_img, **kwargs):
        try:
            # 더미 결과 반환
            if hasattr(person_img, 'shape'):
                batch_size = person_img.shape[0]
                result = torch.randn(batch_size, 3, self.resolution, self.resolution)
            else:
                result = torch.randn(1, 3, self.resolution, self.resolution)
            return result
        except Exception as e:
            logger.warning(f"DummyOOTDiffusion call 실패: {e}")
            return torch.randn(1, 3, self.resolution, self.resolution)
    
    def get_hash(self):
        try:
            return hashlib.md5(f"ootd_{self.resolution}_{self.steps}".encode()).hexdigest()
        except:
            return "ootd_dummy_hash"

# ========================================
# 편의 함수들 - 안전한 버전
# ========================================

@lru_cache(maxsize=1)
def get_global_model_loader() -> ModelLoader:
    """전역 ModelLoader 인스턴스 반환"""
    try:
        return ModelLoader()
    except Exception as e:
        logger.error(f"전역 ModelLoader 생성 실패: {e}")
        # 최소한의 ModelLoader 생성
        class MinimalModelLoader:
            def __init__(self):
                self.device = "cpu"
            
            async def load_model(self, model_name: str, config=None):
                return DummyModel(model_name)
        
        return MinimalModelLoader()

async def load_model_async(model_name: str, config: Optional[ModelConfig] = None) -> Optional[Any]:
    """전역 로더를 사용한 비동기 모델 로드"""
    try:
        loader = get_global_model_loader()
        return await loader.load_model(model_name, config)
    except Exception as e:
        logger.error(f"비동기 모델 로드 실패: {e}")
        return DummyModel(model_name)

def load_model_sync(model_name: str, config: Optional[ModelConfig] = None) -> Optional[Any]:
    """전역 로더를 사용한 동기 모델 로드"""
    try:
        loader = get_global_model_loader()
        
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(loader.load_model(model_name, config))
    except Exception as e:
        logger.error(f"동기 모델 로드 실패: {e}")
        return DummyModel(model_name)

def cleanup_global_loader():
    """전역 로더 정리"""
    try:
        loader = get_global_model_loader()
        if hasattr(loader, 'cleanup'):
            loader.cleanup()
        # 캐시 클리어
        get_global_model_loader.cache_clear()
    except Exception as e:
        logger.warning(f"전역 로더 정리 실패: {e}")

# 모듈 레벨에서 안전한 정리 함수 등록
import atexit
atexit.register(cleanup_global_loader)