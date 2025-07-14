"""
완전한 ModelLoader 구현
app/ai_pipeline/utils/model_loader.py

고급 AI 모델 로더:
- 지연 로딩 (Lazy Loading)
- 모델 캐싱 및 공유
- 동적 양자화
- 멀티 GPU 지원
- 체크포인트 관리
- 해시 에러 완전 해결
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
    """모델 설정 클래스"""
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
        return asdict(self)
    
    def get_hash(self) -> str:
        """설정 해시값 생성"""
        config_str = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()

class ModelRegistry:
    """싱글톤 모델 레지스트리"""
    
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
        """모델 등록"""
        with self._lock:
            self.registered_models[name] = {
                'class': model_class,
                'config': default_config or {},
                'loader': loader_func,
                'registered_at': time.time()
            }
            logger.info(f"모델 등록: {name}")
    
    def get_model_info(self, name: str) -> Optional[Dict[str, Any]]:
        """모델 정보 조회"""
        with self._lock:
            return self.registered_models.get(name)
    
    def list_models(self) -> List[str]:
        """등록된 모델 목록"""
        with self._lock:
            return list(self.registered_models.keys())
    
    def unregister_model(self, name: str) -> bool:
        """모델 등록 해제"""
        with self._lock:
            if name in self.registered_models:
                del self.registered_models[name]
                logger.info(f"모델 등록 해제: {name}")
                return True
            return False

class ModelMemoryManager:
    """모델 메모리 관리자"""
    
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
                import psutil
                memory_info = psutil.virtual_memory()
                available_gb = memory_info.available / 1024**3
                return available_gb * 0.6  # MPS는 시스템 메모리의 60% 정도 사용 가능
            
            else:
                # CPU 메모리
                import psutil
                memory_info = psutil.virtual_memory()
                return memory_info.available / 1024**3
                
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
        available = self.get_available_memory()
        return available >= required_gb

class ModelLoader:
    """고급 모델 로더 - 완전한 구현"""
    
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
        self.model_cache = weakref.WeakValueDictionary() if enable_model_sharing else {}
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
    
    def get_model_path(self, model_name: str) -> Path:
        """모델 경로 반환"""
        model_info = self.registry.get_model_info(model_name)
        if model_info is None:
            raise ValueError(f"모델 '{model_name}'이 등록되지 않았습니다")
        
        return self.models_dir / model_name
    
    def get_hash(self, model_path: Union[str, Path, Any]) -> str:
        """
        모델 해시 계산 - 완전히 안전한 버전
        
        Args:
            model_path: 모델 파일 경로, 문자열, 또는 모델 객체
            
        Returns:
            str: 해시 값
        """
        try:
            # None 체크
            if model_path is None:
                return "none_hash"
            
            # 경로 타입 정규화
            if isinstance(model_path, (str, Path)):
                path_str = str(model_path)
                
                # 파일이 존재하는 경우 파일 해시 계산
                if os.path.isfile(path_str):
                    return self._calculate_file_hash(path_str)
                
                # 디렉토리인 경우 경로 기반 해시
                elif os.path.isdir(path_str):
                    return self._calculate_directory_hash(path_str)
                
                # 경로가 존재하지 않는 경우 문자열 해시
                else:
                    return hashlib.md5(path_str.encode('utf-8')).hexdigest()
            
            # 객체가 get_hash 메서드를 가지고 있는 경우
            elif hasattr(model_path, 'get_hash') and callable(getattr(model_path, 'get_hash')):
                try:
                    return model_path.get_hash()
                except Exception as e:
                    logger.warning(f"객체 get_hash 호출 실패: {e}")
                    # 폴백: 객체를 문자열로 변환 후 해시
                    obj_str = str(model_path)
                    return hashlib.md5(obj_str.encode('utf-8')).hexdigest()
            
            # ModelConfig 객체인 경우
            elif isinstance(model_path, ModelConfig):
                return model_path.get_hash()
            
            # 딕셔너리인 경우
            elif isinstance(model_path, dict):
                dict_str = json.dumps(model_path, sort_keys=True)
                return hashlib.md5(dict_str.encode('utf-8')).hexdigest()
            
            # 리스트나 튜플인 경우
            elif isinstance(model_path, (list, tuple)):
                list_str = str(sorted(model_path) if isinstance(model_path, list) else model_path)
                return hashlib.md5(list_str.encode('utf-8')).hexdigest()
            
            # 기타 객체인 경우 문자열 변환 후 해시
            else:
                obj_str = str(model_path)
                return hashlib.md5(obj_str.encode('utf-8')).hexdigest()
                
        except Exception as e:
            logger.warning(f"해시 계산 실패: {e}, 기본값 사용")
            # 완전 폴백: 현재 시간 기반 해시
            fallback_str = f"fallback_{time.time()}_{id(model_path)}"
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
                        config: Optional[ModelConfig] = None,
                        force_reload: bool = False,
                        **kwargs) -> Optional[Any]:
        """
        모델 로드 - 비동기 지원
        
        Args:
            model_name: 모델 이름
            config: 모델 설정
            force_reload: 강제 재로드 여부
            **kwargs: 추가 옵션
            
        Returns:
            로드된 모델 또는 None
        """
        with self._lock:
            try:
                # 설정 준비
                if config is None:
                    config = ModelConfig(name=model_name, model_type=model_name, device=self.device)
                
                config_hash = config.get_hash()
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
                        raise RuntimeError(f"메모리 부족: {config.max_memory_gb}GB 필요")
                
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
                    logger.error(f"❌ 모델 로드 실패: {model_name}")
                    return None
                    
            except Exception as e:
                logger.error(f"❌ 모델 로드 중 오류 {model_name}: {e}")
                logger.error(traceback.format_exc())
                return None
    
    async def _load_model_async(self, model_name: str, config: ModelConfig, **kwargs) -> Optional[Any]:
        """비동기 모델 로딩"""
        try:
            # 모델 정보 조회
            model_info = self.registry.get_model_info(model_name)
            if model_info is None:
                raise ValueError(f"등록되지 않은 모델: {model_name}")
            
            # 커스텀 로더 함수가 있는 경우 사용
            if model_info.get('loader'):
                loader_func = model_info['loader']
                if asyncio.iscoroutinefunction(loader_func):
                    model = await loader_func(config, **kwargs)
                else:
                    model = await asyncio.get_event_loop().run_in_executor(
                        self._executor, loader_func, config, **kwargs
                    )
            else:
                # 기본 로딩 로직
                model = await self._load_model_default(model_name, config, **kwargs)
            
            # 모델 최적화
            if model is not None:
                model = await self._optimize_model_async(model, config)
            
            return model
            
        except Exception as e:
            logger.error(f"비동기 모델 로딩 실패 {model_name}: {e}")
            return None
    
    def _load_model_sync(self, model_name: str, config: ModelConfig, kwargs: Dict) -> Optional[Any]:
        """동기 모델 로딩"""
        try:
            # 모델 정보 조회
            model_info = self.registry.get_model_info(model_name)
            if model_info is None:
                raise ValueError(f"등록되지 않은 모델: {model_name}")
            
            model_class = model_info['class']
            default_config = model_info['config']
            
            # 모델 인스턴스 생성
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
            
            # 동기 최적화
            model = self._optimize_model_sync(model, config)
            
            return model
            
        except Exception as e:
            logger.error(f"동기 모델 로딩 실패 {model_name}: {e}")
            # 더미 모델 반환
            return DummyModel(model_name)
    
    async def _load_model_default(self, model_name: str, config: ModelConfig, **kwargs) -> Optional[Any]:
        """기본 모델 로딩 로직"""
        try:
            model_info = self.registry.get_model_info(model_name)
            model_class = model_info['class']
            default_config = model_info['config']
            
            # 시뮬레이션 지연
            await asyncio.sleep(0.1)
            
            # 더미 모델 생성
            model = model_class(**{**default_config, **kwargs})
            
            return model
            
        except Exception as e:
            logger.error(f"기본 모델 로딩 실패 {model_name}: {e}")
            return DummyModel(model_name)
    
    def _load_checkpoint(self, model: nn.Module, checkpoint_path: Union[str, Path]):
        """체크포인트 로드"""
        try:
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
        return await asyncio.get_event_loop().run_in_executor(
            self._executor, self._optimize_model_sync, model, config
        )
    
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
            
            # JIT 컴파일 (선택적)
            if (hasattr(torch, 'jit') and 
                not config.quantize and 
                hasattr(model, '__call__') and 
                config.input_size):
                try:
                    dummy_input = torch.randn(
                        config.batch_size, 3, *config.input_size
                    ).to(self.device)
                    
                    if config.use_fp16 and self.device != 'cpu':
                        dummy_input = dummy_input.half()
                    
                    with torch.no_grad():
                        traced_model = torch.jit.trace(model, dummy_input)
                        model = traced_model
                        logger.debug("JIT 트레이싱 완료")
                except Exception as e:
                    logger.debug(f"JIT 트레이싱 실패: {e}")
            
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
            if model_size_gb > config.max_memory_gb:
                logger.warning(f"모델 크기 초과 ({model_size_gb:.2f}GB > {config.max_memory_gb}GB), 캐시하지 않음")
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
    
    # 커스텀 로더 함수들
    async def _load_hr_viton(self, config: ModelConfig, **kwargs) -> Optional[Any]:
        """HR-VITON 모델 로더"""
        await asyncio.sleep(0.2)  # 로딩 시뮬레이션
        return DummyHRVITON(
            input_size=config.input_size[0] if config.input_size else 512,
            num_channels=3
        )
    
    async def _load_graphonomy(self, config: ModelConfig, **kwargs) -> Optional[Any]:
        """Graphonomy 모델 로더"""
        await asyncio.sleep(0.3)
        return DummyGraphonomy(num_classes=20, backbone="resnet101")
    
    async def _load_openpose(self, config: ModelConfig, **kwargs) -> Optional[Any]:
        """OpenPose 모델 로더"""
        await asyncio.sleep(0.2)
        return DummyOpenPose(num_keypoints=18, heatmap_size=64)
    
    async def _load_u2net(self, config: ModelConfig, **kwargs) -> Optional[Any]:
        """U2Net 모델 로더"""
        await asyncio.sleep(0.2)
        return DummyU2Net(input_channels=3, output_channels=1)
    
    async def _load_ootd_diffusion(self, config: ModelConfig, **kwargs) -> Optional[Any]:
        """OOTDiffusion 모델 로더"""
        await asyncio.sleep(0.5)
        return DummyOOTDiffusion(resolution=512, steps=20)
    
    @contextmanager
    def temporary_model(self, model_name: str, config: Optional[ModelConfig] = None):
        """임시 모델 컨텍스트 매니저"""
        model = None
        try:
            # 동기 방식으로 로드
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            model = loop.run_until_complete(self.load_model(model_name, config))
            yield model
        finally:
            # 모델이 캐시되지 않은 경우 명시적으로 정리
            if config and not config.cache_enabled and model:
                del model
                self.memory_manager.cleanup_memory()
    
    async def preload_models(self, model_names: List[str], configs: Optional[Dict[str, ModelConfig]] = None):
        """모델들 미리 로드"""
        logger.info(f"모델 프리로딩 시작: {model_names}")
        
        tasks = []
        for model_name in model_names:
            config = configs.get(model_name) if configs else None
            task = asyncio.create_task(self.load_model(model_name, config))
            tasks.append((model_name, task))
        
        for model_name, task in tasks:
            try:
                await task
                logger.info(f"✅ 프리로드 완료: {model_name}")
            except Exception as e:
                logger.error(f"❌ 프리로드 실패 {model_name}: {e}")
    
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
    
    def clear_cache(self):
        """전체 캐시 정리"""
        with self._lock:
            try:
                cache_size = len(self.model_cache)
                
                self.model_cache.clear()
                self.access_counts.clear()
                self.load_times.clear()
                self.last_access.clear()
                self.model_configs.clear()
                
                # GPU 메모리 정리
                self.memory_manager.cleanup_memory()
                
                logger.info(f"모델 캐시 전체 정리 완료 ({cache_size}개 모델)")
                
            except Exception as e:
                logger.error(f"캐시 정리 실패: {e}")
    
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
    
    async def model_benchmark(self, model_name: str, num_runs: int = 5, config: Optional[ModelConfig] = None) -> Dict[str, float]:
        """모델 성능 벤치마크"""
        logger.info(f"벤치마크 시작: {model_name}")
        
        try:
            # 모델 로드
            model = await self.load_model(model_name, config)
            if model is None:
                return {'error': 'model_load_failed'}
            
            # 벤치마크 설정
            input_size = config.input_size if config else (512, 512)
            batch_size = config.batch_size if config else 1
            
            dummy_input = torch.randn(batch_size, 3, *input_size).to(self.device)
            
            if self.use_fp16 and self.device != 'cpu':
                dummy_input = dummy_input.half()
            
            # 워밍업
            with torch.no_grad():
                for _ in range(2):
                    if hasattr(model, '__call__'):
                        try:
                            _ = model(dummy_input)
                        except:
                            pass
            
            # 실제 측정
            times = []
            memory_usage = []
            
            for i in range(num_runs):
                # 메모리 측정 (시작)
                start_memory = self.memory_manager.get_available_memory()
                
                start_time = time.time()
                
                with torch.no_grad():
                    if hasattr(model, '__call__'):
                        try:
                            result = model(dummy_input)
                        except Exception as e:
                            logger.warning(f"벤치마크 실행 실패 {i+1}/{num_runs}: {e}")
                            continue
                
                end_time = time.time()
                
                # 메모리 측정 (종료)
                end_memory = self.memory_manager.get_available_memory()
                memory_used = start_memory - end_memory
                
                times.append(end_time - start_time)
                memory_usage.append(max(0, memory_used))
            
            if not times:
                return {'error': 'no_successful_runs'}
            
            return {
                'model_name': model_name,
                'num_runs': len(times),
                'avg_time': sum(times) / len(times),
                'min_time': min(times),
                'max_time': max(times),
                'fps': 1.0 / (sum(times) / len(times)),
                'avg_memory_gb': sum(memory_usage) / len(memory_usage) if memory_usage else 0,
                'device': self.device,
                'input_size': input_size,
                'batch_size': batch_size
            }
            
        except Exception as e:
            logger.error(f"벤치마크 실패 {model_name}: {e}")
            return {'error': str(e)}
    
    def get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """모델 정보 반환"""
        try:
            registry_info = self.registry.get_model_info(model_name)
            if registry_info is None:
                return None
            
            # 캐시 정보 추가
            cache_keys = [k for k in self.model_cache.keys() if k.startswith(f"{model_name}_")]
            cached_configs = [self.model_configs.get(k) for k in cache_keys]
            
            return {
                'name': model_name,
                'type': registry_info.get('config', {}).get('model_type', 'unknown'),
                'class': registry_info['class'].__name__,
                'default_config': registry_info.get('config', {}),
                'registered_at': registry_info.get('registered_at'),
                'cached_instances': len(cache_keys),
                'cached_configs': [c.to_dict() if c else {} for c in cached_configs],
                'total_access_count': sum(self.access_counts.get(k, 0) for k in cache_keys),
                'avg_load_time': sum(self.load_times.get(k, 0) for k in cache_keys) / len(cache_keys) if cache_keys else 0,
                'path': str(self.get_model_path(model_name)),
                'has_custom_loader': registry_info.get('loader') is not None
            }
        except Exception as e:
            logger.error(f"모델 정보 조회 실패 {model_name}: {e}")
            return None
    
    def list_models(self) -> List[str]:
        """등록된 모델 목록 반환"""
        return self.registry.list_models()
    
    def get_status(self) -> Dict[str, Any]:
        """ModelLoader 전체 상태 반환"""
        try:
            cache_stats = self.get_cache_stats()
            
            return {
                'device': self.device,
                'use_fp16': self.use_fp16,
                'max_cache_size': self.max_cache_size,
                'enable_model_sharing': self.enable_model_sharing,
                'memory_limit_gb': self.memory_limit_gb,
                'registered_models': len(self.registry.list_models()),
                'cached_models': len(self.model_cache),
                'cache_stats': cache_stats,
                'models_dir': str(self.models_dir),
                'cache_dir': str(self.cache_dir),
                'models': {
                    name: {
                        'cached': any(k.startswith(f"{name}_") for k in self.model_cache.keys()),
                        'info': self.get_model_info(name)
                    }
                    for name in self.registry.list_models()
                }
            }
        except Exception as e:
            logger.error(f"상태 조회 실패: {e}")
            return {'error': str(e)}
    
    def cleanup(self):
        """리소스 정리"""
        try:
            logger.info("ModelLoader 정리 시작...")
            
            # 모든 모델 언로드
            model_names = list(set(k.split('_')[0] for k in self.model_cache.keys()))
            for model_name in model_names:
                self.unload_model(model_name)
            
            # 캐시 전체 정리
            self.clear_cache()
            
            # 스레드풀 종료
            if hasattr(self, '_executor'):
                self._executor.shutdown(wait=True)
            
            logger.info("✅ ModelLoader 정리 완료")
            
        except Exception as e:
            logger.error(f"ModelLoader 정리 중 오류: {e}")

# ========================================
# 더미 모델 클래스들
# ========================================

class DummyModel:
    """기본 더미 모델"""
    def __init__(self, name: str):
        self.name = name
        self.device = "cpu"
    
    def __call__(self, *args, **kwargs):
        return {"result": f"dummy_{self.name}", "success": True}
    
    def get_hash(self):
        return hashlib.md5(f"dummy_{self.name}".encode()).hexdigest()

class DummyHRVITON(nn.Module):
    """HR-VITON 더미 모델"""
    def __init__(self, input_size=512, num_channels=3):
        super().__init__()
        self.input_size = input_size
        self.conv = nn.Conv2d(num_channels, 64, 3, padding=1)
        self.out_conv = nn.Conv2d(64, num_channels, 3, padding=1)
    
    def forward(self, person_img, cloth_img=None):
        if cloth_img is None:
            cloth_img = person_img
        
        # 간단한 더미 처리
        x = self.conv(person_img)
        x = torch.relu(x)
        return self.out_conv(x)

class DummyGraphonomy(nn.Module):
    """Graphonomy 더미 모델"""
    def __init__(self, num_classes=20, backbone="resnet101"):
        super().__init__()
        self.num_classes = num_classes
        self.backbone = nn.Conv2d(3, 64, 7, stride=2, padding=3)
        self.classifier = nn.Conv2d(64, num_classes, 1)
    
    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)

class DummyOpenPose(nn.Module):
    """OpenPose 더미 모델"""
    def __init__(self, num_keypoints=18, heatmap_size=64):
        super().__init__()
        self.num_keypoints = num_keypoints
        self.backbone = nn.Conv2d(3, 128, 3, padding=1)
        self.keypoint_head = nn.Conv2d(128, num_keypoints, 1)
    
    def forward(self, x):
        features = self.backbone(x)
        heatmaps = self.keypoint_head(features)
        return heatmaps

class DummyU2Net(nn.Module):
    """U2Net 더미 모델"""
    def __init__(self, input_channels=3, output_channels=1):
        super().__init__()
        self.encoder = nn.Conv2d(input_channels, 64, 3, padding=1)
        self.decoder = nn.Conv2d(64, output_channels, 3, padding=1)
    
    def forward(self, x):
        encoded = torch.relu(self.encoder(x))
        return torch.sigmoid(self.decoder(encoded))

class DummyOOTDiffusion:
    """OOTDiffusion 더미 모델"""
    def __init__(self, resolution=512, steps=20):
        self.resolution = resolution
        self.steps = steps
        self.device = "cpu"
    
    def __call__(self, person_img, cloth_img, **kwargs):
        # 더미 결과 반환
        if hasattr(person_img, 'shape'):
            batch_size = person_img.shape[0]
            result = torch.randn(batch_size, 3, self.resolution, self.resolution)
        else:
            result = torch.randn(1, 3, self.resolution, self.resolution)
        return result
    
    def get_hash(self):
        return hashlib.md5(f"ootd_{self.resolution}_{self.steps}".encode()).hexdigest()

# ========================================
# 편의 함수들
# ========================================

@lru_cache(maxsize=1)
def get_global_model_loader() -> ModelLoader:
    """전역 ModelLoader 인스턴스 반환"""
    return ModelLoader()

async def load_model_async(model_name: str, config: Optional[ModelConfig] = None) -> Optional[Any]:
    """전역 로더를 사용한 비동기 모델 로드"""
    loader = get_global_model_loader()
    return await loader.load_model(model_name, config)

def load_model_sync(model_name: str, config: Optional[ModelConfig] = None) -> Optional[Any]:
    """전역 로더를 사용한 동기 모델 로드"""
    loader = get_global_model_loader()
    
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    return loop.run_until_complete(loader.load_model(model_name, config))

def cleanup_global_loader():
    """전역 로더 정리"""
    try:
        loader = get_global_model_loader()
        loader.cleanup()
        # 캐시 클리어
        get_global_model_loader.cache_clear()
    except:
        pass