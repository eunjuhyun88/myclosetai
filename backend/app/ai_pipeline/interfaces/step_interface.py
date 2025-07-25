# backend/app/ai_pipeline/interface/step_interface.py
"""
🔥 StepModelInterface v2.0 - 완전 호환성 + 순환참조 해결
======================================================

✅ BaseStepMixin 완전 호환성 보장
✅ TYPE_CHECKING 패턴으로 순환참조 방지
✅ register_model_requirement 메서드 완전 구현
✅ list_available_models 크기순 정렬
✅ 향상된 에러 처리 및 안정성
✅ M3 Max 128GB 메모리 최적화

Author: MyCloset AI Team
Date: 2025-07-24
Version: 2.0 (Complete Compatibility)
"""

import logging
import threading
import asyncio
import time
import gc
import weakref
from typing import Dict, Any, Optional, List, Union, Callable, TYPE_CHECKING
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

# 🔥 TYPE_CHECKING으로 순환참조 방지
if TYPE_CHECKING:
    from ..utils.model_loader import ModelLoader

logger = logging.getLogger(__name__)

# ==============================================
# 🔥 데이터 구조 정의
# ==============================================

class ModelStatus(Enum):
    """모델 상태"""
    NOT_LOADED = "not_loaded"
    LOADING = "loading"
    LOADED = "loaded"
    ERROR = "error"
    VALIDATING = "validating"

class CachePolicy(Enum):
    """캐시 정책"""
    MEMORY_FIRST = "memory_first"
    DISK_FIRST = "disk_first"
    NO_CACHE = "no_cache"
    HYBRID = "hybrid"

@dataclass
class ModelRequirement:
    """모델 요구사항"""
    model_name: str
    model_type: str = "BaseModel"
    device: str = "auto"
    precision: str = "fp16"
    input_size: tuple = (512, 512)
    num_classes: Optional[int] = None
    priority: int = 5
    min_memory_mb: float = 100.0
    max_memory_mb: float = 8192.0
    cache_policy: CachePolicy = CachePolicy.MEMORY_FIRST
    metadata: Dict[str, Any] = field(default_factory=dict)
    registered_at: float = field(default_factory=time.time)

@dataclass
class ModelCacheEntry:
    """모델 캐시 엔트리"""
    model: Any
    status: ModelStatus
    load_time: float
    last_access: float
    access_count: int
    memory_mb: float
    device: str
    step_name: str
    requirement: Optional[ModelRequirement] = None
    validation_passed: bool = True
    error_count: int = 0
    last_error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class InterfaceStatistics:
    """인터페이스 통계"""
    models_registered: int = 0
    models_loaded: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    loading_failures: int = 0
    total_memory_mb: float = 0.0
    average_load_time: float = 0.0
    creation_time: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)

# ==============================================
# 🔥 메모리 관리자
# ==============================================

class ModelMemoryManager:
    """모델 메모리 관리자"""
    
    def __init__(self, max_memory_mb: float = 4096.0):
        self.max_memory_mb = max_memory_mb
        self.current_memory_mb = 0.0
        self.logger = logging.getLogger("ModelMemoryManager")
        self._lock = threading.RLock()
    
    def can_load_model(self, required_memory_mb: float) -> bool:
        """모델 로딩 가능 여부 확인"""
        with self._lock:
            return (self.current_memory_mb + required_memory_mb) <= self.max_memory_mb
    
    def reserve_memory(self, memory_mb: float) -> bool:
        """메모리 예약"""
        with self._lock:
            if self.can_load_model(memory_mb):
                self.current_memory_mb += memory_mb
                return True
            return False
    
    def release_memory(self, memory_mb: float):
        """메모리 해제"""
        with self._lock:
            self.current_memory_mb = max(0.0, self.current_memory_mb - memory_mb)
    
    def force_cleanup(self) -> float:
        """강제 메모리 정리"""
        try:
            released_memory = self.current_memory_mb
            
            # Python GC 실행
            gc.collect()
            
            # PyTorch 메모리 정리 (안전하게)
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                elif hasattr(torch, 'mps') and hasattr(torch.mps, 'empty_cache'):
                    torch.mps.empty_cache()
            except:
                pass
            
            self.current_memory_mb = 0.0
            self.logger.info(f"🧹 강제 메모리 정리: {released_memory:.1f}MB 해제")
            return released_memory
            
        except Exception as e:
            self.logger.error(f"❌ 강제 메모리 정리 실패: {e}")
            return 0.0
    
    def get_memory_info(self) -> Dict[str, float]:
        """메모리 정보 조회"""
        with self._lock:
            return {
                "current_mb": self.current_memory_mb,
                "max_mb": self.max_memory_mb,
                "available_mb": self.max_memory_mb - self.current_memory_mb,
                "usage_percent": (self.current_memory_mb / self.max_memory_mb) * 100
            }

# ==============================================
# 🔥 StepModelInterface v2.0 - 완전 호환성
# ==============================================

class StepModelInterface:
    """
    🔗 Step용 ModelLoader 인터페이스 v2.0 - 완전 호환성
    
    ✅ BaseStepMixin 완전 호환성 보장
    ✅ register_model_requirement 완전 구현
    ✅ list_available_models 크기순 정렬
    ✅ 향상된 캐싱 및 메모리 관리
    ✅ 프로덕션 레벨 안정성
    """
    
    def __init__(self, step_name: str, model_loader: Optional['ModelLoader'] = None):
        self.step_name = step_name
        self.model_loader = model_loader
        self.logger = logging.getLogger(f"StepInterface.{step_name}")
        
        # 모델 관리
        self._model_cache: Dict[str, ModelCacheEntry] = {}
        self._model_requirements: Dict[str, ModelRequirement] = {}
        self._model_status: Dict[str, ModelStatus] = {}
        
        # 메모리 관리
        max_memory = 8192.0 if self._is_m3_max() else 4096.0
        self.memory_manager = ModelMemoryManager(max_memory)
        
        # 동기화
        self._lock = threading.RLock()
        self._async_lock = asyncio.Lock()
        
        # 통계 및 상태
        self.statistics = InterfaceStatistics()
        
        # 설정
        self.auto_cleanup = True
        self.cache_policy = CachePolicy.MEMORY_FIRST
        self.max_cache_entries = 20
        
        # 약한 참조로 모델 추적 (메모리 누수 방지)
        self._weak_model_refs: Dict[str, weakref.ref] = {}
        
        self.logger.info(f"🔗 {step_name} StepInterface v2.0 초기화 완료")
    
    def _is_m3_max(self) -> bool:
        """M3 Max 감지"""
        try:
            import platform
            import subprocess
            if platform.system() == 'Darwin':
                result = subprocess.run(
                    ['sysctl', '-n', 'machdep.cpu.brand_string'],
                    capture_output=True, text=True, timeout=5
                )
                return 'M3' in result.stdout
        except:
            pass
        return False
    
    # ==============================================
    # 🔥 핵심 메서드: register_model_requirement
    # ==============================================
    
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
            **kwargs: 추가 설정 (device, precision, input_size 등)
            
        Returns:
            bool: 등록 성공 여부
        """
        try:
            with self._lock:
                self.logger.info(f"📝 모델 요구사항 등록 시작: {model_name} ({model_type})")
                
                # ModelRequirement 생성
                requirement = ModelRequirement(
                    model_name=model_name,
                    model_type=model_type,
                    device=kwargs.get("device", "auto"),
                    precision=kwargs.get("precision", "fp16"),
                    input_size=kwargs.get("input_size", (512, 512)),
                    num_classes=kwargs.get("num_classes"),
                    priority=kwargs.get("priority", 5),
                    min_memory_mb=kwargs.get("min_memory_mb", 100.0),
                    max_memory_mb=kwargs.get("max_memory_mb", 8192.0),
                    cache_policy=kwargs.get("cache_policy", CachePolicy.MEMORY_FIRST),
                    metadata={
                        "step_name": self.step_name,
                        "registered_by": "register_model_requirement",
                        **kwargs.get("metadata", {})
                    }
                )
                
                # 요구사항 저장
                self._model_requirements[model_name] = requirement
                self._model_status[model_name] = ModelStatus.NOT_LOADED
                
                # 통계 업데이트
                self.statistics.models_registered += 1
                self.statistics.last_activity = time.time()
                
                # ModelLoader에 전달 (가능한 경우)
                if self.model_loader and hasattr(self.model_loader, 'register_model_requirement'):
                    try:
                        loader_success = self.model_loader.register_model_requirement(
                            model_name=model_name,
                            model_type=model_type,
                            step_name=self.step_name,
                            **kwargs
                        )
                        if loader_success:
                            self.logger.debug(f"✅ ModelLoader에 요구사항 전달 성공: {model_name}")
                        else:
                            self.logger.warning(f"⚠️ ModelLoader 요구사항 전달 실패: {model_name}")
                    except Exception as e:
                        self.logger.warning(f"⚠️ ModelLoader 요구사항 전달 중 오류: {e}")
                
                # ModelLoader에 설정 등록 시도 (register_model_config)
                if self.model_loader and hasattr(self.model_loader, 'register_model_config'):
                    try:
                        config = {
                            "model_type": model_type,
                            "model_class": model_type,
                            "device": requirement.device,
                            "precision": requirement.precision,
                            "input_size": requirement.input_size,
                            "num_classes": requirement.num_classes,
                            "metadata": requirement.metadata
                        }
                        self.model_loader.register_model_config(model_name, config)
                        self.logger.debug(f"✅ ModelLoader 설정 등록 성공: {model_name}")
                    except Exception as e:
                        self.logger.debug(f"ModelLoader 설정 등록 실패: {e}")
                
                self.logger.info(f"✅ 모델 요구사항 등록 완료: {model_name} ({model_type})")
                return True
                
        except Exception as e:
            self.statistics.loading_failures += 1
            self.logger.error(f"❌ 모델 요구사항 등록 실패: {model_name} - {e}")
            return False
    
    async def register_model_requirement_async(
        self, 
        model_name: str, 
        model_type: str = "BaseModel",
        **kwargs
    ) -> bool:
        """비동기 모델 요구사항 등록"""
        async with self._async_lock:
            # 동기 메서드를 executor에서 실행
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None,
                self.register_model_requirement,
                model_name,
                model_type,
                **kwargs
            )
    
    # ==============================================
    # 🔥 핵심 메서드: list_available_models
    # ==============================================
    
    def list_available_models(
        self, 
        step_class: Optional[str] = None,
        model_type: Optional[str] = None,
        include_unloaded: bool = True,
        sort_by: str = "size"  # size, name, priority, load_time
    ) -> List[Dict[str, Any]]:
        """
        🔥 사용 가능한 모델 목록 반환 - BaseStepMixin 완전 호환
        
        Args:
            step_class: Step 클래스 필터
            model_type: 모델 타입 필터
            include_unloaded: 로드되지 않은 모델 포함 여부
            sort_by: 정렬 기준
            
        Returns:
            List[Dict[str, Any]]: 모델 목록 (크기순 정렬)
        """
        try:
            models = []
            
            with self._lock:
                # 1. 등록된 요구사항에서 모델 목록 생성
                for model_name, requirement in self._model_requirements.items():
                    # 필터링
                    if step_class and step_class != self.step_name:
                        continue
                    if model_type and requirement.model_type != model_type:
                        continue
                    
                    # 캐시에서 정보 가져오기
                    cache_entry = self._model_cache.get(model_name)
                    is_loaded = cache_entry is not None and cache_entry.status == ModelStatus.LOADED
                    
                    # 로드되지 않은 모델 제외 (설정에 따라)
                    if not include_unloaded and not is_loaded:
                        continue
                    
                    memory_mb = cache_entry.memory_mb if cache_entry else requirement.min_memory_mb
                    device = cache_entry.device if cache_entry else requirement.device
                    
                    model_info = {
                        "name": model_name,
                        "path": f"step_models/{self.step_name}/{model_name}",
                        "size_mb": memory_mb,
                        "model_type": requirement.model_type,
                        "step_class": self.step_name,
                        "loaded": is_loaded,
                        "device": device,
                        "status": self._model_status.get(model_name, ModelStatus.NOT_LOADED).value,
                        "priority": requirement.priority,
                        "metadata": {
                            "step_name": self.step_name,
                            "input_size": requirement.input_size,
                            "num_classes": requirement.num_classes,
                            "precision": requirement.precision,
                            "cache_policy": requirement.cache_policy.value,
                            "access_count": cache_entry.access_count if cache_entry else 0,
                            "last_access": cache_entry.last_access if cache_entry else 0,
                            "load_time": cache_entry.load_time if cache_entry else 0,
                            "error_count": cache_entry.error_count if cache_entry else 0,
                            "validation_passed": cache_entry.validation_passed if cache_entry else True,
                            **requirement.metadata
                        }
                    }
                    models.append(model_info)
                
                # 2. ModelLoader에서 추가 모델 가져오기 (가능한 경우)
                if self.model_loader and hasattr(self.model_loader, 'list_available_models'):
                    try:
                        additional_models = self.model_loader.list_available_models(
                            step_class=self.step_name,
                            model_type=model_type
                        )
                        
                        # 중복 제거하며 추가
                        existing_names = {m["name"] for m in models}
                        for model in additional_models:
                            if model["name"] not in existing_names:
                                # ModelLoader의 모델 정보를 StepInterface 형식으로 변환
                                model_info = {
                                    "name": model["name"],
                                    "path": model.get("path", f"loader_models/{model['name']}"),
                                    "size_mb": model.get("size_mb", 0.0),
                                    "model_type": model.get("model_type", "unknown"),
                                    "step_class": model.get("step_class", self.step_name),
                                    "loaded": model.get("loaded", False),
                                    "device": model.get("device", "auto"),
                                    "status": "loaded" if model.get("loaded", False) else "not_loaded",
                                    "priority": 5,  # 기본 우선순위
                                    "metadata": {
                                        "step_name": self.step_name,
                                        "source": "model_loader",
                                        "access_count": 0,
                                        "last_access": 0,
                                        "load_time": 0,
                                        "error_count": 0,
                                        "validation_passed": True,
                                        **model.get("metadata", {})
                                    }
                                }
                                models.append(model_info)
                                
                    except Exception as e:
                        self.logger.warning(f"⚠️ ModelLoader 모델 목록 조회 실패: {e}")
                
                # 3. 정렬 수행
                if sort_by == "size":
                    models.sort(key=lambda x: x["size_mb"], reverse=True)  # 큰 것부터
                elif sort_by == "name":
                    models.sort(key=lambda x: x["name"])
                elif sort_by == "priority":
                    models.sort(key=lambda x: x["priority"])  # 작은 값이 높은 우선순위
                elif sort_by == "load_time":
                    models.sort(key=lambda x: x["metadata"].get("load_time", 0), reverse=True)
                else:
                    # 기본값: 크기순 정렬
                    models.sort(key=lambda x: x["size_mb"], reverse=True)
                
                self.logger.debug(f"📋 모델 목록 반환: {len(models)}개 (step={step_class}, type={model_type}, sort={sort_by})")
                return models
            
        except Exception as e:
            self.logger.error(f"❌ 모델 목록 조회 실패: {e}")
            return []
    
    # ==============================================
    # 🔥 모델 로딩 메서드들
    # ==============================================
    
    async def get_model_async(self, model_name: str, **kwargs) -> Optional[Any]:
        """비동기 모델 로드 - BaseStepMixin 호환"""
        async with self._async_lock:
            try:
                self.statistics.last_activity = time.time()
                
                # 캐시 확인
                if model_name in self._model_cache:
                    cache_entry = self._model_cache[model_name]
                    if cache_entry.status == ModelStatus.LOADED and cache_entry.model is not None:
                        cache_entry.last_access = time.time()
                        cache_entry.access_count += 1
                        self.statistics.cache_hits += 1
                        self.logger.debug(f"♻️ 캐시된 모델 반환: {model_name}")
                        return cache_entry.model
                    elif cache_entry.status == ModelStatus.ERROR:
                        self.logger.warning(f"⚠️ 이전에 로딩 실패한 모델: {model_name}")
                        return None
                
                # 로딩 상태 설정
                self._model_status[model_name] = ModelStatus.LOADING
                
                # 메모리 요구사항 확인
                requirement = self._model_requirements.get(model_name)
                if requirement:
                    if not self.memory_manager.can_load_model(requirement.max_memory_mb):
                        if self.auto_cleanup:
                            self.logger.info(f"🧹 메모리 부족으로 자동 정리 실행: {model_name}")
                            self._cleanup_least_used_models()
                            
                        if not self.memory_manager.can_load_model(requirement.max_memory_mb):
                            self.logger.error(f"❌ 메모리 부족으로 모델 로딩 불가: {model_name}")
                            self._model_status[model_name] = ModelStatus.ERROR
                            return None
                
                # ModelLoader를 통한 안전한 체크포인트 로드
                model = await self._safe_load_model(model_name, **kwargs)
                
                if model is not None:
                    # 메모리 사용량 추정
                    memory_usage = self._estimate_model_memory(model)
                    
                    # 메모리 예약
                    self.memory_manager.reserve_memory(memory_usage)
                    
                    # 캐시 엔트리 생성
                    cache_entry = ModelCacheEntry(
                        model=model,
                        status=ModelStatus.LOADED,
                        load_time=time.time(),
                        last_access=time.time(),
                        access_count=1,
                        memory_mb=memory_usage,
                        device=getattr(model, 'device', 'cpu') if hasattr(model, 'device') else 'cpu',
                        step_name=self.step_name,
                        requirement=requirement,
                        validation_passed=True,
                        error_count=0,
                        metadata={
                            "loading_method": "async",
                            "kwargs": kwargs
                        }
                    )
                    
                    # 캐시 관리
                    self._manage_cache_size()
                    
                    # 캐시에 저장
                    with self._lock:
                        self._model_cache[model_name] = cache_entry
                        self._model_status[model_name] = ModelStatus.LOADED
                        
                        # 약한 참조 저장 (메모리 누수 방지)
                        self._weak_model_refs[model_name] = weakref.ref(model)
                    
                    # 통계 업데이트
                    self.statistics.models_loaded += 1
                    self.statistics.total_memory_mb += memory_usage
                    self._update_average_load_time(cache_entry.load_time)
                    
                    self.logger.info(f"✅ 모델 로드 성공: {model_name} ({memory_usage:.1f}MB)")
                    return model
                
                # 로딩 실패
                self._model_status[model_name] = ModelStatus.ERROR
                self.statistics.loading_failures += 1
                self.statistics.cache_misses += 1
                self.logger.warning(f"⚠️ 모델 로드 실패: {model_name}")
                return None
                
            except Exception as e:
                self._model_status[model_name] = ModelStatus.ERROR
                self.statistics.loading_failures += 1
                self.logger.error(f"❌ 비동기 모델 로드 실패: {model_name} - {e}")
                return None
    
    def get_model_sync(self, model_name: str, **kwargs) -> Optional[Any]:
        """동기 모델 로드 - BaseStepMixin 호환"""
        try:
            self.statistics.last_activity = time.time()
            
            # 캐시 확인
            with self._lock:
                if model_name in self._model_cache:
                    cache_entry = self._model_cache[model_name]
                    if cache_entry.status == ModelStatus.LOADED and cache_entry.model is not None:
                        cache_entry.last_access = time.time()
                        cache_entry.access_count += 1
                        self.statistics.cache_hits += 1
                        self.logger.debug(f"♻️ 캐시된 모델 반환: {model_name}")
                        return cache_entry.model
                    elif cache_entry.status == ModelStatus.ERROR:
                        self.logger.warning(f"⚠️ 이전에 로딩 실패한 모델: {model_name}")
                        return None
            
            # 로딩 상태 설정
            self._model_status[model_name] = ModelStatus.LOADING
            
            # 메모리 요구사항 확인
            requirement = self._model_requirements.get(model_name)
            if requirement:
                if not self.memory_manager.can_load_model(requirement.max_memory_mb):
                    if self.auto_cleanup:
                        self.logger.info(f"🧹 메모리 부족으로 자동 정리 실행: {model_name}")
                        self._cleanup_least_used_models()
                        
                    if not self.memory_manager.can_load_model(requirement.max_memory_mb):
                        self.logger.error(f"❌ 메모리 부족으로 모델 로딩 불가: {model_name}")
                        self._model_status[model_name] = ModelStatus.ERROR
                        return None
            
            # ModelLoader를 통한 체크포인트 로드
            model = None
            if self.model_loader and hasattr(self.model_loader, 'load_model'):
                try:
                    model = self.model_loader.load_model(model_name, **kwargs)
                except Exception as e:
                    self.logger.error(f"❌ ModelLoader를 통한 로딩 실패: {e}")
            
            if model is not None:
                # 메모리 사용량 추정
                memory_usage = self._estimate_model_memory(model)
                
                # 메모리 예약
                self.memory_manager.reserve_memory(memory_usage)
                
                # 캐시 엔트리 생성
                cache_entry = ModelCacheEntry(
                    model=model,
                    status=ModelStatus.LOADED,
                    load_time=time.time(),
                    last_access=time.time(),
                    access_count=1,
                    memory_mb=memory_usage,
                    device=getattr(model, 'device', 'cpu') if hasattr(model, 'device') else 'cpu',
                    step_name=self.step_name,
                    requirement=requirement,
                    validation_passed=True,
                    error_count=0,
                    metadata={
                        "loading_method": "sync",
                        "kwargs": kwargs
                    }
                )
                
                # 캐시 관리
                self._manage_cache_size()
                
                # 캐시에 저장
                with self._lock:
                    self._model_cache[model_name] = cache_entry
                    self._model_status[model_name] = ModelStatus.LOADED
                    
                    # 약한 참조 저장
                    self._weak_model_refs[model_name] = weakref.ref(model)
                
                # 통계 업데이트
                self.statistics.models_loaded += 1
                self.statistics.total_memory_mb += memory_usage
                self._update_average_load_time(cache_entry.load_time)
                
                self.logger.info(f"✅ 동기 모델 로드 성공: {model_name} ({memory_usage:.1f}MB)")
                return model
            
            # 로딩 실패
            self._model_status[model_name] = ModelStatus.ERROR
            self.statistics.loading_failures += 1
            self.statistics.cache_misses += 1
            self.logger.warning(f"⚠️ 동기 모델 로드 실패: {model_name}")
            return None
            
        except Exception as e:
            self._model_status[model_name] = ModelStatus.ERROR
            self.statistics.loading_failures += 1
            self.logger.error(f"❌ 동기 모델 로드 실패: {model_name} - {e}")
            return None
    
    async def _safe_load_model(self, model_name: str, **kwargs) -> Optional[Any]:
        """안전한 모델 로딩"""
        try:
            if self.model_loader:
                if hasattr(self.model_loader, 'load_model_async'):
                    return await self.model_loader.load_model_async(model_name, **kwargs)
                elif hasattr(self.model_loader, 'load_model'):
                    # 동기 메서드를 비동기로 실행
                    loop = asyncio.get_event_loop()
                    return await loop.run_in_executor(
                        None, 
                        lambda: self.model_loader.load_model(model_name, **kwargs)
                    )
            
            self.logger.error(f"❌ ModelLoader가 없거나 로딩 메서드 없음")
            return None
            
        except Exception as e:
            self.logger.error(f"❌ 안전한 모델 로딩 실패: {e}")
            return None
    
    # BaseStepMixin 호환성을 위한 별칭
    async def get_model(self, model_name: str, **kwargs) -> Optional[Any]:
        """모델 로드 (비동기 우선) - BaseStepMixin 호환"""
        return await self.get_model_async(model_name, **kwargs)
    
    def load_model(self, model_name: str, **kwargs) -> Optional[Any]:
        """모델 로드 (동기) - BaseStepMixin 호환"""
        return self.get_model_sync(model_name, **kwargs)
    
    # ==============================================
    # 🔥 모델 상태 및 관리 메서드들
    # ==============================================
    
    def get_model_status(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        """모델 상태 조회 - BaseStepMixin 호환"""
        try:
            if model_name:
                # 특정 모델 상태
                with self._lock:
                    if model_name in self._model_cache:
                        cache_entry = self._model_cache[model_name]
                        return {
                            "name": model_name,
                            "status": cache_entry.status.value,
                            "loaded": cache_entry.status == ModelStatus.LOADED,
                            "device": cache_entry.device,
                            "memory_mb": cache_entry.memory_mb,
                            "load_time": cache_entry.load_time,
                            "last_access": cache_entry.last_access,
                            "access_count": cache_entry.access_count,
                            "validation_passed": cache_entry.validation_passed,
                            "error_count": cache_entry.error_count,
                            "last_error": cache_entry.last_error,
                            "metadata": cache_entry.metadata
                        }
                    else:
                        status = self._model_status.get(model_name, ModelStatus.NOT_LOADED)
                        requirement = self._model_requirements.get(model_name)
                        return {
                            "name": model_name,
                            "status": status.value,
                            "loaded": False,
                            "device": requirement.device if requirement else "unknown",
                            "memory_mb": 0,
                            "load_time": 0,
                            "last_access": 0,
                            "access_count": 0,
                            "validation_passed": True,
                            "error_count": 0,
                            "last_error": None,
                            "metadata": {}
                        }
            else:
                # 전체 상태
                with self._lock:
                    models_status = {}
                    all_model_names = set(self._model_requirements.keys()) | set(self._model_cache.keys())
                    
                    for name in all_model_names:
                        models_status[name] = self.get_model_status(name)
                    
                    memory_info = self.memory_manager.get_memory_info()
                    
                    return {
                        "step_name": self.step_name,
                        "models": models_status,
                        "total_models": len(self._model_requirements),
                        "loaded_models": len([
                            entry for entry in self._model_cache.values() 
                            if entry.status == ModelStatus.LOADED
                        ]),
                        "cache_entries": len(self._model_cache),
                        "memory_info": memory_info,
                        "statistics": {
                            "models_registered": self.statistics.models_registered,
                            "models_loaded": self.statistics.models_loaded,
                            "cache_hits": self.statistics.cache_hits,
                            "cache_misses": self.statistics.cache_misses,
                            "loading_failures": self.statistics.loading_failures,
                            "average_load_time": self.statistics.average_load_time,
                            "total_memory_mb": self.statistics.total_memory_mb
                        },
                        "creation_time": self.statistics.creation_time,
                        "last_activity": self.statistics.last_activity,
                        "version": "2.0"
                    }
        except Exception as e:
            self.logger.error(f"❌ 모델 상태 조회 실패: {e}")
            return {"error": str(e)}
    
    def unload_model(self, model_name: str) -> bool:
        """모델 언로드"""
        try:
            with self._lock:
                if model_name in self._model_cache:
                    cache_entry = self._model_cache[model_name]
                    
                    # 메모리 해제
                    self.memory_manager.release_memory(cache_entry.memory_mb)
                    
                    # 통계 업데이트
                    self.statistics.total_memory_mb -= cache_entry.memory_mb
                    
                    # 캐시에서 제거
                    del self._model_cache[model_name]
                    
                    # 약한 참조 제거
                    if model_name in self._weak_model_refs:
                        del self._weak_model_refs[model_name]
                    
                    # 상태 업데이트
                    self._model_status[model_name] = ModelStatus.NOT_LOADED
                    
                    # 가비지 컬렉션
                    gc.collect()
                    
                    self.logger.info(f"✅ 모델 언로드: {model_name} ({cache_entry.memory_mb:.1f}MB 해제)")
                    return True
                else:
                    self.logger.warning(f"⚠️ 언로드할 모델이 캐시에 없음: {model_name}")
                    return False
        except Exception as e:
            self.logger.error(f"❌ 모델 언로드 실패: {model_name} - {e}")
            return False
    
    def clear_cache(self, force: bool = False) -> bool:
        """모델 캐시 초기화"""
        try:
            with self._lock:
                if not force and len(self._model_cache) > 0:
                    # 강제가 아닌 경우 확인
                    self.logger.warning(f"⚠️ {len(self._model_cache)}개 모델이 캐시에 있음")
                
                # 모든 모델 언로드
                unloaded_count = 0
                for model_name in list(self._model_cache.keys()):
                    if self.unload_model(model_name):
                        unloaded_count += 1
                
                # 강제 메모리 정리
                released_memory = self.memory_manager.force_cleanup()
                
                # 상태 초기화
                for model_name in self._model_status:
                    self._model_status[model_name] = ModelStatus.NOT_LOADED
                
                # 통계 리셋 (일부)
                self.statistics.total_memory_mb = 0.0
                self.statistics.last_activity = time.time()
                
                self.logger.info(f"🧹 캐시 초기화 완료: {unloaded_count}개 모델 언로드, {released_memory:.1f}MB 해제")
                return True
                
        except Exception as e:
            self.logger.error(f"❌ 캐시 초기화 실패: {e}")
            return False
    
    # ==============================================
    # 🔥 내부 유틸리티 메서드들
    # ==============================================
    
    def _estimate_model_memory(self, model) -> float:
        """모델 메모리 사용량 추정 (MB)"""
        try:
            if model is None:
                return 0.0
                
            # PyTorch 모델
            if hasattr(model, 'parameters'):
                total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                return total_params * 4 / (1024 * 1024)  # float32 기준
            
            # state_dict
            elif isinstance(model, dict):
                total_size = 0
                for tensor in model.values():
                    if hasattr(tensor, 'numel'):
                        total_size += tensor.numel() * 4  # float32 기준
                return total_size / (1024 * 1024)
            
            # 기본 추정치
            else:
                return 100.0
                
        except Exception:
            return 100.0  # 기본값
    
    def _update_average_load_time(self, load_time: float):
        """평균 로딩 시간 업데이트"""
        try:
            current_avg = self.statistics.average_load_time
            loaded_count = self.statistics.models_loaded
            
            if loaded_count <= 1:
                self.statistics.average_load_time = load_time
            else:
                # 이동 평균 계산
                self.statistics.average_load_time = (
                    (current_avg * (loaded_count - 1) + load_time) / loaded_count
                )
        except Exception:
            pass
    
    def _manage_cache_size(self):
        """캐시 크기 관리"""
        try:
            with self._lock:
                if len(self._model_cache) >= self.max_cache_entries:
                    self.logger.info(f"📦 캐시 크기 초과 ({len(self._model_cache)}/{self.max_cache_entries}), 정리 실행")
                    self._cleanup_least_used_models(1)
        except Exception as e:
            self.logger.error(f"❌ 캐시 크기 관리 실패: {e}")
    
    def _cleanup_least_used_models(self, count: int = 1):
        """가장 적게 사용된 모델들 정리"""
        try:
            with self._lock:
                if len(self._model_cache) <= count:
                    return
                
                # 접근 시간과 횟수를 기준으로 정렬
                sorted_models = sorted(
                    self._model_cache.items(),
                    key=lambda x: (x[1].last_access, x[1].access_count)
                )
                
                # 가장 적게 사용된 모델들 언로드
                for i in range(min(count, len(sorted_models))):
                    model_name, cache_entry = sorted_models[i]
                    self.logger.info(f"🧹 사용 빈도 낮은 모델 정리: {model_name} (접근: {cache_entry.access_count}회)")
                    self.unload_model(model_name)
                    
        except Exception as e:
            self.logger.error(f"❌ 최소 사용 모델 정리 실패: {e}")
    
    def _cleanup_dead_references(self):
        """죽은 약한 참조 정리"""
        try:
            with self._lock:
                dead_refs = []
                for model_name, weak_ref in self._weak_model_refs.items():
                    if weak_ref() is None:
                        dead_refs.append(model_name)
                
                for model_name in dead_refs:
                    del self._weak_model_refs[model_name]
                    if model_name in self._model_cache:
                        cache_entry = self._model_cache[model_name]
                        self.memory_manager.release_memory(cache_entry.memory_mb)
                        del self._model_cache[model_name]
                        self._model_status[model_name] = ModelStatus.NOT_LOADED
                
                if dead_refs:
                    self.logger.info(f"🧹 죽은 참조 정리: {len(dead_refs)}개")
                    
        except Exception as e:
            self.logger.error(f"❌ 죽은 참조 정리 실패: {e}")
    
    # ==============================================
    # 🔥 고급 기능들
    # ==============================================
    
    def get_requirements(self) -> Dict[str, ModelRequirement]:
        """등록된 요구사항 반환"""
        with self._lock:
            return self._model_requirements.copy()
    
    def get_cache_info(self) -> Dict[str, Any]:
        """캐시 정보 반환"""
        try:
            with self._lock:
                memory_info = self.memory_manager.get_memory_info()
                
                return {
                    "cache_size": len(self._model_cache),
                    "max_cache_size": self.max_cache_entries,
                    "weak_references": len(self._weak_model_refs),
                    "memory_info": memory_info,
                    "cache_policy": self.cache_policy.value,
                    "auto_cleanup": self.auto_cleanup,
                    "loaded_models": [
                        {
                            "name": name,
                            "memory_mb": entry.memory_mb,
                            "access_count": entry.access_count,
                            "last_access": entry.last_access
                        }
                        for name, entry in self._model_cache.items()
                        if entry.status == ModelStatus.LOADED
                    ]
                }
        except Exception as e:
            self.logger.error(f"❌ 캐시 정보 조회 실패: {e}")
            return {}
    
    def get_statistics(self) -> Dict[str, Any]:
        """상세 통계 정보 반환"""
        try:
            with self._lock:
                return {
                    "models_registered": self.statistics.models_registered,
                    "models_loaded": self.statistics.models_loaded,
                    "cache_hits": self.statistics.cache_hits,
                    "cache_misses": self.statistics.cache_misses,
                    "loading_failures": self.statistics.loading_failures,
                    "total_memory_mb": self.statistics.total_memory_mb,
                    "average_load_time": self.statistics.average_load_time,
                    "creation_time": self.statistics.creation_time,
                    "last_activity": self.statistics.last_activity,
                    "uptime_seconds": time.time() - self.statistics.creation_time,
                    "cache_hit_rate": (
                        self.statistics.cache_hits / 
                        max(1, self.statistics.cache_hits + self.statistics.cache_misses)
                    ) * 100,
                    "success_rate": (
                        self.statistics.models_loaded / 
                        max(1, self.statistics.models_loaded + self.statistics.loading_failures)
                    ) * 100,
                    "memory_efficiency": self.memory_manager.get_memory_info()
                }
        except Exception as e:
            self.logger.error(f"❌ 통계 조회 실패: {e}")
            return {}
    
    def optimize_cache(self):
        """캐시 최적화 실행"""
        try:
            self.logger.info("🔧 캐시 최적화 시작...")
            
            # 죽은 참조 정리
            self._cleanup_dead_references()
            
            # 메모리 사용량이 많은 경우 정리
            memory_info = self.memory_manager.get_memory_info()
            if memory_info["usage_percent"] > 80:
                cleanup_count = max(1, len(self._model_cache) // 4)
                self._cleanup_least_used_models(cleanup_count)
            
            # 가비지 컬렉션
            gc.collect()
            
            self.logger.info("✅ 캐시 최적화 완료")
            
        except Exception as e:
            self.logger.error(f"❌ 캐시 최적화 실패: {e}")
    
    # ==============================================
    # 🔥 정리 메서드
    # ==============================================
    
    def cleanup(self):
        """리소스 정리"""
        try:
            self.logger.info(f"🧹 {self.step_name} Interface 정리 시작...")
            
            # 모든 모델 언로드
            self.clear_cache(force=True)
            
            # 요구사항 정리
            self._model_requirements.clear()
            self._model_status.clear()
            
            # 약한 참조 정리
            self._weak_model_refs.clear()
            
            self.logger.info(f"✅ {self.step_name} Interface 정리 완료")
            
        except Exception as e:
            self.logger.error(f"❌ Interface 정리 실패: {e}")
    
    def __del__(self):
        """소멸자"""
        try:
            self.cleanup()
        except:
            pass

# ==============================================
# 🔥 팩토리 함수들
# ==============================================

def create_step_model_interface(
    step_name: str, 
    model_loader: Optional['ModelLoader'] = None,
    max_cache_entries: int = 20,
    auto_cleanup: bool = True,
    cache_policy: CachePolicy = CachePolicy.MEMORY_FIRST
) -> StepModelInterface:
    """Step Model Interface 생성"""
    try:
        interface = StepModelInterface(step_name, model_loader)
        
        # 설정 적용
        interface.max_cache_entries = max_cache_entries
        interface.auto_cleanup = auto_cleanup
        interface.cache_policy = cache_policy
        
        logger.info(f"✅ Step Interface 생성 완료: {step_name}")
        return interface
        
    except Exception as e:
        logger.error(f"❌ Step Interface 생성 실패: {step_name} - {e}")
        # 폴백 인터페이스
        return StepModelInterface(step_name, None)

def create_optimized_step_interface(
    step_name: str,
    model_loader: Optional['ModelLoader'] = None,
    memory_limit_mb: float = None
) -> StepModelInterface:
    """최적화된 Step Interface 생성 (M3 Max 대응)"""
    try:
        # M3 Max 감지
        is_m3_max = False
        try:
            import platform
            import subprocess
            if platform.system() == 'Darwin':
                result = subprocess.run(
                    ['sysctl', '-n', 'machdep.cpu.brand_string'],
                    capture_output=True, text=True, timeout=5
                )
                is_m3_max = 'M3' in result.stdout
        except:
            pass
        
        # M3 Max에 맞는 설정
        if is_m3_max:
            max_cache = 30
            memory_limit = memory_limit_mb or 16384.0  # 16GB
            auto_cleanup = True
            cache_policy = CachePolicy.HYBRID
        else:
            max_cache = 15
            memory_limit = memory_limit_mb or 4096.0   # 4GB
            auto_cleanup = True
            cache_policy = CachePolicy.MEMORY_FIRST
        
        interface = create_step_model_interface(
            step_name=step_name,
            model_loader=model_loader,
            max_cache_entries=max_cache,
            auto_cleanup=auto_cleanup,
            cache_policy=cache_policy
        )
        
        # 메모리 제한 설정
        interface.memory_manager.max_memory_mb = memory_limit
        
        logger.info(f"✅ 최적화된 Step Interface 생성: {step_name} (M3 Max: {is_m3_max})")
        return interface
        
    except Exception as e:
        logger.error(f"❌ 최적화된 Step Interface 생성 실패: {step_name} - {e}")
        return create_step_model_interface(step_name, model_loader)

# ==============================================
# 🔥 Export
# ==============================================

__all__ = [
    # 메인 클래스들
    'StepModelInterface',
    'ModelMemoryManager',
    
    # 데이터 구조들
    'ModelRequirement',
    'ModelCacheEntry',
    'InterfaceStatistics',
    'ModelStatus',
    'CachePolicy',
    
    # 팩토리 함수들
    'create_step_model_interface',
    'create_optimized_step_interface'
]

# 모듈 로드 완료
logger.info("=" * 80)
logger.info("🔗 StepModelInterface v2.0 - 완전 호환성 + 순환참조 해결")
logger.info("=" * 80)
logger.info("✅ BaseStepMixin 완전 호환성 보장")
logger.info("✅ register_model_requirement 완전 구현")
logger.info("✅ list_available_models 크기순 정렬")
logger.info("✅ 향상된 캐싱 및 메모리 관리")
logger.info("✅ 약한 참조 기반 메모리 누수 방지")
logger.info("✅ M3 Max 128GB 메모리 최적화")
logger.info("✅ 프로덕션 레벨 안정성")
logger.info("=" * 80)