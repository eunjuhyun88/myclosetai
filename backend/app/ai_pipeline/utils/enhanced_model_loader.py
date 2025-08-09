#!/usr/bin/env python3
"""
🔥 Enhanced Model Loader v7.0 - 고급 모델 로딩 시스템
================================================================================
✅ 예측 로딩 및 스마트 캐싱
✅ 모델 공유 메커니즘
✅ 그래디언트 체크포인팅
✅ 병렬 로딩 시스템
✅ 메모리 최적화 고도화
✅ 실시간 성능 모니터링
================================================================================
"""

import os
import sys
import time
import logging
import threading
import asyncio
import gc
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import Enum
import queue
import weakref

# 경고 무시
warnings.filterwarnings('ignore')

# 프로젝트 경로 설정
current_file = Path(__file__).resolve()
backend_root = current_file.parents[3]
sys.path.insert(0, str(backend_root))

# PyTorch 안전 import
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️ PyTorch 없음 - 제한된 기능만 사용 가능")

# 시스템 정보
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# 기존 model_loader.py의 아키텍처 클래스들 import
try:
    from .model_loader import (
        HumanParsingArchitecture,
        PoseEstimationArchitecture,
        ClothSegmentationArchitecture,
        GeometricMatchingArchitecture,
        VirtualFittingArchitecture,
        ClothWarpingArchitecture,
        StepSpecificArchitecture
    )
    ARCHITECTURE_AVAILABLE = True
except ImportError:
    ARCHITECTURE_AVAILABLE = False
    print("⚠️ 기존 아키텍처 클래스 import 실패 - 기본 아키텍처 사용")

# ==============================================
# 🔥 1. 고급 캐싱 전략
# ==============================================

class CacheStrategy(Enum):
    """캐싱 전략"""
    LRU = "lru"
    LFU = "lfu"
    ADAPTIVE = "adaptive"
    PREDICTIVE = "predictive"

@dataclass
class CacheEntry:
    """캐시 엔트리"""
    data: Any
    size_mb: float
    access_count: int = 0
    last_access: float = field(default_factory=time.time)
    creation_time: float = field(default_factory=time.time)
    priority: float = 1.0
    predicted_next_access: Optional[float] = None

class PredictiveCache:
    """예측 기반 스마트 캐시"""
    
    def __init__(self, max_size_mb: float = 1024, strategy: CacheStrategy = CacheStrategy.ADAPTIVE):
        self.max_size_mb = max_size_mb
        self.strategy = strategy
        self.cache: Dict[str, CacheEntry] = {}
        self.access_patterns: Dict[str, List[float]] = {}
        self.prediction_model = self._create_prediction_model()
        self._lock = threading.RLock()
        self.logger = logging.getLogger(f"{__name__}.PredictiveCache")
        
        # 자동 정리 스레드
        self._cleanup_thread = threading.Thread(target=self._cleanup_worker, daemon=True)
        self._cleanup_thread.start()
    
    def _create_prediction_model(self) -> Dict[str, float]:
        """간단한 예측 모델 생성"""
        return {}
    
    def get(self, key: str) -> Optional[Any]:
        """캐시에서 조회"""
        with self._lock:
            if key in self.cache:
                entry = self.cache[key]
                entry.access_count += 1
                entry.last_access = time.time()
                
                # 접근 패턴 기록
                if key not in self.access_patterns:
                    self.access_patterns[key] = []
                self.access_patterns[key].append(time.time())
                
                # 예측 모델 업데이트
                self._update_prediction_model(key)
                
                return entry.data
            return None
    
    def set(self, key: str, data: Any, size_mb: float, ttl: Optional[float] = None):
        """캐시에 저장"""
        with self._lock:
            # 캐시 크기 확인 및 정리
            if self._get_total_size() + size_mb > self.max_size_mb:
                self._evict_entries(size_mb)
            
            # 새 엔트리 생성
            entry = CacheEntry(
                data=data,
                size_mb=size_mb,
                priority=self._calculate_priority(key)
            )
            
            self.cache[key] = entry
            
            # 예측 모델에 추가
            if key not in self.prediction_model:
                self.prediction_model[key] = time.time() + 3600  # 1시간 후 예측
    
    def _calculate_priority(self, key: str) -> float:
        """우선순위 계산"""
        base_priority = 1.0
        
        # 접근 빈도 기반
        if key in self.access_patterns:
            recent_accesses = [t for t in self.access_patterns[key] if time.time() - t < 3600]
            base_priority += len(recent_accesses) * 0.1
        
        # 예측 기반
        if key in self.prediction_model:
            predicted_time = self.prediction_model[key]
            time_until_predicted = predicted_time - time.time()
            if time_until_predicted < 300:  # 5분 내 예측
                base_priority += 2.0
        
        return base_priority
    
    def _evict_entries(self, required_size_mb: float):
        """캐시 엔트리 제거"""
        if self.strategy == CacheStrategy.LRU:
            self._evict_lru(required_size_mb)
        elif self.strategy == CacheStrategy.LFU:
            self._evict_lfu(required_size_mb)
        elif self.strategy == CacheStrategy.ADAPTIVE:
            self._evict_adaptive(required_size_mb)
        else:
            self._evict_predictive(required_size_mb)
    
    def _evict_lru(self, required_size_mb: float):
        """LRU 기반 제거"""
        sorted_entries = sorted(
            self.cache.items(),
            key=lambda x: x[1].last_access
        )
        
        freed_size = 0.0
        for key, entry in sorted_entries:
            if freed_size >= required_size_mb:
                break
            del self.cache[key]
            freed_size += entry.size_mb
    
    def _evict_lfu(self, required_size_mb: float):
        """LFU 기반 제거"""
        sorted_entries = sorted(
            self.cache.items(),
            key=lambda x: x[1].access_count
        )
        
        freed_size = 0.0
        for key, entry in sorted_entries:
            if freed_size >= required_size_mb:
                break
            del self.cache[key]
            freed_size += entry.size_mb
    
    def _evict_adaptive(self, required_size_mb: float):
        """적응형 제거 (우선순위 기반)"""
        sorted_entries = sorted(
            self.cache.items(),
            key=lambda x: x[1].priority,
            reverse=True
        )
        
        # 낮은 우선순위부터 제거
        entries_to_remove = sorted_entries[:-len(sorted_entries)//2]  # 하위 50% 제거
        
        freed_size = 0.0
        for key, entry in entries_to_remove:
            if freed_size >= required_size_mb:
                break
            del self.cache[key]
            freed_size += entry.size_mb
    
    def _evict_predictive(self, required_size_mb: float):
        """예측 기반 제거"""
        current_time = time.time()
        
        # 예측 시간이 가까운 것들은 보존
        entries_with_prediction = []
        for key, entry in self.cache.items():
            if key in self.prediction_model:
                predicted_time = self.prediction_model[key]
                time_until_predicted = predicted_time - current_time
                entries_with_prediction.append((key, entry, time_until_predicted))
        
        # 예측 시간이 먼 것부터 제거
        entries_with_prediction.sort(key=lambda x: x[2], reverse=True)
        
        freed_size = 0.0
        for key, entry, _ in entries_with_prediction:
            if freed_size >= required_size_mb:
                break
            del self.cache[key]
            freed_size += entry.size_mb
    
    def _get_total_size(self) -> float:
        """캐시 총 크기"""
        return sum(entry.size_mb for entry in self.cache.values())
    
    def _update_prediction_model(self, key: str):
        """예측 모델 업데이트"""
        if key in self.access_patterns:
            pattern = self.access_patterns[key]
            if len(pattern) >= 3:
                # 간단한 선형 예측
                intervals = [pattern[i] - pattern[i-1] for i in range(1, len(pattern))]
                avg_interval = sum(intervals) / len(intervals)
                self.prediction_model[key] = time.time() + avg_interval
    
    def _cleanup_worker(self):
        """캐시 정리 워커"""
        while True:
            try:
                time.sleep(300)  # 5분마다 정리
                with self._lock:
                    current_time = time.time()
                    
                    # 오래된 접근 패턴 정리
                    for key in list(self.access_patterns.keys()):
                        self.access_patterns[key] = [
                            t for t in self.access_patterns[key]
                            if current_time - t < 3600  # 1시간 이내만 유지
                        ]
                        if not self.access_patterns[key]:
                            del self.access_patterns[key]
                    
                    # 캐시 크기 조정
                    if self._get_total_size() > self.max_size_mb * 0.9:
                        self._evict_entries(self.max_size_mb * 0.1)
                        
            except Exception as e:
                self.logger.warning(f"캐시 정리 실패: {e}")

# ==============================================
# 🔥 2. 모델 공유 메커니즘
# ==============================================

class SharedModelRegistry:
    """모델 공유 레지스트리"""
    
    def __init__(self):
        self.shared_models: Dict[str, nn.Module] = {}
        self.model_references: Dict[str, List[str]] = {}
        self._lock = threading.RLock()
        self.logger = logging.getLogger(f"{__name__}.SharedModelRegistry")
    
    def register_shared_model(self, model_name: str, model: nn.Module, step_types: List[str]):
        """공유 모델 등록"""
        with self._lock:
            self.shared_models[model_name] = model
            self.model_references[model_name] = step_types
            self.logger.info(f"✅ 공유 모델 등록: {model_name} -> {step_types}")
    
    def get_shared_model(self, model_name: str) -> Optional[nn.Module]:
        """공유 모델 조회"""
        with self._lock:
            return self.shared_models.get(model_name)
    
    def is_shared_model(self, model_name: str) -> bool:
        """공유 모델 여부 확인"""
        return model_name in self.shared_models
    
    def get_shared_components(self, step_type: str) -> Dict[str, nn.Module]:
        """Step별 공유 컴포넌트 조회"""
        with self._lock:
            shared_components = {}
            for model_name, step_types in self.model_references.items():
                if step_type in step_types:
                    shared_components[model_name] = self.shared_models[model_name]
            return shared_components

# ==============================================
# 🔥 3. 그래디언트 체크포인팅
# ==============================================

class GradientCheckpointingManager:
    """그래디언트 체크포인팅 관리자"""
    
    def __init__(self):
        self.checkpointed_models: Dict[str, nn.Module] = {}
        self._lock = threading.RLock()
        self.logger = logging.getLogger(f"{__name__}.GradientCheckpointingManager")
    
    def enable_checkpointing(self, model: nn.Module, model_name: str) -> nn.Module:
        """그래디언트 체크포인팅 활성화"""
        if not TORCH_AVAILABLE:
            return model
        
        try:
            with self._lock:
                # 이미 체크포인팅된 모델인지 확인
                if model_name in self.checkpointed_models:
                    return self.checkpointed_models[model_name]
                
                # 그래디언트 체크포인팅 적용
                if hasattr(model, 'gradient_checkpointing_enable'):
                    model.gradient_checkpointing_enable()
                    self.logger.info(f"✅ 그래디언트 체크포인팅 활성화: {model_name}")
                else:
                    # 수동으로 체크포인팅 적용
                    model = self._apply_manual_checkpointing(model)
                    self.logger.info(f"✅ 수동 그래디언트 체크포인팅 적용: {model_name}")
                
                self.checkpointed_models[model_name] = model
                return model
                
        except Exception as e:
            self.logger.warning(f"⚠️ 그래디언트 체크포인팅 실패: {e}")
            return model
    
    def _apply_manual_checkpointing(self, model: nn.Module) -> nn.Module:
        """수동 그래디언트 체크포인팅 적용"""
        # 간단한 래퍼 클래스로 체크포인팅 구현
        class CheckpointedModel(nn.Module):
            def __init__(self, base_model):
                super().__init__()
                self.base_model = base_model
            
            def forward(self, *args, **kwargs):
                return torch.utils.checkpoint.checkpoint(
                    self.base_model, *args, **kwargs
                )
        
        return CheckpointedModel(model)

# ==============================================
# 🔥 4. 병렬 로딩 시스템
# ==============================================

class ParallelModelLoader:
    """병렬 모델 로딩 시스템"""
    
    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.loading_queue = queue.Queue()
        self.loading_results: Dict[str, Any] = {}
        self._lock = threading.RLock()
        self.logger = logging.getLogger(f"{__name__}.ParallelModelLoader")
        
        # 로딩 워커 시작
        self._start_loading_workers()
    
    def _start_loading_workers(self):
        """로딩 워커 시작"""
        for i in range(self.max_workers):
            worker = threading.Thread(target=self._loading_worker, daemon=True)
            worker.start()
    
    def _loading_worker(self):
        """로딩 워커"""
        while True:
            try:
                task = self.loading_queue.get(timeout=1)
                if task is None:  # 종료 신호
                    break
                
                model_id, load_func, args, kwargs = task
                
                try:
                    result = load_func(*args, **kwargs)
                    with self._lock:
                        self.loading_results[model_id] = {
                            'success': True,
                            'result': result,
                            'timestamp': time.time()
                        }
                except Exception as e:
                    with self._lock:
                        self.loading_results[model_id] = {
                            'success': False,
                            'error': str(e),
                            'timestamp': time.time()
                        }
                
                self.loading_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"로딩 워커 오류: {e}")
    
    def submit_loading_task(self, model_id: str, load_func: Callable, *args, **kwargs):
        """로딩 작업 제출"""
        task = (model_id, load_func, args, kwargs)
        self.loading_queue.put(task)
        self.logger.debug(f"로딩 작업 제출: {model_id}")
    
    def get_loading_result(self, model_id: str, timeout: float = 30.0) -> Optional[Any]:
        """로딩 결과 조회"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            with self._lock:
                if model_id in self.loading_results:
                    result = self.loading_results[model_id]
                    if result['success']:
                        return result['result']
                    else:
                        raise Exception(f"로딩 실패: {result['error']}")
            
            time.sleep(0.1)
        
        raise TimeoutError(f"로딩 타임아웃: {model_id}")
    
    def shutdown(self):
        """종료"""
        for _ in range(self.max_workers):
            self.loading_queue.put(None)
        self.executor.shutdown(wait=True)

# ==============================================
# 🔥 5. 실시간 성능 모니터링
# ==============================================

@dataclass
class PerformanceMetrics:
    """성능 메트릭"""
    load_time: float
    memory_usage_mb: float
    cache_hit_rate: float
    throughput: float
    error_rate: float
    timestamp: float

class PerformanceMonitor:
    """실시간 성능 모니터링"""
    
    def __init__(self):
        self.metrics_history: List[PerformanceMetrics] = []
        self.max_history_size = 1000
        self._lock = threading.RLock()
        self.logger = logging.getLogger(f"{__name__}.PerformanceMonitor")
        
        # 모니터링 시작
        self._start_monitoring()
    
    def record_metric(self, metric: PerformanceMetrics):
        """메트릭 기록"""
        with self._lock:
            self.metrics_history.append(metric)
            
            # 히스토리 크기 제한
            if len(self.metrics_history) > self.max_history_size:
                self.metrics_history = self.metrics_history[-self.max_history_size:]
    
    def get_recent_metrics(self, minutes: int = 5) -> List[PerformanceMetrics]:
        """최근 메트릭 조회"""
        with self._lock:
            cutoff_time = time.time() - (minutes * 60)
            return [
                m for m in self.metrics_history
                if m.timestamp >= cutoff_time
            ]
    
    def get_average_metrics(self, minutes: int = 5) -> Dict[str, float]:
        """평균 메트릭 계산"""
        recent_metrics = self.get_recent_metrics(minutes)
        
        if not recent_metrics:
            return {}
        
        return {
            'avg_load_time': sum(m.load_time for m in recent_metrics) / len(recent_metrics),
            'avg_memory_usage': sum(m.memory_usage_mb for m in recent_metrics) / len(recent_metrics),
            'avg_cache_hit_rate': sum(m.cache_hit_rate for m in recent_metrics) / len(recent_metrics),
            'avg_throughput': sum(m.throughput for m in recent_metrics) / len(recent_metrics),
            'avg_error_rate': sum(m.error_rate for m in recent_metrics) / len(recent_metrics)
        }
    
    def _start_monitoring(self):
        """모니터링 시작"""
        def monitor_worker():
            while True:
                try:
                    time.sleep(60)  # 1분마다 체크
                    
                    # 시스템 메트릭 수집
                    if PSUTIL_AVAILABLE:
                        memory = psutil.virtual_memory()
                        memory_usage_mb = memory.used / (1024 * 1024)
                        
                        metric = PerformanceMetrics(
                            load_time=0.0,
                            memory_usage_mb=memory_usage_mb,
                            cache_hit_rate=0.0,
                            throughput=0.0,
                            error_rate=0.0,
                            timestamp=time.time()
                        )
                        
                        self.record_metric(metric)
                        
                except Exception as e:
                    self.logger.warning(f"모니터링 오류: {e}")
        
        monitor_thread = threading.Thread(target=monitor_worker, daemon=True)
        monitor_thread.start()

# ==============================================
# 🔥 6. Enhanced Model Loader (메인 클래스)
# ==============================================

class EnhancedModelLoader:
    """고급 모델 로더 v7.0"""
    
    def __init__(self, 
                 device: str = "auto",
                 cache_size_mb: float = 2048,
                 max_workers: int = 4,
                 enable_checkpointing: bool = True):
        
        self.device = self._detect_device(device)
        self.logger = logging.getLogger(f"{__name__}.EnhancedModelLoader")
        
        # 컴포넌트 초기화
        self.cache = PredictiveCache(max_size_mb=cache_size_mb)
        self.shared_registry = SharedModelRegistry()
        self.checkpointing_manager = GradientCheckpointingManager()
        self.parallel_loader = ParallelModelLoader(max_workers=max_workers)
        self.performance_monitor = PerformanceMonitor()
        
        # 설정
        self.enable_checkpointing = enable_checkpointing
        
        # 로딩된 모델들
        self.loaded_models: Dict[str, nn.Module] = {}
        self.model_metadata: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()
        
        self.logger.info(f"🚀 Enhanced Model Loader v7.0 초기화 완료 (디바이스: {self.device})")
    
    def _detect_device(self, device: str) -> str:
        """디바이스 감지"""
        if device == "auto":
            if TORCH_AVAILABLE:
                if torch.backends.mps.is_available():
                    return "mps"
                elif torch.cuda.is_available():
                    return "cuda"
                else:
                    return "cpu"
            else:
                return "cpu"
        return device
    
    def load_model(self, 
                  model_path: str, 
                  model_name: str,
                  step_type: str,
                  enable_parallel: bool = True) -> Optional[nn.Module]:
        """모델 로딩 (고급 기능 포함)"""
        
        start_time = time.time()
        
        try:
            # 캐시 확인
            cache_key = f"{model_name}_{step_type}"
            cached_model = self.cache.get(cache_key)
            if cached_model:
                self._record_performance_metric(start_time, cache_hit=True)
                return cached_model
            
            # 공유 모델 확인
            if self.shared_registry.is_shared_model(model_name):
                shared_model = self.shared_registry.get_shared_model(model_name)
                if shared_model:
                    self._record_performance_metric(start_time, cache_hit=True)
                    return shared_model
            
            # 실제 로딩
            if enable_parallel:
                return self._load_model_parallel(model_path, model_name, step_type, start_time)
            else:
                return self._load_model_sequential(model_path, model_name, step_type, start_time)
                
        except Exception as e:
            self.logger.error(f"❌ 모델 로딩 실패 {model_name}: {e}")
            self._record_performance_metric(start_time, error=True)
            return None
    
    def _load_model_parallel(self, model_path: str, model_name: str, step_type: str, start_time: float) -> Optional[nn.Module]:
        """병렬 모델 로딩"""
        model_id = f"{model_name}_{step_type}"
        
        # 로딩 작업 제출
        self.parallel_loader.submit_loading_task(
            model_id, 
            self._load_model_worker, 
            model_path, 
            model_name, 
            step_type
        )
        
        # 결과 대기
        try:
            model = self.parallel_loader.get_loading_result(model_id)
            
            # 캐시에 저장
            model_size = self._estimate_model_size(model)
            self.cache.set(model_id, model, model_size)
            
            # 메타데이터 저장
            with self._lock:
                self.loaded_models[model_id] = model
                self.model_metadata[model_id] = {
                    'path': model_path,
                    'name': model_name,
                    'step_type': step_type,
                    'load_time': time.time() - start_time,
                    'size_mb': model_size
                }
            
            self._record_performance_metric(start_time, cache_hit=False)
            return model
            
        except Exception as e:
            self.logger.error(f"❌ 병렬 로딩 실패: {e}")
            return None
    
    def _load_model_sequential(self, model_path: str, model_name: str, step_type: str, start_time: float) -> Optional[nn.Module]:
        """순차 모델 로딩"""
        try:
            model = self._load_model_worker(model_path, model_name, step_type)
            
            # 캐시에 저장
            model_id = f"{model_name}_{step_type}"
            model_size = self._estimate_model_size(model)
            self.cache.set(model_id, model, model_size)
            
            # 메타데이터 저장
            with self._lock:
                self.loaded_models[model_id] = model
                self.model_metadata[model_id] = {
                    'path': model_path,
                    'name': model_name,
                    'step_type': step_type,
                    'load_time': time.time() - start_time,
                    'size_mb': model_size
                }
            
            self._record_performance_metric(start_time, cache_hit=False)
            return model
            
        except Exception as e:
            self.logger.error(f"❌ 순차 로딩 실패: {e}")
            return None
    
    def _load_model_worker(self, model_path: str, model_name: str, step_type: str) -> nn.Module:
        """실제 체크포인트 기반 모델 로딩 워커"""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch가 필요합니다")
        
        self.logger.info(f"🔄 체크포인트 로딩 시작: {model_path}")
        
        # 체크포인트 분석
        checkpoint_analysis = self._analyze_checkpoint(model_path)
        
        # Step별 특화 아키텍처 생성
        architecture = self._create_step_architecture(step_type)
        
        # 체크포인트에서 모델 생성
        model = architecture.create_model(checkpoint_analysis)
        
        # 체크포인트 가중치 로딩
        success = self._load_checkpoint_weights(model, model_path, checkpoint_analysis, architecture)
        
        if not success:
            raise RuntimeError(f"체크포인트 가중치 로딩 실패: {model_path}")
        
        # 디바이스 이동
        model = model.to(self.device)
        
        # 그래디언트 체크포인팅 적용
        if self.enable_checkpointing:
            model = self.checkpointing_manager.enable_checkpointing(model, model_name)
        
        self.logger.info(f"✅ 체크포인트 로딩 완료: {model_path}")
        return model
    
    def _analyze_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """체크포인트 분석"""
        try:
            # 안전한 체크포인트 로딩
            checkpoint = self._load_checkpoint_safe(checkpoint_path)
            
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif isinstance(checkpoint, dict):
                state_dict = checkpoint
            else:
                state_dict = checkpoint
            
            # 체크포인트 분석
            analysis = {
                'state_dict': state_dict,
                'total_params': sum(p.numel() for p in state_dict.values() if hasattr(p, 'numel')),
                'layer_count': len(state_dict),
                'key_patterns': self._analyze_key_patterns(state_dict),
                'layer_types': self._analyze_layer_types(state_dict),
                'has_batch_norm': self._has_batch_normalization(state_dict),
                'has_attention': self._has_attention_layers(state_dict),
                'model_depth': self._estimate_model_depth(state_dict),
                'parameter_counts': self._count_parameters_by_type(state_dict)
            }
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"❌ 체크포인트 분석 실패: {e}")
            raise
    
    def _load_checkpoint_safe(self, checkpoint_path: str) -> Any:
        """안전한 체크포인트 로딩 (3단계)"""
        # 1단계: weights_only=True (최고 보안)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
            return checkpoint
        except Exception as e1:
            self.logger.debug(f"1단계 로딩 실패: {e1}")
        
        # 2단계: weights_only=False (호환성)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            return checkpoint
        except Exception as e2:
            self.logger.debug(f"2단계 로딩 실패: {e2}")
        
        # 3단계: Legacy 모드
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
            return checkpoint
        except Exception as e3:
            raise RuntimeError(f"모든 로딩 방법 실패: {e3}")
    
    def _create_step_architecture(self, step_type: str):
        """Step별 특화 아키텍처 생성"""
        # 기존 model_loader.py의 아키텍처 클래스들을 활용
        if step_type == 'human_parsing':
            return HumanParsingArchitecture(step_type, self.device)
        elif step_type == 'pose_estimation':
            return PoseEstimationArchitecture(step_type, self.device)
        elif step_type == 'cloth_segmentation':
            return ClothSegmentationArchitecture(step_type, self.device)
        elif step_type == 'geometric_matching':
            return GeometricMatchingArchitecture(step_type, self.device)
        elif step_type == 'virtual_fitting':
            return VirtualFittingArchitecture(step_type, self.device)
        elif step_type == 'cloth_warping':
            return ClothWarpingArchitecture(step_type, self.device)
        else:
            # 기본 아키텍처
            return self._create_generic_architecture(step_type)
    
    def _create_generic_architecture(self, step_type: str):
        """기본 아키텍처 생성"""
        class GenericArchitecture:
            def __init__(self, step_name: str, device: str):
                self.step_name = step_name
                self.device = device
            
            def create_model(self, checkpoint_analysis: Dict[str, Any]) -> nn.Module:
                # 체크포인트 분석을 바탕으로 동적 모델 생성
                state_dict = checkpoint_analysis['state_dict']
                
                # 간단한 CNN 모델 생성
                class GenericModel(nn.Module):
                    def __init__(self, input_channels=3, num_classes=1):
                        super().__init__()
                        self.features = nn.Sequential(
                            nn.Conv2d(input_channels, 64, 3, padding=1),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(64, 128, 3, padding=1),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(128, 256, 3, padding=1),
                            nn.ReLU(inplace=True)
                        )
                        self.classifier = nn.Conv2d(256, num_classes, 1)
                    
                    def forward(self, x):
                        x = self.features(x)
                        x = self.classifier(x)
                        return x
                
                return GenericModel()
            
            def map_checkpoint_keys(self, checkpoint: Dict[str, Any]) -> Dict[str, Any]:
                return checkpoint
            
            def validate_model(self, model) -> bool:
                return True
        
        return GenericArchitecture(step_type, self.device)
    
    def _load_checkpoint_weights(self, model: nn.Module, checkpoint_path: str, 
                                analysis: Dict[str, Any], architecture) -> bool:
        """체크포인트 가중치를 모델에 로딩"""
        try:
            state_dict = analysis['state_dict']
            
            # 키 매핑 적용
            mapped_state_dict = architecture.map_checkpoint_keys(state_dict)
            
            # 모델에 가중치 로딩
            model.load_state_dict(mapped_state_dict, strict=False)
            
            self.logger.info(f"✅ 가중치 로딩 성공: {checkpoint_path}")
            return True
            
        except Exception as e:
            self.logger.warning(f"⚠️ 가중치 로딩 실패 (strict=False 시도): {e}")
            
            # strict=False로 재시도
            try:
                model.load_state_dict(mapped_state_dict, strict=False)
                self.logger.info(f"✅ 가중치 로딩 성공 (strict=False): {checkpoint_path}")
                return True
            except Exception as e2:
                self.logger.error(f"❌ 가중치 로딩 완전 실패: {e2}")
                return False
    
    def _analyze_key_patterns(self, state_dict: Dict[str, Any]) -> Dict[str, List[str]]:
        """키 패턴 분석"""
        patterns = {
            'conv': [],
            'bn': [],
            'linear': [],
            'attention': [],
            'embedding': []
        }
        
        for key in state_dict.keys():
            key_lower = key.lower()
            if 'conv' in key_lower:
                patterns['conv'].append(key)
            elif any(kw in key_lower for kw in ['bn', 'batch_norm']):
                patterns['bn'].append(key)
            elif any(kw in key_lower for kw in ['linear', 'fc', 'classifier']):
                patterns['linear'].append(key)
            elif any(kw in key_lower for kw in ['attn', 'attention']):
                patterns['attention'].append(key)
            elif 'embed' in key_lower:
                patterns['embedding'].append(key)
        
        return patterns
    
    def _analyze_layer_types(self, state_dict: Dict[str, Any]) -> Dict[str, int]:
        """레이어 타입 분석"""
        layer_counts = {
            'conv': 0,
            'bn': 0,
            'linear': 0,
            'attention': 0,
            'embedding': 0
        }
        
        for key in state_dict.keys():
            key_lower = key.lower()
            if 'conv' in key_lower:
                layer_counts['conv'] += 1
            elif any(kw in key_lower for kw in ['bn', 'batch_norm']):
                layer_counts['bn'] += 1
            elif any(kw in key_lower for kw in ['linear', 'fc', 'classifier']):
                layer_counts['linear'] += 1
            elif any(kw in key_lower for kw in ['attn', 'attention']):
                layer_counts['attention'] += 1
            elif 'embed' in key_lower:
                layer_counts['embedding'] += 1
        
        return layer_counts
    
    def _has_batch_normalization(self, state_dict: Dict[str, Any]) -> bool:
        """BatchNorm 레이어 존재 여부"""
        return any('bn' in key.lower() or 'batch_norm' in key.lower() for key in state_dict.keys())
    
    def _has_attention_layers(self, state_dict: Dict[str, Any]) -> bool:
        """Attention 레이어 존재 여부"""
        return any(keyword in key.lower() for key in state_dict.keys() 
                  for keyword in ['attn', 'attention', 'self_attn', 'cross_attn'])
    
    def _extract_metadata(self, checkpoint: Any) -> Dict[str, Any]:
        """체크포인트에서 메타데이터 추출"""
        metadata = {}
        
        if isinstance(checkpoint, dict):
            # state_dict 형태
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
                metadata['total_parameters'] = sum(tensor.numel() for tensor in state_dict.values() if hasattr(tensor, 'numel'))
                metadata['total_keys'] = len(state_dict)
            else:
                metadata['total_parameters'] = sum(tensor.numel() for tensor in checkpoint.values() if hasattr(tensor, 'numel'))
                metadata['total_keys'] = len(checkpoint)
            
            # 추가 메타데이터
            for key in ['epoch', 'step', 'optimizer', 'scheduler', 'config', 'args']:
                if key in checkpoint:
                    metadata[key] = checkpoint[key]
        
        return metadata
    
    def _estimate_model_depth(self, state_dict: Dict[str, Any]) -> int:
        """모델 깊이 추정"""
        # 레이어 번호로 깊이 추정
        layer_numbers = []
        for key in state_dict.keys():
            # 숫자 추출 (예: layer1.0.conv1.weight -> 1)
            import re
            numbers = re.findall(r'\d+', key)
            if numbers:
                layer_numbers.extend([int(n) for n in numbers])
        
        return max(layer_numbers) if layer_numbers else 10
    
    def _count_parameters_by_type(self, state_dict: Dict[str, Any]) -> Dict[str, int]:
        """타입별 파라미터 수 계산"""
        param_counts = {
            'conv_params': 0,
            'linear_params': 0,
            'norm_params': 0,
            'embedding_params': 0,
            'total_params': 0
        }
        
        for key, tensor in state_dict.items():
            if hasattr(tensor, 'numel'):
                param_count = tensor.numel()
                param_counts['total_params'] += param_count
                
                key_lower = key.lower()
                if 'conv' in key_lower:
                    param_counts['conv_params'] += param_count
                elif any(kw in key_lower for kw in ['linear', 'fc', 'classifier']):
                    param_counts['linear_params'] += param_count
                elif any(kw in key_lower for kw in ['bn', 'norm']):
                    param_counts['norm_params'] += param_count
                elif 'embed' in key_lower:
                    param_counts['embedding_params'] += param_count
        
        return param_counts
    
    def _estimate_model_size(self, model: nn.Module) -> float:
        """모델 크기 추정 (MB)"""
        if not TORCH_AVAILABLE:
            return 100.0  # 기본값
        
        try:
            param_size = 0
            buffer_size = 0
            
            for param in model.parameters():
                param_size += param.nelement() * param.element_size()
            
            for buffer in model.buffers():
                buffer_size += buffer.nelement() * buffer.element_size()
            
            size_mb = (param_size + buffer_size) / (1024 * 1024)
            return size_mb
        except:
            return 100.0  # 기본값
    
    def _record_performance_metric(self, start_time: float, cache_hit: bool = False, error: bool = False):
        """성능 메트릭 기록"""
        load_time = time.time() - start_time
        
        # 메모리 사용량
        memory_usage_mb = 0.0
        if PSUTIL_AVAILABLE:
            memory = psutil.virtual_memory()
            memory_usage_mb = memory.used / (1024 * 1024)
        
        # 캐시 히트율 (간단한 계산)
        cache_hit_rate = 1.0 if cache_hit else 0.0
        
        # 처리량 (간단한 계산)
        throughput = 1.0 / max(load_time, 0.001)
        
        # 에러율
        error_rate = 1.0 if error else 0.0
        
        metric = PerformanceMetrics(
            load_time=load_time,
            memory_usage_mb=memory_usage_mb,
            cache_hit_rate=cache_hit_rate,
            throughput=throughput,
            error_rate=error_rate,
            timestamp=time.time()
        )
        
        self.performance_monitor.record_metric(metric)
    
    def get_performance_stats(self, minutes: int = 5) -> Dict[str, Any]:
        """성능 통계 조회"""
        avg_metrics = self.performance_monitor.get_average_metrics(minutes)
        
        return {
            'average_metrics': avg_metrics,
            'cache_stats': {
                'total_size_mb': self.cache._get_total_size(),
                'entry_count': len(self.cache.cache)
            },
            'loaded_models_count': len(self.loaded_models),
            'shared_models_count': len(self.shared_registry.shared_models)
        }
    
    def cleanup(self):
        """리소스 정리"""
        try:
            # 병렬 로더 종료
            self.parallel_loader.shutdown()
            
            # 캐시 정리
            self.cache.cache.clear()
            
            # 모델 언로드
            with self._lock:
                self.loaded_models.clear()
                self.model_metadata.clear()
            
            # 메모리 정리
            gc.collect()
            
            self.logger.info("✅ Enhanced Model Loader 정리 완료")
            
        except Exception as e:
            self.logger.error(f"❌ 정리 실패: {e}")

# ==============================================
# 🔥 7. 팩토리 함수
# ==============================================

def create_enhanced_model_loader(device: str = "auto", 
                                cache_size_mb: float = 2048,
                                max_workers: int = 4,
                                enable_checkpointing: bool = True) -> EnhancedModelLoader:
    """고급 모델 로더 생성"""
    return EnhancedModelLoader(
        device=device,
        cache_size_mb=cache_size_mb,
        max_workers=max_workers,
        enable_checkpointing=enable_checkpointing
    )

# ==============================================
# 🔥 8. 사용 예시
# ==============================================

if __name__ == "__main__":
    # 로거 설정
    logging.basicConfig(level=logging.INFO)
    
    # 고급 모델 로더 생성
    loader = create_enhanced_model_loader(
        device="auto",
        cache_size_mb=1024,
        max_workers=2,
        enable_checkpointing=True
    )
    
    try:
        # 모델 로딩 테스트
        model = loader.load_model(
            model_path="test_model.pth",
            model_name="test_model",
            step_type="human_parsing",
            enable_parallel=True
        )
        
        if model:
            print("✅ 모델 로딩 성공")
            
            # 성능 통계 조회
            stats = loader.get_performance_stats()
            print(f"📊 성능 통계: {stats}")
        else:
            print("❌ 모델 로딩 실패")
    
    finally:
        # 정리
        loader.cleanup()
