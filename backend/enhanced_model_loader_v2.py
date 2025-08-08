#!/usr/bin/env python3
"""
🔥 Enhanced ModelLoader v2.0 → 체크포인트 분석 + auto_detector 통합
================================================================================

✅ 체크포인트 분석 + auto_detector 통합
✅ 지연 로딩 (Lazy Loading)
✅ 스마트 캐싱 시스템
✅ 기존 API 100% 호환성
✅ Step 파일들과 완전 호환

Author: MyCloset AI Team
Date: 2024-08-09
Version: 2.0
"""

import os
import sys
import gc
import time
import json
import logging
import asyncio
import threading
import traceback
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Tuple, Type, Set, Callable
from dataclasses import dataclass, field
from enum import Enum
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache, wraps
from collections import defaultdict, OrderedDict

# PyTorch imports
import torch
import torch.nn as nn

# 기존 모델 로더 호환성을 위한 imports
try:
    from app.ai_pipeline.utils.model_loader import (
        RealStepModelType, RealModelStatus, RealModelPriority,
        RealStepModelInfo, RealStepModelRequirement, RealAIModel,
        RealStepModelInterface, ModelLoader as BaseModelLoader
    )
except ImportError:
    # Fallback for standalone usage
    class RealStepModelType(Enum):
        HUMAN_PARSING = "human_parsing"
        POSE_ESTIMATION = "pose_estimation"
        CLOTH_SEGMENTATION = "cloth_segmentation"
        GEOMETRIC_MATCHING = "geometric_matching"
        CLOTH_WARPING = "cloth_warping"
        VIRTUAL_FITTING = "virtual_fitting"
        POST_PROCESSING = "post_processing"
        QUALITY_ASSESSMENT = "quality_assessment"

    class RealModelStatus(Enum):
        NOT_LOADED = "not_loaded"
        LOADING = "loading"
        LOADED = "loaded"
        ERROR = "error"
        VALIDATING = "validating"

    class RealModelPriority(Enum):
        PRIMARY = 1
        SECONDARY = 2
        FALLBACK = 3
        OPTIONAL = 4

    @dataclass
    class RealStepModelInfo:
        name: str
        path: str
        step_type: RealStepModelType
        priority: RealModelPriority
        device: str
        memory_mb: float = 0.0
        loaded: bool = False
        load_time: float = 0.0
        checkpoint_data: Optional[Any] = None
        model_type: str = "BaseModel"
        size_gb: float = 0.0
        requires_checkpoint: bool = True
        error: Optional[str] = None
        validation_passed: bool = False

    @dataclass
    class RealStepModelRequirement:
        step_name: str
        step_id: int
        step_type: RealStepModelType
        required_models: List[str] = field(default_factory=list)
        optional_models: List[str] = field(default_factory=list)
        primary_model: Optional[str] = None
        model_configs: Dict[str, Any] = field(default_factory=dict)
        batch_size: int = 1
        precision: str = "fp32"
        memory_limit_mb: Optional[float] = None

    class RealAIModel:
        def __init__(self, model_name: str, model_path: str, step_type: RealStepModelType, device: str = "auto"):
            self.model_name = model_name
            self.model_path = model_path
            self.step_type = step_type
            self.device = device
            self.model_instance = None
            self.checkpoint_data = None
            self.loaded = False
            self.load_time = 0.0
            self.error = None

        def load(self, validate: bool = True) -> bool:
            return True

        def get_model_instance(self) -> Optional[Any]:
            return self.model_instance

        def unload(self):
            self.model_instance = None
            self.checkpoint_data = None
            self.loaded = False

    class RealStepModelInterface:
        def __init__(self, model_loader, step_name: str, step_type: RealStepModelType):
            self.model_loader = model_loader
            self.step_name = step_name
            self.step_type = step_type

        def get_model(self, model_name: Optional[str] = None) -> Optional[RealAIModel]:
            return None

        def register_requirements(self, requirements: Dict[str, Any]):
            pass

# auto_model_detector 통합
try:
    from app.ai_pipeline.utils.auto_model_detector import (
        get_global_detector, OptimizedModelDetector, OptimizedDetectedModel,
        quick_model_detection, detect_ultra_large_models, find_model_by_name
    )
    AUTO_DETECTOR_AVAILABLE = True
except ImportError:
    AUTO_DETECTOR_AVAILABLE = False
    get_global_detector = None
    OptimizedModelDetector = None
    OptimizedDetectedModel = None
    quick_model_detection = None
    detect_ultra_large_models = None
    find_model_by_name = None

# ==============================================
# 🔥 통합 데이터 구조
# ==============================================

@dataclass
class IntegratedModelInfo:
    """통합된 모델 정보 (체크포인트 분석 + auto_detector)"""
    name: str
    path: str
    step_type: RealStepModelType
    file_size_mb: float
    file_size_gb: float
    
    # auto_detector 정보
    auto_detector_info: Optional[Dict[str, Any]] = None
    
    # 통합된 메타데이터
    ai_class: str = "BaseRealAIModel"
    confidence_score: float = 0.0
    priority_score: float = 0.0
    is_valid: bool = True
    error: Optional[str] = None
    
    def __post_init__(self):
        """초기화 후 통합 정보 계산"""
        self.file_size_gb = self.file_size_mb / 1024
        
        # auto_detector 정보가 있으면 우선 사용
        if self.auto_detector_info:
            self.ai_class = self.auto_detector_info.get('ai_model_info', {}).get('ai_class', self.ai_class)
            self.confidence_score = self.auto_detector_info.get('confidence', self.confidence_score)
            self.priority_score = self.auto_detector_info.get('priority_info', {}).get('priority_score', self.priority_score)

class ModelCache:
    """스마트 모델 캐싱 시스템"""
    
    def __init__(self, max_size: int = 10, max_memory_gb: float = 8.0):
        self.max_size = max_size
        self.max_memory_gb = max_memory_gb
        self.cache: OrderedDict[str, RealAIModel] = OrderedDict()
        self.access_count: Dict[str, int] = defaultdict(int)
        self.last_access: Dict[str, float] = {}
        self.memory_usage_gb = 0.0
        self._lock = threading.RLock()
    
    def get(self, model_name: str) -> Optional[RealAIModel]:
        """캐시에서 모델 가져오기"""
        with self._lock:
            if model_name in self.cache:
                # LRU 업데이트
                model = self.cache.pop(model_name)
                self.cache[model_name] = model
                
                # 접근 통계 업데이트
                self.access_count[model_name] += 1
                self.last_access[model_name] = time.time()
                
                return model
            return None
    
    def put(self, model_name: str, model: RealAIModel) -> bool:
        """캐시에 모델 추가"""
        with self._lock:
            # 메모리 사용량 계산 (추정)
            model_size_gb = getattr(model, 'size_gb', 0.5)  # 기본값 0.5GB
            
            # 캐시 크기 제한 확인
            if len(self.cache) >= self.max_size:
                self._evict_least_used()
            
            # 메모리 제한 확인
            if self.memory_usage_gb + model_size_gb > self.max_memory_gb:
                self._evict_by_memory()
            
            # 모델 추가
            self.cache[model_name] = model
            self.access_count[model_name] = 1
            self.last_access[model_name] = time.time()
            self.memory_usage_gb += model_size_gb
            
            return True
    
    def _evict_least_used(self):
        """가장 적게 사용된 모델 제거"""
        if not self.cache:
            return
        
        # 접근 횟수와 마지막 접근 시간을 고려한 점수 계산
        scores = {}
        current_time = time.time()
        
        for name in self.cache.keys():
            access_count = self.access_count.get(name, 0)
            last_access = self.last_access.get(name, 0)
            time_factor = max(1, (current_time - last_access) / 3600)  # 시간당 감소
            scores[name] = access_count / time_factor
        
        # 가장 낮은 점수의 모델 제거
        least_used = min(scores.keys(), key=lambda x: scores[x])
        self._remove_model(least_used)
    
    def _evict_by_memory(self):
        """메모리 사용량 기준으로 모델 제거"""
        if not self.cache:
            return
        
        # 가장 큰 모델부터 제거
        model_sizes = {}
        for name, model in self.cache.items():
            model_sizes[name] = getattr(model, 'size_gb', 0.5)
        
        largest_model = max(model_sizes.keys(), key=lambda x: model_sizes[x])
        self._remove_model(largest_model)
    
    def _remove_model(self, model_name: str):
        """모델 제거"""
        if model_name in self.cache:
            model = self.cache.pop(model_name)
            model_size_gb = getattr(model, 'size_gb', 0.5)
            self.memory_usage_gb -= model_size_gb
            
            # 모델 언로드
            if hasattr(model, 'unload'):
                model.unload()
            
            # 통계 정리
            self.access_count.pop(model_name, None)
            self.last_access.pop(model_name, None)
    
    def clear(self):
        """캐시 전체 정리"""
        with self._lock:
            for model_name in list(self.cache.keys()):
                self._remove_model(model_name)
    
    def get_stats(self) -> Dict[str, Any]:
        """캐시 통계 반환"""
        with self._lock:
            return {
                'cache_size': len(self.cache),
                'max_size': self.max_size,
                'memory_usage_gb': self.memory_usage_gb,
                'max_memory_gb': self.max_memory_gb,
                'cached_models': list(self.cache.keys()),
                'access_counts': dict(self.access_count),
                'last_access': dict(self.last_access)
            }

# ==============================================
# 🔥 개선된 ModelLoader 클래스
# ==============================================

class EnhancedModelLoader:
    """체크포인트 분석 + auto_detector 통합 기반 개선된 모델 로더"""
    
    def __init__(self, 
                 device: str = "auto",
                 model_cache_dir: Optional[str] = None,
                 max_cached_models: int = 10,
                 max_memory_gb: float = 8.0,
                 enable_auto_detector: bool = True,
                 **kwargs):
        
        # 로거 설정
        self.logger = logging.getLogger(f"{__name__}.EnhancedModelLoader")
        
        # 기본 설정
        self.device = self._setup_device(device)
        self.model_cache_dir = Path(model_cache_dir) if model_cache_dir else Path("ai_models")
        self.model_cache_dir.mkdir(parents=True, exist_ok=True)
        
        # 컴포넌트 초기화
        self.cache = ModelCache(max_cached_models, max_memory_gb)
        
        # auto_detector 통합
        self.enable_auto_detector = enable_auto_detector and AUTO_DETECTOR_AVAILABLE
        self.auto_detector = None
        if self.enable_auto_detector:
            try:
                self.auto_detector = get_global_detector()
                self.logger.info("✅ auto_detector 통합 성공")
            except Exception as e:
                self.logger.warning(f"⚠️ auto_detector 통합 실패: {e}")
                self.enable_auto_detector = False
        
        # 모델 관리
        self.loaded_models: Dict[str, RealAIModel] = {}
        self.model_info: Dict[str, RealStepModelInfo] = {}
        self.model_status: Dict[str, RealModelStatus] = {}
        self.integrated_model_info: Dict[str, IntegratedModelInfo] = {}
        
        # Step 요구사항
        self.step_requirements: Dict[str, RealStepModelRequirement] = {}
        self.step_interfaces: Dict[str, RealStepModelInterface] = {}
        
        # 성능 메트릭
        self.performance_metrics = {
            'models_loaded': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'total_memory_mb': 0.0,
            'error_count': 0,
            'total_load_time': 0.0,
            'auto_detector_hits': 0
        }
        
        # 동기화
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="EnhancedModelLoader")
        
        # 데이터 로드
        self._load_checkpoint_analysis()
        self._load_auto_detector_data()
        self._integrate_model_data()
        
        self.logger.info(f"🚀 EnhancedModelLoader v2.0 초기화 완료")
        self.logger.info(f"📱 Device: {self.device}")
        self.logger.info(f"💾 Cache: {max_cached_models} models, {max_memory_gb}GB")
        self.logger.info(f"🔍 Auto Detector: {'Enabled' if self.enable_auto_detector else 'Disabled'}")
    
    def _setup_device(self, device: str) -> str:
        """디바이스 설정"""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return device
    
    def _load_checkpoint_analysis(self):
        """체크포인트 분석 결과 로드"""
        try:
            analysis_file = Path("comprehensive_checkpoint_analysis.json")
            if analysis_file.exists():
                with open(analysis_file, 'r', encoding='utf-8') as f:
                    analysis_data = json.load(f)
                
                self.checkpoint_analysis = analysis_data
                self.logger.info(f"✅ 체크포인트 분석 결과 로드: {len(analysis_data.get('checkpoints', {}))}개")
            else:
                self.checkpoint_analysis = {}
                self.logger.warning("⚠️ 체크포인트 분석 파일을 찾을 수 없음")
                
        except Exception as e:
            self.logger.error(f"❌ 체크포인트 분석 로드 실패: {e}")
            self.checkpoint_analysis = {}
    
    def _load_auto_detector_data(self):
        """auto_detector 데이터 로드"""
        if not self.enable_auto_detector or not self.auto_detector:
            self.auto_detector_data = {}
            return
        
        try:
            # auto_detector에서 모든 모델 정보 가져오기
            detected_models = self.auto_detector.detect_all_models()
            self.auto_detector_data = {
                name: model.to_dict() for name, model in detected_models.items()
            }
            self.logger.info(f"✅ auto_detector 데이터 로드: {len(self.auto_detector_data)}개")
            
        except Exception as e:
            self.logger.error(f"❌ auto_detector 데이터 로드 실패: {e}")
            self.auto_detector_data = {}
    
    def _integrate_model_data(self):
        """체크포인트 분석과 auto_detector 데이터 통합"""
        try:
            self.logger.info("🔄 모델 데이터 통합 시작...")
            
            # auto_detector 데이터를 기준으로 통합
            for model_name, auto_info in self.auto_detector_data.items():
                try:
                    # 통합 모델 정보 생성
                    integrated_info = IntegratedModelInfo(
                        name=model_name,
                        path=auto_info.get('path', ''),
                        step_type=self._map_step_type(auto_info.get('step_class', '')),
                        file_size_mb=auto_info.get('size_mb', 0.0),
                        auto_detector_info=auto_info,
                        ai_class=auto_info.get('ai_model_info', {}).get('ai_class', 'BaseRealAIModel'),
                        confidence_score=auto_info.get('confidence', 0.0),
                        priority_score=auto_info.get('priority_info', {}).get('priority_score', 0.0)
                    )
                    
                    self.integrated_model_info[model_name] = integrated_info
                    
                except Exception as e:
                    self.logger.error(f"❌ 모델 통합 실패 {model_name}: {e}")
                    continue
            
            self.logger.info(f"✅ 모델 데이터 통합 완료: {len(self.integrated_model_info)}개")
            
        except Exception as e:
            self.logger.error(f"❌ 모델 데이터 통합 실패: {e}")
    
    def _map_step_type(self, step_class: str) -> RealStepModelType:
        """Step 클래스명을 RealStepModelType으로 매핑"""
        step_class_lower = step_class.lower()
        
        if 'human' in step_class_lower or 'parsing' in step_class_lower:
            return RealStepModelType.HUMAN_PARSING
        elif 'pose' in step_class_lower:
            return RealStepModelType.POSE_ESTIMATION
        elif 'segmentation' in step_class_lower:
            return RealStepModelType.CLOTH_SEGMENTATION
        elif 'geometric' in step_class_lower or 'matching' in step_class_lower:
            return RealStepModelType.GEOMETRIC_MATCHING
        elif 'warping' in step_class_lower:
            return RealStepModelType.CLOTH_WARPING
        elif 'fitting' in step_class_lower or 'virtual' in step_class_lower:
            return RealStepModelType.VIRTUAL_FITTING
        elif 'post' in step_class_lower:
            return RealStepModelType.POST_PROCESSING
        elif 'quality' in step_class_lower or 'assessment' in step_class_lower:
            return RealStepModelType.QUALITY_ASSESSMENT
        
        return RealStepModelType.HUMAN_PARSING  # 기본값
    
    def load_model(self, model_name: str, step_type: Optional[RealStepModelType] = None, **kwargs) -> Optional[RealAIModel]:
        """모델 로딩 (통합 데이터 기반)"""
        try:
            # 캐시 확인
            cached_model = self.cache.get(model_name)
            if cached_model:
                self.performance_metrics['cache_hits'] += 1
                return cached_model
            
            self.performance_metrics['cache_misses'] += 1
            
            # 통합 모델 정보 확인
            integrated_info = self.integrated_model_info.get(model_name)
            if integrated_info:
                self.performance_metrics['auto_detector_hits'] += 1
                
                # 통합 정보에서 경로와 타입 가져오기
                model_path = integrated_info.path
                if not step_type:
                    step_type = integrated_info.step_type
                
                # 모델 로딩
                start_time = time.time()
                
                model = RealAIModel(model_name, model_path, step_type, self.device)
                success = model.load()
                
                if success:
                    # 통합 정보를 모델에 추가
                    model.integrated_info = integrated_info
                    
                    # 캐시에 추가
                    self.cache.put(model_name, model)
                    
                    # 성능 메트릭 업데이트
                    load_time = time.time() - start_time
                    self.performance_metrics['total_load_time'] += load_time
                    self.performance_metrics['models_loaded'] += 1
                    
                    self.logger.info(f"✅ 모델 로딩 완료: {model_name} ({load_time:.2f}s)")
                    return model
            
            # 통합 정보가 없으면 기존 방식으로 시도
            return self._fallback_load_model(model_name, step_type, **kwargs)
            
        except Exception as e:
            self.logger.error(f"❌ 모델 로딩 실패: {model_name} - {e}")
            self.performance_metrics['error_count'] += 1
        
        return None
    
    def _fallback_load_model(self, model_name: str, step_type: Optional[RealStepModelType] = None, **kwargs) -> Optional[RealAIModel]:
        """기존 방식으로 모델 로딩 (폴백)"""
        # 모델 경로 찾기
        model_path = self._find_model_path(model_name)
        if not model_path:
            self.logger.error(f"❌ 모델 경로를 찾을 수 없음: {model_name}")
            return None
        
        # Step 타입 추정
        if not step_type:
            step_type = self._infer_step_type(model_name, model_path)
        
        # 모델 로딩
        start_time = time.time()
        
        model = RealAIModel(model_name, model_path, step_type, self.device)
        success = model.load()
        
        if success:
            # 캐시에 추가
            self.cache.put(model_name, model)
            
            # 성능 메트릭 업데이트
            load_time = time.time() - start_time
            self.performance_metrics['total_load_time'] += load_time
            self.performance_metrics['models_loaded'] += 1
            
            self.logger.info(f"✅ 모델 로딩 완료 (폴백): {model_name} ({load_time:.2f}s)")
            return model
        
        return None
    
    def _find_model_path(self, model_name: str) -> Optional[str]:
        """모델 경로 찾기"""
        # 통합 정보에서 먼저 찾기
        if model_name in self.integrated_model_info:
            return self.integrated_model_info[model_name].path
        
        # 체크포인트 분석 결과에서 경로 찾기
        if self.checkpoint_analysis:
            for path, analysis in self.checkpoint_analysis.get('checkpoints', {}).items():
                if model_name in path or model_name in os.path.basename(path):
                    return path
        
        # 기본 경로 시도
        possible_paths = [
            self.model_cache_dir / f"{model_name}.pth",
            self.model_cache_dir / f"{model_name}.pt",
            self.model_cache_dir / f"{model_name}.bin",
            self.model_cache_dir / f"{model_name}.safetensors"
        ]
        
        for path in possible_paths:
            if path.exists():
                return str(path)
        
        return None
    
    def _infer_step_type(self, model_name: str, model_path: str) -> RealStepModelType:
        """Step 타입 추정"""
        # 파일명 기반 추정
        model_name_lower = model_name.lower()
        path_lower = model_path.lower()
        
        if any(x in model_name_lower for x in ['graphonomy', 'schp', 'deeplab', 'human']):
            return RealStepModelType.HUMAN_PARSING
        elif any(x in model_name_lower for x in ['hrnet', 'openpose', 'yolo', 'pose']):
            return RealStepModelType.POSE_ESTIMATION
        elif any(x in model_name_lower for x in ['sam', 'u2net', 'segmentation']):
            return RealStepModelType.CLOTH_SEGMENTATION
        elif any(x in model_name_lower for x in ['gmm', 'tps', 'raft', 'geometric']):
            return RealStepModelType.GEOMETRIC_MATCHING
        elif any(x in model_name_lower for x in ['warping', 'viton', 'hrviton']):
            return RealStepModelType.CLOTH_WARPING
        elif any(x in model_name_lower for x in ['diffusion', 'ootd', 'stable']):
            return RealStepModelType.VIRTUAL_FITTING
        elif any(x in model_name_lower for x in ['esrgan', 'gfpgan', 'swinir', 'enhance']):
            return RealStepModelType.POST_PROCESSING
        elif any(x in model_name_lower for x in ['clip', 'lpips', 'quality']):
            return RealStepModelType.QUALITY_ASSESSMENT
        
        # 경로 기반 추정
        if 'step_01' in path_lower:
            return RealStepModelType.HUMAN_PARSING
        elif 'step_02' in path_lower:
            return RealStepModelType.POSE_ESTIMATION
        elif 'step_03' in path_lower:
            return RealStepModelType.CLOTH_SEGMENTATION
        elif 'step_04' in path_lower:
            return RealStepModelType.GEOMETRIC_MATCHING
        elif 'step_05' in path_lower:
            return RealStepModelType.CLOTH_WARPING
        elif 'step_06' in path_lower:
            return RealStepModelType.VIRTUAL_FITTING
        elif 'step_07' in path_lower:
            return RealStepModelType.POST_PROCESSING
        elif 'step_08' in path_lower:
            return RealStepModelType.QUALITY_ASSESSMENT
        
        return RealStepModelType.HUMAN_PARSING  # 기본값
    
    def create_step_interface(self, step_name: str, step_requirements: Optional[Dict[str, Any]] = None) -> RealStepModelInterface:
        """Step 인터페이스 생성 (기존 API 호환)"""
        step_type = self._infer_step_type_from_name(step_name)
        
        interface = RealStepModelInterface(self, step_name, step_type)
        
        if step_requirements:
            interface.register_requirements(step_requirements)
        
        self.step_interfaces[step_name] = interface
        return interface
    
    def _infer_step_type_from_name(self, step_name: str) -> RealStepModelType:
        """Step 이름으로부터 타입 추정"""
        step_name_lower = step_name.lower()
        
        if 'human' in step_name_lower or 'parsing' in step_name_lower:
            return RealStepModelType.HUMAN_PARSING
        elif 'pose' in step_name_lower:
            return RealStepModelType.POSE_ESTIMATION
        elif 'segmentation' in step_name_lower or 'cloth' in step_name_lower:
            return RealStepModelType.CLOTH_SEGMENTATION
        elif 'geometric' in step_name_lower or 'matching' in step_name_lower:
            return RealStepModelType.GEOMETRIC_MATCHING
        elif 'warping' in step_name_lower:
            return RealStepModelType.CLOTH_WARPING
        elif 'fitting' in step_name_lower or 'virtual' in step_name_lower:
            return RealStepModelType.VIRTUAL_FITTING
        elif 'post' in step_name_lower or 'enhance' in step_name_lower:
            return RealStepModelType.POST_PROCESSING
        elif 'quality' in step_name_lower or 'assessment' in step_name_lower:
            return RealStepModelType.QUALITY_ASSESSMENT
        
        return RealStepModelType.HUMAN_PARSING
    
    def get_integrated_model_info(self, model_name: str) -> Optional[IntegratedModelInfo]:
        """통합된 모델 정보 반환"""
        return self.integrated_model_info.get(model_name)
    
    def list_available_models(self, step_class: Optional[str] = None, model_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """사용 가능한 모델 목록 반환 (통합 데이터 기반)"""
        results = []
        
        for model_name, integrated_info in self.integrated_model_info.items():
            try:
                # 필터링
                if step_class and integrated_info.step_type.value != step_class.lower():
                    continue
                if model_type and integrated_info.auto_detector_info.get('model_type') != model_type:
                    continue
                
                # 결과 생성
                result = {
                    'name': model_name,
                    'path': integrated_info.path,
                    'step_type': integrated_info.step_type.value,
                    'file_size_mb': integrated_info.file_size_mb,
                    'file_size_gb': integrated_info.file_size_gb,
                    'ai_class': integrated_info.ai_class,
                    'confidence_score': integrated_info.confidence_score,
                    'priority_score': integrated_info.priority_score,
                    'is_valid': integrated_info.is_valid,
                    'has_auto_detector_info': integrated_info.auto_detector_info is not None
                }
                
                results.append(result)
                
            except Exception as e:
                self.logger.error(f"❌ 모델 정보 변환 실패 {model_name}: {e}")
                continue
        
        return results
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """성능 메트릭 반환"""
        cache_stats = self.cache.get_stats()
        
        return {
            **self.performance_metrics,
            'cache_stats': cache_stats,
            'loaded_models_count': len(self.loaded_models),
            'step_interfaces_count': len(self.step_interfaces),
            'integrated_models_count': len(self.integrated_model_info),
            'auto_detector_enabled': self.enable_auto_detector,
            'device': self.device
        }
    
    def cleanup(self):
        """리소스 정리"""
        try:
            # 캐시 정리
            self.cache.clear()
            
            # 스레드 풀 정리
            self._executor.shutdown(wait=True)
            
            self.logger.info("✅ EnhancedModelLoader 정리 완료")
            
        except Exception as e:
            self.logger.error(f"❌ 정리 중 오류: {e}")

# ==============================================
# 🔥 전역 인스턴스 및 유틸리티 함수
# ==============================================

_global_enhanced_model_loader = None

def get_global_enhanced_model_loader() -> EnhancedModelLoader:
    """전역 EnhancedModelLoader 인스턴스 반환"""
    global _global_enhanced_model_loader
    if _global_enhanced_model_loader is None:
        _global_enhanced_model_loader = EnhancedModelLoader()
    return _global_enhanced_model_loader

def initialize_enhanced_model_loader(**kwargs) -> EnhancedModelLoader:
    """EnhancedModelLoader 초기화"""
    global _global_enhanced_model_loader
    _global_enhanced_model_loader = EnhancedModelLoader(**kwargs)
    return _global_enhanced_model_loader

# 기존 API 호환성을 위한 함수들
def get_model(model_name: str) -> Optional[RealAIModel]:
    """모델 가져오기 (기존 API 호환)"""
    loader = get_global_enhanced_model_loader()
    return loader.load_model(model_name)

def create_step_interface(step_name: str, step_requirements: Optional[Dict[str, Any]] = None) -> RealStepModelInterface:
    """Step 인터페이스 생성 (기존 API 호환)"""
    loader = get_global_enhanced_model_loader()
    return loader.create_step_interface(step_name, step_requirements)

def get_performance_metrics() -> Dict[str, Any]:
    """성능 메트릭 반환 (기존 API 호환)"""
    loader = get_global_enhanced_model_loader()
    return loader.get_performance_metrics()

def list_available_models(step_class: Optional[str] = None, model_type: Optional[str] = None) -> List[Dict[str, Any]]:
    """사용 가능한 모델 목록 반환 (기존 API 호환)"""
    loader = get_global_enhanced_model_loader()
    return loader.list_available_models(step_class, model_type)

if __name__ == "__main__":
    # 테스트 코드
    logging.basicConfig(level=logging.INFO)
    
    # EnhancedModelLoader 초기화
    loader = EnhancedModelLoader()
    
    # 성능 메트릭 출력
    metrics = loader.get_performance_metrics()
    print("EnhancedModelLoader v2.0 성능 메트릭:")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    
    # 사용 가능한 모델 목록 출력
    available_models = loader.list_available_models()
    print(f"\n사용 가능한 모델: {len(available_models)}개")
    for model in available_models[:5]:  # 처음 5개만 출력
        print(f"  - {model['name']}: {model['file_size_mb']:.1f}MB ({model['ai_class']})")
    
    # 정리
    loader.cleanup()
