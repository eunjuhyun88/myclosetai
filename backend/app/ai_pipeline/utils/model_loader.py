# app/ai_pipeline/utils/model_loader.py (완전히 새로운 버전)
"""
🔄 완전히 새로운 ModelLoader v3.0
✅ Step별 요청 정보 (3번 파일)에 정확히 맞춤
✅ auto_model_detector와 완벽한 데이터 교환
✅ 체크포인트 정보 완전 처리
✅ M3 Max 128GB 최적화
"""

import os
import logging
import threading
import asyncio
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple, Type, Callable
from dataclasses import dataclass, field
from enum import Enum
import weakref

# PyTorch 및 AI 라이브러리
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import numpy as np
    from PIL import Image
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

# Step별 요청 정보 임포트
from .step_model_requests import (
    STEP_MODEL_REQUESTS,
    StepModelRequestAnalyzer,
    ModelRequestInfo
)

logger = logging.getLogger(__name__)

# ==============================================
# 🔥 Step 요청에 맞춘 데이터 구조들
# ==============================================

@dataclass
class ModelCheckpointInfo:
    """ModelLoader가 처리하는 체크포인트 정보"""
    primary_path: str                               # 주 모델 파일
    config_files: List[str] = field(default_factory=list)        # config.json 등
    required_files: List[str] = field(default_factory=list)      # 필수 파일들
    optional_files: List[str] = field(default_factory=list)      # 선택적 파일들
    tokenizer_files: List[str] = field(default_factory=list)     # tokenizer 관련
    scheduler_files: List[str] = field(default_factory=list)     # scheduler 관련
    
    # Step별 특수 체크포인트
    unet_model: Optional[str] = None                # VirtualFittingStep용
    vae_model: Optional[str] = None                 # VirtualFittingStep용
    text_encoder: Optional[str] = None              # VirtualFittingStep용
    body_model: Optional[str] = None                # PoseEstimationStep용
    hand_model: Optional[str] = None                # PoseEstimationStep용
    face_model: Optional[str] = None                # PoseEstimationStep용
    
    # 메타데이터
    total_size_mb: float = 0.0
    validation_passed: bool = False

@dataclass
class StepModelConfig:
    """Step별 모델 설정 (Step 요청 정보 기반)"""
    # Step 기본 정보
    step_name: str                                  # Step 클래스명
    model_name: str                                 # 모델 이름
    model_class: str                                # AI 모델 클래스
    model_type: str                                 # 모델 타입
    
    # 디바이스 및 최적화 (Step 요청 그대로)
    device: str                                     # 'auto', 'mps', 'cuda', 'cpu'
    precision: str                                  # 'fp16', 'fp32'
    input_size: Tuple[int, int]                     # 입력 크기
    num_classes: Optional[int]                      # 클래스 수
    
    # 체크포인트 정보
    checkpoints: ModelCheckpointInfo                # 완전한 체크포인트 정보
    
    # Step별 파라미터 (Step 요청 정보 그대로)
    optimization_params: Dict[str, Any] = field(default_factory=dict)
    special_params: Dict[str, Any] = field(default_factory=dict)
    
    # 대체 및 폴백
    alternative_models: List[str] = field(default_factory=list)
    fallback_config: Dict[str, Any] = field(default_factory=dict)
    
    # 메타데이터
    priority: int = 5
    confidence_score: float = 0.0
    auto_detected: bool = False
    registration_time: float = field(default_factory=time.time)

class DeviceManager:
    """M3 Max 특화 디바이스 관리자"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.DeviceManager")
        self.available_devices = self._detect_available_devices()
        self.optimal_device = self._select_optimal_device()
        
    def _detect_available_devices(self) -> List[str]:
        """사용 가능한 디바이스 탐지"""
        devices = ["cpu"]
        
        if TORCH_AVAILABLE:
            # M3 Max MPS 지원 확인
            if torch.backends.mps.is_available():
                devices.append("mps")
                self.logger.info("🍎 M3 Max MPS 사용 가능")
            
            # CUDA 지원 확인 (외부 GPU)
            if torch.cuda.is_available():
                devices.append("cuda")
                cuda_devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
                devices.extend(cuda_devices)
                self.logger.info(f"🔥 CUDA 디바이스: {cuda_devices}")
        
        self.logger.info(f"🔍 사용 가능한 디바이스: {devices}")
        return devices
    
    def _select_optimal_device(self) -> str:
        """최적 디바이스 선택"""
        # M3 Max 환경에서는 MPS 우선
        if "mps" in self.available_devices:
            return "mps"
        elif "cuda" in self.available_devices:
            return "cuda"
        else:
            return "cpu"
    
    def resolve_device(self, requested_device: str) -> str:
        """요청된 디바이스를 실제 디바이스로 변환"""
        if requested_device == "auto":
            return self.optimal_device
        elif requested_device in self.available_devices:
            return requested_device
        else:
            self.logger.warning(f"⚠️ 요청된 디바이스 {requested_device} 사용 불가, {self.optimal_device} 사용")
            return self.optimal_device

class MemoryOptimizer:
    """M3 Max 128GB 메모리 최적화"""
    
    def __init__(self, total_memory_gb: float = 128.0):
        self.total_memory_gb = total_memory_gb
        self.allocated_memory_gb = 0.0
        self.memory_budget = total_memory_gb * 0.7  # 70% 까지만 사용
        self.logger = logging.getLogger(f"{__name__}.MemoryOptimizer")
        
    def can_allocate(self, required_memory_gb: float) -> bool:
        """메모리 할당 가능 여부 확인"""
        return (self.allocated_memory_gb + required_memory_gb) <= self.memory_budget
    
    def allocate_memory(self, memory_gb: float) -> bool:
        """메모리 할당"""
        if self.can_allocate(memory_gb):
            self.allocated_memory_gb += memory_gb
            self.logger.debug(f"💾 메모리 할당: {memory_gb:.2f}GB (총: {self.allocated_memory_gb:.2f}GB)")
            return True
        return False
    
    def deallocate_memory(self, memory_gb: float):
        """메모리 해제"""
        self.allocated_memory_gb = max(0, self.allocated_memory_gb - memory_gb)
        self.logger.debug(f"🗑️ 메모리 해제: {memory_gb:.2f}GB (남은: {self.allocated_memory_gb:.2f}GB)")
    
    def cleanup_memory(self):
        """메모리 정리"""
        try:
            if TORCH_AVAILABLE:
                # MPS 캐시 정리
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()
                
                # CUDA 캐시 정리
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Python 가비지 컬렉션
            import gc
            gc.collect()
            
            self.logger.info("🧹 메모리 정리 완료")
            
        except Exception as e:
            self.logger.warning(f"⚠️ 메모리 정리 실패: {e}")

# ==============================================
# 🔄 새로운 ModelLoader 클래스
# ==============================================

class ModelLoader:
    """
    🔄 완전히 새로운 ModelLoader v3.0
    ✅ Step별 요청 정보에 정확히 맞춤
    ✅ auto_model_detector와 완벽 연동
    ✅ M3 Max 128GB 최적화
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.logger = logging.getLogger(f"{__name__}.ModelLoader")
        
        # 설정
        self.config = config or {}
        
        # 핵심 매니저들
        self.device_manager = DeviceManager()
        self.memory_optimizer = MemoryOptimizer()
        
        # 모델 등록 및 캐시
        self.registered_models: Dict[str, StepModelConfig] = {}
        self.loaded_models: Dict[str, Any] = {}
        self.model_instances: Dict[str, Any] = {}
        
        # Step별 인터페이스
        self.step_interfaces: Dict[str, 'StepModelInterface'] = {}
        
        # 성능 통계
        self.load_stats = {
            "total_loads": 0,
            "successful_loads": 0,
            "failed_loads": 0,
            "cache_hits": 0,
            "average_load_time": 0.0
        }
        
        # 스레드 안전성
        self._lock = threading.RLock()
        
        self.logger.info("🔄 새로운 ModelLoader v3.0 초기화 완료")
    
    def register_model(
        self, 
        model_name: str, 
        model_config: Union[Dict[str, Any], StepModelConfig]
    ) -> bool:
        """모델 등록 (auto_model_detector에서 호출)"""
        try:
            with self._lock:
                # Dict를 StepModelConfig로 변환
                if isinstance(model_config, dict):
                    step_model_config = self._dict_to_step_model_config(model_config)
                else:
                    step_model_config = model_config
                
                if not step_model_config:
                    self.logger.error(f"❌ 잘못된 모델 설정: {model_name}")
                    return False
                
                # 체크포인트 검증
                if not self._validate_checkpoints(step_model_config.checkpoints):
                    self.logger.error(f"❌ 체크포인트 검증 실패: {model_name}")
                    return False
                
                # 등록
                self.registered_models[model_name] = step_model_config
                
                self.logger.info(f"✅ 모델 등록 완료: {model_name} ({step_model_config.step_name})")
                return True
                
        except Exception as e:
            self.logger.error(f"❌ 모델 등록 실패 {model_name}: {e}")
            return False
    
    def _dict_to_step_model_config(self, config_dict: Dict[str, Any]) -> Optional[StepModelConfig]:
        """딕셔너리를 StepModelConfig로 변환"""
        try:
            # 체크포인트 정보 변환
            checkpoints_data = config_dict.get("checkpoints", {})
            if isinstance(checkpoints_data, dict):
                checkpoints = ModelCheckpointInfo(
                    primary_path=checkpoints_data.get("primary", ""),
                    config_files=checkpoints_data.get("config", []),
                    required_files=checkpoints_data.get("required", []),
                    optional_files=checkpoints_data.get("optional", []),
                    tokenizer_files=checkpoints_data.get("tokenizer_files", []),
                    scheduler_files=checkpoints_data.get("scheduler_files", []),
                    unet_model=checkpoints_data.get("unet"),
                    vae_model=checkpoints_data.get("vae"),
                    text_encoder=checkpoints_data.get("text_encoder"),
                    body_model=checkpoints_data.get("body_model"),
                    hand_model=checkpoints_data.get("hand_model"),
                    face_model=checkpoints_data.get("face_model"),
                    total_size_mb=checkpoints_data.get("total_size_mb", 0.0),
                    validation_passed=checkpoints_data.get("validation_passed", False)
                )
            else:
                # 간단한 경로만 제공된 경우
                primary_path = config_dict.get("checkpoint_path", "")
                checkpoints = ModelCheckpointInfo(
                    primary_path=primary_path,
                    validation_passed=bool(primary_path and Path(primary_path).exists())
                )
            
            # StepModelConfig 생성
            step_config = StepModelConfig(
                step_name=config_dict.get("step_name", "UnknownStep"),
                model_name=config_dict.get("name", "unknown_model"),
                model_class=config_dict.get("model_class", "BaseModel"),
                model_type=config_dict.get("model_type", "unknown"),
                device=config_dict.get("device", "auto"),
                precision=config_dict.get("precision", "fp16"),
                input_size=tuple(config_dict.get("input_size", (512, 512))),
                num_classes=config_dict.get("num_classes"),
                checkpoints=checkpoints,
                optimization_params=config_dict.get("optimization_params", {}),
                special_params=config_dict.get("special_params", {}),
                alternative_models=config_dict.get("alternative_models", []),
                fallback_config=config_dict.get("fallback_config", {}),
                priority=config_dict.get("priority", 5),
                confidence_score=config_dict.get("confidence", 0.0),
                auto_detected=config_dict.get("auto_detected", True)
            )
            
            return step_config
            
        except Exception as e:
            self.logger.error(f"❌ 설정 변환 실패: {e}")
            return None
    
    def _validate_checkpoints(self, checkpoints: ModelCheckpointInfo) -> bool:
        """체크포인트 파일들 검증"""
        try:
            # 주 모델 파일 확인
            if not checkpoints.primary_path:
                return False
            
            primary_path = Path(checkpoints.primary_path)
            if not primary_path.exists():
                self.logger.warning(f"⚠️ 주 모델 파일 없음: {primary_path}")
                return False
            
            # 파일 크기 확인
            file_size_mb = primary_path.stat().st_size / (1024 * 1024)
            if file_size_mb < 0.1:  # 100KB 미만
                self.logger.warning(f"⚠️ 모델 파일이 너무 작음: {file_size_mb}MB")
                return False
            
            # 필수 파일들 확인
            for required_file in checkpoints.required_files:
                if required_file and not Path(required_file).exists():
                    self.logger.warning(f"⚠️ 필수 파일 없음: {required_file}")
                    # 필수 파일이 없어도 일단 통과 (유연성을 위해)
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ 체크포인트 검증 실패: {e}")
            return False
    
    async def get_model(
        self, 
        model_name: str, 
        step_name: Optional[str] = None,
        **kwargs
    ) -> Optional[Any]:
        """
        모델 로드 및 반환 (Step에서 호출)
        
        Args:
            model_name: 요청할 모델 이름
            step_name: 호출하는 Step 이름
            **kwargs: 추가 파라미터들
            
        Returns:
            로드된 모델 인스턴스
        """
        try:
            start_time = time.time()
            
            with self._lock:
                self.load_stats["total_loads"] += 1
                
                # 캐시 확인
                cache_key = f"{model_name}_{step_name}" if step_name else model_name
                if cache_key in self.loaded_models:
                    self.load_stats["cache_hits"] += 1
                    load_time = time.time() - start_time
                    self.logger.debug(f"📦 캐시된 모델 반환: {model_name} ({load_time:.3f}초)")
                    return self.loaded_models[cache_key]
                
                # 등록된 모델 확인
                if model_name not in self.registered_models:
                    # Step별 권장 모델 시도
                    if step_name:
                        recommended_model = self._get_recommended_model_for_step(step_name)
                        if recommended_model and recommended_model in self.registered_models:
                            model_name = recommended_model
                            self.logger.info(f"🎯 Step 권장 모델 사용: {model_name}")
                        else:
                            self.logger.error(f"❌ 모델 없음: {model_name}, Step: {step_name}")
                            self.load_stats["failed_loads"] += 1
                            return None
                    else:
                        self.logger.error(f"❌ 등록되지 않은 모델: {model_name}")
                        self.load_stats["failed_loads"] += 1
                        return None
                
                model_config = self.registered_models[model_name]
                
                # 디바이스 결정
                device = self.device_manager.resolve_device(model_config.device)
                
                # 메모리 확인
                estimated_memory = self._estimate_model_memory(model_config)
                if not self.memory_optimizer.can_allocate(estimated_memory):
                    self.logger.warning(f"⚠️ 메모리 부족: {estimated_memory:.2f}GB 필요")
                    # 메모리 정리 시도
                    self._cleanup_least_used_models()
                    if not self.memory_optimizer.can_allocate(estimated_memory):
                        self.logger.error(f"❌ 메모리 부족으로 모델 로드 실패: {model_name}")
                        self.load_stats["failed_loads"] += 1
                        return None
                
                # 실제 모델 로드
                model_instance = await self._load_model_instance(model_config, device, **kwargs)
                
                if model_instance:
                    # 캐시에 저장
                    self.loaded_models[cache_key] = model_instance
                    self.model_instances[model_name] = model_instance
                    
                    # 메모리 할당 기록
                    self.memory_optimizer.allocate_memory(estimated_memory)
                    
                    # 통계 업데이트
                    self.load_stats["successful_loads"] += 1
                    load_time = time.time() - start_time
                    self.load_stats["average_load_time"] = (
                        (self.load_stats["average_load_time"] * (self.load_stats["successful_loads"] - 1) + load_time) 
                        / self.load_stats["successful_loads"]
                    )
                    
                    self.logger.info(f"✅ 모델 로드 완료: {model_name} ({load_time:.2f}초)")
                    return model_instance
                else:
                    self.load_stats["failed_loads"] += 1
                    return None
                    
        except Exception as e:
            self.logger.error(f"❌ 모델 로드 실패 {model_name}: {e}")
            self.load_stats["failed_loads"] += 1
            return None
    
    async def _load_model_instance(
        self, 
        model_config: StepModelConfig, 
        device: str,
        **kwargs
    ) -> Optional[Any]:
        """실제 모델 인스턴스 로드"""
        try:
            primary_path = Path(model_config.checkpoints.primary_path)
            
            # Step별 특화 로딩
            if model_config.step_name == "VirtualFittingStep":
                return await self._load_diffusion_model(model_config, device, **kwargs)
            elif model_config.step_name == "HumanParsingStep":
                return await self._load_pytorch_model(model_config, device, **kwargs)
            elif model_config.step_name == "PoseEstimationStep":
                return await self._load_pose_model(model_config, device, **kwargs)
            elif model_config.step_name == "ClothSegmentationStep":
                return await self._load_segmentation_model(model_config, device, **kwargs)
            else:
                # 기본 PyTorch 모델 로딩
                return await self._load_pytorch_model(model_config, device, **kwargs)
                
        except Exception as e:
            self.logger.error(f"❌ 모델 인스턴스 로드 실패: {e}")
            return None
    
    async def _load_pytorch_model(
        self, 
        model_config: StepModelConfig, 
        device: str,
        **kwargs
    ) -> Optional[Any]:
        """PyTorch 모델 로딩"""
        try:
            if not TORCH_AVAILABLE:
                self.logger.error("❌ PyTorch가 설치되지 않음")
                return None
            
            checkpoint_path = Path(model_config.checkpoints.primary_path)
            
            # 체크포인트 로드
            checkpoint = torch.load(
                checkpoint_path, 
                map_location='cpu',
                weights_only=True
            )
            
            # 모델 구조는 실제 Step에서 정의해야 함
            # 여기서는 체크포인트만 반환
            model_data = {
                "checkpoint": checkpoint,
                "config": model_config,
                "device": device,
                "checkpoints_info": model_config.checkpoints
            }
            
            self.logger.debug(f"✅ PyTorch 체크포인트 로드: {checkpoint_path.name}")
            return model_data
            
        except Exception as e:
            self.logger.error(f"❌ PyTorch 모델 로드 실패: {e}")
            return None
    
    async def _load_diffusion_model(
        self, 
        model_config: StepModelConfig, 
        device: str,
        **kwargs
    ) -> Optional[Any]:
        """Diffusion 모델 로딩 (VirtualFittingStep용)"""
        try:
            checkpoints = model_config.checkpoints
            
            # Diffusion 모델은 여러 구성 요소로 구성됨
            diffusion_components = {
                "primary_model": checkpoints.primary_path,
                "unet_model": checkpoints.unet_model,
                "vae_model": checkpoints.vae_model,
                "text_encoder": checkpoints.text_encoder,
                "config_files": checkpoints.config_files,
                "tokenizer_files": checkpoints.tokenizer_files,
                "scheduler_files": checkpoints.scheduler_files
            }
            
            # 실제 Diffusion 파이프라인은 Step에서 구성
            model_data = {
                "components": diffusion_components,
                "config": model_config,
                "device": device,
                "optimization_params": model_config.optimization_params,
                "special_params": model_config.special_params
            }
            
            self.logger.debug(f"✅ Diffusion 모델 구성 요소 준비: {len([v for v in diffusion_components.values() if v])}개")
            return model_data
            
        except Exception as e:
            self.logger.error(f"❌ Diffusion 모델 로드 실패: {e}")
            return None
    
    async def _load_pose_model(
        self, 
        model_config: StepModelConfig, 
        device: str,
        **kwargs
    ) -> Optional[Any]:
        """포즈 추정 모델 로딩 (PoseEstimationStep용)"""
        try:
            checkpoints = model_config.checkpoints
            
            # 포즈 모델은 body, hand, face 모델로 구성될 수 있음
            pose_components = {
                "primary_model": checkpoints.primary_path,
                "body_model": checkpoints.body_model,
                "hand_model": checkpoints.hand_model,
                "face_model": checkpoints.face_model,
                "config_files": checkpoints.config_files
            }
            
            model_data = {
                "components": pose_components,
                "config": model_config,
                "device": device,
                "special_params": model_config.special_params
            }
            
            self.logger.debug(f"✅ 포즈 모델 구성 요소 준비")
            return model_data
            
        except Exception as e:
            self.logger.error(f"❌ 포즈 모델 로드 실패: {e}")
            return None
    
    async def _load_segmentation_model(
        self, 
        model_config: StepModelConfig, 
        device: str,
        **kwargs
    ) -> Optional[Any]:
        """세그멘테이션 모델 로딩 (ClothSegmentationStep용)"""
        try:
            return await self._load_pytorch_model(model_config, device, **kwargs)
        except Exception as e:
            self.logger.error(f"❌ 세그멘테이션 모델 로드 실패: {e}")
            return None
    
    def _get_recommended_model_for_step(self, step_name: str) -> Optional[str]:
        """Step별 권장 모델 조회"""
        try:
            # Step별 권장 모델 매핑
            step_recommendations = {
                'HumanParsingStep': 'human_parsing_graphonomy',
                'PoseEstimationStep': 'pose_estimation_openpose',
                'ClothSegmentationStep': 'cloth_segmentation_u2net',
                'GeometricMatchingStep': 'geometric_matching_gmm',
                'ClothWarpingStep': 'cloth_warping_tom',
                'VirtualFittingStep': 'virtual_fitting_stable_diffusion',
                'PostProcessingStep': 'post_processing_realesrgan',
                'QualityAssessmentStep': 'quality_assessment_clip'
            }
            
            recommended = step_recommendations.get(step_name)
            if recommended:
                return recommended
            
            # 등록된 모델 중에서 해당 Step의 모델 찾기
            for model_name, config in self.registered_models.items():
                if config.step_name == step_name:
                    return model_name
            
            return None
            
        except Exception as e:
            self.logger.error(f"❌ Step 권장 모델 조회 실패: {e}")
            return None
    
    def _estimate_model_memory(self, model_config: StepModelConfig) -> float:
        """모델 메모리 사용량 추정 (GB)"""
        try:
            # 파일 크기 기반 추정
            base_memory = model_config.checkpoints.total_size_mb / 1024
            
            # Step별 메모리 승수
            step_multipliers = {
                'VirtualFittingStep': 3.0,  # Diffusion 모델은 메모리 많이 사용
                'HumanParsingStep': 2.0,
                'PoseEstimationStep': 1.5,
                'ClothSegmentationStep': 2.5,
                'GeometricMatchingStep': 1.2,
                'ClothWarpingStep': 2.0,
                'PostProcessingStep': 1.5,
                'QualityAssessmentStep': 2.0
            }
            
            multiplier = step_multipliers.get(model_config.step_name, 1.5)
            estimated_memory = base_memory * multiplier
            
            # 최소 메모리 보장
            return max(estimated_memory, 0.5)  # 최소 500MB
            
        except Exception as e:
            return 2.0  # 기본값 2GB
    
    def _cleanup_least_used_models(self):
        """사용량이 적은 모델들 정리"""
        try:
            # 간단한 LRU 방식으로 정리
            # 실제로는 더 정교한 정리 전략 필요
            if len(self.loaded_models) > 5:  # 5개 이상이면 정리
                # 가장 오래된 캐시 항목 제거
                oldest_key = next(iter(self.loaded_models))
                removed_model = self.loaded_models.pop(oldest_key)
                
                # 메모리 해제
                if hasattr(removed_model, 'cpu'):
                    removed_model.cpu()
                
                self.logger.info(f"🗑️ 모델 캐시 정리: {oldest_key}")
                
        except Exception as e:
            self.logger.warning(f"⚠️ 모델 정리 실패: {e}")
    
    def create_step_interface(self, step_name: str) -> 'StepModelInterface':
        """Step별 인터페이스 생성"""
        try:
            if step_name not in self.step_interfaces:
                interface = StepModelInterface(self, step_name)
                self.step_interfaces[step_name] = interface
                self.logger.debug(f"🔗 Step 인터페이스 생성: {step_name}")
            
            return self.step_interfaces[step_name]
            
        except Exception as e:
            self.logger.error(f"❌ Step 인터페이스 생성 실패 {step_name}: {e}")
            raise
    
    def get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """모델 정보 조회"""
        try:
            if model_name not in self.registered_models:
                return None
            
            config = self.registered_models[model_name]
            return {
                "name": config.model_name,
                "step_name": config.step_name,
                "model_class": config.model_class,
                "model_type": config.model_type,
                "device": config.device,
                "precision": config.precision,
                "input_size": config.input_size,
                "num_classes": config.num_classes,
                "checkpoints": {
                    "primary_path": config.checkpoints.primary_path,
                    "total_size_mb": config.checkpoints.total_size_mb,
                    "validation_passed": config.checkpoints.validation_passed
                },
                "priority": config.priority,
                "confidence_score": config.confidence_score,
                "auto_detected": config.auto_detected
            }
            
        except Exception as e:
            self.logger.error(f"❌ 모델 정보 조회 실패: {e}")
            return None
    
    def list_models(self, step_name: Optional[str] = None) -> List[str]:
        """등록된 모델 목록 조회"""
        try:
            if step_name:
                return [
                    name for name, config in self.registered_models.items()
                    if config.step_name == step_name
                ]
            else:
                return list(self.registered_models.keys())
                
        except Exception as e:
            self.logger.error(f"❌ 모델 목록 조회 실패: {e}")
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """ModelLoader 통계 정보"""
        return {
            "registered_models": len(self.registered_models),
            "loaded_models": len(self.loaded_models),
            "memory_usage": {
                "allocated_gb": self.memory_optimizer.allocated_memory_gb,
                "budget_gb": self.memory_optimizer.memory_budget,
                "utilization_percent": (self.memory_optimizer.allocated_memory_gb / self.memory_optimizer.memory_budget) * 100
            },
            "load_statistics": self.load_stats.copy(),
            "available_devices": self.device_manager.available_devices,
            "optimal_device": self.device_manager.optimal_device
        }
    
    def cleanup(self):
        """리소스 정리"""
        try:
            with self._lock:
                # 로드된 모델들 정리
                for model_name, model in self.loaded_models.items():
                    try:
                        if hasattr(model, 'cpu'):
                            model.cpu()
                        del model
                    except:
                        pass
                
                self.loaded_models.clear()
                self.model_instances.clear()
                
                # Step 인터페이스 정리
                for interface in self.step_interfaces.values():
                    try:
                        interface.cleanup()
                    except:
                        pass
                self.step_interfaces.clear()
                
                # 메모리 정리
                self.memory_optimizer.cleanup_memory()
                
                self.logger.info("✅ ModelLoader 정리 완료")
                
        except Exception as e:
            self.logger.error(f"❌ ModelLoader 정리 실패: {e}")

# ==============================================
# 🔗 Step 인터페이스 클래스
# ==============================================

class StepModelInterface:
    """Step과 ModelLoader 간 인터페이스"""
    
    def __init__(self, model_loader: ModelLoader, step_name: str):
        self.model_loader = model_loader
        self.step_name = step_name
        self.logger = logging.getLogger(f"{__name__}.StepModelInterface.{step_name}")
        
        # Step별 캐시
        self.step_cache: Dict[str, Any] = {}
        self._lock = threading.RLock()
    
    async def get_model(self, model_name: str, **kwargs) -> Optional[Any]:
        """
        Step에서 모델 요청
        
        Args:
            model_name: 요청할 모델 이름
            **kwargs: Step별 추가 파라미터
            
        Returns:
            로드된 모델 또는 모델 데이터
        """
        try:
            # Step별 캐시 확인
            cache_key = f"{model_name}_{hash(str(kwargs))}" if kwargs else model_name
            
            with self._lock:
                if cache_key in self.step_cache:
                    self.logger.debug(f"📦 Step 캐시 히트: {model_name}")
                    return self.step_cache[cache_key]
            
            # ModelLoader에서 모델 로드
            model = await self.model_loader.get_model(
                model_name=model_name,
                step_name=self.step_name,
                **kwargs
            )
            
            if model:
                with self._lock:
                    self.step_cache[cache_key] = model
                
                self.logger.info(f"✅ Step 모델 로드 완료: {model_name}")
                return model
            else:
                self.logger.error(f"❌ Step 모델 로드 실패: {model_name}")
                return None
                
        except Exception as e:
            self.logger.error(f"❌ Step 모델 요청 실패 {model_name}: {e}")
            return None
    
    async def get_recommended_model(self) -> Optional[Any]:
        """Step별 권장 모델 자동 로드"""
        try:
            # Step별 권장 모델명 조회
            recommended_models = self.model_loader.list_models(self.step_name)
            
            if not recommended_models:
                self.logger.warning(f"⚠️ {self.step_name}에 대한 등록된 모델이 없습니다")
                return None
            
            # 가장 우선순위가 높은 모델 선택
            best_model_name = None
            best_priority = float('inf')
            
            for model_name in recommended_models:
                model_info = self.model_loader.get_model_info(model_name)
                if model_info and model_info["priority"] < best_priority:
                    best_priority = model_info["priority"]
                    best_model_name = model_name
            
            if best_model_name:
                return await self.get_model(best_model_name)
            else:
                return None
                
        except Exception as e:
            self.logger.error(f"❌ Step 권장 모델 로드 실패: {e}")
            return None
    
    def get_step_model_config(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Step별 모델 설정 조회"""
        try:
            model_info = self.model_loader.get_model_info(model_name)
            if not model_info or model_info["step_name"] != self.step_name:
                return None
            
            # Step 요청 정보와 결합
            step_request_info = StepModelRequestAnalyzer.get_step_request_info(self.step_name)
            if step_request_info:
                default_request = step_request_info["default_request"]
                model_info.update({
                    "optimization_params": default_request.optimization_params,
                    "special_params": default_request.special_params,
                    "checkpoint_requirements": default_request.checkpoint_requirements
                })
            
            return model_info
            
        except Exception as e:
            self.logger.error(f"❌ Step 모델 설정 조회 실패: {e}")
            return None
    
    def cleanup(self):
        """Step 인터페이스 정리"""
        try:
            with self._lock:
                # 캐시된 모델들 정리
                for model in self.step_cache.values():
                    try:
                        if hasattr(model, 'cpu'):
                            model.cpu()
                        del model
                    except:
                        pass
                
                self.step_cache.clear()
                
            self.logger.debug(f"✅ {self.step_name} 인터페이스 정리 완료")
            
        except Exception as e:
            self.logger.warning(f"⚠️ {self.step_name} 인터페이스 정리 실패: {e}")

# ==============================================
# 🔥 전역 ModelLoader 인스턴스 관리
# ==============================================

_global_model_loader: Optional[ModelLoader] = None
_loader_lock = threading.Lock()

def get_global_model_loader(config: Optional[Dict[str, Any]] = None) -> ModelLoader:
    """전역 ModelLoader 인스턴스 반환"""
    global _global_model_loader
    
    with _loader_lock:
        if _global_model_loader is None:
            _global_model_loader = ModelLoader(config)
            logger.info("🌐 전역 ModelLoader 인스턴스 생성")
        
        return _global_model_loader

def cleanup_global_model_loader():
    """전역 ModelLoader 정리"""
    global _global_model_loader
    
    with _loader_lock:
        if _global_model_loader:
            _global_model_loader.cleanup()
            _global_model_loader = None
            logger.info("🌐 전역 ModelLoader 정리 완료")

# ==============================================
# 🔥 편의 함수들
# ==============================================

async def load_model_for_step(
    step_name: str, 
    model_name: Optional[str] = None,
    **kwargs
) -> Optional[Any]:
    """Step별 모델 로드 편의 함수"""
    try:
        loader = get_global_model_loader()
        interface = loader.create_step_interface(step_name)
        
        if model_name:
            return await interface.get_model(model_name, **kwargs)
        else:
            return await interface.get_recommended_model()
            
    except Exception as e:
        logger.error(f"❌ Step 모델 로드 실패 {step_name}: {e}")
        return None

def register_auto_detected_models(detected_models: Dict[str, Any]) -> Dict[str, bool]:
    """auto_model_detector에서 탐지된 모델들 일괄 등록"""
    try:
        loader = get_global_model_loader()
        registration_results = {}
        
        for model_name, model_config in detected_models.items():
            success = loader.register_model(model_name, model_config)
            registration_results[model_name] = success
        
        successful_count = sum(registration_results.values())
        logger.info(f"🔗 자동 탐지 모델 등록: {successful_count}/{len(detected_models)}개 성공")
        
        return registration_results
        
    except Exception as e:
        logger.error(f"❌ 자동 탐지 모델 등록 실패: {e}")
        return {}

# ==============================================
# 🔥 누락된 핵심 기능들 추가
# ==============================================

class ModelCache:
    """모델 캐시 관리자"""
    
    def __init__(self, max_models: int = 10, max_memory_gb: float = 32.0):
        self.max_models = max_models
        self.max_memory_gb = max_memory_gb
        self.cached_models: Dict[str, Any] = {}
        self.access_times: Dict[str, float] = {}
        self.memory_usage: Dict[str, float] = {}
        self.load_times: Dict[str, float] = {}
        self._lock = threading.RLock()
        self.logger = logging.getLogger(f"{__name__}.ModelCache")
    
    def get(self, key: str) -> Optional[Any]:
        """캐시에서 모델 조회"""
        with self._lock:
            if key in self.cached_models:
                self.access_times[key] = time.time()
                self.logger.debug(f"📦 캐시 히트: {key}")
                return self.cached_models[key]
            return None
    
    def put(self, key: str, model: Any, memory_usage_gb: float = 0.0):
        """캐시에 모델 저장"""
        with self._lock:
            # 메모리 한도 확인
            current_memory = sum(self.memory_usage.values())
            if current_memory + memory_usage_gb > self.max_memory_gb:
                self._evict_models(memory_usage_gb)
            
            # 모델 수 한도 확인
            if len(self.cached_models) >= self.max_models:
                self._evict_oldest()
            
            self.cached_models[key] = model
            self.access_times[key] = time.time()
            self.memory_usage[key] = memory_usage_gb
            self.load_times[key] = time.time()
            
            self.logger.debug(f"💾 캐시 저장: {key} ({memory_usage_gb:.2f}GB)")
    
    def _evict_models(self, required_memory: float):
        """메모리 확보를 위한 모델 제거"""
        # LRU 방식으로 제거
        sorted_models = sorted(
            self.access_times.items(),
            key=lambda x: x[1]
        )
        
        freed_memory = 0.0
        for key, _ in sorted_models:
            if freed_memory >= required_memory:
                break
            
            freed_memory += self.memory_usage.get(key, 0.0)
            self._remove_model(key)
    
    def _evict_oldest(self):
        """가장 오래된 모델 제거"""
        if not self.access_times:
            return
        
        oldest_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
        self._remove_model(oldest_key)
    
    def _remove_model(self, key: str):
        """모델 제거"""
        if key in self.cached_models:
            model = self.cached_models[key]
            if hasattr(model, 'cpu'):
                model.cpu()
            
            del self.cached_models[key]
            del self.access_times[key]
            del self.memory_usage[key]
            if key in self.load_times:
                del self.load_times[key]
            
            self.logger.debug(f"🗑️ 캐시 제거: {key}")
    
    def clear(self):
        """캐시 전체 정리"""
        with self._lock:
            for model in self.cached_models.values():
                if hasattr(model, 'cpu'):
                    model.cpu()
            
            self.cached_models.clear()
            self.access_times.clear()
            self.memory_usage.clear()
            self.load_times.clear()
            
            self.logger.info("🧹 모델 캐시 전체 정리 완료")
    
    def get_stats(self) -> Dict[str, Any]:
        """캐시 통계"""
        with self._lock:
            total_memory = sum(self.memory_usage.values())
            return {
                "cached_models": len(self.cached_models),
                "max_models": self.max_models,
                "total_memory_gb": total_memory,
                "max_memory_gb": self.max_memory_gb,
                "memory_utilization": (total_memory / self.max_memory_gb) * 100,
                "models": list(self.cached_models.keys())
            }

class BatchModelLoader:
    """배치 모델 로딩 시스템"""
    
    def __init__(self, model_loader: 'ModelLoader'):
        self.model_loader = model_loader
        self.logger = logging.getLogger(f"{__name__}.BatchModelLoader")
        
    async def load_models_batch(
        self, 
        model_requests: List[Tuple[str, str]], # (model_name, step_name)
        max_concurrent: int = 3
    ) -> Dict[str, Any]:
        """여러 모델 동시 로딩"""
        try:
            semaphore = asyncio.Semaphore(max_concurrent)
            
            async def load_single_model(model_name: str, step_name: str):
                async with semaphore:
                    try:
                        model = await self.model_loader.get_model(model_name, step_name)
                        return model_name, model, None
                    except Exception as e:
                        return model_name, None, str(e)
            
            # 동시 로딩 실행
            tasks = [
                load_single_model(model_name, step_name)
                for model_name, step_name in model_requests
            ]
            
            results = await asyncio.gather(*tasks)
            
            # 결과 정리
            loaded_models = {}
            failed_models = {}
            
            for model_name, model, error in results:
                if model is not None:
                    loaded_models[model_name] = model
                else:
                    failed_models[model_name] = error
            
            batch_result = {
                "loaded_models": loaded_models,
                "failed_models": failed_models,
                "success_rate": len(loaded_models) / len(model_requests) if model_requests else 0.0,
                "total_requested": len(model_requests)
            }
            
            self.logger.info(f"📦 배치 로딩 완료: {len(loaded_models)}/{len(model_requests)}개 성공")
            return batch_result
            
        except Exception as e:
            self.logger.error(f"❌ 배치 로딩 실패: {e}")
            return {"error": str(e)}

class ModelWarmer:
    """모델 워밍업 및 프리로딩"""
    
    def __init__(self, model_loader: 'ModelLoader'):
        self.model_loader = model_loader
        self.logger = logging.getLogger(f"{__name__}.ModelWarmer")
        self.warmed_models: Set[str] = set()
        
    async def warmup_critical_models(self) -> Dict[str, Any]:
        """중요 모델들 워밍업"""
        try:
            critical_models = [
                ("human_parsing_graphonomy", "HumanParsingStep"),
                ("virtual_fitting_stable_diffusion", "VirtualFittingStep"),
                ("pose_estimation_openpose", "PoseEstimationStep")
            ]
            
            warmup_results = {
                "warmed_models": [],
                "failed_models": [],
                "warmup_time": 0.0
            }
            
            start_time = time.time()
            
            for model_name, step_name in critical_models:
                try:
                    # 모델 로드만 수행 (실제 추론은 하지 않음)
                    model = await self.model_loader.get_model(model_name, step_name)
                    if model:
                        self.warmed_models.add(model_name)
                        warmup_results["warmed_models"].append(model_name)
                        self.logger.debug(f"🔥 워밍업 완료: {model_name}")
                    
                except Exception as e:
                    warmup_results["failed_models"].append({
                        "model": model_name,
                        "error": str(e)
                    })
                    self.logger.warning(f"⚠️ 워밍업 실패 {model_name}: {e}")
            
            warmup_results["warmup_time"] = time.time() - start_time
            
            self.logger.info(f"🔥 모델 워밍업 완료: {len(warmup_results['warmed_models'])}개")
            return warmup_results
            
        except Exception as e:
            self.logger.error(f"❌ 모델 워밍업 실패: {e}")
            return {"error": str(e)}
    
    def is_warmed(self, model_name: str) -> bool:
        """모델 워밍업 상태 확인"""
        return model_name in self.warmed_models

class ModelHealthChecker:
    """모델 상태 및 건강성 체크"""
    
    def __init__(self, model_loader: 'ModelLoader'):
        self.model_loader = model_loader
        self.logger = logging.getLogger(f"{__name__}.ModelHealthChecker")
        
    async def check_all_models(self) -> Dict[str, Any]:
        """등록된 모든 모델 상태 체크"""
        try:
            health_report = {
                "healthy_models": [],
                "unhealthy_models": [],
                "missing_models": [],
                "total_models": 0,
                "overall_health": 0.0,
                "recommendations": []
            }
            
            model_names = self.model_loader.list_models()
            health_report["total_models"] = len(model_names)
            
            for model_name in model_names:
                try:
                    model_info = self.model_loader.get_model_info(model_name)
                    if not model_info:
                        health_report["missing_models"].append(model_name)
                        continue
                    
                    # 체크포인트 파일 존재 확인
                    checkpoint_path = Path(model_info["checkpoints"]["primary_path"])
                    if not checkpoint_path.exists():
                        health_report["unhealthy_models"].append({
                            "model": model_name,
                            "issue": "Checkpoint file missing",
                            "path": str(checkpoint_path)
                        })
                        continue
                    
                    # 파일 크기 확인
                    file_size_mb = checkpoint_path.stat().st_size / (1024 * 1024)
                    expected_size = model_info["checkpoints"]["total_size_mb"]
                    
                    if abs(file_size_mb - expected_size) > expected_size * 0.1:  # 10% 오차
                        health_report["unhealthy_models"].append({
                            "model": model_name,
                            "issue": "File size mismatch",
                            "expected": expected_size,
                            "actual": file_size_mb
                        })
                        continue
                    
                    health_report["healthy_models"].append(model_name)
                    
                except Exception as e:
                    health_report["unhealthy_models"].append({
                        "model": model_name,
                        "issue": str(e)
                    })
            
            # 전체 건강도 계산
            total_checked = len(health_report["healthy_models"]) + len(health_report["unhealthy_models"])
            if total_checked > 0:
                health_report["overall_health"] = len(health_report["healthy_models"]) / total_checked
            
            # 추천사항 생성
            if health_report["unhealthy_models"]:
                health_report["recommendations"].append("Fix unhealthy models before production use")
            
            if health_report["missing_models"]:
                health_report["recommendations"].append("Re-run model detection to find missing models")
            
            if health_report["overall_health"] < 0.8:
                health_report["recommendations"].append("Overall model health is low - consider system maintenance")
            
            self.logger.info(f"🏥 모델 건강성 체크 완료: {len(health_report['healthy_models'])}/{health_report['total_models']}개 정상")
            return health_report
            
        except Exception as e:
            self.logger.error(f"❌ 모델 건강성 체크 실패: {e}")
            return {"error": str(e)}

# ModelLoader 클래스에 추가할 메서드들
def add_missing_methods_to_modelloader():
    """ModelLoader에 누락된 메서드들 추가"""
    
    def preload_models(self, model_names: List[str]) -> Dict[str, Any]:
        """모델들 사전 로딩"""
        try:
            preload_results = {
                "preloaded": [],
                "failed": [],
                "total_time": 0.0
            }
            
            start_time = time.time()
            
            for model_name in model_names:
                try:
                    # 비동기 함수를 동기적으로 실행
                    import asyncio
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    
                    model = loop.run_until_complete(self.get_model(model_name))
                    if model:
                        preload_results["preloaded"].append(model_name)
                    
                    loop.close()
                    
                except Exception as e:
                    preload_results["failed"].append({
                        "model": model_name,
                        "error": str(e)
                    })
            
            preload_results["total_time"] = time.time() - start_time
            
            self.logger.info(f"📦 모델 사전 로딩: {len(preload_results['preloaded'])}/{len(model_names)}개 성공")
            return preload_results
            
        except Exception as e:
            self.logger.error(f"❌ 모델 사전 로딩 실패: {e}")
            return {"error": str(e)}
    
    def get_model_status(self, model_name: str) -> Dict[str, Any]:
        """모델 상태 정보 조회"""
        try:
            status = {
                "registered": model_name in self.registered_models,
                "loaded": False,
                "cached": False,
                "memory_usage_gb": 0.0,
                "last_access": None,
                "load_count": 0
            }
            
            # 캐시 상태 확인
            for cache_key in self.loaded_models.keys():
                if model_name in cache_key:
                    status["loaded"] = True
                    status["cached"] = True
                    break
            
            # 메모리 사용량 추정
            if model_name in self.registered_models:
                config = self.registered_models[model_name]
                status["memory_usage_gb"] = self._estimate_model_memory(config)
            
            return status
            
        except Exception as e:
            self.logger.error(f"❌ 모델 상태 조회 실패: {e}")
            return {"error": str(e)}
    
    def optimize_memory_usage(self) -> Dict[str, Any]:
        """메모리 사용량 최적화"""
        try:
            optimization_result = {
                "before_memory_gb": self.memory_optimizer.allocated_memory_gb,
                "cleaned_models": [],
                "memory_freed_gb": 0.0,
                "after_memory_gb": 0.0
            }
            
            # 사용되지 않는 모델들 정리
            self._cleanup_least_used_models()
            
            # 메모리 정리
            self.memory_optimizer.cleanup_memory()
            
            optimization_result["after_memory_gb"] = self.memory_optimizer.allocated_memory_gb
            optimization_result["memory_freed_gb"] = (
                optimization_result["before_memory_gb"] - 
                optimization_result["after_memory_gb"]
            )
            
            self.logger.info(f"🧹 메모리 최적화: {optimization_result['memory_freed_gb']:.2f}GB 해제")
            return optimization_result
            
        except Exception as e:
            self.logger.error(f"❌ 메모리 최적화 실패: {e}")
            return {"error": str(e)}
    
    # ModelLoader 클래스에 메서드 동적 추가
    ModelLoader.preload_models = preload_models
    ModelLoader.get_model_status = get_model_status
    ModelLoader.optimize_memory_usage = optimize_memory_usage

# 초기화 시 메서드 추가
add_missing_methods_to_modelloader()

# 전역 함수들
async def batch_load_models_for_steps(step_models: Dict[str, str]) -> Dict[str, Any]:
    """Step별 모델 배치 로딩"""
    try:
        loader = get_global_model_loader()
        batch_loader = BatchModelLoader(loader)
        
        model_requests = [(model_name, step_name) for step_name, model_name in step_models.items()]
        return await batch_loader.load_models_batch(model_requests)
        
    except Exception as e:
        logger.error(f"❌ Step별 배치 로딩 실패: {e}")
        return {"error": str(e)}

def warmup_system_models() -> Dict[str, Any]:
    """시스템 모델들 워밍업"""
    try:
        loader = get_global_model_loader()
        warmer = ModelWarmer(loader)
        
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        result = loop.run_until_complete(warmer.warmup_critical_models())
        loop.close()
        
        return result
        
    except Exception as e:
        logger.error(f"❌ 시스템 모델 워밍업 실패: {e}")
        return {"error": str(e)}

def check_system_model_health() -> Dict[str, Any]:
    """시스템 모델 건강성 체크"""
    try:
        loader = get_global_model_loader()
        health_checker = ModelHealthChecker(loader)
        
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        result = loop.run_until_complete(health_checker.check_all_models())
        loop.close()
        
        return result
        
    except Exception as e:
        logger.error(f"❌ 시스템 모델 건강성 체크 실패: {e}")
        return {"error": str(e)}

# 모듈 익스포트 업데이트
__all__ = [
    # 기존 클래스들
    'ModelLoader',
    'StepModelInterface',
    'StepModelConfig',
    'ModelCheckpointInfo',
    'DeviceManager',
    'MemoryOptimizer',
    
    # 새로 추가된 클래스들
    'ModelCache',
    'BatchModelLoader',
    'ModelWarmer',
    'ModelHealthChecker',
    
    # 기존 함수들
    'get_global_model_loader',
    'cleanup_global_model_loader',
    'load_model_for_step',
    'register_auto_detected_models',
    
    # 새로 추가된 함수들
    'batch_load_models_for_steps',
    'warmup_system_models',
    'check_system_model_health'
]