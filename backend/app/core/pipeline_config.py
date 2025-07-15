"""
🎯 MyCloset AI 파이프라인 설정 관리 시스템 - 모든 기능 완전 복원
- 원본의 모든 설정 및 최적화 기능
- M3 Max 최적화 지원
- 환경별 설정 분리
- 타입 안전성 보장
- 모든 헬퍼 함수들 포함

파일 경로: backend/app/core/pipeline_config.py
"""

import os
import json
import logging
import platform
import subprocess
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
from functools import lru_cache
from abc import ABC, abstractmethod
import torch
from enum import Enum
from dataclasses import dataclass, field

# GPU 설정 안전한 import
try:
    from .gpu_config import gpu_config, DEVICE, DEVICE_INFO
except ImportError:
    logger = logging.getLogger(__name__)
    logger.warning("gpu_config import 실패 - 기본값 사용")
    
    # 기본값 설정
    DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    DEVICE_INFO = {"device": DEVICE, "available": True}
    
    class DummyGPUConfig:
        def __init__(self):
            self.device = DEVICE
            self.device_type = "auto"
    
    gpu_config = DummyGPUConfig()

logger = logging.getLogger(__name__)

# ===============================================================
# 🎯 핵심 Enum 및 설정 클래스들 (원본 완전 복원)
# ===============================================================

class DeviceType(Enum):
    """디바이스 타입 열거형"""
    AUTO = "auto"
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"
    APPLE_SILICON = "apple_silicon"
    NVIDIA = "nvidia"
    INTEL = "intel"

class QualityLevel(Enum):
    """품질 레벨 열거형"""
    FAST = "fast"
    BALANCED = "balanced"
    HIGH = "high"
    ULTRA = "ultra"

class PipelineMode(Enum):
    """파이프라인 모드 열거형"""
    SIMULATION = "simulation"
    PRODUCTION = "production"
    HYBRID = "hybrid"
    DEVELOPMENT = "development"

@dataclass
class SystemInfo:
    """시스템 정보 데이터클래스"""
    device: str
    device_type: str
    memory_gb: float
    is_m3_max: bool
    optimization_enabled: bool
    torch_version: str = field(default_factory=lambda: torch.__version__)
    platform_system: str = field(default_factory=platform.system)
    
    def __post_init__(self):
        """초기화 후 추가 검증"""
        if self.memory_gb < 1:
            raise ValueError("메모리는 최소 1GB 이상이어야 합니다")

@dataclass 
class StepConfig:
    """단계별 설정 데이터클래스"""
    device: str
    device_type: str
    memory_gb: float
    is_m3_max: bool
    optimization_enabled: bool
    quality_level: str
    enable_caching: bool = True
    batch_size: int = 1
    timeout_seconds: int = 300
    config_data: Dict[str, Any] = field(default_factory=dict)

# ===============================================================
# 🎯 최적 생성자 베이스 클래스 (완전 복원)
# ===============================================================

class OptimalConfigBase(ABC):
    """
    🎯 최적화된 설정 베이스 클래스
    모든 Step 클래스와 Manager 클래스의 공통 인터페이스
    """

    def __init__(
        self,
        device: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """
        ✅ 최적 생성자 - 모든 MyCloset AI 컴포넌트 호환
        
        Args:
            device: 사용할 디바이스 (None=자동감지)
            config: 설정 딕셔너리
            **kwargs: 확장 파라미터들
        """
        # 1. 지능적 디바이스 자동 감지
        self.device = self._auto_detect_device(device)
        
        # 2. 기본 설정
        self.config = config or {}
        self.class_name = self.__class__.__name__
        self.logger = logging.getLogger(f"mycloset.{self.class_name}")
        
        # 3. 표준 시스템 파라미터 추출
        self.device_type = kwargs.get('device_type', self._detect_device_type())
        self.memory_gb = kwargs.get('memory_gb', self._detect_memory_gb())
        self.is_m3_max = kwargs.get('is_m3_max', self._detect_m3_max())
        self.optimization_enabled = kwargs.get('optimization_enabled', True)
        self.quality_level = kwargs.get('quality_level', 'balanced')
        
        # 4. 클래스별 특화 파라미터를 config에 병합
        self._merge_class_specific_config(kwargs)
        
        # 5. 상태 초기화
        self.is_initialized = False
        
        self.logger.info(f"🎯 {self.class_name} 최적 생성자 초기화 - 디바이스: {self.device}")

    def _auto_detect_device(self, preferred_device: Optional[str]) -> str:
        """💡 지능적 디바이스 자동 감지"""
        if preferred_device and preferred_device != "auto":
            return preferred_device

        try:
            if torch.backends.mps.is_available():
                return 'mps'  # M3 Max 우선
            elif torch.cuda.is_available():
                return 'cuda'  # NVIDIA GPU
            else:
                return 'cpu'  # 폴백
        except ImportError:
            return 'cpu'

    def _detect_device_type(self) -> str:
        """디바이스 타입 감지"""
        if self.device == 'mps':
            return 'apple_silicon'
        elif self.device == 'cuda':
            return 'nvidia'
        else:
            return 'intel'

    def _detect_memory_gb(self) -> float:
        """메모리 용량 감지"""
        try:
            import psutil
            return round(psutil.virtual_memory().total / (1024**3), 1)
        except ImportError:
            # M3 Max 기본값 또는 추정값
            if self._detect_m3_max():
                return 128.0
            return 16.0

    def _detect_m3_max(self) -> bool:
        """🍎 M3 Max 칩 자동 감지"""
        try:
            if platform.system() == 'Darwin':
                result = subprocess.run(
                    ['sysctl', '-n', 'machdep.cpu.brand_string'], 
                    capture_output=True, text=True, timeout=5
                )
                chip_info = result.stdout.strip()
                return 'M3' in chip_info and ('Max' in chip_info)
        except:
            pass
        return False

    def _merge_class_specific_config(self, kwargs: Dict[str, Any]):
        """⚙️ 클래스별 특화 설정 병합"""
        system_params = {
            'device_type', 'memory_gb', 'is_m3_max', 
            'optimization_enabled', 'quality_level'
        }

        for key, value in kwargs.items():
            if key not in system_params:
                self.config[key] = value

    def get_system_info(self) -> SystemInfo:
        """🔍 시스템 정보 반환"""
        return SystemInfo(
            device=self.device,
            device_type=self.device_type,
            memory_gb=self.memory_gb,
            is_m3_max=self.is_m3_max,
            optimization_enabled=self.optimization_enabled
        )

    def get_step_config(self) -> StepConfig:
        """📋 단계별 설정 반환"""
        return StepConfig(
            device=self.device,
            device_type=self.device_type,
            memory_gb=self.memory_gb,
            is_m3_max=self.is_m3_max,
            optimization_enabled=self.optimization_enabled,
            quality_level=self.quality_level,
            config_data=self.config.copy()
        )

# ===============================================================
# 🎯 파이프라인 설정 관리자 (모든 기능 완전 복원)
# ===============================================================

class PipelineConfig(OptimalConfigBase):
    """
    🎯 MyCloset AI 8단계 파이프라인 설정 관리자
    - 모든 Step 클래스와 Manager 클래스의 설정 제공
    - 환경별 설정 분리
    - M3 Max 최적화
    - 원본의 모든 기능 완전 복원
    """
    
    def __init__(
        self, 
        device: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        config_path: Optional[str] = None,
        quality_level: str = "balanced",
        mode: Union[str, PipelineMode] = PipelineMode.PRODUCTION,
        **kwargs
    ):
        """
        ✅ 최적 생성자 - PipelineConfig용
        """
        # kwargs에서 품질 레벨 덮어쓰기 확인
        if 'quality_level_override' in kwargs:
            quality_level = kwargs.pop('quality_level_override')
        
        # 부모 클래스 초기화
        super().__init__(
            device=device,
            config=config,
            quality_level=quality_level,
            **kwargs
        )
        
        # PipelineConfig 특화 속성들
        self.config_path = config_path or kwargs.get('config_path')
        self.mode = mode if isinstance(mode, PipelineMode) else PipelineMode(mode)
        self.device_info = DEVICE_INFO
        
        # 기본 설정 로드 (최적 생성자 패턴과 통합)
        self.config = self._load_default_config_optimal()
        
        # 외부 설정 파일 로드 (있는 경우)
        if self.config_path and os.path.exists(self.config_path):
            self._load_external_config(self.config_path)
        
        # 환경변수 기반 오버라이드
        self._apply_environment_overrides()
        
        # 디바이스별 최적화 적용
        self._apply_device_optimizations()
        
        # 품질 레벨 적용
        self._apply_quality_preset(self.quality_level)
        
        # 초기화 완료
        self.is_initialized = True
        
        logger.info(f"🔧 최적 생성자 패턴 파이프라인 설정 완료 - 품질: {quality_level}, 디바이스: {self.device}")
        logger.info(f"💻 시스템: {self.device_type}, 메모리: {self.memory_gb}GB, M3 Max: {'✅' if self.is_m3_max else '❌'}")

    def _load_default_config_optimal(self) -> Dict[str, Any]:
        """✅ 최적 생성자 패턴과 통합된 기본 설정 로드 (완전 복원)"""
        
        # kwargs에서 설정된 파라미터들 활용
        enable_caching = self.config.get('enable_caching', True)
        enable_parallel = self.config.get('enable_parallel', True)
        memory_optimization = self.config.get('memory_optimization', True)
        max_concurrent_requests = self.config.get('max_concurrent_requests', 4)
        timeout_seconds = self.config.get('timeout_seconds', 300)
        enable_intermediate_saving = self.config.get('enable_intermediate_saving', False)
        auto_retry = self.config.get('auto_retry', True)
        max_retries = self.config.get('max_retries', 3)
        
        return {
            # 전역 파이프라인 설정 (최적 생성자 패턴 통합)
            "pipeline": {
                "name": "mycloset_virtual_fitting",
                "version": "4.0.0-optimal",
                "constructor_pattern": "optimal",
                "quality_level": self.quality_level,
                "processing_mode": "complete",  # fast, balanced, complete
                "enable_optimization": self.optimization_enabled,
                "enable_caching": enable_caching,
                "enable_parallel": enable_parallel,
                "memory_optimization": memory_optimization,
                "max_concurrent_requests": max_concurrent_requests,
                "timeout_seconds": timeout_seconds,
                "enable_intermediate_saving": enable_intermediate_saving,
                "auto_retry": auto_retry,
                "max_retries": max_retries
            },
            
            # 시스템 정보 (최적 생성자 패턴)
            "system": {
                "device": self.device,
                "device_type": self.device_type,
                "memory_gb": self.memory_gb,
                "is_m3_max": self.is_m3_max,
                "optimization_enabled": self.optimization_enabled,
                "constructor_pattern": "optimal"
            },
            
            # 이미지 처리 설정
            "image": {
                "input_size": (512, 512),
                "output_size": (512, 512),
                "max_resolution": 1024,
                "supported_formats": ["jpg", "jpeg", "png", "webp"],
                "quality": 95,
                "preprocessing": {
                    "normalize": True,
                    "resize_mode": "lanczos",
                    "center_crop": True,
                    "background_removal": True
                }
            },
            
            # 8단계 파이프라인 개별 설정 (완전 복원)
            "steps": {
                # 1단계: 인체 파싱 (Human Parsing)
                "human_parsing": {
                    "model_name": "graphonomy",
                    "model_path": "app/ai_pipeline/models/ai_models/graphonomy",
                    "num_classes": 20,
                    "confidence_threshold": 0.7,
                    "input_size": (512, 512),
                    "batch_size": 1,
                    "cache_enabled": enable_caching,
                    "use_coreml": self.is_m3_max,
                    "enable_quantization": self.optimization_enabled,
                    "preprocessing": {
                        "normalize": True,
                        "mean": [0.485, 0.456, 0.406],
                        "std": [0.229, 0.224, 0.225]
                    },
                    "postprocessing": {
                        "morphology_cleanup": True,
                        "smooth_edges": True,
                        "fill_holes": True
                    }
                },
                
                # 2단계: 포즈 추정 (Pose Estimation)
                "pose_estimation": {
                    "model_name": "mediapipe",
                    "model_complexity": 2,
                    "min_detection_confidence": 0.5,
                    "min_tracking_confidence": 0.5,
                    "static_image_mode": True,
                    "enable_segmentation": True,
                    "smooth_landmarks": True,
                    "keypoints_format": "openpose_18",
                    "fallback_models": ["openpose", "hrnet"],
                    "use_gpu": self.device != 'cpu',
                    "pose_validation": {
                        "min_keypoints": 8,
                        "visibility_threshold": 0.3,
                        "symmetry_check": True
                    }
                },
                
                # 3단계: 의류 세그멘테이션 (Cloth Segmentation)
                "cloth_segmentation": {
                    "model_name": "u2net",
                    "model_path": "app/ai_pipeline/models/ai_models/u2net",
                    "fallback_method": "rembg",
                    "background_removal": True,
                    "edge_refinement": True,
                    "background_threshold": 0.5,
                    "post_process": True,
                    "refine_edges": True,
                    "post_processing": {
                        "morphology_enabled": True,
                        "gaussian_blur": True,
                        "edge_smoothing": True,
                        "noise_removal": True
                    },
                    "quality_assessment": {
                        "enable": True,
                        "min_quality": 0.6,
                        "auto_retry": auto_retry
                    }
                },
                
                # 4단계: 기하학적 매칭 (Geometric Matching)
                "geometric_matching": {
                    "algorithm": "tps_hybrid",  # tps, affine, tps_hybrid
                    "num_control_points": 20,
                    "regularization": 0.001,
                    "matching_method": "hungarian",
                    "tps_points": 25,
                    "matching_threshold": 0.8,
                    "use_advanced_matching": True,
                    "keypoint_extraction": {
                        "method": "contour_based",
                        "num_points": 50,
                        "adaptive_sampling": True
                    },
                    "validation": {
                        "min_matched_points": 4,
                        "outlier_threshold": 2.0,
                        "quality_threshold": 0.7
                    }
                },
                
                # 5단계: 옷 워핑 (Cloth Warping)
                "cloth_warping": {
                    "physics_enabled": True,
                    "fabric_simulation": True,
                    "deformation_strength": 0.8,
                    "wrinkle_simulation": True,
                    "warping_method": "tps",
                    "optimization_level": "high",
                    "fabric_properties": {
                        "cotton": {"stiffness": 0.6, "elasticity": 0.3, "thickness": 0.5},
                        "denim": {"stiffness": 0.9, "elasticity": 0.1, "thickness": 0.8},
                        "silk": {"stiffness": 0.2, "elasticity": 0.4, "thickness": 0.2},
                        "wool": {"stiffness": 0.7, "elasticity": 0.2, "thickness": 0.7},
                        "polyester": {"stiffness": 0.4, "elasticity": 0.6, "thickness": 0.3}
                    },
                    "simulation_steps": 50,
                    "convergence_threshold": 0.001
                },
                
                # 6단계: 가상 피팅 (Virtual Fitting)
                "virtual_fitting": {
                    "model_name": "hr_viton",
                    "model_path": "app/ai_pipeline/models/ai_models/hr_viton",
                    "guidance_scale": 7.5,
                    "num_inference_steps": 50,
                    "strength": 0.8,
                    "eta": 0.0,
                    "composition_method": "neural_blending",
                    "fallback_method": "traditional_blending",
                    "blending_method": "poisson",
                    "seamless_cloning": True,
                    "color_transfer": True,
                    "quality_enhancement": {
                        "color_matching": True,
                        "lighting_adjustment": True,
                        "texture_preservation": True,
                        "edge_smoothing": True
                    }
                },
                
                # 7단계: 후처리 (Post Processing)
                "post_processing": {
                    "enable_super_resolution": self.optimization_enabled,
                    "enhance_faces": True,
                    "color_correction": True,
                    "noise_reduction": True,
                    "super_resolution": {
                        "enabled": self.optimization_enabled,
                        "model": "real_esrgan",
                        "scale_factor": 2,
                        "model_path": "app/ai_pipeline/models/ai_models/real_esrgan"
                    },
                    "face_enhancement": {
                        "enabled": True,
                        "model": "gfpgan",
                        "strength": 0.8,
                        "model_path": "app/ai_pipeline/models/ai_models/gfpgan"
                    },
                    "color_correction": {
                        "enabled": True,
                        "method": "histogram_matching",
                        "strength": 0.6
                    },
                    "noise_reduction": {
                        "enabled": True,
                        "method": "bilateral_filter",
                        "strength": 0.7
                    },
                    "edge_enhancement": {
                        "enabled": True,
                        "method": "unsharp_mask",
                        "strength": 0.5
                    }
                },
                
                # 8단계: 품질 평가 (Quality Assessment)
                "quality_assessment": {
                    "metrics": ["ssim", "lpips", "fid", "is"],
                    "quality_threshold": 0.7,
                    "comprehensive_analysis": True,
                    "generate_suggestions": True,
                    "enable_detailed_analysis": True,
                    "perceptual_metrics": True,
                    "technical_metrics": True,
                    "benchmarking": {
                        "enabled": False,
                        "reference_dataset": None,
                        "save_results": False
                    }
                }
            },
            
            # 모델 경로 설정
            "model_paths": {
                "base_dir": "app/ai_pipeline/models/ai_models",
                "cache_dir": "app/ai_pipeline/cache",
                "checkpoints": {
                    "graphonomy": "graphonomy/checkpoints/graphonomy.pth",
                    "hr_viton": "hr_viton/checkpoints/hr_viton.pth",
                    "u2net": "u2net/checkpoints/u2net.pth",
                    "real_esrgan": "real_esrgan/checkpoints/RealESRGAN_x4plus.pth",
                    "gfpgan": "gfpgan/checkpoints/GFPGANv1.4.pth",
                    "openpose": "openpose/checkpoints/pose_iter_440000.caffemodel"
                }
            },
            
            # 성능 최적화 설정 (최적 생성자 패턴 통합)
            "optimization": {
                "device": self.device,
                "device_type": self.device_type,
                "memory_gb": self.memory_gb,
                "is_m3_max": self.is_m3_max,
                "optimization_enabled": self.optimization_enabled,
                "mixed_precision": self.optimization_enabled,
                "gradient_checkpointing": False,
                "memory_efficient_attention": True,
                "compile_models": False,  # PyTorch 2.0 compile
                "constructor_pattern": "optimal",
                "batch_processing": {
                    "enabled": enable_parallel,
                    "max_batch_size": 4 if self.device != 'cpu' else 1,
                    "dynamic_batching": self.optimization_enabled
                },
                "caching": {
                    "enabled": enable_caching,
                    "ttl": 3600,  # 1시간
                    "max_size": "2GB",
                    "cache_intermediate": enable_intermediate_saving
                }
            },
            
            # 메모리 관리 설정 (최적 생성자 패턴 통합)
            "memory": {
                "max_memory_usage": f"{min(80, int(self.memory_gb * 0.8))}%",
                "memory_gb": self.memory_gb,
                "cleanup_interval": 300,  # 5분
                "aggressive_cleanup": False,
                "optimization": memory_optimization,
                "model_offloading": {
                    "enabled": True,
                    "offload_to": "cpu",
                    "keep_in_memory": ["human_parsing", "pose_estimation"]
                }
            },
            
            # 로깅 및 모니터링
            "logging": {
                "level": "INFO",
                "detailed_timing": True,
                "performance_metrics": True,
                "save_intermediate": enable_intermediate_saving,
                "debug_mode": False,
                "constructor_pattern": "optimal"
            }
        }

    def _load_external_config(self, config_path: str):
        """외부 설정 파일 로드"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                external_config = json.load(f)
            
            # 딥 머지
            self._deep_merge(self.config, external_config)
            logger.info(f"✅ 외부 설정 로드 완료: {config_path}")
            
        except Exception as e:
            logger.warning(f"⚠️ 외부 설정 로드 실패: {e}")

    def _apply_environment_overrides(self):
        """환경변수 기반 설정 오버라이드"""
        
        # 품질 레벨
        quality = os.getenv("PIPELINE_QUALITY_LEVEL", self.quality_level)
        if quality != self.quality_level:
            self.quality_level = quality
            self.config["pipeline"]["quality_level"] = quality
            self._apply_quality_preset(quality)
        
        # 디바이스 설정
        device_override = os.getenv("PIPELINE_DEVICE")
        if device_override and device_override != self.device:
            self.device = device_override
            self.config["optimization"]["device"] = device_override
        
        # 메모리 제한
        memory_limit = os.getenv("PIPELINE_MEMORY_LIMIT")
        if memory_limit:
            self.config["memory"]["max_memory_usage"] = memory_limit
        
        # 동시 처리 수
        max_concurrent = os.getenv("PIPELINE_MAX_CONCURRENT")
        if max_concurrent:
            try:
                self.config["pipeline"]["max_concurrent_requests"] = int(max_concurrent)
            except ValueError:
                pass
        
        # 디버그 모드
        debug_mode = os.getenv("PIPELINE_DEBUG", "false").lower() == "true"
        self.config["logging"]["debug_mode"] = debug_mode
        if debug_mode:
            self.config["logging"]["level"] = "DEBUG"
            self.config["logging"]["save_intermediate"] = True
        
        # 최적화 활성화/비활성화
        optimization_override = os.getenv("PIPELINE_OPTIMIZATION")
        if optimization_override:
            enable_opt = optimization_override.lower() == "true"
            self.optimization_enabled = enable_opt
            self.config["optimization"]["optimization_enabled"] = enable_opt

    def _apply_device_optimizations(self):
        """디바이스별 최적화 적용"""
        
        if self.device == "mps":
            # M3 Max MPS 최적화
            self.config["optimization"].update({
                "mixed_precision": self.optimization_enabled,
                "memory_efficient_attention": True,
                "compile_models": False,  # MPS에서는 컴파일 비활성화
                "batch_processing": {
                    "enabled": True,
                    "max_batch_size": 2,  # MPS 메모리 제한
                    "dynamic_batching": False
                }
            })
            
            # 이미지 크기 조정 (메모리 효율성)
            if self.quality_level in ["fast", "balanced"]:
                self.config["image"]["input_size"] = (512, 512)
                self.config["image"]["max_resolution"] = 1024
            
        elif self.device == "cuda":
            # CUDA 최적화
            self.config["optimization"].update({
                "mixed_precision": self.optimization_enabled,
                "gradient_checkpointing": True,
                "compile_models": self.optimization_enabled,
                "batch_processing": {
                    "enabled": True,
                    "max_batch_size": 8,
                    "dynamic_batching": True
                }
            })
            
        else:
            # CPU 최적화
            self.config["optimization"].update({
                "mixed_precision": False,
                "compile_models": False,
                "batch_processing": {
                    "enabled": False,
                    "max_batch_size": 1
                }
            })
            
            # CPU에서는 더 작은 모델 사용
            self.config["steps"]["virtual_fitting"]["num_inference_steps"] = 20
            self.config["steps"]["post_processing"]["super_resolution"]["enabled"] = False

    def _apply_quality_preset(self, quality_level: str):
        """품질 레벨에 따른 프리셋 적용"""
        
        quality_presets = {
            "fast": {
                "image_size": (256, 256),
                "inference_steps": 20,
                "super_resolution": False,
                "face_enhancement": False,
                "physics_simulation": False,
                "timeout": 60
            },
            "balanced": {
                "image_size": (512, 512),
                "inference_steps": 30,
                "super_resolution": True,
                "face_enhancement": False,
                "physics_simulation": True,
                "timeout": 120
            },
            "high": {
                "image_size": (512, 512),
                "inference_steps": 50,
                "super_resolution": True,
                "face_enhancement": True,
                "physics_simulation": True,
                "timeout": 300
            },
            "ultra": {
                "image_size": (1024, 1024),
                "inference_steps": 100,
                "super_resolution": True,
                "face_enhancement": True,
                "physics_simulation": True,
                "timeout": 600
            }
        }
        
        preset = quality_presets.get(quality_level, quality_presets["high"])
        
        # 이미지 크기
        self.config["image"]["input_size"] = preset["image_size"]
        self.config["image"]["output_size"] = preset["image_size"]
        
        # 추론 단계
        self.config["steps"]["virtual_fitting"]["num_inference_steps"] = preset["inference_steps"]
        
        # 후처리 설정
        self.config["steps"]["post_processing"]["super_resolution"]["enabled"] = preset["super_resolution"]
        self.config["steps"]["post_processing"]["face_enhancement"]["enabled"] = preset["face_enhancement"]
        
        # 물리 시뮬레이션
        self.config["steps"]["cloth_warping"]["physics_enabled"] = preset["physics_simulation"]
        
        # 타임아웃
        self.config["pipeline"]["timeout_seconds"] = preset["timeout"]
        
        logger.info(f"🎯 품질 프리셋 적용: {quality_level}")

    def _deep_merge(self, base_dict: Dict, update_dict: Dict):
        """딕셔너리 딥 머지"""
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._deep_merge(base_dict[key], value)
            else:
                base_dict[key] = value

    # ===============================================================
    # ✅ 최적 생성자 패턴 - 설정 접근 메서드들 (완전 복원)
    # ===============================================================
    
    def get_step_config(self, step_name: str) -> Dict[str, Any]:
        """특정 단계 설정 반환"""
        return self.config["steps"].get(step_name, {})
    
    def get_step_config_for_step(self, step_name: str) -> StepConfig:
        """특정 단계용 설정 반환"""
        step_data = self.config["steps"].get(step_name, {})
        
        return StepConfig(
            device=self.device,
            device_type=self.device_type,
            memory_gb=self.memory_gb,
            is_m3_max=self.is_m3_max,
            optimization_enabled=self.optimization_enabled,
            quality_level=self.quality_level,
            enable_caching=step_data.get('cache_enabled', True),
            batch_size=step_data.get('batch_size', 1),
            timeout_seconds=self.config["pipeline"]["timeout_seconds"],
            config_data=step_data
        )
    
    def get_model_path(self, model_name: str) -> str:
        """모델 파일 경로 반환"""
        base_dir = self.config["model_paths"]["base_dir"]
        checkpoint_path = self.config["model_paths"]["checkpoints"].get(model_name)
        
        if checkpoint_path:
            full_path = os.path.join(base_dir, checkpoint_path)
            return full_path
        else:
            # 기본 경로 생성
            return os.path.join(base_dir, model_name)
    
    def get_optimization_config(self) -> Dict[str, Any]:
        """최적화 설정 반환"""
        return self.config["optimization"]
    
    def get_memory_config(self) -> Dict[str, Any]:
        """메모리 설정 반환"""
        return self.config["memory"]
    
    def get_image_config(self) -> Dict[str, Any]:
        """이미지 처리 설정 반환"""
        return self.config["image"]
    
    def get_pipeline_config(self) -> Dict[str, Any]:
        """파이프라인 전역 설정 반환"""
        return self.config["pipeline"]
    
    def get_system_config(self) -> Dict[str, Any]:
        """✅ 최적 생성자 패턴 - 시스템 설정 반환"""
        return {
            "device": self.device,
            "device_type": self.device_type,
            "memory_gb": self.memory_gb,
            "is_m3_max": self.is_m3_max,
            "optimization_enabled": self.optimization_enabled,
            "quality_level": self.quality_level,
            "constructor_pattern": "optimal"
        }

    # ===============================================================
    # ✅ 최적 생성자 패턴 - 동적 설정 변경 메서드들 (완전 복원)
    # ===============================================================
    
    def update_quality_level(self, quality_level: str):
        """품질 레벨 동적 변경"""
        if quality_level != self.quality_level:
            self.quality_level = quality_level
            self._apply_quality_preset(quality_level)
            logger.info(f"🔄 품질 레벨 변경: {quality_level}")
    
    def update_device(self, device: str):
        """디바이스 동적 변경"""
        if device != self.device:
            self.device = device
            self.config["optimization"]["device"] = device
            self.config["system"]["device"] = device
            self._apply_device_optimizations()
            logger.info(f"🔄 디바이스 변경: {device}")
    
    def update_memory_limit(self, memory_gb: float):
        """✅ 최적 생성자 패턴 - 메모리 제한 동적 변경"""
        self.memory_gb = memory_gb
        self.config["memory"]["memory_gb"] = memory_gb
        self.config["system"]["memory_gb"] = memory_gb
        self.config["memory"]["max_memory_usage"] = f"{min(80, int(memory_gb * 0.8))}%"
        logger.info(f"🔄 메모리 제한 변경: {memory_gb}GB")
    
    def toggle_optimization(self, enabled: bool):
        """✅ 최적 생성자 패턴 - 최적화 토글"""
        self.optimization_enabled = enabled
        self.config["optimization"]["optimization_enabled"] = enabled
        self.config["system"]["optimization_enabled"] = enabled
        
        # 관련 설정들 업데이트
        self.config["optimization"]["mixed_precision"] = enabled
        self.config["steps"]["post_processing"]["enable_super_resolution"] = enabled
        self.config["steps"]["human_parsing"]["enable_quantization"] = enabled
        
        logger.info(f"🔄 최적화 모드: {'활성화' if enabled else '비활성화'}")
    
    def enable_debug_mode(self, enabled: bool = True):
        """디버그 모드 토글"""
        self.config["logging"]["debug_mode"] = enabled
        self.config["logging"]["save_intermediate"] = enabled
        if enabled:
            self.config["logging"]["level"] = "DEBUG"
        logger.info(f"🔄 디버그 모드: {'활성화' if enabled else '비활성화'}")
    
    def set_memory_limit(self, limit: Union[str, float]):
        """메모리 제한 설정 (기존 호환성)"""
        if isinstance(limit, (int, float)):
            self.update_memory_limit(float(limit))
        else:
            self.config["memory"]["max_memory_usage"] = limit
            logger.info(f"🔄 메모리 제한 설정: {limit}")

    # ===============================================================
    # ✅ 최적 생성자 패턴 - 검증 및 진단 메서드들 (완전 복원)
    # ===============================================================
    
    def validate_config(self) -> Dict[str, Any]:
        """설정 유효성 검사"""
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "constructor_pattern": "optimal"
        }
        
        # 필수 모델 경로 확인
        for model_name, checkpoint_path in self.config["model_paths"]["checkpoints"].items():
            full_path = self.get_model_path(model_name)
            if not os.path.exists(os.path.dirname(full_path)):
                validation_result["warnings"].append(f"모델 디렉토리 없음: {full_path}")
        
        # 디바이스 호환성 확인
        if self.device == "mps" and not torch.backends.mps.is_available():
            validation_result["errors"].append("MPS가 요청되었지만 사용할 수 없습니다")
            validation_result["valid"] = False
        
        if self.device == "cuda" and not torch.cuda.is_available():
            validation_result["errors"].append("CUDA가 요청되었지만 사용할 수 없습니다")
            validation_result["valid"] = False
        
        # 메모리 설정 확인
        max_memory = self.config["memory"]["max_memory_usage"]
        if isinstance(max_memory, str) and max_memory.endswith("%"):
            try:
                percent = float(max_memory[:-1])
                if not (0 < percent <= 100):
                    validation_result["errors"].append(f"잘못된 메모리 백분율: {max_memory}")
                    validation_result["valid"] = False
            except ValueError:
                validation_result["errors"].append(f"잘못된 메모리 형식: {max_memory}")
                validation_result["valid"] = False
        
        # 최적 생성자 패턴 검증
        required_system_params = ["device", "device_type", "memory_gb", "is_m3_max", "optimization_enabled"]
        for param in required_system_params:
            if not hasattr(self, param):
                validation_result["errors"].append(f"필수 시스템 파라미터 누락: {param}")
                validation_result["valid"] = False
        
        return validation_result
    
    def get_system_info(self) -> Dict[str, Any]:
        """✅ 최적 생성자 패턴 - 시스템 정보 반환 (오버라이드)"""
        base_info = super().get_system_info().__dict__
        
        # PipelineConfig 특화 정보 추가
        base_info.update({
            "pipeline_version": self.config["pipeline"]["version"],
            "quality_level": self.quality_level,
            "config_path": self.config_path,
            "device_info": self.device_info,
            "memory_config": self.get_memory_config(),
            "optimization_config": self.get_optimization_config(),
            "torch_version": torch.__version__,
            "config_valid": self.validate_config()["valid"],
            "pipeline_mode": self.config["pipeline"]["processing_mode"],
            "constructor_pattern": "optimal"
        })
        
        return base_info
    
    def export_config(self, file_path: str):
        """설정을 JSON 파일로 내보내기"""
        try:
            # 시스템 정보 포함하여 내보내기
            export_data = {
                "config": self.config,
                "system_info": self.get_system_info(),
                "export_timestamp": str(logger.info)
            }
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            logger.info(f"💾 설정 내보내기 완료: {file_path}")
        except Exception as e:
            logger.error(f"❌ 설정 내보내기 실패: {e}")
    
    def __repr__(self):
        return (f"PipelineConfig(device={self.device}, quality={self.quality_level}, "
                f"memory={self.memory_gb}GB, m3_max={self.is_m3_max}, "
                f"optimization={self.optimization_enabled}, constructor='optimal')")

# ===============================================================
# ✅ 최적 생성자 패턴 - 전역 파이프라인 설정 인스턴스들 (완전 복원)
# ===============================================================

@lru_cache()
def get_pipeline_config(
    quality_level: str = "balanced",
    device: Optional[str] = None,
    mode: Union[str, PipelineMode] = PipelineMode.PRODUCTION,
    **kwargs
) -> PipelineConfig:
    """✅ 파이프라인 설정 인스턴스 반환 (캐시됨)"""
    return PipelineConfig(
        device=device,
        quality_level=quality_level,
        mode=mode,
        **kwargs
    )

@lru_cache()
def get_step_configs() -> Dict[str, Dict[str, Any]]:
    """모든 단계 설정 반환 (캐시됨)"""
    config = get_pipeline_config()
    return config.config["steps"]

@lru_cache()
def get_model_paths() -> Dict[str, str]:
    """모든 모델 경로 반환 (캐시됨)"""
    config = get_pipeline_config()
    return {
        model_name: config.get_model_path(model_name)
        for model_name in config.config["model_paths"]["checkpoints"].keys()
    }

def create_custom_config(
    quality_level: str = "balanced",
    device: Optional[str] = None,
    custom_settings: Optional[Dict[str, Any]] = None,
    **kwargs
) -> PipelineConfig:
    """✅ 최적 생성자 패턴 - 커스텀 파이프라인 설정 생성"""
    
    # 커스텀 설정을 kwargs에 병합
    if custom_settings:
        kwargs.update(custom_settings)
    
    config = PipelineConfig(
        device=device,
        quality_level=quality_level,
        **kwargs
    )
    
    return config

def create_step_config_factory(pipeline_config: PipelineConfig):
    """Step 클래스용 설정 팩토리 생성"""
    def get_step_config(step_name: str) -> StepConfig:
        """특정 단계용 설정 반환"""
        return pipeline_config.get_step_config_for_step(step_name)
    
    return get_step_config

# ===============================================================
# ✅ 최적 생성자 패턴 - 하위 호환성 보장 함수들 (완전 복원)
# ===============================================================

def create_optimal_pipeline_config(
    device: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    **kwargs
) -> PipelineConfig:
    """✅ 최적 생성자 패턴 - 새로운 최적 방식"""
    return PipelineConfig(
        device=device,
        config=config,
        **kwargs
    )

def create_legacy_pipeline_config(
    config_path: Optional[str] = None, 
    quality_level: str = "balanced"
) -> PipelineConfig:
    """기존 방식 호환 (최적 생성자 패턴으로 내부 처리)"""
    return PipelineConfig(
        config_path=config_path,
        quality_level=quality_level
    )

# ===============================================================
# 환경별 설정 함수들 - 최적 생성자 패턴 (완전 복원)
# ===============================================================

def configure_for_development():
    """개발 환경 설정 - 최적 생성자 패턴"""
    config = get_pipeline_config(
        quality_level="fast",
        optimization_enabled=False,
        enable_caching=False,
        enable_intermediate_saving=True
    )
    config.enable_debug_mode(True)
    logger.info("🔧 개발 환경 설정 적용 (최적 생성자 패턴)")
    return config

def configure_for_production():
    """프로덕션 환경 설정 - 최적 생성자 패턴"""
    config = get_pipeline_config(
        quality_level="high",
        optimization_enabled=True,
        enable_caching=True,
        memory_optimization=True
    )
    config.enable_debug_mode(False)
    logger.info("🔧 프로덕션 환경 설정 적용 (최적 생성자 패턴)")
    return config

def configure_for_testing():
    """테스트 환경 설정 - 최적 생성자 패턴"""
    config = get_pipeline_config(
        quality_level="fast",
        max_concurrent_requests=1,
        timeout_seconds=60,
        optimization_enabled=False
    )
    config.enable_debug_mode(True)
    logger.info("🔧 테스트 환경 설정 적용 (최적 생성자 패턴)")
    return config

def configure_for_m3_max():
    """✅ M3 Max 최적화 설정 - 최적 생성자 패턴"""
    config = get_pipeline_config(
        device="mps",
        quality_level="high",
        memory_gb=128.0,
        is_m3_max=True,
        optimization_enabled=True,
        enable_caching=True,
        memory_optimization=True
    )
    logger.info("🔧 M3 Max 최적화 설정 적용 (최적 생성자 패턴)")
    return config

# ===============================================================
# 초기화 및 검증 (최적 생성자 패턴) (완전 복원)
# ===============================================================

# 기본 설정 생성 (자동 감지)
_default_config = get_pipeline_config()
_validation_result = _default_config.validate_config()

if not _validation_result["valid"]:
    for error in _validation_result["errors"]:
        logger.error(f"❌ 설정 오류: {error}")
    
    # 경고는 로깅만
    for warning in _validation_result["warnings"]:
        logger.warning(f"⚠️ 설정 경고: {warning}")

logger.info(f"🔧 최적 생성자 패턴 파이프라인 설정 초기화 완료 - 디바이스: {DEVICE}")

# 시스템 정보 로깅
_system_info = _default_config.get_system_info()
logger.info(f"💻 시스템: {_system_info['device']} ({_system_info['quality_level']}) - 최적 생성자 패턴")
logger.info(f"🎯 메모리: {_system_info['memory_gb']}GB, M3 Max: {'✅' if _system_info['is_m3_max'] else '❌'}")

# ===============================================================
# 최적 생성자 패턴 호환성 검증 (완전 복원)
# ===============================================================

def validate_optimal_constructor_compatibility() -> Dict[str, bool]:
    """최적 생성자 패턴 호환성 검증"""
    try:
        # 테스트 설정 생성 - 최적 생성자 패턴
        test_config = create_optimal_pipeline_config(
            device="cpu",  # 명시적 설정
            quality_level="fast",
            device_type="test",
            memory_gb=8.0,
            is_m3_max=False,
            optimization_enabled=False,
            custom_param="test_value"  # 확장 파라미터
        )
        
        # 필수 속성 확인
        required_attrs = [
            'device', 'device_type', 'memory_gb', 'is_m3_max', 
            'optimization_enabled', 'quality_level'
        ]
        attr_check = {attr: hasattr(test_config, attr) for attr in required_attrs}
        
        # 필수 메서드 확인
        required_methods = [
            'get_step_config', 'get_model_path', 'get_system_config',
            'update_quality_level', 'update_device', 'validate_config'
        ]
        method_check = {method: hasattr(test_config, method) for method in required_methods}
        
        # 확장 파라미터 확인
        extension_check = test_config.config.get('custom_param') == 'test_value'
        
        return {
            'attributes': all(attr_check.values()),
            'methods': all(method_check.values()),
            'extensions': extension_check,
            'attr_details': attr_check,
            'method_details': method_check,
            'overall_compatible': (
                all(attr_check.values()) and 
                all(method_check.values()) and 
                extension_check
            ),
            'constructor_pattern': 'optimal'
        }
        
    except Exception as e:
        logger.error(f"최적 생성자 패턴 호환성 검증 실패: {e}")
        return {
            'overall_compatible': False, 
            'error': str(e), 
            'constructor_pattern': 'optimal'
        }

# 모듈 로드 시 호환성 검증
_compatibility_result = validate_optimal_constructor_compatibility()
if _compatibility_result['overall_compatible']:
    logger.info("✅ 최적 생성자 패턴 호환성 검증 완료")
else:
    logger.warning(f"⚠️ 호환성 문제: {_compatibility_result}")

# 모듈 레벨 exports
__all__ = [
    # Enum 클래스들
    "DeviceType", "QualityLevel", "PipelineMode",
    
    # 데이터 클래스들
    "SystemInfo", "StepConfig",
    
    # 베이스 클래스들
    "OptimalConfigBase",
    
    # 메인 설정 클래스
    "PipelineConfig",
    
    # 팩토리 함수들
    "get_pipeline_config", "create_step_config_factory", "get_step_configs", "get_model_paths",
    "create_custom_config", "create_optimal_pipeline_config", "create_legacy_pipeline_config",
    
    # 환경별 설정 함수들
    "configure_for_development", "configure_for_production", "configure_for_testing", "configure_for_m3_max",
    
    # 유틸리티 함수들
    "validate_optimal_constructor_compatibility"
]

logger.info("🎯 MyCloset AI 파이프라인 설정 모듈 로드 완료 - 모든 기능 복원")