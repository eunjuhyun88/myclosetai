"""
AI 파이프라인 설정 관리
8단계 가상 피팅 파이프라인의 전체 설정을 관리합니다.
기존 app/ 구조와 완전히 호환되며, M3 Max 최적화를 포함합니다.
"""

import os
import json
import logging
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
from functools import lru_cache
import torch

from .gpu_config import gpu_config, DEVICE, DEVICE_INFO

logger = logging.getLogger(__name__)

class PipelineConfig:
    """
    8단계 AI 파이프라인 설정 관리
    
    기존 app/ 구조에 맞춰 설계된 완전한 파이프라인 설정 시스템
    - 8단계 가상 피팅 파이프라인 지원
    - M3 Max MPS 최적화
    - 동적 설정 로딩
    - 환경별 설정 분리
    """
    
    def __init__(self, config_path: Optional[str] = None, quality_level: str = "high"):
        """
        Args:
            config_path: 설정 파일 경로 (선택적)
            quality_level: 품질 레벨 (fast, balanced, high, ultra)
        """
        self.quality_level = quality_level
        self.device = DEVICE
        self.device_info = DEVICE_INFO
        
        # 기본 설정 로드
        self.config = self._load_default_config()
        
        # 외부 설정 파일 로드 (있는 경우)
        if config_path and os.path.exists(config_path):
            self._load_external_config(config_path)
        
        # 환경변수 기반 오버라이드
        self._apply_environment_overrides()
        
        # 디바이스별 최적화 적용
        self._apply_device_optimizations()
        
        logger.info(f"🔧 파이프라인 설정 완료 - 품질: {quality_level}, 디바이스: {self.device}")
    
    def _load_default_config(self) -> Dict[str, Any]:
        """기본 설정 로드"""
        return {
            # 전역 파이프라인 설정
            "pipeline": {
                "name": "mycloset_virtual_fitting",
                "version": "2.0.0",
                "quality_level": self.quality_level,
                "processing_mode": "complete",  # fast, balanced, complete
                "enable_optimization": True,
                "enable_caching": True,
                "enable_parallel": True,
                "memory_optimization": True,
                "max_concurrent_requests": 4,
                "timeout_seconds": 300,
                "enable_intermediate_saving": False,
                "auto_retry": True,
                "max_retries": 3
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
            
            # 8단계 파이프라인 개별 설정
            "steps": {
                # 1단계: 인체 파싱 (Human Parsing)
                "human_parsing": {
                    "model_name": "graphonomy",
                    "model_path": "app/ai_pipeline/models/ai_models/graphonomy",
                    "num_classes": 20,
                    "confidence_threshold": 0.7,
                    "input_size": (512, 512),
                    "batch_size": 1,
                    "cache_enabled": True,
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
                    "post_processing": {
                        "morphology_enabled": True,
                        "gaussian_blur": True,
                        "edge_smoothing": True,
                        "noise_removal": True
                    },
                    "quality_assessment": {
                        "enable": True,
                        "min_quality": 0.6,
                        "auto_retry": True
                    }
                },
                
                # 4단계: 기하학적 매칭 (Geometric Matching)
                "geometric_matching": {
                    "algorithm": "tps_hybrid",  # tps, affine, tps_hybrid
                    "num_control_points": 20,
                    "regularization": 0.001,
                    "matching_method": "hungarian",
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
                    "quality_enhancement": {
                        "color_matching": True,
                        "lighting_adjustment": True,
                        "texture_preservation": True,
                        "edge_smoothing": True
                    }
                },
                
                # 7단계: 후처리 (Post Processing)
                "post_processing": {
                    "super_resolution": {
                        "enabled": True,
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
            
            # 성능 최적화 설정
            "optimization": {
                "device": self.device,
                "mixed_precision": True,
                "gradient_checkpointing": False,
                "memory_efficient_attention": True,
                "compile_models": False,  # PyTorch 2.0 compile
                "batch_processing": {
                    "enabled": True,
                    "max_batch_size": 4,
                    "dynamic_batching": True
                },
                "caching": {
                    "enabled": True,
                    "ttl": 3600,  # 1시간
                    "max_size": "2GB",
                    "cache_intermediate": False
                }
            },
            
            # 메모리 관리 설정
            "memory": {
                "max_memory_usage": "80%",
                "cleanup_interval": 300,  # 5분
                "aggressive_cleanup": False,
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
                "save_intermediate": False,
                "debug_mode": False
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
    
    def _apply_device_optimizations(self):
        """디바이스별 최적화 적용"""
        
        if self.device == "mps":
            # M3 Max MPS 최적화
            self.config["optimization"].update({
                "mixed_precision": True,
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
                "mixed_precision": True,
                "gradient_checkpointing": True,
                "compile_models": True,
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
    
    # 설정 접근 메서드들
    
    def get_step_config(self, step_name: str) -> Dict[str, Any]:
        """특정 단계 설정 반환"""
        return self.config["steps"].get(step_name, {})
    
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
    
    # 동적 설정 변경 메서드들
    
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
            self._apply_device_optimizations()
            logger.info(f"🔄 디바이스 변경: {device}")
    
    def enable_debug_mode(self, enabled: bool = True):
        """디버그 모드 토글"""
        self.config["logging"]["debug_mode"] = enabled
        self.config["logging"]["save_intermediate"] = enabled
        if enabled:
            self.config["logging"]["level"] = "DEBUG"
        logger.info(f"🔄 디버그 모드: {'활성화' if enabled else '비활성화'}")
    
    def set_memory_limit(self, limit: Union[str, float]):
        """메모리 제한 설정"""
        self.config["memory"]["max_memory_usage"] = limit
        logger.info(f"🔄 메모리 제한 설정: {limit}")
    
    # 검증 및 진단 메서드들
    
    def validate_config(self) -> Dict[str, Any]:
        """설정 유효성 검사"""
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": []
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
        
        return validation_result
    
    def get_system_info(self) -> Dict[str, Any]:
        """시스템 정보 반환"""
        return {
            "device": self.device,
            "device_info": self.device_info,
            "quality_level": self.quality_level,
            "memory_config": self.get_memory_config(),
            "optimization_config": self.get_optimization_config(),
            "torch_version": torch.__version__,
            "config_valid": self.validate_config()["valid"]
        }
    
    def export_config(self, file_path: str):
        """설정을 JSON 파일로 내보내기"""
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
            logger.info(f"💾 설정 내보내기 완료: {file_path}")
        except Exception as e:
            logger.error(f"❌ 설정 내보내기 실패: {e}")
    
    def __repr__(self):
        return f"PipelineConfig(device={self.device}, quality={self.quality_level})"


# 전역 파이프라인 설정 인스턴스들

@lru_cache()
def get_pipeline_config(quality_level: str = "high") -> PipelineConfig:
    """파이프라인 설정 인스턴스 반환 (캐시됨)"""
    return PipelineConfig(quality_level=quality_level)

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
    quality_level: str = "high",
    device: Optional[str] = None,
    custom_settings: Optional[Dict[str, Any]] = None
) -> PipelineConfig:
    """커스텀 파이프라인 설정 생성"""
    
    config = PipelineConfig(quality_level=quality_level)
    
    # 디바이스 설정
    if device:
        config.update_device(device)
    
    # 커스텀 설정 적용
    if custom_settings:
        config._deep_merge(config.config, custom_settings)
    
    return config

# 초기화 및 검증
_default_config = get_pipeline_config()
_validation_result = _default_config.validate_config()

if not _validation_result["valid"]:
    for error in _validation_result["errors"]:
        logger.error(f"❌ 설정 오류: {error}")
    
    # 경고는 로깅만
    for warning in _validation_result["warnings"]:
        logger.warning(f"⚠️ 설정 경고: {warning}")

logger.info(f"🔧 파이프라인 설정 초기화 완료 - 디바이스: {DEVICE}")

# 시스템 정보 로깅
_system_info = _default_config.get_system_info()
logger.info(f"💻 시스템 정보: {_system_info['device']} ({_system_info['quality_level']})")

# 모듈 레벨 exports
__all__ = [
    "PipelineConfig",
    "get_pipeline_config", 
    "get_step_configs",
    "get_model_paths",
    "create_custom_config"
]

# 환경별 설정 함수들
def configure_for_development():
    """개발 환경 설정"""
    config = get_pipeline_config()
    config.enable_debug_mode(True)
    config.config["pipeline"]["enable_caching"] = False
    config.config["logging"]["detailed_timing"] = True
    logger.info("🔧 개발 환경 설정 적용")

def configure_for_production():
    """프로덕션 환경 설정"""
    config = get_pipeline_config()
    config.enable_debug_mode(False)
    config.config["pipeline"]["enable_caching"] = True
    config.config["memory"]["aggressive_cleanup"] = True
    logger.info("🔧 프로덕션 환경 설정 적용")

def configure_for_testing():
    """테스트 환경 설정"""
    config = get_pipeline_config("fast")  # 빠른 처리
    config.config["pipeline"]["max_concurrent_requests"] = 1
    config.config["pipeline"]["timeout_seconds"] = 60
    config.config["logging"]["level"] = "DEBUG"
    logger.info("🔧 테스트 환경 설정 적용")