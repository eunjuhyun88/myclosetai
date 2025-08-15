#!/usr/bin/env python3
"""
🔥 MyCloset AI - Warping Configuration for Cloth Warping
========================================================

🎯 의류 워핑 전용 설정
✅ 워핑 알고리즘 설정
✅ 품질 설정
✅ 성능 설정
✅ M3 Max 최적화
"""

import torch
from dataclasses import dataclass, field
from typing import Dict, Any, List, Tuple, Optional, Union
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

@dataclass
class TPSWarpingConfig:
    """TPS 워핑 설정"""
    num_control_points: int = 16
    embedding_dim: int = 256
    warping_layers: int = 4
    control_point_net_layers: List[int] = field(default_factory=lambda: [64, 128, 256])
    tps_transform_layers: List[int] = field(default_factory=lambda: [256, 128, 64, 6])
    enable_adaptive_control_points: bool = True
    control_point_regularization: float = 0.01

@dataclass
class GeometricFlowConfig:
    """기하학적 플로우 워핑 설정"""
    embedding_dim: int = 128
    flow_layers: int = 6
    flow_net_channels: List[int] = field(default_factory=lambda: [64, 128, 256, 128, 64])
    flow_refinement_channels: List[int] = field(default_factory=lambda: [64, 32, 2])
    deformation_strength: float = 1.0
    enable_flow_regularization: bool = True
    flow_regularization_weight: float = 0.1

@dataclass
class NeuralWarpingConfig:
    """신경망 기반 워핑 설정"""
    embedding_dim: int = 512
    hidden_layers: int = 8
    attention_heads: int = 4
    hidden_channels: List[int] = field(default_factory=lambda: [256, 512, 1024, 512, 256])
    enable_skip_connections: bool = True
    enable_attention_mechanism: bool = True
    dropout_rate: float = 0.1

@dataclass
class ClothDeformationConfig:
    """의류 변형 네트워크 설정"""
    embedding_dim: int = 256
    deformation_layers: int = 5
    deformation_channels: List[int] = field(default_factory=lambda: [128, 256, 512, 256, 128])
    quality_assessment: bool = True
    enable_geometric_constraints: bool = True
    constraint_weight: float = 0.5

@dataclass
class QualityEnhancementConfig:
    """품질 향상 설정"""
    enable_edge_refinement: bool = True
    enable_artifact_removal: bool = True
    enable_texture_enhancement: bool = True
    enable_color_enhancement: bool = True
    refinement_iterations: int = 3
    artifact_threshold: float = 0.1
    texture_strength: float = 0.5
    color_enhancement_strength: float = 0.6

@dataclass
class HighResolutionConfig:
    """고해상도 처리 설정"""
    target_resolutions: List[Tuple[int, int]] = field(default_factory=lambda: [
        (256, 256), (512, 512), (1024, 1024), (2048, 2048)
    ])
    enable_multi_scale: bool = True
    enable_super_resolution: bool = True
    enable_adaptive_processing: bool = True
    scale_factors: List[float] = field(default_factory=lambda: [1.0, 2.0, 4.0, 8.0])
    quality_threshold: float = 0.8
    memory_efficient: bool = True

@dataclass
class ProcessingConfig:
    """처리 설정"""
    batch_size: int = 4
    num_workers: int = 2
    enable_mixed_precision: bool = True
    enable_gradient_checkpointing: bool = False
    enable_compile_optimization: bool = True
    max_memory_usage_gb: float = 8.0
    cleanup_threshold: float = 0.8

@dataclass
class DeviceConfig:
    """디바이스 설정"""
    device_type: str = "auto"  # auto, cpu, mps, cuda
    enable_mps: bool = True
    enable_cuda: bool = False
    enable_cpu_fallback: bool = True
    memory_fraction: float = 0.9
    enable_memory_pinning: bool = False

@dataclass
class ClothWarpingConfig:
    """의류 워핑 전체 설정"""
    
    # 기본 설정
    model_name: str = "cloth_warping_model"
    version: str = "1.0"
    input_size: Tuple[int, int] = (256, 256)
    output_size: Tuple[int, int] = (256, 256)
    
    # 워핑 알고리즘 설정
    tps_warping: TPSWarpingConfig = field(default_factory=TPSWarpingConfig)
    geometric_flow: GeometricFlowConfig = field(default_factory=GeometricFlowConfig)
    neural_warping: NeuralWarpingConfig = field(default_factory=NeuralWarpingConfig)
    cloth_deformation: ClothDeformationConfig = field(default_factory=ClothDeformationConfig)
    
    # 품질 향상 설정
    quality_enhancement: QualityEnhancementConfig = field(default_factory=QualityEnhancementConfig)
    
    # 고해상도 처리 설정
    high_resolution: HighResolutionConfig = field(default_factory=HighResolutionConfig)
    
    # 처리 설정
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    
    # 디바이스 설정
    device: DeviceConfig = field(default_factory=DeviceConfig)
    
    # 앙상블 설정
    enable_ensemble: bool = True
    ensemble_methods: List[str] = field(default_factory=lambda: ["voting", "weighted", "quality_weighted"])
    ensemble_weights: Dict[str, float] = field(default_factory=lambda: {
        "tps_warping": 0.3,
        "geometric_flow": 0.3,
        "neural_warping": 0.2,
        "cloth_deformation": 0.2
    })
    
    # 로깅 설정
    log_level: str = "INFO"
    enable_tensorboard: bool = True
    log_interval: int = 100
    
    # 체크포인트 설정
    checkpoint_dir: str = "./checkpoints"
    save_interval: int = 1000
    max_checkpoints: int = 5
    
    def __post_init__(self):
        """초기화 후 검증"""
        self._validate_config()
        self._setup_device()
        self._setup_logging()
    
    def _validate_config(self):
        """설정 검증"""
        try:
            # 입력/출력 크기 검증
            if self.input_size[0] <= 0 or self.input_size[1] <= 0:
                raise ValueError("입력 크기는 양수여야 합니다")
            
            if self.output_size[0] <= 0 or self.output_size[1] <= 0:
                raise ValueError("출력 크기는 양수여야 합니다")
            
            # 배치 크기 검증
            if self.processing.batch_size <= 0:
                raise ValueError("배치 크기는 양수여야 합니다")
            
            # 메모리 사용량 검증
            if self.processing.max_memory_usage_gb <= 0:
                raise ValueError("최대 메모리 사용량은 양수여야 합니다")
            
            # 앙상블 가중치 검증
            total_weight = sum(self.ensemble_weights.values())
            if abs(total_weight - 1.0) > 1e-6:
                raise ValueError(f"앙상블 가중치의 합이 1.0이어야 합니다 (현재: {total_weight})")
            
            self.logger.info("✅ 설정 검증 완료")
            
        except Exception as e:
            self.logger.error(f"❌ 설정 검증 실패: {e}")
            raise
    
    def _setup_device(self):
        """디바이스 설정"""
        try:
            if self.device.device_type == "auto":
                if self.device.enable_mps and torch.backends.mps.is_available():
                    self.device.device_type = "mps"
                    self.logger.info("✅ MPS 디바이스 사용")
                elif self.device.enable_cuda and torch.cuda.is_available():
                    self.device.device_type = "cuda"
                    self.logger.info("✅ CUDA 디바이스 사용")
                else:
                    self.device.device_type = "cpu"
                    self.logger.info("✅ CPU 디바이스 사용")
            else:
                self.logger.info(f"✅ {self.device.device_type.upper()} 디바이스 사용")
            
            # 디바이스별 설정 적용
            if self.device.device_type == "mps":
                self._apply_mps_settings()
            elif self.device.device_type == "cuda":
                self._apply_cuda_settings()
            elif self.device.device_type == "cpu":
                self._apply_cpu_settings()
                
        except Exception as e:
            self.logger.error(f"❌ 디바이스 설정 실패: {e}")
            # CPU로 폴백
            self.device.device_type = "cpu"
            self._apply_cpu_settings()
    
    def _apply_mps_settings(self):
        """MPS 설정 적용"""
        try:
            # MPS 최적화 설정
            if hasattr(torch.backends, 'mps'):
                # MPS 백엔드 설정
                pass
            
            # 메모리 효율성 설정
            self.processing.enable_mixed_precision = True
            self.processing.enable_gradient_checkpointing = False
            
            self.logger.info("✅ MPS 설정 적용 완료")
            
        except Exception as e:
            self.logger.warning(f"MPS 설정 적용 실패: {e}")
    
    def _apply_cuda_settings(self):
        """CUDA 설정 적용"""
        try:
            # CUDA 최적화 설정
            if torch.cuda.is_available():
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.deterministic = False
                
                # 메모리 설정
                torch.cuda.set_per_process_memory_fraction(self.device.memory_fraction)
            
            self.logger.info("✅ CUDA 설정 적용 완료")
            
        except Exception as e:
            self.logger.warning(f"CUDA 설정 적용 실패: {e}")
    
    def _apply_cpu_settings(self):
        """CPU 설정 적용"""
        try:
            # CPU 최적화 설정
            torch.set_num_threads(self.processing.num_workers)
            
            # 메모리 효율성 설정
            self.processing.enable_mixed_precision = False
            self.processing.enable_gradient_checkpointing = True
            
            self.logger.info("✅ CPU 설정 적용 완료")
            
        except Exception as e:
            self.logger.warning(f"CPU 설정 적용 실패: {e}")
    
    def _setup_logging(self):
        """로깅 설정"""
        try:
            # 로그 레벨 설정
            logging.getLogger().setLevel(getattr(logging, self.log_level.upper()))
            
            # 텐서보드 설정
            if self.enable_tensorboard:
                try:
                    from torch.utils.tensorboard import SummaryWriter
                    self.logger.info("✅ TensorBoard 지원 활성화")
                except ImportError:
                    self.logger.warning("TensorBoard가 설치되지 않았습니다")
                    self.enable_tensorboard = False
            
            self.logger.info("✅ 로깅 설정 완료")
            
        except Exception as e:
            self.logger.warning(f"로깅 설정 실패: {e}")
    
    def get_device(self) -> torch.device:
        """현재 디바이스 반환"""
        if self.device.device_type == "mps":
            return torch.device("mps")
        elif self.device.device_type == "cuda":
            return torch.device("cuda")
        else:
            return torch.device("cpu")
    
    def get_model_config(self) -> Dict[str, Any]:
        """모델 설정 반환"""
        return {
            "model_name": self.model_name,
            "version": self.version,
            "input_size": self.input_size,
            "output_size": self.output_size,
            "tps_warping": self.tps_warping.__dict__,
            "geometric_flow": self.geometric_flow.__dict__,
            "neural_warping": self.neural_warping.__dict__,
            "cloth_deformation": self.cloth_deformation.__dict__,
            "device_type": self.device.device_type
        }
    
    def get_processing_config(self) -> Dict[str, Any]:
        """처리 설정 반환"""
        return {
            "batch_size": self.processing.batch_size,
            "num_workers": self.processing.num_workers,
            "enable_mixed_precision": self.processing.enable_mixed_precision,
            "enable_gradient_checkpointing": self.processing.enable_gradient_checkpointing,
            "max_memory_usage_gb": self.processing.max_memory_usage_gb,
            "cleanup_threshold": self.processing.cleanup_threshold
        }
    
    def get_quality_config(self) -> Dict[str, Any]:
        """품질 설정 반환"""
        return {
            "quality_enhancement": self.quality_enhancement.__dict__,
            "high_resolution": self.high_resolution.__dict__
        }
    
    def save_config(self, file_path: Union[str, Path]):
        """설정을 파일로 저장"""
        try:
            import json
            
            config_dict = {
                "model_name": self.model_name,
                "version": self.version,
                "input_size": self.input_size,
                "output_size": self.output_size,
                "tps_warping": self.tps_warping.__dict__,
                "geometric_flow": self.geometric_flow.__dict__,
                "neural_warping": self.neural_warping.__dict__,
                "cloth_deformation": self.cloth_deformation.__dict__,
                "quality_enhancement": self.quality_enhancement.__dict__,
                "high_resolution": self.high_resolution.__dict__,
                "processing": self.processing.__dict__,
                "device": self.device.__dict__,
                "ensemble": {
                    "enable_ensemble": self.enable_ensemble,
                    "ensemble_methods": self.ensemble_methods,
                    "ensemble_weights": self.ensemble_weights
                }
            }
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(config_dict, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"✅ 설정 저장 완료: {file_path}")
            
        except Exception as e:
            self.logger.error(f"❌ 설정 저장 실패: {e}")
            raise
    
    @classmethod
    def load_config(cls, file_path: Union[str, Path]) -> 'ClothWarpingConfig':
        """파일에서 설정 로드"""
        try:
            import json
            
            with open(file_path, 'r', encoding='utf-8') as f:
                config_dict = json.load(f)
            
            # 설정 객체 생성
            config = cls()
            
            # 기본 설정 업데이트
            for key, value in config_dict.items():
                if hasattr(config, key):
                    if isinstance(value, dict) and hasattr(getattr(config, key), '__dict__'):
                        # 중첩된 설정 객체 업데이트
                        for sub_key, sub_value in value.items():
                            if hasattr(getattr(config, key), sub_key):
                                setattr(getattr(config, key), sub_key, sub_value)
                    else:
                        setattr(config, key, value)
            
            # 설정 검증 및 디바이스 설정
            config._validate_config()
            config._setup_device()
            config._setup_logging()
            
            logger.info(f"✅ 설정 로드 완료: {file_path}")
            return config
            
        except Exception as e:
            logger.error(f"❌ 설정 로드 실패: {e}")
            raise
    
    def get_summary(self) -> str:
        """설정 요약 반환"""
        summary = f"""
🔥 Cloth Warping Configuration Summary
=====================================

📋 기본 정보:
  - 모델명: {self.model_name}
  - 버전: {self.version}
  - 입력 크기: {self.input_size}
  - 출력 크기: {self.output_size}

🔧 워핑 알고리즘:
  - TPS 워핑: {self.tps_warping.num_control_points} 제어점, {self.tps_warping.embedding_dim}차원
  - 기하학적 플로우: {self.geometric_flow.flow_layers} 레이어, {self.geometric_flow.embedding_dim}차원
  - 신경망 워핑: {self.neural_warping.hidden_layers} 레이어, {self.neural_warping.attention_heads} 어텐션 헤드
  - 의류 변형: {self.cloth_deformation.deformation_layers} 레이어, 품질 평가: {self.cloth_deformation.quality_assessment}

🎯 품질 향상:
  - 엣지 정제: {self.quality_enhancement.enable_edge_refinement}
  - 아티팩트 제거: {self.quality_enhancement.enable_artifact_removal}
  - 텍스처 향상: {self.quality_enhancement.enable_texture_enhancement}
  - 색상 향상: {self.quality_enhancement.enable_color_enhancement}

🚀 고해상도 처리:
  - 멀티스케일: {self.high_resolution.enable_multi_scale}
  - 슈퍼해상도: {self.high_resolution.enable_super_resolution}
  - 목표 해상도: {self.high_resolution.target_resolutions}

⚙️ 처리 설정:
  - 배치 크기: {self.processing.batch_size}
  - 워커 수: {self.processing.num_workers}
  - 혼합 정밀도: {self.processing.enable_mixed_precision}
  - 최대 메모리: {self.processing.max_memory_usage_gb}GB

💻 디바이스:
  - 타입: {self.device.device_type.upper()}
  - MPS 지원: {self.device.enable_mps}
  - CUDA 지원: {self.device.enable_cuda}

🎲 앙상블:
  - 활성화: {self.enable_ensemble}
  - 방법: {', '.join(self.ensemble_methods)}
  - 가중치: {self.ensemble_weights}
"""
        return summary

# 기본 설정 인스턴스 생성
def create_default_warping_config() -> ClothWarpingConfig:
    """기본 워핑 설정 생성"""
    return ClothWarpingConfig()

# 설정 팩토리 함수
def create_warping_config(config_type: str = "default", **kwargs) -> ClothWarpingConfig:
    """
    워핑 설정 생성
    
    Args:
        config_type: 설정 타입 (default, high_quality, fast, memory_efficient)
        **kwargs: 추가 설정
    
    Returns:
        워핑 설정 객체
    """
    if config_type == "default":
        config = create_default_warping_config()
    elif config_type == "high_quality":
        config = create_default_warping_config()
        config.quality_enhancement.enable_edge_refinement = True
        config.quality_enhancement.enable_artifact_removal = True
        config.quality_enhancement.refinement_iterations = 5
        config.high_resolution.enable_super_resolution = True
        config.high_resolution.target_resolutions = [(512, 512), (1024, 1024), (2048, 2048)]
    elif config_type == "fast":
        config = create_default_warping_config()
        config.processing.batch_size = 8
        config.quality_enhancement.refinement_iterations = 1
        config.high_resolution.enable_super_resolution = False
        config.high_resolution.target_resolutions = [(256, 256), (512, 512)]
    elif config_type == "memory_efficient":
        config = create_default_warping_config()
        config.processing.batch_size = 2
        config.processing.max_memory_usage_gb = 4.0
        config.high_resolution.memory_efficient = True
        config.high_resolution.target_resolutions = [(256, 256), (512, 512)]
    else:
        config = create_default_warping_config()
    
    # 추가 설정 적용
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    return config

if __name__ == "__main__":
    # 테스트 코드
    logging.basicConfig(level=logging.INFO)
    
    # 기본 설정 생성
    config = create_default_warping_config()
    
    # 설정 요약 출력
    print(config.get_summary())
    
    # 설정 저장
    config.save_config("./warping_config.json")
    
    # 설정 로드
    loaded_config = ClothWarpingConfig.load_config("./warping_config.json")
    print("✅ 설정 로드 완료")
    
    # 고품질 설정 생성
    high_quality_config = create_warping_config("high_quality")
    print("✅ 고품질 설정 생성 완료")
