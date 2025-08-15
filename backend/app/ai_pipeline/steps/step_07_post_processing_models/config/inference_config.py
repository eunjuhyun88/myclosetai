"""
Post Processing Inference Configuration

후처리 모델들의 추론 설정을 관리하는 클래스입니다.
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass, field
import json
from pathlib import Path


@dataclass
class SwinIRInferenceConfig:
    """SwinIR 추론 설정"""
    tile_size: int = 400
    tile_pad: int = 10
    pre_pad: int = 0
    half: bool = True
    img_range: float = 1.0
    window_size: int = 7


@dataclass
class RealESRGANInferenceConfig:
    """Real-ESRGAN 추론 설정"""
    tile_size: int = 400
    tile_pad: int = 10
    pre_pad: int = 0
    half: bool = True
    outscale: int = 4
    alpha_upsampler: str = 'realesrgan'
    face_enhance: bool = False


@dataclass
class GFPGANInferenceConfig:
    """GFPGAN 추론 설정"""
    bg_upsampler: bool = True
    bg_tile: int = 400
    suffix: Optional[str] = None
    only_center_face: bool = False
    aligned: bool = False
    weight: float = 0.5


@dataclass
class CodeFormerInferenceConfig:
    """CodeFormer 추론 설정"""
    background_enhance: bool = True
    face_upsample: bool = True
    upscale: int = 2
    codeformer_fidelity: float = 0.7
    weight: float = 0.5


@dataclass
class PostProcessingInferenceConfig:
    """후처리 추론 전체 설정"""
    
    # 모델별 추론 설정
    swinir: SwinIRInferenceConfig = field(default_factory=SwinIRInferenceConfig)
    realesrgan: RealESRGANInferenceConfig = field(default_factory=RealESRGANInferenceConfig)
    gfpgan: GFPGANInferenceConfig = field(default_factory=GFPGANInferenceConfig)
    codeformer: CodeFormerInferenceConfig = field(default_factory=CodeFormerInferenceConfig)
    
    # 공통 추론 설정
    batch_size: int = 1
    num_workers: int = 0
    pin_memory: bool = True
    device: str = "auto"  # "auto", "cuda", "cpu"
    
    # 메모리 관리 설정
    max_memory_usage: float = 0.8  # GPU 메모리 사용률 제한 (0.0 ~ 1.0)
    enable_tiling: bool = True  # 큰 이미지에 대한 타일링 처리 활성화
    tile_size_threshold: int = 1024  # 타일링을 적용할 이미지 크기 임계값
    
    @classmethod
    def from_file(cls, config_path: str) -> 'PostProcessingInferenceConfig':
        """파일에서 설정을 로드합니다."""
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
        
        return cls.from_dict(config_data)
    
    @classmethod
    def from_dict(cls, config_data: Dict[str, Any]) -> 'PostProcessingInferenceConfig':
        """딕셔너리에서 설정을 로드합니다."""
        # 모델별 추론 설정 로드
        swinir_config = SwinIRInferenceConfig(**config_data.get('swinir', {}))
        realesrgan_config = RealESRGANInferenceConfig(**config_data.get('realesrgan', {}))
        gfpgan_config = GFPGANInferenceConfig(**config_data.get('gfpgan', {}))
        codeformer_config = CodeFormerInferenceConfig(**config_data.get('codeformer', {}))
        
        # 공통 설정
        common_config = {
            'batch_size': config_data.get('batch_size', 1),
            'num_workers': config_data.get('num_workers', 0),
            'pin_memory': config_data.get('pin_memory', True),
            'device': config_data.get('device', 'auto'),
            'max_memory_usage': config_data.get('max_memory_usage', 0.8),
            'enable_tiling': config_data.get('enable_tiling', True),
            'tile_size_threshold': config_data.get('tile_size_threshold', 1024)
        }
        
        return cls(
            swinir=swinir_config,
            realesrgan=realesrgan_config,
            gfpgan=gfpgan_config,
            codeformer=codeformer_config,
            **common_config
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """설정을 딕셔너리로 변환합니다."""
        return {
            'swinir': self.swinir.__dict__,
            'realesrgan': self.realesrgan.__dict__,
            'gfpgan': self.gfpgan.__dict__,
            'codeformer': self.codeformer.__dict__,
            'batch_size': self.batch_size,
            'num_workers': self.num_workers,
            'pin_memory': self.pin_memory,
            'device': self.device,
            'max_memory_usage': self.max_memory_usage,
            'enable_tiling': self.enable_tiling,
            'tile_size_threshold': self.tile_size_threshold
        }
    
    def save_to_file(self, config_path: str):
        """설정을 파일에 저장합니다."""
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
    
    def get_model_inference_config(self, model_type: str) -> Dict[str, Any]:
        """지정된 모델 타입의 추론 설정을 반환합니다."""
        if model_type == 'swinir':
            return self.swinir.__dict__
        elif model_type == 'realesrgan':
            return self.realesrgan.__dict__
        elif model_type == 'gfpgan':
            return self.gfpgan.__dict__
        elif model_type == 'codeformer':
            return self.codeformer.__dict__
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def update_model_inference_config(self, model_type: str, **kwargs):
        """지정된 모델 타입의 추론 설정을 업데이트합니다."""
        if model_type == 'swinir':
            for key, value in kwargs.items():
                if hasattr(self.swinir, key):
                    setattr(self.swinir, key, value)
        elif model_type == 'realesrgan':
            for key, value in kwargs.items():
                if hasattr(self.realesrgan, key):
                    setattr(self.realesrgan, key, value)
        elif model_type == 'gfpgan':
            for key, value in kwargs.items():
                if hasattr(self.gfpgan, key):
                    setattr(self.gfpgan, key, value)
        elif model_type == 'codeformer':
            for key, value in kwargs.items():
                if hasattr(self.codeformer, key):
                    setattr(self.codeformer, key, value)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def should_use_tiling(self, image_height: int, image_width: int) -> bool:
        """이미지 크기에 따라 타일링 사용 여부를 결정합니다."""
        if not self.enable_tiling:
            return False
        
        max_dimension = max(image_height, image_width)
        return max_dimension > self.tile_size_threshold
    
    def get_optimal_tile_size(self, model_type: str, image_height: int, image_width: int) -> int:
        """이미지 크기에 따른 최적 타일 크기를 반환합니다."""
        if model_type == 'swinir':
            base_tile_size = self.swinir.tile_size
        elif model_type == 'realesrgan':
            base_tile_size = self.realesrgan.tile_size
        elif model_type == 'gfpgan':
            base_tile_size = self.gfpgan.bg_tile
        else:
            base_tile_size = 400  # 기본값
        
        # 이미지 크기에 따라 타일 크기 조정
        max_dimension = max(image_height, image_width)
        if max_dimension <= base_tile_size:
            return max_dimension
        
        # 메모리 효율성을 위해 타일 크기 조정
        if max_dimension > 2048:
            return min(base_tile_size // 2, 200)
        elif max_dimension > 1024:
            return min(base_tile_size, 400)
        else:
            return base_tile_size
    
    def validate(self) -> bool:
        """설정의 유효성을 검증합니다."""
        try:
            # 기본 검증
            assert self.batch_size > 0, "batch_size must be positive"
            assert self.num_workers >= 0, "num_workers must be non-negative"
            assert 0.0 <= self.max_memory_usage <= 1.0, "max_memory_usage must be between 0.0 and 1.0"
            assert self.tile_size_threshold > 0, "tile_size_threshold must be positive"
            
            # 모델별 설정 검증
            assert self.swinir.tile_size > 0, "SwinIR tile_size must be positive"
            assert self.realesrgan.tile_size > 0, "Real-ESRGAN tile_size must be positive"
            assert self.gfpgan.bg_tile > 0, "GFPGAN bg_tile must be positive"
            assert 0.0 <= self.codeformer.codeformer_fidelity <= 1.0, "CodeFormer fidelity must be between 0.0 and 1.0"
            
            return True
        except AssertionError as e:
            print(f"Inference configuration validation failed: {e}")
            return False
    
    def get_memory_efficient_settings(self, available_memory_gb: float) -> Dict[str, Any]:
        """사용 가능한 메모리에 따른 메모리 효율적인 설정을 반환합니다."""
        settings = {}
        
        if available_memory_gb < 4.0:
            # 낮은 메모리 환경
            settings.update({
                'batch_size': 1,
                'enable_tiling': True,
                'tile_size_threshold': 512,
                'swinir': {'tile_size': 200, 'tile_pad': 5},
                'realesrgan': {'tile_size': 200, 'tile_pad': 5},
                'gfpgan': {'bg_tile': 200}
            })
        elif available_memory_gb < 8.0:
            # 중간 메모리 환경
            settings.update({
                'batch_size': 1,
                'enable_tiling': True,
                'tile_size_threshold': 1024,
                'swinir': {'tile_size': 400, 'tile_pad': 10},
                'realesrgan': {'tile_size': 400, 'tile_pad': 10},
                'gfpgan': {'bg_tile': 400}
            })
        else:
            # 높은 메모리 환경
            settings.update({
                'batch_size': 2,
                'enable_tiling': False,
                'tile_size_threshold': 2048
            })
        
        return settings
