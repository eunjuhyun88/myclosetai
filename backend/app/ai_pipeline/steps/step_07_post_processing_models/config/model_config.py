"""
Post Processing Model Configuration

후처리 모델들의 설정을 관리하는 클래스입니다.
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass, field
import json
from pathlib import Path


@dataclass
class SwinIRConfig:
    """SwinIR 모델 설정"""
    img_size: int = 64
    patch_size: int = 1
    in_chans: int = 3
    embed_dim: int = 96
    depths: list = field(default_factory=lambda: [6, 6, 6, 6, 6, 6])
    num_heads: list = field(default_factory=lambda: [6, 6, 6, 6, 6, 6])
    window_size: int = 7
    mlp_ratio: float = 4.0
    qkv_bias: bool = True
    qk_scale: Optional[float] = None
    drop_rate: float = 0.0
    attn_drop_rate: float = 0.0
    drop_path_rate: float = 0.1
    ape: bool = False
    patch_norm: bool = True
    use_checkpoint: bool = False
    upscale: int = 4
    img_range: float = 1.0
    upsampler: str = 'nearest+conv'
    resi_connection: str = '1conv'


@dataclass
class RealESRGANConfig:
    """Real-ESRGAN 모델 설정"""
    num_feat: int = 64
    num_block: int = 23
    num_grow_ch: int = 32
    upscale: int = 4
    num_in_ch: int = 3
    num_out_ch: int = 3
    task: str = 'realesrgan'


@dataclass
class GFPGANConfig:
    """GFPGAN 모델 설정"""
    out_size: int = 512
    num_style_feat: int = 512
    channel_multiplier: int = 2
    decoder_load_path: Optional[str] = None
    fix_decoder: bool = True
    num_mlp: int = 8
    input_is_latent: bool = True
    different_w: bool = True
    narrow: int = 1
    sft_half: bool = True


@dataclass
class CodeFormerConfig:
    """CodeFormer 모델 설정"""
    dim_embd: int = 512
    n_head: int = 8
    n_layers: int = 9
    codebook_size: int = 1024
    latent_dim: int = 256
    channels: list = field(default_factory=lambda: [64, 128, 256, 512])
    img_size: int = 256


@dataclass
class PostProcessingModelConfig:
    """후처리 모델 전체 설정"""
    
    # 모델별 설정
    swinir: SwinIRConfig = field(default_factory=SwinIRConfig)
    realesrgan: RealESRGANConfig = field(default_factory=RealESRGANConfig)
    gfpgan: GFPGANConfig = field(default_factory=GFPGANConfig)
    codeformer: CodeFormerConfig = field(default_factory=CodeFormerConfig)
    
    # 공통 설정
    checkpoint_dir: str = "checkpoints"
    device: str = "auto"  # "auto", "cuda", "cpu"
    model_cache_size: int = 2  # 메모리에 캐시할 모델 수
    
    @classmethod
    def from_file(cls, config_path: str) -> 'PostProcessingModelConfig':
        """파일에서 설정을 로드합니다."""
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
        
        return cls.from_dict(config_data)
    
    @classmethod
    def from_dict(cls, config_data: Dict[str, Any]) -> 'PostProcessingModelConfig':
        """딕셔너리에서 설정을 로드합니다."""
        # 모델별 설정 로드
        swinir_config = SwinIRConfig(**config_data.get('swinir', {}))
        realesrgan_config = RealESRGANConfig(**config_data.get('realesrgan', {}))
        gfpgan_config = GFPGANConfig(**config_data.get('gfpgan', {}))
        codeformer_config = CodeFormerConfig(**config_data.get('codeformer', {}))
        
        # 공통 설정
        common_config = {
            'checkpoint_dir': config_data.get('checkpoint_dir', 'checkpoints'),
            'device': config_data.get('device', 'auto'),
            'model_cache_size': config_data.get('model_cache_size', 2)
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
            'checkpoint_dir': self.checkpoint_dir,
            'device': self.device,
            'model_cache_size': self.model_cache_size
        }
    
    def save_to_file(self, config_path: str):
        """설정을 파일에 저장합니다."""
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
    
    def get_model_config(self, model_type: str) -> Dict[str, Any]:
        """지정된 모델 타입의 설정을 반환합니다."""
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
    
    def update_model_config(self, model_type: str, **kwargs):
        """지정된 모델 타입의 설정을 업데이트합니다."""
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
    
    def validate(self) -> bool:
        """설정의 유효성을 검증합니다."""
        try:
            # 기본 검증
            assert self.model_cache_size > 0, "model_cache_size must be positive"
            assert self.checkpoint_dir, "checkpoint_dir cannot be empty"
            
            # 모델별 설정 검증
            assert self.swinir.upscale > 0, "SwinIR upscale must be positive"
            assert self.realesrgan.upscale > 0, "Real-ESRGAN upscale must be positive"
            assert self.gfpgan.out_size > 0, "GFPGAN out_size must be positive"
            assert self.codeformer.dim_embd > 0, "CodeFormer dim_embd must be positive"
            
            return True
        except AssertionError as e:
            print(f"Configuration validation failed: {e}")
            return False
