"""
Post Processing Models Configuration

후처리 모델들의 설정을 관리하는 클래스입니다.
"""

import os
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
import yaml


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
class InferenceConfig:
    """추론 설정"""
    tile_size: int = 400
    tile_pad: int = 10
    pre_pad: int = 0
    half: bool = True
    bg_upsampler: bool = True
    bg_tile: int = 400
    suffix: Optional[str] = None
    only_center_face: bool = False
    aligned: bool = False
    background_enhance: bool = True
    face_upsample: bool = True
    upscale: int = 2
    codeformer_fidelity: float = 0.7


@dataclass
class PostProcessingConfig:
    """후처리 모델들의 전체 설정"""
    
    # 모델별 설정
    swinir: SwinIRConfig = field(default_factory=SwinIRConfig)
    realesrgan: RealESRGANConfig = field(default_factory=RealESRGANConfig)
    gfpgan: GFPGANConfig = field(default_factory=GFPGANConfig)
    codeformer: CodeFormerConfig = field(default_factory=CodeFormerConfig)
    
    # 추론 설정
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    
    # 체크포인트 설정
    checkpoint_dir: str = "checkpoints"
    
    # 디바이스 설정
    device: str = "auto"  # "auto", "cuda", "cpu"
    
    # 로깅 설정
    log_level: str = "INFO"
    log_file: Optional[str] = None
    
    # 메모리 설정
    max_memory_usage: float = 0.8  # GPU 메모리의 80%까지 사용
    enable_memory_optimization: bool = True
    
    # 배치 처리 설정
    batch_size: int = 1
    enable_batch_processing: bool = True
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'PostProcessingConfig':
        """YAML 파일에서 설정을 로드합니다."""
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"Config file not found: {yaml_path}")
        
        with open(yaml_path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)
        
        return cls.from_dict(config_data)
    
    @classmethod
    def from_dict(cls, config_data: Dict[str, Any]) -> 'PostProcessingConfig':
        """딕셔너리에서 설정을 로드합니다."""
        config = cls()
        
        # 모델별 설정 업데이트
        if 'swinir' in config_data:
            for key, value in config_data['swinir'].items():
                if hasattr(config.swinir, key):
                    setattr(config.swinir, key, value)
        
        if 'realesrgan' in config_data:
            for key, value in config_data['realesrgan'].items():
                if hasattr(config.realesrgan, key):
                    setattr(config.realesrgan, key, value)
        
        if 'gfpgan' in config_data:
            for key, value in config_data['gfpgan'].items():
                if hasattr(config.gfpgan, key):
                    setattr(config.gfpgan, key, value)
        
        if 'codeformer' in config_data:
            for key, value in config_data['codeformer'].items():
                if hasattr(config.codeformer, key):
                    setattr(config.codeformer, key, value)
        
        # 추론 설정 업데이트
        if 'inference' in config_data:
            for key, value in config_data['inference'].items():
                if hasattr(config.inference, key):
                    setattr(config.inference, key, value)
        
        # 일반 설정 업데이트
        for key, value in config_data.items():
            if key not in ['swinir', 'realesrgan', 'gfpgan', 'codeformer', 'inference']:
                if hasattr(config, key):
                    setattr(config, key, value)
        
        return config
    
    def to_dict(self) -> Dict[str, Any]:
        """설정을 딕셔너리로 변환합니다."""
        return {
            'swinir': {
                'img_size': self.swinir.img_size,
                'patch_size': self.swinir.patch_size,
                'in_chans': self.swinir.in_chans,
                'embed_dim': self.swinir.embed_dim,
                'depths': self.swinir.depths,
                'num_heads': self.swinir.num_heads,
                'window_size': self.swinir.window_size,
                'mlp_ratio': self.swinir.mlp_ratio,
                'qkv_bias': self.swinir.qkv_bias,
                'qk_scale': self.swinir.qk_scale,
                'drop_rate': self.swinir.drop_rate,
                'attn_drop_rate': self.swinir.attn_drop_rate,
                'drop_path_rate': self.swinir.drop_path_rate,
                'ape': self.swinir.ape,
                'patch_norm': self.swinir.patch_norm,
                'use_checkpoint': self.swinir.use_checkpoint,
                'upscale': self.swinir.upscale,
                'img_range': self.swinir.img_range,
                'upsampler': self.swinir.upsampler,
                'resi_connection': self.swinir.resi_connection
            },
            'realesrgan': {
                'num_feat': self.realesrgan.num_feat,
                'num_block': self.realesrgan.num_block,
                'upscale': self.realesrgan.upscale,
                'num_in_ch': self.realesrgan.num_in_ch,
                'num_out_ch': self.realesrgan.num_out_ch,
                'task': self.realesrgan.task
            },
            'gfpgan': {
                'out_size': self.gfpgan.out_size,
                'num_style_feat': self.gfpgan.num_style_feat,
                'channel_multiplier': self.gfpgan.channel_multiplier,
                'decoder_load_path': self.gfpgan.decoder_load_path,
                'fix_decoder': self.gfpgan.fix_decoder,
                'num_mlp': self.gfpgan.num_mlp,
                'input_is_latent': self.gfpgan.input_is_latent,
                'different_w': self.gfpgan.different_w,
                'narrow': self.gfpgan.narrow,
                'sft_half': self.gfpgan.sft_half
            },
            'codeformer': {
                'dim_embd': self.codeformer.dim_embd,
                'n_head': self.codeformer.n_head,
                'n_layers': self.codeformer.n_layers,
                'codebook_size': self.codeformer.codebook_size,
                'latent_dim': self.codeformer.latent_dim,
                'channels': self.codeformer.channels,
                'img_size': self.codeformer.img_size
            },
            'inference': {
                'tile_size': self.inference.tile_size,
                'tile_pad': self.inference.tile_pad,
                'pre_pad': self.inference.pre_pad,
                'half': self.inference.half,
                'bg_upsampler': self.inference.bg_upsampler,
                'bg_tile': self.inference.bg_tile,
                'suffix': self.inference.suffix,
                'only_center_face': self.inference.only_center_face,
                'aligned': self.inference.aligned,
                'background_enhance': self.inference.background_enhance,
                'face_upsample': self.inference.face_upsample,
                'upscale': self.inference.upscale,
                'codeformer_fidelity': self.inference.codeformer_fidelity
            },
            'checkpoint_dir': self.checkpoint_dir,
            'device': self.device,
            'log_level': self.log_level,
            'log_file': self.log_file,
            'max_memory_usage': self.max_memory_usage,
            'enable_memory_optimization': self.enable_memory_optimization,
            'batch_size': self.batch_size,
            'enable_batch_processing': self.enable_batch_processing
        }
    
    def to_yaml(self, yaml_path: str):
        """설정을 YAML 파일로 저장합니다."""
        config_dict = self.to_dict()
        
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True, indent=2)
    
    def validate(self) -> bool:
        """설정의 유효성을 검증합니다."""
        # 기본 검증
        if self.checkpoint_dir and not os.path.exists(self.checkpoint_dir):
            return False
        
        # 모델별 설정 검증
        if self.swinir.upscale not in [1, 2, 3, 4]:
            return False
        
        if self.realesrgan.upscale not in [1, 2, 3, 4]:
            return False
        
        if self.inference.codeformer_fidelity < 0.0 or self.inference.codeformer_fidelity > 1.0:
            return False
        
        return True
    
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


# 기본 설정 인스턴스
default_config = PostProcessingConfig()
