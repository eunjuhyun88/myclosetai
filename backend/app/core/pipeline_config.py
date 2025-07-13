"""
MyCloset AI 파이프라인 설정
"""

from dataclasses import dataclass
from typing import Optional, List
import torch

@dataclass
class PipelineConfig:
    """파이프라인 설정 클래스"""
    
    # 디바이스 설정
    device: str = "mps"  # M3 Max MPS 백엔드
    use_fp16: bool = True
    
    # 이미지 처리 설정
    image_size: int = 512
    batch_size: int = 1
    
    # 메모리 설정
    memory_limit_gb: float = 16.0
    enable_caching: bool = True
    
    # 성능 설정
    parallel_steps: bool = True
    max_workers: int = 4
    
    # 품질 설정
    quality_mode: str = "balanced"  # "fast", "balanced", "quality"
    quality_threshold: float = 0.8
    
    # 파일 경로
    model_dir: str = "models/checkpoints"
    upload_dir: str = "static/uploads"
    result_dir: str = "static/results"
    
    def __post_init__(self):
        """초기화 후 검증"""
        # 디바이스 검증
        if self.device == "mps" and not torch.backends.mps.is_available():
            self.device = "cpu"
        elif self.device == "cuda" and not torch.cuda.is_available():
            self.device = "cpu"
        
        # 품질 모드에 따른 설정 조정
        if self.quality_mode == "fast":
            self.image_size = 256
            self.parallel_steps = True
            self.use_fp16 = True
        elif self.quality_mode == "quality":
            self.image_size = 1024
            self.parallel_steps = False
            self.use_fp16 = False
        
        # 메모리에 따른 배치 크기 조정
        if self.memory_limit_gb < 8:
            self.batch_size = 1
            self.image_size = min(self.image_size, 256)
    
    @property
    def torch_device(self) -> torch.device:
        """PyTorch 디바이스 객체 반환"""
        return torch.device(self.device)
    
    def to_dict(self) -> dict:
        """딕셔너리로 변환"""
        return {
            "device": self.device,
            "use_fp16": self.use_fp16,
            "image_size": self.image_size,
            "batch_size": self.batch_size,
            "memory_limit_gb": self.memory_limit_gb,
            "enable_caching": self.enable_caching,
            "parallel_steps": self.parallel_steps,
            "max_workers": self.max_workers,
            "quality_mode": self.quality_mode,
            "quality_threshold": self.quality_threshold,
            "model_dir": self.model_dir,
            "upload_dir": self.upload_dir,
            "result_dir": self.result_dir
        }