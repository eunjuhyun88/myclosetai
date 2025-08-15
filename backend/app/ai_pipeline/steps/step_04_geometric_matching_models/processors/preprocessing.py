#!/usr/bin/env python3
"""
🔥 MyCloset AI - Geometric Matching Preprocessing
==================================================

🎯 기하학적 매칭 전처리 모듈
✅ 이미지 정규화 및 표준화
✅ 특징 강화 및 노이즈 제거
✅ 기하학적 변형 처리
✅ M3 Max 최적화
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
import cv2
from PIL import Image, ImageEnhance, ImageFilter

logger = logging.getLogger(__name__)

@dataclass
class PreprocessingConfig:
    """전처리 설정"""
    input_size: Tuple[int, int] = (256, 256)
    output_size: Tuple[int, int] = (256, 256)
    normalize_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    normalize_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    enable_augmentation: bool = True
    enable_noise_reduction: bool = True
    enable_contrast_enhancement: bool = True
    enable_edge_detection: bool = True
    enable_geometric_correction: bool = True
    use_mps: bool = True

class ImageNormalizer(nn.Module):
    """이미지 정규화 모듈"""
    
    def __init__(self, mean: Tuple[float, float, float], std: Tuple[float, float, float]):
        super().__init__()
        self.mean = torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1)
        
    def to(self, device):
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        return self
        
    def forward(self, x):
        # 정규화 적용
        normalized = (x - self.mean) / self.std
        return normalized

class NoiseReductionNetwork(nn.Module):
    """노이즈 제거 네트워크"""
    
    def __init__(self, input_channels: int = 3):
        super().__init__()
        self.input_channels = input_channels
        
        # 노이즈 제거를 위한 U-Net 구조
        self.encoder = nn.ModuleList([
            nn.Conv2d(input_channels, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU()
        ])
        
        self.bottleneck = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU()
        )
        
        self.decoder = nn.ModuleList([
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, input_channels, 3, padding=1),
            nn.Sigmoid()
        ])
        
    def forward(self, x):
        # Encoder
        encoded = x
        for layer in self.encoder:
            encoded = layer(encoded)
        
        # Bottleneck
        bottleneck = self.bottleneck(encoded)
        
        # Decoder
        decoded = bottleneck
        for layer in self.decoder:
            decoded = layer(decoded)
        
        return decoded

class ContrastEnhancementNetwork(nn.Module):
    """대비 향상 네트워크"""
    
    def __init__(self, input_channels: int = 3):
        super().__init__()
        self.input_channels = input_channels
        
        # 대비 향상을 위한 네트워크
        self.enhancement_net = nn.Sequential(
            nn.Conv2d(input_channels, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, input_channels, 3, padding=1),
            nn.Tanh()
        )
        
        # 적응형 대비 조정
        self.adaptive_contrast = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(input_channels, 32, 1),
            nn.ReLU(),
            nn.Conv2d(32, input_channels, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # 대비 향상
        enhanced = self.enhancement_net(x)
        
        # 적응형 대비 조정
        contrast_weights = self.adaptive_contrast(x)
        adjusted = enhanced * contrast_weights
        
        return adjusted

class EdgeDetectionNetwork(nn.Module):
    """엣지 검출 네트워크"""
    
    def __init__(self, input_channels: int = 3):
        super().__init__()
        self.input_channels = input_channels
        
        # Sobel 필터 기반 엣지 검출
        self.sobel_x = nn.Conv2d(input_channels, input_channels, 3, padding=1, bias=False)
        self.sobel_y = nn.Conv2d(input_channels, input_channels, 3, padding=1, bias=False)
        
        # Sobel 커널 초기화
        self._init_sobel_kernels()
        
        # 엣지 강화 네트워크
        self.edge_enhancement = nn.Sequential(
            nn.Conv2d(input_channels * 2, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, input_channels, 3, padding=1),
            nn.Sigmoid()
        )
        
    def _init_sobel_kernels(self):
        """Sobel 커널을 초기화합니다."""
        # Sobel X 커널
        sobel_x_kernel = torch.tensor([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
        # Sobel Y 커널
        sobel_y_kernel = torch.tensor([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
        # 모든 채널에 적용
        for i in range(self.input_channels):
            self.sobel_x.weight.data[i, i] = sobel_x_kernel
            self.sobel_y.weight.data[i, i] = sobel_y_kernel
        
        # 가중치 고정
        self.sobel_x.weight.requires_grad = False
        self.sobel_y.weight.requires_grad = False
        
    def forward(self, x):
        # Sobel X 방향 엣지
        edge_x = self.sobel_x(x)
        
        # Sobel Y 방향 엣지
        edge_y = self.sobel_y(x)
        
        # 엣지 결합
        edges = torch.cat([edge_x, edge_y], dim=1)
        
        # 엣지 강화
        enhanced_edges = self.edge_enhancement(edges)
        
        return enhanced_edges

class GeometricCorrectionNetwork(nn.Module):
    """기하학적 보정 네트워크"""
    
    def __init__(self, input_channels: int = 3):
        super().__init__()
        self.input_channels = input_channels
        
        # 기하학적 보정을 위한 네트워크
        self.correction_net = nn.Sequential(
            nn.Conv2d(input_channels, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, input_channels, 3, padding=1),
            nn.Tanh()
        )
        
        # 공간 변형 필드
        self.spatial_transform = nn.Sequential(
            nn.Conv2d(input_channels, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 2, 3, padding=1),  # 2 channels for x, y offsets
            nn.Tanh()
        )
        
    def forward(self, x):
        # 기하학적 보정
        corrected = self.correction_net(x)
        
        # 공간 변형 필드 생성
        transform_field = self.spatial_transform(x)
        
        # 변형 필드 적용 (간단한 구현)
        batch_size, channels, height, width = x.shape
        
        # 그리드 생성 (배치 차원 포함)
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, height, device=x.device),
            torch.linspace(-1, 1, width, device=x.device),
            indexing='ij'
        )
        
        # 배치 차원 추가
        grid_x = grid_x.unsqueeze(0).expand(batch_size, -1, -1)
        grid_y = grid_y.unsqueeze(0).expand(batch_size, -1, -1)
        
        # 변형 필드 적용
        grid_x = grid_x + transform_field[:, 0, :, :] * 0.1
        grid_y = grid_y + transform_field[:, 1, :, :] * 0.1
        
        # 그리드 정규화
        grid = torch.stack([grid_x, grid_y], dim=-1)
        
        # 변형 적용
        transformed = F.grid_sample(corrected, grid, mode='bilinear', align_corners=False)
        
        return transformed

class GeometricMatchingPreprocessor(nn.Module):
    """기하학적 매칭 전처리기"""
    
    def __init__(self, config: PreprocessingConfig = None):
        super().__init__()
        self.config = config or PreprocessingConfig()
        self.logger = logging.getLogger(__name__)
        
        # MPS 디바이스 확인
        self.device = torch.device("mps" if torch.backends.mps.is_available() and self.config.use_mps else "cpu")
        self.logger.info(f"🎯 Geometric Matching 전처리기 초기화 (디바이스: {self.device})")
        
        # 이미지 정규화
        self.normalizer = ImageNormalizer(
            self.config.normalize_mean, 
            self.config.normalize_std
        ).to(self.device)
        
        # 노이즈 제거 네트워크
        if self.config.enable_noise_reduction:
            self.noise_reduction_net = NoiseReductionNetwork(3).to(self.device)
        
        # 대비 향상 네트워크
        if self.config.enable_contrast_enhancement:
            self.contrast_enhancement_net = ContrastEnhancementNetwork(3).to(self.device)
        
        # 엣지 검출 네트워크
        if self.config.enable_edge_detection:
            self.edge_detection_net = EdgeDetectionNetwork(3).to(self.device)
        
        # 기하학적 보정 네트워크
        if self.config.enable_geometric_correction:
            self.geometric_correction_net = GeometricCorrectionNetwork(3).to(self.device)
        
        # 최종 출력 조정
        self.output_adjustment = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 3, 3, padding=1),
            nn.Tanh()
        ).to(self.device)
        
        self.logger.info("✅ Geometric Matching 전처리기 초기화 완료")
    
    def forward(self, image: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        이미지를 전처리합니다.
        
        Args:
            image: 입력 이미지 (B, C, H, W)
            
        Returns:
            전처리된 결과 딕셔너리
        """
        batch_size, channels, height, width = image.shape
        
        # 입력 검증
        if channels != 3:
            raise ValueError(f"Expected 3 channels, got {channels}")
        
        # 노이즈 제거
        if self.config.enable_noise_reduction:
            denoised = self.noise_reduction_net(image)
            self.logger.debug("노이즈 제거 완료")
        else:
            denoised = image
        
        # 대비 향상
        if self.config.enable_contrast_enhancement:
            enhanced = self.contrast_enhancement_net(denoised)
            self.logger.debug("대비 향상 완료")
        else:
            enhanced = denoised
        
        # 엣지 검출
        if self.config.enable_edge_detection:
            edges = self.edge_detection_net(enhanced)
            self.logger.debug("엣지 검출 완료")
        else:
            edges = enhanced
        
        # 기하학적 보정
        if self.config.enable_geometric_correction:
            corrected = self.geometric_correction_net(edges)
            self.logger.debug("기하학적 보정 완료")
        else:
            corrected = edges
        
        # 최종 출력 조정
        output = self.output_adjustment(corrected)
        
        # 목표 해상도로 리사이즈
        if output.shape[-2:] != self.config.output_size:
            output = F.interpolate(
                output, 
                size=self.config.output_size, 
                mode='bilinear', 
                align_corners=False
            )
        
        # 정규화 적용
        normalized = self.normalizer(output)
        
        # 결과 반환
        result = {
            'preprocessed_image': normalized,
            'denoised_image': denoised,
            'enhanced_image': enhanced,
            'edge_image': edges,
            'corrected_image': corrected,
            'output_size': self.config.output_size,
            'input_size': (height, width)
        }
        
        return result
    
    def process_batch(self, batch_images: List[torch.Tensor]) -> List[Dict[str, torch.Tensor]]:
        """
        배치 단위로 전처리를 수행합니다.
        
        Args:
            batch_images: 이미지 배치 리스트
            
        Returns:
            전처리된 결과 배치 리스트
        """
        results = []
        
        for i, image in enumerate(batch_images):
            try:
                result = self.forward(image)
                results.append(result)
                self.logger.debug(f"배치 {i} 전처리 완료")
            except Exception as e:
                self.logger.error(f"배치 {i} 전처리 실패: {e}")
                # 에러 발생 시 원본 이미지 반환
                results.append({
                    'preprocessed_image': image,
                    'denoised_image': image,
                    'enhanced_image': image,
                    'edge_image': image,
                    'corrected_image': image,
                    'output_size': image.shape[-2:],
                    'input_size': image.shape[-2:]
                })
        
        return results
    
    def get_preprocessing_stats(self) -> Dict[str, Any]:
        """전처리 통계를 반환합니다."""
        return {
            'noise_reduction_enabled': self.config.enable_noise_reduction,
            'contrast_enhancement_enabled': self.config.enable_contrast_enhancement,
            'edge_detection_enabled': self.config.enable_edge_detection,
            'geometric_correction_enabled': self.config.enable_geometric_correction,
            'input_size': self.config.input_size,
            'output_size': self.config.output_size,
            'device': str(self.device),
            'config': self.config.__dict__
        }

# 사용 예시
if __name__ == "__main__":
    # 설정
    config = PreprocessingConfig(
        input_size=(256, 256),
        output_size=(256, 256),
        enable_augmentation=True,
        enable_noise_reduction=True,
        enable_contrast_enhancement=True,
        enable_edge_detection=True,
        enable_geometric_correction=True,
        use_mps=True
    )
    
    # 전처리기 초기화
    preprocessor = GeometricMatchingPreprocessor(config)
    
    # 테스트 입력
    batch_size = 2
    channels = 3
    height = 256
    width = 256
    
    test_image = torch.randn(batch_size, channels, height, width)
    
    # 전처리 수행
    with torch.no_grad():
        result = preprocessor(test_image)
        
        print("✅ 전처리 완료!")
        print(f"입력 형태: {test_image.shape}")
        print(f"출력 형태: {result['preprocessed_image'].shape}")
        print(f"출력 크기: {result['output_size']}")
        print(f"전처리 통계: {preprocessor.get_preprocessing_stats()}")
