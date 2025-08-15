#!/usr/bin/env python3
"""
🔥 MyCloset AI - Geometric Matching Preprocessor
================================================

🎯 기하학적 매칭 전처리기
✅ 이미지 정규화 및 전처리
✅ 특징 추출 및 강화
✅ 품질 향상 및 노이즈 제거
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
    use_mps: bool = True

class GeometricMatchingPreprocessor(nn.Module):
    """기하학적 매칭 전처리기"""
    
    def __init__(self, config: PreprocessingConfig = None):
        super().__init__()
        self.config = config or PreprocessingConfig()
        self.logger = logging.getLogger(__name__)
        
        # MPS 디바이스 확인
        self.device = torch.device("mps" if torch.backends.mps.is_available() and self.config.use_mps else "cpu")
        self.logger.info(f"🎯 Geometric Matching 전처리기 초기화 (디바이스: {self.device})")
        
        # 노이즈 제거 네트워크
        if self.config.enable_noise_reduction:
            self.noise_reduction_net = self._create_noise_reduction_net()
        
        # 특징 강화 네트워크
        self.feature_enhancement_net = self._create_feature_enhancement_net()
        
        # 품질 향상 네트워크
        self.quality_enhancement_net = self._create_quality_enhancement_net()
        
        # 정규화 레이어
        self.normalize = nn.Parameter(
            torch.tensor([self.config.normalize_mean, self.config.normalize_std]), 
            requires_grad=False
        )
        
        self.logger.info("✅ Geometric Matching 전처리기 초기화 완료")
    
    def _create_noise_reduction_net(self) -> nn.Module:
        """노이즈 제거 네트워크 생성"""
        return nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
            nn.Sigmoid()
        ).to(self.device)
    
    def _create_feature_enhancement_net(self) -> nn.Module:
        """특징 강화 네트워크 생성"""
        return nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
            nn.Tanh()
        ).to(self.device)
    
    def _create_quality_enhancement_net(self) -> nn.Module:
        """품질 향상 네트워크 생성"""
        return nn.Sequential(
            nn.Conv2d(6, 128, kernel_size=3, padding=1),  # 6 channels: 3 for image1 + 3 for image2
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 6, kernel_size=3, padding=1),  # Output: enhanced image1 + enhanced image2
            nn.Sigmoid()
        ).to(self.device)
    
    def forward(self, image1: torch.Tensor, image2: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        이미지 전처리 수행
        
        Args:
            image1: 첫 번째 이미지 (B, C, H, W)
            image2: 두 번째 이미지 (B, C, H, W)
        
        Returns:
            전처리된 이미지들
        """
        # 입력 검증
        if not self._validate_inputs(image1, image2):
            raise ValueError("입력 검증 실패")
        
        # 디바이스 이동
        image1 = image1.to(self.device)
        image2 = image2.to(self.device)
        
        # 1단계: 기본 전처리
        processed_image1 = self._basic_preprocessing(image1)
        processed_image2 = self._basic_preprocessing(image2)
        
        # 2단계: 노이즈 제거
        if self.config.enable_noise_reduction:
            processed_image1 = self._reduce_noise(processed_image1)
            processed_image2 = self._reduce_noise(processed_image2)
        
        # 3단계: 특징 강화
        enhanced_image1 = self._enhance_features(processed_image1)
        enhanced_image2 = self._enhance_features(processed_image2)
        
        # 4단계: 품질 향상
        quality_enhanced_images = self._enhance_quality(enhanced_image1, enhanced_image2)
        
        # 5단계: 최종 정규화
        final_image1 = self._final_normalization(quality_enhanced_images[:, :3, :, :])
        final_image2 = self._final_normalization(quality_enhanced_images[:, 3:, :, :])
        
        # 6단계: 크기 조정
        if self.config.input_size != self.config.output_size:
            final_image1 = F.interpolate(final_image1, size=self.config.output_size, 
                                       mode='bilinear', align_corners=False)
            final_image2 = F.interpolate(final_image2, size=self.config.output_size, 
                                       mode='bilinear', align_corners=False)
        
        # 결과 반환
        result = {
            "processed_image1": final_image1,
            "processed_image2": final_image2,
            "intermediate_features": {
                "enhanced_image1": enhanced_image1,
                "enhanced_image2": enhanced_image2,
                "quality_enhanced": quality_enhanced_images
            }
        }
        
        return result
    
    def _validate_inputs(self, image1: torch.Tensor, image2: torch.Tensor) -> bool:
        """입력 검증"""
        if image1.dim() != 4 or image2.dim() != 4:
            return False
        
        if image1.size(0) != image2.size(0):
            return False
        
        if image1.size(2) != image2.size(2) or image1.size(3) != image2.size(3):
            return False
        
        if image1.size(1) != 3 or image2.size(1) != 3:
            return False
        
        return True
    
    def _basic_preprocessing(self, image: torch.Tensor) -> torch.Tensor:
        """기본 전처리"""
        # 1. 픽셀 값 정규화 (0-1 범위)
        if image.max() > 1.0:
            image = image / 255.0
        
        # 2. 대비 향상
        if self.config.enable_contrast_enhancement:
            image = self._enhance_contrast(image)
        
        # 3. 엣지 검출
        if self.config.enable_edge_detection:
            image = self._detect_edges(image)
        
        return image
    
    def _enhance_contrast(self, image: torch.Tensor) -> torch.Tensor:
        """대비 향상"""
        # 히스토그램 평활화
        batch_size, channels, height, width = image.shape
        enhanced_image = torch.zeros_like(image)
        
        for b in range(batch_size):
            for c in range(channels):
                # 각 채널별로 히스토그램 평활화
                channel = image[b, c, :, :]
                
                # 히스토그램 계산
                hist = torch.histc(channel, bins=256, min=0, max=1)
                cdf = torch.cumsum(hist, dim=0)
                cdf_normalized = cdf / cdf.max()
                
                # LUT 적용
                enhanced_channel = cdf_normalized[(channel * 255).long()]
                enhanced_image[b, c, :, :] = enhanced_channel
        
        return enhanced_image
    
    def _detect_edges(self, image: torch.Tensor) -> torch.Tensor:
        """엣지 검출"""
        # Sobel 엣지 검출
        batch_size, channels, height, width = image.shape
        edge_image = torch.zeros_like(image)
        
        # Sobel 커널
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                              dtype=torch.float32, device=self.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                              dtype=torch.float32, device=self.device).view(1, 1, 3, 3)
        
        for b in range(batch_size):
            for c in range(channels):
                channel = image[b:b+1, c:c+1, :, :]
                
                # X 방향 엣지
                edge_x = F.conv2d(channel, sobel_x, padding=1)
                
                # Y 방향 엣지
                edge_y = F.conv2d(channel, sobel_y, padding=1)
                
                # 엣지 강도
                edge_magnitude = torch.sqrt(edge_x ** 2 + edge_y ** 2)
                
                # 엣지 강화
                enhanced_channel = image[b, c, :, :] + 0.1 * edge_magnitude.squeeze()
                edge_image[b, c, :, :] = torch.clamp(enhanced_channel, 0, 1)
        
        return edge_image
    
    def _reduce_noise(self, image: torch.Tensor) -> torch.Tensor:
        """노이즈 제거"""
        # 1. 가우시안 블러로 노이즈 제거
        blurred = F.avg_pool2d(image, kernel_size=3, stride=1, padding=1)
        
        # 2. 노이즈 제거 네트워크 적용
        noise_reduced = self.noise_reduction_net(blurred)
        
        # 3. 원본과 노이즈 제거된 이미지 결합
        denoised = 0.7 * image + 0.3 * noise_reduced
        
        return torch.clamp(denoised, 0, 1)
    
    def _enhance_features(self, image: torch.Tensor) -> torch.Tensor:
        """특징 강화"""
        # 특징 강화 네트워크 적용
        enhanced = self.feature_enhancement_net(image)
        
        # 원본과 강화된 특징 결합
        enhanced_features = 0.8 * image + 0.2 * enhanced
        
        return torch.clamp(enhanced_features, 0, 1)
    
    def _enhance_quality(self, image1: torch.Tensor, image2: torch.Tensor) -> torch.Tensor:
        """품질 향상"""
        # 두 이미지 결합
        combined_input = torch.cat([image1, image2], dim=1)
        
        # 품질 향상 네트워크 적용
        quality_enhanced = self.quality_enhancement_net(combined_input)
        
        return quality_enhanced
    
    def _final_normalization(self, image: torch.Tensor) -> torch.Tensor:
        """최종 정규화"""
        # ImageNet 정규화 적용
        mean = self.normalize[0].view(1, 3, 1, 1)
        std = self.normalize[1].view(1, 3, 1, 1)
        
        normalized = (image - mean) / std
        
        return normalized
    
    def preprocess_single_image(self, image: torch.Tensor) -> torch.Tensor:
        """단일 이미지 전처리"""
        if image.dim() == 3:
            image = image.unsqueeze(0)  # Add batch dimension
        
        # 더미 이미지로 전처리 수행
        dummy_image = torch.zeros_like(image)
        result = self.forward(image, dummy_image)
        
        return result["processed_image1"]
    
    def preprocess_batch(self, images: List[torch.Tensor]) -> List[torch.Tensor]:
        """배치 이미지 전처리"""
        if not images:
            return []
        
        # 첫 번째 이미지로 배치 크기 확인
        batch_size = len(images)
        first_image = images[0]
        
        # 배치 텐서 생성
        batch_tensor = torch.stack(images)
        
        # 더미 이미지로 전처리 수행
        dummy_batch = torch.zeros_like(batch_tensor)
        result = self.forward(batch_tensor, dummy_batch)
        
        # 결과를 개별 이미지로 분리
        processed_images = []
        for i in range(batch_size):
            processed_images.append(result["processed_image1"][i])
        
        return processed_images
    
    def get_preprocessing_info(self) -> Dict[str, Any]:
        """전처리 정보 반환"""
        return {
            "input_size": self.config.input_size,
            "output_size": self.config.output_size,
            "normalize_mean": self.config.normalize_mean,
            "normalize_std": self.config.normalize_std,
            "enable_augmentation": self.config.enable_augmentation,
            "enable_noise_reduction": self.config.enable_noise_reduction,
            "enable_contrast_enhancement": self.config.enable_contrast_enhancement,
            "enable_edge_detection": self.config.enable_edge_detection,
            "device": str(self.device),
            "total_parameters": sum(p.numel() for p in self.parameters()),
            "trainable_parameters": sum(p.numel() for p in self.parameters() if p.requires_grad)
        }

# 전처리기 인스턴스 생성
def create_geometric_matching_preprocessor(config: PreprocessingConfig = None) -> GeometricMatchingPreprocessor:
    """Geometric Matching 전처리기 생성"""
    return GeometricMatchingPreprocessor(config)

if __name__ == "__main__":
    # 테스트 코드
    logging.basicConfig(level=logging.INFO)
    
    # 전처리기 생성
    preprocessor = create_geometric_matching_preprocessor()
    
    # 테스트 데이터 생성
    batch_size, channels, height, width = 2, 3, 256, 256
    test_image1 = torch.randn(batch_size, channels, height, width)
    test_image2 = torch.randn(batch_size, channels, height, width)
    
    # 전처리 수행
    result = preprocessor(test_image1, test_image2)
    
    print(f"전처리된 이미지1 형태: {result['processed_image1'].shape}")
    print(f"전처리된 이미지2 형태: {result['processed_image2'].shape}")
    
    # 전처리 정보 출력
    preprocess_info = preprocessor.get_preprocessing_info()
    print(f"전처리 정보: {preprocess_info}")
    
    # 단일 이미지 전처리 테스트
    single_image = torch.randn(channels, height, width)
    processed_single = preprocessor.preprocess_single_image(single_image)
    print(f"단일 이미지 전처리 결과 형태: {processed_single.shape}")
    
    # 배치 전처리 테스트
    image_list = [torch.randn(channels, height, width) for _ in range(3)]
    processed_batch = preprocessor.preprocess_batch(image_list)
    print(f"배치 전처리 결과 개수: {len(processed_batch)}")
    for i, img in enumerate(processed_batch):
        print(f"  이미지 {i+1} 형태: {img.shape}")
