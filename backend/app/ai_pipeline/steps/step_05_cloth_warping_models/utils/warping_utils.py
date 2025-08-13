#!/usr/bin/env python3
"""
🔥 Cloth Warping Utils
=======================

의류 워핑을 위한 유틸리티 함수들
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from PIL import Image

class WarpingUtils:
    """의류 워핑 유틸리티 클래스"""
    
    @staticmethod
    def create_warping_grid(height: int, width: int, device: str = "cpu") -> torch.Tensor:
        """워핑을 위한 그리드 생성"""
        y_coords, x_coords = torch.meshgrid(
            torch.linspace(-1, 1, height, device=device),
            torch.linspace(-1, 1, width, device=device),
            indexing='ij'
        )
        grid = torch.stack([x_coords, y_coords], dim=-1)
        return grid.unsqueeze(0)  # (1, H, W, 2)
    
    @staticmethod
    def apply_warping_transform(image: torch.Tensor, transform_matrix: torch.Tensor) -> torch.Tensor:
        """이미지에 워핑 변형 적용"""
        batch_size, channels, height, width = image.shape
        
        # 그리드 생성
        grid = WarpingUtils.create_warping_grid(height, width, image.device)
        grid = grid.expand(batch_size, -1, -1, -1)
        
        # 변형 적용
        warped_grid = torch.matmul(grid.view(batch_size, -1, 2), transform_matrix.transpose(-2, -1))
        warped_grid = warped_grid.view(batch_size, height, width, 2)
        
        # 이미지 워핑
        warped_image = F.grid_sample(image, warped_grid, mode='bilinear', padding_mode='zeros', align_corners=True)
        return warped_image
    
    @staticmethod
    def calculate_warping_loss(original: torch.Tensor, warped: torch.Tensor) -> torch.Tensor:
        """워핑 손실 계산"""
        # L1 손실
        l1_loss = F.l1_loss(original, warped)
        
        # SSIM 손실 (간단한 버전)
        ssim_loss = 1 - WarpingUtils._simple_ssim(original, warped)
        
        # 전체 손실
        total_loss = l1_loss + 0.1 * ssim_loss
        return total_loss
    
    @staticmethod
    def _simple_ssim(x: torch.Tensor, y: torch.Tensor, window_size: int = 11) -> torch.Tensor:
        """간단한 SSIM 계산"""
        # 간단한 구현
        mu_x = F.avg_pool2d(x, window_size, stride=1, padding=window_size//2)
        mu_y = F.avg_pool2d(y, window_size, stride=1, padding=window_size//2)
        
        mu_x_sq = mu_x.pow(2)
        mu_y_sq = mu_y.pow(2)
        mu_xy = mu_x * mu_y
        
        sigma_x_sq = F.avg_pool2d(x.pow(2), window_size, stride=1, padding=window_size//2) - mu_x_sq
        sigma_y_sq = F.avg_pool2d(y.pow(2), window_size, stride=1, padding=window_size//2) - mu_y_sq
        sigma_xy = F.avg_pool2d(x * y, window_size, stride=1, padding=window_size//2) - mu_xy
        
        c1 = 0.01 ** 2
        c2 = 0.03 ** 2
        
        ssim = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / ((mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2))
        return ssim.mean()

class ClothWarpingProcessor:
    """의류 워핑 프로세서"""
    
    def __init__(self, device: str = "cpu"):
        self.device = device
        self.utils = WarpingUtils()
    
    def process_warping(self, person_image: torch.Tensor, cloth_image: torch.Tensor, 
                       pose_keypoints: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """의류 워핑 처리"""
        try:
            # 기본 워핑 (간단한 구현)
            warped_cloth = self._basic_warping(cloth_image, person_image)
            
            # 품질 평가
            quality_score = self._assess_warping_quality(person_image, warped_cloth)
            
            return {
                'success': True,
                'warped_cloth': warped_cloth,
                'quality_score': quality_score,
                'processing_info': {
                    'method': 'basic_warping',
                    'device': self.device
                }
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'processing_info': {
                    'method': 'basic_warping',
                    'device': self.device
                }
            }
    
    def _basic_warping(self, cloth_image: torch.Tensor, person_image: torch.Tensor) -> torch.Tensor:
        """기본 워핑 (간단한 리사이즈)"""
        # 의류 이미지를 사람 이미지 크기에 맞춤
        if cloth_image.shape[-2:] != person_image.shape[-2:]:
            warped_cloth = F.interpolate(
                cloth_image, 
                size=person_image.shape[-2:], 
                mode='bilinear', 
                align_corners=True
            )
        else:
            warped_cloth = cloth_image
        
        return warped_cloth
    
    def _assess_warping_quality(self, person_image: torch.Tensor, warped_cloth: torch.Tensor) -> float:
        """워핑 품질 평가"""
        try:
            # 간단한 품질 메트릭
            loss = self.utils.calculate_warping_loss(person_image, warped_cloth)
            quality_score = max(0.0, 1.0 - loss.item())
            return quality_score
        except:
            return 0.5  # 기본값

# 편의 함수들
def create_warping_processor(device: str = "cpu") -> ClothWarpingProcessor:
    """워핑 프로세서 생성"""
    return ClothWarpingProcessor(device)

def apply_basic_warping(cloth_image: torch.Tensor, target_size: Tuple[int, int]) -> torch.Tensor:
    """기본 워핑 적용"""
    if cloth_image.shape[-2:] != target_size:
        warped = F.interpolate(
            cloth_image, 
            size=target_size, 
            mode='bilinear', 
            align_corners=True
        )
        return warped
    return cloth_image
