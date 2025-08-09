#!/usr/bin/env python3
"""
🔥 MyCloset AI - Step 05: Cloth Warping - Processing Utils
=========================================================

옷감 변형 처리를 위한 유틸리티 함수들을 정의합니다.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Dict, Any, Optional, Tuple, List, Union
from PIL import Image

def preprocess_image(image: Union[np.ndarray, Image.Image, torch.Tensor], 
                    target_size: Tuple[int, int] = (768, 1024)) -> torch.Tensor:
    """이미지 전처리"""
    if isinstance(image, np.ndarray):
        # BGR to RGB
        if image.shape[-1] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
    
    if isinstance(image, Image.Image):
        # 리사이즈
        image = image.resize(target_size, Image.LANCZOS)
        # PIL to tensor
        image = torch.from_numpy(np.array(image)).float() / 255.0
        image = image.permute(2, 0, 1).unsqueeze(0)
    
    if isinstance(image, torch.Tensor):
        if image.dim() == 3:
            image = image.unsqueeze(0)
        if image.size(-1) != target_size[1] or image.size(-2) != target_size[0]:
            image = F.interpolate(image, size=target_size, mode='bilinear', align_corners=True)
    
    return image

def postprocess_image(tensor: torch.Tensor) -> np.ndarray:
    """이미지 후처리"""
    # Tensor to numpy
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    
    # Normalize to [0, 255]
    if tensor.max() <= 1.0:
        tensor = tensor * 255.0
    
    # Convert to numpy
    image = tensor.detach().cpu().numpy()
    image = np.clip(image, 0, 255).astype(np.uint8)
    
    # CHW to HWC
    if image.shape[0] == 3:
        image = np.transpose(image, (1, 2, 0))
    
    return image

def calculate_quality_metrics(original: np.ndarray, warped: np.ndarray) -> Dict[str, float]:
    """품질 메트릭 계산"""
    metrics = {}
    
    # PSNR 계산
    mse = np.mean((original.astype(np.float32) - warped.astype(np.float32)) ** 2)
    if mse > 0:
        metrics['psnr'] = 20 * np.log10(255.0 / np.sqrt(mse))
    else:
        metrics['psnr'] = float('inf')
    
    # SSIM 계산 (간단한 버전)
    metrics['ssim'] = calculate_ssim(original, warped)
    
    # 구조적 유사성
    metrics['structural_similarity'] = calculate_structural_similarity(original, warped)
    
    return metrics

def calculate_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """SSIM 계산 (간단한 구현)"""
    # 그레이스케일 변환
    if img1.shape[-1] == 3:
        img1_gray = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
        img2_gray = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    else:
        img1_gray = img1
        img2_gray = img2
    
    # 간단한 SSIM 계산
    mu1 = np.mean(img1_gray)
    mu2 = np.mean(img2_gray)
    sigma1 = np.var(img1_gray)
    sigma2 = np.var(img2_gray)
    sigma12 = np.mean((img1_gray - mu1) * (img2_gray - mu2))
    
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    
    ssim = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / ((mu1 ** 2 + mu2 ** 2 + c1) * (sigma1 + sigma2 + c2))
    
    return float(ssim)

def calculate_structural_similarity(img1: np.ndarray, img2: np.ndarray) -> float:
    """구조적 유사성 계산"""
    # 그레이스케일 변환
    if img1.shape[-1] == 3:
        img1_gray = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
        img2_gray = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    else:
        img1_gray = img1
        img2_gray = img2
    
    # 구조적 유사성 계산
    structural_sim = np.corrcoef(img1_gray.flatten(), img2_gray.flatten())[0, 1]
    
    return float(structural_sim) if not np.isnan(structural_sim) else 0.0

def apply_transformation_matrix(image: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """변형 행렬 적용"""
    if matrix.shape == (3, 3):
        # 호모그래피 변형
        warped = cv2.warpPerspective(image, matrix, (image.shape[1], image.shape[0]))
    elif matrix.shape == (2, 3):
        # 어파인 변형
        warped = cv2.warpAffine(image, matrix, (image.shape[1], image.shape[0]))
    else:
        raise ValueError(f"Unsupported matrix shape: {matrix.shape}")
    
    return warped

def create_regular_grid(height: int, width: int, device: torch.device) -> torch.Tensor:
    """정규 그리드 생성"""
    y_coords = torch.linspace(-1, 1, height, device=device)
    x_coords = torch.linspace(-1, 1, width, device=device)
    grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')
    grid = torch.stack([grid_x, grid_y], dim=-1)
    return grid

def apply_grid_warping(image: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    """그리드 변형 적용"""
    batch_size = image.size(0)
    grid = grid.unsqueeze(0).expand(batch_size, -1, -1, -1)
    
    warped = F.grid_sample(image, grid, mode='bilinear', padding_mode='border', align_corners=True)
    return warped

def calculate_flow_confidence(flow: torch.Tensor) -> torch.Tensor:
    """플로우 신뢰도 계산"""
    # 플로우 크기 계산
    flow_magnitude = torch.norm(flow, dim=1, keepdim=True)
    
    # 신뢰도 계산 (간단한 메트릭)
    confidence = torch.exp(-flow_magnitude / 10.0)
    
    return confidence

def validate_transformation_matrix(matrix: np.ndarray) -> bool:
    """변형 행렬 유효성 검증"""
    if matrix is None:
        return False
    
    if matrix.shape not in [(2, 3), (3, 3)]:
        return False
    
    # 행렬식이 0이 아닌지 확인
    if matrix.shape == (3, 3):
        det = np.linalg.det(matrix)
        if abs(det) < 1e-6:
            return False
    
    return True

def normalize_keypoints(keypoints: np.ndarray, image_shape: Tuple[int, int]) -> np.ndarray:
    """키포인트 정규화"""
    if keypoints is None or len(keypoints) == 0:
        return np.array([])
    
    height, width = image_shape
    normalized = keypoints.copy()
    normalized[:, 0] = (normalized[:, 0] / width) * 2 - 1  # x 좌표
    normalized[:, 1] = (normalized[:, 1] / height) * 2 - 1  # y 좌표
    
    return normalized

def denormalize_keypoints(keypoints: np.ndarray, image_shape: Tuple[int, int]) -> np.ndarray:
    """키포인트 역정규화"""
    if keypoints is None or len(keypoints) == 0:
        return np.array([])
    
    height, width = image_shape
    denormalized = keypoints.copy()
    denormalized[:, 0] = (denormalized[:, 0] + 1) / 2 * width  # x 좌표
    denormalized[:, 1] = (denormalized[:, 1] + 1) / 2 * height  # y 좌표
    
    return denormalized
