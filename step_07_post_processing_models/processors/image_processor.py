"""
Image Processor for Post Processing Models

후처리 모델들을 위한 이미지 프로세서입니다.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Optional, Union, Tuple
import numpy as np
from PIL import Image
import cv2
import logging

# 프로젝트 로깅 설정 import
from backend.app.ai_pipeline.utils.logging_config import get_logger

logger = get_logger(__name__)


class ImageProcessor:
    """
    후처리 모델들을 위한 이미지 프로세서
    
    이미지 전처리, 후처리, 변환 등의 기능을 제공합니다.
    """
    
    def __init__(self, device: Optional[torch.device] = None):
        """
        Args:
            device: 사용할 디바이스
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 이미지 처리 설정
        self.processing_config = {
            'normalize': True,
            'resize_method': 'bilinear',
            'interpolation_mode': 'bilinear',
            'padding_mode': 'reflect',
            'align_corners': False
        }
        
        logger.info(f"ImageProcessor initialized on device: {self.device}")
    
    def preprocess_image(self, image: Union[np.ndarray, Image.Image, torch.Tensor], 
                        target_size: Optional[Tuple[int, int]] = None,
                        normalize: bool = True,
                        **kwargs) -> torch.Tensor:
        """
        이미지를 전처리합니다.
        
        Args:
            image: 입력 이미지
            target_size: 목표 크기 (H, W)
            normalize: 정규화 여부
            **kwargs: 추가 파라미터
            
        Returns:
            전처리된 이미지 텐서
        """
        try:
            # 이미지를 텐서로 변환
            if isinstance(image, np.ndarray):
                image_tensor = torch.from_numpy(image).float()
            elif isinstance(image, Image.Image):
                image_tensor = torch.from_numpy(np.array(image)).float()
            elif isinstance(image, torch.Tensor):
                image_tensor = image.float()
            else:
                raise ValueError(f"Unsupported image type: {type(image)}")
            
            # 차원 확인 및 조정
            if image_tensor.dim() == 2:  # Grayscale
                image_tensor = image_tensor.unsqueeze(0)  # (1, H, W)
            elif image_tensor.dim() == 3:
                if image_tensor.shape[0] == 3:  # (C, H, W)
                    pass
                else:  # (H, W, C)
                    image_tensor = image_tensor.permute(2, 0, 1)
            elif image_tensor.dim() == 4:  # (B, C, H, W)
                pass
            else:
                raise ValueError(f"Invalid image tensor dimensions: {image_tensor.shape}")
            
            # 배치 차원 추가 (없는 경우)
            if image_tensor.dim() == 3:
                image_tensor = image_tensor.unsqueeze(0)  # (1, C, H, W)
            
            # 크기 조정
            if target_size is not None:
                image_tensor = self._resize_image(image_tensor, target_size, **kwargs)
            
            # 정규화
            if normalize:
                image_tensor = self._normalize_image(image_tensor, **kwargs)
            
            # 디바이스 이동
            image_tensor = image_tensor.to(self.device)
            
            return image_tensor
            
        except Exception as e:
            logger.error(f"Error in image preprocessing: {str(e)}")
            raise RuntimeError(f"Failed to preprocess image: {str(e)}")
    
    def postprocess_image(self, image_tensor: torch.Tensor, 
                         output_format: str = 'numpy',
                         denormalize: bool = True,
                         **kwargs) -> Union[np.ndarray, Image.Image, torch.Tensor]:
        """
        이미지를 후처리합니다.
        
        Args:
            image_tensor: 입력 이미지 텐서
            output_format: 출력 형식 ('numpy', 'pil', 'tensor')
            denormalize: 역정규화 여부
            **kwargs: 추가 파라미터
            
        Returns:
            후처리된 이미지
        """
        try:
            # 디바이스에서 CPU로 이동
            if image_tensor.device != torch.device('cpu'):
                image_tensor = image_tensor.detach().cpu()
            
            # 역정규화
            if denormalize:
                image_tensor = self._denormalize_image(image_tensor, **kwargs)
            
            # 클램핑
            image_tensor = torch.clamp(image_tensor, 0, 1)
            
            # 배치 차원 제거 (단일 이미지인 경우)
            if image_tensor.dim() == 4 and image_tensor.shape[0] == 1:
                image_tensor = image_tensor.squeeze(0)  # (C, H, W)
            
            # 출력 형식 변환
            if output_format == 'numpy':
                return image_tensor.numpy()
            elif output_format == 'pil':
                return self._tensor_to_pil(image_tensor)
            elif output_format == 'tensor':
                return image_tensor
            else:
                raise ValueError(f"Unsupported output format: {output_format}")
                
        except Exception as e:
            logger.error(f"Error in image postprocessing: {str(e)}")
            raise RuntimeError(f"Failed to postprocess image: {str(e)}")
    
    def _resize_image(self, image_tensor: torch.Tensor, 
                     target_size: Tuple[int, int], 
                     **kwargs) -> torch.Tensor:
        """이미지 크기를 조정합니다."""
        method = kwargs.get('method', self.processing_config['resize_method'])
        mode = kwargs.get('mode', self.processing_config['interpolation_mode'])
        align_corners = kwargs.get('align_corners', self.processing_config['align_corners'])
        
        if method == 'bilinear':
            return F.interpolate(image_tensor, size=target_size, mode='bilinear', 
                               align_corners=align_corners)
        elif method == 'nearest':
            return F.interpolate(image_tensor, size=target_size, mode='nearest')
        elif method == 'bicubic':
            return F.interpolate(image_tensor, size=target_size, mode='bicubic', 
                               align_corners=align_corners)
        else:
            return F.interpolate(image_tensor, size=target_size, mode='bilinear', 
                               align_corners=align_corners)
    
    def _normalize_image(self, image_tensor: torch.Tensor, **kwargs) -> torch.Tensor:
        """이미지를 정규화합니다."""
        normalize_method = kwargs.get('normalize_method', 'standard')
        
        if normalize_method == 'standard':
            # [0, 255] -> [0, 1]
            if image_tensor.max() > 1:
                image_tensor = image_tensor / 255.0
        elif normalize_method == 'imagenet':
            # ImageNet 정규화
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            image_tensor = (image_tensor - mean) / std
        elif normalize_method == 'tanh':
            # [-1, 1] 범위로 정규화
            image_tensor = image_tensor * 2 - 1
        
        return image_tensor
    
    def _denormalize_image(self, image_tensor: torch.Tensor, **kwargs) -> torch.Tensor:
        """이미지 역정규화를 수행합니다."""
        normalize_method = kwargs.get('normalize_method', 'standard')
        
        if normalize_method == 'imagenet':
            # ImageNet 역정규화
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            image_tensor = image_tensor * std + mean
        elif normalize_method == 'tanh':
            # [-1, 1] -> [0, 1] 범위로 변환
            image_tensor = (image_tensor + 1) / 2
        
        return image_tensor
    
    def _tensor_to_pil(self, image_tensor: torch.Tensor) -> Image.Image:
        """텐서를 PIL 이미지로 변환합니다."""
        # (C, H, W) -> (H, W, C)
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.permute(1, 2, 0)
        
        # numpy 배열로 변환
        image_array = image_tensor.numpy()
        
        # [0, 1] -> [0, 255]
        if image_array.max() <= 1:
            image_array = (image_array * 255).astype(np.uint8)
        
        return Image.fromarray(image_array)
    
    def apply_augmentation(self, image_tensor: torch.Tensor, 
                          augmentation_type: str, 
                          **kwargs) -> torch.Tensor:
        """
        이미지 증강을 적용합니다.
        
        Args:
            image_tensor: 입력 이미지 텐서
            augmentation_type: 증강 타입
            **kwargs: 증강 파라미터
            
        Returns:
            증강된 이미지 텐서
        """
        try:
            if augmentation_type == 'horizontal_flip':
                return self._horizontal_flip(image_tensor, **kwargs)
            elif augmentation_type == 'vertical_flip':
                return self._vertical_flip(image_tensor, **kwargs)
            elif augmentation_type == 'rotation':
                return self._rotation(image_tensor, **kwargs)
            elif augmentation_type == 'translation':
                return self._translation(image_tensor, **kwargs)
            elif augmentation_type == 'scaling':
                return self._scaling(image_tensor, **kwargs)
            elif augmentation_type == 'noise':
                return self._add_noise(image_tensor, **kwargs)
            elif augmentation_type == 'blur':
                return self._apply_blur(image_tensor, **kwargs)
            elif augmentation_type == 'sharpening':
                return self._apply_sharpening(image_tensor, **kwargs)
            else:
                raise ValueError(f"Unsupported augmentation type: {augmentation_type}")
                
        except Exception as e:
            logger.error(f"Error in image augmentation: {str(e)}")
            raise RuntimeError(f"Failed to apply augmentation: {str(e)}")
    
    def _horizontal_flip(self, image_tensor: torch.Tensor, **kwargs) -> torch.Tensor:
        """수평 뒤집기를 적용합니다."""
        probability = kwargs.get('probability', 0.5)
        if torch.rand(1).item() < probability:
            return torch.flip(image_tensor, dims=[3])  # W 차원
        return image_tensor
    
    def _vertical_flip(self, image_tensor: torch.Tensor, **kwargs) -> torch.Tensor:
        """수직 뒤집기를 적용합니다."""
        probability = kwargs.get('probability', 0.5)
        if torch.rand(1).item() < probability:
            return torch.flip(image_tensor, dims=[2])  # H 차원
        return image_tensor
    
    def _rotation(self, image_tensor: torch.Tensor, **kwargs) -> torch.Tensor:
        """회전을 적용합니다."""
        angle = kwargs.get('angle', 15)
        angle_rad = torch.tensor(angle * np.pi / 180.0)
        
        # 회전 행렬 생성
        cos_a = torch.cos(angle_rad)
        sin_a = torch.sin(angle_rad)
        
        rotation_matrix = torch.tensor([
            [cos_a, -sin_a, 0],
            [sin_a, cos_a, 0]
        ]).unsqueeze(0).repeat(image_tensor.shape[0], 1, 1)
        
        # 그리드 생성
        grid = F.affine_grid(rotation_matrix, image_tensor.size(), align_corners=False)
        
        # 회전 적용
        return F.grid_sample(image_tensor, grid, align_corners=False)
    
    def _translation(self, image_tensor: torch.Tensor, **kwargs) -> torch.Tensor:
        """이동을 적용합니다."""
        translate_x = kwargs.get('translate_x', 0.1)
        translate_y = kwargs.get('translate_y', 0.1)
        
        # 이동 행렬 생성
        translation_matrix = torch.tensor([
            [1, 0, translate_x],
            [0, 1, translate_y]
        ]).unsqueeze(0).repeat(image_tensor.shape[0], 1, 1)
        
        # 그리드 생성
        grid = F.affine_grid(translation_matrix, image_tensor.size(), align_corners=False)
        
        # 이동 적용
        return F.grid_sample(image_tensor, grid, align_corners=False)
    
    def _scaling(self, image_tensor: torch.Tensor, **kwargs) -> torch.Tensor:
        """스케일링을 적용합니다."""
        scale_x = kwargs.get('scale_x', 1.0)
        scale_y = kwargs.get('scale_y', 1.0)
        
        # 스케일링 행렬 생성
        scaling_matrix = torch.tensor([
            [scale_x, 0, 0],
            [0, scale_y, 0]
        ]).unsqueeze(0).repeat(image_tensor.shape[0], 1, 1)
        
        # 그리드 생성
        grid = F.affine_grid(scaling_matrix, image_tensor.size(), align_corners=False)
        
        # 스케일링 적용
        return F.grid_sample(image_tensor, grid, align_corners=False)
    
    def _add_noise(self, image_tensor: torch.Tensor, **kwargs) -> torch.Tensor:
        """노이즈를 추가합니다."""
        noise_type = kwargs.get('noise_type', 'gaussian')
        intensity = kwargs.get('intensity', 0.1)
        
        if noise_type == 'gaussian':
            noise = torch.randn_like(image_tensor) * intensity
            return image_tensor + noise
        elif noise_type == 'salt_pepper':
            # Salt & Pepper 노이즈
            mask = torch.rand_like(image_tensor)
            image_tensor[mask < intensity/2] = 0
            image_tensor[mask > 1 - intensity/2] = 1
            return image_tensor
        else:
            return image_tensor
    
    def _apply_blur(self, image_tensor: torch.Tensor, **kwargs) -> torch.Tensor:
        """블러를 적용합니다."""
        kernel_size = kwargs.get('kernel_size', 3)
        sigma = kwargs.get('sigma', 1.0)
        
        # 가우시안 커널 생성
        kernel = self._create_gaussian_kernel(kernel_size, sigma)
        kernel = kernel.view(1, 1, kernel_size, kernel_size)
        kernel = kernel.repeat(image_tensor.shape[1], 1, 1, 1)
        
        # 블러 적용
        return F.conv2d(image_tensor, kernel, padding=kernel_size//2, groups=image_tensor.shape[1])
    
    def _apply_sharpening(self, image_tensor: torch.Tensor, **kwargs) -> torch.Tensor:
        """샤프닝을 적용합니다."""
        strength = kwargs.get('strength', 0.5)
        
        # 샤프닝 커널
        kernel = torch.tensor([
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0]
        ], dtype=torch.float32).view(1, 1, 3, 3)
        
        kernel = kernel.repeat(image_tensor.shape[1], 1, 1, 1)
        
        # 샤프닝 적용
        sharpened = F.conv2d(image_tensor, kernel, padding=1, groups=image_tensor.shape[1])
        
        # 원본과 블렌딩
        return image_tensor * (1 - strength) + sharpened * strength
    
    def _create_gaussian_kernel(self, kernel_size: int, sigma: float) -> torch.Tensor:
        """가우시안 커널을 생성합니다."""
        kernel = torch.zeros(kernel_size, kernel_size)
        center = kernel_size // 2
        
        for i in range(kernel_size):
            for j in range(kernel_size):
                x, y = i - center, j - center
                kernel[i, j] = torch.exp(-(x**2 + y**2) / (2 * sigma**2))
        
        return kernel / kernel.sum()
    
    def get_processing_info(self) -> Dict[str, Any]:
        """처리 정보를 반환합니다."""
        return {
            'device': str(self.device),
            'processing_config': self.processing_config
        }
    
    def update_processing_config(self, new_config: Dict[str, Any]):
        """처리 설정을 업데이트합니다."""
        self.processing_config.update(new_config)
        logger.info(f"Updated processing config: {self.processing_config}")
