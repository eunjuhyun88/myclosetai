"""
MyCloset AI 데이터 변환기
다양한 데이터 형식 간 변환 유틸리티
"""

import torch
import numpy as np
from PIL import Image
import cv2
from typing import Union, Tuple, Optional
import torchvision.transforms as transforms

class DataConverter:
    """데이터 형식 변환 클래스"""
    
    def __init__(self):
        # 기본 이미지 변환
        self.to_tensor = transforms.ToTensor()
        self.to_pil = transforms.ToPILImage()
        
        # 정규화 변환
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    
    def image_to_tensor(self, image: Union[Image.Image, np.ndarray], size: Optional[int] = None) -> torch.Tensor:
        """이미지를 텐서로 변환"""
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # 리사이즈
        if size:
            image = image.resize((size, size), Image.Resampling.LANCZOS)
        
        # 텐서 변환
        tensor = self.to_tensor(image)
        
        return tensor.unsqueeze(0)  # 배치 차원 추가
    
    def tensor_to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        """텐서를 numpy 배열로 변환"""
        if tensor.dim() == 4:  # [B, C, H, W]
            tensor = tensor.squeeze(0)  # 배치 차원 제거
        
        if tensor.dim() == 3:  # [C, H, W]
            tensor = tensor.permute(1, 2, 0)  # [H, W, C]
        
        # CPU로 이동 후 numpy 변환
        numpy_array = tensor.detach().cpu().numpy()
        
        # [0, 1] → [0, 255] 변환
        if numpy_array.max() <= 1.0:
            numpy_array = (numpy_array * 255).astype(np.uint8)
        
        return numpy_array
    
    def numpy_to_tensor(self, array: np.ndarray, normalize: bool = True) -> torch.Tensor:
        """numpy 배열을 텐서로 변환"""
        if len(array.shape) == 3:  # [H, W, C]
            array = array.transpose(2, 0, 1)  # [C, H, W]
        
        tensor = torch.from_numpy(array).float()
        
        # 정규화
        if normalize and tensor.max() > 1.0:
            tensor = tensor / 255.0
        
        return tensor.unsqueeze(0)  # 배치 차원 추가
    
    def resize_image(self, image: Union[Image.Image, np.ndarray], size: Tuple[int, int]) -> Union[Image.Image, np.ndarray]:
        """이미지 리사이즈"""
        if isinstance(image, Image.Image):
            return image.resize(size, Image.Resampling.LANCZOS)
        else:
            return cv2.resize(image, size)
    
    def normalize_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """텐서 정규화"""
        return self.normalize(tensor)