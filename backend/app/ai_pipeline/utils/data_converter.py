"""
데이터 변환기 - 이미지/텐서 변환
"""

import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from typing import Union

class DataConverter:
    """데이터 형식 변환 클래스"""
    
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def image_to_tensor(self, image: Image.Image, size: int = 512) -> torch.Tensor:
        """PIL Image를 Tensor로 변환"""
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # 크기 조정 및 정규화
        transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
        ])
        
        tensor = transform(image).unsqueeze(0)  # 배치 차원 추가
        return tensor
    
    def tensor_to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        """Tensor를 NumPy 배열로 변환"""
        if tensor.dim() == 4:  # 배치 차원 제거
            tensor = tensor.squeeze(0)
        
        # [-1, 1] -> [0, 255] 변환
        tensor = torch.clamp(tensor, 0, 1)
        numpy_array = tensor.permute(1, 2, 0).cpu().numpy()
        return (numpy_array * 255).astype(np.uint8)
    
    def numpy_to_tensor(self, array: np.ndarray, device: str = "mps") -> torch.Tensor:
        """NumPy 배열을 Tensor로 변환"""
        if array.dtype == np.uint8:
            array = array.astype(np.float32) / 255.0
        
        tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
        return tensor.to(device)
