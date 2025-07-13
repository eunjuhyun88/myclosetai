"""데이터 변환 유틸리티"""
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

class DataConverter:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
        ])
    
    def image_to_tensor(self, image, size=512):
        """PIL 이미지를 텐서로 변환"""
        if isinstance(image, Image.Image):
            image = image.resize((size, size))
            tensor = self.transform(image)
            return tensor.unsqueeze(0)  # 배치 차원 추가
        return image
    
    def tensor_to_numpy(self, tensor):
        """텐서를 numpy 배열로 변환"""
        if torch.is_tensor(tensor):
            # 배치 차원 제거하고 (C, H, W) -> (H, W, C)로 변환
            tensor = tensor.squeeze(0) if tensor.dim() == 4 else tensor
            if tensor.dim() == 3:
                tensor = tensor.permute(1, 2, 0)
            
            # [0, 1] 범위를 [0, 255]로 변환
            array = tensor.cpu().numpy()
            if array.max() <= 1.0:
                array = (array * 255).astype(np.uint8)
            return array
        return tensor
