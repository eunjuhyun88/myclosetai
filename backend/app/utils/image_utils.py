import cv2
import numpy as np
from PIL import Image
import io

class ImageProcessor:
    """Image processing utilities"""
    
    @staticmethod
    def resize_image(image: np.ndarray, size: int) -> np.ndarray:
        """Resize image maintaining aspect ratio"""
        h, w = image.shape[:2]
        
        if h > w:
            new_h = size
            new_w = int(w * size / h)
        else:
            new_w = size
            new_h = int(h * size / w)
        
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        
        # 정사각형으로 패딩
        if new_h != size or new_w != size:
            # 중앙 정렬 패딩
            top = (size - new_h) // 2
            bottom = size - new_h - top
            left = (size - new_w) // 2
            right = size - new_w - left
            
            resized = cv2.copyMakeBorder(
                resized, top, bottom, left, right, 
                cv2.BORDER_CONSTANT, value=[255, 255, 255]
            )
        
        return resized
    
    @staticmethod
    def normalize_image(image: np.ndarray) -> np.ndarray:
        """Normalize image to [0, 1] range"""
        return image.astype(np.float32) / 255.0
    
    @staticmethod
    def denormalize_image(image: np.ndarray) -> np.ndarray:
        """Denormalize image to [0, 255] range"""
        return (image * 255).astype(np.uint8)
    
    @staticmethod
    def enhance_image(image: np.ndarray) -> np.ndarray:
        """Enhance image quality"""
        # 히스토그램 평활화
        if len(image.shape) == 3:
            # 컬러 이미지의 경우 LAB 변환 후 L 채널만 평활화
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            lab[:, :, 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lab[:, :, 0])
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        else:
            # 그레이스케일
            enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(image)
        
        return enhanced
