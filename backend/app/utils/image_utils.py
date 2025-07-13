# backend/app/utils/image_utils.py
"""
Image processing utilities
"""

import cv2
import numpy as np
from PIL import Image
import logging

logger = logging.getLogger(__name__)

class ImageProcessor:
    """Image processing utility class"""
    
    def __init__(self):
        self.supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    
    def resize_image(self, image: np.ndarray, target_size: tuple) -> np.ndarray:
        """
        Resize image to target size
        
        Args:
            image: Input image as numpy array
            target_size: Target size as (width, height)
        
        Returns:
            Resized image
        """
        try:
            if isinstance(target_size, int):
                target_size = (target_size, target_size)
            
            resized = cv2.resize(image, target_size, interpolation=cv2.INTER_LANCZOS4)
            logger.debug(f"Image resized to {target_size}")
            return resized
            
        except Exception as e:
            logger.error(f"Failed to resize image: {e}")
            return image
    
    def normalize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize image values to [0, 1] or [0, 255]
        
        Args:
            image: Input image
            
        Returns:
            Normalized image
        """
        try:
            # 이미 0-255 범위라면 그대로 반환
            if image.dtype == np.uint8:
                return image
            
            # 0-1 범위를 0-255로 변환
            if image.max() <= 1.0:
                normalized = (image * 255).astype(np.uint8)
            else:
                # 다른 범위는 0-255로 정규화
                normalized = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
                normalized = normalized.astype(np.uint8)
            
            logger.debug("Image normalized")
            return normalized
            
        except Exception as e:
            logger.error(f"Failed to normalize image: {e}")
            return image
    
    def crop_center(self, image: np.ndarray, crop_size: tuple) -> np.ndarray:
        """
        Crop image from center
        
        Args:
            image: Input image
            crop_size: Crop size as (width, height)
            
        Returns:
            Cropped image
        """
        try:
            h, w = image.shape[:2]
            crop_w, crop_h = crop_size
            
            # 중앙 좌표 계산
            start_x = max(0, (w - crop_w) // 2)
            start_y = max(0, (h - crop_h) // 2)
            
            end_x = min(w, start_x + crop_w)
            end_y = min(h, start_y + crop_h)
            
            cropped = image[start_y:end_y, start_x:end_x]
            logger.debug(f"Image cropped to {crop_size}")
            return cropped
            
        except Exception as e:
            logger.error(f"Failed to crop image: {e}")
            return image
    
    def enhance_image(self, image: np.ndarray, brightness: float = 1.0, contrast: float = 1.0) -> np.ndarray:
        """
        Enhance image brightness and contrast
        
        Args:
            image: Input image
            brightness: Brightness factor (1.0 = no change)
            contrast: Contrast factor (1.0 = no change)
            
        Returns:
            Enhanced image
        """
        try:
            # 밝기 및 대비 조정
            enhanced = cv2.convertScaleAbs(image, alpha=contrast, beta=brightness * 30)
            logger.debug(f"Image enhanced: brightness={brightness}, contrast={contrast}")
            return enhanced
            
        except Exception as e:
            logger.error(f"Failed to enhance image: {e}")
            return image
    
    def remove_background(self, image: np.ndarray, method: str = "simple") -> np.ndarray:
        """
        Remove image background (simple implementation)
        
        Args:
            image: Input image
            method: Background removal method
            
        Returns:
            Image with background removed
        """
        try:
            if method == "simple":
                # 간단한 배경 제거 (실제로는 더 정교한 방법 필요)
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                
                # 임계값 처리
                _, mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
                
                # 모폴로지 연산
                kernel = np.ones((3, 3), np.uint8)
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
                
                # 마스크 적용
                result = image.copy()
                result[mask == 0] = [255, 255, 255]  # 배경을 흰색으로
                
                logger.debug("Background removed using simple method")
                return result
            
            else:
                logger.warning(f"Unknown background removal method: {method}")
                return image
                
        except Exception as e:
            logger.error(f"Failed to remove background: {e}")
            return image
    
    def detect_person(self, image: np.ndarray) -> dict:
        """
        Detect person in image (placeholder implementation)
        
        Args:
            image: Input image
            
        Returns:
            Detection results
        """
        try:
            # 실제로는 YOLO나 다른 객체 검출 모델 사용
            h, w = image.shape[:2]
            
            # 더미 결과 (중앙에 사람이 있다고 가정)
            person_bbox = {
                "x": w // 4,
                "y": h // 8,
                "width": w // 2,
                "height": h * 3 // 4,
                "confidence": 0.85
            }
            
            logger.debug("Person detected (dummy implementation)")
            return {
                "detected": True,
                "bbox": person_bbox,
                "keypoints": None  # 실제 구현에서는 키포인트 포함
            }
            
        except Exception as e:
            logger.error(f"Failed to detect person: {e}")
            return {"detected": False, "bbox": None, "keypoints": None}
    
    def validate_image(self, image_path: str) -> bool:
        """
        Validate image file
        
        Args:
            image_path: Path to image file
            
        Returns:
            True if valid, False otherwise
        """
        try:
            # 파일 확장자 확인
            import os
            _, ext = os.path.splitext(image_path.lower())
            
            if ext not in self.supported_formats:
                logger.warning(f"Unsupported image format: {ext}")
                return False
            
            # 이미지 로드 테스트
            image = cv2.imread(image_path)
            if image is None:
                logger.warning(f"Cannot load image: {image_path}")
                return False
            
            # 최소 크기 확인
            h, w = image.shape[:2]
            if h < 100 or w < 100:
                logger.warning(f"Image too small: {w}x{h}")
                return False
            
            logger.debug(f"Image validated: {image_path}")
            return True
            
        except Exception as e:
            logger.error(f"Image validation failed: {e}")
            return False
    
    def convert_pil_to_cv2(self, pil_image: Image.Image) -> np.ndarray:
        """
        Convert PIL Image to OpenCV format
        
        Args:
            pil_image: PIL Image
            
        Returns:
            OpenCV image (BGR format)
        """
        try:
            # RGB to BGR 변환
            rgb_image = np.array(pil_image.convert('RGB'))
            bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
            
            logger.debug("PIL image converted to OpenCV format")
            return bgr_image
            
        except Exception as e:
            logger.error(f"Failed to convert PIL to CV2: {e}")
            return np.array([])
    
    def convert_cv2_to_pil(self, cv2_image: np.ndarray) -> Image.Image:
        """
        Convert OpenCV image to PIL format
        
        Args:
            cv2_image: OpenCV image (BGR format)
            
        Returns:
            PIL Image
        """
        try:
            # BGR to RGB 변환
            rgb_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_image)
            
            logger.debug("OpenCV image converted to PIL format")
            return pil_image
            
        except Exception as e:
            logger.error(f"Failed to convert CV2 to PIL: {e}")
            return Image.new('RGB', (100, 100), color='white')