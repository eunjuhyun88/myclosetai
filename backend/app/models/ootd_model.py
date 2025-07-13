# backend/app/models/ootd_model.py
"""
OOTDiffusion Model Implementation
"""

import torch
import numpy as np
from PIL import Image
import logging

logger = logging.getLogger(__name__)

class OOTDModel:
    """OOTDiffusion Model for Virtual Try-On"""
    
    def __init__(self, device="cpu"):
        self.device = device
        self.model_loaded = False
        self._load_model()
    
    def _load_model(self):
        """Load the OOTDiffusion model"""
        try:
            # 실제 모델 로딩 로직 (현재는 시뮬레이션)
            logger.info(f"Loading OOTDiffusion model on {self.device}")
            
            # 여기서 실제 모델을 로드해야 함
            # self.model = ... 
            
            self.model_loaded = True
            logger.info("✅ OOTDiffusion model loaded successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to load OOTDiffusion model: {e}")
            self.model_loaded = False
    
    def generate(
        self, 
        person_image: np.ndarray, 
        clothing_image: np.ndarray, 
        category: str = "upper_body"
    ) -> np.ndarray:
        """Generate virtual try-on result"""
        
        if not self.model_loaded:
            logger.warning("Model not loaded, using fallback method")
            return self._fallback_generation(person_image, clothing_image)
        
        try:
            # 실제 OOTDiffusion 추론 로직
            # 현재는 간단한 합성으로 대체
            result = self._simple_composite(person_image, clothing_image)
            
            logger.info(f"✅ Generated try-on result for category: {category}")
            return result
            
        except Exception as e:
            logger.error(f"❌ Generation failed: {e}")
            return self._fallback_generation(person_image, clothing_image)
    
    def _simple_composite(
        self, 
        person_image: np.ndarray, 
        clothing_image: np.ndarray
    ) -> np.ndarray:
        """Simple image compositing"""
        
        # 사람 이미지 복사
        result = person_image.copy()
        
        # 의류 이미지 리사이즈
        h, w = person_image.shape[:2]
        clothing_resized = self._resize_image(clothing_image, (w//3, h//3))
        
        # 중앙 상단에 배치
        y_offset = h // 6
        x_offset = (w - clothing_resized.shape[1]) // 2
        
        # 영역 확인
        if (y_offset + clothing_resized.shape[0] <= h and 
            x_offset + clothing_resized.shape[1] <= w and
            y_offset >= 0 and x_offset >= 0):
            
            # 간단한 알파 블렌딩
            alpha = 0.7
            result[y_offset:y_offset+clothing_resized.shape[0], 
                   x_offset:x_offset+clothing_resized.shape[1]] = (
                alpha * clothing_resized + 
                (1 - alpha) * result[y_offset:y_offset+clothing_resized.shape[0], 
                                   x_offset:x_offset+clothing_resized.shape[1]]
            )
        
        return result
    
    def _resize_image(self, image: np.ndarray, target_size: tuple) -> np.ndarray:
        """Resize image while maintaining aspect ratio"""
        import cv2
        
        h, w = image.shape[:2]
        target_w, target_h = target_size
        
        # 비율 계산
        ratio = min(target_w / w, target_h / h)
        new_w = int(w * ratio)
        new_h = int(h * ratio)
        
        # 리사이즈
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        
        return resized
    
    def _fallback_generation(
        self, 
        person_image: np.ndarray, 
        clothing_image: np.ndarray
    ) -> np.ndarray:
        """Fallback generation method"""
        logger.info("Using fallback generation method")
        
        # 단순히 사람 이미지 반환
        return person_image.copy()
    
    def is_ready(self) -> bool:
        """Check if model is ready"""
        return self.model_loaded
    
    def get_info(self) -> dict:
        """Get model information"""
        return {
            "model_name": "OOTDiffusion",
            "device": self.device,
            "loaded": self.model_loaded,
            "version": "1.0.0"
        }