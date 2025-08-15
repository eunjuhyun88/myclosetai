"""
🔥 품질 향상 서비스
===================

후처리를 위한 품질 향상 서비스:
1. 이미지 품질 향상
2. 노이즈 제거
3. 선명도 향상
4. 대비 개선

Author: MyCloset AI Team
Date: 2025-08-14
Version: 1.0
"""

import logging
import numpy as np
import cv2
from typing import Dict, Any, Optional, Tuple
from PIL import Image, ImageEnhance

logger = logging.getLogger(__name__)

class QualityEnhancementService:
    """품질 향상 서비스"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.QualityEnhancementService")
        
        # 서비스 통계
        self.service_stats = {
            'total_enhancements': 0,
            'successful_enhancements': 0,
            'failed_enhancements': 0,
            'average_enhancement_time': 0.0
        }
    
    def enhance_image_quality(self, image: np.ndarray, enhancement_type: str = "comprehensive") -> np.ndarray:
        """이미지 품질 향상"""
        try:
            self.logger.info(f"🚀 {enhancement_type} 품질 향상 시작")
            
            if enhancement_type == "comprehensive":
                enhanced = self._comprehensive_enhancement(image)
            elif enhancement_type == "noise_reduction":
                enhanced = self._noise_reduction(image)
            elif enhancement_type == "sharpness":
                enhanced = self._sharpness_enhancement(image)
            elif enhancement_type == "contrast":
                enhanced = self._contrast_enhancement(image)
            else:
                enhanced = self._comprehensive_enhancement(image)
            
            self._update_service_stats(True)
            self.logger.info(f"✅ {enhancement_type} 품질 향상 완료")
            
            return enhanced
            
        except Exception as e:
            self.logger.error(f"❌ {enhancement_type} 품질 향상 실패: {e}")
            self._update_service_stats(False)
            return image
    
    def _comprehensive_enhancement(self, image: np.ndarray) -> np.ndarray:
        """종합 품질 향상"""
        try:
            enhanced = image.copy()
            
            # 1. 노이즈 제거
            enhanced = self._noise_reduction(enhanced)
            
            # 2. 선명도 향상
            enhanced = self._sharpness_enhancement(enhanced)
            
            # 3. 대비 향상
            enhanced = self._contrast_enhancement(enhanced)
            
            # 4. 색상 균형 조정
            enhanced = self._color_balance(enhanced)
            
            return enhanced
            
        except Exception as e:
            self.logger.error(f"❌ 종합 품질 향상 실패: {e}")
            return image
    
    def _noise_reduction(self, image: np.ndarray) -> np.ndarray:
        """노이즈 제거"""
        try:
            # Non-local Means Denoising
            denoised = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
            
            # Bilateral Filter 추가 적용
            denoised = cv2.bilateralFilter(denoised, 9, 75, 75)
            
            return denoised
            
        except Exception as e:
            self.logger.error(f"❌ 노이즈 제거 실패: {e}")
            return image
    
    def _sharpness_enhancement(self, image: np.ndarray) -> np.ndarray:
        """선명도 향상"""
        try:
            # Unsharp Masking
            gaussian = cv2.GaussianBlur(image, (0, 0), 2.0)
            sharpened = cv2.addWeighted(image, 1.5, gaussian, -0.5, 0)
            
            # 추가 선명도 향상
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            sharpened = cv2.filter2D(sharpened, -1, kernel)
            
            return sharpened
            
        except Exception as e:
            self.logger.error(f"❌ 선명도 향상 실패: {e}")
            return image
    
    def _contrast_enhancement(self, image: np.ndarray) -> np.ndarray:
        """대비 향상"""
        try:
            # LAB 색공간으로 변환
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # CLAHE 적용 (L 채널)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            l = clahe.apply(l)
            
            # 채널 합치기
            enhanced = cv2.merge([l, a, b])
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
            
            return enhanced
            
        except Exception as e:
            self.logger.error(f"❌ 대비 향상 실패: {e}")
            return image
    
    def _color_balance(self, image: np.ndarray) -> np.ndarray:
        """색상 균형 조정"""
        try:
            # 색상 균형 조정
            balanced = cv2.convertScaleAbs(image, alpha=1.1, beta=5)
            
            # 감마 보정
            gamma = 1.1
            invGamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
            balanced = cv2.LUT(balanced, table)
            
            return balanced
            
        except Exception as e:
            self.logger.error(f"❌ 색상 균형 조정 실패: {e}")
            return image
    
    def get_enhancement_options(self) -> Dict[str, str]:
        """향상 옵션 반환"""
        return {
            "comprehensive": "종합 품질 향상 (노이즈 제거 + 선명도 + 대비 + 색상)",
            "noise_reduction": "노이즈 제거 전용",
            "sharpness": "선명도 향상 전용",
            "contrast": "대비 향상 전용"
        }
    
    def _update_service_stats(self, success: bool):
        """서비스 통계 업데이트"""
        try:
            self.service_stats['total_enhancements'] += 1
            
            if success:
                self.service_stats['successful_enhancements'] += 1
            else:
                self.service_stats['failed_enhancements'] += 1
                
        except Exception as e:
            self.logger.error(f"❌ 서비스 통계 업데이트 실패: {e}")
    
    def get_service_stats(self) -> Dict[str, Any]:
        """서비스 통계 반환"""
        return self.service_stats.copy()
    
    def reset_service_stats(self):
        """서비스 통계 초기화"""
        self.service_stats = {
            'total_enhancements': 0,
            'successful_enhancements': 0,
            'failed_enhancements': 0,
            'average_enhancement_time': 0.0
        }
