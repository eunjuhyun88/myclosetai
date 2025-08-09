#!/usr/bin/env python3
"""
🔥 MyCloset AI - Step 03: 의류 세그멘테이션 - Special Case Processor
=====================================================================

특수 케이스 처리를 위한 전용 프로세서

Author: MyCloset AI Team  
Date: 2025-08-01
Version: 1.0
"""

import logging
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
import cv2

logger = logging.getLogger(__name__)

class SpecialCaseProcessor:
    """특수 케이스 처리 프로세서"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """초기화"""
        self.config = config or {}
        self.logger = logging.getLogger(f"{__name__}.SpecialCaseProcessor")
        self.enabled = self.config.get('enable_special_case_handling', True)
        
    def detect_special_cases(self, image: np.ndarray) -> Dict[str, Any]:
        """특수 케이스 탐지"""
        try:
            if not self.enabled:
                return {}
            
            special_cases = {}
            
            # 1. 저조도 탐지
            if self._is_low_light(image):
                special_cases['low_light'] = True
                self.logger.info("🔍 저조도 케이스 탐지됨")
            
            # 2. 고노이즈 탐지
            if self._is_high_noise(image):
                special_cases['high_noise'] = True
                self.logger.info("🔍 고노이즈 케이스 탐지됨")
            
            # 3. 블러 탐지
            if self._is_blurry(image):
                special_cases['blurry'] = True
                self.logger.info("🔍 블러 케이스 탐지됨")
            
            # 4. 복잡한 배경 탐지
            if self._has_complex_background(image):
                special_cases['complex_background'] = True
                self.logger.info("🔍 복잡한 배경 케이스 탐지됨")
            
            # 5. 작은 의류 탐지
            if self._has_small_clothing(image):
                special_cases['small_clothing'] = True
                self.logger.info("🔍 작은 의류 케이스 탐지됨")
            
            return special_cases
            
        except Exception as e:
            self.logger.error(f"❌ 특수 케이스 탐지 실패: {e}")
            return {}
    
    def apply_special_case_enhancement(self, image: np.ndarray, special_cases: Dict[str, Any]) -> np.ndarray:
        """특수 케이스 향상 적용"""
        try:
            if not special_cases:
                return image
            
            enhanced_image = image.copy()
            
            # 저조도 향상
            if special_cases.get('low_light'):
                enhanced_image = self._enhance_low_light(enhanced_image)
            
            # 고노이즈 향상
            if special_cases.get('high_noise'):
                enhanced_image = self._enhance_high_noise(enhanced_image)
            
            # 블러 향상
            if special_cases.get('blurry'):
                enhanced_image = self._enhance_blurry(enhanced_image)
            
            # 복잡한 배경 향상
            if special_cases.get('complex_background'):
                enhanced_image = self._enhance_complex_background(enhanced_image)
            
            # 작은 의류 향상
            if special_cases.get('small_clothing'):
                enhanced_image = self._enhance_small_clothing(enhanced_image)
            
            return enhanced_image
            
        except Exception as e:
            self.logger.error(f"❌ 특수 케이스 향상 실패: {e}")
            return image
    
    def _is_low_light(self, image: np.ndarray) -> bool:
        """저조도 탐지"""
        try:
            # 밝기 계산
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            mean_brightness = np.mean(gray)
            
            # 임계값: 100 이하를 저조도로 간주
            return mean_brightness < 100
            
        except Exception as e:
            self.logger.warning(f"⚠️ 저조도 탐지 실패: {e}")
            return False
    
    def _is_high_noise(self, image: np.ndarray) -> bool:
        """고노이즈 탐지"""
        try:
            # 라플라시안 분산으로 노이즈 측정
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # 임계값: 500 이상을 고노이즈로 간주
            return laplacian_var > 500
            
        except Exception as e:
            self.logger.warning(f"⚠️ 고노이즈 탐지 실패: {e}")
            return False
    
    def _is_blurry(self, image: np.ndarray) -> bool:
        """블러 탐지"""
        try:
            # 라플라시안 분산으로 블러 측정
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # 임계값: 100 이하를 블러로 간주
            return laplacian_var < 100
            
        except Exception as e:
            self.logger.warning(f"⚠️ 블러 탐지 실패: {e}")
            return False
    
    def _has_complex_background(self, image: np.ndarray) -> bool:
        """복잡한 배경 탐지"""
        try:
            # 엣지 밀도로 복잡성 측정
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges) / (edges.size * 255)
            
            # 임계값: 0.1 이상을 복잡한 배경으로 간주
            return edge_density > 0.1
            
        except Exception as e:
            self.logger.warning(f"⚠️ 복잡한 배경 탐지 실패: {e}")
            return False
    
    def _has_small_clothing(self, image: np.ndarray) -> bool:
        """작은 의류 탐지"""
        try:
            # 의류 영역 크기 추정 (간단한 색상 기반)
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            
            # 의류 색상 범위 (예: 파란색, 빨간색, 검은색 등)
            clothing_masks = []
            
            # 파란색 의류
            blue_lower = np.array([100, 50, 50])
            blue_upper = np.array([130, 255, 255])
            blue_mask = cv2.inRange(hsv, blue_lower, blue_upper)
            clothing_masks.append(blue_mask)
            
            # 빨간색 의류
            red_lower1 = np.array([0, 50, 50])
            red_upper1 = np.array([10, 255, 255])
            red_lower2 = np.array([170, 50, 50])
            red_upper2 = np.array([180, 255, 255])
            red_mask1 = cv2.inRange(hsv, red_lower1, red_upper1)
            red_mask2 = cv2.inRange(hsv, red_lower2, red_upper2)
            red_mask = cv2.bitwise_or(red_mask1, red_mask2)
            clothing_masks.append(red_mask)
            
            # 모든 의류 마스크 결합
            combined_mask = np.zeros_like(blue_mask)
            for mask in clothing_masks:
                combined_mask = cv2.bitwise_or(combined_mask, mask)
            
            # 의류 영역 비율 계산
            clothing_ratio = np.sum(combined_mask > 0) / combined_mask.size
            
            # 임계값: 0.1 이하를 작은 의류로 간주
            return clothing_ratio < 0.1
            
        except Exception as e:
            self.logger.warning(f"⚠️ 작은 의류 탐지 실패: {e}")
            return False
    
    def _enhance_low_light(self, image: np.ndarray) -> np.ndarray:
        """저조도 향상"""
        try:
            # CLAHE 적용
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            l = clahe.apply(l)
            enhanced = cv2.merge([l, a, b])
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
            
            return enhanced
            
        except Exception as e:
            self.logger.warning(f"⚠️ 저조도 향상 실패: {e}")
            return image
    
    def _enhance_high_noise(self, image: np.ndarray) -> np.ndarray:
        """고노이즈 향상"""
        try:
            # 노이즈 제거
            denoised = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
            return denoised
            
        except Exception as e:
            self.logger.warning(f"⚠️ 고노이즈 향상 실패: {e}")
            return image
    
    def _enhance_blurry(self, image: np.ndarray) -> np.ndarray:
        """블러 향상"""
        try:
            # 선명도 향상
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            sharpened = cv2.filter2D(image, -1, kernel)
            return sharpened
            
        except Exception as e:
            self.logger.warning(f"⚠️ 블러 향상 실패: {e}")
            return image
    
    def _enhance_complex_background(self, image: np.ndarray) -> np.ndarray:
        """복잡한 배경 향상"""
        try:
            # 배경 블러 처리
            blurred = cv2.GaussianBlur(image, (15, 15), 0)
            
            # 전경 마스크 생성 (간단한 방법)
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # 마스크를 3채널로 확장
            mask_3d = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB) / 255.0
            
            # 전경과 배경 결합
            enhanced = image * mask_3d + blurred * (1 - mask_3d)
            enhanced = enhanced.astype(np.uint8)
            
            return enhanced
            
        except Exception as e:
            self.logger.warning(f"⚠️ 복잡한 배경 향상 실패: {e}")
            return image
    
    def _enhance_small_clothing(self, image: np.ndarray) -> np.ndarray:
        """작은 의류 향상"""
        try:
            # 의류 영역 확대 (간단한 방법)
            # 실제로는 더 정교한 의류 탐지 알고리즘이 필요
            enhanced = image.copy()
            
            # 대비 향상
            lab = cv2.cvtColor(enhanced, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            l = clahe.apply(l)
            enhanced = cv2.merge([l, a, b])
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
            
            return enhanced
            
        except Exception as e:
            self.logger.warning(f"⚠️ 작은 의류 향상 실패: {e}")
            return image
