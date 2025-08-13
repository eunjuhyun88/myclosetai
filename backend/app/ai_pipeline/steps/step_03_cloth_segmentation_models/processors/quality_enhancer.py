#!/usr/bin/env python3
"""
🔥 MyCloset AI - Step 03: 의류 세그멘테이션 - Quality Enhancer
=====================================================================

품질 향상을 위한 전용 프로세서

Author: MyCloset AI Team  
Date: 2025-08-01
Version: 1.0
"""

import logging
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
import cv2

logger = logging.getLogger(__name__)

class QualityEnhancer:
    """품질 향상 프로세서"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """초기화"""
        self.config = config or {}
        self.logger = logging.getLogger(f"{__name__}.QualityEnhancer")
        self.enabled = self.config.get('enable_quality_enhancement', True)
        
    def enhance_image_quality(self, image: np.ndarray) -> np.ndarray:
        """이미지 품질 향상"""
        try:
            if not self.enabled:
                return image
            
            if image is None or image.size == 0:
                return image
            
            enhanced_image = image.copy()
            
            # 1. 노이즈 제거
            enhanced_image = self._remove_noise(enhanced_image)
            
            # 2. 선명도 향상
            enhanced_image = self._enhance_sharpness(enhanced_image)
            
            # 3. 대비 향상
            enhanced_image = self._enhance_contrast(enhanced_image)
            
            # 4. 색상 보정
            enhanced_image = self._correct_colors(enhanced_image)
            
            return enhanced_image
            
        except Exception as e:
            self.logger.error(f"❌ 이미지 품질 향상 실패: {e}")
            return image
    
    def enhance_mask_quality(self, mask: np.ndarray) -> np.ndarray:
        """마스크 품질 향상"""
        try:
            if not self.enabled:
                return mask
            
            if mask is None or mask.size == 0:
                return mask
            
            enhanced_mask = mask.copy()
            
            # 1. 노이즈 제거
            enhanced_mask = self._remove_mask_noise(enhanced_mask)
            
            # 2. 경계 정제
            enhanced_mask = self._refine_mask_boundaries(enhanced_mask)
            
            # 3. 홀 채우기
            enhanced_mask = self._fill_mask_holes(enhanced_mask)
            
            # 4. 연결성 개선
            enhanced_mask = self._improve_connectivity(enhanced_mask)
            
            return enhanced_mask
            
        except Exception as e:
            self.logger.error(f"❌ 마스크 품질 향상 실패: {e}")
            return mask
    
    def enhance_segmentation_quality(self, masks: Dict[str, np.ndarray], image: np.ndarray) -> Dict[str, np.ndarray]:
        """세그멘테이션 품질 향상"""
        try:
            if not self.enabled:
                return masks
            
            enhanced_masks = {}
            
            for mask_key, mask in masks.items():
                if mask is None or mask.size == 0:
                    enhanced_masks[mask_key] = mask
                    continue
                
                # 개별 마스크 품질 향상
                enhanced_mask = self.enhance_mask_quality(mask)
                
                # 이미지 기반 추가 정제
                enhanced_mask = self._refine_with_image_context(enhanced_mask, image)
                
                enhanced_masks[mask_key] = enhanced_mask
            
            return enhanced_masks
            
        except Exception as e:
            self.logger.error(f"❌ 세그멘테이션 품질 향상 실패: {e}")
            return masks
    
    def _remove_noise(self, image: np.ndarray) -> np.ndarray:
        """노이즈 제거"""
        try:
            # Non-local means denoising
            denoised = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
            return denoised
            
        except Exception as e:
            self.logger.warning(f"⚠️ 노이즈 제거 실패: {e}")
            return image
    
    def _enhance_sharpness(self, image: np.ndarray) -> np.ndarray:
        """선명도 향상"""
        try:
            # Unsharp masking
            gaussian = cv2.GaussianBlur(image, (0, 0), 2.0)
            sharpened = cv2.addWeighted(image, 1.5, gaussian, -0.5, 0)
            return sharpened
            
        except Exception as e:
            self.logger.warning(f"⚠️ 선명도 향상 실패: {e}")
            return image
    
    def _enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """대비 향상"""
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
            self.logger.warning(f"⚠️ 대비 향상 실패: {e}")
            return image
    
    def _correct_colors(self, image: np.ndarray) -> np.ndarray:
        """색상 보정"""
        try:
            # 자동 화이트 밸런스
            # 1. 그레이 월드 가정
            avg_b = np.mean(image[:, :, 0])
            avg_g = np.mean(image[:, :, 1])
            avg_r = np.mean(image[:, :, 2])
            
            # 2. 화이트 밸런스 적용
            corrected = image.copy().astype(np.float32)
            corrected[:, :, 0] = corrected[:, :, 0] * (avg_g / avg_b)
            corrected[:, :, 2] = corrected[:, :, 2] * (avg_g / avg_r)
            
            # 3. 클리핑 방지
            corrected = np.clip(corrected, 0, 255).astype(np.uint8)
            
            return corrected
            
        except Exception as e:
            self.logger.warning(f"⚠️ 색상 보정 실패: {e}")
            return image
    
    def _remove_mask_noise(self, mask: np.ndarray) -> np.ndarray:
        """마스크 노이즈 제거"""
        try:
            # 모폴로지 연산으로 노이즈 제거
            kernel = np.ones((3, 3), np.uint8)
            denoised = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            return denoised
            
        except Exception as e:
            self.logger.warning(f"⚠️ 마스크 노이즈 제거 실패: {e}")
            return mask
    
    def _refine_mask_boundaries(self, mask: np.ndarray) -> np.ndarray:
        """마스크 경계 정제"""
        try:
            # 경계 스무딩
            refined = cv2.GaussianBlur(mask, (3, 3), 0)
            refined = (refined > 127).astype(np.uint8) * 255
            return refined
            
        except Exception as e:
            self.logger.warning(f"⚠️ 마스크 경계 정제 실패: {e}")
            return mask
    
    def _fill_mask_holes(self, mask: np.ndarray) -> np.ndarray:
        """마스크 홀 채우기"""
        try:
            # 홀 채우기
            kernel = np.ones((5, 5), np.uint8)
            filled = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            return filled
            
        except Exception as e:
            self.logger.warning(f"⚠️ 마스크 홀 채우기 실패: {e}")
            return mask
    
    def _improve_connectivity(self, mask: np.ndarray) -> np.ndarray:
        """연결성 개선"""
        try:
            # 연결 구성요소 분석
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
            
            if num_labels > 1:
                # 가장 큰 연결 구성요소만 유지
                largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
                improved = (labels == largest_label).astype(np.uint8) * 255
                return improved
            
            return mask
            
        except Exception as e:
            self.logger.warning(f"⚠️ 연결성 개선 실패: {e}")
            return mask
    
    def _refine_with_image_context(self, mask: np.ndarray, image: np.ndarray) -> np.ndarray:
        """이미지 컨텍스트로 마스크 정제"""
        try:
            refined_mask = mask.copy()
            
            # 1. 엣지 기반 정제
            edges = cv2.Canny(image, 50, 150)
            edge_kernel = np.ones((2, 2), np.uint8)
            dilated_edges = cv2.dilate(edges, edge_kernel, iterations=1)
            
            # 엣지 근처의 마스크 정제
            refined_mask[dilated_edges > 0] = 0
            
            # 2. 색상 기반 정제
            if image.shape[2] == 3:
                # 의류 색상 범위 확인
                hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
                
                # 의류 색상 마스크 생성 (예: 파란색, 빨간색 등)
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
                
                # 의류 색상 마스크 결합
                clothing_mask = np.zeros_like(blue_mask)
                for cm in clothing_masks:
                    clothing_mask = cv2.bitwise_or(clothing_mask, cm)
                
                # 의류 색상이 아닌 영역 제거
                refined_mask = cv2.bitwise_and(refined_mask, clothing_mask)
            
            return refined_mask
            
        except Exception as e:
            self.logger.warning(f"⚠️ 이미지 컨텍스트 정제 실패: {e}")
            return mask
