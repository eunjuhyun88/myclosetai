#!/usr/bin/env python3
"""
🔥 MyCloset AI - Step 03: 의류 세그멘테이션 - Quality Enhancement
=====================================================================

품질 향상 관련 후처리 기능들을 분리한 모듈

Author: MyCloset AI Team  
Date: 2025-08-01
Version: 1.0
"""

import logging
import numpy as np
import cv2
from typing import Dict, Any, List, Tuple, Optional

logger = logging.getLogger(__name__)

def _fill_holes_and_remove_noise_advanced(self, masks: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """고급 홀 채우기 및 노이즈 제거"""
    try:
        processed_masks = {}
        
        for mask_type, mask in masks.items():
            if mask is None or mask.size == 0:
                continue
            
            # 노이즈 제거
            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            
            # 홀 채우기
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                # 가장 큰 컨투어 찾기
                largest_contour = max(contours, key=cv2.contourArea)
                
                # 홀 채우기
                filled_mask = np.zeros_like(mask)
                cv2.fillPoly(filled_mask, [largest_contour], 1)
                
                # 작은 홀들도 채우기
                filled_mask = cv2.morphologyEx(filled_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
                
                processed_masks[mask_type] = filled_mask
            else:
                processed_masks[mask_type] = mask
        
        return processed_masks
        
    except Exception as e:
        logger.warning(f"고급 홀 채우기 및 노이즈 제거 실패: {e}")
        return masks

def _evaluate_segmentation_quality(self, masks: Dict[str, np.ndarray], image: np.ndarray) -> Dict[str, float]:
    """세그멘테이션 품질 평가"""
    try:
        quality_metrics = {}
        
        for mask_type, mask in masks.items():
            if mask is None or mask.size == 0:
                quality_metrics[mask_type] = 0.0
                continue
            
            # 면적 비율
            total_pixels = mask.size
            mask_pixels = np.sum(mask)
            area_ratio = mask_pixels / total_pixels
            
            # 경계 품질
            edges = cv2.Canny(mask.astype(np.uint8) * 255, 50, 150)
            edge_density = np.sum(edges) / (edges.size * 255)
            
            # 연결성
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            connectivity_score = 1.0 / (len(contours) + 1)  # 컨투어가 적을수록 좋음
            
            # 원형도 (circularity)
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                contour_area = cv2.contourArea(largest_contour)
                contour_perimeter = cv2.arcLength(largest_contour, True)
                
                if contour_perimeter > 0:
                    circularity = 4 * np.pi * contour_area / (contour_perimeter ** 2)
                else:
                    circularity = 0.0
            else:
                circularity = 0.0
            
            # 종합 품질 점수
            quality_score = (
                area_ratio * 0.3 +
                (1 - edge_density) * 0.2 +
                connectivity_score * 0.3 +
                circularity * 0.2
            )
            
            quality_metrics[mask_type] = min(quality_score, 1.0)
        
        return quality_metrics
        
    except Exception as e:
        logger.warning(f"세그멘테이션 품질 평가 실패: {e}")
        return {mask_type: 0.5 for mask_type in masks.keys()}

def _create_segmentation_visualizations(self, image: np.ndarray, masks: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """세그멘테이션 시각화 생성"""
    try:
        visualizations = {}
        
        if image is None or not masks:
            return visualizations
        
        # 원본 이미지 복사
        overlay_image = image.copy()
        
        # 색상 매핑
        colors = [
            [255, 0, 0],    # 빨강
            [0, 255, 0],    # 초록
            [0, 0, 255],    # 파랑
            [255, 255, 0],  # 노랑
            [255, 0, 255],  # 마젠타
            [0, 255, 255]   # 시안
        ]
        
        # 마스크 오버레이 생성
        for i, (mask_type, mask) in enumerate(masks.items()):
            if mask is not None and np.any(mask):
                color = colors[i % len(colors)]
                
                # 마스크를 3채널로 확장
                mask_3d = np.stack([mask, mask, mask], axis=-1)
                
                # 색상 적용
                colored_mask = np.array(color) * mask_3d
                
                # 알파 블렌딩
                alpha = 0.6
                overlay_image = overlay_image * (1 - alpha * mask_3d) + colored_mask * alpha * mask_3d
        
        visualizations['overlay'] = overlay_image.astype(np.uint8)
        
        # 개별 마스크 시각화
        for mask_type, mask in masks.items():
            if mask is not None:
                visualizations[f'mask_{mask_type}'] = (mask * 255).astype(np.uint8)
        
        return visualizations
        
    except Exception as e:
        logger.warning(f"세그멘테이션 시각화 생성 실패: {e}")
        return {}

def _assess_image_quality(self, image: np.ndarray) -> Dict[str, float]:
    """이미지 품질 평가"""
    try:
        quality_scores = {}
        
        if image is None:
            return {'brightness': 0.5, 'contrast': 0.5, 'sharpness': 0.5, 'noise_level': 0.5}
        
        # 밝기 평가
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        brightness = np.mean(gray) / 255.0
        quality_scores['brightness'] = brightness
        
        # 대비 평가
        contrast = np.std(gray) / 255.0
        quality_scores['contrast'] = contrast
        
        # 선명도 평가
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness = np.var(laplacian) / 1000.0  # 정규화
        quality_scores['sharpness'] = min(sharpness, 1.0)
        
        # 노이즈 레벨 평가
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        noise = np.mean(np.abs(gray.astype(np.float32) - blurred.astype(np.float32))) / 255.0
        quality_scores['noise_level'] = min(noise, 1.0)
        
        return quality_scores
        
    except Exception as e:
        logger.warning(f"이미지 품질 평가 실패: {e}")
        return {'brightness': 0.5, 'contrast': 0.5, 'sharpness': 0.5, 'noise_level': 0.5}

def _normalize_lighting(self, image: np.ndarray) -> np.ndarray:
    """조명 정규화"""
    try:
        if image is None:
            return image
        
        # LAB 색공간으로 변환
        if len(image.shape) == 3:
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        else:
            # 그레이스케일인 경우 3채널로 확장
            lab = cv2.cvtColor(cv2.cvtColor(image, cv2.COLOR_GRAY2RGB), cv2.COLOR_RGB2LAB)
        
        # L 채널 정규화
        l_channel = lab[:, :, 0]
        
        # 히스토그램 평활화
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_channel_normalized = clahe.apply(l_channel)
        
        # 정규화된 L 채널로 교체
        lab[:, :, 0] = l_channel_normalized
        
        # RGB로 변환
        normalized_image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        return normalized_image
        
    except Exception as e:
        logger.warning(f"조명 정규화 실패: {e}")
        return image

def _correct_colors(self, image: np.ndarray) -> np.ndarray:
    """색상 보정"""
    try:
        if image is None:
            return image
        
        # 색상 보정을 위한 간단한 방법
        # 화이트 밸런스 적용
        if len(image.shape) == 3:
            # 각 채널의 평균값 계산
            means = np.mean(image, axis=(0, 1))
            
            # 화이트 밸런스 적용
            max_mean = np.max(means)
            if max_mean > 0:
                white_balanced = image * (max_mean / means)
                white_balanced = np.clip(white_balanced, 0, 255).astype(np.uint8)
                return white_balanced
        
        return image
        
    except Exception as e:
        logger.warning(f"색상 보정 실패: {e}")
        return image
