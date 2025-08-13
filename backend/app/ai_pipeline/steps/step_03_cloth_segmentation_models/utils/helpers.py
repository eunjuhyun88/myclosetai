#!/usr/bin/env python3
"""
🔥 MyCloset AI - Step 03: 의류 세그멘테이션 - Helpers
=====================================================================

유틸리티 함수들을 분리한 모듈

Author: MyCloset AI Team  
Date: 2025-08-01
Version: 1.0
"""

import numpy as np
import cv2
import platform
import subprocess
from typing import Dict, Any, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

def detect_m3_max():
    """M3 Max 감지"""
    try:
        if platform.system() == 'Darwin':
            result = subprocess.run(
                ['sysctl', '-n', 'machdep.cpu.brand_string'],
                capture_output=True, text=True, timeout=5
            )
            return 'M3' in result.stdout
    except:
        pass
    return False


def safe_resize_mask(mask: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
    """
    마스크 안전 리사이즈
    
    Args:
        mask: 입력 마스크
        target_shape: 목표 크기
        
    Returns:
        리사이즈된 마스크
    """
    try:
        if mask.shape[:2] == target_shape:
            return mask
        
        # 이진 마스크인 경우
        if len(mask.shape) == 2 or mask.shape[2] == 1:
            resized = cv2.resize(mask, target_shape, interpolation=cv2.INTER_NEAREST)
        else:
            # 다채널 마스크인 경우
            resized = cv2.resize(mask, target_shape, interpolation=cv2.INTER_LINEAR)
        
        return resized
        
    except Exception as e:
        logger.error(f"마스크 리사이즈 실패: {e}")
        return np.zeros(target_shape, dtype=mask.dtype)


def create_segmentation_visualizations(image: np.ndarray, masks: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """
    세그멘테이션 시각화 생성
    
    Args:
        image: 원본 이미지
        masks: 마스크들
        
    Returns:
        시각화 결과들
    """
    try:
        visualizations = {}
        
        for mask_name, mask in masks.items():
            if mask is None or mask.size == 0:
                continue
            
            # 마스크를 원본 이미지 크기로 리사이즈
            target_shape = (image.shape[1], image.shape[0])
            resized_mask = safe_resize_mask(mask, target_shape)
            
            # 마스크를 3채널로 변환
            if len(resized_mask.shape) == 2:
                resized_mask = np.stack([resized_mask] * 3, axis=-1)
            
            # 마스크 오버레이 생성
            overlay = image.copy()
            mask_region = resized_mask > 0.5
            
            # 마스크 영역에 색상 적용
            overlay[mask_region] = overlay[mask_region] * 0.7 + np.array([0, 255, 0]) * 0.3
            
            visualizations[f"{mask_name}_overlay"] = overlay.astype(np.uint8)
            visualizations[f"{mask_name}_mask"] = (resized_mask * 255).astype(np.uint8)
        
        return visualizations
        
    except Exception as e:
        logger.error(f"시각화 생성 실패: {e}")
        return {}


def fill_holes_and_remove_noise_advanced(masks: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    고급 홀 채우기 및 노이즈 제거
    
    Args:
        masks: 마스크들
        
    Returns:
        정제된 마스크들
    """
    try:
        refined_masks = {}
        
        for mask_name, mask in masks.items():
            if mask is None or mask.size == 0:
                continue
            
            # 이진화
            if len(mask.shape) == 3:
                mask = mask[:, :, 0]
            
            # 노이즈 제거
            kernel = np.ones((3, 3), np.uint8)
            mask_cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            
            # 홀 채우기
            mask_filled = cv2.morphologyEx(mask_cleaned, cv2.MORPH_CLOSE, kernel)
            
            # 경계 정리
            mask_refined = cv2.morphologyEx(mask_filled, cv2.MORPH_OPEN, kernel)
            
            refined_masks[mask_name] = mask_refined
        
        return refined_masks
        
    except Exception as e:
        logger.error(f"마스크 정제 실패: {e}")
        return masks


def evaluate_segmentation_quality(masks: Dict[str, np.ndarray], image: np.ndarray) -> Dict[str, float]:
    """
    세그멘테이션 품질 평가
    
    Args:
        masks: 마스크들
        image: 원본 이미지
        
    Returns:
        품질 점수들
    """
    try:
        quality_scores = {}
        
        for mask_name, mask in masks.items():
            if mask is None or mask.size == 0:
                quality_scores[mask_name] = 0.0
                continue
            
            # 마스크를 이진화
            if len(mask.shape) == 3:
                mask = mask[:, :, 0]
            
            binary_mask = (mask > 0.5).astype(np.uint8)
            
            # 1. 연결성 평가
            num_labels, labels = cv2.connectedComponents(binary_mask)
            connectivity_score = 1.0 / num_labels if num_labels > 1 else 1.0
            
            # 2. 경계 부드러움 평가
            edges = cv2.Canny(binary_mask, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            smoothness_score = 1.0 - min(edge_density, 0.1) / 0.1
            
            # 3. 영역 크기 평가
            area = np.sum(binary_mask)
            total_area = binary_mask.size
            area_ratio = area / total_area
            area_score = min(area_ratio * 10, 1.0)  # 적절한 크기 범위
            
            # 4. 전체 품질 점수
            overall_quality = (connectivity_score + smoothness_score + area_score) / 3
            quality_scores[mask_name] = overall_quality
        
        return quality_scores
        
    except Exception as e:
        logger.error(f"품질 평가 실패: {e}")
        return {mask_name: 0.5 for mask_name in masks.keys()}
