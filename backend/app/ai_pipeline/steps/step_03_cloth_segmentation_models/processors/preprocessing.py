#!/usr/bin/env python3
"""
🔥 MyCloset AI - Step 03: 의류 세그멘테이션 - Preprocessing
=====================================================================

전처리 관련 함수들을 분리한 모듈

Author: MyCloset AI Team  
Date: 2025-08-01
Version: 1.0
"""

import numpy as np
import cv2
from typing import Dict, Any, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

def assess_image_quality(image: np.ndarray) -> Dict[str, float]:
    """
    이미지 품질 평가
    
    Args:
        image: 입력 이미지
        
    Returns:
        품질 점수들
    """
    try:
        # 그레이스케일 변환
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # 1. 밝기 평가
        brightness = np.mean(gray)
        brightness_score = min(1.0, brightness / 128.0)
        
        # 2. 대비 평가
        contrast = np.std(gray)
        contrast_score = min(1.0, contrast / 50.0)
        
        # 3. 선명도 평가 (Laplacian variance)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness = np.var(laplacian)
        sharpness_score = min(1.0, sharpness / 500.0)
        
        # 4. 노이즈 평가
        noise = np.mean(np.abs(cv2.GaussianBlur(gray, (5, 5), 0) - gray))
        noise_score = max(0.0, 1.0 - noise / 10.0)
        
        # 5. 전체 품질 점수
        overall_quality = (brightness_score + contrast_score + sharpness_score + noise_score) / 4
        
        return {
            'brightness': brightness_score,
            'contrast': contrast_score,
            'sharpness': sharpness_score,
            'noise': noise_score,
            'overall_quality': overall_quality
        }
        
    except Exception as e:
        logger.error(f"이미지 품질 평가 실패: {e}")
        return {
            'brightness': 0.5,
            'contrast': 0.5,
            'sharpness': 0.5,
            'noise': 0.5,
            'overall_quality': 0.5
        }


def normalize_lighting(image: np.ndarray) -> np.ndarray:
    """
    조명 정규화
    
    Args:
        image: 입력 이미지
        
    Returns:
        정규화된 이미지
    """
    try:
        # LAB 색공간으로 변환
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        # L 채널 추출
        l_channel = lab[:, :, 0]
        
        # CLAHE 적용
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_channel_normalized = clahe.apply(l_channel)
        
        # 정규화된 L 채널로 교체
        lab[:, :, 0] = l_channel_normalized
        
        # BGR로 변환
        normalized_image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        return normalized_image
        
    except Exception as e:
        logger.error(f"조명 정규화 실패: {e}")
        return image


def correct_colors(image: np.ndarray) -> np.ndarray:
    """
    색상 보정
    
    Args:
        image: 입력 이미지
        
    Returns:
        보정된 이미지
    """
    try:
        # 색온도 보정 (자동 화이트 밸런스)
        # 1. 그레이 월드 가정
        b, g, r = cv2.split(image)
        
        # 각 채널의 평균 계산
        b_mean = np.mean(b)
        g_mean = np.mean(g)
        r_mean = np.mean(r)
        
        # 그레이 월드 가정에 따른 스케일 팩터
        gray_mean = (b_mean + g_mean + r_mean) / 3
        b_scale = gray_mean / b_mean if b_mean > 0 else 1.0
        g_scale = gray_mean / g_mean if g_mean > 0 else 1.0
        r_scale = gray_mean / r_mean if r_mean > 0 else 1.0
        
        # 스케일링 적용
        b_corrected = np.clip(b * b_scale, 0, 255).astype(np.uint8)
        g_corrected = np.clip(g * g_scale, 0, 255).astype(np.uint8)
        r_corrected = np.clip(r * r_scale, 0, 255).astype(np.uint8)
        
        # 채널 합치기
        corrected_image = cv2.merge([b_corrected, g_corrected, r_corrected])
        
        return corrected_image
        
    except Exception as e:
        logger.error(f"색상 보정 실패: {e}")
        return image


def determine_quality_level(processed_input: Dict[str, Any], quality_scores: Dict[str, float]) -> str:
    """
    품질 레벨 결정
    
    Args:
        processed_input: 처리된 입력
        quality_scores: 품질 점수들
        
    Returns:
        품질 레벨
    """
    try:
        overall_quality = quality_scores.get('overall_quality', 0.5)
        
        # 품질 레벨 결정
        if overall_quality >= 0.8:
            return 'ultra'
        elif overall_quality >= 0.6:
            return 'high'
        elif overall_quality >= 0.4:
            return 'balanced'
        else:
            return 'fast'
            
    except Exception as e:
        logger.error(f"품질 레벨 결정 실패: {e}")
        return 'balanced'
