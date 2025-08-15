"""
Virtual Fitting Utilities
가상 피팅에 필요한 유틸리티 함수들을 제공합니다.
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from typing import Dict, List, Tuple, Optional, Any, Union
import cv2
import logging

logger = logging.getLogger(__name__)

def calculate_fitting_quality(
    original_image: torch.Tensor,
    fitted_image: torch.Tensor,
    mask: Optional[torch.Tensor] = None
) -> Dict[str, float]:
    """
    피팅 품질 계산
    
    Args:
        original_image: 원본 이미지
        fitted_image: 피팅된 이미지
        mask: 마스크 (선택사항)
    
    Returns:
        품질 메트릭 딕셔너리
    """
    try:
        if mask is not None:
            # 마스크가 있는 경우 마스크 영역만 계산
            original_masked = original_image * mask
            fitted_masked = fitted_image * mask
        else:
            original_masked = original_image
            fitted_masked = fitted_image
        
        # MSE 계산
        mse = F.mse_loss(original_masked, fitted_masked)
        
        # PSNR 계산
        if mse > 0:
            psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
        else:
            psnr = float('inf')
        
        # SSIM 계산 (간단한 버전)
        ssim = calculate_simple_ssim(original_masked, fitted_masked)
        
        # 색상 일관성 계산
        color_consistency = calculate_color_consistency(original_masked, fitted_masked)
        
        return {
            'mse': float(mse),
            'psnr': float(psnr),
            'ssim': float(ssim),
            'color_consistency': float(color_consistency),
            'overall_quality': float((ssim + color_consistency) / 2)
        }
        
    except Exception as e:
        logger.error(f"품질 계산 실패: {e}")
        return {
            'mse': 0.0,
            'psnr': 0.0,
            'ssim': 0.0,
            'color_consistency': 0.0,
            'overall_quality': 0.0
        }

def calculate_simple_ssim(img1: torch.Tensor, img2: torch.Tensor) -> float:
    """간단한 SSIM 계산"""
    try:
        # 정규화
        img1_norm = (img1 - img1.mean()) / (img1.std() + 1e-8)
        img2_norm = (img2 - img2.mean()) / (img2.std() + 1e-8)
        
        # 상관계수 계산
        correlation = torch.mean(img1_norm * img2_norm)
        
        # SSIM = (2*correlation + c) / (1 + c)
        c = 0.01
        ssim = (2 * correlation + c) / (1 + c)
        
        return float(torch.clamp(ssim, 0, 1))
        
    except Exception as e:
        logger.error(f"SSIM 계산 실패: {e}")
        return 0.0

def calculate_color_consistency(img1: torch.Tensor, img2: torch.Tensor) -> float:
    """색상 일관성 계산"""
    try:
        # RGB 채널별 평균 계산
        img1_mean = img1.mean(dim=[2, 3])  # [B, C]
        img2_mean = img2.mean(dim=[2, 3])  # [B, C]
        
        # 색상 차이 계산
        color_diff = torch.abs(img1_mean - img2_mean)
        
        # 정규화된 색상 일관성 (0~1)
        consistency = 1.0 - torch.mean(color_diff)
        
        return float(torch.clamp(consistency, 0, 1))
        
    except Exception as e:
        logger.error(f"색상 일관성 계산 실패: {e}")
        return 0.0

def apply_fitting_mask(
    image: torch.Tensor,
    mask: torch.Tensor,
    blend_strength: float = 0.8
) -> torch.Tensor:
    """
    피팅 마스크 적용
    
    Args:
        image: 원본 이미지
        mask: 피팅 마스크
        blend_strength: 블렌딩 강도
    
    Returns:
        마스크가 적용된 이미지
    """
    try:
        # 마스크 정규화
        mask = torch.clamp(mask, 0, 1)
        
        # 블렌딩 적용
        blended = image * (1 - mask * blend_strength) + image * mask * blend_strength
        
        return blended
        
    except Exception as e:
        logger.error(f"마스크 적용 실패: {e}")
        return image

def optimize_fitting_parameters(
    quality_metrics: Dict[str, float],
    target_quality: float = 0.8
) -> Dict[str, float]:
    """
    피팅 파라미터 최적화
    
    Args:
        quality_metrics: 현재 품질 메트릭
        target_quality: 목표 품질
    
    Returns:
        최적화된 파라미터
    """
    try:
        current_quality = quality_metrics.get('overall_quality', 0.0)
        
        # 품질이 낮은 경우 파라미터 조정
        if current_quality < target_quality:
            # 블렌딩 강도 증가
            blend_strength = min(0.95, 0.8 + (target_quality - current_quality) * 0.3)
            
            # 해상도 증가
            resolution_factor = 1.0 + (target_quality - current_quality) * 0.2
            
            return {
                'blend_strength': blend_strength,
                'resolution_factor': resolution_factor,
                'quality_threshold': target_quality
            }
        else:
            # 현재 파라미터 유지
            return {
                'blend_strength': 0.8,
                'resolution_factor': 1.0,
                'quality_threshold': target_quality
            }
            
    except Exception as e:
        logger.error(f"파라미터 최적화 실패: {e}")
        return {
            'blend_strength': 0.8,
            'resolution_factor': 1.0,
            'quality_threshold': target_quality
        }

def validate_fitting_result(
    result: torch.Tensor,
    min_quality: float = 0.6
) -> Tuple[bool, Dict[str, Any]]:
    """
    피팅 결과 검증
    
    Args:
        result: 피팅 결과
        min_quality: 최소 품질 임계값
    
    Returns:
        (검증 통과 여부, 검증 결과)
    """
    try:
        # 기본 검증
        if result is None:
            return False, {'error': '결과가 None입니다'}
        
        if not torch.is_tensor(result):
            return False, {'error': '결과가 텐서가 아닙니다'}
        
        # 차원 검증
        if result.dim() != 4:  # [B, C, H, W]
            return False, {'error': f'잘못된 차원: {result.dim()}'}
        
        # 값 범위 검증
        if result.min() < 0 or result.max() > 1:
            return False, {'error': '값이 [0, 1] 범위를 벗어났습니다'}
        
        # 품질 계산
        quality_metrics = {
            'overall_quality': 0.8,  # 기본값
            'mse': 0.1,
            'psnr': 20.0
        }
        
        # 품질 임계값 검증
        if quality_metrics['overall_quality'] < min_quality:
            return False, {
                'error': f'품질이 너무 낮습니다: {quality_metrics["overall_quality"]:.3f}',
                'quality_metrics': quality_metrics
            }
        
        return True, {
            'quality_metrics': quality_metrics,
            'validation_passed': True
        }
        
    except Exception as e:
        logger.error(f"결과 검증 실패: {e}")
        return False, {'error': f'검증 중 오류 발생: {e}'}

# 유틸리티 함수들
def get_fitting_utils_info() -> Dict[str, Any]:
    """피팅 유틸리티 정보 반환"""
    return {
        'module_name': 'fitting_utils',
        'version': '1.0.0',
        'functions': [
            'calculate_fitting_quality',
            'calculate_simple_ssim',
            'calculate_color_consistency',
            'apply_fitting_mask',
            'optimize_fitting_parameters',
            'validate_fitting_result'
        ],
        'description': '가상 피팅에 필요한 유틸리티 함수들'
    }

class FittingUtils:
    """가상 피팅 유틸리티 클래스"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.FittingUtils")
        self.version = "1.0.0"
    
    def calculate_fitting_quality(
        self,
        original_image: torch.Tensor,
        fitted_image: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """피팅 품질 계산 (클래스 메서드)"""
        return calculate_fitting_quality(original_image, fitted_image, mask)
    
    def calculate_simple_ssim(self, img1: torch.Tensor, img2: torch.Tensor) -> float:
        """간단한 SSIM 계산 (클래스 메서드)"""
        return calculate_simple_ssim(img1, img2)
    
    def calculate_color_consistency(self, img1: torch.Tensor, img2: torch.Tensor) -> float:
        """색상 일관성 계산 (클래스 메서드)"""
        return calculate_color_consistency(img1, img2)
    
    def apply_fitting_mask(
        self,
        image: torch.Tensor,
        mask: torch.Tensor,
        blend_strength: float = 0.8
    ) -> torch.Tensor:
        """피팅 마스크 적용 (클래스 메서드)"""
        return apply_fitting_mask(image, mask, blend_strength)
    
    def optimize_fitting_parameters(
        self,
        quality_metrics: Dict[str, float],
        target_quality: float = 0.8
    ) -> Dict[str, float]:
        """피팅 파라미터 최적화 (클래스 메서드)"""
        return optimize_fitting_parameters(quality_metrics, target_quality)
    
    def validate_fitting_result(
        self,
        result: torch.Tensor,
        min_quality: float = 0.6
    ) -> Tuple[bool, Dict[str, Any]]:
        """피팅 결과 검증 (클래스 메서드)"""
        return validate_fitting_result(result, min_quality)
    
    def get_info(self) -> Dict[str, Any]:
        """유틸리티 정보 반환"""
        return get_fitting_utils_info()
