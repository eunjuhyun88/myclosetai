"""
Image Enhancer
이미지 품질 향상을 위한 후처리 클래스
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Optional, Union, Tuple
import numpy as np
from PIL import Image
import cv2
import logging

# 프로젝트 로깅 설정 import
from backend.app.ai_pipeline.utils.logging_config import get_logger

logger = get_logger(__name__)

class ImageEnhancer:
    """
    이미지 품질 향상을 위한 후처리 클래스
    """
    
    def __init__(self, device: Optional[torch.device] = None):
        """
        Args:
            device: 사용할 디바이스
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 향상 설정
        self.enhancement_config = {
            'sharpening_strength': 0.5,
            'contrast_enhancement': 0.3,
            'brightness_adjustment': 0.0,
            'saturation_boost': 0.2,
            'denoising_strength': 0.1,
            'edge_enhancement': 0.4
        }
        
        logger.info(f"ImageEnhancer initialized on device: {self.device}")
    
    def enhance_image(self, image: torch.Tensor, 
                     enhancement_type: str = 'comprehensive',
                     config: Optional[Dict[str, Any]] = None) -> torch.Tensor:
        """
        이미지 향상 실행
        
        Args:
            image: 입력 이미지
            enhancement_type: 향상 타입 ('comprehensive', 'sharpening', 'contrast', 'color')
            config: 향상 설정
            
        Returns:
            향상된 이미지
        """
        try:
            # 설정 병합
            if config is None:
                config = {}
            
            enhancement_config = {**self.enhancement_config, **config}
            
            # 이미지를 디바이스로 이동
            image = image.to(self.device)
            
            if enhancement_type == 'comprehensive':
                return self._comprehensive_enhancement(image, enhancement_config)
            elif enhancement_type == 'sharpening':
                return self._sharpen_image(image, enhancement_config)
            elif enhancement_type == 'contrast':
                return self._enhance_contrast(image, enhancement_config)
            elif enhancement_type == 'color':
                return self._enhance_color(image, enhancement_config)
            elif enhancement_type == 'edge':
                return self._enhance_edges(image, enhancement_config)
            else:
                logger.warning(f"알 수 없는 향상 타입: {enhancement_type}")
                return image
                
        except Exception as e:
            logger.error(f"이미지 향상 중 오류 발생: {e}")
            return image
    
    def _comprehensive_enhancement(self, image: torch.Tensor, 
                                  config: Dict[str, Any]) -> torch.Tensor:
        """종합적인 이미지 향상"""
        try:
            enhanced_image = image.clone()
            
            # 1. 노이즈 제거
            if config['denoising_strength'] > 0:
                enhanced_image = self._denoise_image(enhanced_image, config['denoising_strength'])
            
            # 2. 선명도 향상
            if config['sharpening_strength'] > 0:
                enhanced_image = self._sharpen_image(enhanced_image, config)
            
            # 3. 대비 향상
            if config['contrast_enhancement'] > 0:
                enhanced_image = self._enhance_contrast(enhanced_image, config)
            
            # 4. 색상 향상
            if config['saturation_boost'] > 0:
                enhanced_image = self._enhance_color(enhanced_image, config)
            
            # 5. 엣지 향상
            if config['edge_enhancement'] > 0:
                enhanced_image = self._enhance_edges(enhanced_image, config)
            
            # 6. 밝기 조정
            if abs(config['brightness_adjustment']) > 0:
                enhanced_image = self._adjust_brightness(enhanced_image, config['brightness_adjustment'])
            
            return enhanced_image
            
        except Exception as e:
            logger.error(f"종합 향상 중 오류 발생: {e}")
            return image
    
    def _sharpen_image(self, image: torch.Tensor, 
                       config: Dict[str, Any]) -> torch.Tensor:
        """이미지 선명도 향상"""
        try:
            strength = config['sharpening_strength']
            
            # 언샤프 마스크 필터
            kernel = torch.tensor([
                [0, -1, 0],
                [-1, 5, -1],
                [0, -1, 0]
            ], dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(0)
            
            # 컨볼루션 적용
            sharpened = F.conv2d(image.unsqueeze(0), kernel, padding=1)
            
            # 원본과 블렌딩
            enhanced = image * (1 - strength) + sharpened.squeeze(0) * strength
            
            return enhanced.clamp(0, 1)
            
        except Exception as e:
            logger.error(f"선명도 향상 중 오류 발생: {e}")
            return image
    
    def _enhance_contrast(self, image: torch.Tensor, 
                          config: Dict[str, Any]) -> torch.Tensor:
        """대비 향상"""
        try:
            strength = config['contrast_enhancement']
            
            # 히스토그램 평활화 기반 대비 향상
            enhanced = image.clone()
            
            for c in range(image.size(0)):
                channel = image[c]
                
                # 히스토그램 계산
                hist = torch.histc(channel, bins=256, min=0, max=1)
                cdf = torch.cumsum(hist, dim=0)
                cdf_normalized = cdf / cdf.max()
                
                # 대비 향상
                enhanced[c] = cdf_normalized[torch.floor(channel * 255).long()] * strength + \
                              channel * (1 - strength)
            
            return enhanced.clamp(0, 1)
            
        except Exception as e:
            logger.error(f"대비 향상 중 오류 발생: {e}")
            return image
    
    def _enhance_color(self, image: torch.Tensor, 
                       config: Dict[str, Any]) -> torch.Tensor:
        """색상 향상"""
        try:
            saturation_boost = config['saturation_boost']
            
            if image.size(0) != 3:  # RGB가 아닌 경우
                return image
            
            # HSV 변환
            hsv_image = self._rgb_to_hsv(image)
            
            # 채도 향상
            hsv_image[1] = hsv_image[1] * (1 + saturation_boost)
            hsv_image[1] = hsv_image[1].clamp(0, 1)
            
            # RGB로 변환
            enhanced = self._hsv_to_rgb(hsv_image)
            
            return enhanced.clamp(0, 1)
            
        except Exception as e:
            logger.error(f"색상 향상 중 오류 발생: {e}")
            return image
    
    def _enhance_edges(self, image: torch.Tensor, 
                       config: Dict[str, Any]) -> torch.Tensor:
        """엣지 향상"""
        try:
            strength = config['edge_enhancement']
            
            # 라플라시안 필터
            kernel = torch.tensor([
                [0, -1, 0],
                [-1, 4, -1],
                [0, -1, 0]
            ], dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(0)
            
            # 그레이스케일 변환
            if image.size(0) == 3:
                gray = 0.299 * image[0] + 0.587 * image[1] + 0.114 * image[2]
            else:
                gray = image[0]
            
            # 엣지 검출
            edges = F.conv2d(gray.unsqueeze(0).unsqueeze(0), kernel, padding=1)
            edges = edges.squeeze(0).squeeze(0)
            
            # 엣지 강화
            enhanced = image.clone()
            for c in range(image.size(0)):
                enhanced[c] = image[c] + edges * strength
            
            return enhanced.clamp(0, 1)
            
        except Exception as e:
            logger.error(f"엣지 향상 중 오류 발생: {e}")
            return image
    
    def _denoise_image(self, image: torch.Tensor, strength: float) -> torch.Tensor:
        """이미지 노이즈 제거"""
        try:
            # 가우시안 블러를 사용한 노이즈 제거
            kernel_size = int(3 + strength * 4)  # 3-7
            if kernel_size % 2 == 0:
                kernel_size += 1
            
            sigma = strength * 2.0
            
            # 가우시안 커널 생성
            kernel = self._create_gaussian_kernel(kernel_size, sigma)
            
            # 각 채널에 대해 블러 적용
            denoised = image.clone()
            for c in range(image.size(0)):
                denoised[c] = F.conv2d(
                    image[c].unsqueeze(0).unsqueeze(0), 
                    kernel, 
                    padding=kernel_size//2
                ).squeeze(0).squeeze(0)
            
            # 원본과 블렌딩
            result = image * (1 - strength) + denoised * strength
            
            return result.clamp(0, 1)
            
        except Exception as e:
            logger.error(f"노이즈 제거 중 오류 발생: {e}")
            return image
    
    def _adjust_brightness(self, image: torch.Tensor, adjustment: float) -> torch.Tensor:
        """밝기 조정"""
        try:
            # 감마 보정을 사용한 밝기 조정
            if adjustment > 0:
                gamma = 1.0 - adjustment
            else:
                gamma = 1.0 + abs(adjustment)
            
            adjusted = torch.pow(image, gamma)
            
            return adjusted.clamp(0, 1)
            
        except Exception as e:
            logger.error(f"밝기 조정 중 오류 발생: {e}")
            return image
    
    def _create_gaussian_kernel(self, kernel_size: int, sigma: float) -> torch.Tensor:
        """가우시안 커널 생성"""
        try:
            # 1D 가우시안 커널 생성
            x = torch.arange(-kernel_size//2, kernel_size//2 + 1, dtype=torch.float32)
            gaussian_1d = torch.exp(-(x**2) / (2 * sigma**2))
            gaussian_1d = gaussian_1d / gaussian_1d.sum()
            
            # 2D 가우시안 커널 생성
            gaussian_2d = gaussian_1d.unsqueeze(1) * gaussian_1d.unsqueeze(0)
            
            return gaussian_2d.unsqueeze(0).unsqueeze(0)
            
        except Exception as e:
            logger.error(f"가우시안 커널 생성 중 오류 발생: {e}")
            # 기본 커널 반환
            return torch.ones(1, 1, kernel_size, kernel_size) / (kernel_size * kernel_size)
    
    def _rgb_to_hsv(self, rgb_image: torch.Tensor) -> torch.Tensor:
        """RGB를 HSV로 변환"""
        try:
            r, g, b = rgb_image[0], rgb_image[1], rgb_image[2]
            
            # 최대값과 최소값
            max_rgb, _ = torch.max(rgb_image, dim=0)
            min_rgb, _ = torch.min(rgb_image, dim=0)
            diff = max_rgb - min_rgb
            
            # 명도 (Value)
            v = max_rgb
            
            # 채도 (Saturation)
            s = torch.zeros_like(max_rgb)
            s[max_rgb > 0] = diff[max_rgb > 0] / max_rgb[max_rgb > 0]
            
            # 색조 (Hue)
            h = torch.zeros_like(max_rgb)
            
            # R이 최대인 경우
            r_max_mask = (max_rgb == r) & (diff > 0)
            h[r_max_mask] = (60 * ((g[r_max_mask] - b[r_max_mask]) / diff[r_max_mask]) % 360) / 360
            
            # G가 최대인 경우
            g_max_mask = (max_rgb == g) & (diff > 0)
            h[g_max_mask] = (60 * ((b[g_max_mask] - r[g_max_mask]) / diff[g_max_mask] + 2) % 360) / 360
            
            # B가 최대인 경우
            b_max_mask = (max_rgb == b) & (diff > 0)
            h[b_max_mask] = (60 * ((r[b_max_mask] - g[b_max_mask]) / diff[b_max_mask] + 4) % 360) / 360
            
            # 음수 값 처리
            h[h < 0] += 1.0
            
            return torch.stack([h, s, v])
            
        except Exception as e:
            logger.error(f"RGB to HSV 변환 중 오류 발생: {e}")
            return rgb_image
    
    def _hsv_to_rgb(self, hsv_image: torch.Tensor) -> torch.Tensor:
        """HSV를 RGB로 변환"""
        try:
            h, s, v = hsv_image[0], hsv_image[1], hsv_image[2]
            
            # 색조를 0-6 범위로 변환
            h = h * 6
            
            # 정수 부분과 소수 부분
            i = torch.floor(h).long()
            f = h - i.float()
            
            # 보조 값들
            p = v * (1 - s)
            q = v * (1 - f * s)
            t = v * (1 - (1 - f) * s)
            
            # RGB 값 계산
            rgb = torch.zeros_like(hsv_image)
            
            # 각 구간별 RGB 값 설정
            mask_0 = (i == 0) | (i == 6)
            rgb[0, mask_0] = v[mask_0]
            rgb[1, mask_0] = t[mask_0]
            rgb[2, mask_0] = p[mask_0]
            
            mask_1 = i == 1
            rgb[0, mask_1] = q[mask_1]
            rgb[1, mask_1] = v[mask_1]
            rgb[2, mask_1] = p[mask_1]
            
            mask_2 = i == 2
            rgb[0, mask_2] = p[mask_2]
            rgb[1, mask_2] = v[mask_2]
            rgb[2, mask_2] = t[mask_2]
            
            mask_3 = i == 3
            rgb[0, mask_3] = p[mask_3]
            rgb[1, mask_3] = q[mask_3]
            rgb[2, mask_3] = v[mask_3]
            
            mask_4 = i == 4
            rgb[0, mask_4] = t[mask_4]
            rgb[1, mask_4] = p[mask_4]
            rgb[2, mask_4] = v[mask_4]
            
            mask_5 = i == 5
            rgb[0, mask_5] = v[mask_5]
            rgb[1, mask_5] = p[mask_5]
            rgb[2, mask_5] = q[mask_5]
            
            return rgb
            
        except Exception as e:
            logger.error(f"HSV to RGB 변환 중 오류 발생: {e}")
            return hsv_image
    
    def set_enhancement_config(self, **kwargs):
        """향상 설정 업데이트"""
        self.enhancement_config.update(kwargs)
        logger.info("향상 설정 업데이트 완료")
    
    def get_enhancement_config(self) -> Dict[str, Any]:
        """향상 설정 반환"""
        return self.enhancement_config.copy()
    
    def get_available_enhancements(self) -> List[str]:
        """사용 가능한 향상 방법들 반환"""
        return ['comprehensive', 'sharpening', 'contrast', 'color', 'edge']
