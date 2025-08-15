"""
Virtual Fitting Rendering Utilities
가상 피팅에 필요한 렌더링 유틸리티 함수들을 제공합니다.
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from typing import Dict, List, Tuple, Optional, Any, Union
import cv2
import logging

logger = logging.getLogger(__name__)

class RenderingUtils:
    """가상 피팅 렌더링 유틸리티 클래스"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.RenderingUtils")
        self.version = "1.0.0"
    
    def apply_lighting_effects(
        self,
        image: torch.Tensor,
        lighting_type: str = "natural",
        intensity: float = 1.0
    ) -> torch.Tensor:
        """
        조명 효과 적용
        
        Args:
            image: 입력 이미지
            lighting_type: 조명 타입 (natural, warm, cool, dramatic)
            intensity: 조명 강도
        
        Returns:
            조명 효과가 적용된 이미지
        """
        try:
            if lighting_type == "natural":
                return self._apply_natural_lighting(image, intensity)
            elif lighting_type == "warm":
                return self._apply_warm_lighting(image, intensity)
            elif lighting_type == "cool":
                return self._apply_cool_lighting(image, intensity)
            elif lighting_type == "dramatic":
                return self._apply_dramatic_lighting(image, intensity)
            else:
                self.logger.warning(f"알 수 없는 조명 타입: {lighting_type}")
                return image
                
        except Exception as e:
            self.logger.error(f"조명 효과 적용 실패: {e}")
            return image
    
    def _apply_natural_lighting(self, image: torch.Tensor, intensity: float) -> torch.Tensor:
        """자연스러운 조명 효과"""
        try:
            # 밝기 조정
            brightness_factor = 1.0 + (intensity - 1.0) * 0.3
            enhanced = image * brightness_factor
            
            # 대비 조정
            contrast_factor = 1.0 + (intensity - 1.0) * 0.2
            enhanced = (enhanced - 0.5) * contrast_factor + 0.5
            
            return torch.clamp(enhanced, 0, 1)
            
        except Exception as e:
            self.logger.error(f"자연스러운 조명 효과 적용 실패: {e}")
            return image
    
    def _apply_warm_lighting(self, image: torch.Tensor, intensity: float) -> torch.Tensor:
        """따뜻한 조명 효과"""
        try:
            # 따뜻한 색조 (노란색/주황색)
            warm_filter = torch.tensor([1.1, 1.0, 0.9], device=image.device).view(1, 3, 1, 1)
            enhanced = image * warm_filter
            
            # 밝기 조정
            brightness_factor = 1.0 + (intensity - 1.0) * 0.2
            enhanced = enhanced * brightness_factor
            
            return torch.clamp(enhanced, 0, 1)
            
        except Exception as e:
            self.logger.error(f"따뜻한 조명 효과 적용 실패: {e}")
            return image
    
    def _apply_cool_lighting(self, image: torch.Tensor, intensity: float) -> torch.Tensor:
        """차가운 조명 효과"""
        try:
            # 차가운 색조 (파란색/청록색)
            cool_filter = torch.tensor([0.9, 1.0, 1.1], device=image.device).view(1, 3, 1, 1)
            enhanced = image * cool_filter
            
            # 밝기 조정
            brightness_factor = 1.0 + (intensity - 1.0) * 0.2
            enhanced = enhanced * brightness_factor
            
            return torch.clamp(enhanced, 0, 1)
            
        except Exception as e:
            self.logger.error(f"차가운 조명 효과 적용 실패: {e}")
            return image
    
    def _apply_dramatic_lighting(self, image: torch.Tensor, intensity: float) -> torch.Tensor:
        """드라마틱한 조명 효과"""
        try:
            # 대비 강화
            contrast_factor = 1.0 + (intensity - 1.0) * 0.5
            enhanced = (image - 0.5) * contrast_factor + 0.5
            
            # 채도 증가
            saturation_factor = 1.0 + (intensity - 1.0) * 0.3
            enhanced = enhanced * saturation_factor
            
            return torch.clamp(enhanced, 0, 1)
            
        except Exception as e:
            self.logger.error(f"드라마틱한 조명 효과 적용 실패: {e}")
            return image
    
    def apply_shadow_effects(
        self,
        image: torch.Tensor,
        shadow_strength: float = 0.3,
        shadow_position: str = "bottom_right"
    ) -> torch.Tensor:
        """
        그림자 효과 적용
        
        Args:
            image: 입력 이미지
            shadow_strength: 그림자 강도
            shadow_position: 그림자 위치
        
        Returns:
            그림자 효과가 적용된 이미지
        """
        try:
            # 그림자 마스크 생성
            shadow_mask = self._create_shadow_mask(image.shape, shadow_position)
            
            # 그림자 적용
            shadowed = image * (1 - shadow_mask * shadow_strength)
            
            return shadowed
            
        except Exception as e:
            self.logger.error(f"그림자 효과 적용 실패: {e}")
            return image
    
    def _create_shadow_mask(self, shape: Tuple[int, ...], position: str) -> torch.Tensor:
        """그림자 마스크 생성"""
        try:
            batch_size, channels, height, width = shape
            
            # 기본 마스크 생성
            mask = torch.zeros((batch_size, 1, height, width))
            
            if position == "bottom_right":
                # 우하단 그림자
                for i in range(height):
                    for j in range(width):
                        distance = np.sqrt((i - height) ** 2 + (j - width) ** 2)
                        mask[0, 0, i, j] = max(0, 1 - distance / (height + width) * 2)
            
            elif position == "bottom_left":
                # 좌하단 그림자
                for i in range(height):
                    for j in range(width):
                        distance = np.sqrt((i - height) ** 2 + j ** 2)
                        mask[0, 0, i, j] = max(0, 1 - distance / (height + width) * 2)
            
            elif position == "top_right":
                # 우상단 그림자
                for i in range(height):
                    for j in range(width):
                        distance = np.sqrt(i ** 2 + (j - width) ** 2)
                        mask[0, 0, i, j] = max(0, 1 - distance / (height + width) * 2)
            
            elif position == "top_left":
                # 좌상단 그림자
                for i in range(height):
                    for j in range(width):
                        distance = np.sqrt(i ** 2 + j ** 2)
                        mask[0, 0, i, j] = max(0, 1 - distance / (height + width) * 2)
            
            return mask
            
        except Exception as e:
            self.logger.error(f"그림자 마스크 생성 실패: {e}")
            return torch.zeros(shape)
    
    def apply_texture_enhancement(
        self,
        image: torch.Tensor,
        enhancement_strength: float = 0.5
    ) -> torch.Tensor:
        """
        텍스처 향상
        
        Args:
            image: 입력 이미지
            enhancement_strength: 향상 강도
        
        Returns:
            텍스처가 향상된 이미지
        """
        try:
            # 언샤프 마스킹
            blurred = F.avg_pool2d(image, kernel_size=3, stride=1, padding=1)
            enhanced = image + (image - blurred) * enhancement_strength
            
            return torch.clamp(enhanced, 0, 1)
            
        except Exception as e:
            self.logger.error(f"텍스처 향상 실패: {e}")
            return image
    
    def apply_color_grading(
        self,
        image: torch.Tensor,
        color_temperature: float = 0.0,  # -1.0 (차가움) ~ 1.0 (따뜻함)
        saturation: float = 1.0,
        brightness: float = 1.0
    ) -> torch.Tensor:
        """
        컬러 그레이딩 적용
        
        Args:
            image: 입력 이미지
            color_temperature: 색온도
            saturation: 채도
            brightness: 밝기
        
        Returns:
            컬러 그레이딩이 적용된 이미지
        """
        try:
            # 색온도 조정
            if color_temperature != 0.0:
                # 따뜻함 (노란색/주황색)
                if color_temperature > 0:
                    warm_filter = torch.tensor([
                        1.0 + color_temperature * 0.2,
                        1.0,
                        1.0 - color_temperature * 0.1
                    ], device=image.device).view(1, 3, 1, 1)
                    image = image * warm_filter
                
                # 차가움 (파란색/청록색)
                else:
                    cool_filter = torch.tensor([
                        1.0 + color_temperature * 0.1,
                        1.0,
                        1.0 - color_temperature * 0.2
                    ], device=image.device).view(1, 3, 1, 1)
                    image = image * cool_filter
            
            # 채도 조정
            if saturation != 1.0:
                # 그레이스케일 계산
                gray = image.mean(dim=1, keepdim=True)
                image = gray + (image - gray) * saturation
            
            # 밝기 조정
            if brightness != 1.0:
                image = image * brightness
            
            return torch.clamp(image, 0, 1)
            
        except Exception as e:
            self.logger.error(f"컬러 그레이딩 적용 실패: {e}")
            return image
    
    def get_info(self) -> Dict[str, Any]:
        """렌더링 유틸리티 정보 반환"""
        return {
            'module_name': 'rendering_utils',
            'version': self.version,
            'class_name': 'RenderingUtils',
            'methods': [
                'apply_lighting_effects',
                'apply_shadow_effects',
                'apply_texture_enhancement',
                'apply_color_grading'
            ],
            'description': '가상 피팅에 필요한 렌더링 유틸리티'
        }
