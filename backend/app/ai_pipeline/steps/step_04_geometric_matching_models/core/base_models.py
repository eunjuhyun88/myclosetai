"""
Base model classes for geometric matching.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional


class BaseOpticalFlowModel(nn.Module):
    """기본 광학 흐름 모델 클래스"""
    
    def __init__(self):
        super().__init__()
        
    def forward(self, img1, img2):
        """
        기본 forward 메서드
        Args:
            img1: 첫 번째 이미지
            img2: 두 번째 이미지
        Returns:
            광학 흐름 결과
        """
        raise NotImplementedError("Subclasses must implement forward method")
    
    def _validate_inputs(self, img1, img2):
        """입력 검증"""
        if img1.shape != img2.shape:
            raise ValueError("Input images must have the same shape")
        if len(img1.shape) != 4:
            raise ValueError("Input images must be 4D tensors (B, C, H, W)")
    
    def _compute_flow(self, img1, img2):
        """광학 흐름 계산"""
        return self.forward(img1, img2)
    
    def _format_result(self, result, device):
        """결과 포맷팅"""
        if isinstance(result, torch.Tensor):
            return result.to(device)
        return result


class BaseGeometricMatcher(nn.Module):
    """기본 기하학적 매칭 모델 클래스"""
    
    def __init__(self, input_nc=6, **kwargs):
        super().__init__()
        self.input_nc = input_nc
        
    def forward(self, person_image, clothing_image):
        """
        기본 forward 메서드
        Args:
            person_image: 사람 이미지
            clothing_image: 의류 이미지
        Returns:
            기하학적 매칭 결과
        """
        raise NotImplementedError("Subclasses must implement forward method")
    
    def _validate_inputs(self, person_image, clothing_image):
        """입력 검증"""
        if person_image.shape[1:] != clothing_image.shape[1:]:
            raise ValueError("Person and clothing images must have the same spatial dimensions")
    
    def _init_common_components(self, **kwargs):
        """공통 컴포넌트 초기화"""
        pass
    
    def _compute_matching(self, person_image, clothing_image):
        """매칭 계산"""
        return self.forward(person_image, clothing_image)
    
    def _format_result(self, result, device):
        """결과 포맷팅"""
        if isinstance(result, torch.Tensor):
            return result.to(device)
        return result
