#!/usr/bin/env python3
"""
🔥 MyCloset AI - Cloth Warping Preprocessing
============================================

🎯 의류 워핑 전처리 모듈
✅ 의류 이미지 정규화 및 표준화
✅ 사람 이미지 전처리
✅ 워핑 준비 데이터 생성
✅ M3 Max 최적화
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
import cv2
from PIL import Image

logger = logging.getLogger(__name__)

@dataclass
class PreprocessingConfig:
    """전처리 설정"""
    input_size: Tuple[int, int] = (256, 256)
    output_size: Tuple[int, int] = (256, 256)
    normalize_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    normalize_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    enable_cloth_enhancement: bool = True
    enable_person_enhancement: bool = True
    enable_warping_preparation: bool = True
    use_mps: bool = True

class ClothEnhancementNetwork(nn.Module):
    """의류 향상 네트워크"""
    
    def __init__(self, input_channels: int = 3):
        super().__init__()
        self.input_channels = input_channels
        
        # 의류 향상을 위한 네트워크
        self.enhancement_net = nn.Sequential(
            nn.Conv2d(input_channels, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, input_channels, 3, padding=1),
            nn.Tanh()
        )
        
    def forward(self, x):
        # 의류 향상
        enhanced = self.enhancement_net(x)
        return enhanced

class PersonEnhancementNetwork(nn.Module):
    """사람 향상 네트워크"""
    
    def __init__(self, input_channels: int = 3):
        super().__init__()
        self.input_channels = input_channels
        
        # 사람 향상을 위한 네트워크
        self.enhancement_net = nn.Sequential(
            nn.Conv2d(input_channels, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, input_channels, 3, padding=1),
            nn.Tanh()
        )
        
    def forward(self, x):
        # 사람 향상
        enhanced = self.enhancement_net(x)
        return enhanced

class WarpingPreparationNetwork(nn.Module):
    """워핑 준비 네트워크"""
    
    def __init__(self, input_channels: int = 6):  # 3 for cloth + 3 for person
        super().__init__()
        self.input_channels = input_channels
        
        # 워핑 준비를 위한 네트워크
        self.preparation_net = nn.Sequential(
            nn.Conv2d(input_channels, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, input_channels, 3, padding=1),
            nn.Tanh()
        )
        
    def forward(self, x):
        # 워핑 준비
        prepared = self.preparation_net(x)
        return prepared

class ClothWarpingPreprocessor(nn.Module):
    """의류 워핑 전처리기"""
    
    def __init__(self, config: PreprocessingConfig = None):
        super().__init__()
        self.config = config or PreprocessingConfig()
        self.logger = logging.getLogger(__name__)
        
        # MPS 디바이스 확인
        self.device = torch.device("mps" if torch.backends.mps.is_available() and self.config.use_mps else "cpu")
        self.logger.info(f"🎯 Cloth Warping 전처리기 초기화 (디바이스: {self.device})")
        
        # 의류 향상 네트워크
        if self.config.enable_cloth_enhancement:
            self.cloth_enhancer = ClothEnhancementNetwork(3).to(self.device)
        
        # 사람 향상 네트워크
        if self.config.enable_person_enhancement:
            self.person_enhancer = PersonEnhancementNetwork(3).to(self.device)
        
        # 워핑 준비 네트워크
        if self.config.enable_warping_preparation:
            self.warping_preparer = WarpingPreparationNetwork(6).to(self.device)
        
        # 이미지 정규화
        self.normalizer = nn.Parameter(
            torch.tensor([self.config.normalize_mean, self.config.normalize_std], dtype=torch.float32), 
            requires_grad=False
        ).to(self.device)
        
        self.logger.info("✅ Cloth Warping 전처리기 초기화 완료")
    
    def forward(self, cloth_image: torch.Tensor, 
                person_image: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        의류와 사람 이미지를 전처리합니다.
        
        Args:
            cloth_image: 의류 이미지 (B, C, H, W)
            person_image: 사람 이미지 (B, C, H, W)
            
        Returns:
            전처리된 결과 딕셔너리
        """
        batch_size, channels, height, width = cloth_image.shape
        
        # 입력 검증
        if channels != 3:
            raise ValueError(f"Expected 3 channels, got {channels}")
        
        # 의류 향상
        if self.config.enable_cloth_enhancement:
            enhanced_cloth = self.cloth_enhancer(cloth_image)
            self.logger.debug("의류 향상 완료")
        else:
            enhanced_cloth = cloth_image
        
        # 사람 향상
        if self.config.enable_person_enhancement:
            enhanced_person = self.person_enhancer(person_image)
            self.logger.debug("사람 향상 완료")
        else:
            enhanced_person = person_image
        
        # 워핑 준비
        if self.config.enable_warping_preparation:
            combined_input = torch.cat([enhanced_cloth, enhanced_person], dim=1)
            prepared_input = self.warping_preparer(combined_input)
            self.logger.debug("워핑 준비 완료")
        else:
            prepared_input = torch.cat([enhanced_cloth, enhanced_person], dim=1)
        
        # 목표 해상도로 리사이즈
        if prepared_input.shape[-2:] != self.config.output_size:
            prepared_input = F.interpolate(
                prepared_input, 
                size=self.config.output_size, 
                mode='bilinear', 
                align_corners=False
            )
        
        # 정규화 적용
        mean = self.normalizer[0].view(1, 3, 1, 1)
        std = self.normalizer[1].view(1, 3, 1, 1)
        
        # 의류와 사람 이미지 분리하여 정규화
        cloth_normalized = (enhanced_cloth - mean) / std
        person_normalized = (enhanced_person - mean) / std
        
        # 결과 반환
        result = {
            'prepared_input': prepared_input,
            'enhanced_cloth': enhanced_cloth,
            'enhanced_person': enhanced_person,
            'cloth_normalized': cloth_normalized,
            'person_normalized': person_normalized,
            'output_size': self.config.output_size,
            'input_size': (height, width)
        }
        
        return result
    
    def process_batch(self, batch_cloth: List[torch.Tensor], 
                     batch_person: List[torch.Tensor]) -> List[Dict[str, torch.Tensor]]:
        """
        배치 단위로 전처리를 수행합니다.
        
        Args:
            batch_cloth: 의류 이미지 배치 리스트
            batch_person: 사람 이미지 배치 리스트
            
        Returns:
            전처리된 결과 배치 리스트
        """
        results = []
        
        for i, (cloth, person) in enumerate(zip(batch_cloth, batch_person)):
            try:
                result = self.forward(cloth, person)
                results.append(result)
                self.logger.debug(f"배치 {i} 전처리 완료")
            except Exception as e:
                self.logger.error(f"배치 {i} 전처리 실패: {e}")
                # 에러 발생 시 원본 이미지 반환
                results.append({
                    'prepared_input': torch.cat([cloth, person], dim=1),
                    'enhanced_cloth': cloth,
                    'enhanced_person': person,
                    'cloth_normalized': cloth,
                    'person_normalized': person,
                    'output_size': cloth.shape[-2:],
                    'input_size': cloth.shape[-2:]
                })
        
        return results
    
    def get_preprocessing_stats(self) -> Dict[str, Any]:
        """전처리 통계를 반환합니다."""
        return {
            'cloth_enhancement_enabled': self.config.enable_cloth_enhancement,
            'person_enhancement_enabled': self.config.enable_person_enhancement,
            'warping_preparation_enabled': self.config.enable_warping_preparation,
            'input_size': self.config.input_size,
            'output_size': self.config.output_size,
            'device': str(self.device),
            'config': self.config.__dict__
        }

# 사용 예시
if __name__ == "__main__":
    # 설정
    config = PreprocessingConfig(
        input_size=(256, 256),
        output_size=(256, 256),
        enable_cloth_enhancement=True,
        enable_person_enhancement=True,
        enable_warping_preparation=True,
        use_mps=True
    )
    
    # 전처리기 초기화
    preprocessor = ClothWarpingPreprocessor(config)
    
    # 테스트 입력
    batch_size = 2
    channels = 3
    height = 256
    width = 256
    
    test_cloth = torch.randn(batch_size, channels, height, width)
    test_person = torch.randn(batch_size, channels, height, width)
    
    # 전처리 수행
    with torch.no_grad():
        result = preprocessor(test_cloth, test_person)
        
        print("✅ 전처리 완료!")
        print(f"의류 이미지 형태: {test_cloth.shape}")
        print(f"사람 이미지 형태: {test_person.shape}")
        print(f"준비된 입력 형태: {result['prepared_input'].shape}")
        print(f"출력 크기: {result['output_size']}")
        print(f"전처리 통계: {preprocessor.get_preprocessing_stats()}")
