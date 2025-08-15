#!/usr/bin/env python3
"""
🔥 MyCloset AI - Quality Enhancer for Cloth Warping
====================================================

🎯 의류 워핑 품질 향상 프로세서
✅ 워핑 품질 자동 향상
✅ 노이즈 제거 및 선명도 향상
✅ 텍스처 보존 및 개선
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

logger = logging.getLogger(__name__)

@dataclass
class QualityEnhancerConfig:
    """품질 향상 설정"""
    enable_noise_reduction: bool = True
    enable_sharpness_enhancement: bool = True
    enable_texture_preservation: bool = True
    enable_color_enhancement: bool = True
    enable_detail_recovery: bool = True
    noise_reduction_strength: float = 0.5
    sharpness_strength: float = 0.7
    texture_preservation_strength: float = 0.8
    color_enhancement_strength: float = 0.6
    detail_recovery_strength: float = 0.9

class QualityEnhancer(nn.Module):
    """의류 워핑 품질 향상 프로세서"""
    
    def __init__(self, config: QualityEnhancerConfig = None):
        super().__init__()
        self.config = config or QualityEnhancerConfig()
        self.logger = logging.getLogger(__name__)
        
        # MPS 디바이스 확인
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.logger.info(f"🎯 Quality Enhancer 초기화 (디바이스: {self.device})")
        
        # 노이즈 제거 네트워크
        if self.config.enable_noise_reduction:
            self.noise_reduction_net = self._create_noise_reduction_net()
        
        # 선명도 향상 네트워크
        if self.config.enable_sharpness_enhancement:
            self.sharpness_enhancement_net = self._create_sharpness_enhancement_net()
        
        # 텍스처 보존 네트워크
        if self.config.enable_texture_preservation:
            self.texture_preservation_net = self._create_texture_preservation_net()
        
        # 색상 향상 네트워크
        if self.config.enable_color_enhancement:
            self.color_enhancement_net = self._create_color_enhancement_net()
        
        # 세부 복구 네트워크
        if self.config.enable_detail_recovery:
            self.detail_recovery_net = self._create_detail_recovery_net()
        
        self.logger.info("✅ Quality Enhancer 초기화 완료")
    
    def _create_noise_reduction_net(self) -> nn.Module:
        """노이즈 제거 네트워크 생성"""
        return nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 3, kernel_size=3, padding=1),
            nn.Sigmoid()
        ).to(self.device)
    
    def _create_sharpness_enhancement_net(self) -> nn.Module:
        """선명도 향상 네트워크 생성"""
        return nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 3, kernel_size=3, padding=1),
            nn.Tanh()
        ).to(self.device)
    
    def _create_texture_preservation_net(self) -> nn.Module:
        """텍스처 보존 네트워크 생성"""
        return nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
            nn.Tanh()
        ).to(self.device)
    
    def _create_color_enhancement_net(self) -> nn.Module:
        """색상 향상 네트워크 생성"""
        return nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 3, kernel_size=3, padding=1),
            nn.Sigmoid()
        ).to(self.device)
    
    def _create_detail_recovery_net(self) -> nn.Module:
        """세부 복구 네트워크 생성"""
        return nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
            nn.Tanh()
        ).to(self.device)
    
    def forward(self, warped_cloth: torch.Tensor, 
                original_cloth: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """
        품질 향상 수행
        
        Args:
            warped_cloth: 워핑된 의류 이미지 (B, C, H, W)
            original_cloth: 원본 의류 이미지 (B, C, H, W)
        
        Returns:
            품질 향상 결과
        """
        # 입력 검증
        if not self._validate_inputs(warped_cloth):
            raise ValueError("입력 검증 실패")
        
        # 디바이스 이동
        warped_cloth = warped_cloth.to(self.device)
        if original_cloth is not None:
            original_cloth = original_cloth.to(self.device)
        
        # 1단계: 노이즈 제거
        if self.config.enable_noise_reduction:
            denoised_cloth = self._reduce_noise(warped_cloth)
        else:
            denoised_cloth = warped_cloth
        
        # 2단계: 선명도 향상
        if self.config.enable_sharpness_enhancement:
            sharpened_cloth = self._enhance_sharpness(denoised_cloth)
        else:
            sharpened_cloth = denoised_cloth
        
        # 3단계: 텍스처 보존
        if self.config.enable_texture_preservation:
            texture_preserved_cloth = self._preserve_texture(sharpened_cloth, original_cloth)
        else:
            texture_preserved_cloth = sharpened_cloth
        
        # 4단계: 색상 향상
        if self.config.enable_color_enhancement:
            color_enhanced_cloth = self._enhance_color(texture_preserved_cloth)
        else:
            color_enhanced_cloth = texture_preserved_cloth
        
        # 5단계: 세부 복구
        if self.config.enable_detail_recovery:
            detail_recovered_cloth = self._recover_details(color_enhanced_cloth, original_cloth)
        else:
            detail_recovered_cloth = color_enhanced_cloth
        
        # 6단계: 최종 품질 검증
        final_enhanced_cloth = self._final_quality_validation(detail_recovered_cloth)
        
        # 결과 반환
        result = {
            "final_enhanced_cloth": final_enhanced_cloth,
            "denoised_cloth": denoised_cloth,
            "sharpened_cloth": sharpened_cloth,
            "texture_preserved_cloth": texture_preserved_cloth,
            "color_enhanced_cloth": color_enhanced_cloth,
            "detail_recovered_cloth": detail_recovered_cloth,
            "enhancement_steps": {
                "noise_reduction": self.config.enable_noise_reduction,
                "sharpness_enhancement": self.config.enable_sharpness_enhancement,
                "texture_preservation": self.config.enable_texture_preservation,
                "color_enhancement": self.config.enable_color_enhancement,
                "detail_recovery": self.config.enable_detail_recovery
            }
        }
        
        return result
    
    def _validate_inputs(self, warped_cloth: torch.Tensor) -> bool:
        """입력 검증"""
        if warped_cloth.dim() != 4:
            return False
        
        if warped_cloth.size(1) != 3:
            return False
        
        return True
    
    def _reduce_noise(self, warped_cloth: torch.Tensor) -> torch.Tensor:
        """노이즈 제거"""
        try:
            # 노이즈 제거 네트워크 적용
            noise_mask = self.noise_reduction_net(warped_cloth)
            
            # 노이즈 제거 적용
            denoised_cloth = warped_cloth * (1 - noise_mask * self.config.noise_reduction_strength)
            
            # 값 범위 제한
            denoised_cloth = torch.clamp(denoised_cloth, 0, 1)
            
            self.logger.debug("✅ 노이즈 제거 완료")
            return denoised_cloth
            
        except Exception as e:
            self.logger.warning(f"노이즈 제거 실패: {e}")
            return warped_cloth
    
    def _enhance_sharpness(self, warped_cloth: torch.Tensor) -> torch.Tensor:
        """선명도 향상"""
        try:
            # 선명도 향상 네트워크 적용
            sharpness_enhancement = self.sharpness_enhancement_net(warped_cloth)
            
            # 선명도 향상 적용
            sharpened_cloth = warped_cloth + sharpness_enhancement * self.config.sharpness_strength
            
            # 값 범위 제한
            sharpened_cloth = torch.clamp(sharpened_cloth, 0, 1)
            
            self.logger.debug("✅ 선명도 향상 완료")
            return sharpened_cloth
            
        except Exception as e:
            self.logger.warning(f"선명도 향상 실패: {e}")
            return warped_cloth
    
    def _preserve_texture(self, warped_cloth: torch.Tensor, 
                          original_cloth: torch.Tensor = None) -> torch.Tensor:
        """텍스처 보존"""
        try:
            if original_cloth is None:
                return warped_cloth
            
            # 텍스처 보존 네트워크 적용
            texture_preservation = self.texture_preservation_net(warped_cloth)
            
            # 원본 텍스처와 결합
            preserved_cloth = warped_cloth * (1 - self.config.texture_preservation_strength) + \
                             texture_preservation * self.config.texture_preservation_strength
            
            # 값 범위 제한
            preserved_cloth = torch.clamp(preserved_cloth, 0, 1)
            
            self.logger.debug("✅ 텍스처 보존 완료")
            return preserved_cloth
            
        except Exception as e:
            self.logger.warning(f"텍스처 보존 실패: {e}")
            return warped_cloth
    
    def _enhance_color(self, warped_cloth: torch.Tensor) -> torch.Tensor:
        """색상 향상"""
        try:
            # 색상 향상 네트워크 적용
            color_enhancement = self.color_enhancement_net(warped_cloth)
            
            # 색상 향상 적용
            enhanced_cloth = warped_cloth * (1 + color_enhancement * self.config.color_enhancement_strength)
            
            # 값 범위 제한
            enhanced_cloth = torch.clamp(enhanced_cloth, 0, 1)
            
            self.logger.debug("✅ 색상 향상 완료")
            return enhanced_cloth
            
        except Exception as e:
            self.logger.warning(f"색상 향상 실패: {e}")
            return warped_cloth
    
    def _recover_details(self, warped_cloth: torch.Tensor, 
                        original_cloth: torch.Tensor = None) -> torch.Tensor:
        """세부 복구"""
        try:
            if original_cloth is None:
                return warped_cloth
            
            # 세부 복구 네트워크 적용
            detail_recovery = self.detail_recovery_net(warped_cloth)
            
            # 원본 세부 정보와 결합
            recovered_cloth = warped_cloth * (1 - self.config.detail_recovery_strength) + \
                             detail_recovery * self.config.detail_recovery_strength
            
            # 값 범위 제한
            recovered_cloth = torch.clamp(recovered_cloth, 0, 1)
            
            self.logger.debug("✅ 세부 복구 완료")
            return recovered_cloth
            
        except Exception as e:
            self.logger.warning(f"세부 복구 실패: {e}")
            return warped_cloth
    
    def _final_quality_validation(self, enhanced_cloth: torch.Tensor) -> torch.Tensor:
        """최종 품질 검증"""
        try:
            # 품질 메트릭 계산
            quality_score = self._calculate_quality_score(enhanced_cloth)
            
            # 품질이 낮은 경우 추가 보정
            if quality_score < 0.7:
                enhanced_cloth = self._apply_quality_boost(enhanced_cloth)
            
            # 최종 검증
            enhanced_cloth = self._validate_output(enhanced_cloth)
            
            self.logger.debug(f"✅ 최종 품질 검증 완료 (품질 점수: {quality_score:.3f})")
            return enhanced_cloth
            
        except Exception as e:
            self.logger.warning(f"최종 품질 검증 실패: {e}")
            return enhanced_cloth
    
    def _calculate_quality_score(self, cloth: torch.Tensor) -> float:
        """품질 점수 계산"""
        try:
            with torch.no_grad():
                # 선명도 점수
                sharpness_score = self._calculate_sharpness_score(cloth)
                
                # 텍스처 점수
                texture_score = self._calculate_texture_score(cloth)
                
                # 색상 점수
                color_score = self._calculate_color_score(cloth)
                
                # 종합 품질 점수
                quality_score = (sharpness_score + texture_score + color_score) / 3
                
                return float(quality_score.item())
                
        except Exception as e:
            self.logger.warning(f"품질 점수 계산 실패: {e}")
            return 0.5
    
    def _calculate_sharpness_score(self, cloth: torch.Tensor) -> torch.Tensor:
        """선명도 점수 계산"""
        # 라플라시안 필터로 엣지 강도 계산
        laplacian_kernel = torch.tensor([
            [0, -1, 0],
            [-1, 4, -1],
            [0, -1, 0]
        ], dtype=torch.float32, device=cloth.device).unsqueeze(0).unsqueeze(0)
        
        edge_response = F.conv2d(cloth, laplacian_kernel, padding=1)
        sharpness_score = torch.mean(torch.abs(edge_response))
        
        return sharpness_score
    
    def _calculate_texture_score(self, cloth: torch.Tensor) -> torch.Tensor:
        """텍스처 점수 계산"""
        # 로컬 표준편차로 텍스처 품질 측정
        mean_cloth = F.avg_pool2d(cloth, kernel_size=5, stride=1, padding=2)
        variance = F.avg_pool2d(cloth**2, kernel_size=5, stride=1, padding=2) - mean_cloth**2
        texture_score = torch.mean(torch.sqrt(torch.clamp(variance, min=0)))
        
        return texture_score
    
    def _calculate_color_score(self, cloth: torch.Tensor) -> torch.Tensor:
        """색상 점수 계산"""
        # 색상 채널별 표준편차로 색상 품질 측정
        color_std = torch.std(cloth, dim=1)
        color_score = torch.mean(color_std)
        
        return color_score
    
    def _apply_quality_boost(self, cloth: torch.Tensor) -> torch.Tensor:
        """품질 향상 적용"""
        try:
            # 추가적인 품질 향상 처리
            boosted_cloth = cloth
            
            # 가우시안 블러로 노이즈 제거
            boosted_cloth = F.avg_pool2d(boosted_cloth, kernel_size=3, stride=1, padding=1)
            
            # 샤프닝 필터 적용
            sharpened = self._apply_sharpening_filter(boosted_cloth)
            boosted_cloth = boosted_cloth * 0.8 + sharpened * 0.2
            
            # 값 범위 제한
            boosted_cloth = torch.clamp(boosted_cloth, 0, 1)
            
            return boosted_cloth
            
        except Exception as e:
            self.logger.warning(f"품질 향상 적용 실패: {e}")
            return cloth
    
    def _apply_sharpening_filter(self, cloth: torch.Tensor) -> torch.Tensor:
        """샤프닝 필터 적용"""
        # 언샤프 마스킹
        blurred = F.avg_pool2d(cloth, kernel_size=3, stride=1, padding=1)
        sharpened = cloth + (cloth - blurred) * 0.5
        
        return sharpened
    
    def _validate_output(self, cloth: torch.Tensor) -> torch.Tensor:
        """출력 검증"""
        try:
            # 값 범위 검증
            if cloth.min() < 0 or cloth.max() > 1:
                cloth = torch.clamp(cloth, 0, 1)
            
            # NaN 검증
            if torch.isnan(cloth).any():
                cloth = torch.where(torch.isnan(cloth), torch.zeros_like(cloth), cloth)
            
            # 무한값 검증
            if torch.isinf(cloth).any():
                cloth = torch.where(torch.isinf(cloth), torch.zeros_like(cloth), cloth)
            
            return cloth
            
        except Exception as e:
            self.logger.warning(f"출력 검증 실패: {e}")
            return cloth
    
    def get_enhancement_stats(self, input_cloth: torch.Tensor, 
                             output_cloth: torch.Tensor) -> Dict[str, float]:
        """향상 통계 조회"""
        stats = {}
        
        try:
            with torch.no_grad():
                # 품질 향상 정도
                input_quality = self._calculate_quality_score(input_cloth)
                output_quality = self._calculate_quality_score(output_cloth)
                
                stats['input_quality'] = input_quality
                stats['output_quality'] = output_quality
                stats['quality_improvement'] = output_quality - input_quality
                stats['improvement_ratio'] = (output_quality / input_quality) if input_quality > 0 else 1.0
                
                # 세부 메트릭
                stats['sharpness_improvement'] = float(
                    self._calculate_sharpness_score(output_cloth) - self._calculate_sharpness_score(input_cloth)
                )
                stats['texture_improvement'] = float(
                    self._calculate_texture_score(output_cloth) - self._calculate_texture_score(input_cloth)
                )
                stats['color_improvement'] = float(
                    self._calculate_color_score(output_cloth) - self._calculate_color_score(input_cloth)
                )
                
        except Exception as e:
            self.logger.warning(f"향상 통계 계산 실패: {e}")
            stats = {
                'input_quality': 0.0,
                'output_quality': 0.0,
                'quality_improvement': 0.0,
                'improvement_ratio': 1.0,
                'sharpness_improvement': 0.0,
                'texture_improvement': 0.0,
                'color_improvement': 0.0
            }
        
        return stats

# 품질 향상 프로세서 인스턴스 생성
def create_quality_enhancer(config: QualityEnhancerConfig = None) -> QualityEnhancer:
    """Quality Enhancer 생성"""
    return QualityEnhancer(config)

if __name__ == "__main__":
    # 테스트 코드
    logging.basicConfig(level=logging.INFO)
    
    # 설정 생성
    config = QualityEnhancerConfig(
        enable_noise_reduction=True,
        enable_sharpness_enhancement=True,
        enable_texture_preservation=True,
        enable_color_enhancement=True,
        enable_detail_recovery=True
    )
    
    # 프로세서 생성
    processor = create_quality_enhancer(config)
    
    # 테스트 데이터 생성
    batch_size, channels, height, width = 2, 3, 256, 256
    test_cloth = torch.rand(batch_size, channels, height, width)
    original_cloth = torch.rand(batch_size, channels, height, width)
    
    # 품질 향상 수행
    result = processor(test_cloth, original_cloth)
    
    print(f"최종 향상된 의류 형태: {result['final_enhanced_cloth'].shape}")
    print(f"향상 단계: {result['enhancement_steps']}")
    
    # 향상 통계 계산
    stats = processor.get_enhancement_stats(test_cloth, result['final_enhanced_cloth'])
    print(f"향상 통계: {stats}")
