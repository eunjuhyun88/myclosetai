#!/usr/bin/env python3
"""
🔥 MyCloset AI - Virtual Fitting Validator
=========================================

🎯 가상 피팅 검증기
✅ 가상 피팅 결과 검증
✅ 품질 메트릭 계산
✅ 일관성 검사
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
class ValidationConfig:
    """검증 설정"""
    enable_quality_metrics: bool = True
    enable_consistency_check: bool = True
    enable_realism_evaluation: bool = True
    enable_fitting_accuracy: bool = True
    quality_threshold: float = 0.7
    use_mps: bool = True

class VirtualFittingQualityEvaluator(nn.Module):
    """가상 피팅 품질 평가기"""
    
    def __init__(self, input_channels: int = 3):
        super().__init__()
        self.input_channels = input_channels
        
        # 품질 평가를 위한 네트워크
        self.quality_net = nn.Sequential(
            nn.Conv2d(input_channels, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # 품질 평가
        quality_score = self.quality_net(x)
        return quality_score

class VirtualFittingConsistencyChecker(nn.Module):
    """가상 피팅 일관성 검사기"""
    
    def __init__(self, input_channels: int = 6):  # 3 for original + 3 for result
        super().__init__()
        self.input_channels = input_channels
        
        # 일관성 검사를 위한 네트워크
        self.consistency_net = nn.Sequential(
            nn.Conv2d(input_channels, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # 일관성 검사
        consistency_score = self.consistency_net(x)
        return consistency_score

class VirtualFittingRealismEvaluator(nn.Module):
    """가상 피팅 현실성 평가기"""
    
    def __init__(self, input_channels: int = 3):
        super().__init__()
        self.input_channels = input_channels
        
        # 현실성 평가를 위한 네트워크
        self.realism_net = nn.Sequential(
            nn.Conv2d(input_channels, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # 현실성 평가
        realism_score = self.realism_net(x)
        return realism_score

class VirtualFittingAccuracyChecker(nn.Module):
    """가상 피팅 정확도 검사기"""
    
    def __init__(self, input_channels: int = 6):  # 3 for target + 3 for result
        super().__init__()
        self.input_channels = input_channels
        
        # 정확도 검사를 위한 네트워크
        self.accuracy_net = nn.Sequential(
            nn.Conv2d(input_channels, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # 정확도 검사
        accuracy_score = self.accuracy_net(x)
        return accuracy_score

class VirtualFittingValidator(nn.Module):
    """가상 피팅 검증기"""
    
    def __init__(self, config: ValidationConfig = None):
        super().__init__()
        self.config = config or ValidationConfig()
        self.logger = logging.getLogger(__name__)
        
        # MPS 디바이스 확인
        self.device = torch.device("mps" if torch.backends.mps.is_available() and self.config.use_mps else "cpu")
        self.logger.info(f"🎯 Virtual Fitting 검증기 초기화 (디바이스: {self.device})")
        
        # 품질 평가기
        if self.config.enable_quality_metrics:
            self.quality_evaluator = VirtualFittingQualityEvaluator(3).to(self.device)
        
        # 일관성 검사기
        if self.config.enable_consistency_check:
            self.consistency_checker = VirtualFittingConsistencyChecker(6).to(self.device)
        
        # 현실성 평가기
        if self.config.enable_realism_evaluation:
            self.realism_evaluator = VirtualFittingRealismEvaluator(3).to(self.device)
        
        # 정확도 검사기
        if self.config.enable_fitting_accuracy:
            self.accuracy_checker = VirtualFittingAccuracyChecker(6).to(self.device)
        
        self.logger.info("✅ Virtual Fitting 검증기 초기화 완료")
    
    def forward(self, virtual_fitting_result: torch.Tensor,
                original_image: Optional[torch.Tensor] = None,
                target_image: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        가상 피팅 결과를 검증합니다.
        
        Args:
            virtual_fitting_result: 가상 피팅 결과 이미지 (B, C, H, W)
            original_image: 원본 이미지 (B, C, H, W)
            target_image: 목표 이미지 (B, C, H, W)
            
        Returns:
            검증 결과 딕셔너리
        """
        batch_size, channels, height, width = virtual_fitting_result.shape
        
        # 입력 검증
        if channels != 3:
            raise ValueError(f"Expected 3 channels, got {channels}")
        
        # 입력을 디바이스로 이동
        virtual_fitting_result = virtual_fitting_result.to(self.device)
        if original_image is not None:
            original_image = original_image.to(self.device)
        if target_image is not None:
            target_image = target_image.to(self.device)
        
        validation_results = {}
        
        # 품질 평가
        if self.config.enable_quality_metrics:
            quality_score = self.quality_evaluator(virtual_fitting_result)
            validation_results['quality_score'] = quality_score
            self.logger.debug("품질 평가 완료")
        
        # 일관성 검사
        if self.config.enable_consistency_check and original_image is not None:
            combined_input = torch.cat([original_image, virtual_fitting_result], dim=1)
            consistency_score = self.consistency_checker(combined_input)
            validation_results['consistency_score'] = consistency_score
            self.logger.debug("일관성 검사 완료")
        
        # 현실성 평가
        if self.config.enable_realism_evaluation:
            realism_score = self.realism_evaluator(virtual_fitting_result)
            validation_results['realism_score'] = realism_score
            self.logger.debug("현실성 평가 완료")
        
        # 정확도 검사
        if self.config.enable_fitting_accuracy and target_image is not None:
            combined_input = torch.cat([target_image, virtual_fitting_result], dim=1)
            accuracy_score = self.accuracy_checker(combined_input)
            validation_results['accuracy_score'] = accuracy_score
            self.logger.debug("정확도 검사 완료")
        
        # 전체 검증 점수 계산
        if validation_results:
            scores = [score.mean().item() for score in validation_results.values()]
            overall_score = sum(scores) / len(scores)
            validation_results['overall_score'] = torch.tensor([[overall_score]], device=self.device)
            
            # 품질 임계값 확인
            validation_results['quality_passed'] = overall_score >= self.config.quality_threshold
        
        # 결과 반환
        result = {
            'validation_results': validation_results,
            'input_size': (height, width),
            'batch_size': batch_size
        }
        
        return result
    
    def calculate_quality_metrics(self, original_image: torch.Tensor, 
                                 virtual_fitting_result: torch.Tensor) -> Dict[str, float]:
        """품질 메트릭을 계산합니다."""
        if not self.config.enable_quality_metrics:
            return {'status': 'disabled'}
        
        metrics = {}
        
        try:
            with torch.no_grad():
                # PSNR (Peak Signal-to-Noise Ratio) 계산
                mse = torch.mean((original_image - virtual_fitting_result) ** 2)
                if mse > 0:
                    psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
                    metrics['psnr'] = psnr.item() if hasattr(psnr, 'item') else float(psnr)
                else:
                    metrics['psnr'] = float('inf')
                
                # SSIM (Structural Similarity Index) 계산 (간단한 버전)
                def simple_ssim(x, y):
                    mu_x = torch.mean(x)
                    mu_y = torch.mean(y)
                    sigma_x = torch.std(x)
                    sigma_y = torch.std(y)
                    sigma_xy = torch.mean((x - mu_x) * (y - mu_y))
                    
                    c1 = 0.01 ** 2
                    c2 = 0.03 ** 2
                    
                    ssim = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / \
                           ((mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x ** 2 + sigma_y ** 2 + c2))
                    
                    return ssim
                
                ssim = simple_ssim(original_image, virtual_fitting_result)
                metrics['ssim'] = ssim.item() if hasattr(ssim, 'item') else float(ssim)
                
                # LPIPS (Learned Perceptual Image Patch Similarity) 계산 (간단한 버전)
                def simple_lpips(x, y):
                    diff = torch.mean(torch.abs(x - y))
                    return diff
                
                lpips = simple_lpips(original_image, virtual_fitting_result)
                metrics['lpips'] = lpips.item() if hasattr(lpips, 'item') else float(lpips)
                
                metrics['status'] = 'success'
                self.logger.info("품질 메트릭 계산 완료")
                
        except Exception as e:
            metrics['status'] = 'error'
            metrics['error'] = str(e)
            self.logger.error(f"품질 메트릭 계산 실패: {e}")
        
        return metrics
    
    def validate_batch(self, batch_virtual_fitting: List[torch.Tensor],
                      batch_original: Optional[List[torch.Tensor]] = None,
                      batch_target: Optional[List[torch.Tensor]] = None) -> List[Dict[str, torch.Tensor]]:
        """
        배치 단위로 검증을 수행합니다.
        
        Args:
            batch_virtual_fitting: 가상 피팅 결과 배치 리스트
            batch_original: 원본 이미지 배치 리스트
            batch_target: 목표 이미지 배치 리스트
            
        Returns:
            검증 결과 배치 리스트
        """
        results = []
        
        for i, virtual_fitting in enumerate(batch_virtual_fitting):
            try:
                original = batch_original[i] if batch_original else None
                target = batch_target[i] if batch_target else None
                
                result = self.forward(virtual_fitting, original, target)
                results.append(result)
                self.logger.debug(f"배치 {i} 검증 완료")
            except Exception as e:
                self.logger.error(f"배치 {i} 검증 실패: {e}")
                # 에러 발생 시 기본 결과 반환
                results.append({
                    'validation_results': {},
                    'input_size': virtual_fitting.shape[-2:],
                    'batch_size': virtual_fitting.shape[0]
                })
        
        return results
    
    def get_validation_stats(self) -> Dict[str, Any]:
        """검증 통계를 반환합니다."""
        return {
            'quality_metrics_enabled': self.config.enable_quality_metrics,
            'consistency_check_enabled': self.config.enable_consistency_check,
            'realism_evaluation_enabled': self.config.enable_realism_evaluation,
            'fitting_accuracy_enabled': self.config.enable_fitting_accuracy,
            'quality_threshold': self.config.quality_threshold,
            'device': str(self.device),
            'config': self.config.__dict__
        }

# 사용 예시
if __name__ == "__main__":
    # 설정
    config = ValidationConfig(
        enable_quality_metrics=True,
        enable_consistency_check=True,
        enable_realism_evaluation=True,
        enable_fitting_accuracy=True,
        quality_threshold=0.7,
        use_mps=True
    )
    
    # 검증기 초기화
    validator = VirtualFittingValidator(config)
    
    # 테스트 입력
    batch_size = 2
    channels = 3
    height = 256
    width = 256
    
    test_virtual_fitting = torch.randn(batch_size, channels, height, width)
    test_original = torch.randn(batch_size, channels, height, width)
    test_target = torch.randn(batch_size, channels, height, width)
    
    # 검증 수행
    with torch.no_grad():
        result = validator(test_virtual_fitting, test_original, test_target)
        
        print("✅ 검증 완료!")
        print(f"가상 피팅 결과 형태: {test_virtual_fitting.shape}")
        print(f"검증 결과: {result}")
        
        # 품질 메트릭 계산
        metrics = validator.calculate_quality_metrics(test_original, test_virtual_fitting)
        print(f"품질 메트릭: {metrics}")
        
        # 검증 통계
        stats = validator.get_validation_stats()
        print(f"검증 통계: {stats}")
