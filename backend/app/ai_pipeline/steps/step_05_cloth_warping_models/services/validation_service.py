#!/usr/bin/env python3
"""
🔥 MyCloset AI - Cloth Warping Validation Service
================================================

🎯 의류 워핑 검증 서비스
✅ 입력 데이터 검증
✅ 출력 결과 검증
✅ 품질 메트릭 계산
✅ M3 Max 최적화
"""

import logging
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
import cv2
from PIL import Image

logger = logging.getLogger(__name__)

@dataclass
class ValidationServiceConfig:
    """검증 서비스 설정"""
    enable_input_validation: bool = True
    enable_output_validation: bool = True
    enable_quality_metrics: bool = True
    min_image_size: Tuple[int, int] = (64, 64)
    max_image_size: Tuple[int, int] = (4096, 4096)
    use_mps: bool = True

class ClothWarpingValidationService:
    """의류 워핑 검증 서비스"""
    
    def __init__(self, config: ValidationServiceConfig = None):
        self.config = config or ValidationServiceConfig()
        self.logger = logging.getLogger(__name__)
        self.logger.info("🎯 Cloth Warping 검증 서비스 초기화")
        
        # MPS 디바이스 확인
        self.device = torch.device("mps" if torch.backends.mps.is_available() and self.config.use_mps else "cpu")
        
        # 검증 결과 저장
        self.validation_results = {}
        
        self.logger.info("✅ Cloth Warping 검증 서비스 초기화 완료")
    
    def validate_input_data(self, cloth_image: torch.Tensor, 
                           person_image: torch.Tensor) -> Dict[str, Any]:
        """입력 데이터를 검증합니다."""
        if not self.config.enable_input_validation:
            return {'status': 'disabled'}
        
        validation_results = {}
        
        try:
            # 기본 형태 검증
            if not isinstance(cloth_image, torch.Tensor) or not isinstance(person_image, torch.Tensor):
                validation_results['tensor_type'] = 'failed'
                validation_results['error'] = '입력이 torch.Tensor가 아닙니다.'
                return validation_results
            
            # 차원 검증
            if cloth_image.dim() != 4 or person_image.dim() != 4:
                validation_results['dimensions'] = 'failed'
                validation_results['error'] = '입력이 4차원 텐서가 아닙니다 (B, C, H, W).'
                return validation_results
            
            # 배치 크기 검증
            if cloth_image.shape[0] != person_image.shape[0]:
                validation_results['batch_size'] = 'failed'
                validation_results['error'] = '의류와 사람 이미지의 배치 크기가 다릅니다.'
                return validation_results
            
            # 채널 수 검증
            if cloth_image.shape[1] != 3 or person_image.shape[1] != 3:
                validation_results['channels'] = 'failed'
                validation_results['error'] = '입력이 3채널(RGB)이 아닙니다.'
                return validation_results
            
            # 이미지 크기 검증
            cloth_height, cloth_width = cloth_image.shape[2], cloth_image.shape[3]
            person_height, person_width = person_image.shape[2], person_image.shape[3]
            
            min_height, min_width = self.config.min_image_size
            max_height, max_width = self.config.max_image_size
            
            if (cloth_height < min_height or cloth_width < min_width or
                cloth_height > max_height or cloth_width > max_width):
                validation_results['cloth_size'] = 'failed'
                validation_results['error'] = f'의류 이미지 크기가 허용 범위를 벗어났습니다: {cloth_height}x{cloth_width}'
                return validation_results
            
            if (person_height < min_height or person_width < min_width or
                person_height > max_height or person_width > max_width):
                validation_results['person_size'] = 'failed'
                validation_results['error'] = '사람 이미지 크기가 허용 범위를 벗어났습니다.'
                return validation_results
            
            # 값 범위 검증
            if torch.min(cloth_image) < -1.0 or torch.max(cloth_image) > 1.0:
                validation_results['cloth_value_range'] = 'warning'
                validation_results['warning'] = '의류 이미지 값이 [-1, 1] 범위를 벗어났습니다.'
            
            if torch.min(person_image) < -1.0 or torch.max(person_image) > 1.0:
                validation_results['person_value_range'] = 'warning'
                validation_results['warning'] = '사람 이미지 값이 [-1, 1] 범위를 벗어났습니다.'
            
            # 디바이스 검증
            if cloth_image.device != self.device or person_image.device != self.device:
                validation_results['device'] = 'warning'
                validation_results['warning'] = '입력 이미지가 예상 디바이스에 있지 않습니다.'
            
            validation_results.update({
                'status': 'success',
                'cloth_shape': cloth_image.shape,
                'person_shape': person_image.shape,
                'batch_size': cloth_image.shape[0],
                'channels': cloth_image.shape[1],
                'cloth_size': (cloth_height, cloth_width),
                'person_size': (person_height, person_width)
            })
            
            self.logger.info("입력 데이터 검증 완료")
            
        except Exception as e:
            validation_results['status'] = 'error'
            validation_results['error'] = str(e)
            self.logger.error(f"입력 데이터 검증 실패: {e}")
        
        return validation_results
    
    def validate_output_data(self, output_data: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """출력 데이터를 검증합니다."""
        if not self.config.enable_output_validation:
            return {'status': 'disabled'}
        
        validation_results = {}
        
        try:
            # 필수 키 검증
            required_keys = ['warped_cloth', 'quality_score', 'validation_score']
            missing_keys = [key for key in required_keys if key not in output_data]
            
            if missing_keys:
                validation_results['required_keys'] = 'failed'
                validation_results['error'] = f'필수 키가 누락되었습니다: {missing_keys}'
                return validation_results
            
            # 워핑된 의류 이미지 검증
            warped_cloth = output_data['warped_cloth']
            if not isinstance(warped_cloth, torch.Tensor) or warped_cloth.dim() != 4:
                validation_results['warped_cloth_format'] = 'failed'
                validation_results['error'] = '워핑된 의류 이미지가 올바른 형식이 아닙니다.'
                return validation_results
            
            # 품질 점수 검증
            quality_score = output_data['quality_score']
            if not isinstance(quality_score, torch.Tensor):
                validation_results['quality_score_format'] = 'failed'
                validation_results['error'] = '품질 점수가 텐서가 아닙니다.'
                return validation_results
            
            if torch.min(quality_score) < 0.0 or torch.max(quality_score) > 1.0:
                validation_results['quality_score_range'] = 'warning'
                validation_results['warning'] = '품질 점수가 [0, 1] 범위를 벗어났습니다.'
            
            # 검증 점수 검증
            validation_score = output_data['validation_score']
            if not isinstance(validation_score, torch.Tensor):
                validation_results['validation_score_format'] = 'failed'
                validation_results['error'] = '검증 점수가 텐서가 아닙니다.'
                return validation_results
            
            if torch.min(validation_score) < 0.0 or torch.max(validation_score) > 1.0:
                validation_results['validation_score_range'] = 'warning'
                validation_results['warning'] = '검증 점수가 [0, 1] 범위를 벗어났습니다.'
            
            validation_results.update({
                'status': 'success',
                'warped_cloth_shape': warped_cloth.shape,
                'quality_score_shape': quality_score.shape,
                'validation_score_shape': validation_score.shape,
                'quality_score_range': (torch.min(quality_score).item(), torch.max(quality_score).item()),
                'validation_score_range': (torch.min(validation_score).item(), torch.max(validation_score).item())
            })
            
            self.logger.info("출력 데이터 검증 완료")
            
        except Exception as e:
            validation_results['status'] = 'error'
            validation_results['error'] = str(e)
            self.logger.error(f"출력 데이터 검증 실패: {e}")
        
        return validation_results
    
    def calculate_quality_metrics(self, original_cloth: torch.Tensor, 
                                 warped_cloth: torch.Tensor,
                                 target_person: torch.Tensor) -> Dict[str, Any]:
        """품질 메트릭을 계산합니다."""
        if not self.config.enable_quality_metrics:
            return {'status': 'disabled'}
        
        metrics = {}
        
        try:
            with torch.no_grad():
                # PSNR (Peak Signal-to-Noise Ratio) 계산
                mse = torch.mean((original_cloth - warped_cloth) ** 2)
                if mse > 0:
                    psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
                    metrics['psnr'] = psnr.item() if hasattr(psnr, 'item') else float(psnr)
                else:
                    metrics['psnr'] = float('inf')
                
                # SSIM (Structural Similarity Index) 계산 (간단한 버전)
                def simple_ssim(x, y, window_size=11):
                    # 간단한 SSIM 계산
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
                
                ssim = simple_ssim(original_cloth, warped_cloth)
                metrics['ssim'] = ssim.item() if hasattr(ssim, 'item') else float(ssim)
                
                # LPIPS (Learned Perceptual Image Patch Similarity) 계산 (간단한 버전)
                def simple_lpips(x, y):
                    # 간단한 LPIPS 계산 (실제로는 사전 훈련된 네트워크 사용)
                    diff = torch.abs(x - y)
                    lpips = torch.mean(diff)
                    return lpips
                
                lpips = simple_lpips(original_cloth, warped_cloth)
                metrics['lpips'] = lpips.item() if hasattr(lpips, 'item') else float(lpips)
                
                # 워핑 품질 점수 (의류와 사람 이미지 간의 일관성)
                cloth_person_similarity = torch.mean(torch.abs(warped_cloth - target_person))
                metrics['cloth_person_similarity'] = cloth_person_similarity.item() if hasattr(cloth_person_similarity, 'item') else float(cloth_person_similarity)
                
                # 전체 품질 점수 (여러 메트릭의 가중 평균)
                quality_score = (
                    0.4 * (1.0 / (1.0 + metrics['lpips'])) +  # LPIPS (낮을수록 좋음)
                    0.3 * metrics['ssim'] +                     # SSIM (높을수록 좋음)
                    0.2 * (metrics['psnr'] / 50.0) +           # PSNR (높을수록 좋음)
                    0.1 * (1.0 / (1.0 + metrics['cloth_person_similarity']))  # 일관성
                )
                metrics['overall_quality_score'] = quality_score.item() if hasattr(quality_score, 'item') else float(quality_score)
                
                metrics['status'] = 'success'
                self.logger.info("품질 메트릭 계산 완료")
                
        except Exception as e:
            metrics['status'] = 'error'
            metrics['error'] = str(e)
            self.logger.error(f"품질 메트릭 계산 실패: {e}")
        
        return metrics
    
    def validate_entire_pipeline(self, cloth_image: torch.Tensor,
                                person_image: torch.Tensor,
                                output_data: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """전체 파이프라인을 검증합니다."""
        self.logger.info("전체 파이프라인 검증 시작")
        
        validation_results = {}
        
        # 입력 데이터 검증
        input_validation = self.validate_input_data(cloth_image, person_image)
        validation_results['input_validation'] = input_validation
        
        # 출력 데이터 검증
        output_validation = self.validate_output_data(output_data)
        validation_results['output_validation'] = output_validation
        
        # 품질 메트릭 계산
        if 'warped_cloth' in output_data:
            quality_metrics = self.calculate_quality_metrics(
                cloth_image, 
                output_data['warped_cloth'], 
                person_image
            )
            validation_results['quality_metrics'] = quality_metrics
        
        # 전체 검증 상태 결정
        all_passed = True
        for validation_type, result in validation_results.items():
            if result.get('status') == 'failed':
                all_passed = False
                break
        
        validation_results['overall_status'] = 'success' if all_passed else 'failed'
        
        self.validation_results = validation_results
        self.logger.info(f"전체 파이프라인 검증 완료: {'성공' if all_passed else '실패'}")
        
        return validation_results
    
    def get_validation_results(self) -> Dict[str, Any]:
        """검증 결과를 반환합니다."""
        return self.validation_results
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """검증 요약을 반환합니다."""
        if not self.validation_results:
            return {'status': 'no_validation_run'}
        
        summary = {}
        for validation_type, result in self.validation_results.items():
            if isinstance(result, dict):
                summary[validation_type] = result.get('status', 'unknown')
        
        return summary

# 사용 예시
if __name__ == "__main__":
    # 설정
    config = ValidationServiceConfig(
        enable_input_validation=True,
        enable_output_validation=True,
        enable_quality_metrics=True,
        min_image_size=(64, 64),
        max_image_size=(4096, 4096),
        use_mps=True
    )
    
    # 검증 서비스 초기화
    validation_service = ClothWarpingValidationService(config)
    
    # 테스트 데이터 생성
    batch_size = 2
    channels = 3
    height = 256
    width = 256
    
    cloth_image = torch.randn(batch_size, channels, height, width)
    person_image = torch.randn(batch_size, channels, height, width)
    
    # 가상의 출력 데이터 생성
    output_data = {
        'warped_cloth': torch.randn(batch_size, channels, height, width),
        'quality_score': torch.rand(1, 1),
        'validation_score': torch.rand(1, 1)
    }
    
    # 전체 파이프라인 검증
    validation_results = validation_service.validate_entire_pipeline(
        cloth_image, person_image, output_data
    )
    
    # 결과 출력
    print("=== 검증 결과 ===")
    for validation_type, result in validation_results.items():
        print(f"\n{validation_type}:")
        if isinstance(result, dict):
            for key, value in result.items():
                print(f"  {key}: {value}")
    
    # 검증 요약
    summary = validation_service.get_validation_summary()
    print(f"\n=== 검증 요약 ===")
    print(f"전체 상태: {summary}")
