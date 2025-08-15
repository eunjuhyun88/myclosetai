#!/usr/bin/env python3
"""
🔥 MyCloset AI - Special Case Processor for Cloth Warping
==========================================================

🎯 의류 워핑 특수 케이스 처리 프로세서
✅ 복잡한 패턴 처리
✅ 투명/반투명 의류 처리
✅ 특수 소재 처리
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
class SpecialCaseProcessorConfig:
    """특수 케이스 처리 설정"""
    enable_complex_pattern_processing: bool = True
    enable_transparency_handling: bool = True
    enable_special_material_processing: bool = True
    enable_edge_case_detection: bool = True
    enable_adaptive_processing: bool = True
    pattern_complexity_threshold: float = 0.7
    transparency_threshold: float = 0.3
    material_detection_sensitivity: float = 0.8
    edge_case_threshold: float = 0.5

class SpecialCaseProcessor(nn.Module):
    """의류 워핑 특수 케이스 처리 프로세서"""
    
    def __init__(self, config: SpecialCaseProcessorConfig = None):
        super().__init__()
        self.config = config or SpecialCaseProcessorConfig()
        self.logger = logging.getLogger(__name__)
        
        # MPS 디바이스 확인
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.logger.info(f"🎯 Special Case Processor 초기화 (디바이스: {self.device})")
        
        # 복잡한 패턴 처리 네트워크
        if self.config.enable_complex_pattern_processing:
            self.complex_pattern_net = self._create_complex_pattern_net()
        
        # 투명도 처리 네트워크
        if self.config.enable_transparency_handling:
            self.transparency_net = self._create_transparency_net()
        
        # 특수 소재 처리 네트워크
        if self.config.enable_special_material_processing:
            self.special_material_net = self._create_special_material_net()
        
        # 엣지 케이스 감지 네트워크
        if self.config.enable_edge_case_detection:
            self.edge_case_detector = self._create_edge_case_detector()
        
        # 적응형 처리 네트워크
        if self.config.enable_adaptive_processing:
            self.adaptive_processor = self._create_adaptive_processor()
        
        self.logger.info("✅ Special Case Processor 초기화 완료")
    
    def _create_complex_pattern_net(self) -> nn.Module:
        """복잡한 패턴 처리 네트워크 생성"""
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
    
    def _create_transparency_net(self) -> nn.Module:
        """투명도 처리 네트워크 생성"""
        return nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 4, kernel_size=3, padding=1),  # RGB + Alpha
            nn.Sigmoid()
        ).to(self.device)
    
    def _create_special_material_net(self) -> nn.Module:
        """특수 소재 처리 네트워크 생성"""
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
    
    def _create_edge_case_detector(self) -> nn.Module:
        """엣지 케이스 감지 네트워크 생성"""
        return nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        ).to(self.device)
    
    def _create_adaptive_processor(self) -> nn.Module:
        """적응형 처리 네트워크 생성"""
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
    
    def forward(self, warped_cloth: torch.Tensor, 
                original_cloth: torch.Tensor = None,
                target_mask: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """
        특수 케이스 처리 수행
        
        Args:
            warped_cloth: 워핑된 의류 이미지 (B, C, H, W)
            original_cloth: 원본 의류 이미지 (B, C, H, W)
            target_mask: 타겟 마스크 (B, C, H, W)
        
        Returns:
            특수 케이스 처리 결과
        """
        # 입력 검증
        if not self._validate_inputs(warped_cloth):
            raise ValueError("입력 검증 실패")
        
        # 디바이스 이동
        warped_cloth = warped_cloth.to(self.device)
        if original_cloth is not None:
            original_cloth = original_cloth.to(self.device)
        if target_mask is not None:
            target_mask = target_mask.to(self.device)
        
        # 1단계: 특수 케이스 감지
        special_cases = self._detect_special_cases(warped_cloth)
        
        # 2단계: 복잡한 패턴 처리
        if self.config.enable_complex_pattern_processing and special_cases['has_complex_pattern']:
            pattern_processed_cloth = self._process_complex_pattern(warped_cloth)
        else:
            pattern_processed_cloth = warped_cloth
        
        # 3단계: 투명도 처리
        if self.config.enable_transparency_handling and special_cases['has_transparency']:
            transparency_processed_cloth = self._handle_transparency(pattern_processed_cloth)
        else:
            transparency_processed_cloth = pattern_processed_cloth
        
        # 4단계: 특수 소재 처리
        if self.config.enable_special_material_processing and special_cases['has_special_material']:
            material_processed_cloth = self._process_special_material(transparency_processed_cloth)
        else:
            material_processed_cloth = transparency_processed_cloth
        
        # 5단계: 엣지 케이스 처리
        if self.config.enable_edge_case_detection and special_cases['is_edge_case']:
            edge_case_processed_cloth = self._handle_edge_case(material_processed_cloth)
        else:
            edge_case_processed_cloth = material_processed_cloth
        
        # 6단계: 적응형 처리
        if self.config.enable_adaptive_processing:
            final_processed_cloth = self._apply_adaptive_processing(edge_case_processed_cloth, special_cases)
        else:
            final_processed_cloth = edge_case_processed_cloth
        
        # 결과 반환
        result = {
            "final_processed_cloth": final_processed_cloth,
            "pattern_processed_cloth": pattern_processed_cloth,
            "transparency_processed_cloth": transparency_processed_cloth,
            "material_processed_cloth": material_processed_cloth,
            "edge_case_processed_cloth": edge_case_processed_cloth,
            "special_cases": special_cases,
            "processing_config": {
                "complex_pattern_processing": self.config.enable_complex_pattern_processing,
                "transparency_handling": self.config.enable_transparency_handling,
                "special_material_processing": self.config.enable_special_material_processing,
                "edge_case_detection": self.config.enable_edge_case_detection,
                "adaptive_processing": self.config.enable_adaptive_processing
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
    
    def _detect_special_cases(self, warped_cloth: torch.Tensor) -> Dict[str, bool]:
        """특수 케이스 감지"""
        special_cases = {}
        
        try:
            with torch.no_grad():
                # 복잡한 패턴 감지
                pattern_complexity = self._calculate_pattern_complexity(warped_cloth)
                special_cases['has_complex_pattern'] = pattern_complexity > self.config.pattern_complexity_threshold
                
                # 투명도 감지
                transparency_level = self._calculate_transparency_level(warped_cloth)
                special_cases['has_transparency'] = transparency_level > self.config.transparency_threshold
                
                # 특수 소재 감지
                material_specialty = self._calculate_material_specialty(warped_cloth)
                special_cases['has_special_material'] = material_specialty > self.config.material_detection_sensitivity
                
                # 엣지 케이스 감지
                edge_case_score = self._calculate_edge_case_score(warped_cloth)
                special_cases['is_edge_case'] = edge_case_score > self.config.edge_case_threshold
                
                # 특수 케이스 통계
                special_cases['total_special_cases'] = sum(special_cases.values())
                special_cases['case_details'] = {
                    'pattern_complexity': float(pattern_complexity.item()),
                    'transparency_level': float(transparency_level.item()),
                    'material_specialty': float(material_specialty.item()),
                    'edge_case_score': float(edge_case_score.item())
                }
            
            self.logger.debug(f"✅ 특수 케이스 감지 완료: {special_cases['total_special_cases']}개")
            
        except Exception as e:
            self.logger.warning(f"특수 케이스 감지 실패: {e}")
            special_cases = {
                'has_complex_pattern': False,
                'has_transparency': False,
                'has_special_material': False,
                'is_edge_case': False,
                'total_special_cases': 0,
                'case_details': {}
            }
        
        return special_cases
    
    def _calculate_pattern_complexity(self, cloth: torch.Tensor) -> torch.Tensor:
        """패턴 복잡도 계산"""
        try:
            # 그라디언트 강도로 패턴 복잡도 측정
            grad_x = torch.gradient(cloth[:, 0, :, :], dim=2)[0]
            grad_y = torch.gradient(cloth[:, 1, :, :], dim=1)[0]
            
            gradient_magnitude = torch.sqrt(grad_x**2 + grad_y**2)
            pattern_complexity = torch.mean(gradient_magnitude)
            
            return pattern_complexity
            
        except Exception:
            return torch.tensor(0.5)
    
    def _calculate_transparency_level(self, cloth: torch.Tensor) -> torch.Tensor:
        """투명도 레벨 계산"""
        try:
            # 밝기와 대비로 투명도 추정
            brightness = torch.mean(cloth, dim=1)
            contrast = torch.std(cloth, dim=1)
            
            # 투명도 점수 (밝기가 높고 대비가 낮을수록 투명)
            transparency_score = brightness * (1 - contrast)
            transparency_level = torch.mean(transparency_score)
            
            return transparency_level
            
        except Exception:
            return torch.tensor(0.3)
    
    def _calculate_material_specialty(self, cloth: torch.Tensor) -> torch.Tensor:
        """소재 특수성 계산"""
        try:
            # 텍스처 특성으로 소재 특수성 측정
            # 로컬 표준편차의 변화
            local_std = F.avg_pool2d(cloth**2, kernel_size=5, stride=1, padding=2) - \
                       F.avg_pool2d(cloth, kernel_size=5, stride=1, padding=2)**2
            local_std = torch.sqrt(torch.clamp(local_std, min=0))
            
            # 표준편차의 표준편차 (변화의 변화)
            material_specialty = torch.std(local_std)
            
            return material_specialty
            
        except Exception:
            return torch.tensor(0.5)
    
    def _calculate_edge_case_score(self, cloth: torch.Tensor) -> torch.Tensor:
        """엣지 케이스 점수 계산"""
        try:
            # 엣지 케이스 감지 네트워크 적용
            edge_case_score = self.edge_case_detector(cloth)
            return torch.mean(edge_case_score)
            
        except Exception:
            return torch.tensor(0.3)
    
    def _process_complex_pattern(self, warped_cloth: torch.Tensor) -> torch.Tensor:
        """복잡한 패턴 처리"""
        try:
            # 복잡한 패턴 처리 네트워크 적용
            processed_cloth = self.complex_pattern_net(warped_cloth)
            
            # 원본과 결합
            final_cloth = warped_cloth * 0.7 + processed_cloth * 0.3
            final_cloth = torch.clamp(final_cloth, 0, 1)
            
            self.logger.debug("✅ 복잡한 패턴 처리 완료")
            return final_cloth
            
        except Exception as e:
            self.logger.warning(f"복잡한 패턴 처리 실패: {e}")
            return warped_cloth
    
    def _handle_transparency(self, warped_cloth: torch.Tensor) -> torch.Tensor:
        """투명도 처리"""
        try:
            # 투명도 처리 네트워크 적용
            rgba_output = self.transparency_net(warped_cloth)
            
            # RGB와 Alpha 분리
            rgb = rgba_output[:, :3, :, :]
            alpha = rgba_output[:, 3:4, :, :]
            
            # 투명도 적용
            processed_cloth = rgb * alpha + warped_cloth * (1 - alpha)
            processed_cloth = torch.clamp(processed_cloth, 0, 1)
            
            self.logger.debug("✅ 투명도 처리 완료")
            return processed_cloth
            
        except Exception as e:
            self.logger.warning(f"투명도 처리 실패: {e}")
            return warped_cloth
    
    def _process_special_material(self, warped_cloth: torch.Tensor) -> torch.Tensor:
        """특수 소재 처리"""
        try:
            # 특수 소재 처리 네트워크 적용
            processed_cloth = self.special_material_net(warped_cloth)
            
            # 원본과 결합
            final_cloth = warped_cloth * 0.6 + processed_cloth * 0.4
            final_cloth = torch.clamp(final_cloth, 0, 1)
            
            self.logger.debug("✅ 특수 소재 처리 완료")
            return final_cloth
            
        except Exception as e:
            self.logger.warning(f"특수 소재 처리 실패: {e}")
            return warped_cloth
    
    def _handle_edge_case(self, warped_cloth: torch.Tensor) -> torch.Tensor:
        """엣지 케이스 처리"""
        try:
            # 엣지 케이스에 대한 특별한 처리
            # 가우시안 블러로 부드럽게 처리
            blurred = F.avg_pool2d(warped_cloth, kernel_size=5, stride=1, padding=2)
            
            # 원본과 결합
            processed_cloth = warped_cloth * 0.8 + blurred * 0.2
            processed_cloth = torch.clamp(processed_cloth, 0, 1)
            
            self.logger.debug("✅ 엣지 케이스 처리 완료")
            return processed_cloth
            
        except Exception as e:
            self.logger.warning(f"엣지 케이스 처리 실패: {e}")
            return warped_cloth
    
    def _apply_adaptive_processing(self, warped_cloth: torch.Tensor, 
                                  special_cases: Dict[str, bool]) -> torch.Tensor:
        """적응형 처리 적용"""
        try:
            # 특수 케이스에 따른 적응형 처리
            if special_cases['total_special_cases'] > 2:
                # 여러 특수 케이스가 있는 경우 강화된 처리
                processed_cloth = self.adaptive_processor(warped_cloth)
                final_cloth = warped_cloth * 0.5 + processed_cloth * 0.5
            elif special_cases['total_special_cases'] > 0:
                # 일부 특수 케이스가 있는 경우 중간 강도 처리
                processed_cloth = self.adaptive_processor(warped_cloth)
                final_cloth = warped_cloth * 0.8 + processed_cloth * 0.2
            else:
                # 특수 케이스가 없는 경우 원본 유지
                final_cloth = warped_cloth
            
            final_cloth = torch.clamp(final_cloth, 0, 1)
            
            self.logger.debug("✅ 적응형 처리 완료")
            return final_cloth
            
        except Exception as e:
            self.logger.warning(f"적응형 처리 실패: {e}")
            return warped_cloth
    
    def get_processing_stats(self, input_cloth: torch.Tensor, 
                            output_cloth: torch.Tensor,
                            special_cases: Dict[str, bool]) -> Dict[str, Any]:
        """처리 통계 조회"""
        stats = {}
        
        try:
            with torch.no_grad():
                # 기본 처리 통계
                stats['input_shape'] = list(input_cloth.shape)
                stats['output_shape'] = list(output_cloth.shape)
                
                # 특수 케이스 통계
                stats['special_cases_detected'] = special_cases['total_special_cases']
                stats['case_details'] = special_cases.get('case_details', {})
                
                # 품질 메트릭
                stats['psnr'] = self._calculate_psnr(input_cloth, output_cloth)
                stats['ssim'] = self._calculate_ssim(input_cloth, output_cloth)
                
                # 특수 케이스별 처리 효과
                if special_cases.get('has_complex_pattern', False):
                    stats['pattern_processing_effect'] = self._calculate_pattern_processing_effect(
                        input_cloth, output_cloth
                    )
                
                if special_cases.get('has_transparency', False):
                    stats['transparency_processing_effect'] = self._calculate_transparency_processing_effect(
                        input_cloth, output_cloth
                    )
                
        except Exception as e:
            self.logger.warning(f"처리 통계 계산 실패: {e}")
            stats = {
                'input_shape': [0, 0, 0, 0],
                'output_shape': [0, 0, 0, 0],
                'special_cases_detected': 0,
                'case_details': {},
                'psnr': 0.0,
                'ssim': 0.0
            }
        
        return stats
    
    def _calculate_psnr(self, input_tensor: torch.Tensor, output_tensor: torch.Tensor) -> float:
        """PSNR 계산"""
        try:
            mse = F.mse_loss(input_tensor, output_tensor)
            if mse == 0:
                return float('inf')
            
            psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
            return float(psnr.item())
            
        except Exception:
            return 0.0
    
    def _calculate_ssim(self, input_tensor: torch.Tensor, output_tensor: torch.Tensor) -> float:
        """SSIM 계산 (간단한 버전)"""
        try:
            input_mean = input_tensor.mean()
            output_mean = output_tensor.mean()
            
            input_var = input_tensor.var()
            output_var = output_tensor.var()
            
            covariance = ((input_tensor - input_mean) * (output_tensor - output_mean)).mean()
            
            c1 = 0.01 ** 2
            c2 = 0.03 ** 2
            
            ssim = ((2 * input_mean * output_mean + c1) * (2 * covariance + c2)) / \
                   ((input_mean ** 2 + output_mean ** 2 + c1) * (input_var + output_var + c2))
            
            return float(ssim.item())
            
        except Exception:
            return 0.0
    
    def _calculate_pattern_processing_effect(self, input_cloth: torch.Tensor, 
                                           output_cloth: torch.Tensor) -> float:
        """패턴 처리 효과 계산"""
        try:
            input_pattern = self._calculate_pattern_complexity(input_cloth)
            output_pattern = self._calculate_pattern_complexity(output_cloth)
            
            effect = float((output_pattern - input_pattern).item())
            return effect
            
        except Exception:
            return 0.0
    
    def _calculate_transparency_processing_effect(self, input_cloth: torch.Tensor, 
                                                output_cloth: torch.Tensor) -> float:
        """투명도 처리 효과 계산"""
        try:
            input_transparency = self._calculate_transparency_level(input_cloth)
            output_transparency = self._calculate_transparency_level(output_cloth)
            
            effect = float((output_transparency - input_transparency).item())
            return effect
            
        except Exception:
            return 0.0

# 특수 케이스 처리 프로세서 인스턴스 생성
def create_special_case_processor(config: SpecialCaseProcessorConfig = None) -> SpecialCaseProcessor:
    """Special Case Processor 생성"""
    return SpecialCaseProcessor(config)

if __name__ == "__main__":
    # 테스트 코드
    logging.basicConfig(level=logging.INFO)
    
    # 설정 생성
    config = SpecialCaseProcessorConfig(
        enable_complex_pattern_processing=True,
        enable_transparency_handling=True,
        enable_special_material_processing=True,
        enable_edge_case_detection=True,
        enable_adaptive_processing=True
    )
    
    # 프로세서 생성
    processor = create_special_case_processor(config)
    
    # 테스트 데이터 생성
    batch_size, channels, height, width = 2, 3, 256, 256
    test_cloth = torch.rand(batch_size, channels, height, width)
    
    # 특수 케이스 처리 수행
    result = processor(test_cloth)
    
    print(f"최종 처리된 의류 형태: {result['final_processed_cloth'].shape}")
    print(f"특수 케이스: {result['special_cases']}")
    print(f"처리 설정: {result['processing_config']}")
    
    # 처리 통계 계산
    stats = processor.get_processing_stats(test_cloth, result['final_processed_cloth'], result['special_cases'])
    print(f"처리 통계: {stats}")
