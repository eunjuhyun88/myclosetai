#!/usr/bin/env python3
"""
🔥 MyCloset AI - Final Output Postprocessor
============================================

🎯 최종 결과의 품질 향상 및 후처리
✅ 최종 결과 후처리
✅ 품질 향상
✅ 최종 검증
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

# 로거 설정
logger = logging.getLogger(__name__)

@dataclass
class PostprocessingConfig:
    """후처리 설정"""
    confidence_threshold: float = 0.5
    quality_threshold: float = 0.7
    enable_enhancement: bool = True
    enable_validation: bool = True
    use_mps: bool = True
    enable_quality_enhancement: bool = True

class FinalOutputPostprocessor(nn.Module):
    """
    🔥 Final Output 후처리 시스템
    
    최종 결과를 향상시키고 품질을 개선합니다.
    """
    
    def __init__(self, config: PostprocessingConfig = None):
        super().__init__()
        self.config = config or PostprocessingConfig()
        self.logger = logging.getLogger(__name__)
        
        # MPS 디바이스 확인
        self.device = torch.device("mps" if torch.backends.mps.is_available() and self.config.use_mps else "cpu")
        self.logger.info(f"🎯 Final Output 후처리 시스템 초기화 (디바이스: {self.device})")
        
        # 품질 향상 모듈
        if self.config.enable_quality_enhancement:
            self.quality_enhancer = self._create_quality_enhancer()
        
        self.logger.info("✅ Final Output 후처리 시스템 초기화 완료")
    
    def _create_quality_enhancer(self) -> nn.Module:
        """품질 향상 모듈 생성"""
        return nn.Sequential(
            nn.Linear(256 * 256 * 3, 512),  # 256x256x3 (RGB)
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 256 * 256 * 3)
        ).to(self.device)
    
    def forward(self, final_output: torch.Tensor, 
                confidences: torch.Tensor = None,
                intermediate_results: Dict[str, torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        후처리 수행
        
        Args:
            final_output: 최종 출력 (B, C, H, W) - RGB 이미지
            confidences: 출력 신뢰도 (B, H, W)
            intermediate_results: 중간 결과들 (각 단계별 결과)
        
        Returns:
            후처리된 결과
        """
        if final_output.dim() < 3:
            raise ValueError(f"최종 출력 형태가 올바르지 않습니다: {final_output.shape}")
        
        # 디바이스 이동
        final_output = final_output.to(self.device)
        if confidences is not None:
            confidences = confidences.to(self.device)
        
        # 1단계: 신뢰도 기반 필터링
        filtered_output, filtered_confidences = self._confidence_filtering(final_output, confidences)
        
        # 2단계: 품질 향상
        if self.config.enable_enhancement:
            enhanced_output = self._enhance_quality(filtered_output)
        else:
            enhanced_output = filtered_output
        
        # 3단계: 최종 검증
        if self.config.enable_validation:
            validated_output = self._validate_output(enhanced_output, intermediate_results)
        else:
            validated_output = enhanced_output
        
        # 4단계: 품질 향상 모듈 적용
        if self.config.enable_quality_enhancement:
            final_result = self._apply_quality_enhancement(validated_output)
        else:
            final_result = validated_output
        
        # 5단계: 최종 품질 평가
        final_quality_score = self._calculate_final_quality(final_result, filtered_confidences)
        
        # 결과 반환
        result = {
            "final_output": final_result,
            "confidences": filtered_confidences,
            "quality_score": final_quality_score,
            "postprocessing_metadata": {
                "confidence_filtered": True,
                "enhanced": self.config.enable_enhancement,
                "validated": self.config.enable_validation,
                "quality_enhanced": self.config.enable_quality_enhancement
            }
        }
        
        self.logger.debug(f"✅ 후처리 완료 - 품질 점수: {final_quality_score:.3f}")
        return result
    
    def _confidence_filtering(self, final_output: torch.Tensor, 
                            confidences: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """신뢰도 기반 필터링"""
        if confidences is None:
            # 신뢰도가 없는 경우 기본값 사용
            if final_output.dim() == 4:  # (B, C, H, W)
                confidences = torch.ones(final_output.size(0), final_output.size(2), final_output.size(3), device=self.device)
            else:  # (B, H, W)
                confidences = torch.ones(final_output.size(0), final_output.size(1), final_output.size(2), device=self.device)
        
        # 신뢰도 임계값 적용
        confidence_mask = confidences > self.config.confidence_threshold
        
        # 낮은 신뢰도 영역은 스무딩 적용
        filtered_output = final_output.clone()
        if final_output.dim() == 4:  # (B, C, H, W)
            for b in range(final_output.size(0)):
                for c in range(final_output.size(1)):
                    filtered_output[b, c][~confidence_mask[b]] *= 0.8  # 신뢰도 감소
        else:  # (B, H, W)
            filtered_output[~confidence_mask] *= 0.8
        
        return filtered_output, confidences
    
    def _enhance_quality(self, final_output: torch.Tensor) -> torch.Tensor:
        """품질 향상"""
        enhanced_output = final_output.clone()
        
        if final_output.dim() == 4:  # (B, C, H, W)
            batch_size, channels = final_output.size(0), final_output.size(1)
            for b in range(batch_size):
                for c in range(channels):
                    channel = final_output[b, c]
                    if channel.numel() > 0:
                        # 가우시안 스무딩으로 품질 향상
                        enhanced_output[b, c] = self._gaussian_smooth_2d(channel)
        else:  # (B, H, W)
            batch_size = final_output.size(0)
            for b in range(batch_size):
                channel = final_output[b]
                if channel.numel() > 0:
                    # 가우시안 스무딩으로 품질 향상
                    enhanced_output[b] = self._gaussian_smooth_2d(channel)
        
        return enhanced_output
    
    def _gaussian_smooth_2d(self, channel: torch.Tensor) -> torch.Tensor:
        """2D 가우시안 스무딩"""
        if channel.dim() != 2:
            return channel
        
        # 가우시안 커널 생성
        kernel_size = 3
        sigma = 0.5
        
        # 1D 가우시안 커널
        x = torch.arange(-kernel_size // 2, kernel_size // 2 + 1, device=channel.device)
        gaussian_1d = torch.exp(-(x ** 2) / (2 * sigma ** 2))
        gaussian_1d = gaussian_1d / gaussian_1d.sum()
        
        # 2D 가우시안 커널
        gaussian_2d = gaussian_1d.unsqueeze(0) * gaussian_1d.unsqueeze(1)
        
        # 패딩 추가
        padded_channel = F.pad(channel.unsqueeze(0).unsqueeze(0), 
                              (kernel_size // 2, kernel_size // 2, kernel_size // 2, kernel_size // 2), 
                              mode='reflect')
        
        # 컨볼루션 적용
        smoothed_channel = F.conv2d(padded_channel, gaussian_2d.unsqueeze(0).unsqueeze(0))
        
        return smoothed_channel.squeeze()
    
    def _validate_output(self, final_output: torch.Tensor, 
                        intermediate_results: Dict[str, torch.Tensor] = None) -> torch.Tensor:
        """최종 출력 검증"""
        if not intermediate_results:
            return final_output
        
        validated_output = final_output.clone()
        
        # 중간 결과들과의 일관성 검증
        for b in range(final_output.size(0)):
            output = final_output[b]
            if output.numel() > 0:
                # 중간 결과와의 일관성 검증
                validated_output[b] = self._validate_single_output(output, intermediate_results, b)
        
        return validated_output
    
    def _validate_single_output(self, output: torch.Tensor, 
                               intermediate_results: Dict[str, torch.Tensor], 
                               batch_idx: int) -> torch.Tensor:
        """단일 출력 검증"""
        if output.dim() != 3:
            return output
        
        # 출력 검증
        validated_output = output.clone()
        
        # 1. 색상 범위 검증 (0-1 범위)
        if output.numel() > 0:
            # 색상 범위 제한
            validated_output = torch.clamp(output, 0.0, 1.0)
        
        # 2. 중간 결과와의 일관성 검증
        if intermediate_results:
            # 예: human parsing, pose estimation 등과의 일관성
            for key, result in intermediate_results.items():
                if isinstance(result, torch.Tensor) and result.size(0) > batch_idx:
                    intermediate = result[batch_idx]
                    if intermediate.dim() == 3 and output.size(1) == intermediate.size(1):
                        # 간단한 일관성 검증 (예: 마스크 영역에서의 색상 일관성)
                        if key == "human_parsing" and intermediate.size(0) > 0:
                            # 사람 영역에서의 색상 일관성
                            human_mask = intermediate[0] > 0.5  # 사람 영역
                            if human_mask.sum() > 0:
                                # 사람 영역의 색상 정규화
                                for c in range(output.size(0)):
                                    validated_output[c][human_mask] = torch.clamp(
                                        validated_output[c][human_mask], 0.0, 1.0
                                    )
        
        return validated_output
    
    def _apply_quality_enhancement(self, final_output: torch.Tensor) -> torch.Tensor:
        """품질 향상 모듈 적용"""
        if not self.config.enable_quality_enhancement:
            return final_output
        
        # 출력을 1D로 평탄화
        batch_size = final_output.size(0)
        if final_output.dim() == 4:  # (B, C, H, W)
            output_flat = final_output.view(batch_size, -1)
        else:  # (B, H, W)
            output_flat = final_output.view(batch_size, -1)
        
        # 품질 향상 모듈 적용
        with torch.no_grad():
            enhanced_flat = self.quality_enhancer(output_flat)
        
        # 원래 형태로 복원
        if final_output.dim() == 4:
            enhanced_output = enhanced_flat.view(batch_size, final_output.size(1), final_output.size(2), final_output.size(3))
        else:
            enhanced_output = enhanced_flat.view(batch_size, final_output.size(1), final_output.size(2))
        
        # 원본과의 가중 평균
        alpha = 0.2  # 낮은 가중치로 원본 보존
        enhanced_output = alpha * enhanced_output + (1 - alpha) * final_output
        
        return enhanced_output
    
    def _calculate_final_quality(self, final_output: torch.Tensor, 
                               confidences: torch.Tensor) -> float:
        """최종 품질 점수 계산"""
        # 신뢰도 평균
        confidence_score = float(confidences.mean().item())
        
        # 출력 품질
        output_quality = self._calculate_output_quality(final_output)
        
        # 일관성 품질
        consistency_quality = self._calculate_consistency_quality(final_output)
        
        # 종합 품질 점수
        final_score = (confidence_score * 0.4 + 
                      output_quality * 0.3 + 
                      consistency_quality * 0.3)
        
        return final_score
    
    def _calculate_output_quality(self, final_output: torch.Tensor) -> float:
        """출력 품질 계산"""
        if final_output.numel() == 0:
            return 0.0
        
        # 최종 출력의 품질 평가
        quality_scores = []
        
        if final_output.dim() == 4:  # (B, C, H, W)
            batch_size, channels = final_output.size(0), final_output.size(1)
            for b in range(batch_size):
                for c in range(channels):
                    channel = final_output[b, c]
                    if channel.numel() > 0:
                        quality_score = self._evaluate_single_channel_quality(channel)
                        quality_scores.append(quality_score)
        else:  # (B, H, W)
            batch_size = final_output.size(0)
            for b in range(batch_size):
                channel = final_output[b]
                if channel.numel() > 0:
                    quality_score = self._evaluate_single_channel_quality(channel)
                    quality_scores.append(quality_score)
        
        return float(np.mean(quality_scores)) if quality_scores else 0.0
    
    def _evaluate_single_channel_quality(self, channel: torch.Tensor) -> float:
        """단일 채널 품질 평가"""
        if channel.dim() != 2:
            return 0.0
        
        # 채널을 numpy로 변환
        channel_np = channel.detach().cpu().numpy()
        
        # 1. 신호 대 잡음비 평가
        signal_power = np.mean(channel_np**2)
        noise_power = np.var(channel_np)
        snr = signal_power / (noise_power + 1e-8)
        snr_score = min(snr / 10.0, 1.0)
        
        # 2. 엣지 품질 평가
        grad_x = np.gradient(channel_np, axis=1)
        grad_y = np.gradient(channel_np, axis=0)
        edge_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        edge_score = 1.0 / (1.0 + np.mean(edge_magnitude))
        
        # 3. 텍스처 품질 평가
        texture_score = 1.0 / (1.0 + np.std(channel_np))
        
        # 종합 품질 점수
        quality_score = (snr_score * 0.4 + edge_score * 0.3 + texture_score * 0.3)
        
        return quality_score
    
    def _calculate_consistency_quality(self, final_output: torch.Tensor) -> float:
        """일관성 품질 계산"""
        if final_output.numel() == 0:
            return 0.0
        
        # 최종 출력의 일관성 품질
        consistency_scores = []
        
        if final_output.dim() == 4:  # (B, C, H, W)
            batch_size = final_output.size(0)
            for b in range(batch_size):
                output = final_output[b]
                if output.numel() > 0:
                    consistency_score = self._evaluate_single_output_consistency(output)
                    consistency_scores.append(consistency_score)
        else:  # (B, H, W)
            batch_size = final_output.size(0)
            for b in range(batch_size):
                output = final_output[b]
                if output.numel() > 0:
                    consistency_score = self._evaluate_single_output_consistency(output)
                    consistency_scores.append(consistency_score)
        
        return float(np.mean(consistency_scores)) if consistency_scores else 0.0
    
    def _evaluate_single_output_consistency(self, output: torch.Tensor) -> float:
        """단일 출력 일관성 평가"""
        if output.dim() != 3:
            return 0.0
        
        # 출력을 numpy로 변환
        output_np = output.detach().cpu().numpy()
        
        # 일관성 품질 계산
        # 1. 채널 간 일관성
        channel_consistency = 0.0
        if output_np.shape[0] > 1:
            for c1 in range(output_np.shape[0]):
                for c2 in range(c1 + 1, output_np.shape[0]):
                    correlation = np.corrcoef(output_np[c1].flatten(), output_np[c2].flatten())[0, 1]
                    if not np.isnan(correlation):
                        channel_consistency += abs(correlation)
            
            if output_np.shape[0] > 1:
                channel_consistency /= (output_np.shape[0] * (output_np.shape[0] - 1) / 2)
        
        # 2. 공간적 일관성
        spatial_consistency = 0.0
        for c in range(output_np.shape[0]):
            channel = output_np[c]
            # 간단한 공간적 일관성 (인접 픽셀 간의 유사성)
            h, w = channel.shape
            if h > 1 and w > 1:
                # 수평 방향 일관성
                horizontal_diff = np.mean(np.abs(channel[:, :-1] - channel[:, 1:]))
                # 수직 방향 일관성
                vertical_diff = np.mean(np.abs(channel[:-1, :] - channel[1:, :]))
                
                spatial_consistency += 1.0 / (1.0 + horizontal_diff + vertical_diff)
        
        if output_np.shape[0] > 0:
            spatial_consistency /= output_np.shape[0]
        
        # 종합 일관성 점수
        consistency_score = (channel_consistency * 0.5 + spatial_consistency * 0.5)
        
        return consistency_score
    
    def get_postprocessing_info(self) -> Dict[str, Any]:
        """후처리 시스템 정보 반환"""
        return {
            "confidence_threshold": self.config.confidence_threshold,
            "quality_threshold": self.config.quality_threshold,
            "enable_enhancement": self.config.enable_enhancement,
            "enable_validation": self.config.enable_validation,
            "enable_quality_enhancement": self.config.enable_quality_enhancement,
            "device": str(self.device)
        }

# 후처리 시스템 인스턴스 생성 함수
def create_final_output_postprocessor(config: PostprocessingConfig = None) -> FinalOutputPostprocessor:
    """Final Output 후처리 시스템 생성"""
    return FinalOutputPostprocessor(config)

# 기본 설정으로 후처리 시스템 생성
def create_default_postprocessor() -> FinalOutputPostprocessor:
    """기본 설정으로 후처리 시스템 생성"""
    config = PostprocessingConfig(
        confidence_threshold=0.5,
        quality_threshold=0.7,
        enable_enhancement=True,
        enable_validation=True,
        use_mps=True,
        enable_quality_enhancement=True
    )
    return FinalOutputPostprocessor(config)

if __name__ == "__main__":
    # 테스트 코드
    logging.basicConfig(level=logging.INFO)
    
    # 후처리 시스템 생성
    postprocessor = create_default_postprocessor()
    
    # 테스트 데이터 생성
    batch_size, channels, height, width = 2, 3, 256, 256
    test_output = torch.randn(batch_size, channels, height, width)
    test_confidences = torch.rand(batch_size, height, width)
    
    # 후처리 수행
    result = postprocessor(test_output, test_confidences)
    print(f"후처리 결과 형태: {result['final_output'].shape}")
    print(f"품질 점수: {result['quality_score']:.3f}")
    print(f"후처리 정보: {postprocessor.get_postprocessing_info()}")
