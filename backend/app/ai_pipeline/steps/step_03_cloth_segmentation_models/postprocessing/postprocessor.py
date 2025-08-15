#!/usr/bin/env python3
"""
🔥 MyCloset AI - Cloth Segmentation Postprocessor
=================================================

🎯 의류 분할 결과의 품질 향상 및 후처리
✅ 마스크 후처리
✅ 노이즈 제거
✅ 경계 정제
✅ M3 Max 최적화
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
import cv2

# PyTorch import 시도
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    # torch가 없을 때는 기본 타입 사용
    class MockNNModule:
        """Mock nn.Module (torch 없음)"""
        pass
    # nn.Module을 MockNNModule으로 대체
    class nn:
        Module = MockNNModule
    F = None

# 로거 설정
logger = logging.getLogger(__name__)

@dataclass
class PostprocessingConfig:
    """후처리 설정"""
    confidence_threshold: float = 0.5
    noise_reduction_strength: float = 0.3
    boundary_refinement: bool = True
    hole_filling: bool = True
    morphological_operations: bool = True
    use_mps: bool = True
    enable_quality_enhancement: bool = True

class ClothSegmentationPostprocessor(nn.Module):
    """
    🔥 Cloth Segmentation 후처리 시스템
    
    의류 분할 결과를 향상시키고 품질을 개선합니다.
    """
    
    def __init__(self, config: PostprocessingConfig = None):
        super().__init__()
        self.config = config or PostprocessingConfig()
        self.logger = logging.getLogger(__name__)
        
        # MPS 디바이스 확인
        self.device = torch.device("mps" if TORCH_AVAILABLE and torch.backends.mps.is_available() and self.config.use_mps else "cpu")
        self.logger.info(f"🎯 Cloth Segmentation 후처리 시스템 초기화 (디바이스: {self.device})")
        
        # 품질 향상 모듈
        if self.config.enable_quality_enhancement:
            self.quality_enhancer = self._create_quality_enhancer()
        
        self.logger.info("✅ Cloth Segmentation 후처리 시스템 초기화 완료")
    
    def _create_quality_enhancer(self) -> nn.Module:
        """품질 향상 모듈 생성"""
        if not TORCH_AVAILABLE:
            raise ImportError("Torch is not available. Cannot create quality enhancer.")
        return nn.Sequential(
            nn.Linear(256 * 256, 512),  # 256x256 이미지
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 256 * 256)
        ).to(self.device)
    
    def forward(self, masks: torch.Tensor, 
                confidences: torch.Tensor = None,
                image_size: Tuple[int, int] = None) -> Dict[str, torch.Tensor]:
        """
        후처리 수행
        
        Args:
            masks: 분할 마스크 (B, C, H, W) 또는 (B, H, W)
            confidences: 마스크 신뢰도 (B, C) 또는 (B,)
            image_size: 이미지 크기 (H, W)
        
        Returns:
            후처리된 결과
        """
        if not TORCH_AVAILABLE:
            raise ImportError("Torch is not available. Cannot perform postprocessing.")

        if masks.dim() < 3:
            raise ValueError(f"마스크 형태가 올바르지 않습니다: {masks.shape}")
        
        # 디바이스 이동
        masks = masks.to(self.device)
        if confidences is not None:
            confidences = confidences.to(self.device)
        
        # 1단계: 신뢰도 기반 필터링
        filtered_masks, filtered_confidences = self._confidence_filtering(masks, confidences)
        
        # 2단계: 노이즈 제거
        denoised_masks = self._reduce_noise(filtered_masks)
        
        # 3단계: 경계 정제
        refined_masks = self._refine_boundaries(denoised_masks)
        
        # 4단계: 홀 채우기
        if self.config.hole_filling:
            filled_masks = self._fill_holes(refined_masks)
        else:
            filled_masks = refined_masks
        
        # 5단계: 형태학적 연산
        if self.config.morphological_operations:
            morphological_masks = self._apply_morphological_operations(filled_masks)
        else:
            morphological_masks = filled_masks
        
        # 6단계: 품질 향상
        if self.config.enable_quality_enhancement:
            enhanced_masks = self._enhance_quality(morphological_masks)
        else:
            enhanced_masks = morphological_masks
        
        # 7단계: 최종 품질 평가
        final_quality_score = self._calculate_final_quality(enhanced_masks, filtered_confidences)
        
        # 결과 반환
        result = {
            "masks": enhanced_masks,
            "confidences": filtered_confidences,
            "quality_score": final_quality_score,
            "postprocessing_metadata": {
                "confidence_filtered": True,
                "noise_reduced": True,
                "boundaries_refined": True,
                "holes_filled": self.config.hole_filling,
                "morphological_applied": self.config.morphological_operations,
                "quality_enhanced": self.config.enable_quality_enhancement
            }
        }
        
        self.logger.debug(f"✅ 후처리 완료 - 품질 점수: {final_quality_score:.3f}")
        return result
    
    def _confidence_filtering(self, masks: torch.Tensor, 
                            confidences: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """신뢰도 기반 필터링"""
        if confidences is None:
            # 신뢰도가 없는 경우 기본값 사용
            if masks.dim() == 4:  # (B, C, H, W)
                confidences = torch.ones(masks.size(0), masks.size(1), device=self.device)
            else:  # (B, H, W)
                confidences = torch.ones(masks.size(0), device=self.device)
        
        # 신뢰도 임계값 적용
        confidence_mask = confidences > self.config.confidence_threshold
        
        # 낮은 신뢰도 마스크는 0으로 설정
        filtered_masks = masks.clone()
        if masks.dim() == 4:  # (B, C, H, W)
            for b in range(masks.size(0)):
                for c in range(masks.size(1)):
                    if not confidence_mask[b, c]:
                        filtered_masks[b, c] = 0.0
        else:  # (B, H, W)
            for b in range(masks.size(0)):
                if not confidence_mask[b]:
                    filtered_masks[b] = 0.0
        
        return filtered_masks, confidences
    
    def _reduce_noise(self, masks: torch.Tensor) -> torch.Tensor:
        """노이즈 제거"""
        denoised_masks = masks.clone()
        
        if masks.dim() == 4:  # (B, C, H, W)
            batch_size, channels = masks.size(0), masks.size(1)
            for b in range(batch_size):
                for c in range(channels):
                    mask = masks[b, c]
                    if mask.numel() > 0:
                        # 가우시안 스무딩
                        denoised_mask = self._gaussian_smooth_2d(mask)
                        denoised_masks[b, c] = denoised_mask
        else:  # (B, H, W)
            batch_size = masks.size(0)
            for b in range(batch_size):
                mask = masks[b]
                if mask.numel() > 0:
                    # 가우시안 스무딩
                    denoised_mask = self._gaussian_smooth_2d(mask)
                    denoised_masks[b] = denoised_mask
        
        return denoised_masks
    
    def _gaussian_smooth_2d(self, mask: torch.Tensor) -> torch.Tensor:
        """2D 가우시안 스무딩"""
        if mask.dim() != 2:
            return mask
        
        # 가우시안 커널 생성
        kernel_size = 5
        sigma = 1.0
        
        # 1D 가우시안 커널
        x = torch.arange(-kernel_size // 2, kernel_size // 2 + 1, device=mask.device)
        gaussian_1d = torch.exp(-(x ** 2) / (2 * sigma ** 2))
        gaussian_1d = gaussian_1d / gaussian_1d.sum()
        
        # 2D 가우시안 커널
        gaussian_2d = gaussian_1d.unsqueeze(0) * gaussian_1d.unsqueeze(1)
        
        # 패딩 추가
        padded_mask = F.pad(mask.unsqueeze(0).unsqueeze(0), 
                           (kernel_size // 2, kernel_size // 2, kernel_size // 2, kernel_size // 2), 
                           mode='reflect')
        
        # 컨볼루션 적용
        smoothed_mask = F.conv2d(padded_mask, gaussian_2d.unsqueeze(0).unsqueeze(0))
        
        return smoothed_mask.squeeze()
    
    def _refine_boundaries(self, masks: torch.Tensor) -> torch.Tensor:
        """경계 정제"""
        if not self.config.boundary_refinement:
            return masks
        
        refined_masks = masks.clone()
        
        if masks.dim() == 4:  # (B, C, H, W)
            batch_size, channels = masks.size(0), masks.size(1)
            for b in range(batch_size):
                for c in range(channels):
                    mask = masks[b, c]
                    if mask.numel() > 0:
                        # 경계 정제
                        refined_mask = self._refine_single_mask_boundary(mask)
                        refined_masks[b, c] = refined_mask
        else:  # (B, H, W)
            batch_size = masks.size(0)
            for b in range(batch_size):
                mask = masks[b]
                if mask.numel() > 0:
                    # 경계 정제
                    refined_mask = self._refine_single_mask_boundary(mask)
                    refined_masks[b] = refined_mask
        
        return refined_masks
    
    def _refine_single_mask_boundary(self, mask: torch.Tensor) -> torch.Tensor:
        """단일 마스크 경계 정제"""
        if mask.dim() != 2:
            return mask
        
        # 마스크를 numpy로 변환
        mask_np = mask.detach().cpu().numpy()
        
        # 이진화
        binary_mask = (mask_np > 0.5).astype(np.uint8)
        
        # 경계 검출
        edges = cv2.Canny(binary_mask * 255, 50, 150)
        
        # 경계 정제 (모폴로지 연산)
        kernel = np.ones((3, 3), np.uint8)
        refined_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        # 원본 마스크와 정제된 경계 결합
        refined_mask = mask_np.copy()
        refined_mask[refined_edges > 0] = 0.8  # 경계 강화
        
        return torch.from_numpy(refined_mask).to(mask.device)
    
    def _fill_holes(self, masks: torch.Tensor) -> torch.Tensor:
        """홀 채우기"""
        filled_masks = masks.clone()
        
        if masks.dim() == 4:  # (B, C, H, W)
            batch_size, channels = masks.size(0), masks.size(1)
            for b in range(batch_size):
                for c in range(channels):
                    mask = masks[b, c]
                    if mask.numel() > 0:
                        # 홀 채우기
                        filled_mask = self._fill_single_mask_holes(mask)
                        filled_masks[b, c] = filled_mask
        else:  # (B, H, W)
            batch_size = masks.size(0)
            for b in range(batch_size):
                mask = masks[b]
                if mask.numel() > 0:
                    # 홀 채우기
                    filled_mask = self._fill_single_mask_holes(mask)
                    filled_masks[b] = filled_mask
        
        return filled_masks
    
    def _fill_single_mask_holes(self, mask: torch.Tensor) -> torch.Tensor:
        """단일 마스크 홀 채우기"""
        if mask.dim() != 2:
            return mask
        
        # 마스크를 numpy로 변환
        mask_np = mask.detach().cpu().numpy()
        
        # 이진화
        binary_mask = (mask_np > 0.5).astype(np.uint8)
        
        # 홀 채우기
        filled_mask = cv2.fillPoly(binary_mask, [cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0][0]], 1)
        
        # 원본 마스크와 결합
        result_mask = mask_np.copy()
        result_mask[filled_mask > 0] = torch.max(mask).item()
        
        return torch.from_numpy(result_mask).to(mask.device)
    
    def _apply_morphological_operations(self, masks: torch.Tensor) -> torch.Tensor:
        """형태학적 연산 적용"""
        morphological_masks = masks.clone()
        
        if masks.dim() == 4:  # (B, C, H, W)
            batch_size, channels = masks.size(0), masks.size(1)
            for b in range(batch_size):
                for c in range(channels):
                    mask = masks[b, c]
                    if mask.numel() > 0:
                        # 형태학적 연산
                        morphological_mask = self._apply_single_mask_morphology(mask)
                        morphological_masks[b, c] = morphological_mask
        else:  # (B, H, W)
            batch_size = masks.size(0)
            for b in range(batch_size):
                mask = masks[b]
                if mask.numel() > 0:
                    # 형태학적 연산
                    morphological_mask = self._apply_single_mask_morphology(mask)
                    morphological_masks[b] = morphological_mask
        
        return morphological_masks
    
    def _apply_single_mask_morphology(self, mask: torch.Tensor) -> torch.Tensor:
        """단일 마스크 형태학적 연산"""
        if mask.dim() != 2:
            return mask
        
        # 마스크를 numpy로 변환
        mask_np = mask.detach().cpu().numpy()
        
        # 이진화
        binary_mask = (mask_np > 0.5).astype(np.uint8)
        
        # 모폴로지 연산
        kernel = np.ones((3, 3), np.uint8)
        
        # 열기 연산 (Opening) - 노이즈 제거
        opened_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
        
        # 닫기 연산 (Closing) - 홀 채우기
        closed_mask = cv2.morphologyEx(opened_mask, cv2.MORPH_CLOSE, kernel)
        
        # 원본 마스크와 결합
        result_mask = mask_np.copy()
        result_mask[closed_mask > 0] = torch.max(mask).item()
        
        return torch.from_numpy(result_mask).to(mask.device)
    
    def _enhance_quality(self, masks: torch.Tensor) -> torch.Tensor:
        """품질 향상"""
        if not self.config.enable_quality_enhancement:
            return masks
        
        # 마스크를 1D로 평탄화
        batch_size = masks.size(0)
        if masks.dim() == 4:  # (B, C, H, W)
            masks_flat = masks.view(batch_size, -1)
        else:  # (B, H, W)
            masks_flat = masks.view(batch_size, -1)
        
        # 품질 향상 모듈 적용
        if not TORCH_AVAILABLE:
            raise ImportError("Torch is not available. Cannot enhance quality.")
        with torch.no_grad():
            enhanced_flat = self.quality_enhancer(masks_flat)
        
        # 원래 형태로 복원
        if masks.dim() == 4:
            enhanced_masks = enhanced_flat.view(batch_size, masks.size(1), masks.size(2), masks.size(3))
        else:
            enhanced_masks = enhanced_flat.view(batch_size, masks.size(1), masks.size(2))
        
        # 원본과의 가중 평균
        alpha = 0.3
        enhanced_masks = alpha * enhanced_masks + (1 - alpha) * masks
        
        return enhanced_masks
    
    def _calculate_final_quality(self, masks: torch.Tensor, 
                               confidences: torch.Tensor) -> float:
        """최종 품질 점수 계산"""
        # 신뢰도 평균
        confidence_score = float(confidences.mean().item())
        
        # 마스크 품질
        mask_quality = self._calculate_mask_quality(masks)
        
        # 경계 품질
        boundary_quality = self._calculate_boundary_quality(masks)
        
        # 종합 품질 점수
        final_score = (confidence_score * 0.4 + 
                      mask_quality * 0.3 + 
                      boundary_quality * 0.3)
        
        return final_score
    
    def _calculate_mask_quality(self, masks: torch.Tensor) -> float:
        """마스크 품질 계산"""
        if masks.numel() == 0:
            return 0.0
        
        # 마스크의 연결성 및 일관성 평가
        quality_scores = []
        
        if masks.dim() == 4:  # (B, C, H, W)
            batch_size, channels = masks.size(0), masks.size(1)
            for b in range(batch_size):
                for c in range(channels):
                    mask = masks[b, c]
                    if mask.numel() > 0:
                        quality_score = self._evaluate_single_mask_quality(mask)
                        quality_scores.append(quality_score)
        else:  # (B, H, W)
            batch_size = masks.size(0)
            for b in range(batch_size):
                mask = masks[b]
                if mask.numel() > 0:
                    quality_score = self._evaluate_single_mask_quality(mask)
                    quality_scores.append(quality_score)
        
        return float(np.mean(quality_scores)) if quality_scores else 0.0
    
    def _evaluate_single_mask_quality(self, mask: torch.Tensor) -> float:
        """단일 마스크 품질 평가"""
        if mask.dim() != 2:
            return 0.0
        
        # 마스크를 numpy로 변환
        mask_np = mask.detach().cpu().numpy()
        
        # 이진화
        binary_mask = (mask_np > 0.5).astype(np.uint8)
        
        # 1. 면적 비율 평가
        area_ratio = np.sum(binary_mask) / binary_mask.size
        area_score = min(area_ratio * 10, 1.0)
        
        # 2. 연결성 평가
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        connectivity_score = 1.0 / (len(contours) + 1)
        
        # 3. 원형도 평가
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            contour_area = cv2.contourArea(largest_contour)
            contour_perimeter = cv2.arcLength(largest_contour, True)
            
            if contour_perimeter > 0:
                circularity = 4 * np.pi * contour_area / (contour_perimeter ** 2)
            else:
                circularity = 0.0
        else:
            circularity = 0.0
        
        # 종합 품질 점수
        quality_score = (area_score * 0.4 + connectivity_score * 0.3 + circularity * 0.3)
        
        return quality_score
    
    def _calculate_boundary_quality(self, masks: torch.Tensor) -> float:
        """경계 품질 계산"""
        if masks.numel() == 0:
            return 0.0
        
        # 경계의 선명도 계산
        boundary_scores = []
        
        if masks.dim() == 4:  # (B, C, H, W)
            batch_size, channels = masks.size(0), masks.size(1)
            for b in range(batch_size):
                for c in range(channels):
                    mask = masks[b, c]
                    if mask.numel() > 0:
                        boundary_score = self._evaluate_single_mask_boundary(mask)
                        boundary_scores.append(boundary_score)
        else:  # (B, H, W)
            batch_size = masks.size(0)
            for b in range(batch_size):
                mask = masks[b]
                if mask.numel() > 0:
                    boundary_score = self._evaluate_single_mask_boundary(mask)
                    boundary_scores.append(boundary_score)
        
        return float(np.mean(boundary_scores)) if boundary_scores else 0.0
    
    def _evaluate_single_mask_boundary(self, mask: torch.Tensor) -> float:
        """단일 마스크 경계 품질 평가"""
        if mask.dim() != 2:
            return 0.0
        
        # 마스크를 numpy로 변환
        mask_np = mask.detach().cpu().numpy()
        
        # 이진화
        binary_mask = (mask_np > 0.5).astype(np.uint8)
        
        # Canny 엣지 검출
        edges = cv2.Canny(binary_mask * 255, 50, 150)
        
        # 엣지 밀도 계산
        edge_density = np.sum(edges) / (edges.size * 255)
        
        # 낮은 엣지 밀도에 높은 점수 (깔끔한 경계)
        boundary_score = 1.0 - min(edge_density * 5, 1.0)
        
        return boundary_score
    
    def get_postprocessing_info(self) -> Dict[str, Any]:
        """후처리 시스템 정보 반환"""
        return {
            "confidence_threshold": self.config.confidence_threshold,
            "noise_reduction_strength": self.config.noise_reduction_strength,
            "boundary_refinement": self.config.boundary_refinement,
            "hole_filling": self.config.hole_filling,
            "morphological_operations": self.config.morphological_operations,
            "enable_quality_enhancement": self.config.enable_quality_enhancement,
            "device": str(self.device)
        }

# 후처리 시스템 인스턴스 생성 함수
def create_cloth_segmentation_postprocessor(config: PostprocessingConfig = None) -> ClothSegmentationPostprocessor:
    """Cloth Segmentation 후처리 시스템 생성"""
    return ClothSegmentationPostprocessor(config)

# 기본 설정으로 후처리 시스템 생성
def create_default_postprocessor() -> ClothSegmentationPostprocessor:
    """기본 설정으로 후처리 시스템 생성"""
    config = PostprocessingConfig(
        confidence_threshold=0.5,
        noise_reduction_strength=0.3,
        boundary_refinement=True,
        hole_filling=True,
        morphological_operations=True,
        use_mps=True,
        enable_quality_enhancement=True
    )
    return ClothSegmentationPostprocessor(config)

if __name__ == "__main__":
    # 테스트 코드
    logging.basicConfig(level=logging.INFO)
    
    # 후처리 시스템 생성
    postprocessor = create_default_postprocessor()
    
    # 테스트 데이터 생성
    batch_size, channels, height, width = 2, 1, 256, 256
    test_masks = torch.randn(batch_size, channels, height, width)
    test_confidences = torch.rand(batch_size, channels)
    
    # 후처리 수행
    result = postprocessor(test_masks, test_confidences, (height, width))
    print(f"후처리 결과 마스크 형태: {result['masks'].shape}")
    print(f"품질 점수: {result['quality_score']:.3f}")
    print(f"후처리 정보: {postprocessor.get_postprocessing_info()}")
