#!/usr/bin/env python3
"""
🔥 MyCloset AI - Geometric Matching Utils
=========================================

🎯 기하학적 매칭 유틸리티
✅ 이미지 처리 및 변환
✅ 메트릭 계산 및 평가
✅ 시각화 및 결과 분석
✅ M3 Max 최적화
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
import matplotlib.pyplot as plt
import cv2
from PIL import Image, ImageDraw, ImageFont
import math

logger = logging.getLogger(__name__)

@dataclass
class VisualizationConfig:
    """시각화 설정"""
    save_path: Optional[str] = None
    dpi: int = 300
    figsize: Tuple[int, int] = (12, 8)
    colormap: str = 'viridis'
    enable_grid: bool = True
    enable_legend: bool = True

class GeometricMatchingUtils:
    """기하학적 매칭 유틸리티"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.info("🎯 Geometric Matching 유틸리티 초기화")
        
        # 기본 설정
        self.default_colormap = 'viridis'
        self.default_figsize = (12, 8)
        
        self.logger.info("✅ Geometric Matching 유틸리티 초기화 완료")
    
    def tensor_to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        """텐서를 numpy 배열로 변환"""
        if tensor.requires_grad:
            tensor = tensor.detach()
        
        if tensor.is_cuda:
            tensor = tensor.cpu()
        
        return tensor.numpy()
    
    def numpy_to_tensor(self, array: np.ndarray, device: str = 'cpu') -> torch.Tensor:
        """numpy 배열을 텐서로 변환"""
        tensor = torch.from_numpy(array).float()
        if device != 'cpu':
            tensor = tensor.to(device)
        return tensor
    
    def normalize_image(self, image: torch.Tensor, min_val: float = 0.0, max_val: float = 1.0) -> torch.Tensor:
        """이미지 정규화"""
        if image.max() == image.min():
            return torch.zeros_like(image)
        
        normalized = (image - image.min()) / (image.max() - image.min())
        normalized = normalized * (max_val - min_val) + min_val
        
        return torch.clamp(normalized, min_val, max_val)
    
    def compute_ssim(self, img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11) -> float:
        """SSIM (Structural Similarity Index) 계산"""
        try:
            # 간단한 SSIM 구현
            mu1 = F.avg_pool2d(img1, window_size, stride=1, padding=window_size//2)
            mu2 = F.avg_pool2d(img2, window_size, stride=1, padding=window_size//2)
            
            mu1_sq = mu1.pow(2)
            mu2_sq = mu2.pow(2)
            mu1_mu2 = mu1 * mu2
            
            sigma1_sq = F.avg_pool2d(img1 * img1, window_size, stride=1, padding=window_size//2) - mu1_sq
            sigma2_sq = F.avg_pool2d(img2 * img2, window_size, stride=1, padding=window_size//2) - mu2_sq
            sigma12 = F.avg_pool2d(img1 * img2, window_size, stride=1, padding=window_size//2) - mu1_mu2
            
            C1 = 0.01 ** 2
            C2 = 0.03 ** 2
            
            ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                       ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
            
            return float(ssim_map.mean().item())
            
        except Exception as e:
            self.logger.warning(f"SSIM 계산 실패: {e}")
            return 0.0
    
    def compute_psnr(self, img1: torch.Tensor, img2: torch.Tensor, max_val: float = 1.0) -> float:
        """PSNR (Peak Signal-to-Noise Ratio) 계산"""
        try:
            mse = F.mse_loss(img1, img2)
            if mse == 0:
                return float('inf')
            
            psnr = 20 * math.log10(max_val / math.sqrt(mse.item()))
            return psnr
            
        except Exception as e:
            self.logger.warning(f"PSNR 계산 실패: {e}")
            return 0.0
    
    def compute_geometric_consistency(self, displacement_field: torch.Tensor) -> float:
        """기하학적 일관성 계산"""
        try:
            # 변위 필드의 기울기 계산
            grad_x = torch.gradient(displacement_field[:, 0, :, :], dim=2)[0]
            grad_y = torch.gradient(displacement_field[:, 1, :, :], dim=1)[0]
            
            # 기울기 일관성 (작을수록 좋음)
            consistency = torch.mean(torch.abs(grad_x - grad_y))
            
            # 0-1 범위로 정규화 (1이 가장 일관성 있음)
            normalized_consistency = 1.0 / (1.0 + consistency.item())
            
            return normalized_consistency
            
        except Exception as e:
            self.logger.warning(f"기하학적 일관성 계산 실패: {e}")
            return 0.0
    
    def compute_matching_accuracy(self, displacement_field: torch.Tensor, 
                                 ground_truth: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """매칭 정확도 계산"""
        metrics = {}
        
        try:
            # 1. 변위 크기 통계
            displacement_magnitude = torch.sqrt(
                displacement_field[:, 0, :, :] ** 2 + displacement_field[:, 1, :, :] ** 2
            )
            
            metrics['mean_displacement'] = float(displacement_magnitude.mean().item())
            metrics['max_displacement'] = float(displacement_magnitude.max().item())
            metrics['std_displacement'] = float(displacement_magnitude.std().item())
            
            # 2. 변위 분포 분석
            metrics['displacement_variance'] = float(displacement_magnitude.var().item())
            metrics['displacement_skewness'] = self._compute_skewness(displacement_magnitude)
            
            # 3. 그라운드 트루스와 비교 (있는 경우)
            if ground_truth is not None:
                error = torch.norm(displacement_field - ground_truth, dim=1)
                metrics['mean_error'] = float(error.mean().item())
                metrics['max_error'] = float(error.max().item())
                metrics['rmse'] = float(torch.sqrt(torch.mean(error ** 2)).item())
            
            # 4. 기하학적 일관성
            metrics['geometric_consistency'] = self.compute_geometric_consistency(displacement_field)
            
        except Exception as e:
            self.logger.warning(f"매칭 정확도 계산 실패: {e}")
            metrics = {
                'mean_displacement': 0.0,
                'max_displacement': 0.0,
                'std_displacement': 0.0,
                'displacement_variance': 0.0,
                'displacement_skewness': 0.0,
                'geometric_consistency': 0.0
            }
        
        return metrics
    
    def _compute_skewness(self, tensor: torch.Tensor) -> float:
        """왜도 계산"""
        try:
            mean = tensor.mean()
            std = tensor.std()
            
            if std == 0:
                return 0.0
            
            skewness = torch.mean(((tensor - mean) / std) ** 3)
            return float(skewness.item())
            
        except Exception:
            return 0.0
    
    def visualize_displacement_field(self, displacement_field: torch.Tensor, 
                                   save_path: Optional[str] = None,
                                   config: VisualizationConfig = None) -> plt.Figure:
        """변위 필드 시각화"""
        if config is None:
            config = VisualizationConfig()
        
        # 텐서를 numpy로 변환
        disp_np = self.tensor_to_numpy(displacement_field)
        
        # 배치 차원이 있으면 첫 번째만 사용
        if disp_np.ndim == 4:
            disp_np = disp_np[0]
        
        # 변위 크기와 방향 계산
        magnitude = np.sqrt(disp_np[0] ** 2 + disp_np[1] ** 2)
        direction = np.arctan2(disp_np[1], disp_np[0])
        
        # 시각화
        fig, axes = plt.subplots(2, 2, figsize=config.figsize, dpi=config.dpi)
        
        # 1. X 방향 변위
        im1 = axes[0, 0].imshow(disp_np[0], cmap='RdBu_r', vmin=-1, vmax=1)
        axes[0, 0].set_title('X Direction Displacement')
        axes[0, 0].set_axis_off()
        plt.colorbar(im1, ax=axes[0, 0])
        
        # 2. Y 방향 변위
        im2 = axes[0, 1].imshow(disp_np[1], cmap='RdBu_r', vmin=-1, vmax=1)
        axes[0, 1].set_title('Y Direction Displacement')
        axes[0, 1].set_axis_off()
        plt.colorbar(im2, ax=axes[0, 1])
        
        # 3. 변위 크기
        im3 = axes[1, 0].imshow(magnitude, cmap=config.colormap)
        axes[1, 0].set_title('Displacement Magnitude')
        axes[1, 0].set_axis_off()
        plt.colorbar(im3, ax=axes[1, 0])
        
        # 4. 변위 방향
        im4 = axes[1, 1].imshow(direction, cmap='hsv', vmin=-np.pi, vmax=np.pi)
        axes[1, 1].set_title('Displacement Direction')
        axes[1, 1].set_axis_off()
        plt.colorbar(im4, ax=axes[1, 1])
        
        plt.tight_layout()
        
        # 저장
        if save_path:
            plt.savefig(save_path, dpi=config.dpi, bbox_inches='tight')
            self.logger.info(f"시각화 결과 저장: {save_path}")
        
        return fig
    
    def visualize_matching_result(self, image1: torch.Tensor, image2: torch.Tensor, 
                                 matching_result: torch.Tensor,
                                 save_path: Optional[str] = None,
                                 config: VisualizationConfig = None) -> plt.Figure:
        """매칭 결과 시각화"""
        if config is None:
            config = VisualizationConfig()
        
        # 텐서를 numpy로 변환
        img1_np = self.tensor_to_numpy(image1)
        img2_np = self.tensor_to_numpy(image2)
        result_np = self.tensor_to_numpy(matching_result)
        
        # 배치 차원이 있으면 첫 번째만 사용
        if img1_np.ndim == 4:
            img1_np = img1_np[0]
        if img2_np.ndim == 4:
            img2_np = img2_np[0]
        if result_np.ndim == 4:
            result_np = result_np[0]
        
        # 채널 차원을 마지막으로 이동
        if img1_np.shape[0] == 3:
            img1_np = np.transpose(img1_np, (1, 2, 0))
            img2_np = np.transpose(img2_np, (1, 2, 0))
            result_np = np.transpose(result_np, (1, 2, 0))
        
        # 시각화
        fig, axes = plt.subplots(1, 3, figsize=config.figsize, dpi=config.dpi)
        
        # 1. 원본 이미지 1
        axes[0].imshow(img1_np)
        axes[0].set_title('Original Image 1')
        axes[0].set_axis_off()
        
        # 2. 원본 이미지 2
        axes[1].imshow(img2_np)
        axes[1].set_title('Original Image 2')
        axes[1].set_axis_off()
        
        # 3. 매칭 결과
        axes[2].imshow(result_np)
        axes[2].set_title('Matching Result')
        axes[2].set_axis_off()
        
        plt.tight_layout()
        
        # 저장
        if save_path:
            plt.savefig(save_path, dpi=config.dpi, bbox_inches='tight')
            self.logger.info(f"매칭 결과 시각화 저장: {save_path}")
        
        return fig
    
    def create_matching_report(self, displacement_field: torch.Tensor,
                              image1: torch.Tensor, image2: torch.Tensor,
                              matching_result: torch.Tensor,
                              save_path: Optional[str] = None) -> Dict[str, Any]:
        """매칭 리포트 생성"""
        try:
            # 메트릭 계산
            matching_metrics = self.compute_matching_accuracy(displacement_field)
            
            # 품질 메트릭
            ssim_score = self.compute_ssim(image1, matching_result)
            psnr_score = self.compute_psnr(image1, matching_result)
            
            # 리포트 구성
            report = {
                "matching_metrics": matching_metrics,
                "quality_metrics": {
                    "ssim": ssim_score,
                    "psnr": psnr_score
                },
                "summary": {
                    "overall_quality": (ssim_score + matching_metrics['geometric_consistency']) / 2,
                    "recommendation": self._generate_recommendation(matching_metrics, ssim_score)
                }
            }
            
            # 리포트 저장
            if save_path:
                self._save_report(report, save_path)
            
            return report
            
        except Exception as e:
            self.logger.error(f"매칭 리포트 생성 실패: {e}")
            return {"error": str(e)}
    
    def _generate_recommendation(self, matching_metrics: Dict[str, float], 
                                ssim_score: float) -> str:
        """권장사항 생성"""
        overall_score = (ssim_score + matching_metrics['geometric_consistency']) / 2
        
        if overall_score >= 0.8:
            return "Excellent matching quality. Results are highly reliable."
        elif overall_score >= 0.6:
            return "Good matching quality. Results are generally reliable."
        elif overall_score >= 0.4:
            return "Moderate matching quality. Results should be used with caution."
        else:
            return "Poor matching quality. Results are not reliable and should be reviewed."
    
    def _save_report(self, report: Dict[str, Any], save_path: str):
        """리포트 저장"""
        try:
            with open(save_path, 'w', encoding='utf-8') as f:
                import json
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"매칭 리포트 저장: {save_path}")
            
        except Exception as e:
            self.logger.error(f"리포트 저장 실패: {e}")
    
    def get_utils_info(self) -> Dict[str, Any]:
        """유틸리티 정보 반환"""
        return {
            "available_functions": [
                "tensor_to_numpy",
                "numpy_to_tensor", 
                "normalize_image",
                "compute_ssim",
                "compute_psnr",
                "compute_geometric_consistency",
                "compute_matching_accuracy",
                "visualize_displacement_field",
                "visualize_matching_result",
                "create_matching_report"
            ],
            "supported_metrics": [
                "SSIM", "PSNR", "Geometric Consistency", "Displacement Statistics"
            ],
            "visualization_features": [
                "Displacement Field Visualization",
                "Matching Result Comparison",
                "Quality Metrics Analysis"
            ]
        }

# 유틸리티 인스턴스 생성
def create_geometric_matching_utils() -> GeometricMatchingUtils:
    """Geometric Matching 유틸리티 생성"""
    return GeometricMatchingUtils()

if __name__ == "__main__":
    # 테스트 코드
    logging.basicConfig(level=logging.INFO)
    
    # 유틸리티 생성
    utils = create_geometric_matching_utils()
    
    # 유틸리티 정보 출력
    utils_info = utils.get_utils_info()
    print("유틸리티 정보:")
    for key, value in utils_info.items():
        print(f"  {key}: {value}")
    print()
    
    # 테스트 데이터 생성
    batch_size, channels, height, width = 2, 3, 64, 64
    test_image1 = torch.randn(batch_size, channels, height, width)
    test_image2 = torch.randn(batch_size, channels, height, width)
    test_displacement = torch.randn(batch_size, 2, height, width) * 0.1
    
    # 메트릭 계산 테스트
    matching_metrics = utils.compute_matching_accuracy(test_displacement)
    print(f"매칭 메트릭: {matching_metrics}")
    
    # 품질 메트릭 테스트
    ssim_score = utils.compute_ssim(test_image1, test_image2)
    psnr_score = utils.compute_psnr(test_image1, test_image2)
    print(f"SSIM: {ssim_score:.4f}")
    print(f"PSNR: {psnr_score:.4f}")
    
    # 기하학적 일관성 테스트
    consistency = utils.compute_geometric_consistency(test_displacement)
    print(f"기하학적 일관성: {consistency:.4f}")
    
    # 매칭 리포트 생성 테스트
    matching_result = test_image1 + test_displacement.mean(dim=1, keepdim=True)
    report = utils.create_matching_report(test_displacement, test_image1, test_image2, matching_result)
    print(f"매칭 리포트 요약: {report['summary']}")
