"""
Virtual Fitting Visualization Utilities
가상 피팅에 필요한 시각화 유틸리티 함수들을 제공합니다.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Dict, List, Tuple, Optional, Any, Union
import cv2
import logging

logger = logging.getLogger(__name__)

class VisualizationUtils:
    """가상 피팅 시각화 유틸리티 클래스"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.VisualizationUtils")
        self.version = "1.0.0"
    
    def create_fitting_comparison(
        self,
        original_image: np.ndarray,
        fitted_image: np.ndarray,
        mask: Optional[np.ndarray] = None,
        save_path: Optional[str] = None
    ) -> str:
        """
        피팅 전후 비교 시각화 생성
        
        Args:
            original_image: 원본 이미지
            fitted_image: 피팅된 이미지
            mask: 피팅 마스크 (선택사항)
            save_path: 저장 경로 (선택사항)
        
        Returns:
            저장된 파일 경로
        """
        try:
            # 시각화 생성
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('Virtual Fitting Comparison', fontsize=16, fontweight='bold')
            
            # 1. 원본 이미지
            axes[0, 0].imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
            axes[0, 0].set_title('Original Image', fontweight='bold')
            axes[0, 0].axis('off')
            
            # 2. 피팅된 이미지
            axes[0, 1].imshow(cv2.cvtColor(fitted_image, cv2.COLOR_BGR2RGB))
            axes[0, 1].set_title('Fitted Image', fontweight='bold')
            axes[0, 1].axis('off')
            
            # 3. 피팅 마스크 (있는 경우)
            if mask is not None:
                axes[1, 0].imshow(mask, cmap='gray')
                axes[1, 0].set_title('Fitting Mask', fontweight='bold')
                axes[1, 0].axis('off')
            else:
                # 마스크가 없는 경우 원본과 피팅된 이미지의 차이
                diff = cv2.absdiff(original_image, fitted_image)
                axes[1, 0].imshow(cv2.cvtColor(diff, cv2.COLOR_BGR2RGB))
                axes[1, 0].set_title('Difference (Original - Fitted)', fontweight='bold')
                axes[1, 0].axis('off')
            
            # 4. 품질 메트릭
            quality_metrics = self._calculate_visual_quality_metrics(original_image, fitted_image)
            metrics_text = f"""
Quality Metrics:
• Overall Quality: {quality_metrics['overall_quality']:.3f}
• Color Consistency: {quality_metrics['color_consistency']:.3f}
• Edge Preservation: {quality_metrics['edge_preservation']:.3f}
• Texture Quality: {quality_metrics['texture_quality']:.3f}
            """
            axes[1, 1].text(0.1, 0.9, metrics_text, transform=axes[1, 1].transAxes, 
                           fontsize=10, verticalalignment='top', fontfamily='monospace',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
            axes[1, 1].set_title('Quality Metrics', fontweight='bold')
            axes[1, 1].axis('off')
            
            plt.tight_layout()
            
            # 저장
            if save_path is None:
                save_path = f"fitting_comparison_{self.version}.png"
            
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"✅ 피팅 비교 시각화 생성 완료: {save_path}")
            return save_path
            
        except Exception as e:
            self.logger.error(f"❌ 피팅 비교 시각화 생성 실패: {e}")
            return ""
    
    def _calculate_visual_quality_metrics(
        self,
        original_image: np.ndarray,
        fitted_image: np.ndarray
    ) -> Dict[str, float]:
        """시각적 품질 메트릭 계산"""
        try:
            # 색상 일관성
            color_consistency = self._calculate_color_consistency(original_image, fitted_image)
            
            # 엣지 보존
            edge_preservation = self._calculate_edge_preservation(original_image, fitted_image)
            
            # 텍스처 품질
            texture_quality = self._calculate_texture_quality(original_image, fitted_image)
            
            # 전체 품질
            overall_quality = (color_consistency + edge_preservation + texture_quality) / 3
            
            return {
                'color_consistency': color_consistency,
                'edge_preservation': edge_preservation,
                'texture_quality': texture_quality,
                'overall_quality': overall_quality
            }
            
        except Exception as e:
            self.logger.error(f"품질 메트릭 계산 실패: {e}")
            return {
                'color_consistency': 0.5,
                'edge_preservation': 0.5,
                'texture_quality': 0.5,
                'overall_quality': 0.5
            }
    
    def _calculate_color_consistency(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """색상 일관성 계산"""
        try:
            # 그레이스케일 변환
            gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            
            # 히스토그램 비교
            hist1 = cv2.calcHist([gray1], [0], None, [256], [0, 256])
            hist2 = cv2.calcHist([gray2], [0], None, [256], [0, 256])
            
            # 히스토그램 상관계수
            correlation = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
            
            # 0~1 범위로 정규화
            normalized_correlation = (correlation + 1) / 2
            
            return float(normalized_correlation)
            
        except Exception as e:
            self.logger.error(f"색상 일관성 계산 실패: {e}")
            return 0.5
    
    def _calculate_edge_preservation(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """엣지 보존 계산"""
        try:
            # 그레이스케일 변환
            gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            
            # 엣지 감지
            edges1 = cv2.Canny(gray1, 50, 150)
            edges2 = cv2.Canny(gray2, 50, 150)
            
            # 엣지 유사도 계산
            edge_similarity = cv2.matchTemplate(edges1, edges2, cv2.TM_CCOEFF_NORMED)
            max_similarity = np.max(edge_similarity)
            
            # 0~1 범위로 정규화
            normalized_similarity = (max_similarity + 1) / 2
            
            return float(normalized_similarity)
            
        except Exception as e:
            self.logger.error(f"엣지 보존 계산 실패: {e}")
            return 0.5
    
    def _calculate_texture_quality(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """텍스처 품질 계산"""
        try:
            # 그레이스케일 변환
            gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            
            # 로컬 이진 패턴 (LBP) 계산
            lbp1 = self._calculate_lbp(gray1)
            lbp2 = self._calculate_lbp(gray2)
            
            # LBP 히스토그램 비교
            hist1 = cv2.calcHist([lbp1], [0], None, [256], [0, 256])
            hist2 = cv2.calcHist([lbp2], [0], None, [256], [0, 256])
            
            # 히스토그램 상관계수
            correlation = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
            
            # 0~1 범위로 정규화
            normalized_correlation = (correlation + 1) / 2
            
            return float(normalized_correlation)
            
        except Exception as e:
            self.logger.error(f"텍스처 품질 계산 실패: {e}")
            return 0.5
    
    def _calculate_lbp(self, gray_image: np.ndarray) -> np.ndarray:
        """로컬 이진 패턴 계산"""
        try:
            h, w = gray_image.shape
            lbp = np.zeros((h-2, w-2), dtype=np.uint8)
            
            for i in range(1, h-1):
                for j in range(1, w-1):
                    center = gray_image[i, j]
                    code = 0
                    
                    # 8-이웃 픽셀 검사
                    neighbors = [
                        gray_image[i-1, j-1], gray_image[i-1, j], gray_image[i-1, j+1],
                        gray_image[i, j+1], gray_image[i+1, j+1], gray_image[i+1, j],
                        gray_image[i+1, j-1], gray_image[i, j-1]
                    ]
                    
                    for k, neighbor in enumerate(neighbors):
                        if neighbor >= center:
                            code |= (1 << k)
                    
                    lbp[i-1, j-1] = code
            
            return lbp
            
        except Exception as e:
            self.logger.error(f"LBP 계산 실패: {e}")
            return np.zeros_like(gray_image)
    
    def create_fitting_pipeline_visualization(
        self,
        pipeline_steps: List[Dict[str, Any]],
        save_path: Optional[str] = None
    ) -> str:
        """
        피팅 파이프라인 시각화 생성
        
        Args:
            pipeline_steps: 파이프라인 단계 정보
            save_path: 저장 경로 (선택사항)
        
        Returns:
            저장된 파일 경로
        """
        try:
            # 시각화 생성
            num_steps = len(pipeline_steps)
            fig, axes = plt.subplots(2, num_steps, figsize=(4*num_steps, 8))
            fig.suptitle('Virtual Fitting Pipeline Visualization', fontsize=16, fontweight='bold')
            
            for i, step in enumerate(pipeline_steps):
                step_name = step.get('name', f'Step {i+1}')
                step_image = step.get('image')
                step_status = step.get('status', 'unknown')
                
                if step_image is not None:
                    # 이미지 표시
                    axes[0, i].imshow(cv2.cvtColor(step_image, cv2.COLOR_BGR2RGB))
                    axes[0, i].set_title(f'{step_name}\n({step_status})', fontweight='bold')
                    axes[0, i].axis('off')
                    
                    # 단계 정보
                    step_info = step.get('info', {})
                    info_text = f"""
Status: {step_status}
Time: {step.get('time', 'N/A')}s
Quality: {step_info.get('quality', 'N/A')}
                    """
                    axes[1, i].text(0.1, 0.9, info_text, transform=axes[1, i].transAxes, 
                                   fontsize=8, verticalalignment='top', fontfamily='monospace',
                                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
                    axes[1, i].set_title(f'{step_name} Info', fontweight='bold')
                    axes[1, i].axis('off')
                else:
                    # 이미지가 없는 경우
                    axes[0, i].text(0.5, 0.5, f'No Image\nAvailable', 
                                   transform=axes[0, i].transAxes, ha='center', va='center',
                                   fontsize=12, fontweight='bold')
                    axes[0, i].set_title(f'{step_name}', fontweight='bold')
                    axes[0, i].axis('off')
                    
                    axes[1, i].text(0.5, 0.5, f'No Info\nAvailable', 
                                   transform=axes[1, i].transAxes, ha='center', va='center',
                                   fontsize=12, fontweight='bold')
                    axes[1, i].set_title(f'{step_name} Info', fontweight='bold')
                    axes[1, i].axis('off')
            
            plt.tight_layout()
            
            # 저장
            if save_path is None:
                save_path = f"fitting_pipeline_visualization_{self.version}.png"
            
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"✅ 피팅 파이프라인 시각화 생성 완료: {save_path}")
            return save_path
            
        except Exception as e:
            self.logger.error(f"❌ 피팅 파이프라인 시각화 생성 실패: {e}")
            return ""
    
    def get_info(self) -> Dict[str, Any]:
        """시각화 유틸리티 정보 반환"""
        return {
            'module_name': 'visualization_utils',
            'version': self.version,
            'class_name': 'VisualizationUtils',
            'methods': [
                'create_fitting_comparison',
                'create_fitting_pipeline_visualization'
            ],
            'description': '가상 피팅에 필요한 시각화 유틸리티'
        }
