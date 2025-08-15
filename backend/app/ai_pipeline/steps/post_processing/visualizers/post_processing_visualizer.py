"""
🔥 Post Processing 시각화 시스템
================================

후처리 결과를 위한 완전한 시각화 기능:
1. 전처리 vs 후처리 비교
2. 품질 향상 시각화
3. 결과 분석 차트
4. 품질 메트릭 시각화

Author: MyCloset AI Team
Date: 2025-08-14
Version: 1.0
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
from typing import Dict, Any, Optional, Tuple, List, Union
import logging
from PIL import Image
import os

logger = logging.getLogger(__name__)

class PostProcessingVisualizer:
    """후처리 결과 시각화 시스템"""
    
    def __init__(self, save_dir: str = "./post_processing_visualization"):
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.logger = logging.getLogger(f"{__name__}.PostProcessingVisualizer")
        
        # 시각화 통계
        self.visualization_stats = {
            'images_visualized': 0,
            'comparisons_created': 0,
            'quality_analyses': 0
        }
    
    def visualize_preprocessing_vs_postprocessing(self, 
                                               original_image: np.ndarray,
                                               preprocessed_image: np.ndarray,
                                               postprocessed_image: np.ndarray,
                                               save_path: Optional[str] = None) -> str:
        """전처리 vs 후처리 비교 시각화"""
        try:
            self.visualization_stats['comparisons_created'] += 1
            self.logger.info("🔥 전처리 vs 후처리 비교 시각화 시작")
            
            # 시각화 생성
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            fig.suptitle('Preprocessing vs Postprocessing Comparison', fontsize=16, fontweight='bold')
            
            # 1. 원본 이미지
            axes[0].imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
            axes[0].set_title('Original Image', fontweight='bold')
            axes[0].axis('off')
            
            # 2. 전처리 이미지
            axes[1].imshow(cv2.cvtColor(preprocessed_image, cv2.COLOR_BGR2RGB))
            axes[1].set_title('Preprocessed Image', fontweight='bold')
            axes[1].axis('off')
            
            # 3. 후처리 이미지
            axes[2].imshow(cv2.cvtColor(postprocessed_image, cv2.COLOR_BGR2RGB))
            axes[2].set_title('Postprocessed Image', fontweight='bold')
            axes[2].axis('off')
            
            # 저장
            if save_path is None:
                save_path = os.path.join(self.save_dir, f"preprocessing_vs_postprocessing_{self.visualization_stats['comparisons_created']}.png")
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"✅ 전처리 vs 후처리 비교 시각화 완료: {save_path}")
            return save_path
            
        except Exception as e:
            self.logger.error(f"❌ 전처리 vs 후처리 비교 시각화 실패: {e}")
            raise
    
    def visualize_quality_improvement(self, 
                                    quality_metrics: Dict[str, float],
                                    save_path: Optional[str] = None) -> str:
        """품질 향상 시각화"""
        try:
            self.visualization_stats['quality_analyses'] += 1
            self.logger.info("🔥 품질 향상 시각화 시작")
            
            # 메트릭 추출
            metrics = list(quality_metrics.keys())
            values = list(quality_metrics.values())
            
            # 막대 그래프 생성
            plt.figure(figsize=(12, 8))
            bars = plt.bar(metrics, values, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
            
            # 값 표시
            for bar, value in zip(bars, values):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                        f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
            
            plt.title('Quality Metrics Comparison', fontsize=16, fontweight='bold')
            plt.xlabel('Quality Metrics', fontsize=12)
            plt.ylabel('Score', fontsize=12)
            plt.ylim(0, 1.1)
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
            
            # 저장
            if save_path is None:
                save_path = os.path.join(self.save_dir, f"quality_improvement_{self.visualization_stats['quality_analyses']}.png")
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"✅ 품질 향상 시각화 완료: {save_path}")
            return save_path
            
        except Exception as e:
            self.logger.error(f"❌ 품질 향상 시각화 실패: {e}")
            raise
    
    def get_visualization_stats(self) -> Dict[str, Any]:
        """시각화 통계 반환"""
        return self.visualization_stats.copy()
