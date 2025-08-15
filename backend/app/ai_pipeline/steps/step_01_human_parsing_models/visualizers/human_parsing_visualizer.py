"""
🔥 Human Parsing 시각화 시스템
================================

인체 파싱 결과를 위한 완전한 시각화 기능:
1. 원본 이미지와 결과 비교
2. 파싱 마스크 색상별 시각화
3. 전처리 과정 시각화
4. 결과 품질 분석 시각화
5. 인터랙티브 시각화

Author: MyCloset AI Team
Date: 2025-01-27
Version: 1.0 (완전 구현)
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
import seaborn as sns
from typing import Dict, Any, Optional, Tuple, List, Union
import logging
from PIL import Image
import torch
import os

logger = logging.getLogger(__name__)

class HumanParsingVisualizer:
    """인체 파싱 결과 시각화 시스템"""
    
    def __init__(self, save_dir: str = "./visualization_results"):
        self.save_dir = save_dir
        # 디렉토리 자동 생성
        os.makedirs(self.save_dir, exist_ok=True)
        self.logger = logging.getLogger(f"{__name__}.HumanParsingVisualizer")
        
        # 인체 파싱 클래스별 색상 정의 (COCO-20 기준)
        self.parsing_colors = {
            0: [0, 0, 0],        # Background
            1: [255, 0, 0],      # Hat
            2: [255, 85, 0],     # Hair
            3: [255, 170, 0],    # Glove
            4: [255, 255, 0],    # Sunglasses
            5: [170, 255, 0],    # Upper-clothes
            6: [85, 255, 0],     # Dress
            7: [0, 255, 0],      # Coat
            8: [0, 255, 85],     # Socks
            9: [0, 255, 170],    # Pants
            10: [0, 255, 255],   # Jumpsuits
            11: [0, 170, 255],   # Scarf
            12: [0, 85, 255],    # Skirt
            13: [0, 0, 255],     # Face
            14: [85, 0, 255],    # Left-arm
            15: [170, 0, 255],   # Right-arm
            16: [255, 0, 255],   # Left-leg
            17: [255, 0, 170],   # Right-leg
            18: [255, 0, 85],    # Left-shoe
            19: [255, 0, 0]      # Right-shoe
        }
        
        # 클래스 이름 정의
        self.class_names = {
            0: "Background", 1: "Hat", 2: "Hair", 3: "Glove", 4: "Sunglasses",
            5: "Upper-clothes", 6: "Dress", 7: "Coat", 8: "Socks", 9: "Pants",
            10: "Jumpsuits", 11: "Scarf", 12: "Skirt", 13: "Face", 14: "Left-arm",
            15: "Right-arm", 16: "Left-leg", 17: "Right-leg", 18: "Left-shoe", 19: "Right-shoe"
        }
        
        # 시각화 통계
        self.visualization_stats = {
            'images_visualized': 0,
            'comparisons_created': 0,
            'masks_visualized': 0,
            'quality_analyses': 0
        }
    
    def visualize_preprocessing_pipeline(self, 
                                       original_image: np.ndarray,
                                       preprocessing_result: Dict[str, Any],
                                       save_path: Optional[str] = None) -> str:
        """전처리 파이프라인 시각화"""
        try:
            self.visualization_stats['images_visualized'] += 1
            self.logger.info("🔥 전처리 파이프라인 시각화 시작")
            
            # 이미지 준비
            processed_image = preprocessing_result['processed_image']
            crop_info = preprocessing_result['crop_info']
            
            # 시각화 생성
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Human Parsing Preprocessing Pipeline', fontsize=16, fontweight='bold')
            
            # 1. 원본 이미지
            axes[0, 0].imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
            axes[0, 0].set_title('Original Image', fontweight='bold')
            axes[0, 0].axis('off')
            
            # 인체 감지 박스 표시
            if crop_info['human_detected']:
                bbox = crop_info['bbox']
                rect = patches.Rectangle(
                    (bbox[0], bbox[1]), bbox[2]-bbox[0], bbox[3]-bbox[1],
                    linewidth=3, edgecolor='red', facecolor='none'
                )
                axes[0, 0].add_patch(rect)
                axes[0, 0].text(bbox[0], bbox[1]-10, f"Human: {crop_info['confidence']:.2f}", 
                               color='red', fontweight='bold', fontsize=10)
            
            # 2. 크롭된 이미지
            axes[0, 1].imshow(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB))
            axes[0, 1].set_title(f'Cropped & Resized ({crop_info["cropped_size"][1]}x{crop_info["cropped_size"][0]})', 
                                fontweight='bold')
            axes[0, 1].axis('off')
            
            # 3. 전처리 정보
            info_text = f"""
Preprocessing Info:
• Target Size: {preprocessing_result['target_size']}
• Mode: {preprocessing_result['mode']}
• Human Detected: {crop_info['human_detected']}
• Confidence: {crop_info['confidence']:.3f}
• Original Size: {crop_info['original_size'][1]}x{crop_info['original_size'][0]}
• Cropped Size: {crop_info['cropped_size'][1]}x{crop_info['cropped_size'][0]}
            """
            axes[1, 0].text(0.1, 0.9, info_text, transform=axes[1, 0].transAxes, 
                           fontsize=10, verticalalignment='top', fontfamily='monospace',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
            axes[1, 0].set_title('Processing Information', fontweight='bold')
            axes[1, 0].axis('off')
            
            # 4. 품질 향상 비교
            if preprocessing_result['mode'] == 'advanced':
                # 원본과 향상된 이미지 비교
                axes[1, 1].imshow(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB))
                axes[1, 1].set_title('Enhanced Image (Advanced Mode)', fontweight='bold')
                axes[1, 1].axis('off')
            else:
                axes[1, 1].text(0.5, 0.5, 'Basic Mode\n(No Enhancement)', 
                               transform=axes[1, 1].transAxes, ha='center', va='center',
                               fontsize=12, fontweight='bold')
                axes[1, 1].axis('off')
            
            plt.tight_layout()
            
            # 저장
            if save_path is None:
                save_path = f"{self.save_dir}/preprocessing_pipeline_{self.visualization_stats['images_visualized']:03d}.png"
            
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"✅ 전처리 파이프라인 시각화 완료: {save_path}")
            return save_path
            
        except Exception as e:
            self.logger.error(f"❌ 전처리 파이프라인 시각화 실패: {e}")
            return ""
    
    def visualize_parsing_result(self, 
                                original_image: np.ndarray,
                                parsing_mask: Union[np.ndarray, torch.Tensor],
                                confidence: float = 1.0,
                                save_path: Optional[str] = None) -> str:
        """파싱 결과 시각화"""
        try:
            self.visualization_stats['masks_visualized'] += 1
            self.logger.info("🔥 파싱 결과 시각화 시작")
            
            # 텐서를 NumPy로 변환
            if isinstance(parsing_mask, torch.Tensor):
                parsing_mask = parsing_mask.detach().cpu().numpy()
            
            # 배치 차원 제거
            if len(parsing_mask.shape) == 4:
                parsing_mask = parsing_mask[0]  # [B, C, H, W] -> [C, H, W]
            
            # 클래스별 마스크 생성
            num_classes = parsing_mask.shape[0]
            colored_mask = self._create_colored_parsing_mask(parsing_mask)
            
            # 시각화 생성
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle('Human Parsing Results', fontsize=16, fontweight='bold')
            
            # 1. 원본 이미지
            axes[0, 0].imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
            axes[0, 0].set_title('Original Image', fontweight='bold')
            axes[0, 0].axis('off')
            
            # 2. 파싱 마스크 (색상)
            axes[0, 1].imshow(colored_mask)
            axes[0, 1].set_title(f'Parsing Mask (Confidence: {confidence:.3f})', fontweight='bold')
            axes[0, 1].axis('off')
            
            # 3. 파싱 마스크 (오버레이)
            overlay = self._create_overlay(original_image, colored_mask, alpha=0.7)
            axes[0, 2].imshow(overlay)
            axes[0, 2].set_title('Overlay Result', fontweight='bold')
            axes[0, 2].axis('off')
            
            # 4. 클래스별 분포
            class_distribution = self._calculate_class_distribution(parsing_mask)
            axes[1, 0].bar(range(len(class_distribution)), list(class_distribution.values()))
            axes[1, 0].set_title('Class Distribution', fontweight='bold')
            axes[1, 0].set_xlabel('Class ID')
            axes[1, 0].set_ylabel('Pixel Count')
            axes[1, 0].tick_params(axis='x', rotation=45)
            
            # 5. 클래스별 신뢰도 히트맵
            confidence_map = np.max(parsing_mask, axis=0)
            im = axes[1, 1].imshow(confidence_map, cmap='hot', interpolation='nearest')
            axes[1, 1].set_title('Confidence Heatmap', fontweight='bold')
            axes[1, 1].axis('off')
            plt.colorbar(im, ax=axes[1, 1])
            
            # 6. 주요 클래스별 마스크
            main_classes = [5, 9, 13, 14, 15]  # Upper-clothes, Pants, Face, Left-arm, Right-arm
            main_mask = np.zeros_like(parsing_mask[0])
            for class_id in main_classes:
                if class_id < num_classes:
                    main_mask = np.logical_or(main_mask, parsing_mask[class_id] > 0.5)
            
            axes[1, 2].imshow(main_mask, cmap='gray')
            axes[1, 2].set_title('Main Body Parts', fontweight='bold')
            axes[1, 2].axis('off')
            
            plt.tight_layout()
            
            # 저장
            if save_path is None:
                save_path = f"{self.save_dir}/parsing_result_{self.visualization_stats['masks_visualized']:03d}.png"
            
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"✅ 파싱 결과 시각화 완료: {save_path}")
            return save_path
            
        except Exception as e:
            self.logger.error(f"❌ 파싱 결과 시각화 실패: {e}")
            return ""
    
    def _create_colored_parsing_mask(self, parsing_mask: np.ndarray) -> np.ndarray:
        """색상이 있는 파싱 마스크 생성"""
        try:
            H, W = parsing_mask.shape[1], parsing_mask.shape[2]
            colored_mask = np.zeros((H, W, 3), dtype=np.uint8)
            
            for class_id in range(parsing_mask.shape[0]):
                if class_id in self.parsing_colors:
                    mask = parsing_mask[class_id] > 0.5
                    color = np.array(self.parsing_colors[class_id])
                    colored_mask[mask] = color
            
            return colored_mask
            
        except Exception as e:
            self.logger.warning(f"색상 마스크 생성 실패: {e}")
            return np.zeros((parsing_mask.shape[1], parsing_mask.shape[2], 3), dtype=np.uint8)
    
    def _create_overlay(self, image: np.ndarray, mask: np.ndarray, alpha: float = 0.7) -> np.ndarray:
        """마스크를 이미지에 오버레이"""
        try:
            # 이미지 크기 맞추기
            if image.shape[:2] != mask.shape[:2]:
                mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
            
            # 오버레이 생성
            overlay = image.copy()
            mask_rgb = cv2.cvtColor(mask, cv2.COLOR_RGB2BGR)
            overlay = cv2.addWeighted(overlay, 1-alpha, mask_rgb, alpha, 0)
            
            return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
            
        except Exception as e:
            self.logger.warning(f"오버레이 생성 실패: {e}")
            return image
    
    def _calculate_class_distribution(self, parsing_mask: np.ndarray) -> Dict[int, int]:
        """클래스별 픽셀 분포 계산"""
        try:
            distribution = {}
            for class_id in range(parsing_mask.shape[0]):
                if class_id in self.class_names:
                    pixel_count = np.sum(parsing_mask[class_id] > 0.5)
                    distribution[class_id] = pixel_count
            
            return distribution
            
        except Exception as e:
            self.logger.warning(f"클래스 분포 계산 실패: {e}")
            return {}
    
    def create_comparison_visualization(self, 
                                      original_image: np.ndarray,
                                      preprocessing_result: Dict[str, Any],
                                      parsing_result: Dict[str, Any],
                                      save_path: Optional[str] = None) -> str:
        """전체 파이프라인 비교 시각화"""
        try:
            self.visualization_stats['comparisons_created'] += 1
            self.logger.info("🔥 전체 파이프라인 비교 시각화 시작")
            
            # 시각화 생성
            fig, axes = plt.subplots(2, 4, figsize=(20, 10))
            fig.suptitle('Complete Human Parsing Pipeline', fontsize=18, fontweight='bold')
            
            # 1. 원본 이미지
            axes[0, 0].imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
            axes[0, 0].set_title('Original', fontweight='bold')
            axes[0, 0].axis('off')
            
            # 2. 전처리된 이미지
            processed_img = preprocessing_result['processed_image']
            axes[0, 1].imshow(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB))
            axes[0, 1].set_title('Preprocessed', fontweight='bold')
            axes[0, 1].axis('off')
            
            # 3. 파싱 마스크
            if 'parsing_mask' in parsing_result:
                parsing_mask = parsing_result['parsing_mask']
                colored_mask = self._create_colored_parsing_mask(parsing_mask)
                axes[0, 2].imshow(colored_mask)
                axes[0, 2].set_title('Parsing Mask', fontweight='bold')
                axes[0, 2].axis('off')
            
            # 4. 최종 결과 (오버레이)
            if 'parsing_mask' in parsing_result:
                overlay = self._create_overlay(processed_img, colored_mask, alpha=0.6)
                axes[0, 3].imshow(overlay)
                axes[0, 3].set_title('Final Result', fontweight='bold')
                axes[0, 3].axis('off')
            
            # 5. 처리 정보
            info_text = f"""
Processing Info:
• Target Size: {preprocessing_result['target_size']}
• Mode: {preprocessing_result['mode']}
• Human Detected: {preprocessing_result['crop_info']['human_detected']}
• Confidence: {preprocessing_result['crop_info']['confidence']:.3f}
            """
            axes[1, 0].text(0.1, 0.9, info_text, transform=axes[1, 0].transAxes, 
                           fontsize=9, verticalalignment='top', fontfamily='monospace',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
            axes[1, 0].set_title('Processing Info', fontweight='bold')
            axes[1, 0].axis('off')
            
            # 6. 품질 메트릭
            if 'quality_metrics' in parsing_result:
                quality_text = f"""
Quality Metrics:
• Overall Score: {parsing_result['quality_metrics'].get('overall_score', 'N/A'):.3f}
• Boundary Quality: {parsing_result['quality_metrics'].get('boundary_quality', 'N/A'):.3f}
• Segmentation Quality: {parsing_result['quality_metrics'].get('segmentation_quality', 'N/A'):.3f}
                """
            else:
                quality_text = "Quality metrics not available"
            
            axes[1, 1].text(0.1, 0.9, quality_text, transform=axes[1, 1].transAxes, 
                           fontsize=9, verticalalignment='top', fontfamily='monospace',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
            axes[1, 1].set_title('Quality Metrics', fontweight='bold')
            axes[1, 1].axis('off')
            
            # 7. 클래스 분포
            if 'parsing_mask' in parsing_result:
                class_dist = self._calculate_class_distribution(parsing_result['parsing_mask'])
                if class_dist:
                    axes[1, 2].bar(range(len(class_dist)), list(class_dist.values()))
                    axes[1, 2].set_title('Class Distribution', fontweight='bold')
                    axes[1, 2].set_xlabel('Class ID')
                    axes[1, 2].set_ylabel('Pixel Count')
                    axes[1, 2].tick_params(axis='x', rotation=45)
            
            # 8. 처리 통계
            stats_text = f"""
Processing Stats:
• Images Processed: {self.visualization_stats['images_visualized']}
• Masks Visualized: {self.visualization_stats['masks_visualized']}
• Comparisons Created: {self.visualization_stats['comparisons_created']}
• Quality Analyses: {self.visualization_stats['quality_analyses']}
            """
            axes[1, 3].text(0.1, 0.9, stats_text, transform=axes[1, 3].transAxes, 
                           fontsize=9, verticalalignment='top', fontfamily='monospace',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))
            axes[1, 3].set_title('Processing Stats', fontweight='bold')
            axes[1, 3].axis('off')
            
            plt.tight_layout()
            
            # 저장
            if save_path is None:
                save_path = f"{self.save_dir}/complete_pipeline_{self.visualization_stats['comparisons_created']:03d}.png"
            
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"✅ 전체 파이프라인 비교 시각화 완료: {save_path}")
            return save_path
            
        except Exception as e:
            self.logger.error(f"❌ 전체 파이프라인 비교 시각화 실패: {e}")
            return ""
    
    def get_visualization_stats(self) -> Dict[str, Any]:
        """시각화 통계 반환"""
        return self.visualization_stats.copy()
    
    def reset_stats(self):
        """통계 초기화"""
        self.visualization_stats = {
            'images_visualized': 0,
            'comparisons_created': 0,
            'masks_visualized': 0,
            'quality_analyses': 0
        }
