"""
🔥 Pose Estimation 시각화 시스템
=================================

포즈 추정 결과를 위한 완전한 시각화 기능:
1. 원본 이미지와 포즈 결과 비교
2. 키포인트 및 스켈레톤 시각화
3. 전처리 과정 시각화
4. 포즈 품질 분석 시각화
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

class PoseEstimationVisualizer:
    """포즈 추정 결과 시각화 시스템"""
    
    def __init__(self, save_dir: str = "./visualization_results"):
        self.save_dir = save_dir
        # 디렉토리 자동 생성
        os.makedirs(self.save_dir, exist_ok=True)
        self.logger = logging.getLogger(f"{__name__}.PoseEstimationVisualizer")
        
        # COCO-17 키포인트 정의
        self.coco_keypoints = {
            0: "nose", 1: "left_eye", 2: "right_eye", 3: "left_ear", 4: "right_ear",
            5: "left_shoulder", 6: "right_shoulder", 7: "left_elbow", 8: "right_elbow",
            9: "left_wrist", 10: "right_wrist", 11: "left_hip", 12: "right_hip",
            13: "left_knee", 14: "right_knee", 15: "left_ankle", 16: "right_ankle"
        }
        
        # 스켈레톤 연결 (COCO-17)
        self.skeleton_connections = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # 머리
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # 상체
            (5, 11), (6, 12), (11, 12),  # 몸통
            (11, 13), (13, 15), (12, 14), (14, 16)  # 하체
        ]
        
        # 키포인트 색상
        self.keypoint_colors = {
            'head': [255, 0, 0],      # 빨강
            'torso': [0, 255, 0],     # 초록
            'arms': [0, 0, 255],      # 파랑
            'legs': [255, 255, 0]     # 노랑
        }
        
        # 시각화 통계
        self.visualization_stats = {
            'images_visualized': 0,
            'pose_visualized': 0,
            'comparisons_created': 0,
            'quality_analyses': 0
        }
    
    def visualize_preprocessing_pipeline(self, 
                                       original_image: np.ndarray,
                                       preprocessing_result: Dict[str, Any],
                                       save_path: Optional[str] = None) -> str:
        """전처리 파이프라인 시각화"""
        try:
            self.visualization_stats['images_visualized'] += 1
            self.logger.info("🔥 포즈 추정 전처리 파이프라인 시각화 시작")
            
            # 이미지 준비
            processed_image = preprocessing_result['processed_image']
            alignment_info = preprocessing_result['alignment_info']
            
            # 시각화 생성
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle('Pose Estimation Preprocessing Pipeline', fontsize=16, fontweight='bold')
            
            # 1. 원본 이미지
            axes[0, 0].imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
            axes[0, 0].set_title('Original Image', fontweight='bold')
            axes[0, 0].axis('off')
            
            # 인체 감지 박스 표시
            if alignment_info['human_detected']:
                bbox = alignment_info['bbox']
                rect = patches.Rectangle(
                    (bbox[0], bbox[1]), bbox[2]-bbox[0], bbox[3]-bbox[1],
                    linewidth=3, edgecolor='red', facecolor='none'
                )
                axes[0, 0].add_patch(rect)
                axes[0, 0].text(bbox[0], bbox[1]-10, f"Human: {alignment_info['confidence']:.2f}", 
                               color='red', fontweight='bold', fontsize=10)
            
            # 2. 크롭된 이미지
            axes[0, 1].imshow(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB))
            axes[0, 1].set_title(f'Pose-Aligned ({alignment_info["aligned_size"][1]}x{alignment_info["aligned_size"][0]})', 
                                fontweight='bold')
            axes[0, 1].axis('off')
            
            # 3. 포즈 정렬 정보
            info_text = f"""
Pose Alignment Info:
• Target Size: {preprocessing_result['target_size']}
• Mode: {preprocessing_result['mode']}
• Human Detected: {alignment_info['human_detected']}
• Confidence: {alignment_info['confidence']:.3f}
• Pose Centered: {alignment_info['pose_centered']}
• Original Size: {alignment_info['original_size'][1]}x{alignment_info['original_size'][0]}
• Aligned Size: {alignment_info['aligned_size'][1]}x{alignment_info['aligned_size'][0]}
            """
            axes[0, 2].text(0.1, 0.9, info_text, transform=axes[0, 2].transAxes, 
                           fontsize=9, verticalalignment='top', fontfamily='monospace',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
            axes[0, 2].set_title('Alignment Information', fontweight='bold')
            axes[0, 2].axis('off')
            
            # 4. 포즈 파라미터
            pose_params = preprocessing_result['pose_params']
            params_text = f"""
Pose Parameters:
• Joint Enhancement: {pose_params['joint_enhancement']}
• Background Removal: {pose_params['background_removal']}
• Pose Normalization: {pose_params['pose_normalization']}
• Lighting Correction: {pose_params['lighting_correction']}
            """
            axes[1, 0].text(0.1, 0.9, params_text, transform=axes[1, 0].transAxes, 
                           fontsize=9, verticalalignment='top', fontfamily='monospace',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
            axes[1, 0].set_title('Pose Parameters', fontweight='bold')
            axes[1, 0].axis('off')
            
            # 5. 품질 향상 비교
            if preprocessing_result['mode'] == 'advanced':
                # 원본과 향상된 이미지 비교
                axes[1, 1].imshow(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB))
                axes[1, 1].set_title('Enhanced for Pose Estimation', fontweight='bold')
                axes[1, 1].axis('off')
            else:
                axes[1, 1].text(0.5, 0.5, 'Basic Mode\n(No Enhancement)', 
                               transform=axes[1, 1].transAxes, ha='center', va='center',
                               fontsize=12, fontweight='bold')
                axes[1, 1].axis('off')
            
            # 6. 처리 통계
            stats_text = f"""
Processing Stats:
• Images Processed: {self.visualization_stats['images_visualized']}
• Pose Visualized: {self.visualization_stats['pose_visualized']}
• Comparisons Created: {self.visualization_stats['comparisons_created']}
• Quality Analyses: {self.visualization_stats['quality_analyses']}
            """
            axes[1, 2].text(0.1, 0.9, stats_text, transform=axes[1, 2].transAxes, 
                           fontsize=9, verticalalignment='top', fontfamily='monospace',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))
            axes[1, 2].set_title('Processing Stats', fontweight='bold')
            axes[1, 2].axis('off')
            
            plt.tight_layout()
            
            # 저장
            if save_path is None:
                save_path = f"{self.save_dir}/pose_preprocessing_pipeline_{self.visualization_stats['images_visualized']:03d}.png"
            
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"✅ 포즈 추정 전처리 파이프라인 시각화 완료: {save_path}")
            return save_path
            
        except Exception as e:
            self.logger.error(f"❌ 포즈 추정 전처리 파이프라인 시각화 실패: {e}")
            return ""
    
    def visualize_pose_result(self, 
                             original_image: np.ndarray,
                             pose_keypoints: Union[np.ndarray, torch.Tensor],
                             confidence_scores: Optional[np.ndarray] = None,
                             save_path: Optional[str] = None) -> str:
        """포즈 추정 결과 시각화"""
        try:
            self.visualization_stats['pose_visualized'] += 1
            self.logger.info("🔥 포즈 추정 결과 시각화 시작")
            
            # 텐서를 NumPy로 변환
            if isinstance(pose_keypoints, torch.Tensor):
                pose_keypoints = pose_keypoints.detach().cpu().numpy()
            
            # 배치 차원 제거
            if len(pose_keypoints.shape) == 3:
                pose_keypoints = pose_keypoints[0]  # [B, N, 2] -> [N, 2]
            
            # 신뢰도 점수 준비
            if confidence_scores is None:
                confidence_scores = np.ones(pose_keypoints.shape[0])
            elif isinstance(confidence_scores, torch.Tensor):
                confidence_scores = confidence_scores.detach().cpu().numpy()
            
            # 시각화 생성
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle('Pose Estimation Results', fontsize=16, fontweight='bold')
            
            # 1. 원본 이미지 + 키포인트
            axes[0, 0].imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
            self._draw_keypoints(axes[0, 0], pose_keypoints, confidence_scores)
            axes[0, 0].set_title('Original + Keypoints', fontweight='bold')
            axes[0, 0].axis('off')
            
            # 2. 스켈레톤 시각화
            axes[0, 1].imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
            self._draw_skeleton(axes[0, 1], pose_keypoints, confidence_scores)
            axes[0, 1].set_title('Skeleton Visualization', fontweight='bold')
            axes[0, 1].axis('off')
            
            # 3. 신뢰도 히트맵
            confidence_map = self._create_confidence_heatmap(pose_keypoints, confidence_scores, original_image.shape[:2])
            axes[0, 2].imshow(confidence_map, cmap='hot', alpha=0.7)
            axes[0, 2].set_title('Confidence Heatmap', fontweight='bold')
            axes[0, 2].axis('off')
            
            # 4. 키포인트별 신뢰도
            valid_keypoints = confidence_scores > 0.1
            if np.any(valid_keypoints):
                valid_indices = np.where(valid_keypoints)[0]
                valid_confidences = confidence_scores[valid_keypoints]
                valid_names = [self.coco_keypoints.get(i, f"KP_{i}") for i in valid_indices]
                
                axes[1, 0].bar(range(len(valid_confidences)), valid_confidences)
                axes[1, 0].set_title('Keypoint Confidence Scores', fontweight='bold')
                axes[1, 0].set_xlabel('Keypoint')
                axes[1, 0].set_ylabel('Confidence')
                axes[1, 0].set_xticks(range(len(valid_names)))
                axes[1, 0].set_xticklabels(valid_names, rotation=45, ha='right')
                axes[1, 0].set_ylim(0, 1)
            else:
                axes[1, 0].text(0.5, 0.5, 'No valid keypoints detected', 
                               transform=axes[1, 0].transAxes, ha='center', va='center',
                               fontsize=12, fontweight='bold')
                axes[1, 0].set_title('Keypoint Confidence Scores', fontweight='bold')
                axes[1, 0].axis('off')
            
            # 5. 포즈 품질 분석
            quality_metrics = self._calculate_pose_quality(pose_keypoints, confidence_scores)
            quality_text = f"""
Pose Quality Metrics:
• Overall Score: {quality_metrics['overall_score']:.3f}
• Symmetry Score: {quality_metrics['symmetry_score']:.3f}
• Confidence Score: {quality_metrics['confidence_score']:.3f}
• Coverage Score: {quality_metrics['coverage_score']:.3f}
• Valid Keypoints: {quality_metrics['valid_keypoints']}/{len(pose_keypoints)}
            """
            axes[1, 1].text(0.1, 0.9, quality_text, transform=axes[1, 1].transAxes, 
                           fontsize=9, verticalalignment='top', fontfamily='monospace',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
            axes[1, 1].set_title('Pose Quality Analysis', fontweight='bold')
            axes[1, 1].axis('off')
            
            # 6. 포즈 분류
            pose_category = self._classify_pose(pose_keypoints, confidence_scores)
            category_text = f"""
Pose Classification:
• Category: {pose_category['category']}
• Confidence: {pose_category['confidence']:.3f}
• Description: {pose_category['description']}
• Key Features: {', '.join(pose_category['key_features'])}
            """
            axes[1, 2].text(0.1, 0.9, category_text, transform=axes[1, 2].transAxes, 
                           fontsize=9, verticalalignment='top', fontfamily='monospace',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
            axes[1, 2].set_title('Pose Classification', fontweight='bold')
            axes[1, 2].axis('off')
            
            plt.tight_layout()
            
            # 저장
            if save_path is None:
                save_path = f"{self.save_dir}/pose_result_{self.visualization_stats['pose_visualized']:03d}.png"
            
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"✅ 포즈 추정 결과 시각화 완료: {save_path}")
            return save_path
            
        except Exception as e:
            self.logger.error(f"❌ 포즈 추정 결과 시각화 실패: {e}")
            return ""
    
    def _draw_keypoints(self, ax, keypoints: np.ndarray, confidence_scores: np.ndarray):
        """키포인트 그리기"""
        try:
            for i, (kp, conf) in enumerate(zip(keypoints, confidence_scores)):
                if conf > 0.1:  # 신뢰도 임계값
                    x, y = kp[0], kp[1]
                    
                    # 키포인트 타입에 따른 색상 선택
                    if i in [0, 1, 2, 3, 4]:  # 머리
                        color = self.keypoint_colors['head']
                    elif i in [5, 6, 11, 12]:  # 몸통
                        color = self.keypoint_colors['torso']
                    elif i in [7, 8, 9, 10]:  # 팔
                        color = self.keypoint_colors['arms']
                    else:  # 다리
                        color = self.keypoint_colors['legs']
                    
                    # 키포인트 그리기
                    ax.scatter(x, y, c=[np.array(color)/255], s=100, alpha=0.8, edgecolors='white', linewidth=2)
                    
                    # 키포인트 번호 표시
                    ax.text(x+5, y+5, str(i), fontsize=8, fontweight='bold', 
                           color='white', bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.7))
                    
        except Exception as e:
            self.logger.warning(f"키포인트 그리기 실패: {e}")
    
    def _draw_skeleton(self, ax, keypoints: np.ndarray, confidence_scores: np.ndarray):
        """스켈레톤 그리기"""
        try:
            for connection in self.skeleton_connections:
                start_idx, end_idx = connection
                
                # 두 키포인트 모두 유효한 경우에만 연결
                if (confidence_scores[start_idx] > 0.1 and 
                    confidence_scores[end_idx] > 0.1):
                    
                    start_kp = keypoints[start_idx]
                    end_kp = keypoints[end_idx]
                    
                    # 연결선 그리기
                    ax.plot([start_kp[0], end_kp[0]], [start_kp[1], end_kp[1]], 
                           color='red', linewidth=2, alpha=0.8)
                    
        except Exception as e:
            self.logger.warning(f"스켈레톤 그리기 실패: {e}")
    
    def _create_confidence_heatmap(self, keypoints: np.ndarray, confidence_scores: np.ndarray, 
                                  image_shape: Tuple[int, int]) -> np.ndarray:
        """신뢰도 히트맵 생성"""
        try:
            heatmap = np.zeros(image_shape[:2], dtype=np.float32)
            
            for kp, conf in zip(keypoints, confidence_scores):
                if conf > 0.1:
                    x, y = int(kp[0]), int(kp[1])
                    if 0 <= x < image_shape[1] and 0 <= y < image_shape[0]:
                        # 가우시안 커널로 신뢰도 분산
                        y_coords, x_coords = np.ogrid[:image_shape[0], :image_shape[1]]
                        dist_sq = (x_coords - x)**2 + (y_coords - y)**2
                        gaussian = np.exp(-dist_sq / (2 * 20**2))  # 표준편차 20
                        heatmap += gaussian * conf
            
            return heatmap
            
        except Exception as e:
            self.logger.warning(f"신뢰도 히트맵 생성 실패: {e}")
            return np.zeros(image_shape[:2], dtype=np.float32)
    
    def _calculate_pose_quality(self, keypoints: np.ndarray, confidence_scores: np.ndarray) -> Dict[str, float]:
        """포즈 품질 계산"""
        try:
            # 유효한 키포인트 수
            valid_mask = confidence_scores > 0.1
            valid_count = np.sum(valid_mask)
            
            # 전체 신뢰도 점수
            confidence_score = np.mean(confidence_scores[valid_mask]) if valid_count > 0 else 0.0
            
            # 대칭성 점수 (좌우 대칭 키포인트 비교)
            symmetry_pairs = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)]
            symmetry_score = 0.0
            valid_pairs = 0
            
            for left, right in symmetry_pairs:
                if (left < len(keypoints) and right < len(keypoints) and
                    confidence_scores[left] > 0.1 and confidence_scores[right] > 0.1):
                    
                    # Y축 기준 대칭성 계산
                    left_y = keypoints[left][1]
                    right_y = keypoints[right][1]
                    y_diff = abs(left_y - right_y)
                    
                    # 대칭성 점수 (차이가 작을수록 높은 점수)
                    pair_score = max(0, 1 - y_diff / 50)  # 50픽셀 차이를 0점으로
                    symmetry_score += pair_score
                    valid_pairs += 1
            
            symmetry_score = symmetry_score / valid_pairs if valid_pairs > 0 else 0.0
            
            # 커버리지 점수 (키포인트가 이미지 전체에 분산되어 있는지)
            if valid_count > 0:
                valid_keypoints = keypoints[valid_mask]
                x_range = np.max(valid_keypoints[:, 0]) - np.min(valid_keypoints[:, 0])
                y_range = np.max(valid_keypoints[:, 1]) - np.min(valid_keypoints[:, 1])
                
                # 이미지 크기 대비 키포인트 분산도
                coverage_score = min(1.0, (x_range + y_range) / 200)  # 200픽셀을 최대값으로
            else:
                coverage_score = 0.0
            
            # 전체 점수 (가중 평균)
            overall_score = (confidence_score * 0.4 + 
                           symmetry_score * 0.3 + 
                           coverage_score * 0.3)
            
            return {
                'overall_score': overall_score,
                'symmetry_score': symmetry_score,
                'confidence_score': confidence_score,
                'coverage_score': coverage_score,
                'valid_keypoints': valid_count
            }
            
        except Exception as e:
            self.logger.warning(f"포즈 품질 계산 실패: {e}")
            return {
                'overall_score': 0.0,
                'symmetry_score': 0.0,
                'confidence_score': 0.0,
                'coverage_score': 0.0,
                'valid_keypoints': 0
            }
    
    def _classify_pose(self, keypoints: np.ndarray, confidence_scores: np.ndarray) -> Dict[str, Any]:
        """포즈 분류"""
        try:
            valid_mask = confidence_scores > 0.1
            valid_count = np.sum(valid_mask)
            
            if valid_count < 5:
                return {
                    'category': 'Insufficient',
                    'confidence': 0.0,
                    'description': 'Too few keypoints detected',
                    'key_features': ['Low detection count']
                }
            
            # 주요 키포인트 추출
            if valid_mask[5] and valid_mask[6]:  # 어깨
                shoulder_width = abs(keypoints[5][0] - keypoints[6][0])
            else:
                shoulder_width = 0
            
            if valid_mask[11] and valid_mask[12]:  # 엉덩이
                hip_width = abs(keypoints[11][0] - keypoints[12][0])
            else:
                hip_width = 0
            
            if valid_mask[13] and valid_mask[14]:  # 무릎
                knee_width = abs(keypoints[13][0] - keypoints[14][0])
            else:
                knee_width = 0
            
            # 포즈 분류
            if shoulder_width > 0 and hip_width > 0:
                if shoulder_width > hip_width * 1.2:
                    category = 'Standing'
                    description = 'Person standing with arms extended'
                    key_features = ['Extended arms', 'Upright posture']
                elif hip_width > shoulder_width * 1.2:
                    category = 'Sitting'
                    description = 'Person sitting or crouching'
                    key_features = ['Lower body spread', 'Compressed upper body']
                else:
                    category = 'Neutral'
                    description = 'Person in neutral standing position'
                    key_features = ['Balanced proportions', 'Natural stance']
            else:
                category = 'Partial'
                description = 'Partial pose detection'
                key_features = ['Limited keypoints', 'Incomplete detection']
            
            # 분류 신뢰도
            classification_confidence = min(1.0, valid_count / 17)  # 17개 키포인트 기준
            
            return {
                'category': category,
                'confidence': classification_confidence,
                'description': description,
                'key_features': key_features
            }
            
        except Exception as e:
            self.logger.warning(f"포즈 분류 실패: {e}")
            return {
                'category': 'Unknown',
                'confidence': 0.0,
                'description': 'Classification failed',
                'key_features': ['Error in classification']
            }
    
    def create_comparison_visualization(self, 
                                      original_image: np.ndarray,
                                      preprocessing_result: Dict[str, Any],
                                      pose_result: Dict[str, Any],
                                      save_path: Optional[str] = None) -> str:
        """전체 파이프라인 비교 시각화"""
        try:
            self.visualization_stats['comparisons_created'] += 1
            self.logger.info("🔥 전체 포즈 추정 파이프라인 비교 시각화 시작")
            
            # 시각화 생성
            fig, axes = plt.subplots(2, 4, figsize=(20, 10))
            fig.suptitle('Complete Pose Estimation Pipeline', fontsize=18, fontweight='bold')
            
            # 1. 원본 이미지
            axes[0, 0].imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
            axes[0, 0].set_title('Original', fontweight='bold')
            axes[0, 0].axis('off')
            
            # 2. 전처리된 이미지
            processed_img = preprocessing_result['processed_image']
            axes[0, 1].imshow(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB))
            axes[0, 1].set_title('Preprocessed', fontweight='bold')
            axes[0, 1].axis('off')
            
            # 3. 포즈 결과
            if 'keypoints' in pose_result:
                axes[0, 2].imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
                self._draw_keypoints(axes[0, 2], pose_result['keypoints'], pose_result.get('confidence_scores', np.ones(17)))
                axes[0, 2].set_title('Pose Detection', fontweight='bold')
                axes[0, 2].axis('off')
            
            # 4. 최종 결과 (스켈레톤)
            if 'keypoints' in pose_result:
                axes[0, 3].imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
                self._draw_skeleton(axes[0, 3], pose_result['keypoints'], pose_result.get('confidence_scores', np.ones(17)))
                axes[0, 3].set_title('Final Skeleton', fontweight='bold')
                axes[0, 3].axis('off')
            
            # 5. 전처리 정보
            info_text = f"""
Preprocessing Info:
• Target Size: {preprocessing_result['target_size']}
• Mode: {preprocessing_result['mode']}
• Human Detected: {preprocessing_result['alignment_info']['human_detected']}
• Pose Centered: {preprocessing_result['alignment_info']['pose_centered']}
            """
            axes[1, 0].text(0.1, 0.9, info_text, transform=axes[1, 0].transAxes, 
                           fontsize=9, verticalalignment='top', fontfamily='monospace',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
            axes[1, 0].set_title('Preprocessing Info', fontweight='bold')
            axes[1, 0].axis('off')
            
            # 6. 포즈 품질
            if 'keypoints' in pose_result:
                quality_metrics = self._calculate_pose_quality(
                    pose_result['keypoints'], 
                    pose_result.get('confidence_scores', np.ones(17))
                )
                quality_text = f"""
Pose Quality:
• Overall Score: {quality_metrics['overall_score']:.3f}
• Symmetry: {quality_metrics['symmetry_score']:.3f}
• Confidence: {quality_metrics['confidence_score']:.3f}
• Coverage: {quality_metrics['coverage_score']:.3f}
                """
            else:
                quality_text = "Pose data not available"
            
            axes[1, 1].text(0.1, 0.9, quality_text, transform=axes[1, 1].transAxes, 
                           fontsize=9, verticalalignment='top', fontfamily='monospace',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
            axes[1, 1].set_title('Pose Quality', fontweight='bold')
            axes[1, 1].axis('off')
            
            # 7. 포즈 분류
            if 'keypoints' in pose_result:
                pose_category = self._classify_pose(
                    pose_result['keypoints'], 
                    pose_result.get('confidence_scores', np.ones(17))
                )
                category_text = f"""
Pose Classification:
• Category: {pose_category['category']}
• Confidence: {pose_category['confidence']:.3f}
• Description: {pose_category['description']}
                """
            else:
                category_text = "Pose data not available"
            
            axes[1, 2].text(0.1, 0.9, category_text, transform=axes[1, 2].transAxes, 
                           fontsize=9, verticalalignment='top', fontfamily='monospace',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))
            axes[1, 2].set_title('Pose Classification', fontweight='bold')
            axes[1, 2].axis('off')
            
            # 8. 처리 통계
            stats_text = f"""
Processing Stats:
• Images Processed: {self.visualization_stats['images_visualized']}
• Pose Visualized: {self.visualization_stats['pose_visualized']}
• Comparisons Created: {self.visualization_stats['comparisons_created']}
• Quality Analyses: {self.visualization_stats['quality_analyses']}
            """
            axes[1, 3].text(0.1, 0.9, stats_text, transform=axes[1, 3].transAxes, 
                           fontsize=9, verticalalignment='top', fontfamily='monospace',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral"))
            axes[1, 3].set_title('Processing Stats', fontweight='bold')
            axes[1, 3].axis('off')
            
            plt.tight_layout()
            
            # 저장
            if save_path is None:
                save_path = f"{self.save_dir}/complete_pose_pipeline_{self.visualization_stats['comparisons_created']:03d}.png"
            
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"✅ 전체 포즈 추정 파이프라인 비교 시각화 완료: {save_path}")
            return save_path
            
        except Exception as e:
            self.logger.error(f"❌ 전체 포즈 추정 파이프라인 비교 시각화 실패: {e}")
            return ""
    
    def get_visualization_stats(self) -> Dict[str, Any]:
        """시각화 통계 반환"""
        return self.visualization_stats.copy()
    
    def reset_stats(self):
        """통계 초기화"""
        self.visualization_stats = {
            'images_visualized': 0,
            'pose_visualized': 0,
            'comparisons_created': 0,
            'quality_analyses': 0
        }
