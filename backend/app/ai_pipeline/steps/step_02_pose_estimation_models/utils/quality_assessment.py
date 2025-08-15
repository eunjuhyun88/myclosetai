"""
🔥 Pose Estimation Quality Assessment - 포즈 추정 품질 평가
=======================================================

포즈 추정 결과의 품질을 평가하는 시스템

주요 기능:
- 키포인트 품질 평가
- 신뢰도 분석
- 공간적 일관성 검사
- 전반적 품질 점수 계산
"""

import logging
import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Optional, List, Tuple

logger = logging.getLogger(__name__)

class PoseEstimationQualityAssessment:
    """포즈 추정 품질 평가 시스템"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # 기본 설정
        self.confidence_threshold = self.config.get('confidence_threshold', 0.3)
        self.spatial_consistency_threshold = self.config.get('spatial_consistency_threshold', 0.5)
        self.quality_weights = self.config.get('quality_weights', {
            'confidence': 0.4,
            'spatial_consistency': 0.3,
            'temporal_stability': 0.2,
            'keypoint_distribution': 0.1
        })
        
        logger.info("✅ Pose Estimation Quality Assessment 초기화 완료")
    
    def assess_quality(self, keypoints: torch.Tensor, **kwargs) -> Dict[str, Any]:
        """
        포즈 추정 결과 품질 평가
        
        Args:
            keypoints: 키포인트 텐서 (B, N, 3) - x, y, confidence
            **kwargs: 추가 파라미터
        
        Returns:
            quality_result: 품질 평가 결과
        """
        try:
            logger.info("🚀 포즈 추정 품질 평가 시작")
            
            if keypoints is None:
                logger.warning("⚠️ 키포인트가 None입니다")
                return {
                    'quality_score': 0.0,
                    'confidence': 0.0,
                    'spatial_consistency': 0.0,
                    'temporal_stability': 0.0,
                    'keypoint_distribution': 0.0,
                    'overall_quality': 'poor'
                }
            
            # 1. 신뢰도 평가
            confidence_score = self._assess_confidence(keypoints)
            
            # 2. 공간적 일관성 평가
            spatial_consistency_score = self._assess_spatial_consistency(keypoints)
            
            # 3. 시간적 안정성 평가
            temporal_stability_score = self._assess_temporal_stability(keypoints)
            
            # 4. 키포인트 분포 평가
            keypoint_distribution_score = self._assess_keypoint_distribution(keypoints)
            
            # 5. 종합 품질 점수 계산
            overall_score = self._calculate_overall_quality(
                confidence_score,
                spatial_consistency_score,
                temporal_stability_score,
                keypoint_distribution_score
            )
            
            # 6. 품질 등급 결정
            quality_grade = self._determine_quality_grade(overall_score)
            
            quality_result = {
                'quality_score': overall_score,
                'confidence': confidence_score,
                'spatial_consistency': spatial_consistency_score,
                'temporal_stability': temporal_stability_score,
                'keypoint_distribution': keypoint_distribution_score,
                'overall_quality': quality_grade,
                'assessment_details': {
                    'confidence_threshold': self.confidence_threshold,
                    'spatial_consistency_threshold': self.spatial_consistency_threshold,
                    'quality_weights': self.quality_weights
                }
            }
            
            logger.info(f"✅ 품질 평가 완료: 전체 점수={overall_score:.3f}, 등급={quality_grade}")
            return quality_result
            
        except Exception as e:
            logger.error(f"❌ 품질 평가 실패: {e}")
            return {
                'quality_score': 0.0,
                'confidence': 0.0,
                'spatial_consistency': 0.0,
                'temporal_stability': 0.0,
                'keypoint_distribution': 0.0,
                'overall_quality': 'poor',
                'error': str(e)
            }
    
    def _assess_confidence(self, keypoints: torch.Tensor) -> float:
        """신뢰도 평가"""
        try:
            # 신뢰도 값 추출 (마지막 차원)
            confidences = keypoints[..., 2]
            
            # 평균 신뢰도 계산
            mean_confidence = torch.mean(confidences).item()
            
            # 높은 신뢰도 키포인트 비율 계산
            high_confidence_ratio = torch.mean(
                (confidences >= self.confidence_threshold).float()
            ).item()
            
            # 신뢰도 점수 계산 (평균 + 높은 신뢰도 비율)
            confidence_score = (mean_confidence + high_confidence_ratio) / 2.0
            
            return min(confidence_score, 1.0)
            
        except Exception as e:
            logger.warning(f"신뢰도 평가 실패: {e}")
            return 0.5
    
    def _assess_spatial_consistency(self, keypoints: torch.Tensor) -> float:
        """공간적 일관성 평가"""
        try:
            # 좌표 값 추출 (x, y)
            coordinates = keypoints[..., :2]
            
            # 배치별로 처리
            batch_scores = []
            
            for b in range(coordinates.size(0)):
                frame_coordinates = coordinates[b]  # (N, 2)
                
                # 인접한 키포인트 간의 거리 계산
                distances = []
                for k in range(1, frame_coordinates.size(0)):
                    prev_kp = frame_coordinates[k-1]
                    curr_kp = frame_coordinates[k]
                    
                    # 유클리드 거리 계산
                    distance = torch.sqrt(
                        (curr_kp[0] - prev_kp[0])**2 + 
                        (curr_kp[1] - prev_kp[1])**2
                    )
                    distances.append(distance.item())
                
                if distances:
                    # 거리의 표준편차 계산 (낮을수록 일관성 높음)
                    distances_array = np.array(distances)
                    distance_std = np.std(distances_array)
                    
                    # 표준편차를 점수로 변환 (낮을수록 높은 점수)
                    consistency_score = max(0.0, 1.0 - distance_std / self.spatial_consistency_threshold)
                    batch_scores.append(consistency_score)
            
            # 배치 평균 계산
            if batch_scores:
                return np.mean(batch_scores)
            else:
                return 0.5
                
        except Exception as e:
            logger.warning(f"공간적 일관성 평가 실패: {e}")
            return 0.5
    
    def _assess_temporal_stability(self, keypoints: torch.Tensor) -> float:
        """시간적 안정성 평가"""
        try:
            # 배치 크기가 1이면 시간적 변화 없음
            if keypoints.size(0) <= 1:
                return 1.0
            
            # 좌표 값 추출 (x, y)
            coordinates = keypoints[..., :2]
            
            # 프레임 간 변화량 계산
            frame_changes = []
            
            for b in range(1, coordinates.size(0)):
                prev_frame = coordinates[b-1]  # (N, 2)
                curr_frame = coordinates[b]    # (N, 2)
                
                # 프레임 간 변화량 계산
                frame_change = torch.mean(
                    torch.sqrt(
                        (curr_frame[:, 0] - prev_frame[:, 0])**2 + 
                        (curr_frame[:, 1] - prev_frame[:, 1])**2
                    )
                ).item()
                
                frame_changes.append(frame_change)
            
            if frame_changes:
                # 변화량의 평균 계산
                mean_change = np.mean(frame_changes)
                
                # 변화량을 안정성 점수로 변환 (낮을수록 높은 점수)
                stability_score = max(0.0, 1.0 - mean_change / 0.5)  # 0.5는 임계값
                return stability_score
            else:
                return 0.5
                
        except Exception as e:
            logger.warning(f"시간적 안정성 평가 실패: {e}")
            return 0.5
    
    def _assess_keypoint_distribution(self, keypoints: torch.Tensor) -> float:
        """키포인트 분포 평가"""
        try:
            # 좌표 값 추출 (x, y)
            coordinates = keypoints[..., :2]
            
            # 배치별로 처리
            batch_scores = []
            
            for b in range(coordinates.size(0)):
                frame_coordinates = coordinates[b]  # (N, 2)
                
                # x, y 좌표의 분산 계산
                x_coords = frame_coordinates[:, 0]
                y_coords = frame_coordinates[:, 1]
                
                x_variance = torch.var(x_coords).item()
                y_variance = torch.var(y_coords).item()
                
                # 분산이 너무 작으면 (모든 키포인트가 한 곳에 몰림) 낮은 점수
                # 분산이 너무 크면 (키포인트가 너무 분산됨) 낮은 점수
                # 적절한 분산 범위에서 높은 점수
                
                # x, y 분산의 적절성 평가
                x_score = self._evaluate_variance(x_variance)
                y_score = self._evaluate_variance(y_variance)
                
                # 평균 분포 점수
                distribution_score = (x_score + y_score) / 2.0
                batch_scores.append(distribution_score)
            
            # 배치 평균 계산
            if batch_scores:
                return np.mean(batch_scores)
            else:
                return 0.5
                
        except Exception as e:
            logger.warning(f"키포인트 분포 평가 실패: {e}")
            return 0.5
    
    def _evaluate_variance(self, variance: float) -> float:
        """분산 값 평가"""
        # 적절한 분산 범위: 0.01 ~ 0.25
        if variance < 0.01:  # 너무 작음
            return 0.3
        elif variance > 0.25:  # 너무 큼
            return 0.4
        else:  # 적절함
            return 1.0
    
    def _calculate_overall_quality(self, 
                                 confidence_score: float,
                                 spatial_consistency_score: float,
                                 temporal_stability_score: float,
                                 keypoint_distribution_score: float) -> float:
        """종합 품질 점수 계산"""
        try:
            # 가중 평균 계산
            overall_score = (
                self.quality_weights['confidence'] * confidence_score +
                self.quality_weights['spatial_consistency'] * spatial_consistency_score +
                self.quality_weights['temporal_stability'] * temporal_stability_score +
                self.quality_weights['keypoint_distribution'] * keypoint_distribution_score
            )
            
            return min(overall_score, 1.0)
            
        except Exception as e:
            logger.warning(f"종합 품질 점수 계산 실패: {e}")
            return 0.5
    
    def _determine_quality_grade(self, overall_score: float) -> str:
        """품질 등급 결정"""
        if overall_score >= 0.8:
            return 'excellent'
        elif overall_score >= 0.6:
            return 'good'
        elif overall_score >= 0.4:
            return 'fair'
        else:
            return 'poor'
    
    def get_assessment_config(self) -> Dict[str, Any]:
        """평가 설정 반환"""
        return {
            'confidence_threshold': self.confidence_threshold,
            'spatial_consistency_threshold': self.spatial_consistency_threshold,
            'quality_weights': self.quality_weights
        }
    
    def update_config(self, new_config: Dict[str, Any]):
        """설정 업데이트"""
        try:
            if 'confidence_threshold' in new_config:
                self.confidence_threshold = new_config['confidence_threshold']
            
            if 'spatial_consistency_threshold' in new_config:
                self.spatial_consistency_threshold = new_config['spatial_consistency_threshold']
            
            if 'quality_weights' in new_config:
                self.quality_weights.update(new_config['quality_weights'])
            
            logger.info("✅ 품질 평가 설정 업데이트 완료")
            
        except Exception as e:
            logger.error(f"❌ 설정 업데이트 실패: {e}")
