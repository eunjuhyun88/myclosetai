"""
🔥 Geometric Matching 품질 평가 시스템
=====================================

기하학적 매칭 결과의 품질을 평가하고 개선 방향을 제시

Author: MyCloset AI Team
Date: 2025-08-15
Version: 1.0
"""

import logging
import torch
import numpy as np
from typing import Dict, Any, List, Tuple
import cv2
from scipy import ndimage

logger = logging.getLogger(__name__)

class GeometricMatchingQualityAssessment:
    """기하학적 매칭 품질 평가 및 개선 시스템"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.QualityAssessment")
        self.quality_metrics = {
            'keypoint_accuracy': 0.0,
            'matching_consistency': 0.0,
            'geometric_coherence': 0.0,
            'transformation_quality': 0.0,
            'overall_quality': 0.0
        }
    
    def assess_quality(self, matching_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        기하학적 매칭 결과의 품질을 종합적으로 평가
        
        Args:
            matching_result: 매칭 결과 딕셔너리
            
        Returns:
            quality_report: 품질 평가 리포트
        """
        try:
            self.logger.info("🔍 기하학적 매칭 품질 평가 시작")
            
            # 기본 품질 메트릭 계산
            quality_scores = self._calculate_quality_metrics(matching_result)
            
            # 개선 제안 생성
            improvement_suggestions = self._generate_improvement_suggestions(quality_scores)
            
            # 최종 품질 리포트 생성
            quality_report = {
                'quality_scores': quality_scores,
                'improvement_suggestions': improvement_suggestions,
                'overall_quality': quality_scores['overall_quality'],
                'quality_level': self._get_quality_level(quality_scores['overall_quality']),
                'assessment_timestamp': str(np.datetime64('now'))
            }
            
            self.logger.info(f"✅ 품질 평가 완료 - 전체 품질: {quality_scores['overall_quality']:.3f}")
            return quality_report
            
        except Exception as e:
            self.logger.error(f"❌ 품질 평가 실패: {e}")
            return {
                'quality_scores': self.quality_metrics,
                'error': str(e),
                'overall_quality': 0.0
            }
    
    def _calculate_quality_metrics(self, matching_result: Dict[str, Any]) -> Dict[str, float]:
        """품질 메트릭 계산"""
        try:
            # 키포인트 정확도 평가
            keypoint_accuracy = self._assess_keypoint_accuracy(matching_result)
            
            # 매칭 일관성 평가
            matching_consistency = self._assess_matching_consistency(matching_result)
            
            # 기하학적 일관성 평가
            geometric_coherence = self._assess_geometric_coherence(matching_result)
            
            # 변환 품질 평가
            transformation_quality = self._assess_transformation_quality(matching_result)
            
            # 전체 품질 점수 계산 (가중 평균)
            overall_quality = (
                keypoint_accuracy * 0.3 +
                matching_consistency * 0.3 +
                geometric_coherence * 0.2 +
                transformation_quality * 0.2
            )
            
            return {
                'keypoint_accuracy': keypoint_accuracy,
                'matching_consistency': matching_consistency,
                'geometric_coherence': geometric_coherence,
                'transformation_quality': transformation_quality,
                'overall_quality': overall_quality
            }
            
        except Exception as e:
            self.logger.warning(f"품질 메트릭 계산 실패: {e}")
            return self.quality_metrics
    
    def _assess_keypoint_accuracy(self, matching_result: Dict[str, Any]) -> float:
        """키포인트 정확도 평가"""
        try:
            if 'keypoints' not in matching_result:
                return 0.5
            
            keypoints = matching_result['keypoints']
            
            # 텐서를 numpy로 변환
            if torch.is_tensor(keypoints):
                keypoints = keypoints.detach().cpu().numpy()
            
            # 키포인트 정확도 계산
            accuracy_score = self._calculate_keypoint_accuracy(keypoints)
            
            return min(1.0, max(0.0, accuracy_score))
            
        except Exception as e:
            self.logger.warning(f"키포인트 정확도 평가 실패: {e}")
            return 0.5
    
    def _assess_matching_consistency(self, matching_result: Dict[str, Any]) -> float:
        """매칭 일관성 평가"""
        try:
            if 'matching_scores' not in matching_result:
                return 0.5
            
            matching_scores = matching_result['matching_scores']
            
            # 텐서를 numpy로 변환
            if torch.is_tensor(matching_scores):
                matching_scores = matching_scores.detach().cpu().numpy()
            
            # 매칭 일관성 계산
            consistency_score = self._calculate_matching_consistency(matching_scores)
            
            return min(1.0, max(0.0, consistency_score))
            
        except Exception as e:
            self.logger.warning(f"매칭 일관성 평가 실패: {e}")
            return 0.5
    
    def _assess_geometric_coherence(self, matching_result: Dict[str, Any]) -> float:
        """기하학적 일관성 평가"""
        try:
            if 'geometric_features' not in matching_result:
                return 0.5
            
            geometric_features = matching_result['geometric_features']
            
            # 텐서를 numpy로 변환
            if torch.is_tensor(geometric_features):
                geometric_features = geometric_features.detach().cpu().numpy()
            
            # 기하학적 일관성 계산
            coherence_score = self._calculate_geometric_coherence(geometric_features)
            
            return min(1.0, max(0.0, coherence_score))
            
        except Exception as e:
            self.logger.warning(f"기하학적 일관성 평가 실패: {e}")
            return 0.5
    
    def _assess_transformation_quality(self, matching_result: Dict[str, Any]) -> float:
        """변환 품질 평가"""
        try:
            if 'transformation_matrix' not in matching_result:
                return 0.5
            
            transformation_matrix = matching_result['transformation_matrix']
            
            # 텐서를 numpy로 변환
            if torch.is_tensor(transformation_matrix):
                transformation_matrix = transformation_matrix.detach().cpu().numpy()
            
            # 변환 품질 계산
            quality_score = self._calculate_transformation_quality(transformation_matrix)
            
            return min(1.0, max(0.0, quality_score))
            
        except Exception as e:
            self.logger.warning(f"변환 품질 평가 실패: {e}")
            return 0.5
    
    def _calculate_keypoint_accuracy(self, keypoints: np.ndarray) -> float:
        """키포인트 정확도 계산"""
        try:
            if keypoints.size == 0:
                return 0.5
            
            # 키포인트의 공간적 분포 평가
            if len(keypoints.shape) == 3:  # (batch, num_keypoints, 2)
                batch_size, num_keypoints, _ = keypoints.shape
                
                accuracy_scores = []
                for b in range(batch_size):
                    batch_keypoints = keypoints[b]
                    
                    # 키포인트 간 거리 계산
                    distances = []
                    for i in range(num_keypoints):
                        for j in range(i + 1, num_keypoints):
                            dist = np.linalg.norm(batch_keypoints[i] - batch_keypoints[j])
                            distances.append(dist)
                    
                    if distances:
                        # 거리의 일관성 평가
                        mean_dist = np.mean(distances)
                        std_dist = np.std(distances)
                        
                        if mean_dist > 0:
                            consistency = 1.0 - (std_dist / mean_dist)
                            accuracy_scores.append(max(0, consistency))
                        else:
                            accuracy_scores.append(0.5)
                    else:
                        accuracy_scores.append(0.5)
                
                return np.mean(accuracy_scores) if accuracy_scores else 0.5
            
            return 0.5
            
        except Exception as e:
            self.logger.warning(f"키포인트 정확도 계산 실패: {e}")
            return 0.5
    
    def _calculate_matching_consistency(self, matching_scores: np.ndarray) -> float:
        """매칭 일관성 계산"""
        try:
            if matching_scores.size == 0:
                return 0.5
            
            # 매칭 점수의 일관성 평가
            if len(matching_scores.shape) == 3:  # (batch, source, target)
                batch_size, num_source, num_target = matching_scores.shape
                
                consistency_scores = []
                for b in range(batch_size):
                    batch_scores = matching_scores[b]
                    
                    # 각 소스에 대해 최고 점수 찾기
                    best_scores = np.max(batch_scores, axis=1)
                    
                    # 점수의 일관성 평가
                    if best_scores.size > 0:
                        mean_score = np.mean(best_scores)
                        std_score = np.std(best_scores)
                        
                        if mean_score > 0:
                            consistency = 1.0 - (std_score / mean_score)
                            consistency_scores.append(max(0, consistency))
                        else:
                            consistency_scores.append(0.5)
                    else:
                        consistency_scores.append(0.5)
                
                return np.mean(consistency_scores) if consistency_scores else 0.5
            
            return 0.5
            
        except Exception as e:
            self.logger.warning(f"매칭 일관성 계산 실패: {e}")
            return 0.5
    
    def _calculate_geometric_coherence(self, geometric_features: np.ndarray) -> float:
        """기하학적 일관성 계산"""
        try:
            if geometric_features.size == 0:
                return 0.5
            
            # 기하학적 특징의 일관성 평가
            if len(geometric_features.shape) == 3:  # (batch, num_features, feature_dim)
                batch_size, num_features, feature_dim = geometric_features.shape
                
                coherence_scores = []
                for b in range(batch_size):
                    batch_features = geometric_features[b]
                    
                    # 특징 간의 유사도 계산
                    similarities = []
                    for i in range(num_features):
                        for j in range(i + 1, num_features):
                            # 코사인 유사도
                            dot_product = np.dot(batch_features[i], batch_features[j])
                            norm_i = np.linalg.norm(batch_features[i])
                            norm_j = np.linalg.norm(batch_features[j])
                            
                            if norm_i > 0 and norm_j > 0:
                                similarity = dot_product / (norm_i * norm_j)
                                similarities.append(similarity)
                    
                    if similarities:
                        # 유사도의 일관성 평가
                        mean_similarity = np.mean(similarities)
                        coherence_scores.append(max(0, mean_similarity))
                    else:
                        coherence_scores.append(0.5)
                
                return np.mean(coherence_scores) if coherence_scores else 0.5
            
            return 0.5
            
        except Exception as e:
            self.logger.warning(f"기하학적 일관성 계산 실패: {e}")
            return 0.5
    
    def _calculate_transformation_quality(self, transformation_matrix: np.ndarray) -> float:
        """변환 품질 계산"""
        try:
            if transformation_matrix.size == 0:
                return 0.5
            
            # 변환 행렬의 품질 평가
            if len(transformation_matrix.shape) == 3:  # (batch, 3, 3)
                batch_size = transformation_matrix.shape[0]
                
                quality_scores = []
                for b in range(batch_size):
                    matrix = transformation_matrix[b]
                    
                    # 행렬의 조건수 계산 (안정성 측정)
                    try:
                        condition_number = np.linalg.cond(matrix)
                        # 조건수가 작을수록 안정적
                        stability = 1.0 / (1.0 + condition_number)
                        quality_scores.append(stability)
                    except:
                        quality_scores.append(0.5)
                
                return np.mean(quality_scores) if quality_scores else 0.5
            
            return 0.5
            
        except Exception as e:
            self.logger.warning(f"변환 품질 계산 실패: {e}")
            return 0.5
    
    def _generate_improvement_suggestions(self, quality_scores: Dict[str, float]) -> List[str]:
        """품질 개선 제안 생성"""
        suggestions = []
        
        if quality_scores['keypoint_accuracy'] < 0.7:
            suggestions.append("키포인트 정확도 개선: 키포인트 검출 알고리즘 강화 필요")
        
        if quality_scores['matching_consistency'] < 0.7:
            suggestions.append("매칭 일관성 개선: 매칭 알고리즘의 일관성 강화 필요")
        
        if quality_scores['geometric_coherence'] < 0.7:
            suggestions.append("기하학적 일관성 개선: 기하학적 특징 추출 방식 개선 필요")
        
        if quality_scores['transformation_quality'] < 0.7:
            suggestions.append("변환 품질 개선: 변환 행렬 계산 방식 개선 필요")
        
        if not suggestions:
            suggestions.append("현재 품질이 양호합니다. 추가 개선이 필요하지 않습니다.")
        
        return suggestions
    
    def _get_quality_level(self, overall_quality: float) -> str:
        """전체 품질 수준 판정"""
        if overall_quality >= 0.9:
            return "Excellent"
        elif overall_quality >= 0.8:
            return "Good"
        elif overall_quality >= 0.7:
            return "Fair"
        elif overall_quality >= 0.6:
            return "Poor"
        else:
            return "Very Poor"
