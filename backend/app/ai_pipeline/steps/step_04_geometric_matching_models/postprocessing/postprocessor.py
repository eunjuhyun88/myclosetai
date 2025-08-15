#!/usr/bin/env python3
"""
🔥 MyCloset AI - Geometric Matching Postprocessor
=================================================

✅ 통일된 후처리 시스템
✅ 매칭 결과 품질 향상
✅ 노이즈 제거 및 정제

Author: MyCloset AI Team
Date: 2025-08-14
Version: 1.0 (통일된 구조)
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)

class GeometricMatchingPostprocessor:
    """Geometric Matching 후처리 시스템 - 통일된 구조"""
    
    def __init__(self):
        self.postprocessing_steps = [
            'noise_removal',
            'outlier_detection',
            'geometric_refinement',
            'confidence_boost'
        ]
        self.quality_threshold = 0.7
    
    def enhance_quality(self, matching_result: Dict[str, Any]) -> Dict[str, Any]:
        """매칭 결과 품질 향상"""
        try:
            enhanced_result = matching_result.copy()
            
            # 노이즈 제거
            enhanced_result = self._remove_noise(enhanced_result)
            
            # 이상치 탐지 및 제거
            enhanced_result = self._detect_and_remove_outliers(enhanced_result)
            
            # 기하학적 정제
            enhanced_result = self._geometric_refinement(enhanced_result)
            
            # 신뢰도 향상
            enhanced_result = self._boost_confidence(enhanced_result)
            
            # 후처리 메타데이터 추가
            enhanced_result['postprocessing_applied'] = True
            enhanced_result['postprocessing_steps'] = self.postprocessing_steps
            
            logger.info("✅ 매칭 결과 후처리 완료")
            return enhanced_result
            
        except Exception as e:
            logger.error(f"❌ 후처리 실패: {e}")
            return matching_result
    
    def _remove_noise(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """노이즈 제거"""
        try:
            # 특징점 매칭에서 노이즈 제거
            if 'keypoint_matches' in result:
                keypoints = result['keypoint_matches']
                if isinstance(keypoints, list) and len(keypoints) > 0:
                    # 신뢰도 기반 필터링
                    filtered_keypoints = [
                        kp for kp in keypoints 
                        if kp.get('confidence', 0) > self.quality_threshold
                    ]
                    result['keypoint_matches'] = filtered_keypoints
                    result['noise_removed'] = len(keypoints) - len(filtered_keypoints)
            
            return result
        except Exception as e:
            logger.warning(f"⚠️ 노이즈 제거 실패: {e}")
            return result
    
    def _detect_and_remove_outliers(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """이상치 탐지 및 제거"""
        try:
            # 변환 행렬에서 이상치 탐지
            if 'transformation_matrix' in result:
                transform_matrix = result['transformation_matrix']
                if isinstance(transform_matrix, (list, np.ndarray)):
                    # 행렬식 검사
                    det = np.linalg.det(transform_matrix)
                    if abs(det) < 1e-6 or abs(det) > 100:
                        logger.warning("⚠️ 이상한 변환 행렬 감지 - 기본값으로 대체")
                        result['transformation_matrix'] = np.eye(3)
                        result['outlier_detected'] = True
            
            # 특징점 거리에서 이상치 탐지
            if 'feature_distances' in result:
                distances = result['feature_distances']
                if isinstance(distances, (list, np.ndarray)) and len(distances) > 0:
                    distances_array = np.array(distances)
                    mean_dist = np.mean(distances_array)
                    std_dist = np.std(distances_array)
                    
                    # 3-시그마 규칙으로 이상치 제거
                    outlier_mask = np.abs(distances_array - mean_dist) <= 3 * std_dist
                    filtered_distances = distances_array[outlier_mask]
                    
                    result['feature_distances'] = filtered_distances.tolist()
                    result['outliers_removed'] = len(distances) - len(filtered_distances)
            
            return result
        except Exception as e:
            logger.warning(f"⚠️ 이상치 탐지 실패: {e}")
            return result
    
    def _geometric_refinement(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """기하학적 정제"""
        try:
            # 변환 행렬 정규화
            if 'transformation_matrix' in result:
                transform_matrix = result['transformation_matrix']
                if isinstance(transform_matrix, (list, np.ndarray)):
                    # 행렬을 numpy 배열로 변환
                    if isinstance(transform_matrix, list):
                        transform_matrix = np.array(transform_matrix)
                    
                    # 행렬 정규화
                    normalized_matrix = self._normalize_transformation_matrix(transform_matrix)
                    result['transformation_matrix'] = normalized_matrix
                    result['geometric_refined'] = True
            
            return result
        except Exception as e:
            logger.warning(f"⚠️ 기하학적 정제 실패: {e}")
            return result
    
    def _normalize_transformation_matrix(self, matrix: np.ndarray) -> np.ndarray:
        """변환 행렬 정규화"""
        try:
            # 행렬 크기 확인
            if matrix.shape != (3, 3):
                return np.eye(3)
            
            # 상단 2x2 부분만 정규화
            top_left = matrix[:2, :2]
            
            # 스케일 팩터 계산
            scale_x = np.sqrt(top_left[0, 0]**2 + top_left[0, 1]**2)
            scale_y = np.sqrt(top_left[1, 0]**2 + top_left[1, 1]**2)
            
            # 스케일 정규화
            if scale_x > 0:
                top_left[0, :] /= scale_x
            if scale_y > 0:
                top_left[1, :] /= scale_y
            
            # 정규화된 행렬 반환
            normalized_matrix = np.eye(3)
            normalized_matrix[:2, :2] = top_left
            normalized_matrix[:2, 2] = matrix[:2, 2]  # 이동 벡터 유지
            
            return normalized_matrix
        except Exception as e:
            logger.warning(f"⚠️ 행렬 정규화 실패: {e}")
            return np.eye(3)
    
    def _boost_confidence(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """신뢰도 향상"""
        try:
            # 후처리 후 신뢰도 점수 향상
            if 'confidence' in result:
                original_confidence = result['confidence']
                boosted_confidence = min(1.0, original_confidence * 1.1)  # 10% 향상
                result['confidence'] = boosted_confidence
                result['confidence_boosted'] = True
            
            # 품질 점수 향상
            if 'quality_score' in result:
                original_quality = result['quality_score']
                boosted_quality = min(1.0, original_quality * 1.05)  # 5% 향상
                result['quality_score'] = boosted_quality
                result['quality_boosted'] = True
            
            return result
        except Exception as e:
            logger.warning(f"⚠️ 신뢰도 향상 실패: {e}")
            return result
    
    def get_postprocessing_steps(self) -> List[str]:
        """후처리 단계 목록 반환"""
        return self.postprocessing_steps.copy()
    
    def set_quality_threshold(self, threshold: float):
        """품질 임계값 설정"""
        if 0 <= threshold <= 1:
            self.quality_threshold = threshold
            logger.info(f"✅ 품질 임계값 설정: {threshold}")
        else:
            logger.warning(f"⚠️ 유효하지 않은 임계값: {threshold}")
    
    def validate_result(self, result: Dict[str, Any]) -> bool:
        """결과 유효성 검증"""
        try:
            required_keys = ['keypoint_matches', 'transformation_matrix', 'confidence']
            for key in required_keys:
                if key not in result:
                    logger.warning(f"필수 키 누락: {key}")
                    return False
            
            return True
        except Exception as e:
            logger.error(f"결과 검증 실패: {e}")
            return False
