"""
품질 평가 유틸리티들
"""

import logging
import numpy as np
from typing import Dict, Any, Optional, Tuple, List

logger = logging.getLogger(__name__)


def validate_matching_result(result: Dict[str, Any]) -> bool:
    """매칭 결과 검증"""
    try:
        # 필수 키 확인
        required_keys = ['transformation_matrix', 'confidence', 'quality_score']
        for key in required_keys:
            if key not in result:
                logger.warning(f"필수 키 누락: {key}")
                return False
        
        # 변환 행렬 검증
        if 'transformation_matrix' in result:
            transform_matrix = result['transformation_matrix']
            if not isinstance(transform_matrix, (np.ndarray, list)):
                logger.warning("변환 행렬이 올바른 형식이 아닙니다.")
                return False
            
            # 행렬 크기 확인
            if isinstance(transform_matrix, np.ndarray):
                if transform_matrix.shape != (3, 3):
                    logger.warning("변환 행렬이 3x3이 아닙니다.")
                    return False
        
        # 신뢰도 점수 검증
        confidence = result.get('confidence', 0)
        if not (0 <= confidence <= 1):
            logger.warning("신뢰도 점수가 0-1 범위를 벗어났습니다.")
            return False
        
        # 품질 점수 검증
        quality_score = result.get('quality_score', 0)
        if not (0 <= quality_score <= 1):
            logger.warning("품질 점수가 0-1 범위를 벗어났습니다.")
            return False
        
        return True
    
    except Exception as e:
        logger.error(f"매칭 결과 검증 실패: {e}")
        return False


def compute_quality_metrics(result: Dict[str, Any], inference_results: Dict[str, Any]) -> Dict[str, Any]:
    """품질 메트릭 계산"""
    try:
        quality_metrics = {}
        
        # 기본 품질 점수
        quality_metrics['base_quality'] = result.get('quality_score', 0.5)
        
        # 신뢰도 점수
        quality_metrics['confidence'] = result.get('confidence', 0.5)
        
        # 변환 행렬 품질
        if 'transformation_matrix' in result:
            transform_quality = _compute_transformation_quality(result['transformation_matrix'])
            quality_metrics['transformation_quality'] = transform_quality
        
        # 키포인트 매칭 품질
        if 'keypoint_matches' in result:
            keypoint_quality = _compute_keypoint_quality(result['keypoint_matches'])
            quality_metrics['keypoint_quality'] = keypoint_quality
        
        # 광학 흐름 품질
        if 'optical_flow' in result:
            flow_quality = _compute_flow_quality(result['optical_flow'])
            quality_metrics['flow_quality'] = flow_quality
        
        # 종합 품질 점수
        quality_metrics['overall_quality'] = _compute_overall_quality(quality_metrics)
        
        return quality_metrics
    
    except Exception as e:
        logger.error(f"품질 메트릭 계산 실패: {e}")
        return {'overall_quality': 0.5}


def evaluate_geometric_matching_quality(result: Dict[str, Any]) -> Dict[str, Any]:
    """기하학적 매칭 품질 평가"""
    try:
        quality_evaluation = {}
        
        # 변환 행렬 평가
        if 'transformation_matrix' in result:
            transform_eval = _evaluate_transformation_matrix(result['transformation_matrix'])
            quality_evaluation['transformation_evaluation'] = transform_eval
        
        # 키포인트 매칭 평가
        if 'keypoint_matches' in result:
            keypoint_eval = _evaluate_keypoint_matches(result['keypoint_matches'])
            quality_evaluation['keypoint_evaluation'] = keypoint_eval
        
        # 광학 흐름 평가
        if 'optical_flow' in result:
            flow_eval = _evaluate_optical_flow(result['optical_flow'])
            quality_evaluation['flow_evaluation'] = flow_eval
        
        # 종합 평가
        quality_evaluation['overall_evaluation'] = _compute_overall_evaluation(quality_evaluation)
        
        return quality_evaluation
    
    except Exception as e:
        logger.error(f"기하학적 매칭 품질 평가 실패: {e}")
        return {'overall_evaluation': 'unknown'}


def _compute_transformation_quality(transform_matrix) -> float:
    """변환 행렬 품질 계산"""
    try:
        if isinstance(transform_matrix, list):
            transform_matrix = np.array(transform_matrix)
        
        if transform_matrix.shape != (3, 3):
            return 0.0
        
        # 행렬식 계산
        det = np.linalg.det(transform_matrix)
        
        # 스케일 팩터 계산
        scale_x = np.sqrt(transform_matrix[0, 0]**2 + transform_matrix[0, 1]**2)
        scale_y = np.sqrt(transform_matrix[1, 0]**2 + transform_matrix[1, 1]**2)
        
        # 품질 점수 계산
        quality_score = 1.0
        
        # 행렬식 검증
        if abs(det) < 0.1 or abs(det) > 10:
            quality_score *= 0.5
        
        # 스케일 팩터 검증
        if scale_x < 0.1 or scale_x > 10 or scale_y < 0.1 or scale_y > 10:
            quality_score *= 0.5
        
        return max(0.0, min(1.0, quality_score))
    
    except Exception as e:
        logger.error(f"변환 행렬 품질 계산 실패: {e}")
        return 0.5


def _compute_keypoint_quality(keypoint_matches) -> float:
    """키포인트 매칭 품질 계산"""
    try:
        if not keypoint_matches:
            return 0.0
        
        # 매칭 개수
        num_matches = len(keypoint_matches)
        
        # 매칭 신뢰도 평균
        confidences = [match.get('confidence', 0) for match in keypoint_matches]
        avg_confidence = np.mean(confidences) if confidences else 0
        
        # 품질 점수 계산
        quality_score = (num_matches / 20) * avg_confidence  # 20개 키포인트 기준
        
        return max(0.0, min(1.0, quality_score))
    
    except Exception as e:
        logger.error(f"키포인트 품질 계산 실패: {e}")
        return 0.5


def _compute_flow_quality(optical_flow) -> float:
    """광학 흐름 품질 계산"""
    try:
        if isinstance(optical_flow, np.ndarray):
            # 흐름 크기 계산
            flow_magnitude = np.sqrt(optical_flow[..., 0]**2 + optical_flow[..., 1]**2)
            
            # 평균 흐름 크기
            avg_magnitude = np.mean(flow_magnitude)
            
            # 흐름 일관성 (표준편차)
            flow_std = np.std(flow_magnitude)
            
            # 품질 점수 계산
            quality_score = 1.0
            
            # 흐름 크기 검증
            if avg_magnitude > 100:  # 너무 큰 흐름
                quality_score *= 0.5
            
            # 흐름 일관성 검증
            if flow_std > 50:  # 너무 불규칙한 흐름
                quality_score *= 0.5
            
            return max(0.0, min(1.0, quality_score))
        
        return 0.5
    
    except Exception as e:
        logger.error(f"광학 흐름 품질 계산 실패: {e}")
        return 0.5


def _compute_overall_quality(quality_metrics: Dict[str, float]) -> float:
    """종합 품질 점수 계산"""
    try:
        weights = {
            'base_quality': 0.3,
            'confidence': 0.2,
            'transformation_quality': 0.3,
            'keypoint_quality': 0.1,
            'flow_quality': 0.1
        }
        
        overall_score = 0.0
        total_weight = 0.0
        
        for metric, weight in weights.items():
            if metric in quality_metrics:
                overall_score += quality_metrics[metric] * weight
                total_weight += weight
        
        if total_weight > 0:
            return overall_score / total_weight
        else:
            return 0.5
    
    except Exception as e:
        logger.error(f"종합 품질 점수 계산 실패: {e}")
        return 0.5


def _evaluate_transformation_matrix(transform_matrix) -> Dict[str, Any]:
    """변환 행렬 평가"""
    try:
        evaluation = {
            'is_valid': True,
            'det_score': 0.0,
            'scale_score': 0.0,
            'overall_score': 0.0
        }
        
        if isinstance(transform_matrix, list):
            transform_matrix = np.array(transform_matrix)
        
        if transform_matrix.shape != (3, 3):
            evaluation['is_valid'] = False
            return evaluation
        
        # 행렬식 평가
        det = np.linalg.det(transform_matrix)
        evaluation['det_score'] = 1.0 if 0.1 <= abs(det) <= 10 else 0.0
        
        # 스케일 평가
        scale_x = np.sqrt(transform_matrix[0, 0]**2 + transform_matrix[0, 1]**2)
        scale_y = np.sqrt(transform_matrix[1, 0]**2 + transform_matrix[1, 1]**2)
        
        if 0.1 <= scale_x <= 10 and 0.1 <= scale_y <= 10:
            evaluation['scale_score'] = 1.0
        
        # 종합 점수
        evaluation['overall_score'] = (evaluation['det_score'] + evaluation['scale_score']) / 2
        
        return evaluation
    
    except Exception as e:
        logger.error(f"변환 행렬 평가 실패: {e}")
        return {'is_valid': False, 'overall_score': 0.0}


def _evaluate_keypoint_matches(keypoint_matches) -> Dict[str, Any]:
    """키포인트 매칭 평가"""
    try:
        evaluation = {
            'num_matches': len(keypoint_matches),
            'avg_confidence': 0.0,
            'match_quality': 'unknown'
        }
        
        if keypoint_matches:
            confidences = [match.get('confidence', 0) for match in keypoint_matches]
            evaluation['avg_confidence'] = np.mean(confidences)
            
            # 매칭 품질 평가
            if evaluation['num_matches'] >= 10 and evaluation['avg_confidence'] >= 0.7:
                evaluation['match_quality'] = 'excellent'
            elif evaluation['num_matches'] >= 5 and evaluation['avg_confidence'] >= 0.5:
                evaluation['match_quality'] = 'good'
            elif evaluation['num_matches'] >= 3:
                evaluation['match_quality'] = 'fair'
            else:
                evaluation['match_quality'] = 'poor'
        
        return evaluation
    
    except Exception as e:
        logger.error(f"키포인트 매칭 평가 실패: {e}")
        return {'num_matches': 0, 'avg_confidence': 0.0, 'match_quality': 'unknown'}


def _evaluate_optical_flow(optical_flow) -> Dict[str, Any]:
    """광학 흐름 평가"""
    try:
        evaluation = {
            'avg_magnitude': 0.0,
            'flow_consistency': 0.0,
            'flow_quality': 'unknown'
        }
        
        if isinstance(optical_flow, np.ndarray):
            # 흐름 크기 계산
            flow_magnitude = np.sqrt(optical_flow[..., 0]**2 + optical_flow[..., 1]**2)
            evaluation['avg_magnitude'] = float(np.mean(flow_magnitude))
            evaluation['flow_consistency'] = float(1.0 / (1.0 + np.std(flow_magnitude)))
            
            # 흐름 품질 평가
            if evaluation['avg_magnitude'] < 50 and evaluation['flow_consistency'] > 0.7:
                evaluation['flow_quality'] = 'excellent'
            elif evaluation['avg_magnitude'] < 100 and evaluation['flow_consistency'] > 0.5:
                evaluation['flow_quality'] = 'good'
            elif evaluation['avg_magnitude'] < 200:
                evaluation['flow_quality'] = 'fair'
            else:
                evaluation['flow_quality'] = 'poor'
        
        return evaluation
    
    except Exception as e:
        logger.error(f"광학 흐름 평가 실패: {e}")
        return {'avg_magnitude': 0.0, 'flow_consistency': 0.0, 'flow_quality': 'unknown'}


def _compute_overall_evaluation(quality_evaluation: Dict[str, Any]) -> str:
    """종합 평가 계산"""
    try:
        scores = []
        
        # 변환 행렬 평가
        if 'transformation_evaluation' in quality_evaluation:
            transform_score = quality_evaluation['transformation_evaluation'].get('overall_score', 0)
            scores.append(transform_score)
        
        # 키포인트 매칭 평가
        if 'keypoint_evaluation' in quality_evaluation:
            keypoint_quality = quality_evaluation['keypoint_evaluation'].get('match_quality', 'unknown')
            if keypoint_quality == 'excellent':
                scores.append(1.0)
            elif keypoint_quality == 'good':
                scores.append(0.7)
            elif keypoint_quality == 'fair':
                scores.append(0.4)
            else:
                scores.append(0.1)
        
        # 광학 흐름 평가
        if 'flow_evaluation' in quality_evaluation:
            flow_quality = quality_evaluation['flow_evaluation'].get('flow_quality', 'unknown')
            if flow_quality == 'excellent':
                scores.append(1.0)
            elif flow_quality == 'good':
                scores.append(0.7)
            elif flow_quality == 'fair':
                scores.append(0.4)
            else:
                scores.append(0.1)
        
        if scores:
            avg_score = np.mean(scores)
            if avg_score >= 0.8:
                return 'excellent'
            elif avg_score >= 0.6:
                return 'good'
            elif avg_score >= 0.4:
                return 'fair'
            else:
                return 'poor'
        else:
            return 'unknown'
    
    except Exception as e:
        logger.error(f"종합 평가 계산 실패: {e}")
        return 'unknown'
