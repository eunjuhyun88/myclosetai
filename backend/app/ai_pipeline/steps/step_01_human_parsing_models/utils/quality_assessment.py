"""
🔥 Human Parsing 품질 평가 시스템
================================

Human Parsing 결과의 품질을 평가하고 개선 방향을 제시

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

class HumanParsingQualityAssessment:
    """Human Parsing 품질 평가 및 개선 시스템"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.QualityAssessment")
        self.quality_metrics = {
            'boundary_consistency': 0.0,
            'semantic_coherence': 0.0,
            'spatial_continuity': 0.0,
            'confidence_reliability': 0.0,
            'overall_quality': 0.0
        }
    
    def assess_quality(self, parsing_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Human Parsing 결과의 품질을 종합적으로 평가
        
        Args:
            parsing_result: 파싱 결과 딕셔너리
            
        Returns:
            quality_report: 품질 평가 리포트
        """
        try:
            self.logger.info("🔍 Human Parsing 품질 평가 시작")
            
            # 기본 품질 메트릭 계산
            quality_scores = self._calculate_quality_metrics(parsing_result)
            
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
    
    def _calculate_quality_metrics(self, parsing_result: Dict[str, Any]) -> Dict[str, float]:
        """품질 메트릭 계산"""
        try:
            # 경계 일관성 평가
            boundary_consistency = self._assess_boundary_consistency(parsing_result)
            
            # 의미적 일관성 평가
            semantic_coherence = self._assess_semantic_coherence(parsing_result)
            
            # 공간적 연속성 평가
            spatial_continuity = self._assess_spatial_continuity(parsing_result)
            
            # 신뢰도 평가
            confidence_reliability = self._assess_confidence_reliability(parsing_result)
            
            # 전체 품질 점수 계산 (가중 평균)
            overall_quality = (
                boundary_consistency * 0.3 +
                semantic_coherence * 0.3 +
                spatial_continuity * 0.2 +
                confidence_reliability * 0.2
            )
            
            return {
                'boundary_consistency': boundary_consistency,
                'semantic_coherence': semantic_coherence,
                'spatial_continuity': spatial_continuity,
                'confidence_reliability': confidence_reliability,
                'overall_quality': overall_quality
            }
            
        except Exception as e:
            self.logger.warning(f"품질 메트릭 계산 실패: {e}")
            return self.quality_metrics
    
    def _assess_boundary_consistency(self, parsing_result: Dict[str, Any]) -> float:
        """경계 일관성 평가"""
        try:
            if 'parsing_map' not in parsing_result:
                return 0.5
            
            parsing_map = parsing_result['parsing_map']
            
            # 텐서를 numpy로 변환
            if torch.is_tensor(parsing_map):
                parsing_map = parsing_map.detach().cpu().numpy()
            
            # 경계 검출
            boundaries = self._extract_boundaries(parsing_map)
            
            # 경계의 일관성 평가
            boundary_consistency = self._calculate_boundary_consistency(boundaries)
            
            return min(1.0, max(0.0, boundary_consistency))
            
        except Exception as e:
            self.logger.warning(f"경계 일관성 평가 실패: {e}")
            return 0.5
    
    def _assess_semantic_coherence(self, parsing_result: Dict[str, Any]) -> float:
        """의미적 일관성 평가"""
        try:
            if 'parsing_map' not in parsing_result:
                return 0.5
            
            parsing_map = parsing_result['parsing_map']
            
            # 텐서를 numpy로 변환
            if torch.is_tensor(parsing_map):
                parsing_map = parsing_map.detach().cpu().numpy()
            
            # 의미적 일관성 계산
            semantic_coherence = self._calculate_semantic_coherence(parsing_map)
            
            return min(1.0, max(0.0, semantic_coherence))
            
        except Exception as e:
            self.logger.warning(f"의미적 일관성 평가 실패: {e}")
            return 0.5
    
    def _assess_spatial_continuity(self, parsing_result: Dict[str, Any]) -> float:
        """공간적 연속성 평가"""
        try:
            if 'parsing_map' not in parsing_result:
                return 0.5
            
            parsing_map = parsing_result['parsing_map']
            
            # 텐서를 numpy로 변환
            if torch.is_tensor(parsing_map):
                parsing_map = parsing_map.detach().cpu().numpy()
            
            # 공간적 연속성 계산
            spatial_continuity = self._calculate_spatial_continuity(parsing_map)
            
            return min(1.0, max(0.0, spatial_continuity))
            
        except Exception as e:
            self.logger.warning(f"공간적 연속성 평가 실패: {e}")
            return 0.5
    
    def _assess_confidence_reliability(self, parsing_result: Dict[str, Any]) -> float:
        """신뢰도 평가"""
        try:
            if 'confidence' not in parsing_result:
                return 0.5
            
            confidence = parsing_result['confidence']
            
            # 텐서를 numpy로 변환
            if torch.is_tensor(confidence):
                confidence = confidence.detach().cpu().numpy()
            
            # 신뢰도 신뢰성 계산
            confidence_reliability = self._calculate_confidence_reliability(confidence)
            
            return min(1.0, max(0.0, confidence_reliability))
            
        except Exception as e:
            self.logger.warning(f"신뢰도 평가 실패: {e}")
            return 0.5
    
    def _extract_boundaries(self, parsing_map: np.ndarray) -> np.ndarray:
        """파싱 맵에서 경계 추출"""
        try:
            # 각 클래스별로 경계 검출
            boundaries = np.zeros_like(parsing_map, dtype=np.uint8)
            
            for class_id in np.unique(parsing_map):
                if class_id == 0:  # 배경 제외
                    continue
                
                class_mask = (parsing_map == class_id).astype(np.uint8)
                class_boundaries = cv2.Canny(class_mask, 50, 150)
                boundaries = np.logical_or(boundaries, class_boundaries)
            
            return boundaries.astype(np.uint8)
            
        except Exception as e:
            self.logger.warning(f"경계 추출 실패: {e}")
            return np.zeros_like(parsing_map, dtype=np.uint8)
    
    def _calculate_boundary_consistency(self, boundaries: np.ndarray) -> float:
        """경계 일관성 계산"""
        try:
            if boundaries.sum() == 0:
                return 0.5
            
            # 경계의 연속성 계산
            kernel = np.ones((3, 3), np.uint8)
            dilated = cv2.dilate(boundaries, kernel, iterations=1)
            eroded = cv2.erode(boundaries, kernel, iterations=1)
            
            # 경계의 일관성 점수
            consistency_score = np.sum(boundaries) / np.sum(dilated)
            
            return float(consistency_score)
            
        except Exception as e:
            self.logger.warning(f"경계 일관성 계산 실패: {e}")
            return 0.5
    
    def _calculate_semantic_coherence(self, parsing_map: np.ndarray) -> float:
        """의미적 일관성 계산"""
        try:
            # 각 클래스의 연결성 계산
            coherence_scores = []
            
            for class_id in np.unique(parsing_map):
                if class_id == 0:  # 배경 제외
                    continue
                
                class_mask = (parsing_map == class_id).astype(np.uint8)
                
                # 연결된 컴포넌트 수 계산
                num_components, _ = cv2.connectedComponents(class_mask)
                
                # 단일 컴포넌트일수록 일관성 높음
                if num_components == 1:
                    coherence_scores.append(1.0)
                else:
                    coherence_scores.append(1.0 / num_components)
            
            if not coherence_scores:
                return 0.5
            
            return np.mean(coherence_scores)
            
        except Exception as e:
            self.logger.warning(f"의미적 일관성 계산 실패: {e}")
            return 0.5
    
    def _calculate_spatial_continuity(self, parsing_map: np.ndarray) -> float:
        """공간적 연속성 계산"""
        try:
            # 공간적 연속성 계산 (모폴로지 연산 사용)
            kernel = np.ones((5, 5), np.uint8)
            
            continuity_scores = []
            
            for class_id in np.unique(parsing_map):
                if class_id == 0:  # 배경 제외
                    continue
                
                class_mask = (parsing_map == class_id).astype(np.uint8)
                
                # 모폴로지 연산으로 연속성 평가
                opened = cv2.morphologyEx(class_mask, cv2.MORPH_OPEN, kernel)
                closed = cv2.morphologyEx(class_mask, cv2.MORPH_CLOSE, kernel)
                
                # 원본과의 유사도 계산
                similarity = np.sum(opened == class_mask) / class_mask.size
                continuity_scores.append(similarity)
            
            if not continuity_scores:
                return 0.5
            
            return np.mean(continuity_scores)
            
        except Exception as e:
            self.logger.warning(f"공간적 연속성 계산 실패: {e}")
            return 0.5
    
    def _calculate_confidence_reliability(self, confidence: np.ndarray) -> float:
        """신뢰도 신뢰성 계산"""
        try:
            if confidence.size == 0:
                return 0.5
            
            # 신뢰도 분포의 일관성 계산
            confidence_std = np.std(confidence)
            confidence_mean = np.mean(confidence)
            
            # 표준편차가 작을수록 신뢰도가 일관적
            if confidence_mean == 0:
                return 0.5
            
            cv_score = confidence_std / confidence_mean  # 변동계수
            reliability = max(0, 1 - cv_score)
            
            return min(1.0, max(0.0, reliability))
            
        except Exception as e:
            self.logger.warning(f"신뢰도 신뢰성 계산 실패: {e}")
            return 0.5
    
    def _generate_improvement_suggestions(self, quality_scores: Dict[str, float]) -> List[str]:
        """품질 개선 제안 생성"""
        suggestions = []
        
        if quality_scores['boundary_consistency'] < 0.7:
            suggestions.append("경계 일관성 개선: 경계 검출 알고리즘 강화 필요")
        
        if quality_scores['semantic_coherence'] < 0.7:
            suggestions.append("의미적 일관성 개선: 클래스별 연결성 강화 필요")
        
        if quality_scores['spatial_continuity'] < 0.7:
            suggestions.append("공간적 연속성 개선: 모폴로지 후처리 강화 필요")
        
        if quality_scores['confidence_reliability'] < 0.7:
            suggestions.append("신뢰도 신뢰성 개선: 신뢰도 계산 방식 개선 필요")
        
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
