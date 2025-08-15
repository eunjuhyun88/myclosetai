#!/usr/bin/env python3
"""
🔥 MyCloset AI - Step 03: Cloth Segmentation Quality Assessment
===============================================================

🎯 의류 분할 결과 품질 평가
✅ 세그멘테이션 정확도 평가
✅ 경계 품질 평가
✅ 일관성 검사
✅ 종합 품질 점수 계산
"""

import logging
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass
import numpy as np

# PyTorch import 시도
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

logger = logging.getLogger(__name__)

@dataclass
class QualityAssessmentConfig:
    """품질 평가 설정"""
    accuracy_threshold: float = 0.8
    boundary_threshold: float = 0.7
    consistency_threshold: float = 0.6
    use_advanced_metrics: bool = True
    quality_weights: Dict[str, float] = None
    
    def __post_init__(self):
        if self.quality_weights is None:
            self.quality_weights = {
                'accuracy': 0.4,
                'boundary': 0.3,
                'consistency': 0.2,
                'completeness': 0.1
            }

class ClothSegmentationQualityAssessment:
    """
    🔥 의류 분할 품질 평가 시스템
    
    의류 분할 결과의 품질을 종합적으로 평가합니다.
    """
    
    def __init__(self, config: QualityAssessmentConfig = None):
        self.config = config or QualityAssessmentConfig()
        self.logger = logging.getLogger(__name__)
        
        # 품질 메트릭 히스토리
        self.quality_history = []
        self.max_history_size = 10
        
        self.logger.info("🎯 의류 분할 품질 평가 시스템 초기화 완료")
    
    def assess_segmentation_quality(self, 
                                  segmentation_mask: Union[torch.Tensor, np.ndarray],
                                  ground_truth: Optional[Union[torch.Tensor, np.ndarray]] = None,
                                  cloth_categories: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        세그멘테이션 품질 평가
        
        Args:
            segmentation_mask: 분할 마스크
            ground_truth: 정답 마스크 (있는 경우)
            cloth_categories: 의류 카테고리 리스트
        
        Returns:
            품질 평가 결과
        """
        try:
            # numpy로 변환
            if TORCH_AVAILABLE and isinstance(segmentation_mask, torch.Tensor):
                mask_np = segmentation_mask.detach().cpu().numpy()
            else:
                mask_np = np.array(segmentation_mask)
            
            # 1단계: 정확도 평가
            accuracy_score = self._assess_accuracy(mask_np, ground_truth)
            
            # 2단계: 경계 품질 평가
            boundary_score = self._assess_boundary_quality(mask_np)
            
            # 3단계: 일관성 평가
            consistency_score = self._assess_consistency(mask_np)
            
            # 4단계: 완성도 평가
            completeness_score = self._assess_completeness(mask_np, cloth_categories)
            
            # 5단계: 종합 품질 점수 계산
            overall_score = self._calculate_overall_quality(
                accuracy_score, boundary_score, consistency_score, completeness_score
            )
            
            # 결과 정리
            result = {
                'overall_quality': overall_score,
                'accuracy_score': accuracy_score,
                'boundary_score': boundary_score,
                'consistency_score': consistency_score,
                'completeness_score': completeness_score,
                'quality_level': self._get_quality_level(overall_score),
                'recommendations': self._generate_recommendations(
                    accuracy_score, boundary_score, consistency_score, completeness_score
                )
            }
            
            # 품질 히스토리 업데이트
            self._update_quality_history(result)
            
            self.logger.info(f"✅ 의류 분할 품질 평가 완료 (종합 점수: {overall_score:.3f})")
            return result
            
        except Exception as e:
            self.logger.error(f"❌ 품질 평가 실패: {e}")
            return {
                'overall_quality': 0.0,
                'accuracy_score': 0.0,
                'boundary_score': 0.0,
                'consistency_score': 0.0,
                'completeness_score': 0.0,
                'quality_level': 'poor',
                'error': str(e)
            }
    
    def _assess_accuracy(self, mask: np.ndarray, ground_truth: Optional[np.ndarray] = None) -> float:
        """정확도 평가"""
        try:
            if ground_truth is not None:
                # 정답과 비교
                if TORCH_AVAILABLE and isinstance(ground_truth, torch.Tensor):
                    gt_np = ground_truth.detach().cpu().numpy()
                else:
                    gt_np = np.array(ground_truth)
                
                # IoU 계산
                intersection = np.logical_and(mask > 0, gt_np > 0).sum()
                union = np.logical_or(mask > 0, gt_np > 0).sum()
                
                if union > 0:
                    iou = intersection / union
                    return float(iou)
                else:
                    return 0.0
            else:
                # 정답이 없는 경우 기본 정확도
                return 0.7
                
        except Exception as e:
            self.logger.warning(f"정확도 평가 실패: {e}")
            return 0.5
    
    def _assess_boundary_quality(self, mask: np.ndarray) -> float:
        """경계 품질 평가"""
        try:
            # 경계 픽셀 찾기
            from scipy import ndimage
            
            # 경계 검출
            boundary = ndimage.binary_erosion(mask > 0) != (mask > 0)
            
            if boundary.sum() == 0:
                return 0.5
            
            # 경계의 연속성 평가
            boundary_components = ndimage.label(boundary)[0]
            continuity_score = 1.0 / (1.0 + boundary_components)
            
            # 경계의 부드러움 평가
            smoothness_score = self._calculate_boundary_smoothness(mask)
            
            # 종합 경계 점수
            boundary_score = (continuity_score + smoothness_score) / 2.0
            
            return float(np.clip(boundary_score, 0.0, 1.0))
            
        except Exception as e:
            self.logger.warning(f"경계 품질 평가 실패: {e}")
            return 0.5
    
    def _calculate_boundary_smoothness(self, mask: np.ndarray) -> float:
        """경계 부드러움 계산"""
        try:
            # 간단한 경계 부드러움 계산
            # 경계 주변의 픽셀 변화량을 측정
            
            # Sobel 필터로 경계 강도 계산
            from scipy import ndimage
            
            sobel_x = ndimage.sobel(mask, axis=1)
            sobel_y = ndimage.sobel(mask, axis=0)
            gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
            
            # 경계 강도의 표준편차 (낮을수록 부드러움)
            boundary_strength = gradient_magnitude[gradient_magnitude > 0]
            
            if len(boundary_strength) == 0:
                return 0.5
            
            # 정규화된 부드러움 점수
            smoothness = 1.0 / (1.0 + np.std(boundary_strength))
            
            return float(np.clip(smoothness, 0.0, 1.0))
            
        except Exception:
            return 0.5
    
    def _assess_consistency(self, mask: np.ndarray) -> float:
        """일관성 평가"""
        try:
            # 마스크의 일관성 평가
            # 1. 연결성 검사
            from scipy import ndimage
            
            labeled_mask, num_components = ndimage.label(mask > 0)
            
            if num_components == 0:
                return 0.0
            
            # 2. 컴포넌트 크기 일관성
            component_sizes = [np.sum(labeled_mask == i) for i in range(1, num_components + 1)]
            
            if len(component_sizes) == 1:
                consistency_score = 1.0
            else:
                # 크기 분포의 표준편차 (낮을수록 일관성 높음)
                size_std = np.std(component_sizes)
                size_mean = np.mean(component_sizes)
                
                if size_mean > 0:
                    cv = size_std / size_mean  # 변동계수
                    consistency_score = 1.0 / (1.0 + cv)
                else:
                    consistency_score = 0.0
            
            # 3. 공간적 일관성
            spatial_consistency = self._assess_spatial_consistency(mask)
            
            # 종합 일관성 점수
            final_consistency = (consistency_score + spatial_consistency) / 2.0
            
            return float(np.clip(final_consistency, 0.0, 1.0))
            
        except Exception as e:
            self.logger.warning(f"일관성 평가 실패: {e}")
            return 0.5
    
    def _assess_spatial_consistency(self, mask: np.ndarray) -> float:
        """공간적 일관성 평가"""
        try:
            # 간단한 공간적 일관성 계산
            # 마스크의 형태적 특성 평가
            
            # 원형도 (circularity) 계산
            if mask.sum() > 0:
                # 경계 길이 계산
                from scipy import ndimage
                boundary = ndimage.binary_erosion(mask > 0) != (mask > 0)
                perimeter = boundary.sum()
                
                if perimeter > 0:
                    area = mask.sum()
                    circularity = 4 * np.pi * area / (perimeter ** 2)
                    # 정규화된 원형도 점수
                    circularity_score = np.clip(circularity, 0.0, 1.0)
                else:
                    circularity_score = 0.5
            else:
                circularity_score = 0.0
            
            return float(circularity_score)
            
        except Exception:
            return 0.5
    
    def _assess_completeness(self, mask: np.ndarray, cloth_categories: Optional[List[str]] = None) -> float:
        """완성도 평가"""
        try:
            # 마스크의 완성도 평가
            # 1. 기본 완성도 (0이 아닌 픽셀 비율)
            total_pixels = mask.size
            non_zero_pixels = np.sum(mask > 0)
            
            if total_pixels > 0:
                coverage_ratio = non_zero_pixels / total_pixels
            else:
                coverage_ratio = 0.0
            
            # 2. 카테고리별 완성도 (있는 경우)
            category_completeness = 1.0
            if cloth_categories and len(cloth_categories) > 0:
                # 간단한 카테고리 완성도 계산
                expected_categories = len(cloth_categories)
                detected_categories = len(np.unique(mask[mask > 0]))
                
                if expected_categories > 0:
                    category_completeness = min(detected_categories / expected_categories, 1.0)
            
            # 종합 완성도 점수
            completeness_score = (coverage_ratio + category_completeness) / 2.0
            
            return float(np.clip(completeness_score, 0.0, 1.0))
            
        except Exception as e:
            self.logger.warning(f"완성도 평가 실패: {e}")
            return 0.5
    
    def _calculate_overall_quality(self, accuracy: float, boundary: float, 
                                 consistency: float, completeness: float) -> float:
        """종합 품질 점수 계산"""
        try:
            weights = self.config.quality_weights
            
            overall_score = (
                weights['accuracy'] * accuracy +
                weights['boundary'] * boundary +
                weights['consistency'] * consistency +
                weights['completeness'] * completeness
            )
            
            return float(np.clip(overall_score, 0.0, 1.0))
            
        except Exception as e:
            self.logger.warning(f"종합 품질 점수 계산 실패: {e}")
            # 기본 가중 평균
            return (accuracy + boundary + consistency + completeness) / 4.0
    
    def _get_quality_level(self, score: float) -> str:
        """품질 수준 결정"""
        if score >= 0.9:
            return "excellent"
        elif score >= 0.8:
            return "very_good"
        elif score >= 0.7:
            return "good"
        elif score >= 0.6:
            return "fair"
        elif score >= 0.5:
            return "poor"
        else:
            return "very_poor"
    
    def _generate_recommendations(self, accuracy: float, boundary: float, 
                                consistency: float, completeness: float) -> List[str]:
        """개선 권장사항 생성"""
        recommendations = []
        
        if accuracy < self.config.accuracy_threshold:
            recommendations.append("세그멘테이션 정확도 향상 필요")
        
        if boundary < self.config.boundary_threshold:
            recommendations.append("경계 품질 개선 필요")
        
        if consistency < self.config.consistency_threshold:
            recommendations.append("일관성 향상 필요")
        
        if completeness < 0.6:
            recommendations.append("완성도 개선 필요")
        
        if not recommendations:
            recommendations.append("현재 품질이 양호합니다")
        
        return recommendations
    
    def _update_quality_history(self, result: Dict[str, Any]):
        """품질 히스토리 업데이트"""
        self.quality_history.append(result)
        
        # 히스토리 크기 제한
        if len(self.quality_history) > self.max_history_size:
            self.quality_history.pop(0)
    
    def get_quality_trend(self) -> Dict[str, Any]:
        """품질 트렌드 분석"""
        try:
            if len(self.quality_history) < 2:
                return {"trend": "insufficient_data", "change": 0.0}
            
            recent_scores = [result['overall_quality'] for result in self.quality_history]
            
            # 트렌드 계산
            if len(recent_scores) >= 2:
                trend = "improving" if recent_scores[-1] > recent_scores[0] else "declining"
                change = recent_scores[-1] - recent_scores[0]
            else:
                trend = "stable"
                change = 0.0
            
            return {
                "trend": trend,
                "change": float(change),
                "history_length": len(self.quality_history),
                "average_score": float(np.mean(recent_scores)),
                "best_score": float(np.max(recent_scores)),
                "worst_score": float(np.min(recent_scores))
            }
            
        except Exception as e:
            self.logger.warning(f"품질 트렌드 분석 실패: {e}")
            return {"trend": "error", "change": 0.0}
    
    def reset_quality_history(self):
        """품질 히스토리 초기화"""
        self.quality_history.clear()
        self.logger.info("✅ 품질 히스토리 초기화 완료")
    
    def get_config(self) -> QualityAssessmentConfig:
        """현재 설정 반환"""
        return self.config
    
    def update_config(self, new_config: QualityAssessmentConfig):
        """설정 업데이트"""
        self.config = new_config
        self.logger.info("✅ 품질 평가 설정 업데이트 완료")

# 기본 품질 평가 시스템 생성 함수
def create_cloth_segmentation_quality_assessment(config: QualityAssessmentConfig = None) -> ClothSegmentationQualityAssessment:
    """의류 분할 품질 평가 시스템 생성"""
    return ClothSegmentationQualityAssessment(config)

if __name__ == "__main__":
    # 테스트 코드
    logging.basicConfig(level=logging.INFO)
    
    # 품질 평가 시스템 생성
    quality_assessor = create_cloth_segmentation_quality_assessment()
    
    # 테스트 데이터
    test_mask = np.random.randint(0, 2, (256, 256)).astype(np.float32)
    
    # 품질 평가 수행
    result = quality_assessor.assess_segmentation_quality(test_mask)
    
    print(f"품질 평가 결과: {result['overall_quality']:.3f}")
    print(f"품질 수준: {result['quality_level']}")
    print(f"권장사항: {result['recommendations']}")
