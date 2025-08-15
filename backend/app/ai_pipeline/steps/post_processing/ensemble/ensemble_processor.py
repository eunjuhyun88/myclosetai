"""
🔥 앙상블 프로세서
==================

후처리를 위한 앙상블 시스템:
1. 다중 메트릭 통합
2. 품질 점수 앙상블
3. 신뢰도 기반 가중치
4. 앙상블 최적화

Author: MyCloset AI Team
Date: 2025-08-14
Version: 1.0
"""

import logging
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class EnsembleMethod(ABC):
    """앙상블 방법 기본 클래스"""
    
    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    @abstractmethod
    def combine(self, metrics: Dict[str, float], weights: Optional[Dict[str, float]] = None) -> float:
        """메트릭 결합"""
        pass
    
    def get_method_info(self) -> Dict[str, Any]:
        """메서드 정보 반환"""
        return {
            'name': self.name,
            'description': self.__doc__ or f"{self.name} 앙상블 방법",
            'type': self.__class__.__name__
        }

class SimpleAverageEnsemble(EnsembleMethod):
    """단순 평균 앙상블"""
    
    def __init__(self):
        super().__init__("SimpleAverage")
    
    def combine(self, metrics: Dict[str, float], weights: Optional[Dict[str, float]] = None) -> float:
        """단순 평균으로 메트릭 결합"""
        try:
            if not metrics:
                return 0.0
            
            values = list(metrics.values())
            return float(np.mean(values))
            
        except Exception as e:
            self.logger.error(f"❌ 단순 평균 앙상블 실패: {e}")
            return 0.0

class WeightedAverageEnsemble(EnsembleMethod):
    """가중 평균 앙상블"""
    
    def __init__(self):
        super().__init__("WeightedAverage")
    
    def combine(self, metrics: Dict[str, float], weights: Optional[Dict[str, float]] = None) -> float:
        """가중 평균으로 메트릭 결합"""
        try:
            if not metrics:
                return 0.0
            
            if weights is None:
                # 균등 가중치
                weights = {key: 1.0 for key in metrics.keys()}
            
            # 가중치 정규화
            total_weight = sum(weights.values())
            if total_weight == 0:
                return 0.0
            
            normalized_weights = {key: weight / total_weight for key, weight in weights.items()}
            
            # 가중 평균 계산
            weighted_sum = sum(metrics[key] * normalized_weights[key] for key in metrics.keys())
            
            return float(weighted_sum)
            
        except Exception as e:
            self.logger.error(f"❌ 가중 평균 앙상블 실패: {e}")
            return 0.0

class QualityBasedEnsemble(EnsembleMethod):
    """품질 기반 앙상블"""
    
    def __init__(self):
        super().__init__("QualityBased")
    
    def combine(self, metrics: Dict[str, float], weights: Optional[Dict[str, float]] = None) -> float:
        """품질 기반으로 메트릭 결합"""
        try:
            if not metrics:
                return 0.0
            
            # 각 메트릭의 품질 점수 계산
            quality_scores = {}
            
            for metric_name, value in metrics.items():
                quality_score = self._calculate_quality_score(metric_name, value)
                quality_scores[metric_name] = quality_score
            
            # 품질 점수를 가중치로 사용
            total_quality = sum(quality_scores.values())
            if total_quality == 0:
                return 0.0
            
            normalized_weights = {key: score / total_quality for key, score in quality_scores.items()}
            
            # 품질 기반 가중 평균 계산
            weighted_sum = sum(metrics[key] * normalized_weights[key] for key in metrics.keys())
            
            return float(weighted_sum)
            
        except Exception as e:
            self.logger.error(f"❌ 품질 기반 앙상블 실패: {e}")
            return 0.0
    
    def _calculate_quality_score(self, metric_name: str, value: float) -> float:
        """메트릭별 품질 점수 계산"""
        try:
            if metric_name == 'psnr':
                # PSNR: 높을수록 좋음 (20dB 이상이 좋음)
                return max(0.0, min(1.0, value / 40.0))
            elif metric_name == 'ssim':
                # SSIM: 0~1 범위, 높을수록 좋음
                return max(0.0, min(1.0, value))
            elif metric_name == 'contrast':
                # 대비: 1.0 근처가 좋음
                if value < 0.5 or value > 2.0:
                    return 0.0
                elif value < 0.8 or value > 1.5:
                    return 0.5
                else:
                    return 1.0
            elif metric_name == 'sharpness':
                # 선명도: 1.0 근처가 좋음
                if value < 0.5 or value > 2.0:
                    return 0.0
                elif value < 0.8 or value > 1.5:
                    return 0.5
                else:
                    return 1.0
            elif metric_name == 'color_balance':
                # 색상 균형: 1.0 근처가 좋음
                if value < 0.8 or value > 1.3:
                    return 0.0
                elif value < 0.9 or value > 1.2:
                    return 0.5
                else:
                    return 1.0
            else:
                # 알 수 없는 메트릭은 기본값
                return 0.5
                
        except Exception as e:
            self.logger.error(f"❌ 품질 점수 계산 실패: {metric_name} - {e}")
            return 0.5

class ConfidenceBasedEnsemble(EnsembleMethod):
    """신뢰도 기반 앙상블"""
    
    def __init__(self):
        super().__init__("ConfidenceBased")
    
    def combine(self, metrics: Dict[str, float], weights: Optional[Dict[str, float]] = None) -> float:
        """신뢰도 기반으로 메트릭 결합"""
        try:
            if not metrics:
                return 0.0
            
            # 각 메트릭의 신뢰도 계산
            confidence_scores = {}
            
            for metric_name, value in metrics.items():
                confidence = self._calculate_confidence(metric_name, value)
                confidence_scores[metric_name] = confidence
            
            # 신뢰도를 가중치로 사용
            total_confidence = sum(confidence_scores.values())
            if total_confidence == 0:
                return 0.0
            
            normalized_weights = {key: confidence / total_confidence for key, confidence in confidence_scores.items()}
            
            # 신뢰도 기반 가중 평균 계산
            weighted_sum = sum(metrics[key] * normalized_weights[key] for key in metrics.keys())
            
            return float(weighted_sum)
            
        except Exception as e:
            self.logger.error(f"❌ 신뢰도 기반 앙상블 실패: {e}")
            return 0.0
    
    def _calculate_confidence(self, metric_name: str, value: float) -> float:
        """메트릭별 신뢰도 계산"""
        try:
            if metric_name == 'psnr':
                # PSNR: 30dB 이상이 높은 신뢰도
                if value >= 30:
                    return 1.0
                elif value >= 25:
                    return 0.8
                elif value >= 20:
                    return 0.6
                else:
                    return 0.3
            elif metric_name == 'ssim':
                # SSIM: 0.9 이상이 높은 신뢰도
                if value >= 0.9:
                    return 1.0
                elif value >= 0.8:
                    return 0.8
                elif value >= 0.7:
                    return 0.6
                else:
                    return 0.3
            elif metric_name in ['contrast', 'sharpness', 'color_balance']:
                # 개선도 메트릭: 1.0 근처가 높은 신뢰도
                if 0.9 <= value <= 1.1:
                    return 1.0
                elif 0.8 <= value <= 1.2:
                    return 0.8
                elif 0.7 <= value <= 1.3:
                    return 0.6
                else:
                    return 0.3
            else:
                # 알 수 없는 메트릭은 기본 신뢰도
                return 0.5
                
        except Exception as e:
            self.logger.error(f"❌ 신뢰도 계산 실패: {metric_name} - {e}")
            return 0.5

class EnsembleProcessor:
    """앙상블 프로세서"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.EnsembleProcessor")
        
        # 사용 가능한 앙상블 방법들
        self.ensemble_methods = {
            'simple_average': SimpleAverageEnsemble(),
            'weighted_average': WeightedAverageEnsemble(),
            'quality_based': QualityBasedEnsemble(),
            'confidence_based': ConfidenceBasedEnsemble()
        }
        
        # 앙상블 통계
        self.ensemble_stats = {
            'total_ensembles': 0,
            'successful_ensembles': 0,
            'failed_ensembles': 0,
            'method_usage': {}
        }
    
    def get_available_methods(self) -> List[str]:
        """사용 가능한 앙상블 방법 목록 반환"""
        return list(self.ensemble_methods.keys())
    
    def get_method_info(self, method_name: str) -> Optional[Dict[str, Any]]:
        """앙상블 방법 정보 반환"""
        try:
            if method_name not in self.ensemble_methods:
                return None
            
            return self.ensemble_methods[method_name].get_method_info()
            
        except Exception as e:
            self.logger.error(f"❌ 앙상블 방법 정보 조회 실패: {method_name} - {e}")
            return None
    
    def combine_metrics(self, 
                       metrics: Dict[str, float],
                       method: str = 'quality_based',
                       weights: Optional[Dict[str, float]] = None) -> Optional[float]:
        """메트릭 앙상블 결합"""
        try:
            if method not in self.ensemble_methods:
                self.logger.error(f"❌ 알 수 없는 앙상블 방법: {method}")
                return None
            
            self.logger.info(f"🚀 {method} 앙상블 시작")
            
            # 앙상블 실행
            ensemble_method = self.ensemble_methods[method]
            result = ensemble_method.combine(metrics, weights)
            
            # 통계 업데이트
            self._update_ensemble_stats(method, True)
            
            self.logger.info(f"✅ {method} 앙상블 완료: {result:.4f}")
            return result
            
        except Exception as e:
            self.logger.error(f"❌ {method} 앙상블 실패: {e}")
            self._update_ensemble_stats(method, False)
            return None
    
    def combine_with_multiple_methods(self, 
                                    metrics: Dict[str, float],
                                    methods: List[str] = None,
                                    weights: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """여러 방법으로 메트릭 결합"""
        try:
            if methods is None:
                methods = list(self.ensemble_methods.keys())
            
            self.logger.info(f"🚀 다중 방법 앙상블 시작: {methods}")
            
            results = {}
            
            for method in methods:
                if method in self.ensemble_methods:
                    result = self.combine_metrics(metrics, method, weights)
                    if result is not None:
                        results[method] = result
            
            # 최종 통합 결과 (품질 기반 방법 사용)
            if results:
                final_result = self.combine_metrics(results, 'quality_based')
                results['final_ensemble'] = final_result
            
            self.logger.info(f"✅ 다중 방법 앙상블 완료: {len(results)}개 방법")
            return results
            
        except Exception as e:
            self.logger.error(f"❌ 다중 방법 앙상블 실패: {e}")
            return {}
    
    def optimize_weights(self, 
                        metrics: Dict[str, float],
                        target_score: float = 1.0,
                        method: str = 'weighted_average') -> Dict[str, float]:
        """가중치 최적화"""
        try:
            self.logger.info(f"🚀 가중치 최적화 시작: {method}")
            
            if method != 'weighted_average':
                self.logger.warning(f"⚠️ 가중치 최적화는 weighted_average 방법에서만 지원됩니다.")
                return None
            
            # 간단한 가중치 최적화 (그리드 서치)
            best_weights = None
            best_score = float('inf')
            
            # 가중치 범위 설정 (0.1 ~ 2.0)
            weight_range = np.arange(0.1, 2.1, 0.1)
            
            for w1 in weight_range:
                for w2 in weight_range:
                    for w3 in weight_range:
                        for w4 in weight_range:
                            for w5 in weight_range:
                                # 가중치 정규화
                                weights = {
                                    'psnr': w1, 'ssim': w2, 'contrast': w3,
                                    'sharpness': w4, 'color_balance': w5
                                }
                                
                                # 앙상블 실행
                                result = self.combine_metrics(metrics, method, weights)
                                
                                if result is not None:
                                    # 목표 점수와의 차이 계산
                                    score_diff = abs(result - target_score)
                                    
                                    if score_diff < best_score:
                                        best_score = score_diff
                                        best_weights = weights.copy()
            
            if best_weights is not None:
                self.logger.info(f"✅ 가중치 최적화 완료: 최적 점수 차이 = {best_score:.4f}")
                return best_weights
            else:
                self.logger.warning("⚠️ 가중치 최적화 실패")
                return None
                
        except Exception as e:
            self.logger.error(f"❌ 가중치 최적화 실패: {e}")
            return None
    
    def add_custom_method(self, method_name: str, method: EnsembleMethod):
        """사용자 정의 앙상블 방법 추가"""
        try:
            if method_name in self.ensemble_methods:
                self.logger.warning(f"⚠️ 앙상블 방법 {method_name}이 이미 존재합니다. 덮어씁니다.")
            
            self.ensemble_methods[method_name] = method
            self.logger.info(f"✅ 사용자 정의 앙상블 방법 추가: {method_name}")
            
        except Exception as e:
            self.logger.error(f"❌ 사용자 정의 앙상블 방법 추가 실패: {method_name} - {e}")
    
    def remove_method(self, method_name: str) -> bool:
        """앙상블 방법 제거"""
        try:
            if method_name not in self.ensemble_methods:
                self.logger.warning(f"⚠️ 앙상블 방법 {method_name}이 존재하지 않습니다.")
                return False
            
            del self.ensemble_methods[method_name]
            self.logger.info(f"✅ 앙상블 방법 제거: {method_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ 앙상블 방법 제거 실패: {method_name} - {e}")
            return False
    
    def _update_ensemble_stats(self, method: str, success: bool):
        """앙상블 통계 업데이트"""
        try:
            self.ensemble_stats['total_ensembles'] += 1
            
            if success:
                self.ensemble_stats['successful_ensembles'] += 1
            else:
                self.ensemble_stats['failed_ensembles'] += 1
            
            # 방법별 사용 통계
            if method not in self.ensemble_stats['method_usage']:
                self.ensemble_stats['method_usage'][method] = {'total': 0, 'successful': 0, 'failed': 0}
            
            self.ensemble_stats['method_usage'][method]['total'] += 1
            if success:
                self.ensemble_stats['method_usage'][method]['successful'] += 1
            else:
                self.ensemble_stats['method_usage'][method]['failed'] += 1
                
        except Exception as e:
            self.logger.error(f"❌ 앙상블 통계 업데이트 실패: {e}")
    
    def get_ensemble_stats(self) -> Dict[str, Any]:
        """앙상블 통계 반환"""
        return self.ensemble_stats.copy()
    
    def reset_ensemble_stats(self):
        """앙상블 통계 초기화"""
        self.ensemble_stats = {
            'total_ensembles': 0,
            'successful_ensembles': 0,
            'failed_ensembles': 0,
            'method_usage': {}
        }
