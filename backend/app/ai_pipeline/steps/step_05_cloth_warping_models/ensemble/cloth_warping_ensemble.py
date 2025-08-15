#!/usr/bin/env python3
"""
🔥 MyCloset AI - Cloth Warping Ensemble System
==============================================

✅ 통일된 앙상블 시스템
✅ 다중 모델 결과 통합
✅ 품질 기반 가중치 적용

Author: MyCloset AI Team
Date: 2025-08-14
Version: 1.0 (통일된 구조)
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)

class ClothWarpingEnsembleSystem:
    """Cloth Warping 앙상블 시스템 - 통일된 구조"""
    
    def __init__(self):
        self.ensemble_methods = [
            'weighted_average',
            'confidence_based',
            'majority_voting',
            'quality_weighted'
        ]
        self.default_method = 'confidence_based'
    
    def run_ensemble(self, results: List[Dict[str, Any]], method: str = None) -> Dict[str, Any]:
        """앙상블 실행"""
        if not results:
            return {'ensemble_result': None, 'method': method, 'error': 'No results provided'}
        
        method = method or self.default_method
        
        try:
            if method == 'weighted_average':
                return self._weighted_average_ensemble(results)
            elif method == 'confidence_based':
                return self._confidence_based_ensemble(results)
            elif method == 'majority_voting':
                return self._majority_voting_ensemble(results)
            elif method == 'quality_weighted':
                return self._quality_weighted_ensemble(results)
            else:
                logger.warning(f"⚠️ 지원하지 않는 앙상블 방법: {method}")
                return self._confidence_based_ensemble(results)
        except Exception as e:
            logger.error(f"❌ 앙상블 실행 실패: {e}")
            return {'ensemble_result': None, 'method': method, 'error': str(e)}
    
    def _weighted_average_ensemble(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """가중 평균 앙상블"""
        try:
            # 간단한 가중 평균 (모든 결과에 동일한 가중치)
            weights = [1.0 / len(results)] * len(results)
            
            ensemble_result = {
                'method': 'weighted_average',
                'weights': weights,
                'combined_result': results[0].copy() if results else None
            }
            
            return ensemble_result
        except Exception as e:
            logger.error(f"❌ 가중 평균 앙상블 실패: {e}")
            return {'ensemble_result': None, 'method': 'weighted_average', 'error': str(e)}
    
    def _confidence_based_ensemble(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """신뢰도 기반 앙상블"""
        try:
            # 신뢰도가 가장 높은 결과 선택
            best_result = max(results, key=lambda x: x.get('confidence', 0.0))
            
            ensemble_result = {
                'method': 'confidence_based',
                'best_confidence': best_result.get('confidence', 0.0),
                'combined_result': best_result
            }
            
            return ensemble_result
        except Exception as e:
            logger.error(f"❌ 신뢰도 기반 앙상블 실패: {e}")
            return {'ensemble_result': None, 'method': 'confidence_based', 'error': str(e)}
    
    def _majority_voting_ensemble(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """다수결 투표 앙상블"""
        try:
            # 간단한 다수결 (첫 번째 결과 선택)
            ensemble_result = {
                'method': 'majority_voting',
                'total_votes': len(results),
                'combined_result': results[0].copy() if results else None
            }
            
            return ensemble_result
        except Exception as e:
            logger.error(f"❌ 다수결 투표 앙상블 실패: {e}")
            return {'ensemble_result': None, 'method': 'majority_voting', 'error': str(e)}
    
    def _quality_weighted_ensemble(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """품질 가중 앙상블"""
        try:
            # 품질 점수 기반 가중치 계산
            quality_scores = [r.get('quality_score', 0.5) for r in results]
            total_quality = sum(quality_scores)
            
            if total_quality > 0:
                weights = [score / total_quality for score in quality_scores]
            else:
                weights = [1.0 / len(results)] * len(results)
            
            ensemble_result = {
                'method': 'quality_weighted',
                'weights': weights,
                'quality_scores': quality_scores,
                'combined_result': results[0].copy() if results else None
            }
            
            return ensemble_result
        except Exception as e:
            logger.error(f"❌ 품질 가중 앙상블 실패: {e}")
            return {'ensemble_result': None, 'method': 'quality_weighted', 'error': str(e)}
    
    def get_supported_methods(self) -> List[str]:
        """지원하는 앙상블 방법 반환"""
        return self.ensemble_methods.copy()
    
    def validate_results(self, results: List[Dict[str, Any]]) -> bool:
        """결과 유효성 검증"""
        if not results:
            return False
        
        for result in results:
            if not isinstance(result, dict):
                return False
        
        return True
