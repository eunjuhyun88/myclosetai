#!/usr/bin/env python3
"""
🔥 MyCloset AI - Step 03: 의류 세그멘테이션 - Ensemble Core
=====================================================================

앙상블 핵심 기능

Author: MyCloset AI Team  
Date: 2025-08-01
Version: 1.0
"""

import logging
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
import cv2
import time

logger = logging.getLogger(__name__)

class EnsembleCore:
    """앙상블 핵심 기능"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """초기화"""
        self.config = config or {}
        self.logger = logging.getLogger(f"{__name__}.EnsembleCore")
        
    def run_hybrid_ensemble(self, image: np.ndarray, models: Dict[str, Any]) -> Dict[str, Any]:
        """하이브리드 앙상블 실행"""
        try:
            results = []
            methods_used = []
            execution_times = []
            
            # 각 모델로 세그멘테이션 실행
            for model_name, model in models.items():
                if model is not None:
                    start_time = time.time()
                    result = self._run_single_model(model, image, model_name)
                    execution_time = time.time() - start_time
                    
                    if result.get('success', False):
                        results.append(result)
                        methods_used.append(model_name)
                        execution_times.append(execution_time)
            
            # 결과 결합
            if results:
                combined_result = self._combine_ensemble_results(
                    results, methods_used, execution_times, image
                )
                return combined_result
            else:
                return self._create_fallback_result(image.shape)
                
        except Exception as e:
            self.logger.error(f"❌ 하이브리드 앙상블 실패: {e}")
            return self._create_fallback_result(image.shape)
    
    def _run_single_model(self, model: Any, image: np.ndarray, model_name: str) -> Dict[str, Any]:
        """단일 모델 실행"""
        try:
            if hasattr(model, 'predict'):
                return model.predict(image)
            else:
                return {'success': False, 'error': f'모델 {model_name}에 predict 메서드가 없습니다'}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _combine_ensemble_results(self, results: List[Dict[str, Any]], 
                                methods_used: List[str], 
                                execution_times: List[float],
                                image: np.ndarray) -> Dict[str, Any]:
        """앙상블 결과 결합"""
        try:
            if not results:
                return self._create_fallback_result(image.shape)
            
            # 가중 평균으로 마스크 결합
            combined_masks = {}
            total_weights = 0
            
            for i, result in enumerate(results):
                if result.get('success', False) and 'masks' in result:
                    weight = self._calculate_model_weight(
                        result.get('confidence', 0.5),
                        execution_times[i],
                        methods_used[i]
                    )
                    
                    for mask_key, mask in result['masks'].items():
                        if mask_key not in combined_masks:
                            combined_masks[mask_key] = np.zeros_like(mask, dtype=np.float32)
                        
                        combined_masks[mask_key] += mask.astype(np.float32) * weight
                        total_weights += weight
            
            # 정규화
            if total_weights > 0:
                for mask_key in combined_masks:
                    combined_masks[mask_key] = (combined_masks[mask_key] / total_weights).astype(np.uint8)
            
            # 최종 결과 생성
            final_result = {
                'success': True,
                'masks': combined_masks,
                'confidence': np.mean([r.get('confidence', 0.5) for r in results]),
                'method': 'hybrid_ensemble',
                'methods_used': methods_used,
                'execution_times': execution_times
            }
            
            return final_result
            
        except Exception as e:
            self.logger.error(f"❌ 앙상블 결과 결합 실패: {e}")
            return self._create_fallback_result(image.shape)
    
    def _calculate_model_weight(self, confidence: float, execution_time: float, method: str) -> float:
        """모델 가중치 계산"""
        try:
            # 신뢰도 가중치 (0.3)
            confidence_weight = confidence * 0.3
            
            # 실행 시간 가중치 (0.2) - 빠를수록 높은 가중치
            time_weight = max(0.1, 1.0 / (execution_time + 0.1)) * 0.2
            
            # 모델 타입 가중치 (0.3)
            method_weights = {
                'u2net': 0.3,
                'sam': 0.25,
                'deeplabv3': 0.2,
                'fallback': 0.1
            }
            method_weight = method_weights.get(method.lower(), 0.1) * 0.3
            
            # 마스크 품질 가중치 (0.2)
            quality_weight = 0.2
            
            total_weight = confidence_weight + time_weight + method_weight + quality_weight
            return total_weight
            
        except Exception as e:
            self.logger.error(f"❌ 가중치 계산 실패: {e}")
            return 0.1
    
    def _create_fallback_result(self, image_shape: Tuple[int, ...]) -> Dict[str, Any]:
        """폴백 결과 생성"""
        return {
            'success': False,
            'masks': {'all_clothes': np.zeros(image_shape[:2], dtype=np.uint8)},
            'confidence': 0.0,
            'method': 'fallback',
            'error': '앙상블 실패'
        }
