#!/usr/bin/env python3
"""
🔥 Output Integration Utils - 출력 통합 유틸리티
================================================================================

✅ 출력 통합 함수
✅ 데이터 검증
✅ 품질 평가
✅ 메트릭 계산

Author: MyCloset AI Team
Date: 2025-08-13
Version: 1.0
"""

import logging
import traceback
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import numpy as np

logger = logging.getLogger(__name__)

class OutputIntegrationUtils:
    """출력 통합 유틸리티"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
    
    def validate_step_outputs(self, step_outputs: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """단계별 출력 검증"""
        try:
            errors = []
            
            if not step_outputs:
                errors.append("단계별 출력이 비어있습니다")
                return False, errors
            
            # 각 단계 결과 검증
            for step_name, step_result in step_outputs.items():
                if not isinstance(step_result, dict):
                    errors.append(f"{step_name}: 결과가 딕셔너리가 아닙니다")
                    continue
                
                # 필수 필드 검증
                required_fields = ['status', 'step_version']
                for field in required_fields:
                    if field not in step_result:
                        errors.append(f"{step_name}: {field} 필드가 없습니다")
                
                # 상태 검증
                if 'status' in step_result:
                    status = step_result['status']
                    if status not in ['success', 'error', 'partial']:
                        errors.append(f"{step_name}: 잘못된 상태값: {status}")
            
            return len(errors) == 0, errors
            
        except Exception as e:
            logger.error(f"출력 검증 실패: {e}")
            return False, [f"검증 오류: {e}"]
    
    def calculate_integration_metrics(self, step_outputs: Dict[str, Any]) -> Dict[str, Any]:
        """통합 메트릭 계산"""
        try:
            metrics = {
                'total_steps': len(step_outputs),
                'successful_steps': 0,
                'failed_steps': 0,
                'total_processing_time': 0.0,
                'average_quality_score': 0.0,
                'average_confidence_score': 0.0
            }
            
            quality_scores = []
            confidence_scores = []
            
            for step_result in step_outputs.values():
                if isinstance(step_result, dict):
                    # 상태별 카운트
                    status = step_result.get('status', 'unknown')
                    if status == 'success':
                        metrics['successful_steps'] += 1
                    elif status == 'error':
                        metrics['failed_steps'] += 1
                    
                    # 처리 시간 누적
                    processing_time = step_result.get('processing_time', 0.0)
                    metrics['total_processing_time'] += processing_time
                    
                    # 품질 점수 수집
                    if 'quality_score' in step_result:
                        quality_score = step_result['quality_score']
                        if isinstance(quality_score, (int, float)):
                            quality_scores.append(float(quality_score))
                    
                    # 신뢰도 점수 수집
                    if 'confidence_score' in step_result:
                        confidence_score = step_result['confidence_score']
                        if isinstance(confidence_score, (int, float)):
                            confidence_scores.append(float(confidence_score))
            
            # 평균 계산
            if quality_scores:
                metrics['average_quality_score'] = sum(quality_scores) / len(quality_scores)
            if confidence_scores:
                metrics['average_confidence_score'] = sum(confidence_scores) / len(confidence_scores)
            
            # 성공률 계산
            if metrics['total_steps'] > 0:
                metrics['success_rate'] = metrics['successful_steps'] / metrics['total_steps']
            else:
                metrics['success_rate'] = 0.0
            
            return metrics
            
        except Exception as e:
            logger.error(f"메트릭 계산 실패: {e}")
            return {'error': str(e)}
    
    def generate_integration_summary(self, step_outputs: Dict[str, Any], 
                                   metrics: Dict[str, Any]) -> Dict[str, Any]:
        """통합 요약 생성"""
        try:
            summary = {
                'integration_timestamp': datetime.now().isoformat(),
                'pipeline_status': self._determine_pipeline_status(metrics),
                'key_achievements': self._identify_achievements(metrics),
                'areas_for_improvement': self._identify_improvements(metrics),
                'recommendations': self._generate_recommendations(metrics),
                'next_steps': self._suggest_next_steps(metrics)
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"요약 생성 실패: {e}")
            return {'error': str(e)}
    
    def _determine_pipeline_status(self, metrics: Dict[str, Any]) -> str:
        """파이프라인 상태 결정"""
        try:
            success_rate = metrics.get('success_rate', 0.0)
            avg_quality = metrics.get('average_quality_score', 0.0)
            
            if success_rate >= 0.9 and avg_quality >= 0.8:
                return 'excellent'
            elif success_rate >= 0.8 and avg_quality >= 0.7:
                return 'good'
            elif success_rate >= 0.7 and avg_quality >= 0.6:
                return 'acceptable'
            elif success_rate >= 0.6:
                return 'needs_improvement'
            else:
                return 'critical_issues'
                
        except Exception as e:
            logger.warning(f"상태 결정 실패: {e}")
            return 'unknown'
    
    def _identify_achievements(self, metrics: Dict[str, Any]) -> List[str]:
        """성과 식별"""
        achievements = []
        
        try:
            success_rate = metrics.get('success_rate', 0.0)
            avg_quality = metrics.get('average_quality_score', 0.0)
            total_steps = metrics.get('total_steps', 0)
            
            if success_rate >= 0.9:
                achievements.append("높은 성공률 달성")
            if success_rate == 1.0:
                achievements.append("모든 단계 완벽 실행")
            if avg_quality >= 0.8:
                achievements.append("높은 품질 달성")
            if total_steps >= 5:
                achievements.append("복잡한 파이프라인 성공적 처리")
                
        except Exception as e:
            logger.warning(f"성과 식별 실패: {e}")
        
        return achievements
    
    def _identify_improvements(self, metrics: Dict[str, Any]) -> List[str]:
        """개선 영역 식별"""
        improvements = []
        
        try:
            success_rate = metrics.get('success_rate', 0.0)
            avg_quality = metrics.get('average_quality_score', 0.0)
            failed_steps = metrics.get('failed_steps', 0)
            
            if success_rate < 0.8:
                improvements.append("성공률 개선 필요")
            if avg_quality < 0.7:
                improvements.append("품질 향상 필요")
            if failed_steps > 0:
                improvements.append("실패한 단계 재처리 필요")
                
        except Exception as e:
            logger.warning(f"개선 영역 식별 실패: {e}")
        
        return improvements
    
    def _generate_recommendations(self, metrics: Dict[str, Any]) -> List[str]:
        """권장사항 생성"""
        recommendations = []
        
        try:
            success_rate = metrics.get('success_rate', 0.0)
            avg_quality = metrics.get('average_quality_score', 0.0)
            
            if success_rate < 0.7:
                recommendations.append("전체 파이프라인 재실행 고려")
            if avg_quality < 0.6:
                recommendations.append("품질이 낮은 단계 재처리")
            if success_rate >= 0.9 and avg_quality >= 0.8:
                recommendations.append("현재 설정 유지 권장")
                
        except Exception as e:
            logger.warning(f"권장사항 생성 실패: {e}")
        
        return recommendations
    
    def _suggest_next_steps(self, metrics: Dict[str, Any]) -> List[str]:
        """다음 단계 제안"""
        next_steps = []
        
        try:
            success_rate = metrics.get('success_rate', 0.0)
            
            if success_rate >= 0.8:
                next_steps.append("결과 검증 및 품질 확인")
                next_steps.append("최종 출력 생성")
            else:
                next_steps.append("실패한 단계 디버깅")
                next_steps.append("파이프라인 재실행")
                
        except Exception as e:
            logger.warning(f"다음 단계 제안 실패: {e}")
        
        return next_steps
    
    def merge_step_data(self, step_outputs: Dict[str, Any]) -> Dict[str, Any]:
        """단계별 데이터 병합"""
        try:
            merged_data = {
                'ai_results': {},
                'traditional_metrics': {},
                'additional_data': {}
            }
            
            for step_name, step_result in step_outputs.items():
                if isinstance(step_result, dict) and step_result.get('status') == 'success':
                    # AI 결과 추출
                    if 'ai_quality_assessment' in step_result:
                        merged_data['ai_results'][step_name] = step_result['ai_quality_assessment']
                    
                    # 전통적 메트릭 추출
                    if 'traditional_metrics' in step_result:
                        merged_data['traditional_metrics'][step_name] = step_result['traditional_metrics']
                    
                    # 기타 데이터 추출
                    for key, value in step_result.items():
                        if key not in ['status', 'step_version', 'processing_time', 'device_used']:
                            merged_data['additional_data'][step_name] = {key: value}
            
            return merged_data
            
        except Exception as e:
            logger.error(f"데이터 병합 실패: {e}")
            return {'error': str(e)}
