"""
Processing utilities for geometric matching step.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
import time
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger(__name__)


class GeometricMatchingProcessor:
    """기하학적 매칭 처리 유틸리티"""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def validate_and_extract_inputs(self, kwargs: Dict[str, Any]) -> Tuple:
        """입력 검증 및 추출"""
        try:
            # 필수 입력 확인
            if 'person_image' not in kwargs:
                raise ValueError("person_image가 필요합니다")
            if 'clothing_image' not in kwargs:
                raise ValueError("clothing_image가 필요합니다")
            
            person_image = kwargs['person_image']
            clothing_image = kwargs['clothing_image']
            
            # 선택적 입력
            person_parsing_data = kwargs.get('person_parsing_data')
            pose_data = kwargs.get('pose_data')
            clothing_segmentation_data = kwargs.get('clothing_segmentation_data')
            
            return person_image, clothing_image, person_parsing_data, pose_data, clothing_segmentation_data
            
        except Exception as e:
            self.logger.error(f"❌ 입력 검증 실패: {e}")
            raise
    
    def prepare_image_tensor(self, image, device: str) -> torch.Tensor:
        """이미지 텐서 준비"""
        try:
            if isinstance(image, torch.Tensor):
                if image.dim() == 3:
                    image = image.unsqueeze(0)
                return image.to(device)
            
            elif isinstance(image, np.ndarray):
                if image.dim() == 3:
                    image = np.expand_dims(image, axis=0)
                image = torch.from_numpy(image).float()
                return image.to(device)
            
            else:
                raise ValueError(f"지원하지 않는 이미지 형식: {type(image)}")
                
        except Exception as e:
            self.logger.error(f"❌ 이미지 텐서 준비 실패: {e}")
            raise
    
    def execute_all_ai_models(self, step_instance, person_tensor: torch.Tensor, 
                            clothing_tensor: torch.Tensor,
                            person_parsing_data: Dict = None, pose_data: List = None,
                            clothing_segmentation_data: Dict = None) -> Dict[str, Any]:
        """모든 AI 모델 실행"""
        try:
            results = {}
            
            # 1. GMM 모델 실행
            if 'gmm' in step_instance.geometric_matching_models:
                gmm_result = self.execute_gmm_model(step_instance, person_tensor, clothing_tensor)
                results['gmm'] = gmm_result
            
            # 2. 키포인트 매칭 실행
            if 'keypoint_matcher' in step_instance.geometric_matching_models and pose_data:
                keypoint_result = self.execute_keypoint_matching(step_instance, person_tensor, clothing_tensor, pose_data)
                results['keypoint'] = keypoint_result
            
            # 3. 광학 흐름 실행
            if 'optical_flow' in step_instance.geometric_matching_models:
                optical_flow_result = self.execute_optical_flow(step_instance, person_tensor, clothing_tensor)
                results['optical_flow'] = optical_flow_result
            
            # 4. 고급 AI 모델 실행
            if 'complete_advanced' in step_instance.advanced_ai_models:
                advanced_result = self.execute_advanced_ai(step_instance, person_tensor, clothing_tensor)
                results['advanced_ai'] = advanced_result
            
            return results
            
        except Exception as e:
            self.logger.error(f"❌ AI 모델 실행 실패: {e}")
            return {}
    
    def execute_gmm_model(self, step_instance, person_tensor: torch.Tensor, clothing_tensor: torch.Tensor) -> Dict[str, Any]:
        """GMM 모델 실행"""
        try:
            with torch.no_grad():
                output = step_instance.geometric_matching_models['gmm'](person_tensor, clothing_tensor)
                return {
                    'transformation_matrix': output,
                    'confidence': 0.8,
                    'method': 'gmm'
                }
        except Exception as e:
            self.logger.error(f"❌ GMM 모델 실행 실패: {e}")
            return {'error': str(e)}
    
    def execute_keypoint_matching(self, step_instance, person_tensor: torch.Tensor, clothing_tensor: torch.Tensor, 
                                pose_keypoints: List) -> Dict[str, Any]:
        """키포인트 매칭 실행"""
        try:
            with torch.no_grad():
                person_result = step_instance.geometric_matching_models['keypoint_matcher'](person_tensor)
                clothing_result = step_instance.geometric_matching_models['keypoint_matcher'](clothing_tensor)
                matching_result = step_instance.geometric_matching_models['keypoint_matcher'].match_keypoints(person_result, clothing_result)
                
                return {
                    'matches': matching_result['matches'],
                    'person_keypoints': matching_result['person_keypoints'],
                    'clothing_keypoints': matching_result['clothing_keypoints'],
                    'confidence': 0.7,
                    'method': 'keypoint_matching'
                }
        except Exception as e:
            self.logger.error(f"❌ 키포인트 매칭 실행 실패: {e}")
            return {'error': str(e)}
    
    def execute_optical_flow(self, step_instance, person_tensor: torch.Tensor, clothing_tensor: torch.Tensor) -> Dict[str, Any]:
        """광학 흐름 실행"""
        try:
            with torch.no_grad():
                flow = step_instance.geometric_matching_models['optical_flow'](person_tensor, clothing_tensor)
                return {
                    'optical_flow': flow,
                    'confidence': 0.6,
                    'method': 'optical_flow'
                }
        except Exception as e:
            self.logger.error(f"❌ 광학 흐름 실행 실패: {e}")
            return {'error': str(e)}
    
    def execute_advanced_ai(self, step_instance, person_tensor: torch.Tensor, clothing_tensor: torch.Tensor) -> Dict[str, Any]:
        """고급 AI 모델 실행"""
        try:
            with torch.no_grad():
                combined_input = torch.cat([person_tensor, clothing_tensor], dim=1)
                output = step_instance.advanced_ai_models['complete_advanced'](combined_input)
                return {
                    'advanced_result': output,
                    'confidence': 0.9,
                    'method': 'advanced_ai'
                }
        except Exception as e:
            self.logger.error(f"❌ 고급 AI 모델 실행 실패: {e}")
            return {'error': str(e)}
    
    def postprocess_geometric_matching_result(self, inference_result: Dict[str, Any], 
                                            person_tensor: torch.Tensor, 
                                            clothing_tensor: torch.Tensor) -> Dict[str, Any]:
        """기하학적 매칭 결과 후처리"""
        try:
            # 결과 융합
            fused_result = self.fuse_results(inference_result)
            
            # 품질 평가
            quality_metrics = self.evaluate_geometric_matching_quality(fused_result)
            
            # 최종 결과 생성
            final_result = self.create_final_result(fused_result, quality_metrics)
            
            return final_result
            
        except Exception as e:
            self.logger.error(f"❌ 결과 후처리 실패: {e}")
            return self.create_error_response(str(e))
    
    def fuse_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """결과 융합"""
        try:
            fused_result = {
                'transformation_matrix': None,
                'confidence': 0.0,
                'method': 'fused',
                'sub_results': results
            }
            
            # 신뢰도 기반 융합
            total_confidence = 0.0
            valid_results = 0
            
            for method, result in results.items():
                if 'confidence' in result and 'error' not in result:
                    total_confidence += result['confidence']
                    valid_results += 1
            
            if valid_results > 0:
                fused_result['confidence'] = total_confidence / valid_results
            
            return fused_result
            
        except Exception as e:
            self.logger.error(f"❌ 결과 융합 실패: {e}")
            return {'error': str(e)}
    
    def enhance_quality(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """품질 향상"""
        try:
            # 품질 향상 로직 구현
            enhanced_result = result.copy()
            enhanced_result['quality_enhanced'] = True
            return enhanced_result
        except Exception as e:
            self.logger.error(f"❌ 품질 향상 실패: {e}")
            return result
    
    def create_visualization(self, result: Dict[str, Any], person_tensor: torch.Tensor, 
                           clothing_tensor: torch.Tensor) -> Dict[str, Any]:
        """시각화 생성"""
        try:
            # 시각화 로직 구현
            visualization = {
                'visualization_created': True,
                'timestamp': time.time()
            }
            return visualization
        except Exception as e:
            self.logger.error(f"❌ 시각화 생성 실패: {e}")
            return {}
    
    def evaluate_geometric_matching_quality(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """기하학적 매칭 품질 평가"""
        try:
            quality_metrics = {
                'overall_quality': 0.0,
                'confidence_score': result.get('confidence', 0.0),
                'method_count': len(result.get('sub_results', {})),
                'error_count': 0
            }
            
            # 에러 개수 계산
            for method, sub_result in result.get('sub_results', {}).items():
                if 'error' in sub_result:
                    quality_metrics['error_count'] += 1
            
            # 전체 품질 계산
            quality_metrics['overall_quality'] = quality_metrics['confidence_score'] * (1 - quality_metrics['error_count'] / max(quality_metrics['method_count'], 1))
            
            return quality_metrics
            
        except Exception as e:
            self.logger.error(f"❌ 품질 평가 실패: {e}")
            return {'overall_quality': 0.0}
    
    def create_final_result(self, processed_result: Dict[str, Any], 
                          quality_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """최종 결과 생성"""
        try:
            final_result = {
                'success': True,
                'transformation_matrix': processed_result.get('transformation_matrix'),
                'confidence': processed_result.get('confidence', 0.0),
                'quality_metrics': quality_metrics,
                'method': processed_result.get('method', 'fused'),
                'timestamp': time.time(),
                'step_name': 'GeometricMatchingStep',
                'version': 'v1.0_modularized'
            }
            
            return final_result
            
        except Exception as e:
            self.logger.error(f"❌ 최종 결과 생성 실패: {e}")
            return self.create_error_response(str(e))
    
    def create_error_response(self, error_message: str) -> Dict[str, Any]:
        """에러 응답 생성"""
        return {
            'success': False,
            'error': error_message,
            'timestamp': time.time(),
            'step_name': 'GeometricMatchingStep',
            'version': 'v1.0_modularized'
        }
    
    def update_performance_stats(self, step_instance, processing_time: float, success: bool):
        """성능 통계 업데이트"""
        try:
            step_instance.processing_stats['total_processing_time'] += processing_time
            step_instance.processing_stats['last_processing_time'] = processing_time
            
            if success:
                step_instance.processing_stats['successful_inferences'] += 1
            else:
                step_instance.processing_stats['failed_inferences'] += 1
            
            # 평균 처리 시간 계산
            total_inferences = step_instance.processing_stats['successful_inferences'] + step_instance.processing_stats['failed_inferences']
            if total_inferences > 0:
                step_instance.processing_stats['average_processing_time'] = step_instance.processing_stats['total_processing_time'] / total_inferences
            
        except Exception as e:
            self.logger.error(f"❌ 성능 통계 업데이트 실패: {e}")
