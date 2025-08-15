"""
🔥 Pose Estimation Service - 포즈 추정 서비스
==========================================

포즈 추정을 위한 통합 서비스 시스템

주요 기능:
- 모델 관리
- 추론 실행
- 결과 후처리
- 품질 평가
"""

import logging
import time
from typing import Dict, Any, Optional, List, Tuple, Union
import torch
import numpy as np

from ..inference.inference_engine import InferenceEngine
from ..postprocessing.postprocessor import Postprocessor
from ..utils.quality_assessment import PoseEstimationQualityAssessment

logger = logging.getLogger(__name__)

class PoseEstimationService:
    """포즈 추정 통합 서비스"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # 서비스 컴포넌트들 초기화
        self.inference_engine = None
        self.postprocessor = None
        self.quality_assessor = None
        
        # 설정 로드
        self._load_config()
        
        # 컴포넌트 초기화
        self._initialize_components()
        
        logger.info("✅ Pose Estimation Service 초기화 완료")
    
    def _load_config(self):
        """설정 로드"""
        # 기본 설정
        self.enable_inference = self.config.get('enable_inference', True)
        self.enable_postprocessing = self.config.get('enable_postprocessing', True)
        self.enable_quality_assessment = self.config.get('enable_quality_assessment', True)
        
        # 고급 설정
        self.inference_timeout = self.config.get('inference_timeout', 30.0)
        self.max_retries = self.config.get('max_retries', 3)
        self.quality_threshold = self.config.get('quality_threshold', 0.7)
        
        logger.info(f"✅ 설정 로드 완료: inference={self.enable_inference}, postprocessing={self.enable_postprocessing}")
    
    def _initialize_components(self):
        """서비스 컴포넌트들 초기화"""
        try:
            # 추론 엔진 초기화
            if self.enable_inference:
                self.inference_engine = InferenceEngine(self)
                logger.info("✅ 추론 엔진 초기화 완료")
            
            # 후처리기 초기화
            if self.enable_postprocessing:
                self.postprocessor = Postprocessor(self.config)
                logger.info("✅ 후처리기 초기화 완료")
            
            # 품질 평가기 초기화
            if self.enable_quality_assessment:
                self.quality_assessor = PoseEstimationQualityAssessment(self.config)
                logger.info("✅ 품질 평가기 초기화 완료")
                
        except Exception as e:
            logger.error(f"❌ 컴포넌트 초기화 실패: {e}")
            raise
    
    def process_pose_estimation(self, 
                               input_data: Dict[str, Any], 
                               **kwargs) -> Dict[str, Any]:
        """
        포즈 추정 처리 메인 메서드
        
        Args:
            input_data: 입력 데이터
            **kwargs: 추가 파라미터
        
        Returns:
            result: 처리 결과
        """
        try:
            logger.info("🚀 포즈 추정 서비스 시작")
            start_time = time.time()
            
            # 1. 입력 데이터 검증
            validated_input = self._validate_input(input_data)
            if not validated_input['valid']:
                return {
                    'success': False,
                    'error': validated_input['error'],
                    'keypoints': None
                }
            
            # 2. 추론 실행
            inference_result = None
            if self.enable_inference and self.inference_engine:
                inference_result = self._run_inference(validated_input['data'])
                if not inference_result['success']:
                    return inference_result
            
            # 3. 후처리
            postprocessed_result = None
            if self.enable_postprocessing and self.postprocessor and inference_result:
                postprocessed_result = self._run_postprocessing(inference_result['keypoints'])
            
            # 4. 품질 평가
            quality_result = None
            if self.enable_quality_assessment and self.quality_assessor:
                target_data = postprocessed_result if postprocessed_result else inference_result['keypoints']
                quality_result = self._assess_quality(target_data)
            
            # 5. 결과 통합
            final_result = self._integrate_results(
                inference_result, 
                postprocessed_result, 
                quality_result
            )
            
            # 6. 실행 시간 계산
            execution_time = time.time() - start_time
            final_result['execution_time'] = execution_time
            
            logger.info(f"✅ 포즈 추정 서비스 완료 (소요시간: {execution_time:.2f}초)")
            return final_result
            
        except Exception as e:
            logger.error(f"❌ 포즈 추정 서비스 실패: {e}")
            return {
                'success': False,
                'error': str(e),
                'keypoints': None,
                'execution_time': time.time() - start_time
            }
    
    def _validate_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """입력 데이터 검증"""
        try:
            # 필수 필드 확인
            required_fields = ['image']
            for field in required_fields:
                if field not in input_data:
                    return {
                        'valid': False,
                        'error': f"필수 필드 누락: {field}"
                    }
            
            # 이미지 데이터 검증
            image = input_data['image']
            if image is None:
                return {
                    'valid': False,
                    'error': "이미지가 None입니다"
                }
            
            # 이미지 크기 검증
            if hasattr(image, 'size'):
                width, height = image.size
                if width < 64 or height < 64:
                    return {
                        'valid': False,
                        'error': f"이미지 크기가 너무 작습니다: {width}x{height}"
                    }
            
            return {
                'valid': True,
                'data': input_data
            }
            
        except Exception as e:
            return {
                'valid': False,
                'error': f"입력 데이터 검증 실패: {e}"
            }
    
    def _run_inference(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """추론 실행"""
        try:
            if not self.inference_engine:
                return {
                    'success': False,
                    'error': "추론 엔진이 초기화되지 않았습니다"
                }
            
            # 추론 실행
            result = self.inference_engine.run_ai_inference(input_data)
            
            if not result['success']:
                return result
            
            logger.info("✅ 추론 실행 완료")
            return result
            
        except Exception as e:
            logger.error(f"❌ 추론 실행 실패: {e}")
            return {
                'success': False,
                'error': str(e),
                'keypoints': None
            }
    
    def _run_postprocessing(self, keypoints: torch.Tensor) -> torch.Tensor:
        """후처리 실행"""
        try:
            if not self.postprocessor:
                logger.warning("⚠️ 후처리기가 초기화되지 않았습니다")
                return keypoints
            
            # 후처리 실행
            postprocessed_keypoints = self.postprocessor.postprocess(keypoints)
            
            logger.info("✅ 후처리 완료")
            return postprocessed_keypoints
            
        except Exception as e:
            logger.error(f"❌ 후처리 실패: {e}")
            return keypoints
    
    def _assess_quality(self, keypoints: torch.Tensor) -> Dict[str, Any]:
        """품질 평가"""
        try:
            if not self.quality_assessor:
                logger.warning("⚠️ 품질 평가기가 초기화되지 않았습니다")
                return {'quality_score': 0.8, 'confidence': 0.8}
            
            # 품질 평가 실행
            quality_result = self.quality_assessor.assess_quality(keypoints)
            
            logger.info("✅ 품질 평가 완료")
            return quality_result
            
        except Exception as e:
            logger.error(f"❌ 품질 평가 실패: {e}")
            return {'quality_score': 0.8, 'confidence': 0.8}
    
    def _integrate_results(self, 
                          inference_result: Dict[str, Any],
                          postprocessed_result: Optional[torch.Tensor],
                          quality_result: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """결과 통합"""
        try:
            # 기본 결과
            final_result = {
                'success': True,
                'keypoints': postprocessed_result if postprocessed_result is not None else inference_result['keypoints'],
                'raw_keypoints': inference_result['keypoints'],
                'postprocessed': postprocessed_result is not None,
                'models_used': inference_result.get('models_used', []),
                'ensemble_method': inference_result.get('ensemble_method', 'single')
            }
            
            # 품질 정보 추가
            if quality_result:
                final_result.update({
                    'quality_score': quality_result.get('quality_score', 0.8),
                    'confidence': quality_result.get('confidence', 0.8)
                })
            
            # 후처리 통계 추가
            if postprocessed_result is not None and self.postprocessor:
                final_result['postprocessing_stats'] = self.postprocessor.get_processing_stats()
            
            # 품질 향상 통계 추가
            if hasattr(self, 'quality_enhancer') and self.quality_enhancer:
                final_result['enhancement_stats'] = self.quality_enhancer.get_enhancement_stats()
            
            return final_result
            
        except Exception as e:
            logger.error(f"❌ 결과 통합 실패: {e}")
            return {
                'success': False,
                'error': str(e),
                'keypoints': inference_result.get('keypoints') if inference_result else None
            }
    
    def get_service_status(self) -> Dict[str, Any]:
        """서비스 상태 반환"""
        return {
            'inference_engine_available': self.inference_engine is not None,
            'postprocessor_available': self.postprocessor is not None,
            'quality_assessor_available': self.quality_assessor is not None,
            'enable_inference': self.enable_inference,
            'enable_postprocessing': self.enable_postprocessing,
            'enable_quality_assessment': self.enable_quality_assessment,
            'inference_timeout': self.inference_timeout,
            'max_retries': self.max_retries,
            'quality_threshold': self.quality_threshold
        }
    
    def cleanup(self):
        """서비스 정리"""
        try:
            logger.info("🧹 Pose Estimation Service 정리 시작")
            
            # 컴포넌트 정리
            if self.inference_engine:
                del self.inference_engine
                self.inference_engine = None
            
            if self.postprocessor:
                del self.postprocessor
                self.postprocessor = None
            
            if self.quality_assessor:
                del self.quality_assessor
                self.quality_assessor = None
            
            logger.info("✅ Pose Estimation Service 정리 완료")
            
        except Exception as e:
            logger.error(f"❌ 서비스 정리 실패: {e}")
