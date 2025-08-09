#!/usr/bin/env python3
"""
🔥 MyCloset AI - Step 03: 의류 세그멘테이션 - Inference Core
=====================================================================

AI 추론 실행 및 관리 핵심 기능들

Author: MyCloset AI Team  
Date: 2025-08-01
Version: 1.0
"""

import logging
import time
import gc
from typing import Dict, Any, Optional, List, Tuple

try:
    import numpy as np
    import cv2
    NUMPY_AVAILABLE = True
    CV2_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    CV2_AVAILABLE = False
    np = None
    cv2 = None

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from ..config import SegmentationMethod, ClothCategory, QualityLevel
from ..ensemble import _run_hybrid_ensemble_sync, _combine_ensemble_results

logger = logging.getLogger(__name__)

class InferenceCore:
    """
    🔥 AI 추론 실행 및 관리 핵심 기능들
    
    분리된 기능들:
    - AI 모델 추론 실행
    - 앙상블 추론 관리
    - 메모리 안전성 보장
    - 결과 검증 및 후처리
    """
    
    def __init__(self, models: Dict[str, Any], device: str = "cpu"):
        """초기화"""
        self.models = models
        self.device = device
        self.logger = logging.getLogger(f"{__name__}.InferenceCore")
        self.inference_stats = {
            'total_inferences': 0,
            'successful_inferences': 0,
            'failed_inferences': 0,
            'average_inference_time': 0.0,
            'last_inference_time': 0.0
        }
        
    def run_ai_inference(self, 
                        image: np.ndarray, 
                        method: SegmentationMethod = SegmentationMethod.U2NET_CLOTH,
                        person_parsing: Optional[Dict[str, Any]] = None,
                        pose_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """AI 추론 실행"""
        try:
            start_time = time.time()
            self.logger.info(f"🔄 AI 추론 시작: {method.value}")
            
            # 입력 검증
            if not self._validate_input(image):
                return self._create_error_result("입력 이미지 검증 실패")
            
            # 메모리 안전성 체크
            if not self._check_memory_safety():
                return self._create_error_result("메모리 부족")
            
            # 추론 실행
            if method == SegmentationMethod.HYBRID_AI:
                result = self._run_hybrid_inference(image, person_parsing, pose_info)
            else:
                result = self._run_single_model_inference(image, method, person_parsing, pose_info)
            
            # 결과 검증
            if not self._validate_result(result):
                return self._create_error_result("추론 결과 검증 실패")
            
            # 통계 업데이트
            inference_time = time.time() - start_time
            self._update_inference_stats(True, inference_time)
            
            self.logger.info(f"✅ AI 추론 완료: {method.value} ({inference_time:.2f}s)")
            return result
            
        except Exception as e:
            self.logger.error(f"❌ AI 추론 실패: {e}")
            self._update_inference_stats(False, 0.0)
            return self._create_error_result(f"추론 실패: {e}")

    def _run_single_model_inference(self, 
                                  image: np.ndarray, 
                                  method: SegmentationMethod,
                                  person_parsing: Optional[Dict[str, Any]] = None,
                                  pose_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """단일 모델 추론 실행"""
        try:
            model_key = method.value
            
            if model_key not in self.models:
                return self._create_error_result(f"모델이 로딩되지 않음: {model_key}")
            
            model = self.models[model_key]
            
            # 모델별 추론 실행
            if 'u2net' in model_key:
                result = self._run_u2net_inference(model, image, person_parsing, pose_info)
            elif 'sam' in model_key:
                result = self._run_sam_inference(model, image, person_parsing, pose_info)
            elif 'deeplabv3' in model_key:
                result = self._run_deeplabv3_inference(model, image, person_parsing, pose_info)
            else:
                return self._create_error_result(f"알 수 없는 모델 타입: {model_key}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ 단일 모델 추론 실패: {e}")
            return self._create_error_result(f"단일 모델 추론 실패: {e}")

    def _run_hybrid_inference(self, 
                            image: np.ndarray,
                            person_parsing: Optional[Dict[str, Any]] = None,
                            pose_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """하이브리드 앙상블 추론 실행"""
        try:
            self.logger.info("🔄 하이브리드 앙상블 추론 시작")
            
            # 앙상블 실행
            result = _run_hybrid_ensemble_sync(
                self, image, person_parsing or {}, pose_info or {}
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ 하이브리드 추론 실패: {e}")
            return self._create_error_result(f"하이브리드 추론 실패: {e}")

    def _run_u2net_inference(self, 
                           model: Any, 
                           image: np.ndarray,
                           person_parsing: Optional[Dict[str, Any]] = None,
                           pose_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """U2Net 추론 실행"""
        try:
            # 이미지 전처리
            processed_image = self._preprocess_image_for_u2net(image)
            
            # 추론 실행
            with torch.no_grad():
                prediction = model.predict(processed_image)
            
            # 결과 후처리
            result = self._postprocess_u2net_result(prediction, image)
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ U2Net 추론 실패: {e}")
            return self._create_error_result(f"U2Net 추론 실패: {e}")

    def _run_sam_inference(self, 
                          model: Any, 
                          image: np.ndarray,
                          person_parsing: Optional[Dict[str, Any]] = None,
                          pose_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """SAM 추론 실행"""
        try:
            # 프롬프트 생성
            prompts = self._generate_sam_prompts(image, person_parsing, pose_info)
            
            # 추론 실행
            with torch.no_grad():
                prediction = model.predict(image, prompts=prompts)
            
            # 결과 후처리
            result = self._postprocess_sam_result(prediction, image)
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ SAM 추론 실패: {e}")
            return self._create_error_result(f"SAM 추론 실패: {e}")

    def _run_deeplabv3_inference(self, 
                                model: Any, 
                                image: np.ndarray,
                                person_parsing: Optional[Dict[str, Any]] = None,
                                pose_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """DeepLabV3+ 추론 실행"""
        try:
            # 이미지 전처리
            processed_image = self._preprocess_image_for_deeplabv3(image)
            
            # 추론 실행
            with torch.no_grad():
                prediction = model.predict(processed_image)
            
            # 결과 후처리
            result = self._postprocess_deeplabv3_result(prediction, image)
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ DeepLabV3+ 추론 실패: {e}")
            return self._create_error_result(f"DeepLabV3+ 추론 실패: {e}")

    def _preprocess_image_for_u2net(self, image: np.ndarray) -> np.ndarray:
        """U2Net용 이미지 전처리"""
        try:
            # 이미지 크기 조정
            if image.shape[:2] != (512, 512):
                image = cv2.resize(image, (512, 512))
            
            # 정규화
            if image.dtype != np.float32:
                image = image.astype(np.float32) / 255.0
            
            return image
            
        except Exception as e:
            self.logger.error(f"❌ U2Net 이미지 전처리 실패: {e}")
            return image

    def _preprocess_image_for_deeplabv3(self, image: np.ndarray) -> np.ndarray:
        """DeepLabV3+용 이미지 전처리"""
        try:
            # 이미지 크기 조정
            if image.shape[:2] != (512, 512):
                image = cv2.resize(image, (512, 512))
            
            # 정규화
            if image.dtype != np.float32:
                image = image.astype(np.float32) / 255.0
            
            return image
            
        except Exception as e:
            self.logger.error(f"❌ DeepLabV3+ 이미지 전처리 실패: {e}")
            return image

    def _generate_sam_prompts(self, 
                            image: np.ndarray,
                            person_parsing: Optional[Dict[str, Any]] = None,
                            pose_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """SAM 프롬프트 생성"""
        try:
            prompts = {
                'points': [],
                'boxes': [],
                'masks': []
            }
            
            # 기본 프롬프트 생성 (이미지 중심점)
            h, w = image.shape[:2]
            center_point = [w // 2, h // 2]
            prompts['points'].append(center_point)
            
            return prompts
            
        except Exception as e:
            self.logger.error(f"❌ SAM 프롬프트 생성 실패: {e}")
            return {'points': [], 'boxes': [], 'masks': []}

    def _postprocess_u2net_result(self, prediction: Dict[str, Any], original_image: np.ndarray) -> Dict[str, Any]:
        """U2Net 결과 후처리"""
        try:
            # 마스크 추출
            masks = prediction.get('masks', {})
            
            # 결과 구성
            result = {
                'success': True,
                'method': 'u2net_cloth',
                'masks': masks,
                'confidence': prediction.get('confidence', 0.0),
                'processing_time': prediction.get('processing_time', 0.0),
                'original_image_shape': original_image.shape
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ U2Net 결과 후처리 실패: {e}")
            return self._create_error_result(f"U2Net 결과 후처리 실패: {e}")

    def _postprocess_sam_result(self, prediction: Dict[str, Any], original_image: np.ndarray) -> Dict[str, Any]:
        """SAM 결과 후처리"""
        try:
            # 마스크 추출
            masks = prediction.get('masks', {})
            
            # 결과 구성
            result = {
                'success': True,
                'method': 'sam_huge',
                'masks': masks,
                'confidence': prediction.get('confidence', 0.0),
                'processing_time': prediction.get('processing_time', 0.0),
                'original_image_shape': original_image.shape
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ SAM 결과 후처리 실패: {e}")
            return self._create_error_result(f"SAM 결과 후처리 실패: {e}")

    def _postprocess_deeplabv3_result(self, prediction: Dict[str, Any], original_image: np.ndarray) -> Dict[str, Any]:
        """DeepLabV3+ 결과 후처리"""
        try:
            # 마스크 추출
            masks = prediction.get('masks', {})
            
            # 결과 구성
            result = {
                'success': True,
                'method': 'deeplabv3_plus',
                'masks': masks,
                'confidence': prediction.get('confidence', 0.0),
                'processing_time': prediction.get('processing_time', 0.0),
                'original_image_shape': original_image.shape
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ DeepLabV3+ 결과 후처리 실패: {e}")
            return self._create_error_result(f"DeepLabV3+ 결과 후처리 실패: {e}")

    def _validate_input(self, image: np.ndarray) -> bool:
        """입력 검증"""
        try:
            if image is None:
                return False
            
            if not isinstance(image, np.ndarray):
                return False
            
            if len(image.shape) != 3:
                return False
            
            if image.shape[2] not in [1, 3]:
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ 입력 검증 실패: {e}")
            return False

    def _validate_result(self, result: Dict[str, Any]) -> bool:
        """결과 검증"""
        try:
            if not isinstance(result, dict):
                return False
            
            if 'success' not in result:
                return False
            
            if not result['success']:
                return False
            
            if 'masks' not in result:
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ 결과 검증 실패: {e}")
            return False

    def _check_memory_safety(self) -> bool:
        """메모리 안전성 체크"""
        try:
            import psutil
            memory_usage = psutil.virtual_memory().percent
            return memory_usage < 90
        except ImportError:
            return True

    def _create_error_result(self, error_message: str) -> Dict[str, Any]:
        """에러 결과 생성"""
        return {
            'success': False,
            'error': error_message,
            'masks': {},
            'confidence': 0.0,
            'processing_time': 0.0
        }

    def _update_inference_stats(self, success: bool, inference_time: float):
        """추론 통계 업데이트"""
        try:
            self.inference_stats['total_inferences'] += 1
            
            if success:
                self.inference_stats['successful_inferences'] += 1
            else:
                self.inference_stats['failed_inferences'] += 1
            
            # 평균 추론 시간 업데이트
            total_successful = self.inference_stats['successful_inferences']
            if total_successful > 0:
                current_avg = self.inference_stats['average_inference_time']
                new_avg = (current_avg * (total_successful - 1) + inference_time) / total_successful
                self.inference_stats['average_inference_time'] = new_avg
            
            self.inference_stats['last_inference_time'] = inference_time
            
        except Exception as e:
            self.logger.error(f"❌ 추론 통계 업데이트 실패: {e}")

    def get_inference_stats(self) -> Dict[str, Any]:
        """추론 통계 반환"""
        return self.inference_stats.copy()

    def reset_inference_stats(self):
        """추론 통계 초기화"""
        self.inference_stats = {
            'total_inferences': 0,
            'successful_inferences': 0,
            'failed_inferences': 0,
            'average_inference_time': 0.0,
            'last_inference_time': 0.0
        }
