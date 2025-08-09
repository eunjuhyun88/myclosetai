# backend/app/ai_pipeline/steps/pose_estimation_step_integrated.py
"""
🔥 PoseEstimationStep 통합 시스템
================================================================================

✅ 3단계 모델 로딩 시스템
✅ 앙상블 추론 시스템
✅ Central Hub 통합
✅ BaseStepMixin 완전 호환

Author: MyCloset AI Team
Date: 2025-01-27
Version: 1.0 (통합 시스템)
"""

import os
import sys
import gc
import time
import json
import logging
import traceback
import warnings
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Tuple
from dataclasses import dataclass

# PyTorch 안전 import
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    F = None

# NumPy 안전 import
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

# MPS 지원 확인
MPS_AVAILABLE = TORCH_AVAILABLE and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()

# 기본 디바이스 설정
DEFAULT_DEVICE = "mps" if MPS_AVAILABLE else ("cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu")

# BaseStepMixin import
try:
    from .base_step_mixin import BaseStepMixin
except ImportError:
    from app.ai_pipeline.steps.base_step_mixin import BaseStepMixin

# 통합 로더 import
from .pose_estimation_integrated_loader import get_integrated_loader, PoseEstimationIntegratedLoader

logger = logging.getLogger(__name__)

@dataclass
class PoseEstimationResult:
    """포즈 추정 결과"""
    keypoints: List[List[float]] = None
    confidence_scores: List[float] = None
    model_used: str = ""
    processing_time: float = 0.0
    success: bool = False
    error_message: str = ""
    ensemble_info: Dict[str, Any] = None

class PoseEstimationStepIntegrated(BaseStepMixin):
    """Pose Estimation 통합 Step 클래스"""
    
    def __init__(self, device: str = DEFAULT_DEVICE, **kwargs):
        super().__init__(**kwargs)
        
        # 기본 설정
        self.device = self._setup_device(device)
        self.logger = logger
        
        # 통합 로더 초기화
        self.integrated_loader: Optional[PoseEstimationIntegratedLoader] = None
        self.loaded_models: Dict[str, Any] = {}
        
        # 앙상블 설정
        self.ensemble_config = {
            'enable_ensemble': True,
            'ensemble_models': ['hrnet', 'openpose', 'yolo_pose', 'mediapipe'],
            'ensemble_method': 'weighted_average',
            'confidence_threshold': 0.7
        }
        
        # Step 정보 설정
        self.step_name = "pose_estimation"
        self.step_version = "2.0"
        self.step_description = "Pose Estimation 통합 시스템"
        
        self.logger.info(f"🚀 PoseEstimationStepIntegrated 초기화 완료 (디바이스: {self.device})")
    
    def _setup_device(self, device: str) -> str:
        """디바이스 설정"""
        if device == "auto":
            if MPS_AVAILABLE:
                return "mps"
            elif TORCH_AVAILABLE and torch.cuda.is_available():
                return "cuda"
            else:
                return "cpu"
        return device
    
    async def initialize(self):
        """Step 초기화"""
        try:
            self.logger.info("🔄 PoseEstimationStepIntegrated 초기화 시작")
            
            # 통합 로더 초기화
            self.integrated_loader = get_integrated_loader(device=self.device, logger=self.logger)
            
            # 모델 로딩
            if self.integrated_loader.load_models_integrated():
                self.loaded_models = self.integrated_loader.get_loaded_models()
                self.logger.info(f"✅ 모델 로딩 완료: {list(self.loaded_models.keys())}")
            else:
                self.logger.warning("⚠️ 모델 로딩 실패 - 기본 모델 사용")
            
            self.logger.info("✅ PoseEstimationStepIntegrated 초기화 완료")
            
        except Exception as e:
            self.logger.error(f"❌ PoseEstimationStepIntegrated 초기화 실패: {e}")
            raise
    
    def process(self, **kwargs) -> Dict[str, Any]:
        """메인 처리 메서드"""
        try:
            self.logger.info("🔄 PoseEstimationStepIntegrated 처리 시작")
            start_time = time.time()
            
            # 입력 데이터 검증
            input_data = self._validate_input(kwargs)
            if not input_data:
                return self._create_error_response("입력 데이터 검증 실패")
            
            # 앙상블 추론 실행
            result = self._run_ensemble_inference(input_data)
            
            # 결과 후처리
            processed_result = self._postprocess_result(result)
            
            processing_time = time.time() - start_time
            self.logger.info(f"✅ PoseEstimationStepIntegrated 처리 완료 ({processing_time:.2f}초)")
            
            return processed_result
            
        except Exception as e:
            self.logger.error(f"❌ PoseEstimationStepIntegrated 처리 실패: {e}")
            return self._create_error_response(str(e))
    
    def _validate_input(self, kwargs: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """입력 데이터 검증"""
        try:
            # 필수 필드 확인
            required_fields = ['image']
            for field in required_fields:
                if field not in kwargs:
                    self.logger.error(f"❌ 필수 필드 누락: {field}")
                    return None
            
            # 이미지 데이터 검증
            image = kwargs['image']
            if image is None:
                self.logger.error("❌ 이미지 데이터가 None입니다")
                return None
            
            return {
                'image': image,
                'config': kwargs.get('config', {}),
                'options': kwargs.get('options', {})
            }
            
        except Exception as e:
            self.logger.error(f"❌ 입력 데이터 검증 실패: {e}")
            return None
    
    def _run_ensemble_inference(self, input_data: Dict[str, Any]) -> PoseEstimationResult:
        """앙상블 추론 실행"""
        try:
            self.logger.info("🔄 앙상블 추론 시작")
            start_time = time.time()
            
            image = input_data['image']
            config = input_data.get('config', {})
            
            # 개별 모델 추론 결과 수집
            model_results = {}
            
            for model_name, model in self.loaded_models.items():
                try:
                    self.logger.info(f"🔄 {model_name} 모델 추론 시작")
                    
                    if hasattr(model, 'detect_poses'):
                        result = model.detect_poses(image)
                        if result.get('success', False):
                            model_results[model_name] = result
                            self.logger.info(f"✅ {model_name} 추론 성공")
                        else:
                            self.logger.warning(f"⚠️ {model_name} 추론 실패: {result.get('error', 'Unknown error')}")
                    else:
                        self.logger.warning(f"⚠️ {model_name} 모델에 detect_poses 메서드가 없습니다")
                        
                except Exception as e:
                    self.logger.error(f"❌ {model_name} 추론 중 오류: {e}")
            
            # 앙상블 결과 통합
            ensemble_result = self._combine_ensemble_results(model_results)
            
            processing_time = time.time() - start_time
            ensemble_result.processing_time = processing_time
            
            self.logger.info(f"✅ 앙상블 추론 완료 ({processing_time:.2f}초)")
            return ensemble_result
            
        except Exception as e:
            self.logger.error(f"❌ 앙상블 추론 실패: {e}")
            return PoseEstimationResult(
                success=False,
                error_message=str(e)
            )
    
    def _combine_ensemble_results(self, model_results: Dict[str, Any]) -> PoseEstimationResult:
        """앙상블 결과 통합"""
        try:
            if not model_results:
                return PoseEstimationResult(
                    success=False,
                    error_message="사용 가능한 모델 결과가 없습니다"
                )
            
            # 단일 모델 결과인 경우
            if len(model_results) == 1:
                model_name = list(model_results.keys())[0]
                result = model_results[model_name]
                return PoseEstimationResult(
                    keypoints=result.get('keypoints', []),
                    confidence_scores=result.get('confidence_scores', []),
                    model_used=model_name,
                    success=True,
                    ensemble_info={'method': 'single_model', 'models_used': [model_name]}
                )
            
            # 다중 모델 앙상블
            ensemble_method = self.ensemble_config.get('ensemble_method', 'weighted_average')
            
            if ensemble_method == 'weighted_average':
                return self._weighted_average_ensemble(model_results)
            elif ensemble_method == 'confidence_weighted':
                return self._confidence_weighted_ensemble(model_results)
            else:
                return self._simple_average_ensemble(model_results)
                
        except Exception as e:
            self.logger.error(f"❌ 앙상블 결과 통합 실패: {e}")
            return PoseEstimationResult(
                success=False,
                error_message=f"앙상블 통합 실패: {e}"
            )
    
    def _weighted_average_ensemble(self, model_results: Dict[str, Any]) -> PoseEstimationResult:
        """가중 평균 앙상블"""
        try:
            # 모델별 가중치 설정
            model_weights = {
                'hrnet': 0.4,
                'openpose': 0.3,
                'yolo_pose': 0.2,
                'mediapipe': 0.1
            }
            
            # 키포인트별 가중 평균 계산
            all_keypoints = []
            total_weight = 0
            
            for model_name, result in model_results.items():
                weight = model_weights.get(model_name, 0.1)
                keypoints = result.get('keypoints', [])
                
                if keypoints:
                    all_keypoints.append((keypoints, weight))
                    total_weight += weight
            
            if not all_keypoints:
                return PoseEstimationResult(
                    success=False,
                    error_message="유효한 키포인트가 없습니다"
                )
            
            # 가중 평균 계산
            num_keypoints = len(all_keypoints[0][0])
            ensemble_keypoints = []
            ensemble_confidences = []
            
            for i in range(num_keypoints):
                weighted_x = 0
                weighted_y = 0
                weighted_conf = 0
                
                for keypoints, weight in all_keypoints:
                    if i < len(keypoints):
                        kp = keypoints[i]
                        if len(kp) >= 3:
                            weighted_x += kp[0] * weight
                            weighted_y += kp[1] * weight
                            weighted_conf += kp[2] * weight
                
                if total_weight > 0:
                    ensemble_keypoints.append([
                        weighted_x / total_weight,
                        weighted_y / total_weight,
                        weighted_conf / total_weight
                    ])
                    ensemble_confidences.append(weighted_conf / total_weight)
                else:
                    ensemble_keypoints.append([0.0, 0.0, 0.0])
                    ensemble_confidences.append(0.0)
            
            return PoseEstimationResult(
                keypoints=ensemble_keypoints,
                confidence_scores=ensemble_confidences,
                model_used="ensemble",
                success=True,
                ensemble_info={
                    'method': 'weighted_average',
                    'models_used': list(model_results.keys()),
                    'weights': model_weights
                }
            )
            
        except Exception as e:
            self.logger.error(f"❌ 가중 평균 앙상블 실패: {e}")
            return PoseEstimationResult(
                success=False,
                error_message=f"가중 평균 앙상블 실패: {e}"
            )
    
    def _confidence_weighted_ensemble(self, model_results: Dict[str, Any]) -> PoseEstimationResult:
        """신뢰도 가중 앙상블"""
        try:
            # 신뢰도 기반 가중치 계산
            model_confidences = {}
            for model_name, result in model_results.items():
                confidence_scores = result.get('confidence_scores', [])
                if confidence_scores:
                    avg_confidence = sum(confidence_scores) / len(confidence_scores)
                    model_confidences[model_name] = avg_confidence
                else:
                    model_confidences[model_name] = 0.5
            
            # 신뢰도 정규화
            total_confidence = sum(model_confidences.values())
            if total_confidence > 0:
                model_weights = {name: conf / total_confidence for name, conf in model_confidences.items()}
            else:
                model_weights = {name: 1.0 / len(model_confidences) for name in model_confidences.keys()}
            
            # 가중 평균 계산 (위의 메서드와 동일한 로직)
            return self._weighted_average_ensemble(model_results)
            
        except Exception as e:
            self.logger.error(f"❌ 신뢰도 가중 앙상블 실패: {e}")
            return PoseEstimationResult(
                success=False,
                error_message=f"신뢰도 가중 앙상블 실패: {e}"
            )
    
    def _simple_average_ensemble(self, model_results: Dict[str, Any]) -> PoseEstimationResult:
        """단순 평균 앙상블"""
        try:
            # 모든 모델에 동일한 가중치 적용
            model_weights = {name: 1.0 / len(model_results) for name in model_results.keys()}
            
            # 가중 평균 계산 (위의 메서드와 동일한 로직)
            return self._weighted_average_ensemble(model_results)
            
        except Exception as e:
            self.logger.error(f"❌ 단순 평균 앙상블 실패: {e}")
            return PoseEstimationResult(
                success=False,
                error_message=f"단순 평균 앙상블 실패: {e}"
            )
    
    def _postprocess_result(self, result: PoseEstimationResult) -> Dict[str, Any]:
        """결과 후처리"""
        try:
            if not result.success:
                return self._create_error_response(result.error_message)
            
            # 결과 검증
            if not result.keypoints or len(result.keypoints) == 0:
                return self._create_error_response("키포인트가 감지되지 않았습니다")
            
            # API 응답 형식으로 변환
            response = {
                "success": True,
                "step_name": self.step_name,
                "step_version": self.step_version,
                "data": {
                    "keypoints": result.keypoints,
                    "confidence_scores": result.confidence_scores,
                    "model_used": result.model_used,
                    "num_keypoints": len(result.keypoints),
                    "overall_confidence": sum(result.confidence_scores) / len(result.confidence_scores) if result.confidence_scores else 0.0
                },
                "metadata": {
                    "processing_time": result.processing_time,
                    "ensemble_info": result.ensemble_info,
                    "device_used": self.device,
                    "models_loaded": list(self.loaded_models.keys())
                }
            }
            
            return response
            
        except Exception as e:
            self.logger.error(f"❌ 결과 후처리 실패: {e}")
            return self._create_error_response(f"결과 후처리 실패: {e}")
    
    def _create_error_response(self, error_message: str) -> Dict[str, Any]:
        """에러 응답 생성"""
        return {
            "success": False,
            "step_name": self.step_name,
            "step_version": self.step_version,
            "error": {
                "message": error_message,
                "type": "pose_estimation_error"
            },
            "data": {
                "keypoints": [],
                "confidence_scores": [],
                "model_used": "",
                "num_keypoints": 0,
                "overall_confidence": 0.0
            },
            "metadata": {
                "processing_time": 0.0,
                "device_used": self.device,
                "models_loaded": list(self.loaded_models.keys())
            }
        }
    
    def get_model_status(self) -> Dict[str, Any]:
        """모델 상태 반환"""
        return {
            "step_name": self.step_name,
            "device": self.device,
            "models_loaded": list(self.loaded_models.keys()),
            "ensemble_config": self.ensemble_config,
            "total_models": len(self.loaded_models)
        }
    
    async def cleanup(self):
        """리소스 정리"""
        try:
            self.logger.info("🔄 PoseEstimationStepIntegrated 리소스 정리 시작")
            
            if self.integrated_loader:
                self.integrated_loader.cleanup_resources()
            
            self.loaded_models.clear()
            
            if TORCH_AVAILABLE:
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            gc.collect()
            
            self.logger.info("✅ PoseEstimationStepIntegrated 리소스 정리 완료")
            
        except Exception as e:
            self.logger.error(f"❌ 리소스 정리 실패: {e}")

# 팩토리 함수
async def create_pose_estimation_step_integrated(
    device: str = DEFAULT_DEVICE,
    config: Optional[Dict[str, Any]] = None,
    **kwargs
) -> PoseEstimationStepIntegrated:
    """PoseEstimationStepIntegrated 인스턴스 생성"""
    step = PoseEstimationStepIntegrated(device=device, **kwargs)
    await step.initialize()
    return step

def create_pose_estimation_step_integrated_sync(
    device: str = DEFAULT_DEVICE,
    config: Optional[Dict[str, Any]] = None,
    **kwargs
) -> PoseEstimationStepIntegrated:
    """동기 버전 PoseEstimationStepIntegrated 인스턴스 생성"""
    import asyncio
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # 새로운 이벤트 루프에서 실행
            step = PoseEstimationStepIntegrated(device=device, **kwargs)
            # 초기화는 나중에 수행
            return step
        else:
            # 기존 루프에서 실행
            return loop.run_until_complete(
                create_pose_estimation_step_integrated(device=device, config=config, **kwargs)
            )
    except RuntimeError:
        # 이벤트 루프가 없는 경우
        step = PoseEstimationStepIntegrated(device=device, **kwargs)
        return step
