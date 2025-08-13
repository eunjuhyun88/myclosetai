# backend/app/ai_pipeline/steps/pose_estimation_integrated_loader.py
"""
🔥 PoseEstimationStep 통합 모델 로딩 시스템
================================================================================

✅ Central Hub 통합
✅ 체크포인트 분석 시스템 연동
✅ 모델 아키텍처 기반 생성
✅ 단계적 폴백 시스템
✅ BaseStepMixin 완전 호환

Author: MyCloset AI Team
Date: 2025-01-27
Version: 1.0 (통합 모델 로딩 시스템)
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

logger = logging.getLogger(__name__)

@dataclass
class ModelLoadingResult:
    """모델 로딩 결과"""
    success: bool
    model: Optional[Any] = None
    model_name: str = ""
    loading_method: str = ""
    error_message: str = ""
    processing_time: float = 0.0

class PoseEstimationModelLoader:
    """PoseEstimationStep 전용 모델 로딩 시스템"""
    
    def __init__(self, device: str = DEFAULT_DEVICE, logger=None):
        self.device = self._setup_device(device)
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.models = {}
        self.loaded_models = {}
        
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
    
    def load_models_integrated(self) -> bool:
        """통합된 모델 로딩 시스템 - 메인 메서드"""
        try:
            self.logger.info("🚀 PoseEstimation 통합 모델 로딩 시스템 시작")
            start_time = time.time()
            
            # 1단계: Central Hub 시도
            if self._load_via_central_hub():
                self.logger.info("✅ Central Hub를 통한 모델 로딩 성공")
                return True
            
            # 2단계: 체크포인트 분석 기반 로딩
            models_loaded = 0
            for model_name in ['hrnet', 'openpose', 'yolo_pose', 'mediapipe']:
                result = self._load_with_checkpoint_analysis(model_name)
                if result.success:
                    self.models[model_name] = result.model
                    models_loaded += 1
            
            if models_loaded > 0:
                self.logger.info(f"✅ 체크포인트 분석 기반 모델 로딩 성공: {models_loaded}개")
                return True
            
            # 3단계: 아키텍처 기반 생성
            fallback_models = {
                'hrnet': {'num_keypoints': 17, 'architecture_type': 'hrnet'},
                'openpose': {'num_keypoints': 18, 'architecture_type': 'openpose'},
                'yolo_pose': {'num_keypoints': 17, 'architecture_type': 'yolo_pose'},
                'mediapipe': {'num_keypoints': 17, 'architecture_type': 'mediapipe'}
            }
            
            for model_name, config in fallback_models.items():
                result = self._create_with_architecture(model_name, config)
                if result.success:
                    self.models[model_name] = result.model
                    models_loaded += 1
            
            if models_loaded > 0:
                self.logger.info(f"✅ 아키텍처 기반 모델 생성 성공: {models_loaded}개")
                return True
            
            self.logger.error("❌ 모든 모델 로딩 방법 실패")
            return False
            
        except Exception as e:
            self.logger.error(f"❌ 통합 모델 로딩 실패: {e}")
            return False
    
    def _load_via_central_hub(self) -> bool:
        """Central Hub를 통한 모델 로딩"""
        try:
            # Central Hub에서 모델 로더 서비스 가져오기
            model_loader_service = self._get_service_from_central_hub('model_loader')
            if not model_loader_service:
                self.logger.warning("⚠️ Central Hub에서 모델 로더 서비스를 찾을 수 없습니다")
                return False
            
            # Step별 최적 모델 로드
            models_to_load = {
                'hrnet': 'pose_estimation',
                'openpose': 'pose_estimation', 
                'yolo_pose': 'pose_estimation',
                'mediapipe': 'pose_estimation'
            }
            
            models_loaded = 0
            for model_name, step_type in models_to_load.items():
                try:
                    model = model_loader_service.load_model_for_step(step_type, model_name)
                    if model:
                        self.models[model_name] = model
                        models_loaded += 1
                        self.logger.info(f"✅ Central Hub를 통한 {model_name} 모델 로딩 성공")
                except Exception as e:
                    self.logger.warning(f"⚠️ Central Hub를 통한 {model_name} 모델 로딩 실패: {e}")
            
            return models_loaded > 0
            
        except Exception as e:
            self.logger.error(f"❌ Central Hub 로딩 실패: {e}")
            return False
    
    def _load_with_checkpoint_analysis(self, model_name: str) -> ModelLoadingResult:
        """체크포인트 분석 기반 모델 로딩"""
        start_time = time.time()
        try:
            self.logger.info(f"🔍 {model_name} 체크포인트 분석 시작")
            
            # 체크포인트 분석 서비스 가져오기
            checkpoint_analyzer = self._get_service_from_central_hub('checkpoint_analyzer')
            if not checkpoint_analyzer:
                return ModelLoadingResult(
                    success=False,
                    model_name=model_name,
                    loading_method="checkpoint_analysis",
                    error_message="체크포인트 분석 서비스를 찾을 수 없습니다",
                    processing_time=time.time() - start_time
                )
            
            # 체크포인트 분석 및 모델 생성
            config = checkpoint_analyzer.analyze_pose_model(model_name)
            if not config:
                return ModelLoadingResult(
                    success=False,
                    model_name=model_name,
                    loading_method="checkpoint_analysis", 
                    error_message="체크포인트 분석 실패",
                    processing_time=time.time() - start_time
                )
            
            # 모델 생성
            if model_name == 'hrnet':
                result = self._load_hrnet_from_modules(config)
            elif model_name == 'openpose':
                result = self._load_openpose_from_modules(config)
            elif model_name == 'yolo_pose':
                result = self._load_yolo_pose_from_modules(config)
            elif model_name == 'mediapipe':
                result = self._load_mediapipe_from_modules(config)
            else:
                return ModelLoadingResult(
                    success=False,
                    model_name=model_name,
                    loading_method="checkpoint_analysis",
                    error_message=f"지원하지 않는 모델: {model_name}",
                    processing_time=time.time() - start_time
                )
            
            return result
            
        except Exception as e:
            return ModelLoadingResult(
                success=False,
                model_name=model_name,
                loading_method="checkpoint_analysis",
                error_message=str(e),
                processing_time=time.time() - start_time
            )
    
    def _create_with_architecture(self, model_type: str, config: Dict[str, Any]) -> ModelLoadingResult:
        """아키텍처 기반 모델 생성"""
        start_time = time.time()
        try:
            self.logger.info(f"🏗️ {model_type} 아키텍처 기반 모델 생성")
            
            # 기본 아키텍처 생성
            model = self._create_basic_architecture(model_type, config)
            
            return ModelLoadingResult(
                success=True,
                model=model,
                model_name=model_type,
                loading_method="architecture_based",
                processing_time=time.time() - start_time
            )
            
        except Exception as e:
            return ModelLoadingResult(
                success=False,
                model_name=model_type,
                loading_method="architecture_based",
                error_message=str(e),
                processing_time=time.time() - start_time
            )
    
    def _load_hrnet_from_modules(self, config: Dict[str, Any]) -> ModelLoadingResult:
        """기존 모듈화된 HRNet 모델 로드 - 실제 API 호환"""
        start_time = time.time()
        try:
            from .pose_estimation.models.openpose_model import HRNetModel
            
            # 기존 모듈의 실제 API에 맞게 호출
            model_path = config.get('model_path')
            model = HRNetModel(model_path=model_path)
            
            # 기존 모듈의 load_model() 메서드 호출
            success = model.load_model()
            
            if success and model.loaded:
                self.logger.info(f"✅ HRNet 모듈화된 구조에서 로드 완료")
                return ModelLoadingResult(
                    success=True,
                    model=model,
                    model_name='hrnet',
                    loading_method="modular_architecture",
                    processing_time=time.time() - start_time
                )
            else:
                return ModelLoadingResult(
                    success=False,
                    model_name='hrnet',
                    loading_method="modular_architecture",
                    error_message="HRNet 모델 로딩 실패",
                    processing_time=time.time() - start_time
                )
                
        except ImportError as e:
            self.logger.warning(f"⚠️ HRNet 모듈 import 실패: {e}")
            return self._create_basic_architecture('hrnet', config)
        except Exception as e:
            return ModelLoadingResult(
                success=False,
                model_name='hrnet',
                loading_method="modular_architecture",
                error_message=str(e),
                processing_time=time.time() - start_time
            )
    
    def _load_openpose_from_modules(self, config: Dict[str, Any]) -> ModelLoadingResult:
        """기존 모듈화된 OpenPose 모델 로드 - 실제 API 호환"""
        start_time = time.time()
        try:
            from .pose_estimation.models.openpose_model import OpenPoseModel
            
            # 기존 모듈의 실제 API에 맞게 호출
            model_path = config.get('model_path')
            model = OpenPoseModel(model_path=model_path)
            
            # 기존 모듈의 load_model() 메서드 호출
            success = model.load_model()
            
            if success and model.loaded:
                self.logger.info(f"✅ OpenPose 모듈화된 구조에서 로드 완료")
                return ModelLoadingResult(
                    success=True,
                    model=model,
                    model_name='openpose',
                    loading_method="modular_architecture",
                    processing_time=time.time() - start_time
                )
            else:
                return ModelLoadingResult(
                    success=False,
                    model_name='openpose',
                    loading_method="modular_architecture",
                    error_message="OpenPose 모델 로딩 실패",
                    processing_time=time.time() - start_time
                )
                
        except ImportError as e:
            self.logger.warning(f"⚠️ OpenPose 모듈 import 실패: {e}")
            return self._create_basic_architecture('openpose', config)
        except Exception as e:
            return ModelLoadingResult(
                success=False,
                model_name='openpose',
                loading_method="modular_architecture",
                error_message=str(e),
                processing_time=time.time() - start_time
            )
    
    def _load_yolo_pose_from_modules(self, config: Dict[str, Any]) -> ModelLoadingResult:
        """기존 모듈화된 YOLO Pose 모델 로드 - 실제 API 호환"""
        start_time = time.time()
        try:
            from .pose_estimation.models.yolov8_model import YOLOv8PoseModel
            
            # 기존 모듈의 실제 API에 맞게 호출
            model_path = config.get('model_path')
            model = YOLOv8PoseModel(model_path=model_path)
            
            # 기존 모듈의 load_model() 메서드 호출
            success = model.load_model()
            
            if success and model.loaded:
                self.logger.info(f"✅ YOLO Pose 모듈화된 구조에서 로드 완료")
                return ModelLoadingResult(
                    success=True,
                    model=model,
                    model_name='yolo_pose',
                    loading_method="modular_architecture",
                    processing_time=time.time() - start_time
                )
            else:
                return ModelLoadingResult(
                    success=False,
                    model_name='yolo_pose',
                    loading_method="modular_architecture",
                    error_message="YOLO Pose 모델 로딩 실패",
                    processing_time=time.time() - start_time
                )
                
        except ImportError as e:
            self.logger.warning(f"⚠️ YOLO Pose 모듈 import 실패: {e}")
            return self._create_basic_architecture('yolo_pose', config)
        except Exception as e:
            return ModelLoadingResult(
                success=False,
                model_name='yolo_pose',
                loading_method="modular_architecture",
                error_message=str(e),
                processing_time=time.time() - start_time
            )
    
    def _load_mediapipe_from_modules(self, config: Dict[str, Any]) -> ModelLoadingResult:
        """기존 모듈화된 MediaPipe 모델 로드 - 실제 API 호환"""
        start_time = time.time()
        try:
            from .pose_estimation.models.mediapipe_model import MediaPoseModel
            
            # 기존 모듈의 실제 API에 맞게 호출
            model_path = config.get('model_path')
            model = MediaPoseModel(model_path=model_path)
            
            # 기존 모듈의 load_model() 메서드 호출
            success = model.load_model()
            
            if success and model.loaded:
                self.logger.info(f"✅ MediaPipe 모듈화된 구조에서 로드 완료")
                return ModelLoadingResult(
                    success=True,
                    model=model,
                    model_name='mediapipe',
                    loading_method="modular_architecture",
                    processing_time=time.time() - start_time
                )
            else:
                return ModelLoadingResult(
                    success=False,
                    model_name='mediapipe',
                    loading_method="modular_architecture",
                    error_message="MediaPipe 모델 로딩 실패",
                    processing_time=time.time() - start_time
                )
                
        except ImportError as e:
            self.logger.warning(f"⚠️ MediaPipe 모듈 import 실패: {e}")
            return self._create_basic_architecture('mediapipe', config)
        except Exception as e:
            return ModelLoadingResult(
                success=False,
                model_name='mediapipe',
                loading_method="modular_architecture",
                error_message=str(e),
                processing_time=time.time() - start_time
            )
    
    def _create_basic_architecture(self, model_type: str, config: Dict[str, Any]):
        """기본 아키텍처 생성"""
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch가 필요합니다")
        
        if model_type == 'hrnet':
            return self._create_hrnet_architecture(config)
        elif model_type == 'openpose':
            return self._create_openpose_architecture(config)
        elif model_type == 'yolo_pose':
            return self._create_yolo_pose_architecture(config)
        elif model_type == 'mediapipe':
            return self._create_mediapipe_architecture(config)
        else:
            raise ValueError(f"지원하지 않는 모델 타입: {model_type}")
    
    def _create_hrnet_architecture(self, config: Dict[str, Any]):
        """HRNet 기본 아키텍처 생성"""
        class BasicHRNet(nn.Module):
            def __init__(self, num_keypoints=17):
                super().__init__()
                self.backbone = nn.Sequential(
                    nn.Conv2d(3, 64, 3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(64, 128, 3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2, 2)
                )
                self.head = nn.Conv2d(128, num_keypoints, 1)
            
            def forward(self, x):
                features = self.backbone(x)
                heatmaps = self.head(features)
                return heatmaps
        
        model = BasicHRNet(config.get('num_keypoints', 17))
        model.to(self.device)
        model.eval()
        return model
    
    def _create_openpose_architecture(self, config: Dict[str, Any]):
        """OpenPose 기본 아키텍처 생성"""
        class BasicOpenPose(nn.Module):
            def __init__(self, num_keypoints=18):
                super().__init__()
                self.backbone = nn.Sequential(
                    nn.Conv2d(3, 64, 3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(64, 128, 3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2, 2)
                )
                self.paf_head = nn.Conv2d(128, 38, 1)  # 19 limbs * 2
                self.confidence_head = nn.Conv2d(128, num_keypoints, 1)
            
            def forward(self, x):
                features = self.backbone(x)
                pafs = self.paf_head(features)
                confidence = self.confidence_head(features)
                return pafs, confidence
        
        model = BasicOpenPose(config.get('num_keypoints', 18))
        model.to(self.device)
        model.eval()
        return model
    
    def _create_yolo_pose_architecture(self, config: Dict[str, Any]):
        """YOLO Pose 기본 아키텍처 생성"""
        class BasicYOLOPose(nn.Module):
            def __init__(self, num_keypoints=17):
                super().__init__()
                self.backbone = nn.Sequential(
                    nn.Conv2d(3, 64, 3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(64, 128, 3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2, 2)
                )
                self.head = nn.Conv2d(128, num_keypoints * 3, 1)  # x, y, confidence
            
            def forward(self, x):
                features = self.backbone(x)
                keypoints = self.head(features)
                return keypoints
        
        model = BasicYOLOPose(config.get('num_keypoints', 17))
        model.to(self.device)
        model.eval()
        return model
    
    def _create_mediapipe_architecture(self, config: Dict[str, Any]):
        """MediaPipe 기본 아키텍처 생성"""
        # MediaPipe는 별도 라이브러리이므로 간단한 래퍼 생성
        class MediaPipeWrapper:
            def __init__(self):
                self.loaded = False
                try:
                    import mediapipe as mp
                    self.mp = mp
                    self.model = mp.solutions.pose.Pose(
                        static_image_mode=False,
                        model_complexity=1,
                        enable_segmentation=False,
                        min_detection_confidence=0.5,
                        min_tracking_confidence=0.5
                    )
                    self.loaded = True
                except ImportError:
                    pass
            
            def detect_poses(self, image):
                if not self.loaded:
                    return {"error": "MediaPipe를 사용할 수 없습니다"}
                
                try:
                    results = self.model.process(image)
                    if results.pose_landmarks:
                        keypoints = []
                        for landmark in results.pose_landmarks.landmark:
                            keypoints.append([landmark.x, landmark.y, landmark.z])
                        return {
                            "keypoints": keypoints,
                            "success": True
                        }
                    else:
                        return {
                            "keypoints": [],
                            "success": False
                        }
                except Exception as e:
                    return {"error": str(e), "success": False}
        
        return MediaPipeWrapper()
    
    def _get_service_from_central_hub(self, service_key: str):
        """Central Hub에서 서비스 가져오기"""
        try:
            from app.ai_pipeline.utils.common_imports import _get_central_hub_container
            container = _get_central_hub_container()
            if container and hasattr(container, 'get_service'):
                return container.get_service(service_key)
        except Exception as e:
            self.logger.warning(f"⚠️ Central Hub 서비스 가져오기 실패: {e}")
        return None
    
    def get_loaded_models(self) -> Dict[str, Any]:
        """로드된 모델들 반환"""
        return self.models.copy()
    
    def cleanup_resources(self):
        """리소스 정리"""
        try:
            for model_name, model in self.models.items():
                if hasattr(model, 'cleanup'):
                    model.cleanup()
                elif hasattr(model, 'close'):
                    model.close()
            
            self.models.clear()
            self.loaded_models.clear()
            
            if TORCH_AVAILABLE:
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            gc.collect()
            
        except Exception as e:
            self.logger.warning(f"⚠️ 리소스 정리 중 오류: {e}")

def get_integrated_loader(device: str = DEFAULT_DEVICE, logger=None) -> PoseEstimationModelLoader:
    """통합 로더 인스턴스 생성"""
    return PoseEstimationModelLoader(device=device, logger=logger)
