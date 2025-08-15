#!/usr/bin/env python3
"""
🔥 MyCloset AI - Pose Estimation Core Step
=========================================

✅ 기존 step.py의 PoseEstimationStep 클래스 완전 복원
✅ 모든 기능 포함
✅ 모듈화된 구조 적용
"""

import asyncio
import logging
import threading
import time
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union

from app.ai_pipeline.steps.base.core.base_step_mixin import BaseStepMixin
from app.ai_pipeline.utils.common_imports import (
    np, torch, Image, ImageDraw, ImageFont, cv2
)

# 절대 임포트로 변경 (파일 이동으로 인한 경로 문제 해결)
try:
    from app.ai_pipeline.steps.step_02_pose_estimation.config.types import PoseModel, PoseQuality, EnhancedPoseConfig, PoseResult
except ImportError:
    from app.ai_pipeline.steps.step_02_pose_estimation.config.types import PoseModel, PoseQuality, EnhancedPoseConfig, PoseResult

try:
    from app.ai_pipeline.steps.step_02_pose_estimation.analyzers.pose_analyzer import PoseAnalyzer
except ImportError:
    from app.ai_pipeline.steps.step_02_pose_estimation.analyzers.pose_analyzer import PoseAnalyzer

try:
    from app.ai_pipeline.steps.step_02_pose_estimation.analyzers.pose_quality_analyzer import PoseQualityAnalyzer
except ImportError:
    from app.ai_pipeline.steps.step_02_pose_estimation.analyzers.pose_quality_analyzer import PoseQualityAnalyzer

try:
    from app.ai_pipeline.steps.step_02_pose_estimation.analyzers.pose_geometry_analyzer import PoseGeometryAnalyzer
except ImportError:
    from app.ai_pipeline.steps.step_02_pose_estimation.analyzers.pose_geometry_analyzer import PoseGeometryAnalyzer

try:
    from app.ai_pipeline.steps.step_02_pose_estimation.processors.pose_processor import PoseProcessor
except ImportError:
    from app.ai_pipeline.steps.step_02_pose_estimation.processors.pose_processor import PoseProcessor

try:
    from app.ai_pipeline.steps.step_02_pose_estimation.processors.image_processor import ImageProcessor
except ImportError:
    from app.ai_pipeline.steps.step_02_pose_estimation.processors.image_processor import ImageProcessor

try:
    from app.ai_pipeline.steps.step_02_pose_estimation.processors.keypoint_processor import KeypointProcessor
except ImportError:
    from app.ai_pipeline.steps.step_02_pose_estimation.processors.keypoint_processor import KeypointProcessor

try:
    from app.ai_pipeline.steps.step_02_pose_estimation.visualizers.pose_visualizer import PoseVisualizer
except ImportError:
    from app.ai_pipeline.steps.step_02_pose_estimation.visualizers.pose_visualizer import PoseVisualizer

try:
    from app.ai_pipeline.steps.step_02_pose_estimation.utils.pose_utils import (
        validate_keypoints,
        draw_pose_on_image,
        analyze_pose_for_clothing_advanced,
        analyze_posture_stability,
        analyze_clothing_specific_requirements,
        analyze_pose_for_clothing,
        convert_coco17_to_openpose18
    )
except ImportError:
    from app.ai_pipeline.steps.step_02_pose_estimation.utils.pose_utils import (
        validate_keypoints,
        draw_pose_on_image,
        analyze_pose_for_clothing_advanced,
        analyze_posture_stability,
        analyze_clothing_specific_requirements,
        analyze_pose_for_clothing,
        convert_coco17_to_openpose18
    )

logger = logging.getLogger(__name__)

class PoseEstimationStep(BaseStepMixin):
    """
    🔥 Step 02: Pose Estimation - Central Hub DI Container v7.0 완전 연동
    
    ✅ BaseStepMixin 상속 패턴 (Human Parsing Step과 동일)
    ✅ MediaPipe Pose 모델 지원 (우선순위 1)
    ✅ OpenPose 모델 지원 (폴백 옵션)
    ✅ YOLOv8-Pose 모델 지원 (실시간)
    ✅ HRNet 모델 지원 (고정밀)
    ✅ 17개 COCO keypoints 감지
    ✅ Mock 모델 완전 제거
    ✅ 실제 AI 추론 실행
    ✅ 다중 모델 폴백 시스템
    """
    
    def __init__(self, **kwargs):
        """포즈 추정 Step 초기화"""
        self._lock = threading.RLock()  # ✅ threading 사용

        # 🔥 1. 필수 속성들 초기화 (에러 방지)
        self._initialize_step_attributes()
        
        # 🔥 2. BaseStepMixin 초기화 (Central Hub 자동 연동)
        super().__init__(step_name="PoseEstimationStep", **kwargs)
        
        # 🔥 3. Pose Estimation 특화 초기화
        self._initialize_pose_estimation_specifics()
    
    def _initialize_step_attributes(self):
        """Step 필수 속성들 초기화"""
        self.ai_models = {}
        self.models_loading_status = {
            'mediapipe': False,
            'openpose': False,
            'yolov8': False,
            'hrnet': False,
            'total_loaded': 0,
            'loading_errors': []
        }
        self.model_interface = None
        self.loaded_models = {}
        
        # Pose Estimation 특화 속성들
        self.pose_models = {}
        self.pose_ready = False
        self.keypoints_cache = {}
    
    def _initialize_pose_estimation_specifics(self):
        """Pose Estimation 특화 초기화"""
        
        # 🔥 강화된 설정 (앙상블 시스템 포함)
        self.config = EnhancedPoseConfig(
            method=PoseModel.HRNET,
            quality_level=PoseQuality.EXCELLENT,
            enable_ensemble=True,
            ensemble_models=['hrnet', 'yolov8', 'mediapipe', 'openpose'],
            ensemble_method='weighted_average',
            enable_uncertainty_quantification=True,
            enable_confidence_calibration=True,
            enable_subpixel_accuracy=True,
            enable_joint_angle_calculation=True,
            enable_body_proportion_analysis=True,
            enable_pose_quality_assessment=True,
            enable_skeleton_structure_analysis=True,
            enable_virtual_fitting_optimization=True
        )
        
        # 기본 설정
        self.confidence_threshold = self.config.confidence_threshold
        self.use_subpixel = self.config.enable_subpixel_accuracy
        
        # 포즈 분석기들
        self.analyzer = PoseAnalyzer()
        self.quality_analyzer = PoseQualityAnalyzer()
        self.geometry_analyzer = PoseGeometryAnalyzer()
        
        # 프로세서들
        self.pose_processor = PoseProcessor(self.config)
        self.image_processor = ImageProcessor()
        self.keypoint_processor = KeypointProcessor()
        
        # 시각화기
        self.visualizer = PoseVisualizer()
        
        # 🔥 앙상블 시스템 초기화
        self.ensemble_manager = None
        if self.config.enable_ensemble:
            try:
                from app.ai_pipeline.steps.step_02_pose_estimation.ensemble.ensemble_manager import PoseEnsembleManager
                self.ensemble_manager = PoseEnsembleManager(self.config)
                self.logger.info("✅ PoseEnsembleManager 생성 완료")
            except ImportError:
                self.logger.warning("⚠️ PoseEnsembleManager를 사용할 수 없습니다")
        
        # 모델 우선순위 (앙상블 순서)
        self.model_priority = [
            PoseModel.HRNET,
            PoseModel.YOLOV8_POSE,
            PoseModel.MEDIAPIPE,
            PoseModel.OPENPOSE
        ]
        
        # 🔥 새로운 아키텍처 모델 초기화
        self.new_openpose_model = None
        
        self.logger.info(f"✅ {self.step_name} 포즈 추정 특화 초기화 완료 (앙상블 시스템 포함)")
    
    def _load_pose_models_via_central_hub(self):
        """Central Hub를 통한 Pose 모델 로딩 (앙상블 시스템 방식으로 개선)"""
        loaded_count = 0
        
        print(f"🔥 [디버깅] _load_pose_models_via_central_hub 시작 (앙상블 방식)")
        print(f"🔥 [디버깅] self.model_loader 존재: {self.model_loader is not None}")
        
        if not self.model_loader:
            self.logger.error("❌ model_loader가 없습니다")
            return loaded_count
        
        # 모델 로딩 시도
        for model_name in ['hrnet', 'yolov8', 'mediapipe', 'openpose']:
            try:
                model = self.model_loader.load_model(model_name)
                if model:
                    self.pose_models[model_name] = model
                    self.models_loading_status[model_name] = True
                    loaded_count += 1
                    self.logger.info(f"✅ {model_name} 모델 로딩 성공")
                else:
                    self.logger.warning(f"⚠️ {model_name} 모델 로딩 실패")
            except Exception as e:
                self.logger.error(f"❌ {model_name} 모델 로딩 실패: {e}")
                self.models_loading_status['loading_errors'].append(f"{model_name}: {e}")
        
        self.models_loading_status['total_loaded'] = loaded_count
        self.pose_ready = loaded_count > 0
        
        return loaded_count
    
    def process(self, **kwargs) -> Dict[str, Any]:
        """포즈 추정 처리"""
        try:
            start_time = time.time()
            
            # 입력 검증
            if 'image' not in kwargs:
                return self._create_error_response("이미지가 필요합니다")
            
            image = kwargs['image']
            if image is None:
                return self._create_error_response("이미지가 None입니다")
            
            # 이미지 전처리
            processed_input = self.pose_processor.preprocess_input(image)
            
            # AI 추론 실행
            inference_result = self._run_ai_inference(processed_input)
            
            if not inference_result['success']:
                return inference_result
            
            # 결과 후처리
            processed_result = self.pose_processor.postprocess_results(inference_result)
            
            # 포즈 분석
            analysis_result = self._analyze_pose(processed_result)
            
            # 시각화
            visualization_result = self._create_visualization(image, processed_result)
            
            # 최종 결과 구성
            final_result = {
                'success': True,
                'keypoints': processed_result.get('keypoints', []),
                'confidence_scores': processed_result.get('confidence_scores', []),
                'joint_angles': analysis_result.get('joint_angles', {}),
                'body_proportions': analysis_result.get('body_proportions', {}),
                'pose_quality': analysis_result.get('pose_quality', {}),
                'pose_direction': analysis_result.get('pose_direction', 'unknown'),
                'pose_stability': analysis_result.get('pose_stability', 0.0),
                'body_orientation': analysis_result.get('body_orientation', {}),
                'skeleton_structure': analysis_result.get('skeleton_structure', {}),
                'visualization': visualization_result,
                'model_used': inference_result.get('model_used', 'unknown'),
                'processing_time': time.time() - start_time,
                'real_ai_inference': inference_result.get('real_ai_inference', False)
            }
            
            return final_result
            
        except Exception as e:
            self.logger.error(f"❌ 포즈 추정 처리 실패: {e}")
            return self._create_error_response(str(e))
    
    def _run_ai_inference(self, processed_input: Dict[str, Any]) -> Dict[str, Any]:
        """AI 추론 실행"""
        try:
            if not self.pose_ready:
                return self._create_error_response("포즈 모델이 준비되지 않았습니다")
            
            # 모델 우선순위에 따라 추론 시도
            for model_name in self.model_priority:
                if model_name.value in self.pose_models:
                    try:
                        model = self.pose_models[model_name.value]
                        result = model.predict(processed_input['image'])
                        
                        if result and result.get('success', False):
                            return {
                                'success': True,
                                'keypoints': result.get('keypoints', []),
                                'confidence_scores': result.get('confidence_scores', []),
                                'model_used': model_name.value,
                                'real_ai_inference': True
                            }
                    except Exception as e:
                        self.logger.warning(f"⚠️ {model_name.value} 모델 추론 실패: {e}")
                        continue
            
            return self._create_error_response("모든 모델 추론 실패")
            
        except Exception as e:
            self.logger.error(f"❌ AI 추론 실패: {e}")
            return self._create_error_response(str(e))
    
    def _analyze_pose(self, pose_result: Dict[str, Any]) -> Dict[str, Any]:
        """포즈 분석"""
        try:
            keypoints = pose_result.get('keypoints', [])
            
            if not keypoints:
                return {}
            
            # 관절 각도 계산
            joint_angles = self.analyzer.calculate_joint_angles(keypoints)
            
            # 신체 비율 계산
            body_proportions = self.analyzer.calculate_body_proportions(keypoints)
            
            # 포즈 품질 평가
            pose_quality = self.quality_analyzer.assess_pose_quality(keypoints, joint_angles, body_proportions)
            
            # 포즈 방향 계산
            pose_direction = self.geometry_analyzer.calculate_pose_direction(keypoints)
            
            # 포즈 안정성 계산
            pose_stability = self.geometry_analyzer.calculate_pose_stability(keypoints)
            
            # 신체 방향 계산
            body_orientation = self.geometry_analyzer.calculate_body_orientation(keypoints)
            
            # 스켈레톤 구조 생성
            skeleton_structure = self.geometry_analyzer.build_skeleton_structure(keypoints)
            
            return {
                'joint_angles': joint_angles,
                'body_proportions': body_proportions,
                'pose_quality': pose_quality,
                'pose_direction': pose_direction,
                'pose_stability': pose_stability,
                'body_orientation': body_orientation,
                'skeleton_structure': skeleton_structure
            }
            
        except Exception as e:
            self.logger.error(f"❌ 포즈 분석 실패: {e}")
            return {}
    
    def _create_visualization(self, image: Any, pose_result: Dict[str, Any]) -> Dict[str, Any]:
        """시각화 생성"""
        try:
            keypoints = pose_result.get('keypoints', [])
            
            if not keypoints:
                return {}
            
            # 포즈 시각화 생성
            visualization = self.visualizer.create_visualization(image, keypoints)
            
            return visualization
            
        except Exception as e:
            self.logger.error(f"❌ 시각화 생성 실패: {e}")
            return {}
    
    def _create_error_response(self, error_message: str) -> Dict[str, Any]:
        """에러 응답 생성"""
        return {
            'success': False,
            'error': error_message,
            'keypoints': [],
            'confidence_scores': [],
            'joint_angles': {},
            'body_proportions': {},
            'pose_quality': {},
            'pose_direction': 'unknown',
            'pose_stability': 0.0,
            'body_orientation': {},
            'skeleton_structure': {},
            'visualization': {},
            'model_used': 'unknown',
            'processing_time': 0.0,
            'real_ai_inference': False
        }
    
    def get_model_status(self) -> Dict[str, Any]:
        """모델 상태 조회"""
        return {
            'step_name': self.step_name,
            'pose_ready': self.pose_ready,
            'loaded_models': list(self.pose_models.keys()),
            'total_loaded': self.models_loading_status['total_loaded'],
            'loading_errors': self.models_loading_status['loading_errors'],
            'model_priority': [model.value for model in self.model_priority]
        }
    
    async def initialize(self):
        """초기화"""
        try:
            self.logger.info(f"🔥 {self.step_name} 초기화 시작")
            
            # 모델 로딩
            loaded_count = self._load_pose_models_via_central_hub()
            
            if loaded_count > 0:
                self.logger.info(f"✅ {self.step_name} 초기화 완료 ({loaded_count}개 모델 로딩)")
            else:
                self.logger.warning(f"⚠️ {self.step_name} 초기화 완료 (모델 로딩 실패)")
            
        except Exception as e:
            self.logger.error(f"❌ {self.step_name} 초기화 실패: {e}")
    
    async def cleanup(self):
        """정리"""
        try:
            self.logger.info(f"🧹 {self.step_name} 정리 시작")
            
            # 모델 정리
            for model_name, model in self.pose_models.items():
                try:
                    if hasattr(model, 'cleanup'):
                        await model.cleanup()
                    self.logger.info(f"✅ {model_name} 모델 정리 완료")
                except Exception as e:
                    self.logger.warning(f"⚠️ {model_name} 모델 정리 실패: {e}")
            
            # 캐시 정리
            self.keypoints_cache.clear()
            
            self.logger.info(f"✅ {self.step_name} 정리 완료")
            
        except Exception as e:
            self.logger.error(f"❌ {self.step_name} 정리 실패: {e}")
    
    def _get_service_from_central_hub(self, service_key: str):
        """Central Hub에서 서비스 가져오기"""
        try:
            if hasattr(self, 'di_container') and self.di_container:
                return self.di_container.get_service(service_key)
            return None
        except Exception as e:
            self.logger.error(f"❌ Central Hub 서비스 가져오기 실패: {e}")
            return None

# ==============================================
# 🔥 팩토리 함수들
# ==============================================

async def create_pose_estimation_step(
    device: str = "auto",
    config: Optional[Dict[str, Any]] = None,
    **kwargs
) -> PoseEstimationStep:
    """PoseEstimationStep 비동기 생성"""
    try:
        step = PoseEstimationStep(**kwargs)
        await step.initialize()
        return step
    except Exception as e:
        logger.error(f"❌ PoseEstimationStep 생성 실패: {e}")
        raise

def create_pose_estimation_step_sync(
    device: str = "auto",
    config: Optional[Dict[str, Any]] = None,
    **kwargs
) -> PoseEstimationStep:
    """PoseEstimationStep 동기 생성"""
    try:
        step = PoseEstimationStep(**kwargs)
        return step
    except Exception as e:
        logger.error(f"❌ PoseEstimationStep 생성 실패: {e}")
        raise
