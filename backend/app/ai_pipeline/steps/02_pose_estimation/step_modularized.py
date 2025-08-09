#!/usr/bin/env python3
"""
🔥 MyCloset AI - Step 02: Pose Estimation - Modularized Version
================================================================

✅ 기존 step.py 기능 그대로 보존
✅ 분리된 모듈들 사용 (config/, models/, ensemble/, utils/, processors/, analyzers/)
✅ 모듈화된 구조 적용
✅ 중복 코드 제거
✅ 유지보수성 향상

파일 위치: backend/app/ai_pipeline/steps/02_pose_estimation/step_modularized.py
작성자: MyCloset AI Team  
날짜: 2025-08-01
버전: v8.0 (Modularized)
"""

# 🔥 공통 imports 시스템 사용
from app.ai_pipeline.utils.common_imports import (
    # 표준 라이브러리
    os, sys, gc, time, asyncio, logging, threading, traceback,
    hashlib, json, base64, math, warnings, np,
    Path, Dict, Any, Optional, Tuple, List, Union, Callable, TYPE_CHECKING,
    dataclass, field, Enum, IntEnum, BytesIO, ThreadPoolExecutor,
    lru_cache, wraps, asynccontextmanager,
    
    # 에러 처리 시스템
    MyClosetAIException, ModelLoadingError, ImageProcessingError, DataValidationError, ConfigurationError,
    error_tracker, track_exception, get_error_summary, create_exception_response, convert_to_mycloset_exception,
    ErrorCodes, EXCEPTIONS_AVAILABLE,
    
    # Mock Data Diagnostic
    detect_mock_data, diagnose_step_data, MOCK_DIAGNOSTIC_AVAILABLE,
    
    # AI/ML 라이브러리
    torch, nn, F, transforms, TORCH_AVAILABLE, MPS_AVAILABLE,
    Image, cv2, scipy,
    PIL_AVAILABLE, CV2_AVAILABLE, SCIPY_AVAILABLE,
    
    # MediaPipe 및 기타 라이브러리
    MEDIAPIPE_AVAILABLE, mp, ULTRALYTICS_AVAILABLE, YOLO,
    
    # 유틸리티 함수
    detect_m3_max, get_available_libraries, log_library_status,
    
    # 상수
    DEVICE_CPU, DEVICE_CUDA, DEVICE_MPS,
    DEFAULT_INPUT_SIZE, DEFAULT_CONFIDENCE_THRESHOLD, DEFAULT_QUALITY_THRESHOLD,
    
    # Central Hub DI Container
    _get_central_hub_container
)

# 🔥 분리된 모듈들 import
from .config import (
    PoseModel, PoseQuality, EnhancedPoseConfig, PoseResult,
    COCO_17_KEYPOINTS, OPENPOSE_18_KEYPOINTS, SKELETON_CONNECTIONS, KEYPOINT_COLORS
)

from .models import (
    MediaPoseModel, YOLOv8PoseModel, OpenPoseModel, HRNetModel
)

from .ensemble import (
    PoseEnsembleSystem, PoseEnsembleManager
)

from .utils import (
    draw_pose_on_image, analyze_pose_for_clothing, 
    convert_coco17_to_openpose18, validate_keypoints
)

from .processors import PoseProcessor
from .analyzers import PoseAnalyzer

# BaseStepMixin import
from app.ai_pipeline.steps.base_step_mixin import BaseStepMixin

# 경고 무시 설정
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=ImportWarning)

logger = logging.getLogger(__name__)

# ==============================================
# 🔥 PoseEstimationStep - 모듈화된 버전
# ==============================================

class PoseEstimationStep(BaseStepMixin):
    """
    🔥 Pose Estimation Step - 모듈화된 버전
    
    ✅ 기존 step.py 기능 그대로 보존
    ✅ 분리된 모듈들 사용
    ✅ 중복 코드 제거
    ✅ 유지보수성 향상
    """
    
    def __init__(self, **kwargs):
        """초기화"""
        super().__init__(**kwargs)
        
        # Step 기본 정보
        self.step_name = "pose_estimation"
        self.step_id = 2
        self.step_description = "포즈 추정 - 17개 COCO keypoints 감지"
        
        # 설정 초기화
        self.config = EnhancedPoseConfig()
        
        # 모듈화된 컴포넌트들 초기화
        self.processor = PoseProcessor(self.config)
        self.analyzer = PoseAnalyzer()
        
        # 모델들 초기화
        self.models = {}
        self.ensemble_manager = None
        
        # 상태 관리
        self.models_loading_status = {}
        self.loaded_models = {}
        
        # 성능 통계
        self.performance_stats = {
            'total_processed': 0,
            'successful_processed': 0,
            'failed_processed': 0,
            'average_processing_time': 0.0,
            'last_processing_time': None
        }
        
        # 초기화 완료
        self._initialize_step_attributes()
        self._initialize_pose_estimation_specifics()
        
        logger.info(f"✅ PoseEstimationStep 초기화 완료 (버전: v8.0 - Modularized)")
    
    def _initialize_step_attributes(self):
        """Step 기본 속성 초기화"""
        try:
            # Central Hub Container 연결
            self.central_hub_container = _get_central_hub_container()
            
            # 기본 설정
            self.device = DEVICE_MPS if TORCH_AVAILABLE and MPS_AVAILABLE else DEVICE_CPU
            self.input_size = self.config.input_size
            self.confidence_threshold = self.config.confidence_threshold
            
            # 모델 로딩 상태 초기화
            self.models_loading_status = {
                'mediapipe': False,
                'yolov8': False,
                'openpose': False,
                'hrnet': False
            }
            
            logger.info(f"✅ Step 기본 속성 초기화 완료")
            
        except Exception as e:
            logger.error(f"❌ Step 기본 속성 초기화 실패: {e}")
            if EXCEPTIONS_AVAILABLE:
                error = ConfigurationError(f"Step 기본 속성 초기화 실패: {e}", ErrorCodes.CONFIGURATION_ERROR)
                track_exception(error, {'step': self.step_name}, 2)
    
    def _initialize_pose_estimation_specifics(self):
        """Pose Estimation 특화 초기화"""
        try:
            # 앙상블 매니저 초기화
            if self.config.enable_ensemble:
                self.ensemble_manager = PoseEnsembleManager(self.config)
                logger.info("✅ 앙상블 매니저 초기화 완료")
            
            # 모델들 초기화
            self.models = {
                'mediapipe': MediaPoseModel(),
                'yolov8': YOLOv8PoseModel(),
                'openpose': OpenPoseModel(),
                'hrnet': HRNetModel()
            }
            
            logger.info(f"✅ Pose Estimation 특화 초기화 완료")
            
        except Exception as e:
            logger.error(f"❌ Pose Estimation 특화 초기화 실패: {e}")
            if EXCEPTIONS_AVAILABLE:
                error = ConfigurationError(f"Pose Estimation 특화 초기화 실패: {e}", ErrorCodes.CONFIGURATION_ERROR)
                track_exception(error, {'step': self.step_name}, 2)
    
    def process(self, **kwargs) -> Dict[str, Any]:
        """
        🔥 Pose Estimation 처리 - 모듈화된 버전
        
        Args:
            **kwargs: 입력 데이터 (이미지, 설정 등)
            
        Returns:
            Dict[str, Any]: 포즈 추정 결과
        """
        start_time = time.time()
        
        try:
            logger.info(f"🔥 Pose Estimation 처리 시작 (버전: v8.0 - Modularized)")
            
            # 입력 데이터 검증 및 변환 (분리된 processor 사용)
            processed_input = self.processor.preprocess_input(kwargs)
            if not processed_input:
                return self._create_error_response("입력 데이터 처리 실패")
            
            # AI 추론 실행
            inference_result = self._run_ai_inference(processed_input)
            if not inference_result or 'error' in inference_result:
                return self._create_error_response(inference_result.get('error', 'AI 추론 실패'))
            
            # 결과 분석 (분리된 analyzer 사용)
            analysis_result = self.analyzer.analyze_pose(inference_result)
            
            # 결과 후처리 (분리된 processor 사용)
            final_result = self.processor.postprocess_results(inference_result, analysis_result, processed_input)
            
            # 성능 통계 업데이트
            processing_time = time.time() - start_time
            self._update_performance_stats(processing_time, True)
            
            logger.info(f"✅ Pose Estimation 처리 완료 (시간: {processing_time:.2f}초)")
            
            return final_result
            
        except Exception as e:
            processing_time = time.time() - start_time
            self._update_performance_stats(processing_time, False)
            
            logger.error(f"❌ Pose Estimation 처리 실패: {e}")
            if EXCEPTIONS_AVAILABLE:
                error = ImageProcessingError(f"Pose Estimation 처리 실패: {e}", ErrorCodes.IMAGE_PROCESSING_FAILED)
                track_exception(error, {'step': self.step_name, 'processing_time': processing_time}, 2)
            
            return self._create_error_response(str(e))
    
    def _run_ai_inference(self, processed_input: Dict[str, Any]) -> Dict[str, Any]:
        """AI 추론 실행"""
        try:
            image = processed_input.get('image')
            if image is None:
                return {'error': '이미지가 없습니다'}
            
            # 앙상블 모드인 경우
            if self.config.enable_ensemble and self.ensemble_manager:
                logger.info("🔥 앙상블 모드로 추론 실행")
                return self.ensemble_manager.run_ensemble_inference(image, self.device)
            
            # 단일 모델 모드
            model_name = self.config.method.value
            if model_name in self.models and self.models_loading_status.get(model_name, False):
                logger.info(f"🔥 {model_name} 모델로 추론 실행")
                model = self.models[model_name]
                if hasattr(model, 'detect_poses'):
                    return model.detect_poses(image)
                else:
                    return {'error': f'{model_name} 모델에 detect_poses 메서드가 없습니다'}
            else:
                # 폴백: MediaPipe 사용
                logger.info("🔄 MediaPipe 폴백 모델 사용")
                if 'mediapipe' in self.models:
                    return self.models['mediapipe'].detect_poses(image)
                else:
                    return {'error': '사용 가능한 모델이 없습니다'}
                    
        except Exception as e:
            logger.error(f"❌ AI 추론 실행 실패: {e}")
            return {'error': str(e)}
    
    def _update_performance_stats(self, processing_time: float, success: bool):
        """성능 통계 업데이트"""
        try:
            self.performance_stats['total_processed'] += 1
            
            if success:
                self.performance_stats['successful_processed'] += 1
            else:
                self.performance_stats['failed_processed'] += 1
            
            # 평균 처리 시간 업데이트
            total = self.performance_stats['total_processed']
            current_avg = self.performance_stats['average_processing_time']
            
            self.performance_stats['average_processing_time'] = (
                (current_avg * (total - 1) + processing_time) / total
            )
            
            self.performance_stats['last_processing_time'] = time.time()
            
        except Exception as e:
            logger.debug(f"성능 통계 업데이트 실패: {e}")
    
    def _create_error_response(self, error_message: str) -> Dict[str, Any]:
        """에러 응답 생성"""
        return {
            'success': False,
            'error': error_message,
            'step_name': self.step_name,
            'step_id': self.step_id,
            'processing_time': 0.0
        }
    
    def get_model_status(self) -> Dict[str, Any]:
        """모델 상태 조회"""
        return {
            'step_name': self.step_name,
            'models_loading_status': self.models_loading_status,
            'ensemble_enabled': self.config.enable_ensemble,
            'device_used': self.device,
            'performance_stats': self.performance_stats
        }
    
    async def initialize(self):
        """비동기 초기화"""
        try:
            logger.info("🔄 PoseEstimationStep 비동기 초기화 시작")
            
            # 모델들 로딩
            self._load_pose_models_via_central_hub()
            
            logger.info("✅ PoseEstimationStep 비동기 초기화 완료")
            
        except Exception as e:
            logger.error(f"❌ PoseEstimationStep 비동기 초기화 실패: {e}")
    
    def _load_pose_models_via_central_hub(self):
        """Central Hub를 통한 모델 로딩"""
        try:
            logger.info("🔥 Central Hub를 통한 Pose 모델들 로딩 시작")
            
            # Central Hub에서 ModelLoader 조회
            model_loader = self._get_service_from_central_hub('model_loader')
            if not model_loader:
                logger.warning("⚠️ Central Hub에서 ModelLoader를 찾을 수 없음 - 직접 로딩 시도")
                return self._load_models_directly()
            
            # 각 모델 로딩
            for model_name, model in self.models.items():
                try:
                    if hasattr(model, 'load_model'):
                        success = model.load_model()
                        self.models_loading_status[model_name] = success
                        if success:
                            logger.info(f"✅ {model_name} 모델 로딩 성공")
                        else:
                            logger.warning(f"⚠️ {model_name} 모델 로딩 실패")
                    else:
                        logger.warning(f"⚠️ {model_name} 모델에 load_model 메서드가 없음")
                except Exception as e:
                    logger.error(f"❌ {model_name} 모델 로딩 중 오류: {e}")
                    self.models_loading_status[model_name] = False
            
            # 앙상블 매니저 로딩
            if self.ensemble_manager:
                try:
                    self.ensemble_manager.load_ensemble_models(model_loader)
                    logger.info("✅ 앙상블 매니저 모델 로딩 완료")
                except Exception as e:
                    logger.error(f"❌ 앙상블 매니저 모델 로딩 실패: {e}")
            
            logger.info("🔥 Central Hub를 통한 Pose 모델들 로딩 완료")
            
        except Exception as e:
            logger.error(f"❌ Central Hub를 통한 모델 로딩 실패: {e}")
            if EXCEPTIONS_AVAILABLE:
                error = ModelLoadingError(f"Central Hub를 통한 모델 로딩 실패: {e}", ErrorCodes.MODEL_LOADING_FAILED)
                track_exception(error, {'step': self.step_name}, 2)
    
    def _load_models_directly(self):
        """직접 모델 로딩 (폴백)"""
        try:
            logger.info("🔄 직접 모델 로딩 시작 (폴백)")
            
            for model_name, model in self.models.items():
                try:
                    if hasattr(model, 'load_model'):
                        success = model.load_model()
                        self.models_loading_status[model_name] = success
                        if success:
                            logger.info(f"✅ {model_name} 모델 직접 로딩 성공")
                        else:
                            logger.warning(f"⚠️ {model_name} 모델 직접 로딩 실패")
                except Exception as e:
                    logger.error(f"❌ {model_name} 모델 직접 로딩 실패: {e}")
                    self.models_loading_status[model_name] = False
            
            logger.info("🔄 직접 모델 로딩 완료")
            
        except Exception as e:
            logger.error(f"❌ 직접 모델 로딩 실패: {e}")
    
    def _get_service_from_central_hub(self, service_key: str):
        """Central Hub에서 서비스 조회"""
        try:
            if self.central_hub_container:
                return self.central_hub_container.get_service(service_key)
            return None
        except Exception as e:
            logger.debug(f"Central Hub 서비스 조회 실패: {e}")
            return None
    
    async def cleanup(self):
        """정리"""
        try:
            logger.info("🧹 PoseEstimationStep 정리 시작")
            
            # 모델들 정리
            for model_name, model in self.models.items():
                if hasattr(model, 'cleanup'):
                    try:
                        model.cleanup()
                        logger.info(f"✅ {model_name} 모델 정리 완료")
                    except Exception as e:
                        logger.warning(f"⚠️ {model_name} 모델 정리 실패: {e}")
            
            # 앙상블 매니저 정리
            if self.ensemble_manager and hasattr(self.ensemble_manager, 'cleanup'):
                try:
                    self.ensemble_manager.cleanup()
                    logger.info("✅ 앙상블 매니저 정리 완료")
                except Exception as e:
                    logger.warning(f"⚠️ 앙상블 매니저 정리 실패: {e}")
            
            logger.info("✅ PoseEstimationStep 정리 완료")
            
        except Exception as e:
            logger.error(f"❌ PoseEstimationStep 정리 실패: {e}")

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

# ==============================================
# 🔥 모듈 초기화
# ==============================================

logger.info("✅ PoseEstimationStep 모듈화된 버전 로드 완료 (버전: v8.0 - Modularized)")
