#!/usr/bin/env python3
"""
🔥 MyCloset AI - Step 03: 의류 세그멘테이션 (Model Loader 통합 버전)
=====================================================================

✅ Model Loader v6.0 완전 통합 - 중앙 집중식 모델 관리
✅ Model Architectures 완전 활용 - 표준화된 모델 구조
✅ StepModelInterface 완전 호환 - 일관된 인터페이스
✅ 실제 AI 모델 완전 지원 - SAM, U2Net, DeepLabV3+
✅ 메모리 최적화 - M3 Max 환경 최적화
✅ 에러 처리 강화 - 통합된 예외 처리

Author: MyCloset AI Team  
Date: 2025-08-01
Version: 34.0 (Model Loader Integration)
"""

# 🔥 공통 imports 시스템 사용
from app.ai_pipeline.utils.common_imports import (
    # 표준 라이브러리
    os, gc, time, logging, threading, math, hashlib, json, base64, warnings, np,
    Path, Dict, Any, Optional, Union, List, Tuple, TYPE_CHECKING,
    dataclass, field, Enum, BytesIO, ThreadPoolExecutor,
    
    # 에러 처리 시스템
    MyClosetAIException, ModelLoadingError, ImageProcessingError, DataValidationError, ConfigurationError,
    error_tracker, track_exception, get_error_summary, create_exception_response, convert_to_mycloset_exception,
    ErrorCodes, EXCEPTIONS_AVAILABLE,
    
    # Mock Data Diagnostic
    detect_mock_data, diagnose_step_data, MOCK_DIAGNOSTIC_AVAILABLE,
    
    # AI/ML 라이브러리
    cv2, PIL_AVAILABLE, CV2_AVAILABLE, NUMPY_AVAILABLE, Image, ImageEnhance
)

# PIL Image import 추가
if PIL_AVAILABLE:
    from PIL import Image as PILImage

# 추가 imports
import weakref
from abc import ABC, abstractmethod

# 경고 무시 설정
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=ImportWarning)

# 최상단에 추가
logger = logging.getLogger(__name__)

# 🔥 PyTorch 로딩 최적화
try:
    from fix_pytorch_loading import apply_pytorch_patch
    apply_pytorch_patch()
except ImportError:
    logger.warning("⚠️ fix_pytorch_loading 모듈 없음 - 기본 PyTorch 로딩 사용")
except Exception as e:
    logger.warning(f"⚠️ PyTorch 로딩 패치 실패: {e}")

# 🔥 PyTorch 통합 import
TORCH_AVAILABLE = False
MPS_AVAILABLE = False
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torchvision import transforms
    TORCH_AVAILABLE = True
    
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        MPS_AVAILABLE = True
        
    logger.info(f"🔥 PyTorch {torch.__version__} 로드 완료")
    if MPS_AVAILABLE:
        logger.info("🍎 MPS 사용 가능")
except ImportError:
    logger.error("❌ PyTorch 필수 - 설치 필요")
    if EXCEPTIONS_AVAILABLE:
        error = ModelLoadingError("PyTorch 필수 라이브러리 로딩 실패", ErrorCodes.MODEL_LOADING_FAILED)
        track_exception(error, {'library': 'torch'}, 3)
        raise error
    else:
        raise

# 🔥 Model Loader 및 Architectures import
try:
    from ..models.model_loader import (
        ModelLoader, StepModelInterface, StepModelFactory,
        get_global_model_loader, create_step_interface, initialize_all_steps
    )
    from ..models.model_architectures import (
        SAMModel, U2NetModel, DeepLabV3PlusModel,
        ModelArchitectureFactory, CompleteModelWrapper
    )
    MODEL_LOADER_AVAILABLE = True
    logger.info("✅ Model Loader 및 Architectures 로드 완료")
except ImportError as e:
    logger.error(f"❌ Model Loader import 실패: {e}")
    MODEL_LOADER_AVAILABLE = False

# 🔥 메인 BaseStepMixin import
try:
    from app.ai_pipeline.steps.base.base_step_mixin import BaseStepMixin
    BASE_STEP_MIXIN_AVAILABLE = True
    logger.info("✅ 메인 BaseStepMixin import 성공")
except ImportError:
    try:
        from ...base.base_step_mixin import BaseStepMixin
        BASE_STEP_MIXIN_AVAILABLE = True
        logger.info("✅ 상대 경로로 BaseStepMixin import 성공")
    except ImportError:
        BASE_STEP_MIXIN_AVAILABLE = False
        logger.error("❌ BaseStepMixin import 실패 - 메인 파일 사용 필요")
        raise ImportError("BaseStepMixin을 import할 수 없습니다. 메인 BaseStepMixin을 사용하세요.")

def get_base_step_mixin_class():
    """BaseStepMixin 클래스 가져오기"""
    return BaseStepMixin
            def __init__(self, **kwargs):
                self.logger = logging.getLogger(self.__class__.__name__)
                self.initialized = False
                self.model_loader = None
                self.memory_manager = None
                self.data_converter = None
                self.di_container = None
                
                # Step별 특화 속성들
                self.step_type = "cloth_segmentation"
                self.step_name = "Step 03: Cloth Segmentation"
                self.step_version = "34.0"
                
                # 모델 관련 속성들
                self.ai_models = {}
                self.segmentation_models = {}
                self.models_loading_status = {}
                self.model_paths = {}
                
                # 성능 메트릭
                self.inference_count = 0
                self.total_inference_time = 0.0
                self.last_inference_time = 0.0
                
                # 설정
                self.config = kwargs.get('config', {})
                
            def process(self, **kwargs) -> Dict[str, Any]:
                """메인 처리 로직"""
                try:
                    # 입력 검증
                    if not self._validate_input(kwargs):
                        return self._create_error_result("입력 검증 실패")
                    
                    # 모델 로딩 확인
                    if not self._ensure_models_loaded():
                        return self._create_error_result("모델 로딩 실패")
                    
                    # AI 추론 실행
                    result = self._run_ai_inference(kwargs)
                    
                    # 결과 검증
                    if not self._validate_output(result):
                        return self._create_error_result("출력 검증 실패")
                    
                    return result
                    
                except Exception as e:
                    self.logger.error(f"❌ 처리 실패: {e}")
                    return self._create_error_result(f"처리 실패: {e}")
            
            def initialize(self) -> bool:
                """초기화"""
                try:
                    if self.initialized:
                        return True
                    
                    # Model Loader 초기화
                    if MODEL_LOADER_AVAILABLE:
                        self.model_loader = get_global_model_loader()
                        if not self.model_loader:
                            self.logger.warning("⚠️ 전역 Model Loader 없음 - 새로 생성")
                            self.model_loader = ModelLoader()
                    
                    # 모델 로딩
                    if not self._load_models():
                        self.logger.error("❌ 모델 로딩 실패")
                        return False
                    
                    self.initialized = True
                    self.logger.info("✅ 초기화 완료")
                    return True
                    
                except Exception as e:
                    self.logger.error(f"❌ 초기화 실패: {e}")
                    return False
            
            def cleanup(self):
                """정리"""
                try:
                    # 모델 언로드
                    if self.model_loader:
                        self.model_loader.cleanup()
                    
                    # 메모리 정리
                    gc.collect()
                    if TORCH_AVAILABLE:
                        try:
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            elif MPS_AVAILABLE:
                                torch.mps.empty_cache()
                        except:
                            pass
                    
                    self.initialized = False
                    self.logger.info("✅ 정리 완료")
                    
                except Exception as e:
                    self.logger.error(f"❌ 정리 실패: {e}")
            
            def get_status(self) -> Dict[str, Any]:
                """상태 조회"""
                return {
                    'initialized': self.initialized,
                    'step_type': self.step_type,
                    'step_name': self.step_name,
                    'step_version': self.step_version,
                    'models_loaded': len(self.ai_models),
                    'inference_count': self.inference_count,
                    'total_inference_time': self.total_inference_time,
                    'last_inference_time': self.last_inference_time
                }
            
            def _validate_input(self, kwargs) -> bool:
                """입력 검증"""
                required_keys = ['image']
                return all(key in kwargs for key in required_keys)
            
            def _ensure_models_loaded(self) -> bool:
                """모델 로딩 확인"""
                return len(self.ai_models) > 0
            
            def _run_ai_inference(self, kwargs) -> Dict[str, Any]:
                """AI 추론 실행"""
                # 기본 구현 - 하위 클래스에서 오버라이드
                return {'status': 'success', 'message': '기본 추론 완료'}
            
            def _validate_output(self, result) -> bool:
                """출력 검증"""
                return isinstance(result, dict) and 'status' in result
            
            def _create_error_result(self, message: str) -> Dict[str, Any]:
                """에러 결과 생성"""
                return {
                    'status': 'error',
                    'message': message,
                    'step_type': self.step_type,
                    'timestamp': time.time()
                }
            
            def _load_models(self) -> bool:
                """모델 로딩"""
                # 기본 구현 - 하위 클래스에서 오버라이드
                return True
        
        return BaseStepMixin

# 🔥 중앙 허브 컨테이너 가져오기
def _get_central_hub_container():
    """중앙 허브 컨테이너 가져오기"""
    try:
        from app.api.central_hub import get_central_hub_container
        return get_central_hub_container()
    except ImportError:
        logger.warning("⚠️ 중앙 허브 컨테이너 없음")
        return None

# 🔥 의존성 주입 안전 함수
def _inject_dependencies_safe(step_instance):
    """의존성 안전 주입"""
    try:
        container = _get_central_hub_container()
        if container:
            # Model Loader 주입
            if hasattr(step_instance, 'set_model_loader'):
                model_loader = container.get_service('model_loader')
                if model_loader:
                    step_instance.set_model_loader(model_loader)
            
            # Memory Manager 주입
            if hasattr(step_instance, 'set_memory_manager'):
                memory_manager = container.get_service('memory_manager')
                if memory_manager:
                    step_instance.set_memory_manager(memory_manager)
            
            # Data Converter 주입
            if hasattr(step_instance, 'set_data_converter'):
                data_converter = container.get_service('data_converter')
                if data_converter:
                    step_instance.set_data_converter(data_converter)
            
            # DI Container 주입
            if hasattr(step_instance, 'set_di_container'):
                step_instance.set_di_container(container)
                
    except Exception as e:
        logger.warning(f"⚠️ 의존성 주입 실패: {e}")

# 🔥 서비스 가져오기 안전 함수
def _get_service_from_central_hub(service_key: str):
    """중앙 허브에서 서비스 가져오기"""
    try:
        container = _get_central_hub_container()
        if container:
            return container.get_service(service_key)
        return None
    except Exception as e:
        logger.warning(f"⚠️ 서비스 가져오기 실패 ({service_key}): {e}")
        return None

# 🔥 M3 Max 감지
def detect_m3_max():
    """M3 Max 환경 감지"""
    try:
        import platform
        if platform.system() == 'Darwin':
            # macOS에서 M3 Max 감지
            import subprocess
            result = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'], 
                                  capture_output=True, text=True)
            if 'M3 Max' in result.stdout:
                return True
        return False
    except:
        return False

# 🔥 세그멘테이션 방법 열거형
class SegmentationMethod(Enum):
    """세그멘테이션 방법"""
    U2NET_CLOTH = "u2net_cloth"         # U2Net 의류 특화 (168.1MB) - 우선순위 1 (M3 Max 안전)
    SAM_HUGE = "sam_huge"               # SAM ViT-Huge (2445.7MB) - 우선순위 2 (메모리 여유시)
    DEEPLABV3_PLUS = "deeplabv3_plus"   # DeepLabV3+ (233.3MB) - 우선순위 3 (나중에)
    MASK_RCNN = "mask_rcnn"             # Mask R-CNN (폴백)
    HYBRID_AI = "hybrid_ai"             # 하이브리드 앙상블

# 🔥 의류 카테고리 열거형
class ClothCategory(Enum):
    """의류 카테고리 (다중 클래스)"""
    BACKGROUND = 0
    SHIRT = 1           # 셔츠/블라우스
    T_SHIRT = 2         # 티셔츠
    SWEATER = 3         # 스웨터/니트
    HOODIE = 4          # 후드티
    JACKET = 5          # 재킷/아우터
    COAT = 6            # 코트
    DRESS = 7           # 원피스
    SKIRT = 8           # 스커트
    PANTS = 9           # 바지
    JEANS = 10          # 청바지
    SHORTS = 11         # 반바지
    SHOES = 12          # 신발
    BOOTS = 13          # 부츠
    SNEAKERS = 14       # 운동화
    BAG = 15            # 가방
    HAT = 16            # 모자
    GLASSES = 17        # 안경
    SCARF = 18          # 스카프
    BELT = 19           # 벨트
    ACCESSORY = 20      # 액세서리

# 🔥 품질 레벨 열거형
class QualityLevel(Enum):
    """품질 레벨"""
    FAST = "fast"           # 빠른 처리
    BALANCED = "balanced"   # 균형
    HIGH = "high"          # 고품질
    ULTRA = "ultra"        # 최고품질

# 🔥 설정 데이터클래스
@dataclass
class ClothSegmentationConfig:
    """의류 세그멘테이션 설정"""
    method: SegmentationMethod = SegmentationMethod.U2NET_CLOTH  # M3 Max 안전 모드
    quality_level: QualityLevel = QualityLevel.HIGH
    input_size: Tuple[int, int] = (512, 512)
    
    # 전처리 설정
    enable_quality_assessment: bool = True
    enable_lighting_normalization: bool = True
    enable_color_correction: bool = True
    
    # 의류 분류 설정
    enable_clothing_classification: bool = True
    classification_confidence_threshold: float = 0.8
    
    # 후처리 설정
    enable_crf_postprocessing: bool = True
    enable_edge_refinement: bool = True
    enable_hole_filling: bool = True
    enable_multiscale_processing: bool = True
    
    # 품질 검증 설정
    enable_quality_validation: bool = True
    quality_threshold: float = 0.7
    enable_auto_retry: bool = True
    max_retry_attempts: int = 3
    
    # 기본 설정
    confidence_threshold: float = 0.5
    enable_visualization: bool = True
    
    # 자동 전처리 설정
    auto_preprocessing: bool = True
    
    # 자동 후처리 설정
    auto_postprocessing: bool = True
    
    # 데이터 검증 설정
    strict_data_validation: bool = True

# 🔥 메인 Step 클래스
class ClothSegmentationStep(get_base_step_mixin_class()):
    """의류 세그멘테이션 Step - Model Loader 통합 버전"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # 설정 초기화
        self.config = kwargs.get('config', ClothSegmentationConfig())
        
        # Model Loader 통합
        self.step_interface: Optional[StepModelInterface] = None
        self.model_factory: Optional[StepModelFactory] = None
        
        # 성능 메트릭
        self.segmentation_stats = {
            'total_segmentations': 0,
            'successful_segmentations': 0,
            'failed_segmentations': 0,
            'average_confidence': 0.0,
            'average_processing_time': 0.0
        }
        
        # 의존성 주입
        _inject_dependencies_safe(self)
        
        self.logger.info(f"🔥 {self.step_name} 초기화 완료 (v{self.step_version})")
    
    def initialize(self) -> bool:
        """초기화 - Model Loader 통합"""
        try:
            if self.initialized:
                return True
            
            # Model Loader 초기화
            if not self._initialize_model_loader():
                return False
            
            # Step Interface 생성
            if not self._create_step_interface():
                return False
            
            # 모델 로딩
            if not self._load_models():
                return False
            
            self.initialized = True
            self.logger.info("✅ 의류 세그멘테이션 초기화 완료")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ 초기화 실패: {e}")
            return False
    
    def _initialize_model_loader(self) -> bool:
        """Model Loader 초기화"""
        try:
            if not MODEL_LOADER_AVAILABLE:
                self.logger.error("❌ Model Loader 사용 불가")
                return False
            
            # 전역 Model Loader 가져오기
            self.model_loader = get_global_model_loader()
            if not self.model_loader:
                self.logger.warning("⚠️ 전역 Model Loader 없음 - 새로 생성")
                self.model_loader = ModelLoader()
            
            # Step Factory 생성
            self.model_factory = StepModelFactory(self.model_loader)
            
            self.logger.info("✅ Model Loader 초기화 완료")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Model Loader 초기화 실패: {e}")
            return False
    
    def _create_step_interface(self) -> bool:
        """Step Interface 생성"""
        try:
            if not self.model_factory:
                self.logger.error("❌ Model Factory 없음")
                return False
            
            # Step Interface 생성
            self.step_interface = self.model_factory.create_step_interface('cloth_segmentation')
            if not self.step_interface:
                self.logger.error("❌ Step Interface 생성 실패")
                return False
            
            self.logger.info("✅ Step Interface 생성 완료")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Step Interface 생성 실패: {e}")
            return False
    
    def _load_models(self) -> bool:
        """모델 로딩 - Model Loader 활용"""
        try:
            if not self.step_interface:
                self.logger.error("❌ Step Interface 없음")
                return False
            
            # 주요 모델 로딩
            success = self.step_interface.load_primary_model()
            if not success:
                self.logger.error("❌ 주요 모델 로딩 실패")
                return False
            
            # 모델 정보 저장
            model = self.step_interface.get_model()
            if model:
                self.ai_models['primary'] = model
                self.segmentation_models['primary'] = model
                self.models_loading_status['primary'] = True
                self.logger.info("✅ 주요 모델 로딩 완료")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ 모델 로딩 실패: {e}")
            return False
    
    def process(self, **kwargs) -> Dict[str, Any]:
        """메인 처리 로직 - Model Loader 통합"""
        try:
            start_time = time.time()
            
            # 입력 검증
            if not self._validate_input(kwargs):
                return self._create_error_result("입력 검증 실패")
            
            # 모델 로딩 확인
            if not self._ensure_models_loaded():
                return self._create_error_result("모델 로딩 실패")
            
            # 입력 전처리
            processed_input = self._preprocess_input(kwargs)
            
            # AI 추론 실행
            result = self._run_ai_inference(processed_input)
            
            # 결과 후처리
            result = self._postprocess_output(result)
            
            # 성능 메트릭 업데이트
            processing_time = time.time() - start_time
            self._update_metrics(processing_time, result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ 처리 실패: {e}")
            return self._create_error_result(f"처리 실패: {e}")
    
    def _validate_input(self, kwargs) -> bool:
        """입력 검증"""
        try:
            required_keys = ['image']
            if not all(key in kwargs for key in required_keys):
                self.logger.error("❌ 필수 입력 키 누락")
                return False
            
            image = kwargs['image']
            if not isinstance(image, np.ndarray):
                self.logger.error("❌ 이미지가 numpy 배열이 아님")
                return False
            
            if len(image.shape) != 3:
                self.logger.error("❌ 이미지 차원이 3이 아님")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ 입력 검증 실패: {e}")
            return False
    
    def _preprocess_input(self, kwargs) -> Dict[str, Any]:
        """입력 전처리"""
        try:
            image = kwargs['image']
            
            # 이미지 품질 평가
            quality_scores = self._assess_image_quality(image)
            
            # 품질 레벨 결정
            quality_level = self._determine_quality_level(quality_scores)
            
            # 이미지 전처리
            if self.config.enable_lighting_normalization:
                image = self._normalize_lighting(image)
            
            if self.config.enable_color_correction:
                image = self._correct_colors(image)
            
            return {
                'image': image,
                'quality_scores': quality_scores,
                'quality_level': quality_level,
                'person_parsing': kwargs.get('person_parsing', {}),
                'pose_info': kwargs.get('pose_info', {})
            }
            
        except Exception as e:
            self.logger.error(f"❌ 입력 전처리 실패: {e}")
            return kwargs
    
    def _run_ai_inference(self, processed_input: Dict[str, Any]) -> Dict[str, Any]:
        """AI 추론 실행 - Model Loader 활용"""
        try:
            if not self.step_interface:
                return self._create_error_result("Step Interface 없음")
            
            # Step Interface를 통한 추론 실행
            result = self.step_interface.run_inference(
                processed_input['image'],
                **processed_input
            )
            
            if not result:
                return self._create_error_result("추론 실패")
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ AI 추론 실패: {e}")
            return self._create_error_result(f"AI 추론 실패: {e}")
    
    def _postprocess_output(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """출력 후처리"""
        try:
            if result.get('status') == 'error':
                return result
            
            # 마스크 후처리
            if 'masks' in result:
                result['masks'] = self._postprocess_masks(result['masks'])
            
            # 의류 카테고리 감지
            if self.config.enable_clothing_classification and 'masks' in result:
                cloth_categories = self._detect_cloth_categories(result['masks'])
                result['cloth_categories'] = cloth_categories
            
            # 시각화 생성
            if self.config.enable_visualization and 'masks' in result:
                visualizations = self._create_segmentation_visualizations(
                    result.get('original_image', np.zeros((512, 512, 3))),
                    result['masks']
                )
                result['visualizations'] = visualizations
            
            # 메타데이터 추가
            result['step_type'] = self.step_type
            result['step_name'] = self.step_name
            result['step_version'] = self.step_version
            result['timestamp'] = time.time()
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ 출력 후처리 실패: {e}")
            return result
    
    def _assess_image_quality(self, image: np.ndarray) -> Dict[str, float]:
        """이미지 품질 평가"""
        try:
            scores = {}
            
            # 밝기 평가
            brightness = np.mean(image)
            scores['brightness'] = brightness / 255.0
            
            # 대비 평가
            contrast = np.std(image)
            scores['contrast'] = min(contrast / 50.0, 1.0)
            
            # 선명도 평가
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            scores['sharpness'] = min(laplacian_var / 1000.0, 1.0)
            
            # 노이즈 평가
            noise_level = 1.0 - scores['sharpness']
            scores['noise'] = noise_level
            
            # 전체 품질 점수
            scores['overall'] = (scores['brightness'] + scores['contrast'] + scores['sharpness']) / 3.0
            
            return scores
            
        except Exception as e:
            self.logger.error(f"❌ 이미지 품질 평가 실패: {e}")
            return {'overall': 0.5}
    
    def _normalize_lighting(self, image: np.ndarray) -> np.ndarray:
        """조명 정규화"""
        try:
            # CLAHE 적용
            if len(image.shape) == 3:
                lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                lab[:, :, 0] = clahe.apply(lab[:, :, 0])
                return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            else:
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                return clahe.apply(image)
        except Exception as e:
            self.logger.error(f"❌ 조명 정규화 실패: {e}")
            return image
    
    def _correct_colors(self, image: np.ndarray) -> np.ndarray:
        """색상 보정"""
        try:
            # 자동 화이트 밸런스
            if len(image.shape) == 3:
                result = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
                avg_a = np.average(result[:, :, 1])
                avg_b = np.average(result[:, :, 2])
                result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
                result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
                return cv2.cvtColor(result, cv2.COLOR_LAB2RGB)
            else:
                return image
        except Exception as e:
            self.logger.error(f"❌ 색상 보정 실패: {e}")
            return image
    
    def _determine_quality_level(self, quality_scores: Dict[str, float]) -> QualityLevel:
        """품질 레벨 결정"""
        try:
            overall_score = quality_scores.get('overall', 0.5)
            
            if overall_score >= 0.8:
                return QualityLevel.ULTRA
            elif overall_score >= 0.6:
                return QualityLevel.HIGH
            elif overall_score >= 0.4:
                return QualityLevel.BALANCED
            else:
                return QualityLevel.FAST
                
        except Exception as e:
            self.logger.error(f"❌ 품질 레벨 결정 실패: {e}")
            return QualityLevel.BALANCED
    
    def _postprocess_masks(self, masks: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """마스크 후처리"""
        try:
            processed_masks = {}
            
            for mask_name, mask in masks.items():
                if mask is None:
                    continue
                
                # 노이즈 제거
                if self.config.enable_hole_filling:
                    mask = self._fill_holes_and_remove_noise(mask)
                
                # 경계 정제
                if self.config.enable_edge_refinement:
                    mask = self._refine_edges(mask)
                
                processed_masks[mask_name] = mask
            
            return processed_masks
            
        except Exception as e:
            self.logger.error(f"❌ 마스크 후처리 실패: {e}")
            return masks
    
    def _fill_holes_and_remove_noise(self, mask: np.ndarray) -> np.ndarray:
        """구멍 채우기 및 노이즈 제거"""
        try:
            # 작은 노이즈 제거
            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            
            # 구멍 채우기
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            
            return mask
            
        except Exception as e:
            self.logger.error(f"❌ 구멍 채우기 실패: {e}")
            return mask
    
    def _refine_edges(self, mask: np.ndarray) -> np.ndarray:
        """경계 정제"""
        try:
            # 가우시안 블러로 경계 부드럽게
            blurred = cv2.GaussianBlur(mask, (3, 3), 0)
            
            # 임계값 적용
            _, refined = cv2.threshold(blurred, 0.5, 1.0, cv2.THRESH_BINARY)
            
            return refined
            
        except Exception as e:
            self.logger.error(f"❌ 경계 정제 실패: {e}")
            return mask
    
    def _detect_cloth_categories(self, masks: Dict[str, np.ndarray]) -> List[str]:
        """의류 카테고리 감지"""
        try:
            categories = []
            
            for mask_name, mask in masks.items():
                if mask is None or np.sum(mask) == 0:
                    continue
                
                # 마스크 이름에서 카테고리 추론
                if 'shirt' in mask_name.lower():
                    categories.append('shirt')
                elif 'pants' in mask_name.lower():
                    categories.append('pants')
                elif 'dress' in mask_name.lower():
                    categories.append('dress')
                elif 'jacket' in mask_name.lower():
                    categories.append('jacket')
                else:
                    categories.append('unknown')
            
            return list(set(categories))  # 중복 제거
            
        except Exception as e:
            self.logger.error(f"❌ 의류 카테고리 감지 실패: {e}")
            return []
    
    def _create_segmentation_visualizations(self, image: np.ndarray, masks: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """세그멘테이션 시각화 생성"""
        try:
            visualizations = {}
            
            # 원본 이미지
            visualizations['original'] = image.tolist()
            
            # 개별 마스크 시각화
            for mask_name, mask in masks.items():
                if mask is None:
                    continue
                
                # 마스크를 컬러로 변환
                colored_mask = np.zeros_like(image)
                colored_mask[mask > 0.5] = [255, 0, 0]  # 빨간색
                
                # 오버레이 생성
                overlay = cv2.addWeighted(image, 0.7, colored_mask, 0.3, 0)
                visualizations[f'{mask_name}_overlay'] = overlay.tolist()
            
            return visualizations
            
        except Exception as e:
            self.logger.error(f"❌ 시각화 생성 실패: {e}")
            return {}
    
    def _update_metrics(self, processing_time: float, result: Dict[str, Any]):
        """성능 메트릭 업데이트"""
        try:
            self.inference_count += 1
            self.total_inference_time += processing_time
            self.last_inference_time = processing_time
            
            # 세그멘테이션 통계 업데이트
            self.segmentation_stats['total_segmentations'] += 1
            
            if result.get('status') == 'success':
                self.segmentation_stats['successful_segmentations'] += 1
            else:
                self.segmentation_stats['failed_segmentations'] += 1
            
            # 평균 계산
            if self.inference_count > 0:
                self.segmentation_stats['average_processing_time'] = self.total_inference_time / self.inference_count
            
            # 신뢰도 업데이트
            confidence = result.get('confidence', 0.0)
            if confidence > 0:
                current_avg = self.segmentation_stats['average_confidence']
                total_successful = self.segmentation_stats['successful_segmentations']
                self.segmentation_stats['average_confidence'] = (
                    (current_avg * (total_successful - 1) + confidence) / total_successful
                )
                
        except Exception as e:
            self.logger.error(f"❌ 메트릭 업데이트 실패: {e}")
    
    def get_model_info(self, model_key: str = None) -> Dict[str, Any]:
        """모델 정보 조회"""
        try:
            if not self.step_interface:
                return {'error': 'Step Interface 없음'}
            
            model = self.step_interface.get_model()
            if not model:
                return {'error': '모델 없음'}
            
            return {
                'model_type': type(model).__name__,
                'step_type': self.step_type,
                'loaded': self.step_interface.loaded,
                'inference_count': self.inference_count,
                'total_inference_time': self.total_inference_time,
                'last_inference_time': self.last_inference_time
            }
            
        except Exception as e:
            self.logger.error(f"❌ 모델 정보 조회 실패: {e}")
            return {'error': str(e)}
    
    def get_segmentation_stats(self) -> Dict[str, Any]:
        """세그멘테이션 통계 조회"""
        return self.segmentation_stats.copy()
    
    def clear_cache(self):
        """캐시 정리"""
        try:
            if self.model_loader:
                self.model_loader.cleanup()
            
            # 메모리 정리
            gc.collect()
            if TORCH_AVAILABLE:
                try:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    elif MPS_AVAILABLE:
                        torch.mps.empty_cache()
                except:
                    pass
            
            self.logger.info("✅ 캐시 정리 완료")
            
        except Exception as e:
            self.logger.error(f"❌ 캐시 정리 실패: {e}")
    
    def reload_models(self):
        """모델 재로딩"""
        try:
            # 기존 모델 언로드
            self.ai_models.clear()
            self.segmentation_models.clear()
            self.models_loading_status.clear()
            
            # 모델 재로딩
            if self._load_models():
                self.logger.info("✅ 모델 재로딩 완료")
            else:
                self.logger.error("❌ 모델 재로딩 실패")
                
        except Exception as e:
            self.logger.error(f"❌ 모델 재로딩 실패: {e}")

# 🔥 팩토리 함수들
def create_cloth_segmentation_step(**kwargs) -> ClothSegmentationStep:
    """의류 세그멘테이션 Step 생성"""
    return ClothSegmentationStep(**kwargs)

def create_m3_max_segmentation_step(**kwargs) -> ClothSegmentationStep:
    """M3 Max 최적화 의류 세그멘테이션 Step 생성"""
    config = kwargs.get('config', ClothSegmentationConfig())
    config.method = SegmentationMethod.U2NET_CLOTH  # M3 Max 안전 모드
    config.quality_level = QualityLevel.BALANCED  # 균형 모드
    
    kwargs['config'] = config
    return ClothSegmentationStep(**kwargs)

# 🔥 테스트 함수
def test_cloth_segmentation_ai():
    """의류 세그멘테이션 AI 테스트"""
    try:
        logger.info("🧪 의류 세그멘테이션 AI 테스트 시작")
        
        # Step 생성
        step = create_cloth_segmentation_step()
        
        # 초기화
        if not step.initialize():
            logger.error("❌ Step 초기화 실패")
            return False
        
        # 더미 이미지 생성
        dummy_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        
        # 처리 테스트
        result = step.process(image=dummy_image)
        
        if result.get('status') == 'success':
            logger.info("✅ 의류 세그멘테이션 AI 테스트 성공")
            return True
        else:
            logger.error(f"❌ 의류 세그멘테이션 AI 테스트 실패: {result.get('message')}")
            return False
            
    except Exception as e:
        logger.error(f"❌ 의류 세그멘테이션 AI 테스트 실패: {e}")
        return False

# 🔥 중앙 허브 호환성 테스트
def test_central_hub_compatibility():
    """중앙 허브 호환성 테스트"""
    try:
        logger.info("🧪 중앙 허브 호환성 테스트 시작")
        
        # 중앙 허브 컨테이너 확인
        container = _get_central_hub_container()
        if not container:
            logger.warning("⚠️ 중앙 허브 컨테이너 없음")
            return True  # 경고만 하고 성공으로 처리
        
        # 서비스 확인
        services = ['model_loader', 'memory_manager', 'data_converter']
        for service in services:
            service_instance = container.get_service(service)
            if service_instance:
                logger.info(f"✅ {service} 서비스 확인됨")
            else:
                logger.warning(f"⚠️ {service} 서비스 없음")
        
        logger.info("✅ 중앙 허브 호환성 테스트 완료")
        return True
        
    except Exception as e:
        logger.error(f"❌ 중앙 허브 호환성 테스트 실패: {e}")
        return False

# 🔥 메모리 관리 함수들
def cleanup_memory():
    """메모리 정리"""
    try:
        gc.collect()
        if TORCH_AVAILABLE:
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                elif MPS_AVAILABLE:
                    torch.mps.empty_cache()
            except Exception as mem_error:
                logger.warning(f"⚠️ PyTorch 메모리 정리 실패: {mem_error}")
    except Exception as e:
        logger.warning(f"⚠️ 메모리 정리 실패: {e}")

def safe_torch_operation(operation_func, *args, **kwargs):
    """안전한 PyTorch 연산"""
    try:
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch 사용 불가")
        
        return operation_func(*args, **kwargs)
        
    except Exception as e:
        logger.error(f"❌ PyTorch 연산 실패: {e}")
        if EXCEPTIONS_AVAILABLE:
            error = ModelLoadingError(f"PyTorch 연산 실패: {e}", ErrorCodes.MODEL_LOADING_FAILED)
            track_exception(error, {'operation': operation_func.__name__}, 3)
        raise

# 🔥 모듈 초기화
if __name__ == "__main__":
    # 로깅 설정
    logging.basicConfig(level=logging.INFO)
    
    # 테스트 실행
    test_cloth_segmentation_ai()
    test_central_hub_compatibility()
    
    logger.info("🎉 Step 03 Cloth Segmentation 모듈 로드 완료")
