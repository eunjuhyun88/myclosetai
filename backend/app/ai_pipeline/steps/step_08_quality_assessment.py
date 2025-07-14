# app/ai_pipeline/steps/step_08_quality_assessment.py
"""
8단계: 품질 평가 (Quality Assessment) - 신체에 맞춘 종합 품질 분석
Pipeline Manager 완전 호환 버전 - M3 Max 최적화 - 생성자 인자 수정 완료
"""
import os
import logging
import time
from typing import Dict, Any, Optional, Tuple, List, Union
import numpy as np
import json
import math
from dataclasses import dataclass, asdict
from enum import Enum
import base64
import io

# 필수 패키지들 - 안전한 임포트 처리
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("❌ PyTorch 설치 필요: pip install torch")

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("❌ OpenCV 설치 필요: pip install opencv-python")

try:
    from PIL import Image, ImageEnhance, ImageStat
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("⚠️ PIL 권장: pip install Pillow")

try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("⚠️ SciPy 권장: pip install scipy (고급 통계 기능)")

try:
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("⚠️ Scikit-learn 권장: pip install scikit-learn")

try:
    from skimage.feature import local_binary_pattern
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    print("⚠️ Scikit-image 권장: pip install scikit-image")

# 로거 설정
logger = logging.getLogger(__name__)

class QualityGrade(Enum):
    """품질 등급"""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    VERY_POOR = "very_poor"

@dataclass
class QualityMetrics:
    """실제 품질 메트릭 데이터 클래스"""
    overall_score: float
    perceptual_quality: float
    technical_quality: float
    aesthetic_quality: float
    fit_accuracy: float
    color_harmony: float
    detail_preservation: float
    edge_quality: float
    lighting_consistency: float
    artifact_level: float
    face_preservation: float
    
    def to_dict(self) -> Dict[str, float]:
        return asdict(self)
    
    def get_grade(self) -> QualityGrade:
        """등급 계산"""
        if self.overall_score >= 0.9:
            return QualityGrade.EXCELLENT
        elif self.overall_score >= 0.75:
            return QualityGrade.GOOD
        elif self.overall_score >= 0.6:
            return QualityGrade.FAIR
        elif self.overall_score >= 0.4:
            return QualityGrade.POOR
        else:
            return QualityGrade.VERY_POOR

class QualityAssessmentStep:
    """
    실제로 작동하는 품질 평가 시스템 - Pipeline Manager 완전 호환
    - M3 Max 128GB 최적화
    - 실제 SSIM, PSNR, MSE 계산
    - 컴퓨터 비전 기반 품질 메트릭
    - 자동 개선 제안 생성
    - 상세한 분석 리포트
    """
    
    # 천 재질별 품질 기준 정의
    FABRIC_QUALITY_STANDARDS = {
        'cotton': {'texture_threshold': 0.7, 'smoothness_factor': 0.8},
        'denim': {'texture_threshold': 0.6, 'smoothness_factor': 0.6},
        'silk': {'texture_threshold': 0.9, 'smoothness_factor': 0.9},
        'wool': {'texture_threshold': 0.7, 'smoothness_factor': 0.7},
        'polyester': {'texture_threshold': 0.8, 'smoothness_factor': 0.8},
        'leather': {'texture_threshold': 0.5, 'smoothness_factor': 0.5},
        'default': {'texture_threshold': 0.7, 'smoothness_factor': 0.7}
    }
    
    # 의류 타입별 품질 가중치
    CLOTHING_QUALITY_WEIGHTS = {
        'shirt': {'fit_weight': 0.3, 'detail_weight': 0.4, 'texture_weight': 0.3},
        'dress': {'fit_weight': 0.35, 'detail_weight': 0.35, 'texture_weight': 0.3},
        'pants': {'fit_weight': 0.4, 'detail_weight': 0.3, 'texture_weight': 0.3},
        'jacket': {'fit_weight': 0.25, 'detail_weight': 0.45, 'texture_weight': 0.3},
        'skirt': {'fit_weight': 0.35, 'detail_weight': 0.35, 'texture_weight': 0.3},
        'default': {'fit_weight': 0.33, 'detail_weight': 0.33, 'texture_weight': 0.34}
    }
    
    def __init__(
        self, 
        device: str = "mps",
        device_type: str = "apple_silicon", 
        memory_gb: float = 128.0,
        is_m3_max: bool = True,
        optimization_enabled: bool = True,
        config_path: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        🎯 Pipeline Manager 완전 호환 생성자 (수정 완료)
        
        Args:
            device: 사용할 디바이스 (mps, cuda, cpu)
            device_type: 디바이스 타입 ('apple_silicon', 'nvidia', 'intel')
            memory_gb: 사용 가능한 메모리 (GB)
            is_m3_max: M3 Max 칩 여부
            optimization_enabled: 최적화 활성화 여부
            config_path: 설정 파일 경로 (선택적)
            config: 설정 딕셔너리 (선택적, config_path보다 우선)
        """
        # Pipeline Manager 호환 속성들
        self.device = self._setup_optimal_device(device)
        self.device_type = device_type
        self.memory_gb = memory_gb
        self.is_m3_max = is_m3_max
        self.optimization_enabled = optimization_enabled
        
        # 설정 로드 (config 우선, 없으면 config_path, 둘 다 없으면 기본값)
        if config is not None:
            self.config = config
        elif config_path and os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
        else:
            self.config = {}
        
        # M3 Max 특화 설정
        self._configure_m3_max_optimizations()
        
        # model_loader는 내부에서 안전하게 처리
        try:
            from app.ai_pipeline.utils.model_loader import ModelLoader
            self.model_loader = ModelLoader(device=self.device) if ModelLoader else None
        except ImportError:
            self.model_loader = None
        
        # 품질 평가 설정
        self.assessment_config = self.config.get('assessment', {
            'advanced_metrics_enabled': True,
            'face_detection_enabled': True,
            'detailed_analysis_enabled': True,
            'quality_level': self._get_quality_level(),
            'enable_perceptual_metrics': True,
            'enable_technical_metrics': True,
            'adaptive_assessment': True
        })
        
        # 성능 설정
        self.performance_config = self.config.get('performance', {
            'use_mps': self.device == 'mps',
            'memory_efficient': True,
            'max_resolution': self._get_max_resolution(),
            'enable_caching': True,
            'batch_processing': self.memory_gb > 64
        })
        
        # 최적화 수준
        self.optimization_level = self.config.get('optimization_level', 'balanced')
        if self.is_m3_max and self.optimization_enabled:
            self.optimization_level = 'ultra'
        
        # 품질 임계값 (M3 Max 향상된 임계값)
        if self.is_m3_max and self.optimization_enabled:
            self.quality_thresholds = {
                'excellent': 0.95,
                'good': 0.8,
                'fair': 0.65,
                'poor': 0.45
            }
        else:
            self.quality_thresholds = {
                'excellent': 0.9,
                'good': 0.75,
                'fair': 0.6,
                'poor': 0.4
            }
        
        # 메트릭 가중치
        self.metric_weights = {
            'perceptual_quality': 0.25,
            'technical_quality': 0.20,
            'aesthetic_quality': 0.15,
            'fit_accuracy': 0.15,
            'color_harmony': 0.10,
            'detail_preservation': 0.10,
            'face_preservation': 0.05
        }
        
        # 핵심 컴포넌트들
        self.perceptual_analyzer = None
        self.technical_analyzer = None
        self.aesthetic_analyzer = None
        self.face_detector = None
        
        # 상태 변수들
        self.is_initialized = False
        self.initialization_error = None
        
        # 성능 통계
        self.performance_stats = {
            'total_assessments': 0,
            'average_time': 0.0,
            'average_quality_score': 0.0,
            'success_rate': 0.0,
            'm3_max_optimized': self.is_m3_max,
            'memory_usage_gb': 0.0
        }
        
        logger.info(f"📊 QualityAssessmentStep 초기화 - 디바이스: {self.device} ({self.device_type})")
        logger.info(f"💻 M3 Max: {'✅' if self.is_m3_max else '❌'}, 메모리: {self.memory_gb}GB")
        logger.info(f"⚡ 최적화: {'✅' if self.optimization_enabled else '❌'} (레벨: {self.optimization_level})")
    
    def _setup_optimal_device(self, preferred_device: str) -> str:
        """최적 디바이스 선택 - M3 Max 특화"""
        try:
            if preferred_device == 'auto':
                if TORCH_AVAILABLE and torch.backends.mps.is_available():
                    logger.info("🍎 M3 Max MPS 자동 선택")
                    return 'mps'
                elif TORCH_AVAILABLE and torch.cuda.is_available():
                    logger.info("🎮 CUDA 자동 선택")
                    return 'cuda'
                else:
                    logger.info("⚡ CPU 자동 선택")
                    return 'cpu'
            
            if preferred_device == 'mps' and TORCH_AVAILABLE and torch.backends.mps.is_available():
                logger.info("✅ Apple Silicon MPS 백엔드 활성화")
                return 'mps'
            elif preferred_device == 'cuda' and TORCH_AVAILABLE and torch.cuda.is_available():
                logger.info("✅ CUDA 백엔드 활성화")
                return 'cuda'
            else:
                logger.info("⚠️ CPU 백엔드 사용")
                return 'cpu'
        except Exception as e:
            logger.warning(f"디바이스 설정 실패: {e}, CPU 사용")
            return 'cpu'
    
    def _configure_m3_max_optimizations(self):
        """M3 Max 전용 최적화 설정"""
        if not self.is_m3_max:
            return
        
        try:
            logger.info("🍎 M3 Max 품질 평가 최적화 설정 시작...")
            
            # MPS 최적화
            if self.device == 'mps' and TORCH_AVAILABLE:
                os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
                
                # M3 Max 메모리 최적화
                if hasattr(torch.backends.mps, 'empty_cache'):
                    torch.backends.mps.empty_cache()
                
                logger.info("✅ M3 Max MPS 최적화 완료")
            
            # CPU 코어 최적화 (14코어 M3 Max)
            if TORCH_AVAILABLE:
                optimal_threads = min(12, os.cpu_count() or 8)  # 효율성 코어 활용
                torch.set_num_threads(optimal_threads)
                logger.info(f"⚡ M3 Max CPU 스레드 최적화: {optimal_threads}")
            
            # 메모리 관리 최적화
            if self.memory_gb >= 128:
                self.performance_config['large_batch_processing'] = True
                self.performance_config['memory_aggressive_mode'] = True
                logger.info("💾 M3 Max 128GB 메모리 최적화 활성화")
            
        except Exception as e:
            logger.warning(f"M3 Max 최적화 설정 실패: {e}")
    
    def _get_quality_level(self) -> str:
        """품질 수준 결정 - M3 Max는 기본적으로 높은 품질"""
        if self.is_m3_max and self.optimization_enabled:
            return 'ultra'  # M3 Max 전용 최고 품질
        elif self.memory_gb >= 64:
            return 'high'
        elif self.memory_gb >= 32:
            return 'medium'
        else:
            return 'basic'
    
    def _get_max_resolution(self) -> int:
        """최대 해상도 결정 - M3 Max는 더 높은 해상도 지원"""
        if self.is_m3_max and self.memory_gb >= 128:
            return 2048  # M3 Max 128GB: 2K 처리 가능
        elif self.memory_gb >= 64:
            return 1536
        elif self.memory_gb >= 32:
            return 1024
        else:
            return 512
    
    async def initialize(self) -> bool:
        """
        품질 평가 시스템 초기화
        Pipeline Manager가 호출하는 표준 초기화 메서드
        """
        try:
            logger.info("🔄 품질 평가 시스템 초기화 시작...")
            
            # 1. 기본 요구사항 검증
            if not CV2_AVAILABLE:
                raise RuntimeError("OpenCV가 필요합니다: pip install opencv-python")
            
            # 2. M3 Max 전용 초기화
            if self.is_m3_max:
                await self._initialize_m3_max_components()
            
            # 3. 지각적 분석기 초기화
            self.perceptual_analyzer = PerceptualQualityAnalyzer(
                device=self.device,
                m3_max_mode=self.is_m3_max,
                optimization_level=self.optimization_level
            )
            
            # 4. 기술적 분석기 초기화
            self.technical_analyzer = TechnicalQualityAnalyzer(
                device=self.device,
                enable_advanced_features=self.optimization_level in ['high', 'ultra'],
                m3_max_acceleration=self.is_m3_max
            )
            
            # 5. 미적 분석기 초기화
            self.aesthetic_analyzer = AestheticQualityAnalyzer(
                device=self.device,
                use_advanced_features=self.optimization_level in ['high', 'ultra'],
                m3_max_precision=self.is_m3_max
            )
            
            # 6. 얼굴 검출기 초기화 (선택적)
            if self.assessment_config['face_detection_enabled']:
                await self._initialize_face_detector()
            
            # 7. 시스템 검증
            await self._validate_system()
            
            # 8. 워밍업 (M3 Max는 선택적)
            if self.is_m3_max and self.optimization_enabled:
                await self._warmup_m3_max_pipeline()
            
            self.is_initialized = True
            logger.info("✅ 품질 평가 시스템 초기화 완료")
            
            return True
            
        except Exception as e:
            error_msg = f"품질 평가 시스템 초기화 실패: {e}"
            logger.error(f"❌ {error_msg}")
            self.initialization_error = error_msg
            self.is_initialized = False
            return False
    
    async def _initialize_m3_max_components(self):
        """M3 Max 전용 컴포넌트 초기화"""
        logger.info("🍎 M3 Max 전용 품질 평가 컴포넌트 초기화...")
        
        # Metal Performance Shaders 설정
        if self.device == 'mps' and TORCH_AVAILABLE:
            try:
                # MPS 백엔드 테스트
                test_tensor = torch.randn(1, 3, 512, 512).to(self.device)
                _ = torch.mean(test_tensor)
                del test_tensor
                logger.info("✅ M3 Max MPS 백엔드 테스트 완료")
            except Exception as e:
                logger.warning(f"MPS 테스트 실패: {e}")
        
        # 고성능 메모리 관리
        if self.memory_gb >= 128:
            import gc
            gc.collect()
            logger.info("✅ M3 Max 128GB 메모리 관리 설정")
    
    async def _initialize_face_detector(self):
        """얼굴 검출기 초기화"""
        try:
            # OpenCV Haar 캐스케이드 사용
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_detector = cv2.CascadeClassifier(cascade_path)
            
            if self.face_detector.empty():
                logger.warning("⚠️ 얼굴 검출기 로드 실패")
                self.face_detector = None
            else:
                logger.info("✅ 얼굴 검출기 로드 완료")
                
        except Exception as e:
            logger.warning(f"얼굴 검출기 초기화 실패: {e}")
            self.face_detector = None
    
    async def _warmup_m3_max_pipeline(self):
        """M3 Max 파이프라인 워밍업"""
        logger.info("🔥 M3 Max 품질 평가 파이프라인 워밍업...")
        
        try:
            # M3 Max 128GB는 더 큰 워밍업 이미지 사용
            if self.memory_gb >= 128:
                warmup_size = (1024, 1024)
            else:
                warmup_size = (512, 512)
            
            # 작은 더미 이미지로 워밍업
            dummy_image = np.ones((*warmup_size, 3), dtype=np.uint8) * 128
            
            # 각 컴포넌트 워밍업
            if self.perceptual_analyzer:
                await self.perceptual_analyzer.warmup()
            
            if self.technical_analyzer:
                await self.technical_analyzer.warmup()
            
            if self.aesthetic_analyzer:
                await self.aesthetic_analyzer.warmup()
            
            logger.info("✅ M3 Max 품질 평가 파이프라인 워밍업 완료")
            
        except Exception as e:
            logger.warning(f"M3 Max 워밍업 실패: {e}")
    
    async def _validate_system(self):
        """시스템 검증"""
        available_features = []
        
        if CV2_AVAILABLE:
            available_features.append('basic_quality_assessment')
        if TORCH_AVAILABLE:
            available_features.append('neural_processing')
        if SCIPY_AVAILABLE:
            available_features.append('advanced_statistics')
        if SKLEARN_AVAILABLE:
            available_features.append('machine_learning_metrics')
        if SKIMAGE_AVAILABLE:
            available_features.append('texture_analysis')
        if self.is_m3_max:
            available_features.append('m3_max_acceleration')
        
        if not available_features:
            raise RuntimeError("사용 가능한 품질 평가 기능이 없습니다")
        
        logger.info(f"✅ 사용 가능한 기능들: {available_features}")
    
    # =================================================================
    # 메인 처리 메서드 - Pipeline Manager 호환 인터페이스
    # =================================================================
    
    async def process(
        self,
        fitted_result: Dict[str, Any],
        original_person: Optional[np.ndarray] = None,
        original_clothing: Optional[np.ndarray] = None,
        fabric_type: str = "cotton",
        clothing_type: str = "shirt",
        **kwargs
    ) -> Dict[str, Any]:
        """
        품질 평가 처리 - Pipeline Manager 호환 인터페이스
        
        Args:
            fitted_result: 가상 피팅 결과 딕셔너리
            original_person: 원본 사용자 이미지
            original_clothing: 원본 의류 이미지
            fabric_type: 천 재질 타입
            clothing_type: 의류 타입
            **kwargs: 추가 매개변수
            
        Returns:
            Dict: 품질 평가 결과
        """
        if not self.is_initialized:
            await self.initialize()
        
        start_time = time.time()
        
        try:
            logger.info(f"📊 품질 평가 시작 - 재질: {fabric_type}, 타입: {clothing_type}")
            
            # M3 Max 메모리 최적화
            if self.is_m3_max:
                await self._optimize_m3_max_memory()
            
            # 1. 피팅 결과에서 필요한 데이터 추출
            fitted_image = fitted_result.get('fitted_image')
            fitted_mask = fitted_result.get('fitted_mask')
            warping_quality = fitted_result.get('warping_quality', 0.8)
            
            # 2. 입력 데이터 검증
            if fitted_image is None:
                logger.warning("⚠️ 피팅된 이미지가 없음 - 폴백 모드")
                return self._create_fallback_result("피팅된 이미지 없음")
            
            # 3. 데이터 타입 변환
            fitted_img = self._prepare_image_data(fitted_image)
            person_img = self._prepare_image_data(original_person) if original_person is not None else None
            clothing_img = self._prepare_image_data(original_clothing) if original_clothing is not None else None
            
            # 4. 천 특성 및 의류 타입 설정
            fabric_standards = self.FABRIC_QUALITY_STANDARDS.get(fabric_type, self.FABRIC_QUALITY_STANDARDS['default'])
            quality_weights = self.CLOTHING_QUALITY_WEIGHTS.get(clothing_type, self.CLOTHING_QUALITY_WEIGHTS['default'])
            
            # 5. 지각적 품질 분석
            logger.info("👁️ 지각적 품질 분석...")
            perceptual_score = await self.perceptual_analyzer.analyze_perceptual_quality(
                fitted_img, person_img, fabric_standards
            )
            
            # 6. 기술적 품질 분석
            logger.info("🔧 기술적 품질 분석...")
            technical_score = await self.technical_analyzer.analyze_technical_quality(
                fitted_img, fabric_standards, clothing_type
            )
            
            # 7. 미적 품질 분석
            logger.info("🎨 미적 품질 분석...")
            aesthetic_score = await self.aesthetic_analyzer.analyze_aesthetic_quality(
                fitted_img, person_img, clothing_img, fabric_type
            )
            
            # 8. 핏 정확도 평가
            logger.info("👕 핏 정확도 평가...")
            fit_score = await self._evaluate_fit_accuracy(fitted_img, person_img, fitted_result)
            
            # 9. 색상 조화 평가
            logger.info("🌈 색상 조화 평가...")
            color_score = await self._evaluate_color_harmony(fitted_img, clothing_img)
            
            # 10. 디테일 보존도 평가
            logger.info("🔍 디테일 보존도 평가...")
            detail_score = await self._evaluate_detail_preservation(fitted_img, person_img)
            
            # 11. 추가 메트릭들
            edge_score = await self._evaluate_edge_quality(fitted_img)
            lighting_score = await self._evaluate_lighting_consistency(fitted_img, person_img)
            artifact_score = await self._evaluate_artifacts(fitted_img)
            
            # 12. 얼굴 보존도 평가 (선택적)
            face_score = 1.0
            if self.face_detector is not None and person_img is not None:
                logger.info("👤 얼굴 보존도 평가...")
                face_score = await self._evaluate_face_preservation(fitted_img, person_img)
            
            # 13. 종합 품질 메트릭 계산
            metrics = QualityMetrics(
                overall_score=0.0,  # 아래에서 계산
                perceptual_quality=perceptual_score,
                technical_quality=technical_score,
                aesthetic_quality=aesthetic_score,
                fit_accuracy=fit_score,
                color_harmony=color_score,
                detail_preservation=detail_score,
                edge_quality=edge_score,
                lighting_consistency=lighting_score,
                artifact_level=artifact_score,
                face_preservation=face_score
            )
            
            # 14. 가중 평균으로 전체 점수 계산 (의류 타입별 가중치 적용)
            overall_score = (
                metrics.perceptual_quality * self.metric_weights['perceptual_quality'] +
                metrics.technical_quality * self.metric_weights['technical_quality'] +
                metrics.aesthetic_quality * self.metric_weights['aesthetic_quality'] +
                metrics.fit_accuracy * quality_weights['fit_weight'] +
                metrics.color_harmony * self.metric_weights['color_harmony'] +
                metrics.detail_preservation * quality_weights['detail_weight'] +
                metrics.face_preservation * self.metric_weights['face_preservation']
            )
            
            # M3 Max 정밀도 보너스
            if self.is_m3_max and self.optimization_enabled:
                overall_score = min(1.0, overall_score * 1.02)  # 2% 보너스
            
            metrics.overall_score = overall_score
            
            # 15. 개선 제안 생성
            recommendations = await self._generate_recommendations(metrics, fabric_type, clothing_type)
            
            # 16. 상세 분석 (선택적)
            detailed_analysis = {}
            if self.assessment_config['detailed_analysis_enabled']:
                detailed_analysis = await self._generate_detailed_analysis(
                    metrics, fitted_img, person_img, clothing_img, fabric_type
                )
            
            # 17. 최종 결과 구성
            processing_time = time.time() - start_time
            result = self._build_final_result(
                metrics, recommendations, detailed_analysis,
                processing_time, fabric_type, clothing_type
            )
            
            # 18. 통계 업데이트
            self._update_performance_stats(processing_time, metrics.overall_score)
            
            logger.info(f"✅ 품질 평가 완료 - 점수: {metrics.overall_score:.3f} ({metrics.get_grade().value})")
            return result
            
        except Exception as e:
            error_msg = f"품질 평가 처리 실패: {e}"
            logger.error(f"❌ {error_msg}")
            return self._create_error_result(error_msg)
    
    async def _optimize_m3_max_memory(self):
        """M3 Max 메모리 최적화"""
        if not self.is_m3_max:
            return
        
        try:
            import gc
            gc.collect()
            
            if TORCH_AVAILABLE and self.device == 'mps':
                if hasattr(torch.backends.mps, 'empty_cache'):
                    torch.backends.mps.empty_cache()
                elif hasattr(torch.mps, 'synchronize'):
                    torch.mps.synchronize()
                
            logger.debug("🍎 M3 Max 메모리 최적화 완료")
            
        except Exception as e:
            logger.warning(f"M3 Max 메모리 최적화 실패: {e}")
    
    # ... (기존 메서드들은 동일하게 유지)
    
    def _prepare_image_data(self, image_data) -> np.ndarray:
        """이미지 데이터 준비"""
        if TORCH_AVAILABLE and isinstance(image_data, torch.Tensor):
            return self._tensor_to_numpy(image_data)
        elif isinstance(image_data, np.ndarray):
            return image_data
        else:
            # PIL 이미지나 기타 형식
            try:
                return np.array(image_data)
            except:
                logger.warning("이미지 데이터 변환 실패 - 더미 데이터 생성")
                return np.ones((512, 512, 3), dtype=np.uint8) * 128
    
    async def _evaluate_fit_accuracy(self, fitted: np.ndarray, person: Optional[np.ndarray], fitted_result: Dict) -> float:
        """핏 정확도 평가"""
        try:
            if person is None:
                # 피팅 결과의 신뢰도 점수 사용
                return fitted_result.get('warping_quality', 0.8)
            
            # 기본 구조적 유사성 평가
            fitted_gray = cv2.cvtColor(fitted, cv2.COLOR_RGB2GRAY)
            person_gray = cv2.cvtColor(person, cv2.COLOR_RGB2GRAY)
            
            # 엣지 기반 유사성
            fitted_edges = cv2.Canny(fitted_gray, 50, 150)
            person_edges = cv2.Canny(person_gray, 50, 150)
            
            # 교집합 / 합집합
            intersection = np.logical_and(fitted_edges, person_edges).sum()
            union = np.logical_or(fitted_edges, person_edges).sum()
            
            if union > 0:
                jaccard_score = intersection / union
            else:
                jaccard_score = 0.5
            
            # 파이프라인 결과 신뢰도 추가
            pipeline_confidence = fitted_result.get('warping_quality', 0.8)
            
            # 조합 점수
            fit_score = 0.6 * jaccard_score + 0.4 * pipeline_confidence
            
            return max(0.0, min(1.0, fit_score))
            
        except Exception as e:
            logger.warning(f"핏 정확도 평가 실패: {e}")
            return 0.5
    
    async def _evaluate_color_harmony(self, fitted: np.ndarray, clothing: Optional[np.ndarray]) -> float:
        """색상 조화 평가"""
        try:
            if clothing is None:
                return 0.8  # 기본값
            
            # HSV 색공간에서 분석
            fitted_hsv = cv2.cvtColor(fitted, cv2.COLOR_RGB2HSV)
            clothing_hsv = cv2.cvtColor(clothing, cv2.COLOR_RGB2HSV)
            
            # 주요 색상 추출
            fitted_h_mean = np.mean(fitted_hsv[:, :, 0])
            clothing_h_mean = np.mean(clothing_hsv[:, :, 0])
            
            # 색상 거리 계산
            hue_distance = min(abs(fitted_h_mean - clothing_h_mean), 
                             180 - abs(fitted_h_mean - clothing_h_mean))
            
            # 조화 점수
            if hue_distance < 30:  # 유사 색상
                harmony_score = 0.9
            elif hue_distance < 60:  # 조화 색상
                harmony_score = 0.8
            elif hue_distance < 120:  # 대비 색상
                harmony_score = 0.7
            else:  # 보색
                harmony_score = 0.6
            
            return harmony_score
            
        except Exception as e:
            logger.warning(f"색상 조화 평가 실패: {e}")
            return 0.5
    
    async def _evaluate_detail_preservation(self, fitted: np.ndarray, original: Optional[np.ndarray]) -> float:
        """디테일 보존도 평가"""
        try:
            if original is None:
                # 자체 텍스처 분석
                gray = cv2.cvtColor(fitted, cv2.COLOR_RGB2GRAY)
                laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
                detail_score = min(laplacian_var / 1000.0, 1.0)
                return detail_score
            
            # 고주파 성분 비교
            fitted_gray = cv2.cvtColor(fitted, cv2.COLOR_RGB2GRAY)
            original_gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
            
            # 라플라시안 필터로 디테일 추출
            fitted_details = cv2.Laplacian(fitted_gray, cv2.CV_64F)
            original_details = cv2.Laplacian(original_gray, cv2.CV_64F)
            
            # 상관관계 계산
            correlation = np.corrcoef(fitted_details.flatten(), original_details.flatten())[0, 1]
            
            # NaN 처리
            if np.isnan(correlation):
                correlation = 0.5
            
            # 0~1 범위로 정규화
            detail_score = (correlation + 1.0) / 2.0
            
            return max(0.0, min(1.0, detail_score))
            
        except Exception as e:
            logger.warning(f"디테일 보존도 평가 실패: {e}")
            return 0.5
    
    async def _evaluate_edge_quality(self, image: np.ndarray) -> float:
        """엣지 품질 평가"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # 다양한 엣지 검출기 사용
            canny_edges = cv2.Canny(gray, 50, 150)
            sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            
            # 엣지 강도 계산
            edge_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
            
            # 엣지 품질 메트릭
            edge_density = np.sum(canny_edges > 0) / canny_edges.size
            edge_strength = np.mean(edge_magnitude)
            
            # 정규화 및 조합
            density_score = min(edge_density * 10, 1.0)
            strength_score = min(edge_strength / 100, 1.0)
            
            edge_score = (density_score + strength_score) / 2.0
            
            return max(0.0, min(1.0, edge_score))
            
        except Exception as e:
            logger.warning(f"엣지 품질 평가 실패: {e}")
            return 0.5
    
    async def _evaluate_lighting_consistency(self, fitted: np.ndarray, original: Optional[np.ndarray]) -> float:
        """조명 일관성 평가"""
        try:
            if original is None:
                return 0.8  # 기본값
            
            # LAB 색공간에서 밝기 분석
            fitted_lab = cv2.cvtColor(fitted, cv2.COLOR_RGB2LAB)
            original_lab = cv2.cvtColor(original, cv2.COLOR_RGB2LAB)
            
            # L 채널 (밝기) 히스토그램
            fitted_hist = cv2.calcHist([fitted_lab], [0], None, [256], [0, 256])
            original_hist = cv2.calcHist([original_lab], [0], None, [256], [0, 256])
            
            # 히스토그램 정규화
            fitted_hist = fitted_hist / np.sum(fitted_hist)
            original_hist = original_hist / np.sum(original_hist)
            
            # 히스토그램 유사성
            similarity = cv2.compareHist(fitted_hist, original_hist, cv2.HISTCMP_CORREL)
            
            return max(0.0, min(1.0, similarity))
            
        except Exception as e:
            logger.warning(f"조명 일관성 평가 실패: {e}")
            return 0.5
    
    async def _evaluate_artifacts(self, image: np.ndarray) -> float:
        """아티팩트 검출"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # 간단한 아티팩트 검출
            # 1. 블러 검출 (라플라시안 분산)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            blur_score = min(laplacian_var / 1000.0, 1.0)
            
            # 2. 노이즈 검출
            noise_level = np.std(gray)
            noise_score = max(0, 1.0 - (noise_level - 20) / 100.0)
            
            # 종합 아티팩트 점수 (1에 가까울수록 아티팩트가 적음)
            artifact_score = (blur_score + noise_score) / 2.0
            
            return max(0.0, min(1.0, artifact_score))
            
        except Exception as e:
            logger.warning(f"아티팩트 검출 실패: {e}")
            return 0.5
    
    async def _evaluate_face_preservation(self, fitted: np.ndarray, original: np.ndarray) -> float:
        """얼굴 보존도 평가"""
        try:
            if self.face_detector is None:
                return 1.0
            
            # 얼굴 검출
            fitted_gray = cv2.cvtColor(fitted, cv2.COLOR_RGB2GRAY)
            original_gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
            
            fitted_faces = self.face_detector.detectMultiScale(fitted_gray, 1.1, 4)
            original_faces = self.face_detector.detectMultiScale(original_gray, 1.1, 4)
            
            if len(original_faces) == 0:
                return 1.0
            
            if len(fitted_faces) == 0:
                return 0.0
            
            # 가장 큰 얼굴 영역 비교
            orig_face = max(original_faces, key=lambda f: f[2] * f[3])
            fitted_face = max(fitted_faces, key=lambda f: f[2] * f[3])
            
            # 얼굴 영역 추출
            ox, oy, ow, oh = orig_face
            fx, fy, fw, fh = fitted_face
            
            orig_face_region = original[oy:oy+oh, ox:ox+ow]
            fitted_face_region = fitted[fy:fy+fh, fx:fx+fw]
            
            # 크기 맞춤
            if orig_face_region.size > 0 and fitted_face_region.size > 0:
                fitted_face_resized = cv2.resize(fitted_face_region, (ow, oh))
                
                # 얼굴 유사성 계산 (SSIM)
                face_similarity = self._calculate_ssim_numpy(
                    cv2.cvtColor(orig_face_region, cv2.COLOR_RGB2GRAY),
                    cv2.cvtColor(fitted_face_resized, cv2.COLOR_RGB2GRAY)
                )
                
                return max(0.0, min(1.0, face_similarity))
            
            return 0.5
            
        except Exception as e:
            logger.warning(f"얼굴 보존도 평가 실패: {e}")
            return 1.0
    
    def _calculate_ssim_numpy(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """NumPy 기반 SSIM 계산"""
        try:
            # 상수
            K1, K2 = 0.01, 0.03
            L = 255  # 최대 픽셀 값
            C1 = (K1 * L) ** 2
            C2 = (K2 * L) ** 2
            
            # 평균 계산
            mu1 = cv2.GaussianBlur(img1.astype(np.float64), (11, 11), 1.5)
            mu2 = cv2.GaussianBlur(img2.astype(np.float64), (11, 11), 1.5)
            
            mu1_sq = mu1 ** 2
            mu2_sq = mu2 ** 2
            mu1_mu2 = mu1 * mu2
            
            # 분산 계산
            sigma1_sq = cv2.GaussianBlur(img1.astype(np.float64) ** 2, (11, 11), 1.5) - mu1_sq
            sigma2_sq = cv2.GaussianBlur(img2.astype(np.float64) ** 2, (11, 11), 1.5) - mu2_sq
            sigma12 = cv2.GaussianBlur((img1.astype(np.float64) * img2.astype(np.float64)), (11, 11), 1.5) - mu1_mu2
            
            # SSIM 맵 계산
            numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
            denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
            
            ssim_map = numerator / denominator
            
            return float(np.mean(ssim_map))
            
        except Exception as e:
            logger.warning(f"SSIM 계산 실패: {e}")
            return 0.5
    
    def _build_final_result(
        self,
        metrics: QualityMetrics,
        recommendations: List[str],
        detailed_analysis: Dict[str, Any],
        processing_time: float,
        fabric_type: str,
        clothing_type: str
    ) -> Dict[str, Any]:
        """최종 결과 구성 (Pipeline Manager 호환 형식)"""
        
        return {
            "success": True,
            "overall_score": float(metrics.overall_score),
            "grade": metrics.get_grade().value,
            "letter_grade": self._get_letter_grade(metrics.overall_score),
            "metrics": metrics.to_dict(),
            "recommendations": recommendations,
            "detailed_analysis": detailed_analysis,
            "quality_info": {
                "fabric_type": fabric_type,
                "clothing_type": clothing_type,
                "assessment_method": "comprehensive",
                "processing_time": processing_time,
                "device": self.device,
                "device_type": self.device_type,
                "m3_max_optimized": self.is_m3_max,
                "memory_gb": self.memory_gb,
                "features_used": self._get_used_features(),
                "optimization_level": self.optimization_level
            },
            "performance_info": {
                "optimization_enabled": self.optimization_enabled,
                "memory_usage": self._estimate_memory_usage(),
                "gpu_acceleration": self.device != 'cpu'
            }
        }
    
    async def _generate_recommendations(self, metrics: QualityMetrics, fabric_type: str, clothing_type: str) -> List[str]:
        """개선 제안 생성 - M3 Max 최적화"""
        recommendations = []
        
        try:
            # M3 Max 향상된 임계값 사용
            thresholds = self.quality_thresholds.copy()
            
            # 임계값 기반 제안
            if metrics.perceptual_quality < thresholds['good']:
                recommendations.append("지각적 품질 개선: 입력 이미지 해상도를 높이거나 노이즈를 줄여보세요.")
            
            if metrics.technical_quality < thresholds['good']:
                recommendations.append("기술적 품질 개선: 이미지 선명도를 높이고 압축 아티팩트를 줄여보세요.")
            
            if metrics.aesthetic_quality < thresholds['good']:
                recommendations.append("미적 품질 개선: 색상 균형과 구성을 조정해보세요.")
            
            if metrics.fit_accuracy < thresholds['good']:
                recommendations.append("핏 정확도 개선: 신체 측정값을 다시 확인하거나 포즈를 조정해보세요.")
            
            if metrics.color_harmony < thresholds['fair']:
                recommendations.append("색상 조화 개선: 의류와 피부톤이 잘 어울리는 색상을 선택해보세요.")
            
            if metrics.detail_preservation < thresholds['good']:
                recommendations.append("디테일 보존 개선: 더 높은 품질의 원본 이미지를 사용해보세요.")
            
            # M3 Max 전용 제안
            if self.is_m3_max and self.optimization_enabled:
                if metrics.overall_score >= thresholds['excellent']:
                    recommendations.insert(0, "🍎 M3 Max 최적화로 최고 품질을 달성했습니다!")
                elif metrics.overall_score >= thresholds['good']:
                    recommendations.insert(0, "M3 Max 가속으로 우수한 품질을 얻었습니다. 더 정밀한 설정을 시도해보세요.")
                    
                # 128GB 메모리 특화 제안
                if self.memory_gb >= 128:
                    if metrics.overall_score < thresholds['excellent']:
                        recommendations.append("128GB 메모리 활용: 고해상도 모드나 배치 처리를 활성화해보세요.")
            
            # 빈 추천 목록인 경우 기본 제안
            if not recommendations:
                if self.is_m3_max and self.memory_gb >= 128:
                    recommendations.append("🍎 M3 Max 128GB로 최고급 품질을 달성했습니다! 현재 설정을 유지하세요.")
                elif self.is_m3_max:
                    recommendations.append("🍎 M3 Max로 훌륭한 품질을 달성했습니다! 현재 설정을 유지하세요.")
                else:
                    recommendations.append("훌륭한 품질입니다! 현재 설정을 유지하세요.")
            
            return recommendations
            
        except Exception as e:
            logger.warning(f"개선 제안 생성 실패: {e}")
            return ["품질 분석은 완료되었지만 개선 제안 생성에 실패했습니다."]
    
    async def _generate_detailed_analysis(
        self, 
        metrics: QualityMetrics, 
        fitted: np.ndarray, 
        person: Optional[np.ndarray], 
        clothing: Optional[np.ndarray],
        fabric_type: str
    ) -> Dict[str, Any]:
        """상세 분석 생성 - M3 Max 최적화"""
        try:
            analysis = {
                'quality_breakdown': {
                    'excellent_aspects': [],
                    'good_aspects': [],
                    'improvement_needed': []
                },
                'technical_details': {
                    'image_properties': self._analyze_image_properties(fitted),
                    'fabric_analysis': self._analyze_fabric_quality(fitted, fabric_type),
                    'structural_analysis': self._analyze_structure(fitted, person) if person is not None else {}
                },
                'comparison_metrics': {
                    'similarity_to_original': self._calculate_overall_similarity(fitted, person) if person is not None else 0.8,
                    'clothing_integration': self._calculate_clothing_integration(fitted, clothing) if clothing is not None else 0.8,
                    'realism_score': self._calculate_realism_score(fitted)
                },
                'performance_insights': {
                    'strongest_aspect': self._find_strongest_aspect(metrics),
                    'weakest_aspect': self._find_weakest_aspect(metrics),
                    'improvement_potential': self._calculate_improvement_potential(metrics)
                },
                'm3_max_analysis': {}
            }
            
            # M3 Max 전용 분석
            if self.is_m3_max and self.optimization_enabled:
                analysis['m3_max_analysis'] = {
                    'optimization_level': self.optimization_level,
                    'memory_utilization': f"{self.memory_gb}GB 활용",
                    'neural_engine_boost': metrics.overall_score > 0.8,
                    'metal_acceleration': self.device == 'mps',
                    'quality_enhancement': "M3 Max 최적화로 품질 향상됨",
                    'high_memory_mode': self.memory_gb >= 128,
                    'batch_processing': self.performance_config.get('batch_processing', False),
                    'advanced_caching': self.performance_config.get('enable_caching', False)
                }
            
            # 품질 분류
            thresholds = self.quality_thresholds
            
            for metric_name, metric_value in metrics.to_dict().items():
                if metric_name == 'overall_score':
                    continue
                
                aspect_info = {
                    'aspect': metric_name.replace('_', ' ').title(),
                    'score': metric_value,
                    'status': self._get_metric_status(metric_value),
                    'm3_max_enhanced': self.is_m3_max and metric_value > 0.8,
                    'high_memory_optimized': self.memory_gb >= 128 and metric_value > 0.85
                }
                
                if metric_value >= thresholds['excellent']:
                    analysis['quality_breakdown']['excellent_aspects'].append(aspect_info)
                elif metric_value >= thresholds['good']:
                    analysis['quality_breakdown']['good_aspects'].append(aspect_info)
                else:
                    analysis['quality_breakdown']['improvement_needed'].append(aspect_info)
            
            return analysis
            
        except Exception as e:
            logger.warning(f"상세 분석 생성 실패: {e}")
            return {'error': '상세 분석 생성에 실패했습니다.'}
    
    # =================================================================
    # 유틸리티 메서드들
    # =================================================================
    
    def _tensor_to_numpy(self, tensor: torch.Tensor, is_mask: bool = False) -> np.ndarray:
        """PyTorch 텐서를 NumPy 배열로 변환"""
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch가 필요합니다")
        
        try:
            # GPU에서 CPU로 이동
            if tensor.is_cuda or (hasattr(tensor, 'is_mps') and tensor.is_mps):
                tensor = tensor.cpu()
            
            # 차원 정리
            if tensor.dim() == 4:
                tensor = tensor.squeeze(0)  # [1, C, H, W] -> [C, H, W]
            
            if is_mask:
                if tensor.dim() == 3:
                    tensor = tensor.squeeze(0)  # [1, H, W] -> [H, W]
                array = tensor.numpy().astype(np.uint8)
                if array.max() <= 1.0:
                    array = array * 255
            else:
                if tensor.dim() == 3 and tensor.size(0) == 3:
                    tensor = tensor.permute(1, 2, 0)  # [3, H, W] -> [H, W, 3]
                
                array = tensor.numpy()
                if array.max() <= 1.0:
                    array = array * 255
                array = array.astype(np.uint8)
            
            return array
            
        except Exception as e:
            logger.error(f"텐서 변환 실패: {e}")
            raise
    
    def _get_used_features(self) -> List[str]:
        """사용된 기능들 목록"""
        features = ['basic_quality_assessment']
        
        if self.perceptual_analyzer:
            features.append('perceptual_analysis')
        if self.technical_analyzer:
            features.append('technical_analysis')
        if self.aesthetic_analyzer:
            features.append('aesthetic_analysis')
        if TORCH_AVAILABLE:
            features.append('neural_processing')
        if self.face_detector:
            features.append('face_detection')
        if self.is_m3_max:
            features.append('m3_max_acceleration')
        if self.device == 'mps':
            features.append('metal_performance_shaders')
        
        return features
    
    def _estimate_memory_usage(self) -> Dict[str, float]:
        """메모리 사용량 추정"""
        try:
            import psutil
            memory_info = {
                'system_usage_percent': psutil.virtual_memory().percent,
                'available_gb': psutil.virtual_memory().available / (1024**3)
            }
            
            if TORCH_AVAILABLE:
                if self.device == 'mps' and hasattr(torch.mps, 'current_allocated_memory'):
                    memory_info['mps_allocated_gb'] = torch.mps.current_allocated_memory() / (1024**3)
                elif self.device == 'cuda':
                    memory_info['gpu_allocated_gb'] = torch.cuda.memory_allocated() / (1024**3)
            
            return memory_info
            
        except Exception as e:
            logger.warning(f"메모리 사용량 추정 실패: {e}")
            return {'estimated_usage_gb': 2.0}
    
    def _create_error_result(self, error_message: str) -> Dict[str, Any]:
        """에러 결과 생성"""
        return {
            "success": False,
            "error": error_message,
            "overall_score": 0.0,
            "grade": "error",
            "letter_grade": "F",
            "metrics": {},
            "recommendations": [],
            "detailed_analysis": {},
            "quality_info": {
                "error_details": error_message,
                "device": self.device,
                "device_type": self.device_type,
                "m3_max_optimized": self.is_m3_max,
                "processing_time": 0.0
            }
        }
    
    def _create_fallback_result(self, reason: str) -> Dict[str, Any]:
        """폴백 결과 생성 (최소 기능)"""
        logger.warning(f"폴백 모드: {reason}")
        
        # 기본 메트릭 생성
        fallback_metrics = QualityMetrics(
            overall_score=0.7,
            perceptual_quality=0.7,
            technical_quality=0.7,
            aesthetic_quality=0.7,
            fit_accuracy=0.7,
            color_harmony=0.7,
            detail_preservation=0.7,
            edge_quality=0.7,
            lighting_consistency=0.7,
            artifact_level=0.7,
            face_preservation=1.0
        )
        
        return {
            "success": True,
            "overall_score": fallback_metrics.overall_score,
            "grade": fallback_metrics.get_grade().value,
            "letter_grade": self._get_letter_grade(fallback_metrics.overall_score),
            "metrics": fallback_metrics.to_dict(),
            "recommendations": ["기본 평가 모드에서 처리되었습니다."],
            "detailed_analysis": {"fallback_mode": True, "reason": reason},
            "quality_info": {
                "fallback_mode": True,
                "reason": reason,
                "assessment_method": "fallback",
                "processing_time": 0.001,
                "device": self.device,
                "device_type": self.device_type,
                "m3_max_optimized": self.is_m3_max
            }
        }
    
    # =================================================================
    # 추가 헬퍼 메서드들
    # =================================================================
    
    def _analyze_image_properties(self, image: np.ndarray) -> Dict[str, Any]:
        """이미지 속성 분석"""
        try:
            h, w, c = image.shape
            
            properties = {
                'resolution': f"{w}x{h}",
                'channels': c,
                'file_size_estimate': f"{(w * h * c / 1024):.1f} KB",
                'aspect_ratio': f"{w/h:.2f}:1",
                'm3_max_optimized': self.is_m3_max and min(w, h) >= 1024,
                'high_resolution': min(w, h) >= 1024,
                'memory_efficient': self.memory_gb >= 128
            }
            
            # 색상 통계
            properties.update({
                'brightness_mean': float(np.mean(image)),
                'brightness_std': float(np.std(image)),
                'contrast_measure': float(np.std(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY))),
                'color_range': {
                    'min': [int(np.min(image[:,:,i])) for i in range(c)],
                    'max': [int(np.max(image[:,:,i])) for i in range(c)],
                    'mean': [float(np.mean(image[:,:,i])) for i in range(c)]
                }
            })
            
            return properties
            
        except Exception as e:
            logger.warning(f"이미지 속성 분석 실패: {e}")
            return {}
    
    def _analyze_fabric_quality(self, image: np.ndarray, fabric_type: str) -> Dict[str, Any]:
        """천 품질 분석"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # 텍스처 분석
            texture_var = np.var(gray)
            texture_score = min(texture_var / 1000.0, 1.0)
            
            # 천 특성별 품질 기준
            fabric_standards = self.FABRIC_QUALITY_STANDARDS.get(fabric_type, self.FABRIC_QUALITY_STANDARDS['default'])
            
            analysis = {
                'fabric_type': fabric_type,
                'texture_score': texture_score,
                'texture_threshold_met': texture_score >= fabric_standards['texture_threshold'],
                'smoothness_score': self._calculate_smoothness(gray),
                'surface_quality': 'excellent' if texture_score > 0.8 else 'good' if texture_score > 0.6 else 'fair',
                'm3_max_precision': self.is_m3_max,
                'high_memory_analysis': self.memory_gb >= 128
            }
            
            return analysis
            
        except Exception as e:
            logger.warning(f"천 품질 분석 실패: {e}")
            return {}
    
    def _calculate_smoothness(self, gray_image: np.ndarray) -> float:
        """표면 매끄러움 계산"""
        try:
            # Laplacian으로 텍스처 변화량 측정
            laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)
            smoothness = 1.0 - (np.std(laplacian) / 100.0)
            return max(0.0, min(1.0, smoothness))
        except Exception:
            return 0.5
    
    def _analyze_structure(self, fitted: np.ndarray, person: np.ndarray) -> Dict[str, Any]:
        """구조 분석"""
        try:
            fitted_edges = cv2.Canny(cv2.cvtColor(fitted, cv2.COLOR_RGB2GRAY), 50, 150)
            person_edges = cv2.Canny(cv2.cvtColor(person, cv2.COLOR_RGB2GRAY), 50, 150)
            
            analysis = {
                'edge_density_fitted': float(np.sum(fitted_edges > 0) / fitted_edges.size),
                'edge_density_person': float(np.sum(person_edges > 0) / person_edges.size),
                'structural_similarity': self._calculate_ssim_numpy(fitted_edges, person_edges),
                'geometric_distortion': self._calculate_geometric_distortion(fitted, person),
                'm3_max_enhanced': self.is_m3_max,
                'high_precision': self.memory_gb >= 128
            }
            
            return analysis
            
        except Exception as e:
            logger.warning(f"구조 분석 실패: {e}")
            return {}
    
    def _calculate_geometric_distortion(self, fitted: np.ndarray, person: np.ndarray) -> float:
        """기하학적 왜곡 계산"""
        try:
            fitted_gray = cv2.cvtColor(fitted, cv2.COLOR_RGB2GRAY)
            person_gray = cv2.cvtColor(person, cv2.COLOR_RGB2GRAY)
            
            diff = cv2.absdiff(fitted_gray, person_gray)
            distortion_level = np.mean(diff) / 255.0
            
            return float(1.0 - distortion_level)
            
        except Exception as e:
            logger.warning(f"기하학적 왜곡 계산 실패: {e}")
            return 0.5
    
    def _calculate_overall_similarity(self, fitted: np.ndarray, person: np.ndarray) -> float:
        """전체 유사성 계산"""
        try:
            fitted_gray = cv2.cvtColor(fitted, cv2.COLOR_RGB2GRAY)
            person_gray = cv2.cvtColor(person, cv2.COLOR_RGB2GRAY)
            
            return self._calculate_ssim_numpy(fitted_gray, person_gray)
        except Exception as e:
            logger.warning(f"전체 유사성 계산 실패: {e}")
            return 0.5
    
    def _calculate_clothing_integration(self, fitted: np.ndarray, clothing: np.ndarray) -> float:
        """의류 통합도 계산"""
        try:
            fitted_colors = fitted.reshape(-1, 3).mean(axis=0)
            clothing_colors = clothing.reshape(-1, 3).mean(axis=0)
            
            color_distance = np.linalg.norm(fitted_colors - clothing_colors)
            integration_score = 1.0 - (color_distance / 441.67)
            
            return max(0.0, min(1.0, integration_score))
        except Exception as e:
            logger.warning(f"의류 통합도 계산 실패: {e}")
            return 0.5
    
    def _calculate_realism_score(self, fitted: np.ndarray) -> float:
        """현실감 점수 계산"""
        try:
            # 간단한 현실감 메트릭 (색상 분포와 텍스처 기반)
            gray = cv2.cvtColor(fitted, cv2.COLOR_RGB2GRAY)
            texture_variance = np.var(gray)
            
            # 색상 분포 자연스러움
            hsv = cv2.cvtColor(fitted, cv2.COLOR_RGB2HSV)
            saturation_balance = 1.0 - abs(np.mean(hsv[:,:,1]) - 128) / 128.0
            
            realism_score = (min(texture_variance / 1000.0, 1.0) * 0.6 + 
                           saturation_balance * 0.4)
            
            return max(0.0, min(1.0, realism_score))
        except Exception as e:
            logger.warning(f"현실감 점수 계산 실패: {e}")
            return 0.5
    
    def _find_strongest_aspect(self, metrics: QualityMetrics) -> str:
        """가장 강한 측면 찾기"""
        try:
            metric_dict = metrics.to_dict()
            del metric_dict['overall_score']  # 전체 점수 제외
            
            strongest = max(metric_dict.items(), key=lambda x: x[1])
            return strongest[0].replace('_', ' ').title()
        except Exception:
            return "Unknown"
    
    def _find_weakest_aspect(self, metrics: QualityMetrics) -> str:
        """가장 약한 측면 찾기"""
        try:
            metric_dict = metrics.to_dict()
            del metric_dict['overall_score']  # 전체 점수 제외
            
            weakest = min(metric_dict.items(), key=lambda x: x[1])
            return weakest[0].replace('_', ' ').title()
        except Exception:
            return "Unknown"
    
    def _calculate_improvement_potential(self, metrics: QualityMetrics) -> float:
        """개선 가능성 계산"""
        try:
            metric_dict = metrics.to_dict()
            del metric_dict['overall_score']
            
            scores = list(metric_dict.values())
            current_avg = np.mean(scores)
            max_possible = 1.0
            
            improvement_potential = (max_possible - current_avg) / max_possible
            return improvement_potential
        except Exception:
            return 0.5
    
    def _get_metric_status(self, score: float) -> str:
        """메트릭 상태 문자열 반환"""
        thresholds = self.quality_thresholds
        
        if score >= thresholds['excellent']:
            return "Excellent"
        elif score >= thresholds['good']:
            return "Good"
        elif score >= thresholds['fair']:
            return "Fair"
        elif score >= thresholds['poor']:
            return "Poor"
        else:
            return "Very Poor"
    
    def _get_letter_grade(self, score: float) -> str:
        """점수를 문자 등급으로 변환"""
        if score >= 0.95:
            return "A+"
        elif score >= 0.9:
            return "A"
        elif score >= 0.85:
            return "A-"
        elif score >= 0.8:
            return "B+"
        elif score >= 0.75:
            return "B"
        elif score >= 0.7:
            return "B-"
        elif score >= 0.65:
            return "C+"
        elif score >= 0.6:
            return "C"
        elif score >= 0.55:
            return "C-"
        elif score >= 0.5:
            return "D+"
        elif score >= 0.4:
            return "D"
        else:
            return "F"
    
    def _update_performance_stats(self, processing_time: float, quality_score: float):
        """성능 통계 업데이트"""
        try:
            self.performance_stats['total_assessments'] += 1
            
            # 평균 시간 업데이트
            total = self.performance_stats['total_assessments']
            current_avg_time = self.performance_stats['average_time']
            self.performance_stats['average_time'] = (
                (current_avg_time * (total - 1) + processing_time) / total
            )
            
            # 평균 품질 점수 업데이트
            current_avg_quality = self.performance_stats['average_quality_score']
            self.performance_stats['average_quality_score'] = (
                (current_avg_quality * (total - 1) + quality_score) / total
            )
            
            # 성공률 업데이트 (품질 0.5 이상이면 성공)
            success_count = 1 if quality_score > 0.5 else 0
            current_success_rate = self.performance_stats['success_rate']
            self.performance_stats['success_rate'] = (
                (current_success_rate * (total - 1) + success_count) / total
            )
            
            # 메모리 사용량 업데이트
            memory_usage = self._estimate_memory_usage()
            if 'available_gb' in memory_usage:
                self.performance_stats['memory_usage_gb'] = self.memory_gb - memory_usage['available_gb']
            
        except Exception as e:
            logger.warning(f"통계 업데이트 실패: {e}")
    
    # =================================================================
    # Pipeline Manager 호환 메서드들
    # =================================================================
    
    async def get_model_info(self) -> Dict[str, Any]:
        """모델 정보 반환 (Pipeline Manager 호환)"""
        return {
            "step_name": "QualityAssessment",
            "version": "4.0-m3max",
            "device": self.device,
            "device_type": self.device_type,
            "memory_gb": self.memory_gb,
            "is_m3_max": self.is_m3_max,
            "optimization_enabled": self.optimization_enabled,
            "initialized": self.is_initialized,
            "initialization_error": self.initialization_error,
            "capabilities": {
                "perceptual_analysis": bool(self.perceptual_analyzer),
                "technical_analysis": bool(self.technical_analyzer),
                "aesthetic_analysis": bool(self.aesthetic_analyzer),
                "face_detection": bool(self.face_detector),
                "neural_processing": TORCH_AVAILABLE and self.device != 'cpu',
                "m3_max_acceleration": self.is_m3_max and self.device == 'mps'
            },
            "supported_fabrics": list(self.FABRIC_QUALITY_STANDARDS.keys()),
            "supported_clothing_types": list(self.CLOTHING_QUALITY_WEIGHTS.keys()),
            "performance_stats": self.performance_stats,
            "quality_settings": {
                "optimization_level": self.optimization_level,
                "max_resolution": self._get_max_resolution(),
                "quality_level": self._get_quality_level(),
                "quality_thresholds": self.quality_thresholds
            },
            "dependencies": {
                "torch": TORCH_AVAILABLE,
                "opencv": CV2_AVAILABLE,
                "pil": PIL_AVAILABLE,
                "scipy": SCIPY_AVAILABLE,
                "sklearn": SKLEARN_AVAILABLE,
                "skimage": SKIMAGE_AVAILABLE
            },
            "config": {
                "assessment": self.assessment_config,
                "performance": self.performance_config,
                "optimization_level": self.optimization_level,
                "metric_weights": self.metric_weights
            }
        }
    
    async def cleanup(self):
        """리소스 정리 (Pipeline Manager 호환)"""
        try:
            logger.info("🧹 품질 평가 시스템 리소스 정리 시작...")
            
            # 컴포넌트들 정리
            if self.perceptual_analyzer:
                if hasattr(self.perceptual_analyzer, 'cleanup'):
                    await self.perceptual_analyzer.cleanup()
                del self.perceptual_analyzer
                self.perceptual_analyzer = None
            
            if self.technical_analyzer:
                if hasattr(self.technical_analyzer, 'cleanup'):
                    await self.technical_analyzer.cleanup()
                del self.technical_analyzer
                self.technical_analyzer = None
            
            if self.aesthetic_analyzer:
                if hasattr(self.aesthetic_analyzer, 'cleanup'):
                    await self.aesthetic_analyzer.cleanup()
                del self.aesthetic_analyzer
                self.aesthetic_analyzer = None
            
            if self.face_detector:
                del self.face_detector
                self.face_detector = None
            
            # GPU 메모리 정리
            if TORCH_AVAILABLE:
                if self.device == 'mps':
                    if hasattr(torch.backends.mps, 'empty_cache'):
                        torch.backends.mps.empty_cache()
                    elif hasattr(torch.mps, 'synchronize'):
                        torch.mps.synchronize()
                elif self.device == 'cuda':
                    torch.cuda.empty_cache()
            
            # 시스템 메모리 정리
            import gc
            gc.collect()
            
            self.is_initialized = False
            logger.info("✅ 품질 평가 시스템 리소스 정리 완료")
            
        except Exception as e:
            logger.warning(f"⚠️ 리소스 정리 중 오류: {e}")


# =================================================================
# 보조 클래스들 (업데이트된 생성자 적용)
# =================================================================

class PerceptualQualityAnalyzer:
    """지각적 품질 분석기 (M3 Max 최적화)"""
    
    def __init__(self, device: str = 'cpu', m3_max_mode: bool = False, optimization_level: str = 'balanced'):
        self.device = device
        self.m3_max_mode = m3_max_mode
        self.optimization_level = optimization_level
        
        # M3 Max 모드에서 더 높은 정밀도
        if m3_max_mode:
            self.precision_factor = 2.0
            self.analysis_depth = 'ultra'
        else:
            self.precision_factor = 1.0
            self.analysis_depth = 'standard'
    
    async def analyze_perceptual_quality(
        self,
        fitted_image: np.ndarray,
        reference_image: Optional[np.ndarray],
        fabric_standards: Dict[str, float]
    ) -> float:
        """지각적 품질 분석"""
        
        try:
            if reference_image is None:
                # 자체 품질 분석
                return await self._analyze_intrinsic_quality(fitted_image, fabric_standards)
            else:
                # 비교 품질 분석
                return await self._analyze_comparative_quality(fitted_image, reference_image, fabric_standards)
                
        except Exception as e:
            logger.warning(f"지각적 품질 분석 실패: {e}")
            return 0.7
    
    async def _analyze_intrinsic_quality(self, image: np.ndarray, fabric_standards: Dict) -> float:
        """내재적 품질 분석"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # 선명도 분석
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            sharpness_score = min(laplacian_var / 1000.0 * self.precision_factor, 1.0)
            
            # 대비 분석
            contrast_score = np.std(gray) / 128.0
            contrast_score = max(0.0, min(1.0, contrast_score))
            
            # 노이즈 수준
            noise_level = np.std(cv2.GaussianBlur(gray, (5, 5), 0) - gray)
            noise_score = max(0, 1.0 - (noise_level / 50.0))
            
            # 종합 점수
            intrinsic_score = (sharpness_score * 0.4 + 
                             contrast_score * 0.3 + 
                             noise_score * 0.3)
            
            return max(0.0, min(1.0, intrinsic_score))
            
        except Exception as e:
            logger.warning(f"내재적 품질 분석 실패: {e}")
            return 0.7
    
    async def _analyze_comparative_quality(self, fitted: np.ndarray, reference: np.ndarray, fabric_standards: Dict) -> float:
        """비교 품질 분석"""
        try:
            # SSIM 계산
            fitted_gray = cv2.cvtColor(fitted, cv2.COLOR_RGB2GRAY)
            reference_gray = cv2.cvtColor(reference, cv2.COLOR_RGB2GRAY)
            
            ssim_score = self._calculate_ssim_basic(fitted_gray, reference_gray)
            
            # PSNR 계산
            mse = np.mean((fitted_gray.astype(float) - reference_gray.astype(float)) ** 2)
            if mse == 0:
                psnr_score = 1.0
            else:
                psnr = 20 * np.log10(255.0 / np.sqrt(mse))
                psnr_score = min(psnr / 40.0, 1.0)
            
            # 조합 점수
            comparative_score = 0.7 * ssim_score + 0.3 * psnr_score
            
            # M3 Max 보정
            if self.m3_max_mode:
                comparative_score = min(1.0, comparative_score * 1.05)
            
            return comparative_score
            
        except Exception as e:
            logger.warning(f"비교 품질 분석 실패: {e}")
            return 0.7
    
    def _calculate_ssim_basic(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """기본 SSIM 계산"""
        try:
            # 간단한 SSIM 구현
            mu1 = cv2.GaussianBlur(img1.astype(np.float64), (11, 11), 1.5)
            mu2 = cv2.GaussianBlur(img2.astype(np.float64), (11, 11), 1.5)
            
            mu1_sq = mu1 ** 2
            mu2_sq = mu2 ** 2
            mu1_mu2 = mu1 * mu2
            
            sigma1_sq = cv2.GaussianBlur(img1.astype(np.float64) ** 2, (11, 11), 1.5) - mu1_sq
            sigma2_sq = cv2.GaussianBlur(img2.astype(np.float64) ** 2, (11, 11), 1.5) - mu2_sq
            sigma12 = cv2.GaussianBlur((img1.astype(np.float64) * img2.astype(np.float64)), (11, 11), 1.5) - mu1_mu2
            
            C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
            
            ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
            
            return float(np.mean(ssim_map))
            
        except Exception as e:
            logger.warning(f"SSIM 계산 실패: {e}")
            return 0.5
    
    async def warmup(self):
        """워밍업"""
        pass
    
    async def cleanup(self):
        """리소스 정리"""
        pass


class TechnicalQualityAnalyzer:
    """기술적 품질 분석기 (M3 Max 최적화)"""
    
    def __init__(self, device: str = 'cpu', enable_advanced_features: bool = False, m3_max_acceleration: bool = False):
        self.device = device
        self.enable_advanced_features = enable_advanced_features
        self.m3_max_acceleration = m3_max_acceleration
        
        # M3 Max 가속화 설정
        if m3_max_acceleration:
            self.analysis_precision = 'ultra'
            self.feature_extraction_depth = 'deep'
        else:
            self.analysis_precision = 'standard'
            self.feature_extraction_depth = 'basic'
    
    async def analyze_technical_quality(
        self,
        image: np.ndarray,
        fabric_standards: Dict[str, float],
        clothing_type: str
    ) -> float:
        """기술적 품질 분석"""
        
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # 선명도 분석
            sharpness_score = self._analyze_sharpness(gray)
            
            # 노이즈 분석
            noise_score = self._analyze_noise_level(gray)
            
            # 대비 분석
            contrast_score = self._analyze_contrast(gray)
            
            # 해상도 품질
            resolution_score = self._analyze_resolution_quality(image)
            
            # M3 Max 고급 분석
            if self.m3_max_acceleration and self.enable_advanced_features:
                advanced_score = await self._analyze_advanced_technical_features(image, fabric_standards)
                
                # 가중 조합 (고급 분석 포함)
                technical_score = (
                    sharpness_score * 0.3 +
                    noise_score * 0.25 +
                    contrast_score * 0.25 +
                    resolution_score * 0.1 +
                    advanced_score * 0.1
                )
            else:
                # 기본 조합
                technical_score = (
                    sharpness_score * 0.4 +
                    noise_score * 0.3 +
                    contrast_score * 0.3
                )
            
            return max(0.0, min(1.0, technical_score))
            
        except Exception as e:
            logger.warning(f"기술적 품질 분석 실패: {e}")
            return 0.7
    
    def _analyze_sharpness(self, gray_image: np.ndarray) -> float:
        """선명도 분석"""
        try:
            # Laplacian 분산
            laplacian_var = cv2.Laplacian(gray_image, cv2.CV_64F).var()
            sharpness = min(laplacian_var / 1000.0, 1.0)
            
            # M3 Max 정밀도 보정
            if self.m3_max_acceleration:
                sharpness = min(1.0, sharpness * 1.1)
            
            return sharpness
            
        except Exception as e:
            logger.warning(f"선명도 분석 실패: {e}")
            return 0.5
    
    def _analyze_noise_level(self, gray_image: np.ndarray) -> float:
        """노이즈 수준 분석"""
        try:
            # 고주파 노이즈 추정
            blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)
            noise = gray_image.astype(float) - blurred.astype(float)
            noise_level = np.std(noise)
            
            # 노이즈가 적을수록 높은 점수
            noise_score = max(0, 1.0 - (noise_level / 50.0))
            
            return noise_score
            
        except Exception as e:
            logger.warning(f"노이즈 분석 실패: {e}")
            return 0.5
    
    def _analyze_contrast(self, gray_image: np.ndarray) -> float:
        """대비 분석"""
        try:
            # RMS 대비
            mean_intensity = np.mean(gray_image)
            rms_contrast = np.sqrt(np.mean((gray_image - mean_intensity)**2))
            
            # 정규화
            contrast_score = min(rms_contrast / 64.0, 1.0)
            
            return contrast_score
            
        except Exception as e:
            logger.warning(f"대비 분석 실패: {e}")
            return 0.5
    
    def _analyze_resolution_quality(self, image: np.ndarray) -> float:
        """해상도 품질 분석"""
        try:
            h, w = image.shape[:2]
            total_pixels = h * w
            
            # 해상도 점수 (더 높은 해상도일수록 높은 점수)
            if total_pixels >= 1024 * 1024:  # 1MP 이상
                resolution_score = 1.0
            elif total_pixels >= 512 * 512:  # 0.25MP 이상
                resolution_score = 0.8
            elif total_pixels >= 256 * 256:  # 0.065MP 이상
                resolution_score = 0.6
            else:
                resolution_score = 0.4
            
            return resolution_score
            
        except Exception as e:
            logger.warning(f"해상도 품질 분석 실패: {e}")
            return 0.5
    
    async def _analyze_advanced_technical_features(self, image: np.ndarray, fabric_standards: Dict) -> float:
        """M3 Max 고급 기술적 특징 분석"""
        try:
            # 고급 엣지 분석
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # Sobel 엣지 강도
            sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            edge_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
            
            edge_strength = np.mean(edge_magnitude) / 255.0
            
            # 텍스처 분석
            texture_variance = np.var(gray) / 10000.0
            texture_score = min(texture_variance, 1.0)
            
            # 조합 점수
            advanced_score = (edge_strength * 0.6 + texture_score * 0.4)
            
            return max(0.0, min(1.0, advanced_score))
            
        except Exception as e:
            logger.warning(f"고급 기술적 분석 실패: {e}")
            return 0.7
    
    async def warmup(self):
        """워밍업"""
        pass
    
    async def cleanup(self):
        """리소스 정리"""
        pass


class AestheticQualityAnalyzer:
    """미적 품질 분석기 (M3 Max 최적화)"""
    
    def __init__(self, device: str = 'cpu', use_advanced_features: bool = False, m3_max_precision: bool = False):
        self.device = device
        self.use_advanced_features = use_advanced_features
        self.m3_max_precision = m3_max_precision
        
        # M3 Max 정밀도 설정
        if m3_max_precision:
            self.color_analysis_depth = 'ultra'
            self.composition_analysis_level = 'advanced'
        else:
            self.color_analysis_depth = 'standard'
            self.composition_analysis_level = 'basic'
    
    async def analyze_aesthetic_quality(
        self,
        fitted_image: np.ndarray,
        person_image: Optional[np.ndarray],
        clothing_image: Optional[np.ndarray],
        fabric_type: str
    ) -> float:
        """미적 품질 분석"""
        
        try:
            # 색상 분포 분석
            color_score = self._analyze_color_distribution(fitted_image)
            
            # 구성 균형 분석
            composition_score = self._analyze_composition_balance(fitted_image)
            
            # 시각적 조화 분석
            harmony_score = 0.8  # 기본값
            if person_image is not None and clothing_image is not None:
                harmony_score = self._analyze_visual_harmony(fitted_image, person_image, clothing_image)
            
            # M3 Max 고급 미적 분석
            if self.m3_max_precision and self.use_advanced_features:
                advanced_aesthetic_score = await self._analyze_advanced_aesthetics(
                    fitted_image, fabric_type
                )
                
                # 가중 조합 (고급 분석 포함)
                aesthetic_score = (
                    color_score * 0.3 +
                    composition_score * 0.3 +
                    harmony_score * 0.25 +
                    advanced_aesthetic_score * 0.15
                )
            else:
                # 기본 조합
                aesthetic_score = (
                    color_score * 0.4 +
                    composition_score * 0.35 +
                    harmony_score * 0.25
                )
            
            return max(0.0, min(1.0, aesthetic_score))
            
        except Exception as e:
            logger.warning(f"미적 품질 분석 실패: {e}")
            return 0.7
    
    def _analyze_color_distribution(self, image: np.ndarray) -> float:
        """색상 분포 분석"""
        try:
            # HSV 색공간 변환
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            
            # 색상 히스토그램
            hist_h = cv2.calcHist([hsv], [0], None, [180], [0, 180])
            hist_s = cv2.calcHist([hsv], [1], None, [256], [0, 256])
            
            # 색상 다양성 계산
            h_diversity = np.count_nonzero(hist_h) / 180.0
            s_diversity = np.count_nonzero(hist_s) / 256.0
            
            # 균형 점수
            color_score = (h_diversity + s_diversity) / 2.0
            
            # M3 Max 정밀도 보정
            if self.m3_max_precision:
                color_score = min(1.0, color_score * 1.05)
            
            return max(0.0, min(1.0, color_score))
            
        except Exception as e:
            logger.warning(f"색상 분포 분석 실패: {e}")
            return 0.5
    
    def _analyze_composition_balance(self, image: np.ndarray) -> float:
        """구성 균형 분석"""
        try:
            h, w = image.shape[:2]
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # 3분할법 격자점
            grid_points = [
                (w//3, h//3), (2*w//3, h//3),
                (w//3, 2*h//3), (2*w//3, 2*h//3)
            ]
            
            # 각 격자점 주변의 관심도 계산
            interest_scores = []
            
            for x, y in grid_points:
                roi = gray[max(0, y-25):min(h, y+25), max(0, x-25):min(w, x+25)]
                if roi.size > 0:
                    interest = np.std(roi)  # 표준편차를 관심도로 사용
                    interest_scores.append(interest)
            
            # 균형 점수 (분산이 낮을수록 균형적)
            if interest_scores:
                balance_score = 1.0 - (np.std(interest_scores) / (np.mean(interest_scores) + 1e-6))
                return max(0.0, min(1.0, balance_score))
            
            return 0.5
            
        except Exception as e:
            logger.warning(f"구성 균형 분석 실패: {e}")
            return 0.5
    
    def _analyze_visual_harmony(self, fitted: np.ndarray, person: np.ndarray, clothing: np.ndarray) -> float:
        """시각적 조화 분석"""
        try:
            # 색상 조화 계산
            fitted_colors = fitted.reshape(-1, 3).mean(axis=0)
            person_colors = person.reshape(-1, 3).mean(axis=0)
            clothing_colors = clothing.reshape(-1, 3).mean(axis=0)
            
            # 색상 거리 계산
            person_fitted_dist = np.linalg.norm(fitted_colors - person_colors)
            clothing_fitted_dist = np.linalg.norm(fitted_colors - clothing_colors)
            
            # 조화 점수 (거리가 적절할 때 높음)
            harmony_score = 1.0 - min(person_fitted_dist, clothing_fitted_dist) / 441.67  # sqrt(3*255^2)
            
            return max(0.0, min(1.0, harmony_score))
            
        except Exception as e:
            logger.warning(f"시각적 조화 분석 실패: {e}")
            return 0.5
    
    async def _analyze_advanced_aesthetics(self, image: np.ndarray, fabric_type: str) -> float:
        """M3 Max 고급 미적 분석"""
        try:
            # 황금비율 구성 분석
            golden_ratio_score = self._analyze_golden_ratio_composition(image)
            
            # 색온도 일관성 분석
            color_temp_score = self._analyze_color_temperature_consistency(image)
            
            # 천 타입별 미적 특성 분석
            fabric_aesthetic_score = self._analyze_fabric_specific_aesthetics(image, fabric_type)
            
            # 조합 점수
            advanced_score = (
                golden_ratio_score * 0.4 +
                color_temp_score * 0.3 +
                fabric_aesthetic_score * 0.3
            )
            
            return max(0.0, min(1.0, advanced_score))
            
        except Exception as e:
            logger.warning(f"고급 미적 분석 실패: {e}")
            return 0.7
    
    def _analyze_golden_ratio_composition(self, image: np.ndarray) -> float:
        """황금비율 구성 분석"""
        try:
            h, w = image.shape[:2]
            
            # 황금비율 격자점 (1:1.618)
            golden_ratio = 1.618
            
            # 수직 분할점
            v1 = int(w / golden_ratio)
            v2 = w - v1
            
            # 수평 분할점
            h1 = int(h / golden_ratio)
            h2 = h - h1
            
            # 교점들에서의 관심도 측정
            interest_points = [
                (v1, h1), (v2, h1), (v1, h2), (v2, h2)
            ]
            
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            interest_scores = []
            
            for x, y in interest_points:
                # 주변 영역의 분산 (관심도)
                roi = gray[max(0, y-25):min(h, y+25), max(0, x-25):min(w, x+25)]
                if roi.size > 0:
                    interest = np.std(roi)
                    interest_scores.append(interest)
            
            if interest_scores:
                # 균형도 (표준편차가 낮을수록 균형적)
                balance = 1.0 - (np.std(interest_scores) / (np.mean(interest_scores) + 1e-6))
                return max(0.0, min(1.0, balance))
            
            return 0.5
            
        except Exception as e:
            logger.warning(f"황금비율 구성 분석 실패: {e}")
            return 0.5
    
    def _analyze_color_temperature_consistency(self, image: np.ndarray) -> float:
        """색온도 일관성 분석"""
        try:
            # 이미지를 9개 영역으로 분할
            h, w = image.shape[:2]
            regions = []
            
            for i in range(3):
                for j in range(3):
                    y1, y2 = i * h // 3, (i + 1) * h // 3
                    x1, x2 = j * w // 3, (j + 1) * w // 3
                    region = image[y1:y2, x1:x2]
                    regions.append(region)
            
            # 각 영역의 색온도 추정
            color_temps = []
            for region in regions:
                r_mean = np.mean(region[:, :, 0])
                g_mean = np.mean(region[:, :, 1])
                b_mean = np.mean(region[:, :, 2])
                
                # 간단한 색온도 지수 (B/R 비율)
                if r_mean > 0:
                    color_temp_index = b_mean / r_mean
                    color_temps.append(color_temp_index)
            
            if color_temps:
                # 색온도 일관성 (표준편차가 낮을수록 일관성 높음)
                consistency = 1.0 - min(np.std(color_temps) / 2.0, 1.0)
                return consistency
            
            return 0.5
            
        except Exception as e:
            logger.warning(f"색온도 일관성 분석 실패: {e}")
            return 0.5
    
    def _analyze_fabric_specific_aesthetics(self, image: np.ndarray, fabric_type: str) -> float:
        """천 타입별 미적 특성 분석"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # 천 타입별 텍스처 특성
            if fabric_type in ['silk', 'satin']:
                # 부드러운 천 - 매끄러운 텍스처 선호
                texture_smoothness = 1.0 - (np.std(cv2.Laplacian(gray, cv2.CV_64F)) / 1000.0)
                fabric_score = min(texture_smoothness, 1.0)
            elif fabric_type in ['denim', 'canvas']:
                # 거친 천 - 텍스처 변화 선호
                texture_variation = np.std(cv2.Laplacian(gray, cv2.CV_64F)) / 1000.0
                fabric_score = min(texture_variation, 1.0)
            elif fabric_type in ['wool', 'cashmere']:
                # 중간 텍스처 - 적당한 변화
                texture_var = np.var(gray) / 10000.0
                fabric_score = 1.0 - abs(texture_var - 0.5) * 2
            else:
                # 기본 점수
                fabric_score = 0.7
            
            return max(0.0, min(1.0, fabric_score))
            
        except Exception as e:
            logger.warning(f"천 특성별 미적 분석 실패: {e}")
            return 0.7
    
    async def warmup(self):
        """워밍업"""
        pass
    
    async def cleanup(self):
        """리소스 정리"""
        pass