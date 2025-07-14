# app/ai_pipeline/steps/step_08_quality_assessment.py
"""
🎯 완전히 작동하는 8단계: 품질 평가 (Quality Assessment)
실제 메트릭 계산 + 자동 개선 제안 + 상세 분석 - model_loader 수정 버전
"""
import os
import time
import logging
import asyncio
from typing import Dict, Any, Optional, List, Tuple, Union
import numpy as np
import cv2
from PIL import Image, ImageEnhance, ImageStat
import json
from dataclasses import dataclass, asdict
from enum import Enum
import base64
import io
from scipy import stats
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# PyTorch는 선택적 사용
try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    print("⚠️ PyTorch 없음 - 기본 메트릭으로 실행")
    TORCH_AVAILABLE = False

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

class RealQualityAssessmentStep:
    """
    🎯 실제로 작동하는 품질 평가 시스템
    
    특징:
    - 실제 SSIM, PSNR, MSE 계산
    - 컴퓨터 비전 기반 품질 메트릭
    - 자동 개선 제안 생성
    - 상세한 분석 리포트
    - 얼굴 보존도 평가
    - 색상 조화 분석
    """
    
    def __init__(self, device: str = 'cpu', config: Dict[str, Any] = None):
        """
        Args:
            device: 디바이스 ('cpu', 'mps', 'cuda')
            config: 설정 딕셔너리
        """
        self.device = device
        self.config = config or {}
        
        # model_loader는 내부에서 생성하거나 전역에서 가져옴
        from app.ai_pipeline.utils.model_loader import get_global_model_loader
        self.model_loader = get_global_model_loader()
        
        # 평가 설정
        self.enable_advanced_metrics = self.config.get('enable_advanced_metrics', True)
        self.enable_face_detection = self.config.get('enable_face_detection', True)
        self.enable_detailed_analysis = self.config.get('enable_detailed_analysis', True)
        
        # 품질 임계값
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
        
        # 얼굴 검출기
        self.face_detector = None
        
        # PyTorch 최적화
        self.use_torch = TORCH_AVAILABLE and self.config.get('use_torch', True)
        
        self.is_initialized = False
        
        logger.info(f"📊 실제 품질 평가 시스템 초기화 - 디바이스: {device}")
    
    async def initialize(self) -> bool:
        """실제 초기화"""
        try:
            logger.info("🔄 품질 평가 시스템 로딩 중...")
            
            # 얼굴 검출기 초기화
            if self.enable_face_detection:
                await self._initialize_face_detector()
            
            self.is_initialized = True
            logger.info("✅ 품질 평가 시스템 초기화 완료")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ 품질 평가 시스템 초기화 실패: {e}")
            self.is_initialized = False
            return False
    
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
    
    async def process(
        self,
        fitted_image: Union[np.ndarray, Image.Image, str],
        original_person: Union[np.ndarray, Image.Image, str],
        original_clothing: Union[np.ndarray, Image.Image, str],
        pipeline_results: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        실제 품질 평가 처리
        
        Args:
            fitted_image: 최종 가상 피팅 결과
            original_person: 원본 사용자 이미지
            original_clothing: 원본 의류 이미지
            pipeline_results: 파이프라인 중간 결과 (선택적)
            
        Returns:
            종합 품질 평가 결과
        """
        if not self.is_initialized:
            raise RuntimeError("품질 평가 시스템이 초기화되지 않았습니다.")
        
        start_time = time.time()
        
        try:
            # 1. 입력 이미지 전처리
            fitted_np = await self._prepare_image(fitted_image)
            person_np = await self._prepare_image(original_person)
            clothing_np = await self._prepare_image(original_clothing)
            
            logger.info("📊 종합 품질 평가 시작...")
            
            # 2. 지각적 품질 평가 (SSIM, PSNR 등)
            logger.info("👁️ 지각적 품질 평가 중...")
            perceptual_score = await self._evaluate_perceptual_quality(fitted_np, person_np)
            
            # 3. 기술적 품질 평가 (선명도, 노이즈 등)
            logger.info("🔧 기술적 품질 평가 중...")
            technical_score = await self._evaluate_technical_quality(fitted_np)
            
            # 4. 미적 품질 평가 (색상 조화, 구성 등)
            logger.info("🎨 미적 품질 평가 중...")
            aesthetic_score = await self._evaluate_aesthetic_quality(fitted_np, person_np, clothing_np)
            
            # 5. 핏 정확도 평가
            logger.info("👕 핏 정확도 평가 중...")
            fit_score = await self._evaluate_fit_accuracy(fitted_np, person_np, pipeline_results)
            
            # 6. 색상 조화 평가
            logger.info("🌈 색상 조화 평가 중...")
            color_score = await self._evaluate_color_harmony(fitted_np, clothing_np)
            
            # 7. 디테일 보존도 평가
            logger.info("🔍 디테일 보존도 평가 중...")
            detail_score = await self._evaluate_detail_preservation(fitted_np, person_np)
            
            # 8. 엣지 품질 평가
            logger.info("📐 엣지 품질 평가 중...")
            edge_score = await self._evaluate_edge_quality(fitted_np)
            
            # 9. 조명 일관성 평가
            logger.info("💡 조명 일관성 평가 중...")
            lighting_score = await self._evaluate_lighting_consistency(fitted_np, person_np)
            
            # 10. 아티팩트 검출
            logger.info("🔎 아티팩트 검출 중...")
            artifact_score = await self._evaluate_artifacts(fitted_np)
            
            # 11. 얼굴 보존도 평가 (선택적)
            face_score = 1.0
            if self.face_detector is not None:
                logger.info("👤 얼굴 보존도 평가 중...")
                face_score = await self._evaluate_face_preservation(fitted_np, person_np)
            
            # 12. 종합 품질 메트릭 계산
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
            
            # 가중 평균으로 전체 점수 계산
            overall_score = (
                metrics.perceptual_quality * self.metric_weights['perceptual_quality'] +
                metrics.technical_quality * self.metric_weights['technical_quality'] +
                metrics.aesthetic_quality * self.metric_weights['aesthetic_quality'] +
                metrics.fit_accuracy * self.metric_weights['fit_accuracy'] +
                metrics.color_harmony * self.metric_weights['color_harmony'] +
                metrics.detail_preservation * self.metric_weights['detail_preservation'] +
                metrics.face_preservation * self.metric_weights['face_preservation']
            )
            
            metrics.overall_score = overall_score
            
            # 13. 개선 제안 생성
            recommendations = await self._generate_recommendations(metrics)
            
            # 14. 상세 분석 (선택적)
            detailed_analysis = {}
            if self.enable_detailed_analysis:
                detailed_analysis = await self._generate_detailed_analysis(
                    metrics, fitted_np, person_np, clothing_np
                )
            
            processing_time = time.time() - start_time
            
            # 15. 최종 결과 구성
            result = {
                'success': True,
                'overall_score': float(overall_score),
                'grade': metrics.get_grade().value,
                'letter_grade': self._get_letter_grade(overall_score),
                'metrics': metrics.to_dict(),
                'recommendations': recommendations,
                'detailed_analysis': detailed_analysis,
                'processing_time': processing_time,
                'timestamp': time.time(),
                'config_used': {
                    'device': self.device,
                    'advanced_metrics': self.enable_advanced_metrics,
                    'face_detection': self.enable_face_detection,
                    'detailed_analysis': self.enable_detailed_analysis
                }
            }
            
            logger.info(f"✅ 품질 평가 완료 - 점수: {overall_score:.3f} ({metrics.get_grade().value})")
            logger.info(f"⏱️ 처리 시간: {processing_time:.2f}초")
            
            return result
            
        except Exception as e:
            error_msg = f"품질 평가 처리 실패: {str(e)}"
            logger.error(error_msg)
            
            return {
                'success': False,
                'error': error_msg,
                'processing_time': time.time() - start_time
            }
    
    async def _prepare_image(self, image: Union[np.ndarray, Image.Image, str]) -> np.ndarray:
        """이미지 전처리"""
        try:
            if isinstance(image, str):
                # Base64 디코딩
                if image.startswith('data:image'):
                    header, data = image.split(',', 1)
                    image_data = base64.b64decode(data)
                else:
                    image_data = base64.b64decode(image)
                
                # PIL 이미지로 변환
                pil_image = Image.open(io.BytesIO(image_data))
                image_np = np.array(pil_image)
                
            elif isinstance(image, Image.Image):
                image_np = np.array(image)
                
            elif isinstance(image, np.ndarray):
                image_np = image.copy()
                
            else:
                raise ValueError(f"지원하지 않는 이미지 타입: {type(image)}")
            
            # RGB 변환 (필요한 경우)
            if len(image_np.shape) == 3 and image_np.shape[2] == 4:
                # RGBA -> RGB
                image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)
            elif len(image_np.shape) == 3 and image_np.shape[2] == 3:
                # BGR -> RGB (OpenCV 기본)
                if image_np.max() <= 1.0:  # 정규화된 이미지
                    image_np = (image_np * 255).astype(np.uint8)
            
            # 크기 정규화 (선택적)
            if image_np.shape[:2] != (512, 512):
                image_np = cv2.resize(image_np, (512, 512))
            
            return image_np
            
        except Exception as e:
            logger.error(f"이미지 전처리 실패: {e}")
            # 기본 더미 이미지 반환
            return np.zeros((512, 512, 3), dtype=np.uint8)
    
    async def _evaluate_perceptual_quality(self, fitted: np.ndarray, original: np.ndarray) -> float:
        """지각적 품질 평가 (SSIM 기반)"""
        try:
            # 그레이스케일 변환
            fitted_gray = cv2.cvtColor(fitted, cv2.COLOR_RGB2GRAY)
            original_gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
            
            # SSIM 계산 (scikit-image 없이 직접 구현)
            ssim_score = self._calculate_ssim(fitted_gray, original_gray)
            
            # PSNR 계산
            mse = np.mean((fitted_gray.astype(float) - original_gray.astype(float)) ** 2)
            if mse == 0:
                psnr_score = 1.0
            else:
                psnr_score = min(20 * np.log10(255.0 / np.sqrt(mse)) / 40.0, 1.0)
            
            # 조합 점수
            perceptual_score = 0.7 * ssim_score + 0.3 * psnr_score
            
            return max(0.0, min(1.0, perceptual_score))
            
        except Exception as e:
            logger.warning(f"지각적 품질 평가 실패: {e}")
            return 0.5
    
    def _calculate_ssim(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """SSIM 직접 계산"""
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
    
    async def _evaluate_technical_quality(self, image: np.ndarray) -> float:
        """기술적 품질 평가"""
        try:
            # 선명도 (Laplacian 분산)
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            sharpness_score = min(laplacian_var / 1000.0, 1.0)
            
            # 노이즈 수준 (이미지 표준편차 기반)
            noise_level = np.std(gray)
            noise_score = max(0, 1.0 - (noise_level - 20) / 100.0)
            
            # 대비 품질
            contrast_score = np.std(gray) / 128.0
            contrast_score = max(0.0, min(1.0, contrast_score))
            
            # 종합 기술적 품질
            technical_score = (sharpness_score * 0.4 + 
                             noise_score * 0.3 + 
                             contrast_score * 0.3)
            
            return max(0.0, min(1.0, technical_score))
            
        except Exception as e:
            logger.warning(f"기술적 품질 평가 실패: {e}")
            return 0.5
    
    async def _evaluate_aesthetic_quality(self, fitted: np.ndarray, person: np.ndarray, clothing: np.ndarray) -> float:
        """미적 품질 평가"""
        try:
            # 색상 분포 평가
            color_score = self._evaluate_color_distribution(fitted)
            
            # 구성 균형 평가
            composition_score = self._evaluate_composition(fitted)
            
            # 전체적인 조화 평가
            harmony_score = self._evaluate_visual_harmony(fitted, person, clothing)
            
            # 종합 미적 점수
            aesthetic_score = (color_score * 0.4 + 
                             composition_score * 0.3 + 
                             harmony_score * 0.3)
            
            return max(0.0, min(1.0, aesthetic_score))
            
        except Exception as e:
            logger.warning(f"미적 품질 평가 실패: {e}")
            return 0.5
    
    def _evaluate_color_distribution(self, image: np.ndarray) -> float:
        """색상 분포 평가"""
        try:
            # HSV로 변환
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            
            # 색상 히스토그램
            hist_h = cv2.calcHist([hsv], [0], None, [180], [0, 180])
            hist_s = cv2.calcHist([hsv], [1], None, [256], [0, 256])
            
            # 색상 다양성 계산
            h_diversity = np.count_nonzero(hist_h) / 180.0
            s_diversity = np.count_nonzero(hist_s) / 256.0
            
            # 균형 점수
            diversity_score = (h_diversity + s_diversity) / 2.0
            
            return max(0.0, min(1.0, diversity_score))
            
        except Exception as e:
            logger.warning(f"색상 분포 평가 실패: {e}")
            return 0.5
    
    def _evaluate_composition(self, image: np.ndarray) -> float:
        """구성 균형 평가"""
        try:
            # 황금비율 기반 구성 평가
            h, w = image.shape[:2]
            
            # 3분할법 격자점
            grid_points = [
                (w//3, h//3), (2*w//3, h//3),
                (w//3, 2*h//3), (2*w//3, 2*h//3)
            ]
            
            # 각 격자점 주변의 관심도 계산
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            interest_scores = []
            
            for x, y in grid_points:
                roi = gray[max(0, y-50):min(h, y+50), max(0, x-50):min(w, x+50)]
                if roi.size > 0:
                    interest = np.std(roi)  # 표준편차를 관심도로 사용
                    interest_scores.append(interest)
            
            # 균형 점수 (분산이 낮을수록 균형적)
            if interest_scores:
                balance_score = 1.0 - (np.std(interest_scores) / (np.mean(interest_scores) + 1e-6))
                return max(0.0, min(1.0, balance_score))
            
            return 0.5
            
        except Exception as e:
            logger.warning(f"구성 평가 실패: {e}")
            return 0.5
    
    def _evaluate_visual_harmony(self, fitted: np.ndarray, person: np.ndarray, clothing: np.ndarray) -> float:
        """시각적 조화 평가"""
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
            logger.warning(f"시각적 조화 평가 실패: {e}")
            return 0.5
    
    async def _evaluate_fit_accuracy(self, fitted: np.ndarray, person: np.ndarray, pipeline_results: Optional[Dict]) -> float:
        """핏 정확도 평가"""
        try:
            # 가상 피팅 결과가 원본 사람과 얼마나 잘 맞는지 평가
            
            # 1. 사이즈 일관성 검사
            size_score = self._evaluate_size_consistency(fitted, person)
            
            # 2. 변형 품질 검사
            deformation_score = self._evaluate_deformation_quality(fitted, person)
            
            # 3. 경계선 품질 검사
            boundary_score = self._evaluate_boundary_quality(fitted)
            
            # 4. 파이프라인 결과 활용 (있는 경우)
            pipeline_score = 0.8  # 기본값
            if pipeline_results:
                pipeline_score = self._extract_pipeline_confidence(pipeline_results)
            
            # 종합 핏 정확도
            fit_score = (size_score * 0.3 + 
                        deformation_score * 0.3 + 
                        boundary_score * 0.2 + 
                        pipeline_score * 0.2)
            
            return max(0.0, min(1.0, fit_score))
            
        except Exception as e:
            logger.warning(f"핏 정확도 평가 실패: {e}")
            return 0.5
    
    def _evaluate_size_consistency(self, fitted: np.ndarray, person: np.ndarray) -> float:
        """사이즈 일관성 평가"""
        try:
            # 신체 윤곽 검출
            fitted_edges = cv2.Canny(cv2.cvtColor(fitted, cv2.COLOR_RGB2GRAY), 50, 150)
            person_edges = cv2.Canny(cv2.cvtColor(person, cv2.COLOR_RGB2GRAY), 50, 150)
            
            # 윤곽 영역 비교
            fitted_area = np.sum(fitted_edges > 0)
            person_area = np.sum(person_edges > 0)
            
            if person_area > 0:
                size_ratio = min(fitted_area, person_area) / max(fitted_area, person_area)
                return size_ratio
            
            return 0.5
            
        except Exception as e:
            logger.warning(f"사이즈 일관성 평가 실패: {e}")
            return 0.5
    
    def _evaluate_deformation_quality(self, fitted: np.ndarray, person: np.ndarray) -> float:
        """변형 품질 평가"""
        try:
            # 구조적 유사성 검사
            fitted_gray = cv2.cvtColor(fitted, cv2.COLOR_RGB2GRAY)
            person_gray = cv2.cvtColor(person, cv2.COLOR_RGB2GRAY)
            
            # 특징점 기반 유사성
            structural_similarity = self._calculate_ssim(fitted_gray, person_gray)
            
            return structural_similarity
            
        except Exception as e:
            logger.warning(f"변형 품질 평가 실패: {e}")
            return 0.5
    
    def _evaluate_boundary_quality(self, image: np.ndarray) -> float:
        """경계선 품질 평가"""
        try:
            # 엣지 검출
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            
            # 경계선 연속성 평가
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # 가장 큰 윤곽의 연속성 검사
                largest_contour = max(contours, key=cv2.contourArea)
                perimeter = cv2.arcLength(largest_contour, True)
                area = cv2.contourArea(largest_contour)
                
                if area > 0:
                    compactness = (perimeter ** 2) / (4 * np.pi * area)
                    boundary_score = 1.0 / (1.0 + compactness)
                    return max(0.0, min(1.0, boundary_score))
            
            return 0.5
            
        except Exception as e:
            logger.warning(f"경계선 품질 평가 실패: {e}")
            return 0.5
    
    def _extract_pipeline_confidence(self, pipeline_results: Dict) -> float:
        """파이프라인 결과에서 신뢰도 추출"""
        try:
            confidences = []
            
            # 각 단계의 신뢰도 수집
            for step_name, step_result in pipeline_results.items():
                if isinstance(step_result, dict):
                    confidence = step_result.get('confidence', 0.5)
                    confidences.append(confidence)
            
            if confidences:
                # 가중 평균 계산
                return sum(confidences) / len(confidences)
            
            return 0.5
            
        except Exception as e:
            logger.warning(f"파이프라인 신뢰도 추출 실패: {e}")
            return 0.5
    
    async def _evaluate_color_harmony(self, fitted: np.ndarray, clothing: np.ndarray) -> float:
        """색상 조화 평가"""
        try:
            # HSV 색공간에서 분석
            fitted_hsv = cv2.cvtColor(fitted, cv2.COLOR_RGB2HSV)
            clothing_hsv = cv2.cvtColor(clothing, cv2.COLOR_RGB2HSV)
            
            # 주요 색상 추출
            fitted_colors = self._extract_dominant_colors(fitted_hsv)
            clothing_colors = self._extract_dominant_colors(clothing_hsv)
            
            # 색상 조화 계산
            harmony_score = self._calculate_color_harmony(fitted_colors, clothing_colors)
            
            return max(0.0, min(1.0, harmony_score))
            
        except Exception as e:
            logger.warning(f"색상 조화 평가 실패: {e}")
            return 0.5
    
    def _extract_dominant_colors(self, hsv_image: np.ndarray, k: int = 3) -> List[np.ndarray]:
        """주요 색상 추출"""
        try:
            # 이미지를 1D로 변환
            data = hsv_image.reshape(-1, 3)
            data = data[np.random.choice(data.shape[0], min(1000, data.shape[0]), replace=False)]
            
            # K-means 클러스터링 (sklearn 없이 간단 구현)
            centers = []
            for _ in range(k):
                idx = np.random.randint(0, len(data))
                centers.append(data[idx])
            
            centers = np.array(centers)
            
            # 간단한 K-means 반복
            for _ in range(10):
                # 클러스터 할당
                distances = np.sqrt(((data - centers[:, np.newaxis])**2).sum(axis=2))
                closest_cluster = np.argmin(distances, axis=0)
                
                # 중심점 업데이트
                for i in range(k):
                    if np.any(closest_cluster == i):
                        centers[i] = data[closest_cluster == i].mean(axis=0)
            
            return centers.tolist()
            
        except Exception as e:
            logger.warning(f"주요 색상 추출 실패: {e}")
            return [[128, 128, 128]]  # 기본 회색
    
    def _calculate_color_harmony(self, colors1: List, colors2: List) -> float:
        """색상 조화 계산"""
        try:
            min_distance = float('inf')
            
            for c1 in colors1:
                for c2 in colors2:
                    # HSV 공간에서 색상 거리
                    h_dist = min(abs(c1[0] - c2[0]), 180 - abs(c1[0] - c2[0]))
                    s_dist = abs(c1[1] - c2[1])
                    v_dist = abs(c1[2] - c2[2])
                    
                    distance = np.sqrt(h_dist**2 + s_dist**2 + v_dist**2)
                    min_distance = min(min_distance, distance)
            
            # 거리를 0-1 점수로 변환
            harmony_score = 1.0 - (min_distance / 300.0)  # 정규화
            
            return max(0.0, min(1.0, harmony_score))
            
        except Exception as e:
            logger.warning(f"색상 조화 계산 실패: {e}")
            return 0.5
    
    async def _evaluate_detail_preservation(self, fitted: np.ndarray, original: np.ndarray) -> float:
        """디테일 보존도 평가"""
        try:
            # 고주파 성분 비교
            fitted_gray = cv2.cvtColor(fitted, cv2.COLOR_RGB2GRAY)
            original_gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
            
            # 라플라시안 필터로 디테일 추출
            fitted_details = cv2.Laplacian(fitted_gray, cv2.CV_64F)
            original_details = cv2.Laplacian(original_gray, cv2.CV_64F)
            
            # 디테일 유사성 계산
            correlation = self._calculate_correlation(fitted_details, original_details)
            
            return max(0.0, min(1.0, correlation))
            
        except Exception as e:
            logger.warning(f"디테일 보존도 평가 실패: {e}")
            return 0.5
    
    def _calculate_correlation(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """이미지 간 상관관계 계산"""
        try:
            # 정규화
            img1_norm = (img1 - np.mean(img1)) / (np.std(img1) + 1e-8)
            img2_norm = (img2 - np.mean(img2)) / (np.std(img2) + 1e-8)
            
            # 상관계수 계산
            correlation = np.mean(img1_norm * img2_norm)
            
            return (correlation + 1.0) / 2.0  # -1~1을 0~1로 변환
            
        except Exception as e:
            logger.warning(f"상관관계 계산 실패: {e}")
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
            density_score = min(edge_density * 10, 1.0)  # 적절한 엣지 밀도
            strength_score = min(edge_strength / 100, 1.0)  # 적절한 엣지 강도
            
            edge_score = (density_score + strength_score) / 2.0
            
            return max(0.0, min(1.0, edge_score))
            
        except Exception as e:
            logger.warning(f"엣지 품질 평가 실패: {e}")
            return 0.5
    
    async def _evaluate_lighting_consistency(self, fitted: np.ndarray, original: np.ndarray) -> float:
        """조명 일관성 평가"""
        try:
            # RGB를 LAB 색공간으로 변환 (더 정확한 밝기 분석)
            fitted_lab = cv2.cvtColor(fitted, cv2.COLOR_RGB2LAB)
            original_lab = cv2.cvtColor(original, cv2.COLOR_RGB2LAB)
            
            # L 채널 (밝기) 비교
            fitted_l = fitted_lab[:, :, 0]
            original_l = original_lab[:, :, 0]
            
            # 밝기 분포 유사성
            fitted_hist = cv2.calcHist([fitted_l], [0], None, [256], [0, 256])
            original_hist = cv2.calcHist([original_l], [0], None, [256], [0, 256])
            
            # 히스토그램 정규화
            fitted_hist = fitted_hist / np.sum(fitted_hist)
            original_hist = original_hist / np.sum(original_hist)
            
            # 히스토그램 유사성 (Bhattacharyya distance)
            similarity = cv2.compareHist(fitted_hist, original_hist, cv2.HISTCMP_BHATTACHARYYA)
            lighting_score = 1.0 - similarity
            
            return max(0.0, min(1.0, lighting_score))
            
        except Exception as e:
            logger.warning(f"조명 일관성 평가 실패: {e}")
            return 0.5
    
    async def _evaluate_artifacts(self, image: np.ndarray) -> float:
        """아티팩트 검출"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # 1. 블로킹 아티팩트 검출
            blocking_score = self._detect_blocking_artifacts(gray)
            
            # 2. 링잉 아티팩트 검출
            ringing_score = self._detect_ringing_artifacts(gray)
            
            # 3. 블러 아티팩트 검출
            blur_score = self._detect_blur_artifacts(gray)
            
            # 종합 아티팩트 점수 (낮을수록 좋음, 1에서 뺌)
            artifact_level = 1.0 - (blocking_score + ringing_score + blur_score) / 3.0
            
            return max(0.0, min(1.0, artifact_level))
            
        except Exception as e:
            logger.warning(f"아티팩트 검출 실패: {e}")
            return 0.5
    
    def _detect_blocking_artifacts(self, gray: np.ndarray) -> float:
        """블로킹 아티팩트 검출"""
        try:
            # 8x8 블록 경계에서의 불연속성 검사
            h, w = gray.shape
            blocking_measure = 0.0
            count = 0
            
            # 수직 경계
            for i in range(8, h, 8):
                if i < h - 1:
                    diff = np.abs(gray[i, :].astype(float) - gray[i-1, :].astype(float))
                    blocking_measure += np.mean(diff)
                    count += 1
            
            # 수평 경계
            for j in range(8, w, 8):
                if j < w - 1:
                    diff = np.abs(gray[:, j].astype(float) - gray[:, j-1].astype(float))
                    blocking_measure += np.mean(diff)
                    count += 1
            
            if count > 0:
                blocking_measure /= count
                return min(blocking_measure / 50.0, 1.0)  # 정규화
            
            return 0.0
            
        except Exception as e:
            logger.warning(f"블로킹 아티팩트 검출 실패: {e}")
            return 0.0
    
    def _detect_ringing_artifacts(self, gray: np.ndarray) -> float:
        """링잉 아티팩트 검출"""
        try:
            # 라플라시안으로 엣지 주변의 진동 패턴 검출
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            
            # 링잉은 강한 엣지 주변에서 발생
            edges = cv2.Canny(gray, 50, 150)
            
            # 엣지 주변 영역에서 라플라시안 분산 측정
            kernel = np.ones((5, 5), np.uint8)
            edge_regions = cv2.dilate(edges, kernel, iterations=1)
            
            ringing_measure = np.std(laplacian[edge_regions > 0])
            
            return min(ringing_measure / 100.0, 1.0)  # 정규화
            
        except Exception as e:
            logger.warning(f"링잉 아티팩트 검출 실패: {e}")
            return 0.0
    
    def _detect_blur_artifacts(self, gray: np.ndarray) -> float:
        """블러 아티팩트 검출"""
        try:
            # 라플라시안 분산으로 블러 정도 측정
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # 분산이 낮으면 블러가 많음
            blur_measure = 1.0 - min(laplacian_var / 1000.0, 1.0)
            
            return max(0.0, min(1.0, blur_measure))
            
        except Exception as e:
            logger.warning(f"블러 아티팩트 검출 실패: {e}")
            return 0.0
    
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
                # 원본에 얼굴이 없으면 평가 불가
                return 1.0
            
            if len(fitted_faces) == 0:
                # 피팅 결과에 얼굴이 없으면 보존 실패
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
                
                # 얼굴 유사성 계산
                face_similarity = self._calculate_face_similarity(orig_face_region, fitted_face_resized)
                
                return max(0.0, min(1.0, face_similarity))
            
            return 0.5
            
        except Exception as e:
            logger.warning(f"얼굴 보존도 평가 실패: {e}")
            return 1.0
    
    def _calculate_face_similarity(self, face1: np.ndarray, face2: np.ndarray) -> float:
        """얼굴 유사성 계산"""
        try:
            # 구조적 유사성 사용
            face1_gray = cv2.cvtColor(face1, cv2.COLOR_RGB2GRAY) if len(face1.shape) == 3 else face1
            face2_gray = cv2.cvtColor(face2, cv2.COLOR_RGB2GRAY) if len(face2.shape) == 3 else face2
            
            similarity = self._calculate_ssim(face1_gray, face2_gray)
            
            return similarity
            
        except Exception as e:
            logger.warning(f"얼굴 유사성 계산 실패: {e}")
            return 0.5
    
    async def _generate_recommendations(self, metrics: QualityMetrics) -> List[str]:
        """개선 제안 생성"""
        recommendations = []
        
        try:
            # 임계값 기반 제안
            thresholds = {
                'perceptual_quality': 0.7,
                'technical_quality': 0.6,
                'aesthetic_quality': 0.65,
                'fit_accuracy': 0.75,
                'color_harmony': 0.6,
                'detail_preservation': 0.7,
                'edge_quality': 0.6,
                'lighting_consistency': 0.65,
                'artifact_level': 0.8,
                'face_preservation': 0.8
            }
            
            if metrics.perceptual_quality < thresholds['perceptual_quality']:
                recommendations.append("지각적 품질 개선: 입력 이미지 해상도를 높이거나 노이즈를 줄여보세요.")
            
            if metrics.technical_quality < thresholds['technical_quality']:
                recommendations.append("기술적 품질 개선: 이미지 선명도를 높이고 압축 아티팩트를 줄여보세요.")
            
            if metrics.aesthetic_quality < thresholds['aesthetic_quality']:
                recommendations.append("미적 품질 개선: 색상 균형과 구성을 조정해보세요.")
            
            if metrics.fit_accuracy < thresholds['fit_accuracy']:
                recommendations.append("핏 정확도 개선: 신체 측정값을 다시 확인하거나 포즈를 조정해보세요.")
            
            if metrics.color_harmony < thresholds['color_harmony']:
                recommendations.append("색상 조화 개선: 의류와 피부톤이 잘 어울리는 색상을 선택해보세요.")
            
            if metrics.detail_preservation < thresholds['detail_preservation']:
                recommendations.append("디테일 보존 개선: 더 높은 품질의 원본 이미지를 사용해보세요.")
            
            if metrics.edge_quality < thresholds['edge_quality']:
                recommendations.append("엣지 품질 개선: 배경이 깔끔한 이미지를 사용해보세요.")
            
            if metrics.lighting_consistency < thresholds['lighting_consistency']:
                recommendations.append("조명 일관성 개선: 균일한 조명 환경에서 촬영된 이미지를 사용해보세요.")
            
            if metrics.artifact_level < thresholds['artifact_level']:
                recommendations.append("아티팩트 감소: 처리 품질 설정을 높이거나 다른 알고리즘을 시도해보세요.")
            
            if metrics.face_preservation < thresholds['face_preservation']:
                recommendations.append("얼굴 보존 개선: 얼굴이 명확히 보이는 정면 사진을 사용해보세요.")
            
            # 전체 점수 기반 제안
            if metrics.overall_score < 0.5:
                recommendations.insert(0, "전체적인 품질이 낮습니다. 더 높은 해상도의 이미지와 적절한 조명을 사용해보세요.")
            elif metrics.overall_score < 0.7:
                recommendations.insert(0, "좋은 결과입니다. 몇 가지 세부사항을 개선하면 더 나은 결과를 얻을 수 있습니다.")
            
            # 빈 추천 목록인 경우 기본 제안
            if not recommendations:
                recommendations.append("훌륭한 품질입니다! 현재 설정을 유지하세요.")
            
            return recommendations
            
        except Exception as e:
            logger.warning(f"개선 제안 생성 실패: {e}")
            return ["품질 분석은 완료되었지만 개선 제안 생성에 실패했습니다."]
    
    async def _generate_detailed_analysis(
        self, 
        metrics: QualityMetrics, 
        fitted: np.ndarray, 
        person: np.ndarray, 
        clothing: np.ndarray
    ) -> Dict[str, Any]:
        """상세 분석 생성"""
        try:
            analysis = {
                'quality_breakdown': {
                    'excellent_aspects': [],
                    'good_aspects': [],
                    'improvement_needed': []
                },
                'technical_details': {
                    'image_properties': self._analyze_image_properties(fitted),
                    'color_analysis': self._analyze_colors(fitted, clothing),
                    'structural_analysis': self._analyze_structure(fitted, person)
                },
                'comparison_metrics': {
                    'similarity_to_original': self._calculate_overall_similarity(fitted, person),
                    'clothing_integration': self._calculate_clothing_integration(fitted, clothing),
                    'realism_score': self._calculate_realism_score(fitted)
                },
                'performance_insights': {
                    'strongest_aspect': self._find_strongest_aspect(metrics),
                    'weakest_aspect': self._find_weakest_aspect(metrics),
                    'improvement_potential': self._calculate_improvement_potential(metrics)
                }
            }
            
            # 품질 분류
            for metric_name, metric_value in metrics.to_dict().items():
                if metric_name == 'overall_score':
                    continue
                
                if metric_value >= 0.8:
                    analysis['quality_breakdown']['excellent_aspects'].append({
                        'aspect': metric_name.replace('_', ' ').title(),
                        'score': metric_value,
                        'status': 'excellent'
                    })
                elif metric_value >= 0.6:
                    analysis['quality_breakdown']['good_aspects'].append({
                        'aspect': metric_name.replace('_', ' ').title(),
                        'score': metric_value,
                        'status': 'good'
                    })
                else:
                    analysis['quality_breakdown']['improvement_needed'].append({
                        'aspect': metric_name.replace('_', ' ').title(),
                        'score': metric_value,
                        'status': 'needs_improvement'
                    })
            
            return analysis
            
        except Exception as e:
            logger.warning(f"상세 분석 생성 실패: {e}")
            return {'error': '상세 분석 생성에 실패했습니다.'}
    
    def _analyze_image_properties(self, image: np.ndarray) -> Dict[str, Any]:
        """이미지 속성 분석"""
        try:
            h, w, c = image.shape
            
            # 기본 속성
            properties = {
                'resolution': f"{w}x{h}",
                'channels': c,
                'file_size_estimate': f"{(w * h * c / 1024):.1f} KB",
                'aspect_ratio': f"{w/h:.2f}:1"
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
    
    def _analyze_colors(self, fitted: np.ndarray, clothing: np.ndarray) -> Dict[str, Any]:
        """색상 분석"""
        try:
            # 주요 색상 추출
            fitted_hsv = cv2.cvtColor(fitted, cv2.COLOR_RGB2HSV)
            clothing_hsv = cv2.cvtColor(clothing, cv2.COLOR_RGB2HSV)
            
            fitted_colors = self._extract_dominant_colors(fitted_hsv)
            clothing_colors = self._extract_dominant_colors(clothing_hsv)
            
            analysis = {
                'dominant_colors_fitted': fitted_colors,
                'dominant_colors_clothing': clothing_colors,
                'color_temperature': self._estimate_color_temperature(fitted),
                'saturation_level': float(np.mean(fitted_hsv[:,:,1])),
                'brightness_level': float(np.mean(fitted_hsv[:,:,2]))
            }
            
            return analysis
            
        except Exception as e:
            logger.warning(f"색상 분석 실패: {e}")
            return {}
    
    def _estimate_color_temperature(self, image: np.ndarray) -> str:
        """색온도 추정"""
        try:
            # RGB 평균값으로 색온도 추정
            r_mean = np.mean(image[:,:,0])
            g_mean = np.mean(image[:,:,1])
            b_mean = np.mean(image[:,:,2])
            
            # 간단한 색온도 분류
            if b_mean > r_mean * 1.1:
                return "Cool (차가운 톤)"
            elif r_mean > b_mean * 1.1:
                return "Warm (따뜻한 톤)"
            else:
                return "Neutral (중성 톤)"
                
        except Exception as e:
            logger.warning(f"색온도 추정 실패: {e}")
            return "Unknown"
    
    def _analyze_structure(self, fitted: np.ndarray, person: np.ndarray) -> Dict[str, Any]:
        """구조 분석"""
        try:
            # 엣지 분석
            fitted_edges = cv2.Canny(cv2.cvtColor(fitted, cv2.COLOR_RGB2GRAY), 50, 150)
            person_edges = cv2.Canny(cv2.cvtColor(person, cv2.COLOR_RGB2GRAY), 50, 150)
            
            analysis = {
                'edge_density_fitted': float(np.sum(fitted_edges > 0) / fitted_edges.size),
                'edge_density_person': float(np.sum(person_edges > 0) / person_edges.size),
                'structural_similarity': self._calculate_ssim(fitted_edges, person_edges),
                'geometric_distortion': self._calculate_geometric_distortion(fitted, person)
            }
            
            return analysis
            
        except Exception as e:
            logger.warning(f"구조 분석 실패: {e}")
            return {}
    
    def _calculate_geometric_distortion(self, fitted: np.ndarray, person: np.ndarray) -> float:
        """기하학적 왜곡 계산"""
        try:
            # 키포인트 기반 왜곡 측정 (간단 버전)
            fitted_gray = cv2.cvtColor(fitted, cv2.COLOR_RGB2GRAY)
            person_gray = cv2.cvtColor(person, cv2.COLOR_RGB2GRAY)
            
            # 구조적 차이 측정
            diff = cv2.absdiff(fitted_gray, person_gray)
            distortion_level = np.mean(diff) / 255.0
            
            return float(1.0 - distortion_level)  # 1에 가까울수록 왜곡이 적음
            
        except Exception as e:
            logger.warning(f"기하학적 왜곡 계산 실패: {e}")
            return 0.5
    
    def _calculate_overall_similarity(self, fitted: np.ndarray, person: np.ndarray) -> float:
        """전체 유사성 계산"""
        try:
            # 다중 메트릭 조합
            structural_sim = self._calculate_ssim(
                cv2.cvtColor(fitted, cv2.COLOR_RGB2GRAY),
                cv2.cvtColor(person, cv2.COLOR_RGB2GRAY)
            )
            
            # 색상 유사성
            fitted_mean = np.mean(fitted.reshape(-1, 3), axis=0)
            person_mean = np.mean(person.reshape(-1, 3), axis=0)
            color_sim = 1.0 - np.linalg.norm(fitted_mean - person_mean) / 441.67
            
            # 조합
            overall_sim = 0.6 * structural_sim + 0.4 * color_sim
            
            return max(0.0, min(1.0, overall_sim))
            
        except Exception as e:
            logger.warning(f"전체 유사성 계산 실패: {e}")
            return 0.5
    
    def _calculate_clothing_integration(self, fitted: np.ndarray, clothing: np.ndarray) -> float:
        """의류 통합도 계산"""
        try:
            # 의류가 자연스럽게 통합되었는지 평가
            fitted_colors = np.mean(fitted.reshape(-1, 3), axis=0)
            clothing_colors = np.mean(clothing.reshape(-1, 3), axis=0)
            
            # 색상 일관성
            color_consistency = 1.0 - np.linalg.norm(fitted_colors - clothing_colors) / 441.67
            
            # 경계선 품질
            boundary_quality = self._evaluate_boundary_quality(fitted)
            
            # 조합
            integration_score = 0.5 * color_consistency + 0.5 * boundary_quality
            
            return max(0.0, min(1.0, integration_score))
            
        except Exception as e:
            logger.warning(f"의류 통합도 계산 실패: {e}")
            return 0.5
    
    def _calculate_realism_score(self, image: np.ndarray) -> float:
        """현실성 점수 계산"""
        try:
            # 현실적인 이미지인지 평가
            
            # 1. 조명 자연스러움
            lighting_score = self._evaluate_lighting_naturalness(image)
            
            # 2. 텍스처 품질
            texture_score = self._evaluate_texture_quality(image)
            
            # 3. 그림자 일관성
            shadow_score = self._evaluate_shadow_consistency(image)
            
            # 조합
            realism_score = (lighting_score + texture_score + shadow_score) / 3.0
            
            return max(0.0, min(1.0, realism_score))
            
        except Exception as e:
            logger.warning(f"현실성 점수 계산 실패: {e}")
            return 0.5
    
    def _evaluate_lighting_naturalness(self, image: np.ndarray) -> float:
        """조명 자연스러움 평가"""
        try:
            # LAB 색공간에서 L 채널 분석
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            l_channel = lab[:, :, 0]
            
            # 조명 분포 분석
            hist = cv2.calcHist([l_channel], [0], None, [256], [0, 256])
            hist_norm = hist / np.sum(hist)
            
            # 자연스러운 조명은 적절한 분포를 가짐
            entropy = -np.sum(hist_norm * np.log(hist_norm + 1e-8))
            naturalness = min(entropy / 8.0, 1.0)  # 정규화
            
            return naturalness
            
        except Exception as e:
            logger.warning(f"조명 자연스러움 평가 실패: {e}")
            return 0.5
    
    def _evaluate_texture_quality(self, image: np.ndarray) -> float:
        """텍스처 품질 평가"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # LBP (Local Binary Pattern) 유사 계산
            texture_variance = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # 적절한 텍스처 복잡도
            texture_score = min(texture_variance / 1000.0, 1.0)
            
            return texture_score
            
        except Exception as e:
            logger.warning(f"텍스처 품질 평가 실패: {e}")
            return 0.5
    
    def _evaluate_shadow_consistency(self, image: np.ndarray) -> float:
        """그림자 일관성 평가"""
        try:
            # 그림자 영역 검출 (간단 버전)
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            v_channel = hsv[:, :, 2]
            
            # 어두운 영역 (그림자 후보) 검출
            shadow_mask = v_channel < np.percentile(v_channel, 20)
            
            # 그림자 영역의 연속성 평가
            contours, _ = cv2.findContours(
                shadow_mask.astype(np.uint8), 
                cv2.RETR_EXTERNAL, 
                cv2.CHAIN_APPROX_SIMPLE
            )
            
            if contours:
                # 가장 큰 그림자 영역의 형태 분석
                largest_shadow = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest_shadow)
                perimeter = cv2.arcLength(largest_shadow, True)
                
                if perimeter > 0:
                    compactness = (4 * np.pi * area) / (perimeter ** 2)
                    consistency_score = min(compactness * 2, 1.0)
                    return consistency_score
            
            return 0.7  # 기본값
            
        except Exception as e:
            logger.warning(f"그림자 일관성 평가 실패: {e}")
            return 0.5
    
    def _find_strongest_aspect(self, metrics: QualityMetrics) -> Dict[str, Any]:
        """가장 강한 측면 찾기"""
        try:
            metric_dict = metrics.to_dict()
            metric_dict.pop('overall_score', None)  # 전체 점수 제외
            
            if metric_dict:
                best_metric = max(metric_dict.items(), key=lambda x: x[1])
                return {
                    'aspect': best_metric[0].replace('_', ' ').title(),
                    'score': best_metric[1],
                    'description': self._get_aspect_description(best_metric[0])
                }
            
            return {'aspect': 'Unknown', 'score': 0.0, 'description': ''}
            
        except Exception as e:
            logger.warning(f"가장 강한 측면 찾기 실패: {e}")
            return {'aspect': 'Unknown', 'score': 0.0, 'description': ''}
    
    def _find_weakest_aspect(self, metrics: QualityMetrics) -> Dict[str, Any]:
        """가장 약한 측면 찾기"""
        try:
            metric_dict = metrics.to_dict()
            metric_dict.pop('overall_score', None)  # 전체 점수 제외
            
            if metric_dict:
                worst_metric = min(metric_dict.items(), key=lambda x: x[1])
                return {
                    'aspect': worst_metric[0].replace('_', ' ').title(),
                    'score': worst_metric[1],
                    'description': self._get_aspect_description(worst_metric[0]),
                    'improvement_suggestion': self._get_improvement_suggestion(worst_metric[0])
                }
            
            return {'aspect': 'Unknown', 'score': 0.0, 'description': '', 'improvement_suggestion': ''}
            
        except Exception as e:
            logger.warning(f"가장 약한 측면 찾기 실패: {e}")
            return {'aspect': 'Unknown', 'score': 0.0, 'description': '', 'improvement_suggestion': ''}
    
    def _get_aspect_description(self, aspect_name: str) -> str:
        """측면 설명 반환"""
        descriptions = {
            'perceptual_quality': '사람이 인지하는 이미지의 전반적인 품질',
            'technical_quality': '선명도, 노이즈, 대비 등 기술적 측면의 품질',
            'aesthetic_quality': '색상 조화, 구성 등 미적 측면의 품질',
            'fit_accuracy': '의류가 신체에 얼마나 자연스럽게 맞는지',
            'color_harmony': '의류와 신체, 배경 간의 색상 조화',
            'detail_preservation': '원본 이미지의 세부사항이 얼마나 보존되었는지',
            'edge_quality': '객체 경계선의 선명도와 자연스러움',
            'lighting_consistency': '전체 이미지의 조명 일관성',
            'artifact_level': '압축, 블러 등 인공적 결함의 정도',
            'face_preservation': '얼굴 특징이 얼마나 잘 보존되었는지'
        }
        
        return descriptions.get(aspect_name, '해당 측면에 대한 설명이 없습니다.')
    
    def _get_improvement_suggestion(self, aspect_name: str) -> str:
        """개선 제안 반환"""
        suggestions = {
            'perceptual_quality': '더 높은 해상도의 입력 이미지를 사용하고 노이즈를 줄여보세요.',
            'technical_quality': '이미지 압축을 줄이고 선명한 원본을 사용해보세요.',
            'aesthetic_quality': '색상 균형을 조정하고 구성을 개선해보세요.',
            'fit_accuracy': '정확한 신체 측정값을 입력하고 적절한 포즈를 취해보세요.',
            'color_harmony': '피부톤과 잘 어울리는 의류 색상을 선택해보세요.',
            'detail_preservation': '더 높은 품질의 원본 이미지를 사용해보세요.',
            'edge_quality': '배경이 깔끔하고 대비가 좋은 이미지를 사용해보세요.',
            'lighting_consistency': '균일한 조명 환경에서 촬영해보세요.',
            'artifact_level': '처리 품질 설정을 높이거나 다른 알고리즘을 시도해보세요.',
            'face_preservation': '얼굴이 명확히 보이는 정면 사진을 사용해보세요.'
        }
        
        return suggestions.get(aspect_name, '구체적인 개선 방법을 찾을 수 없습니다.')
    
    def _calculate_improvement_potential(self, metrics: QualityMetrics) -> Dict[str, Any]:
        """개선 잠재력 계산"""
        try:
            metric_dict = metrics.to_dict()
            metric_dict.pop('overall_score', None)
            
            if not metric_dict:
                return {}
            
            # 평균 점수와 최저 점수 차이
            scores = list(metric_dict.values())
            avg_score = sum(scores) / len(scores)
            min_score = min(scores)
            max_score = max(scores)
            
            improvement_potential = max_score - min_score
            
            # 개선 우선순위 (낮은 점수 순)
            sorted_metrics = sorted(metric_dict.items(), key=lambda x: x[1])
            priority_areas = [
                {
                    'aspect': metric[0].replace('_', ' ').title(),
                    'current_score': metric[1],
                    'potential_gain': avg_score - metric[1] if metric[1] < avg_score else 0,
                    'priority': 'high' if metric[1] < 0.6 else 'medium' if metric[1] < 0.75 else 'low'
                }
                for metric in sorted_metrics[:3]  # 상위 3개
            ]
            
            return {
                'improvement_potential': improvement_potential,
                'current_average': avg_score,
                'target_score': min(avg_score + improvement_potential * 0.5, 1.0),
                'priority_areas': priority_areas,
                'overall_status': "overall" if improvement_potential < 0.2 else min_score[0].replace("_", " ")
            }
            
        except Exception as e:
            logger.warning(f"개선 잠재력 계산 실패: {e}")
            return {}
    
    def _get_letter_grade(self, score: float) -> str:
        """문자 등급 반환"""
        if score >= 0.97:
            return "A+"
        elif score >= 0.93:
            return "A"
        elif score >= 0.90:
            return "A-"
        elif score >= 0.87:
            return "B+"
        elif score >= 0.83:
            return "B"
        elif score >= 0.80:
            return "B-"
        elif score >= 0.77:
            return "C+"
        elif score >= 0.73:
            return "C"
        elif score >= 0.70:
            return "C-"
        elif score >= 0.65:
            return "D+"
        elif score >= 0.60:
            return "D"
        elif score >= 0.50:
            return "D-"
        else:
            return "F"
    
    def _get_metric_status(self, score: float) -> str:
        """메트릭 상태 반환"""
        if score >= 0.9:
            return "excellent"
        elif score >= 0.75:
            return "good"
        elif score >= 0.6:
            return "fair"
        elif score >= 0.4:
            return "poor"
        else:
            return "very_poor"
    
    async def get_model_info(self) -> Dict[str, Any]:
        """모델 정보 반환"""
        return {
            "step_name": "RealQualityAssessment",
            "version": "1.0",
            "device": self.device,
            "initialized": self.is_initialized,
            "torch_available": TORCH_AVAILABLE,
            "face_detection_enabled": self.face_detector is not None,
            "advanced_metrics_enabled": self.enable_advanced_metrics,
            "detailed_analysis_enabled": self.enable_detailed_analysis,
            "supported_metrics": list(QualityMetrics.__annotations__.keys()),
            "quality_thresholds": self.quality_thresholds,
            "metric_weights": self.metric_weights
        }
    
    async def cleanup(self):
        """리소스 정리"""
        self.face_detector = None
        self.is_initialized = False
        logger.info("🧹 실제 품질 평가 시스템 리소스 정리 완료")


# === 사용 예시 ===
async def test_real_quality_assessment():
    """실제 품질 평가 테스트"""
    
    # 1. 시스템 초기화
    quality_assessor = RealQualityAssessmentStep(
        device='cpu',  # 또는 'mps' (M3 Max)
        config={
            'enable_advanced_metrics': True,
            'enable_face_detection': True,
            'enable_detailed_analysis': True,
            'use_torch': TORCH_AVAILABLE
        }
    )
    
    success = await quality_assessor.initialize()
    if not success:
        print("❌ 품질 평가 시스템 초기화 실패")
        return
    
    # 2. 테스트 이미지 생성
    print("📝 테스트 이미지 생성 중...")
    
    # 더미 이미지 생성
    fitted_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    person_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    clothing_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    
    # 3. 품질 평가 실행
    print("🚀 품질 평가 시작...")
    
    result = await quality_assessor.process(
        fitted_image=fitted_image,
        original_person=person_image,
        original_clothing=clothing_image,
        pipeline_results={
            'step_01': {'confidence': 0.85},
            'step_02': {'confidence': 0.78},
            'step_03': {'confidence': 0.92}
        }
    )
    
    # 4. 결과 출력
    if result['success']:
        print("\n" + "="*50)
        print("📊 품질 평가 결과")
        print("="*50)
        
        print(f"🎯 전체 점수: {result['overall_score']:.3f}")
        print(f"📈 등급: {result['grade'].upper()} ({result['letter_grade']})")
        print(f"⏱️ 처리 시간: {result['processing_time']:.2f}초")
        
        print(f"\n📋 세부 메트릭:")
        for metric, value in result['metrics'].items():
            if metric != 'overall_score':
                status = quality_assessor._get_metric_status(value)
                print(f"  • {metric.replace('_', ' ').title()}: {value:.3f} ({status})")
        
        print(f"\n💡 개선 제안:")
        for i, rec in enumerate(result['recommendations'], 1):
            print(f"  {i}. {rec}")
        
        if result.get('detailed_analysis'):
            analysis = result['detailed_analysis']
            
            print(f"\n🔍 상세 분석:")
            
            if 'performance_insights' in analysis:
                insights = analysis['performance_insights']
                print(f"  • 최고 측면: {insights.get('strongest_aspect', {}).get('aspect', 'N/A')}")
                print(f"  • 개선 필요: {insights.get('weakest_aspect', {}).get('aspect', 'N/A')}")
            
            if 'comparison_metrics' in analysis:
                comp = analysis['comparison_metrics']
                print(f"  • 원본 유사도: {comp.get('similarity_to_original', 0):.3f}")
                print(f"  • 의류 통합도: {comp.get('clothing_integration', 0):.3f}")
                print(f"  • 현실성 점수: {comp.get('realism_score', 0):.3f}")
        
        # 리포트 저장
        with open("quality_report.md", "w", encoding='utf-8') as f:
            f.write(f"# 가상 피팅 품질 평가 리포트\n\n")
            f.write(f"**전체 점수**: {result['overall_score']:.3f} ({result['letter_grade']})\n\n")
            f.write(f"**등급**: {result['grade'].upper()}\n\n")
            f.write(f"**처리 시간**: {result['processing_time']:.2f}초\n\n")
            
            f.write(f"## 세부 메트릭\n\n")
            for metric, value in result['metrics'].items():
                if metric != 'overall_score':
                    f.write(f"- **{metric.replace('_', ' ').title()}**: {value:.3f}\n")
            
            f.write(f"\n## 개선 제안\n\n")
            for i, suggestion in enumerate(result['recommendations'], 1):
                priority = "HIGH" if i <= 2 else "MEDIUM" if i <= 4 else "LOW"
                f.write(f"**[{priority}]** {suggestion['issue']}\n")
                f.write(f"   → {suggestion['suggestion']}\n\n")
            
            # 리포트 저장
            with open("quality_report.md", "w", encoding='utf-8') as f:
                f.write(report)
            
            print("📋 상세 리포트 저장: quality_report.md")
            
    else:
        print(f"❌ 평가 실패: {result['error']}")

    await quality_assessor.cleanup()


# === FastAPI 통합 예시 ===
class QualityAssessmentAPI:
    """품질 평가 API 래퍼"""
    
    def __init__(self):
        self.assessor = None
    
    async def initialize(self):
        """API 초기화"""
        self.assessor = RealQualityAssessmentStep(
            device='cpu',  # 또는 환경에 따라 'mps'
            config={
                'enable_advanced_metrics': True,
                'enable_face_detection': True,
                'enable_detailed_analysis': True
            }
        )
        return await self.assessor.initialize()
    
    async def assess_quality(
        self,
        fitted_image_base64: str,
        person_image_base64: str,
        clothing_image_base64: str,
        pipeline_results: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """품질 평가 API 엔드포인트"""
        
        if not self.assessor or not self.assessor.is_initialized:
            raise RuntimeError("품질 평가 시스템이 초기화되지 않았습니다.")
        
        return await self.assessor.process(
            fitted_image=fitted_image_base64,
            original_person=person_image_base64,
            original_clothing=clothing_image_base64,
            pipeline_results=pipeline_results
        )
    
    async def cleanup(self):
        """API 정리"""
        if self.assessor:
            await self.assessor.cleanup()


# === 별칭 클래스 (기존 코드 호환성) ===
QualityAssessmentStep = RealQualityAssessmentStep


if __name__ == "__main__":
    print("🎯 실제 품질 평가 시스템 테스트")
    print("=" * 50)
    
    # 기본 테스트
    asyncio.run(test_real_quality_assessment())
    
    print("\n" + "=" * 50)
    print("✅ 테스트 완료")