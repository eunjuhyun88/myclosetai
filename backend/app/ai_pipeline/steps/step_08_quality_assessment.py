"""
🎯 완전히 작동하는 8단계: 품질 평가 (Quality Assessment)
실제 메트릭 계산 + 자동 개선 제안 + 상세 분석
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
            color_harmony = await self._evaluate_color_harmony(fitted_np, person_np, clothing_np)
            
            # 7. 디테일 보존도 평가
            logger.info("🔍 디테일 보존도 평가 중...")
            detail_preservation = await self._evaluate_detail_preservation(fitted_np, clothing_np)
            
            # 8. 엣지 품질 평가
            logger.info("📐 엣지 품질 평가 중...")
            edge_quality = await self._evaluate_edge_quality(fitted_np)
            
            # 9. 조명 일관성 평가
            logger.info("💡 조명 일관성 평가 중...")
            lighting_consistency = await self._evaluate_lighting_consistency(fitted_np, person_np)
            
            # 10. 아티팩트 레벨 평가
            logger.info("🔎 아티팩트 검출 중...")
            artifact_level = await self._evaluate_artifacts(fitted_np)
            
            # 11. 얼굴 보존도 평가
            logger.info("😊 얼굴 보존도 평가 중...")
            face_preservation = await self._evaluate_face_preservation(fitted_np, person_np)
            
            # 12. 종합 점수 계산
            logger.info("📈 종합 점수 계산 중...")
            overall_score = self._calculate_overall_score({
                'perceptual_quality': perceptual_score,
                'technical_quality': technical_score,
                'aesthetic_quality': aesthetic_score,
                'fit_accuracy': fit_score,
                'color_harmony': color_harmony,
                'detail_preservation': detail_preservation,
                'face_preservation': face_preservation
            })
            
            # 품질 메트릭 객체 생성
            quality_metrics = QualityMetrics(
                overall_score=overall_score,
                perceptual_quality=perceptual_score,
                technical_quality=technical_score,
                aesthetic_quality=aesthetic_score,
                fit_accuracy=fit_score,
                color_harmony=color_harmony,
                detail_preservation=detail_preservation,
                edge_quality=edge_quality,
                lighting_consistency=lighting_consistency,
                artifact_level=artifact_level,
                face_preservation=face_preservation
            )
            
            # 13. 개선 제안 생성
            logger.info("💡 개선 제안 생성 중...")
            improvement_suggestions = await self._generate_improvement_suggestions(quality_metrics)
            
            # 14. 상세 분석 (선택적)
            detailed_analysis = None
            if self.enable_detailed_analysis:
                logger.info("🔍 상세 분석 수행 중...")
                detailed_analysis = await self._perform_detailed_analysis(
                    fitted_np, person_np, clothing_np, quality_metrics
                )
            
            processing_time = time.time() - start_time
            
            # 최종 결과 구성
            result = {
                "success": True,
                "quality_metrics": quality_metrics.to_dict(),
                "quality_grade": quality_metrics.get_grade().value,
                "letter_grade": self._get_letter_grade(overall_score),
                "score_percentage": round(overall_score * 100, 1),
                
                # 개선 제안
                "improvement_suggestions": improvement_suggestions,
                
                # 상세 분석
                "detailed_analysis": detailed_analysis,
                
                # 개별 메트릭 상세
                "metric_breakdown": {
                    "perceptual": {
                        "score": perceptual_score,
                        "description": "지각적 품질 (SSIM, 시각적 유사도)",
                        "status": self._get_metric_status(perceptual_score)
                    },
                    "technical": {
                        "score": technical_score,
                        "description": "기술적 품질 (선명도, 노이즈)",
                        "status": self._get_metric_status(technical_score)
                    },
                    "aesthetic": {
                        "score": aesthetic_score,
                        "description": "미적 품질 (구성, 시각적 매력)",
                        "status": self._get_metric_status(aesthetic_score)
                    },
                    "fit": {
                        "score": fit_score,
                        "description": "착용감 정확도",
                        "status": self._get_metric_status(fit_score)
                    },
                    "color_harmony": {
                        "score": color_harmony,
                        "description": "색상 조화",
                        "status": self._get_metric_status(color_harmony)
                    },
                    "detail_preservation": {
                        "score": detail_preservation,
                        "description": "디테일 보존도",
                        "status": self._get_metric_status(detail_preservation)
                    }
                },
                
                # 처리 정보
                "assessment_info": {
                    "processing_time": processing_time,
                    "evaluation_method": "comprehensive_multi_metric",
                    "metrics_computed": 11,
                    "device_used": self.device,
                    "face_detection_enabled": self.face_detector is not None,
                    "advanced_metrics_enabled": self.enable_advanced_metrics
                }
            }
            
            logger.info(
                f"✅ 품질 평가 완료 - "
                f"종합점수: {overall_score:.3f} ({quality_metrics.get_grade().value}), "
                f"처리시간: {processing_time:.3f}초"
            )
            
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"❌ 품질 평가 실패: {e}")
            
            return {
                "success": False,
                "error": str(e),
                "processing_time": processing_time
            }
    
    async def _prepare_image(self, image: Union[np.ndarray, Image.Image, str]) -> np.ndarray:
        """이미지 전처리"""
        
        if isinstance(image, str):
            # Base64 문자열 또는 파일 경로
            if image.startswith('data:image') or len(image) > 100:
                # Base64 디코딩
                if 'base64,' in image:
                    image_data = image.split('base64,')[1]
                else:
                    image_data = image
                
                img_bytes = base64.b64decode(image_data)
                img = Image.open(io.BytesIO(img_bytes))
                return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            else:
                # 파일 경로
                return cv2.imread(image)
                
        elif isinstance(image, Image.Image):
            return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
        elif isinstance(image, np.ndarray):
            return image
            
        else:
            raise ValueError(f"지원하지 않는 이미지 타입: {type(image)}")
    
    async def _evaluate_perceptual_quality(self, fitted: np.ndarray, reference: np.ndarray) -> float:
        """지각적 품질 평가 (SSIM 기반)"""
        
        try:
            # 크기 맞추기
            if fitted.shape != reference.shape:
                reference = cv2.resize(reference, (fitted.shape[1], fitted.shape[0]))
            
            # SSIM 계산
            ssim_score = self._calculate_ssim(fitted, reference)
            
            # PSNR 계산
            psnr_score = self._calculate_psnr(fitted, reference)
            psnr_normalized = min(1.0, psnr_score / 40.0)  # 40dB를 1.0으로 정규화
            
            # MSE 기반 유사도
            mse = mean_squared_error(
                fitted.flatten(), 
                reference.flatten()
            )
            mse_score = 1.0 / (1.0 + mse / 1000.0)  # MSE를 0-1 범위로 변환
            
            # 종합 지각적 품질
            perceptual_quality = (ssim_score * 0.5 + psnr_normalized * 0.3 + mse_score * 0.2)
            
            return max(0.0, min(1.0, perceptual_quality))
            
        except Exception as e:
            logger.warning(f"지각적 품질 평가 실패: {e}")
            return 0.7
    
    async def _evaluate_technical_quality(self, image: np.ndarray) -> float:
        """기술적 품질 평가"""
        
        try:
            # 1. 선명도 평가 (라플라시안 분산)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            sharpness_score = min(1.0, laplacian_var / 500.0)
            
            # 2. 노이즈 레벨 평가
            noise_score = self._evaluate_noise_level(image)
            
            # 3. 대비 평가
            contrast_score = self._evaluate_contrast(image)
            
            # 4. 밝기 적절성
            brightness_score = self._evaluate_brightness(image)
            
            # 5. 색상 포화도
            saturation_score = self._evaluate_saturation(image)
            
            # 종합 기술적 품질
            technical_quality = (
                sharpness_score * 0.3 +
                (1.0 - noise_score) * 0.2 +  # 노이즈가 적을수록 좋음
                contrast_score * 0.2 +
                brightness_score * 0.15 +
                saturation_score * 0.15
            )
            
            return max(0.0, min(1.0, technical_quality))
            
        except Exception as e:
            logger.warning(f"기술적 품질 평가 실패: {e}")
            return 0.7
    
    async def _evaluate_aesthetic_quality(
        self, 
        fitted: np.ndarray, 
        person: np.ndarray, 
        clothing: np.ndarray
    ) -> float:
        """미적 품질 평가"""
        
        try:
            # 1. 색상 다양성
            color_diversity = self._evaluate_color_diversity(fitted)
            
            # 2. 구성 품질 (삼등분법)
            composition_score = self._evaluate_composition(fitted)
            
            # 3. 시각적 균형
            visual_balance = self._evaluate_visual_balance(fitted)
            
            # 4. 전체적인 매력도
            appeal_score = self._evaluate_visual_appeal(fitted)
            
            # 종합 미적 품질
            aesthetic_quality = (
                color_diversity * 0.25 +
                composition_score * 0.25 +
                visual_balance * 0.25 +
                appeal_score * 0.25
            )
            
            return max(0.0, min(1.0, aesthetic_quality))
            
        except Exception as e:
            logger.warning(f"미적 품질 평가 실패: {e}")
            return 0.7
    
    async def _evaluate_fit_accuracy(
        self, 
        fitted: np.ndarray, 
        person: np.ndarray, 
        pipeline_results: Optional[Dict[str, Any]]
    ) -> float:
        """핏 정확도 평가"""
        
        try:
            # 파이프라인 결과에서 핏 정보 추출
            if pipeline_results:
                # 포즈 일관성 확인
                pose_consistency = pipeline_results.get('pose_result', {}).get('pose_confidence', 0.8)
                
                # 워핑 품질 확인  
                warping_quality = pipeline_results.get('warping_result', {}).get('quality_metrics', {}).get('overall_quality', 0.8)
                
                # 파이프라인 기반 핏 점수
                pipeline_fit_score = (pose_consistency + warping_quality) / 2
            else:
                pipeline_fit_score = 0.8
            
            # 시각적 핏 분석
            visual_fit_score = self._analyze_visual_fit(fitted, person)
            
            # 종합 핏 점수
            fit_accuracy = (pipeline_fit_score * 0.6 + visual_fit_score * 0.4)
            
            return max(0.0, min(1.0, fit_accuracy))
            
        except Exception as e:
            logger.warning(f"핏 정확도 평가 실패: {e}")
            return 0.7
    
    async def _evaluate_color_harmony(
        self, 
        fitted: np.ndarray, 
        person: np.ndarray, 
        clothing: np.ndarray
    ) -> float:
        """색상 조화 평가"""
        
        try:
            # 1. HSV 색공간에서 분석
            fitted_hsv = cv2.cvtColor(fitted, cv2.COLOR_BGR2HSV)
            person_hsv = cv2.cvtColor(person, cv2.COLOR_BGR2HSV)
            clothing_hsv = cv2.cvtColor(clothing, cv2.COLOR_BGR2HSV)
            
            # 2. 색조(Hue) 조화 분석
            fitted_hue = np.mean(fitted_hsv[:, :, 0])
            person_hue = np.mean(person_hsv[:, :, 0])
            clothing_hue = np.mean(clothing_hsv[:, :, 0])
            
            # 색조 차이 계산 (원형 거리)
            hue_diff_person = min(abs(fitted_hue - person_hue), 180 - abs(fitted_hue - person_hue))
            hue_diff_clothing = min(abs(fitted_hue - clothing_hue), 180 - abs(fitted_hue - clothing_hue))
            
            hue_harmony = 1.0 - (hue_diff_person + hue_diff_clothing) / (2 * 90)  # 90도를 최대 차이로
            hue_harmony = max(0.0, hue_harmony)
            
            # 3. 채도 일관성
            fitted_sat = np.mean(fitted_hsv[:, :, 1])
            person_sat = np.mean(person_hsv[:, :, 1])
            
            sat_consistency = 1.0 - abs(fitted_sat - person_sat) / 255.0
            
            # 4. 명도 일관성
            fitted_val = np.mean(fitted_hsv[:, :, 2])
            person_val = np.mean(person_hsv[:, :, 2])
            
            val_consistency = 1.0 - abs(fitted_val - person_val) / 255.0
            
            # 종합 색상 조화
            color_harmony = (hue_harmony * 0.5 + sat_consistency * 0.3 + val_consistency * 0.2)
            
            return max(0.0, min(1.0, color_harmony))
            
        except Exception as e:
            logger.warning(f"색상 조화 평가 실패: {e}")
            return 0.7
    
    async def _evaluate_detail_preservation(self, fitted: np.ndarray, clothing: np.ndarray) -> float:
        """디테일 보존도 평가"""
        
        try:
            # 크기 맞추기
            if fitted.shape != clothing.shape:
                clothing = cv2.resize(clothing, (fitted.shape[1], fitted.shape[0]))
            
            # 고주파 성분 비교
            fitted_gray = cv2.cvtColor(fitted, cv2.COLOR_BGR2GRAY)
            clothing_gray = cv2.cvtColor(clothing, cv2.COLOR_BGR2GRAY)
            
            # 라플라시안으로 디테일 추출
            fitted_detail = cv2.Laplacian(fitted_gray, cv2.CV_64F)
            clothing_detail = cv2.Laplacian(clothing_gray, cv2.CV_64F)
            
            # 디테일 강도 비교
            fitted_detail_strength = np.std(fitted_detail)
            clothing_detail_strength = np.std(clothing_detail)
            
            if clothing_detail_strength > 0:
                preservation_ratio = fitted_detail_strength / clothing_detail_strength
                preservation_score = 1.0 - abs(preservation_ratio - 1.0)
            else:
                preservation_score = 0.5
            
            return max(0.0, min(1.0, preservation_score))
            
        except Exception as e:
            logger.warning(f"디테일 보존도 평가 실패: {e}")
            return 0.7
    
    async def _evaluate_edge_quality(self, image: np.ndarray) -> float:
        """엣지 품질 평가"""
        
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Canny 엣지 검출
            edges = cv2.Canny(gray, 50, 150)
            
            # 윤곽선 찾기
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return 0.5
            
            # 가장 큰 윤곽선들의 품질 평가
            contour_areas = [cv2.contourArea(c) for c in contours]
            contour_areas.sort(reverse=True)
            
            # 상위 윤곽선들의 평균 크기
            top_contours = contour_areas[:min(5, len(contour_areas))]
            avg_contour_size = np.mean(top_contours) if top_contours else 0
            
            # 이미지 크기 대비 정규화
            image_area = image.shape[0] * image.shape[1]
            normalized_size = avg_contour_size / image_area
            
            # 적절한 윤곽선 크기 범위로 평가
            if 0.01 <= normalized_size <= 0.1:
                edge_quality = 1.0
            else:
                edge_quality = max(0.0, 1.0 - abs(normalized_size - 0.05) * 10)
            
            return edge_quality
            
        except Exception as e:
            logger.warning(f"엣지 품질 평가 실패: {e}")
            return 0.7
    
    async def _evaluate_lighting_consistency(self, fitted: np.ndarray, reference: np.ndarray) -> float:
        """조명 일관성 평가"""
        
        try:
            # 크기 맞추기
            if fitted.shape != reference.shape:
                reference = cv2.resize(reference, (fitted.shape[1], fitted.shape[0]))
            
            # LAB 색공간에서 명도 채널 비교
            fitted_lab = cv2.cvtColor(fitted, cv2.COLOR_BGR2LAB)
            ref_lab = cv2.cvtColor(reference, cv2.COLOR_BGR2LAB)
            
            fitted_l = fitted_lab[:, :, 0]
            ref_l = ref_lab[:, :, 0]
            
            # 명도 분포 비교
            fitted_mean = np.mean(fitted_l)
            ref_mean = np.mean(ref_l)
            
            fitted_std = np.std(fitted_l)
            ref_std = np.std(ref_l)
            
            # 평균 명도 차이
            mean_diff = abs(fitted_mean - ref_mean) / 255.0
            
            # 명도 분산 차이
            std_diff = abs(fitted_std - ref_std) / (ref_std + 1e-6)
            std_diff = min(1.0, std_diff)
            
            # 조명 일관성 점수
            lighting_consistency = 1.0 - (mean_diff + std_diff) / 2.0
            
            return max(0.0, min(1.0, lighting_consistency))
            
        except Exception as e:
            logger.warning(f"조명 일관성 평가 실패: {e}")
            return 0.7
    
    async def _evaluate_artifacts(self, image: np.ndarray) -> float:
        """아티팩트 평가 (낮을수록 좋음)"""
        
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # 1. 블록킹 아티팩트 검출
            # DCT 기반 블록킹 검출
            h, w = gray.shape
            block_size = 8
            
            blocking_score = 0.0
            block_count = 0
            
            for y in range(0, h - block_size, block_size):
                for x in range(0, w - block_size, block_size):
                    block = gray[y:y+block_size, x:x+block_size]
                    
                    # 블록 경계에서의 불연속성 검사
                    if x + block_size < w:
                        right_diff = np.mean(np.abs(block[:, -1] - gray[y:y+block_size, x+block_size]))
                        blocking_score += right_diff
                        block_count += 1
                    
                    if y + block_size < h:
                        bottom_diff = np.mean(np.abs(block[-1, :] - gray[y+block_size, x:x+block_size]))
                        blocking_score += bottom_diff
                        block_count += 1
            
            if block_count > 0:
                blocking_score = blocking_score / block_count / 255.0
            
            # 2. 링잉 아티팩트 검출 (고주파 노이즈)
            sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            
            gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
            ringing_score = np.std(gradient_magnitude) / 255.0
            
            # 종합 아티팩트 레벨
            artifact_level = (blocking_score * 0.6 + ringing_score * 0.4)
            
            return max(0.0, min(1.0, artifact_level))
            
        except Exception as e:
            logger.warning(f"아티팩트 평가 실패: {e}")
            return 0.2
    
    async def _evaluate_face_preservation(self, fitted: np.ndarray, reference: np.ndarray) -> float:
        """얼굴 보존도 평가"""
        
        try:
            if self.face_detector is None:
                return 0.8  # 기본값
            
            # 그레이스케일 변환
            fitted_gray = cv2.cvtColor(fitted, cv2.COLOR_BGR2GRAY)
            ref_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
            
            # 크기 맞추기
            if fitted.shape != reference.shape:
                ref_gray = cv2.resize(ref_gray, (fitted.shape[1], fitted.shape[0]))
                reference = cv2.resize(reference, (fitted.shape[1], fitted.shape[0]))
            
            # 얼굴 검출
            fitted_faces = self.face_detector.detectMultiScale(fitted_gray, 1.1, 4)
            ref_faces = self.face_detector.detectMultiScale(ref_gray, 1.1, 4)
            
            if len(fitted_faces) == 0 or len(ref_faces) == 0:
                return 0.8  # 얼굴이 검출되지 않으면 기본값
            
            # 가장 큰 얼굴 영역 비교
            fitted_face = max(fitted_faces, key=lambda x: x[2] * x[3])
            ref_face = max(ref_faces, key=lambda x: x[2] * x[3])
            
            # 얼굴 영역 추출
            fx, fy, fw, fh = fitted_face
            rx, ry, rw, rh = ref_face
            
            fitted_face_region = fitted[fy:fy+fh, fx:fx+fw]
            ref_face_region = reference[ry:ry+rh, rx:rx+rw]
            
            # 크기 맞추기
            if fitted_face_region.shape != ref_face_region.shape:
                ref_face_region = cv2.resize(ref_face_region, (fw, fh))
            
            # 얼굴 영역 SSIM 계산
            face_ssim = self._calculate_ssim(fitted_face_region, ref_face_region)
            
            return face_ssim
            
        except Exception as e:
            logger.warning(f"얼굴 보존도 평가 실패: {e}")
            return 0.8
    
    def _calculate_overall_score(self, metrics: Dict[str, float]) -> float:
        """종합 점수 계산"""
        
        total_score = 0.0
        total_weight = 0.0
        
        for metric_name, weight in self.metric_weights.items():
            if metric_name in metrics:
                total_score += metrics[metric_name] * weight
                total_weight += weight
        
        if total_weight > 0:
            return total_score / total_weight
        else:
            return 0.7
    
    async def _generate_improvement_suggestions(self, metrics: QualityMetrics) -> List[Dict[str, Any]]:
        """개선 제안 생성"""
        
        suggestions = []
        
        # 지각적 품질 개선
        if metrics.perceptual_quality < 0.7:
            suggestions.append({
                "category": "perceptual",
                "priority": "high",
                "issue": "시각적 품질이 낮습니다",
                "suggestion": "더 고해상도 이미지를 사용하거나 품질 설정을 높여보세요",
                "technical_detail": f"현재 지각적 품질: {metrics.perceptual_quality:.2f}",
                "target_improvement": "0.8 이상"
            })
        
        # 기술적 품질 개선
        if metrics.technical_quality < 0.7:
            suggestions.append({
                "category": "technical",
                "priority": "high",
                "issue": "이미지의 선명도나 노이즈 레벨에 문제가 있습니다",
                "suggestion": "조명이 좋은 환경에서 촬영하거나 노이즈 제거 처리를 적용해보세요",
                "technical_detail": f"현재 기술적 품질: {metrics.technical_quality:.2f}",
                "target_improvement": "0.75 이상"
            })
        
        # 핏 정확도 개선
        if metrics.fit_accuracy < 0.6:
            suggestions.append({
                "category": "fit",
                "priority": "high",
                "issue": "의류 핏이 부정확합니다",
                "suggestion": "전신이 잘 보이는 정면 사진을 사용하고, 정확한 신체 치수를 입력해보세요",
                "technical_detail": f"현재 핏 정확도: {metrics.fit_accuracy:.2f}",
                "target_improvement": "0.7 이상"
            })
        
        # 색상 조화 개선
        if metrics.color_harmony < 0.6:
            suggestions.append({
                "category": "color",
                "priority": "medium",
                "issue": "색상 조화가 부족합니다",
                "suggestion": "자연광에서 촬영된 이미지를 사용하거나 색상 보정을 적용해보세요",
                "technical_detail": f"현재 색상 조화: {metrics.color_harmony:.2f}",
                "target_improvement": "0.7 이상"
            })
        
        # 디테일 보존 개선
        if metrics.detail_preservation < 0.5:
            suggestions.append({
                "category": "detail",
                "priority": "medium",
                "issue": "의류 디테일이 손실되었습니다",
                "suggestion": "더 선명하고 고해상도의 의류 이미지를 사용해보세요",
                "technical_detail": f"현재 디테일 보존도: {metrics.detail_preservation:.2f}",
                "target_improvement": "0.6 이상"
            })
        
        # 얼굴 보존 개선
        if metrics.face_preservation < 0.7:
            suggestions.append({
                "category": "face",
                "priority": "medium",
                "issue": "얼굴 영역이 부자연스럽게 변형되었습니다",
                "suggestion": "얼굴이 정면을 향하고 명확하게 보이는 사진을 사용해보세요",
                "technical_detail": f"현재 얼굴 보존도: {metrics.face_preservation:.2f}",
                "target_improvement": "0.8 이상"
            })
        
        # 아티팩트 개선
        if metrics.artifact_level > 0.3:
            suggestions.append({
                "category": "artifact",
                "priority": "low",
                "issue": "이미지에 인공적인 흔적이 있습니다",
                "suggestion": "품질 설정을 높이거나 처리 시간을 늘려보세요",
                "technical_detail": f"현재 아티팩트 레벨: {metrics.artifact_level:.2f}",
                "target_improvement": "0.2 이하"
            })
        
        # 우선순위별 정렬
        priority_order = {"high": 0, "medium": 1, "low": 2}
        suggestions.sort(key=lambda x: priority_order[x["priority"]])
        
        # 성공적인 경우 격려 메시지
        if not suggestions:
            suggestions.append({
                "category": "success",
                "priority": "info",
                "issue": "우수한 품질입니다!",
                "suggestion": "모든 메트릭이 좋은 수준에 도달했습니다. 추가 개선사항이 없습니다.",
                "technical_detail": f"종합 점수: {metrics.overall_score:.2f}",
                "target_improvement": "현재 수준 유지"
            })
        
        return suggestions
    
    async def _perform_detailed_analysis(
        self,
        fitted: np.ndarray,
        person: np.ndarray,
        clothing: np.ndarray,
        metrics: QualityMetrics
    ) -> Dict[str, Any]:
        """상세 분석 수행"""
        
        analysis = {}
        
        try:
            # 1. 색상 분포 분석
            analysis["color_distribution"] = self._analyze_color_distribution(fitted)
            
            # 2. 히스토그램 분석
            analysis["histogram_analysis"] = self._analyze_histograms(fitted, person, clothing)
            
            # 3. 텍스처 분석
            analysis["texture_analysis"] = self._analyze_texture_statistics(fitted)
            
            # 4. 엣지 분석
            analysis["edge_analysis"] = self._analyze_edge_statistics(fitted)
            
            # 5. 품질 진단
            analysis["quality_diagnosis"] = self._diagnose_quality_issues(metrics)
            
            # 6. 개선 잠재력 분석
            analysis["improvement_potential"] = self._analyze_improvement_potential(metrics)
            
            return analysis
            
        except Exception as e:
            logger.warning(f"상세 분석 실패: {e}")
            return {"error": str(e)}
    
    # === 헬퍼 메소드들 ===
    
    def _calculate_ssim(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """SSIM 계산"""
        try:
            # 크기 맞추기
            if img1.shape != img2.shape:
                img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
            
            # 그레이스케일 변환
            if len(img1.shape) == 3:
                gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
                gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            else:
                gray1 = img1
                gray2 = img2
            
            # SSIM 계산
            C1 = (0.01 * 255) ** 2
            C2 = (0.03 * 255) ** 2
            
            mu1 = cv2.GaussianBlur(gray1, (11, 11), 1.5)
            mu2 = cv2.GaussianBlur(gray2, (11, 11), 1.5)
            
            mu1_sq = mu1 * mu1
            mu2_sq = mu2 * mu2
            mu1_mu2 = mu1 * mu2
            
            sigma1_sq = cv2.GaussianBlur(gray1 * gray1, (11, 11), 1.5) - mu1_sq
            sigma2_sq = cv2.GaussianBlur(gray2 * gray2, (11, 11), 1.5) - mu2_sq
            sigma12 = cv2.GaussianBlur(gray1 * gray2, (11, 11), 1.5) - mu1_mu2
            
            ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
            
            return float(np.mean(ssim_map))
            
        except Exception as e:
            logger.warning(f"SSIM 계산 실패: {e}")
            return 0.8
    
    def _calculate_psnr(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """PSNR 계산"""
        try:
            if img1.shape != img2.shape:
                img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
            
            mse = np.mean((img1.astype(float) - img2.astype(float)) ** 2)
            
            if mse == 0:
                return 40.0  # 완전히 동일한 경우
            
            max_pixel = 255.0
            psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
            
            return float(psnr)
            
        except Exception as e:
            logger.warning(f"PSNR 계산 실패: {e}")
            return 30.0
    
    def _evaluate_noise_level(self, image: np.ndarray) -> float:
        """노이즈 레벨 평가"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 미디안 필터와 원본의 차이로 노이즈 추정
        median_filtered = cv2.medianBlur(gray, 5)
        noise = np.abs(gray.astype(float) - median_filtered.astype(float))
        
        noise_level = np.mean(noise) / 255.0
        return min(1.0, noise_level)
    
    def _evaluate_contrast(self, image: np.ndarray) -> float:
        """대비 평가"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        contrast = np.std(gray) / 255.0
        
        # 적절한 대비 범위 (0.1-0.3)로 평가
        if 0.1 <= contrast <= 0.3:
            return 1.0
        else:
            return max(0.0, 1.0 - abs(contrast - 0.2) * 5)
    
    def _evaluate_brightness(self, image: np.ndarray) -> float:
        """밝기 적절성 평가"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray) / 255.0
        
        # 적절한 밝기 범위 (0.3-0.7)로 평가
        if 0.3 <= brightness <= 0.7:
            return 1.0
        else:
            return max(0.0, 1.0 - abs(brightness - 0.5) * 2)
    
    def _evaluate_saturation(self, image: np.ndarray) -> float:
        """채도 평가"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        saturation = np.mean(hsv[:, :, 1]) / 255.0
        
        # 적절한 채도 범위 (0.2-0.6)로 평가
        if 0.2 <= saturation <= 0.6:
            return 1.0
        else:
            return max(0.0, 1.0 - abs(saturation - 0.4) * 2.5)
    
    def _evaluate_color_diversity(self, image: np.ndarray) -> float:
        """색상 다양성 평가"""
        # 이미지를 RGB로 변환하고 고유 색상 수 계산
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 다운샘플링으로 계산 속도 향상
        small_img = cv2.resize(rgb_image, (50, 50))
        pixels = small_img.reshape(-1, 3)
        
        # 고유 색상 수
        unique_colors = len(np.unique(pixels, axis=0))
        
        # 정규화 (최대 2500개 색상 기준)
        diversity = min(1.0, unique_colors / 1000)
        
        return diversity
    
    def _evaluate_composition(self, image: np.ndarray) -> float:
        """구성 품질 평가 (삼등분법)"""
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 엣지 검출
        edges = cv2.Canny(gray, 50, 150)
        
        # 삼등분선 위치
        third_h1, third_h2 = h // 3, 2 * h // 3
        third_w1, third_w2 = w // 3, 2 * w // 3
        
        # 삼등분선 근처의 엣지 밀도
        edge_density = 0.0
        
        # 수직 삼등분선
        edge_density += np.sum(edges[:, max(0, third_w1-10):min(w, third_w1+10)])
        edge_density += np.sum(edges[:, max(0, third_w2-10):min(w, third_w2+10)])
        
        # 수평 삼등분선
        edge_density += np.sum(edges[max(0, third_h1-10):min(h, third_h1+10), :])
        edge_density += np.sum(edges[max(0, third_h2-10):min(h, third_h2+10), :])
        
        # 정규화
        composition_score = min(1.0, edge_density / (h * w * 0.01))
        
        return composition_score
    
    def _evaluate_visual_balance(self, image: np.ndarray) -> float:
        """시각적 균형 평가"""
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 좌우 균형
        left_half = gray[:, :w//2]
        right_half = gray[:, w//2:]
        
        left_mean = np.mean(left_half)
        right_mean = np.mean(right_half)
        
        horizontal_balance = 1.0 - abs(left_mean - right_mean) / 255.0
        
        # 상하 균형
        top_half = gray[:h//2, :]
        bottom_half = gray[h//2:, :]
        
        top_mean = np.mean(top_half)
        bottom_mean = np.mean(bottom_half)
        
        vertical_balance = 1.0 - abs(top_mean - bottom_mean) / 255.0
        
        # 종합 균형
        visual_balance = (horizontal_balance + vertical_balance) / 2
        
        return visual_balance
    
    def _evaluate_visual_appeal(self, image: np.ndarray) -> float:
        """시각적 매력도 평가"""
        # 여러 요소의 조합
        contrast = self._evaluate_contrast(image)
        brightness = self._evaluate_brightness(image)
        saturation = self._evaluate_saturation(image)
        color_diversity = self._evaluate_color_diversity(image)
        
        # 종합 매력도
        appeal = (contrast * 0.3 + brightness * 0.2 + saturation * 0.3 + color_diversity * 0.2)
        
        return appeal
    
    def _analyze_visual_fit(self, fitted: np.ndarray, person: np.ndarray) -> float:
        """시각적 핏 분석"""
        try:
            # 윤곽선 기반 분석
            fitted_gray = cv2.cvtColor(fitted, cv2.COLOR_BGR2GRAY)
            person_gray = cv2.cvtColor(person, cv2.COLOR_BGR2GRAY)
            
            if fitted.shape != person.shape:
                person_gray = cv2.resize(person_gray, (fitted.shape[1], fitted.shape[0]))
            
            # 엣지 검출
            fitted_edges = cv2.Canny(fitted_gray, 50, 150)
            person_edges = cv2.Canny(person_gray, 50, 150)
            
            # 윤곽선 유사도
            edge_diff = np.abs(fitted_edges.astype(float) - person_edges.astype(float))
            edge_similarity = 1.0 - np.mean(edge_diff) / 255.0
            
            return max(0.0, min(1.0, edge_similarity))
            
        except Exception as e:
            logger.warning(f"시각적 핏 분석 실패: {e}")
            return 0.7
    
    def _analyze_color_distribution(self, image: np.ndarray) -> Dict[str, Any]:
        """색상 분포 분석"""
        try:
            # RGB 히스토그램
            hist_r = cv2.calcHist([image], [2], None, [256], [0, 256])  # BGR 순서
            hist_g = cv2.calcHist([image], [1], None, [256], [0, 256])
            hist_b = cv2.calcHist([image], [0], None, [256], [0, 256])
            
            return {
                "mean_rgb": [float(np.mean(image[:, :, 2])), float(np.mean(image[:, :, 1])), float(np.mean(image[:, :, 0]))],
                "std_rgb": [float(np.std(image[:, :, 2])), float(np.std(image[:, :, 1])), float(np.std(image[:, :, 0]))],
                "histogram_peaks": {
                    "red": int(np.argmax(hist_r)),
                    "green": int(np.argmax(hist_g)),
                    "blue": int(np.argmax(hist_b))
                }
            }
        except Exception as e:
            return {"error": str(e)}
    
    def _analyze_histograms(self, fitted: np.ndarray, person: np.ndarray, clothing: np.ndarray) -> Dict[str, Any]:
        """히스토그램 분석"""
        try:
            # 각 이미지의 히스토그램 비교
            def calc_hist_similarity(img1, img2):
                if img1.shape != img2.shape:
                    img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
                
                hist1 = cv2.calcHist([img1], [0, 1, 2], None, [50, 50, 50], [0, 256, 0, 256, 0, 256])
                hist2 = cv2.calcHist([img2], [0, 1, 2], None, [50, 50, 50], [0, 256, 0, 256, 0, 256])
                
                return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
            
            return {
                "person_similarity": float(calc_hist_similarity(fitted, person)),
                "clothing_similarity": float(calc_hist_similarity(fitted, clothing))
            }
        except Exception as e:
            return {"error": str(e)}
    
    def _analyze_texture_statistics(self, image: np.ndarray) -> Dict[str, Any]:
        """텍스처 통계 분석"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # 텍스처 메트릭
            contrast = float(np.std(gray))
            sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            
            # LBP (Local Binary Pattern) 간소화 버전
            uniformity = float(np.sum(cv2.calcHist([gray], [0], None, [256], [0, 256])**2) / (gray.shape[0] * gray.shape[1])**2)
            
            return {
                "contrast": contrast,
                "sharpness": sharpness,
                "uniformity": uniformity
            }
        except Exception as e:
            return {"error": str(e)}
    
    def _analyze_edge_statistics(self, image: np.ndarray) -> Dict[str, Any]:
        """엣지 통계 분석"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # 다양한 엣지 검출
            sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            canny_edges = cv2.Canny(gray, 50, 150)
            
            return {
                "sobel_x_mean": float(np.mean(np.abs(sobel_x))),
                "sobel_y_mean": float(np.mean(np.abs(sobel_y))),
                "edge_density": float(np.sum(canny_edges > 0) / (gray.shape[0] * gray.shape[1])),
                "edge_strength": float(np.mean(np.sqrt(sobel_x**2 + sobel_y**2)))
            }
        except Exception as e:
            return {"error": str(e)}
    
    def _diagnose_quality_issues(self, metrics: QualityMetrics) -> Dict[str, Any]:
        """품질 문제 진단"""
        issues = []
        strengths = []
        
        metric_dict = metrics.to_dict()
        
        for metric_name, value in metric_dict.items():
            if metric_name == "overall_score":
                continue
                
            readable_name = metric_name.replace("_", " ").title()
            
            if value >= 0.8:
                strengths.append(f"{readable_name}: {value:.2f}")
            elif value < 0.6:
                issues.append(f"{readable_name}: {value:.2f}")
        
        return {
            "identified_issues": issues,
            "strengths": strengths,
            "critical_issues": len([i for i in metric_dict.values() if i < 0.5]),
            "excellent_metrics": len([i for i in metric_dict.values() if i >= 0.9])
        }
    
    def _analyze_improvement_potential(self, metrics: QualityMetrics) -> Dict[str, Any]:
        """개선 잠재력 분석"""
        metric_dict = metrics.to_dict()
        
        # 가장 낮은 점수와 가장 높은 점수 찾기
        min_metric = min(metric_dict.items(), key=lambda x: x[1] if x[0] != "overall_score" else 1.0)
        max_metric = max(metric_dict.items(), key=lambda x: x[1] if x[0] != "overall_score" else 0.0)
        
        improvement_potential = max_metric[1] - min_metric[1]
        
        return {
            "weakest_metric": {
                "name": min_metric[0].replace("_", " ").title(),
                "score": min_metric[1],
                "improvement_needed": max(0, 0.8 - min_metric[1])
            },
            "strongest_metric": {
                "name": max_metric[0].replace("_", " ").title(),
                "score": max_metric[1]
            },
            "improvement_potential": improvement_potential,
            "recommended_focus": "overall" if improvement_potential < 0.2 else min_metric[0].replace("_", " ")
        }
    
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
    
    # 2. 테스트 이미지 생성 (실제로는 실제 이미지 사용)
    def create_test_image(color=(128, 128, 128), noise_level=0):
        img = np.full((400, 300, 3), color, dtype=np.uint8)
        if noise_level > 0:
            noise = np.random.randint(-noise_level, noise_level, img.shape, dtype=np.int16)
            img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        return img
    
    # 테스트 이미지들
    fitted_image = create_test_image((120, 150, 180), noise_level=10)  # 약간의 노이즈
    person_image = create_test_image((130, 140, 160), noise_level=5)   # 원본 사람
    clothing_image = create_test_image((100, 160, 200), noise_level=0) # 깨끗한 의류
    
    # 파이프라인 결과 시뮬레이션
    pipeline_results = {
        'pose_result': {'pose_confidence': 0.85},
        'warping_result': {'quality_metrics': {'overall_quality': 0.78}},
        'parsing_result': {'confidence': 0.82}
    }
    
    # 3. 품질 평가 실행
    result = await quality_assessor.process(
        fitted_image=fitted_image,
        original_person=person_image,
        original_clothing=clothing_image,
        pipeline_results=pipeline_results
    )
    
    if result["success"]:
        print(f"✅ 품질 평가 성공!")
        print(f"📊 종합 점수: {result['score_percentage']}% ({result['letter_grade']})")
        print(f"🏆 품질 등급: {result['quality_grade']}")
        print(f"⏱️ 처리 시간: {result['assessment_info']['processing_time']:.3f}초")
        
        print("\n📈 메트릭 상세:")
        for metric, details in result['metric_breakdown'].items():
            print(f"  {details['description']}: {details['score']:.3f} ({details['status']})")
        
        print(f"\n💡 개선 제안 ({len(result['improvement_suggestions'])}개):")
        for suggestion in result['improvement_suggestions'][:3]:  # 상위 3개만 표시
            print(f"  [{suggestion['priority'].upper()}] {suggestion['issue']}")
            print(f"    → {suggestion['suggestion']}")
        
        if result['detailed_analysis']:
            print(f"\n🔍 상세 분석:")
            diagnosis = result['detailed_analysis'].get('quality_diagnosis', {})
            if diagnosis:
                print(f"  강점: {len(diagnosis.get('strengths', []))}개")
                print(f"  개선 필요: {len(diagnosis.get('identified_issues', []))}개")
        
        # JSON으로 저장
        with open("quality_assessment_result.json", "w") as f:
            json.dump(result, f, indent=2, default=str)
        
        print("💾 상세 결과 저장: quality_assessment_result.json")
        
    else:
        print(f"❌ 품질 평가 실패: {result['error']}")
    
    # 4. 리소스 정리
    await quality_assessor.cleanup()


# === 실제 이미지 파일로 테스트하는 함수 ===
async def test_with_real_images():
    """실제 이미지 파일로 테스트"""
    
    quality_assessor = RealQualityAssessmentStep(
        device='cpu',
        config={'enable_detailed_analysis': True}
    )
    
    await quality_assessor.initialize()
    
    # 실제 이미지 파일 경로들
    fitted_path = "output/fitted_result.jpg"      # 가상 피팅 결과
    person_path = "test_images/person.jpg"        # 원본 사람 사진
    clothing_path = "test_images/clothing.jpg"    # 원본 의류 사진
    
    # 파일 존재 확인
    if all(os.path.exists(path) for path in [fitted_path, person_path, clothing_path]):
        result = await quality_assessor.process(
            fitted_image=fitted_path,
            original_person=person_path,
            original_clothing=clothing_path
        )
        
        if result["success"]:
            print(f"🎯 실제 이미지 품질 평가 완료!")
            print(f"📊 종합 점수: {result['score_percentage']}%")
            print(f"🏆 등급: {result['quality_grade']} ({result['letter_grade']})")
            
            # 상세 리포트 생성
            report = f"""
# 가상 피팅 품질 평가 리포트

## 종합 평가
- **점수**: {result['score_percentage']}% ({result['letter_grade']})
- **등급**: {result['quality_grade']}
- **처리 시간**: {result['assessment_info']['processing_time']:.3f}초

## 메트릭 상세
"""
            for metric, details in result['metric_breakdown'].items():
                report += f"- **{details['description']}**: {details['score']:.3f} ({details['status']})\n"
            
            report += f"\n## 개선 제안\n"
            for i, suggestion in enumerate(result['improvement_suggestions'], 1):
                report += f"{i}. **[{suggestion['priority'].upper()}]** {suggestion['issue']}\n"
                report += f"   → {suggestion['suggestion']}\n\n"
            
            # 리포트 저장
            with open("quality_report.md", "w", encoding='utf-8') as f:
                f.write(report)
            
            print("📋 상세 리포트 저장: quality_report.md")
            
        else:
            print(f"❌ 평가 실패: {result['error']}")
    else:
        print("⚠️ 테스트 이미지 파일들이 없습니다.")
        print("필요한 파일들:")
        print(f"  - {fitted_path}")
        print(f"  - {person_path}")
        print(f"  - {clothing_path}")
    
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


if __name__ == "__main__":
    print("🎯 실제 품질 평가 시스템 테스트")
    print("=" * 50)
    
    # 기본 테스트
    asyncio.run(test_real_quality_assessment())
    
    print("\n" + "=" * 50)
    print("📁 실제 이미지 파일 테스트")
    
    # 실제 이미지 테스트
    asyncio.run(test_with_real_images())