"""
Virtual Fitting Step
가상 피팅을 위한 메인 스텝입니다.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, Union, List, Tuple
from pathlib import Path
import numpy as np
from PIL import Image
import time
import torch

from ...base.base_step import BaseStep
from ...base.base_step_mixin import BaseStepMixin
from .models.virtual_fitting_engine import VirtualFittingEngine
from .config.types import (
    FittingParameters, BodyMeasurements, ClothingInfo, 
    FittingResult, QualityMetrics
)
from .config.config import get_fitting_config, get_model_config

logger = logging.getLogger(__name__)

class VirtualFittingStep(BaseStep, BaseStepMixin):
    """가상 피팅 스텝 - 논문 구조 그대로 구현"""
    
    def __init__(self, 
                 device: str = "auto",
                 quality_level: str = "high",
                 model_type: str = "hybrid",
                 enable_ensemble: bool = True,
                 checkpoint_paths: Optional[Dict[str, str]] = None,
                 **kwargs):
        """
        가상 피팅 스텝 초기화
        
        Args:
            device: 디바이스 (auto, cpu, cuda, mps)
            quality_level: 품질 레벨 (low, balanced, high, ultra)
            model_type: 모델 타입 (hr_viton, ootd, viton_hd, hybrid)
            enable_ensemble: 앙상블 활성화 여부
            checkpoint_paths: 체크포인트 경로 딕셔너리
        """
        super().__init__()
        BaseStepMixin.__init__(self)
        
        self.device = device
        self.quality_level = quality_level
        self.model_type = model_type
        self.enable_ensemble = enable_ensemble
        self.checkpoint_paths = checkpoint_paths or {}
        
        # 설정 로드
        self.config = get_fitting_config(quality_level, model_type)
        
        # 가상 피팅 엔진 초기화
        self.fitting_engine = None
        self._initialize_engine()
        
        # 스텝 정보
        self.step_name = "virtual_fitting"
        self.step_description = "가상 피팅을 통한 의류 피팅 생성"
        self.step_version = "1.0.0"
        
        logger.info(f"Virtual Fitting Step 초기화 완료: {model_type}, {quality_level}")
    
    def _initialize_engine(self):
        """피팅 엔진 초기화"""
        try:
            self.fitting_engine = VirtualFittingEngine(
                device=self.device,
                quality_level=self.quality_level,
                model_type=self.model_type,
                enable_ensemble=self.enable_ensemble,
                checkpoint_paths=self.checkpoint_paths
            )
            logger.info("가상 피팅 엔진 초기화 성공")
        except Exception as e:
            logger.error(f"가상 피팅 엔진 초기화 실패: {e}")
            raise
    
    async def process(self, 
                      person_image: Union[np.ndarray, Image.Image], 
                      clothing_image: Union[np.ndarray, Image.Image],
                      person_parsing: Optional[Union[np.ndarray, Image.Image]] = None,
                      clothing_parsing: Optional[Union[np.ndarray, Image.Image]] = None,
                      body_measurements: Optional[Dict[str, float]] = None,
                      clothing_info: Optional[Dict[str, Any]] = None,
                      fitting_parameters: Optional[Dict[str, Any]] = None,
                      **kwargs) -> Dict[str, Any]:
        """
        가상 피팅 처리
        
        Args:
            person_image: 사람 이미지
            clothing_image: 의류 이미지
            person_parsing: 사람 파싱 (선택사항)
            clothing_parsing: 의류 파싱 (선택사항)
            body_measurements: 신체 측정값 (선택사항)
            clothing_info: 의류 정보 (선택사항)
            fitting_parameters: 피팅 파라미터 (선택사항)
        
        Returns:
            처리 결과 딕셔너리
        """
        start_time = time.time()
        
        try:
            # 입력 검증
            self._validate_inputs(person_image, clothing_image)
            
            # 파라미터 변환
            body_measurements_obj = self._convert_body_measurements(body_measurements)
            clothing_info_obj = self._convert_clothing_info(clothing_info)
            fitting_params_obj = self._convert_fitting_parameters(fitting_parameters)
            
            # 가상 피팅 수행
            fitting_result = self.fitting_engine.fit_clothing(
                person_image=person_image,
                clothing_image=clothing_image,
                person_parsing=person_parsing,
                clothing_parsing=clothing_parsing,
                body_measurements=body_measurements,
                clothing_info=clothing_info,
                **kwargs
            )
            
            # 품질 분석 및 등급 평가
            quality_analysis = self._analyze_quality(fitting_result)
            
            # 결과 구성
            processing_time = time.time() - start_time
            
            result = {
                "success": True,
                "step_name": self.step_name,
                "step_version": self.step_version,
                "processing_time": processing_time,
                "fitting_result": {
                    "output_image": self._tensor_to_numpy(fitting_result.output_image),
                    "quality_metrics": fitting_result.quality_metrics.__dict__,
                    "model_used": fitting_result.model_used,
                    "confidence": fitting_result.confidence,
                    "metadata": fitting_result.metadata
                },
                "quality_analysis": quality_analysis,
                "input_info": {
                    "person_image_shape": self._get_image_shape(person_image),
                    "clothing_image_shape": self._get_image_shape(clothing_image),
                    "has_person_parsing": person_parsing is not None,
                    "has_clothing_parsing": clothing_parsing is not None,
                    "body_measurements": body_measurements,
                    "clothing_info": clothing_info
                }
            }
            
            logger.info(f"가상 피팅 완료: {processing_time:.2f}초, 품질 등급: {quality_analysis['quality_grade']}")
            
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"가상 피팅 실패: {e}")
            
            return {
                "success": False,
                "step_name": self.step_name,
                "step_version": self.step_version,
                "processing_time": processing_time,
                "error": str(e),
                "error_type": type(e).__name__
            }
    
    def _validate_inputs(self, person_image: Union[np.ndarray, Image.Image], 
                        clothing_image: Union[np.ndarray, Image.Image]):
        """입력 검증"""
        if person_image is None:
            raise ValueError("사람 이미지가 필요합니다")
        
        if clothing_image is None:
            raise ValueError("의류 이미지가 필요합니다")
        
        # 이미지 형식 검증
        if not self._is_valid_image(person_image):
            raise ValueError("사람 이미지 형식이 올바르지 않습니다")
        
        if not self._is_valid_image(clothing_image):
            raise ValueError("의류 이미지 형식이 올바르지 않습니다")
    
    def _is_valid_image(self, image: Union[np.ndarray, Image.Image]) -> bool:
        """이미지 유효성 검사"""
        try:
            if isinstance(image, np.ndarray):
                return image.ndim in [2, 3] and image.size > 0
            elif isinstance(image, Image.Image):
                return image.size[0] > 0 and image.size[1] > 0
            else:
                return False
        except:
            return False
    
    def _convert_body_measurements(self, measurements: Optional[Dict[str, float]]) -> Optional[BodyMeasurements]:
        """신체 측정값 변환"""
        if measurements is None:
            return None
        
        try:
            return BodyMeasurements(**measurements)
        except Exception as e:
            logger.warning(f"신체 측정값 변환 실패: {e}")
            return None
    
    def _convert_clothing_info(self, info: Optional[Dict[str, Any]]) -> Optional[ClothingInfo]:
        """의류 정보 변환"""
        if info is None:
            return None
        
        try:
            return ClothingInfo(**info)
        except Exception as e:
            logger.warning(f"의류 정보 변환 실패: {e}")
            return None
    
    def _convert_fitting_parameters(self, params: Optional[Dict[str, Any]]) -> Optional[FittingParameters]:
        """피팅 파라미터 변환"""
        if params is None:
            return None
        
        try:
            return FittingParameters(**params)
        except Exception as e:
            logger.warning(f"피팅 파라미터 변환 실패: {e}")
            return None
    
    def _analyze_quality(self, fitting_result: FittingResult) -> Dict[str, Any]:
        """품질 분석 및 등급 평가"""
        quality_metrics = fitting_result.quality_metrics
        
        # 품질 등급 평가
        quality_grade = self._evaluate_quality_grade(quality_metrics)
        
        # 피팅 등급 평가
        fit_grade = self._evaluate_fit_grade(quality_metrics)
        
        # 세부 품질 분석
        detailed_analysis = self._detailed_quality_analysis(quality_metrics)
        
        return {
            "quality_grade": quality_grade,
            "fit_grade": fit_grade,
            "overall_score": self._calculate_overall_score(quality_metrics),
            "detailed_analysis": detailed_analysis,
            "recommendations": self._generate_recommendations(quality_metrics)
        }
    
    def _evaluate_quality_grade(self, metrics: QualityMetrics) -> str:
        """품질 등급 평가"""
        # SSIM 기반 등급
        if metrics.ssim >= 0.9:
            ssim_grade = "A+"
        elif metrics.ssim >= 0.8:
            ssim_grade = "A"
        elif metrics.ssim >= 0.7:
            ssim_grade = "B+"
        elif metrics.ssim >= 0.6:
            ssim_grade = "B"
        elif metrics.ssim >= 0.5:
            ssim_grade = "C+"
        else:
            ssim_grade = "C"
        
        # PSNR 기반 등급
        if metrics.psnr >= 40:
            psnr_grade = "A+"
        elif metrics.psnr >= 35:
            psnr_grade = "A"
        elif metrics.psnr >= 30:
            psnr_grade = "B+"
        elif metrics.psnr >= 25:
            psnr_grade = "B"
        elif metrics.psnr >= 20:
            psnr_grade = "C+"
        else:
            psnr_grade = "C"
        
        # 종합 등급
        if ssim_grade == "A+" and psnr_grade == "A+":
            return "A+"
        elif ssim_grade in ["A+", "A"] and psnr_grade in ["A+", "A"]:
            return "A"
        elif ssim_grade in ["A+", "A", "B+"] and psnr_grade in ["A+", "A", "B+"]:
            return "B+"
        elif ssim_grade in ["A+", "A", "B+", "B"] and psnr_grade in ["A+", "A", "B+", "B"]:
            return "B"
        elif ssim_grade in ["A+", "A", "B+", "B", "C+"] and psnr_grade in ["A+", "A", "B+", "B", "C+"]:
            return "C+"
        else:
            return "C"
    
    def _evaluate_fit_grade(self, metrics: QualityMetrics) -> str:
        """피팅 등급 평가"""
        # 색상 일관성 및 질감 보존 기반
        color_score = metrics.color_consistency
        texture_score = metrics.texture_preservation
        edge_score = metrics.edge_quality
        blending_score = metrics.blending_quality
        
        # 종합 점수
        fit_score = (color_score + texture_score + edge_score + blending_score) / 4
        
        if fit_score >= 0.9:
            return "Perfect"
        elif fit_score >= 0.8:
            return "Excellent"
        elif fit_score >= 0.7:
            return "Good"
        elif fit_score >= 0.6:
            return "Fair"
        elif fit_score >= 0.5:
            return "Poor"
        else:
            return "Very Poor"
    
    def _detailed_quality_analysis(self, metrics: QualityMetrics) -> Dict[str, Any]:
        """세부 품질 분석"""
        return {
            "structural_similarity": {
                "score": metrics.ssim,
                "grade": "A+" if metrics.ssim >= 0.9 else "A" if metrics.ssim >= 0.8 else "B+" if metrics.ssim >= 0.7 else "B" if metrics.ssim >= 0.6 else "C",
                "description": self._get_ssim_description(metrics.ssim)
            },
            "peak_signal_noise_ratio": {
                "score": metrics.psnr,
                "grade": "A+" if metrics.psnr >= 40 else "A" if metrics.psnr >= 35 else "B+" if metrics.psnr >= 30 else "B" if metrics.psnr >= 25 else "C",
                "description": self._get_psnr_description(metrics.psnr)
            },
            "color_consistency": {
                "score": metrics.color_consistency,
                "grade": "A+" if metrics.color_consistency >= 0.9 else "A" if metrics.color_consistency >= 0.8 else "B+" if metrics.color_consistency >= 0.7 else "B" if metrics.color_consistency >= 0.6 else "C",
                "description": self._get_color_description(metrics.color_consistency)
            },
            "texture_preservation": {
                "score": metrics.texture_preservation,
                "grade": "A+" if metrics.texture_preservation >= 0.9 else "A" if metrics.texture_preservation >= 0.8 else "B+" if metrics.texture_preservation >= 0.7 else "B" if metrics.texture_preservation >= 0.6 else "C",
                "description": self._get_texture_description(metrics.texture_preservation)
            },
            "edge_quality": {
                "score": metrics.edge_quality,
                "grade": "A+" if metrics.edge_quality >= 0.9 else "A" if metrics.edge_quality >= 0.8 else "B+" if metrics.edge_quality >= 0.7 else "B" if metrics.edge_quality >= 0.6 else "C",
                "description": self._get_edge_description(metrics.edge_quality)
            },
            "blending_quality": {
                "score": metrics.blending_quality,
                "grade": "A+" if metrics.blending_quality >= 0.9 else "A" if metrics.blending_quality >= 0.8 else "B+" if metrics.blending_quality >= 0.7 else "B" if metrics.blending_quality >= 0.6 else "C",
                "description": self._get_blending_description(metrics.blending_quality)
            }
        }
    
    def _get_ssim_description(self, ssim: float) -> str:
        """SSIM 설명"""
        if ssim >= 0.9:
            return "매우 높은 구조적 유사성, 원본과 거의 동일"
        elif ssim >= 0.8:
            return "높은 구조적 유사성, 매우 좋은 품질"
        elif ssim >= 0.7:
            return "좋은 구조적 유사성, 만족스러운 품질"
        elif ssim >= 0.6:
            return "보통의 구조적 유사성, 개선 여지 있음"
        else:
            return "낮은 구조적 유사성, 품질 개선 필요"
    
    def _get_psnr_description(self, psnr: float) -> str:
        """PSNR 설명"""
        if psnr >= 40:
            return "매우 높은 신호 대 잡음비, 우수한 품질"
        elif psnr >= 35:
            return "높은 신호 대 잡음비, 매우 좋은 품질"
        elif psnr >= 30:
            return "좋은 신호 대 잡음비, 만족스러운 품질"
        elif psnr >= 25:
            return "보통의 신호 대 잡음비, 개선 여지 있음"
        else:
            return "낮은 신호 대 잡음비, 품질 개선 필요"
    
    def _get_color_description(self, color_score: float) -> str:
        """색상 일관성 설명"""
        if color_score >= 0.9:
            return "매우 높은 색상 일관성, 자연스러운 색상"
        elif color_score >= 0.8:
            return "높은 색상 일관성, 좋은 색상 보존"
        elif color_score >= 0.7:
            return "좋은 색상 일관성, 만족스러운 색상"
        elif color_score >= 0.6:
            return "보통의 색상 일관성, 약간의 색상 차이"
        else:
            return "낮은 색상 일관성, 색상 보정 필요"
    
    def _get_texture_description(self, texture_score: float) -> str:
        """질감 보존 설명"""
        if texture_score >= 0.9:
            return "매우 높은 질감 보존, 원본 질감 완벽 재현"
        elif texture_score >= 0.8:
            return "높은 질감 보존, 매우 좋은 질감 표현"
        elif texture_score >= 0.7:
            return "좋은 질감 보존, 만족스러운 질감"
        elif texture_score >= 0.6:
            return "보통의 질감 보존, 약간의 질감 손실"
        else:
            return "낮은 질감 보존, 질감 개선 필요"
    
    def _get_edge_description(self, edge_score: float) -> str:
        """엣지 품질 설명"""
        if edge_score >= 0.9:
            return "매우 높은 엣지 품질, 선명하고 깔끔한 경계"
        elif edge_score >= 0.8:
            return "높은 엣지 품질, 좋은 경계 표현"
        elif edge_score >= 0.7:
            return "좋은 엣지 품질, 만족스러운 경계"
        elif edge_score >= 0.6:
            return "보통의 엣지 품질, 약간의 경계 흐림"
        else:
            return "낮은 엣지 품질, 경계 개선 필요"
    
    def _get_blending_description(self, blending_score: float) -> str:
        """블렌딩 품질 설명"""
        if blending_score >= 0.9:
            return "매우 높은 블렌딩 품질, 자연스러운 합성"
        elif blending_score >= 0.8:
            return "높은 블렌딩 품질, 매우 좋은 합성"
        elif blending_score >= 0.7:
            return "좋은 블렌딩 품질, 만족스러운 합성"
        elif blending_score >= 0.6:
            return "보통의 블렌딩 품질, 약간의 합성 흔적"
        else:
            return "낮은 블렌딩 품질, 합성 개선 필요"
    
    def _calculate_overall_score(self, metrics: QualityMetrics) -> float:
        """종합 점수 계산"""
        # 가중 평균
        weights = {
            'ssim': 0.25,
            'psnr': 0.20,
            'color_consistency': 0.15,
            'texture_preservation': 0.15,
            'edge_quality': 0.15,
            'blending_quality': 0.10
        }
        
        # PSNR 정규화 (0-1)
        normalized_psnr = min(1.0, metrics.psnr / 50.0)
        
        overall_score = (
            weights['ssim'] * metrics.ssim +
            weights['psnr'] * normalized_psnr +
            weights['color_consistency'] * metrics.color_consistency +
            weights['texture_preservation'] * metrics.texture_preservation +
            weights['edge_quality'] * metrics.edge_quality +
            weights['blending_quality'] * metrics.blending_quality
        )
        
        return round(overall_score, 3)
    
    def _generate_recommendations(self, metrics: QualityMetrics) -> List[str]:
        """개선 권장사항 생성"""
        recommendations = []
        
        if metrics.ssim < 0.7:
            recommendations.append("구조적 유사성 개선을 위해 더 높은 품질 레벨 사용 권장")
        
        if metrics.psnr < 30:
            recommendations.append("신호 대 잡음비 개선을 위해 노이즈 감소 후처리 적용 권장")
        
        if metrics.color_consistency < 0.8:
            recommendations.append("색상 일관성 개선을 위해 색상 보정 후처리 적용 권장")
        
        if metrics.texture_preservation < 0.8:
            recommendations.append("질감 보존 개선을 위해 고해상도 모델 사용 권장")
        
        if metrics.edge_quality < 0.8:
            recommendations.append("엣지 품질 개선을 위해 엣지 강화 후처리 적용 권장")
        
        if metrics.blending_quality < 0.8:
            recommendations.append("블렌딩 품질 개선을 위해 블렌딩 최적화 후처리 적용 권장")
        
        if not recommendations:
            recommendations.append("현재 품질이 매우 우수합니다. 추가 개선이 필요하지 않습니다.")
        
        return recommendations
    
    def _tensor_to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        """텐서를 numpy 배열로 변환"""
        if isinstance(tensor, torch.Tensor):
            # 디바이스에서 CPU로 이동
            if tensor.device.type != 'cpu':
                tensor = tensor.cpu()
            
            # 정규화 (-1, 1) -> (0, 255)
            if tensor.min() < 0:
                tensor = (tensor + 1) * 127.5
            else:
                tensor = tensor * 255
            
            # numpy로 변환
            array = tensor.detach().numpy()
            
            # 차원 정리
            if array.ndim == 4:
                array = array[0]  # 배치 차원 제거
            
            # 채널 순서 조정 (C, H, W) -> (H, W, C)
            if array.ndim == 3 and array.shape[0] == 3:
                array = np.transpose(array, (1, 2, 0))
            
            # 클리핑
            array = np.clip(array, 0, 255).astype(np.uint8)
            
            return array
        else:
            return tensor
    
    def _get_image_shape(self, image: Union[np.ndarray, Image.Image]) -> Tuple[int, int, int]:
        """이미지 형태 반환"""
        if isinstance(image, np.ndarray):
            if image.ndim == 3:
                return (image.shape[0], image.shape[1], image.shape[2])
            else:
                return (image.shape[0], image.shape[1], 1)
        elif isinstance(image, Image.Image):
            return (image.size[1], image.size[0], len(image.getbands()))
        else:
            return (0, 0, 0)
    
    def get_step_info(self) -> Dict[str, Any]:
        """스텝 정보 반환"""
        return {
            "step_name": self.step_name,
            "step_description": self.step_description,
            "step_version": self.step_version,
            "device": str(self.device),
            "model_type": self.model_type,
            "quality_level": self.quality_level,
            "enable_ensemble": self.enable_ensemble
        }
    
    def switch_model(self, model_type: str):
        """모델 변경"""
        if self.fitting_engine:
            self.fitting_engine.switch_model(model_type)
            self.model_type = model_type
            logger.info(f"모델 변경: {model_type}")
    
    def switch_quality_level(self, quality_level: str):
        """품질 레벨 변경"""
        if self.fitting_engine:
            self.fitting_engine.switch_quality_level(quality_level)
            self.quality_level = quality_level
            logger.info(f"품질 레벨 변경: {quality_level}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """모델 정보 반환"""
        if self.fitting_engine:
            return self.fitting_engine.get_model_info()
        return {}
    
    async def warmup(self):
        """워밍업 수행"""
        try:
            # 더미 이미지로 추론 수행
            dummy_person = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
            dummy_clothing = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
            
            await self.process(dummy_person, dummy_clothing)
            logger.info("워밍업 완료")
            
        except Exception as e:
            logger.warning(f"워밍업 실패: {e}")
    
    def cleanup(self):
        """리소스 정리"""
        if self.fitting_engine:
            self.fitting_engine.cleanup()
        logger.info("리소스 정리 완료")
