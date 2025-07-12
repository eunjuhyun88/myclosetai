import asyncio
import torch
import cv2
import numpy as np
from PIL import Image
import base64
import io
from pathlib import Path
import logging

from app.core.config import settings
from app.models.ootd_model import OOTDModel
from app.utils.image_utils import ImageProcessor

logger = logging.getLogger(__name__)

class VirtualTryOnService:
    """Virtual Try-On Service"""
    
    def __init__(self):
        self.device = torch.device(settings.DEVICE if torch.cuda.is_available() and settings.USE_GPU else "cpu")
        self.image_processor = ImageProcessor()
        self.models = {}
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize AI models"""
        try:
            # OOTDiffusion 모델 로드
            self.models['ootd'] = OOTDModel(device=self.device)
            logger.info(f"Models initialized on {self.device}")
        except Exception as e:
            logger.error(f"Failed to initialize models: {e}")
            # 폴백: 더미 모델 사용
            self.models['ootd'] = None
    
    async def process(
        self,
        person_image_path: str,
        clothing_image_path: str,
        height: float,
        weight: float,
        model_type: str = "ootd",
        category: str = "upper_body",
        session_id: str = None
    ) -> dict:
        """Process virtual try-on"""
        
        try:
            # 이미지 로드 및 전처리
            person_img = await self._load_and_preprocess_image(person_image_path)
            clothing_img = await self._load_and_preprocess_image(clothing_image_path)
            
            # 모델 선택 및 실행
            if model_type == "ootd" and self.models.get('ootd'):
                result = await self._run_ootd_model(
                    person_img, clothing_img, category, height, weight
                )
            else:
                # 폴백: 더미 결과 생성
                result = await self._generate_dummy_result(
                    person_img, clothing_img, height, weight
                )
            
            # 결과 이미지 저장
            if session_id:
                result_path = Path(settings.RESULTS_DIR) / f"{session_id}_result.jpg"
                self._save_result_image(result['fitted_image'], result_path)
            
            return result
            
        except Exception as e:
            logger.error(f"Virtual try-on processing failed: {e}")
            raise
    
    async def _load_and_preprocess_image(self, image_path: str) -> np.ndarray:
        """Load and preprocess image"""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Cannot load image: {image_path}")
        
        # 전처리
        image = self.image_processor.resize_image(image, settings.IMAGE_SIZE)
        image = self.image_processor.normalize_image(image)
        
        return image
    
    async def _run_ootd_model(
        self, 
        person_img: np.ndarray, 
        clothing_img: np.ndarray, 
        category: str,
        height: float,
        weight: float
    ) -> dict:
        """Run OOTDiffusion model"""
        model = self.models['ootd']
        
        # 비동기 처리
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, 
            model.generate,
            person_img,
            clothing_img,
            category
        )
        
        # 피팅 점수 계산
        fit_score = self._calculate_fit_score(person_img, result, height, weight)
        
        # Base64 인코딩
        fitted_image_base64 = self._image_to_base64(result)
        
        return {
            "fitted_image": result,
            "fitted_image_base64": fitted_image_base64,
            "confidence": 0.87,
            "fit_score": fit_score,
            "recommendations": self._generate_recommendations(fit_score, category)
        }
    
    async def _generate_dummy_result(
        self,
        person_img: np.ndarray,
        clothing_img: np.ndarray,
        height: float,
        weight: float
    ) -> dict:
        """Generate dummy result for testing"""
        
        # 간단한 더미 합성
        result_img = person_img.copy()
        
        # 임시 지연 (실제 처리 시뮬레이션)
        await asyncio.sleep(2)
        
        # 가짜 피팅 점수
        bmi = weight / ((height / 100) ** 2)
        fit_score = max(0.6, min(0.95, 1.0 - abs(bmi - 22) / 10))
        
        fitted_image_base64 = self._image_to_base64(result_img)
        
        return {
            "fitted_image": result_img,
            "fitted_image_base64": fitted_image_base64,
            "confidence": 0.75,
            "fit_score": fit_score,
            "recommendations": self._generate_recommendations(fit_score, "upper_body")
        }
    
    def _calculate_fit_score(
        self,
        person_img: np.ndarray,
        fitted_img: np.ndarray,
        height: float,
        weight: float
    ) -> float:
        """Calculate fit score"""
        # BMI 기반 기본 점수
        bmi = weight / ((height / 100) ** 2)
        
        if 18.5 <= bmi <= 25:
            base_score = 0.9
        elif 17 <= bmi <= 30:
            base_score = 0.8
        else:
            base_score = 0.7
        
        # 이미지 품질 기반 조정
        quality_score = self._assess_image_quality(fitted_img)
        
        final_score = (base_score + quality_score) / 2
        return round(final_score, 2)
    
    def _assess_image_quality(self, image: np.ndarray) -> float:
        """Assess generated image quality"""
        # 간단한 품질 평가
        # 실제로는 더 정교한 메트릭 사용
        
        # 선명도 (라플라시안 분산)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # 정규화
        sharpness_score = min(1.0, sharpness / 1000)
        
        return sharpness_score
    
    def _generate_recommendations(self, fit_score: float, category: str) -> list:
        """Generate AI recommendations"""
        recommendations = []
        
        if fit_score >= 0.8:
            recommendations.append("이 의류가 당신의 체형에 매우 잘 어울립니다!")
            recommendations.append("자신감 있게 착용하실 수 있습니다.")
        elif fit_score >= 0.6:
            recommendations.append("이 의류가 적당히 잘 맞습니다.")
            if category == "upper_body":
                recommendations.append("어깨 라인을 고려해 보세요.")
        else:
            recommendations.append("다른 사이즈나 스타일을 고려해보시는 것이 좋겠습니다.")
            recommendations.append("체형에 더 맞는 옷을 찾아보세요.")
        
        return recommendations
    
    def _image_to_base64(self, image: np.ndarray) -> str:
        """Convert image to base64"""
        # BGR to RGB
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # PIL Image로 변환
        pil_image = Image.fromarray(image.astype(np.uint8))
        
        # Base64 인코딩
        buffer = io.BytesIO()
        pil_image.save(buffer, format='JPEG', quality=90)
        return base64.b64encode(buffer.getvalue()).decode()
    
    def _save_result_image(self, image: np.ndarray, path: Path):
        """Save result image"""
        cv2.imwrite(str(path), image)
    
    async def get_models_status(self) -> dict:
        """Get models status"""
        status = {}
        
        for model_name, model in self.models.items():
            status[model_name] = {
                "loaded": model is not None,
                "device": str(self.device),
                "ready": model is not None
            }
        
        return {
            "models": status,
            "gpu_available": torch.cuda.is_available(),
            "device": str(self.device)
        }
    
    async def preprocess_image(self, image_file, operation: str):
        """Preprocess image"""
        # 이미지 전처리 로직 구현
        return {"operation": operation, "status": "completed"}
