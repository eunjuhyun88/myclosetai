# backend/app/services/virtual_fitter.py
import torch
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import numpy as np
import asyncio
import cv2
from typing import Optional, Dict, Any
import logging
import os
import random
from datetime import datetime

from app.core.gpu_config import DEVICE, MODEL_CONFIG

logger = logging.getLogger(__name__)

class VirtualFitter:
    def __init__(self):
        self.models_loaded = False
        self.demo_mode = True
        self.device = DEVICE
        self.model_config = MODEL_CONFIG
        
        # 지원하는 모델 타입들
        self.supported_models = {
            "demo": "데모 모드 (빠른 합성)",
            "ootd": "OOTDiffusion (고품질)",
            "viton": "VITON-HD (실시간)",
            "acgpn": "ACGPN (정밀 피팅)"
        }
        
    async def initialize_models(self):
        """AI 모델 초기화"""
        try:
            logger.info("🤖 AI 모델 초기화 시작...")
            
            # 모델 로딩 시뮬레이션 (실제로는 모델 파일 로드)
            await asyncio.sleep(2)
            
            # 실제 모델들이 준비되면 여기서 로드
            # self.ootd_model = self._load_ootd_model()
            # self.viton_model = self._load_viton_model()
            
            self.models_loaded = True
            logger.info("✅ AI 모델 초기화 완료")
            
        except Exception as e:
            logger.error(f"❌ 모델 초기화 실패: {e}")
            self.demo_mode = True
            self.models_loaded = False
    
    async def demo_fitting(
        self, 
        person_image: Image.Image, 
        clothing_image: Image.Image,
        height: float,
        weight: float
    ) -> Image.Image:
        """데모 가상 피팅 (고급 합성 버전)"""
        try:
            logger.info("🎭 데모 가상 피팅 시작")
            
            # 1. 기본 설정
            result = person_image.copy()
            draw = ImageDraw.Draw(result)
            
            # 2. 의류 위치 계산 (체형 기반)
            person_width, person_height = person_image.size
            clothing_position = self._calculate_clothing_position(
                person_width, person_height, height, weight
            )
            
            # 3. 의류 이미지 변형
            transformed_clothing = self._transform_clothing_for_body(
                clothing_image, clothing_position, height, weight
            )
            
            # 4. 자연스러운 합성
            result = self._blend_clothing_naturally(
                result, transformed_clothing, clothing_position
            )
            
            # 5. 후처리 효과
            result = self._apply_post_effects(result)
            
            # 6. 워터마크 추가
            result = self._add_watermark(result, "MyCloset AI Demo")
            
            logger.info("✅ 데모 피팅 완료")
            return result
            
        except Exception as e:
            logger.error(f"❌ 데모 피팅 오류: {e}")
            # 기본 합성으로 대체
            return self._basic_overlay(person_image, clothing_image)
    
    def _calculate_clothing_position(
        self, 
        person_width: int, 
        person_height: int, 
        height: float, 
        weight: float
    ) -> Dict[str, Any]:
        """체형에 따른 의류 위치 계산"""
        
        # BMI 계산
        bmi = weight / ((height / 100) ** 2)
        
        # 체형에 따른 스케일링
        if bmi < 18.5:
            scale_factor = 0.85  # 마른 체형
        elif bmi > 25:
            scale_factor = 1.15  # 통통한 체형
        else:
            scale_factor = 1.0   # 보통 체형
        
        # 의류 위치 및 크기
        clothing_width = int(person_width * 0.6 * scale_factor)
        clothing_height = int(person_height * 0.4 * scale_factor)
        
        # 상의 위치 (가슴 부분)
        x_position = (person_width - clothing_width) // 2
        y_position = int(person_height * 0.25)  # 상체 25% 지점
        
        return {
            "x": x_position,
            "y": y_position,
            "width": clothing_width,
            "height": clothing_height,
            "scale_factor": scale_factor,
            "bmi": bmi
        }
    
    def _transform_clothing_for_body(
        self, 
        clothing_image: Image.Image, 
        position: Dict[str, Any],
        height: float,
        weight: float
    ) -> Image.Image:
        """체형에 맞게 의류 변형"""
        
        # 1. 기본 리사이즈
        resized = clothing_image.resize(
            (position["width"], position["height"]), 
            Image.Resampling.LANCZOS
        )
        
        # 2. 체형에 따른 왜곡 효과
        bmi = position["bmi"]
        
        if bmi < 18.5:
            # 마른 체형: 약간 수직으로 늘림
            resized = resized.resize(
                (int(position["width"] * 0.95), int(position["height"] * 1.05)),
                Image.Resampling.LANCZOS
            )
        elif bmi > 25:
            # 통통한 체형: 약간 수평으로 늘림
            resized = resized.resize(
                (int(position["width"] * 1.05), int(position["height"] * 0.95)),
                Image.Resampling.LANCZOS
            )
        
        # 3. 자연스러운 곡률 효과 추가
        resized = self._add_fabric_curve(resized)
        
        return resized
    
    def _add_fabric_curve(self, clothing_image: Image.Image) -> Image.Image:
        """천의 자연스러운 곡률 효과"""
        try:
            # OpenCV로 변환
            cv_image = cv2.cvtColor(np.array(clothing_image), cv2.COLOR_RGB2BGR)
            
            # 약간의 barrel distortion 효과
            height, width = cv_image.shape[:2]
            
            # 왜곡 매트릭스 생성
            map_x = np.zeros((height, width), dtype=np.float32)
            map_y = np.zeros((height, width), dtype=np.float32)
            
            center_x, center_y = width // 2, height // 2
            
            for y in range(height):
                for x in range(width):
                    # 중심으로부터의 거리
                    dx = x - center_x
                    dy = y - center_y
                    r = np.sqrt(dx*dx + dy*dy)
                    
                    # 약한 barrel distortion
                    factor = 1 + 0.00002 * r * r
                    
                    map_x[y, x] = center_x + dx * factor
                    map_y[y, x] = center_y + dy * factor
            
            # 왜곡 적용
            curved = cv2.remap(cv_image, map_x, map_y, cv2.INTER_LINEAR)
            
            # PIL로 변환
            return Image.fromarray(cv2.cvtColor(curved, cv2.COLOR_BGR2RGB))
            
        except Exception as e:
            logger.warning(f"곡률 효과 실패: {e}")
            return clothing_image
    
    def _blend_clothing_naturally(
        self, 
        person_image: Image.Image, 
        clothing_image: Image.Image, 
        position: Dict[str, Any]
    ) -> Image.Image:
        """자연스러운 의류 합성"""
        
        # 1. 알파 블렌딩을 위한 마스크 생성
        mask = self._create_blending_mask(clothing_image)
        
        # 2. 그림자 효과 추가
        shadow = self._create_shadow_effect(clothing_image)
        person_image.paste(shadow, 
                          (position["x"] + 3, position["y"] + 3), 
                          shadow)
        
        # 3. 의류 합성
        person_image.paste(clothing_image, 
                          (position["x"], position["y"]), 
                          mask)
        
        return person_image
    
    def _create_blending_mask(self, clothing_image: Image.Image) -> Image.Image:
        """블렌딩용 마스크 생성"""
        # 그레이스케일로 변환
        gray = clothing_image.convert('L')
        
        # 가장자리 페이드 효과
        mask = gray.copy()
        
        # 가장자리를 점진적으로 투명하게
        width, height = mask.size
        for y in range(height):
            for x in range(width):
                # 가장자리로부터의 거리
                edge_dist = min(x, y, width-x-1, height-y-1)
                fade_zone = 10  # 페이드 영역 크기
                
                if edge_dist < fade_zone:
                    # 가장자리일수록 투명하게
                    alpha = int(255 * (edge_dist / fade_zone))
                    current_alpha = mask.getpixel((x, y))
                    new_alpha = min(current_alpha, alpha)
                    mask.putpixel((x, y), new_alpha)
        
        return mask
    
    def _create_shadow_effect(self, clothing_image: Image.Image) -> Image.Image:
        """그림자 효과 생성"""
        # 그림자용 이미지 생성
        shadow = clothing_image.convert('RGBA')
        
        # 어둡게 만들기
        shadow_data = []
        for pixel in shadow.getdata():
            if pixel[3] > 0:  # 투명하지 않은 픽셀
                # 어둡게 만들고 투명도 조정
                shadow_data.append((30, 30, 30, 100))
            else:
                shadow_data.append((0, 0, 0, 0))
        
        shadow.putdata(shadow_data)
        
        # 블러 효과 추가
        shadow = shadow.filter(ImageFilter.GaussianBlur(radius=2))
        
        return shadow
    
    def _apply_post_effects(self, image: Image.Image) -> Image.Image:
        """후처리 효과 적용"""
        
        # 1. 약간의 색상 보정
        enhanced = image.copy()
        
        # 2. 미세한 노이즈 추가 (자연스러움)
        enhanced = self._add_subtle_noise(enhanced)
        
        # 3. 약간의 샤프닝
        enhanced = enhanced.filter(ImageFilter.UnsharpMask(radius=1, percent=50))
        
        return enhanced
    
    def _add_subtle_noise(self, image: Image.Image) -> Image.Image:
        """미세한 노이즈 추가"""
        try:
            # NumPy 배열로 변환
            img_array = np.array(image)
            
            # 가우시안 노이즈 생성
            noise = np.random.normal(0, 2, img_array.shape)
            
            # 노이즈 추가
            noisy = img_array + noise
            noisy = np.clip(noisy, 0, 255).astype(np.uint8)
            
            return Image.fromarray(noisy)
            
        except Exception as e:
            logger.warning(f"노이즈 추가 실패: {e}")
            return image
    
    def _add_watermark(self, image: Image.Image, text: str) -> Image.Image:
        """워터마크 추가"""
        try:
            draw = ImageDraw.Draw(image)
            
            # 기본 폰트 사용
            try:
                # 시스템 폰트 시도
                font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 16)
            except:
                font = ImageFont.load_default()
            
            # 텍스트 위치 (오른쪽 하단)
            text_width = draw.textlength(text, font=font)
            x = image.width - text_width - 10
            y = image.height - 30
            
            # 배경 박스
            draw.rectangle([x-5, y-5, x+text_width+5, y+20], 
                          fill=(0, 0, 0, 128))
            
            # 텍스트
            draw.text((x, y), text, fill=(255, 255, 255), font=font)
            
            return image
            
        except Exception as e:
            logger.warning(f"워터마크 추가 실패: {e}")
            return image
    
    def _basic_overlay(
        self, 
        person_image: Image.Image, 
        clothing_image: Image.Image
    ) -> Image.Image:
        """기본 오버레이 (백업용)"""
        result = person_image.copy()
        
        # 간단한 합성
        clothing_resized = clothing_image.resize((200, 200))
        result.paste(clothing_resized, (150, 100))
        
        # 텍스트 추가
        draw = ImageDraw.Draw(result)
        draw.text((10, 10), "Basic Demo Mode", fill='white')
        
        return result
    
    async def ai_fitting(
        self,
        person_image: Image.Image,
        clothing_image: Image.Image,
        height: float,
        weight: float,
        model_type: str
    ) -> Image.Image:
        """실제 AI 모델 피팅 (추후 구현)"""
        
        logger.info(f"🤖 AI 피팅 모델: {model_type}")
        
        if model_type == "ootd":
            return await self._ootd_fitting(person_image, clothing_image, height, weight)
        elif model_type == "viton":
            return await self._viton_fitting(person_image, clothing_image, height, weight)
        elif model_type == "acgpn":
            return await self._acgpn_fitting(person_image, clothing_image, height, weight)
        else:
            # 데모 모드로 대체
            return await self.demo_fitting(person_image, clothing_image, height, weight)
    
    async def _ootd_fitting(
        self,
        person_image: Image.Image,
        clothing_image: Image.Image,
        height: float,
        weight: float
    ) -> Image.Image:
        """OOTDiffusion 모델 사용 (미구현)"""
        
        # TODO: 실제 OOTDiffusion 모델 로직
        # 현재는 고급 데모로 대체
        logger.info("🔄 OOTDiffusion 모델 (데모 모드)")
        
        # 처리 시간 시뮬레이션
        await asyncio.sleep(3)
        
        return await self.demo_fitting(person_image, clothing_image, height, weight)
    
    async def _viton_fitting(
        self,
        person_image: Image.Image,
        clothing_image: Image.Image,
        height: float,
        weight: float
    ) -> Image.Image:
        """VITON-HD 모델 사용 (미구현)"""
        
        # TODO: 실제 VITON-HD 모델 로직
        logger.info("🔄 VITON-HD 모델 (데모 모드)")
        
        # 처리 시간 시뮬레이션
        await asyncio.sleep(2)
        
        return await self.demo_fitting(person_image, clothing_image, height, weight)
    
    async def _acgpn_fitting(
        self,
        person_image: Image.Image,
        clothing_image: Image.Image,
        height: float,
        weight: float
    ) -> Image.Image:
        """ACGPN 모델 사용 (미구현)"""
        
        # TODO: 실제 ACGPN 모델 로직
        logger.info("🔄 ACGPN 모델 (데모 모드)")
        
        # 처리 시간 시뮬레이션
        await asyncio.sleep(4)
        
        return await self.demo_fitting(person_image, clothing_image, height, weight)
    
    def get_model_status(self) -> Dict[str, Any]:
        """모델 상태 반환"""
        return {
            "models_loaded": self.models_loaded,
            "demo_mode": self.demo_mode,
            "device": self.device,
            "supported_models": self.supported_models,
            "current_config": self.model_config
        }