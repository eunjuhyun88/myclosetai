# backend/app/services/image_processor.py
import cv2
import numpy as np
from PIL import Image, ImageOps, ImageEnhance
import torch
from typing import Dict, Any, Tuple, Optional
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

class ImageProcessor:
    def __init__(self):
        self.target_size = (512, 512)
        self.max_size = (1024, 1024)
        self.supported_formats = ['RGB', 'RGBA']
        
    async def process_person_image(self, image_path: str) -> Image.Image:
        """사용자 이미지 전처리"""
        try:
            logger.info(f"📸 사용자 이미지 처리 시작: {image_path}")
            
            # 1. 이미지 로드 및 기본 검증
            image = self._load_and_validate_image(image_path)
            
            # 2. 사이즈 조정
            image = self._resize_image(image, self.target_size)
            
            # 3. 색상 보정
            image = self._enhance_image(image)
            
            # 4. 노이즈 제거
            image = self._denoise_image(image)
            
            logger.info("✅ 사용자 이미지 처리 완료")
            return image
            
        except Exception as e:
            logger.error(f"❌ 사용자 이미지 처리 오류: {e}")
            raise
    
    async def process_clothing_image(self, image_path: str) -> Image.Image:
        """의류 이미지 전처리"""
        try:
            logger.info(f"👕 의류 이미지 처리 시작: {image_path}")
            
            # 1. 이미지 로드
            image = self._load_and_validate_image(image_path)
            
            # 2. 배경 제거 (간단한 버전)
            image = self._remove_background_simple(image)
            
            # 3. 사이즈 조정
            image = self._resize_image(image, self.target_size)
            
            # 4. 색상 정규화
            image = self._normalize_colors(image)
            
            logger.info("✅ 의류 이미지 처리 완료")
            return image
            
        except Exception as e:
            logger.error(f"❌ 의류 이미지 처리 오류: {e}")
            raise
    
    def _load_and_validate_image(self, image_path: str) -> Image.Image:
        """이미지 로드 및 검증"""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"이미지 파일을 찾을 수 없습니다: {image_path}")
        
        try:
            image = Image.open(image_path)
            
            # RGBA를 RGB로 변환
            if image.mode == 'RGBA':
                # 흰색 배경으로 변환
                background = Image.new('RGB', image.size, (255, 255, 255))
                background.paste(image, mask=image.split()[-1])
                image = background
            elif image.mode != 'RGB':
                image = image.convert('RGB')
            
            # 이미지 크기 확인
            width, height = image.size
            if width < 100 or height < 100:
                raise ValueError("이미지 크기가 너무 작습니다 (최소 100x100)")
            
            if width > 4000 or height > 4000:
                logger.warning("이미지 크기가 큽니다. 리사이즈가 필요합니다.")
            
            return image
            
        except Exception as e:
            raise ValueError(f"이미지 로드 실패: {e}")
    
    def _resize_image(self, image: Image.Image, target_size: Tuple[int, int]) -> Image.Image:
        """이미지 리사이즈 (비율 유지)"""
        # 비율을 유지하면서 리사이즈
        image.thumbnail(target_size, Image.Resampling.LANCZOS)
        
        # 정확한 크기로 패딩
        new_image = Image.new('RGB', target_size, (255, 255, 255))
        
        # 중앙에 배치
        x_offset = (target_size[0] - image.width) // 2
        y_offset = (target_size[1] - image.height) // 2
        new_image.paste(image, (x_offset, y_offset))
        
        return new_image
    
    def _enhance_image(self, image: Image.Image) -> Image.Image:
        """이미지 품질 개선"""
        try:
            # 밝기 조정
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(1.1)  # 10% 밝게
            
            # 대비 조정
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.1)  # 10% 대비 증가
            
            # 선명도 조정
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(1.05)  # 5% 선명하게
            
            return image
            
        except Exception as e:
            logger.warning(f"이미지 개선 실패: {e}")
            return image
    
    def _denoise_image(self, image: Image.Image) -> Image.Image:
        """노이즈 제거"""
        try:
            # PIL을 OpenCV로 변환
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # 가우시안 블러로 노이즈 제거
            denoised = cv2.GaussianBlur(cv_image, (3, 3), 0)
            
            # 다시 PIL로 변환
            denoised_pil = Image.fromarray(cv2.cvtColor(denoised, cv2.COLOR_BGR2RGB))
            
            return denoised_pil
            
        except Exception as e:
            logger.warning(f"노이즈 제거 실패: {e}")
            return image
    
    def _remove_background_simple(self, image: Image.Image) -> Image.Image:
        """간단한 배경 제거"""
        try:
            # OpenCV로 변환
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # 그레이스케일 변환
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            
            # 임계값 처리
            _, binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
            
            # 형태학적 연산으로 노이즈 제거
            kernel = np.ones((3, 3), np.uint8)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
            
            # 마스크 생성 (배경은 검은색, 객체는 흰색)
            mask = 255 - binary
            
            # 마스크 적용
            result = cv_image.copy()
            result[mask == 0] = [255, 255, 255]  # 배경을 흰색으로
            
            # PIL로 변환
            result_pil = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
            
            return result_pil
            
        except Exception as e:
            logger.warning(f"배경 제거 실패: {e}")
            return image
    
    def _normalize_colors(self, image: Image.Image) -> Image.Image:
        """색상 정규화"""
        try:
            # 색상 균형 조정
            enhancer = ImageEnhance.Color(image)
            image = enhancer.enhance(1.1)  # 채도 10% 증가
            
            return image
            
        except Exception as e:
            logger.warning(f"색상 정규화 실패: {e}")
            return image
    
    async def analyze_person(self, image_path: str) -> Dict[str, Any]:
        """사용자 이미지 분석"""
        try:
            image = Image.open(image_path)
            width, height = image.size
            
            # 간단한 분석 (실제로는 더 정교한 분석 필요)
            analysis = {
                "image_info": {
                    "width": width,
                    "height": height,
                    "aspect_ratio": round(width / height, 2),
                    "format": image.format or "Unknown"
                },
                "pose_detected": True,  # 실제로는 포즈 검출 알고리즘 필요
                "body_parts": {
                    "head": {"detected": True, "confidence": 0.95},
                    "torso": {"detected": True, "confidence": 0.92},
                    "arms": {"detected": True, "confidence": 0.88},
                    "legs": {"detected": True, "confidence": 0.85}
                },
                "quality_score": self._calculate_image_quality(image),
                "estimated_size": "M",  # 실제로는 체형 분석 필요
                "confidence": 0.9
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"사용자 이미지 분석 실패: {e}")
            return {
                "error": str(e),
                "pose_detected": False,
                "confidence": 0.0
            }
    
    async def analyze_clothing(self, image_path: str) -> Dict[str, Any]:
        """의류 이미지 분석"""
        try:
            image = Image.open(image_path)
            
            # 간단한 의류 분석
            analysis = {
                "category": self._detect_clothing_category(image),
                "style": "캐주얼",  # 실제로는 스타일 분류 알고리즘 필요
                "colors": self._extract_dominant_colors(image),
                "pattern": "단색",  # 실제로는 패턴 분석 필요
                "quality_score": self._calculate_image_quality(image),
                "background_removed": True,
                "size_info": {
                    "width": image.width,
                    "height": image.height
                }
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"의류 이미지 분석 실패: {e}")
            return {
                "error": str(e),
                "category": "unknown"
            }
    
    def _calculate_image_quality(self, image: Image.Image) -> float:
        """이미지 품질 점수 계산"""
        try:
            # 해상도 점수
            resolution_score = min(1.0, (image.width * image.height) / (512 * 512))
            
            # 밝기 분산 (선명도 추정)
            gray = image.convert('L')
            np_gray = np.array(gray)
            brightness_variance = np.var(np_gray) / (255 ** 2)
            sharpness_score = min(1.0, brightness_variance * 10)
            
            # 전체 품질 점수
            quality_score = (resolution_score * 0.4 + sharpness_score * 0.6)
            
            return round(quality_score, 2)
            
        except:
            return 0.5  # 기본값
    
    def _detect_clothing_category(self, image: Image.Image) -> str:
        """의류 카테고리 감지 (간단한 버전)"""
        # 실제로는 CNN 모델 필요
        width, height = image.size
        
        if height > width * 1.5:
            return "하의"  # 세로가 긴 경우
        elif width > height * 1.2:
            return "액세서리"  # 가로가 긴 경우
        else:
            return "상의"  # 기본값
    
    def _extract_dominant_colors(self, image: Image.Image) -> list:
        """주요 색상 추출"""
        try:
            # 이미지 크기 축소 (성능 향상)
            small_image = image.resize((50, 50))
            
            # 색상 히스토그램 분석
            colors = small_image.getcolors(maxcolors=256*256*256)
            
            if colors:
                # 가장 많이 사용된 색상들 추출
                sorted_colors = sorted(colors, reverse=True)
                dominant_colors = []
                
                for count, color in sorted_colors[:3]:  # 상위 3개 색상
                    if isinstance(color, tuple) and len(color) == 3:
                        color_name = self._rgb_to_color_name(color)
                        dominant_colors.append(color_name)
                
                return dominant_colors
            
        except:
            pass
        
        return ["알 수 없음"]
    
    def _rgb_to_color_name(self, rgb: Tuple[int, int, int]) -> str:
        """RGB 값을 색상 이름으로 변환"""
        r, g, b = rgb
        
        # 간단한 색상 분류
        if r > 200 and g > 200 and b > 200:
            return "흰색"
        elif r < 50 and g < 50 and b < 50:
            return "검은색"
        elif r > g and r > b:
            return "빨간색"
        elif g > r and g > b:
            return "초록색"
        elif b > r and b > g:
            return "파란색"
        elif r > 150 and g > 150:
            return "노란색"
        else:
            return "기타"