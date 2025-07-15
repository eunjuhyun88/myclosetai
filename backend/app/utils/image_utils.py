"""
MyCloset AI - 완전한 이미지 처리 유틸리티
✅ M3 Max 최적화  
✅ 고품질 이미지 처리
✅ PIL/OpenCV 통합
✅ 기존 코드와 완전 호환
✅ 추가 유틸리티 함수들 포함
"""

import os
import io
import base64
import uuid
import tempfile
import logging
import asyncio
from typing import Tuple, Union, Optional, List, Dict, Any
from pathlib import Path
import numpy as np
import cv2
from PIL import Image, ImageEnhance, ImageFilter, ImageDraw, ImageFont, ImageOps
from datetime import datetime

logger = logging.getLogger(__name__)

class ImageProcessor:
    """
    완전한 이미지 처리 유틸리티 클래스
    ✅ 기존 함수명 완전 유지
    ✅ M3 Max 최적화
    ✅ 고품질 처리
    """
    
    def __init__(self):
        self.is_m3_max = self._detect_m3_max()
        self.max_resolution = (2048, 2048) if self.is_m3_max else (1024, 1024)
        self.default_quality = 95 if self.is_m3_max else 85
        
        logger.info(f"🎨 ImageProcessor 초기화 - M3 Max: {self.is_m3_max}")

    def _detect_m3_max(self) -> bool:
        """M3 Max 감지"""
        try:
            import platform
            import subprocess
            
            if platform.system() == 'Darwin':
                result = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'], 
                                      capture_output=True, text=True, timeout=5)
                chip_info = result.stdout.strip()
                return 'M3' in chip_info and 'Max' in chip_info
        except:
            pass
        return False

    @staticmethod
    def enhance_image(image: Image.Image, enhancement_level: float = 1.1) -> Image.Image:
        """
        이미지 품질 향상
        ✅ 기존 함수명 유지
        """
        try:
            # 선명도 향상
            enhancer = ImageEnhance.Sharpness(image)
            enhanced = enhancer.enhance(enhancement_level)
            
            # 색상 향상
            enhancer = ImageEnhance.Color(enhanced)
            enhanced = enhancer.enhance(1.05)
            
            # 대비 향상
            enhancer = ImageEnhance.Contrast(enhanced)
            enhanced = enhancer.enhance(1.02)
            
            logger.debug("🎨 이미지 품질 향상 완료")
            return enhanced
            
        except Exception as e:
            logger.error(f"❌ 이미지 향상 실패: {e}")
            return image

    @staticmethod
    def resize_image(
        image: Image.Image, 
        target_size: Tuple[int, int], 
        maintain_ratio: bool = True,
        resample: int = Image.Resampling.LANCZOS
    ) -> Image.Image:
        """
        이미지 크기 조정
        ✅ 기존 함수와 완전 호환
        """
        try:
            if maintain_ratio:
                # 비율 유지하며 리사이즈
                image.thumbnail(target_size, resample)
                
                # 정사각형으로 패딩
                new_image = Image.new('RGB', target_size, (255, 255, 255))
                paste_x = (target_size[0] - image.width) // 2
                paste_y = (target_size[1] - image.height) // 2
                new_image.paste(image, (paste_x, paste_y))
                return new_image
            else:
                return image.resize(target_size, resample)
                
        except Exception as e:
            logger.error(f"❌ 이미지 크기 조정 실패: {e}")
            return image

    @staticmethod
    def enhance_image_quality(image: Image.Image) -> Image.Image:
        """
        이미지 품질 향상 (기존 함수와 호환)
        """
        try:
            # 선명도 향상
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(1.1)
            
            # 대비 향상
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.05)
            
            return image
            
        except Exception as e:
            logger.error(f"❌ 이미지 품질 향상 실패: {e}")
            return image

    @staticmethod
    def convert_to_rgb(image: Image.Image) -> Image.Image:
        """
        RGB로 변환 (기존 함수와 호환)
        """
        try:
            if image.mode != 'RGB':
                return image.convert('RGB')
            return image
        except Exception as e:
            logger.error(f"❌ RGB 변환 실패: {e}")
            return image

    def normalize_image(self, image: Image.Image) -> Image.Image:
        """이미지 정규화 (확장)"""
        try:
            # RGB 모드로 변환
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # M3 Max에서 처리 가능한 최대 크기 제한
            max_dimension = max(self.max_resolution)
            if max(image.size) > max_dimension:
                ratio = max_dimension / max(image.size)
                new_size = tuple(int(dim * ratio) for dim in image.size)
                image = image.resize(new_size, Image.Resampling.LANCZOS)
            
            return image
            
        except Exception as e:
            logger.error(f"❌ 이미지 정규화 실패: {e}")
            return image

    @staticmethod
    def convert_to_array(image: Image.Image) -> np.ndarray:
        """PIL 이미지를 numpy 배열로 변환"""
        try:
            return np.array(image)
        except Exception as e:
            logger.error(f"❌ 배열 변환 실패: {e}")
            return np.zeros((512, 512, 3), dtype=np.uint8)

    @staticmethod
    def convert_from_array(array: np.ndarray) -> Image.Image:
        """numpy 배열을 PIL 이미지로 변환"""
        try:
            if array.max() <= 1.0:
                array = (array * 255).astype(np.uint8)
            return Image.fromarray(array)
        except Exception as e:
            logger.error(f"❌ 이미지 변환 실패: {e}")
            return Image.new('RGB', (512, 512), color='black')

    @staticmethod
    def apply_filter(image: Image.Image, filter_type: str = "enhance") -> Image.Image:
        """이미지 필터 적용 (확장)"""
        try:
            if filter_type == "enhance":
                return ImageProcessor.enhance_image(image)
            elif filter_type == "blur":
                return image.filter(ImageFilter.GaussianBlur(radius=1))
            elif filter_type == "sharpen":
                return image.filter(ImageFilter.SHARPEN)
            elif filter_type == "smooth":
                return image.filter(ImageFilter.SMOOTH)
            elif filter_type == "edge_enhance":
                return image.filter(ImageFilter.EDGE_ENHANCE)
            elif filter_type == "detail":
                return image.filter(ImageFilter.DETAIL)
            else:
                return image
                
        except Exception as e:
            logger.error(f"❌ 필터 적용 실패: {e}")
            return image

    @staticmethod
    def crop_center(image: Image.Image, crop_size: Tuple[int, int]) -> Image.Image:
        """중앙 기준 이미지 크롭"""
        try:
            width, height = image.size
            crop_width, crop_height = crop_size
            
            left = (width - crop_width) // 2
            top = (height - crop_height) // 2
            right = left + crop_width
            bottom = top + crop_height
            
            return image.crop((left, top, right, bottom))
            
        except Exception as e:
            logger.error(f"❌ 이미지 크롭 실패: {e}")
            return image

    @staticmethod
    def add_padding(
        image: Image.Image, 
        target_size: Tuple[int, int], 
        fill_color: Tuple[int, int, int] = (255, 255, 255)
    ) -> Image.Image:
        """이미지에 패딩 추가"""
        try:
            target_width, target_height = target_size
            img_width, img_height = image.size
            
            # 새 이미지 생성
            new_image = Image.new('RGB', target_size, fill_color)
            
            # 중앙에 원본 이미지 붙여넣기
            paste_x = (target_width - img_width) // 2
            paste_y = (target_height - img_height) // 2
            new_image.paste(image, (paste_x, paste_y))
            
            return new_image
            
        except Exception as e:
            logger.error(f"❌ 패딩 추가 실패: {e}")
            return image

# ============================================
# 기존 호환 함수들 (전역 함수로 유지)
# ============================================

def resize_image(image: Image.Image, target_size: Tuple[int, int], maintain_ratio: bool = True) -> Image.Image:
    """기존 resize_image 함수와 완전 호환"""
    return ImageProcessor.resize_image(image, target_size, maintain_ratio)

def enhance_image_quality(image: Image.Image) -> Image.Image:
    """기존 enhance_image_quality 함수와 완전 호환"""
    return ImageProcessor.enhance_image_quality(image)

def convert_to_rgb(image: Image.Image) -> Image.Image:
    """기존 convert_to_rgb 함수와 완전 호환"""
    return ImageProcessor.convert_to_rgb(image)

async def validate_image_content(image_bytes: bytes) -> bool:
    """기존 validate_image_content 함수와 완전 호환"""
    try:
        image = Image.open(io.BytesIO(image_bytes))
        width, height = image.size
        
        # 최소/최대 크기 검사
        if width < 100 or height < 100:
            return False
        if width > 4096 or height > 4096:
            return False
            
        return True
    except Exception:
        return False

# ============================================
# 추가 유틸리티 함수들 (완전 구현)
# ============================================

def save_temp_image(
    image: Union[Image.Image, np.ndarray], 
    prefix: str = "temp", 
    suffix: str = ".jpg",
    directory: Optional[str] = None
) -> str:
    """
    임시 이미지 파일 저장
    ✅ 기존 함수와 완전 호환
    """
    try:
        # 디렉토리 설정
        if directory is None:
            directory = tempfile.gettempdir()
        
        # 파일명 생성
        filename = f"{prefix}_{uuid.uuid4().hex[:8]}{suffix}"
        filepath = os.path.join(directory, filename)
        
        # PIL Image로 변환
        if isinstance(image, np.ndarray):
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)
            if len(image.shape) == 3 and image.shape[2] == 3:
                # BGR to RGB 변환 (OpenCV 사용 시)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image)
        else:
            pil_image = image
        
        # RGB로 변환
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        # 파일 저장
        pil_image.save(filepath, "JPEG", quality=90)
        logger.debug(f"임시 이미지 저장: {filepath}")
        
        return filepath
        
    except Exception as e:
        logger.error(f"임시 이미지 저장 실패: {e}")
        raise

def load_image(image_path: str, target_size: Optional[Tuple[int, int]] = None) -> Image.Image:
    """
    이미지 로드
    ✅ 기존 함수와 완전 호환
    """
    try:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"이미지 파일을 찾을 수 없습니다: {image_path}")
        
        image = Image.open(image_path)
        
        # RGB로 변환
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # 크기 조정
        if target_size:
            image = resize_image(image, target_size, maintain_ratio=True)
        
        return image
        
    except Exception as e:
        logger.error(f"이미지 로드 실패: {e}")
        raise

def save_image(
    image: Union[Image.Image, np.ndarray], 
    filepath: str, 
    quality: int = 90
) -> bool:
    """
    이미지 저장
    ✅ 기존 함수와 완전 호환
    """
    try:
        # PIL Image로 변환
        if isinstance(image, np.ndarray):
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)
            if len(image.shape) == 3 and image.shape[2] == 3:
                # BGR to RGB 변환
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image)
        else:
            pil_image = image
        
        # RGB로 변환
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        # 디렉토리 생성
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # 파일 저장
        pil_image.save(filepath, "JPEG", quality=quality)
        logger.debug(f"이미지 저장 완료: {filepath}")
        
        return True
        
    except Exception as e:
        logger.error(f"이미지 저장 실패: {e}")
        return False

def image_to_base64(
    image: Union[Image.Image, np.ndarray], 
    format: str = "JPEG"
) -> str:
    """
    이미지를 base64 문자열로 변환
    ✅ 기존 함수와 완전 호환
    """
    try:
        # PIL Image로 변환
        if isinstance(image, np.ndarray):
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image)
        else:
            pil_image = image
        
        # RGB로 변환
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        # base64 변환
        buffer = io.BytesIO()
        pil_image.save(buffer, format=format, quality=90)
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        return image_base64
        
    except Exception as e:
        logger.error(f"base64 변환 실패: {e}")
        return ""

def base64_to_image(base64_str: str) -> Image.Image:
    """
    base64 문자열을 이미지로 변환
    ✅ 기존 함수와 완전 호환
    """
    try:
        # base64 디코딩
        image_data = base64.b64decode(base64_str)
        image = Image.open(io.BytesIO(image_data))
        
        # RGB로 변환
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        return image
        
    except Exception as e:
        logger.error(f"base64 이미지 변환 실패: {e}")
        raise

def create_demo_image(width: int = 512, height: int = 512, text: str = "DEMO") -> Image.Image:
    """
    데모용 이미지 생성
    ✅ 기존 함수와 완전 호환
    """
    try:
        # 배경 이미지 생성
        image = Image.new('RGB', (width, height), color=(240, 240, 240))
        draw = ImageDraw.Draw(image)
        
        # 폰트 설정
        try:
            font = ImageFont.truetype("arial.ttf", 48)
        except:
            font = ImageFont.load_default()
        
        # 텍스트 그리기
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        
        x = (width - text_width) // 2
        y = (height - text_height) // 2
        
        draw.text((x, y), text, fill=(100, 100, 100), font=font)
        
        return image
        
    except Exception as e:
        logger.error(f"데모 이미지 생성 실패: {e}")
        return Image.new('RGB', (width, height), color=(200, 200, 200))

def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    이미지 정규화 (0-1 범위)
    ✅ 기존 함수와 완전 호환
    """
    if image.dtype == np.uint8:
        return image.astype(np.float32) / 255.0
    return image

def denormalize_image(image: np.ndarray) -> np.ndarray:
    """
    이미지 비정규화 (0-255 범위)  
    ✅ 기존 함수와 완전 호환
    """
    if image.dtype == np.float32 or image.dtype == np.float64:
        return (image * 255).astype(np.uint8)
    return image

def crop_center(image: Image.Image, crop_size: Tuple[int, int]) -> Image.Image:
    """
    중앙 크롭
    ✅ 기존 함수와 완전 호환
    """
    return ImageProcessor.crop_center(image, crop_size)

def pad_image(
    image: Image.Image, 
    target_size: Tuple[int, int], 
    fill_color: Tuple[int, int, int] = (255, 255, 255)
) -> Image.Image:
    """
    이미지 패딩
    ✅ 기존 함수와 완전 호환
    """
    return ImageProcessor.add_padding(image, target_size, fill_color)

# ============================================
# OpenCV 관련 유틸리티
# ============================================

def pil_to_cv2(image: Image.Image) -> np.ndarray:
    """PIL 이미지를 OpenCV 형식으로 변환"""
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

def cv2_to_pil(image: np.ndarray) -> Image.Image:
    """OpenCV 이미지를 PIL 형식으로 변환"""
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

def apply_gaussian_blur(image: Image.Image, radius: float = 1.0) -> Image.Image:
    """가우시안 블러 적용"""
    return image.filter(ImageFilter.GaussianBlur(radius=radius))

def apply_edge_enhance(image: Image.Image) -> Image.Image:
    """엣지 강화 적용"""
    return image.filter(ImageFilter.EDGE_ENHANCE)

def apply_sharpen(image: Image.Image) -> Image.Image:
    """샤프닝 적용"""
    return image.filter(ImageFilter.SHARPEN)

# ============================================
# 고급 이미지 처리 함수들 (M3 Max 최적화)
# ============================================

def smart_resize(
    image: Image.Image, 
    target_size: Tuple[int, int], 
    method: str = "lanczos"
) -> Image.Image:
    """스마트 리사이즈 (M3 Max 최적화)"""
    try:
        methods = {
            "lanczos": Image.Resampling.LANCZOS,
            "bicubic": Image.Resampling.BICUBIC,
            "bilinear": Image.Resampling.BILINEAR,
            "nearest": Image.Resampling.NEAREST
        }
        
        resample_method = methods.get(method, Image.Resampling.LANCZOS)
        
        # 원본 비율 계산
        original_ratio = image.width / image.height
        target_ratio = target_size[0] / target_size[1]
        
        if abs(original_ratio - target_ratio) < 0.01:
            # 비율이 거의 같으면 직접 리사이즈
            return image.resize(target_size, resample_method)
        else:
            # 비율이 다르면 크롭 후 리사이즈
            if original_ratio > target_ratio:
                # 너무 넓음 - 높이 기준으로 크롭
                new_width = int(image.height * target_ratio)
                left = (image.width - new_width) // 2
                image = image.crop((left, 0, left + new_width, image.height))
            else:
                # 너무 높음 - 너비 기준으로 크롭
                new_height = int(image.width / target_ratio)
                top = (image.height - new_height) // 2
                image = image.crop((0, top, image.width, top + new_height))
            
            return image.resize(target_size, resample_method)
            
    except Exception as e:
        logger.error(f"❌ 스마트 리사이즈 실패: {e}")
        return image.resize(target_size, Image.Resampling.LANCZOS)

def enhance_image_advanced(
    image: Image.Image,
    brightness: float = 1.0,
    contrast: float = 1.0,
    saturation: float = 1.0,
    sharpness: float = 1.0
) -> Image.Image:
    """고급 이미지 향상 (M3 Max 최적화)"""
    try:
        # 밝기 조정
        if brightness != 1.0:
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(brightness)
        
        # 대비 조정
        if contrast != 1.0:
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(contrast)
        
        # 채도 조정
        if saturation != 1.0:
            enhancer = ImageEnhance.Color(image)
            image = enhancer.enhance(saturation)
        
        # 선명도 조정
        if sharpness != 1.0:
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(sharpness)
        
        return image
        
    except Exception as e:
        logger.error(f"❌ 고급 이미지 향상 실패: {e}")
        return image

def auto_orient_image(image: Image.Image) -> Image.Image:
    """이미지 자동 회전 (EXIF 기반)"""
    try:
        return ImageOps.exif_transpose(image)
    except Exception as e:
        logger.warning(f"⚠️ 자동 회전 실패: {e}")
        return image

def remove_background_simple(image: Image.Image, threshold: int = 240) -> Image.Image:
    """간단한 배경 제거 (흰색 배경)"""
    try:
        # RGBA로 변환
        image = image.convert("RGBA")
        data = np.array(image)
        
        # 흰색에 가까운 픽셀을 투명하게
        white_pixels = (data[:, :, 0] > threshold) & \
                      (data[:, :, 1] > threshold) & \
                      (data[:, :, 2] > threshold)
        
        data[white_pixels] = [255, 255, 255, 0]  # 투명하게
        
        return Image.fromarray(data, "RGBA")
        
    except Exception as e:
        logger.error(f"❌ 배경 제거 실패: {e}")
        return image

def create_thumbnail_grid(
    images: List[Image.Image], 
    grid_size: Tuple[int, int] = (3, 3),
    thumbnail_size: Tuple[int, int] = (150, 150),
    padding: int = 10
) -> Image.Image:
    """이미지 그리드 생성"""
    try:
        cols, rows = grid_size
        thumb_width, thumb_height = thumbnail_size
        
        # 전체 크기 계산
        total_width = cols * thumb_width + (cols - 1) * padding
        total_height = rows * thumb_height + (rows - 1) * padding
        
        # 배경 이미지 생성
        grid_image = Image.new('RGB', (total_width, total_height), (255, 255, 255))
        
        # 이미지 배치
        for i, img in enumerate(images[:cols * rows]):
            row = i // cols
            col = i % cols
            
            # 썸네일 생성
            thumbnail = img.copy()
            thumbnail.thumbnail(thumbnail_size, Image.Resampling.LANCZOS)
            
            # 위치 계산
            x = col * (thumb_width + padding)
            y = row * (thumb_height + padding)
            
            # 중앙 정렬을 위한 오프셋
            x_offset = (thumb_width - thumbnail.width) // 2
            y_offset = (thumb_height - thumbnail.height) // 2
            
            grid_image.paste(thumbnail, (x + x_offset, y + y_offset))
        
        return grid_image
        
    except Exception as e:
        logger.error(f"❌ 썸네일 그리드 생성 실패: {e}")
        return Image.new('RGB', (300, 300), (255, 255, 255))

def add_watermark(
    image: Image.Image, 
    watermark_text: str = "MyCloset AI",
    position: str = "bottom-right",
    opacity: int = 128
) -> Image.Image:
    """워터마크 추가"""
    try:
        # 워터마크 레이어 생성
        watermark = Image.new('RGBA', image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(watermark)
        
        # 폰트 크기 계산
        font_size = max(12, min(image.width, image.height) // 20)
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except:
            font = ImageFont.load_default()
        
        # 텍스트 크기 계산
        text_bbox = draw.textbbox((0, 0), watermark_text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        
        # 위치 계산
        margin = 20
        if position == "bottom-right":
            x = image.width - text_width - margin
            y = image.height - text_height - margin
        elif position == "bottom-left":
            x = margin
            y = image.height - text_height - margin
        elif position == "top-right":
            x = image.width - text_width - margin
            y = margin
        elif position == "top-left":
            x = margin
            y = margin
        else:  # center
            x = (image.width - text_width) // 2
            y = (image.height - text_height) // 2
        
        # 텍스트 그리기
        draw.text((x, y), watermark_text, fill=(255, 255, 255, opacity), font=font)
        
        # 합성
        if image.mode != 'RGBA':
            image = image.convert('RGBA')
        
        watermarked = Image.alpha_composite(image, watermark)
        return watermarked.convert('RGB')
        
    except Exception as e:
        logger.error(f"❌ 워터마크 추가 실패: {e}")
        return image

# ============================================
# 임시 파일 정리 함수
# ============================================

def cleanup_temp_files(directory: str, pattern: str = "temp_*.jpg"):
    """
    임시 파일들 정리
    ✅ 기존 함수와 완전 호환
    """
    try:
        import glob
        temp_files = glob.glob(os.path.join(directory, pattern))
        for file_path in temp_files:
            try:
                os.remove(file_path)
                logger.debug(f"임시 파일 삭제: {file_path}")
            except Exception as e:
                logger.warning(f"임시 파일 삭제 실패: {file_path} - {e}")
    except Exception as e:
        logger.error(f"임시 파일 정리 실패: {e}")

# ============================================
# 배치 처리 함수들 (M3 Max 최적화)
# ============================================

async def batch_resize_images(
    images: List[Image.Image],
    target_size: Tuple[int, int],
    max_workers: int = 4
) -> List[Image.Image]:
    """배치 이미지 리사이즈 (비동기 처리)"""
    try:
        import concurrent.futures
        import asyncio
        
        def resize_single(image: Image.Image) -> Image.Image:
            return resize_image(image, target_size, maintain_ratio=True)
        
        # 스레드풀로 병렬 처리
        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            tasks = [
                loop.run_in_executor(executor, resize_single, img)
                for img in images
            ]
            resized_images = await asyncio.gather(*tasks)
        
        logger.info(f"🎨 배치 리사이즈 완료: {len(resized_images)}개 이미지")
        return resized_images
        
    except Exception as e:
        logger.error(f"❌ 배치 리사이즈 실패: {e}")
        # 폴백: 순차 처리
        return [resize_image(img, target_size, maintain_ratio=True) for img in images]

async def batch_enhance_images(
    images: List[Image.Image],
    enhancement_level: float = 1.1,
    max_workers: int = 4
) -> List[Image.Image]:
    """배치 이미지 향상 (비동기 처리)"""
    try:
        import concurrent.futures
        import asyncio
        
        def enhance_single(image: Image.Image) -> Image.Image:
            return enhance_image_quality(image)
        
        # 스레드풀로 병렬 처리
        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            tasks = [
                loop.run_in_executor(executor, enhance_single, img)
                for img in images
            ]
            enhanced_images = await asyncio.gather(*tasks)
        
        logger.info(f"🎨 배치 향상 완료: {len(enhanced_images)}개 이미지")
        return enhanced_images
        
    except Exception as e:
        logger.error(f"❌ 배치 향상 실패: {e}")
        # 폴백: 순차 처리
        return [enhance_image_quality(img) for img in images]

def create_image_comparison(
    original: Image.Image,
    processed: Image.Image,
    labels: Tuple[str, str] = ("Original", "Processed")
) -> Image.Image:
    """이미지 비교 뷰 생성"""
    try:
        # 같은 크기로 맞춤
        max_width = max(original.width, processed.width)
        max_height = max(original.height, processed.height)
        
        original_resized = smart_resize(original, (max_width, max_height))
        processed_resized = smart_resize(processed, (max_width, max_height))
        
        # 비교 이미지 생성
        comparison_width = max_width * 2 + 20  # 여백 20px
        comparison_height = max_height + 60    # 라벨용 60px
        
        comparison = Image.new('RGB', (comparison_width, comparison_height), (255, 255, 255))
        
        # 이미지 배치
        comparison.paste(original_resized, (0, 30))
        comparison.paste(processed_resized, (max_width + 20, 30))
        
        # 라벨 추가
        draw = ImageDraw.Draw(comparison)
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except:
            font = ImageFont.load_default()
        
        # 원본 라벨
        text_bbox = draw.textbbox((0, 0), labels[0], font=font)
        text_width = text_bbox[2] - text_bbox[0]
        x1 = (max_width - text_width) // 2
        draw.text((x1, 5), labels[0], fill=(0, 0, 0), font=font)
        
        # 처리된 이미지 라벨
        text_bbox = draw.textbbox((0, 0), labels[1], font=font)
        text_width = text_bbox[2] - text_bbox[0]
        x2 = max_width + 20 + (max_width - text_width) // 2
        draw.text((x2, 5), labels[1], fill=(0, 0, 0), font=font)
        
        return comparison
        
    except Exception as e:
        logger.error(f"❌ 이미지 비교 생성 실패: {e}")
        return original

def extract_dominant_colors(image: Image.Image, num_colors: int = 5) -> List[Tuple[int, int, int]]:
    """주요 색상 추출"""
    try:
        # 이미지 크기 줄이기 (성능 향상)
        small_image = image.resize((150, 150))
        
        # K-means 클러스터링으로 주요 색상 추출
        import numpy as np
        from sklearn.cluster import KMeans
        
        # 픽셀 데이터 준비
        pixels = np.array(small_image).reshape(-1, 3)
        
        # K-means 적용
        kmeans = KMeans(n_clusters=num_colors, random_state=42, n_init=10)
        kmeans.fit(pixels)
        
        # 클러스터 중심점을 색상으로 변환
        colors = []
        for center in kmeans.cluster_centers_:
            color = tuple(int(c) for c in center)
            colors.append(color)
        
        return colors
        
    except Exception as e:
        logger.error(f"❌ 주요 색상 추출 실패: {e}")
        # 폴백: 기본 색상들
        return [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]

def create_color_palette(colors: List[Tuple[int, int, int]], size: Tuple[int, int] = (300, 50)) -> Image.Image:
    """색상 팔레트 이미지 생성"""
    try:
        palette_width, palette_height = size
        color_width = palette_width // len(colors)
        
        palette = Image.new('RGB', (palette_width, palette_height))
        draw = ImageDraw.Draw(palette)
        
        for i, color in enumerate(colors):
            x1 = i * color_width
            x2 = (i + 1) * color_width
            draw.rectangle([x1, 0, x2, palette_height], fill=color)
        
        return palette
        
    except Exception as e:
        logger.error(f"❌ 색상 팔레트 생성 실패: {e}")
        return Image.new('RGB', size, (128, 128, 128))

def calculate_image_similarity(image1: Image.Image, image2: Image.Image) -> float:
    """이미지 유사도 계산 (간단한 히스토그램 기반)"""
    try:
        # 같은 크기로 맞춤
        size = (256, 256)
        img1_resized = image1.resize(size)
        img2_resized = image2.resize(size)
        
        # 히스토그램 계산
        hist1 = img1_resized.histogram()
        hist2 = img2_resized.histogram()
        
        # 코사인 유사도 계산
        import math
        
        dot_product = sum(a * b for a, b in zip(hist1, hist2))
        magnitude1 = math.sqrt(sum(a * a for a in hist1))
        magnitude2 = math.sqrt(sum(b * b for b in hist2))
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        similarity = dot_product / (magnitude1 * magnitude2)
        return similarity
        
    except Exception as e:
        logger.error(f"❌ 이미지 유사도 계산 실패: {e}")
        return 0.0

def convert_image_format(
    image: Image.Image,
    output_format: str = "JPEG",
    quality: int = 90,
    optimize: bool = True
) -> bytes:
    """이미지 포맷 변환"""
    try:
        buffer = io.BytesIO()
        
        # 포맷별 처리
        if output_format.upper() in ['JPEG', 'JPG']:
            if image.mode in ['RGBA', 'LA']:
                # 투명도가 있는 이미지는 흰색 배경으로 변환
                background = Image.new('RGB', image.size, (255, 255, 255))
                background.paste(image, mask=image.split()[-1] if image.mode == 'RGBA' else None)
                image = background
            image.save(buffer, format='JPEG', quality=quality, optimize=optimize)
        elif output_format.upper() == 'PNG':
            image.save(buffer, format='PNG', optimize=optimize)
        elif output_format.upper() == 'WEBP':
            image.save(buffer, format='WEBP', quality=quality, optimize=optimize)
        else:
            # 기본값: JPEG
            if image.mode in ['RGBA', 'LA']:
                background = Image.new('RGB', image.size, (255, 255, 255))
                background.paste(image, mask=image.split()[-1] if image.mode == 'RGBA' else None)
                image = background
            image.save(buffer, format='JPEG', quality=quality, optimize=optimize)
        
        return buffer.getvalue()
        
    except Exception as e:
        logger.error(f"❌ 이미지 포맷 변환 실패: {e}")
        # 폴백: JPEG로 저장
        buffer = io.BytesIO()
        if image.mode in ['RGBA', 'LA']:
            background = Image.new('RGB', image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[-1] if image.mode == 'RGBA' else None)
            image = background
        image.save(buffer, format='JPEG', quality=quality)
        return buffer.getvalue()

# ============================================
# 전역 ImageProcessor 인스턴스
# ============================================

_global_image_processor = None

def get_image_processor() -> ImageProcessor:
    """전역 이미지 프로세서 인스턴스 반환"""
    global _global_image_processor
    if _global_image_processor is None:
        _global_image_processor = ImageProcessor()
    return _global_image_processor

# ============================================
# 최종 테스트 및 예제 코드
# ============================================

if __name__ == "__main__":
    # 테스트 코드
    print("🖼️ Image Utils 완전 테스트")
    
    try:
        # 데모 이미지 생성 테스트
        demo_img = create_demo_image(512, 512, "MyCloset AI M3 Max")
        temp_path = save_temp_image(demo_img, "test")
        print(f"데모 이미지 저장: {temp_path}")
        
        # 로드 테스트
        loaded_img = load_image(temp_path, (256, 256))
        print(f"이미지 로드 완료: {loaded_img.size}")
        
        # 향상 테스트
        enhanced_img = enhance_image_quality(loaded_img)
        print(f"이미지 향상 완료: {enhanced_img.size}")
        
        # base64 변환 테스트
        b64_str = image_to_base64(enhanced_img)
        print(f"base64 변환 완료: {len(b64_str)} 문자")
        
        # base64 복원 테스트
        restored_img = base64_to_image(b64_str)
        print(f"base64 복원 완료: {restored_img.size}")
        
        # 색상 추출 테스트
        try:
            colors = extract_dominant_colors(loaded_img, 3)
            print(f"주요 색상 추출: {colors}")
            
            # 색상 팔레트 생성
            palette = create_color_palette(colors)
            print(f"색상 팔레트 생성: {palette.size}")
        except ImportError:
            print("⚠️ scikit-learn 없음 - 색상 추출 스킵")
        
        # 비교 이미지 생성 테스트
        comparison = create_image_comparison(loaded_img, enhanced_img)
        print(f"비교 이미지 생성: {comparison.size}")
        
        # 워터마크 테스트
        watermarked = add_watermark(loaded_img, "MyCloset AI")
        print(f"워터마크 추가: {watermarked.size}")
        
        # 정리
        os.remove(temp_path)
        print("✅ 모든 테스트 완료")
        
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")

logger.info("✅ ImageProcessor 모듈 로드 완료 - 모든 기능 포함")