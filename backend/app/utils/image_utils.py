# app/utils/image_utils.py
"""
이미지 처리 유틸리티 함수들 - 완전한 구현
"""
import os
import io
import base64
import uuid
import tempfile
from typing import Tuple, Union, Optional
from pathlib import Path
import numpy as np
import cv2
from PIL import Image, ImageEnhance, ImageFilter, ImageDraw, ImageFont
import logging

logger = logging.getLogger(__name__)

def resize_image(image: Image.Image, target_size: Tuple[int, int], maintain_ratio: bool = True) -> Image.Image:
    """이미지 크기 조정"""
    if maintain_ratio:
        image.thumbnail(target_size, Image.Resampling.LANCZOS)
        
        # 정사각형으로 패딩
        new_image = Image.new('RGB', target_size, (255, 255, 255))
        paste_x = (target_size[0] - image.width) // 2
        paste_y = (target_size[1] - image.height) // 2
        new_image.paste(image, (paste_x, paste_y))
        return new_image
    else:
        return image.resize(target_size, Image.Resampling.LANCZOS)

def enhance_image_quality(image: Image.Image) -> Image.Image:
    """이미지 품질 향상"""
    # 선명도 향상
    enhancer = ImageEnhance.Sharpness(image)
    image = enhancer.enhance(1.1)
    
    # 대비 향상
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(1.05)
    
    return image

def convert_to_rgb(image: Image.Image) -> Image.Image:
    """RGB로 변환"""
    if image.mode != 'RGB':
        return image.convert('RGB')
    return image

async def validate_image_content(image_bytes: bytes) -> bool:
    """이미지 내용 검증"""
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

# ===== 누락된 함수들 추가 =====

def save_temp_image(image: Union[Image.Image, np.ndarray], 
                   prefix: str = "temp", 
                   suffix: str = ".jpg",
                   directory: Optional[str] = None) -> str:
    """임시 이미지 파일 저장"""
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
    """이미지 로드"""
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

def save_image(image: Union[Image.Image, np.ndarray], 
               filepath: str, 
               quality: int = 90) -> bool:
    """이미지 저장"""
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

def image_to_base64(image: Union[Image.Image, np.ndarray], 
                   format: str = "JPEG") -> str:
    """이미지를 base64 문자열로 변환"""
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
    """base64 문자열을 이미지로 변환"""
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
    """데모용 이미지 생성"""
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
    """이미지 정규화 (0-1 범위)"""
    if image.dtype == np.uint8:
        return image.astype(np.float32) / 255.0
    return image

def denormalize_image(image: np.ndarray) -> np.ndarray:
    """이미지 비정규화 (0-255 범위)"""
    if image.dtype == np.float32 or image.dtype == np.float64:
        return (image * 255).astype(np.uint8)
    return image

def crop_center(image: Image.Image, crop_size: Tuple[int, int]) -> Image.Image:
    """중앙 크롭"""
    width, height = image.size
    crop_width, crop_height = crop_size
    
    left = (width - crop_width) // 2
    top = (height - crop_height) // 2
    right = left + crop_width
    bottom = top + crop_height
    
    return image.crop((left, top, right, bottom))

def pad_image(image: Image.Image, target_size: Tuple[int, int], fill_color: Tuple[int, int, int] = (255, 255, 255)) -> Image.Image:
    """이미지 패딩"""
    width, height = image.size
    target_width, target_height = target_size
    
    if width == target_width and height == target_height:
        return image
    
    # 새 이미지 생성
    new_image = Image.new('RGB', target_size, fill_color)
    
    # 중앙에 배치
    paste_x = (target_width - width) // 2
    paste_y = (target_height - height) // 2
    new_image.paste(image, (paste_x, paste_y))
    
    return new_image

# OpenCV 관련 유틸리티
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

# 임시 파일 정리 함수
def cleanup_temp_files(directory: str, pattern: str = "temp_*.jpg"):
    """임시 파일들 정리"""
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

if __name__ == "__main__":
    # 테스트 코드
    print("🖼️ Image Utils 테스트")
    
    # 데모 이미지 생성 테스트
    demo_img = create_demo_image(512, 512, "MyCloset AI")
    temp_path = save_temp_image(demo_img, "test")
    print(f"데모 이미지 저장: {temp_path}")
    
    # 로드 테스트
    loaded_img = load_image(temp_path, (256, 256))
    print(f"이미지 로드 완료: {loaded_img.size}")
    
    # base64 변환 테스트
    b64_str = image_to_base64(loaded_img)
    print(f"base64 변환 완료: {len(b64_str)} 문자")
    
    # 정리
    os.remove(temp_path)
    print("✅ 모든 테스트 완료")