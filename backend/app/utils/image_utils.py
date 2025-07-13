"""
이미지 처리 유틸리티 함수들
"""
import io
from typing import Tuple
from PIL import Image, ImageEnhance, ImageFilter

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
