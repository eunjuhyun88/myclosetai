from fastapi import UploadFile
from PIL import Image
import io
from typing import List

ALLOWED_EXTENSIONS = ["jpg", "jpeg", "png", "webp", "bmp"]
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB

def validate_image(file: UploadFile) -> bool:
    """이미지 파일 검증"""
    
    # 파일 확장자 검사
    if not file.filename:
        return False
        
    extension = file.filename.split(".")[-1].lower()
    if extension not in ALLOWED_EXTENSIONS:
        return False
    
    # MIME 타입 검사
    if not file.content_type or not file.content_type.startswith("image/"):
        return False
    
    return True

def validate_measurements(height: float, weight: float) -> bool:
    """신체 측정값 검증"""
    
    # 합리적인 범위 검사
    if not (100 <= height <= 250):  # cm
        return False
        
    if not (30 <= weight <= 300):   # kg
        return False
    
    return True

async def validate_image_content(image_bytes: bytes) -> bool:
    """이미지 내용 검증"""
    try:
        image = Image.open(io.BytesIO(image_bytes))
        
        # 이미지 크기 검사
        width, height = image.size
        if width < 100 or height < 100:
            return False
            
        if width > 4096 or height > 4096:
            return False
        
        return True
    except:
        return False
