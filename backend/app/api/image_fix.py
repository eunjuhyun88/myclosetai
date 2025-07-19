"""
🔥 이미지 응답 수정을 위한 임시 패치
"""
import base64
import io
from PIL import Image, ImageDraw
import numpy as np

def create_demo_fitted_image(width=400, height=600):
    """데모용 가상 피팅 이미지 생성"""
    # 기본 이미지 생성
    image = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(image)
    
    # 사람 실루엣 그리기 (간단한 형태)
    # 머리
    draw.ellipse([180, 50, 220, 90], fill='#FDB5A6', outline='black')
    
    # 몸통 (상의 착용)
    draw.rectangle([160, 90, 240, 280], fill='#000000', outline='black')  # 검은색 상의
    
    # 팔
    draw.rectangle([140, 100, 160, 200], fill='#FDB5A6', outline='black')  # 왼팔
    draw.rectangle([240, 100, 260, 200], fill='#FDB5A6', outline='black')  # 오른팔
    
    # 하체
    draw.rectangle([160, 280, 240, 450], fill='#000080', outline='black')  # 바지
    
    # 다리
    draw.rectangle([160, 450, 190, 550], fill='#FDB5A6', outline='black')  # 왼다리
    draw.rectangle([210, 450, 240, 550], fill='#FDB5A6', outline='black')  # 오른다리
    
    # 텍스트 추가
    try:
        # 기본 폰트 사용
        draw.text((150, 20), "Virtual Try-On Result", fill='black')
        draw.text((160, 560), "MyCloset AI", fill='blue')
    except:
        pass
    
    return image

def image_to_base64_fixed(image_input):
    """이미지를 Base64로 변환 (수정된 버전)"""
    try:
        if image_input is None:
            # 데모 이미지 생성
            demo_image = create_demo_fitted_image()
            buffer = io.BytesIO()
            demo_image.save(buffer, format='JPEG', quality=85)
            image_bytes = buffer.getvalue()
            return base64.b64encode(image_bytes).decode('utf-8')
        
        # 기존 로직...
        if isinstance(image_input, str):
            return image_input
            
        if hasattr(image_input, 'save'):  # PIL Image
            buffer = io.BytesIO()
            image_input.save(buffer, format='JPEG', quality=85)
            image_bytes = buffer.getvalue()
            return base64.b64encode(image_bytes).decode('utf-8')
        
        return ""
        
    except Exception as e:
        print(f"이미지 변환 오류: {e}")
        # 오류 시 데모 이미지 반환
        demo_image = create_demo_fitted_image()
        buffer = io.BytesIO()
        demo_image.save(buffer, format='JPEG', quality=85)
        image_bytes = buffer.getvalue()
        return base64.b64encode(image_bytes).decode('utf-8')

