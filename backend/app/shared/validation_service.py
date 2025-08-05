# backend/app/shared/validation_service.py
"""
🔥 MyCloset AI Validation Service
================================================================================

검증 서비스를 위한 공통 모듈입니다.

- validate_upload_file: 업로드 파일 검증
- validate_measurements: 측정값 검증

Author: MyCloset AI Team
Date: 2025-08-01
Version: 1.0
"""

import logging
import io
from typing import Tuple, Dict, Any, Optional
from PIL import Image
from fastapi import UploadFile

logger = logging.getLogger(__name__)

# 허용된 이미지 형식
ALLOWED_IMAGE_FORMATS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff'}
ALLOWED_IMAGE_MIME_TYPES = {
    'image/jpeg', 'image/jpg', 'image/png', 'image/webp', 
    'image/bmp', 'image/tiff'
}

# 파일 크기 제한 (10MB)
MAX_FILE_SIZE = 10 * 1024 * 1024

# 이미지 크기 제한
MIN_IMAGE_SIZE = (100, 100)
MAX_IMAGE_SIZE = (4096, 4096)


async def validate_upload_file(
    file: UploadFile,
    file_type: str = "image"
) -> Tuple[bool, str, Optional[bytes]]:
    """업로드 파일 검증"""
    
    try:
        # 파일 존재 확인
        if not file:
            return False, "파일이 제공되지 않았습니다.", None
        
        # 파일명 확인
        if not file.filename:
            return False, "파일명이 없습니다.", None
        
        # 파일 크기 확인
        file_size = 0
        file_content = b""
        
        # 파일 내용 읽기
        while chunk := await file.read(8192):  # 8KB씩 읽기
            file_content += chunk
            file_size += len(chunk)
            
            # 파일 크기 제한 확인
            if file_size > MAX_FILE_SIZE:
                return False, f"파일 크기가 너무 큽니다. 최대 {MAX_FILE_SIZE // (1024*1024)}MB까지 허용됩니다.", None
        
        # 파일 포인터를 처음으로 되돌리기
        await file.seek(0)
        
        # 이미지 파일인 경우 추가 검증
        if file_type == "image":
            # MIME 타입 확인
            if file.content_type not in ALLOWED_IMAGE_MIME_TYPES:
                return False, f"지원되지 않는 이미지 형식입니다. 허용된 형식: {', '.join(ALLOWED_IMAGE_MIME_TYPES)}", None
            
            # 파일 확장자 확인
            file_extension = file.filename.lower().split('.')[-1] if '.' in file.filename else ''
            if f'.{file_extension}' not in ALLOWED_IMAGE_FORMATS:
                return False, f"지원되지 않는 파일 확장자입니다. 허용된 확장자: {', '.join(ALLOWED_IMAGE_FORMATS)}", None
            
            # 이미지 유효성 확인
            try:
                image = Image.open(io.BytesIO(file_content))
                image.verify()  # 이미지 무결성 확인
                
                # 이미지 크기 확인
                image = Image.open(io.BytesIO(file_content))  # 다시 열기
                width, height = image.size
                
                if width < MIN_IMAGE_SIZE[0] or height < MIN_IMAGE_SIZE[1]:
                    return False, f"이미지가 너무 작습니다. 최소 크기: {MIN_IMAGE_SIZE[0]}x{MIN_IMAGE_SIZE[1]}", None
                
                if width > MAX_IMAGE_SIZE[0] or height > MAX_IMAGE_SIZE[1]:
                    return False, f"이미지가 너무 큽니다. 최대 크기: {MAX_IMAGE_SIZE[0]}x{MAX_IMAGE_SIZE[1]}", None
                
                logger.info(f"✅ 이미지 검증 성공: {file.filename} ({width}x{height})")
                
            except Exception as e:
                return False, f"이미지 파일이 손상되었거나 읽을 수 없습니다: {str(e)}", None
        
        return True, "파일 검증 성공", file_content
        
    except Exception as e:
        logger.error(f"❌ 파일 검증 중 에러 발생: {e}")
        return False, f"파일 검증 중 에러가 발생했습니다: {str(e)}", None


def validate_measurements(
    height: float,
    weight: float,
    chest: Optional[float] = None,
    waist: Optional[float] = None,
    hips: Optional[float] = None
) -> Tuple[bool, str, Dict[str, Any]]:
    """신체 측정값 검증"""
    
    try:
        errors = []
        validated_data = {}
        
        # 키 검증 (140cm ~ 220cm)
        if not (140 <= height <= 220):
            errors.append(f"키는 140cm에서 220cm 사이여야 합니다. 현재: {height}cm")
        else:
            validated_data['height'] = height
        
        # 몸무게 검증 (40kg ~ 150kg)
        if not (40 <= weight <= 150):
            errors.append(f"몸무게는 40kg에서 150kg 사이여야 합니다. 현재: {weight}kg")
        else:
            validated_data['weight'] = weight
        
        # 가슴둘레 검증 (선택적, 0이 아닌 경우만)
        if chest is not None and chest > 0:
            if not (50 <= chest <= 150):
                errors.append(f"가슴둘레는 50cm에서 150cm 사이여야 합니다. 현재: {chest}cm")
            else:
                validated_data['chest'] = chest
        
        # 허리둘레 검증 (선택적, 0이 아닌 경우만)
        if waist is not None and waist > 0:
            if not (50 <= waist <= 150):
                errors.append(f"허리둘레는 50cm에서 150cm 사이여야 합니다. 현재: {waist}cm")
            else:
                validated_data['waist'] = waist
        
        # 엉덩이둘레 검증 (선택적, 0이 아닌 경우만)
        if hips is not None and hips > 0:
            if not (50 <= hips <= 150):
                errors.append(f"엉덩이둘레는 50cm에서 150cm 사이여야 합니다. 현재: {hips}cm")
            else:
                validated_data['hips'] = hips
        
        # BMI 계산
        if 'height' in validated_data and 'weight' in validated_data:
            height_m = validated_data['height'] / 100
            bmi = validated_data['weight'] / (height_m * height_m)
            validated_data['bmi'] = round(bmi, 2)
            
            # BMI 카테고리
            if bmi < 18.5:
                validated_data['bmi_category'] = "저체중"
            elif bmi < 25:
                validated_data['bmi_category'] = "정상"
            elif bmi < 30:
                validated_data['bmi_category'] = "과체중"
            else:
                validated_data['bmi_category'] = "비만"
        
        if errors:
            return False, "; ".join(errors), validated_data
        
        logger.info(f"✅ 측정값 검증 성공: 키={height}cm, 몸무게={weight}kg, BMI={validated_data.get('bmi', 'N/A')}")
        return True, "측정값 검증 성공", validated_data
        
    except Exception as e:
        logger.error(f"❌ 측정값 검증 중 에러 발생: {e}")
        return False, f"측정값 검증 중 에러가 발생했습니다: {str(e)}", {}


def validate_session_id(session_id: str) -> Tuple[bool, str]:
    """세션 ID 검증"""
    
    try:
        if not session_id:
            return False, "세션 ID가 제공되지 않았습니다."
        
        if not isinstance(session_id, str):
            return False, "세션 ID는 문자열이어야 합니다."
        
        if len(session_id) < 10:
            return False, "세션 ID가 너무 짧습니다."
        
        if len(session_id) > 100:
            return False, "세션 ID가 너무 깁니다."
        
        # 세션 ID 형식 확인 (session_타임스탬프_랜덤ID)
        if not session_id.startswith('session_'):
            return False, "세션 ID 형식이 올바르지 않습니다."
        
        return True, "세션 ID 검증 성공"
        
    except Exception as e:
        logger.error(f"❌ 세션 ID 검증 중 에러 발생: {e}")
        return False, f"세션 ID 검증 중 에러가 발생했습니다: {str(e)}"


def validate_step_parameters(
    step_id: int,
    parameters: Dict[str, Any]
) -> Tuple[bool, str, Dict[str, Any]]:
    """스텝 파라미터 검증"""
    
    try:
        errors = []
        validated_params = {}
        
        # 스텝 ID 검증
        if not (1 <= step_id <= 8):
            errors.append(f"스텝 ID는 1에서 8 사이여야 합니다. 현재: {step_id}")
        
        # 공통 파라미터 검증
        if 'confidence_threshold' in parameters:
            confidence = parameters['confidence_threshold']
            if not isinstance(confidence, (int, float)) or not (0.1 <= confidence <= 1.0):
                errors.append(f"신뢰도 임계값은 0.1에서 1.0 사이의 숫자여야 합니다. 현재: {confidence}")
            else:
                validated_params['confidence_threshold'] = confidence
        
        # 스텝별 파라미터 검증
        if step_id == 1:  # Human Parsing
            if 'enhance_quality' in parameters:
                if not isinstance(parameters['enhance_quality'], bool):
                    errors.append("enhance_quality는 boolean 값이어야 합니다.")
                else:
                    validated_params['enhance_quality'] = parameters['enhance_quality']
        
        elif step_id == 2:  # Pose Estimation
            if 'clothing_type' in parameters:
                clothing_type = parameters['clothing_type']
                valid_types = ['shirt', 'pants', 'dress', 'jacket', 'auto_detect']
                if clothing_type not in valid_types:
                    errors.append(f"clothing_type은 다음 중 하나여야 합니다: {', '.join(valid_types)}")
                else:
                    validated_params['clothing_type'] = clothing_type
        
        elif step_id == 3:  # Cloth Segmentation
            if 'analysis_detail' in parameters:
                detail = parameters['analysis_detail']
                valid_details = ['low', 'medium', 'high']
                if detail not in valid_details:
                    errors.append(f"analysis_detail은 다음 중 하나여야 합니다: {', '.join(valid_details)}")
                else:
                    validated_params['analysis_detail'] = detail
        
        if errors:
            return False, "; ".join(errors), validated_params
        
        return True, "스텝 파라미터 검증 성공", validated_params
        
    except Exception as e:
        logger.error(f"❌ 스텝 파라미터 검증 중 에러 발생: {e}")
        return False, f"스텝 파라미터 검증 중 에러가 발생했습니다: {str(e)}", {} 