#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🖼️ MyCloset AI - 이미지 처리 모듈 (완전한 Python 구조)
===========================================================
✅ 올바른 Python 구조로 완전 재구성
✅ 함수들을 논리적 순서로 배치
✅ 모든 기능 완전 구현 (잘린 부분 없음)
✅ conda 환경 & M3 Max 최적화
✅ 순환참조 완전 방지
✅ 타입 힌팅 및 문서화 완료

Author: MyCloset AI Team
Date: 2025-07-21
Version: 2.0 (Complete Restructure)
"""

# =============================================================================
# 🔥 1. 표준 라이브러리 임포트
# =============================================================================
import io
import os
import sys
import logging
import base64
import tempfile
import uuid
import math
from typing import Dict, Any, Optional, Tuple, List, Union, Callable
from pathlib import Path
from abc import ABC, abstractmethod

# =============================================================================
# 🔥 2. 서드파티 라이브러리 조건부 임포트 (안전한 처리)
# =============================================================================

# 로거 설정 (최우선)
logger = logging.getLogger(__name__)

# NumPy 안전 임포트
try:
    import numpy as np
    NUMPY_AVAILABLE = True
    logger.info("✅ NumPy 사용 가능")
    
    # NumPy 2.x 호환성 체크
    numpy_version = tuple(map(int, np.__version__.split('.')[:2]))
    if numpy_version >= (2, 0):
        logger.warning(f"⚠️ NumPy {np.__version__} 감지 - 1.x 권장")
        logger.warning("🔧 해결방법: conda install numpy=1.24.3 -y --force-reinstall")
        
except ImportError as e:
    NUMPY_AVAILABLE = False
    np = None
    logger.warning(f"⚠️ NumPy 없음: {e}")

# PIL/Pillow 안전 임포트
try:
    from PIL import Image, ImageEnhance, ImageFilter, ImageDraw, ImageFont
    PIL_AVAILABLE = True
    logger.info("✅ PIL/Pillow 사용 가능")
except ImportError as e:
    PIL_AVAILABLE = False
    Image = None
    logger.warning(f"⚠️ PIL/Pillow 없음: {e}")

# OpenCV 안전 임포트
try:
    import cv2
    CV2_AVAILABLE = True
    logger.info("✅ OpenCV 사용 가능")
except ImportError as e:
    CV2_AVAILABLE = False
    cv2 = None
    logger.warning(f"⚠️ OpenCV 없음: {e}")

# PyTorch 안전 임포트
try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
    logger.info("✅ PyTorch 사용 가능")
    
    # M3 Max MPS 체크
    if hasattr(torch, 'mps') and torch.mps.is_available():
        logger.info("🚀 M3 Max MPS 가속 사용 가능")
except ImportError as e:
    TORCH_AVAILABLE = False
    torch = None
    F = None
    logger.warning(f"⚠️ PyTorch 없음: {e}")

# =============================================================================
# 🔥 3. 상수 및 설정
# =============================================================================

# 기본 설정값
DEFAULT_IMAGE_SIZE = (512, 512)
DEFAULT_DEVICE = "mps" if TORCH_AVAILABLE and hasattr(torch, 'mps') and torch.mps.is_available() else "cpu"
DEFAULT_DTYPE = torch.float16 if TORCH_AVAILABLE else None

# 지원되는 이미지 포맷
SUPPORTED_FORMATS = ['JPEG', 'PNG', 'WEBP', 'BMP', 'TIFF']
SUPPORTED_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff']

# 메모리 최적화 설정
MAX_IMAGE_SIZE = (2048, 2048)  # M3 Max 기준 최대 권장 크기
MEMORY_THRESHOLD_MB = 500  # 메모리 임계값

# =============================================================================
# 🔥 4. 기본 헬퍼 함수들 (가장 기초적인 것들)
# =============================================================================

def _ensure_numpy() -> bool:
    """NumPy 가용성 확인"""
    if not NUMPY_AVAILABLE:
        logger.error("❌ NumPy가 필요합니다. conda install numpy=1.24.3")
        return False
    return True

def _ensure_pil() -> bool:
    """PIL 가용성 확인"""
    if not PIL_AVAILABLE:
        logger.error("❌ PIL/Pillow가 필요합니다. conda install pillow")
        return False
    return True

def _ensure_torch() -> bool:
    """PyTorch 가용성 확인"""
    if not TORCH_AVAILABLE:
        logger.error("❌ PyTorch가 필요합니다. conda install pytorch")
        return False
    return True

def _get_optimal_device() -> str:
    """최적 디바이스 자동 선택"""
    if TORCH_AVAILABLE:
        if hasattr(torch, 'mps') and torch.mps.is_available():
            return "mps"  # M3 Max
        elif torch.cuda.is_available():
            return "cuda"
    return "cpu"

def _validate_image_input(image: Any) -> bool:
    """이미지 입력 검증"""
    if image is None:
        return False
    
    # PIL Image
    if hasattr(image, 'size') and hasattr(image, 'mode'):
        return True
    
    # NumPy array
    if NUMPY_AVAILABLE and isinstance(image, np.ndarray):
        return len(image.shape) >= 2
    
    # PyTorch tensor
    if TORCH_AVAILABLE and torch.is_tensor(image):
        return len(image.shape) >= 2
    
    # 파일 경로
    if isinstance(image, (str, Path)):
        return Path(image).exists()
    
    return False

# =============================================================================
# 🔥 5. 핵심 이미지 처리 함수들 (올바른 순서)
# =============================================================================

def load_image(filepath: Union[str, Path], target_format: str = "RGB") -> Optional[Any]:
    """
    이미지 파일 로드
    
    Args:
        filepath: 이미지 파일 경로
        target_format: 타겟 포맷 ('RGB', 'RGBA', 'L')
    
    Returns:
        PIL Image 또는 None
    """
    try:
        if not _ensure_pil():
            return None
        
        filepath = Path(filepath)
        if not filepath.exists():
            logger.error(f"❌ 파일이 존재하지 않음: {filepath}")
            return None
        
        # 확장자 체크
        if filepath.suffix.lower() not in SUPPORTED_EXTENSIONS:
            logger.warning(f"⚠️ 지원하지 않는 확장자: {filepath.suffix}")
        
        # 이미지 로드
        image = Image.open(filepath)
        
        # 포맷 변환
        if target_format and image.mode != target_format:
            if target_format == 'RGB' and image.mode == 'RGBA':
                # 투명도가 있는 경우 흰색 배경으로 합성
                background = Image.new('RGB', image.size, (255, 255, 255))
                background.paste(image, mask=image.split()[-1] if image.mode == 'RGBA' else None)
                image = background
            else:
                image = image.convert(target_format)
        
        logger.debug(f"✅ 이미지 로드 완료: {filepath} ({image.size}, {image.mode})")
        return image
        
    except Exception as e:
        logger.error(f"❌ 이미지 로드 실패: {e}")
        return None

def save_image(image: Any, filepath: Union[str, Path], 
              format: str = None, quality: int = 95, **kwargs) -> bool:
    """
    이미지 파일로 저장
    
    Args:
        image: 저장할 이미지 (PIL, numpy, tensor)
        filepath: 저장 경로
        format: 저장 포맷 (자동 감지 가능)
        quality: JPEG 품질 (1-100)
        **kwargs: 추가 저장 옵션
    
    Returns:
        저장 성공 여부
    """
    try:
        if not _ensure_pil():
            return False
        
        filepath = Path(filepath)
        
        # PIL Image로 변환
        pil_image = convert_to_pil(image)
        if pil_image is None:
            logger.error(f"❌ PIL 이미지 변환 실패: {type(image)}")
            return False
        
        # 포맷 자동 감지
        if format is None:
            format = filepath.suffix.upper().lstrip('.')
            if format == 'JPG':
                format = 'JPEG'
        
        # RGB 모드 확인 (JPEG는 투명도 지원 안함)
        if format.upper() == 'JPEG' and pil_image.mode in ['RGBA', 'LA']:
            background = Image.new('RGB', pil_image.size, (255, 255, 255))
            if pil_image.mode == 'RGBA':
                background.paste(pil_image, mask=pil_image.split()[-1])
            else:
                background.paste(pil_image)
            pil_image = background
        
        # 디렉토리 생성
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # 저장 옵션 설정
        save_kwargs = {'format': format, **kwargs}
        if format.upper() in ['JPEG', 'WEBP']:
            save_kwargs['quality'] = quality
            save_kwargs['optimize'] = True
        
        # 저장 실행
        pil_image.save(filepath, **save_kwargs)
        
        logger.debug(f"✅ 이미지 저장 완료: {filepath} ({format})")
        return True
        
    except Exception as e:
        logger.error(f"❌ 이미지 저장 실패: {e}")
        return False

def convert_to_pil(image: Any) -> Optional[Any]:
    """
    다양한 이미지 타입을 PIL Image로 변환
    
    Args:
        image: 변환할 이미지 (numpy, tensor, PIL 등)
    
    Returns:
        PIL Image 또는 None
    """
    try:
        if not _ensure_pil():
            return None
        
        # 이미 PIL Image인 경우
        if hasattr(image, 'save') and hasattr(image, 'size'):
            return image
        
        # NumPy array인 경우
        if NUMPY_AVAILABLE and isinstance(image, np.ndarray):
            # 데이터 타입 정규화
            if image.dtype != np.uint8:
                if image.max() <= 1.0:
                    image = (image * 255).astype(np.uint8)
                else:
                    image = np.clip(image, 0, 255).astype(np.uint8)
            
            # 차원에 따른 처리
            if len(image.shape) == 3:
                if image.shape[2] == 3:
                    return Image.fromarray(image, 'RGB')
                elif image.shape[2] == 4:
                    return Image.fromarray(image, 'RGBA')
                elif image.shape[2] == 1:
                    return Image.fromarray(image.squeeze(), 'L')
            elif len(image.shape) == 2:
                return Image.fromarray(image, 'L')
        
        # PyTorch tensor인 경우
        if TORCH_AVAILABLE and torch.is_tensor(image):
            return tensor_to_pil(image)
        
        # 파일 경로인 경우
        if isinstance(image, (str, Path)):
            return load_image(image)
        
        logger.warning(f"⚠️ 지원하지 않는 이미지 타입: {type(image)}")
        return None
        
    except Exception as e:
        logger.error(f"❌ PIL 변환 실패: {e}")
        return None

def convert_to_numpy(image: Any) -> Optional[np.ndarray]:
    """
    다양한 이미지 타입을 NumPy array로 변환
    
    Args:
        image: 변환할 이미지
    
    Returns:
        NumPy array 또는 None
    """
    try:
        if not _ensure_numpy():
            return None
        
        # 이미 NumPy array인 경우
        if isinstance(image, np.ndarray):
            return image
        
        # PIL Image인 경우
        if hasattr(image, 'size') and hasattr(image, 'mode'):
            return np.array(image)
        
        # PyTorch tensor인 경우
        if TORCH_AVAILABLE and torch.is_tensor(image):
            # CPU로 이동 후 numpy 변환
            if image.is_cuda or (hasattr(image, 'is_mps') and image.is_mps):
                image = image.cpu()
            
            # 차원 조정
            if image.dim() == 4:  # (N, C, H, W)
                image = image.squeeze(0)
            if image.dim() == 3:  # (C, H, W)
                image = image.permute(1, 2, 0)  # (H, W, C)
            
            # detach 후 numpy 변환
            if hasattr(image, 'detach'):
                image = image.detach()
            
            return image.numpy()
        
        # 파일 경로인 경우
        if isinstance(image, (str, Path)):
            pil_image = load_image(image)
            if pil_image:
                return np.array(pil_image)
        
        logger.warning(f"⚠️ 지원하지 않는 이미지 타입: {type(image)}")
        return None
        
    except Exception as e:
        logger.error(f"❌ NumPy 변환 실패: {e}")
        return None

def tensor_to_pil(tensor: torch.Tensor) -> Optional[Any]:
    """
    PyTorch tensor를 PIL Image로 변환
    
    Args:
        tensor: PyTorch tensor (C,H,W) 또는 (N,C,H,W)
    
    Returns:
        PIL Image 또는 None
    """
    try:
        if not _ensure_torch() or not _ensure_pil():
            return None
        
        if not torch.is_tensor(tensor):
            logger.error("❌ PyTorch tensor가 아닙니다")
            return None
        
        # 차원 조정
        if tensor.dim() == 4:  # (N, C, H, W)
            tensor = tensor.squeeze(0)
            
        if tensor.dim() == 3:  # (C, H, W)
            tensor = tensor.permute(1, 2, 0)  # (H, W, C)
        
        # CPU로 이동
        if tensor.is_cuda or (hasattr(tensor, 'is_mps') and tensor.is_mps):
            tensor = tensor.cpu()
        
        # detach 및 numpy 변환
        if hasattr(tensor, 'detach'):
            tensor = tensor.detach()
        
        array = tensor.numpy()
        
        # 값 범위 조정 (0-1 → 0-255)
        if array.dtype != np.uint8:
            if array.max() <= 1.0:
                array = (array * 255).astype(np.uint8)
            else:
                array = np.clip(array, 0, 255).astype(np.uint8)
        
        # PIL Image 생성
        if len(array.shape) == 3:
            if array.shape[2] == 3:
                return Image.fromarray(array, 'RGB')
            elif array.shape[2] == 1:
                return Image.fromarray(array.squeeze(), 'L')
            elif array.shape[2] == 4:
                return Image.fromarray(array, 'RGBA')
        elif len(array.shape) == 2:
            return Image.fromarray(array, 'L')
        
        logger.error(f"❌ 지원하지 않는 텐서 형태: {array.shape}")
        return None
        
    except Exception as e:
        logger.error(f"❌ tensor→PIL 변환 실패: {e}")
        return None

def pil_to_tensor(image: Any, device: str = None, normalize: bool = True) -> Optional[torch.Tensor]:
    """
    PIL Image를 PyTorch tensor로 변환
    
    Args:
        image: PIL Image 또는 변환 가능한 이미지
        device: 타겟 디바이스
        normalize: 0-1 범위로 정규화 여부
    
    Returns:
        PyTorch tensor (N,C,H,W) 또는 None
    """
    try:
        if not _ensure_torch():
            return None
        
        if device is None:
            device = _get_optimal_device()
        
        # PIL Image로 변환
        pil_image = convert_to_pil(image)
        if pil_image is None:
            return None
        
        # NumPy array로 변환
        array = np.array(pil_image).astype(np.float32)
        
        # 정규화
        if normalize and array.max() > 1.0:
            array = array / 255.0
        
        # 차원 조정
        if len(array.shape) == 2:  # 그레이스케일 (H, W)
            array = np.expand_dims(array, axis=-1)  # (H, W, 1)
        
        if len(array.shape) == 3:  # (H, W, C)
            # (H, W, C) → (C, H, W) → (1, C, H, W)
            tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
        else:
            logger.error(f"❌ 지원하지 않는 배열 형태: {array.shape}")
            return None
        
        # 디바이스로 이동
        tensor = tensor.to(device)
        
        logger.debug(f"✅ PIL→tensor 변환 완료: {tensor.shape}, device: {device}")
        return tensor
        
    except Exception as e:
        logger.error(f"❌ PIL→tensor 변환 실패: {e}")
        return None

# =============================================================================
# 🔥 6. 이미지 크기 및 형태 조정 함수들
# =============================================================================

def resize_image(image: Any, target_size: Tuple[int, int], 
                keep_aspect_ratio: bool = True, 
                resample_method: str = "LANCZOS") -> Optional[Any]:
    """
    이미지 크기 조정
    
    Args:
        image: 입력 이미지
        target_size: 타겟 크기 (width, height)
        keep_aspect_ratio: 비율 유지 여부
        resample_method: 리샘플링 방법
    
    Returns:
        크기 조정된 이미지
    """
    try:
        pil_image = convert_to_pil(image)
        if pil_image is None:
            return None
        
        original_size = pil_image.size
        
        # 비율 유지하면서 크기 조정
        if keep_aspect_ratio:
            # 타겟 크기에 맞게 비율 계산
            scale = min(target_size[0] / original_size[0], 
                       target_size[1] / original_size[1])
            
            new_size = (int(original_size[0] * scale), 
                       int(original_size[1] * scale))
            
            # 크기 조정
            resample = getattr(Image.Resampling, resample_method, Image.Lanczos)
            resized = pil_image.resize(new_size, resample)
            
            # 패딩 추가 (중앙 배치)
            if new_size != target_size:
                # 새 이미지 생성 (검은색 배경)
                padded = Image.new(pil_image.mode, target_size, (0, 0, 0))
                
                # 중앙에 배치
                offset = ((target_size[0] - new_size[0]) // 2,
                         (target_size[1] - new_size[1]) // 2)
                padded.paste(resized, offset)
                
                return padded
            else:
                return resized
        else:
            # 비율 무시하고 크기 조정
            resample = getattr(Image.Resampling, resample_method, Image.Lanczos)
            return pil_image.resize(target_size, resample)
        
    except Exception as e:
        logger.error(f"❌ 이미지 크기 조정 실패: {e}")
        return image

def crop_image(image: Any, bbox: Tuple[int, int, int, int]) -> Optional[Any]:
    """
    이미지 크롭
    
    Args:
        image: 입력 이미지
        bbox: 크롭 영역 (left, top, right, bottom)
    
    Returns:
        크롭된 이미지
    """
    try:
        pil_image = convert_to_pil(image)
        if pil_image is None:
            return None
        
        # 크롭 영역 검증
        width, height = pil_image.size
        left, top, right, bottom = bbox
        
        left = max(0, min(left, width))
        top = max(0, min(top, height))
        right = max(left, min(right, width))
        bottom = max(top, min(bottom, height))
        
        # 크롭 실행
        cropped = pil_image.crop((left, top, right, bottom))
        
        logger.debug(f"✅ 이미지 크롭 완료: {bbox} → {cropped.size}")
        return cropped
        
    except Exception as e:
        logger.error(f"❌ 이미지 크롭 실패: {e}")
        return image

def pad_image(image: Any, padding: Union[int, Tuple[int, int, int, int]], 
             fill_color: Tuple[int, int, int] = (0, 0, 0)) -> Optional[Any]:
    """
    이미지 패딩 추가
    
    Args:
        image: 입력 이미지
        padding: 패딩 크기 (전체) 또는 (left, top, right, bottom)
        fill_color: 패딩 색상
    
    Returns:
        패딩이 추가된 이미지
    """
    try:
        pil_image = convert_to_pil(image)
        if pil_image is None:
            return None
        
        # 패딩 값 정규화
        if isinstance(padding, int):
            padding = (padding, padding, padding, padding)
        elif len(padding) == 2:
            padding = (padding[0], padding[1], padding[0], padding[1])
        
        left, top, right, bottom = padding
        
        # 새 크기 계산
        old_width, old_height = pil_image.size
        new_width = old_width + left + right
        new_height = old_height + top + bottom
        
        # 새 이미지 생성
        padded = Image.new(pil_image.mode, (new_width, new_height), fill_color)
        
        # 원본 이미지 붙이기
        padded.paste(pil_image, (left, top))
        
        logger.debug(f"✅ 이미지 패딩 완료: {padding}")
        return padded
        
    except Exception as e:
        logger.error(f"❌ 이미지 패딩 실패: {e}")
        return image

# =============================================================================
# 🔥 7. 이미지 정규화 및 전처리 함수들
# =============================================================================

def normalize_image(image: Any, mean: Tuple[float, float, float] = (0.485, 0.456, 0.406), 
                   std: Tuple[float, float, float] = (0.229, 0.224, 0.225)) -> Optional[Any]:
    """
    이미지 정규화 (ImageNet 표준)
    
    Args:
        image: 입력 이미지
        mean: 평균값 (채널별)
        std: 표준편차 (채널별)
    
    Returns:
        정규화된 이미지
    """
    try:
        if TORCH_AVAILABLE and torch.is_tensor(image):
            # PyTorch tensor 정규화
            normalized = image.clone().float()
            
            if normalized.dim() == 4:  # (N, C, H, W)
                for i in range(min(3, normalized.shape[1])):
                    normalized[:, i, :, :] = (normalized[:, i, :, :] - mean[i]) / std[i]
            elif normalized.dim() == 3:  # (C, H, W)
                for i in range(min(3, normalized.shape[0])):
                    normalized[i, :, :] = (normalized[i, :, :] - mean[i]) / std[i]
            
            return normalized
            
        elif NUMPY_AVAILABLE:
            # NumPy array 정규화
            array = convert_to_numpy(image)
            if array is None:
                return None
            
            normalized = array.astype(np.float32) / 255.0  # 0-1 정규화
            
            # ImageNet 정규화 적용
            if len(normalized.shape) == 3 and normalized.shape[2] >= 3:
                for i in range(3):
                    normalized[:, :, i] = (normalized[:, :, i] - mean[i]) / std[i]
            
            return normalized
        
        logger.warning("⚠️ 정규화를 위해 PyTorch 또는 NumPy가 필요합니다")
        return image
        
    except Exception as e:
        logger.error(f"❌ 이미지 정규화 실패: {e}")
        return image

def denormalize_image(image: Any, mean: Tuple[float, float, float] = (0.485, 0.456, 0.406), 
                     std: Tuple[float, float, float] = (0.229, 0.224, 0.225)) -> Optional[Any]:
    """
    이미지 역정규화
    
    Args:
        image: 정규화된 이미지
        mean: 원래 평균값
        std: 원래 표준편차
    
    Returns:
        역정규화된 이미지
    """
    try:
        if TORCH_AVAILABLE and torch.is_tensor(image):
            # PyTorch tensor 역정규화
            denormalized = image.clone().float()
            
            if denormalized.dim() == 4:  # (N, C, H, W)
                for i in range(min(3, denormalized.shape[1])):
                    denormalized[:, i, :, :] = denormalized[:, i, :, :] * std[i] + mean[i]
            elif denormalized.dim() == 3:  # (C, H, W)
                for i in range(min(3, denormalized.shape[0])):
                    denormalized[i, :, :] = denormalized[i, :, :] * std[i] + mean[i]
            
            # 0-1 범위로 클리핑
            denormalized = torch.clamp(denormalized, 0, 1)
            return denormalized
            
        elif NUMPY_AVAILABLE:
            # NumPy array 역정규화
            array = convert_to_numpy(image)
            if array is None:
                return None
            
            denormalized = array.copy().astype(np.float32)
            
            # ImageNet 역정규화 적용
            if len(denormalized.shape) == 3 and denormalized.shape[2] >= 3:
                for i in range(3):
                    denormalized[:, :, i] = denormalized[:, :, i] * std[i] + mean[i]
            
            # 0-1 범위로 클리핑 후 0-255로 변환
            denormalized = np.clip(denormalized, 0, 1) * 255
            return denormalized.astype(np.uint8)
        
        logger.warning("⚠️ 역정규화를 위해 PyTorch 또는 NumPy가 필요합니다")
        return image
        
    except Exception as e:
        logger.error(f"❌ 이미지 역정규화 실패: {e}")
        return image

def preprocess_image(image: Any, target_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE,
                    device: str = None, normalize: bool = True, 
                    to_tensor: bool = True) -> Optional[Any]:
    """
    통합 이미지 전처리 함수
    
    Args:
        image: 입력 이미지 (파일경로, PIL, numpy, tensor)
        target_size: 타겟 크기 (width, height)
        device: 디바이스 ("mps", "cuda", "cpu")
        normalize: 정규화 여부
        to_tensor: tensor로 변환 여부
    
    Returns:
        전처리된 이미지
    """
    try:
        logger.debug(f"이미지 전처리 시작: {type(image)} → {target_size}")
        
        if device is None:
            device = _get_optimal_device()
        
        # 1. 이미지 로드/변환
        if isinstance(image, (str, Path)):
            pil_image = load_image(image)
        else:
            pil_image = convert_to_pil(image)
        
        if pil_image is None:
            logger.error("❌ 이미지 로드/변환 실패")
            return None
        
        # 2. 크기 조정
        if pil_image.size != target_size:
            pil_image = resize_image(pil_image, target_size, keep_aspect_ratio=True)
        
        # 3. RGB 변환
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        # 4. tensor 변환 여부에 따른 처리
        if to_tensor:
            if not _ensure_torch():
                logger.warning("⚠️ PyTorch 없음, NumPy array 반환")
                array = np.array(pil_image).astype(np.float32)
                if normalize:
                    array = array / 255.0
                return array
            
            # tensor로 변환
            tensor = pil_to_tensor(pil_image, device, normalize)
            if tensor is None:
                logger.error("❌ tensor 변환 실패")
                return None
            
            logger.debug(f"✅ 전처리 완료: {tensor.shape}, device: {device}")
            return tensor
        else:
            # PIL 또는 numpy로 반환
            if normalize:
                array = np.array(pil_image).astype(np.float32) / 255.0
                return array
            else:
                return pil_image
                
    except Exception as e:
        logger.error(f"❌ 이미지 전처리 실패: {e}")
        return None

# =============================================================================
# 🔥 8. Base64 변환 함수들
# =============================================================================

def image_to_base64(image: Any, format: str = "JPEG", quality: int = 95) -> str:
    """
    이미지를 Base64 문자열로 변환
    
    Args:
        image: 입력 이미지
        format: 저장 포맷
        quality: 압축 품질
    
    Returns:
        Base64 문자열
    """
    try:
        if not _ensure_pil():
            return ""
        
        pil_image = convert_to_pil(image)
        if pil_image is None:
            logger.error("❌ PIL 변환 실패")
            return ""
        
        # RGB 변환 (JPEG 호환성)
        if format.upper() == 'JPEG' and pil_image.mode in ['RGBA', 'LA']:
            background = Image.new('RGB', pil_image.size, (255, 255, 255))
            if pil_image.mode == 'RGBA':
                background.paste(pil_image, mask=pil_image.split()[-1])
            else:
                background.paste(pil_image)
            pil_image = background
        
        # Base64 변환
        buffer = io.BytesIO()
        save_kwargs = {'format': format}
        if format.upper() in ['JPEG', 'WEBP']:
            save_kwargs['quality'] = quality
            save_kwargs['optimize'] = True
        
        pil_image.save(buffer, **save_kwargs)
        img_str = base64.b64encode(buffer.getvalue()).decode()
        
        logger.debug(f"✅ Base64 변환 완료: {len(img_str)} 문자")
        return img_str
        
    except Exception as e:
        logger.error(f"❌ Base64 변환 실패: {e}")
        return ""

def base64_to_image(base64_str: str) -> Optional[Any]:
    """
    Base64 문자열을 이미지로 변환
    
    Args:
        base64_str: Base64 문자열
    
    Returns:
        PIL Image 또는 None
    """
    try:
        if not _ensure_pil():
            return None
        
        # Base64 디코딩
        img_data = base64.b64decode(base64_str)
        img_buffer = io.BytesIO(img_data)
        pil_image = Image.open(img_buffer)
        
        # RGB 변환
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        logger.debug(f"✅ Base64→이미지 변환 완료: {pil_image.size}")
        return pil_image
        
    except Exception as e:
        logger.error(f"❌ Base64→이미지 변환 실패: {e}")
        return None

# =============================================================================
# 🔥 9. 이미지 후처리 함수들
# =============================================================================

def postprocess_segmentation(output: Any, threshold: float = 0.5) -> Optional[Any]:
    """
    세그멘테이션 결과 후처리
    
    Args:
        output: 모델 출력 (확률 맵)
        threshold: 이진화 임계값
    
    Returns:
        이진 마스크 (0-255)
    """
    try:
        logger.debug(f"세그멘테이션 후처리 시작: {type(output)}")
        
        # PyTorch tensor 처리
        if TORCH_AVAILABLE and torch.is_tensor(output):
            # CPU로 이동
            if output.is_cuda or (hasattr(output, 'is_mps') and output.is_mps):
                output = output.cpu()
            
            # numpy 변환
            if hasattr(output, 'detach'):
                output_np = output.detach().numpy()
            else:
                output_np = output.numpy()
        else:
            output_np = convert_to_numpy(output)
        
        if output_np is None:
            logger.error("❌ NumPy 변환 실패")
            return None
        
        # 차원 조정
        if output_np.ndim == 4:  # (N, C, H, W)
            output_np = output_np.squeeze(0)
        
        if output_np.ndim == 3:  # (C, H, W)
            if output_np.shape[0] == 1:
                output_np = output_np.squeeze(0)
            else:
                # 다중 클래스인 경우 argmax
                output_np = np.argmax(output_np, axis=0)
                # 배경(0) 제외한 영역을 1로 설정
                output_np = (output_np > 0).astype(np.float32)
        
        # 이진화
        if output_np.dtype != np.uint8:
            binary_mask = (output_np > threshold).astype(np.uint8) * 255
        else:
            binary_mask = (output_np > int(threshold * 255)).astype(np.uint8) * 255
        
        logger.debug(f"✅ 후처리 완료: {binary_mask.shape}, 범위: {binary_mask.min()}-{binary_mask.max()}")
        return binary_mask
        
    except Exception as e:
        logger.error(f"❌ 세그멘테이션 후처리 실패: {e}")
        return None

def postprocess_pose_keypoints(output: Any, confidence_threshold: float = 0.3) -> Dict[str, Any]:
    """
    포즈 키포인트 후처리
    
    Args:
        output: 히트맵 출력 (C, H, W)
        confidence_threshold: 신뢰도 임계값
    
    Returns:
        키포인트 정보 딕셔너리
    """
    try:
        result = {
            "keypoints": [],
            "connections": [],
            "valid_keypoints": 0,
            "confidence_scores": []
        }
        
        if not _ensure_numpy():
            return result
        
        # NumPy로 변환
        if TORCH_AVAILABLE and torch.is_tensor(output):
            if output.is_cuda or (hasattr(output, 'is_mps') and output.is_mps):
                output = output.cpu()
            heatmaps = output.detach().numpy()
        else:
            heatmaps = convert_to_numpy(output)
        
        if heatmaps is None:
            return result
        
        # 차원 조정
        if heatmaps.ndim == 4:  # (N, C, H, W)
            heatmaps = heatmaps.squeeze(0)
        
        num_keypoints = min(heatmaps.shape[0], 18)  # COCO 18개 키포인트
        height, width = heatmaps.shape[1], heatmaps.shape[2]
        
        # 각 키포인트 위치 찾기
        keypoints = []
        confidence_scores = []
        
        for i in range(num_keypoints):
            heatmap = heatmaps[i]
            
            # 최대값 위치 찾기
            max_val = np.max(heatmap)
            if max_val > confidence_threshold:
                max_idx = np.unravel_index(np.argmax(heatmap), heatmap.shape)
                y, x = max_idx
                
                keypoints.append((int(x), int(y), float(max_val)))
                confidence_scores.append(float(max_val))
            else:
                keypoints.append((0, 0, 0.0))
                confidence_scores.append(0.0)
        
        # COCO 키포인트 연결 정의
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),     # 머리
            (1, 5), (5, 6), (6, 7),             # 왼팔
            (1, 8), (8, 9), (9, 10),            # 오른팔
            (1, 11), (11, 12), (12, 13),        # 왼다리
            (1, 14), (14, 15), (15, 16)         # 오른다리
        ]
        
        # 유효한 연결만 필터링
        valid_connections = []
        for conn in connections:
            if (conn[0] < len(keypoints) and conn[1] < len(keypoints) and 
                keypoints[conn[0]][2] > confidence_threshold and 
                keypoints[conn[1]][2] > confidence_threshold):
                valid_connections.append(conn)
        
        result.update({
            "keypoints": keypoints,
            "connections": valid_connections,
            "valid_keypoints": sum(1 for kp in keypoints if kp[2] > confidence_threshold),
            "confidence_scores": confidence_scores
        })
        
        logger.debug(f"✅ 포즈 후처리 완료: {result['valid_keypoints']}개 유효 키포인트")
        return result
        
    except Exception as e:
        logger.error(f"❌ 포즈 후처리 실패: {e}")
        return result

# =============================================================================
# 🔥 10. 이미지 품질 향상 함수들
# =============================================================================

def enhance_image_contrast(image: Any, factor: float = 1.2) -> Optional[Any]:
    """이미지 대비 향상"""
    try:
        if not _ensure_pil():
            return image
        
        pil_image = convert_to_pil(image)
        if pil_image is None:
            return image
        
        enhancer = ImageEnhance.Contrast(pil_image)
        enhanced = enhancer.enhance(factor)
        
        logger.debug(f"✅ 대비 향상: factor={factor}")
        return enhanced
        
    except Exception as e:
        logger.error(f"❌ 대비 향상 실패: {e}")
        return image

def enhance_image_brightness(image: Any, factor: float = 1.1) -> Optional[Any]:
    """이미지 밝기 향상"""
    try:
        if not _ensure_pil():
            return image
        
        pil_image = convert_to_pil(image)
        if pil_image is None:
            return image
        
        enhancer = ImageEnhance.Brightness(pil_image)
        enhanced = enhancer.enhance(factor)
        
        logger.debug(f"✅ 밝기 향상: factor={factor}")
        return enhanced
        
    except Exception as e:
        logger.error(f"❌ 밝기 향상 실패: {e}")
        return image

def enhance_image_sharpness(image: Any, factor: float = 1.1) -> Optional[Any]:
    """이미지 선명도 향상"""
    try:
        if not _ensure_pil():
            return image
        
        pil_image = convert_to_pil(image)
        if pil_image is None:
            return image
        
        enhancer = ImageEnhance.Sharpness(pil_image)
        enhanced = enhancer.enhance(factor)
        
        logger.debug(f"✅ 선명도 향상: factor={factor}")
        return enhanced
        
    except Exception as e:
        logger.error(f"❌ 선명도 향상 실패: {e}")
        return image

def apply_gaussian_blur(image: Any, radius: float = 1.0) -> Optional[Any]:
    """가우시안 블러 적용"""
    try:
        if not _ensure_pil():
            return image
        
        pil_image = convert_to_pil(image)
        if pil_image is None:
            return image
        
        blurred = pil_image.filter(ImageFilter.GaussianBlur(radius=radius))
        
        logger.debug(f"✅ 가우시안 블러: radius={radius}")
        return blurred
        
    except Exception as e:
        logger.error(f"❌ 가우시안 블러 실패: {e}")
        return image

# =============================================================================
# 🔥 11. 고급 이미지 처리 함수들
# =============================================================================

def apply_clahe_enhancement(image: Any, clip_limit: float = 2.0, 
                           tile_grid_size: Tuple[int, int] = (8, 8)) -> Optional[Any]:
    """CLAHE (대비 제한 적응 히스토그램 평활화) 적용"""
    try:
        if not CV2_AVAILABLE or not _ensure_numpy():
            logger.warning("⚠️ OpenCV 또는 NumPy 필요")
            return image
        
        array = convert_to_numpy(image)
        if array is None:
            return image
        
        # CLAHE 객체 생성
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        
        # 컬러 이미지인 경우 LAB 공간에서 처리
        if len(array.shape) == 3 and array.shape[2] == 3:
            # RGB → LAB 변환
            lab = cv2.cvtColor(array, cv2.COLOR_RGB2LAB)
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])  # L 채널에만 적용
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        elif len(array.shape) == 2:
            # 그레이스케일
            enhanced = clahe.apply(array)
        else:
            logger.warning("⚠️ 지원하지 않는 이미지 형태")
            return image
        
        # PIL Image로 변환
        if _ensure_pil():
            return Image.fromarray(enhanced)
        else:
            return enhanced
            
    except Exception as e:
        logger.error(f"❌ CLAHE 적용 실패: {e}")
        return image

def detect_dominant_colors(image: Any, k: int = 5) -> List[Tuple[int, int, int]]:
    """이미지에서 주요 색상 k개 추출"""
    try:
        if not _ensure_numpy():
            return []
        
        array = convert_to_numpy(image)
        if array is None or len(array.shape) != 3:
            return []
        
        # 픽셀을 1차원으로 변환
        pixels = array.reshape((-1, 3))
        
        # 고유 색상과 개수 계산
        unique_colors, counts = np.unique(
            pixels.view(np.dtype((np.void, pixels.dtype.itemsize * pixels.shape[1]))), 
            return_counts=True
        )
        
        # 상위 k개 색상 추출
        top_indices = np.argsort(counts)[-k:][::-1]
        dominant_colors = []
        
        for idx in top_indices:
            color_bytes = unique_colors[idx].view(pixels.dtype).reshape(pixels.shape[1])
            dominant_colors.append(tuple(color_bytes.astype(int)))
        
        logger.debug(f"✅ 주요 색상 {k}개 추출 완료")
        return dominant_colors
        
    except Exception as e:
        logger.error(f"❌ 주요 색상 추출 실패: {e}")
        return []

def calculate_image_similarity(image1: Any, image2: Any, method: str = "mse") -> float:
    """두 이미지 간의 유사도 계산"""
    try:
        if not _ensure_numpy():
            return 0.0
        
        # 이미지들을 numpy array로 변환
        arr1 = convert_to_numpy(image1)
        arr2 = convert_to_numpy(image2)
        
        if arr1 is None or arr2 is None:
            return 0.0
        
        # 크기 맞추기 (더 작은 크기로)
        if arr1.shape != arr2.shape:
            min_h = min(arr1.shape[0], arr2.shape[0])
            min_w = min(arr1.shape[1], arr2.shape[1])
            
            if len(arr1.shape) == 3:
                arr1 = arr1[:min_h, :min_w, :]
                arr2 = arr2[:min_h, :min_w, :]
            else:
                arr1 = arr1[:min_h, :min_w]
                arr2 = arr2[:min_h, :min_w]
        
        # 유사도 계산
        if method == "mse":
            # Mean Squared Error (낮을수록 유사)
            mse = np.mean((arr1.astype(float) - arr2.astype(float)) ** 2)
            similarity = 1.0 / (1.0 + mse / 255.0)
        elif method == "cosine":
            # 코사인 유사도
            arr1_flat = arr1.flatten().astype(float)
            arr2_flat = arr2.flatten().astype(float)
            
            dot_product = np.dot(arr1_flat, arr2_flat)
            norm1 = np.linalg.norm(arr1_flat)
            norm2 = np.linalg.norm(arr2_flat)
            
            if norm1 == 0 or norm2 == 0:
                similarity = 0.0
            else:
                similarity = dot_product / (norm1 * norm2)
        else:
            logger.warning(f"⚠️ 지원하지 않는 유사도 방법: {method}")
            return 0.0
        
        logger.debug(f"✅ 이미지 유사도: {similarity:.3f} ({method})")
        return float(similarity)
        
    except Exception as e:
        logger.error(f"❌ 이미지 유사도 계산 실패: {e}")
        return 0.0

# =============================================================================
# 🔥 12. 시각화 및 디버깅 함수들
# =============================================================================

def create_image_grid(images: List[Any], grid_size: Optional[Tuple[int, int]] = None, 
                     padding: int = 2, background_color: Tuple[int, int, int] = (255, 255, 255)) -> Optional[Any]:
    """이미지들을 격자로 배열"""
    try:
        if not _ensure_pil() or not images:
            return None
        
        # 격자 크기 자동 계산
        if grid_size is None:
            grid_cols = int(math.ceil(math.sqrt(len(images))))
            grid_rows = int(math.ceil(len(images) / grid_cols))
        else:
            grid_rows, grid_cols = grid_size
        
        # 모든 이미지를 PIL로 변환
        pil_images = []
        for img in images:
            pil_img = convert_to_pil(img)
            if pil_img:
                pil_images.append(pil_img)
        
        if not pil_images:
            return None
        
        # 최대 크기 계산
        max_width = max(img.width for img in pil_images)
        max_height = max(img.height for img in pil_images)
        
        # 모든 이미지를 같은 크기로 조정
        resized_images = []
        for img in pil_images:
            resized = img.resize((max_width, max_height), Image.Resampling.LANCZOS)
            resized_images.append(resized)
        
        # 격자 이미지 생성
        grid_width = grid_cols * max_width + (grid_cols + 1) * padding
        grid_height = grid_rows * max_height + (grid_rows + 1) * padding
        
        grid_image = Image.new('RGB', (grid_width, grid_height), background_color)
        
        # 이미지들 배치
        for i, img in enumerate(resized_images):
            if i >= grid_rows * grid_cols:
                break
            
            row = i // grid_cols
            col = i % grid_cols
            
            x = col * (max_width + padding) + padding
            y = row * (max_height + padding) + padding
            
            grid_image.paste(img, (x, y))
        
        logger.debug(f"✅ 이미지 격자 생성: {grid_size}, {len(resized_images)}개")
        return grid_image
        
    except Exception as e:
        logger.error(f"❌ 이미지 격자 생성 실패: {e}")
        return None

def add_text_to_image(image: Any, text: str, position: Tuple[int, int] = (10, 10), 
                     font_size: int = 20, color: Tuple[int, int, int] = (0, 0, 0)) -> Optional[Any]:
    """이미지에 텍스트 추가"""
    try:
        if not _ensure_pil():
            return image
        
        pil_image = convert_to_pil(image)
        if pil_image is None:
            return image
        
        # 복사본 생성
        result = pil_image.copy()
        draw = ImageDraw.Draw(result)
        
        # 폰트 설정
        try:
            # 시스템 폰트 시도
            font = ImageFont.truetype("arial.ttf", font_size)
        except:
            try:
                font = ImageFont.load_default()
            except:
                font = None
        
        # 텍스트 추가
        draw.text(position, text, fill=color, font=font)
        
        logger.debug(f"✅ 텍스트 추가: '{text}' at {position}")
        return result
        
    except Exception as e:
        logger.error(f"❌ 텍스트 추가 실패: {e}")
        return image

def create_comparison_image(image1: Any, image2: Any, 
                          labels: Tuple[str, str] = ("Original", "Processed")) -> Optional[Any]:
    """두 이미지를 나란히 비교하는 이미지 생성"""
    try:
        if not _ensure_pil():
            return None
        
        pil1 = convert_to_pil(image1)
        pil2 = convert_to_pil(image2)
        
        if pil1 is None or pil2 is None:
            return None
        
        # 같은 크기로 조정
        max_width = max(pil1.width, pil2.width)
        max_height = max(pil1.height, pil2.height)
        
        pil1 = pil1.resize((max_width, max_height), Image.Resampling.LANCZOS)
        pil2 = pil2.resize((max_width, max_height), Image.Resampling.LANCZOS)
        
        # 비교 이미지 생성
        padding = 20
        text_height = 30
        
        comparison_width = max_width * 2 + padding * 3
        comparison_height = max_height + text_height + padding * 2
        
        comparison = Image.new('RGB', (comparison_width, comparison_height), (255, 255, 255))
        
        # 이미지들 배치
        comparison.paste(pil1, (padding, text_height + padding))
        comparison.paste(pil2, (max_width + padding * 2, text_height + padding))
        
        # 라벨 추가
        comparison = add_text_to_image(comparison, labels[0], (padding, 5))
        comparison = add_text_to_image(comparison, labels[1], (max_width + padding * 2, 5))
        
        logger.debug(f"✅ 비교 이미지 생성: {labels}")
        return comparison
        
    except Exception as e:
        logger.error(f"❌ 비교 이미지 생성 실패: {e}")
        return None

# =============================================================================
# 🔥 13. 메모리 관리 및 최적화 함수들
# =============================================================================

def cleanup_image_memory() -> bool:
    """이미지 처리 관련 메모리 정리"""
    try:
        logger.debug("이미지 메모리 정리 시작")
        
        # Python garbage collection
        import gc
        collected = gc.collect()
        
        # PyTorch 캐시 정리
        if TORCH_AVAILABLE:
            if hasattr(torch, 'cuda') and torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.debug("CUDA 캐시 정리")
            
            if hasattr(torch, 'mps') and hasattr(torch.mps, 'empty_cache'):
                try:
                    torch.mps.empty_cache()
                    logger.debug("MPS 캐시 정리")
                except:
                    pass
        
        logger.info(f"✅ 메모리 정리 완료: {collected}개 객체 수집")
        return True
        
    except Exception as e:
        logger.error(f"❌ 메모리 정리 실패: {e}")
        return False

def estimate_memory_usage(image: Any) -> Dict[str, float]:
    """이미지 메모리 사용량 추정"""
    try:
        usage = {"bytes": 0, "mb": 0, "gb": 0}
        
        if hasattr(image, 'size') and hasattr(image, 'mode'):  # PIL Image
            width, height = image.size
            channels = len(image.getbands())
            bytes_per_pixel = 1 if image.mode == 'L' else 3 if image.mode == 'RGB' else 4
            total_bytes = width * height * bytes_per_pixel
            
        elif NUMPY_AVAILABLE and hasattr(image, 'nbytes'):  # NumPy array
            total_bytes = image.nbytes
            
        elif TORCH_AVAILABLE and torch.is_tensor(image):  # PyTorch tensor
            total_bytes = image.numel() * image.element_size()
            
        else:
            logger.warning("⚠️ 메모리 사용량 계산 불가")
            return usage
        
        usage.update({
            "bytes": total_bytes,
            "mb": total_bytes / (1024 * 1024),
            "gb": total_bytes / (1024 * 1024 * 1024)
        })
        
        logger.debug(f"메모리 사용량: {usage['mb']:.2f} MB")
        return usage
        
    except Exception as e:
        logger.error(f"❌ 메모리 사용량 계산 실패: {e}")
        return {"bytes": 0, "mb": 0, "gb": 0, "error": str(e)}

def optimize_image_memory(image: Any, target_size: Optional[Tuple[int, int]] = None, 
                         quality: int = 85) -> Optional[Any]:
    """이미지 메모리 사용량 최적화"""
    try:
        if not _ensure_pil():
            return image
        
        pil_image = convert_to_pil(image)
        if pil_image is None:
            return image
        
        # 크기 조정
        if target_size and pil_image.size != target_size:
            pil_image = resize_image(pil_image, target_size, keep_aspect_ratio=True)
        
        # 메모리 사용량 체크
        memory_usage = estimate_memory_usage(pil_image)
        
        # 임계값 초과시 압축 적용
        if memory_usage['mb'] > MEMORY_THRESHOLD_MB:
            # JPEG 압축 적용
            buffer = io.BytesIO()
            if pil_image.mode == 'RGBA':
                # 투명도 제거
                background = Image.new('RGB', pil_image.size, (255, 255, 255))
                background.paste(pil_image, mask=pil_image.split()[-1])
                pil_image = background
            
            pil_image.save(buffer, format='JPEG', quality=quality, optimize=True)
            buffer.seek(0)
            pil_image = Image.open(buffer)
            
            logger.debug(f"✅ 메모리 최적화 적용: quality={quality}")
        
        return pil_image
        
    except Exception as e:
        logger.error(f"❌ 메모리 최적화 실패: {e}")
        return image

def validate_image_format(image: Any) -> Dict[str, Any]:
    """이미지 포맷 및 속성 검증"""
    try:
        result = {
            "valid": False,
            "type": str(type(image)),
            "format": None,
            "size": None,
            "mode": None,
            "channels": None,
            "dtype": None,
            "memory_usage_mb": 0.0
        }
        
        if hasattr(image, 'size') and hasattr(image, 'mode'):  # PIL Image
            memory_usage = estimate_memory_usage(image)
            result.update({
                "valid": True,
                "format": "PIL",
                "size": image.size,
                "mode": image.mode,
                "channels": len(image.getbands()),
                "memory_usage_mb": memory_usage['mb']
            })
            
        elif NUMPY_AVAILABLE and isinstance(image, np.ndarray):  # NumPy array
            memory_usage = estimate_memory_usage(image)
            result.update({
                "valid": True,
                "format": "NumPy",
                "size": (image.shape[1], image.shape[0]) if len(image.shape) >= 2 else image.shape,
                "channels": image.shape[2] if len(image.shape) == 3 else 1,
                "dtype": str(image.dtype),
                "memory_usage_mb": memory_usage['mb']
            })
            
        elif TORCH_AVAILABLE and torch.is_tensor(image):  # PyTorch tensor
            memory_usage = estimate_memory_usage(image)
            result.update({
                "valid": True,
                "format": "PyTorch",
                "size": (image.shape[-1], image.shape[-2]) if len(image.shape) >= 2 else image.shape,
                "channels": image.shape[-3] if len(image.shape) >= 3 else 1,
                "dtype": str(image.dtype),
                "memory_usage_mb": memory_usage['mb']
            })
        
        logger.debug(f"이미지 검증: {result}")
        return result
        
    except Exception as e:
        logger.error(f"❌ 이미지 검증 실패: {e}")
        return {"valid": False, "error": str(e)}

# =============================================================================
# 🔥 14. Step별 특화 처리 함수들
# =============================================================================

def preprocess_pose_input(image: Any, target_size: Tuple[int, int] = (368, 368)) -> Optional[Any]:
    """포즈 추정용 이미지 전처리"""
    return preprocess_image(image, target_size, normalize=True, to_tensor=True)

def preprocess_human_parsing_input(image: Any, target_size: Tuple[int, int] = (512, 512)) -> Optional[Any]:
    """인체 파싱용 이미지 전처리"""
    return preprocess_image(image, target_size, normalize=True, to_tensor=True)

def preprocess_cloth_segmentation_input(image: Any, target_size: Tuple[int, int] = (320, 320)) -> Optional[Any]:
    """의류 세그멘테이션용 이미지 전처리"""
    return preprocess_image(image, target_size, normalize=True, to_tensor=True)

def preprocess_virtual_fitting_input(person_img: Any, cloth_img: Any, 
                                   target_size: Tuple[int, int] = (512, 512)) -> Tuple[Optional[Any], Optional[Any]]:
    """가상 피팅용 이미지 전처리"""
    person_tensor = preprocess_image(person_img, target_size, normalize=True, to_tensor=True)
    cloth_tensor = preprocess_image(cloth_img, target_size, normalize=True, to_tensor=True)
    return person_tensor, cloth_tensor

def postprocess_human_parsing(output: Any, num_classes: int = 20, 
                             colormap: Optional[List[Tuple[int, int, int]]] = None) -> Optional[Any]:
    """인체 파싱 결과 후처리 (컬러맵 적용)"""
    try:
        if not _ensure_numpy():
            return output
        
        # 출력을 numpy array로 변환
        if TORCH_AVAILABLE and torch.is_tensor(output):
            if output.is_cuda or (hasattr(output, 'is_mps') and output.is_mps):
                output = output.cpu()
            pred = output.detach().numpy()
        else:
            pred = convert_to_numpy(output)
        
        if pred is None:
            return output
        
        # 차원 조정
        if pred.ndim == 4:  # (N, C, H, W)
            pred = pred.squeeze(0)
        if pred.ndim == 3:  # (C, H, W)
            pred = np.argmax(pred, axis=0)
        
        # 기본 컬러맵 생성
        if colormap is None:
            colormap = []
            for i in range(num_classes):
                if i == 0:  # 배경은 검은색
                    colormap.append((0, 0, 0))
                else:
                    # HSV 색공간에서 균등 분포
                    hue = int(i * 360 / num_classes)
                    # 간단한 HSV to RGB 변환
                    c, x = 255, int(255 * (1 - abs((hue / 60) % 2 - 1)))
                    if 0 <= hue < 60: rgb = (c, x, 0)
                    elif 60 <= hue < 120: rgb = (x, c, 0)
                    elif 120 <= hue < 180: rgb = (0, c, x)
                    elif 180 <= hue < 240: rgb = (0, x, c)
                    elif 240 <= hue < 300: rgb = (x, 0, c)
                    else: rgb = (c, 0, x)
                    colormap.append(rgb)
        
        # 컬러맵 적용
        height, width = pred.shape
        colored = np.zeros((height, width, 3), dtype=np.uint8)
        
        for class_id in range(min(num_classes, len(colormap))):
            mask = (pred == class_id)
            colored[mask] = colormap[class_id]
        
        logger.debug(f"✅ 인체 파싱 후처리 완료: {num_classes}개 클래스")
        return colored
        
    except Exception as e:
        logger.error(f"❌ 인체 파싱 후처리 실패: {e}")
        return output

def create_pose_visualization(image: Any, keypoints_result: Dict[str, Any]) -> Optional[Any]:
    """포즈 키포인트 시각화"""
    try:
        if not _ensure_pil():
            return image
        
        pil_image = convert_to_pil(image)
        if pil_image is None:
            return image
        
        # 복사본 생성
        vis_image = pil_image.copy()
        draw = ImageDraw.Draw(vis_image)
        
        keypoints = keypoints_result.get("keypoints", [])
        connections = keypoints_result.get("connections", [])
        
        # 연결선 그리기
        for conn in connections:
            if conn[0] < len(keypoints) and conn[1] < len(keypoints):
                pt1 = keypoints[conn[0]]
                pt2 = keypoints[conn[1]]
                
                if pt1[2] > 0 and pt2[2] > 0:  # 유효한 키포인트들만
                    draw.line([pt1[0], pt1[1], pt2[0], pt2[1]], 
                             fill=(0, 255, 0), width=3)
        
        # 키포인트 그리기
        for i, (x, y, conf) in enumerate(keypoints):
            if conf > 0:
                # 신뢰도에 따른 색상
                color = (255, int(255 * conf), 0)
                radius = 5
                
                # 원 그리기
                draw.ellipse([x-radius, y-radius, x+radius, y+radius], 
                           fill=color, outline=(0, 0, 0), width=2)
        
        logger.debug("✅ 포즈 시각화 완료")
        return vis_image
        
    except Exception as e:
        logger.error(f"❌ 포즈 시각화 실패: {e}")
        return image

# =============================================================================
# 🔥 15. 배치 처리 함수들
# =============================================================================

def create_batch(images: List[Any], device: str = None) -> Optional[Any]:
    """이미지 리스트를 배치 텐서로 변환"""
    try:
        if not _ensure_torch() or not images:
            return None
        
        if device is None:
            device = _get_optimal_device()
        
        tensors = []
        for i, img in enumerate(images):
            if torch.is_tensor(img):
                tensor = img
                if tensor.dim() == 3:  # (C, H, W)
                    tensor = tensor.unsqueeze(0)  # (1, C, H, W)
            else:
                tensor = pil_to_tensor(img, device, normalize=True)
                if tensor is None:
                    continue
            
            tensors.append(tensor)
        
        if not tensors:
            return None
        
        # 배치로 결합
        batch = torch.cat(tensors, dim=0).to(device)
        
        logger.debug(f"✅ 배치 생성: {batch.shape}, device: {device}")
        return batch
        
    except Exception as e:
        logger.error(f"❌ 배치 생성 실패: {e}")
        return None

def split_batch(batch: torch.Tensor) -> List[torch.Tensor]:
    """배치 텐서를 개별 텐서들로 분할"""
    try:
        if not TORCH_AVAILABLE or not torch.is_tensor(batch):
            return []
        
        if batch.dim() != 4:  # (N, C, H, W)
            logger.warning("⚠️ 4차원 배치 텐서가 아닙니다")
            return []
        
        tensors = [batch[i:i+1] for i in range(batch.shape[0])]
        
        logger.debug(f"✅ 배치 분할: {len(tensors)}개 텐서")
        return tensors
        
    except Exception as e:
        logger.error(f"❌ 배치 분할 실패: {e}")
        return []

# =============================================================================
# 🔥 16. 모듈 정보 및 내보내기
# =============================================================================

__version__ = "2.0.0"
__author__ = "MyCloset AI Team"
__description__ = "완전한 이미지 처리 모듈 - 올바른 Python 구조"

# 모든 공개 함수들
__all__ = [
    # 기본 이미지 I/O
    'load_image',
    'save_image',
    
    # 이미지 변환
    'convert_to_pil',
    'convert_to_numpy',
    'tensor_to_pil',
    'pil_to_tensor',
    
    # 크기 및 형태 조정
    'resize_image',
    'crop_image',
    'pad_image',
    
    # 정규화 및 전처리
    'normalize_image',
    'denormalize_image',
    'preprocess_image',
    
    # Base64 변환
    'image_to_base64',
    'base64_to_image',
    
    # 후처리
    'postprocess_segmentation',
    'postprocess_pose_keypoints',
    'postprocess_human_parsing',
    
    # 품질 향상
    'enhance_image_contrast',
    'enhance_image_brightness',
    'enhance_image_sharpness',
    'apply_gaussian_blur',
    
    # 고급 처리
    'apply_clahe_enhancement',
    'detect_dominant_colors',
    'calculate_image_similarity',
    
    # 시각화
    'create_image_grid',
    'add_text_to_image',
    'create_comparison_image',
    'create_pose_visualization',
    
    # 메모리 관리
    'cleanup_image_memory',
    'estimate_memory_usage',
    'optimize_image_memory',
    'validate_image_format',
    
    # Step별 특화
    'preprocess_pose_input',
    'preprocess_human_parsing_input',
    'preprocess_cloth_segmentation_input',
    'preprocess_virtual_fitting_input',
    
    # 배치 처리
    'create_batch',
    'split_batch',
    
    # 유틸리티
    '_get_optimal_device',
    '_validate_image_input',
    
    # 상수
    'DEFAULT_IMAGE_SIZE',
    'DEFAULT_DEVICE',
    'SUPPORTED_FORMATS',
    'SUPPORTED_EXTENSIONS',
    'NUMPY_AVAILABLE',
    'PIL_AVAILABLE',
    'CV2_AVAILABLE',
    'TORCH_AVAILABLE'
]

# 초기화 로그
logger.info(f"🖼️ 이미지 처리 모듈 v{__version__} 로드 완료")
logger.info(f"📦 사용 가능한 함수: {len(__all__)}개")
logger.info(f"🔧 라이브러리 지원 상태:")
logger.info(f"   - NumPy: {'✅' if NUMPY_AVAILABLE else '❌'}")
logger.info(f"   - PIL/Pillow: {'✅' if PIL_AVAILABLE else '❌'}")
logger.info(f"   - OpenCV: {'✅' if CV2_AVAILABLE else '❌'}")
logger.info(f"   - PyTorch: {'✅' if TORCH_AVAILABLE else '❌'}")
logger.info(f"🚀 최적 디바이스: {_get_optimal_device()}")

# =============================================================================
# 🔥 17. 사용 예시 (주석으로 문서화)
# =============================================================================

"""
🎯 사용 예시:

# 1. 기본 이미지 처리
from backend.app.ai_pipeline.utils.image_processing import (
    load_image, save_image, preprocess_image, postprocess_segmentation
)

# 이미지 로드 및 전처리
image = load_image('input.jpg')
processed = preprocess_image(image, target_size=(512, 512), to_tensor=True)

# 세그멘테이션 후처리
mask = postprocess_segmentation(model_output, threshold=0.5)

# 2. 이미지 변환
from backend.app.ai_pipeline.utils.image_processing import (
    tensor_to_pil, pil_to_tensor, convert_to_numpy
)

pil_image = tensor_to_pil(tensor)
tensor = pil_to_tensor(pil_image, device='mps')
array = convert_to_numpy(image)

# 3. Base64 변환
from backend.app.ai_pipeline.utils.image_processing import (
    image_to_base64, base64_to_image
)

base64_str = image_to_base64(image, format='JPEG', quality=95)
image = base64_to_image(base64_str)

# 4. 배치 처리
from backend.app.ai_pipeline.utils.image_processing import create_batch

batch_tensor = create_batch([img1, img2, img3], device='mps')

# 5. 이미지 향상
from backend.app.ai_pipeline.utils.image_processing import (
    enhance_image_contrast, apply_clahe_enhancement
)

enhanced = enhance_image_contrast(image, factor=1.2)
clahe_enhanced = apply_clahe_enhancement(image, clip_limit=2.0)

# 6. Step별 특화 처리
from backend.app.ai_pipeline.utils.image_processing import (
    preprocess_pose_input, postprocess_human_parsing, create_pose_visualization
)

pose_input = preprocess_pose_input(image, target_size=(368, 368))
colored_parsing = postprocess_human_parsing(parsing_output, num_classes=20)
pose_vis = create_pose_visualization(image, keypoints_result)

# 7. 시각화
from backend.app.ai_pipeline.utils.image_processing import (
    create_image_grid, create_comparison_image, add_text_to_image
)

grid = create_image_grid([img1, img2, img3, img4], grid_size=(2, 2))
comparison = create_comparison_image(original, processed, ('Before', 'After'))
labeled = add_text_to_image(image, 'MyCloset AI', position=(10, 10))

# 8. 메모리 관리
from backend.app.ai_pipeline.utils.image_processing import (
    cleanup_image_memory, estimate_memory_usage, optimize_image_memory
)

memory_info = estimate_memory_usage(image)
optimized = optimize_image_memory(image, target_size=(512, 512))
cleanup_image_memory()
"""