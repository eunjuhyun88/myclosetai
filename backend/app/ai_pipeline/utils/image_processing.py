# backend/app/ai_pipeline/utils/image_processing.py
"""
🖼️ MyCloset AI - 이미지 처리 함수들 (순환참조 방지 버전)
=========================================================
✅ model_loader.py에서 분리된 이미지 처리 함수들
✅ 순환참조 완전 방지 - 독립적인 모듈
✅ PIL, OpenCV, NumPy 기반 이미지 처리
✅ PyTorch 텐서 변환 지원
✅ M3 Max 128GB 최적화
✅ conda 환경 완벽 지원
✅ 기존 함수명 100% 유지

Author: MyCloset AI Team
Date: 2025-07-21
Version: 1.0 (Separated from model_loader.py)
"""

import io
import logging
import base64
import tempfile
import os
from typing import Dict, Any, Optional, Tuple, List, Union
from pathlib import Path

# 조건부 임포트 (안전한 처리)
try:
    import numpy as np
    NUMPY_AVAILABLE = True
    logger = logging.getLogger(__name__)
    logger.info("✅ NumPy 사용 가능")
    
    # NumPy 2.x 호환성 처리
    major_version = int(np.__version__.split('.')[0])
    if major_version >= 2:
        logger.warning(f"⚠️ NumPy {np.__version__} 감지됨. NumPy 1.x 권장")
        logger.warning("🔧 해결방법: conda install numpy=1.24.3 -y --force-reinstall")
except ImportError as e:
    NUMPY_AVAILABLE = False
    np = None
    logger = logging.getLogger(__name__)
    logger.warning(f"⚠️ NumPy 없음: {e}")

try:
    from PIL import Image, ImageEnhance, ImageFilter, ImageDraw, ImageFont
    PIL_AVAILABLE = True
    logger.info("✅ PIL/Pillow 사용 가능")
except ImportError as e:
    PIL_AVAILABLE = False
    logger.warning(f"⚠️ PIL/Pillow 없음: {e}")

try:
    import cv2
    CV2_AVAILABLE = True
    logger.info("✅ OpenCV 사용 가능")
except ImportError as e:
    CV2_AVAILABLE = False
    logger.warning(f"⚠️ OpenCV 없음: {e}")

try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
    logger.info("✅ PyTorch 사용 가능")
except ImportError as e:
    TORCH_AVAILABLE = False
    torch = None
    logger.warning(f"⚠️ PyTorch 없음: {e}")

# ==============================================
# 🔥 기본 이미지 전처리 함수들
# ==============================================

def preprocess_image(
    image: Union[str, 'Image.Image', 'np.ndarray'],
    target_size: Tuple[int, int] = (512, 512),
    device: str = "mps",
    normalize: bool = True,
    to_tensor: bool = True
) -> Any:
    """
    이미지 전처리 함수 - 완전 구현
    
    Args:
        image: 입력 이미지 (파일 경로, PIL Image, numpy array)
        target_size: 타겟 크기 (width, height)
        device: 디바이스 ("mps", "cuda", "cpu")
        normalize: 정규화 여부 (0-1 범위로)
        to_tensor: PyTorch tensor로 변환 여부
    
    Returns:
        전처리된 이미지 (tensor 또는 numpy array)
    """
    try:
        logger.debug(f"이미지 전처리 시작: {type(image)}, 타겟 크기: {target_size}")
        
        # 1. 이미지 로드 및 변환
        if isinstance(image, (str, Path)):
            # 파일 경로인 경우
            if PIL_AVAILABLE:
                try:
                    image = Image.open(image).convert('RGB')
                    logger.debug("✅ PIL로 이미지 로드 성공")
                except Exception as e:
                    logger.error(f"❌ PIL 이미지 로드 실패: {e}")
                    if CV2_AVAILABLE and NUMPY_AVAILABLE:
                        image = cv2.imread(str(image))
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        logger.debug("✅ OpenCV로 이미지 로드 성공")
                    else:
                        raise ImportError("이미지 로드를 위해 PIL 또는 OpenCV가 필요합니다")
            else:
                raise ImportError("이미지 로드를 위해 PIL이 필요합니다")
        
        # 2. PIL Image 처리
        if hasattr(image, 'resize'):  # PIL Image
            logger.debug("PIL Image 처리 중...")
            image = image.resize(target_size, Image.Resampling.LANCZOS if hasattr(Image, 'Resampling') else Image.LANCZOS)
            
            if NUMPY_AVAILABLE:
                img_array = np.array(image).astype(np.float32)
                logger.debug(f"PIL → NumPy 변환: {img_array.shape}")
            else:
                # NumPy 없는 경우 수동 변환
                width, height = image.size
                img_array = []
                for y in range(height):
                    row = []
                    for x in range(width):
                        pixel = image.getpixel((x, y))
                        if isinstance(pixel, int):  # 그레이스케일
                            row.append([pixel, pixel, pixel])
                        else:  # RGB
                            row.append(list(pixel))
                    img_array.append(row)
                logger.debug("PIL → 리스트 변환 완료")
        
        # 3. OpenCV/NumPy 처리
        elif CV2_AVAILABLE and NUMPY_AVAILABLE and hasattr(image, 'shape'):  # OpenCV/numpy array
            logger.debug(f"OpenCV/NumPy 배열 처리 중: {image.shape}")
            if len(image.shape) == 3 and image.shape[2] == 3:
                # RGB 이미지
                img_array = cv2.resize(image, target_size).astype(np.float32)
            elif len(image.shape) == 2:
                # 그레이스케일
                img_array = cv2.resize(image, target_size)
                img_array = np.stack([img_array] * 3, axis=-1).astype(np.float32)
            else:
                raise ValueError(f"지원하지 않는 이미지 형태: {image.shape}")
        
        # 4. 폴백 처리
        else:
            logger.warning("⚠️ 정규화 지원하지 않는 타입, 원본 반환")
            return image
    except Exception as e:
        logger.error(f"❌ 이미지 정규화 실패: {e}")
        return image

def denormalize_image(image: Any, mean: Tuple[float, float, float] = (0.485, 0.456, 0.406), 
                     std: Tuple[float, float, float] = (0.229, 0.224, 0.225)) -> Any:
    """이미지 역정규화"""
    try:
        logger.debug(f"이미지 역정규화: mean={mean}, std={std}")
        
        if TORCH_AVAILABLE and hasattr(image, 'dim'):
            # PyTorch tensor
            image_denormalized = image.clone()
            if image_denormalized.dim() == 4:  # (N, C, H, W)
                for i in range(3):
                    image_denormalized[:, i, :, :] = image_denormalized[:, i, :, :] * std[i] + mean[i]
            elif image_denormalized.dim() == 3:  # (C, H, W)
                for i in range(3):
                    image_denormalized[i, :, :] = image_denormalized[i, :, :] * std[i] + mean[i]
            logger.debug("✅ PyTorch 텐서 역정규화 완료")
            return image_denormalized
        elif NUMPY_AVAILABLE and hasattr(image, 'shape'):
            # numpy array
            image_denormalized = image.astype(np.float32).copy()
            if len(image.shape) == 4:  # (N, H, W, C)
                for i in range(3):
                    image_denormalized[:, :, :, i] = image_denormalized[:, :, :, i] * std[i] + mean[i]
            elif len(image.shape) == 3:  # (H, W, C)
                for i in range(3):
                    image_denormalized[:, :, i] = image_denormalized[:, :, i] * std[i] + mean[i]
            logger.debug("✅ NumPy 배열 역정규화 완료")
            return image_denormalized
        else:
            logger.warning("⚠️ 역정규화 지원하지 않는 타입, 원본 반환")
            return image
    except Exception as e:
        logger.error(f"❌ 이미지 역정규화 실패: {e}")
        return image

def create_batch(images: List[Any], device: str = "mps") -> Any:
    """이미지 리스트를 배치로 변환"""
    try:
        logger.debug(f"배치 생성: {len(images)}개 이미지 → device: {device}")
        
        if not images:
            logger.warning("⚠️ 빈 이미지 리스트, 기본 텐서 반환")
            if TORCH_AVAILABLE:
                return torch.zeros(1, 3, 512, 512, device=device)
            else:
                return []
        
        if TORCH_AVAILABLE:
            # 모든 이미지를 tensor로 변환
            tensors = []
            for i, img in enumerate(images):
                logger.debug(f"이미지 {i+1}/{len(images)} 처리 중...")
                
                if hasattr(img, 'dim'):  # 이미 tensor
                    if img.dim() == 3:  # (C, H, W)
                        tensors.append(img.unsqueeze(0))
                    else:
                        tensors.append(img)
                else:
                    # PIL 또는 numpy → tensor
                    tensor = pil_to_tensor(img, device)
                    tensors.append(tensor)
            
            # 배치로 결합
            if tensors:
                batch = torch.cat(tensors, dim=0)
                batch = batch.to(device)
                logger.debug(f"✅ 배치 생성 완료: {batch.shape}")
                return batch
            else:
                logger.warning("⚠️ 텐서 변환 실패, 기본 텐서 반환")
                return torch.zeros(1, 3, 512, 512, device=device)
        else:
            logger.warning("⚠️ PyTorch 없음, 원본 리스트 반환")
            return images
            
    except Exception as e:
        logger.error(f"❌ 배치 생성 실패: {e}")
        if TORCH_AVAILABLE:
            return torch.zeros(len(images) if images else 1, 3, 512, 512, device=device)
        else:
            return images

# ==============================================
# 🔥 Base64 변환 함수들
# ==============================================

def image_to_base64(image: Any, format: str = "JPEG", quality: int = 95) -> str:
    """이미지를 Base64 문자열로 변환"""
    try:
        logger.debug(f"이미지→Base64 변환: format={format}, quality={quality}")
        
        if not PIL_AVAILABLE:
            logger.error("❌ PIL 필요함")
            return ""
        
        # PIL Image로 변환
        if hasattr(image, 'save'):  # 이미 PIL Image
            pil_image = image
        elif NUMPY_AVAILABLE and hasattr(image, 'shape'):  # NumPy array
            if image.dtype != np.uint8:
                if image.max() <= 1.0:
                    image = (image * 255).astype(np.uint8)
                else:
                    image = np.clip(image, 0, 255).astype(np.uint8)
            pil_image = Image.fromarray(image)
        elif TORCH_AVAILABLE and hasattr(image, 'cpu'):  # PyTorch tensor
            pil_image = tensor_to_pil(image)
        else:
            logger.error(f"❌ 지원하지 않는 이미지 타입: {type(image)}")
            return ""
        
        # RGB 모드로 변환
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        # Base64 변환
        buffer = io.BytesIO()
        pil_image.save(buffer, format=format, quality=quality)
        img_str = base64.b64encode(buffer.getvalue()).decode()
        
        logger.debug(f"✅ Base64 변환 완료: {len(img_str)} 문자")
        return img_str
        
    except Exception as e:
        logger.error(f"❌ 이미지→Base64 변환 실패: {e}")
        return ""

def base64_to_image(base64_str: str) -> Any:
    """Base64 문자열을 이미지로 변환"""
    try:
        logger.debug(f"Base64→이미지 변환: {len(base64_str)} 문자")
        
        if not PIL_AVAILABLE:
            logger.error("❌ PIL 필요함")
            return None
        
        # Base64 디코딩
        img_data = base64.b64decode(base64_str)
        img_buffer = io.BytesIO(img_data)
        pil_image = Image.open(img_buffer).convert('RGB')
        
        logger.debug(f"✅ Base64→이미지 변환 완료: {pil_image.size}")
        return pil_image
        
    except Exception as e:
        logger.error(f"❌ Base64→이미지 변환 실패: {e}")
        return None

def numpy_to_base64(array: 'np.ndarray', format: str = "JPEG", quality: int = 95) -> str:
    """NumPy 배열을 Base64로 변환"""
    try:
        if not NUMPY_AVAILABLE:
            logger.error("❌ NumPy 필요함")
            return ""
        
        return image_to_base64(array, format, quality)
        
    except Exception as e:
        logger.error(f"❌ NumPy→Base64 변환 실패: {e}")
        return ""

def base64_to_numpy(base64_str: str) -> Any:
    """Base64를 NumPy 배열로 변환"""
    try:
        if not NUMPY_AVAILABLE:
            logger.error("❌ NumPy 필요함")
            return None
        
        pil_image = base64_to_image(base64_str)
        if pil_image:
            return np.array(pil_image)
        else:
            return None
            
    except Exception as e:
        logger.error(f"❌ Base64→NumPy 변환 실패: {e}")
        return None

# ==============================================
# 🔥 이미지 품질 향상 함수들
# ==============================================

def enhance_image_contrast(image: Any, factor: float = 1.2) -> Any:
    """이미지 대비 향상"""
    try:
        if PIL_AVAILABLE and hasattr(image, 'save'):
            enhancer = ImageEnhance.Contrast(image)
            enhanced = enhancer.enhance(factor)
            logger.debug(f"✅ 대비 향상 완료: factor={factor}")
            return enhanced
        else:
            logger.warning("⚠️ PIL 이미지가 아님, 원본 반환")
            return image
    except Exception as e:
        logger.error(f"❌ 대비 향상 실패: {e}")
        return image

def enhance_image_brightness(image: Any, factor: float = 1.1) -> Any:
    """이미지 밝기 향상"""
    try:
        if PIL_AVAILABLE and hasattr(image, 'save'):
            enhancer = ImageEnhance.Brightness(image)
            enhanced = enhancer.enhance(factor)
            logger.debug(f"✅ 밝기 향상 완료: factor={factor}")
            return enhanced
        else:
            logger.warning("⚠️ PIL 이미지가 아님, 원본 반환")
            return image
    except Exception as e:
        logger.error(f"❌ 밝기 향상 실패: {e}")
        return image

def enhance_image_sharpness(image: Any, factor: float = 1.1) -> Any:
    """이미지 선명도 향상"""
    try:
        if PIL_AVAILABLE and hasattr(image, 'save'):
            enhancer = ImageEnhance.Sharpness(image)
            enhanced = enhancer.enhance(factor)
            logger.debug(f"✅ 선명도 향상 완료: factor={factor}")
            return enhanced
        else:
            logger.warning("⚠️ PIL 이미지가 아님, 원본 반환")
            return image
    except Exception as e:
        logger.error(f"❌ 선명도 향상 실패: {e}")
        return image

def apply_gaussian_blur(image: Any, radius: float = 1.0) -> Any:
    """가우시안 블러 적용"""
    try:
        if PIL_AVAILABLE and hasattr(image, 'save'):
            blurred = image.filter(ImageFilter.GaussianBlur(radius=radius))
            logger.debug(f"✅ 가우시안 블러 적용 완료: radius={radius}")
            return blurred
        else:
            logger.warning("⚠️ PIL 이미지가 아님, 원본 반환")
            return image
    except Exception as e:
        logger.error(f"❌ 가우시안 블러 적용 실패: {e}")
        return image

# ==============================================
# 🔥 이미지 검증 및 분석 함수들
# ==============================================

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
            "dtype": None
        }
        
        if hasattr(image, 'size'):  # PIL Image
            result.update({
                "valid": True,
                "format": "PIL",
                "size": image.size,
                "mode": image.mode,
                "channels": len(image.getbands())
            })
        elif NUMPY_AVAILABLE and hasattr(image, 'shape'):  # NumPy array
            result.update({
                "valid": True,
                "format": "NumPy",
                "size": (image.shape[1], image.shape[0]) if len(image.shape) >= 2 else image.shape,
                "channels": image.shape[2] if len(image.shape) == 3 else 1,
                "dtype": str(image.dtype)
            })
        elif TORCH_AVAILABLE and hasattr(image, 'shape'):  # PyTorch tensor
            result.update({
                "valid": True,
                "format": "PyTorch",
                "size": (image.shape[-1], image.shape[-2]) if len(image.shape) >= 2 else image.shape,
                "channels": image.shape[-3] if len(image.shape) >= 3 else 1,
                "dtype": str(image.dtype)
            })
        
        logger.debug(f"이미지 검증 결과: {result}")
        return result
        
    except Exception as e:
        logger.error(f"❌ 이미지 검증 실패: {e}")
        return {"valid": False, "error": str(e)}

def get_image_statistics(image: Any) -> Dict[str, Any]:
    """이미지 통계 정보"""
    try:
        stats = {"error": None}
        
        if NUMPY_AVAILABLE and hasattr(image, 'shape'):
            if hasattr(image, 'cpu'):  # PyTorch tensor
                array = image.cpu().numpy()
            else:
                array = image
            
            stats.update({
                "mean": float(np.mean(array)),
                "std": float(np.std(array)),
                "min": float(np.min(array)),
                "max": float(np.max(array)),
                "shape": array.shape
            })
        elif hasattr(image, 'size'):  # PIL Image
            if NUMPY_AVAILABLE:
                array = np.array(image)
                stats.update({
                    "mean": float(np.mean(array)),
                    "std": float(np.std(array)),
                    "min": float(np.min(array)),
                    "max": float(np.max(array)),
                    "size": image.size,
                    "mode": image.mode
                })
        
        logger.debug(f"이미지 통계: {stats}")
        return stats
        
    except Exception as e:
        logger.error(f"❌ 이미지 통계 계산 실패: {e}")
        return {"error": str(e)}

# ==============================================
# 🔥 메모리 관리 함수들
# ==============================================

def cleanup_image_memory():
    """이미지 처리 관련 메모리 정리"""
    try:
        logger.debug("이미지 메모리 정리 시작")
        
        # Python garbage collection
        import gc
        collected = gc.collect()
        logger.debug(f"Python GC: {collected}개 객체 수집")
        
        # PyTorch 캐시 정리
        if TORCH_AVAILABLE:
            if hasattr(torch, 'cuda') and torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.debug("CUDA 캐시 정리 완료")
            
            if hasattr(torch, 'mps') and hasattr(torch.mps, 'empty_cache'):
                try:
                    torch.mps.empty_cache()
                    logger.debug("MPS 캐시 정리 완료")
                except:
                    pass
        
        logger.info("✅ 이미지 메모리 정리 완료")
        return True
        
    except Exception as e:
        logger.error(f"❌ 이미지 메모리 정리 실패: {e}")
        return False

def estimate_memory_usage(image: Any) -> Dict[str, float]:
    """이미지 메모리 사용량 추정"""
    try:
        usage = {"bytes": 0, "mb": 0, "error": None}
        
        if hasattr(image, 'size'):  # PIL Image
            width, height = image.size
            channels = len(image.getbands())
            bytes_per_pixel = 1 if image.mode == 'L' else 3 if image.mode == 'RGB' else 4
            total_bytes = width * height * bytes_per_pixel
        elif NUMPY_AVAILABLE and hasattr(image, 'nbytes'):  # NumPy array
            total_bytes = image.nbytes
        elif TORCH_AVAILABLE and hasattr(image, 'element_size'):  # PyTorch tensor
            total_bytes = image.numel() * image.element_size()
        else:
            total_bytes = 0
        
        usage.update({
            "bytes": total_bytes,
            "mb": total_bytes / (1024 * 1024)
        })
        
        logger.debug(f"메모리 사용량 추정: {usage['mb']:.2f} MB")
        return usage
        
    except Exception as e:
        logger.error(f"❌ 메모리 사용량 추정 실패: {e}")
        return {"bytes": 0, "mb": 0, "error": str(e)}

# ==============================================
# 🔥 모듈 정보 및 내보내기
# ==============================================

__version__ = "1.0.0"
__author__ = "MyCloset AI Team"
__description__ = "이미지 처리 함수들 - model_loader.py에서 분리"

__all__ = [
    # 기본 전처리 함수들
    'preprocess_image',
    'postprocess_segmentation',
    
    # 특화 전처리 함수들
    'preprocess_pose_input',
    'preprocess_human_parsing_input',
    'preprocess_cloth_segmentation_input',
    'preprocess_virtual_fitting_input',
    
    # 이미지 변환 함수들
    'tensor_to_pil',
    'pil_to_tensor',
    'resize_image',
    'normalize_image',
    'denormalize_image',
    'create_batch',
    
    # Base64 변환 함수들
    'image_to_base64',
    'base64_to_image',
    'numpy_to_base64',
    'base64_to_numpy',
    
    # 이미지 품질 향상 함수들
    'enhance_image_contrast',
    'enhance_image_brightness',
    'enhance_image_sharpness',
    'apply_gaussian_blur',
    
    # 검증 및 분석 함수들
    'validate_image_format',
    'get_image_statistics',
    
    # 메모리 관리 함수들
    'cleanup_image_memory',
    'estimate_memory_usage',
    
    # 상수들
    'NUMPY_AVAILABLE',
    'PIL_AVAILABLE',
    'CV2_AVAILABLE',
    'TORCH_AVAILABLE'
]

logger.info(f"🖼️ 이미지 처리 모듈 v{__version__} 로드 완료")
logger.info(f"📦 사용 가능한 함수: {len(__all__)}개")
logger.info(f"⚡ 라이브러리 지원:")
logger.info(f"   - NumPy: {'✅' if NUMPY_AVAILABLE else '❌'}")
logger.info(f"   - PIL/Pillow: {'✅' if PIL_AVAILABLE else '❌'}")
logger.info(f"   - OpenCV: {'✅' if CV2_AVAILABLE else '❌'}")
logger.info(f"   - PyTorch: {'✅' if TORCH_AVAILABLE else '❌'}")

# ==============================================
# 🔥 사용 예시 (주석)
# ==============================================

"""
🎯 사용 예시:

# 1. 기본 이미지 전처리
from backend.app.ai_pipeline.utils.image_processing import preprocess_image

processed = preprocess_image(
    image='path/to/image.jpg',
    target_size=(512, 512),
    normalize=True,
    to_tensor=True
)

# 2. 세그멘테이션 후처리
from backend.app.ai_pipeline.utils.image_processing import postprocess_segmentation

binary_mask = postprocess_segmentation(model_output, threshold=0.5)

# 3. 텐서 ↔ PIL 변환
from backend.app.ai_pipeline.utils.image_processing import tensor_to_pil, pil_to_tensor

pil_image = tensor_to_pil(tensor)
tensor = pil_to_tensor(pil_image, device='mps')

# 4. Base64 변환
from backend.app.ai_pipeline.utils.image_processing import image_to_base64, base64_to_image

base64_str = image_to_base64(image, format='JPEG', quality=95)
image = base64_to_image(base64_str)

# 5. 배치 생성
from backend.app.ai_pipeline.utils.image_processing import create_batch

batch_tensor = create_batch([image1, image2, image3], device='mps')

# 6. 이미지 향상
from backend.app.ai_pipeline.utils.image_processing import enhance_image_contrast

enhanced = enhance_image_contrast(image, factor=1.2)
"""️ 폴백 처리 - 기본 크기의 제로 배열 생성")
            if NUMPY_AVAILABLE:
                img_array = np.zeros((target_size[1], target_size[0], 3), dtype=np.float32)
            else:
                img_array = [[[0.0, 0.0, 0.0] for _ in range(target_size[0])] for _ in range(target_size[1])]
        
        # 5. 정규화
        if normalize:
            logger.debug("이미지 정규화 적용")
            if NUMPY_AVAILABLE and hasattr(img_array, 'dtype'):
                if img_array.max() > 1.0:
                    img_array = img_array / 255.0
            elif isinstance(img_array, list):
                # 리스트 형태인 경우
                for i, row in enumerate(img_array):
                    for j, pixel in enumerate(row):
                        img_array[i][j] = [p/255.0 if p > 1.0 else p for p in pixel]
        
        # 6. PyTorch tensor 변환
        if to_tensor and TORCH_AVAILABLE:
            logger.debug("PyTorch 텐서로 변환")
            if NUMPY_AVAILABLE and hasattr(img_array, 'shape'):
                # numpy array → tensor
                img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)  # NHWC → NCHW
                img_tensor = img_tensor.to(device)
                logger.debug(f"✅ 텐서 변환 완료: {img_tensor.shape}, device: {device}")
                return img_tensor
            else:
                # 리스트 → tensor
                if isinstance(img_array, list):
                    height = len(img_array)
                    width = len(img_array[0]) if height > 0 else target_size[0]
                    channels = len(img_array[0][0]) if height > 0 and width > 0 else 3
                    
                    tensor_data = torch.zeros(1, channels, height, width)
                    for h in range(height):
                        for w in range(width):
                            for c in range(channels):
                                if h < len(img_array) and w < len(img_array[h]) and c < len(img_array[h][w]):
                                    tensor_data[0, c, h, w] = img_array[h][w][c]
                    
                    tensor_data = tensor_data.to(device)
                    logger.debug(f"✅ 리스트→텐서 변환 완료: {tensor_data.shape}")
                    return tensor_data
        
        # 7. NumPy 배열 또는 리스트로 반환
        logger.debug(f"최종 반환: {type(img_array)}")
        return img_array
            
    except Exception as e:
        logger.error(f"❌ 이미지 전처리 실패: {e}")
        # 폴백: 기본 크기의 제로 데이터
        if to_tensor and TORCH_AVAILABLE:
            return torch.zeros(1, 3, target_size[1], target_size[0], device=device)
        elif NUMPY_AVAILABLE:
            return np.zeros((target_size[1], target_size[0], 3), dtype=np.float32)
        else:
            return [[[0.0, 0.0, 0.0] for _ in range(target_size[0])] for _ in range(target_size[1])]

def postprocess_segmentation(output: Any, threshold: float = 0.5) -> Any:
    """
    세그멘테이션 결과 후처리 함수 - 완전 구현
    
    Args:
        output: 모델 출력 (tensor, numpy array, 또는 리스트)
        threshold: 이진화 임계값
    
    Returns:
        후처리된 마스크 (0-255 값의 이미지)
    """
    try:
        logger.debug(f"세그멘테이션 후처리 시작: {type(output)}")
        
        # 1. PyTorch tensor 처리
        if TORCH_AVAILABLE and hasattr(output, 'cpu'):
            output_np = output.cpu().numpy()
            logger.debug("PyTorch 텐서 → NumPy 변환")
        elif TORCH_AVAILABLE and hasattr(output, 'detach'):
            output_np = output.detach().cpu().numpy()
            logger.debug("PyTorch 텐서 (gradient) → NumPy 변환")
        elif NUMPY_AVAILABLE and hasattr(output, 'shape'):
            output_np = output
            logger.debug("NumPy 배열 사용")
        else:
            # 리스트나 기타 형태
            output_np = output
            logger.debug("리스트/기타 형태 처리")
        
        # 2. 차원 조정
        if NUMPY_AVAILABLE and hasattr(output_np, 'shape'):
            logger.debug(f"원본 shape: {output_np.shape}")
            
            # 배치 차원 제거
            if output_np.ndim == 4:  # (N, C, H, W)
                output_np = output_np.squeeze(0)
                logger.debug(f"배치 차원 제거: {output_np.shape}")
            
            if output_np.ndim == 3:  # (C, H, W)
                if output_np.shape[0] == 1:  # 단일 채널
                    output_np = output_np.squeeze(0)
                    logger.debug(f"채널 차원 제거: {output_np.shape}")
                else:  # 다중 채널인 경우 첫 번째 채널 사용
                    output_np = output_np[0]
                    logger.debug(f"첫 번째 채널 선택: {output_np.shape}")
            
            # 3. 이진화 적용
            binary_mask = (output_np > threshold).astype(np.uint8) * 255
            logger.debug(f"이진화 완료: {binary_mask.shape}, 값 범위: {binary_mask.min()}-{binary_mask.max()}")
            
            return binary_mask
        
        else:
            # NumPy 없는 경우 리스트 처리
            logger.debug("리스트 기반 후처리")
            
            def process_value(val):
                if isinstance(val, (list, tuple)):
                    # 중첩 구조인 경우 재귀적으로 처리
                    return [process_value(v) for v in val]
                else:
                    # 단일 값 처리
                    return 255 if float(val) > threshold else 0
            
            if isinstance(output, (list, tuple)):
                # 중첩 리스트 구조 처리
                if len(output) > 0 and isinstance(output[0], (list, tuple)):
                    # 2D 이상 구조
                    if len(output[0]) > 0 and isinstance(output[0][0], (list, tuple)):
                        # 3D 구조 (첫 번째 채널 사용)
                        output = output[0] if isinstance(output[0][0], (list, tuple)) else output
                    
                    result = [[255 if float(pixel) > threshold else 0 for pixel in row] for row in output]
                    logger.debug("2D 리스트 후처리 완료")
                    return result
                else:
                    # 1D 구조
                    result = [255 if float(val) > threshold else 0 for val in output]
                    logger.debug("1D 리스트 후처리 완료")
                    return result
            else:
                # 단일 값
                result = 255 if float(output) > threshold else 0
                logger.debug("단일 값 후처리 완료")
                return result
            
    except Exception as e:
        logger.error(f"❌ 세그멘테이션 후처리 실패: {e}")
        # 폴백: 기본 크기의 제로 마스크
        if NUMPY_AVAILABLE:
            return np.zeros((512, 512), dtype=np.uint8)
        else:
            return [[0 for _ in range(512)] for _ in range(512)]

# ==============================================
# 🔥 특화된 전처리 함수들
# ==============================================

def preprocess_pose_input(image: Any, target_size: Tuple[int, int] = (368, 368)) -> Any:
    """포즈 추정용 이미지 전처리"""
    return preprocess_image(image, target_size, normalize=True, to_tensor=True)

def preprocess_human_parsing_input(image: Any, target_size: Tuple[int, int] = (512, 512)) -> Any:
    """인체 파싱용 이미지 전처리"""
    return preprocess_image(image, target_size, normalize=True, to_tensor=True)

def preprocess_cloth_segmentation_input(image: Any, target_size: Tuple[int, int] = (320, 320)) -> Any:
    """의류 세그멘테이션용 이미지 전처리"""
    return preprocess_image(image, target_size, normalize=True, to_tensor=True)

def preprocess_virtual_fitting_input(person_img: Any, cloth_img: Any, target_size: Tuple[int, int] = (512, 512)) -> Tuple[Any, Any]:
    """가상 피팅용 이미지 전처리"""
    person_tensor = preprocess_image(person_img, target_size, normalize=True, to_tensor=True)
    cloth_tensor = preprocess_image(cloth_img, target_size, normalize=True, to_tensor=True)
    return person_tensor, cloth_tensor

# ==============================================
# 🔥 이미지 변환 함수들
# ==============================================

def tensor_to_pil(tensor: Any) -> Any:
    """
    텐서를 PIL 이미지로 변환
    
    Args:
        tensor: PyTorch tensor (C, H, W) 또는 (N, C, H, W)
    
    Returns:
        PIL Image 또는 numpy array
    """
    try:
        logger.debug(f"텐서→PIL 변환 시작: {type(tensor)}")
        
        if not TORCH_AVAILABLE:
            logger.warning("⚠️ PyTorch 없음, 원본 반환")
            return tensor
        
        # tensor 처리
        if hasattr(tensor, 'dim'):
            logger.debug(f"텐서 차원: {tensor.dim()}, 크기: {tensor.shape}")
            
            if tensor.dim() == 4:  # (N, C, H, W)
                tensor = tensor.squeeze(0)
                logger.debug(f"배치 차원 제거: {tensor.shape}")
            
            if tensor.dim() == 3:  # (C, H, W)
                tensor = tensor.permute(1, 2, 0)  # (H, W, C)
                logger.debug(f"차원 순서 변경: {tensor.shape}")
            
            # CPU로 이동
            if hasattr(tensor, 'cpu'):
                tensor = tensor.cpu()
                logger.debug("CPU로 이동")
            
            # numpy 변환
            if hasattr(tensor, 'numpy'):
                tensor_np = tensor.numpy()
                logger.debug("NumPy 변환 완료")
            elif hasattr(tensor, 'detach'):
                tensor_np = tensor.detach().numpy()
                logger.debug("Detach 후 NumPy 변환 완료")
            else:
                tensor_np = tensor
        else:
            tensor_np = tensor
        
        # 값 범위 조정
        if NUMPY_AVAILABLE and hasattr(tensor_np, 'dtype'):
            logger.debug(f"값 범위 조정: dtype={tensor_np.dtype}, 범위={tensor_np.min():.3f}-{tensor_np.max():.3f}")
            
            if tensor_np.dtype != np.uint8:
                # 0-1 범위를 0-255로 변환
                if tensor_np.max() <= 1.0:
                    tensor_np = (tensor_np * 255).astype(np.uint8)
                    logger.debug("0-1 → 0-255 변환")
                else:
                    tensor_np = np.clip(tensor_np, 0, 255).astype(np.uint8)
                    logger.debug("클리핑 후 uint8 변환")
        
        # PIL Image 생성
        if PIL_AVAILABLE:
            try:
                if NUMPY_AVAILABLE and hasattr(tensor_np, 'shape'):
                    if len(tensor_np.shape) == 3 and tensor_np.shape[2] == 3:
                        pil_image = Image.fromarray(tensor_np, 'RGB')
                        logger.debug("✅ PIL RGB 이미지 생성 완료")
                        return pil_image
                    elif len(tensor_np.shape) == 2:
                        pil_image = Image.fromarray(tensor_np, 'L')
                        logger.debug("✅ PIL 그레이스케일 이미지 생성 완료")
                        return pil_image
                    else:
                        logger.warning(f"⚠️ 지원하지 않는 shape: {tensor_np.shape}")
                        return tensor_np
                else:
                    # NumPy 없는 경우 기본 처리
                    logger.debug("NumPy 없음, 원본 반환")
                    return tensor_np
            except Exception as e:
                logger.error(f"❌ PIL 이미지 생성 실패: {e}")
                return tensor_np
        else:
            logger.warning("⚠️ PIL 없음, NumPy 배열 반환")
            return tensor_np
            
    except Exception as e:
        logger.error(f"❌ tensor→PIL 변환 실패: {e}")
        return None

def pil_to_tensor(image: Any, device: str = "mps") -> Any:
    """
    PIL 이미지를 텐서로 변환
    
    Args:
        image: PIL Image 또는 numpy array
        device: 대상 디바이스
    
    Returns:
        PyTorch tensor (N, C, H, W)
    """
    try:
        logger.debug(f"PIL→텐서 변환 시작: {type(image)}")
        
        if not TORCH_AVAILABLE:
            logger.warning("⚠️ PyTorch 없음, 원본 반환")
            return image
        
        # PIL Image 처리
        if hasattr(image, 'size'):  # PIL Image
            width, height = image.size
            logger.debug(f"PIL 이미지 크기: {width}x{height}")
            
            if NUMPY_AVAILABLE:
                img_array = np.array(image).astype(np.float32) / 255.0
                
                if len(img_array.shape) == 3:  # RGB
                    tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)  # (H,W,C) → (N,C,H,W)
                else:  # 그레이스케일
                    tensor = torch.from_numpy(img_array).unsqueeze(0).unsqueeze(0)  # (H,W) → (N,C,H,W)
                
                tensor = tensor.to(device)
                logger.debug(f"✅ PIL→텐서 변환 완료: {tensor.shape}, device: {device}")
                return tensor
            else:
                # NumPy 없는 경우 수동 변환
                if image.mode == 'RGB':
                    channels = 3
                elif image.mode == 'L':
                    channels = 1
                else:
                    channels = 3
                    image = image.convert('RGB')
                
                tensor = torch.zeros(1, channels, height, width, device=device)
                
                for y in range(height):
                    for x in range(width):
                        pixel = image.getpixel((x, y))
                        if isinstance(pixel, int):  # 그레이스케일
                            tensor[0, 0, y, x] = pixel / 255.0
                        else:  # RGB
                            for c, val in enumerate(pixel[:channels]):
                                tensor[0, c, y, x] = val / 255.0
                
                logger.debug(f"✅ 수동 PIL→텐서 변환 완료: {tensor.shape}")
                return tensor
        
        # numpy array 처리
        elif NUMPY_AVAILABLE and hasattr(image, 'shape'):
            logger.debug(f"NumPy 배열 처리: {image.shape}")
            
            img_array = image.astype(np.float32)
            if img_array.max() > 1.0:
                img_array = img_array / 255.0
            
            if len(image.shape) == 3:  # (H, W, C)
                tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)
            elif len(image.shape) == 2:  # (H, W) 그레이스케일
                tensor = torch.from_numpy(img_array).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
            else:
                raise ValueError(f"지원하지 않는 배열 차원: {image.shape}")
            
            tensor = tensor.to(device)
            logger.debug(f"✅ NumPy→텐서 변환 완료: {tensor.shape}")
            return tensor
        
        # 폴백: 기본 텐서
        logger.warning("⚠️ 변환 실패, 기본 텐서 반환")
        return torch.zeros(1, 3, 512, 512, device=device)
            
    except Exception as e:
        logger.error(f"❌ PIL→tensor 변환 실패: {e}")
        if TORCH_AVAILABLE:
            return torch.zeros(1, 3, 512, 512, device=device)
        else:
            return None

# ==============================================
# 🔥 이미지 유틸리티 함수들
# ==============================================

def resize_image(image: Any, target_size: Tuple[int, int]) -> Any:
    """이미지 크기 조정"""
    try:
        logger.debug(f"이미지 크기 조정: {type(image)} → {target_size}")
        
        if hasattr(image, 'resize'):  # PIL Image
            resized = image.resize(target_size, Image.Resampling.LANCZOS if hasattr(Image, 'Resampling') else Image.LANCZOS)
            logger.debug("✅ PIL 크기 조정 완료")
            return resized
        elif CV2_AVAILABLE and NUMPY_AVAILABLE and hasattr(image, 'shape'):
            resized = cv2.resize(image, target_size)
            logger.debug("✅ OpenCV 크기 조정 완료")
            return resized
        else:
            # 기본 처리 (크기 조정 없이 반환)
            logger.warning("⚠️ 크기 조정 불가, 원본 반환")
            return image
    except Exception as e:
        logger.error(f"❌ 이미지 크기 조정 실패: {e}")
        return image

def normalize_image(image: Any, mean: Tuple[float, float, float] = (0.485, 0.456, 0.406), 
                   std: Tuple[float, float, float] = (0.229, 0.224, 0.225)) -> Any:
    """이미지 정규화 (ImageNet 기본값)"""
    try:
        logger.debug(f"이미지 정규화: mean={mean}, std={std}")
        
        if TORCH_AVAILABLE and hasattr(image, 'dim'):
            # PyTorch tensor
            image_normalized = image.clone()
            if image_normalized.dim() == 4:  # (N, C, H, W)
                for i in range(3):
                    image_normalized[:, i, :, :] = (image_normalized[:, i, :, :] - mean[i]) / std[i]
            elif image_normalized.dim() == 3:  # (C, H, W)
                for i in range(3):
                    image_normalized[i, :, :] = (image_normalized[i, :, :] - mean[i]) / std[i]
            logger.debug("✅ PyTorch 텐서 정규화 완료")
            return image_normalized
        elif NUMPY_AVAILABLE and hasattr(image, 'shape'):
            # numpy array
            image_normalized = image.astype(np.float32).copy()
            if len(image.shape) == 4:  # (N, H, W, C)
                for i in range(3):
                    image_normalized[:, :, :, i] = (image_normalized[:, :, :, i] - mean[i]) / std[i]
            elif len(image.shape) == 3:  # (H, W, C)
                for i in range(3):
                    image_normalized[:, :, i] = (image_normalized[:, :, i] - mean[i]) / std[i]
            logger.debug("✅ NumPy 배열 정규화 완료")
            return image_normalized
        else:
            logger.warning("⚠