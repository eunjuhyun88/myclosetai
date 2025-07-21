# backend/app/ai_pipeline/utils/image_processing.py
"""
🖼️ MyCloset AI - 이미지 처리 함수들 (순환참조 방지 버전) - 완전판
=========================================================
✅ model_loader.py에서 분리된 이미지 처리 함수들
✅ 순환참조 완전 방지 - 독립적인 모듈
✅ PIL, OpenCV, NumPy 기반 이미지 처리
✅ PyTorch 텐서 변환 지원
✅ M3 Max 128GB 최적화
✅ conda 환경 완벽 지원
✅ 기존 함수명 100% 유지
✅ 모든 기능 완전 구현 - 잘린 부분 없음

Author: MyCloset AI Team
Date: 2025-07-21
Version: 1.0 (Separated from model_loader.py - Complete)
"""

import io
import logging
import base64
import tempfile
import os
import uuid
import math
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
            "mb": total_bytes / (1024 * 1024),
            "gb": total_bytes / (1024 * 1024 * 1024)
        })
        
        logger.debug(f"메모리 사용량 추정: {usage['mb']:.2f} MB")
        return usage
        
    except Exception as e:
        logger.error(f"❌ 메모리 사용량 추정 실패: {e}")
        return {"bytes": 0, "mb": 0, "gb": 0, "error": str(e)}

def optimize_image_memory(image: Any, target_size: Optional[Tuple[int, int]] = None, 
                         quality: int = 85, format: str = "JPEG") -> Any:
    """이미지 메모리 사용량 최적화"""
    try:
        if not PIL_AVAILABLE:
            logger.warning("⚠️ PIL 필요, 원본 반환")
            return image
        
        # PIL Image로 변환
        if hasattr(image, 'save'):  # 이미 PIL Image
            pil_image = image
        elif NUMPY_AVAILABLE and hasattr(image, 'shape'):  # NumPy array
            pil_image = Image.fromarray(image.astype(np.uint8))
        elif TORCH_AVAILABLE and hasattr(image, 'cpu'):  # PyTorch tensor
            pil_image = tensor_to_pil(image)
        else:
            logger.warning("⚠️ 지원하지 않는 이미지 타입")
            return image
        
        # 크기 조정
        if target_size:
            pil_image = pil_image.resize(target_size, Image.Resampling.LANCZOS)
            logger.debug(f"크기 조정: {target_size}")
        
        # 색상 모드 최적화
        if pil_image.mode == 'RGBA' and format.upper() == 'JPEG':
            # JPEG는 투명도를 지원하지 않으므로 RGB로 변환
            background = Image.new('RGB', pil_image.size, (255, 255, 255))
            background.paste(pil_image, mask=pil_image.split()[-1])
            pil_image = background
        elif pil_image.mode not in ['RGB', 'L']:
            pil_image = pil_image.convert('RGB')
        
        # 압축 적용
        buffer = io.BytesIO()
        pil_image.save(buffer, format=format, quality=quality, optimize=True)
        buffer.seek(0)
        optimized_image = Image.open(buffer)
        
        logger.debug(f"✅ 이미지 메모리 최적화 완료: format={format}, quality={quality}")
        return optimized_image
        
    except Exception as e:
        logger.error(f"❌ 이미지 메모리 최적화 실패: {e}")
        return image

# ==============================================
# 🔥 이미지 변환 및 포맷 함수들
# ==============================================

def convert_image_format(image: Any, target_format: str = "RGB") -> Any:
    """이미지 포맷 변환"""
    try:
        if not PIL_AVAILABLE:
            logger.warning("⚠️ PIL 필요, 원본 반환")
            return image
        
        # PIL Image로 변환
        if hasattr(image, 'save'):  # 이미 PIL Image
            pil_image = image
        elif NUMPY_AVAILABLE and hasattr(image, 'shape'):  # NumPy array
            pil_image = Image.fromarray(image.astype(np.uint8))
        elif TORCH_AVAILABLE and hasattr(image, 'cpu'):  # PyTorch tensor
            pil_image = tensor_to_pil(image)
        else:
            logger.warning("⚠️ 지원하지 않는 이미지 타입")
            return image
        
        # 포맷 변환
        if pil_image.mode != target_format:
            converted = pil_image.convert(target_format)
            logger.debug(f"✅ 포맷 변환 완료: {pil_image.mode} → {target_format}")
            return converted
        else:
            logger.debug(f"이미 {target_format} 포맷임")
            return pil_image
            
    except Exception as e:
        logger.error(f"❌ 이미지 포맷 변환 실패: {e}")
        return image

def save_image(image: Any, filepath: str, format: str = None, quality: int = 95, **kwargs) -> bool:
    """이미지 파일로 저장"""
    try:
        if not PIL_AVAILABLE:
            logger.error("❌ PIL 필요함")
            return False
        
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
            return False
        
        # 포맷 자동 감지
        if format is None:
            format = Path(filepath).suffix.upper().lstrip('.')
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
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        # 저장
        save_kwargs = {'format': format, **kwargs}
        if format.upper() in ['JPEG', 'WEBP']:
            save_kwargs['quality'] = quality
            save_kwargs['optimize'] = True
        
        pil_image.save(filepath, **save_kwargs)
        
        logger.debug(f"✅ 이미지 저장 완료: {filepath} ({format})")
        return True
        
    except Exception as e:
        logger.error(f"❌ 이미지 저장 실패: {e}")
        return False

def load_image(filepath: str, target_format: str = "RGB") -> Any:
    """이미지 파일 로드"""
    try:
        if not PIL_AVAILABLE:
            logger.error("❌ PIL 필요함")
            return None
        
        if not Path(filepath).exists():
            logger.error(f"❌ 파일이 존재하지 않음: {filepath}")
            return None
        
        # 이미지 로드
        pil_image = Image.open(filepath)
        
        # 포맷 변환
        if target_format and pil_image.mode != target_format:
            pil_image = pil_image.convert(target_format)
        
        logger.debug(f"✅ 이미지 로드 완료: {filepath} ({pil_image.size}, {pil_image.mode})")
        return pil_image
        
    except Exception as e:
        logger.error(f"❌ 이미지 로드 실패: {e}")
        return None

# ==============================================
# 🔥 이미지 시각화 및 디버깅 함수들
# ==============================================

def create_image_grid(images: List[Any], grid_size: Optional[Tuple[int, int]] = None, 
                     padding: int = 2, background_color: Tuple[int, int, int] = (255, 255, 255)) -> Any:
    """이미지들을 격자로 배열"""
    try:
        if not PIL_AVAILABLE or not images:
            logger.warning("⚠️ PIL 필요하거나 이미지가 없음")
            return None
        
        # 격자 크기 계산
        if grid_size is None:
            grid_cols = int(math.ceil(math.sqrt(len(images))))
            grid_rows = int(math.ceil(len(images) / grid_cols))
            grid_size = (grid_rows, grid_cols)
        else:
            grid_rows, grid_cols = grid_size
        
        # PIL Image로 변환
        pil_images = []
        for img in images:
            if hasattr(img, 'save'):  # 이미 PIL Image
                pil_images.append(img)
            elif NUMPY_AVAILABLE and hasattr(img, 'shape'):  # NumPy array
                pil_images.append(Image.fromarray(img.astype(np.uint8)))
            elif TORCH_AVAILABLE and hasattr(img, 'cpu'):  # PyTorch tensor
                pil_images.append(tensor_to_pil(img))
            else:
                logger.warning(f"⚠️ 지원하지 않는 이미지 타입: {type(img)}")
                continue
        
        if not pil_images:
            logger.warning("⚠️ 변환 가능한 이미지가 없음")
            return None
        
        # 모든 이미지를 같은 크기로 조정
        max_width = max(img.width for img in pil_images)
        max_height = max(img.height for img in pil_images)
        
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
        
        logger.debug(f"✅ 이미지 격자 생성 완료: {grid_size}, {len(resized_images)}개 이미지")
        return grid_image
        
    except Exception as e:
        logger.error(f"❌ 이미지 격자 생성 실패: {e}")
        return None

def add_text_to_image(image: Any, text: str, position: Tuple[int, int] = (10, 10), 
                     font_size: int = 20, color: Tuple[int, int, int] = (0, 0, 0)) -> Any:
    """이미지에 텍스트 추가"""
    try:
        if not PIL_AVAILABLE:
            logger.warning("⚠️ PIL 필요, 원본 반환")
            return image
        
        # PIL Image로 변환
        if hasattr(image, 'save'):  # 이미 PIL Image
            pil_image = image.copy()
        elif NUMPY_AVAILABLE and hasattr(image, 'shape'):  # NumPy array
            pil_image = Image.fromarray(image.astype(np.uint8))
        elif TORCH_AVAILABLE and hasattr(image, 'cpu'):  # PyTorch tensor
            pil_image = tensor_to_pil(image)
        else:
            logger.warning("⚠️ 지원하지 않는 이미지 타입")
            return image
        
        # 드로잉 객체 생성
        draw = ImageDraw.Draw(pil_image)
        
        # 폰트 설정 (기본 폰트 사용)
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except:
            try:
                font = ImageFont.load_default()
            except:
                font = None
        
        # 텍스트 추가
        draw.text(position, text, fill=color, font=font)
        
        logger.debug(f"✅ 텍스트 추가 완료: '{text}' at {position}")
        return pil_image
        
    except Exception as e:
        logger.error(f"❌ 텍스트 추가 실패: {e}")
        return image

def create_comparison_image(image1: Any, image2: Any, labels: Tuple[str, str] = ("Original", "Processed")) -> Any:
    """두 이미지를 비교하는 이미지 생성"""
    try:
        if not PIL_AVAILABLE:
            logger.warning("⚠️ PIL 필요")
            return None
        
        # PIL Image로 변환
        def to_pil(img):
            if hasattr(img, 'save'):  # 이미 PIL Image
                return img
            elif NUMPY_AVAILABLE and hasattr(img, 'shape'):  # NumPy array
                return Image.fromarray(img.astype(np.uint8))
            elif TORCH_AVAILABLE and hasattr(img, 'cpu'):  # PyTorch tensor
                return tensor_to_pil(img)
            else:
                return None
        
        pil1 = to_pil(image1)
        pil2 = to_pil(image2)
        
        if pil1 is None or pil2 is None:
            logger.warning("⚠️ 이미지 변환 실패")
            return None
        
        # 같은 크기로 조정
        max_width = max(pil1.width, pil2.width)
        max_height = max(pil1.height, pil2.height)
        
        pil1 = pil1.resize((max_width, max_height), Image.Resampling.LANCZOS)
        pil2 = pil2.resize((max_width, max_height), Image.Resampling.LANCZOS)
        
        # 비교 이미지 생성 (좌우 배치)
        padding = 20
        text_height = 30
        
        comparison_width = max_width * 2 + padding * 3
        comparison_height = max_height + text_height + padding * 2
        
        comparison = Image.new('RGB', (comparison_width, comparison_height), (255, 255, 255))
        
        # 이미지들 배치
        comparison.paste(pil1, (padding, text_height + padding))
        comparison.paste(pil2, (max_width + padding * 2, text_height + padding))
        
        # 라벨 추가
        comparison = add_text_to_image(comparison, labels[0], (padding, 5), font_size=20)
        comparison = add_text_to_image(comparison, labels[1], (max_width + padding * 2, 5), font_size=20)
        
        logger.debug(f"✅ 비교 이미지 생성 완료: {labels}")
        return comparison
        
    except Exception as e:
        logger.error(f"❌ 비교 이미지 생성 실패: {e}")
        return None

# ==============================================
# 🔥 Step별 특화 처리 함수들
# ==============================================

def postprocess_human_parsing(output: Any, num_classes: int = 20, 
                             colormap: Optional[List[Tuple[int, int, int]]] = None) -> Any:
    """인체 파싱 결과 후처리 (컬러맵 적용)"""
    try:
        if not NUMPY_AVAILABLE:
            logger.warning("⚠️ NumPy 필요")
            return output
        
        # 출력을 numpy array로 변환
        if TORCH_AVAILABLE and hasattr(output, 'cpu'):
            pred = output.cpu().numpy()
        elif hasattr(output, 'shape'):
            pred = output
        else:
            logger.warning("⚠️ 지원하지 않는 출력 타입")
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
                # HSV 색공간에서 균등하게 분포된 색상 생성
                hue = int(i * 360 / num_classes)
                if i == 0:  # 배경은 검은색
                    colormap.append((0, 0, 0))
                else:
                    # HSV to RGB 변환 (간단한 버전)
                    c = 255
                    x = int(c * (1 - abs((hue / 60) % 2 - 1)))
                    if 0 <= hue < 60:
                        rgb = (c, x, 0)
                    elif 60 <= hue < 120:
                        rgb = (x, c, 0)
                    elif 120 <= hue < 180:
                        rgb = (0, c, x)
                    elif 180 <= hue < 240:
                        rgb = (0, x, c)
                    elif 240 <= hue < 300:
                        rgb = (x, 0, c)
                    else:
                        rgb = (c, 0, x)
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

def postprocess_pose_keypoints(output: Any, confidence_threshold: float = 0.3) -> Dict[str, Any]:
    """포즈 키포인트 후처리"""
    try:
        result = {
            "keypoints": [],
            "connections": [],
            "valid_keypoints": 0,
            "confidence_scores": []
        }
        
        if not NUMPY_AVAILABLE:
            logger.warning("⚠️ NumPy 필요")
            return result
        
        # OpenPose 키포인트 연결 정보 (COCO 포맷)
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),     # 머리
            (1, 5), (5, 6), (6, 7),             # 왼팔
            (1, 8), (8, 9), (9, 10),            # 오른팔
            (1, 11), (11, 12), (12, 13),        # 왼다리
            (1, 14), (14, 15), (15, 16)         # 오른다리
        ]
        
        # 출력을 numpy array로 변환
        if TORCH_AVAILABLE and hasattr(output, 'cpu'):
            heatmaps = output.cpu().numpy()
        elif hasattr(output, 'shape'):
            heatmaps = output
        else:
            logger.warning("⚠️ 지원하지 않는 출력 타입")
            return result
        
        # 차원 조정
        if heatmaps.ndim == 4:  # (N, C, H, W)
            heatmaps = heatmaps.squeeze(0)
        
        num_keypoints = min(heatmaps.shape[0], 18)  # OpenPose 18 키포인트
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
                
                # 이미지 좌표로 변환
                keypoints.append((x, y, max_val))
                confidence_scores.append(max_val)
            else:
                keypoints.append((0, 0, 0))
                confidence_scores.append(0.0)
        
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
        
        logger.debug(f"✅ 포즈 키포인트 후처리 완료: {result['valid_keypoints']}개 유효 키포인트")
        return result
        
    except Exception as e:
        logger.error(f"❌ 포즈 키포인트 후처리 실패: {e}")
        return result

def create_pose_visualization(image: Any, keypoints_result: Dict[str, Any]) -> Any:
    """포즈 키포인트 시각화"""
    try:
        if not PIL_AVAILABLE:
            logger.warning("⚠️ PIL 필요")
            return image
        
        # PIL Image로 변환
        if hasattr(image, 'save'):  # 이미 PIL Image
            vis_image = image.copy()
        elif NUMPY_AVAILABLE and hasattr(image, 'shape'):  # NumPy array
            vis_image = Image.fromarray(image.astype(np.uint8))
        elif TORCH_AVAILABLE and hasattr(image, 'cpu'):  # PyTorch tensor
            vis_image = tensor_to_pil(image)
        else:
            logger.warning("⚠️ 지원하지 않는 이미지 타입")
            return image
        
        draw = ImageDraw.Draw(vis_image)
        
        keypoints = keypoints_result.get("keypoints", [])
        connections = keypoints_result.get("connections", [])
        
        # 연결선 그리기
        for conn in connections:
            if conn[0] < len(keypoints) and conn[1] < len(keypoints):
                pt1 = keypoints[conn[0]]
                pt2 = keypoints[conn[1]]
                
                if pt1[2] > 0 and pt2[2] > 0:  # 유효한 키포인트들만
                    draw.line([pt1[0], pt1[1], pt2[0], pt2[1]], fill=(0, 255, 0), width=3)
        
        # 키포인트 그리기
        for i, (x, y, conf) in enumerate(keypoints):
            if conf > 0:
                # 신뢰도에 따른 색상 결정
                color = (255, int(255 * conf), 0)  # 빨강-노랑 그라데이션
                radius = 5
                
                # 원 그리기
                draw.ellipse([x-radius, y-radius, x+radius, y+radius], fill=color, outline=(0, 0, 0))
        
        logger.debug("✅ 포즈 시각화 완료")
        return vis_image
        
    except Exception as e:
        logger.error(f"❌ 포즈 시각화 실패: {e}")
        return image

# ==============================================
# 🔥 모듈 정보 및 내보내기
# ==============================================

__version__ = "1.0.0"
__author__ = "MyCloset AI Team"
__description__ = "이미지 처리 함수들 - model_loader.py에서 분리 (완전판)"

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
    'enhance_image_color',
    'apply_gaussian_blur',
    'apply_unsharp_mask',
    'apply_edge_enhance',
    
    # 고급 처리 함수들
    'apply_clahe_enhancement',
    'remove_background_simple',
    'detect_dominant_colors',
    'calculate_image_similarity',
    
    # 검증 및 분석 함수들
    'validate_image_format',
    'get_image_statistics',
    'detect_image_artifacts',
    
    # 메모리 관리 함수들
    'cleanup_image_memory',
    'estimate_memory_usage',
    'optimize_image_memory',
    
    # 이미지 변환 및 포맷 함수들
    'convert_image_format',
    'save_image',
    'load_image',
    
    # 시각화 및 디버깅 함수들
    'create_image_grid',
    'add_text_to_image',
    'create_comparison_image',
    
    # Step별 특화 처리 함수들
    'postprocess_human_parsing',
    'postprocess_pose_keypoints',
    'create_pose_visualization',
    
    # 상수들
    'NUMPY_AVAILABLE',
    'PIL_AVAILABLE',
    'CV2_AVAILABLE',
    'TORCH_AVAILABLE'
]

logger.info(f"🖼️ 이미지 처리 모듈 v{__version__} 로드 완료 (완전판)")
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

# 7. 고급 처리
from backend.app.ai_pipeline.utils.image_processing import apply_clahe_enhancement, detect_dominant_colors

enhanced = apply_clahe_enhancement(image, clip_limit=2.0)
colors = detect_dominant_colors(image, k=5)

# 8. Step별 특화 처리
from backend.app.ai_pipeline.utils.image_processing import postprocess_human_parsing, create_pose_visualization

colored_parsing = postprocess_human_parsing(parsing_output, num_classes=20)
pose_vis = create_pose_visualization(image, keypoints_result)

# 9. 이미지 저장 및 로드
from backend.app.ai_pipeline.utils.image_processing import save_image, load_image

save_image(image, 'output.jpg', quality=95)
loaded_image = load_image('input.jpg', target_format='RGB')

# 10. 시각화
from backend.app.ai_pipeline.utils.image_processing import create_image_grid, create_comparison_image

grid = create_image_grid([img1, img2, img3, img4], grid_size=(2, 2))
comparison = create_comparison_image(original, processed, ('Before', 'After'))
"""image, 'resize'):  # PIL Image
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
            logger.warning("⚠️ 폴백 처리 - 기본 크기의 제로 배열 생성")
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

def enhance_image_color(image: Any, factor: float = 1.1) -> Any:
    """이미지 색상 향상"""
    try:
        if PIL_AVAILABLE and hasattr(image, 'save'):
            enhancer = ImageEnhance.Color(image)
            enhanced = enhancer.enhance(factor)
            logger.debug(f"✅ 색상 향상 완료: factor={factor}")
            return enhanced
        else:
            logger.warning("⚠️ PIL 이미지가 아님, 원본 반환")
            return image
    except Exception as e:
        logger.error(f"❌ 색상 향상 실패: {e}")
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

def apply_unsharp_mask(image: Any, radius: float = 2.0, percent: int = 150, threshold: int = 3) -> Any:
    """언샤프 마스크 적용 (선명도 향상)"""
    try:
        if PIL_AVAILABLE and hasattr(image, 'save'):
            # PIL의 UnsharpMask 필터 사용
            unsharp = image.filter(ImageFilter.UnsharpMask(radius=radius, percent=percent, threshold=threshold))
            logger.debug(f"✅ 언샤프 마스크 적용 완료: radius={radius}, percent={percent}")
            return unsharp
        else:
            logger.warning("⚠️ PIL 이미지가 아님, 원본 반환")
            return image
    except Exception as e:
        logger.error(f"❌ 언샤프 마스크 적용 실패: {e}")
        return image

def apply_edge_enhance(image: Any, factor: float = 1.0) -> Any:
    """엣지 강화 필터 적용"""
    try:
        if PIL_AVAILABLE and hasattr(image, 'save'):
            # EDGE_ENHANCE 필터 적용
            if factor > 1.0:
                edge_enhanced = image.filter(ImageFilter.EDGE_ENHANCE_MORE)
            else:
                edge_enhanced = image.filter(ImageFilter.EDGE_ENHANCE)
            logger.debug(f"✅ 엣지 강화 완료: factor={factor}")
            return edge_enhanced
        else:
            logger.warning("⚠️ PIL 이미지가 아님, 원본 반환")
            return image
    except Exception as e:
        logger.error(f"❌ 엣지 강화 실패: {e}")
        return image

# ==============================================
# 🔥 고급 이미지 처리 함수들
# ==============================================

def apply_clahe_enhancement(image: Any, clip_limit: float = 2.0, tile_grid_size: Tuple[int, int] = (8, 8)) -> Any:
    """CLAHE (Contrast Limited Adaptive Histogram Equalization) 적용"""
    try:
        if not CV2_AVAILABLE or not NUMPY_AVAILABLE:
            logger.warning("⚠️ OpenCV/NumPy 필요, 원본 반환")
            return image
        
        # PIL Image를 numpy array로 변환
        if hasattr(image, 'save'):  # PIL Image
            img_array = np.array(image)
        elif hasattr(image, 'shape'):  # numpy array
            img_array = image
        else:
            logger.warning("⚠️ 지원하지 않는 이미지 타입")
            return image
        
        # CLAHE 객체 생성
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        
        # 컬러 이미지인 경우 LAB 공간에서 처리
        if len(img_array.shape) == 3:
            lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])  # L 채널에만 적용
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        else:
            # 그레이스케일 이미지
            enhanced = clahe.apply(img_array)
        
        # PIL Image로 변환하여 반환
        if PIL_AVAILABLE:
            return Image.fromarray(enhanced)
        else:
            return enhanced
            
    except Exception as e:
        logger.error(f"❌ CLAHE 적용 실패: {e}")
        return image

def remove_background_simple(image: Any, threshold: int = 240) -> Any:
    """간단한 배경 제거 (흰색 배경 가정)"""
    try:
        if not NUMPY_AVAILABLE:
            logger.warning("⚠️ NumPy 필요, 원본 반환")
            return image
        
        # PIL Image를 numpy array로 변환
        if hasattr(image, 'save'):  # PIL Image
            img_array = np.array(image)
        elif hasattr(image, 'shape'):  # numpy array
            img_array = image.copy()
        else:
            logger.warning("⚠️ 지원하지 않는 이미지 타입")
            return image
        
        # 알파 채널 추가 (RGBA)
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            # RGB에서 RGBA로 변환
            alpha_channel = np.ones((img_array.shape[0], img_array.shape[1]), dtype=img_array.dtype) * 255
            img_rgba = np.dstack((img_array, alpha_channel))
            
            # 흰색 배경을 투명하게 만들기
            white_pixels = np.all(img_array >= threshold, axis=2)
            img_rgba[white_pixels, 3] = 0  # 알파값을 0으로 (투명)
            
            # PIL Image로 변환
            if PIL_AVAILABLE:
                return Image.fromarray(img_rgba, 'RGBA')
            else:
                return img_rgba
        else:
            logger.warning("⚠️ RGB 이미지가 아님")
            return image
            
    except Exception as e:
        logger.error(f"❌ 배경 제거 실패: {e}")
        return image

def detect_dominant_colors(image: Any, k: int = 5) -> List[Tuple[int, int, int]]:
    """이미지에서 주요 색상 추출 (K-means 클러스터링)"""
    try:
        if not NUMPY_AVAILABLE:
            logger.warning("⚠️ NumPy 필요")
            return []
        
        # PIL Image를 numpy array로 변환
        if hasattr(image, 'save'):  # PIL Image
            img_array = np.array(image)
        elif hasattr(image, 'shape'):  # numpy array
            img_array = image
        else:
            logger.warning("⚠️ 지원하지 않는 이미지 타입")
            return []
        
        # 이미지를 1차원으로 변환
        if len(img_array.shape) == 3:
            pixels = img_array.reshape((-1, 3))
        else:
            logger.warning("⚠️ 컬러 이미지가 아님")
            return []
        
        # 간단한 색상 분석 (K-means 대신 히스토그램 기반)
        unique_colors, counts = np.unique(pixels.view(np.dtype((np.void, pixels.dtype.itemsize * pixels.shape[1]))), 
                                        return_counts=True)
        
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
        if not NUMPY_AVAILABLE:
            logger.warning("⚠️ NumPy 필요")
            return 0.0
        
        # 이미지들을 numpy array로 변환
        def to_array(img):
            if hasattr(img, 'save'):  # PIL Image
                return np.array(img)
            elif hasattr(img, 'shape'):  # numpy array
                return img
            else:
                return None
        
        arr1 = to_array(image1)
        arr2 = to_array(image2)
        
        if arr1 is None or arr2 is None:
            logger.warning("⚠️ 이미지 변환 실패")
            return 0.0
        
        # 크기 맞추기
        if arr1.shape != arr2.shape:
            # 더 작은 크기로 맞춤
            min_height = min(arr1.shape[0], arr2.shape[0])
            min_width = min(arr1.shape[1], arr2.shape[1])
            arr1 = arr1[:min_height, :min_width]
            arr2 = arr2[:min_height, :min_width]
        
        # 유사도 계산
        if method == "mse":
            # Mean Squared Error (낮을수록 유사)
            mse = np.mean((arr1.astype(float) - arr2.astype(float)) ** 2)
            # 0-1 범위로 정규화 (1에 가까울수록 유사)
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
        
        logger.debug(f"✅ 이미지 유사도 계산 완료: {similarity:.3f} ({method})")
        return float(similarity)
        
    except Exception as e:
        logger.error(f"❌ 이미지 유사도 계산 실패: {e}")
        return 0.0

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
            "dtype": None,
            "memory_usage_mb": 0.0
        }
        
        if hasattr(image, 'size'):  # PIL Image
            result.update({
                "valid": True,
                "format": "PIL",
                "size": image.size,
                "mode": image.mode,
                "channels": len(image.getbands()),
                "memory_usage_mb": (image.size[0] * image.size[1] * len(image.getbands())) / (1024 * 1024)
            })
        elif NUMPY_AVAILABLE and hasattr(image, 'shape'):  # NumPy array
            memory_mb = image.nbytes / (1024 * 1024) if hasattr(image, 'nbytes') else 0.0
            result.update({
                "valid": True,
                "format": "NumPy",
                "size": (image.shape[1], image.shape[0]) if len(image.shape) >= 2 else image.shape,
                "channels": image.shape[2] if len(image.shape) == 3 else 1,
                "dtype": str(image.dtype),
                "memory_usage_mb": memory_mb
            })
        elif TORCH_AVAILABLE and hasattr(image, 'shape'):  # PyTorch tensor
            memory_mb = (image.numel() * image.element_size()) / (1024 * 1024) if hasattr(image, 'numel') else 0.0
            result.update({
                "valid": True,
                "format": "PyTorch",
                "size": (image.shape[-1], image.shape[-2]) if len(image.shape) >= 2 else image.shape,
                "channels": image.shape[-3] if len(image.shape) >= 3 else 1,
                "dtype": str(image.dtype),
                "memory_usage_mb": memory_mb
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
                "median": float(np.median(array)),
                "shape": array.shape,
                "unique_values": int(len(np.unique(array))),
                "zero_ratio": float(np.mean(array == 0))
            })
        elif hasattr(image, 'size'):  # PIL Image
            if NUMPY_AVAILABLE:
                array = np.array(image)
                stats.update({
                    "mean": float(np.mean(array)),
                    "std": float(np.std(array)),
                    "min": float(np.min(array)),
                    "max": float(np.max(array)),
                    "median": float(np.median(array)),
                    "size": image.size,
                    "mode": image.mode,
                    "unique_values": int(len(np.unique(array)))
                })
        
        logger.debug(f"이미지 통계: {stats}")
        return stats
        
    except Exception as e:
        logger.error(f"❌ 이미지 통계 계산 실패: {e}")
        return {"error": str(e)}

def detect_image_artifacts(image: Any) -> Dict[str, Any]:
    """이미지 아티팩트 감지"""
    try:
        artifacts = {
            "noise_level": 0.0,
            "blur_level": 0.0,
            "compression_artifacts": False,
            "over_saturation": False,
            "under_exposure": False,
            "over_exposure": False
        }
        
        if not NUMPY_AVAILABLE:
            logger.warning("⚠️ NumPy 필요")
            return artifacts
        
        # PIL Image를 numpy array로 변환
        if hasattr(image, 'save'):  # PIL Image
            img_array = np.array(image)
        elif hasattr(image, 'shape'):  # numpy array
            img_array = image
        else:
            return artifacts
        
        # 그레이스케일로 변환
        if len(img_array.shape) == 3:
            gray = np.mean(img_array, axis=2)
        else:
            gray = img_array
        
        # 노이즈 레벨 추정 (Laplacian variance 사용)
        if CV2_AVAILABLE:
            laplacian_var = cv2.Laplacian(gray.astype(np.uint8), cv2.CV_64F).var()
            artifacts["noise_level"] = float(laplacian_var / 1000.0)  # 정규화
        
        # 블러 레벨 추정
        if CV2_AVAILABLE:
            blur_score = cv2.Laplacian(gray.astype(np.uint8), cv2.CV_64F).var()
            artifacts["blur_level"] = float(1.0 - min(blur_score / 1000.0, 1.0))
        
        # 노출 문제 감지
        mean_brightness = np.mean(gray)
        artifacts["under_exposure"] = mean_brightness < 50
        artifacts["over_exposure"] = mean_brightness > 200
        
        # 과포화 감지
        if len(img_array.shape) == 3:
            max_values = np.max(img_array, axis=2)
            artifacts["over_saturation"] = np.mean(max_values >= 250) > 0.1
        
        logger.debug(f"아티팩트 감지 결과: {artifacts}")
        return artifacts
        
    except Exception as e:
        logger.error(f"❌ 아티팩트 감지 실패: {e}")
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
        usage = {"bytes": 0, "mb": 0, "gb": 0, "error": None}
        
        if hasattr(