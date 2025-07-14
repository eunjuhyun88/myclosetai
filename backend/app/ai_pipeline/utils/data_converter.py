"""
고성능 데이터 변환 유틸리티
- GPU 가속 지원
- 배치 처리 최적화
- 메모리 효율적 변환
- 다양한 포맷 지원
"""
import torch
import numpy as np
from PIL import Image, ImageOps, ImageEnhance
import torchvision.transforms as transforms
import cv2
import io
import base64
from typing import Union, List, Optional, Tuple, Dict, Any
import logging
from pathlib import Path
# 고급 이미지 증강
try:
    import albumentations as A
    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    ALBUMENTATIONS_AVAILABLE = False
    print("⚠️ albumentations 없음 - 기본 증강만 사용")
from torchvision.transforms.functional import to_pil_image, to_tensor

logger = logging.getLogger(__name__)

class DataConverter:
    """고성능 데이터 변환기"""
    
    def __init__(self, device: str = "mps", use_fp16: bool = True, batch_size: int = 8):
        self.device = torch.device(device)
        self.use_fp16 = use_fp16
        self.batch_size = batch_size
        
        # 기본 변환 파이프라인
        self.base_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        # 고급 증강 파이프라인 (Albumentations - 선택적)
        if ALBUMENTATIONS_AVAILABLE:
            self.augmentation_pipeline = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.2),
                A.HueSaturationValue(p=0.2),
                A.RandomGamma(p=0.2),
                A.CLAHE(p=0.1),
                A.Sharpen(p=0.1),
            ])
        else:
            self.augmentation_pipeline = None
        
        # 캐시된 변환 저장소
        self.transform_cache = {}
        
        logger.info(f"DataConverter 초기화 완료 - Device: {device}, FP16: {use_fp16}")
    
    def image_to_tensor(self, 
                       image: Union[str, Path, Image.Image, np.ndarray, bytes], 
                       size: Union[int, Tuple[int, int]] = 512,
                       normalize: bool = True,
                       augment: bool = False) -> torch.Tensor:
        """다양한 입력을 텐서로 변환"""
        try:
            # 입력 타입별 처리
            if isinstance(image, (str, Path)):
                pil_image = Image.open(image).convert('RGB')
            elif isinstance(image, bytes):
                pil_image = Image.open(io.BytesIO(image)).convert('RGB')
            elif isinstance(image, np.ndarray):
                pil_image = Image.fromarray(image).convert('RGB')
            elif isinstance(image, Image.Image):
                pil_image = image.convert('RGB')
            else:
                raise ValueError(f"지원하지 않는 이미지 타입: {type(image)}")
            
            # 크기 조정
            if isinstance(size, int):
                size = (size, size)
            pil_image = pil_image.resize(size, Image.Resampling.LANCZOS)
            
            # 증강 적용 (옵션)
            if augment and self.augmentation_pipeline is not None:
                image_array = np.array(pil_image)
                augmented = self.augmentation_pipeline(image=image_array)
                pil_image = Image.fromarray(augmented['image'])
            
            # 텐서 변환
            if normalize:
                tensor = self.base_transform(pil_image)
            else:
                tensor = to_tensor(pil_image)
            
            # GPU로 이동 및 FP16 변환
            tensor = tensor.to(self.device)
            if self.use_fp16 and self.device.type != 'cpu':
                tensor = tensor.half()
            
            # 배치 차원 추가
            return tensor.unsqueeze(0)
            
        except Exception as e:
            logger.error(f"이미지 텐서 변환 실패: {e}")
            raise
    
    def batch_images_to_tensors(self, 
                               images: List[Union[str, Path, Image.Image, np.ndarray]],
                               size: Union[int, Tuple[int, int]] = 512,
                               normalize: bool = True) -> torch.Tensor:
        """배치 이미지를 텐서로 변환"""
        tensors = []
        
        for image in images:
            tensor = self.image_to_tensor(image, size, normalize, augment=False)
            tensors.append(tensor.squeeze(0))  # 배치 차원 제거
        
        # 배치로 스택
        batch_tensor = torch.stack(tensors)
        return batch_tensor
    
    def tensor_to_image(self, 
                       tensor: torch.Tensor,
                       denormalize: bool = True,
                       format: str = 'PIL') -> Union[Image.Image, np.ndarray]:
        """텐서를 이미지로 변환"""
        try:
            # 배치 차원 처리
            if tensor.dim() == 4:
                tensor = tensor.squeeze(0)
            
            # CPU로 이동
            tensor = tensor.detach().cpu()
            
            # FP16에서 FP32로 변환
            if tensor.dtype == torch.float16:
                tensor = tensor.float()
            
            # 정규화 해제
            if denormalize:
                tensor = tensor * 0.5 + 0.5  # [-1, 1] -> [0, 1]
            
            # 값 범위 클리핑
            tensor = torch.clamp(tensor, 0, 1)
            
            # PIL 이미지로 변환
            pil_image = to_pil_image(tensor)
            
            if format.upper() == 'PIL':
                return pil_image
            elif format.upper() == 'NUMPY':
                return np.array(pil_image)
            elif format.upper() == 'CV2':
                return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            else:
                raise ValueError(f"지원하지 않는 포맷: {format}")
                
        except Exception as e:
            logger.error(f"텐서 이미지 변환 실패: {e}")
            raise
    
    def image_to_base64(self, 
                       image: Union[torch.Tensor, Image.Image, np.ndarray],
                       format: str = 'JPEG',
                       quality: int = 90) -> str:
        """이미지를 Base64로 인코딩"""
        try:
            # PIL 이미지로 변환
            if isinstance(image, torch.Tensor):
                pil_image = self.tensor_to_image(image)
            elif isinstance(image, np.ndarray):
                pil_image = Image.fromarray(image)
            elif isinstance(image, Image.Image):
                pil_image = image
            else:
                raise ValueError(f"지원하지 않는 이미지 타입: {type(image)}")
            
            # 메모리 버퍼에 저장
            buffer = io.BytesIO()
            if format.upper() == 'JPEG':
                pil_image = pil_image.convert('RGB')
                pil_image.save(buffer, format='JPEG', quality=quality, optimize=True)
            elif format.upper() == 'PNG':
                pil_image.save(buffer, format='PNG', optimize=True)
            else:
                raise ValueError(f"지원하지 않는 포맷: {format}")
            
            # Base64 인코딩
            buffer.seek(0)
            encoded = base64.b64encode(buffer.getvalue()).decode('utf-8')
            return encoded
            
        except Exception as e:
            logger.error(f"Base64 인코딩 실패: {e}")
            raise
    
    def base64_to_image(self, 
                       base64_str: str,
                       format: str = 'PIL') -> Union[Image.Image, np.ndarray]:
        """Base64를 이미지로 디코딩"""
        try:
            # Base64 디코딩
            image_data = base64.b64decode(base64_str)
            pil_image = Image.open(io.BytesIO(image_data)).convert('RGB')
            
            if format.upper() == 'PIL':
                return pil_image
            elif format.upper() == 'NUMPY':
                return np.array(pil_image)
            elif format.upper() == 'CV2':
                return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            else:
                raise ValueError(f"지원하지 않는 포맷: {format}")
                
        except Exception as e:
            logger.error(f"Base64 디코딩 실패: {e}")
            raise
    
    def resize_with_aspect_ratio(self, 
                                image: Union[Image.Image, np.ndarray],
                                target_size: Tuple[int, int],
                                fill_color: Tuple[int, int, int] = (255, 255, 255)) -> Union[Image.Image, np.ndarray]:
        """종횡비 유지하며 리사이즈"""
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
            return_numpy = True
        else:
            return_numpy = False
        
        # 종횡비 계산
        orig_width, orig_height = image.size
        target_width, target_height = target_size
        
        # 스케일 계산
        scale = min(target_width / orig_width, target_height / orig_height)
        new_width = int(orig_width * scale)
        new_height = int(orig_height * scale)
        
        # 리사이즈
        resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # 패딩 추가
        padded = Image.new('RGB', target_size, fill_color)
        paste_x = (target_width - new_width) // 2
        paste_y = (target_height - new_height) // 2
        padded.paste(resized, (paste_x, paste_y))
        
        if return_numpy:
            return np.array(padded)
        return padded
    
    def enhance_image_quality(self, 
                             image: Union[Image.Image, np.ndarray],
                             brightness: float = 1.0,
                             contrast: float = 1.0,
                             saturation: float = 1.0,
                             sharpness: float = 1.0) -> Union[Image.Image, np.ndarray]:
        """이미지 품질 향상"""
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
            return_numpy = True
        else:
            return_numpy = False
        
        # 품질 향상 적용
        if brightness != 1.0:
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(brightness)
        
        if contrast != 1.0:
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(contrast)
        
        if saturation != 1.0:
            enhancer = ImageEnhance.Color(image)
            image = enhancer.enhance(saturation)
        
        if sharpness != 1.0:
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(sharpness)
        
        if return_numpy:
            return np.array(image)
        return image
    
    def create_mask_from_alpha(self, image: Image.Image) -> np.ndarray:
        """알파 채널에서 마스크 생성"""
        if image.mode != 'RGBA':
            raise ValueError("이미지에 알파 채널이 없습니다")
        
        alpha = np.array(image)[:, :, 3]
        mask = (alpha > 128).astype(np.uint8) * 255
        return mask
    
    def apply_color_transfer(self, 
                           source: np.ndarray, 
                           target: np.ndarray) -> np.ndarray:
        """색상 전이 적용"""
        # LAB 색공간으로 변환
        source_lab = cv2.cvtColor(source, cv2.COLOR_RGB2LAB).astype(np.float32)
        target_lab = cv2.cvtColor(target, cv2.COLOR_RGB2LAB).astype(np.float32)
        
        # 각 채널별 평균과 표준편차 계산
        source_mean = np.mean(source_lab, axis=(0, 1))
        source_std = np.std(source_lab, axis=(0, 1))
        target_mean = np.mean(target_lab, axis=(0, 1))
        target_std = np.std(target_lab, axis=(0, 1))
        
        # 색상 전이 적용
        for i in range(3):
            source_lab[:, :, i] = (source_lab[:, :, i] - source_mean[i]) * (target_std[i] / source_std[i]) + target_mean[i]
        
        # RGB로 변환
        source_lab = np.clip(source_lab, 0, 255).astype(np.uint8)
        result = cv2.cvtColor(source_lab, cv2.COLOR_LAB2RGB)
        
        return result
    
    def clear_cache(self):
        """캐시 정리"""
        self.transform_cache.clear()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("DataConverter 캐시 정리 완료")
    
    def get_memory_info(self) -> Dict[str, Any]:
        """메모리 사용량 정보"""
        info = {
            'cache_size': len(self.transform_cache),
            'device': str(self.device),
            'use_fp16': self.use_fp16
        }
        
        if torch.cuda.is_available():
            info['gpu_memory_allocated'] = torch.cuda.memory_allocated() / 1024**3  # GB
            info['gpu_memory_reserved'] = torch.cuda.memory_reserved() / 1024**3  # GB
        elif torch.backends.mps.is_available():
            info['mps_allocated'] = torch.mps.current_allocated_memory() / 1024**3  # GB
        
        return info