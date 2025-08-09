#!/usr/bin/env python3
"""
🔥 MyCloset AI - Pose Estimation Image Processor
===============================================

✅ 이미지 처리 기능 분리
✅ 기존 step.py 기능 보존
✅ 모듈화된 구조 적용
"""

import logging
from app.ai_pipeline.utils.common_imports import (
    np, Image, cv2, torch, TORCH_AVAILABLE, PIL_AVAILABLE, CV2_AVAILABLE,
    Dict, Any, Optional, Tuple, List, Union
)

logger = logging.getLogger(__name__)

class ImageProcessor:
    """이미지 처리기"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.ImageProcessor")
    
    def preprocess_image(self, image: Union[torch.Tensor, np.ndarray, Image.Image], 
                        target_size: Tuple[int, int] = (512, 512)) -> np.ndarray:
        """이미지 전처리"""
        try:
            # PIL Image로 변환
            if isinstance(image, Image.Image):
                image_np = np.array(image)
            elif isinstance(image, torch.Tensor):
                image_np = image.cpu().numpy()
            elif isinstance(image, np.ndarray):
                image_np = image.copy()
            else:
                raise ValueError(f"지원하지 않는 이미지 타입: {type(image)}")
            
            # RGB로 변환
            if len(image_np.shape) == 3 and image_np.shape[2] == 3:
                pass  # 이미 RGB
            elif len(image_np.shape) == 3 and image_np.shape[2] == 4:
                image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)
            elif len(image_np.shape) == 2:
                image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
            else:
                raise ValueError(f"지원하지 않는 이미지 채널: {image_np.shape}")
            
            # 리사이즈
            if image_np.shape[:2] != target_size:
                image_np = cv2.resize(image_np, target_size, interpolation=cv2.INTER_LINEAR)
            
            return image_np
            
        except Exception as e:
            self.logger.error(f"❌ 이미지 전처리 실패: {e}")
            raise
    
    def preprocess_image_with_scale(self, image: Union[torch.Tensor, np.ndarray, Image.Image], 
                                   target_size: Tuple[int, int] = (512, 512)) -> Tuple[np.ndarray, float]:
        """스케일 정보와 함께 이미지 전처리"""
        try:
            original_size = None
            
            # 원본 크기 저장
            if isinstance(image, Image.Image):
                original_size = image.size[::-1]  # (height, width)
                image_np = np.array(image)
            elif isinstance(image, torch.Tensor):
                original_size = image.shape[-2:]  # (height, width)
                image_np = image.cpu().numpy()
            elif isinstance(image, np.ndarray):
                original_size = image.shape[:2]  # (height, width)
                image_np = image.copy()
            else:
                raise ValueError(f"지원하지 않는 이미지 타입: {type(image)}")
            
            # 전처리
            processed_image = self.preprocess_image(image_np, target_size)
            
            # 스케일 팩터 계산
            if original_size:
                scale_factor = min(target_size[0] / original_size[0], target_size[1] / original_size[1])
            else:
                scale_factor = 1.0
            
            return processed_image, scale_factor
            
        except Exception as e:
            self.logger.error(f"❌ 이미지 전처리 (스케일 포함) 실패: {e}")
            raise
    
    def normalize_image(self, image: np.ndarray) -> np.ndarray:
        """이미지 정규화"""
        try:
            if image.dtype != np.float32:
                image = image.astype(np.float32)
            
            # 0-1 범위로 정규화
            if image.max() > 1.0:
                image = image / 255.0
            
            return image
            
        except Exception as e:
            self.logger.error(f"❌ 이미지 정규화 실패: {e}")
            raise
    
    def denormalize_image(self, image: np.ndarray) -> np.ndarray:
        """이미지 역정규화"""
        try:
            if image.dtype != np.float32:
                image = image.astype(np.float32)
            
            # 0-255 범위로 역정규화
            if image.max() <= 1.0:
                image = image * 255.0
            
            return image.astype(np.uint8)
            
        except Exception as e:
            self.logger.error(f"❌ 이미지 역정규화 실패: {e}")
            raise
    
    def apply_augmentation(self, image: np.ndarray, augmentation_type: str = "none") -> np.ndarray:
        """이미지 증강"""
        try:
            if augmentation_type == "none":
                return image
            
            augmented_image = image.copy()
            
            if augmentation_type == "flip_horizontal":
                augmented_image = cv2.flip(augmented_image, 1)
            elif augmentation_type == "flip_vertical":
                augmented_image = cv2.flip(augmented_image, 0)
            elif augmentation_type == "rotate_90":
                augmented_image = cv2.rotate(augmented_image, cv2.ROTATE_90_CLOCKWISE)
            elif augmentation_type == "rotate_180":
                augmented_image = cv2.rotate(augmented_image, cv2.ROTATE_180)
            elif augmentation_type == "rotate_270":
                augmented_image = cv2.rotate(augmented_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
            
            return augmented_image
            
        except Exception as e:
            self.logger.error(f"❌ 이미지 증강 실패: {e}")
            return image
    
    def extract_patches(self, image: np.ndarray, patch_size: Tuple[int, int] = (64, 64), 
                       stride: int = 32) -> List[np.ndarray]:
        """이미지에서 패치 추출"""
        try:
            patches = []
            height, width = image.shape[:2]
            patch_h, patch_w = patch_size
            
            for y in range(0, height - patch_h + 1, stride):
                for x in range(0, width - patch_w + 1, stride):
                    patch = image[y:y + patch_h, x:x + patch_w]
                    patches.append(patch)
            
            return patches
            
        except Exception as e:
            self.logger.error(f"❌ 패치 추출 실패: {e}")
            return []
    
    def merge_patches(self, patches: List[np.ndarray], original_size: Tuple[int, int], 
                     patch_size: Tuple[int, int] = (64, 64), stride: int = 32) -> np.ndarray:
        """패치들을 원본 이미지 크기로 병합"""
        try:
            height, width = original_size
            patch_h, patch_w = patch_size
            
            # 빈 이미지 생성
            merged_image = np.zeros((height, width, 3), dtype=np.float32)
            count_image = np.zeros((height, width, 3), dtype=np.float32)
            
            patch_idx = 0
            for y in range(0, height - patch_h + 1, stride):
                for x in range(0, width - patch_w + 1, stride):
                    if patch_idx < len(patches):
                        patch = patches[patch_idx]
                        merged_image[y:y + patch_h, x:x + patch_w] += patch
                        count_image[y:y + patch_h, x:x + patch_w] += 1
                        patch_idx += 1
            
            # 평균 계산
            count_image[count_image == 0] = 1  # 0으로 나누기 방지
            merged_image = merged_image / count_image
            
            return merged_image
            
        except Exception as e:
            self.logger.error(f"❌ 패치 병합 실패: {e}")
            return np.zeros(original_size + (3,), dtype=np.float32)
