#!/usr/bin/env python3
"""
🔥 MyCloset AI - Data Processing Mixin
=====================================

데이터 전처리 및 후처리 관련 기능을 담당하는 Mixin 클래스
- 이미지 전처리 (리사이즈, 정규화, 텐서 변환)
- 모델별 입력 준비 (SAM, Diffusion, OOTD 등)
- 후처리 (Softmax, Argmax, NMS 등)
- 데이터 검증 및 변환

Author: MyCloset AI Team
Date: 2025-08-14
Version: 2.0
"""

import logging
from typing import Dict, Any, Optional, Tuple, List, Union

# 선택적 import
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    Image = None

class DataProcessingMixin:
    """데이터 전처리 및 후처리 관련 기능을 담당하는 Mixin"""
    
    def _apply_preprocessing_sync(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """전처리 적용 (동기 버전)"""
        try:
            if not hasattr(self, 'config') or not self.config.auto_preprocessing:
                return input_data
            
            processed = input_data.copy()
            
            # 이미지 리사이즈
            if self.config.input_size:
                processed = self._resize_images(processed, self.config.input_size)
            
            # 정규화 적용
            if self.config.normalization_type:
                processed = self._apply_normalization(processed, self.config.normalization_type)
            
            # 텐서 변환
            if self.config.convert_to_tensor:
                processed = self._convert_to_tensor(processed)
            
            # 모델별 특화 전처리
            if hasattr(self, 'model_type'):
                processed = self._apply_model_specific_preprocessing(processed)
            
            self.logger.debug(f"✅ {self.step_name} 전처리 완료")
            return processed
            
        except Exception as e:
            self.logger.error(f"❌ {self.step_name} 전처리 실패: {e}")
            return input_data
    
    def _resize_images(self, data: Dict[str, Any], target_size: Tuple[int, int]) -> Dict[str, Any]:
        """이미지 리사이즈"""
        result = data.copy()
        
        for key, value in data.items():
            try:
                if PIL_AVAILABLE and isinstance(value, Image.Image):
                    result[key] = value.resize(target_size, Image.Resampling.LANCZOS)
                elif NUMPY_AVAILABLE and isinstance(value, np.ndarray):
                    if len(value.shape) == 3:
                        # (H, W, C) → (C, H, W) 변환 후 리사이즈
                        if value.shape[2] in [1, 3, 4]:
                            value = np.transpose(value, (2, 0, 1))
                        
                        # 간단한 리사이즈 (실제로는 더 정교한 방법 사용 권장)
                        from skimage.transform import resize
                        try:
                            resized = resize(value, (value.shape[0], target_size[1], target_size[0]))
                            result[key] = resized
                        except ImportError:
                            # skimage가 없는 경우 기본 방법 사용
                            result[key] = value
                    else:
                        result[key] = value
                else:
                    result[key] = value
                    
            except Exception as e:
                self.logger.debug(f"이미지 리사이즈 실패 ({key}): {e}")
                result[key] = value
        
        return result
    
    def _apply_normalization(self, data: Dict[str, Any], norm_type: str) -> Dict[str, Any]:
        """정규화 적용"""
        result = data.copy()
        
        for key, value in data.items():
            try:
                if norm_type == "imagenet":
                    result[key] = self._normalize_imagenet(value)
                elif norm_type == "clip":
                    result[key] = self._normalize_clip(value)
                elif norm_type == "diffusion":
                    result[key] = self._normalize_diffusion(value)
                elif norm_type == "centered":
                    result[key] = self._normalize_centered(value)
                else:
                    result[key] = value
                    
            except Exception as e:
                self.logger.debug(f"정규화 실패 ({key}): {e}")
                result[key] = value
        
        return result
    
    def _normalize_imagenet(self, value: Any) -> Any:
        """ImageNet 정규화"""
        if NUMPY_AVAILABLE and isinstance(value, np.ndarray):
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            
            if len(value.shape) == 3 and value.shape[0] == 3:
                # (C, H, W) 형태
                for i in range(3):
                    value[i] = (value[i] - mean[i]) / std[i]
            elif len(value.shape) == 3 and value.shape[2] == 3:
                # (H, W, C) 형태
                value = (value - mean) / std
            
        return value
    
    def _normalize_clip(self, value: Any) -> Any:
        """CLIP 정규화"""
        if NUMPY_AVAILABLE and isinstance(value, np.ndarray):
            mean = np.array([0.48145466, 0.4578275, 0.40821073])
            std = np.array([0.26862954, 0.26130258, 0.27577711])
            
            if len(value.shape) == 3 and value.shape[0] == 3:
                for i in range(3):
                    value[i] = (value[i] - mean[i]) / std[i]
            elif len(value.shape) == 3 and value.shape[2] == 3:
                value = (value - mean) / std
        
        return value
    
    def _normalize_diffusion(self, value: Any) -> Any:
        """Diffusion 모델 정규화"""
        if NUMPY_AVAILABLE and isinstance(value, np.ndarray):
            # [-1, 1] 범위로 정규화
            if value.max() > 1.0 or value.min() < 0.0:
                value = (value - 0.5) * 2.0
        
        return value
    
    def _normalize_centered(self, value: Any) -> Any:
        """중앙 정규화"""
        if NUMPY_AVAILABLE and isinstance(value, np.ndarray):
            # [-1, 1] 범위로 정규화
            if value.max() > 1.0 or value.min() < 0.0:
                value = (value - 0.5) * 2.0
        
        return value
    
    def _convert_to_tensor(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """데이터를 텐서로 변환"""
        result = data.copy()
        
        for key, value in data.items():
            try:
                if TORCH_AVAILABLE and NUMPY_AVAILABLE and isinstance(value, np.ndarray):
                    result[key] = torch.from_numpy(value).float()
                elif TORCH_AVAILABLE and PIL_AVAILABLE and isinstance(value, Image.Image):
                    array = np.array(value).astype(np.float32)
                    if len(array.shape) == 3 and array.shape[2] in [1, 3, 4]:
                        array = np.transpose(array, (2, 0, 1))
                    result[key] = torch.from_numpy(array).float()
                else:
                    result[key] = value
                    
            except Exception as e:
                self.logger.debug(f"텐서 변환 실패 ({key}): {e}")
                result[key] = value
        
        return result
    
    def _apply_model_specific_preprocessing(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """모델별 특화 전처리"""
        try:
            model_type = getattr(self, 'model_type', '').lower()
            
            if 'sam' in model_type:
                return self._prepare_sam_prompts(data)
            elif 'diffusion' in model_type:
                return self._prepare_diffusion_input(data)
            elif 'ootd' in model_type:
                return self._prepare_ootd_inputs(data)
            elif 'sr' in model_type or 'super' in model_type:
                return self._prepare_sr_input(data)
            else:
                return data
                
        except Exception as e:
            self.logger.debug(f"모델별 전처리 실패: {e}")
            return data
    
    def _prepare_sam_prompts(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """SAM 프롬프트 준비"""
        result = data.copy()
        
        if 'prompt_points' not in result and 'image' in result:
            if PIL_AVAILABLE and isinstance(result['image'], Image.Image):
                w, h = result['image'].size
                result['prompt_points'] = np.array([[w//2, h//2]])
                result['prompt_labels'] = np.array([1])
        
        return result
    
    def _prepare_diffusion_input(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Diffusion 모델 입력 준비"""
        result = data.copy()
        
        if 'guidance_scale' not in result:
            result['guidance_scale'] = 7.5
        if 'num_inference_steps' not in result:
            result['num_inference_steps'] = 20
        if 'strength' not in result:
            result['strength'] = 0.8
        
        return result
    
    def _prepare_ootd_inputs(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """OOTD Diffusion 입력 준비"""
        result = data.copy()
        
        if 'fitting_mode' not in result:
            result['fitting_mode'] = 'hd'
        
        return result
    
    def _prepare_sr_input(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Super Resolution 입력 준비"""
        result = data.copy()
        
        if 'tile_size' not in result:
            result['tile_size'] = 512
        if 'overlap' not in result:
            result['overlap'] = 64
        
        return result
    
    def _apply_postprocessing_sync(self, ai_result: Dict[str, Any]) -> Dict[str, Any]:
        """후처리 적용 (동기 버전)"""
        try:
            if not hasattr(self, 'config') or not self.config.auto_postprocessing:
                return ai_result
            
            processed = ai_result.copy()
            
            # 후처리 단계들 적용
            if hasattr(self, 'postprocessing_steps'):
                for step_name in self.postprocessing_steps:
                    processed = self._apply_postprocessing_step(processed, step_name)
            
            self.logger.debug(f"✅ {self.step_name} 후처리 완료")
            return processed
            
        except Exception as e:
            self.logger.error(f"❌ {self.step_name} 후처리 실패: {e}")
            return ai_result
    
    def _apply_postprocessing_step(self, data: Dict[str, Any], step_name: str) -> Dict[str, Any]:
        """개별 후처리 단계 적용"""
        try:
            if step_name == "softmax":
                return self._apply_softmax(data)
            elif step_name == "argmax":
                return self._apply_argmax(data)
            elif step_name == "resize_original":
                return self._resize_to_original(data)
            elif step_name == "to_numpy":
                return self._convert_to_numpy(data)
            elif step_name == "threshold_0.5":
                return self._apply_threshold(data, 0.5)
            elif step_name == "nms":
                return self._apply_nms(data)
            elif step_name in ["denormalize_diffusion", "denormalize_centered"]:
                return self._denormalize_diffusion(data)
            elif step_name == "denormalize":
                return self._denormalize_imagenet(data)
            elif step_name in ["clip_values", "clip_0_1"]:
                return self._clip_values(data, 0.0, 1.0)
            else:
                self.logger.debug(f"⚠️ 알 수 없는 후처리 단계: {step_name}")
                return data
                
        except Exception as e:
            self.logger.debug(f"후처리 단계 실패 ({step_name}): {e}")
            return data
    
    def _apply_softmax(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Softmax 적용"""
        result = data.copy()
        
        for key, value in data.items():
            try:
                if TORCH_AVAILABLE and isinstance(value, torch.Tensor):
                    result[key] = torch.softmax(value, dim=-1)
                elif NUMPY_AVAILABLE and isinstance(value, np.ndarray):
                    from scipy.special import softmax
                    result[key] = softmax(value, axis=-1)
                else:
                    result[key] = value
            except Exception as e:
                self.logger.debug(f"Softmax 적용 실패 ({key}): {e}")
                result[key] = value
        
        return result
    
    def _apply_argmax(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Argmax 적용"""
        result = data.copy()
        
        for key, value in data.items():
            try:
                if TORCH_AVAILABLE and isinstance(value, torch.Tensor):
                    result[key] = torch.argmax(value, dim=-1)
                elif NUMPY_AVAILABLE and isinstance(value, np.ndarray):
                    result[key] = np.argmax(value, axis=-1)
                else:
                    result[key] = value
            except Exception as e:
                self.logger.debug(f"Argmax 적용 실패 ({key}): {e}")
                result[key] = value
        
        return result
    
    def _apply_threshold(self, data: Dict[str, Any], threshold: float) -> Dict[str, Any]:
        """임계값 적용"""
        result = data.copy()
        
        for key, value in data.items():
            try:
                if TORCH_AVAILABLE and isinstance(value, torch.Tensor):
                    result[key] = (value > threshold).float()
                elif NUMPY_AVAILABLE and isinstance(value, np.ndarray):
                    result[key] = (value > threshold).astype(np.float32)
                else:
                    result[key] = value
            except Exception as e:
                self.logger.debug(f"임계값 적용 실패 ({key}): {e}")
                result[key] = value
        
        return result
    
    def _clip_values(self, data: Dict[str, Any], min_val: float, max_val: float) -> Dict[str, Any]:
        """값 범위 제한"""
        result = data.copy()
        
        for key, value in data.items():
            try:
                if TORCH_AVAILABLE and isinstance(value, torch.Tensor):
                    result[key] = torch.clamp(value, min_val, max_val)
                elif NUMPY_AVAILABLE and isinstance(value, np.ndarray):
                    result[key] = np.clip(value, min_val, max_val)
                else:
                    result[key] = value
            except Exception as e:
                self.logger.debug(f"값 범위 제한 실패 ({key}): {e}")
                result[key] = value
        
        return result
