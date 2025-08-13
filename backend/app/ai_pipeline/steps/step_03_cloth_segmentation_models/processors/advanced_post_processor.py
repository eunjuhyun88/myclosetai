#!/usr/bin/env python3
"""
🔥 MyCloset AI - Step 03: 의류 세그멘테이션 - Advanced Post Processor
=====================================================================

고급 후처리를 위한 전용 프로세서

Author: MyCloset AI Team  
Date: 2025-08-01
Version: 1.0
"""

import logging
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
import cv2

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)

class AdvancedPostProcessor:
    """고급 후처리 프로세서"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """초기화"""
        self.config = config or {}
        self.logger = logging.getLogger(f"{__name__}.AdvancedPostProcessor")
        self.enabled = self.config.get('enable_advanced_postprocessing', True)
        
    @staticmethod
    def apply_crf_postprocessing(mask: np.ndarray, image: np.ndarray, num_iterations: int = 15) -> np.ndarray:
        """CRF 후처리 적용"""
        try:
            # CRF 라이브러리가 없는 경우 기본 후처리로 대체
            if not TORCH_AVAILABLE:
                return AdvancedPostProcessor._apply_basic_crf(mask, image)
            
            # PyTorch 기반 CRF 구현
            return AdvancedPostProcessor._apply_pytorch_crf(mask, image, num_iterations)
            
        except Exception as e:
            logger.warning(f"⚠️ CRF 후처리 실패: {e}")
            return mask
    
    @staticmethod
    def apply_multiscale_processing(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """멀티스케일 처리 적용"""
        try:
            if mask is None or mask.size == 0:
                return mask
            
            # 여러 스케일에서 처리
            scales = [0.5, 1.0, 2.0]
            processed_masks = []
            
            for scale in scales:
                # 스케일 조정
                if scale != 1.0:
                    scaled_image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
                    scaled_mask = cv2.resize(mask, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
                else:
                    scaled_image = image
                    scaled_mask = mask
                
                # 스케일별 처리
                processed_mask = AdvancedPostProcessor._process_single_scale(scaled_image, scaled_mask)
                processed_masks.append(processed_mask)
            
            # 결과 결합
            if len(processed_masks) > 1:
                # 가중 평균으로 결합
                weights = [0.2, 0.6, 0.2]  # 중간 스케일에 높은 가중치
                combined_mask = np.zeros_like(mask, dtype=np.float32)
                
                for i, processed_mask in enumerate(processed_masks):
                    if processed_mask.shape != mask.shape:
                        processed_mask = cv2.resize(processed_mask, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_NEAREST)
                    combined_mask += processed_mask.astype(np.float32) * weights[i]
                
                # 이진화
                final_mask = (combined_mask > 127).astype(np.uint8) * 255
            else:
                final_mask = processed_masks[0]
            
            return final_mask
            
        except Exception as e:
            logger.warning(f"⚠️ 멀티스케일 처리 실패: {e}")
            return mask
    
    @staticmethod
    def apply_edge_refinement(masks: Dict[str, np.ndarray], image: np.ndarray) -> Dict[str, np.ndarray]:
        """엣지 정제 적용"""
        try:
            refined_masks = {}
            
            for mask_key, mask in masks.items():
                if mask is None or mask.size == 0:
                    refined_masks[mask_key] = mask
                    continue
                
                # 엣지 정제
                refined_mask = AdvancedPostProcessor._refine_edges(mask, image)
                refined_masks[mask_key] = refined_mask
            
            return refined_masks
            
        except Exception as e:
            logger.warning(f"⚠️ 엣지 정제 실패: {e}")
            return masks
    
    @staticmethod
    def _apply_basic_crf(mask: np.ndarray, image: np.ndarray) -> np.ndarray:
        """기본 CRF 후처리"""
        try:
            # 간단한 조건부 랜덤 필드 시뮬레이션
            refined_mask = mask.copy()
            
            # 1. 모폴로지 연산
            kernel = np.ones((3, 3), np.uint8)
            refined_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_CLOSE, kernel)
            refined_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_OPEN, kernel)
            
            # 2. 엣지 기반 정제
            edges = cv2.Canny(image, 50, 150)
            edge_mask = cv2.dilate(edges, kernel, iterations=1)
            
            # 엣지 근처의 마스크 정제
            refined_mask[edge_mask > 0] = 0
            
            # 3. 연결성 분석
            contours, _ = cv2.findContours(refined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # 작은 영역 제거
            min_area = 100
            for contour in contours:
                if cv2.contourArea(contour) < min_area:
                    cv2.fillPoly(refined_mask, [contour], 0)
            
            return refined_mask
            
        except Exception as e:
            logger.warning(f"⚠️ 기본 CRF 실패: {e}")
            return mask
    
    @staticmethod
    def _apply_pytorch_crf(mask: np.ndarray, image: np.ndarray, num_iterations: int) -> np.ndarray:
        """PyTorch 기반 CRF 후처리"""
        try:
            if not TORCH_AVAILABLE:
                return AdvancedPostProcessor._apply_basic_crf(mask, image)
            
            # PyTorch 텐서로 변환
            mask_tensor = torch.from_numpy(mask).float() / 255.0
            image_tensor = torch.from_numpy(image).float() / 255.0
            
            # CRF 시뮬레이션 (실제 CRF 라이브러리 대신)
            refined_mask = AdvancedPostProcessor._simulate_crf(mask_tensor, image_tensor, num_iterations)
            
            # NumPy로 변환
            refined_mask = (refined_mask.numpy() * 255).astype(np.uint8)
            
            return refined_mask
            
        except Exception as e:
            logger.warning(f"⚠️ PyTorch CRF 실패: {e}")
            return AdvancedPostProcessor._apply_basic_crf(mask, image)
    
    @staticmethod
    def _simulate_crf(mask_tensor: torch.Tensor, image_tensor: torch.Tensor, num_iterations: int) -> torch.Tensor:
        """CRF 시뮬레이션"""
        try:
            # 간단한 CRF 시뮬레이션
            refined_mask = mask_tensor.clone()
            
            for _ in range(num_iterations):
                # 가우시안 필터 적용
                refined_mask = F.avg_pool2d(refined_mask.unsqueeze(0), kernel_size=3, stride=1, padding=1).squeeze(0)
                
                # 이미지 기반 가중치 적용
                if image_tensor.dim() == 3:
                    gray_image = torch.mean(image_tensor, dim=0)
                    weight = torch.exp(-torch.abs(gray_image - 0.5))
                    refined_mask = refined_mask * weight
            
            return refined_mask
            
        except Exception as e:
            logger.warning(f"⚠️ CRF 시뮬레이션 실패: {e}")
            return mask_tensor
    
    @staticmethod
    def _process_single_scale(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """단일 스케일 처리"""
        try:
            # 기본 후처리
            processed_mask = mask.copy()
            
            # 1. 노이즈 제거
            kernel = np.ones((3, 3), np.uint8)
            processed_mask = cv2.morphologyEx(processed_mask, cv2.MORPH_OPEN, kernel)
            
            # 2. 홀 채우기
            processed_mask = cv2.morphologyEx(processed_mask, cv2.MORPH_CLOSE, kernel)
            
            # 3. 경계 스무딩
            processed_mask = cv2.GaussianBlur(processed_mask, (3, 3), 0)
            processed_mask = (processed_mask > 127).astype(np.uint8) * 255
            
            return processed_mask
            
        except Exception as e:
            logger.warning(f"⚠️ 단일 스케일 처리 실패: {e}")
            return mask
    
    @staticmethod
    def _refine_edges(mask: np.ndarray, image: np.ndarray) -> np.ndarray:
        """엣지 정제"""
        try:
            refined_mask = mask.copy()
            
            # 1. 엣지 탐지
            edges = cv2.Canny(image, 50, 150)
            
            # 2. 마스크 엣지와 이미지 엣지 비교
            mask_edges = cv2.Canny(mask, 50, 150)
            
            # 3. 엣지 정제
            edge_kernel = np.ones((2, 2), np.uint8)
            refined_edges = cv2.dilate(edges, edge_kernel, iterations=1)
            
            # 엣지 근처의 마스크 정제
            refined_mask[refined_edges > 0] = 0
            
            # 4. 모폴로지 연산으로 정제
            kernel = np.ones((3, 3), np.uint8)
            refined_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_CLOSE, kernel)
            
            return refined_mask
            
        except Exception as e:
            logger.warning(f"⚠️ 엣지 정제 실패: {e}")
            return mask
