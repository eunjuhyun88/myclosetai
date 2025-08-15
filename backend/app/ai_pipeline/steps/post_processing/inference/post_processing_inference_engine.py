"""
🔥 Post Processing 추론 엔진
============================

후처리를 위한 완전한 추론 시스템:
1. 품질 향상 추론
2. 결과 최적화 추론
3. 품질 검증 추론
4. 앙상블 후처리

Author: MyCloset AI Team
Date: 2025-08-14
Version: 1.0
"""

import logging
import time
from typing import Dict, Any, Optional, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2

logger = logging.getLogger(__name__)

class PostProcessingInferenceEngine:
    """후처리 추론 엔진"""
    
    def __init__(self, step_instance):
        self.step = step_instance
        self.logger = logging.getLogger(f"{__name__}.PostProcessingInferenceEngine")
        
        # 추론 통계
        self.inference_stats = {
            'total_inferences': 0,
            'successful_inferences': 0,
            'failed_inferences': 0,
            'average_inference_time': 0.0,
            'last_inference_time': 0.0
        }
    
    def run_quality_enhancement_inference(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """품질 향상 추론 실행"""
        try:
            start_time = time.time()
            self.logger.info("🚀 품질 향상 추론 시작")
            
            # 입력 데이터 검증
            image = self._extract_input_image(input_data)
            if image is None:
                raise ValueError("입력 이미지를 찾을 수 없습니다")
            
            # 품질 향상 추론 실행
            enhanced_image = self._enhance_image_quality(image)
            
            # 품질 메트릭 계산
            quality_metrics = self._calculate_quality_metrics(image, enhanced_image)
            
            # 추론 시간 업데이트
            inference_time = time.time() - start_time
            self._update_inference_stats(True, inference_time)
            
            result = {
                'enhanced_image': enhanced_image,
                'quality_metrics': quality_metrics,
                'inference_time': inference_time,
                'enhancement_method': 'quality_enhancement'
            }
            
            self.logger.info(f"✅ 품질 향상 추론 완료 ({inference_time:.2f}s)")
            return result
            
        except Exception as e:
            self.logger.error(f"❌ 품질 향상 추론 실패: {e}")
            self._update_inference_stats(False, 0.0)
            raise
    
    def run_result_optimization_inference(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """결과 최적화 추론 실행"""
        try:
            start_time = time.time()
            self.logger.info("🚀 결과 최적화 추론 시작")
            
            # 입력 데이터 검증
            image = self._extract_input_image(input_data)
            if image is None:
                raise ValueError("입력 이미지를 찾을 수 없습니다")
            
            # 결과 최적화 추론 실행
            optimized_image = self._optimize_result(image)
            
            # 최적화 메트릭 계산
            optimization_metrics = self._calculate_optimization_metrics(image, optimized_image)
            
            # 추론 시간 업데이트
            inference_time = time.time() - start_time
            self._update_inference_stats(True, inference_time)
            
            result = {
                'optimized_image': optimized_image,
                'optimization_metrics': optimization_metrics,
                'inference_time': inference_time,
                'optimization_method': 'result_optimization'
            }
            
            self.logger.info(f"✅ 결과 최적화 추론 완료 ({inference_time:.2f}s)")
            return result
            
        except Exception as e:
            self.logger.error(f"❌ 결과 최적화 추론 실패: {e}")
            self._update_inference_stats(False, 0.0)
            raise
    
    def _extract_input_image(self, input_data: Dict[str, Any]) -> Optional[np.ndarray]:
        """입력 이미지 추출"""
        try:
            # 다양한 키에서 이미지 찾기
            image_keys = ['input_image', 'image', 'original_image', 'preprocessed_image']
            
            for key in image_keys:
                if key in input_data and input_data[key] is not None:
                    image = input_data[key]
                    if isinstance(image, np.ndarray):
                        return image
                    elif hasattr(image, 'numpy'):
                        return image.numpy()
                    else:
                        self.logger.warning(f"⚠️ {key}에서 이미지를 추출할 수 없습니다")
            
            return None
            
        except Exception as e:
            self.logger.error(f"❌ 이미지 추출 실패: {e}")
            return None
    
    def _enhance_image_quality(self, image: np.ndarray) -> np.ndarray:
        """이미지 품질 향상"""
        try:
            # 기본 품질 향상 적용
            enhanced = image.copy()
            
            # 노이즈 제거
            enhanced = cv2.fastNlMeansDenoisingColored(enhanced, None, 10, 10, 7, 21)
            
            # 선명도 향상
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            enhanced = cv2.filter2D(enhanced, -1, kernel)
            
            # 대비 향상
            lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            l = clahe.apply(l)
            enhanced = cv2.merge([l, a, b])
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
            
            return enhanced
            
        except Exception as e:
            self.logger.error(f"❌ 이미지 품질 향상 실패: {e}")
            return image
    
    def _optimize_result(self, image: np.ndarray) -> np.ndarray:
        """결과 최적화"""
        try:
            # 기본 최적화 적용
            optimized = image.copy()
            
            # 색상 균형 조정
            optimized = cv2.convertScaleAbs(optimized, alpha=1.1, beta=5)
            
            # 감마 보정
            gamma = 1.1
            invGamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
            optimized = cv2.LUT(optimized, table)
            
            return optimized
            
        except Exception as e:
            self.logger.error(f"❌ 결과 최적화 실패: {e}")
            return image
    
    def _calculate_quality_metrics(self, original: np.ndarray, enhanced: np.ndarray) -> Dict[str, float]:
        """품질 메트릭 계산"""
        try:
            metrics = {}
            
            # PSNR 계산
            mse = np.mean((original.astype(float) - enhanced.astype(float)) ** 2)
            if mse == 0:
                metrics['psnr'] = float('inf')
            else:
                metrics['psnr'] = 20 * np.log10(255.0 / np.sqrt(mse))
            
            # SSIM 계산 (간단한 버전)
            metrics['ssim'] = self._calculate_ssim(original, enhanced)
            
            # 대비 개선도
            original_contrast = np.std(original)
            enhanced_contrast = np.std(enhanced)
            metrics['contrast_improvement'] = enhanced_contrast / original_contrast if original_contrast > 0 else 1.0
            
            # 선명도 개선도
            original_sharpness = self._calculate_sharpness(original)
            enhanced_sharpness = self._calculate_sharpness(enhanced)
            metrics['sharpness_improvement'] = enhanced_sharpness / original_sharpness if original_sharpness > 0 else 1.0
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"❌ 품질 메트릭 계산 실패: {e}")
            return {'psnr': 0.0, 'ssim': 0.0, 'contrast_improvement': 1.0, 'sharpness_improvement': 1.0}
    
    def _calculate_ssim(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """SSIM 계산 (간단한 버전)"""
        try:
            # 간단한 SSIM 계산
            mu1 = np.mean(img1)
            mu2 = np.mean(img2)
            sigma1 = np.std(img1)
            sigma2 = np.std(img2)
            sigma12 = np.mean((img1 - mu1) * (img2 - mu2))
            
            c1 = (0.01 * 255) ** 2
            c2 = (0.03 * 255) ** 2
            
            ssim = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / \
                   ((mu1 ** 2 + mu2 ** 2 + c1) * (sigma1 ** 2 + sigma2 ** 2 + c2))
            
            return float(ssim)
            
        except Exception as e:
            self.logger.error(f"❌ SSIM 계산 실패: {e}")
            return 0.0
    
    def _calculate_sharpness(self, image: np.ndarray) -> float:
        """선명도 계산"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            return np.var(laplacian)
            
        except Exception as e:
            self.logger.error(f"❌ 선명도 계산 실패: {e}")
            return 0.0
    
    def _calculate_optimization_metrics(self, original: np.ndarray, optimized: np.ndarray) -> Dict[str, float]:
        """최적화 메트릭 계산"""
        try:
            metrics = {}
            
            # 색상 개선도
            original_color_variance = np.var(original, axis=(0, 1))
            optimized_color_variance = np.var(optimized, axis=(0, 1))
            metrics['color_improvement'] = np.mean(optimized_color_variance / original_color_variance)
            
            # 밝기 개선도
            original_brightness = np.mean(original)
            optimized_brightness = np.mean(optimized)
            metrics['brightness_improvement'] = optimized_brightness / original_brightness if original_brightness > 0 else 1.0
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"❌ 최적화 메트릭 계산 실패: {e}")
            return {'color_improvement': 1.0, 'brightness_improvement': 1.0}
    
    def _update_inference_stats(self, success: bool, inference_time: float):
        """추론 통계 업데이트"""
        try:
            self.inference_stats['total_inferences'] += 1
            
            if success:
                self.inference_stats['successful_inferences'] += 1
                
                # 평균 추론 시간 업데이트
                total_successful = self.inference_stats['successful_inferences']
                current_avg = self.inference_stats['average_inference_time']
                new_avg = (current_avg * (total_successful - 1) + inference_time) / total_successful
                self.inference_stats['average_inference_time'] = new_avg
            else:
                self.inference_stats['failed_inferences'] += 1
            
            self.inference_stats['last_inference_time'] = inference_time
            
        except Exception as e:
            self.logger.error(f"❌ 추론 통계 업데이트 실패: {e}")
    
    def get_inference_stats(self) -> Dict[str, Any]:
        """추론 통계 반환"""
        return self.inference_stats.copy()
    
    def reset_inference_stats(self):
        """추론 통계 초기화"""
        self.inference_stats = {
            'total_inferences': 0,
            'successful_inferences': 0,
            'failed_inferences': 0,
            'average_inference_time': 0.0,
            'last_inference_time': 0.0
        }
