"""
🔥 품질 메트릭 핵심 시스템
==========================

후처리를 위한 핵심 품질 메트릭:
1. PSNR (Peak Signal-to-Noise Ratio)
2. SSIM (Structural Similarity Index)
3. 대비 및 선명도 메트릭
4. 색상 품질 메트릭

Author: MyCloset AI Team
Date: 2025-08-14
Version: 1.0
"""

import logging
import numpy as np
import cv2
from typing import Dict, Any, Optional, Tuple, List
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class QualityMetric(ABC):
    """품질 메트릭 기본 클래스"""
    
    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    @abstractmethod
    def calculate(self, original: np.ndarray, processed: np.ndarray) -> float:
        """메트릭 계산"""
        pass
    
    def get_metric_info(self) -> Dict[str, Any]:
        """메트릭 정보 반환"""
        return {
            'name': self.name,
            'description': self.__doc__ or f"{self.name} 메트릭",
            'range': self.get_range(),
            'optimal_value': self.get_optimal_value()
        }
    
    @abstractmethod
    def get_range(self) -> Tuple[float, float]:
        """메트릭 범위 반환"""
        pass
    
    @abstractmethod
    def get_optimal_value(self) -> float:
        """최적값 반환"""
        pass

class PSNRMetric(QualityMetric):
    """PSNR (Peak Signal-to-Noise Ratio) 메트릭"""
    
    def __init__(self):
        super().__init__("PSNR")
    
    def calculate(self, original: np.ndarray, processed: np.ndarray) -> float:
        """PSNR 계산"""
        try:
            # MSE 계산
            mse = np.mean((original.astype(float) - processed.astype(float)) ** 2)
            
            if mse == 0:
                return float('inf')
            
            # PSNR 계산 (8비트 이미지 기준)
            max_pixel_value = 255.0
            psnr = 20 * np.log10(max_pixel_value / np.sqrt(mse))
            
            return float(psnr)
            
        except Exception as e:
            self.logger.error(f"❌ PSNR 계산 실패: {e}")
            return 0.0
    
    def get_range(self) -> Tuple[float, float]:
        """PSNR 범위 반환"""
        return (0.0, float('inf'))
    
    def get_optimal_value(self) -> float:
        """PSNR 최적값 반환"""
        return float('inf')

class SSIMMetric(QualityMetric):
    """SSIM (Structural Similarity Index) 메트릭"""
    
    def __init__(self):
        super().__init__("SSIM")
    
    def calculate(self, original: np.ndarray, processed: np.ndarray) -> float:
        """SSIM 계산"""
        try:
            # 그레이스케일로 변환
            if len(original.shape) == 3:
                original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
            else:
                original_gray = original
            
            if len(processed.shape) == 3:
                processed_gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
            else:
                processed_gray = processed
            
            # SSIM 계산
            ssim = self._calculate_ssim(original_gray, processed_gray)
            
            return float(ssim)
            
        except Exception as e:
            self.logger.error(f"❌ SSIM 계산 실패: {e}")
            return 0.0
    
    def _calculate_ssim(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """SSIM 계산 (간단한 버전)"""
        try:
            # 평균값 계산
            mu1 = np.mean(img1)
            mu2 = np.mean(img2)
            
            # 표준편차 계산
            sigma1 = np.std(img1)
            sigma2 = np.std(img2)
            
            # 공분산 계산
            sigma12 = np.mean((img1 - mu1) * (img2 - mu2))
            
            # 상수값 설정
            c1 = (0.01 * 255) ** 2
            c2 = (0.03 * 255) ** 2
            
            # SSIM 계산
            numerator = (2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)
            denominator = (mu1 ** 2 + mu2 ** 2 + c1) * (sigma1 ** 2 + sigma2 ** 2 + c2)
            
            ssim = numerator / denominator
            
            return float(ssim)
            
        except Exception as e:
            self.logger.error(f"❌ SSIM 계산 중 오류: {e}")
            return 0.0
    
    def get_range(self) -> Tuple[float, float]:
        """SSIM 범위 반환"""
        return (0.0, 1.0)
    
    def get_optimal_value(self) -> float:
        """SSIM 최적값 반환"""
        return 1.0

class ContrastMetric(QualityMetric):
    """대비 메트릭"""
    
    def __init__(self):
        super().__init__("Contrast")
    
    def calculate(self, original: np.ndarray, processed: np.ndarray) -> float:
        """대비 메트릭 계산"""
        try:
            # 원본 대비
            original_contrast = np.std(original)
            
            # 처리된 이미지 대비
            processed_contrast = np.std(processed)
            
            # 대비 개선도 계산
            if original_contrast > 0:
                contrast_improvement = processed_contrast / original_contrast
            else:
                contrast_improvement = 1.0
            
            return float(contrast_improvement)
            
        except Exception as e:
            self.logger.error(f"❌ 대비 메트릭 계산 실패: {e}")
            return 1.0
    
    def get_range(self) -> Tuple[float, float]:
        """대비 메트릭 범위 반환"""
        return (0.0, float('inf'))
    
    def get_optimal_value(self) -> float:
        """대비 메트릭 최적값 반환"""
        return 1.2  # 약간의 대비 향상이 좋음

class SharpnessMetric(QualityMetric):
    """선명도 메트릭"""
    
    def __init__(self):
        super().__init__("Sharpness")
    
    def calculate(self, original: np.ndarray, processed: np.ndarray) -> float:
        """선명도 메트릭 계산"""
        try:
            # 원본 선명도
            original_sharpness = self._calculate_sharpness(original)
            
            # 처리된 이미지 선명도
            processed_sharpness = self._calculate_sharpness(processed)
            
            # 선명도 개선도 계산
            if original_sharpness > 0:
                sharpness_improvement = processed_sharpness / original_sharpness
            else:
                sharpness_improvement = 1.0
            
            return float(sharpness_improvement)
            
        except Exception as e:
            self.logger.error(f"❌ 선명도 메트릭 계산 실패: {e}")
            return 1.0
    
    def _calculate_sharpness(self, image: np.ndarray) -> float:
        """이미지 선명도 계산"""
        try:
            # 그레이스케일로 변환
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            # Laplacian 필터로 선명도 계산
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            sharpness = np.var(laplacian)
            
            return float(sharpness)
            
        except Exception as e:
            self.logger.error(f"❌ 선명도 계산 실패: {e}")
            return 0.0
    
    def get_range(self) -> Tuple[float, float]:
        """선명도 메트릭 범위 반환"""
        return (0.0, float('inf'))
    
    def get_optimal_value(self) -> float:
        """선명도 메트릭 최적값 반환"""
        return 1.3  # 적당한 선명도 향상이 좋음

class ColorBalanceMetric(QualityMetric):
    """색상 균형 메트릭"""
    
    def __init__(self):
        super().__init__("ColorBalance")
    
    def calculate(self, original: np.ndarray, processed: np.ndarray) -> float:
        """색상 균형 메트릭 계산"""
        try:
            # 원본 색상 분산
            original_color_variance = np.var(original, axis=(0, 1))
            
            # 처리된 이미지 색상 분산
            processed_color_variance = np.var(processed, axis=(0, 1))
            
            # 색상 균형 개선도 계산
            if np.any(original_color_variance > 0):
                color_improvement = np.mean(processed_color_variance / original_color_variance)
            else:
                color_improvement = 1.0
            
            return float(color_improvement)
            
        except Exception as e:
            self.logger.error(f"❌ 색상 균형 메트릭 계산 실패: {e}")
            return 1.0
    
    def get_range(self) -> Tuple[float, float]:
        """색상 균형 메트릭 범위 반환"""
        return (0.0, float('inf'))
    
    def get_optimal_value(self) -> float:
        """색상 균형 메트릭 최적값 반환"""
        return 1.1  # 약간의 색상 개선이 좋음

class QualityMetricsCalculator:
    """품질 메트릭 계산기"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.QualityMetricsCalculator")
        
        # 기본 메트릭들
        self.metrics = {
            'psnr': PSNRMetric(),
            'ssim': SSIMMetric(),
            'contrast': ContrastMetric(),
            'sharpness': SharpnessMetric(),
            'color_balance': ColorBalanceMetric()
        }
        
        # 계산 통계
        self.calculation_stats = {
            'total_calculations': 0,
            'successful_calculations': 0,
            'failed_calculations': 0
        }
    
    def calculate_all_metrics(self, original: np.ndarray, processed: np.ndarray) -> Dict[str, float]:
        """모든 품질 메트릭 계산"""
        try:
            self.logger.info("🚀 모든 품질 메트릭 계산 시작")
            
            results = {}
            
            for metric_name, metric in self.metrics.items():
                try:
                    value = metric.calculate(original, processed)
                    results[metric_name] = value
                    self.logger.debug(f"✅ {metric_name}: {value:.4f}")
                    
                except Exception as e:
                    self.logger.error(f"❌ {metric_name} 계산 실패: {e}")
                    results[metric_name] = 0.0
            
            # 통계 업데이트
            self._update_calculation_stats(True)
            
            self.logger.info(f"✅ 모든 품질 메트릭 계산 완료: {len(results)}개")
            return results
            
        except Exception as e:
            self.logger.error(f"❌ 품질 메트릭 계산 실패: {e}")
            self._update_calculation_stats(False)
            return {}
    
    def calculate_specific_metric(self, metric_name: str, original: np.ndarray, processed: np.ndarray) -> Optional[float]:
        """특정 메트릭 계산"""
        try:
            if metric_name not in self.metrics:
                self.logger.error(f"❌ 알 수 없는 메트릭: {metric_name}")
                return None
            
            metric = self.metrics[metric_name]
            value = metric.calculate(original, processed)
            
            self.logger.info(f"✅ {metric_name} 계산 완료: {value:.4f}")
            return value
            
        except Exception as e:
            self.logger.error(f"❌ {metric_name} 계산 실패: {e}")
            return None
    
    def get_metric_info(self, metric_name: str) -> Optional[Dict[str, Any]]:
        """메트릭 정보 반환"""
        try:
            if metric_name not in self.metrics:
                return None
            
            return self.metrics[metric_name].get_metric_info()
            
        except Exception as e:
            self.logger.error(f"❌ 메트릭 정보 조회 실패: {metric_name} - {e}")
            return None
    
    def get_all_metrics_info(self) -> Dict[str, Dict[str, Any]]:
        """모든 메트릭 정보 반환"""
        try:
            info = {}
            
            for metric_name, metric in self.metrics.items():
                info[metric_name] = metric.get_metric_info()
            
            return info
            
        except Exception as e:
            self.logger.error(f"❌ 모든 메트릭 정보 조회 실패: {e}")
            return {}
    
    def add_custom_metric(self, metric_name: str, metric: QualityMetric):
        """사용자 정의 메트릭 추가"""
        try:
            if metric_name in self.metrics:
                self.logger.warning(f"⚠️ 메트릭 {metric_name}이 이미 존재합니다. 덮어씁니다.")
            
            self.metrics[metric_name] = metric
            self.logger.info(f"✅ 사용자 정의 메트릭 추가: {metric_name}")
            
        except Exception as e:
            self.logger.error(f"❌ 사용자 정의 메트릭 추가 실패: {metric_name} - {e}")
    
    def remove_metric(self, metric_name: str) -> bool:
        """메트릭 제거"""
        try:
            if metric_name not in self.metrics:
                self.logger.warning(f"⚠️ 메트릭 {metric_name}이 존재하지 않습니다.")
                return False
            
            del self.metrics[metric_name]
            self.logger.info(f"✅ 메트릭 제거: {metric_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ 메트릭 제거 실패: {metric_name} - {e}")
            return False
    
    def _update_calculation_stats(self, success: bool):
        """계산 통계 업데이트"""
        try:
            self.calculation_stats['total_calculations'] += 1
            
            if success:
                self.calculation_stats['successful_calculations'] += 1
            else:
                self.calculation_stats['failed_calculations'] += 1
                
        except Exception as e:
            self.logger.error(f"❌ 계산 통계 업데이트 실패: {e}")
    
    def get_calculation_stats(self) -> Dict[str, Any]:
        """계산 통계 반환"""
        return self.calculation_stats.copy()
    
    def reset_calculation_stats(self):
        """계산 통계 초기화"""
        self.calculation_stats = {
            'total_calculations': 0,
            'successful_calculations': 0,
            'failed_calculations': 0
        }
