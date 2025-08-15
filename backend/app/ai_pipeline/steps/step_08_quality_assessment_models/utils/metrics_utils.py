#!/usr/bin/env python3
"""
🔥 MyCloset AI - Quality Assessment Metrics Utils
=================================================

품질 평가를 위한 메트릭 계산 유틸리티 함수들
- 이미지 품질 메트릭 계산
- 통계 분석
- 품질 점수 산출

Author: MyCloset AI Team
Date: 2025-08-14
Version: 1.0
"""

import os
import sys
import logging
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union

# 로깅 설정
logger = logging.getLogger(__name__)

# 라이브러리 가용성 확인
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    cv2 = None

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    Image = None

try:
    from skimage.metrics import structural_similarity as ssim
    from skimage.metrics import peak_signal_noise_ratio as psnr
    from skimage import filters, restoration, exposure
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    ssim = None
    psnr = None

# ==============================================
# 🔥 기본 품질 메트릭 함수들
# ==============================================

def calculate_brightness(image: np.ndarray) -> float:
    """이미지 밝기 계산"""
    try:
        if len(image.shape) == 3:
            # RGB 이미지의 경우 그레이스케일로 변환
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if CV2_AVAILABLE else np.mean(image, axis=2)
        else:
            gray = image
        
        return float(np.mean(gray))
    except Exception as e:
        logger.warning(f"밝기 계산 실패: {e}")
        return 0.0

def calculate_contrast(image: np.ndarray) -> float:
    """이미지 대비 계산"""
    try:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if CV2_AVAILABLE else np.mean(image, axis=2)
        else:
            gray = image
        
        return float(np.std(gray))
    except Exception as e:
        logger.warning(f"대비 계산 실패: {e}")
        return 0.0

def calculate_sharpness(image: np.ndarray) -> float:
    """이미지 선명도 계산 (Laplacian variance)"""
    try:
        if not CV2_AVAILABLE:
            return 0.0
        
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        return float(laplacian_var)
    except Exception as e:
        logger.warning(f"선명도 계산 실패: {e}")
        return 0.0

def calculate_noise_level(image: np.ndarray) -> float:
    """이미지 노이즈 레벨 추정"""
    try:
        if not CV2_AVAILABLE:
            return 0.0
        
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # 가우시안 블러를 사용한 노이즈 추정
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        noise = gray.astype(np.float64) - blurred.astype(np.float64)
        noise_level = np.std(noise)
        
        return float(noise_level)
    except Exception as e:
        logger.warning(f"노이즈 레벨 계산 실패: {e}")
        return 0.0

def calculate_color_distribution(image: np.ndarray) -> Dict[str, float]:
    """이미지 색상 분포 분석"""
    try:
        if len(image.shape) != 3:
            return {'error': '그레이스케일 이미지입니다'}
        
        # RGB 채널별 통계
        r_channel = image[:, :, 0]
        g_channel = image[:, :, 1]
        b_channel = image[:, :, 2]
        
        color_stats = {
            'red_mean': float(np.mean(r_channel)),
            'red_std': float(np.std(r_channel)),
            'green_mean': float(np.mean(g_channel)),
            'green_std': float(np.std(g_channel)),
            'blue_mean': float(np.mean(b_channel)),
            'blue_std': float(np.std(b_channel)),
            'color_balance': float(np.std([np.mean(r_channel), np.mean(g_channel), np.mean(b_channel)]))
        }
        
        return color_stats
        
    except Exception as e:
        logger.warning(f"색상 분포 분석 실패: {e}")
        return {'error': str(e)}

# ==============================================
# 🔥 고급 품질 메트릭 함수들
# ==============================================

def calculate_ssim_score(image1: np.ndarray, image2: np.ndarray) -> float:
    """SSIM (구조적 유사성) 점수 계산"""
    try:
        if not SKIMAGE_AVAILABLE:
            return 0.0
        
        # 이미지 크기 맞추기
        if image1.shape != image2.shape:
            # 간단한 리사이즈 (실제로는 더 정교한 방법 사용)
            h, w = min(image1.shape[0], image2.shape[0]), min(image1.shape[1], image2.shape[1])
            img1_resized = image1[:h, :w]
            img2_resized = image2[:h, :w]
        else:
            img1_resized, img2_resized = image1, image2
        
        # SSIM 계산
        ssim_score = ssim(img1_resized, img2_resized, multichannel=True)
        return float(ssim_score)
        
    except Exception as e:
        logger.warning(f"SSIM 계산 실패: {e}")
        return 0.0

def calculate_psnr_score(image1: np.ndarray, image2: np.ndarray) -> float:
    """PSNR (피크 신호 대 잡음비) 점수 계산"""
    try:
        if not SKIMAGE_AVAILABLE:
            return 0.0
        
        # 이미지 크기 맞추기
        if image1.shape != image2.shape:
            h, w = min(image1.shape[0], image2.shape[0]), min(image1.shape[1], image2.shape[1])
            img1_resized = image1[:h, :w]
            img2_resized = image2[:h, :w]
        else:
            img1_resized, img2_resized = image1, image2
        
        # PSNR 계산
        psnr_score = psnr(img1_resized, img2_resized)
        return float(psnr_score)
        
    except Exception as e:
        logger.warning(f"PSNR 계산 실패: {e}")
        return 0.0

def calculate_edge_density(image: np.ndarray) -> float:
    """이미지 엣지 밀도 계산"""
    try:
        if not CV2_AVAILABLE:
            return 0.0
        
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Canny 엣지 검출
        edges = cv2.Canny(gray, 50, 150)
        
        # 엣지 픽셀 비율 계산
        edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
        
        return float(edge_density * 100)  # 백분율로 반환
        
    except Exception as e:
        logger.warning(f"엣지 밀도 계산 실패: {e}")
        return 0.0

def calculate_texture_complexity(image: np.ndarray) -> float:
    """이미지 텍스처 복잡도 계산"""
    try:
        if not CV2_AVAILABLE:
            return 0.0
        
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Gabor 필터를 사용한 텍스처 분석
        # 간단한 버전: 로컬 표준편차의 평균
        kernel_size = 5
        kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
        local_mean = cv2.filter2D(gray.astype(np.float32), -1, kernel)
        local_var = cv2.filter2D((gray.astype(np.float32) - local_mean) ** 2, -1, kernel)
        local_std = np.sqrt(local_var)
        
        texture_complexity = np.mean(local_std)
        return float(texture_complexity)
        
    except Exception as e:
        logger.warning(f"텍스처 복잡도 계산 실패: {e}")
        return 0.0

# ==============================================
# 🔥 통합 품질 평가 함수들
# ==============================================

def calculate_comprehensive_quality_score(image: np.ndarray, 
                                       reference_image: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """종합적인 품질 점수 계산"""
    try:
        quality_metrics = {}
        
        # 기본 메트릭
        quality_metrics['brightness'] = calculate_brightness(image)
        quality_metrics['contrast'] = calculate_contrast(image)
        quality_metrics['sharpness'] = calculate_sharpness(image)
        quality_metrics['noise_level'] = calculate_noise_level(image)
        quality_metrics['edge_density'] = calculate_edge_density(image)
        quality_metrics['texture_complexity'] = calculate_texture_complexity(image)
        
        # 색상 메트릭
        color_stats = calculate_color_distribution(image)
        if 'error' not in color_stats:
            quality_metrics.update(color_stats)
        
        # 참조 이미지가 있는 경우 상대적 메트릭
        if reference_image is not None:
            quality_metrics['ssim'] = calculate_ssim_score(image, reference_image)
            quality_metrics['psnr'] = calculate_psnr_score(image, reference_image)
        
        # 전체 품질 점수 계산
        overall_score = _calculate_weighted_quality_score(quality_metrics)
        
        return {
            'success': True,
            'metrics': quality_metrics,
            'overall_score': overall_score,
            'quality_grade': _get_quality_grade(overall_score)
        }
        
    except Exception as e:
        logger.error(f"종합 품질 점수 계산 실패: {e}")
        return {
            'success': False,
            'error': str(e)
        }

def _calculate_weighted_quality_score(metrics: Dict[str, float]) -> float:
    """가중치를 적용한 품질 점수 계산"""
    try:
        # 메트릭별 가중치 정의
        weights = {
            'brightness': 0.15,
            'contrast': 0.15,
            'sharpness': 0.25,
            'noise_level': 0.20,
            'edge_density': 0.10,
            'texture_complexity': 0.05,
            'ssim': 0.10,
            'psnr': 0.10
        }
        
        total_score = 0.0
        total_weight = 0.0
        
        for metric_name, value in metrics.items():
            if metric_name in weights and isinstance(value, (int, float)):
                # 노이즈는 낮을수록 좋음 (역수 처리)
                if metric_name == 'noise_level':
                    normalized_value = max(0, 100 - value)
                else:
                    normalized_value = min(100, value)
                
                total_score += normalized_value * weights[metric_name]
                total_weight += weights[metric_name]
        
        return total_score / total_weight if total_weight > 0 else 0.0
        
    except Exception as e:
        logger.error(f"가중 품질 점수 계산 실패: {e}")
        return 0.0

def _get_quality_grade(score: float) -> str:
    """품질 등급 결정"""
    if score >= 90:
        return 'A+'
    elif score >= 80:
        return 'A'
    elif score >= 70:
        return 'B+'
    elif score >= 60:
        return 'B'
    elif score >= 50:
        return 'C+'
    elif score >= 40:
        return 'C'
    else:
        return 'D'

# ==============================================
# 🔥 배치 처리 함수들
# ==============================================

def batch_quality_assessment(images: List[np.ndarray], 
                           reference_images: Optional[List[np.ndarray]] = None) -> List[Dict[str, Any]]:
    """배치 이미지 품질 평가"""
    results = []
    
    for i, image in enumerate(images):
        reference = reference_images[i] if reference_images and i < len(reference_images) else None
        result = calculate_comprehensive_quality_score(image, reference)
        result['image_index'] = i
        results.append(result)
    
    return results

def generate_quality_report(assessment_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """품질 평가 보고서 생성"""
    try:
        if not assessment_results:
            return {'error': '평가 결과가 없습니다.'}
        
        # 통계 계산
        scores = []
        grades = []
        successful_assessments = 0
        
        for result in assessment_results:
            if result.get('success', False):
                successful_assessments += 1
                scores.append(result.get('overall_score', 0))
                grades.append(result.get('quality_grade', 'F'))
        
        if not scores:
            return {'error': '유효한 평가 결과가 없습니다.'}
        
        # 등급별 분포
        grade_distribution = {}
        for grade in grades:
            grade_distribution[grade] = grade_distribution.get(grade, 0) + 1
        
        return {
            'total_images': len(assessment_results),
            'successful_assessments': successful_assessments,
            'average_score': sum(scores) / len(scores),
            'min_score': min(scores),
            'max_score': max(scores),
            'grade_distribution': grade_distribution,
            'overall_quality': _get_quality_grade(sum(scores) / len(scores))
        }
        
    except Exception as e:
        logger.error(f"품질 보고서 생성 실패: {e}")
        return {'error': str(e)}

class MetricsUtils:
    """품질 평가 메트릭 유틸리티 클래스"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.MetricsUtils")
    
    def calculate_image_metrics(self, image: np.ndarray) -> Dict[str, float]:
        """이미지 메트릭 계산"""
        try:
            metrics = {}
            
            # 기본 메트릭들
            metrics['brightness'] = calculate_brightness(image)
            metrics['contrast'] = calculate_contrast(image)
            metrics['sharpness'] = calculate_sharpness(image)
            metrics['noise_level'] = calculate_noise_level(image)
            
            # 고급 메트릭들
            if SKIMAGE_AVAILABLE:
                try:
                    # SSIM과 PSNR은 참조 이미지가 필요하므로 기본값 설정
                    metrics['ssim'] = 0.8  # 기본값
                    metrics['psnr'] = 25.0  # 기본값
                except:
                    metrics['ssim'] = 0.0
                    metrics['psnr'] = 0.0
            else:
                metrics['ssim'] = 0.0
                metrics['psnr'] = 0.0
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"이미지 메트릭 계산 실패: {e}")
            return {}
    
    def compare_images(self, image1: np.ndarray, image2: np.ndarray) -> Dict[str, float]:
        """두 이미지 비교"""
        try:
            if not SKIMAGE_AVAILABLE:
                return {'error': 'skimage 라이브러리가 필요합니다.'}
            
            # 이미지 크기 통일
            if image1.shape != image2.shape:
                # image2를 image1 크기로 리사이즈
                if CV2_AVAILABLE:
                    image2_resized = cv2.resize(image2, (image1.shape[1], image1.shape[0]))
                else:
                    # PIL을 사용한 리사이즈
                    from PIL import Image
                    img1 = Image.fromarray(image1)
                    img2 = Image.fromarray(image2)
                    img2_resized = img2.resize(img1.size, Image.Resampling.LANCZOS)
                    image2_resized = np.array(img2_resized)
            else:
                image2_resized = image2
            
            # 그레이스케일 변환
            if len(image1.shape) == 3:
                gray1 = cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY) if CV2_AVAILABLE else np.mean(image1, axis=2)
                gray2 = cv2.cvtColor(image2_resized, cv2.COLOR_RGB2GRAY) if CV2_AVAILABLE else np.mean(image2_resized, axis=2)
            else:
                gray1 = image1
                gray2 = image2_resized
            
            # 메트릭 계산
            ssim_score = ssim(gray1, gray2) if ssim else 0.0
            psnr_score = psnr(gray1, gray2) if psnr else 0.0
            
            return {
                'ssim': ssim_score,
                'psnr': psnr_score,
                'brightness_diff': abs(calculate_brightness(gray1) - calculate_brightness(gray2)),
                'contrast_diff': abs(calculate_contrast(gray1) - calculate_contrast(gray2))
            }
            
        except Exception as e:
            self.logger.error(f"이미지 비교 실패: {e}")
            return {'error': str(e)}
    
    def generate_quality_summary(self, metrics: Dict[str, float]) -> Dict[str, Any]:
        """품질 요약 생성"""
        try:
            summary = {
                'overall_score': 0.0,
                'quality_grade': 'F',
                'strengths': [],
                'weaknesses': [],
                'recommendations': []
            }
            
            # 점수 계산
            score = 0.0
            total_metrics = 0
            
            for metric_name, value in metrics.items():
                if isinstance(value, (int, float)) and metric_name not in ['ssim', 'psnr']:
                    # 메트릭별 점수 정규화 (0-100)
                    if metric_name == 'brightness':
                        normalized_score = min(100, max(0, (value / 128) * 100))
                    elif metric_name == 'contrast':
                        normalized_score = min(100, max(0, value / 50 * 100))
                    elif metric_name == 'sharpness':
                        normalized_score = min(100, max(0, value / 1000 * 100))
                    elif metric_name == 'noise_level':
                        normalized_score = max(0, 100 - value)
                    else:
                        normalized_score = min(100, max(0, value))
                    
                    score += normalized_score
                    total_metrics += 1
            
            if total_metrics > 0:
                summary['overall_score'] = score / total_metrics
                summary['quality_grade'] = _get_quality_grade(summary['overall_score'])
            
            # 강점과 약점 분석
            for metric_name, value in metrics.items():
                if isinstance(value, (int, float)):
                    if metric_name == 'brightness':
                        if 40 <= value <= 200:
                            summary['strengths'].append(f"적절한 밝기 ({value:.1f})")
                        else:
                            summary['weaknesses'].append(f"밝기 조정 필요 ({value:.1f})")
                    
                    elif metric_name == 'contrast':
                        if value >= 30:
                            summary['strengths'].append(f"좋은 대비 ({value:.1f})")
                        else:
                            summary['weaknesses'].append(f"대비 향상 필요 ({value:.1f})")
                    
                    elif metric_name == 'sharpness':
                        if value >= 100:
                            summary['strengths'].append(f"선명한 이미지 ({value:.1f})")
                        else:
                            summary['weaknesses'].append(f"선명도 향상 필요 ({value:.1f})")
            
            # 권장사항 생성
            if summary['overall_score'] < 60:
                summary['recommendations'].append("전체적인 이미지 품질 향상이 필요합니다")
            if '밝기 조정 필요' in summary['weaknesses']:
                summary['recommendations'].append("밝기와 노출을 조정하세요")
            if '대비 향상 필요' in summary['weaknesses']:
                summary['recommendations'].append("대비와 채도를 향상시키세요")
            if '선명도 향상 필요' in summary['weaknesses']:
                summary['recommendations'].append("선명도와 해상도를 향상시키세요")
            
            return summary
            
        except Exception as e:
            self.logger.error(f"품질 요약 생성 실패: {e}")
            return {'error': str(e)}

# 전역 함수들 export
__all__ = [
    'calculate_brightness',
    'calculate_contrast', 
    'calculate_sharpness',
    'calculate_noise_level',
    'calculate_comprehensive_quality_score',
    'batch_quality_assessment',
    'generate_quality_report',
    'MetricsUtils'
]
