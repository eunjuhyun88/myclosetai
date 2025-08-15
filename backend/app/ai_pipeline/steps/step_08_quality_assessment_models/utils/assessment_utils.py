#!/usr/bin/env python3
"""
🔥 MyCloset AI - Quality Assessment Utils
=========================================

품질 평가를 위한 유틸리티 함수들
- 이미지 품질 메트릭 계산
- 품질 점수 산출
- 품질 향상 제안

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
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    ssim = None
    psnr = None

# ==============================================
# 🔥 품질 평가 메트릭 함수들
# ==============================================

def calculate_image_quality_metrics(image: np.ndarray, reference_image: Optional[np.ndarray] = None) -> Dict[str, float]:
    """이미지 품질 메트릭 계산"""
    metrics = {}
    
    try:
        if CV2_AVAILABLE:
            # Laplacian variance (선명도)
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            metrics['sharpness'] = float(laplacian_var)
            
            # 노이즈 레벨 추정
            noise_level = estimate_noise_level(gray)
            metrics['noise_level'] = float(noise_level)
        
        # 참조 이미지가 있는 경우 상대적 메트릭 계산
        if reference_image is not None and SKIMAGE_AVAILABLE:
            try:
                # SSIM (구조적 유사성)
                ssim_score = ssim(image, reference_image, multichannel=True)
                metrics['ssim'] = float(ssim_score)
                
                # PSNR (피크 신호 대 잡음비)
                psnr_score = psnr(image, reference_image)
                metrics['psnr'] = float(psnr_score)
            except Exception as e:
                logger.warning(f"상대적 메트릭 계산 실패: {e}")
        
        # 기본 통계 메트릭
        metrics['mean_intensity'] = float(np.mean(image))
        metrics['std_intensity'] = float(np.std(image))
        metrics['contrast'] = float(np.max(image) - np.min(image))
        
    except Exception as e:
        logger.error(f"품질 메트릭 계산 실패: {e}")
        metrics = {'error': str(e)}
    
    return metrics

def estimate_noise_level(image: np.ndarray) -> float:
    """이미지 노이즈 레벨 추정"""
    try:
        if CV2_AVAILABLE:
            # 가우시안 블러를 사용한 노이즈 추정
            blurred = cv2.GaussianBlur(image, (5, 5), 0)
            noise = image.astype(np.float64) - blurred.astype(np.float64)
            noise_level = np.std(noise)
            return float(noise_level)
        else:
            # 간단한 차분 기반 노이즈 추정
            diff = np.diff(image, axis=1)
            noise_level = np.std(diff)
            return float(noise_level)
    except Exception as e:
        logger.warning(f"노이즈 레벨 추정 실패: {e}")
        return 0.0

def calculate_overall_quality_score(metrics: Dict[str, float]) -> Dict[str, Any]:
    """전체 품질 점수 계산"""
    try:
        score = 0.0
        max_score = 100.0
        factors = {}
        
        # 선명도 점수 (0-30점)
        if 'sharpness' in metrics:
            sharpness_score = min(30.0, metrics['sharpness'] / 100.0)
            factors['sharpness'] = sharpness_score
            score += sharpness_score
        
        # 노이즈 점수 (0-25점)
        if 'noise_level' in metrics:
            noise_score = max(0.0, 25.0 - (metrics['noise_level'] / 10.0))
            factors['noise'] = noise_score
            score += noise_score
        
        # 대비 점수 (0-20점)
        if 'contrast' in metrics:
            contrast_score = min(20.0, metrics['contrast'] / 50.0)
            factors['contrast'] = contrast_score
            score += contrast_score
        
        # SSIM 점수 (0-15점)
        if 'ssim' in metrics:
            ssim_score = metrics['ssim'] * 15.0
            factors['ssim'] = ssim_score
            score += ssim_score
        
        # PSNR 점수 (0-10점)
        if 'psnr' in metrics:
            psnr_score = min(10.0, metrics['psnr'] / 10.0)
            factors['psnr'] = psnr_score
            score += psnr_score
        
        # 품질 등급 결정
        if score >= 90:
            grade = "A+"
        elif score >= 80:
            grade = "A"
        elif score >= 70:
            grade = "B+"
        elif score >= 60:
            grade = "B"
        elif score >= 50:
            grade = "C+"
        elif score >= 40:
            grade = "C"
        else:
            grade = "D"
        
        return {
            'overall_score': float(score),
            'max_score': float(max_score),
            'percentage': float((score / max_score) * 100),
            'grade': grade,
            'factors': factors,
            'metrics': metrics
        }
        
    except Exception as e:
        logger.error(f"전체 품질 점수 계산 실패: {e}")
        return {
            'overall_score': 0.0,
            'max_score': 100.0,
            'percentage': 0.0,
            'grade': 'F',
            'error': str(e)
        }

def generate_quality_improvement_suggestions(metrics: Dict[str, float], score_result: Dict[str, Any]) -> List[str]:
    """품질 향상 제안 생성"""
    suggestions = []
    
    try:
        overall_score = score_result.get('overall_score', 0)
        
        # 전반적인 품질이 낮은 경우
        if overall_score < 50:
            suggestions.append("이미지 품질이 매우 낮습니다. 촬영 환경을 개선하거나 고품질 카메라를 사용하세요.")
        
        # 선명도 관련 제안
        if 'sharpness' in metrics and metrics['sharpness'] < 100:
            suggestions.append("이미지가 흐릿합니다. 카메라를 고정하거나 셔터 속도를 높이세요.")
        
        # 노이즈 관련 제안
        if 'noise_level' in metrics and metrics['noise_level'] > 20:
            suggestions.append("노이즈가 많습니다. ISO 설정을 낮추거나 조명을 개선하세요.")
        
        # 대비 관련 제안
        if 'contrast' in metrics and metrics['contrast'] < 100:
            suggestions.append("대비가 낮습니다. 조명을 조정하거나 HDR 모드를 사용하세요.")
        
        # SSIM 관련 제안
        if 'ssim' in metrics and metrics['ssim'] < 0.7:
            suggestions.append("구조적 품질이 낮습니다. 이미지 압축을 줄이거나 고해상도로 촬영하세요.")
        
        # PSNR 관련 제안
        if 'psnr' in metrics and metrics['psnr'] < 20:
            suggestions.append("신호 대 잡음비가 낮습니다. 촬영 환경을 개선하거나 후처리를 적용하세요.")
        
        # 제안이 없는 경우
        if not suggestions:
            suggestions.append("이미지 품질이 양호합니다. 현재 설정을 유지하세요.")
            
    except Exception as e:
        logger.error(f"품질 향상 제안 생성 실패: {e}")
        suggestions = ["품질 분석 중 오류가 발생했습니다."]
    
    return suggestions

def assess_image_quality(image: np.ndarray, reference_image: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """이미지 품질 종합 평가"""
    try:
        # 품질 메트릭 계산
        metrics = calculate_image_quality_metrics(image, reference_image)
        
        # 전체 품질 점수 계산
        score_result = calculate_overall_quality_score(metrics)
        
        # 품질 향상 제안 생성
        suggestions = generate_quality_improvement_suggestions(metrics, score_result)
        
        return {
            'success': True,
            'metrics': metrics,
            'score_result': score_result,
            'suggestions': suggestions,
            'timestamp': str(np.datetime64('now'))
        }
        
    except Exception as e:
        logger.error(f"이미지 품질 평가 실패: {e}")
        return {
            'success': False,
            'error': str(e),
            'timestamp': str(np.datetime64('now'))
        }

# ==============================================
# 🔥 유틸리티 함수들
# ==============================================

def validate_image_input(image: np.ndarray) -> bool:
    """이미지 입력 유효성 검사"""
    try:
        if image is None:
            return False
        
        if not isinstance(image, np.ndarray):
            return False
        
        if len(image.shape) < 2 or len(image.shape) > 3:
            return False
        
        if image.size == 0:
            return False
        
        return True
        
    except Exception:
        return False

def normalize_image_for_analysis(image: np.ndarray) -> np.ndarray:
    """분석을 위한 이미지 정규화"""
    try:
        if not validate_image_input(image):
            raise ValueError("유효하지 않은 이미지 입력")
        
        # 0-255 범위로 정규화
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)
        
        return image
        
    except Exception as e:
        logger.error(f"이미지 정규화 실패: {e}")
        return image

def get_quality_assessment_summary(assessment_result: Dict[str, Any]) -> str:
    """품질 평가 결과 요약"""
    try:
        if not assessment_result.get('success', False):
            return "품질 평가 실패"
        
        score_result = assessment_result.get('score_result', {})
        overall_score = score_result.get('overall_score', 0)
        grade = score_result.get('grade', 'F')
        percentage = score_result.get('percentage', 0)
        
        summary = f"품질 점수: {overall_score:.1f}/100 ({percentage:.1f}%) - 등급: {grade}"
        
        # 주요 메트릭 추가
        metrics = assessment_result.get('metrics', {})
        if 'sharpness' in metrics:
            summary += f"\n선명도: {metrics['sharpness']:.1f}"
        if 'noise_level' in metrics:
            summary += f"\n노이즈: {metrics['noise_level']:.1f}"
        if 'contrast' in metrics:
            summary += f"\n대비: {metrics['contrast']:.1f}"
        
        return summary
        
    except Exception as e:
        logger.error(f"품질 평가 요약 생성 실패: {e}")
        return "품질 평가 요약 생성 실패"

# ==============================================
# 🔥 AssessmentUtils 클래스
# ==============================================

class AssessmentUtils:
    """품질 평가 유틸리티 클래스"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.assessment_history = []
        
    def assess_image_quality_batch(self, images: List[np.ndarray], 
                                 reference_images: Optional[List[np.ndarray]] = None) -> List[Dict[str, Any]]:
        """배치 이미지 품질 평가"""
        results = []
        
        for i, image in enumerate(images):
            reference = reference_images[i] if reference_images else None
            result = assess_image_quality(image, reference)
            result['image_index'] = i
            results.append(result)
            self.assessment_history.append(result)
        
        return results
    
    def get_average_quality_score(self, assessment_results: List[Dict[str, Any]]) -> float:
        """평균 품질 점수 계산"""
        if not assessment_results:
            return 0.0
        
        total_score = 0.0
        valid_results = 0
        
        for result in assessment_results:
            if result.get('success', False):
                score = result.get('score_result', {}).get('overall_score', 0)
                total_score += score
                valid_results += 1
        
        return total_score / valid_results if valid_results > 0 else 0.0
    
    def generate_quality_report(self, assessment_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """품질 평가 보고서 생성"""
        if not assessment_results:
            return {'error': '평가 결과가 없습니다.'}
        
        # 통계 계산
        scores = []
        grades = []
        successful_assessments = 0
        
        for result in assessment_results:
            if result.get('success', False):
                successful_assessments += 1
                score_result = result.get('score_result', {})
                scores.append(score_result.get('overall_score', 0))
                grades.append(score_result.get('grade', 'F'))
        
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
            'overall_quality': self._get_overall_quality_rating(scores)
        }
    
    def _get_overall_quality_rating(self, scores: List[float]) -> str:
        """전체 품질 등급 결정"""
        if not scores:
            return 'F'
        
        avg_score = sum(scores) / len(scores)
        
        if avg_score >= 90:
            return 'A+'
        elif avg_score >= 80:
            return 'A'
        elif avg_score >= 70:
            return 'B+'
        elif avg_score >= 60:
            return 'B'
        elif avg_score >= 50:
            return 'C+'
        elif avg_score >= 40:
            return 'C'
        else:
            return 'D'
    
    def export_assessment_history(self, file_path: str) -> bool:
        """평가 히스토리 내보내기"""
        try:
            import json
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(self.assessment_history, f, ensure_ascii=False, indent=2)
            self.logger.info(f"평가 히스토리 내보내기 성공: {file_path}")
            return True
        except Exception as e:
            self.logger.error(f"평가 히스토리 내보내기 실패: {e}")
            return False
    
    def clear_assessment_history(self):
        """평가 히스토리 초기화"""
        self.assessment_history.clear()
        self.logger.info("평가 히스토리 초기화 완료")

