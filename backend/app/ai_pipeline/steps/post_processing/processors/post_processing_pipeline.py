"""
🔥 후처리 파이프라인 프로세서
=============================

후처리를 위한 완전한 파이프라인 시스템:
1. 품질 향상 파이프라인
2. 결과 최적화 파이프라인
3. 품질 검증 파이프라인
4. 앙상블 후처리 파이프라인

Author: MyCloset AI Team
Date: 2025-08-14
Version: 1.0
"""

import logging
import time
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import cv2

logger = logging.getLogger(__name__)

class PostProcessingPipeline:
    """후처리 파이프라인"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.PostProcessingPipeline")
        
        # 파이프라인 통계
        self.pipeline_stats = {
            'total_pipelines': 0,
            'successful_pipelines': 0,
            'failed_pipelines': 0,
            'average_pipeline_time': 0.0,
            'pipeline_types': {}
        }
        
        # 파이프라인 단계 정의
        self.pipeline_steps = {
            'quality_enhancement': [
                'noise_reduction',
                'sharpness_enhancement',
                'contrast_enhancement',
                'color_balance'
            ],
            'result_optimization': [
                'brightness_optimization',
                'color_optimization',
                'gamma_correction',
                'final_tuning'
            ],
            'quality_validation': [
                'metric_calculation',
                'quality_assessment',
                'result_validation',
                'performance_analysis'
            ]
        }
    
    def run_quality_enhancement_pipeline(self, image: np.ndarray, config: Dict[str, Any] = None) -> Dict[str, Any]:
        """품질 향상 파이프라인 실행"""
        try:
            start_time = time.time()
            self.logger.info("🚀 품질 향상 파이프라인 시작")
            
            if config is None:
                config = self._get_default_enhancement_config()
            
            # 파이프라인 단계별 실행
            result = self._execute_pipeline_steps(image, self.pipeline_steps['quality_enhancement'], config)
            
            # 파이프라인 통계 업데이트
            pipeline_time = time.time() - start_time
            self._update_pipeline_stats('quality_enhancement', True, pipeline_time)
            
            self.logger.info(f"✅ 품질 향상 파이프라인 완료 ({pipeline_time:.2f}s)")
            return result
            
        except Exception as e:
            self.logger.error(f"❌ 품질 향상 파이프라인 실패: {e}")
            self._update_pipeline_stats('quality_enhancement', False, 0.0)
            raise
    
    def run_result_optimization_pipeline(self, image: np.ndarray, config: Dict[str, Any] = None) -> Dict[str, Any]:
        """결과 최적화 파이프라인 실행"""
        try:
            start_time = time.time()
            self.logger.info("🚀 결과 최적화 파이프라인 시작")
            
            if config is None:
                config = self._get_default_optimization_config()
            
            # 파이프라인 단계별 실행
            result = self._execute_pipeline_steps(image, self.pipeline_steps['result_optimization'], config)
            
            # 파이프라인 통계 업데이트
            pipeline_time = time.time() - start_time
            self._update_pipeline_stats('result_optimization', True, pipeline_time)
            
            self.logger.info(f"✅ 결과 최적화 파이프라인 완료 ({pipeline_time:.2f}s)")
            return result
            
        except Exception as e:
            self.logger.error(f"❌ 결과 최적화 파이프라인 실패: {e}")
            self._update_pipeline_stats('result_optimization', False, 0.0)
            raise
    
    def run_quality_validation_pipeline(self, original_image: np.ndarray, processed_image: np.ndarray) -> Dict[str, Any]:
        """품질 검증 파이프라인 실행"""
        try:
            start_time = time.time()
            self.logger.info("🚀 품질 검증 파이프라인 시작")
            
            # 품질 메트릭 계산
            quality_metrics = self._calculate_comprehensive_quality_metrics(original_image, processed_image)
            
            # 품질 평가
            quality_assessment = self._assess_quality(quality_metrics)
            
            # 결과 검증
            validation_result = self._validate_results(quality_metrics, quality_assessment)
            
            # 성능 분석
            performance_analysis = self._analyze_performance(quality_metrics)
            
            result = {
                'quality_metrics': quality_metrics,
                'quality_assessment': quality_assessment,
                'validation_result': validation_result,
                'performance_analysis': performance_analysis
            }
            
            # 파이프라인 통계 업데이트
            pipeline_time = time.time() - start_time
            self._update_pipeline_stats('quality_validation', True, pipeline_time)
            
            self.logger.info(f"✅ 품질 검증 파이프라인 완료 ({pipeline_time:.2f}s)")
            return result
            
        except Exception as e:
            self.logger.error(f"❌ 품질 검증 파이프라인 실패: {e}")
            self._update_pipeline_stats('quality_validation', False, 0.0)
            raise
    
    def _execute_pipeline_steps(self, image: np.ndarray, steps: List[str], config: Dict[str, Any]) -> Dict[str, Any]:
        """파이프라인 단계별 실행"""
        try:
            current_image = image.copy()
            step_results = {}
            
            for step in steps:
                self.logger.info(f"📋 파이프라인 단계 실행: {step}")
                
                if step == 'noise_reduction':
                    current_image = self._apply_noise_reduction(current_image, config.get('noise_reduction', {}))
                elif step == 'sharpness_enhancement':
                    current_image = self._apply_sharpness_enhancement(current_image, config.get('sharpness_enhancement', {}))
                elif step == 'contrast_enhancement':
                    current_image = self._apply_contrast_enhancement(current_image, config.get('contrast_enhancement', {}))
                elif step == 'color_balance':
                    current_image = self._apply_color_balance(current_image, config.get('color_balance', {}))
                elif step == 'brightness_optimization':
                    current_image = self._apply_brightness_optimization(current_image, config.get('brightness_optimization', {}))
                elif step == 'color_optimization':
                    current_image = self._apply_color_optimization(current_image, config.get('color_optimization', {}))
                elif step == 'gamma_correction':
                    current_image = self._apply_gamma_correction(current_image, config.get('gamma_correction', {}))
                elif step == 'final_tuning':
                    current_image = self._apply_final_tuning(current_image, config.get('final_tuning', {}))
                
                step_results[step] = current_image.copy()
            
            return {
                'final_result': current_image,
                'step_results': step_results,
                'pipeline_steps': steps
            }
            
        except Exception as e:
            self.logger.error(f"❌ 파이프라인 단계 실행 실패: {e}")
            raise
    
    def _apply_noise_reduction(self, image: np.ndarray, config: Dict[str, Any]) -> np.ndarray:
        """노이즈 제거 적용"""
        try:
            # Non-local Means Denoising
            denoised = cv2.fastNlMeansDenoisingColored(
                image, 
                None, 
                config.get('h', 10),
                config.get('hColor', 10),
                config.get('templateWindowSize', 7),
                config.get('searchWindowSize', 21)
            )
            
            # Bilateral Filter 추가 적용
            denoised = cv2.bilateralFilter(
                denoised,
                config.get('d', 9),
                config.get('sigmaColor', 75),
                config.get('sigmaSpace', 75)
            )
            
            return denoised
            
        except Exception as e:
            self.logger.error(f"❌ 노이즈 제거 적용 실패: {e}")
            return image
    
    def _apply_sharpness_enhancement(self, image: np.ndarray, config: Dict[str, Any]) -> np.ndarray:
        """선명도 향상 적용"""
        try:
            # Unsharp Masking
            gaussian = cv2.GaussianBlur(image, (0, 0), config.get('sigma', 2.0))
            sharpened = cv2.addWeighted(
                image, 
                config.get('alpha', 1.5), 
                gaussian, 
                config.get('beta', -0.5), 
                0
            )
            
            # 추가 선명도 향상
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            sharpened = cv2.filter2D(sharpened, -1, kernel)
            
            return sharpened
            
        except Exception as e:
            self.logger.error(f"❌ 선명도 향상 적용 실패: {e}")
            return image
    
    def _apply_contrast_enhancement(self, image: np.ndarray, config: Dict[str, Any]) -> np.ndarray:
        """대비 향상 적용"""
        try:
            # LAB 색공간으로 변환
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # CLAHE 적용 (L 채널)
            clahe = cv2.createCLAHE(
                clipLimit=config.get('clipLimit', 3.0),
                tileGridSize=tuple(config.get('tileGridSize', [8, 8]))
            )
            l = clahe.apply(l)
            
            # 채널 합치기
            enhanced = cv2.merge([l, a, b])
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
            
            return enhanced
            
        except Exception as e:
            self.logger.error(f"❌ 대비 향상 적용 실패: {e}")
            return image
    
    def _apply_color_balance(self, image: np.ndarray, config: Dict[str, Any]) -> np.ndarray:
        """색상 균형 조정 적용"""
        try:
            # 색상 균형 조정
            balanced = cv2.convertScaleAbs(
                image, 
                alpha=config.get('alpha', 1.1), 
                beta=config.get('beta', 5)
            )
            
            # 감마 보정
            gamma = config.get('gamma', 1.1)
            invGamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
            balanced = cv2.LUT(balanced, table)
            
            return balanced
            
        except Exception as e:
            self.logger.error(f"❌ 색상 균형 조정 적용 실패: {e}")
            return image
    
    def _apply_brightness_optimization(self, image: np.ndarray, config: Dict[str, Any]) -> np.ndarray:
        """밝기 최적화 적용"""
        try:
            # 밝기 조정
            optimized = cv2.convertScaleAbs(
                image,
                alpha=config.get('alpha', 1.0),
                beta=config.get('beta', 10)
            )
            
            return optimized
            
        except Exception as e:
            self.logger.error(f"❌ 밝기 최적화 적용 실패: {e}")
            return image
    
    def _apply_color_optimization(self, image: np.ndarray, config: Dict[str, Any]) -> np.ndarray:
        """색상 최적화 적용"""
        try:
            # 색상 채널별 조정
            b, g, r = cv2.split(image)
            
            # 각 채널에 대한 가중치 적용
            b = cv2.convertScaleAbs(b, alpha=config.get('blue_alpha', 1.0))
            g = cv2.convertScaleAbs(g, alpha=config.get('green_alpha', 1.0))
            r = cv2.convertScaleAbs(r, alpha=config.get('red_alpha', 1.0))
            
            optimized = cv2.merge([b, g, r])
            return optimized
            
        except Exception as e:
            self.logger.error(f"❌ 색상 최적화 적용 실패: {e}")
            return image
    
    def _apply_gamma_correction(self, image: np.ndarray, config: Dict[str, Any]) -> np.ndarray:
        """감마 보정 적용"""
        try:
            gamma = config.get('gamma', 1.1)
            invGamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
            corrected = cv2.LUT(image, table)
            
            return corrected
            
        except Exception as e:
            self.logger.error(f"❌ 감마 보정 적용 실패: {e}")
            return image
    
    def _apply_final_tuning(self, image: np.ndarray, config: Dict[str, Any]) -> np.ndarray:
        """최종 튜닝 적용"""
        try:
            # 최종 품질 조정
            tuned = image.copy()
            
            # 약간의 블러로 노이즈 제거
            if config.get('final_blur', False):
                tuned = cv2.GaussianBlur(tuned, (3, 3), 0.5)
            
            # 최종 색상 조정
            tuned = cv2.convertScaleAbs(
                tuned,
                alpha=config.get('final_alpha', 1.0),
                beta=config.get('final_beta', 0)
            )
            
            return tuned
            
        except Exception as e:
            self.logger.error(f"❌ 최종 튜닝 적용 실패: {e}")
            return image
    
    def _calculate_comprehensive_quality_metrics(self, original: np.ndarray, processed: np.ndarray) -> Dict[str, float]:
        """종합 품질 메트릭 계산"""
        try:
            metrics = {}
            
            # PSNR 계산
            mse = np.mean((original.astype(float) - processed.astype(float)) ** 2)
            if mse == 0:
                metrics['psnr'] = float('inf')
            else:
                metrics['psnr'] = 20 * np.log10(255.0 / np.sqrt(mse))
            
            # SSIM 계산
            metrics['ssim'] = self._calculate_ssim(original, processed)
            
            # 대비 개선도
            original_contrast = np.std(original)
            processed_contrast = np.std(processed)
            metrics['contrast_improvement'] = processed_contrast / original_contrast if original_contrast > 0 else 1.0
            
            # 선명도 개선도
            original_sharpness = self._calculate_sharpness(original)
            processed_sharpness = self._calculate_sharpness(processed)
            metrics['sharpness_improvement'] = processed_sharpness / original_sharpness if original_sharpness > 0 else 1.0
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"❌ 종합 품질 메트릭 계산 실패: {e}")
            return {'psnr': 0.0, 'ssim': 0.0, 'contrast_improvement': 1.0, 'sharpness_improvement': 1.0}
    
    def _calculate_ssim(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """SSIM 계산"""
        try:
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
    
    def _assess_quality(self, quality_metrics: Dict[str, float]) -> Dict[str, Any]:
        """품질 평가"""
        try:
            assessment = {}
            
            # PSNR 평가
            psnr = quality_metrics.get('psnr', 0)
            if psnr > 30:
                assessment['psnr_grade'] = 'Excellent'
            elif psnr > 25:
                assessment['psnr_grade'] = 'Good'
            elif psnr > 20:
                assessment['psnr_grade'] = 'Fair'
            else:
                assessment['psnr_grade'] = 'Poor'
            
            # SSIM 평가
            ssim = quality_metrics.get('ssim', 0)
            if ssim > 0.9:
                assessment['ssim_grade'] = 'Excellent'
            elif ssim > 0.8:
                assessment['ssim_grade'] = 'Good'
            elif ssim > 0.7:
                assessment['ssim_grade'] = 'Fair'
            else:
                assessment['ssim_grade'] = 'Poor'
            
            # 종합 등급
            if assessment.get('psnr_grade') == 'Excellent' and assessment.get('ssim_grade') == 'Excellent':
                assessment['overall_grade'] = 'A+'
            elif assessment.get('psnr_grade') in ['Excellent', 'Good'] and assessment.get('ssim_grade') in ['Excellent', 'Good']:
                assessment['overall_grade'] = 'A'
            elif assessment.get('psnr_grade') in ['Good', 'Fair'] and assessment.get('ssim_grade') in ['Good', 'Fair']:
                assessment['overall_grade'] = 'B'
            else:
                assessment['overall_grade'] = 'C'
            
            return assessment
            
        except Exception as e:
            self.logger.error(f"❌ 품질 평가 실패: {e}")
            return {'overall_grade': 'Unknown'}
    
    def _validate_results(self, quality_metrics: Dict[str, float], quality_assessment: Dict[str, Any]) -> Dict[str, Any]:
        """결과 검증"""
        try:
            validation = {
                'is_valid': True,
                'warnings': [],
                'recommendations': []
            }
            
            # PSNR 검증
            psnr = quality_metrics.get('psnr', 0)
            if psnr < 20:
                validation['warnings'].append("PSNR이 너무 낮습니다. 품질 향상이 필요합니다.")
                validation['is_valid'] = False
            
            # SSIM 검증
            ssim = quality_metrics.get('ssim', 0)
            if ssim < 0.7:
                validation['warnings'].append("SSIM이 너무 낮습니다. 구조적 유사성이 부족합니다.")
                validation['is_valid'] = False
            
            # 개선도 검증
            contrast_improvement = quality_metrics.get('contrast_improvement', 1.0)
            if contrast_improvement < 0.8:
                validation['warnings'].append("대비 개선도가 부족합니다.")
                validation['recommendations'].append("대비 향상 파라미터를 조정해보세요.")
            
            sharpness_improvement = quality_metrics.get('sharpness_improvement', 1.0)
            if sharpness_improvement < 0.8:
                validation['warnings'].append("선명도 개선도가 부족합니다.")
                validation['recommendations'].append("선명도 향상 파라미터를 조정해보세요.")
            
            return validation
            
        except Exception as e:
            self.logger.error(f"❌ 결과 검증 실패: {e}")
            return {'is_valid': False, 'warnings': ['검증 중 오류가 발생했습니다.'], 'recommendations': []}
    
    def _analyze_performance(self, quality_metrics: Dict[str, float]) -> Dict[str, Any]:
        """성능 분석"""
        try:
            analysis = {
                'performance_score': 0.0,
                'strengths': [],
                'weaknesses': [],
                'improvement_areas': []
            }
            
            # 성능 점수 계산
            psnr_score = min(quality_metrics.get('psnr', 0) / 40.0, 1.0)  # 40dB 이상을 만점
            ssim_score = quality_metrics.get('ssim', 0)
            contrast_score = min(quality_metrics.get('contrast_improvement', 1.0), 1.5) / 1.5
            sharpness_score = min(quality_metrics.get('sharpness_improvement', 1.0), 1.5) / 1.5
            
            performance_score = (psnr_score + ssim_score + contrast_score + sharpness_score) / 4.0
            analysis['performance_score'] = performance_score
            
            # 강점 분석
            if psnr_score > 0.8:
                analysis['strengths'].append("높은 PSNR로 우수한 품질")
            if ssim_score > 0.9:
                analysis['strengths'].append("높은 SSIM으로 구조적 유사성 우수")
            if contrast_score > 0.9:
                analysis['strengths'].append("대비 개선 효과 우수")
            if sharpness_score > 0.9:
                analysis['strengths'].append("선명도 개선 효과 우수")
            
            # 약점 분석
            if psnr_score < 0.6:
                analysis['weaknesses'].append("PSNR이 낮아 품질 개선 필요")
            if ssim_score < 0.8:
                analysis['weaknesses'].append("SSIM이 낮아 구조적 유사성 개선 필요")
            if contrast_score < 0.8:
                analysis['weaknesses'].append("대비 개선 효과 부족")
            if sharpness_score < 0.8:
                analysis['weaknesses'].append("선명도 개선 효과 부족")
            
            # 개선 영역
            if psnr_score < 0.7:
                analysis['improvement_areas'].append("노이즈 제거 및 품질 향상")
            if ssim_score < 0.8:
                analysis['improvement_areas'].append("구조적 보존 강화")
            if contrast_score < 0.8:
                analysis['improvement_areas'].append("대비 향상 파라미터 조정")
            if sharpness_score < 0.8:
                analysis['improvement_areas'].append("선명도 향상 파라미터 조정")
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"❌ 성능 분석 실패: {e}")
            return {'performance_score': 0.0, 'strengths': [], 'weaknesses': [], 'improvement_areas': []}
    
    def _get_default_enhancement_config(self) -> Dict[str, Any]:
        """기본 향상 설정 반환"""
        return {
            'noise_reduction': {'h': 10, 'hColor': 10, 'templateWindowSize': 7, 'searchWindowSize': 21, 'd': 9, 'sigmaColor': 75, 'sigmaSpace': 75},
            'sharpness_enhancement': {'sigma': 2.0, 'alpha': 1.5, 'beta': -0.5},
            'contrast_enhancement': {'clipLimit': 3.0, 'tileGridSize': [8, 8]},
            'color_balance': {'alpha': 1.1, 'beta': 5, 'gamma': 1.1}
        }
    
    def _get_default_optimization_config(self) -> Dict[str, Any]:
        """기본 최적화 설정 반환"""
        return {
            'brightness_optimization': {'alpha': 1.0, 'beta': 10},
            'color_optimization': {'blue_alpha': 1.0, 'green_alpha': 1.0, 'red_alpha': 1.0},
            'gamma_correction': {'gamma': 1.1},
            'final_tuning': {'final_blur': False, 'final_alpha': 1.0, 'final_beta': 0}
        }
    
    def _update_pipeline_stats(self, pipeline_type: str, success: bool, pipeline_time: float):
        """파이프라인 통계 업데이트"""
        try:
            self.pipeline_stats['total_pipelines'] += 1
            
            if success:
                self.pipeline_stats['successful_pipelines'] += 1
                
                # 평균 파이프라인 시간 업데이트
                total_successful = self.pipeline_stats['successful_pipelines']
                current_avg = self.pipeline_stats['average_pipeline_time']
                new_avg = (current_avg * (total_successful - 1) + pipeline_time) / total_successful
                self.pipeline_stats['average_pipeline_time'] = new_avg
            else:
                self.pipeline_stats['failed_pipelines'] += 1
            
            # 파이프라인 타입별 통계
            if pipeline_type not in self.pipeline_stats['pipeline_types']:
                self.pipeline_stats['pipeline_types'][pipeline_type] = {'total': 0, 'successful': 0, 'failed': 0}
            
            self.pipeline_stats['pipeline_types'][pipeline_type]['total'] += 1
            if success:
                self.pipeline_stats['pipeline_types'][pipeline_type]['successful'] += 1
            else:
                self.pipeline_stats['pipeline_types'][pipeline_type]['failed'] += 1
                
        except Exception as e:
            self.logger.error(f"❌ 파이프라인 통계 업데이트 실패: {e}")
    
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """파이프라인 통계 반환"""
        return self.pipeline_stats.copy()
    
    def reset_pipeline_stats(self):
        """파이프라인 통계 초기화"""
        self.pipeline_stats = {
            'total_pipelines': 0,
            'successful_pipelines': 0,
            'failed_pipelines': 0,
            'average_pipeline_time': 0.0,
            'pipeline_types': {}
        }
