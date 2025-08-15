"""
🔥 최종 후처리 프로세서
=======================

후처리를 위한 최종 처리 시스템:
1. 최종 품질 검증
2. 결과 최적화
3. 최종 출력 생성
4. 품질 보고서 생성

Author: MyCloset AI Team
Date: 2025-08-14
Version: 1.0
"""

import logging
import time
import numpy as np
import cv2
from typing import Dict, Any, Optional, List, Tuple
import json
import os

logger = logging.getLogger(__name__)

class FinalProcessor:
    """최종 후처리 프로세서"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.FinalProcessor")
        
        # 처리 통계
        self.processing_stats = {
            'total_processings': 0,
            'successful_processings': 0,
            'failed_processings': 0,
            'average_processing_time': 0.0
        }
        
        # 품질 임계값
        self.quality_thresholds = {
            'psnr_min': 25.0,
            'ssim_min': 0.8,
            'contrast_min': 0.8,
            'sharpness_min': 0.8,
            'color_balance_min': 0.8
        }
    
    def process_final_output(self, 
                           original_image: np.ndarray,
                           processed_image: np.ndarray,
                           quality_metrics: Dict[str, float],
                           output_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """최종 출력 처리"""
        try:
            start_time = time.time()
            self.logger.info("🚀 최종 출력 처리 시작")
            
            # 품질 검증
            validation_result = self._validate_final_quality(quality_metrics)
            
            # 결과 최적화
            optimized_image = self._optimize_final_result(processed_image, validation_result)
            
            # 최종 품질 메트릭 계산
            final_metrics = self._calculate_final_metrics(original_image, optimized_image)
            
            # 품질 보고서 생성
            quality_report = self._generate_quality_report(quality_metrics, final_metrics, validation_result)
            
            # 출력 설정 적용
            final_output = self._apply_output_config(optimized_image, output_config)
            
            # 처리 시간 업데이트
            processing_time = time.time() - start_time
            self._update_processing_stats(True, processing_time)
            
            result = {
                'final_image': final_output,
                'original_metrics': quality_metrics,
                'final_metrics': final_metrics,
                'validation_result': validation_result,
                'quality_report': quality_report,
                'processing_time': processing_time,
                'output_config': output_config or {}
            }
            
            self.logger.info(f"✅ 최종 출력 처리 완료 ({processing_time:.2f}s)")
            return result
            
        except Exception as e:
            self.logger.error(f"❌ 최종 출력 처리 실패: {e}")
            self._update_processing_stats(False, 0.0)
            raise
    
    def _validate_final_quality(self, quality_metrics: Dict[str, float]) -> Dict[str, Any]:
        """최종 품질 검증"""
        try:
            self.logger.info("🔍 최종 품질 검증 시작")
            
            validation_result = {
                'is_acceptable': True,
                'overall_score': 0.0,
                'passed_metrics': [],
                'failed_metrics': [],
                'warnings': [],
                'recommendations': []
            }
            
            # 각 메트릭 검증
            metric_scores = {}
            
            for metric_name, threshold_key in [
                ('psnr', 'psnr_min'),
                ('ssim', 'ssim_min'),
                ('contrast', 'contrast_min'),
                ('sharpness', 'sharpness_min'),
                ('color_balance', 'color_balance_min')
            ]:
                if metric_name in quality_metrics:
                    value = quality_metrics[metric_name]
                    threshold = self.quality_thresholds[threshold_key]
                    
                    # 메트릭별 점수 계산
                    if metric_name == 'psnr':
                        # PSNR: 높을수록 좋음
                        score = min(1.0, value / 40.0)  # 40dB 이상을 만점
                    elif metric_name == 'ssim':
                        # SSIM: 0~1 범위
                        score = value
                    else:
                        # 개선도 메트릭: 1.0 근처가 좋음
                        if 0.9 <= value <= 1.1:
                            score = 1.0
                        elif 0.8 <= value <= 1.2:
                            score = 0.8
                        elif 0.7 <= value <= 1.3:
                            score = 0.6
                        else:
                            score = 0.3
                    
                    metric_scores[metric_name] = score
                    
                    # 임계값 검증
                    if metric_name == 'psnr':
                        if value >= threshold:
                            validation_result['passed_metrics'].append(metric_name)
                        else:
                            validation_result['failed_metrics'].append(metric_name)
                            validation_result['warnings'].append(f"{metric_name}이 임계값({threshold})을 만족하지 않습니다")
                    else:
                        if value >= threshold:
                            validation_result['passed_metrics'].append(metric_name)
                        else:
                            validation_result['failed_metrics'].append(metric_name)
                            validation_result['warnings'].append(f"{metric_name}이 임계값({threshold})을 만족하지 않습니다")
            
            # 전체 점수 계산
            if metric_scores:
                validation_result['overall_score'] = np.mean(list(metric_scores.values()))
                
                # 전체 품질 판정
                if validation_result['overall_score'] >= 0.8:
                    validation_result['is_acceptable'] = True
                elif validation_result['overall_score'] >= 0.6:
                    validation_result['is_acceptable'] = True
                    validation_result['warnings'].append("전체 품질이 보통 수준입니다")
                else:
                    validation_result['is_acceptable'] = False
                    validation_result['warnings'].append("전체 품질이 낮습니다")
            
            # 개선 권장사항 생성
            if validation_result['failed_metrics']:
                for metric in validation_result['failed_metrics']:
                    if metric == 'psnr':
                        validation_result['recommendations'].append("노이즈 제거 및 품질 향상 파라미터 조정")
                    elif metric == 'ssim':
                        validation_result['recommendations'].append("구조적 보존 강화")
                    elif metric == 'contrast':
                        validation_result['recommendations'].append("대비 향상 파라미터 조정")
                    elif metric == 'sharpness':
                        validation_result['recommendations'].append("선명도 향상 파라미터 조정")
                    elif metric == 'color_balance':
                        validation_result['recommendations'].append("색상 균형 조정")
            
            self.logger.info(f"✅ 최종 품질 검증 완료: 전체 점수 = {validation_result['overall_score']:.3f}")
            return validation_result
            
        except Exception as e:
            self.logger.error(f"❌ 최종 품질 검증 실패: {e}")
            return {
                'is_acceptable': False,
                'overall_score': 0.0,
                'passed_metrics': [],
                'failed_metrics': [],
                'warnings': ['품질 검증 중 오류가 발생했습니다'],
                'recommendations': []
            }
    
    def _optimize_final_result(self, processed_image: np.ndarray, validation_result: Dict[str, Any]) -> np.ndarray:
        """최종 결과 최적화"""
        try:
            self.logger.info("🚀 최종 결과 최적화 시작")
            
            optimized = processed_image.copy()
            
            # 품질 점수에 따른 추가 최적화
            overall_score = validation_result.get('overall_score', 0.0)
            
            if overall_score < 0.7:
                # 품질이 낮은 경우 추가 최적화
                self.logger.info("📈 낮은 품질로 인한 추가 최적화 적용")
                
                # 약간의 블러로 노이즈 제거
                optimized = cv2.GaussianBlur(optimized, (3, 3), 0.5)
                
                # 색상 보정
                optimized = cv2.convertScaleAbs(optimized, alpha=1.05, beta=3)
                
                # 감마 보정
                gamma = 1.05
                invGamma = 1.0 / gamma
                table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
                optimized = cv2.LUT(optimized, table)
            
            elif overall_score < 0.8:
                # 품질이 보통인 경우 약간의 최적화
                self.logger.info("📊 보통 품질로 인한 약간의 최적화 적용")
                
                # 색상 균형 조정
                optimized = cv2.convertScaleAbs(optimized, alpha=1.02, beta=1)
            
            else:
                # 품질이 좋은 경우 최소한의 최적화
                self.logger.info("✨ 높은 품질로 인한 최소한의 최적화 적용")
                
                # 약간의 선명도 향상
                kernel = np.array([[-0.5,-0.5,-0.5], [-0.5,5,-0.5], [-0.5,-0.5,-0.5]])
                optimized = cv2.filter2D(optimized, -1, kernel)
            
            self.logger.info("✅ 최종 결과 최적화 완료")
            return optimized
            
        except Exception as e:
            self.logger.error(f"❌ 최종 결과 최적화 실패: {e}")
            return processed_image
    
    def _calculate_final_metrics(self, original: np.ndarray, final: np.ndarray) -> Dict[str, float]:
        """최종 품질 메트릭 계산"""
        try:
            self.logger.info("📊 최종 품질 메트릭 계산 시작")
            
            # 기본 메트릭 계산
            final_metrics = {}
            
            # PSNR
            mse = np.mean((original.astype(float) - final.astype(float)) ** 2)
            if mse == 0:
                final_metrics['psnr'] = float('inf')
            else:
                final_metrics['psnr'] = 20 * np.log10(255.0 / np.sqrt(mse))
            
            # SSIM (간단한 버전)
            final_metrics['ssim'] = self._calculate_ssim(original, final)
            
            # 대비 개선도
            original_contrast = np.std(original)
            final_contrast = np.std(final)
            final_metrics['contrast'] = final_contrast / original_contrast if original_contrast > 0 else 1.0
            
            # 선명도 개선도
            original_sharpness = self._calculate_sharpness(original)
            final_sharpness = self._calculate_sharpness(final)
            final_metrics['sharpness'] = final_sharpness / original_sharpness if original_sharpness > 0 else 1.0
            
            # 색상 균형 개선도
            original_color_variance = np.var(original, axis=(0, 1))
            final_color_variance = np.var(final, axis=(0, 1))
            final_metrics['color_balance'] = np.mean(final_color_variance / original_color_variance) if np.any(original_color_variance > 0) else 1.0
            
            self.logger.info("✅ 최종 품질 메트릭 계산 완료")
            return final_metrics
            
        except Exception as e:
            self.logger.error(f"❌ 최종 품질 메트릭 계산 실패: {e}")
            return {}
    
    def _calculate_ssim(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """SSIM 계산"""
        try:
            # 그레이스케일로 변환
            if len(img1.shape) == 3:
                img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            else:
                img1_gray = img1
            
            if len(img2.shape) == 3:
                img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            else:
                img2_gray = img2
            
            # SSIM 계산
            mu1 = np.mean(img1_gray)
            mu2 = np.mean(img2_gray)
            sigma1 = np.std(img1_gray)
            sigma2 = np.std(img2_gray)
            sigma12 = np.mean((img1_gray - mu1) * (img2_gray - mu2))
            
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
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            return np.var(laplacian)
            
        except Exception as e:
            self.logger.error(f"❌ 선명도 계산 실패: {e}")
            return 0.0
    
    def _generate_quality_report(self, 
                                original_metrics: Dict[str, float],
                                final_metrics: Dict[str, float],
                                validation_result: Dict[str, Any]) -> Dict[str, Any]:
        """품질 보고서 생성"""
        try:
            self.logger.info("📋 품질 보고서 생성 시작")
            
            report = {
                'timestamp': time.time(),
                'summary': {
                    'overall_quality_score': validation_result.get('overall_score', 0.0),
                    'is_acceptable': validation_result.get('is_acceptable', False),
                    'total_metrics': len(original_metrics),
                    'passed_metrics': len(validation_result.get('passed_metrics', [])),
                    'failed_metrics': len(validation_result.get('failed_metrics', []))
                },
                'metric_comparison': {},
                'validation_details': validation_result,
                'recommendations': validation_result.get('recommendations', []),
                'processing_info': {
                    'processor_version': '1.0.0',
                    'quality_thresholds': self.quality_thresholds.copy()
                }
            }
            
            # 메트릭 비교
            for metric_name in original_metrics.keys():
                if metric_name in final_metrics:
                    report['metric_comparison'][metric_name] = {
                        'original': original_metrics[metric_name],
                        'final': final_metrics[metric_name],
                        'improvement': final_metrics[metric_name] - original_metrics[metric_name] if metric_name != 'psnr' else final_metrics[metric_name] - original_metrics[metric_name]
                    }
            
            self.logger.info("✅ 품질 보고서 생성 완료")
            return report
            
        except Exception as e:
            self.logger.error(f"❌ 품질 보고서 생성 실패: {e}")
            return {
                'error': f'품질 보고서 생성 실패: {e}',
                'timestamp': time.time()
            }
    
    def _apply_output_config(self, image: np.ndarray, output_config: Optional[Dict[str, Any]]) -> np.ndarray:
        """출력 설정 적용"""
        try:
            if output_config is None:
                return image
            
            self.logger.info("⚙️ 출력 설정 적용 시작")
            
            output_image = image.copy()
            
            # 크기 조정
            if 'resize' in output_config:
                resize_config = output_config['resize']
                if 'width' in resize_config and 'height' in resize_config:
                    width = resize_config['width']
                    height = resize_config['height']
                    output_image = cv2.resize(output_image, (width, height))
                    self.logger.info(f"📏 크기 조정: {width}x{height}")
            
            # 품질 조정
            if 'quality' in output_config:
                quality = output_config['quality']
                if 0 <= quality <= 100:
                    # JPEG 품질 시뮬레이션 (간단한 버전)
                    if quality < 50:
                        # 낮은 품질: 약간의 블러
                        output_image = cv2.GaussianBlur(output_image, (3, 3), 0.5)
                    elif quality > 80:
                        # 높은 품질: 선명도 향상
                        kernel = np.array([[-0.5,-0.5,-0.5], [-0.5,5,-0.5], [-0.5,-0.5,-0.5]])
                        output_image = cv2.filter2D(output_image, -1, kernel)
                    
                    self.logger.info(f"🎯 품질 조정: {quality}%")
            
            # 색상 공간 변환
            if 'color_space' in output_config:
                color_space = output_config['color_space']
                if color_space == 'grayscale':
                    output_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2GRAY)
                    output_image = cv2.cvtColor(output_image, cv2.COLOR_GRAY2BGR)
                    self.logger.info("🎨 그레이스케일 변환")
                elif color_space == 'hsv':
                    output_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2HSV)
                    output_image = cv2.cvtColor(output_image, cv2.COLOR_HSV2BGR)
                    self.logger.info("🎨 HSV 색상 공간 변환")
            
            self.logger.info("✅ 출력 설정 적용 완료")
            return output_image
            
        except Exception as e:
            self.logger.error(f"❌ 출력 설정 적용 실패: {e}")
            return image
    
    def save_quality_report(self, quality_report: Dict[str, Any], output_path: str) -> bool:
        """품질 보고서 저장"""
        try:
            self.logger.info(f"💾 품질 보고서 저장 시작: {output_path}")
            
            # JSON 형식으로 저장
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(quality_report, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"✅ 품질 보고서 저장 완료: {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ 품질 보고서 저장 실패: {e}")
            return False
    
    def set_quality_thresholds(self, thresholds: Dict[str, float]):
        """품질 임계값 설정"""
        try:
            for key, value in thresholds.items():
                if key in self.quality_thresholds:
                    self.quality_thresholds[key] = value
                    self.logger.info(f"✅ {key} 임계값 설정: {value}")
                else:
                    self.logger.warning(f"⚠️ 알 수 없는 임계값: {key}")
                    
        except Exception as e:
            self.logger.error(f"❌ 품질 임계값 설정 실패: {e}")
    
    def get_quality_thresholds(self) -> Dict[str, float]:
        """품질 임계값 반환"""
        return self.quality_thresholds.copy()
    
    def _update_processing_stats(self, success: bool, processing_time: float):
        """처리 통계 업데이트"""
        try:
            self.processing_stats['total_processings'] += 1
            
            if success:
                self.processing_stats['successful_processings'] += 1
                
                # 평균 처리 시간 업데이트
                total_successful = self.processing_stats['successful_processings']
                current_avg = self.processing_stats['average_processing_time']
                new_avg = (current_avg * (total_successful - 1) + processing_time) / total_successful
                self.processing_stats['average_processing_time'] = new_avg
            else:
                self.processing_stats['failed_processings'] += 1
                
        except Exception as e:
            self.logger.error(f"❌ 처리 통계 업데이트 실패: {e}")
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """처리 통계 반환"""
        return self.processing_stats.copy()
    
    def reset_processing_stats(self):
        """처리 통계 초기화"""
        self.processing_stats = {
            'total_processings': 0,
            'successful_processings': 0,
            'failed_processings': 0,
            'average_processing_time': 0.0
        }
