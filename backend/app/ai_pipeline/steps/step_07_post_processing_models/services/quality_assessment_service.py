"""
Quality Assessment Service
이미지 품질을 평가하고 분석하는 서비스 클래스
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Optional, Union, Tuple
import numpy as np
from PIL import Image
import cv2
import logging
import time
from dataclasses import dataclass

# 로깅 설정
import logging

logger = logging.getLogger(__name__)

@dataclass
class QualityMetrics:
    """품질 메트릭 결과를 저장하는 데이터 클래스"""
    psnr: float
    ssim: float
    perceptual_score: float
    sharpness: float
    noise_level: float
    overall_score: float
    processing_time: float

class QualityAssessmentService:
    """
    이미지 품질을 평가하고 분석하는 서비스 클래스
    """
    
    def __init__(self, device: Optional[torch.device] = None):
        """
        Args:
            device: 사용할 디바이스
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 품질 평가 설정
        self.quality_config = {
            'psnr_threshold': 30.0,
            'ssim_threshold': 0.8,
            'perceptual_threshold': 0.7,
            'sharpness_threshold': 0.6,
            'noise_threshold': 0.3
        }
        
        # 품질 메트릭 가중치
        self.metric_weights = {
            'psnr': 0.25,
            'ssim': 0.25,
            'perceptual': 0.2,
            'sharpness': 0.15,
            'noise': 0.15
        }
        
        logger.info(f"QualityAssessmentService initialized on device: {self.device}")
    
    def calculate_psnr(self, img1: torch.Tensor, img2: torch.Tensor) -> float:
        """
        PSNR (Peak Signal-to-Noise Ratio) 계산
        
        Args:
            img1: 첫 번째 이미지 텐서
            img2: 두 번째 이미지 텐서
            
        Returns:
            PSNR 값
        """
        try:
            # 이미지를 0-1 범위로 정규화
            img1 = img1.clamp(0, 1)
            img2 = img2.clamp(0, 1)
            
            # MSE 계산
            mse = F.mse_loss(img1, img2)
            
            if mse == 0:
                return float('inf')
            
            # PSNR 계산 (최대값을 1로 가정)
            psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
            
            return psnr.item()
        except Exception as e:
            logger.error(f"PSNR 계산 중 오류 발생: {e}")
            return 0.0
    
    def calculate_ssim(self, img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11) -> float:
        """
        SSIM (Structural Similarity Index) 계산
        
        Args:
            img1: 첫 번째 이미지 텐서
            img2: 두 번째 이미지 텐서
            window_size: SSIM 계산을 위한 윈도우 크기
            
        Returns:
            SSIM 값
        """
        try:
            # 이미지를 0-1 범위로 정규화
            img1 = img1.clamp(0, 1)
            img2 = img2.clamp(0, 1)
            
            # 단일 채널로 변환 (그레이스케일)
            if img1.dim() == 3 and img1.size(0) == 3:
                img1 = 0.299 * img1[0] + 0.587 * img1[1] + 0.114 * img1[2]
            if img2.dim() == 3 and img2.size(0) == 3:
                img2 = 0.299 * img2[0] + 0.587 * img2[1] + 0.114 * img2[2]
            
            # 배치 차원 추가
            img1 = img1.unsqueeze(0).unsqueeze(0)
            img2 = img2.unsqueeze(0).unsqueeze(0)
            
            # SSIM 계산
            ssim = self._ssim(img1, img2, window_size)
            
            return ssim.item()
        except Exception as e:
            logger.error(f"SSIM 계산 중 오류 발생: {e}")
            return 0.0
    
    def _ssim(self, img1: torch.Tensor, img2: torch.Tensor, window_size: int) -> torch.Tensor:
        """SSIM 계산의 내부 구현"""
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        # 가우시안 윈도우 생성
        window = self._create_window(window_size, img1.size(1)).to(img1.device)
        
        mu1 = F.conv2d(img1, window, padding=window_size//2, groups=img1.size(1))
        mu2 = F.conv2d(img2, window, padding=window_size//2, groups=img2.size(1))
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size//2, groups=img1.size(1)) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size//2, groups=img2.size(1)) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=window_size//2, groups=img1.size(1)) - mu1_mu2
        
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        return ssim_map.mean()
    
    def _create_window(self, window_size: int, channels: int) -> torch.Tensor:
        """가우시안 윈도우 생성"""
        def _gaussian(window_size, sigma):
            gauss = torch.Tensor([np.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
            return gauss/gauss.sum()
        
        _1D_window = _gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channels, 1, window_size, window_size).contiguous()
        return window
    
    def calculate_perceptual_score(self, img1: torch.Tensor, img2: torch.Tensor) -> float:
        """
        지각적 품질 점수 계산 (간단한 구현)
        
        Args:
            img1: 첫 번째 이미지 텐서
            img2: 두 번째 이미지 텐서
            
        Returns:
            지각적 품질 점수
        """
        try:
            # 이미지를 0-1 범위로 정규화
            img1 = img1.clamp(0, 1)
            img2 = img2.clamp(0, 1)
            
            # 색상 차이 계산
            color_diff = torch.mean(torch.abs(img1 - img2))
            
            # 지각적 점수 (차이가 적을수록 높은 점수)
            perceptual_score = 1.0 - color_diff.item()
            
            return max(0.0, perceptual_score)
        except Exception as e:
            logger.error(f"지각적 품질 점수 계산 중 오류 발생: {e}")
            return 0.0
    
    def calculate_sharpness(self, img: torch.Tensor) -> float:
        """
        이미지 선명도 계산
        
        Args:
            img: 이미지 텐서
            
        Returns:
            선명도 점수
        """
        try:
            # 이미지를 0-1 범위로 정규화
            img = img.clamp(0, 1)
            
            # 단일 채널로 변환
            if img.dim() == 3 and img.size(0) == 3:
                img = 0.299 * img[0] + 0.587 * img[1] + 0.114 * img[2]
            
            # 라플라시안 필터를 사용한 선명도 계산
            laplacian_kernel = torch.tensor([
                [0, -1, 0],
                [-1, 4, -1],
                [0, -1, 0]
            ], dtype=torch.float32, device=img.device).unsqueeze(0).unsqueeze(0)
            
            # 컨볼루션 적용
            sharpness_map = F.conv2d(img.unsqueeze(0).unsqueeze(0), laplacian_kernel, padding=1)
            
            # 선명도 점수 계산
            sharpness_score = torch.mean(torch.abs(sharpness_map)).item()
            
            # 0-1 범위로 정규화
            sharpness_score = min(1.0, sharpness_score / 2.0)
            
            return sharpness_score
        except Exception as e:
            logger.error(f"선명도 계산 중 오류 발생: {e}")
            return 0.0
    
    def calculate_noise_level(self, img: torch.Tensor) -> float:
        """
        이미지 노이즈 레벨 계산
        
        Args:
            img: 이미지 텐서
            
        Returns:
            노이즈 레벨 점수 (낮을수록 좋음)
        """
        try:
            # 이미지를 0-1 범위로 정규화
            img = img.clamp(0, 1)
            
            # 단일 채널로 변환
            if img.dim() == 3 and img.size(0) == 3:
                img = 0.299 * img[0] + 0.587 * img[1] + 0.114 * img[2]
            
            # 가우시안 블러를 사용한 노이즈 추정
            blurred = F.avg_pool2d(img.unsqueeze(0).unsqueeze(0), kernel_size=3, stride=1, padding=1)
            
            # 원본과 블러된 이미지의 차이로 노이즈 추정
            noise_map = torch.abs(img.unsqueeze(0).unsqueeze(0) - blurred)
            
            # 노이즈 레벨 계산
            noise_level = torch.mean(noise_map).item()
            
            # 0-1 범위로 정규화 (낮을수록 좋음)
            noise_score = 1.0 - min(1.0, noise_level * 10.0)
            
            return noise_score
        except Exception as e:
            logger.error(f"노이즈 레벨 계산 중 오류 발생: {e}")
            return 0.0
    
    def calculate_overall_quality(self, metrics: QualityMetrics) -> float:
        """
        전체 품질 점수 계산
        
        Args:
            metrics: 품질 메트릭
            
        Returns:
            전체 품질 점수
        """
        try:
            # 가중 평균으로 전체 점수 계산
            overall_score = (
                self.metric_weights['psnr'] * min(1.0, metrics.psnr / 50.0) +
                self.metric_weights['ssim'] * metrics.ssim +
                self.metric_weights['perceptual'] * metrics.perceptual_score +
                self.metric_weights['sharpness'] * metrics.sharpness +
                self.metric_weights['noise'] * metrics.noise_level
            )
            
            return overall_score
        except Exception as e:
            logger.error(f"전체 품질 점수 계산 중 오류 발생: {e}")
            return 0.0
    
    def assess_image_quality(self, original_img: torch.Tensor, processed_img: torch.Tensor) -> QualityMetrics:
        """
        이미지 품질 평가
        
        Args:
            original_img: 원본 이미지 텐서
            processed_img: 처리된 이미지 텐서
            
        Returns:
            품질 메트릭 결과
        """
        start_time = time.time()
        
        try:
            logger.info("이미지 품질 평가 시작")
            
            # PSNR 계산
            psnr = self.calculate_psnr(original_img, processed_img)
            
            # SSIM 계산
            ssim = self.calculate_ssim(original_img, processed_img)
            
            # 지각적 품질 점수 계산
            perceptual_score = self.calculate_perceptual_score(original_img, processed_img)
            
            # 선명도 계산 (처리된 이미지 기준)
            sharpness = self.calculate_sharpness(processed_img)
            
            # 노이즈 레벨 계산 (처리된 이미지 기준)
            noise_level = self.calculate_noise_level(processed_img)
            
            # 처리 시간 계산
            processing_time = time.time() - start_time
            
            # 품질 메트릭 생성
            metrics = QualityMetrics(
                psnr=psnr,
                ssim=ssim,
                perceptual_score=perceptual_score,
                sharpness=sharpness,
                noise_level=noise_level,
                overall_score=0.0,  # 나중에 계산
                processing_time=processing_time
            )
            
            # 전체 품질 점수 계산
            metrics.overall_score = self.calculate_overall_quality(metrics)
            
            logger.info(f"품질 평가 완료 - PSNR: {psnr:.2f}, SSIM: {ssim:.3f}, "
                       f"전체 점수: {metrics.overall_score:.3f}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"이미지 품질 평가 중 오류 발생: {e}")
            processing_time = time.time() - start_time
            
            # 오류 시 기본 메트릭 반환
            return QualityMetrics(
                psnr=0.0,
                ssim=0.0,
                perceptual_score=0.0,
                sharpness=0.0,
                noise_level=0.0,
                overall_score=0.0,
                processing_time=processing_time
            )
    
    def batch_quality_assessment(self, original_imgs: List[torch.Tensor], 
                                processed_imgs: List[torch.Tensor]) -> List[QualityMetrics]:
        """
        배치 이미지 품질 평가
        
        Args:
            original_imgs: 원본 이미지 텐서 리스트
            processed_imgs: 처리된 이미지 텐서 리스트
            
        Returns:
            품질 메트릭 결과 리스트
        """
        try:
            if len(original_imgs) != len(processed_imgs):
                raise ValueError("원본 이미지와 처리된 이미지의 개수가 일치하지 않습니다")
            
            logger.info(f"배치 품질 평가 시작 - {len(original_imgs)}개 이미지")
            
            results = []
            for i, (orig_img, proc_img) in enumerate(zip(original_imgs, processed_imgs)):
                logger.info(f"이미지 {i+1}/{len(original_imgs)} 품질 평가 중...")
                metrics = self.assess_image_quality(orig_img, proc_img)
                results.append(metrics)
            
            logger.info("배치 품질 평가 완료")
            return results
            
        except Exception as e:
            logger.error(f"배치 품질 평가 중 오류 발생: {e}")
            return []
    
    def get_quality_summary(self, metrics_list: List[QualityMetrics]) -> Dict[str, Any]:
        """
        품질 메트릭 요약 정보 반환
        
        Args:
            metrics_list: 품질 메트릭 리스트
            
        Returns:
            요약 정보 딕셔너리
        """
        try:
            if not metrics_list:
                return {}
            
            summary = {
                'total_images': len(metrics_list),
                'average_psnr': np.mean([m.psnr for m in metrics_list]),
                'average_ssim': np.mean([m.ssim for m in metrics_list]),
                'average_perceptual': np.mean([m.perceptual_score for m in metrics_list]),
                'average_sharpness': np.mean([m.sharpness for m in metrics_list]),
                'average_noise': np.mean([m.noise_level for m in metrics_list]),
                'average_overall': np.mean([m.overall_score for m in metrics_list]),
                'average_processing_time': np.mean([m.processing_time for m in metrics_list]),
                'best_image_index': np.argmax([m.overall_score for m in metrics_list]),
                'worst_image_index': np.argmin([m.overall_score for m in metrics_list])
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"품질 요약 생성 중 오류 발생: {e}")
            return {}
    
    def is_quality_acceptable(self, metrics: QualityMetrics) -> bool:
        """
        품질이 허용 가능한지 확인
        
        Args:
            metrics: 품질 메트릭
            
        Returns:
            품질이 허용 가능하면 True
        """
        try:
            return (
                metrics.psnr >= self.quality_config['psnr_threshold'] and
                metrics.ssim >= self.quality_config['ssim_threshold'] and
                metrics.perceptual_score >= self.quality_config['perceptual_threshold'] and
                metrics.sharpness >= self.quality_config['sharpness_threshold'] and
                metrics.noise_level >= self.quality_config['noise_threshold']
            )
        except Exception as e:
            logger.error(f"품질 허용 가능 여부 확인 중 오류 발생: {e}")
            return False

class PostProcessingQualityAssessmentService:
    """후처리 품질 평가 서비스"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # MPS 디바이스 확인
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.logger.info(f"🎯 Post Processing 품질 평가 서비스 초기화 (디바이스: {self.device})")
        
        # 기본 품질 평가 서비스 초기화
        self.quality_service = QualityAssessmentService(device=self.device)
        
        # 후처리 품질 평가 설정
        self.post_processing_config = {
            'enable_psnr': True,
            'enable_ssim': True,
            'enable_perceptual': True,
            'enable_sharpness': True,
            'enable_noise': True,
            'quality_threshold': 0.7,
            'auto_quality_check': True
        }
        
        # 설정 병합
        self.post_processing_config.update(self.config)
        
        # 품질 평가 통계
        self.quality_stats = {
            'total_assessments': 0,
            'passed_assessments': 0,
            'failed_assessments': 0,
            'average_quality_score': 0.0,
            'total_processing_time': 0.0
        }
        
        self.logger.info("✅ Post Processing 품질 평가 서비스 초기화 완료")
    
    def assess_post_processing_quality(self, original_image: torch.Tensor, 
                                     post_processed_image: torch.Tensor) -> Dict[str, Any]:
        """
        후처리 품질을 평가합니다.
        
        Args:
            original_image: 원본 이미지
            post_processed_image: 후처리된 이미지
            
        Returns:
            품질 평가 결과
        """
        try:
            start_time = time.time()
            
            # 입력을 디바이스로 이동
            original_image = original_image.to(self.device)
            post_processed_image = post_processed_image.to(self.device)
            
            # 품질 평가 수행
            quality_metrics = self.quality_service.assess_image_quality(
                original_image, post_processed_image
            )
            
            # 품질 통과 여부 확인
            is_acceptable = self.quality_service.is_quality_acceptable(quality_metrics)
            
            # 통계 업데이트
            self._update_quality_stats(quality_metrics, is_acceptable)
            
            # 결과 반환
            result = {
                'quality_metrics': quality_metrics,
                'is_acceptable': is_acceptable,
                'quality_score': quality_metrics.overall_score,
                'assessment_time': time.time() - start_time,
                'device': str(self.device)
            }
            
            self.logger.info(f"후처리 품질 평가 완료 (점수: {quality_metrics.overall_score:.4f}, 통과: {is_acceptable})")
            return result
            
        except Exception as e:
            self.logger.error(f"후처리 품질 평가 실패: {e}")
            return {
                'quality_metrics': None,
                'is_acceptable': False,
                'quality_score': 0.0,
                'assessment_time': 0.0,
                'error': str(e),
                'device': str(self.device)
            }
    
    def assess_batch_post_processing_quality(self, original_images: List[torch.Tensor],
                                           post_processed_images: List[torch.Tensor]) -> Dict[str, Any]:
        """
        배치 후처리 품질을 평가합니다.
        
        Args:
            original_images: 원본 이미지 리스트
            post_processed_images: 후처리된 이미지 리스트
            
        Returns:
            배치 품질 평가 결과
        """
        try:
            if len(original_images) != len(post_processed_images):
                raise ValueError("원본 이미지와 후처리 이미지의 개수가 일치하지 않습니다")
            
            start_time = time.time()
            
            # 개별 품질 평가
            individual_results = []
            for i, (orig_img, proc_img) in enumerate(zip(original_images, post_processed_images)):
                try:
                    result = self.assess_post_processing_quality(orig_img, proc_img)
                    individual_results.append(result)
                    self.logger.debug(f"이미지 {i+1} 품질 평가 완료")
                except Exception as e:
                    self.logger.error(f"이미지 {i+1} 품질 평가 실패: {e}")
                    individual_results.append({
                        'quality_metrics': None,
                        'is_acceptable': False,
                        'quality_score': 0.0,
                        'assessment_time': 0.0,
                        'error': str(e)
                    })
            
            # 배치 요약 생성
            batch_summary = self._create_batch_summary(individual_results)
            
            # 전체 처리 시간
            total_time = time.time() - start_time
            
            result = {
                'individual_results': individual_results,
                'batch_summary': batch_summary,
                'total_assessment_time': total_time,
                'batch_size': len(original_images),
                'device': str(self.device)
            }
            
            self.logger.info(f"배치 후처리 품질 평가 완료 (배치 크기: {len(original_images)}, 총 시간: {total_time:.4f}초)")
            return result
            
        except Exception as e:
            self.logger.error(f"배치 후처리 품질 평가 실패: {e}")
            return {
                'individual_results': [],
                'batch_summary': {},
                'total_assessment_time': 0.0,
                'batch_size': 0,
                'error': str(e),
                'device': str(self.device)
            }
    
    def _create_batch_summary(self, individual_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """배치 요약을 생성합니다."""
        try:
            if not individual_results:
                return {}
            
            # 통과한 평가 수
            passed_count = sum(1 for result in individual_results if result.get('is_acceptable', False))
            
            # 품질 점수들
            quality_scores = [result.get('quality_score', 0.0) for result in individual_results if result.get('quality_score') is not None]
            
            # 평가 시간들
            assessment_times = [result.get('assessment_time', 0.0) for result in individual_results if result.get('assessment_time') is not None]
            
            summary = {
                'total_images': len(individual_results),
                'passed_count': passed_count,
                'failed_count': len(individual_results) - passed_count,
                'pass_rate': passed_count / len(individual_results) if individual_results else 0.0,
                'average_quality_score': sum(quality_scores) / len(quality_scores) if quality_scores else 0.0,
                'min_quality_score': min(quality_scores) if quality_scores else 0.0,
                'max_quality_score': max(quality_scores) if quality_scores else 0.0,
                'average_assessment_time': sum(assessment_times) / len(assessment_times) if assessment_times else 0.0,
                'total_assessment_time': sum(assessment_times)
            }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"배치 요약 생성 실패: {e}")
            return {'error': str(e)}
    
    def _update_quality_stats(self, quality_metrics: QualityMetrics, is_acceptable: bool):
        """품질 통계를 업데이트합니다."""
        try:
            self.quality_stats['total_assessments'] += 1
            
            if is_acceptable:
                self.quality_stats['passed_assessments'] += 1
            else:
                self.quality_stats['failed_assessments'] += 1
            
            # 평균 품질 점수 업데이트
            total_score = self.quality_stats['average_quality_score'] * (self.quality_stats['total_assessments'] - 1)
            total_score += quality_metrics.overall_score
            self.quality_stats['average_quality_score'] = total_score / self.quality_stats['total_assessments']
            
            # 총 처리 시간 업데이트
            self.quality_stats['total_processing_time'] += quality_metrics.processing_time
            
        except Exception as e:
            self.logger.error(f"품질 통계 업데이트 실패: {e}")
    
    def get_quality_stats(self) -> Dict[str, Any]:
        """품질 평가 통계를 반환합니다."""
        return {
            **self.quality_stats,
            'service_config': self.post_processing_config,
            'device': str(self.device)
        }
    
    def reset_quality_stats(self):
        """품질 평가 통계를 초기화합니다."""
        self.quality_stats = {
            'total_assessments': 0,
            'passed_assessments': 0,
            'failed_assessments': 0,
            'average_quality_score': 0.0,
            'total_processing_time': 0.0
        }
        self.logger.info("품질 평가 통계 초기화 완료")
    
    def cleanup(self):
        """리소스 정리를 수행합니다."""
        try:
            # 품질 평가 통계 초기화
            self.reset_quality_stats()
            
            self.logger.info("Post Processing 품질 평가 서비스 리소스 정리 완료")
            
        except Exception as e:
            self.logger.error(f"리소스 정리 실패: {e}")

# 사용 예시
if __name__ == "__main__":
    # 설정
    config = {
        'enable_psnr': True,
        'enable_ssim': True,
        'enable_perceptual': True,
        'enable_sharpness': True,
        'enable_noise': True,
        'quality_threshold': 0.7,
        'auto_quality_check': True
    }
    
    # Post Processing 품질 평가 서비스 초기화
    quality_service = PostProcessingQualityAssessmentService(config)
    
    # 테스트 입력
    batch_size = 2
    channels = 3
    height = 256
    width = 256
    
    test_original = [torch.randn(channels, height, width) for _ in range(batch_size)]
    test_processed = [torch.randn(channels, height, width) for _ in range(batch_size)]
    
    # 개별 품질 평가
    for i in range(batch_size):
        result = quality_service.assess_post_processing_quality(test_original[i], test_processed[i])
        print(f"이미지 {i+1} 품질 평가: {result['quality_score']:.4f}")
    
    # 배치 품질 평가
    batch_result = quality_service.assess_batch_post_processing_quality(test_original, test_processed)
    print(f"배치 품질 평가 요약: {batch_result['batch_summary']}")
    
    # 품질 평가 통계
    stats = quality_service.get_quality_stats()
    print(f"품질 평가 통계: {stats}")
    
    # 리소스 정리
    quality_service.cleanup()
