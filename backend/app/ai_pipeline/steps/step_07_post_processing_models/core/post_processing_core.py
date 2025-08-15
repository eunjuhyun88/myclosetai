"""
Post Processing Core
후처리 파이프라인의 핵심 기능을 담당하는 클래스
"""

import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional, Union, Tuple
import numpy as np
from PIL import Image
import logging
import time
from dataclasses import dataclass
from pathlib import Path

# 프로젝트 로깅 설정 import
from backend.app.ai_pipeline.utils.logging_config import get_logger

logger = get_logger(__name__)

@dataclass
class ProcessingResult:
    """처리 결과를 저장하는 데이터 클래스"""
    input_image: torch.Tensor
    output_image: torch.Tensor
    processing_time: float
    quality_metrics: Dict[str, float]
    model_info: Dict[str, Any]
    error_message: Optional[str] = None

class PostProcessingCore:
    """
    후처리 파이프라인의 핵심 기능을 담당하는 클래스
    """
    
    def __init__(self, device: Optional[torch.device] = None):
        """
        Args:
            device: 사용할 디바이스
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 핵심 설정
        self.core_config = {
            'enable_quality_assessment': True,
            'enable_ensemble_processing': True,
            'enable_adaptive_processing': True,
            'max_processing_time': 300,  # 5분
            'quality_threshold': 0.7,
            'fallback_strategy': 'best_available'
        }
        
        # 처리 파이프라인
        self.processing_pipeline = []
        
        # 품질 메트릭 히스토리
        self.quality_history = []
        
        # 성능 통계
        self.performance_stats = {
            'total_processed': 0,
            'successful_processing': 0,
            'failed_processing': 0,
            'average_processing_time': 0.0,
            'total_processing_time': 0.0
        }
        
        logger.info(f"PostProcessingCore initialized on device: {self.device}")
    
    def add_processing_step(self, step_name: str, step_function: callable, 
                           step_config: Dict[str, Any]) -> bool:
        """
        처리 단계 추가
        
        Args:
            step_name: 단계 이름
            step_function: 처리 함수
            step_config: 단계 설정
            
        Returns:
            추가 성공 여부
        """
        try:
            step_info = {
                'name': step_name,
                'function': step_function,
                'config': step_config,
                'enabled': True
            }
            
            self.processing_pipeline.append(step_info)
            logger.info(f"처리 단계 추가 완료: {step_name}")
            return True
            
        except Exception as e:
            logger.error(f"처리 단계 추가 중 오류 발생: {e}")
            return False
    
    def remove_processing_step(self, step_name: str) -> bool:
        """
        처리 단계 제거
        
        Args:
            step_name: 제거할 단계 이름
            
        Returns:
            제거 성공 여부
        """
        try:
            for i, step in enumerate(self.processing_pipeline):
                if step['name'] == step_name:
                    del self.processing_pipeline[i]
                    logger.info(f"처리 단계 제거 완료: {step_name}")
                    return True
            
            logger.warning(f"처리 단계를 찾을 수 없습니다: {step_name}")
            return False
            
        except Exception as e:
            logger.error(f"처리 단계 제거 중 오류 발생: {e}")
            return False
    
    def enable_processing_step(self, step_name: str) -> bool:
        """처리 단계 활성화"""
        try:
            for step in self.processing_pipeline:
                if step['name'] == step_name:
                    step['enabled'] = True
                    logger.info(f"처리 단계 활성화 완료: {step_name}")
                    return True
            
            logger.warning(f"처리 단계를 찾을 수 없습니다: {step_name}")
            return False
            
        except Exception as e:
            logger.error(f"처리 단계 활성화 중 오류 발생: {e}")
            return False
    
    def disable_processing_step(self, step_name: str) -> bool:
        """처리 단계 비활성화"""
        try:
            for step in self.processing_pipeline:
                if step['name'] == step_name:
                    step['enabled'] = False
                    logger.info(f"처리 단계 비활성화 완료: {step_name}")
                    return True
            
            logger.warning(f"처리 단계를 찾을 수 없습니다: {step_name}")
            return False
            
        except Exception as e:
            logger.error(f"처리 단계 비활성화 중 오류 발생: {e}")
            return False
    
    def process_image(self, input_image: torch.Tensor, 
                     processing_config: Optional[Dict[str, Any]] = None) -> ProcessingResult:
        """
        이미지 처리 실행
        
        Args:
            input_image: 입력 이미지
            processing_config: 처리 설정
            
        Returns:
            처리 결과
        """
        start_time = time.time()
        
        try:
            logger.info("이미지 처리 시작")
            
            # 설정 병합
            if processing_config is None:
                processing_config = {}
            
            config = {**self.core_config, **processing_config}
            
            # 입력 이미지 검증
            if not self._validate_input_image(input_image):
                raise ValueError("입력 이미지가 유효하지 않습니다")
            
            # 이미지를 디바이스로 이동
            input_image = input_image.to(self.device)
            
            # 처리 파이프라인 실행
            current_image = input_image
            step_results = []
            
            for step in self.processing_pipeline:
                if not step['enabled']:
                    continue
                
                try:
                    step_start_time = time.time()
                    
                    # 단계별 처리 실행
                    step_result = step['function'](current_image, step['config'])
                    
                    if step_result is not None:
                        current_image = step_result
                        step_time = time.time() - step_start_time
                        
                        step_results.append({
                            'step_name': step['name'],
                            'processing_time': step_time,
                            'success': True
                        })
                        
                        logger.debug(f"단계 {step['name']} 처리 완료 (소요시간: {step_time:.3f}s)")
                    else:
                        step_results.append({
                            'step_name': step['name'],
                            'processing_time': 0.0,
                            'success': False
                        })
                        
                        logger.warning(f"단계 {step['name']} 처리 실패")
                        
                except Exception as e:
                    step_results.append({
                        'step_name': step['name'],
                        'processing_time': 0.0,
                        'success': False,
                        'error': str(e)
                    })
                    
                    logger.error(f"단계 {step['name']} 처리 중 오류 발생: {e}")
                    
                    # 오류 처리 전략에 따라 처리
                    if config['fallback_strategy'] == 'stop_on_error':
                        raise e
                    elif config['fallback_strategy'] == 'best_available':
                        logger.info("가용한 최선의 결과로 계속 진행")
                        break
            
            # 처리 시간 계산
            total_processing_time = time.time() - start_time
            
            # 품질 평가
            quality_metrics = {}
            if config['enable_quality_assessment']:
                quality_metrics = self._assess_quality(input_image, current_image)
                
                # 품질 히스토리에 추가
                self.quality_history.append({
                    'timestamp': time.time(),
                    'quality_metrics': quality_metrics,
                    'processing_time': total_processing_time
                })
            
            # 성능 통계 업데이트
            self._update_performance_stats(total_processing_time, True)
            
            # 결과 생성
            result = ProcessingResult(
                input_image=input_image,
                output_image=current_image,
                processing_time=total_processing_time,
                quality_metrics=quality_metrics,
                model_info={
                    'pipeline_steps': step_results,
                    'device': str(self.device),
                    'config': config
                }
            )
            
            logger.info(f"이미지 처리 완료 (소요시간: {total_processing_time:.3f}s)")
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            self._update_performance_stats(processing_time, False)
            
            logger.error(f"이미지 처리 중 오류 발생: {e}")
            
            # 오류 결과 반환
            return ProcessingResult(
                input_image=input_image,
                output_image=input_image,  # 원본 반환
                processing_time=processing_time,
                quality_metrics={},
                model_info={'error': str(e)},
                error_message=str(e)
            )
    
    def process_batch(self, input_images: List[torch.Tensor], 
                     processing_config: Optional[Dict[str, Any]] = None) -> List[ProcessingResult]:
        """
        배치 이미지 처리
        
        Args:
            input_images: 입력 이미지 리스트
            processing_config: 처리 설정
            
        Returns:
            처리 결과 리스트
        """
        try:
            logger.info(f"배치 처리 시작 - {len(input_images)}개 이미지")
            
            results = []
            for i, input_image in enumerate(input_images):
                logger.info(f"이미지 {i+1}/{len(input_images)} 처리 중...")
                
                result = self.process_image(input_image, processing_config)
                results.append(result)
            
            logger.info("배치 처리 완료")
            return results
            
        except Exception as e:
            logger.error(f"배치 처리 중 오류 발생: {e}")
            raise
    
    def _validate_input_image(self, image: torch.Tensor) -> bool:
        """입력 이미지 유효성 검증"""
        try:
            if not isinstance(image, torch.Tensor):
                return False
            
            if image.dim() != 3:
                return False
            
            if image.size(0) not in [1, 3, 4]:  # 그레이스케일, RGB, RGBA
                return False
            
            if image.min() < 0 or image.max() > 1:
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"이미지 유효성 검증 중 오류 발생: {e}")
            return False
    
    def _assess_quality(self, original_image: torch.Tensor, 
                        processed_image: torch.Tensor) -> Dict[str, float]:
        """이미지 품질 평가"""
        try:
            # 간단한 품질 메트릭 계산
            quality_metrics = {}
            
            # PSNR 계산
            mse = torch.mean((original_image - processed_image) ** 2)
            if mse > 0:
                psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
                quality_metrics['psnr'] = psnr.item()
            else:
                quality_metrics['psnr'] = float('inf')
            
            # SSIM 계산 (간단한 구현)
            quality_metrics['ssim'] = self._calculate_simple_ssim(original_image, processed_image)
            
            # 색상 일관성
            color_diff = torch.mean(torch.abs(original_image - processed_image))
            quality_metrics['color_consistency'] = max(0, 1.0 - color_diff.item())
            
            # 전체 품질 점수
            quality_metrics['overall_score'] = np.mean([
                min(1.0, quality_metrics['psnr'] / 50.0),
                quality_metrics['ssim'],
                quality_metrics['color_consistency']
            ])
            
            return quality_metrics
            
        except Exception as e:
            logger.error(f"품질 평가 중 오류 발생: {e}")
            return {'overall_score': 0.0}
    
    def _calculate_simple_ssim(self, img1: torch.Tensor, img2: torch.Tensor) -> float:
        """간단한 SSIM 계산"""
        try:
            # 단일 채널로 변환
            if img1.size(0) == 3:
                img1_gray = 0.299 * img1[0] + 0.587 * img1[1] + 0.114 * img1[2]
            else:
                img1_gray = img1[0]
            
            if img2.size(0) == 3:
                img2_gray = 0.299 * img2[0] + 0.587 * img2[1] + 0.114 * img2[2]
            else:
                img2_gray = img2[0]
            
            # 간단한 구조적 유사성 계산
            mu1 = torch.mean(img1_gray)
            mu2 = torch.mean(img2_gray)
            
            sigma1 = torch.std(img1_gray)
            sigma2 = torch.std(img2_gray)
            
            sigma12 = torch.mean((img1_gray - mu1) * (img2_gray - mu2))
            
            C1 = 0.01 ** 2
            C2 = 0.03 ** 2
            
            ssim = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1 ** 2 + sigma2 ** 2 + C2))
            
            return ssim.item()
            
        except Exception as e:
            logger.error(f"SSIM 계산 중 오류 발생: {e}")
            return 0.0
    
    def _update_performance_stats(self, processing_time: float, success: bool):
        """성능 통계 업데이트"""
        try:
            self.performance_stats['total_processed'] += 1
            self.performance_stats['total_processing_time'] += processing_time
            
            if success:
                self.performance_stats['successful_processing'] += 1
            else:
                self.performance_stats['failed_processing'] += 1
            
            # 평균 처리 시간 업데이트
            total_successful = self.performance_stats['successful_processing']
            if total_successful > 0:
                self.performance_stats['average_processing_time'] = \
                    self.performance_stats['total_processing_time'] / total_successful
                    
        except Exception as e:
            logger.error(f"성능 통계 업데이트 중 오류 발생: {e}")
    
    def get_processing_pipeline_info(self) -> List[Dict[str, Any]]:
        """처리 파이프라인 정보 반환"""
        try:
            pipeline_info = []
            for step in self.processing_pipeline:
                pipeline_info.append({
                    'name': step['name'],
                    'enabled': step['enabled'],
                    'config': step['config']
                })
            return pipeline_info
        except Exception as e:
            logger.error(f"파이프라인 정보 조회 중 오류 발생: {e}")
            return []
    
    def get_quality_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """품질 히스토리 반환"""
        try:
            if limit is None:
                return self.quality_history
            else:
                return self.quality_history[-limit:]
        except Exception as e:
            logger.error(f"품질 히스토리 조회 중 오류 발생: {e}")
            return []
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """성능 통계 반환"""
        try:
            stats = self.performance_stats.copy()
            
            # 추가 통계 계산
            if stats['total_processed'] > 0:
                stats['success_rate'] = stats['successful_processing'] / stats['total_processed']
                stats['failure_rate'] = stats['failed_processing'] / stats['total_processed']
            else:
                stats['success_rate'] = 0.0
                stats['failure_rate'] = 0.0
            
            return stats
        except Exception as e:
            logger.error(f"성능 통계 조회 중 오류 발생: {e}")
            return {}
    
    def reset_statistics(self):
        """통계 초기화"""
        try:
            self.quality_history.clear()
            self.performance_stats = {
                'total_processed': 0,
                'successful_processing': 0,
                'failed_processing': 0,
                'average_processing_time': 0.0,
                'total_processing_time': 0.0
            }
            logger.info("통계 초기화 완료")
        except Exception as e:
            logger.error(f"통계 초기화 중 오류 발생: {e}")
    
    def set_core_config(self, **kwargs):
        """핵심 설정 업데이트"""
        self.core_config.update(kwargs)
        logger.info("핵심 설정 업데이트 완료")
    
    def get_core_config(self) -> Dict[str, Any]:
        """핵심 설정 반환"""
        return self.core_config.copy()
