#!/usr/bin/env python3
"""
🔥 MyCloset AI - Virtual Fitting Quality Service
===============================================

🎯 가상 피팅 품질 관리 서비스
✅ 품질 향상 서비스
✅ 품질 검증 서비스
✅ 품질 모니터링 서비스
✅ M3 Max 최적화
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
import cv2
from PIL import Image
import time
import psutil
import gc

logger = logging.getLogger(__name__)

@dataclass
class QualityServiceConfig:
    """품질 서비스 설정"""
    enable_quality_enhancement: bool = True
    enable_quality_validation: bool = True
    enable_quality_monitoring: bool = True
    quality_threshold: float = 0.7
    monitoring_interval: float = 1.0
    use_mps: bool = True

class VirtualFittingQualityService:
    """가상 피팅 품질 관리 서비스"""
    
    def __init__(self, config: QualityServiceConfig = None):
        self.config = config or QualityServiceConfig()
        self.logger = logging.getLogger(__name__)
        
        # MPS 디바이스 확인
        self.device = torch.device("mps" if torch.backends.mps.is_available() and self.config.use_mps else "cpu")
        self.logger.info(f"🎯 Virtual Fitting 품질 관리 서비스 초기화 (디바이스: {self.device})")
        
        # 품질 향상기
        if self.config.enable_quality_enhancement:
            try:
                import sys
                import os
                sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'processors'))
                from virtual_fitting_quality_enhancer import VirtualFittingQualityEnhancer
                from virtual_fitting_validator import VirtualFittingValidator
                
                self.quality_enhancer = VirtualFittingQualityEnhancer()
                self.validator = VirtualFittingValidator()
                self.logger.info("품질 향상기 및 검증기 로드 완료")
            except ImportError as e:
                self.logger.error(f"품질 향상기 로드 실패: {e}")
                self.quality_enhancer = None
                self.validator = None
        
        # 품질 모니터링
        self.quality_history = []
        self.last_monitoring_time = time.time()
        
        self.logger.info("✅ Virtual Fitting 품질 관리 서비스 초기화 완료")
    
    def enhance_quality(self, virtual_fitting_image: torch.Tensor) -> Dict[str, Any]:
        """
        가상 피팅 이미지의 품질을 향상시킵니다.
        
        Args:
            virtual_fitting_image: 가상 피팅 이미지 (B, C, H, W)
            
        Returns:
            품질 향상 결과
        """
        if not self.config.enable_quality_enhancement or self.quality_enhancer is None:
            return {
                'status': 'disabled',
                'enhanced_image': virtual_fitting_image,
                'message': '품질 향상이 비활성화되어 있습니다.'
            }
        
        try:
            start_time = time.time()
            
            # 품질 향상 수행
            enhanced_result = self.quality_enhancer(virtual_fitting_image)
            
            processing_time = time.time() - start_time
            
            result = {
                'status': 'success',
                'enhanced_image': enhanced_result['enhanced_image'],
                'processing_time': processing_time,
                'enhancement_strength': enhanced_result['enhancement_strength'],
                'input_size': enhanced_result['input_size']
            }
            
            self.logger.info(f"품질 향상 완료 (처리 시간: {processing_time:.4f}초)")
            return result
            
        except Exception as e:
            self.logger.error(f"품질 향상 실패: {e}")
            return {
                'status': 'error',
                'enhanced_image': virtual_fitting_image,
                'error': str(e),
                'message': '품질 향상 중 오류가 발생했습니다.'
            }
    
    def validate_quality(self, virtual_fitting_image: torch.Tensor,
                        original_image: Optional[torch.Tensor] = None,
                        target_image: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """
        가상 피팅 이미지의 품질을 검증합니다.
        
        Args:
            virtual_fitting_image: 가상 피팅 이미지 (B, C, H, W)
            original_image: 원본 이미지 (B, C, H, W)
            target_image: 목표 이미지 (B, C, H, W)
            
        Returns:
            품질 검증 결과
        """
        if not self.config.enable_quality_validation or self.validator is None:
            return {
                'status': 'disabled',
                'message': '품질 검증이 비활성화되어 있습니다.'
            }
        
        try:
            start_time = time.time()
            
            # 품질 검증 수행
            validation_result = self.validator(virtual_fitting_image, original_image, target_image)
            
            # 품질 메트릭 계산
            if original_image is not None:
                quality_metrics = self.validator.calculate_quality_metrics(original_image, virtual_fitting_image)
            else:
                quality_metrics = {'status': 'no_original_image'}
            
            processing_time = time.time() - start_time
            
            result = {
                'status': 'success',
                'validation_result': validation_result,
                'quality_metrics': quality_metrics,
                'processing_time': processing_time,
                'quality_passed': validation_result['validation_results'].get('quality_passed', False)
            }
            
            # 품질 히스토리에 추가
            self.quality_history.append({
                'timestamp': time.time(),
                'quality_score': validation_result['validation_results'].get('overall_score', 0.0),
                'quality_passed': result['quality_passed']
            })
            
            self.logger.info(f"품질 검증 완료 (처리 시간: {processing_time:.4f}초, 통과: {result['quality_passed']})")
            return result
            
        except Exception as e:
            self.logger.error(f"품질 검증 실패: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'message': '품질 검증 중 오류가 발생했습니다.'
            }
    
    def monitor_quality(self) -> Dict[str, Any]:
        """품질 모니터링을 수행합니다."""
        if not self.config.enable_quality_monitoring:
            return {
                'status': 'disabled',
                'message': '품질 모니터링이 비활성화되어 있습니다.'
            }
        
        current_time = time.time()
        
        # 모니터링 간격 확인
        if current_time - self.last_monitoring_time < self.config.monitoring_interval:
            return {
                'status': 'skipped',
                'message': f'모니터링 간격이 충족되지 않았습니다. (필요: {self.config.monitoring_interval}초)'
            }
        
        try:
            # 시스템 리소스 모니터링
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            
            # PyTorch 메모리 모니터링
            if torch.cuda.is_available():
                torch_memory = torch.cuda.memory_stats()
                gpu_memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
                gpu_memory_reserved = torch.cuda.memory_reserved() / 1024**3  # GB
            else:
                torch_memory = {}
                gpu_memory_allocated = 0.0
                gpu_memory_reserved = 0.0
            
            # 품질 통계 계산
            if self.quality_history:
                recent_quality = self.quality_history[-10:]  # 최근 10개
                avg_quality = sum(item['quality_score'] for item in recent_quality) / len(recent_quality)
                pass_rate = sum(1 for item in recent_quality if item['quality_passed']) / len(recent_quality)
            else:
                avg_quality = 0.0
                pass_rate = 0.0
            
            monitoring_result = {
                'status': 'success',
                'timestamp': current_time,
                'system_metrics': {
                    'cpu_percent': cpu_percent,
                    'memory_percent': memory.percent,
                    'memory_available_gb': memory.available / 1024**3
                },
                'torch_memory': {
                    'gpu_memory_allocated_gb': gpu_memory_allocated,
                    'gpu_memory_reserved_gb': gpu_memory_reserved
                },
                'quality_metrics': {
                    'average_quality': avg_quality,
                    'pass_rate': pass_rate,
                    'total_validations': len(self.quality_history)
                }
            }
            
            self.last_monitoring_time = current_time
            self.logger.info(f"품질 모니터링 완료 (CPU: {cpu_percent:.1f}%, 메모리: {memory.percent:.1f}%)")
            
            return monitoring_result
            
        except Exception as e:
            self.logger.error(f"품질 모니터링 실패: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'message': '품질 모니터링 중 오류가 발생했습니다.'
            }
    
    def get_quality_report(self) -> Dict[str, Any]:
        """품질 보고서를 생성합니다."""
        try:
            # 최근 품질 통계
            if self.quality_history:
                recent_quality = self.quality_history[-50:]  # 최근 50개
                avg_quality = sum(item['quality_score'] for item in recent_quality) / len(recent_quality)
                pass_rate = sum(1 for item in recent_quality if item['quality_passed']) / len(recent_quality)
                
                # 품질 트렌드 분석
                if len(recent_quality) >= 2:
                    first_half = recent_quality[:len(recent_quality)//2]
                    second_half = recent_quality[len(recent_quality)//2:]
                    
                    first_avg = sum(item['quality_score'] for item in first_half) / len(first_half)
                    second_avg = sum(item['quality_score'] for item in second_half) / len(second_half)
                    
                    trend = "향상" if second_avg > first_avg else "하락" if second_avg < first_avg else "유지"
                else:
                    trend = "분석 불가"
            else:
                avg_quality = 0.0
                pass_rate = 0.0
                trend = "데이터 없음"
            
            report = {
                'status': 'success',
                'timestamp': time.time(),
                'quality_summary': {
                    'average_quality': avg_quality,
                    'pass_rate': pass_rate,
                    'trend': trend,
                    'total_validations': len(self.quality_history)
                },
                'service_status': {
                    'quality_enhancement_enabled': self.config.enable_quality_enhancement,
                    'quality_validation_enabled': self.config.enable_quality_validation,
                    'quality_monitoring_enabled': self.config.enable_quality_monitoring
                },
                'device_info': str(self.device)
            }
            
            return report
            
        except Exception as e:
            self.logger.error(f"품질 보고서 생성 실패: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'message': '품질 보고서 생성 중 오류가 발생했습니다.'
            }
    
    def cleanup(self):
        """리소스 정리를 수행합니다."""
        try:
            # 품질 히스토리 정리 (최근 1000개만 유지)
            if len(self.quality_history) > 1000:
                self.quality_history = self.quality_history[-1000:]
            
            # 가비지 컬렉션
            gc.collect()
            
            self.logger.info("품질 관리 서비스 리소스 정리 완료")
            
        except Exception as e:
            self.logger.error(f"리소스 정리 실패: {e}")

# 사용 예시
if __name__ == "__main__":
    # 설정
    config = QualityServiceConfig(
        enable_quality_enhancement=True,
        enable_quality_validation=True,
        enable_quality_monitoring=True,
        quality_threshold=0.7,
        monitoring_interval=1.0,
        use_mps=True
    )
    
    # 품질 관리 서비스 초기화
    quality_service = VirtualFittingQualityService(config)
    
    # 테스트 입력
    batch_size = 2
    channels = 3
    height = 256
    width = 256
    
    test_virtual_fitting = torch.randn(batch_size, channels, height, width)
    test_original = torch.randn(batch_size, channels, height, width)
    
    # 품질 향상
    enhancement_result = quality_service.enhance_quality(test_virtual_fitting)
    print(f"품질 향상 결과: {enhancement_result['status']}")
    
    # 품질 검증
    validation_result = quality_service.validate_quality(test_virtual_fitting, test_original)
    print(f"품질 검증 결과: {validation_result['status']}")
    
    # 품질 모니터링
    monitoring_result = quality_service.monitor_quality()
    print(f"품질 모니터링 결과: {monitoring_result['status']}")
    
    # 품질 보고서
    report = quality_service.get_quality_report()
    print(f"품질 보고서: {report['status']}")
    
    # 리소스 정리
    quality_service.cleanup()
