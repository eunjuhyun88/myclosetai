"""
Batch Processor
배치 이미지 처리를 담당하는 클래스
"""

import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional, Union, Tuple
import numpy as np
from PIL import Image
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import gc

# 프로젝트 로깅 설정 import
import logging

logger = logging.getLogger(__name__)

@dataclass
class BatchResult:
    """배치 처리 결과를 저장하는 데이터 클래스"""
    input_images: List[torch.Tensor]
    output_images: List[torch.Tensor]
    processing_times: List[float]
    success_flags: List[bool]
    error_messages: List[Optional[str]]
    batch_size: int
    total_processing_time: float

class BatchProcessor:
    """
    배치 이미지 처리를 담당하는 클래스
    """
    
    def __init__(self, device: Optional[torch.device] = None):
        """
        Args:
            device: 사용할 디바이스
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 배치 처리 설정
        self.batch_config = {
            'max_batch_size': 8,
            'enable_parallel_processing': True,
            'max_workers': 4,
            'memory_limit_mb': 2048,  # 2GB
            'timeout_seconds': 300,  # 5분
            'enable_progress_tracking': True,
            'fallback_batch_size': 2
        }
        
        # 성능 통계
        self.performance_stats = {
            'total_batches': 0,
            'total_images': 0,
            'successful_batches': 0,
            'failed_batches': 0,
            'average_batch_time': 0.0,
            'total_processing_time': 0.0,
            'memory_usage_history': []
        }
        
        logger.info(f"BatchProcessor initialized on device: {self.device}")
    
    def process_batch(self, input_images: List[torch.Tensor], 
                     processing_function: callable,
                     processing_config: Optional[Dict[str, Any]] = None) -> BatchResult:
        """
        배치 이미지 처리 실행
        
        Args:
            input_images: 입력 이미지 리스트
            processing_function: 처리 함수
            processing_config: 처리 설정
            
        Returns:
            배치 처리 결과
        """
        start_time = time.time()
        
        try:
            logger.info(f"배치 처리 시작 - {len(input_images)}개 이미지")
            
            # 설정 병합
            if processing_config is None:
                processing_config = {}
            
            config = {**self.batch_config, **processing_config}
            
            # 배치 크기 결정
            optimal_batch_size = self._determine_optimal_batch_size(input_images, config)
            
            # 배치 분할
            batches = self._split_into_batches(input_images, optimal_batch_size)
            
            # 배치별 처리
            all_results = []
            batch_times = []
            success_flags = []
            error_messages = []
            
            for i, batch in enumerate(batches):
                logger.info(f"배치 {i+1}/{len(batches)} 처리 중... (크기: {len(batch)})")
                
                try:
                    batch_start_time = time.time()
                    
                    if config['enable_parallel_processing']:
                        batch_result = self._process_batch_parallel(batch, processing_function, config)
                    else:
                        batch_result = self._process_batch_sequential(batch, processing_function, config)
                    
                    batch_time = time.time() - batch_start_time
                    
                    all_results.extend(batch_result)
                    batch_times.extend([batch_time] * len(batch))
                    success_flags.extend([True] * len(batch))
                    error_messages.extend([None] * len(batch))
                    
                    logger.info(f"배치 {i+1} 처리 완료 (소요시간: {batch_time:.3f}s)")
                    
                except Exception as e:
                    batch_time = time.time() - batch_start_time
                    error_msg = f"배치 {i+1} 처리 실패: {str(e)}"
                    
                    # 오류 시 원본 이미지 반환
                    all_results.extend(batch)
                    batch_times.extend([batch_time] * len(batch))
                    success_flags.extend([False] * len(batch))
                    error_messages.extend([error_msg] * len(batch))
                    
                    logger.error(error_msg)
            
            # 전체 처리 시간 계산
            total_processing_time = time.time() - start_time
            
            # 결과 생성
            result = BatchResult(
                input_images=input_images,
                output_images=all_results,
                processing_times=batch_times,
                success_flags=success_flags,
                error_messages=error_messages,
                batch_size=optimal_batch_size,
                total_processing_time=total_processing_time
            )
            
            # 성능 통계 업데이트
            self._update_performance_stats(len(batches), len(input_images), 
                                         total_processing_time, True)
            
            logger.info(f"배치 처리 완료 (총 소요시간: {total_processing_time:.3f}s)")
            return result
            
        except Exception as e:
            total_processing_time = time.time() - start_time
            self._update_performance_stats(0, len(input_images), total_processing_time, False)
            
            logger.error(f"배치 처리 중 오류 발생: {e}")
            
            # 오류 시 원본 이미지 반환
            return BatchResult(
                input_images=input_images,
                output_images=input_images,
                processing_times=[total_processing_time] * len(input_images),
                success_flags=[False] * len(input_images),
                error_messages=[str(e)] * len(input_images),
                batch_size=1,
                total_processing_time=total_processing_time
            )
    
    def _determine_optimal_batch_size(self, images: List[torch.Tensor], 
                                     config: Dict[str, Any]) -> int:
        """최적 배치 크기 결정"""
        try:
            # 메모리 사용량 계산
            total_memory = sum(self._estimate_image_memory(img) for img in images)
            available_memory = config['memory_limit_mb'] * 1024 * 1024  # MB to bytes
            
            # 안전 마진 (80%)
            safe_memory = available_memory * 0.8
            
            if total_memory <= safe_memory:
                # 모든 이미지를 한 번에 처리 가능
                optimal_size = len(images)
            else:
                # 메모리에 맞는 배치 크기 계산
                avg_memory_per_image = total_memory / len(images)
                optimal_size = int(safe_memory / avg_memory_per_image)
            
            # 설정된 최대 배치 크기 제한
            optimal_size = min(optimal_size, config['max_batch_size'])
            
            # 최소 배치 크기 보장
            optimal_size = max(optimal_size, 1)
            
            logger.info(f"최적 배치 크기 결정: {optimal_size}")
            return optimal_size
            
        except Exception as e:
            logger.error(f"최적 배치 크기 결정 중 오류 발생: {e}")
            return config['fallback_batch_size']
    
    def _estimate_image_memory(self, image: torch.Tensor) -> int:
        """이미지 메모리 사용량 추정"""
        try:
            # 텐서 크기 계산
            element_size = image.element_size()  # bytes per element
            total_elements = image.numel()
            
            # 메모리 사용량 (bytes)
            memory_usage = element_size * total_elements
            
            return memory_usage
            
        except Exception as e:
            logger.error(f"이미지 메모리 사용량 추정 중 오류 발생: {e}")
            return 1024 * 1024  # 기본값: 1MB
    
    def _split_into_batches(self, images: List[torch.Tensor], 
                            batch_size: int) -> List[List[torch.Tensor]]:
        """이미지를 배치로 분할"""
        try:
            batches = []
            for i in range(0, len(images), batch_size):
                batch = images[i:i + batch_size]
                batches.append(batch)
            
            return batches
            
        except Exception as e:
            logger.error(f"배치 분할 중 오류 발생: {e}")
            return [images]  # 전체를 하나의 배치로
    
    def _process_batch_sequential(self, batch: List[torch.Tensor], 
                                 processing_function: callable,
                                 config: Dict[str, Any]) -> List[torch.Tensor]:
        """순차적 배치 처리"""
        try:
            results = []
            
            for i, image in enumerate(batch):
                if config['enable_progress_tracking']:
                    logger.debug(f"이미지 {i+1}/{len(batch)} 처리 중...")
                
                try:
                    # 이미지를 디바이스로 이동
                    device_image = image.to(self.device)
                    
                    # 처리 함수 실행
                    result = processing_function(device_image)
                    
                    # 결과를 CPU로 이동
                    if result.device != torch.device('cpu'):
                        result = result.cpu()
                    
                    results.append(result)
                    
                except Exception as e:
                    logger.error(f"이미지 {i+1} 처리 중 오류 발생: {e}")
                    results.append(image)  # 원본 반환
            
            return results
            
        except Exception as e:
            logger.error(f"순차적 배치 처리 중 오류 발생: {e}")
            return batch  # 원본 반환
    
    def _process_batch_parallel(self, batch: List[torch.Tensor], 
                               processing_function: callable,
                               config: Dict[str, Any]) -> List[torch.Tensor]:
        """병렬 배치 처리"""
        try:
            max_workers = min(config['max_workers'], len(batch))
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # 각 이미지에 대한 처리 작업 제출
                future_to_index = {
                    executor.submit(self._process_single_image, image, processing_function): i
                    for i, image in enumerate(batch)
                }
                
                # 결과 수집
                results = [None] * len(batch)
                
                for future in as_completed(future_to_index, timeout=config['timeout_seconds']):
                    index = future_to_index[future]
                    
                    try:
                        result = future.result()
                        results[index] = result
                    except Exception as e:
                        logger.error(f"이미지 {index+1} 병렬 처리 중 오류 발생: {e}")
                        results[index] = batch[index]  # 원본 반환
                
                # None 값 처리
                for i, result in enumerate(results):
                    if result is None:
                        results[i] = batch[i]  # 원본 반환
                
                return results
                
        except Exception as e:
            logger.error(f"병렬 배치 처리 중 오류 발생: {e}")
            return self._process_batch_sequential(batch, processing_function, config)
    
    def _process_single_image(self, image: torch.Tensor, 
                             processing_function: callable) -> torch.Tensor:
        """단일 이미지 처리"""
        try:
            # 이미지를 디바이스로 이동
            device_image = image.to(self.device)
            
            # 처리 함수 실행
            result = processing_function(device_image)
            
            # 결과를 CPU로 이동
            if result.device != torch.device('cpu'):
                result = result.cpu()
            
            return result
            
        except Exception as e:
            logger.error(f"단일 이미지 처리 중 오류 발생: {e}")
            return image  # 원본 반환
    
    def _update_performance_stats(self, num_batches: int, num_images: int, 
                                 processing_time: float, success: bool):
        """성능 통계 업데이트"""
        try:
            self.performance_stats['total_batches'] += num_batches
            self.performance_stats['total_images'] += num_images
            self.performance_stats['total_processing_time'] += processing_time
            
            if success:
                self.performance_stats['successful_batches'] += num_batches
            else:
                self.performance_stats['failed_batches'] += num_batches
            
            # 평균 배치 시간 업데이트
            total_successful = self.performance_stats['successful_batches']
            if total_successful > 0:
                self.performance_stats['average_batch_time'] = \
                    self.performance_stats['total_processing_time'] / total_successful
                    
        except Exception as e:
            logger.error(f"성능 통계 업데이트 중 오류 발생: {e}")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """성능 통계 반환"""
        try:
            stats = self.performance_stats.copy()
            
            # 추가 통계 계산
            if stats['total_batches'] > 0:
                stats['success_rate'] = stats['successful_batches'] / stats['total_batches']
                stats['failure_rate'] = stats['failed_batches'] / stats['total_batches']
            else:
                stats['success_rate'] = 0.0
                stats['failure_rate'] = 0.0
            
            if stats['total_images'] > 0:
                stats['average_images_per_batch'] = stats['total_images'] / stats['total_batches']
            else:
                stats['average_images_per_batch'] = 0.0
            
            return stats
            
        except Exception as e:
            logger.error(f"성능 통계 조회 중 오류 발생: {e}")
            return {}
    
    def reset_statistics(self):
        """통계 초기화"""
        try:
            self.performance_stats = {
                'total_batches': 0,
                'total_images': 0,
                'successful_batches': 0,
                'failed_batches': 0,
                'average_batch_time': 0.0,
                'total_processing_time': 0.0,
                'memory_usage_history': []
            }
            logger.info("배치 처리 통계 초기화 완료")
        except Exception as e:
            logger.error(f"통계 초기화 중 오류 발생: {e}")
    
    def set_batch_config(self, **kwargs):
        """배치 설정 업데이트"""
        self.batch_config.update(kwargs)
        logger.info("배치 설정 업데이트 완료")
    
    def get_batch_config(self) -> Dict[str, Any]:
        """배치 설정 반환"""
        return self.batch_config.copy()
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """현재 메모리 사용량 반환"""
        try:
            if torch.cuda.is_available():
                memory_info = {
                    'device': 'CUDA',
                    'allocated_mb': torch.cuda.memory_allocated() / (1024 * 1024),
                    'cached_mb': torch.cuda.memory_reserved() / (1024 * 1024),
                    'total_mb': torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
                }
            else:
                memory_info = {
                    'device': 'CPU',
                    'allocated_mb': 0.0,
                    'cached_mb': 0.0,
                    'total_mb': 0.0
                }
            
            return memory_info
            
        except Exception as e:
            logger.error(f"메모리 사용량 조회 중 오류 발생: {e}")
            return {'error': str(e)}

class PostProcessingBatchProcessor(nn.Module):
    """후처리 배치 프로세서"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__()
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # MPS 디바이스 확인
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.logger.info(f"🎯 Post Processing 배치 프로세서 초기화 (디바이스: {self.device})")
        
        # 배치 처리 설정
        self.batch_config = {
            'max_batch_size': 8,
            'enable_parallel_processing': True,
            'max_workers': 4,
            'memory_limit_mb': 2048,
            'timeout_seconds': 300,
            'enable_progress_tracking': True,
            'fallback_batch_size': 2
        }
        
        # 설정 병합
        self.batch_config.update(self.config)
        
        # 배치 처리기 초기화
        self.batch_processor = BatchProcessor(device=self.device)
        self.batch_processor.set_batch_config(**self.batch_config)
        
        # 후처리 네트워크
        self.post_processing_net = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 3, 3, padding=1),
            nn.Tanh()
        ).to(self.device)
        
        self.logger.info("✅ Post Processing 배치 프로세서 초기화 완료")
    
    def forward(self, batch_images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        배치 이미지 후처리를 수행합니다.
        
        Args:
            batch_images: 후처리할 이미지 배치 (B, C, H, W)
            
        Returns:
            후처리된 배치 결과
        """
        batch_size, channels, height, width = batch_images.shape
        
        # 입력 검증
        if channels != 3:
            raise ValueError(f"Expected 3 channels, got {channels}")
        
        # 입력을 디바이스로 이동
        batch_images = batch_images.to(self.device)
        
        # 후처리 네트워크 적용
        processed_batch = self.post_processing_net(batch_images)
        
        # 결과 반환
        result = {
            'processed_batch': processed_batch,
            'batch_size': batch_size,
            'input_size': (height, width),
            'device': str(self.device)
        }
        
        return result
    
    def process_batch(self, batch_images: List[torch.Tensor]) -> Dict[str, Any]:
        """
        배치 단위로 후처리를 수행합니다.
        
        Args:
            batch_images: 후처리할 이미지 배치 리스트
            
        Returns:
            배치 처리 결과
        """
        try:
            # 배치 처리 수행
            result = self.batch_processor.process_batch(
                batch_images, 
                self._process_single_image,
                self.batch_config
            )
            
            return {
                'status': 'success',
                'batch_result': result,
                'config': self.batch_config
            }
            
        except Exception as e:
            self.logger.error(f"배치 처리 실패: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'message': '배치 처리 중 오류가 발생했습니다.'
            }
    
    def _process_single_image(self, image: torch.Tensor) -> torch.Tensor:
        """단일 이미지를 후처리합니다."""
        try:
            # 단일 이미지 후처리
            processed = self.forward(image.unsqueeze(0))
            return processed['processed_batch'].squeeze(0)
            
        except Exception as e:
            self.logger.error(f"단일 이미지 후처리 실패: {e}")
            return image
    
    def get_batch_stats(self) -> Dict[str, Any]:
        """배치 처리 통계를 반환합니다."""
        try:
            batch_stats = self.batch_processor.get_performance_stats()
            memory_usage = self.batch_processor.get_memory_usage()
            
            return {
                **batch_stats,
                'memory_usage': memory_usage,
                'device': str(self.device),
                'config': self.batch_config
            }
            
        except Exception as e:
            self.logger.error(f"배치 통계 조회 실패: {e}")
            return {'error': str(e)}
    
    def cleanup(self):
        """리소스 정리를 수행합니다."""
        try:
            # 배치 처리기 정리
            self.batch_processor.reset_statistics()
            
            # 가비지 컬렉션
            gc.collect()
            
            self.logger.info("Post Processing 배치 프로세서 리소스 정리 완료")
            
        except Exception as e:
            self.logger.error(f"리소스 정리 실패: {e}")

# 사용 예시
if __name__ == "__main__":
    # 설정
    config = {
        'max_batch_size': 4,
        'enable_parallel_processing': True,
        'max_workers': 2,
        'memory_limit_mb': 1024,
        'timeout_seconds': 60,
        'enable_progress_tracking': True,
        'fallback_batch_size': 1
    }
    
    # Post Processing 배치 프로세서 초기화
    batch_processor = PostProcessingBatchProcessor(config)
    
    # 테스트 입력
    batch_size = 4
    channels = 3
    height = 256
    width = 256
    
    test_batch = torch.randn(batch_size, channels, height, width)
    
    # 배치 후처리 수행
    with torch.no_grad():
        result = batch_processor(test_batch)
        print(f"배치 후처리 결과: {result['processed_batch'].shape}")
    
    # 배치 처리 통계
    stats = batch_processor.get_batch_stats()
    print(f"배치 처리 통계: {stats}")
    
    # 리소스 정리
    batch_processor.cleanup()
