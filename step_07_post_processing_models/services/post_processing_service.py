"""
Post Processing Service

후처리 모델들을 관리하고 실행하는 서비스 클래스입니다.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional, Union, Tuple
import numpy as np
from PIL import Image
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# 프로젝트 로깅 설정 import
from backend.app.ai_pipeline.utils.logging_config import get_logger

logger = get_logger(__name__)


class PostProcessingService:
    """
    후처리 모델들을 관리하고 실행하는 서비스 클래스
    
    모델 로딩, 추론, 결과 관리 등의 기능을 제공합니다.
    """
    
    def __init__(self, model_loader, inference_engine, device: Optional[torch.device] = None):
        """
        Args:
            model_loader: 모델 로더 인스턴스
            inference_engine: 추론 엔진 인스턴스
            device: 사용할 디바이스
        """
        self.model_loader = model_loader
        self.inference_engine = inference_engine
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 서비스 설정
        self.service_config = {
            'max_concurrent_models': 2,
            'enable_caching': True,
            'cache_size': 100,
            'timeout': 300,  # 5분
            'retry_count': 3
        }
        
        # 모델 캐시
        self.model_cache = {}
        self.result_cache = {}
        
        # 성능 메트릭
        self.performance_metrics = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'average_processing_time': 0.0,
            'total_processing_time': 0.0
        }
        
        # 스레드 풀
        self.executor = ThreadPoolExecutor(max_workers=self.service_config['max_concurrent_models'])
        
        logger.info(f"PostProcessingService initialized on device: {self.device}")
    
    def process_image(self, image: Union[np.ndarray, Image.Image, torch.Tensor], 
                     model_type: str, 
                     task_type: str = 'general',
                     **kwargs) -> Dict[str, Any]:
        """
        이미지를 후처리 모델로 처리합니다.
        
        Args:
            image: 입력 이미지
            model_type: 사용할 모델 타입
            task_type: 작업 타입
            **kwargs: 추가 파라미터
            
        Returns:
            처리 결과 딕셔너리
        """
        start_time = time.time()
        request_id = f"req_{int(start_time * 1000)}"
        
        try:
            logger.info(f"Processing image with {model_type} for {task_type} task")
            
            # 캐시 확인
            cache_key = self._generate_cache_key(image, model_type, task_type, kwargs)
            if self.service_config['enable_caching'] and cache_key in self.result_cache:
                logger.info("Returning cached result")
                return self.result_cache[cache_key]
            
            # 모델 로드
            model = self._load_model_safely(model_type)
            
            # 추론 실행
            result = self._execute_inference(model, image, task_type, **kwargs)
            
            # 결과 후처리
            processed_result = self._postprocess_result(result, model_type, task_type)
            
            # 성능 메트릭 업데이트
            processing_time = time.time() - start_time
            self._update_performance_metrics(True, processing_time)
            
            # 결과 캐싱
            if self.service_config['enable_caching']:
                self._cache_result(cache_key, processed_result)
            
            # 응답 생성
            response = {
                'request_id': request_id,
                'status': 'success',
                'model_type': model_type,
                'task_type': task_type,
                'result': processed_result,
                'processing_time': processing_time,
                'timestamp': start_time
            }
            
            logger.info(f"Successfully processed image with {model_type} in {processing_time:.2f}s")
            return response
            
        except Exception as e:
            processing_time = time.time() - start_time
            self._update_performance_metrics(False, processing_time)
            
            error_response = {
                'request_id': request_id,
                'status': 'error',
                'model_type': model_type,
                'task_type': task_type,
                'error': str(e),
                'processing_time': processing_time,
                'timestamp': start_time
            }
            
            logger.error(f"Failed to process image with {model_type}: {str(e)}")
            return error_response
    
    def batch_process(self, images: List[Union[np.ndarray, Image.Image, torch.Tensor]], 
                     model_type: str,
                     task_type: str = 'general',
                     **kwargs) -> List[Dict[str, Any]]:
        """
        여러 이미지를 배치로 처리합니다.
        
        Args:
            images: 입력 이미지 리스트
            model_type: 사용할 모델 타입
            task_type: 작업 타입
            **kwargs: 추가 파라미터
            
        Returns:
            처리 결과 리스트
        """
        logger.info(f"Batch processing {len(images)} images with {model_type}")
        
        # 병렬 처리
        futures = []
        for i, image in enumerate(images):
            future = self.executor.submit(
                self.process_image, 
                image, 
                model_type, 
                task_type, 
                **kwargs
            )
            futures.append((i, future))
        
        # 결과 수집
        results = [None] * len(images)
        for i, future in futures:
            try:
                result = future.result(timeout=self.service_config['timeout'])
                results[i] = result
            except Exception as e:
                logger.error(f"Error processing image {i}: {str(e)}")
                results[i] = {
                    'status': 'error',
                    'error': str(e),
                    'image_index': i
                }
        
        return results
    
    def process_with_ensemble(self, image: Union[np.ndarray, Image.Image, torch.Tensor], 
                            model_types: List[str],
                            ensemble_method: str = 'weighted_average',
                            task_type: str = 'general',
                            **kwargs) -> Dict[str, Any]:
        """
        여러 모델을 앙상블로 처리합니다.
        
        Args:
            image: 입력 이미지
            model_types: 사용할 모델 타입들
            ensemble_method: 앙상블 방법
            task_type: 작업 타입
            **kwargs: 추가 파라미터
            
        Returns:
            앙상블 처리 결과
        """
        logger.info(f"Processing image with ensemble: {model_types}")
        
        # 각 모델로 처리
        model_results = {}
        for model_type in model_types:
            try:
                result = self.process_image(image, model_type, task_type, **kwargs)
                if result['status'] == 'success':
                    model_results[model_type] = result['result']
            except Exception as e:
                logger.warning(f"Failed to process with {model_type}: {str(e)}")
                continue
        
        if not model_results:
            raise RuntimeError("No models successfully processed the image")
        
        # 앙상블 적용
        ensemble_result = self._apply_ensemble(model_results, ensemble_method, **kwargs)
        
        return {
            'status': 'success',
            'ensemble_method': ensemble_method,
            'model_results': model_results,
            'ensemble_result': ensemble_result,
            'timestamp': time.time()
        }
    
    def _load_model_safely(self, model_type: str) -> nn.Module:
        """모델을 안전하게 로드합니다."""
        try:
            # 캐시 확인
            if model_type in self.model_cache:
                logger.info(f"Using cached model: {model_type}")
                return self.model_cache[model_type]
            
            # 모델 로드
            model = self.model_loader.load_model(model_type)
            
            # 캐시에 저장
            if len(self.model_cache) < self.service_config['max_concurrent_models']:
                self.model_cache[model_type] = model
                logger.info(f"Cached model: {model_type}")
            
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model {model_type}: {str(e)}")
            raise RuntimeError(f"Model loading failed: {str(e)}")
    
    def _execute_inference(self, model: nn.Module, 
                          image: Union[np.ndarray, Image.Image, torch.Tensor], 
                          task_type: str,
                          **kwargs) -> torch.Tensor:
        """추론을 실행합니다."""
        try:
            with torch.no_grad():
                result = model(image)
            return result
        except Exception as e:
            logger.error(f"Inference execution failed: {str(e)}")
            raise RuntimeError(f"Inference failed: {str(e)}")
    
    def _postprocess_result(self, result: torch.Tensor, 
                           model_type: str, 
                           task_type: str) -> np.ndarray:
        """결과를 후처리합니다."""
        try:
            # 텐서를 numpy 배열로 변환
            if isinstance(result, torch.Tensor):
                result = result.detach().cpu().numpy()
            
            # 모델별 특화 후처리
            if model_type == 'swinir':
                result = self._postprocess_swinir(result)
            elif model_type == 'realesrgan':
                result = self._postprocess_realesrgan(result)
            elif model_type == 'gfpgan':
                result = self._postprocess_gfpgan(result)
            elif model_type == 'codeformer':
                result = self._postprocess_codeformer(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Result postprocessing failed: {str(e)}")
            raise RuntimeError(f"Postprocessing failed: {str(e)}")
    
    def _postprocess_swinir(self, result: np.ndarray) -> np.ndarray:
        """SwinIR 결과 후처리"""
        # 클램핑 및 정규화
        result = np.clip(result, 0, 1)
        return result
    
    def _postprocess_realesrgan(self, result: np.ndarray) -> np.ndarray:
        """Real-ESRGAN 결과 후처리"""
        # 클램핑 및 정규화
        result = np.clip(result, 0, 1)
        return result
    
    def _postprocess_gfpgan(self, result: np.ndarray) -> np.ndarray:
        """GFPGAN 결과 후처리"""
        # 클램핑 및 정규화
        result = np.clip(result, 0, 1)
        return result
    
    def _postprocess_codeformer(self, result: np.ndarray) -> np.ndarray:
        """CodeFormer 결과 후처리"""
        # 클램핑 및 정규화
        result = np.clip(result, 0, 1)
        return result
    
    def _apply_ensemble(self, model_results: Dict[str, np.ndarray], 
                       ensemble_method: str,
                       **kwargs) -> np.ndarray:
        """앙상블을 적용합니다."""
        if ensemble_method == 'weighted_average':
            return self._weighted_average_ensemble(model_results, **kwargs)
        elif ensemble_method == 'simple_average':
            return self._simple_average_ensemble(model_results)
        elif ensemble_method == 'max':
            return self._max_ensemble(model_results)
        else:
            return self._simple_average_ensemble(model_results)
    
    def _weighted_average_ensemble(self, model_results: Dict[str, np.ndarray], 
                                 **kwargs) -> np.ndarray:
        """가중 평균 앙상블"""
        weights = kwargs.get('weights', {})
        if not weights:
            # 기본 가중치
            weights = {model_type: 1.0 / len(model_results) for model_type in model_results.keys()}
        
        # 가중 평균 계산
        ensemble_result = None
        total_weight = 0
        
        for model_type, result in model_results.items():
            weight = weights.get(model_type, 1.0)
            if ensemble_result is None:
                ensemble_result = result * weight
            else:
                ensemble_result += result * weight
            total_weight += weight
        
        if total_weight > 0:
            ensemble_result /= total_weight
        
        return ensemble_result
    
    def _simple_average_ensemble(self, model_results: Dict[str, np.ndarray]) -> np.ndarray:
        """단순 평균 앙상블"""
        results = list(model_results.values())
        return np.mean(results, axis=0)
    
    def _max_ensemble(self, model_results: Dict[str, np.ndarray]) -> np.ndarray:
        """최대값 앙상블"""
        results = list(model_results.values())
        return np.maximum.reduce(results)
    
    def _generate_cache_key(self, image: Union[np.ndarray, Image.Image, torch.Tensor], 
                           model_type: str, 
                           task_type: str, 
                           kwargs: Dict[str, Any]) -> str:
        """캐시 키를 생성합니다."""
        # 간단한 해시 기반 키 생성
        import hashlib
        
        # 이미지 해시
        if isinstance(image, np.ndarray):
            image_hash = hashlib.md5(image.tobytes()).hexdigest()[:8]
        elif isinstance(image, Image.Image):
            image_hash = hashlib.md5(np.array(image).tobytes()).hexdigest()[:8]
        else:
            image_hash = hashlib.md5(image.cpu().numpy().tobytes()).hexdigest()[:8]
        
        # 파라미터 해시
        param_str = f"{model_type}_{task_type}_{sorted(kwargs.items())}"
        param_hash = hashlib.md5(param_str.encode()).hexdigest()[:8]
        
        return f"{image_hash}_{param_hash}"
    
    def _cache_result(self, cache_key: str, result: np.ndarray):
        """결과를 캐시에 저장합니다."""
        if len(self.result_cache) >= self.service_config['cache_size']:
            # 가장 오래된 항목 제거
            oldest_key = next(iter(self.result_cache))
            del self.result_cache[oldest_key]
        
        self.result_cache[cache_key] = result
    
    def _update_performance_metrics(self, success: bool, processing_time: float):
        """성능 메트릭을 업데이트합니다."""
        self.performance_metrics['total_requests'] += 1
        self.performance_metrics['total_processing_time'] += processing_time
        
        if success:
            self.performance_metrics['successful_requests'] += 1
        else:
            self.performance_metrics['failed_requests'] += 1
        
        # 평균 처리 시간 업데이트
        if self.performance_metrics['successful_requests'] > 0:
            self.performance_metrics['average_processing_time'] = (
                self.performance_metrics['total_processing_time'] / 
                self.performance_metrics['successful_requests']
            )
    
    def get_service_info(self) -> Dict[str, Any]:
        """서비스 정보를 반환합니다."""
        return {
            'device': str(self.device),
            'service_config': self.service_config,
            'performance_metrics': self.performance_metrics,
            'cached_models': list(self.model_cache.keys()),
            'cache_size': len(self.result_cache)
        }
    
    def update_service_config(self, new_config: Dict[str, Any]):
        """서비스 설정을 업데이트합니다."""
        self.service_config.update(new_config)
        logger.info(f"Updated service config: {self.service_config}")
    
    def clear_cache(self, cache_type: str = 'all'):
        """캐시를 정리합니다."""
        if cache_type in ['all', 'models']:
            self.model_cache.clear()
            logger.info("Cleared model cache")
        
        if cache_type in ['all', 'results']:
            self.result_cache.clear()
            logger.info("Cleared result cache")
    
    def get_performance_report(self) -> Dict[str, Any]:
        """성능 리포트를 반환합니다."""
        total_requests = self.performance_metrics['total_requests']
        success_rate = 0.0
        if total_requests > 0:
            success_rate = self.performance_metrics['successful_requests'] / total_requests
        
        return {
            'total_requests': total_requests,
            'successful_requests': self.performance_metrics['successful_requests'],
            'failed_requests': self.performance_metrics['failed_requests'],
            'success_rate': success_rate,
            'average_processing_time': self.performance_metrics['average_processing_time'],
            'total_processing_time': self.performance_metrics['total_processing_time']
        }
    
    def shutdown(self):
        """서비스를 종료합니다."""
        self.executor.shutdown(wait=True)
        self.clear_cache('all')
        logger.info("PostProcessingService shutdown completed")
