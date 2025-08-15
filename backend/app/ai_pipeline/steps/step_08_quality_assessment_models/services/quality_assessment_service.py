"""
🔥 Quality Assessment Service
============================

품질 평가 서비스의 핵심 로직을 담당하는 서비스 클래스입니다.
논문 기반의 AI 모델 구조에 맞춰 구현되었습니다.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Union, List, Tuple
import numpy as np
from PIL import Image
import logging
import time
from datetime import datetime

# 프로젝트 로깅 설정 import
try:
    from backend.app.core.logging_config import get_logger
    logger = get_logger(__name__)
except ImportError:
    logger = logging.getLogger(__name__)

class QualityAssessmentService:
    """
    품질 평가 서비스의 핵심 로직을 담당하는 서비스 클래스
    """

    def __init__(self, model_loader=None, processor=None, inference_engine=None):
        """
        Args:
            model_loader: 모델 로더 인스턴스
            processor: 이미지 프로세서 인스턴스
            inference_engine: 추론 엔진 인스턴스
        """
        self.model_loader = model_loader
        self.processor = processor
        self.inference_engine = inference_engine
        
        # 서비스 설정
        self.service_config = {
            'default_model': 'qualitynet',
            'batch_size': 32,
            'enable_caching': True,
            'quality_thresholds': {
                'excellent': 0.8,
                'good': 0.6,
                'fair': 0.4,
                'poor': 0.0
            }
        }
        
        # 캐시 (메모리 효율성을 위해)
        self._quality_cache = {}
        self._max_cache_size = 1000
        
        logger.info("✅ QualityAssessmentService initialized")

    def assess_single_image(self, image: Union[np.ndarray, Image.Image, torch.Tensor],
                          model_type: str = None,
                          **kwargs) -> Dict[str, Any]:
        """
        단일 이미지의 품질을 평가합니다.
        
        Args:
            image: 입력 이미지
            model_type: 사용할 모델 타입
            **kwargs: 추가 파라미터
            
        Returns:
            품질 평가 결과 딕셔너리
        """
        try:
            start_time = time.time()
            
            # 모델 타입 설정
            if model_type is None:
                model_type = self.service_config['default_model']
            
            # 캐시 확인
            cache_key = self._generate_cache_key(image, model_type)
            if self.service_config['enable_caching'] and cache_key in self._quality_cache:
                logger.info("✅ Quality assessment result found in cache")
                return self._quality_cache[cache_key]
            
            # 이미지 전처리
            if self.processor:
                processed_image = self.processor.preprocess_for_quality_assessment(
                    image, target_size=(224, 224), normalize=True
                )
            else:
                processed_image = image
            
            # 품질 평가 실행
            if self.inference_engine:
                quality_result = self.inference_engine.assess_image_quality(
                    processed_image, model_type, **kwargs
                )
            else:
                # 기본 품질 평가 (간단한 메트릭)
                quality_result = self._basic_quality_assessment(processed_image)
            
            # 결과 후처리
            result = self._postprocess_quality_result(quality_result, model_type)
            result['processing_time'] = time.time() - start_time
            result['timestamp'] = datetime.now().isoformat()
            
            # 캐시에 저장
            if self.service_config['enable_caching']:
                self._add_to_cache(cache_key, result)
            
            logger.info(f"✅ Single image quality assessment completed: {result['quality_grade']}")
            return result
            
        except Exception as e:
            logger.error(f"❌ Single image quality assessment failed: {e}")
            return {
                'error': str(e),
                'quality_score': 0.0,
                'quality_grade': 'Error',
                'processing_time': time.time() - start_time if 'start_time' in locals() else 0.0,
                'timestamp': datetime.now().isoformat()
            }

    def assess_batch_images(self, images: List[Union[np.ndarray, Image.Image, torch.Tensor]],
                          model_type: str = None,
                          **kwargs) -> List[Dict[str, Any]]:
        """
        여러 이미지의 품질을 일괄 평가합니다.
        
        Args:
            images: 입력 이미지 리스트
            model_type: 사용할 모델 타입
            **kwargs: 추가 파라미터
            
        Returns:
            품질 평가 결과 리스트
        """
        try:
            start_time = time.time()
            batch_size = self.service_config['batch_size']
            results = []
            
            # 배치 단위로 처리
            for i in range(0, len(images), batch_size):
                batch_images = images[i:i + batch_size]
                batch_results = []
                
                for j, image in enumerate(batch_images):
                    try:
                        result = self.assess_single_image(image, model_type, **kwargs)
                        result['batch_index'] = i + j
                        batch_results.append(result)
                    except Exception as e:
                        logger.error(f"❌ Failed to assess image {i + j}: {e}")
                        batch_results.append({
                            'batch_index': i + j,
                            'error': str(e),
                            'quality_score': 0.0,
                            'quality_grade': 'Error',
                            'timestamp': datetime.now().isoformat()
                        })
                
                results.extend(batch_results)
                
                # 배치 처리 진행상황 로깅
                logger.info(f"✅ Batch {i//batch_size + 1} completed: {len(batch_results)} images")
            
            # 전체 통계 계산
            total_time = time.time() - start_time
            self._log_batch_statistics(results, total_time)
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Batch quality assessment failed: {e}")
            return []

    def _basic_quality_assessment(self, image: torch.Tensor) -> Dict[str, Any]:
        """
        기본 품질 평가 (추론 엔진이 없을 때 사용)
        """
        try:
            # 간단한 품질 메트릭 계산
            if len(image.shape) == 4:
                image = image.squeeze(0)
            
            # 밝기
            brightness = image.mean().item()
            
            # 대비
            contrast = image.std().item()
            
            # 선명도 (간단한 에지 검출)
            if image.shape[0] == 3:  # RGB
                gray = 0.299 * image[0] + 0.587 * image[1] + 0.114 * image[2]
            else:
                gray = image[0]
            
            # Sobel 필터로 에지 검출
            sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                                 dtype=torch.float32, device=image.device)
            sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                                 dtype=torch.float32, device=image.device)
            
            edge_x = torch.conv2d(gray.unsqueeze(0).unsqueeze(0), 
                                sobel_x.unsqueeze(0).unsqueeze(0), padding=1)
            edge_y = torch.conv2d(gray.unsqueeze(0).unsqueeze(0), 
                                sobel_y.unsqueeze(0).unsqueeze(0), padding=1)
            
            edge_magnitude = torch.sqrt(edge_x**2 + edge_y**2)
            sharpness = edge_magnitude.mean().item()
            
            # 종합 품질 점수 계산 (0-1 범위)
            quality_score = min(1.0, (brightness * 0.3 + contrast * 0.3 + sharpness * 0.4))
            
            return {
                'quality_score': quality_score,
                'brightness': brightness,
                'contrast': contrast,
                'sharpness': sharpness,
                'model_type': 'basic'
            }
            
        except Exception as e:
            logger.error(f"❌ Basic quality assessment failed: {e}")
            return {
                'quality_score': 0.0,
                'model_type': 'basic',
                'error': str(e)
            }

    def _postprocess_quality_result(self, result: Dict[str, Any], 
                                  model_type: str) -> Dict[str, Any]:
        """
        품질 평가 결과를 후처리합니다.
        """
        try:
            # 품질 등급 결정
            quality_score = result.get('quality_score', 0.0)
            thresholds = self.service_config['quality_thresholds']
            
            if quality_score >= thresholds['excellent']:
                quality_grade = 'Excellent'
            elif quality_score >= thresholds['good']:
                quality_grade = 'Good'
            elif quality_score >= thresholds['fair']:
                quality_grade = 'Fair'
            else:
                quality_grade = 'Poor'
            
            # 결과에 품질 등급 추가
            result['quality_grade'] = quality_grade
            
            # 신뢰도 점수 추가 (모델 타입별)
            confidence_scores = {
                'qualitynet': 0.95,
                'brisque': 0.90,
                'niqe': 0.88,
                'piqe': 0.92,
                'basic': 0.70
            }
            
            result['confidence'] = confidence_scores.get(model_type, 0.80)
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Result postprocessing failed: {e}")
            result['quality_grade'] = 'Unknown'
            result['confidence'] = 0.0
            return result

    def _generate_cache_key(self, image: Union[np.ndarray, Image.Image, torch.Tensor],
                           model_type: str) -> str:
        """
        캐시 키를 생성합니다.
        """
        try:
            # 이미지 해시 생성 (간단한 방법)
            if isinstance(image, np.ndarray):
                # numpy array의 평균값과 표준편차로 해시 생성
                hash_value = f"{image.mean():.6f}_{image.std():.6f}_{image.shape}"
            elif isinstance(image, Image.Image):
                # PIL Image의 크기와 모드로 해시 생성
                hash_value = f"{image.size}_{image.mode}"
            elif isinstance(image, torch.Tensor):
                # torch tensor의 통계로 해시 생성
                hash_value = f"{image.mean().item():.6f}_{image.std().item():.6f}_{image.shape}"
            else:
                hash_value = str(hash(str(image)))
            
            return f"{model_type}_{hash_value}"
            
        except Exception as e:
            logger.warning(f"⚠️ Cache key generation failed: {e}")
            return f"{model_type}_{hash(str(image))}"

    def _add_to_cache(self, key: str, result: Dict[str, Any]):
        """
        결과를 캐시에 추가합니다.
        """
        try:
            # 캐시 크기 제한 확인
            if len(self._quality_cache) >= self._max_cache_size:
                # 가장 오래된 항목 제거
                oldest_key = next(iter(self._quality_cache))
                del self._quality_cache[oldest_key]
                logger.debug("🗑️ Oldest cache entry removed")
            
            self._quality_cache[key] = result
            logger.debug(f"💾 Result cached: {key}")
            
        except Exception as e:
            logger.warning(f"⚠️ Cache addition failed: {e}")

    def _log_batch_statistics(self, results: List[Dict[str, Any]], total_time: float):
        """
        배치 처리 통계를 로깅합니다.
        """
        try:
            if not results:
                return
            
            # 성공한 평가 수
            successful_results = [r for r in results if 'error' not in r]
            error_count = len(results) - len(successful_results)
            
            # 품질 등급별 분포
            grade_counts = {}
            for result in successful_results:
                grade = result.get('quality_grade', 'Unknown')
                grade_counts[grade] = grade_counts.get(grade, 0) + 1
            
            # 평균 품질 점수
            quality_scores = [r.get('quality_score', 0.0) for r in successful_results if 'quality_score' in r]
            avg_quality = np.mean(quality_scores) if quality_scores else 0.0
            
            # 통계 로깅
            logger.info(f"📊 Batch Statistics:")
            logger.info(f"   Total images: {len(results)}")
            logger.info(f"   Successful: {len(successful_results)}")
            logger.info(f"   Errors: {error_count}")
            logger.info(f"   Average quality score: {avg_quality:.3f}")
            logger.info(f"   Quality grades: {grade_counts}")
            logger.info(f"   Total processing time: {total_time:.2f}s")
            logger.info(f"   Average time per image: {total_time/len(results):.3f}s")
            
        except Exception as e:
            logger.warning(f"⚠️ Statistics logging failed: {e}")

    def get_service_info(self) -> Dict[str, Any]:
        """
        서비스 정보를 반환합니다.
        """
        return {
            'service_name': 'QualityAssessmentService',
            'version': '1.0.0',
            'supported_models': self.inference_engine.supported_models if self.inference_engine else [],
            'cache_enabled': self.service_config['enable_caching'],
            'cache_size': len(self._quality_cache),
            'max_cache_size': self._max_cache_size,
            'default_model': self.service_config['default_model'],
            'quality_thresholds': self.service_config['quality_thresholds']
        }

    def clear_cache(self):
        """
        캐시를 정리합니다.
        """
        try:
            cache_size = len(self._quality_cache)
            self._quality_cache.clear()
            logger.info(f"🗑️ Cache cleared: {cache_size} entries removed")
        except Exception as e:
            logger.error(f"❌ Cache clearing failed: {e}")

    def update_service_config(self, **kwargs):
        """
        서비스 설정을 업데이트합니다.
        """
        try:
            for key, value in kwargs.items():
                if key in self.service_config:
                    self.service_config[key] = value
                    logger.info(f"✅ Service config updated: {key} = {value}")
                else:
                    logger.warning(f"⚠️ Unknown config key: {key}")
        except Exception as e:
            logger.error(f"❌ Service config update failed: {e}")
