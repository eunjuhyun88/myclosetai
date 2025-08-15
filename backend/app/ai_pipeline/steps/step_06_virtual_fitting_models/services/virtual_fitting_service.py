"""
🔥 Virtual Fitting Service
==========================

가상 피팅 서비스의 핵심 로직을 담당하는 서비스 클래스입니다.
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

class VirtualFittingService:
    """
    가상 피팅 서비스의 핵심 로직을 담당하는 서비스 클래스
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
            'default_model': 'virtual_fitting',
            'batch_size': 8,
            'enable_caching': True,
            'fitting_quality_threshold': 0.7,
            'enable_real_time_preview': True,
            'max_fitting_attempts': 3
        }
        
        # 캐시 (메모리 효율성을 위해)
        self._fitting_cache = {}
        self._max_cache_size = 300
        
        logger.info("✅ VirtualFittingService initialized")

    def perform_virtual_fitting(self, person_image: Union[np.ndarray, Image.Image, torch.Tensor],
                              clothing_image: Union[np.ndarray, Image.Image, torch.Tensor],
                              model_type: str = None,
                              **kwargs) -> Dict[str, Any]:
        """
        가상 피팅을 수행합니다.
        
        Args:
            person_image: 사람 이미지
            clothing_image: 의류 이미지
            model_type: 사용할 모델 타입
            **kwargs: 추가 파라미터
            
        Returns:
            가상 피팅 결과 딕셔너리
        """
        try:
            start_time = time.time()
            
            # 모델 타입 설정
            if model_type is None:
                model_type = self.service_config['default_model']
            
            # 캐시 확인
            cache_key = self._generate_cache_key(person_image, clothing_image, model_type)
            if self.service_config['enable_caching'] and cache_key in self._fitting_cache:
                logger.info("✅ Virtual fitting result found in cache")
                return self._fitting_cache[cache_key]
            
            # 입력 데이터 전처리
            if self.processor:
                processed_data = self.processor.preprocess_for_virtual_fitting(
                    person_image, clothing_image, **kwargs
                )
            else:
                processed_data = {
                    'person_image': person_image,
                    'clothing_image': clothing_image
                }
            
            # 가상 피팅 실행
            if self.inference_engine:
                fitting_result = self.inference_engine.perform_virtual_fitting(
                    processed_data, model_type, **kwargs
                )
            else:
                # 기본 가상 피팅 (간단한 처리)
                fitting_result = self._basic_virtual_fitting(processed_data, model_type)
            
            # 결과 후처리
            if self.processor:
                final_output = self.processor.postprocess_virtual_fitting(
                    fitting_result['fitted_image'], **kwargs
                )
            else:
                final_output = fitting_result['fitted_image']
            
            # 결과 구성
            result = {
                'fitted_image': final_output,
                'person_image': processed_data['person_image'],
                'clothing_image': processed_data['clothing_image'],
                'fitting_quality': fitting_result.get('fitting_quality', 0.0),
                'fitting_confidence': fitting_result.get('confidence', 0.0),
                'model_type': model_type,
                'processing_time': time.time() - start_time,
                'timestamp': datetime.now().isoformat()
            }
            
            # 캐시에 저장
            if self.service_config['enable_caching']:
                self._add_to_cache(cache_key, result)
            
            logger.info(f"✅ Virtual fitting completed: quality={result['fitting_quality']:.3f}")
            return result
            
        except Exception as e:
            logger.error(f"❌ Virtual fitting failed: {e}")
            return {
                'error': str(e),
                'fitting_quality': 0.0,
                'processing_time': time.time() - start_time if 'start_time' in locals() else 0.0,
                'timestamp': datetime.now().isoformat()
            }

    def perform_batch_fitting(self, person_images: List[Union[np.ndarray, Image.Image, torch.Tensor]],
                            clothing_images: List[Union[np.ndarray, Image.Image, torch.Tensor]],
                            model_type: str = None,
                            **kwargs) -> List[Dict[str, Any]]:
        """
        여러 이미지에 대해 가상 피팅을 일괄 수행합니다.
        
        Args:
            person_images: 사람 이미지 리스트
            clothing_images: 의류 이미지 리스트
            model_type: 사용할 모델 타입
            **kwargs: 추가 파라미터
            
        Returns:
            가상 피팅 결과 리스트
        """
        try:
            if len(person_images) != len(clothing_images):
                raise ValueError("Person images and clothing images must have the same length")
            
            start_time = time.time()
            batch_size = self.service_config['batch_size']
            results = []
            
            # 배치 단위로 처리
            for i in range(0, len(person_images), batch_size):
                batch_person = person_images[i:i + batch_size]
                batch_clothing = clothing_images[i:i + batch_size]
                batch_results = []
                
                for j, (person_img, clothing_img) in enumerate(zip(batch_person, batch_clothing)):
                    try:
                        result = self.perform_virtual_fitting(
                            person_img, clothing_img, model_type, **kwargs
                        )
                        result['batch_index'] = i + j
                        batch_results.append(result)
                    except Exception as e:
                        logger.error(f"❌ Failed to perform fitting for batch {i + j}: {e}")
                        batch_results.append({
                            'batch_index': i + j,
                            'error': str(e),
                            'fitting_quality': 0.0,
                            'timestamp': datetime.now().isoformat()
                        })
                
                results.extend(batch_results)
                
                # 배치 처리 진행상황 로깅
                logger.info(f"✅ Batch {i//batch_size + 1} completed: {len(batch_results)} fittings")
            
            # 전체 통계 계산
            total_time = time.time() - start_time
            self._log_batch_statistics(results, total_time)
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Batch virtual fitting failed: {e}")
            return []

    def _basic_virtual_fitting(self, processed_data: Dict[str, torch.Tensor], 
                              model_type: str) -> Dict[str, Any]:
        """
        기본 가상 피팅 (추론 엔진이 없을 때 사용)
        """
        try:
            person_image = processed_data['person_image']
            clothing_image = processed_data['clothing_image']
            
            # 간단한 이미지 합성 (실제로는 더 정교한 알고리즘 사용)
            if len(person_image.shape) == 4:
                person_image = person_image.squeeze(0)
            if len(clothing_image.shape) == 4:
                clothing_image = clothing_image.squeeze(0)
            
            # 의류를 사람 이미지에 오버레이
            # 실제 구현에서는 신체 부위별 마스킹과 블렌딩 사용
            fitted_image = self._overlay_clothing_on_person(person_image, clothing_image)
            
            # 피팅 품질 계산
            fitting_quality = self._calculate_fitting_quality(fitted_image)
            
            return {
                'fitted_image': fitted_image,
                'fitting_quality': fitting_quality,
                'confidence': 0.7,
                'model_type': 'basic'
            }
            
        except Exception as e:
            logger.error(f"❌ Basic virtual fitting failed: {e}")
            return {
                'fitted_image': processed_data['person_image'],
                'fitting_quality': 0.0,
                'confidence': 0.0,
                'model_type': 'basic',
                'error': str(e)
            }

    def _overlay_clothing_on_person(self, person_image: torch.Tensor, 
                                   clothing_image: torch.Tensor) -> torch.Tensor:
        """
        의류를 사람 이미지에 오버레이합니다.
        """
        try:
            # 간단한 알파 블렌딩
            alpha = 0.8  # 의류 투명도
            
            # 의류 이미지 크기 조정
            if clothing_image.shape[-2:] != person_image.shape[-2:]:
                clothing_image = F.interpolate(
                    clothing_image.unsqueeze(0), 
                    size=person_image.shape[-2:], 
                    mode='bilinear', 
                    align_corners=False
                ).squeeze(0)
            
            # 오버레이
            fitted_image = person_image * (1 - alpha) + clothing_image * alpha
            
            return torch.clamp(fitted_image, 0, 1)
            
        except Exception as e:
            logger.warning(f"⚠️ Clothing overlay failed: {e}")
            return person_image

    def _calculate_fitting_quality(self, fitted_image: torch.Tensor) -> float:
        """
        피팅 품질을 계산합니다.
        """
        try:
            # 기본 품질 메트릭
            if len(fitted_image.shape) == 4:
                fitted_image = fitted_image.squeeze(0)
            
            # 밝기
            brightness = fitted_image.mean().item()
            
            # 대비
            contrast = fitted_image.std().item()
            
            # 선명도 (간단한 에지 검출)
            if fitted_image.shape[0] == 3:  # RGB
                gray = 0.299 * fitted_image[0] + 0.587 * fitted_image[1] + 0.114 * fitted_image[2]
            else:
                gray = fitted_image[0]
            
            # Sobel 필터로 에지 검출
            sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                                 dtype=torch.float32, device=fitted_image.device)
            sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                                 dtype=torch.float32, device=fitted_image.device)
            
            edge_x = torch.conv2d(gray.unsqueeze(0).unsqueeze(0), 
                                sobel_x.unsqueeze(0).unsqueeze(0), padding=1)
            edge_y = torch.conv2d(gray.unsqueeze(0).unsqueeze(0), 
                                sobel_y.unsqueeze(0).unsqueeze(0), padding=1)
            
            edge_magnitude = torch.sqrt(edge_x**2 + edge_y**2)
            sharpness = edge_magnitude.mean().item()
            
            # 종합 품질 점수 계산 (0-1 범위)
            quality_score = min(1.0, (brightness * 0.3 + contrast * 0.3 + sharpness * 0.4))
            
            return quality_score
            
        except Exception as e:
            logger.warning(f"⚠️ Fitting quality calculation failed: {e}")
            return 0.5

    def _generate_cache_key(self, person_image: Union[np.ndarray, Image.Image, torch.Tensor],
                           clothing_image: Union[np.ndarray, Image.Image, torch.Tensor],
                           model_type: str) -> str:
        """
        캐시 키를 생성합니다.
        """
        try:
            # 이미지 해시 생성 (간단한 방법)
            person_hash = self._generate_image_hash(person_image)
            clothing_hash = self._generate_image_hash(clothing_image)
            
            return f"{model_type}_{person_hash}_{clothing_hash}"
            
        except Exception as e:
            logger.warning(f"⚠️ Cache key generation failed: {e}")
            return f"{model_type}_{hash(str(person_image))}_{hash(str(clothing_image))}"

    def _generate_image_hash(self, image: Union[np.ndarray, Image.Image, torch.Tensor]) -> str:
        """
        이미지 해시를 생성합니다.
        """
        try:
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
            
            return hash_value
            
        except Exception as e:
            logger.warning(f"⚠️ Image hash generation failed: {e}")
            return str(hash(str(image)))

    def _add_to_cache(self, key: str, result: Dict[str, Any]):
        """
        결과를 캐시에 추가합니다.
        """
        try:
            # 캐시 크기 제한 확인
            if len(self._fitting_cache) >= self._max_cache_size:
                # 가장 오래된 항목 제거
                oldest_key = next(iter(self._fitting_cache))
                del self._fitting_cache[oldest_key]
                logger.debug("🗑️ Oldest cache entry removed")
            
            self._fitting_cache[key] = result
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
            
            # 성공한 피팅 수
            successful_results = [r for r in results if 'error' not in r]
            error_count = len(results) - len(successful_results)
            
            # 품질 점수 통계
            quality_scores = [r.get('fitting_quality', 0.0) for r in successful_results if 'fitting_quality' in r]
            avg_quality = np.mean(quality_scores) if quality_scores else 0.0
            max_quality = max(quality_scores) if quality_scores else 0.0
            min_quality = min(quality_scores) if quality_scores else 0.0
            
            # 통계 로깅
            logger.info(f"📊 Batch Fitting Statistics:")
            logger.info(f"   Total fittings: {len(results)}")
            logger.info(f"   Successful: {len(successful_results)}")
            logger.info(f"   Errors: {error_count}")
            logger.info(f"   Average quality: {avg_quality:.3f}")
            logger.info(f"   Quality range: {min_quality:.3f} - {max_quality:.3f}")
            logger.info(f"   Total processing time: {total_time:.2f}s")
            logger.info(f"   Average time per fitting: {total_time/len(results):.3f}s")
            
        except Exception as e:
            logger.warning(f"⚠️ Statistics logging failed: {e}")

    def get_service_info(self) -> Dict[str, Any]:
        """
        서비스 정보를 반환합니다.
        """
        return {
            'service_name': 'VirtualFittingService',
            'version': '1.0.0',
            'supported_models': self.inference_engine.supported_models if self.inference_engine else [],
            'cache_enabled': self.service_config['enable_caching'],
            'cache_size': len(self._fitting_cache),
            'max_cache_size': self._max_cache_size,
            'default_model': self.service_config['default_model'],
            'fitting_quality_threshold': self.service_config['fitting_quality_threshold'],
            'max_fitting_attempts': self.service_config['max_fitting_attempts']
        }

    def clear_cache(self):
        """
        캐시를 정리합니다.
        """
        try:
            cache_size = len(self._fitting_cache)
            self._fitting_cache.clear()
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

    def validate_fitting_result(self, fitted_image: torch.Tensor, 
                              threshold: float = None) -> Dict[str, Any]:
        """
        피팅 결과를 검증합니다.
        
        Args:
            fitted_image: 피팅된 이미지
            threshold: 품질 임계값
            
        Returns:
            검증 결과
        """
        try:
            if threshold is None:
                threshold = self.service_config['fitting_quality_threshold']
            
            # 품질 점수 계산
            quality_score = self._calculate_fitting_quality(fitted_image)
            
            # 검증 결과
            validation_result = {
                'quality_score': quality_score,
                'meets_threshold': quality_score >= threshold,
                'threshold': threshold,
                'validation_passed': quality_score >= threshold,
                'quality_grade': self._get_quality_grade(quality_score)
            }
            
            return validation_result
            
        except Exception as e:
            logger.error(f"❌ Fitting result validation failed: {e}")
            return {
                'quality_score': 0.0,
                'validation_passed': False,
                'error': str(e)
            }

    def _get_quality_grade(self, quality_score: float) -> str:
        """
        품질 점수에 따른 등급을 반환합니다.
        """
        if quality_score >= 0.9:
            return 'Excellent'
        elif quality_score >= 0.7:
            return 'Good'
        elif quality_score >= 0.5:
            return 'Fair'
        else:
            return 'Poor'
