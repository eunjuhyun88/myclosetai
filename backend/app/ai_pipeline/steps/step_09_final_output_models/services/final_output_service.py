"""
🔥 Final Output Service
=======================

최종 출력 생성 서비스의 핵심 로직을 담당하는 서비스 클래스입니다.
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
from pathlib import Path

# 프로젝트 로깅 설정 import
try:
    from backend.app.core.logging_config import get_logger
    logger = get_logger(__name__)
except ImportError:
    logger = logging.getLogger(__name__)

class FinalOutputService:
    """
    최종 출력 생성 서비스의 핵심 로직을 담당하는 서비스 클래스
    """

    def __init__(self, model_loader=None, processor=None, inference_engine=None):
        """
        Args:
            model_loader: 모델 로더 인스턴스
            processor: 데이터 프로세서 인스턴스
            inference_engine: 추론 엔진 인스턴스
        """
        self.model_loader = model_loader
        self.processor = processor
        self.inference_engine = inference_engine
        
        # 서비스 설정
        self.service_config = {
            'default_model': 'final_generator',
            'batch_size': 16,
            'enable_caching': True,
            'quality_threshold': 0.8,
            'output_formats': ['png', 'jpg', 'tiff'],
            'compression_quality': 95
        }
        
        # 캐시 (메모리 효율성을 위해)
        self._output_cache = {}
        self._max_cache_size = 500
        
        logger.info("✅ FinalOutputService initialized")

    def generate_final_output(self, input_data: Union[np.ndarray, Image.Image, torch.Tensor],
                            model_type: str = None,
                            output_format: str = 'png',
                            **kwargs) -> Dict[str, Any]:
        """
        최종 출력을 생성합니다.
        
        Args:
            input_data: 입력 데이터
            model_type: 사용할 모델 타입
            output_format: 출력 형식
            **kwargs: 추가 파라미터
            
        Returns:
            최종 출력 결과 딕셔너리
        """
        try:
            start_time = time.time()
            
            # 모델 타입 설정
            if model_type is None:
                model_type = self.service_config['default_model']
            
            # 캐시 확인
            cache_key = self._generate_cache_key(input_data, model_type, output_format)
            if self.service_config['enable_caching'] and cache_key in self._output_cache:
                logger.info("✅ Final output result found in cache")
                return self._output_cache[cache_key]
            
            # 입력 데이터 전처리
            if self.processor:
                processed_input = self.processor.preprocess_for_final_output(
                    input_data, **kwargs
                )
            else:
                processed_input = input_data
            
            # 최종 출력 생성
            if self.inference_engine:
                output_result = self.inference_engine.generate_final_output(
                    processed_input, model_type, **kwargs
                )
            else:
                # 기본 출력 생성 (간단한 처리)
                output_result = self._basic_output_generation(processed_input, model_type)
            
            # 출력 후처리
            if self.processor:
                final_output = self.processor.postprocess_final_output(
                    output_result['output'], **kwargs
                )
            else:
                final_output = output_result['output']
            
            # 결과 구성
            result = {
                'output': final_output,
                'quality_score': output_result.get('quality_score', 0.0),
                'quality_grade': output_result.get('quality_grade', 'Unknown'),
                'model_type': model_type,
                'output_format': output_format,
                'processing_time': time.time() - start_time,
                'timestamp': datetime.now().isoformat()
            }
            
            # 캐시에 저장
            if self.service_config['enable_caching']:
                self._add_to_cache(cache_key, result)
            
            logger.info(f"✅ Final output generation completed: {result['quality_grade']}")
            return result
            
        except Exception as e:
            logger.error(f"❌ Final output generation failed: {e}")
            return {
                'error': str(e),
                'quality_score': 0.0,
                'quality_grade': 'Error',
                'processing_time': time.time() - start_time if 'start_time' in locals() else 0.0,
                'timestamp': datetime.now().isoformat()
            }

    def generate_batch_outputs(self, input_data_list: List[Union[np.ndarray, Image.Image, torch.Tensor]],
                             model_type: str = None,
                             output_format: str = 'png',
                             **kwargs) -> List[Dict[str, Any]]:
        """
        여러 입력에 대해 최종 출력을 일괄 생성합니다.
        
        Args:
            input_data_list: 입력 데이터 리스트
            model_type: 사용할 모델 타입
            output_format: 출력 형식
            **kwargs: 추가 파라미터
            
        Returns:
            최종 출력 결과 리스트
        """
        try:
            start_time = time.time()
            batch_size = self.service_config['batch_size']
            results = []
            
            # 배치 단위로 처리
            for i in range(0, len(input_data_list), batch_size):
                batch_inputs = input_data_list[i:i + batch_size]
                batch_results = []
                
                for j, input_data in enumerate(batch_inputs):
                    try:
                        result = self.generate_final_output(
                            input_data, model_type, output_format, **kwargs
                        )
                        result['batch_index'] = i + j
                        batch_results.append(result)
                    except Exception as e:
                        logger.error(f"❌ Failed to generate output for input {i + j}: {e}")
                        batch_results.append({
                            'batch_index': i + j,
                            'error': str(e),
                            'quality_score': 0.0,
                            'quality_grade': 'Error',
                            'timestamp': datetime.now().isoformat()
                        })
                
                results.extend(batch_results)
                
                # 배치 처리 진행상황 로깅
                logger.info(f"✅ Batch {i//batch_size + 1} completed: {len(batch_results)} outputs")
            
            # 전체 통계 계산
            total_time = time.time() - start_time
            self._log_batch_statistics(results, total_time)
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Batch output generation failed: {e}")
            return []

    def _basic_output_generation(self, input_data: torch.Tensor, 
                                model_type: str) -> Dict[str, Any]:
        """
        기본 출력 생성 (추론 엔진이 없을 때 사용)
        """
        try:
            # 간단한 품질 향상
            if len(input_data.shape) == 4:
                input_data = input_data.squeeze(0)
            
            # 기본 품질 메트릭 계산
            brightness = input_data.mean().item()
            contrast = input_data.std().item()
            
            # 선명도 향상 (간단한 언샤프 마스크)
            if input_data.shape[0] == 3:  # RGB
                gray = 0.299 * input_data[0] + 0.587 * input_data[1] + 0.114 * input_data[2]
            else:
                gray = input_data[0]
            
            # Sobel 필터로 에지 검출
            sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                                 dtype=torch.float32, device=input_data.device)
            sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                                 dtype=torch.float32, device=input_data.device)
            
            edge_x = torch.conv2d(gray.unsqueeze(0).unsqueeze(0), 
                                sobel_x.unsqueeze(0).unsqueeze(0), padding=1)
            edge_y = torch.conv2d(gray.unsqueeze(0).unsqueeze(0), 
                                sobel_y.unsqueeze(0).unsqueeze(0), padding=1)
            
            edge_magnitude = torch.sqrt(edge_x**2 + edge_y**2)
            sharpness = edge_magnitude.mean().item()
            
            # 종합 품질 점수 계산 (0-1 범위)
            quality_score = min(1.0, (brightness * 0.3 + contrast * 0.3 + sharpness * 0.4))
            
            # 기본 출력 (입력과 동일)
            output = input_data
            
            return {
                'output': output,
                'quality_score': quality_score,
                'model_type': 'basic'
            }
            
        except Exception as e:
            logger.error(f"❌ Basic output generation failed: {e}")
            return {
                'output': input_data,
                'quality_score': 0.0,
                'model_type': 'basic',
                'error': str(e)
            }

    def _generate_cache_key(self, input_data: Union[np.ndarray, Image.Image, torch.Tensor],
                           model_type: str,
                           output_format: str) -> str:
        """
        캐시 키를 생성합니다.
        """
        try:
            # 입력 데이터 해시 생성 (간단한 방법)
            if isinstance(input_data, np.ndarray):
                # numpy array의 평균값과 표준편차로 해시 생성
                hash_value = f"{input_data.mean():.6f}_{input_data.std():.6f}_{input_data.shape}"
            elif isinstance(input_data, Image.Image):
                # PIL Image의 크기와 모드로 해시 생성
                hash_value = f"{input_data.size}_{input_data.mode}"
            elif isinstance(input_data, torch.Tensor):
                # torch tensor의 통계로 해시 생성
                hash_value = f"{input_data.mean().item():.6f}_{input_data.std().item():.6f}_{input_data.shape}"
            else:
                hash_value = str(hash(str(input_data)))
            
            return f"{model_type}_{output_format}_{hash_value}"
            
        except Exception as e:
            logger.warning(f"⚠️ Cache key generation failed: {e}")
            return f"{model_type}_{output_format}_{hash(str(input_data))}"

    def _add_to_cache(self, key: str, result: Dict[str, Any]):
        """
        결과를 캐시에 추가합니다.
        """
        try:
            # 캐시 크기 제한 확인
            if len(self._output_cache) >= self._max_cache_size:
                # 가장 오래된 항목 제거
                oldest_key = next(iter(self._output_cache))
                del self._output_cache[oldest_key]
                logger.debug("🗑️ Oldest cache entry removed")
            
            self._output_cache[key] = result
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
            
            # 성공한 생성 수
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
            logger.info(f"   Total inputs: {len(results)}")
            logger.info(f"   Successful: {len(successful_results)}")
            logger.info(f"   Errors: {error_count}")
            logger.info(f"   Average quality score: {avg_quality:.3f}")
            logger.info(f"   Quality grades: {grade_counts}")
            logger.info(f"   Total processing time: {total_time:.2f}s")
            logger.info(f"   Average time per input: {total_time/len(results):.3f}s")
            
        except Exception as e:
            logger.warning(f"⚠️ Statistics logging failed: {e}")

    def get_service_info(self) -> Dict[str, Any]:
        """
        서비스 정보를 반환합니다.
        """
        return {
            'service_name': 'FinalOutputService',
            'version': '1.0.0',
            'supported_models': self.inference_engine.supported_models if self.inference_engine else [],
            'cache_enabled': self.service_config['enable_caching'],
            'cache_size': len(self._output_cache),
            'max_cache_size': self._max_cache_size,
            'default_model': self.service_config['default_model'],
            'output_formats': self.service_config['output_formats'],
            'quality_threshold': self.service_config['quality_threshold']
        }

    def clear_cache(self):
        """
        캐시를 정리합니다.
        """
        try:
            cache_size = len(self._output_cache)
            self._output_cache.clear()
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

    def validate_output_quality(self, output: torch.Tensor, 
                              threshold: float = None) -> Dict[str, Any]:
        """
        출력 품질을 검증합니다.
        
        Args:
            output: 출력 텐서
            threshold: 품질 임계값
            
        Returns:
            품질 검증 결과
        """
        try:
            if threshold is None:
                threshold = self.service_config['quality_threshold']
            
            # 품질 메트릭 계산
            if len(output.shape) == 4:
                output = output.squeeze(0)
            
            # 밝기
            brightness = output.mean().item()
            
            # 대비
            contrast = output.std().item()
            
            # 선명도
            if output.shape[0] == 3:  # RGB
                gray = 0.299 * output[0] + 0.587 * output[1] + 0.114 * output[2]
            else:
                gray = output[0]
            
            # 라플라시안 필터
            laplacian_kernel = torch.tensor([
                [0, 1, 0],
                [1, -4, 1],
                [0, 1, 0]
            ], dtype=torch.float32, device=output.device).unsqueeze(0).unsqueeze(0)
            
            laplacian = torch.conv2d(gray.unsqueeze(1), laplacian_kernel, padding=1)
            sharpness = laplacian.var().item()
            
            # 종합 품질 점수
            quality_score = min(1.0, (brightness * 0.3 + contrast * 0.3 + sharpness * 0.4))
            
            # 품질 검증 결과
            validation_result = {
                'quality_score': quality_score,
                'brightness': brightness,
                'contrast': contrast,
                'sharpness': sharpness,
                'meets_threshold': quality_score >= threshold,
                'threshold': threshold,
                'validation_passed': quality_score >= threshold
            }
            
            return validation_result
            
        except Exception as e:
            logger.error(f"❌ Output quality validation failed: {e}")
            return {
                'quality_score': 0.0,
                'validation_passed': False,
                'error': str(e)
            }

    def export_output(self, output: torch.Tensor, 
                     output_format: str = 'png',
                     file_path: str = None,
                     **kwargs) -> Dict[str, Any]:
        """
        출력을 파일로 내보냅니다.
        
        Args:
            output: 출력 텐서
            output_format: 출력 형식
            file_path: 파일 경로
            **kwargs: 추가 파라미터
            
        Returns:
            내보내기 결과
        """
        try:
            # 텐서를 numpy array로 변환
            if len(output.shape) == 4:
                output = output.squeeze(0)
            
            if output.shape[0] == 3:  # RGB
                output_array = output.permute(1, 2, 0).cpu().numpy()
            else:
                output_array = output.squeeze(0).cpu().numpy()
            
            # 정규화
            if output_array.max() <= 1.0:
                output_array = (output_array * 255).astype(np.uint8)
            else:
                output_array = output_array.astype(np.uint8)
            
            # PIL Image로 변환
            if output_array.shape[2] == 3:
                image = Image.fromarray(output_array, 'RGB')
            else:
                image = Image.fromarray(output_array, 'L')
            
            # 파일 경로 설정
            if file_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                file_path = f"final_output_{timestamp}.{output_format}"
            
            # 파일 저장
            save_kwargs = {}
            if output_format == 'jpg':
                save_kwargs['quality'] = self.service_config['compression_quality']
            
            image.save(file_path, format=output_format.upper(), **save_kwargs)
            
            logger.info(f"✅ Output exported: {file_path}")
            
            return {
                'file_path': file_path,
                'output_format': output_format,
                'file_size_mb': Path(file_path).stat().st_size / (1024 * 1024),
                'export_successful': True
            }
            
        except Exception as e:
            logger.error(f"❌ Output export failed: {e}")
            return {
                'export_successful': False,
                'error': str(e)
            }
