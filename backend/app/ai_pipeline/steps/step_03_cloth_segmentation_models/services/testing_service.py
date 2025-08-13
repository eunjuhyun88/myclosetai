#!/usr/bin/env python3
"""
🔥 MyCloset AI - Step 03: 의류 세그멘테이션 - Testing Service
=====================================================================

테스팅을 위한 전용 서비스

Author: MyCloset AI Team  
Date: 2025-08-01
Version: 1.0
"""

import logging
import time
import numpy as np
from typing import Dict, Any, Optional, List
from PIL import Image

logger = logging.getLogger(__name__)

class TestingService:
    """테스팅 서비스"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """초기화"""
        self.config = config or {}
        self.logger = logging.getLogger(f"{__name__}.TestingService")
        self.test_results = {}
        
    def run_basic_tests(self, step_instance: Any) -> Dict[str, Any]:
        """기본 테스트 실행"""
        try:
            test_results = {}
            
            # 1. 초기화 테스트
            init_result = self._test_initialization(step_instance)
            test_results['initialization'] = init_result
            
            # 2. 모델 로딩 테스트
            model_result = self._test_model_loading(step_instance)
            test_results['model_loading'] = model_result
            
            # 3. 기본 처리 테스트
            processing_result = self._test_basic_processing(step_instance)
            test_results['basic_processing'] = processing_result
            
            # 4. 메모리 테스트
            memory_result = self._test_memory_usage(step_instance)
            test_results['memory_usage'] = memory_result
            
            self.test_results = test_results
            return test_results
            
        except Exception as e:
            self.logger.error(f"❌ 기본 테스트 실패: {e}")
            return {'error': str(e)}
    
    def _test_initialization(self, step_instance: Any) -> Dict[str, Any]:
        """초기화 테스트"""
        try:
            start_time = time.time()
            
            # 초기화 실행
            success = step_instance.initialize()
            init_time = time.time() - start_time
            
            return {
                'success': success,
                'time': init_time,
                'status': 'passed' if success else 'failed'
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'status': 'error'
            }
    
    def _test_model_loading(self, step_instance: Any) -> Dict[str, Any]:
        """모델 로딩 테스트"""
        try:
            # 모델 상태 확인
            models_loaded = 0
            total_models = 0
            
            if hasattr(step_instance, 'models_loading_status'):
                for model_name, status in step_instance.models_loading_status.items():
                    total_models += 1
                    if status:
                        models_loaded += 1
            
            success_rate = (models_loaded / total_models * 100) if total_models > 0 else 0
            
            return {
                'models_loaded': models_loaded,
                'total_models': total_models,
                'success_rate': success_rate,
                'status': 'passed' if success_rate > 0 else 'failed'
            }
            
        except Exception as e:
            return {
                'error': str(e),
                'status': 'error'
            }
    
    def _test_basic_processing(self, step_instance: Any) -> Dict[str, Any]:
        """기본 처리 테스트"""
        try:
            # 테스트 이미지 생성
            test_image = Image.new('RGB', (512, 512), (128, 128, 128))
            test_image_array = np.array(test_image)
            
            # 기본 처리 실행
            start_time = time.time()
            
            test_input = {
                'image': test_image_array,
                'person_parsing': {},
                'pose_info': {}
            }
            
            result = step_instance.process(**test_input)
            processing_time = time.time() - start_time
            
            success = result.get('success', False)
            
            return {
                'success': success,
                'processing_time': processing_time,
                'confidence': result.get('confidence', 0),
                'status': 'passed' if success else 'failed'
            }
            
        except Exception as e:
            return {
                'error': str(e),
                'status': 'error'
            }
    
    def _test_memory_usage(self, step_instance: Any) -> Dict[str, Any]:
        """메모리 사용량 테스트"""
        try:
            # 메모리 사용량 확인 (간단한 방법)
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            
            return {
                'memory_mb': memory_mb,
                'status': 'passed' if memory_mb < 2000 else 'warning'  # 2GB 이상시 경고
            }
            
        except Exception as e:
            return {
                'error': str(e),
                'status': 'error'
            }
    
    def run_performance_tests(self, step_instance: Any) -> Dict[str, Any]:
        """성능 테스트 실행"""
        try:
            performance_results = {}
            
            # 1. 처리 속도 테스트
            speed_result = self._test_processing_speed(step_instance)
            performance_results['processing_speed'] = speed_result
            
            # 2. 메모리 효율성 테스트
            memory_result = self._test_memory_efficiency(step_instance)
            performance_results['memory_efficiency'] = memory_result
            
            # 3. 정확도 테스트
            accuracy_result = self._test_accuracy(step_instance)
            performance_results['accuracy'] = accuracy_result
            
            return performance_results
            
        except Exception as e:
            self.logger.error(f"❌ 성능 테스트 실패: {e}")
            return {'error': str(e)}
    
    def _test_processing_speed(self, step_instance: Any) -> Dict[str, Any]:
        """처리 속도 테스트"""
        try:
            # 여러 크기의 이미지로 테스트
            test_sizes = [(256, 256), (512, 512), (1024, 1024)]
            speed_results = {}
            
            for width, height in test_sizes:
                test_image = Image.new('RGB', (width, height), (128, 128, 128))
                test_image_array = np.array(test_image)
                
                start_time = time.time()
                result = step_instance.process(image=test_image_array)
                processing_time = time.time() - start_time
                
                speed_results[f"{width}x{height}"] = {
                    'time': processing_time,
                    'success': result.get('success', False)
                }
            
            return speed_results
            
        except Exception as e:
            return {'error': str(e)}
    
    def _test_memory_efficiency(self, step_instance: Any) -> Dict[str, Any]:
        """메모리 효율성 테스트"""
        try:
            # 메모리 사용량 모니터링
            import psutil
            process = psutil.Process()
            
            # 처리 전 메모리
            memory_before = process.memory_info().rss / 1024 / 1024
            
            # 테스트 처리
            test_image = Image.new('RGB', (512, 512), (128, 128, 128))
            test_image_array = np.array(test_image)
            result = step_instance.process(image=test_image_array)
            
            # 처리 후 메모리
            memory_after = process.memory_info().rss / 1024 / 1024
            memory_increase = memory_after - memory_before
            
            return {
                'memory_before_mb': memory_before,
                'memory_after_mb': memory_after,
                'memory_increase_mb': memory_increase,
                'efficiency': 'good' if memory_increase < 100 else 'poor'
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def _test_accuracy(self, step_instance: Any) -> Dict[str, Any]:
        """정확도 테스트"""
        try:
            # 간단한 정확도 테스트 (실제로는 더 복잡한 검증이 필요)
            test_image = Image.new('RGB', (512, 512), (128, 128, 128))
            test_image_array = np.array(test_image)
            
            result = step_instance.process(image=test_image_array)
            
            confidence = result.get('confidence', 0)
            masks_count = len(result.get('masks', {}))
            
            return {
                'confidence': confidence,
                'masks_count': masks_count,
                'accuracy_score': confidence * 100  # 간단한 점수
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def generate_test_report(self) -> Dict[str, Any]:
        """테스트 리포트 생성"""
        try:
            report = {
                'timestamp': time.time(),
                'test_results': self.test_results,
                'summary': self._generate_summary()
            }
            
            return report
            
        except Exception as e:
            self.logger.error(f"❌ 테스트 리포트 생성 실패: {e}")
            return {'error': str(e)}
    
    def _generate_summary(self) -> Dict[str, Any]:
        """테스트 요약 생성"""
        try:
            if not self.test_results:
                return {'status': 'no_tests_run'}
            
            passed_tests = 0
            total_tests = 0
            
            for test_name, result in self.test_results.items():
                total_tests += 1
                if result.get('status') == 'passed':
                    passed_tests += 1
            
            success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
            
            return {
                'total_tests': total_tests,
                'passed_tests': passed_tests,
                'failed_tests': total_tests - passed_tests,
                'success_rate': success_rate,
                'overall_status': 'passed' if success_rate >= 80 else 'failed'
            }
            
        except Exception as e:
            return {'error': str(e)}
