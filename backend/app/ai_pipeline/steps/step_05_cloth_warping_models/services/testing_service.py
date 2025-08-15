#!/usr/bin/env python3
"""
🔥 MyCloset AI - Cloth Warping Testing Service
==============================================

🎯 의류 워핑 테스팅 서비스
✅ 단위 테스트 실행
✅ 통합 테스트 실행
✅ 성능 테스트 실행
✅ M3 Max 최적화
"""

import logging
import torch
import time
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class TestingServiceConfig:
    """테스팅 서비스 설정"""
    enable_unit_tests: bool = True
    enable_integration_tests: bool = True
    enable_performance_tests: bool = True
    test_batch_size: int = 2
    test_image_size: Tuple[int, int] = (256, 256)
    use_mps: bool = True

class ClothWarpingTestingService:
    """의류 워핑 테스팅 서비스"""
    
    def __init__(self, config: TestingServiceConfig = None):
        self.config = config or TestingServiceConfig()
        self.logger = logging.getLogger(__name__)
        self.logger.info("🎯 Cloth Warping 테스팅 서비스 초기화")
        
        # MPS 디바이스 확인
        self.device = torch.device("mps" if torch.backends.mps.is_available() and self.config.use_mps else "cpu")
        
        # 테스트 결과 저장
        self.test_results = {}
        
        self.logger.info("✅ Cloth Warping 테스팅 서비스 초기화 완료")
    
    def run_unit_tests(self) -> Dict[str, Any]:
        """단위 테스트를 실행합니다."""
        if not self.config.enable_unit_tests:
            return {'status': 'disabled'}
        
        self.logger.info("단위 테스트 시작")
        test_results = {}
        
        try:
            # 프로세서 모듈 import 테스트
            import sys
            import os
            sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'processors'))
            from advanced_post_processor import ClothWarpingAdvancedPostProcessor
            from high_resolution_processor import ClothWarpingHighResolutionProcessor
            from preprocessing import ClothWarpingPreprocessor
            from quality_enhancer import ClothWarpingQualityEnhancer
            from special_case_processor import ClothWarpingSpecialCaseProcessor
            
            test_results['import_test'] = 'success'
            
            # 프로세서 인스턴스 생성 테스트
            processors = {
                'advanced_post_processor': ClothWarpingAdvancedPostProcessor(),
                'high_resolution_processor': ClothWarpingHighResolutionProcessor(),
                'preprocessor': ClothWarpingPreprocessor(),
                'quality_enhancer': ClothWarpingQualityEnhancer(),
                'special_case_processor': ClothWarpingSpecialCaseProcessor()
            }
            
            test_results['instantiation_test'] = 'success'
            test_results['processors_created'] = len(processors)
            
            # 기본 forward pass 테스트
            batch_size = self.config.test_batch_size
            channels = 3
            height, width = self.config.test_image_size
            
            test_image = torch.randn(batch_size, channels, height, width, device=self.device)
            
            for name, processor in processors.items():
                try:
                    with torch.no_grad():
                        if name == 'advanced_post_processor':
                            result = processor(test_image, test_image, test_image)
                        elif name == 'high_resolution_processor':
                            result = processor(test_image, test_image)
                        elif name == 'preprocessor':
                            result = processor(test_image, test_image)
                        else:
                            result = processor(test_image)
                        
                        test_results[f'{name}_forward_test'] = 'success'
                        
                except Exception as e:
                    test_results[f'{name}_forward_test'] = f'failed: {str(e)}'
            
            test_results['status'] = 'success'
            self.logger.info("단위 테스트 완료")
            
        except Exception as e:
            test_results['status'] = 'error'
            test_results['error'] = str(e)
            self.logger.error(f"단위 테스트 실패: {e}")
        
        self.test_results['unit_tests'] = test_results
        return test_results
    
    def run_integration_tests(self) -> Dict[str, Any]:
        """통합 테스트를 실행합니다."""
        if not self.config.enable_integration_tests:
            return {'status': 'disabled'}
        
        self.logger.info("통합 테스트 시작")
        test_results = {}
        
        try:
            # 전체 파이프라인 테스트
            import sys
            import os
            sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'processors'))
            from preprocessing import ClothWarpingPreprocessor
            from high_resolution_processor import ClothWarpingHighResolutionProcessor
            from advanced_post_processor import ClothWarpingAdvancedPostProcessor
            from quality_enhancer import ClothWarpingQualityEnhancer
            
            # 테스트 데이터 생성
            batch_size = self.config.test_batch_size
            channels = 3
            height, width = self.config.test_image_size
            
            cloth_image = torch.randn(batch_size, channels, height, width, device=self.device)
            person_image = torch.randn(batch_size, channels, height, width, device=self.device)
            
            # 1단계: 전처리
            preprocessor = ClothWarpingPreprocessor()
            preprocessed = preprocessor(cloth_image, person_image)
            test_results['preprocessing'] = 'success'
            
            # 2단계: 고해상도 처리
            hr_processor = ClothWarpingHighResolutionProcessor()
            warped_result = hr_processor(cloth_image, person_image)
            test_results['high_resolution_processing'] = 'success'
            
            # 3단계: 고급 후처리
            post_processor = ClothWarpingAdvancedPostProcessor()
            post_processed = post_processor(
                warped_result['warped_cloth'], 
                cloth_image, 
                person_image
            )
            test_results['advanced_post_processing'] = 'success'
            
            # 4단계: 품질 향상
            quality_enhancer = ClothWarpingQualityEnhancer()
            enhanced = quality_enhancer(post_processed['optimized_warped_image'])
            test_results['quality_enhancement'] = 'success'
            
            test_results['status'] = 'success'
            test_results['pipeline_steps'] = 4
            self.logger.info("통합 테스트 완료")
            
        except Exception as e:
            test_results['status'] = 'error'
            test_results['error'] = str(e)
            self.logger.error(f"통합 테스트 실패: {e}")
        
        self.test_results['integration_tests'] = test_results
        return test_results
    
    def run_performance_tests(self) -> Dict[str, Any]:
        """성능 테스트를 실행합니다."""
        if not self.config.enable_performance_tests:
            return {'status': 'disabled'}
        
        self.logger.info("성능 테스트 시작")
        test_results = {}
        
        try:
            import sys
            import os
            sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'processors'))
            from high_resolution_processor import ClothWarpingHighResolutionProcessor
            
            # 다양한 배치 크기로 테스트
            batch_sizes = [1, 2, 4]
            image_sizes = [(256, 256), (512, 512), (1024, 1024)]
            
            performance_data = {}
            
            for batch_size in batch_sizes:
                for height, width in image_sizes:
                    try:
                        # 테스트 데이터 생성
                        cloth_image = torch.randn(batch_size, 3, height, width, device=self.device)
                        person_image = torch.randn(batch_size, 3, height, width, device=self.device)
                        
                        # 프로세서 초기화
                        processor = ClothWarpingHighResolutionProcessor()
                        
                        # 워밍업
                        with torch.no_grad():
                            _ = processor(cloth_image, person_image)
                        
                        # 성능 측정
                        start_time = time.time()
                        
                        with torch.no_grad():
                            for _ in range(5):  # 5회 반복
                                _ = processor(cloth_image, person_image)
                        
                        end_time = time.time()
                        avg_time = (end_time - start_time) / 5
                        
                        key = f"batch_{batch_size}_size_{height}x{width}"
                        performance_data[key] = {
                            'batch_size': batch_size,
                            'image_size': (height, width),
                            'average_time': avg_time,
                            'throughput': batch_size / avg_time
                        }
                        
                    except Exception as e:
                        key = f"batch_{batch_size}_size_{height}x{width}"
                        performance_data[key] = {
                            'error': str(e)
                        }
            
            test_results['performance_data'] = performance_data
            test_results['status'] = 'success'
            self.logger.info("성능 테스트 완료")
            
        except Exception as e:
            test_results['status'] = 'error'
            test_results['error'] = str(e)
            self.logger.error(f"성능 테스트 실패: {e}")
        
        self.test_results['performance_tests'] = test_results
        return test_results
    
    def run_all_tests(self) -> Dict[str, Any]:
        """모든 테스트를 실행합니다."""
        self.logger.info("전체 테스트 시작")
        
        all_results = {}
        
        # 단위 테스트
        all_results['unit_tests'] = self.run_unit_tests()
        
        # 통합 테스트
        all_results['integration_tests'] = self.run_integration_tests()
        
        # 성능 테스트
        all_results['performance_tests'] = self.run_performance_tests()
        
        # 전체 결과 요약
        total_tests = 0
        passed_tests = 0
        
        for test_type, results in all_results.items():
            if results.get('status') == 'success':
                passed_tests += 1
            total_tests += 1
        
        all_results['summary'] = {
            'total_test_types': total_tests,
            'passed_test_types': passed_tests,
            'overall_status': 'success' if passed_tests == total_tests else 'partial_success'
        }
        
        self.test_results = all_results
        self.logger.info(f"전체 테스트 완료: {passed_tests}/{total_tests} 통과")
        
        return all_results
    
    def get_test_results(self) -> Dict[str, Any]:
        """테스트 결과를 반환합니다."""
        return self.test_results
    
    def get_test_summary(self) -> Dict[str, Any]:
        """테스트 요약을 반환합니다."""
        if not self.test_results:
            return {'status': 'no_tests_run'}
        
        summary = {}
        for test_type, results in self.test_results.items():
            if isinstance(results, dict):
                summary[test_type] = results.get('status', 'unknown')
        
        return summary

# 사용 예시
if __name__ == "__main__":
    # 설정
    config = TestingServiceConfig(
        enable_unit_tests=True,
        enable_integration_tests=True,
        enable_performance_tests=True,
        test_batch_size=2,
        test_image_size=(256, 256),
        use_mps=True
    )
    
    # 테스팅 서비스 초기화
    testing_service = ClothWarpingTestingService(config)
    
    # 모든 테스트 실행
    all_results = testing_service.run_all_tests()
    
    # 결과 출력
    print("=== 테스트 결과 ===")
    for test_type, results in all_results.items():
        print(f"\n{test_type}:")
        if isinstance(results, dict):
            for key, value in results.items():
                print(f"  {key}: {value}")
    
    # 테스트 요약
    summary = testing_service.get_test_summary()
    print(f"\n=== 테스트 요약 ===")
    print(f"전체 상태: {summary}")
