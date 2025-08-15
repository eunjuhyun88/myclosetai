"""
🚀 확장된 AI 파이프라인 통합 테스트 시스템
============================================

새로 구현된 Step 04, 05, 06을 포함한 완전한 파이프라인 테스트:
1. Step 04: Geometric Matching (기하학적 매칭)
2. Step 05: Cloth Warping (의류 변형)
3. Step 06: Virtual Fitting (가상 피팅)

Author: MyCloset AI Team
Date: 2025-01-27
Version: 2.0 (확장 구현)
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import logging
import os
from pathlib import Path
from typing import Dict, Any, List, Tuple
import sys

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

# 새로운 전처리 시스템 import
try:
    import sys
    import os
    
    # 현재 디렉토리를 Python 경로에 추가
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    sys.path.insert(0, parent_dir)
    
    from step_04_geometric_matching_models.preprocessing.geometric_matching_preprocessor import GeometricMatchingPreprocessor
    from step_05_cloth_warping_models.preprocessing.cloth_warping_preprocessor import ClothWarpingPreprocessor
    from step_06_virtual_fitting_models.preprocessing.virtual_fitting_preprocessor import VirtualFittingPreprocessor
    
    # 기존 전처리 시스템 import
    from step_01_human_parsing_models.preprocessing.preprocessor import HumanParsingPreprocessor
    from step_02_pose_estimation_models.preprocessing.pose_estimation_preprocessor import PoseEstimationPreprocessor
    from step_03_cloth_segmentation_models.preprocessing.cloth_segmentation_preprocessor import ClothSegmentationPreprocessor
    
    # 새로운 시각화 시스템 import
    from step_04_geometric_matching_models.visualizers.geometric_matching_visualizer import GeometricMatchingVisualizer
    
    EXTENDED_IMPORTS_SUCCESS = True
    logger.info("✅ 모든 확장 모듈 import 성공")
    
except ImportError as e:
    EXTENDED_IMPORTS_SUCCESS = False
    logger.warning(f"⚠️ 일부 확장 모듈 import 실패: {e}")
    logger.warning("기본 모듈만으로 테스트를 진행합니다.")
    
    # 기본 모듈만 import 시도
    try:
        from step_01_human_parsing_models.preprocessing.preprocessor import HumanParsingPreprocessor
        from step_02_pose_estimation_models.preprocessing.pose_estimation_preprocessor import PoseEstimationPreprocessor
        from step_03_cloth_segmentation_models.preprocessing.cloth_segmentation_preprocessor import ClothSegmentationPreprocessor
        BASIC_IMPORTS_SUCCESS = True
        logger.info("✅ 기본 모듈 import 성공")
    except ImportError as basic_e:
        BASIC_IMPORTS_SUCCESS = False
        logger.error(f"❌ 기본 모듈 import도 실패: {basic_e}")

class ExtendedPipelineTester:
    """확장된 AI 파이프라인 테스트 시스템"""
    
    def __init__(self, output_dir: str = "./extended_pipeline_test_results"):
        self.output_dir = output_dir
        self.logger = logging.getLogger(f"{__name__}.ExtendedPipelineTester")
        
        # 출력 디렉토리 생성
        os.makedirs(output_dir, exist_ok=True)
        
        # 테스트 결과 저장
        self.test_results = {
            'total_steps': 6,
            'successful_steps': 0,
            'failed_steps': 0,
            'processing_times': {},
            'step_results': {}
        }
        
        # 전처리 시스템 초기화
        self._initialize_preprocessors()
        
        # 시각화 시스템 초기화
        self._initialize_visualizers()
    
    def _initialize_preprocessors(self):
        """전처리 시스템 초기화"""
        try:
            # 기존 전처리 시스템
            self.human_parsing_preprocessor = HumanParsingPreprocessor()
            self.pose_estimation_preprocessor = PoseEstimationPreprocessor()
            self.cloth_segmentation_preprocessor = ClothSegmentationPreprocessor()
            logger.info("✅ 기본 전처리 시스템 초기화 완료")
            
            # 새로운 전처리 시스템
            if EXTENDED_IMPORTS_SUCCESS:
                self.geometric_matching_preprocessor = GeometricMatchingPreprocessor()
                self.cloth_warping_preprocessor = ClothWarpingPreprocessor()
                self.virtual_fitting_preprocessor = VirtualFittingPreprocessor()
                logger.info("✅ 모든 전처리 시스템 초기화 완료")
            else:
                logger.warning("⚠️ 확장 전처리 시스템은 초기화되지 않음")
                
        except Exception as e:
            self.logger.error(f"❌ 전처리 시스템 초기화 실패: {e}")
            # 기본 전처리 시스템도 초기화 실패 시 속성 설정
            if not hasattr(self, 'human_parsing_preprocessor'):
                self.human_parsing_preprocessor = None
            if not hasattr(self, 'pose_estimation_preprocessor'):
                self.pose_estimation_preprocessor = None
            if not hasattr(self, 'cloth_segmentation_preprocessor'):
                self.cloth_segmentation_preprocessor = None
    
    def _initialize_visualizers(self):
        """시각화 시스템 초기화"""
        try:
            # 새로운 시각화 시스템
            if EXTENDED_IMPORTS_SUCCESS:
                self.geometric_matching_visualizer = GeometricMatchingVisualizer(
                    save_dir=f"{self.output_dir}/step_04_geometric_matching"
                )
                logger.info("✅ 시각화 시스템 초기화 완료")
            else:
                logger.warning("⚠️ 시각화 시스템 초기화 실패")
                
        except Exception as e:
            self.logger.error(f"❌ 시각화 시스템 초기화 실패: {e}")
    
    def create_test_image(self, size: Tuple[int, int] = (512, 512)) -> np.ndarray:
        """테스트용 더미 이미지 생성"""
        try:
            # 기본 이미지 생성
            image = np.zeros((size[0], size[1], 3), dtype=np.uint8)
            
            # 그라데이션 배경
            for i in range(size[0]):
                for j in range(size[1]):
                    image[i, j] = [
                        int(255 * i / size[0]),  # R
                        int(255 * j / size[1]),  # G
                        int(255 * (i + j) / (size[0] + size[1]))  # B
                    ]
            
            # 원형 인체 모양 추가
            center = (size[1] // 2, size[0] // 2)
            radius = min(size) // 4
            cv2.circle(image, center, radius, (100, 150, 200), -1)
            
            # 의류 영역 추가
            clothing_center = (center[0] + radius//2, center[1] + radius//2)
            clothing_radius = radius // 2
            cv2.circle(image, clothing_center, clothing_radius, (200, 100, 150), -1)
            
            # 텍스처 패턴 추가
            for i in range(0, size[0], 20):
                cv2.line(image, (0, i), (size[1], i), (50, 50, 50), 1)
            for j in range(0, size[1], 20):
                cv2.line(image, (j, 0), (j, size[0]), (50, 50, 50), 1)
            
            self.logger.info(f"✅ 테스트 이미지 생성 완료: {size[1]}x{size[0]}")
            return image
            
        except Exception as e:
            self.logger.error(f"❌ 테스트 이미지 생성 실패: {e}")
            return np.zeros((size[0], size[1], 3), dtype=np.uint8)
    
    def test_human_parsing_pipeline(self, test_image: np.ndarray) -> Dict[str, Any]:
        """Step 01: Human Parsing 파이프라인 테스트"""
        try:
            if self.human_parsing_preprocessor is None:
                raise RuntimeError("Human Parsing 전처리 시스템이 초기화되지 않았습니다")
            
            start_time = time.time()
            self.logger.info("🔥 Step 01: Human Parsing 파이프라인 테스트 시작")
            
            # 전처리 실행
            preprocessing_result = self.human_parsing_preprocessor.preprocess_image(test_image, mode='advanced')
            
            # 처리 시간 계산
            processing_time = time.time() - start_time
            
            # 결과 저장
            result = {
                'step_name': 'human_parsing',
                'success': preprocessing_result['success'],
                'processing_time': processing_time,
                'preprocessing_result': preprocessing_result,
                'processing_stats': self.human_parsing_preprocessor.get_processing_stats()
            }
            
            if preprocessing_result['success']:
                self.test_results['successful_steps'] += 1
                self.logger.info(f"✅ Step 01: Human Parsing 성공 ({processing_time:.2f}초)")
            else:
                self.test_results['failed_steps'] += 1
                self.logger.error(f"❌ Step 01: Human Parsing 실패")
            
            # 결과 저장
            self.test_results['step_results']['human_parsing'] = result
            self.test_results['processing_times']['human_parsing'] = processing_time
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Step 01: Human Parsing 테스트 실패: {e}")
            self.test_results['failed_steps'] += 1
            return {
                'step_name': 'human_parsing',
                'success': False,
                'error': str(e),
                'processing_time': 0
            }
    
    def test_pose_estimation_pipeline(self, test_image: np.ndarray) -> Dict[str, Any]:
        """Step 02: Pose Estimation 파이프라인 테스트"""
        try:
            if self.pose_estimation_preprocessor is None:
                raise RuntimeError("Pose Estimation 전처리 시스템이 초기화되지 않았습니다")
            
            start_time = time.time()
            self.logger.info("🔥 Step 02: Pose Estimation 파이프라인 테스트 시작")
            
            # 전처리 실행
            preprocessing_result = self.pose_estimation_preprocessor.preprocess_image(test_image, mode='advanced')
            
            # 처리 시간 계산
            processing_time = time.time() - start_time
            
            # 결과 저장
            result = {
                'step_name': 'pose_estimation',
                'success': preprocessing_result['success'],
                'processing_time': processing_time,
                'preprocessing_result': preprocessing_result,
                'processing_stats': self.pose_estimation_preprocessor.get_processing_stats()
            }
            
            if preprocessing_result['success']:
                self.test_results['successful_steps'] += 1
                self.logger.info(f"✅ Step 02: Pose Estimation 성공 ({processing_time:.2f}초)")
            else:
                self.test_results['failed_steps'] += 1
                self.logger.error(f"❌ Step 02: Pose Estimation 실패")
            
            # 결과 저장
            self.test_results['step_results']['pose_estimation'] = result
            self.test_results['processing_times']['pose_estimation'] = processing_time
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Step 02: Pose Estimation 테스트 실패: {e}")
            self.test_results['failed_steps'] += 1
            return {
                'step_name': 'pose_estimation',
                'success': False,
                'error': str(e),
                'processing_time': 0
            }
    
    def test_cloth_segmentation_pipeline(self, test_image: np.ndarray) -> Dict[str, Any]:
        """Step 03: Cloth Segmentation 파이프라인 테스트"""
        try:
            if self.cloth_segmentation_preprocessor is None:
                raise RuntimeError("Cloth Segmentation 전처리 시스템이 초기화되지 않았습니다")
            
            start_time = time.time()
            self.logger.info("🔥 Step 03: Cloth Segmentation 파이프라인 테스트 시작")
            
            # 전처리 실행
            preprocessing_result = self.cloth_segmentation_preprocessor.preprocess_image(test_image, mode='advanced')
            
            # 처리 시간 계산
            processing_time = time.time() - start_time
            
            # 결과 저장
            result = {
                'step_name': 'cloth_segmentation',
                'success': preprocessing_result['success'],
                'processing_time': processing_time,
                'preprocessing_result': preprocessing_result,
                'processing_stats': self.cloth_segmentation_preprocessor.get_processing_stats()
            }
            
            if preprocessing_result['success']:
                self.test_results['successful_steps'] += 1
                self.logger.info(f"✅ Step 03: Cloth Segmentation 성공 ({processing_time:.2f}초)")
            else:
                self.test_results['failed_steps'] += 1
                self.logger.error(f"❌ Step 03: Cloth Segmentation 실패")
            
            # 결과 저장
            self.test_results['step_results']['cloth_segmentation'] = result
            self.test_results['processing_times']['cloth_segmentation'] = processing_time
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Step 03: Cloth Segmentation 테스트 실패: {e}")
            self.test_results['failed_steps'] += 1
            return {
                'step_name': 'cloth_segmentation',
                'success': False,
                'error': str(e),
                'processing_time': 0
            }
    
    def test_geometric_matching_pipeline(self, test_image: np.ndarray) -> Dict[str, Any]:
        """Step 04: Geometric Matching 파이프라인 테스트"""
        try:
            if not EXTENDED_IMPORTS_SUCCESS:
                raise ImportError("확장 모듈을 사용할 수 없습니다")
            
            start_time = time.time()
            self.logger.info("🔥 Step 04: Geometric Matching 파이프라인 테스트 시작")
            
            # 전처리 실행
            preprocessing_result = self.geometric_matching_preprocessor.preprocess_image(test_image, mode='advanced')
            
            # 시각화 실행
            visualization_path = self.geometric_matching_visualizer.visualize_preprocessing_pipeline(
                test_image, preprocessing_result
            )
            
            # 처리 시간 계산
            processing_time = time.time() - start_time
            
            # 결과 저장
            result = {
                'step_name': 'geometric_matching',
                'success': preprocessing_result['success'],
                'processing_time': processing_time,
                'preprocessing_result': preprocessing_result,
                'visualization_path': visualization_path,
                'processing_stats': self.geometric_matching_preprocessor.get_processing_stats()
            }
            
            if preprocessing_result['success']:
                self.test_results['successful_steps'] += 1
                self.logger.info(f"✅ Step 04: Geometric Matching 성공 ({processing_time:.2f}초)")
            else:
                self.test_results['failed_steps'] += 1
                self.logger.error(f"❌ Step 04: Geometric Matching 실패")
            
            # 결과 저장
            self.test_results['step_results']['geometric_matching'] = result
            self.test_results['processing_times']['geometric_matching'] = processing_time
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Step 04: Geometric Matching 테스트 실패: {e}")
            self.test_results['failed_steps'] += 1
            return {
                'step_name': 'geometric_matching',
                'success': False,
                'error': str(e),
                'processing_time': 0
            }
    
    def test_cloth_warping_pipeline(self, test_image: np.ndarray) -> Dict[str, Any]:
        """Step 05: Cloth Warping 파이프라인 테스트"""
        try:
            if not EXTENDED_IMPORTS_SUCCESS:
                raise ImportError("확장 모듈을 사용할 수 없습니다")
            
            start_time = time.time()
            self.logger.info("🔥 Step 05: Cloth Warping 파이프라인 테스트 시작")
            
            # 전처리 실행
            preprocessing_result = self.cloth_warping_preprocessor.preprocess_image(test_image, mode='advanced')
            
            # 처리 시간 계산
            processing_time = time.time() - start_time
            
            # 결과 저장
            result = {
                'step_name': 'cloth_warping',
                'success': preprocessing_result['success'],
                'processing_time': processing_time,
                'preprocessing_result': preprocessing_result,
                'processing_stats': self.cloth_warping_preprocessor.get_processing_stats()
            }
            
            if preprocessing_result['success']:
                self.test_results['successful_steps'] += 1
                self.logger.info(f"✅ Step 05: Cloth Warping 성공 ({processing_time:.2f}초)")
            else:
                self.test_results['failed_steps'] += 1
                self.logger.error(f"❌ Step 05: Cloth Warping 실패")
            
            # 결과 저장
            self.test_results['step_results']['cloth_warping'] = result
            self.test_results['processing_times']['cloth_warping'] = processing_time
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Step 05: Cloth Warping 테스트 실패: {e}")
            self.test_results['failed_steps'] += 1
            return {
                'step_name': 'cloth_warping',
                'success': False,
                'error': str(e),
                'processing_time': 0
            }
    
    def test_virtual_fitting_pipeline(self, test_image: np.ndarray) -> Dict[str, Any]:
        """Step 06: Virtual Fitting 파이프라인 테스트"""
        try:
            if not EXTENDED_IMPORTS_SUCCESS:
                raise ImportError("확장 모듈을 사용할 수 없습니다")
            
            start_time = time.time()
            self.logger.info("🔥 Step 06: Virtual Fitting 파이프라인 테스트 시작")
            
            # 전처리 실행
            preprocessing_result = self.virtual_fitting_preprocessor.preprocess_image(test_image, mode='advanced')
            
            # 처리 시간 계산
            processing_time = time.time() - start_time
            
            # 결과 저장
            result = {
                'step_name': 'virtual_fitting',
                'success': preprocessing_result['success'],
                'processing_time': processing_time,
                'preprocessing_result': preprocessing_result,
                'processing_stats': self.virtual_fitting_preprocessor.get_processing_stats()
            }
            
            if preprocessing_result['success']:
                self.test_results['successful_steps'] += 1
                self.logger.info(f"✅ Step 06: Virtual Fitting 성공 ({processing_time:.2f}초)")
            else:
                self.test_results['failed_steps'] += 1
                self.logger.error(f"❌ Step 06: Virtual Fitting 실패")
            
            # 결과 저장
            self.test_results['step_results']['virtual_fitting'] = result
            self.test_results['processing_times']['virtual_fitting'] = processing_time
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Step 06: Virtual Fitting 테스트 실패: {e}")
            self.test_results['failed_steps'] += 1
            return {
                'step_name': 'virtual_fitting',
                'success': False,
                'error': str(e),
                'processing_time': 0
            }
    
    def run_extended_pipeline_test(self) -> Dict[str, Any]:
        """확장된 파이프라인 전체 테스트 실행"""
        try:
            self.logger.info("🚀 확장된 AI 파이프라인 전체 테스트 시작")
            overall_start_time = time.time()
            
            # 테스트 이미지 생성
            test_image = self.create_test_image()
            
            # 각 Step 테스트 실행
            step_results = []
            
            # Step 01-03 (기존)
            step_results.append(self.test_human_parsing_pipeline(test_image))
            step_results.append(self.test_pose_estimation_pipeline(test_image))
            step_results.append(self.test_cloth_segmentation_pipeline(test_image))
            
            # Step 04-06 (새로 구현)
            if EXTENDED_IMPORTS_SUCCESS:
                step_results.append(self.test_geometric_matching_pipeline(test_image))
                step_results.append(self.test_cloth_warping_pipeline(test_image))
                step_results.append(self.test_virtual_fitting_pipeline(test_image))
            else:
                self.logger.warning("⚠️ Step 04-06은 확장 모듈이 없어 테스트를 건너뜁니다")
                self.test_results['total_steps'] = 3
            
            # 전체 처리 시간 계산
            overall_processing_time = time.time() - overall_start_time
            
            # 최종 결과 요약
            final_result = {
                'overall_success': self.test_results['successful_steps'] == self.test_results['total_steps'],
                'total_steps': self.test_results['total_steps'],
                'successful_steps': self.test_results['successful_steps'],
                'failed_steps': self.test_results['failed_steps'],
                'overall_processing_time': overall_processing_time,
                'step_results': step_results,
                'test_results': self.test_results
            }
            
            # 결과 출력
            self._print_test_summary(final_result)
            
            return final_result
            
        except Exception as e:
            self.logger.error(f"❌ 확장된 파이프라인 테스트 실패: {e}")
            return {
                'overall_success': False,
                'error': str(e),
                'total_steps': 0,
                'successful_steps': 0,
                'failed_steps': 0
            }
    
    def _print_test_summary(self, final_result: Dict[str, Any]):
        """테스트 결과 요약 출력"""
        print("\n" + "="*80)
        print("🎯 확장된 AI 파이프라인 테스트 결과 요약")
        print("="*80)
        
        if final_result['overall_success']:
            print(f"📊 전체 결과: ✅ 성공")
        else:
            print(f"📊 전체 결과: ❌ 실패")
        
        print(f"🔢 총 Step 수: {final_result['total_steps']}")
        print(f"✅ 성공한 Step: {final_result['successful_steps']}")
        print(f"❌ 실패한 Step: {final_result['failed_steps']}")
        print(f"⏱️  전체 처리 시간: {final_result['overall_processing_time']:.2f}초")
        
        print(f"\n📋 Step별 상세 결과:")
        print("-" * 60)
        
        for step_result in final_result['step_results']:
            step_name = step_result['step_name']
            success = step_result['success']
            processing_time = step_result.get('processing_time', 0)
            stats = step_result.get('processing_stats', {})
            
            status = "✅ 성공" if success else "❌ 실패"
            print(f"{step_name:<20} : {status:<10} ({processing_time:.2f}초)")
            
            if stats:
                stats_str = ", ".join([f"{k}: {v}" for k, v in stats.items()])
                print(f"{'':<20}   📈 처리 통계: {stats_str}")
        
        print(f"\n📁 결과 파일 위치:")
        print(f"  📂 전체 결과: {self.output_dir}")
        
        if EXTENDED_IMPORTS_SUCCESS:
            print(f"  📂 geometric_matching: {self.output_dir}/step_04_geometric_matching")
        
        print("="*80)

def main():
    """메인 실행 함수"""
    try:
        # 테스터 초기화
        tester = ExtendedPipelineTester()
        
        # 확장된 파이프라인 테스트 실행
        result = tester.run_extended_pipeline_test()
        
        # 성공/실패 코드 반환
        return 0 if result['overall_success'] else 1
        
    except Exception as e:
        logger.error(f"❌ 메인 실행 실패: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
