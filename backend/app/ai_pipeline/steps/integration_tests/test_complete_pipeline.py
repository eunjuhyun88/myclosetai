"""
🔥 완전한 AI 파이프라인 통합 테스트
====================================

모든 Step의 전처리 및 시각화 시스템을 테스트:
1. Step 01: Human Parsing
2. Step 02: Pose Estimation  
3. Step 03: Cloth Segmentation

Author: MyCloset AI Team
Date: 2025-01-27
Version: 1.0 (완전 구현)
"""

import sys
import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Dict, Any, List
import logging
import time
from pathlib import Path

# 상위 디렉토리 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Step별 모듈 import
from step_01_human_parsing_models.preprocessing.preprocessor import HumanParsingPreprocessor
from step_01_human_parsing_models.visualizers.human_parsing_visualizer import HumanParsingVisualizer

from step_02_pose_estimation_models.preprocessing.pose_estimation_preprocessor import PoseEstimationPreprocessor
from step_02_pose_estimation_models.visualizers.pose_estimation_visualizer import PoseEstimationVisualizer

from step_03_cloth_segmentation_models.preprocessing.cloth_segmentation_preprocessor import ClothSegmentationPreprocessor

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CompletePipelineTester:
    """완전한 AI 파이프라인 테스터"""
    
    def __init__(self, test_image_path: str = None):
        self.test_image_path = test_image_path
        self.logger = logging.getLogger(f"{__name__}.CompletePipelineTester")
        
        # 결과 저장 디렉토리
        self.results_dir = Path("./pipeline_test_results")
        self.results_dir.mkdir(exist_ok=True)
        
        # 테스트 통계
        self.test_stats = {
            'total_tests': 0,
            'successful_tests': 0,
            'failed_tests': 0,
            'processing_times': []
        }
        
        # Step별 전처리기 초기화
        self.preprocessors = {}
        self.visualizers = {}
        self._initialize_components()
    
    def _initialize_components(self):
        """Step별 컴포넌트 초기화"""
        try:
            self.logger.info("🔥 Step별 컴포넌트 초기화 시작")
            
            # Step 01: Human Parsing
            self.preprocessors['human_parsing'] = HumanParsingPreprocessor(target_size=(512, 512))
            self.visualizers['human_parsing'] = HumanParsingVisualizer(
                save_dir=str(self.results_dir / "step_01_human_parsing")
            )
            
            # Step 02: Pose Estimation
            self.preprocessors['pose_estimation'] = PoseEstimationPreprocessor(target_size=(368, 368))
            self.visualizers['pose_estimation'] = PoseEstimationVisualizer(
                save_dir=str(self.results_dir / "step_02_pose_estimation")
            )
            
            # Step 03: Cloth Segmentation
            self.preprocessors['cloth_segmentation'] = ClothSegmentationPreprocessor(target_size=(512, 512))
            
            self.logger.info("✅ 모든 컴포넌트 초기화 완료")
            
        except Exception as e:
            self.logger.error(f"❌ 컴포넌트 초기화 실패: {e}")
            raise
    
    def create_test_image(self, size: tuple = (640, 480)) -> np.ndarray:
        """테스트용 이미지 생성"""
        try:
            # 그라데이션 배경
            image = np.zeros((size[1], size[0], 3), dtype=np.uint8)
            
            # 수직 그라데이션
            for y in range(size[1]):
                for x in range(size[0]):
                    # 인체 모양 시뮬레이션
                    center_x = size[0] // 2
                    center_y = size[1] // 2
                    
                    # 머리 (원)
                    head_radius = 50
                    if (x - center_x)**2 + (y - center_y + 100)**2 < head_radius**2:
                        image[y, x] = [255, 200, 200]  # 피부색
                    # 몸통 (직사각형)
                    elif (abs(x - center_x) < 80 and 
                          center_y - 50 < y < center_y + 150):
                        image[y, x] = [100, 150, 255]  # 파란색 옷
                    # 팔
                    elif ((abs(x - (center_x - 100)) < 30 and center_y - 30 < y < center_y + 100) or
                          (abs(x - (center_x + 100)) < 30 and center_y - 30 < y < center_y + 100)):
                        image[y, x] = [255, 200, 200]  # 피부색
                    # 다리
                    elif (abs(x - center_x) < 60 and center_y + 150 < y < center_y + 250):
                        image[y, x] = [100, 100, 100]  # 검은색 바지
                    # 배경 그라데이션
                    else:
                        intensity = int(128 + 64 * np.sin(x / 100) + 64 * np.cos(y / 100))
                        image[y, x] = [intensity, intensity, intensity]
            
            return image
            
        except Exception as e:
            self.logger.error(f"테스트 이미지 생성 실패: {e}")
            return np.random.randint(0, 255, (size[1], size[0], 3), dtype=np.uint8)
    
    def test_human_parsing_pipeline(self, test_image: np.ndarray) -> Dict[str, Any]:
        """Step 01: Human Parsing 파이프라인 테스트"""
        try:
            self.logger.info("🔥 Step 01: Human Parsing 파이프라인 테스트 시작")
            start_time = time.time()
            
            # 1. 전처리 테스트
            preprocessor = self.preprocessors['human_parsing']
            preprocessing_result = preprocessor.preprocess_image(test_image, mode='advanced')
            
            if not preprocessing_result['success']:
                raise Exception(f"전처리 실패: {preprocessing_result.get('error', 'Unknown error')}")
            
            # 2. 시각화 테스트
            visualizer = self.visualizers['human_parsing']
            
            # 전처리 파이프라인 시각화
            preprocessing_viz_path = visualizer.visualize_preprocessing_pipeline(
                test_image, preprocessing_result
            )
            
            # 가상의 파싱 결과 생성 (실제 모델 대신)
            fake_parsing_mask = np.random.rand(20, 512, 512)  # 20개 클래스
            fake_parsing_mask = (fake_parsing_mask > 0.5).astype(np.float32)
            
            # 파싱 결과 시각화
            parsing_viz_path = visualizer.visualize_parsing_result(
                test_image, fake_parsing_mask, confidence=0.85
            )
            
            # 전체 파이프라인 비교 시각화
            fake_parsing_result = {
                'parsing_mask': fake_parsing_mask,
                'quality_metrics': {
                    'overall_score': 0.85,
                    'boundary_quality': 0.82,
                    'segmentation_quality': 0.88
                }
            }
            
            comparison_viz_path = visualizer.create_comparison_visualization(
                test_image, preprocessing_result, fake_parsing_result
            )
            
            processing_time = time.time() - start_time
            
            result = {
                'success': True,
                'step': 'human_parsing',
                'preprocessing_result': preprocessing_result,
                'visualization_paths': {
                    'preprocessing': preprocessing_viz_path,
                    'parsing_result': parsing_viz_path,
                    'comparison': comparison_viz_path
                },
                'processing_time': processing_time,
                'stats': preprocessor.get_processing_stats()
            }
            
            self.logger.info(f"✅ Step 01 테스트 완료 (소요시간: {processing_time:.2f}초)")
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Step 01 테스트 실패: {e}")
            return {
                'success': False,
                'step': 'human_parsing',
                'error': str(e),
                'processing_time': 0
            }
    
    def test_pose_estimation_pipeline(self, test_image: np.ndarray) -> Dict[str, Any]:
        """Step 02: Pose Estimation 파이프라인 테스트"""
        try:
            self.logger.info("🔥 Step 02: Pose Estimation 파이프라인 테스트 시작")
            start_time = time.time()
            
            # 1. 전처리 테스트
            preprocessor = self.preprocessors['pose_estimation']
            preprocessing_result = preprocessor.preprocess_image(test_image, mode='advanced')
            
            if not preprocessing_result['success']:
                raise Exception(f"전처리 실패: {preprocessing_result.get('error', 'Unknown error')}")
            
            # 2. 시각화 테스트
            visualizer = self.visualizers['pose_estimation']
            
            # 전처리 파이프라인 시각화
            preprocessing_viz_path = visualizer.visualize_preprocessing_pipeline(
                test_image, preprocessing_result
            )
            
            # 가상의 포즈 결과 생성 (실제 모델 대신)
            fake_keypoints = np.random.rand(17, 2) * 368  # 17개 키포인트, 368x368 이미지
            fake_confidence_scores = np.random.rand(17) * 0.5 + 0.5  # 0.5~1.0 신뢰도
            
            # 포즈 결과 시각화
            pose_viz_path = visualizer.visualize_pose_result(
                test_image, fake_keypoints, fake_confidence_scores
            )
            
            # 전체 파이프라인 비교 시각화
            fake_pose_result = {
                'keypoints': fake_keypoints,
                'confidence_scores': fake_confidence_scores
            }
            
            comparison_viz_path = visualizer.create_comparison_visualization(
                test_image, preprocessing_result, fake_pose_result
            )
            
            processing_time = time.time() - start_time
            
            result = {
                'success': True,
                'step': 'pose_estimation',
                'preprocessing_result': preprocessing_result,
                'visualization_paths': {
                    'preprocessing': preprocessing_viz_path,
                    'pose_result': pose_viz_path,
                    'comparison': comparison_viz_path
                },
                'processing_time': processing_time,
                'stats': preprocessor.get_processing_stats()
            }
            
            self.logger.info(f"✅ Step 02 테스트 완료 (소요시간: {processing_time:.2f}초)")
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Step 02 테스트 실패: {e}")
            return {
                'success': False,
                'step': 'pose_estimation',
                'error': str(e),
                'processing_time': 0
            }
    
    def test_cloth_segmentation_pipeline(self, test_image: np.ndarray) -> Dict[str, Any]:
        """Step 03: Cloth Segmentation 파이프라인 테스트"""
        try:
            self.logger.info("🔥 Step 03: Cloth Segmentation 파이프라인 테스트 시작")
            start_time = time.time()
            
            # 1. 전처리 테스트
            preprocessor = self.preprocessors['cloth_segmentation']
            preprocessing_result = preprocessor.preprocess_image(test_image, mode='advanced')
            
            if not preprocessing_result['success']:
                raise Exception(f"전처리 실패: {preprocessing_result.get('error', 'Unknown error')}")
            
            processing_time = time.time() - start_time
            
            result = {
                'success': True,
                'step': 'cloth_segmentation',
                'preprocessing_result': preprocessing_result,
                'processing_time': processing_time,
                'stats': preprocessor.get_processing_stats()
            }
            
            self.logger.info(f"✅ Step 03 테스트 완료 (소요시간: {processing_time:.2f}초)")
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Step 03 테스트 실패: {e}")
            return {
                'success': False,
                'step': 'cloth_segmentation',
                'error': str(e),
                'processing_time': 0
            }
    
    def run_complete_pipeline_test(self) -> Dict[str, Any]:
        """완전한 파이프라인 테스트 실행"""
        try:
            self.logger.info("🚀 완전한 AI 파이프라인 테스트 시작")
            overall_start_time = time.time()
            
            # 테스트 이미지 생성 또는 로드
            if self.test_image_path and os.path.exists(self.test_image_path):
                test_image = cv2.imread(self.test_image_path)
                test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
                self.logger.info(f"테스트 이미지 로드: {self.test_image_path}")
            else:
                test_image = self.create_test_image()
                self.logger.info("테스트 이미지 생성 완료")
            
            # 각 Step별 테스트 실행
            test_results = {}
            
            # Step 01: Human Parsing
            test_results['human_parsing'] = self.test_human_parsing_pipeline(test_image)
            
            # Step 02: Pose Estimation
            test_results['pose_estimation'] = self.test_pose_estimation_pipeline(test_image)
            
            # Step 03: Cloth Segmentation
            test_results['cloth_segmentation'] = self.test_cloth_segmentation_pipeline(test_image)
            
            # 전체 통계 계산
            overall_processing_time = time.time() - overall_start_time
            successful_steps = sum(1 for result in test_results.values() if result['success'])
            total_steps = len(test_results)
            
            # 테스트 통계 업데이트
            self.test_stats['total_tests'] = total_steps
            self.test_stats['successful_tests'] = successful_steps
            self.test_stats['failed_tests'] = total_steps - successful_steps
            self.test_stats['processing_times'].extend([
                result.get('processing_time', 0) for result in test_results.values()
            ])
            
            # 최종 결과 요약
            final_result = {
                'overall_success': successful_steps == total_steps,
                'total_steps': total_steps,
                'successful_steps': successful_steps,
                'failed_steps': total_steps - successful_steps,
                'overall_processing_time': overall_processing_time,
                'step_results': test_results,
                'test_stats': self.test_stats.copy()
            }
            
            # 결과 요약 출력
            self._print_test_summary(final_result)
            
            self.logger.info("🎉 완전한 AI 파이프라인 테스트 완료")
            return final_result
            
        except Exception as e:
            self.logger.error(f"❌ 완전한 파이프라인 테스트 실패: {e}")
            return {
                'overall_success': False,
                'error': str(e),
                'total_steps': 0,
                'successful_steps': 0,
                'failed_steps': 0
            }
    
    def _print_test_summary(self, final_result: Dict[str, Any]):
        """테스트 결과 요약 출력"""
        try:
            print("\n" + "="*80)
            print("🎯 완전한 AI 파이프라인 테스트 결과 요약")
            print("="*80)
            
            print(f"📊 전체 결과: {'✅ 성공' if final_result['overall_success'] else '❌ 실패'}")
            print(f"🔢 총 Step 수: {final_result['total_steps']}")
            print(f"✅ 성공한 Step: {final_result['successful_steps']}")
            print(f"❌ 실패한 Step: {final_result['failed_steps']}")
            print(f"⏱️  전체 처리 시간: {final_result['overall_processing_time']:.2f}초")
            
            print("\n📋 Step별 상세 결과:")
            print("-" * 60)
            
            for step_name, step_result in final_result['step_results'].items():
                status = "✅ 성공" if step_result['success'] else "❌ 실패"
                processing_time = step_result.get('processing_time', 0)
                
                print(f"{step_name:20s}: {status:10s} ({processing_time:.2f}초)")
                
                if step_result['success'] and 'stats' in step_result:
                    stats = step_result['stats']
                    print(f"{'':20s}  📈 처리 통계: {stats}")
            
            print("\n📁 결과 파일 위치:")
            print(f"  📂 전체 결과: {self.results_dir}")
            
            # 시각화 파일 경로 출력
            for step_name, step_result in final_result['step_results'].items():
                if step_result['success'] and 'visualization_paths' in step_result:
                    viz_paths = step_result['visualization_paths']
                    print(f"  📂 {step_name}:")
                    for viz_type, viz_path in viz_paths.items():
                        if viz_path:
                            print(f"    - {viz_type}: {viz_path}")
            
            print("="*80)
            
        except Exception as e:
            self.logger.error(f"테스트 요약 출력 실패: {e}")

def main():
    """메인 실행 함수"""
    try:
        logger.info("🚀 완전한 AI 파이프라인 테스트 시작")
        
        # 테스터 초기화
        tester = CompletePipelineTester()
        
        # 완전한 파이프라인 테스트 실행
        result = tester.run_complete_pipeline_test()
        
        if result['overall_success']:
            logger.info("🎉 모든 테스트가 성공적으로 완료되었습니다!")
            return 0
        else:
            logger.error(f"❌ 일부 테스트가 실패했습니다. 실패한 Step: {result['failed_steps']}")
            return 1
            
    except Exception as e:
        logger.error(f"❌ 테스트 실행 중 오류 발생: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
