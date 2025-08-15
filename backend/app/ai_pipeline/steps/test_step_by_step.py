#!/usr/bin/env python3
"""
🔥 Step별 AI 추론 완전성 테스트
================================

각 step 폴더별로 AI 추론이 완전히 작동하는지 테스트:
1. Step 01: Human Parsing
2. Step 02: Pose Estimation  
3. Step 03: Cloth Segmentation
4. Step 04: Geometric Matching
5. Step 05: Cloth Warping
6. Step 06: Virtual Fitting

Author: MyCloset AI Team
Date: 2025-01-27
Version: 1.0
"""

import sys
import os
import logging
import time
import traceback
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
from PIL import Image

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class StepByStepTester:
    """Step별 AI 추론 테스터"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.StepByStepTester")
        self.test_results = {}
        self.test_image = self._create_test_image()
        
        # 결과 저장 디렉토리
        self.results_dir = Path("./step_test_results")
        self.results_dir.mkdir(exist_ok=True)
        
    def _create_test_image(self) -> np.ndarray:
        """테스트용 이미지 생성"""
        # 512x512 테스트 이미지 생성
        image = np.zeros((512, 512, 3), dtype=np.uint8)
        
        # 중앙에 원형 영역 생성 (사람 모양 시뮬레이션)
        center_y, center_x = 256, 256
        radius = 150
        
        for y in range(512):
            for x in range(512):
                dist = np.sqrt((y - center_y)**2 + (x - center_x)**2)
                if dist < radius:
                    # 피부색
                    image[y, x] = [255, 200, 150]
                else:
                    # 배경색
                    image[y, x] = [100, 150, 200]
        
        return image
    
    def test_step_01_human_parsing(self) -> Dict[str, Any]:
        """Step 01: Human Parsing 테스트"""
        try:
            self.logger.info("🔍 Step 01: Human Parsing 테스트 시작")
            
            # Step import 시도
            try:
                from step_01_human_parsing_models.step_01_human_parsing import HumanParsingStep
                step_available = True
            except ImportError as e:
                self.logger.error(f"❌ HumanParsingStep import 실패: {e}")
                step_available = False
            
            if not step_available:
                return {
                    'success': False,
                    'error': 'HumanParsingStep import 실패',
                    'step_name': 'human_parsing'
                }
            
            # Step 인스턴스 생성
            start_time = time.time()
            step = HumanParsingStep()
            init_time = time.time() - start_time
            
            # 모델 상태 확인
            model_status = step.get_model_status()
            
            # 실제 추론 테스트
            inference_start = time.time()
            result = step.process(person_image=self.test_image)
            inference_time = time.time() - inference_start
            
            # 결과 검증
            success = result.get('success', False)
            has_parsing_mask = 'parsing_mask' in result
            has_confidence = 'confidence' in result
            
            test_result = {
                'success': success,
                'step_name': 'human_parsing',
                'initialization_time': init_time,
                'inference_time': inference_time,
                'model_status': model_status,
                'has_parsing_mask': has_parsing_mask,
                'has_confidence': has_confidence,
                'result_keys': list(result.keys()) if isinstance(result, dict) else [],
                'error': result.get('error', None) if not success else None
            }
            
            self.logger.info(f"✅ Step 01 테스트 완료: {success}")
            return test_result
            
        except Exception as e:
            self.logger.error(f"❌ Step 01 테스트 실패: {e}")
            return {
                'success': False,
                'error': str(e),
                'step_name': 'human_parsing',
                'traceback': traceback.format_exc()
            }
    
    def test_step_02_pose_estimation(self) -> Dict[str, Any]:
        """Step 02: Pose Estimation 테스트"""
        try:
            self.logger.info("🔍 Step 02: Pose Estimation 테스트 시작")
            
            # Step import 시도
            try:
                from step_02_pose_estimation_models.step_02_pose_estimation import PoseEstimationStep
                step_available = True
            except ImportError as e:
                self.logger.error(f"❌ PoseEstimationStep import 실패: {e}")
                step_available = False
            
            if not step_available:
                return {
                    'success': False,
                    'error': 'PoseEstimationStep import 실패',
                    'step_name': 'pose_estimation'
                }
            
            # Step 인스턴스 생성
            start_time = time.time()
            step = PoseEstimationStep()
            init_time = time.time() - start_time
            
            # 실제 추론 테스트
            inference_start = time.time()
            result = step.process(person_image=self.test_image)
            inference_time = time.time() - inference_start
            
            # 결과 검증
            success = result.get('success', False)
            has_keypoints = 'keypoints' in result or 'pose_keypoints' in result
            has_confidence = 'confidence' in result
            
            test_result = {
                'success': success,
                'step_name': 'pose_estimation',
                'initialization_time': init_time,
                'inference_time': inference_time,
                'has_keypoints': has_keypoints,
                'has_confidence': has_confidence,
                'result_keys': list(result.keys()) if isinstance(result, dict) else [],
                'error': result.get('error', None) if not success else None
            }
            
            self.logger.info(f"✅ Step 02 테스트 완료: {success}")
            return test_result
            
        except Exception as e:
            self.logger.error(f"❌ Step 02 테스트 실패: {e}")
            return {
                'success': False,
                'error': str(e),
                'step_name': 'pose_estimation',
                'traceback': traceback.format_exc()
            }
    
    def test_step_03_cloth_segmentation(self) -> Dict[str, Any]:
        """Step 03: Cloth Segmentation 테스트"""
        try:
            self.logger.info("🔍 Step 03: Cloth Segmentation 테스트 시작")
            
            # Step import 시도
            try:
                from step_03_cloth_segmentation_models.step_03_cloth_segmentation import ClothSegmentationStep
                step_available = True
            except ImportError as e:
                self.logger.error(f"❌ ClothSegmentationStep import 실패: {e}")
                step_available = False
            
            if not step_available:
                return {
                    'success': False,
                    'error': 'ClothSegmentationStep import 실패',
                    'step_name': 'cloth_segmentation'
                }
            
            # Step 인스턴스 생성
            start_time = time.time()
            step = ClothSegmentationStep()
            init_time = time.time() - start_time
            
            # 실제 추론 테스트
            inference_start = time.time()
            result = step.process(cloth_image=self.test_image)
            inference_time = time.time() - inference_start
            
            # 결과 검증
            success = result.get('success', False)
            has_mask = 'segmentation_mask' in result or 'cloth_mask' in result
            has_confidence = 'confidence' in result
            
            test_result = {
                'success': success,
                'step_name': 'cloth_segmentation',
                'initialization_time': init_time,
                'inference_time': inference_time,
                'has_mask': has_mask,
                'has_confidence': has_confidence,
                'result_keys': list(result.keys()) if isinstance(result, dict) else [],
                'error': result.get('error', None) if not success else None
            }
            
            self.logger.info(f"✅ Step 03 테스트 완료: {success}")
            return test_result
            
        except Exception as e:
            self.logger.error(f"❌ Step 03 테스트 실패: {e}")
            return {
                'success': False,
                'error': str(e),
                'step_name': 'cloth_segmentation',
                'traceback': traceback.format_exc()
            }
    
    def test_step_04_geometric_matching(self) -> Dict[str, Any]:
        """Step 04: Geometric Matching 테스트"""
        try:
            self.logger.info("🔍 Step 04: Geometric Matching 테스트 시작")
            
            # Step import 시도
            try:
                from step_04_geometric_matching_models.step_04_geometric_matching import GeometricMatchingStep
                step_available = True
            except ImportError as e:
                self.logger.error(f"❌ GeometricMatchingStep import 실패: {e}")
                step_available = False
            
            if not step_available:
                return {
                    'success': False,
                    'error': 'GeometricMatchingStep import 실패',
                    'step_name': 'geometric_matching'
                }
            
            # Step 인스턴스 생성
            start_time = time.time()
            step = GeometricMatchingStep()
            init_time = time.time() - start_time
            
            # 실제 추론 테스트
            inference_start = time.time()
            result = step.process(
                person_image=self.test_image,
                cloth_image=self.test_image
            )
            inference_time = time.time() - inference_start
            
            # 결과 검증
            success = result.get('success', False)
            has_matching = 'matching_result' in result or 'geometric_matching' in result
            has_confidence = 'confidence' in result
            
            test_result = {
                'success': success,
                'step_name': 'geometric_matching',
                'initialization_time': init_time,
                'inference_time': inference_time,
                'has_matching': has_matching,
                'has_confidence': has_confidence,
                'result_keys': list(result.keys()) if isinstance(result, dict) else [],
                'error': result.get('error', None) if not success else None
            }
            
            self.logger.info(f"✅ Step 04 테스트 완료: {success}")
            return test_result
            
        except Exception as e:
            self.logger.error(f"❌ Step 04 테스트 실패: {e}")
            return {
                'success': False,
                'error': str(e),
                'step_name': 'geometric_matching',
                'traceback': traceback.format_exc()
            }
    
    def test_step_05_cloth_warping(self) -> Dict[str, Any]:
        """Step 05: Cloth Warping 테스트"""
        try:
            self.logger.info("🔍 Step 05: Cloth Warping 테스트 시작")
            
            # Step import 시도
            try:
                from step_05_cloth_warping_models.step_05_cloth_warping import ClothWarpingStep
                step_available = True
            except ImportError as e:
                self.logger.error(f"❌ ClothWarpingStep import 실패: {e}")
                step_available = False
            
            if not step_available:
                return {
                    'success': False,
                    'error': 'ClothWarpingStep import 실패',
                    'step_name': 'cloth_warping'
                }
            
            # Step 인스턴스 생성
            start_time = time.time()
            step = ClothWarpingStep()
            init_time = time.time() - start_time
            
            # 실제 추론 테스트
            inference_start = time.time()
            result = step.process(
                cloth_image=self.test_image,
                person_image=self.test_image
            )
            inference_time = time.time() - inference_start
            
            # 결과 검증
            success = result.get('success', False)
            has_warped = 'warped_cloth' in result or 'warping_result' in result
            has_confidence = 'confidence' in result
            
            test_result = {
                'success': success,
                'step_name': 'cloth_warping',
                'initialization_time': init_time,
                'inference_time': inference_time,
                'has_warped': has_warped,
                'has_confidence': has_confidence,
                'result_keys': list(result.keys()) if isinstance(result, dict) else [],
                'error': result.get('error', None) if not success else None
            }
            
            self.logger.info(f"✅ Step 05 테스트 완료: {success}")
            return test_result
            
        except Exception as e:
            self.logger.error(f"❌ Step 05 테스트 실패: {e}")
            return {
                'success': False,
                'error': str(e),
                'step_name': 'cloth_warping',
                'traceback': traceback.format_exc()
            }
    
    def test_step_06_virtual_fitting(self) -> Dict[str, Any]:
        """Step 06: Virtual Fitting 테스트"""
        try:
            self.logger.info("🔍 Step 06: Virtual Fitting 테스트 시작")
            
            # Step import 시도
            try:
                from step_06_virtual_fitting_models.step_06_virtual_fitting import VirtualFittingStep
                step_available = True
            except ImportError as e:
                self.logger.error(f"❌ VirtualFittingStep import 실패: {e}")
                step_available = False
            
            if not step_available:
                return {
                    'success': False,
                    'error': 'VirtualFittingStep import 실패',
                    'step_name': 'virtual_fitting'
                }
            
            # Step 인스턴스 생성
            start_time = time.time()
            step = VirtualFittingStep()
            init_time = time.time() - start_time
            
            # 실제 추론 테스트
            inference_start = time.time()
            result = step.process(
                person_image=self.test_image,
                cloth_image=self.test_image
            )
            inference_time = time.time() - inference_start
            
            # 결과 검증
            success = result.get('success', False)
            has_fitting = 'fitting_result' in result or 'virtual_fitting' in result
            has_confidence = 'confidence' in result
            
            test_result = {
                'success': success,
                'step_name': 'virtual_fitting',
                'initialization_time': init_time,
                'inference_time': inference_time,
                'has_fitting': has_fitting,
                'has_confidence': has_confidence,
                'result_keys': list(result.keys()) if isinstance(result, dict) else [],
                'error': result.get('error', None) if not success else None
            }
            
            self.logger.info(f"✅ Step 06 테스트 완료: {success}")
            return test_result
            
        except Exception as e:
            self.logger.error(f"❌ Step 06 테스트 실패: {e}")
            return {
                'success': False,
                'error': str(e),
                'step_name': 'virtual_fitting',
                'traceback': traceback.format_exc()
            }
    
    def run_all_tests(self) -> Dict[str, Any]:
        """모든 step 테스트 실행"""
        self.logger.info("🚀 모든 Step 테스트 시작")
        
        # 각 step 테스트 실행
        steps_to_test = [
            self.test_step_01_human_parsing,
            self.test_step_02_pose_estimation,
            self.test_step_03_cloth_segmentation,
            self.test_step_04_geometric_matching,
            self.test_step_05_cloth_warping,
            self.test_step_06_virtual_fitting
        ]
        
        for test_func in steps_to_test:
            try:
                result = test_func()
                self.test_results[result['step_name']] = result
            except Exception as e:
                self.logger.error(f"❌ {test_func.__name__} 실행 실패: {e}")
                self.test_results[test_func.__name__] = {
                    'success': False,
                    'error': str(e),
                    'step_name': test_func.__name__.replace('test_', '')
                }
        
        # 전체 결과 요약
        total_steps = len(self.test_results)
        successful_steps = sum(1 for r in self.test_results.values() if r.get('success', False))
        
        summary = {
            'total_steps': total_steps,
            'successful_steps': successful_steps,
            'failed_steps': total_steps - successful_steps,
            'success_rate': (successful_steps / total_steps * 100) if total_steps > 0 else 0,
            'step_results': self.test_results
        }
        
        self.logger.info(f"✅ 모든 테스트 완료: {successful_steps}/{total_steps} 성공")
        return summary
    
    def print_results(self, results: Dict[str, Any]):
        """테스트 결과 출력"""
        print("\n" + "="*80)
        print("🔥 STEP별 AI 추론 완전성 테스트 결과")
        print("="*80)
        
        print(f"\n📊 전체 요약:")
        print(f"   총 Step 수: {results['total_steps']}")
        print(f"   성공한 Step: {results['successful_steps']}")
        print(f"   실패한 Step: {results['failed_steps']}")
        print(f"   성공률: {results['success_rate']:.1f}%")
        
        print(f"\n🔍 Step별 상세 결과:")
        for step_name, result in results['step_results'].items():
            status = "✅ 성공" if result.get('success', False) else "❌ 실패"
            print(f"\n   {step_name.upper()}: {status}")
            
            if result.get('success', False):
                print(f"     초기화 시간: {result.get('initialization_time', 0):.3f}초")
                print(f"     추론 시간: {result.get('inference_time', 0):.3f}초")
                print(f"     결과 키: {result.get('result_keys', [])}")
            else:
                print(f"     오류: {result.get('error', '알 수 없는 오류')}")
        
        print("\n" + "="*80)

def main():
    """메인 함수"""
    try:
        # 테스터 생성 및 테스트 실행
        tester = StepByStepTester()
        results = tester.run_all_tests()
        
        # 결과 출력
        tester.print_results(results)
        
        # 결과를 파일로 저장
        import json
        results_file = tester.results_dir / "step_test_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)
        
        print(f"\n📁 상세 결과가 저장되었습니다: {results_file}")
        
    except Exception as e:
        logger.error(f"❌ 테스트 실행 실패: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
