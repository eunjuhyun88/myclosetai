#!/usr/bin/env python3
"""
🔥 MyCloset AI - Step 03: 의류 세그멘테이션 - 모듈화된 Step 테스트
=====================================================================

분리된 모듈들을 통합한 ClothSegmentationStep의 실제 테스트

Author: MyCloset AI Team  
Date: 2025-08-01
Version: 1.0
"""

import logging
import time
import sys
import os

# 현재 디렉토리를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_test_image(width: int = 512, height: int = 512):
    """테스트용 이미지 생성"""
    try:
        import numpy as np
        import cv2
        
        # 랜덤 이미지 생성
        test_image = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        
        # 간단한 도형 그리기 (의류 시뮬레이션)
        # 원 그리기 (상의 시뮬레이션)
        cv2.circle(test_image, (width//2, height//3), 80, (255, 0, 0), -1)
        
        # 직사각형 그리기 (하의 시뮬레이션)
        cv2.rectangle(test_image, (width//4, height//2), (3*width//4, 3*height//4), (0, 255, 0), -1)
        
        logger.info(f"✅ 테스트 이미지 생성 완료: {test_image.shape}")
        return test_image
        
    except Exception as e:
        logger.error(f"❌ 테스트 이미지 생성 실패: {e}")
        # 기본 이미지 반환
        import numpy as np
        return np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)

def test_step_import():
    """Step import 테스트"""
    try:
        logger.info("🧪 Step import 테스트 시작...")
        
        from step_modularized import (
            ClothSegmentationStepModularized,
            create_cloth_segmentation_step_modularized,
            create_m3_max_segmentation_step_modularized
        )
        
        logger.info("✅ Step import 성공")
        return True
        
    except Exception as e:
        logger.error(f"❌ Step import 실패: {e}")
        return False

def test_step_creation():
    """Step 생성 테스트"""
    try:
        logger.info("🧪 Step 생성 테스트 시작...")
        
        from step_modularized import create_cloth_segmentation_step_modularized
        
        # Step 생성
        step = create_cloth_segmentation_step_modularized()
        
        if step is None:
            logger.error("❌ Step 생성 실패")
            return None
        
        logger.info(f"✅ Step 생성 성공: {step.step_name}")
        return step
        
    except Exception as e:
        logger.error(f"❌ Step 생성 실패: {e}")
        return None

def test_step_initialization(step):
    """Step 초기화 테스트"""
    try:
        logger.info("🧪 Step 초기화 테스트 시작...")
        
        # 초기화
        success = step.initialize()
        
        if not success:
            logger.error("❌ Step 초기화 실패")
            return False
        
        logger.info("✅ Step 초기화 성공")
        
        # 상태 확인
        status = step.get_status()
        logger.info(f"📊 Step 상태: {status}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Step 초기화 실패: {e}")
        return False

def test_step_processing(step):
    """Step 처리 테스트"""
    try:
        logger.info("🧪 Step 처리 테스트 시작...")
        
        # 테스트 이미지 생성
        test_image = create_test_image()
        
        # 처리 실행
        start_time = time.time()
        result = step.process(image=test_image)
        processing_time = time.time() - start_time
        
        logger.info(f"⏱️ 처리 시간: {processing_time:.2f}초")
        
        if not result.get('success', False):
            logger.error(f"❌ Step 처리 실패: {result.get('error', 'Unknown error')}")
            return False
        
        logger.info("✅ Step 처리 성공")
        
        # 결과 분석
        logger.info(f"📊 처리 결과:")
        logger.info(f"  - 사용된 방법: {result.get('method_used', 'unknown')}")
        logger.info(f"  - 신뢰도: {result.get('confidence', 0.0):.2f}")
        logger.info(f"  - 마스크 개수: {len(result.get('masks', {}))}")
        logger.info(f"  - 의류 카테고리: {result.get('cloth_categories', [])}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Step 처리 실패: {e}")
        return False

def test_step_cleanup(step):
    """Step 정리 테스트"""
    try:
        logger.info("🧪 Step 정리 테스트 시작...")
        
        # 정리
        step.cleanup()
        
        logger.info("✅ Step 정리 성공")
        return True
        
    except Exception as e:
        logger.error(f"❌ Step 정리 실패: {e}")
        return False

def test_modular_components():
    """분리된 모듈들 테스트"""
    try:
        logger.info("🧪 분리된 모듈들 테스트 시작...")
        
        # Core 모듈 테스트
        try:
            from core.step_core import ClothSegmentationStepCore
            logger.info("✅ step_core import 성공")
        except Exception as e:
            logger.warning(f"⚠️ step_core import 실패: {e}")
        
        # Services 모듈 테스트
        try:
            from services.model_loader_service import ModelLoaderService
            logger.info("✅ model_loader_service import 성공")
        except Exception as e:
            logger.warning(f"⚠️ model_loader_service import 실패: {e}")
        
        # Utils 모듈 테스트
        try:
            from utils.step_utils import detect_m3_max, cleanup_memory
            logger.info("✅ step_utils import 성공")
        except Exception as e:
            logger.warning(f"⚠️ step_utils import 실패: {e}")
        
        # Config 모듈 테스트
        try:
            from config import SegmentationMethod, ClothCategory, QualityLevel
            logger.info("✅ config import 성공")
        except Exception as e:
            logger.warning(f"⚠️ config import 실패: {e}")
        
        logger.info("✅ 모듈 테스트 완료")
        return True
        
    except Exception as e:
        logger.error(f"❌ 모듈 테스트 실패: {e}")
        return False

def main():
    """메인 테스트 함수"""
    logger.info("🚀 모듈화된 Step 테스트 시작")
    
    # 테스트 결과 저장
    test_results = {}
    
    # 1. Import 테스트
    logger.info("\n" + "="*50)
    test_results['import'] = test_step_import()
    
    # 2. 모듈 테스트
    logger.info("\n" + "="*50)
    test_results['modules'] = test_modular_components()
    
    # 3. Step 생성 테스트
    logger.info("\n" + "="*50)
    step = test_step_creation()
    test_results['creation'] = step is not None
    
    if step:
        # 4. Step 초기화 테스트
        logger.info("\n" + "="*50)
        test_results['initialization'] = test_step_initialization(step)
        
        # 5. Step 처리 테스트
        logger.info("\n" + "="*50)
        test_results['processing'] = test_step_processing(step)
        
        # 6. Step 정리 테스트
        logger.info("\n" + "="*50)
        test_results['cleanup'] = test_step_cleanup(step)
    else:
        test_results['initialization'] = False
        test_results['processing'] = False
        test_results['cleanup'] = False
    
    # 결과 요약
    logger.info("\n" + "="*50)
    logger.info("📊 테스트 결과 요약:")
    
    total_tests = len(test_results)
    passed_tests = sum(test_results.values())
    
    for test_name, result in test_results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"  - {test_name}: {status}")
    
    logger.info(f"\n🎯 전체 결과: {passed_tests}/{total_tests} 테스트 통과")
    
    if passed_tests == total_tests:
        logger.info("🎉 모든 테스트 통과! 모듈화가 성공적으로 완료되었습니다!")
    else:
        logger.warning("⚠️ 일부 테스트가 실패했습니다. 추가 검토가 필요합니다.")
    
    return passed_tests == total_tests

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
