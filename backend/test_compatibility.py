# 호환성 테스트 스크립트
# backend/test_compatibility.py

"""
PipelineManager와 Step 클래스들 간의 호환성 테스트
"""

import sys
import os
import asyncio
import logging
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_step_imports():
    """Step 클래스들 import 테스트"""
    print("🔧 Step 클래스들 import 테스트...")
    
    try:
        from app.ai_pipeline.steps.step_01_human_parsing import HumanParsingStep
        from app.ai_pipeline.steps.step_02_pose_estimation import PoseEstimationStep
        from app.ai_pipeline.steps.step_03_cloth_segmentation import ClothSegmentationStep
        from app.ai_pipeline.steps.step_04_geometric_matching import GeometricMatchingStep
        from app.ai_pipeline.steps.step_05_cloth_warping import ClothWarpingStep
        from app.ai_pipeline.steps.step_06_virtual_fitting import VirtualFittingStep
        from app.ai_pipeline.steps.step_07_post_processing import PostProcessingStep
        from app.ai_pipeline.steps.step_08_quality_assessment import QualityAssessmentStep
        
        print("✅ 모든 Step 클래스들 import 성공")
        return True
        
    except ImportError as e:
        print(f"❌ Step 클래스 import 실패: {e}")
        return False

async def test_step_constructors():
    """Step 클래스들 생성자 테스트"""
    print("🏗️ Step 클래스들 생성자 테스트...")
    
    try:
        from app.ai_pipeline.steps.step_01_human_parsing import HumanParsingStep
        from app.ai_pipeline.steps.step_02_pose_estimation import PoseEstimationStep
        
        # 통일된 생성자 패턴으로 인스턴스 생성 테스트
        step1 = HumanParsingStep(device="cpu", config={}, optimization_enabled=True)
        step2 = PoseEstimationStep(device="cpu", config={}, optimization_enabled=True)
        
        print(f"✅ HumanParsingStep 생성 성공: {step1.__class__.__name__}")
        print(f"✅ PoseEstimationStep 생성 성공: {step2.__class__.__name__}")
        
        return True
        
    except Exception as e:
        print(f"❌ Step 클래스 생성자 테스트 실패: {e}")
        return False

async def test_pipeline_manager_import():
    """PipelineManager import 테스트"""
    print("📋 PipelineManager import 테스트...")
    
    try:
        from app.ai_pipeline.pipeline_manager import (
            PipelineManager,
            create_m3_max_pipeline,
            create_production_pipeline
        )
        
        print("✅ PipelineManager import 성공")
        return True
        
    except ImportError as e:
        print(f"❌ PipelineManager import 실패: {e}")
        return False

async def test_pipeline_creation():
    """파이프라인 생성 테스트"""
    print("🚀 파이프라인 생성 테스트...")
    
    try:
        from app.ai_pipeline.pipeline_manager import create_production_pipeline
        
        # 파이프라인 생성
        pipeline = create_production_pipeline(device="cpu")
        
        print(f"✅ 파이프라인 생성 성공: {pipeline.__class__.__name__}")
        print(f"📊 디바이스: {pipeline.device}")
        print(f"📋 Step 순서: {pipeline.step_order}")
        
        return True
        
    except Exception as e:
        print(f"❌ 파이프라인 생성 실패: {e}")
        return False

async def test_pipeline_initialization():
    """파이프라인 초기화 테스트"""
    print("⚙️ 파이프라인 초기화 테스트...")
    
    try:
        from app.ai_pipeline.pipeline_manager import create_production_pipeline
        
        # 파이프라인 생성 및 초기화
        pipeline = create_production_pipeline(device="cpu")
        
        print("🔄 초기화 시작...")
        success = await pipeline.initialize()
        
        if success:
            print("✅ 파이프라인 초기화 성공")
            
            # 상태 확인
            status = pipeline.get_pipeline_status()
            steps_loaded = len([s for s in status['steps_status'].values() if s['loaded']])
            total_steps = len(status['steps_status'])
            
            print(f"📊 로드된 단계: {steps_loaded}/{total_steps}")
            print(f"🎯 초기화 상태: {status['initialized']}")
            
            return True
        else:
            print("❌ 파이프라인 초기화 실패")
            return False
            
    except Exception as e:
        print(f"❌ 파이프라인 초기화 테스트 실패: {e}")
        return False

async def test_service_layer():
    """서비스 레이어 테스트"""
    print("🏢 서비스 레이어 테스트...")
    
    try:
        from app.services.pipeline_service import get_complete_pipeline_service
        from app.services.step_service import get_step_service_manager
        
        print("✅ 서비스 레이어 import 성공")
        
        # 서비스 인스턴스 생성 테스트
        pipeline_service = await get_complete_pipeline_service()
        step_service = await get_step_service_manager()
        
        print(f"✅ 파이프라인 서비스: {pipeline_service.__class__.__name__}")
        print(f"✅ 단계 서비스: {step_service.__class__.__name__}")
        
        return True
        
    except Exception as e:
        print(f"❌ 서비스 레이어 테스트 실패: {e}")
        return False

async def main():
    """메인 테스트 실행"""
    print("🧪 MyCloset AI 호환성 테스트 시작")
    print("=" * 50)
    
    tests = [
        ("Step Import", test_step_imports),
        ("Step Constructor", test_step_constructors),
        ("PipelineManager Import", test_pipeline_manager_import),
        ("Pipeline Creation", test_pipeline_creation),
        ("Pipeline Initialization", test_pipeline_initialization),
        ("Service Layer", test_service_layer)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n📋 {test_name} 테스트:")
        print("-" * 30)
        
        try:
            result = await test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"💥 {test_name} 테스트 중 예외 발생: {e}")
            results.append((test_name, False))
    
    # 결과 요약
    print("\n" + "=" * 50)
    print("📊 테스트 결과 요약:")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ 통과" if result else "❌ 실패"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print("-" * 50)
    print(f"📈 전체 결과: {passed}/{total} ({passed/total*100:.1f}%) 통과")
    
    if passed == total:
        print("🎉 모든 테스트 통과! 호환성 문제 없음")
        return True
    else:
        print("⚠️ 일부 테스트 실패. 추가 수정 필요")
        return False

if __name__ == "__main__":
    asyncio.run(main())