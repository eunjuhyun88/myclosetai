#!/usr/bin/env python3
"""
MyCloset AI Import 체인 디버깅 스크립트 (Conda 환경용)
모든 Import 문제를 진단하고 해결방안 제시

실행방법:
conda activate mycloset-ai  # 또는 사용 중인 환경명
cd backend
python debug_imports.py
"""

import sys
import os
import traceback
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

print("🔍 MyCloset AI Import 체인 디버깅")
print("=" * 50)

def test_import(module_path: str, description: str) -> bool:
    """개별 import 테스트"""
    try:
        print(f"🔧 {description} 테스트 중...")
        
        if module_path == "torch":
            import torch
            print(f"   ✅ PyTorch: {torch.__version__}")
            if torch.backends.mps.is_available():
                print(f"   ✅ MPS 사용 가능")
            return True
            
        elif module_path == "fastapi":
            from fastapi import FastAPI, File, UploadFile
            print(f"   ✅ FastAPI import 성공")
            return True
            
        elif module_path == "PIL":
            from PIL import Image
            print(f"   ✅ PIL import 성공")
            return True
            
        elif module_path == "app.core":
            from app.core.config import DEVICE
            print(f"   ✅ Core config import 성공: {DEVICE}")
            return True
            
        elif module_path == "app.ai_pipeline.steps":
            from app.ai_pipeline.steps.step_01_human_parsing import HumanParsingStep
            from app.ai_pipeline.steps.step_02_pose_estimation import PoseEstimationStep
            from app.ai_pipeline.steps.step_03_cloth_segmentation import ClothSegmentationStep
            from app.ai_pipeline.steps.step_04_geometric_matching import GeometricMatchingStep
            from app.ai_pipeline.steps.step_05_cloth_warping import ClothWarpingStep
            from app.ai_pipeline.steps.step_06_virtual_fitting import VirtualFittingStep
            from app.ai_pipeline.steps.step_07_post_processing import PostProcessingStep
            from app.ai_pipeline.steps.step_08_quality_assessment import QualityAssessmentStep
            print(f"   ✅ 모든 8개 Step 클래스 import 성공")
            return True
            
        elif module_path == "app.ai_pipeline.pipeline_manager":
            from app.ai_pipeline.pipeline_manager import (
                PipelineManager, create_m3_max_pipeline, ProcessingResult
            )
            print(f"   ✅ PipelineManager import 성공")
            return True
            
        elif module_path == "app.services.step_service":
            from app.services.step_service import (
                StepServiceManager, get_step_service_manager
            )
            print(f"   ✅ StepService import 성공")
            return True
            
        elif module_path == "app.api.step_routes":
            from app.api.step_routes import router
            print(f"   ✅ Step Routes import 성공")
            return True
            
        else:
            exec(f"import {module_path}")
            print(f"   ✅ {module_path} import 성공")
            return True
            
    except ImportError as e:
        print(f"   ❌ {description} import 실패:")
        print(f"      오류: {e}")
        print(f"      모듈: {module_path}")
        return False
    except Exception as e:
        print(f"   💥 {description} 예상치 못한 오류:")
        print(f"      오류: {e}")
        print(f"      타입: {type(e).__name__}")
        return False

def test_step_creation():
    """Step 클래스 생성 테스트"""
    print("🏗️ Step 클래스 생성 테스트...")
    
    try:
        from app.ai_pipeline.steps.step_01_human_parsing import HumanParsingStep
        from app.ai_pipeline.steps.step_04_geometric_matching import GeometricMatchingStep
        
        # 기본 파라미터
        base_params = {
            'device': 'cpu',
            'device_type': 'cpu', 
            'memory_gb': 16.0,
            'is_m3_max': False,
            'optimization_enabled': True,
            'quality_level': 'balanced'
        }
        
        # HumanParsingStep 생성 테스트
        print("   🔧 HumanParsingStep 생성 중...")
        step1 = HumanParsingStep(**base_params)
        print("   ✅ HumanParsingStep 생성 성공")
        
        # GeometricMatchingStep 생성 테스트 (특별 처리)
        print("   🔧 GeometricMatchingStep 생성 중...")
        geometric_params = base_params.copy()
        config_dict = {'quality_level': geometric_params.pop('quality_level')}
        geometric_params['config'] = config_dict
        
        step4 = GeometricMatchingStep(**geometric_params)
        print("   ✅ GeometricMatchingStep 생성 성공")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Step 클래스 생성 실패: {e}")
        print(f"   📋 상세 오류: {traceback.format_exc()}")
        return False

def main():
    """메인 디버깅 함수"""
    
    results = {}
    
    # 1. 기본 라이브러리들
    print("\n1️⃣ 기본 라이브러리 테스트")
    results['torch'] = test_import("torch", "PyTorch")
    results['fastapi'] = test_import("fastapi", "FastAPI")
    results['PIL'] = test_import("PIL", "PIL/Pillow")
    
    # 2. 프로젝트 Core 모듈
    print("\n2️⃣ Core 모듈 테스트")
    results['core'] = test_import("app.core", "Core Config")
    
    # 3. AI Pipeline Steps
    print("\n3️⃣ AI Pipeline Steps 테스트")
    results['steps'] = test_import("app.ai_pipeline.steps", "AI Steps")
    
    # 4. PipelineManager
    print("\n4️⃣ PipelineManager 테스트")
    results['pipeline_manager'] = test_import("app.ai_pipeline.pipeline_manager", "PipelineManager")
    
    # 5. Services
    print("\n5️⃣ Services 테스트")
    results['services'] = test_import("app.services.step_service", "Step Service")
    
    # 6. API Routes
    print("\n6️⃣ API Routes 테스트")
    results['routes'] = test_import("app.api.step_routes", "Step Routes")
    
    # 7. Step 클래스 생성 테스트
    print("\n7️⃣ Step 클래스 생성 테스트")
    if results['steps']:
        results['step_creation'] = test_step_creation()
    else:
        results['step_creation'] = False
        print("   ⚠️ Steps import 실패로 생성 테스트 건너뛰기")
    
    # 결과 요약
    print("\n" + "=" * 50)
    print("📊 Import 체인 디버깅 결과 요약")
    print("=" * 50)
    
    success_count = sum(1 for success in results.values() if success)
    total_count = len(results)
    
    for test_name, success in results.items():
        status = "✅" if success else "❌"
        print(f"{status} {test_name}")
    
    print(f"\n📈 성공률: {success_count}/{total_count} ({success_count/total_count*100:.1f}%)")
    
    # 해결방안 제시
    if success_count < total_count:
        print("\n🔧 해결방안:")
        
        if not results.get('torch'):
            print("  1. PyTorch 재설치: conda install pytorch torchvision -c pytorch")
            
        if not results.get('fastapi'):
            print("  2. FastAPI 재설치: conda install fastapi uvicorn -c conda-forge")
            print("     또는: pip install fastapi uvicorn")
            
        if not results.get('PIL'):
            print("  3. Pillow 재설치: conda install pillow -c conda-forge")
            
        if not results.get('core'):
            print("  3. Core 모듈 문제: app/core/__init__.py 확인")
            
        if not results.get('steps'):
            print("  4. AI Steps 문제: app/ai_pipeline/steps/__init__.py 확인")
            
        if not results.get('pipeline_manager'):
            print("  5. PipelineManager 문제: circular import 확인")
            
        if not results.get('services'):
            print("  6. Services 문제: 의존성 체인 확인")
            
        if not results.get('routes'):
            print("  7. Routes 문제: API 레이어 확인")
            
        if not results.get('step_creation'):
            print("  8. Step 생성 문제: 생성자 파라미터 확인")
    else:
        print("\n🎉 모든 Import가 성공했습니다!")
        print("   문제는 다른 곳에 있을 수 있습니다.")
        print("   main.py 또는 서버 실행 시 런타임 오류를 확인해보세요.")

if __name__ == "__main__":
    main()