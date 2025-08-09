#!/usr/bin/env python3
"""
🔍 체크포인트 로딩 디버깅 스크립트
================================================================================
✅ 단계별 디버깅
✅ 실제 파일 존재 확인
✅ 체크포인트 분석 테스트
✅ 모델 로딩 테스트
================================================================================
"""

import os
import sys
import time
from pathlib import Path

# 프로젝트 경로 설정
current_file = Path(__file__).resolve()
backend_root = current_file.parent
sys.path.insert(0, str(backend_root))
sys.path.insert(0, str(backend_root / "app"))

def check_ai_models_directory():
    """AI 모델 디렉토리 확인"""
    print("🔍 AI 모델 디렉토리 확인...")
    
    ai_models_root = backend_root / "ai_models"
    print(f"   경로: {ai_models_root}")
    print(f"   존재: {ai_models_root.exists()}")
    
    if ai_models_root.exists():
        # 하위 디렉토리 확인
        subdirs = [d for d in ai_models_root.iterdir() if d.is_dir()]
        print(f"   하위 디렉토리 수: {len(subdirs)}")
        
        for subdir in subdirs[:5]:  # 처음 5개만
            print(f"     - {subdir.name}")
            
            # 체크포인트 파일 확인
            checkpoint_files = list(subdir.glob("*.pth")) + list(subdir.glob("*.pt")) + list(subdir.glob("*.safetensors"))
            print(f"       체크포인트 파일: {len(checkpoint_files)}개")
            
            for file in checkpoint_files[:3]:  # 처음 3개만
                size_mb = file.stat().st_size / (1024 * 1024)
                print(f"         - {file.name} ({size_mb:.1f}MB)")
    
    return ai_models_root.exists()

def test_checkpoint_analysis():
    """체크포인트 분석 테스트"""
    print("\n🔍 체크포인트 분석 테스트...")
    
    try:
        from app.ai_pipeline.utils.model_loader import analyze_checkpoint
        
        # 체크포인트 파일 찾기
        ai_models_root = backend_root / "ai_models"
        checkpoint_files = []
        
        for pattern in ["*.pth", "*.pt", "*.safetensors"]:
            checkpoint_files.extend(ai_models_root.rglob(pattern))
        
        if not checkpoint_files:
            print("   ❌ 체크포인트 파일을 찾을 수 없습니다")
            return False
        
        # 첫 번째 파일로 테스트
        test_file = checkpoint_files[0]
        print(f"   테스트 파일: {test_file}")
        print(f"   파일 크기: {test_file.stat().st_size / (1024 * 1024):.1f}MB")
        
        # 체크포인트 분석
        print("   📊 체크포인트 분석 중...")
        analysis = analyze_checkpoint(test_file)
        
        if analysis:
            print("   ✅ 분석 성공!")
            print(f"     파라미터 수: {analysis.get('total_params', 0):,}")
            print(f"     레이어 수: {analysis.get('layer_count', 0)}")
            print(f"     아키텍처: {analysis.get('architecture_type', 'Unknown')}")
            return True
        else:
            print("   ❌ 분석 실패")
            return False
            
    except Exception as e:
        print(f"   ❌ 분석 오류: {e}")
        return False

def test_model_loader_creation():
    """Model Loader 생성 테스트"""
    print("\n🔍 Model Loader 생성 테스트...")
    
    try:
        from app.ai_pipeline.utils.model_loader import get_model_loader_v6
        
        print("   🚀 Model Loader 생성 중...")
        model_loader = get_model_loader_v6(device="auto")
        
        if model_loader:
            print("   ✅ Model Loader 생성 성공!")
            print(f"     디바이스: {model_loader.device}")
            return True
        else:
            print("   ❌ Model Loader 생성 실패")
            return False
            
    except Exception as e:
        print(f"   ❌ Model Loader 생성 오류: {e}")
        return False

def test_simple_model_loading():
    """간단한 모델 로딩 테스트"""
    print("\n🔍 간단한 모델 로딩 테스트...")
    
    try:
        from app.ai_pipeline.utils.model_loader import load_model_for_step
        
        # 체크포인트 파일 찾기
        ai_models_root = backend_root / "ai_models"
        checkpoint_files = []
        
        for pattern in ["*.pth", "*.pt", "*.safetensors"]:
            checkpoint_files.extend(ai_models_root.rglob(pattern))
        
        if not checkpoint_files:
            print("   ❌ 체크포인트 파일을 찾을 수 없습니다")
            return False
        
        # 첫 번째 파일로 테스트
        test_file = checkpoint_files[0]
        model_name = test_file.stem
        
        # Step 타입 추정
        step_type = "human_parsing"  # 기본값
        if "pose" in test_file.name.lower():
            step_type = "pose_estimation"
        elif "segmentation" in test_file.name.lower():
            step_type = "cloth_segmentation"
        elif "geometric" in test_file.name.lower():
            step_type = "geometric_matching"
        elif "warping" in test_file.name.lower():
            step_type = "cloth_warping"
        elif "fitting" in test_file.name.lower():
            step_type = "virtual_fitting"
        
        print(f"   테스트 파일: {test_file}")
        print(f"   모델명: {model_name}")
        print(f"   Step 타입: {step_type}")
        
        # 모델 로딩
        print("   🔄 모델 로딩 중...")
        start_time = time.time()
        
        model = load_model_for_step(
            step_type=step_type,
            model_name=model_name,
            checkpoint_path=str(test_file)
        )
        
        load_time = time.time() - start_time
        
        if model is not None:
            print(f"   ✅ 로딩 성공! ({load_time:.2f}초)")
            print(f"     모델 타입: {type(model).__name__}")
            
            # 간단한 추론 테스트
            try:
                import torch
                dummy_input = torch.randn(1, 3, 256, 256)
                with torch.no_grad():
                    output = model(dummy_input)
                print(f"     추론 성공: {output.shape}")
            except Exception as e:
                print(f"     추론 실패: {e}")
            
            return True
        else:
            print("   ❌ 로딩 실패")
            return False
            
    except Exception as e:
        print(f"   ❌ 로딩 오류: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_dynamic_model_creator():
    """DynamicModelCreator 테스트"""
    print("\n🔍 DynamicModelCreator 테스트...")
    
    try:
        from app.ai_pipeline.utils.model_loader import DynamicModelCreator
        
        creator = DynamicModelCreator()
        
        # 체크포인트 파일 찾기
        ai_models_root = backend_root / "ai_models"
        checkpoint_files = []
        
        for pattern in ["*.pth", "*.pt", "*.safetensors"]:
            checkpoint_files.extend(ai_models_root.rglob(pattern))
        
        if not checkpoint_files:
            print("   ❌ 체크포인트 파일을 찾을 수 없습니다")
            return False
        
        # 첫 번째 파일로 테스트
        test_file = checkpoint_files[0]
        step_type = "human_parsing"
        
        print(f"   테스트 파일: {test_file}")
        
        model = creator.create_model_from_checkpoint(
            checkpoint_path=test_file,
            step_type=step_type,
            device="auto"
        )
        
        if model is not None:
            print("   ✅ DynamicModelCreator 성공!")
            print(f"     모델 타입: {type(model).__name__}")
            return True
        else:
            print("   ❌ DynamicModelCreator 실패")
            return False
            
    except Exception as e:
        print(f"   ❌ DynamicModelCreator 오류: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🔍 체크포인트 로딩 디버깅")
    print("="*80)
    
    # 1. AI 모델 디렉토리 확인
    dir_exists = check_ai_models_directory()
    
    if not dir_exists:
        print("❌ AI 모델 디렉토리가 존재하지 않습니다")
        sys.exit(1)
    
    # 2. 체크포인트 분석 테스트
    analysis_success = test_checkpoint_analysis()
    
    # 3. Model Loader 생성 테스트
    loader_success = test_model_loader_creation()
    
    # 4. 간단한 모델 로딩 테스트
    loading_success = test_simple_model_loading()
    
    # 5. DynamicModelCreator 테스트
    creator_success = test_dynamic_model_creator()
    
    # 결과 요약
    print("\n" + "="*80)
    print("📊 디버깅 결과 요약")
    print("="*80)
    print(f"AI 모델 디렉토리: {'✅' if dir_exists else '❌'}")
    print(f"체크포인트 분석: {'✅' if analysis_success else '❌'}")
    print(f"Model Loader 생성: {'✅' if loader_success else '❌'}")
    print(f"모델 로딩: {'✅' if loading_success else '❌'}")
    print(f"DynamicModelCreator: {'✅' if creator_success else '❌'}")
    
    if all([dir_exists, analysis_success, loader_success, loading_success, creator_success]):
        print("\n🎉 모든 테스트 성공!")
    else:
        print("\n⚠️ 일부 테스트 실패 - 문제가 있는 단계를 확인하세요")
