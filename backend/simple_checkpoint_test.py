#!/usr/bin/env python3
"""
🔥 간단한 체크포인트 로딩 테스트
================================================================================
✅ 실제 .pth 파일 로딩 테스트
✅ PyTorch torch.load 직접 테스트
✅ 체크포인트 구조 확인
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

def test_pytorch_load():
    """PyTorch torch.load 직접 테스트"""
    print("🔍 PyTorch torch.load 직접 테스트...")
    
    try:
        import torch
        
        # 체크포인트 파일 찾기
        checkpoint_file = backend_root / "ai_models" / "openpose.pth"
        
        if not checkpoint_file.exists():
            print(f"   ❌ 체크포인트 파일이 존재하지 않습니다: {checkpoint_file}")
            return False
        
        print(f"   📁 체크포인트 파일: {checkpoint_file}")
        print(f"   📊 파일 크기: {checkpoint_file.stat().st_size / (1024 * 1024):.1f}MB")
        
        # torch.load로 직접 로딩
        print("   🔄 torch.load로 로딩 중...")
        start_time = time.time()
        
        checkpoint = torch.load(checkpoint_file, map_location='cpu')
        
        load_time = time.time() - start_time
        print(f"   ✅ 로딩 성공! ({load_time:.2f}초)")
        
        # 체크포인트 구조 확인
        if isinstance(checkpoint, dict):
            print(f"   📊 체크포인트 타입: dict")
            print(f"   🔑 키 개수: {len(checkpoint)}")
            
            # 처음 10개 키 출력
            for i, key in enumerate(list(checkpoint.keys())[:10]):
                if hasattr(checkpoint[key], 'shape'):
                    print(f"     {key}: {checkpoint[key].shape}")
                else:
                    print(f"     {key}: {type(checkpoint[key])}")
        else:
            print(f"   📊 체크포인트 타입: {type(checkpoint)}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ 로딩 오류: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_loader_simple():
    """Model Loader 간단 테스트"""
    print("\n🔍 Model Loader 간단 테스트...")
    
    try:
        from app.ai_pipeline.utils.model_loader import analyze_checkpoint
        
        # 체크포인트 파일
        checkpoint_file = backend_root / "ai_models" / "openpose.pth"
        
        if not checkpoint_file.exists():
            print(f"   ❌ 체크포인트 파일이 존재하지 않습니다: {checkpoint_file}")
            return False
        
        print(f"   📁 체크포인트 파일: {checkpoint_file}")
        
        # 체크포인트 분석
        print("   📊 체크포인트 분석 중...")
        analysis = analyze_checkpoint(checkpoint_file)
        
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
        import traceback
        traceback.print_exc()
        return False

def test_step_specific_architecture():
    """Step별 특화 아키텍처 테스트"""
    print("\n🔍 Step별 특화 아키텍처 테스트...")
    
    try:
        from app.ai_pipeline.utils.model_loader import PoseEstimationArchitecture
        
        # Pose Estimation 아키텍처 생성
        print("   🏗️ PoseEstimationArchitecture 생성 중...")
        architecture = PoseEstimationArchitecture("pose_estimation", device="cpu")
        
        # 더미 체크포인트 분석
        dummy_analysis = {
            'architecture_type': 'hrnet',
            'num_keypoints': 17,
            'total_params': 1000000
        }
        
        # 모델 생성
        print("   🔄 모델 생성 중...")
        model = architecture.create_model(dummy_analysis)
        
        if model is not None:
            print("   ✅ 모델 생성 성공!")
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
            print("   ❌ 모델 생성 실패")
            return False
            
    except Exception as e:
        print(f"   ❌ 아키텍처 테스트 오류: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🔥 간단한 체크포인트 로딩 테스트")
    print("="*80)
    
    # 1. PyTorch 직접 로딩 테스트
    pytorch_success = test_pytorch_load()
    
    # 2. Model Loader 분석 테스트
    analysis_success = test_model_loader_simple()
    
    # 3. Step별 아키텍처 테스트
    architecture_success = test_step_specific_architecture()
    
    # 결과 요약
    print("\n" + "="*80)
    print("📊 테스트 결과 요약")
    print("="*80)
    print(f"PyTorch 직접 로딩: {'✅' if pytorch_success else '❌'}")
    print(f"체크포인트 분석: {'✅' if analysis_success else '❌'}")
    print(f"Step별 아키텍처: {'✅' if architecture_success else '❌'}")
    
    if all([pytorch_success, analysis_success, architecture_success]):
        print("\n🎉 모든 테스트 성공!")
    else:
        print("\n⚠️ 일부 테스트 실패 - 문제가 있는 단계를 확인하세요")
