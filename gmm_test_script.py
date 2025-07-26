#!/usr/bin/env python3
"""
GMM 모델 테스트 스크립트 (torch.jit 없이 직접 로딩)
"""

import torch
import os
from pathlib import Path

def test_gmm_models():
    """GMM 모델들 직접 테스트"""
    print("🔥 GMM 모델 테스트 시작 (torch.jit 없음)")
    print("="*50)
    
    # 모델 파일 경로들
    model_files = {
        "GMM Core": "backend/ai_models/step_04_geometric_matching/gmm_final.pth",
        "TPS Network": "backend/ai_models/step_04_geometric_matching/tps_network.pth", 
        "SAM Model": "backend/ai_models/step_04_geometric_matching/sam_vit_h_4b8939.pth",
        "PyTorch Model": "backend/ai_models/step_04_geometric_matching/pytorch_model.bin"
    }
    
    success_count = 0
    
    for model_name, file_path in model_files.items():
        print(f"\n📦 {model_name} 테스트")
        path = Path(file_path)
        
        if not path.exists():
            print(f"❌ 파일 없음: {file_path}")
            continue
            
        try:
            # torch.jit 변환 없이 직접 로딩
            model = torch.load(file_path, map_location='cpu')
            print(f"✅ 직접 로딩 성공: {path.name}")
            
            # 모델 정보 출력
            if isinstance(model, dict):
                print(f"  📋 딕셔너리 키: {list(model.keys())[:5]}...")
                if 'state_dict' in model:
                    print(f"  🔧 state_dict 키 수: {len(model['state_dict'])}")
            else:
                print(f"  🤖 모델 타입: {type(model)}")
                
            success_count += 1
            
        except Exception as e:
            print(f"❌ 로딩 실패: {e}")
    
    print(f"\n🎉 결과: {success_count}/{len(model_files)} 성공")
    return success_count == len(model_files)

def test_geometric_matching_step():
    """GeometricMatchingStep 클래스 테스트"""
    print("\n🔧 GeometricMatchingStep 클래스 테스트")
    print("="*50)
    
    try:
        from backend.app.ai_pipeline.steps.step_04_geometric_matching import GeometricMatchingStep
        print("✅ GeometricMatchingStep import 성공")
        
        # 인스턴스 생성
        step_04 = GeometricMatchingStep(step_id=4)
        print("✅ 인스턴스 생성 성공")
        
        # 초기화 시도
        success = step_04.initialize()
        if hasattr(success, '__await__'):  # async 함수인 경우
            import asyncio
            success = asyncio.run(success)
            
        print(f"{'✅ 초기화 성공' if success else '❌ 초기화 실패'}")
        
        # 모델 상태 확인
        if hasattr(step_04, 'gmm_model'):
            print(f"🤖 GMM 모델: {step_04.gmm_model is not None}")
        if hasattr(step_04, 'tps_model'):
            print(f"🌐 TPS 모델: {step_04.tps_model is not None}")
            
        return success
        
    except Exception as e:
        print(f"❌ GeometricMatchingStep 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """메인 함수"""
    print("🍎 M3 Max conda 환경에서 GMM 모델 테스트")
    print("🐍 conda 환경:", os.environ.get('CONDA_DEFAULT_ENV', 'none'))
    print("🔧 PyTorch:", torch.__version__)
    print("⚡ MPS 사용가능:", torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False)
    
    # 1. 모델 파일 직접 테스트
    models_ok = test_gmm_models()
    
    # 2. Step 클래스 테스트
    step_ok = test_geometric_matching_step()
    
    print(f"\n🏁 최종 결과:")
    print(f"  📁 모델 파일들: {'✅' if models_ok else '❌'}")
    print(f"  🏗️ Step 클래스: {'✅' if step_ok else '❌'}")
    
    if models_ok and step_ok:
        print("\n🎉 GMM 모델 준비 완료! torch.jit 변환 없이 바로 사용 가능!")
    else:
        print("\n⚠️ 일부 문제가 있습니다.")

if __name__ == "__main__":
    main()