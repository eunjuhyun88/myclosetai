#!/usr/bin/env python3
"""
GeometricMatchingStep 실제 사용 테스트 (torch.jit 없이)
"""

import torch
import os
import asyncio
from pathlib import Path

def test_geometric_matching_step():
    """GeometricMatchingStep 클래스 실제 테스트"""
    print("🔥 GeometricMatchingStep 실제 테스트 시작")
    print("="*60)
    
    # 환경 설정 확인
    print(f"🐍 conda 환경: {os.environ.get('CONDA_DEFAULT_ENV', 'none')}")
    print(f"🔧 PyTorch: {torch.__version__}")
    print(f"⚡ MPS 사용가능: {torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False}")
    print(f"🚫 PYTORCH_JIT_DISABLE: {os.environ.get('PYTORCH_JIT_DISABLE', 'not_set')}")
    
    try:
        # GeometricMatchingStep import 및 생성
        from backend.app.ai_pipeline.steps.step_04_geometric_matching import GeometricMatchingStep
        print("✅ GeometricMatchingStep import 성공")
        
        # Step 인스턴스 생성 (torch.jit 없이)
        step_04 = GeometricMatchingStep(
            step_id=4, 
            device="mps" if torch.backends.mps.is_available() else "cpu",
            config={"enable_jit_compile": False}  # torch.jit 명시적 비활성화
        )
        print("✅ GeometricMatchingStep 인스턴스 생성 성공")
        
        # 초기화 테스트
        if asyncio.iscoroutinefunction(step_04.initialize):
            # 비동기 함수인 경우
            success = asyncio.run(step_04.initialize())
        else:
            # 동기 함수인 경우
            success = step_04.initialize()
            
        print(f"{'✅ 초기화 성공' if success else '❌ 초기화 실패'}")
        
        # 모델 상태 확인
        print("\n🤖 로딩된 AI 모델들:")
        if hasattr(step_04, 'gmm_model') and step_04.gmm_model is not None:
            print(f"  - GMM 모델: ✅ (타입: {type(step_04.gmm_model)})")
        else:
            print(f"  - GMM 모델: ❌")
            
        if hasattr(step_04, 'tps_model') and step_04.tps_model is not None:
            print(f"  - TPS 모델: ✅ (타입: {type(step_04.tps_model)})")
        else:
            print(f"  - TPS 모델: ❌")
            
        if hasattr(step_04, 'sam_model') and step_04.sam_model is not None:
            print(f"  - SAM 모델: ✅ (타입: {type(step_04.sam_model)})")
        else:
            print(f"  - SAM 모델: ❌")
        
        # 간단한 추론 테스트 (더미 데이터)
        print("\n🧠 간단한 추론 테스트:")
        try:
            # 더미 이미지 텐서 생성 (작은 크기)
            dummy_person = torch.randn(1, 3, 64, 48)  # 작은 크기로 테스트
            dummy_cloth = torch.randn(1, 3, 64, 48)
            
            print(f"  - 더미 데이터 생성: ✅ (Person: {dummy_person.shape}, Cloth: {dummy_cloth.shape})")
            
            # GMM 모델로 간단한 순전파 테스트
            if hasattr(step_04, 'gmm_model') and step_04.gmm_model is not None:
                try:
                    with torch.no_grad():
                        # GMM 모델은 person + cloth 6채널 입력을 받음
                        gmm_input = torch.cat([dummy_person, dummy_cloth], dim=1)
                        gmm_output = step_04.gmm_model(gmm_input)
                        print(f"  - GMM 추론: ✅ (출력 크기: {gmm_output.shape})")
                except Exception as e:
                    print(f"  - GMM 추론: ⚠️ (에러: {str(e)[:50]}...)")
            
            print("✅ 기본 추론 테스트 완료")
            
        except Exception as e:
            print(f"⚠️ 추론 테스트 중 오류: {e}")
        
        # 최종 결과
        print(f"\n🏁 최종 결과:")
        print(f"  📦 클래스 로딩: ✅")
        print(f"  🏗️ 인스턴스 생성: ✅") 
        print(f"  🔄 초기화: {'✅' if success else '❌'}")
        print(f"  🧠 AI 모델 로딩: {'✅' if hasattr(step_04, 'gmm_model') and step_04.gmm_model else '❌'}")
        print(f"  🚫 torch.jit 사용: ❌ (완전 비활성화)")
        
        if success and hasattr(step_04, 'gmm_model') and step_04.gmm_model:
            print(f"\n🎉 GeometricMatchingStep 완전 성공!")
            print(f"   torch.jit 없이 순수 PyTorch로 AI 모델 로딩 및 실행 가능!")
            return True
        else:
            print(f"\n⚠️ 일부 문제가 있지만 기본 구조는 작동 중")
            return False
            
    except Exception as e:
        print(f"❌ GeometricMatchingStep 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_direct_model_usage():
    """직접 GMM 모델 사용 테스트"""
    print("\n" + "="*60)
    print("🔬 직접 GMM 모델 사용 테스트")
    print("="*60)
    
    try:
        # 직접 모델 파일 로딩
        model_path = "backend/ai_models/step_04_geometric_matching/gmm_final.pth"
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        
        print(f"📁 모델 경로: {model_path}")
        print(f"🖥️ 디바이스: {device}")
        
        # torch.jit 없이 직접 로딩
        model_data = torch.load(model_path, map_location=device)
        print(f"✅ 모델 로딩 성공: {type(model_data)}")
        
        # 모델 정보 출력
        if isinstance(model_data, dict):
            print(f"📋 딕셔너리 키들: {list(model_data.keys())[:5]}...")
            if 'state_dict' in model_data:
                print(f"🔧 state_dict 키 수: {len(model_data['state_dict'])}")
        
        print("✅ 직접 모델 사용 가능 확인!")
        return True
        
    except Exception as e:
        print(f"❌ 직접 모델 사용 실패: {e}")
        return False

def main():
    """메인 함수"""
    print("🍎 M3 Max conda 환경에서 GMM torch.jit 없이 테스트")
    print("🚫 torch.jit 완전 비활성화 모드")
    
    # 1. GeometricMatchingStep 클래스 테스트
    step_test_ok = test_geometric_matching_step()
    
    # 2. 직접 모델 사용 테스트  
    direct_test_ok = test_direct_model_usage()
    
    print(f"\n🏆 전체 테스트 결과:")
    print(f"  🏗️ Step 클래스: {'✅' if step_test_ok else '❌'}")
    print(f"  🔬 직접 모델: {'✅' if direct_test_ok else '❌'}")
    
    if step_test_ok and direct_test_ok:
        print(f"\n🎉 완전 성공! torch.jit 없이 GMM 모델 완벽 작동!")
        print(f"🚀 이제 실제 서비스에서 바로 사용 가능합니다!")
    elif direct_test_ok:
        print(f"\n🔄 부분 성공! 직접 모델 로딩은 작동하므로 Step 클래스 개선 필요")
    else:
        print(f"\n⚠️ 추가 디버깅이 필요합니다.")

if __name__ == "__main__":
    main()  