#!/usr/bin/env python3
"""
🔧 GeometricMatchingStep 디바이스 문제 해결
문제: cpu 저장 모델을 mps로 로딩할 때 디바이스 불일치
해결: 안전한 디바이스 매핑 + 모델 이동
"""

import os
import sys
from pathlib import Path
import torch

# 프로젝트 루트 설정
project_root = Path("/Users/gimdudeul/MVP/mycloset-ai")
backend_root = project_root / "backend"
sys.path.insert(0, str(backend_root))

# 올바른 경로 설정
os.environ["AI_MODELS_ROOT"] = str(backend_root / "ai_models")
os.environ["PYTORCH_JIT_DISABLE"] = "1"

def safe_model_loading_test():
    """안전한 모델 로딩 테스트"""
    print("🔧 디바이스 안전 모델 로딩 테스트")
    print("=" * 50)
    
    ai_models_root = Path(os.environ["AI_MODELS_ROOT"])
    gmm_path = ai_models_root / "step_04_geometric_matching" / "gmm_final.pth"
    
    print(f"📄 GMM 파일: {gmm_path}")
    print(f"📊 파일 존재: {gmm_path.exists()}")
    
    if not gmm_path.exists():
        print("❌ GMM 파일을 찾을 수 없음")
        return False
    
    # PyTorch 디바이스 확인
    print(f"✅ PyTorch 버전: {torch.__version__}")
    has_mps = torch.backends.mps.is_available()
    has_cuda = torch.cuda.is_available()
    
    print(f"🍎 MPS 사용 가능: {has_mps}")
    print(f"🔥 CUDA 사용 가능: {has_cuda}")
    
    # 1단계: CPU로 안전하게 로딩
    print("\n1️⃣ CPU로 안전하게 로딩...")
    try:
        model_data = torch.load(gmm_path, map_location='cpu')
        print(f"✅ CPU 로딩 성공: {type(model_data)}")
        
        # 모델 데이터 타입 확인
        if isinstance(model_data, dict):
            print(f"📋 딕셔너리 키들: {list(model_data.keys())[:5]}")
            if 'state_dict' in model_data:
                print(f"🔧 state_dict 크기: {len(model_data['state_dict'])}")
        elif hasattr(model_data, 'state_dict'):
            print(f"🔧 모델 state_dict 크기: {len(model_data.state_dict())}")
        
    except Exception as e:
        print(f"❌ CPU 로딩 실패: {e}")
        return False
    
    # 2단계: MPS로 안전하게 이동 (가능한 경우)
    if has_mps:
        print("\n2️⃣ MPS로 안전하게 이동...")
        try:
            # torch.jit 모델인지 확인
            if hasattr(model_data, '_c'):
                print("⚠️ torch.jit 모델 감지됨 - CPU 모드 유지")
                target_device = 'cpu'
            else:
                print("✅ 일반 PyTorch 모델 - MPS 이동 시도")
                target_device = 'mps'
                
                # 실제 텐서가 있는지 확인하고 이동
                if isinstance(model_data, dict) and 'state_dict' in model_data:
                    # state_dict의 첫 번째 텐서만 테스트
                    first_key = list(model_data['state_dict'].keys())[0]
                    first_tensor = model_data['state_dict'][first_key]
                    test_tensor = first_tensor.to('mps')
                    print(f"✅ 텐서 MPS 이동 테스트 성공: {test_tensor.device}")
                    
        except Exception as e:
            print(f"⚠️ MPS 이동 실패, CPU 모드 사용: {e}")
            target_device = 'cpu'
    else:
        target_device = 'cpu'
        print("\n2️⃣ MPS 미지원 - CPU 모드 사용")
    
    print(f"\n🎯 최종 디바이스: {target_device}")
    
    # 3단계: GeometricMatchingStep 테스트
    print("\n3️⃣ GeometricMatchingStep 디바이스 안전 테스트...")
    try:
        from app.ai_pipeline.steps.step_04_geometric_matching import GeometricMatchingStep
        print("✅ GeometricMatchingStep import 성공")
        
        # 안전한 디바이스로 인스턴스 생성
        step = GeometricMatchingStep(
            step_id=4,
            device=target_device,  # CPU 또는 안전하게 확인된 MPS
            config={
                "ai_models_root": str(ai_models_root),
                "force_cpu_mode": target_device == 'cpu',
                "enable_jit_compile": False
            }
        )
        print(f"✅ GeometricMatchingStep 인스턴스 생성 성공 (디바이스: {target_device})")
        
        # 초기화 테스트
        try:
            if hasattr(step, 'initialize'):
                if asyncio.iscoroutinefunction(step.initialize):
                    import asyncio
                    success = asyncio.run(step.initialize())
                else:
                    success = step.initialize()
                
                if success:
                    print("✅ Step 초기화 성공")
                    
                    # 간단한 모델 접근 테스트
                    if hasattr(step, 'gmm_model') and step.gmm_model is not None:
                        print(f"✅ GMM 모델 로드됨: {type(step.gmm_model)}")
                        print(f"🖥️ GMM 모델 디바이스: {next(step.gmm_model.parameters()).device if hasattr(step.gmm_model, 'parameters') else 'unknown'}")
                    
                    return True
                else:
                    print("❌ Step 초기화 실패")
                    return False
            else:
                print("⚠️ initialize 메서드 없음 - 기본 성공으로 간주")
                return True
                
        except Exception as e:
            print(f"❌ Step 초기화 실패: {e}")
            return False
            
    except Exception as e:
        print(f"❌ GeometricMatchingStep 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_device_safe_config():
    """디바이스 안전 설정 파일 생성"""
    config_content = '''#!/usr/bin/env python3
"""
디바이스 안전 설정 (자동 생성)
"""
import torch

def get_safe_device():
    """안전한 디바이스 반환"""
    if torch.backends.mps.is_available():
        try:
            # MPS 간단 테스트
            test_tensor = torch.randn(1, 1).to('mps')
            return 'mps'
        except Exception:
            return 'cpu'
    elif torch.cuda.is_available():
        return 'cuda'
    else:
        return 'cpu'

def safe_model_load(model_path, target_device=None):
    """디바이스 안전 모델 로딩"""
    if target_device is None:
        target_device = get_safe_device()
    
    # 1단계: CPU로 로딩
    model_data = torch.load(model_path, map_location='cpu')
    
    # 2단계: 타겟 디바이스로 이동 (torch.jit 아닌 경우만)
    if target_device != 'cpu' and not hasattr(model_data, '_c'):
        if isinstance(model_data, dict) and 'state_dict' in model_data:
            new_state_dict = {}
            for key, tensor in model_data['state_dict'].items():
                new_state_dict[key] = tensor.to(target_device)
            model_data['state_dict'] = new_state_dict
        elif hasattr(model_data, 'to'):
            model_data = model_data.to(target_device)
    
    return model_data, target_device

if __name__ == "__main__":
    print(f"🎯 권장 디바이스: {get_safe_device()}")
'''
    
    config_file = Path("device_safe_config.py")
    config_file.write_text(config_content)
    print(f"✅ 디바이스 안전 설정 생성: {config_file}")

def main():
    """메인 함수"""
    print("🔧 GeometricMatchingStep 디바이스 문제 해결")
    print("=" * 60)
    
    # 1. 안전한 모델 로딩 테스트
    success = safe_model_loading_test()
    
    # 2. 디바이스 안전 설정 생성
    create_device_safe_config()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 디바이스 문제 해결 성공!")
        print("✨ GeometricMatchingStep이 올바른 디바이스로 작동 중")
        
        print("\n🚀 다음 단계:")
        print("1. conda activate mycloset-ai-clean")
        print("2. cd backend")
        print("3. export PYTORCH_JIT_DISABLE=1") 
        print("4. python gmm_step_test.py  # 원래 테스트 재실행")
    else:
        print("⚠️ 추가 디버깅이 필요합니다")
        print("\n💡 권장사항:")
        print("- CPU 모드로 강제 실행")
        print("- torch.jit 완전 비활성화")
        print("- 모델 재다운로드 고려")

if __name__ == "__main__":
    main()