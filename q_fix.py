#!/usr/bin/env python3
"""
🔧 GeometricMatchingStep 경로 문제 즉시 해결
현재 위치: /Users/gimdudeul/MVP/mycloset-ai
실제 파일: ./backend/ai_models/step_04_geometric_matching/gmm_final.pth
"""

import os
import sys
from pathlib import Path

def fix_path_issue():
    """경로 문제 즉시 해결"""
    print("🔧 GeometricMatchingStep 경로 문제 해결 시작")
    print("=" * 60)
    
    # 현재 위치: mycloset-ai 디렉토리
    current_dir = Path.cwd()
    print(f"📍 현재 위치: {current_dir}")
    
    # 실제 파일 경로들
    backend_ai_models = current_dir / "backend" / "ai_models"
    step04_dir = backend_ai_models / "step_04_geometric_matching"
    gmm_file = step04_dir / "gmm_final.pth"
    
    print(f"📁 백엔드 AI 모델 디렉토리: {backend_ai_models}")
    print(f"📁 Step 04 디렉토리: {step04_dir}")
    print(f"📄 GMM 파일: {gmm_file}")
    
    # 파일 존재 확인
    files_to_check = {
        "GMM 모델": gmm_file,
        "TPS 모델": step04_dir / "tps_network.pth", 
        "SAM 모델": backend_ai_models / "step_03_cloth_segmentation" / "sam_vit_h_4b8939.pth",
        "Step 04 디렉토리": step04_dir
    }
    
    print("\n📊 파일 존재 확인:")
    for name, path in files_to_check.items():
        exists = path.exists()
        size = ""
        if exists and path.is_file():
            size_mb = path.stat().st_size / (1024*1024)
            size = f" ({size_mb:.1f}MB)"
        print(f"  {'✅' if exists else '❌'} {name}: {exists}{size}")
    
    # 환경 변수 설정 스크립트 생성
    env_script = current_dir / "set_correct_paths.py"
    
    env_content = f'''#!/usr/bin/env python3
"""
올바른 경로 설정 (자동 생성)
"""
import os
from pathlib import Path

# 올바른 경로들
PROJECT_ROOT = Path("{current_dir.absolute()}")
BACKEND_ROOT = PROJECT_ROOT / "backend"  
AI_MODELS_ROOT = BACKEND_ROOT / "ai_models"

# Step 04 모델 경로들  
STEP04_MODELS = {{
    "gmm_final": AI_MODELS_ROOT / "step_04_geometric_matching" / "gmm_final.pth",
    "tps_network": AI_MODELS_ROOT / "step_04_geometric_matching" / "tps_network.pth",
    "sam_shared": AI_MODELS_ROOT / "step_03_cloth_segmentation" / "sam_vit_h_4b8939.pth"
}}

def get_step04_model_path(model_name: str):
    """Step 04 모델 경로 반환"""
    return str(STEP04_MODELS.get(model_name, ""))

def patch_geometric_matching_step():
    """GeometricMatchingStep 경로 패치"""
    import sys
    sys.path.insert(0, str(BACKEND_ROOT))
    
    # 환경 변수로 올바른 경로 설정
    os.environ["AI_MODELS_ROOT"] = str(AI_MODELS_ROOT)
    os.environ["STEP04_GMM_PATH"] = str(STEP04_MODELS["gmm_final"])
    os.environ["STEP04_TPS_PATH"] = str(STEP04_MODELS["tps_network"])
    
    print(f"✅ AI_MODELS_ROOT: {{AI_MODELS_ROOT}}")
    print(f"✅ GMM 경로: {{STEP04_MODELS['gmm_final']}}")
    
if __name__ == "__main__":
    patch_geometric_matching_step()
    
    # 파일 존재 확인
    print("\\n📊 모델 파일 확인:")
    for name, path in STEP04_MODELS.items():
        exists = Path(path).exists()
        print(f"  {{'✅' if exists else '❌'}} {{name}}: {{path}}")
'''
    
    env_script.write_text(env_content)
    print(f"\n✅ 경로 설정 스크립트 생성: {env_script}")
    
    # 테스트 스크립트 생성
    test_script = current_dir / "test_gmm_with_correct_paths.py"
    
    test_content = f'''#!/usr/bin/env python3
"""
올바른 경로로 GMM 테스트
"""
import os
import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path("{current_dir.absolute()}")
backend_root = project_root / "backend"
sys.path.insert(0, str(backend_root))

# 올바른 경로 설정
os.environ["AI_MODELS_ROOT"] = str(backend_root / "ai_models")
os.environ["PYTORCH_JIT_DISABLE"] = "1"

def test_with_correct_paths():
    """올바른 경로로 테스트"""
    print("🧪 올바른 경로로 GMM 테스트 시작")
    print("=" * 50)
    
    # 경로 확인
    ai_models_root = Path(os.environ["AI_MODELS_ROOT"])
    gmm_path = ai_models_root / "step_04_geometric_matching" / "gmm_final.pth"
    
    print(f"📁 AI 모델 루트: {{ai_models_root}}")
    print(f"📄 GMM 파일: {{gmm_path}}")
    print(f"📊 GMM 파일 존재: {{gmm_path.exists()}}")
    
    if gmm_path.exists():
        size_mb = gmm_path.stat().st_size / (1024*1024)
        print(f"📏 GMM 파일 크기: {{size_mb:.1f}}MB")
        
        # PyTorch로 직접 로딩 테스트
        try:
            import torch
            print(f"✅ PyTorch {{torch.__version__}} 로드 성공")
            
            device = "mps" if torch.backends.mps.is_available() else "cpu"
            print(f"🖥️ 디바이스: {{device}}")
            
            # 직접 모델 로딩
            model_data = torch.load(gmm_path, map_location=device)
            print(f"✅ GMM 모델 로딩 성공: {{type(model_data)}}")
            
            # GeometricMatchingStep import 테스트
            try:
                from app.ai_pipeline.steps.step_04_geometric_matching import GeometricMatchingStep
                print("✅ GeometricMatchingStep import 성공")
                
                # 올바른 경로로 인스턴스 생성
                step = GeometricMatchingStep(
                    step_id=4,
                    device=device,
                    config={{"ai_models_root": str(ai_models_root)}}
                )
                print("✅ GeometricMatchingStep 인스턴스 생성 성공")
                
                return True
                
            except Exception as e:
                print(f"❌ GeometricMatchingStep 문제: {{e}}")
                return False
                
        except Exception as e:
            print(f"❌ PyTorch 로딩 실패: {{e}}")
            return False
    else:
        print("❌ GMM 파일을 찾을 수 없음")
        return False

if __name__ == "__main__":
    success = test_with_correct_paths()
    if success:
        print("\\n🎉 경로 문제 해결 성공!")
    else:
        print("\\n⚠️ 추가 디버깅 필요")
'''
    
    test_script.write_text(test_content)
    print(f"✅ 테스트 스크립트 생성: {test_script}")
    
    print("\n🚀 다음 단계:")
    print("1. python set_correct_paths.py      # 경로 확인")  
    print("2. python test_gmm_with_correct_paths.py  # 올바른 경로로 테스트")
    
    return True

if __name__ == "__main__":
    fix_path_issue()