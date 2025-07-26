#!/usr/bin/env python3
"""
올바른 경로로 GMM 테스트
"""
import os
import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path("/Users/gimdudeul/MVP/mycloset-ai")
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
    
    print(f"📁 AI 모델 루트: {ai_models_root}")
    print(f"📄 GMM 파일: {gmm_path}")
    print(f"📊 GMM 파일 존재: {gmm_path.exists()}")
    
    if gmm_path.exists():
        size_mb = gmm_path.stat().st_size / (1024*1024)
        print(f"📏 GMM 파일 크기: {size_mb:.1f}MB")
        
        # PyTorch로 직접 로딩 테스트
        try:
            import torch
            print(f"✅ PyTorch {torch.__version__} 로드 성공")
            
            device = "mps" if torch.backends.mps.is_available() else "cpu"
            print(f"🖥️ 디바이스: {device}")
            
            # 직접 모델 로딩
            model_data = torch.load(gmm_path, map_location=device)
            print(f"✅ GMM 모델 로딩 성공: {type(model_data)}")
            
            # GeometricMatchingStep import 테스트
            try:
                from app.ai_pipeline.steps.step_04_geometric_matching import GeometricMatchingStep
                print("✅ GeometricMatchingStep import 성공")
                
                # 올바른 경로로 인스턴스 생성
                step = GeometricMatchingStep(
                    step_id=4,
                    device=device,
                    config={"ai_models_root": str(ai_models_root)}
                )
                print("✅ GeometricMatchingStep 인스턴스 생성 성공")
                
                return True
                
            except Exception as e:
                print(f"❌ GeometricMatchingStep 문제: {e}")
                return False
                
        except Exception as e:
            print(f"❌ PyTorch 로딩 실패: {e}")
            return False
    else:
        print("❌ GMM 파일을 찾을 수 없음")
        return False

if __name__ == "__main__":
    success = test_with_correct_paths()
    if success:
        print("\n🎉 경로 문제 해결 성공!")
    else:
        print("\n⚠️ 추가 디버깅 필요")
