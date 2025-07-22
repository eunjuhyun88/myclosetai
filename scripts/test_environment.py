# test_environment.py
# MyCloset AI - 환경 검증 및 테스트 스크립트

import sys
import os
import warnings
from pathlib import Path

print("🧪 MyCloset AI - 환경 검증 및 테스트")
print("=" * 50)

# 1. 기본 환경 정보
print("📊 기본 환경 정보:")
print(f"   Python: {sys.version}")
print(f"   conda 환경: {os.environ.get('CONDA_DEFAULT_ENV', 'None')}")
print(f"   현재 경로: {os.getcwd()}")

# 2. conda 환경 검증
conda_env = os.environ.get('CONDA_DEFAULT_ENV')
if conda_env == 'mycloset-ai':
    print("✅ conda 환경: mycloset-ai 활성화됨")
elif conda_env:
    print(f"⚠️  conda 환경: {conda_env} (mycloset-ai 아님)")
else:
    print("❌ conda 환경 활성화되지 않음")

# 3. 핵심 라이브러리 테스트 (libjpeg 경고 포함)
print("\n🔍 핵심 라이브러리 테스트:")

# PIL/Pillow
try:
    from PIL import Image
    print("✅ PIL/Pillow: 정상")
    
    # 이미지 생성 테스트
    test_img = Image.new('RGB', (100, 100), 'red')
    print("   - 이미지 생성: 정상")
except Exception as e:
    print(f"❌ PIL/Pillow: {e}")

# NumPy
try:
    import numpy as np
    print(f"✅ NumPy: {np.__version__}")
except Exception as e:
    print(f"❌ NumPy: {e}")

# PyTorch (MPS 지원 확인)
try:
    import torch
    print(f"✅ PyTorch: {torch.__version__}")
    print(f"   - MPS 사용 가능: {torch.backends.mps.is_available()}")
    print(f"   - CUDA 사용 가능: {torch.cuda.is_available()}")
except Exception as e:
    print(f"❌ PyTorch: {e}")

# torchvision (libjpeg 경고 캡처)
print("\n🔥 torchvision 상세 테스트 (libjpeg 경고 확인):")
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    
    try:
        import torchvision
        print(f"✅ torchvision: {torchvision.__version__}")
        
        # 이미지 IO 모듈 로드
        import torchvision.io
        print("✅ torchvision.io: 정상")
        
        # 변환 모듈 테스트
        import torchvision.transforms as transforms
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224))
        ])
        
        # 실제 변환 테스트
        test_image = Image.new('RGB', (100, 100), 'blue')
        tensor = transform(test_image)
        print(f"✅ 이미지 변환: {tensor.shape}")
        
        # 경고 분석
        libjpeg_warnings = [warn for warn in w if 'libjpeg' in str(warn.message).lower()]
        if libjpeg_warnings:
            print(f"⚠️  libjpeg 경고 발생: {len(libjpeg_warnings)}개")
            print("   경고 내용:")
            for warn in libjpeg_warnings[:2]:  # 처음 2개만 표시
                print(f"   - {str(warn.message)[:100]}...")
        else:
            print("🎉 libjpeg 경고 없음 - 완전 해결!")
            
    except Exception as e:
        print(f"❌ torchvision: {e}")

# 4. FastAPI 및 웹 프레임워크
print("\n🌐 웹 프레임워크 테스트:")
try:
    import fastapi
    print(f"✅ FastAPI: {fastapi.__version__}")
except Exception as e:
    print(f"❌ FastAPI: {e}")

try:
    import uvicorn
    print(f"✅ Uvicorn: {uvicorn.__version__}")
except Exception as e:
    print(f"❌ Uvicorn: {e}")

# 5. MyCloset AI 서비스 로딩 테스트
print("\n🎯 MyCloset AI 서비스 테스트:")

# 프로젝트 경로 확인
project_root = Path.cwd()
backend_path = project_root / "backend"
if backend_path.exists():
    sys.path.insert(0, str(backend_path))
    print(f"✅ 백엔드 경로: {backend_path}")
else:
    print(f"❌ 백엔드 경로 찾을 수 없음: {backend_path}")

# 서비스 import 테스트
try:
    from app.services import get_service_status
    status = get_service_status()
    print("✅ 서비스 로딩: 성공")
    
    print("📊 서비스 상태:")
    print(f"   - conda 환경: {status['conda_environment']['active']}")
    print(f"   - M3 Max 최적화: {status['conda_environment']['m3_max_optimized']}")
    print(f"   - step_service: {status['services']['step_service']}")
    print(f"   - 총 서비스 수: {status['total_available_services']}")
    
except Exception as e:
    print(f"⚠️  서비스 로딩 실패: {e}")
    print("   services/__init__.py 파일을 확인해주세요")

# 6. AI 모델 경로 확인
print("\n🤖 AI 모델 경로 확인:")
ai_models_path = project_root / "ai_models"
if ai_models_path.exists():
    print(f"✅ AI 모델 경로: {ai_models_path}")
    
    # 모델 파일들 확인
    model_files = list(ai_models_path.rglob("*.pth")) + list(ai_models_path.rglob("*.ckpt"))
    print(f"   - 발견된 모델 파일: {len(model_files)}개")
    
    if model_files:
        total_size = sum(f.stat().st_size for f in model_files) / (1024**3)
        print(f"   - 총 모델 크기: {total_size:.1f}GB")
        
        for model_file in model_files[:3]:  # 처음 3개만 표시
            size_mb = model_file.stat().st_size / (1024**2)
            print(f"   - {model_file.name}: {size_mb:.1f}MB")
    
else:
    print(f"❌ AI 모델 경로 없음: {ai_models_path}")

# 7. 메모리 및 시스템 정보
print("\n💾 시스템 정보:")
try:
    import psutil
    memory = psutil.virtual_memory()
    print(f"✅ 총 메모리: {memory.total / (1024**3):.1f}GB")
    print(f"   - 사용 가능: {memory.available / (1024**3):.1f}GB")
    print(f"   - 사용률: {memory.percent:.1f}%")
except Exception as e:
    print(f"⚠️  시스템 정보 확인 실패: {e}")

# 8. 최종 권장사항
print("\n🎯 최종 검증 결과 및 권장사항:")
print("=" * 50)

# conda 환경 체크
if conda_env == 'mycloset-ai':
    print("✅ conda 환경 설정 완료")
else:
    print("❌ conda activate mycloset-ai 실행 필요")

# libjpeg 체크
if 'libjpeg_warnings' in locals() and not libjpeg_warnings:
    print("✅ libjpeg 문제 해결 완료")
elif 'libjpeg_warnings' in locals():
    print("⚠️  libjpeg 수정 스크립트 실행 권장")
else:
    print("🔍 libjpeg 상태 확인 필요")

# 서비스 체크
try:
    if 'status' in locals() and status['services']['step_service']:
        print("✅ MyCloset AI 서비스 준비 완료")
    else:
        print("⚠️  서비스 설정 확인 필요")
except:
    print("❌ 서비스 초기화 필요")

print("\n🚀 다음 단계:")
if conda_env == 'mycloset-ai':
    print("1. 백엔드 서버 실행: cd backend && python main.py")
    print("2. API 테스트: http://localhost:8000/docs")
else:
    print("1. conda activate mycloset-ai")
    print("2. libjpeg 수정 스크립트 실행")
    print("3. 서비스 설정 확인")

print("\n🎉 환경 검증 완료!")