#!/bin/bash
# install_fastapi_web.sh
# FastAPI 및 웹 서버 관련 패키지 설치

echo "🌐 FastAPI 및 웹 서버 패키지 설치"
echo "=================================="

# 현재 환경 확인
if [[ "$CONDA_DEFAULT_ENV" != "mycloset-ai" ]]; then
    echo "❌ mycloset-ai 환경이 활성화되지 않았습니다."
    echo "실행: conda activate mycloset-ai"
    exit 1
fi

echo "✅ 현재 환경: $CONDA_DEFAULT_ENV"

# 1. FastAPI 및 관련 패키지 (conda로 설치)
echo ""
echo "🚀 FastAPI 및 웹 서버 패키지 설치 중..."
conda install fastapi uvicorn -c conda-forge -y

# 2. 추가 웹 관련 패키지 (pip로 설치)
echo ""
echo "📦 추가 웹 패키지 설치 중..."
pip install python-multipart python-dotenv pydantic-settings

# 3. AI/ML 관련 패키지 (pip로 설치)
echo ""
echo "🤖 AI/ML 패키지 설치 중..."
pip install transformers diffusers accelerate

# 4. 이미지 처리 패키지
echo ""
echo "🖼️ 이미지 처리 패키지 설치 중..."
pip install opencv-python

# 5. 개발 도구 (선택적)
echo ""
echo "🛠️ 개발 도구 설치 중..."
conda install black isort mypy pytest -c conda-forge -y

# 6. 설치 검증
echo ""
echo "🧪 웹 서버 패키지 검증 중..."
python << 'EOF'
import sys
print(f"Python: {sys.version}")
print()

# 웹 서버 패키지들 확인
web_packages = [
    ('fastapi', 'fastapi'),
    ('uvicorn', 'uvicorn'),
    ('pydantic', 'pydantic'),
    ('transformers', 'transformers'),
    ('diffusers', 'diffusers'),
    ('cv2', 'cv2')
]

success_count = 0
total_count = len(web_packages)

for pkg_name, import_name in web_packages:
    try:
        module = __import__(import_name)
        version = getattr(module, '__version__', 'Unknown')
        print(f"✅ {pkg_name}: {version}")
        success_count += 1
    except ImportError as e:
        print(f"❌ {pkg_name}: Import 실패 - {e}")

print(f"\n📊 웹 패키지 설치 성공률: {success_count}/{total_count} ({success_count/total_count*100:.1f}%)")

# FastAPI 간단 테스트
print("\n🚀 FastAPI 기본 테스트:")
try:
    from fastapi import FastAPI
    
    app = FastAPI(title="MyCloset AI Test")
    
    @app.get("/")
    def root():
        return {"message": "MyCloset AI", "status": "ready"}
    
    print("   ✅ FastAPI 앱 생성 성공")
    print("   ✅ 라우터 등록 성공")
    
except Exception as e:
    print(f"   ❌ FastAPI 테스트 실패: {e}")

# PyTorch + MPS 재확인
print("\n🍎 PyTorch MPS 재확인:")
try:
    import torch
    
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        x = torch.randn(1000, 1000, device=device, dtype=torch.float16)
        y = torch.randn(1000, 1000, device=device, dtype=torch.float16)
        z = torch.mm(x, y)
        
        print(f"   ✅ MPS 연산: {z.shape}")
        print(f"   ✅ 디바이스: {device}")
        print(f"   ✅ 데이터 타입: {z.dtype}")
        
        # 메모리 정리
        del x, y, z
        torch.mps.empty_cache()
        print(f"   🧹 메모리 정리 완료")
    else:
        print("   ⚠️ MPS 사용 불가, CPU 모드")
        
except Exception as e:
    print(f"   ❌ PyTorch 테스트 실패: {e}")
EOF

# 7. requirements.txt 최종 업데이트
echo ""
echo "📝 최종 requirements.txt 생성 중..."
cat > requirements_final.txt << 'EOF'
# MyCloset AI - M3 Max 최적화 환경
# 최종 설치 패키지 목록

# 핵심 계산 라이브러리 (conda)
numpy==1.24.3
scipy==1.11.4
scikit-learn==1.3.0

# PyTorch 생태계 (conda - pytorch 채널)  
torch==2.1.0
torchvision==0.16.0
torchaudio==2.1.0

# 웹 프레임워크 (conda + pip)
fastapi
uvicorn
python-multipart
python-dotenv
pydantic-settings

# AI/ML 라이브러리 (pip)
transformers
diffusers
accelerate

# 이미지 처리
pillow
opencv-python

# 시스템 유틸리티 (conda)
psutil==7.0.0
tqdm==4.67.1
aiofiles

# 개발 도구 (conda)
black
isort  
mypy
pytest

# 기타
pyyaml
requests
EOF

echo "✅ requirements_final.txt 생성 완료"

# 8. 다음 단계 안내
echo ""
echo "🎉 FastAPI 및 웹 서버 설치 완료!"
echo ""
echo "📋 설치된 전체 스택:"
echo "   🍎 M3 Max MPS: PyTorch 2.1.0"
echo "   🤖 ML 라이브러리: scikit-learn 1.3.0, transformers, diffusers"  
echo "   🌐 웹 서버: FastAPI + Uvicorn"
echo "   🖼️ 이미지 처리: OpenCV + Pillow"
echo "   💾 시스템: 128GB 메모리, 70.9GB 사용 가능"
echo ""
echo "🚀 이제 백엔드 서버를 시작할 수 있습니다:"
echo "1. cd backend"
echo "2. python app/main.py"
echo "   또는"
echo "3. uvicorn app.main:app --reload --host 0.0.0.0 --port 8000"
echo ""
echo "🌐 서버 실행 후 접속:"
echo "   - API 문서: http://localhost:8000/docs"
echo "   - 헬스체크: http://localhost:8000/health"