#!/bin/bash

# PyTorch M3 Max 최적화 설치 스크립트
echo "🍎 PyTorch M3 Max 최적화 설치"
echo "============================="

# 현재 conda 환경 확인
if [[ "${CONDA_DEFAULT_ENV}" != "mycloset-ai" ]]; then
    echo "❌ mycloset-ai 환경이 활성화되지 않았습니다."
    echo "다음 명령으로 환경을 활성화하세요:"
    echo "conda activate mycloset-ai"
    exit 1
fi

echo "✅ 현재 환경: ${CONDA_DEFAULT_ENV}"
echo "✅ Python: $(python --version)"

# 1단계: 기존 PyTorch 관련 패키지 완전 제거
echo ""
echo "=== 1단계: 기존 PyTorch 패키지 제거 ==="
packages_to_remove=(
    "torch"
    "torchvision" 
    "torchaudio"
    "pytorch"
    "pytorch-cuda"
)

for package in "${packages_to_remove[@]}"; do
    echo "🗑️ 제거 중: $package"
    conda remove $package -y 2>/dev/null || true
    pip uninstall $package -y 2>/dev/null || true
done

# pip 캐시 정리
pip cache purge

echo "✅ 기존 PyTorch 패키지 제거 완료"

# 2단계: M3 Max 최적화 PyTorch 설치
echo ""
echo "=== 2단계: M3 Max 최적화 PyTorch 설치 ==="

# Apple Silicon용 PyTorch 설치 (공식 채널)
echo "🍎 Apple Silicon 최적화 PyTorch 설치 중..."
conda install pytorch torchvision torchaudio -c pytorch -y

# 설치 확인
echo ""
echo "=== 3단계: PyTorch 설치 확인 ==="
python3 -c "
import sys
print('=== PyTorch 설치 확인 ===')

try:
    import torch
    print(f'✅ PyTorch 버전: {torch.__version__}')
    
    # MPS 지원 확인
    if torch.backends.mps.is_available():
        print('✅ M3 Max MPS 지원: 사용 가능')
        device = torch.device('mps')
        
        # 간단한 연산 테스트
        x = torch.randn(100, 100, device=device)
        y = torch.mm(x, x.T)
        print(f'✅ MPS 연산 테스트: {y.shape}')
        
        # float16 테스트 (M3 Max 최적화)
        x_fp16 = x.to(torch.float16)
        print('✅ float16 지원: 정상')
        
    else:
        print('⚠️ MPS 지원: 사용 불가 (CPU 모드)')
    
    # torchvision 확인
    import torchvision
    print(f'✅ torchvision 버전: {torchvision.__version__}')
    
except ImportError as e:
    print(f'❌ PyTorch import 실패: {e}')
    sys.exit(1)
except Exception as e:
    print(f'❌ PyTorch 테스트 실패: {e}')
    sys.exit(1)

print('')
print('🎉 PyTorch M3 Max 설치 및 테스트 완료!')
"

# 4단계: 추가 AI 라이브러리 설치
echo ""
echo "=== 4단계: 추가 AI 라이브러리 설치 ==="

# HuggingFace 생태계
echo "🤗 HuggingFace 라이브러리 설치 중..."
pip install transformers==4.35.0
pip install diffusers==0.21.4
pip install accelerate==0.24.1

# 이미지 처리
echo "🖼️ 이미지 처리 라이브러리 설치 중..."
pip install opencv-python==4.8.1.78
pip install Pillow==10.1.0

# 과학 계산
echo "🔬 과학 계산 라이브러리 설치 중..."
conda install scipy=1.11.4 scikit-learn=1.3.0 -c conda-forge -y

echo "✅ 추가 라이브러리 설치 완료"

# 5단계: 최종 테스트
echo ""
echo "=== 5단계: 통합 테스트 ==="
python3 -c "
print('=== 통합 라이브러리 테스트 ===')

# 핵심 라이브러리들
libraries = [
    ('torch', 'PyTorch'),
    ('torchvision', 'TorchVision'),
    ('transformers', 'Transformers'),
    ('diffusers', 'Diffusers'),
    ('cv2', 'OpenCV'),
    ('PIL', 'Pillow'),
    ('numpy', 'NumPy'),
    ('scipy', 'SciPy')
]

success_count = 0
for lib, name in libraries:
    try:
        module = __import__(lib)
        version = getattr(module, '__version__', 'Unknown')
        print(f'✅ {name}: {version}')
        success_count += 1
    except ImportError:
        print(f'❌ {name}: Import 실패')

print(f'')
print(f'📊 설치 성공률: {success_count}/{len(libraries)} ({success_count/len(libraries)*100:.1f}%)')

# M3 Max 최적화 확인
print('')
print('=== M3 Max 최적화 확인 ===')
try:
    import torch
    import psutil
    
    # 시스템 정보
    memory = psutil.virtual_memory()
    print(f'💾 총 메모리: {memory.total / (1024**3):.1f}GB')
    print(f'💾 사용 가능: {memory.available / (1024**3):.1f}GB')
    
    # GPU 정보
    if torch.backends.mps.is_available():
        print('🍎 M3 Max GPU: 사용 가능')
        print('⚡ Metal Performance Shaders: 활성화')
        
        # 권장 설정
        print('')
        print('💡 권장 설정:')
        print('  - 배치 크기: 8-16 (대용량 메모리 활용)')
        print('  - 정밀도: float16 (M3 Max 최적화)')
        print('  - 디바이스: mps')
    else:
        print('⚠️ MPS 사용 불가 - CPU 모드로 실행')
        
except Exception as e:
    print(f'❌ M3 Max 테스트 실패: {e}')
"

echo ""
echo "🎉 PyTorch M3 Max 최적화 설치 완료!"
echo ""
echo "📋 설치된 구성요소:"
echo "  🍎 PyTorch (M3 Max MPS 지원)"
echo "  🖼️ TorchVision"
echo "  🤗 Transformers & Diffusers"
echo "  📊 과학 계산 라이브러리"
echo ""
echo "🚀 다음 단계:"
echo "1. 서버 재시작: python app/main.py"
echo "2. MPS 디바이스 확인: 로그에서 'mps' 확인"
echo "3. API 테스트: curl http://localhost:8000/health"
echo ""

# 환경변수 설정 (선택적)
echo "💡 성능 최적화를 위한 환경변수 설정 (선택적):"
echo "export PYTORCH_ENABLE_MPS_FALLBACK=1"
echo "export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0"
