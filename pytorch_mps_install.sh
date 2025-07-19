#!/bin/bash

# ===============================================================
# M3 Max PyTorch MPS 설치 스크립트
# Apple Silicon 최적화를 위한 PyTorch 교체
# ===============================================================

set -e

echo "🍎 M3 Max PyTorch MPS 설치 시작"
echo "==============================="
echo ""

# 현재 환경 확인
echo "🔍 현재 환경 확인:"
echo "  Conda 환경: ${CONDA_DEFAULT_ENV:-'없음'}"
echo "  Python: $(python --version)"
echo "  시스템: $(uname -m)"
echo ""

# mycloset-ai 환경 확인
if [[ "${CONDA_DEFAULT_ENV}" != "mycloset-ai" ]]; then
    echo "❌ mycloset-ai 환경이 활성화되지 않았습니다."
    echo "다음 명령으로 환경을 활성화하세요:"
    echo "conda activate mycloset-ai"
    exit 1
fi

# 1단계: 기존 CPU PyTorch 제거
echo "🗑️ 1단계: 기존 CPU 전용 PyTorch 제거"
echo "현재 설치된 PyTorch 확인:"
conda list | grep -E "(torch|pytorch)"
echo ""

echo "기존 PyTorch 패키지 제거 중..."
conda remove pytorch torchvision torchaudio cpuonly -y 2>/dev/null || true
pip uninstall torch torchvision torchaudio -y 2>/dev/null || true

echo "✅ 기존 PyTorch 제거 완료"
echo ""

# 2단계: MPS 지원 PyTorch 설치
echo "⚡ 2단계: M3 Max MPS 지원 PyTorch 설치"
echo "Apple Silicon 최적화 PyTorch 설치 중..."

# 공식 PyTorch 채널에서 MPS 지원 버전 설치
conda install pytorch::pytorch torchvision torchaudio -c pytorch -y

echo "✅ MPS 지원 PyTorch 설치 완료"
echo ""

# 3단계: MPS 지원 확인
echo "🧪 3단계: MPS 지원 테스트"

python3 -c "
import sys
import torch
import platform

print('=== M3 Max PyTorch MPS 테스트 ===')
print(f'PyTorch 버전: {torch.__version__}')
print(f'Python 버전: {sys.version}')
print(f'시스템: {platform.machine()}')
print()

# MPS 사용 가능 여부
if torch.backends.mps.is_available():
    print('✅ MPS (Metal Performance Shaders) 사용 가능!')
    
    try:
        # MPS 디바이스 테스트
        device = torch.device('mps')
        print(f'✅ MPS 디바이스: {device}')
        
        # 간단한 연산 테스트
        print('🧪 MPS 연산 테스트 중...')
        x = torch.randn(1000, 1000, device=device)
        y = torch.randn(1000, 1000, device=device)
        z = torch.mm(x, y)
        
        print(f'✅ MPS 연산 성공: {z.shape}')
        print(f'✅ 결과 디바이스: {z.device}')
        
        # float16 지원 테스트 (M3 Max 최적화)
        x_fp16 = x.to(torch.float16)
        y_fp16 = y.to(torch.float16)
        z_fp16 = torch.mm(x_fp16, y_fp16)
        print(f'✅ Float16 지원: {z_fp16.dtype}')
        
        # 성능 벤치마크
        import time
        
        # CPU vs MPS 성능 비교
        print()
        print('⚡ 성능 벤치마크:')
        
        # CPU 테스트
        x_cpu = torch.randn(2000, 2000)
        y_cpu = torch.randn(2000, 2000)
        
        start_time = time.time()
        z_cpu = torch.mm(x_cpu, y_cpu)
        cpu_time = time.time() - start_time
        print(f'  CPU 시간: {cpu_time:.4f}초')
        
        # MPS 테스트
        x_mps = torch.randn(2000, 2000, device=device)
        y_mps = torch.randn(2000, 2000, device=device)
        
        start_time = time.time()
        z_mps = torch.mm(x_mps, y_mps)
        torch.mps.synchronize()  # MPS 동기화
        mps_time = time.time() - start_time
        print(f'  MPS 시간: {mps_time:.4f}초')
        
        speedup = cpu_time / mps_time
        print(f'  🚀 MPS 가속비: {speedup:.2f}x')
        
        if speedup > 1.5:
            print('✅ M3 Max GPU 가속이 정상적으로 작동합니다!')
        else:
            print('⚠️ GPU 가속 효과가 제한적입니다.')
        
    except Exception as e:
        print(f'❌ MPS 테스트 실패: {e}')
        sys.exit(1)
        
else:
    print('❌ MPS를 사용할 수 없습니다.')
    print('PyTorch 버전과 macOS 버전을 확인하세요.')
    sys.exit(1)

print()
print('🎉 M3 Max PyTorch MPS 설치 및 테스트 완료!')
print('🔥 이제 AI 모델들이 M3 Max GPU를 활용할 수 있습니다.')
"

if [[ $? -eq 0 ]]; then
    echo ""
    echo "🎊 성공! MPS 지원 PyTorch 설치 완료"
    echo ""
    echo "📊 설치된 패키지 확인:"
    conda list | grep -E "(torch|pytorch)"
    echo ""
    echo "🚀 다음 단계:"
    echo "  1. AI 파이프라인 다시 실행"
    echo "  2. 로그에서 'mps' 디바이스 확인"
    echo "  3. M3 Max GPU 활용 여부 모니터링"
    echo ""
    echo "💡 백엔드 재시작:"
    echo "  python3 app/main.py"
else
    echo ""
    echo "❌ MPS 설치 또는 테스트 실패"
    echo "🔧 문제 해결:"
    echo "  1. macOS 버전 확인 (12.3 이상 필요)"
    echo "  2. Xcode Command Line Tools 설치"
    echo "  3. PyTorch 재설치: conda install pytorch -c pytorch -y"
fi

echo ""
echo "🍎 M3 Max PyTorch MPS 설치 스크립트 완료"