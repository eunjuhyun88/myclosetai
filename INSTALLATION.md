# 🚀 MyCloset AI 설치 가이드

> 🍎 **M3 Max 최적화** AI 가상 피팅 시스템 완전 설치 가이드

## 📋 목차

1. [시스템 요구사항](#-시스템-요구사항)
2. [빠른 설치 (권장)](#-빠른-설치-권장)
3. [수동 설치](#-수동-설치)
4. [AI 모델 설치](#-ai-모델-설치)
5. [실행 방법](#-실행-방법)
6. [트러블슈팅](#-트러블슈팅)

---

## 🛠️ 시스템 요구사항

### 🔥 최소 요구사항
- **OS**: macOS 12+, Ubuntu 20.04+, Windows 10+
- **메모리**: 16GB RAM
- **저장공간**: 20GB 여유 공간
- **Python**: 3.10.18
- **Node.js**: 18.0.0+

### 🍎 권장 환경 (M3 Max 최적화)
- **Mac**: Apple M3 Max (128GB 통합 메모리)
- **메모리**: 128GB RAM
- **저장공간**: 100GB+ SSD
- **conda**: miniconda3 또는 anaconda

### 🐧 Linux/Windows
- **GPU**: NVIDIA RTX 4090 또는 동급 이상
- **VRAM**: 24GB+ 
- **CUDA**: 12.1+

---

## ⚡ 빠른 설치 (권장)

### 1️⃣ Repository 복제
```bash
git clone https://github.com/eunjuhyun88/mycloset-ai.git
cd mycloset-ai
```

### 2️⃣ 자동 설치 실행
```bash
# 🍎 M3 Max 사용자
python setup.py install --user-install

# 🐧 Linux/Windows 사용자  
python setup.py install --conda-env=mycloset-ai
```

### 3️⃣ 설치 확인
```bash
# conda 환경 활성화
conda activate mycloset-ai-clean

# 설치 검증
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'MPS Available: {torch.backends.mps.is_available()}')"
```

---

## 🔧 수동 설치

자동 설치가 실패하거나 커스텀 설치가 필요한 경우:

### 1️⃣ conda 환경 생성
```bash
# environment.yml로 환경 생성
conda env create -f environment.yml

# 환경 활성화
conda activate mycloset-ai-clean
```

### 2️⃣ 백엔드 패키지 설치
```bash
cd backend
pip install -r requirements.txt
```

### 3️⃣ 프론트엔드 패키지 설치
```bash
cd ../frontend
npm install
```

### 4️⃣ 환경 변수 설정
```bash
# .env 파일 생성
cp .env.example .env

# 설정 수정 (경로, API 키 등)
nano .env
```

---

## 🤖 AI 모델 설치

### 자동 다운로드 (권장)
```bash
# 필수 모델만 (약 1.2GB)
python install_models.py --essential

# 모든 모델 (약 10GB)
python install_models.py --all

# 특정 모델만
python install_models.py --model human_parsing_schp
```

### 수동 다운로드
필요한 경우 Hugging Face에서 직접 다운로드:
```bash
# Human Parsing 모델
huggingface-cli download mattmdjaga/segformer_b2_clothes --local-dir backend/ai_models/human_parsing/

# OOTDiffusion 모델  
huggingface-cli download levihsu/OOTDiffusion --local-dir backend/ai_models/virtual_fitting/
```

### 모델 설치 확인
```bash
python install_models.py --check
```

---

## 🚀 실행 방법

### 1️⃣ 백엔드 서버 실행
```bash
# conda 환경 활성화
conda activate mycloset-ai-clean

# 백엔드 실행
cd backend
python app/main.py
```

### 2️⃣ 프론트엔드 개발 서버 실행
```bash
# 새 터미널에서
cd frontend
npm run dev
```

### 3️⃣ 브라우저에서 접속
- 프론트엔드: http://localhost:5173
- 백엔드 API: http://localhost:8000
- API 문서: http://localhost:8000/docs

---

## 🔍 설치 검증

### 시스템 정보 확인
```bash
# Python 환경 확인
python --version
conda info --envs

# PyTorch 설정 확인
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA Available: {torch.cuda.is_available()}')
print(f'MPS Available: {torch.backends.mps.is_available() if hasattr(torch.backends, \"mps\") else False}')
print(f'Device: {\"mps\" if torch.backends.mps.is_available() else \"cuda\" if torch.cuda.is_available() else \"cpu\"}')
"
```

### FastAPI 서버 테스트
```bash
# 헬스체크
curl http://localhost:8000/health

# 시스템 정보
curl http://localhost:8000/api/health/system
```

### AI 파이프라인 테스트
```bash
# 테스트 이미지로 파이프라인 실행
python -c "
from app.services.ai_pipeline import PipelineService
service = PipelineService()
print('AI Pipeline Ready:', service.is_ready())
"
```

---

## 🚨 트러블슈팅

### 🐍 conda 관련 문제

#### conda 환경 생성 실패
```bash
# conda 업데이트
conda update conda

# 환경 강제 재생성
conda env remove -n mycloset-ai-clean
conda env create -f environment.yml
```

#### conda 환경 활성화 안됨
```bash
# conda 초기화
conda init bash  # 또는 zsh

# 새 터미널에서 재시도
conda activate mycloset-ai-clean
```

### 🔥 PyTorch/MPS 관련 문제

#### M3 Max MPS 미지원 오류
```bash
# PyTorch nightly 버전 설치
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cpu

# 환경 변수 설정
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

#### CUDA 관련 오류 (Linux/Windows)
```bash
# CUDA 버전 확인
nvidia-smi

# PyTorch CUDA 버전 재설치
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 📦 패키지 설치 문제

#### pip 의존성 충돌
```bash
# pip 캐시 정리
pip cache purge

# 강제 재설치
pip install --force-reinstall --no-deps torch
```

#### npm 설치 실패
```bash
# npm 캐시 정리
npm cache clean --force

# node_modules 삭제 후 재설치
rm -rf node_modules package-lock.json
npm install
```

### 🤖 AI 모델 다운로드 문제

#### Hugging Face 다운로드 실패
```bash
# 토큰 설정 (계정이 있는 경우)
huggingface-cli login

# 다시 시도
python install_models.py --essential --force
```

#### 디스크 공간 부족
```bash
# 불필요한 파일 정리
conda clean --all
pip cache purge

# 임시 파일 정리
rm -rf ~/.cache/huggingface/
```

### 🌐 서버 실행 문제

#### 포트 충돌
```bash
# 다른 포트로 실행
PORT=8080 python app/main.py

# 또는 .env 파일 수정
echo "PORT=8080" >> .env
```

#### 메모리 부족 오류
```bash
# 메모리 사용량 확인
python -c "
import psutil
print(f'RAM: {psutil.virtual_memory().percent}%')
print(f'Available: {psutil.virtual_memory().available / 1024**3:.1f}GB')
"

# .env에서 메모리 설정 조정
MEMORY_FRACTION=0.5
MAX_MEMORY_GB=16.0
```

### 🔧 개발 환경 문제

#### VS Code IntelliSense 안됨
```bash
# Python 인터프리터 설정
# Cmd+Shift+P → "Python: Select Interpreter"
# conda 환경의 python 선택: ~/miniconda3/envs/mycloset-ai-clean/bin/python
```

#### 환경 변수 인식 안됨
```bash
# .env 파일 위치 확인
ls -la .env

# 수동으로 환경 변수 설정
export PYTHONPATH=$PWD/backend
export DEVICE=mps
```

---

## 📞 지원 및 문의

### 🐛 버그 리포트
- GitHub Issues: https://github.com/eunjuhyun88/mycloset-ai/issues

### 💬 커뮤니티 지원
- Discord: [MyCloset AI Community](https://discord.gg/mycloset-ai)
- Reddit: r/MyClosetAI

### 📚 추가 문서
- [API 문서](./docs/API.md)
- [아키텍처 가이드](./docs/ARCHITECTURE.md)
- [개발자 가이드](./docs/DEVELOPMENT.md)

---

## 🎯 다음 단계

설치가 완료되었다면:

1. **[사용법 가이드](./docs/USAGE.md)** 확인
2. **[예제 이미지](./examples/)** 로 테스트
3. **[API 문서](http://localhost:8000/docs)** 탐색
4. **[커스터마이징 가이드](./docs/CUSTOMIZATION.md)** 읽기

---

> 💡 **팁**: 설치 중 문제가 발생하면 `python setup.py develop --conda-env=mycloset-ai-clean` 명령어로 개발 모드로 설치해보세요.

> 🍎 **M3 Max 사용자**: 최적의 성능을 위해 메모리 설정을 `MEMORY_FRACTION=0.9`로 설정하세요.