#!/bin/bash

# =============================================================================
# MyCloset AI - 프로젝트 폴더 완전 재정리 스크립트
# 현재 복잡한 루트 디렉토리를 깔끔하게 정리
# =============================================================================

set -e

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m'

log_info() { echo -e "${BLUE}ℹ️  $1${NC}"; }
log_success() { echo -e "${GREEN}✅ $1${NC}"; }
log_warning() { echo -e "${YELLOW}⚠️  $1${NC}"; }
log_error() { echo -e "${RED}❌ $1${NC}"; }
log_header() { echo -e "${PURPLE}🚀 $1${NC}"; }

log_header "MyCloset AI 프로젝트 폴더 재정리"
echo "=================================================================="
log_info "작업 디렉토리: $(pwd)"
log_info "정리 시작 시간: $(date)"
echo ""

# 1. 백업 생성
log_info "1. 안전을 위한 백업 생성 중..."
backup_dir="backup_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$backup_dir"

# 중요한 설정 파일들만 백업
if [ -f ".env" ]; then cp .env "$backup_dir/"; fi
if [ -f "Makefile" ]; then cp Makefile "$backup_dir/"; fi
if [ -f "structure.txt" ]; then cp structure.txt "$backup_dir/"; fi

log_success "백업 생성 완료: $backup_dir"

# 2. scripts 디렉토리 정리
log_info "2. scripts 디렉토리 정리 중..."

# scripts 하위 디렉토리 생성
mkdir -p scripts/{setup,dev,maintenance,models,utils}

# 설정 관련 스크립트들 이동
mv setup_conda_env.sh scripts/setup/ 2>/dev/null || true
mv pytorch_mps_install.sh scripts/setup/ 2>/dev/null || true

# 개발 관련 스크립트들 이동
mv check_scan.py scripts/dev/ 2>/dev/null || true
mv check_scan.sh scripts/dev/ 2>/dev/null || true
mv log_monitoring_script.sh scripts/dev/ 2>/dev/null || true

# 유지보수 관련 스크립트들 이동
mv fix_*.sh scripts/maintenance/ 2>/dev/null || true
mv immediate_fix_script.sh scripts/maintenance/ 2>/dev/null || true
mv port_fix_script.sh scripts/maintenance/ 2>/dev/null || true
mv project_analysis_script.sh scripts/maintenance/ 2>/dev/null || true

# 모델 관련 스크립트들 이동
mv *scanner*.py scripts/models/ 2>/dev/null || true
mv download_*.py scripts/models/ 2>/dev/null || true
mv consolidate_models.py scripts/models/ 2>/dev/null || true
mv enhanced_model_downloader.py scripts/models/ 2>/dev/null || true
mv huggingface_consolidator.py scripts/models/ 2>/dev/null || true
mv search_and_relocate_models.py scripts/models/ 2>/dev/null || true
mv simple_mover.py scripts/models/ 2>/dev/null || true
mv scan_models.py scripts/models/ 2>/dev/null || true
mv quick_find_models.sh scripts/models/ 2>/dev/null || true
mv model_path_consolidation.sh scripts/models/ 2>/dev/null || true

# 유틸리티 스크립트들 이동
mv project_structure_analyzer.py scripts/utils/ 2>/dev/null || true
mv complete_step_05_ai_fix.py scripts/utils/ 2>/dev/null || true

log_success "scripts 디렉토리 정리 완료"

# 3. 로그 및 리포트 파일 정리
log_info "3. 로그 및 리포트 파일 정리 중..."

# logs 디렉토리가 없으면 생성
mkdir -p logs

# 로그 파일들 이동
mv *.log logs/ 2>/dev/null || true
mv *.pid logs/ 2>/dev/null || true

# reports 디렉토리에 리포트 파일들 이동
mkdir -p reports
mv *.json reports/ 2>/dev/null || true
mv *.txt reports/ 2>/dev/null || true
mv mps_compatibility_test_report.txt reports/ 2>/dev/null || true

log_success "로그 및 리포트 파일 정리 완료"

# 4. 임시 파일 및 캐시 정리
log_info "4. 임시 파일 및 캐시 정리 중..."

# 임시 Python 파일들 정리
mkdir -p temp
mv python temp/ 2>/dev/null || true

# 데이터베이스 파일들 이동
mkdir -p data
mv *.db data/ 2>/dev/null || true

# 테스트 이미지 파일들 정리
mkdir -p assets/test_images
mv test_*.jpg assets/test_images/ 2>/dev/null || true

log_success "임시 파일 및 캐시 정리 완료"

# 5. 중요한 디렉토리 구조 확인 및 정리
log_info "5. 핵심 디렉토리 구조 확인 중..."

# 필수 디렉토리들 확인
essential_dirs=(
    "backend/app/core"
    "backend/app/api" 
    "backend/app/services"
    "backend/app/models"
    "backend/app/utils"
    "backend/app/ai_pipeline/steps"
    "backend/app/ai_pipeline/utils"
    "backend/app/ai_pipeline/models"
    "backend/static/uploads"
    "backend/static/results"
    "frontend/src/components"
    "frontend/src/pages"
    "frontend/src/hooks"
    "frontend/src/types"
    "frontend/src/utils"
)

for dir in "${essential_dirs[@]}"; do
    if [ ! -d "$dir" ]; then
        mkdir -p "$dir"
        touch "$dir/.gitkeep"
        log_info "생성: $dir"
    fi
done

log_success "핵심 디렉토리 구조 확인 완료"

# 6. .gitignore 업데이트
log_info "6. .gitignore 업데이트 중..."

cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
share/python-wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# Virtual Environments
venv/
env/
ENV/
env.bak/
venv.bak/

# Conda
.conda/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db

# Logs
logs/
*.log
*.pid

# AI Models (large files)
ai_models/
*.pth
*.pt
*.ckpt
*.safetensors
*.bin
*.onnx
*.pkl
*.h5

# Environment variables
.env
.env.local
.env.development.local
.env.test.local
.env.production.local

# Node.js
node_modules/
npm-debug.log*
yarn-debug.log*
yarn-error.log*

# Frontend build
frontend/dist/
frontend/build/
frontend/.next/

# Backend uploads and results
backend/static/uploads/*
backend/static/results/*
!backend/static/uploads/.gitkeep
!backend/static/results/.gitkeep

# Database
*.db
*.sqlite
*.sqlite3

# Cache
cache/
.cache/
*.cache

# Temporary files
temp/
tmp/
*.tmp
*.temp

# Reports and backups
reports/*.json
backup_*/

# MacOS specific
.AppleDouble
.LSOverride
EOF

log_success ".gitignore 업데이트 완료"

# 7. README 파일 생성
log_info "7. README 파일 업데이트 중..."

cat > README.md << 'EOF'
# 👗 MyCloset AI - AI 가상 피팅 플랫폼

AI 기술을 활용한 스마트 가상 피팅 서비스

## 🚀 빠른 시작

### 1. 환경 설정
```bash
# Conda 환경 생성 및 활성화
bash scripts/setup/setup_conda_env.sh
conda activate mycloset-m3

# 의존성 설치
cd backend && pip install -r requirements.txt
cd ../frontend && npm install
```

### 2. 개발 서버 실행
```bash
# 백엔드 (터미널 1)
cd backend && python app/main.py

# 프론트엔드 (터미널 2)  
cd frontend && npm run dev
```

## 📁 프로젝트 구조

```
mycloset-ai/
├── backend/           # FastAPI 백엔드
│   ├── app/          # 메인 애플리케이션
│   │   ├── ai_pipeline/  # AI 파이프라인 (8단계)
│   │   ├── api/      # API 라우터
│   │   ├── core/     # 핵심 설정
│   │   ├── models/   # 데이터 모델
│   │   └── services/ # 비즈니스 로직
│   ├── ai_models/    # AI 모델 파일들
│   └── static/       # 정적 파일
├── frontend/         # React 프론트엔드
│   ├── src/
│   │   ├── components/  # UI 컴포넌트
│   │   ├── pages/    # 페이지
│   │   └── hooks/    # 커스텀 훅
├── scripts/          # 유틸리티 스크립트
│   ├── setup/        # 환경 설정
│   ├── models/       # 모델 관리
│   ├── dev/          # 개발 도구
│   └── maintenance/  # 유지보수
├── logs/             # 로그 파일
├── reports/          # 분석 리포트
└── data/             # 데이터베이스 파일
```

## 🤖 AI 파이프라인 (8단계)

1. **Human Parsing** - 인체 부위 분석
2. **Pose Estimation** - 자세 추정  
3. **Cloth Segmentation** - 의류 분할
4. **Geometric Matching** - 기하학적 매칭
5. **Cloth Warping** - 의류 변형
6. **Virtual Fitting** - 가상 피팅
7. **Post Processing** - 후처리
8. **Quality Assessment** - 품질 평가

## 🛠️ 개발 도구

```bash
# 프로젝트 상태 체크
bash scripts/dev/check_structure.sh

# 모델 스캔
python scripts/models/complete_scanner.py

# 로그 모니터링  
bash scripts/dev/log_monitoring_script.sh
```

## 📋 요구사항

- Python 3.9+
- Node.js 18+
- macOS (M1/M2/M3 권장)
- 16GB+ RAM
- 10GB+ 저장공간

## 🔧 문제 해결

문제가 발생하면 다음을 확인하세요:

1. conda 환경 활성화: `conda activate mycloset-m3`
2. 모델 파일 확인: `ls backend/ai_models/`
3. 로그 확인: `tail -f logs/*.log`

## 📞 지원

- 이슈: GitHub Issues
- 문서: `/docs` 폴더 참고
EOF

log_success "README 파일 업데이트 완료"

# 8. 최종 정리 및 검증
log_info "8. 최종 정리 및 검증 중..."

# 빈 디렉토리에 .gitkeep 추가
find . -type d -empty -not -path "./.git/*" -exec touch {}/.gitkeep \; 2>/dev/null || true

# 실행 권한 설정
find scripts/ -name "*.sh" -exec chmod +x {} \; 2>/dev/null || true

log_success "최종 정리 완료"

# 9. 결과 보고
echo ""
log_header "🎉 프로젝트 폴더 재정리 완료!"
echo "=================================================================="

log_success "✨ 정리된 구조:"
echo "📁 scripts/        - 모든 스크립트 체계적 분류"
echo "📁 logs/           - 로그 파일 통합"  
echo "📁 reports/        - 분석 리포트 통합"
echo "📁 data/           - 데이터베이스 파일"
echo "📁 assets/         - 테스트 이미지 등"
echo "📁 temp/           - 임시 파일"
echo "📁 $backup_dir/  - 백업 파일"

echo ""
log_info "📊 파일 통계:"
echo "- Scripts: $(find scripts/ -name "*.py" -o -name "*.sh" | wc -l)개"
echo "- Logs: $(find logs/ -name "*.log" | wc -l)개" 
echo "- Reports: $(find reports/ -name "*.json" -o -name "*.txt" | wc -l)개"

echo ""
log_warning "⚠️ 다음 단계:"
echo "1. git add . && git commit -m '프로젝트 구조 재정리'"
echo "2. conda activate mycloset-m3"
echo "3. bash scripts/setup/setup_conda_env.sh"
echo "4. bash scripts/dev/check_structure.sh"

echo ""
log_success "🚀 재정리 완료! 이제 깔끔한 프로젝트 구조입니다."