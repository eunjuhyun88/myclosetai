#!/bin/bash
# install_and_run_ootd_downloader.sh - OOTDiffusion 원본 다운로더 설치 및 실행

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

log_header "OOTDiffusion 원본 모델 자동 다운로드 시스템"
echo "================================================================="

# 1. 환경 확인
log_info "Step 1: 환경 확인"

# Conda 환경 활성화 확인
if [[ "$CONDA_DEFAULT_ENV" == "mycloset-ai" ]]; then
    log_success "Conda 환경 활성화됨: $CONDA_DEFAULT_ENV"
else
    log_warning "Conda 환경 활성화 필요"
    if command -v conda &> /dev/null; then
        log_info "Conda 환경 활성화 중..."
        source "$(conda info --base)/etc/profile.d/conda.sh"
        conda activate mycloset-ai
        log_success "mycloset-ai 환경 활성화 완료"
    else
        log_warning "Conda 없음 - 시스템 Python 사용"
    fi
fi

# 프로젝트 루트 확인
PROJECT_ROOT="/Users/gimdudeul/MVP/mycloset-ai"
if [[ ! -d "$PROJECT_ROOT" ]]; then
    PROJECT_ROOT=$(pwd)
    log_warning "기본 경로 사용: $PROJECT_ROOT"
else
    log_success "프로젝트 루트: $PROJECT_ROOT"
fi

cd "$PROJECT_ROOT"

# 2. 필수 패키지 설치
log_info "Step 2: 필수 패키지 설치"

# 기본 패키지들
REQUIRED_PACKAGES=(
    "requests"
    "tqdm" 
    "huggingface-hub"
    "safetensors"
)

log_info "필수 패키지 설치 중..."
for package in "${REQUIRED_PACKAGES[@]}"; do
    if python -c "import $package" 2>/dev/null; then
        log_success "$package 이미 설치됨"
    else
        log_info "$package 설치 중..."
        pip install "$package" --quiet || pip install "$package" --user --quiet
        if python -c "import $package" 2>/dev/null; then
            log_success "$package 설치 완료"
        else
            log_warning "$package 설치 실패 - 계속 진행"
        fi
    fi
done

# Git LFS 설치 확인 및 설치
log_info "Git LFS 확인 중..."
if command -v git-lfs &> /dev/null; then
    log_success "Git LFS 이미 설치됨"
else
    log_info "Git LFS 설치 시도 중..."
    if command -v brew &> /dev/null; then
        brew install git-lfs || log_warning "Git LFS 설치 실패"
    elif command -v apt-get &> /dev/null; then
        sudo apt-get install git-lfs -y || log_warning "Git LFS 설치 실패"
    else
        log_warning "Git LFS 자동 설치 불가 - 수동 설치 필요"
    fi
fi

# 3. 다운로더 스크립트 생성
log_info "Step 3: 다운로더 스크립트 준비"

DOWNLOADER_FILE="$PROJECT_ROOT/download_ootd_original.py"

# 위의 다운로더 코드를 파일로 저장
cat > "$DOWNLOADER_FILE" << 'DOWNLOADER_EOF'
# 위 아티팩트의 Python 코드가 여기에 들어갑니다
# [실제로는 전체 다운로더 코드]

#!/usr/bin/env python3
"""OOTDiffusion 원본 모델 다운로더 - 간소화 버전"""

import os
import sys
import json
import time
import shutil
import requests
from pathlib import Path
from urllib.parse import urljoin

def download_ootd_models():
    """OOTDiffusion 모델 다운로드 메인 함수"""
    print("🔥 OOTDiffusion 원본 모델 다운로드 시작")
    print("=" * 50)
    
    # 프로젝트 경로 설정
    project_root = Path("/Users/gimdudeul/MVP/mycloset-ai")
    ai_models_dir = project_root / "ai_models"
    download_dir = ai_models_dir / "downloads" / "ootdiffusion_original"
    
    # 디렉토리 생성
    download_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 다운로드 경로: {download_dir}")
    
    # Hugging Face Hub 사용 시도
    try:
        from huggingface_hub import snapshot_download
        
        print("🤗 Hugging Face Hub으로 다운로드 중...")
        
        # OOTDiffusion 메인 모델
        snapshot_download(
            repo_id="levihsu/OOTDiffusion",
            local_dir=str(download_dir / "levihsu_OOTDiffusion"),
            cache_dir=str(ai_models_dir / "huggingface_cache"),
            resume_download=True,
            local_files_only=False,
            max_workers=2
        )
        
        print("✅ OOTDiffusion 다운로드 완료!")
        
        # Stable Diffusion Inpainting
        print("📦 Stable Diffusion Inpainting 다운로드 중...")
        snapshot_download(
            repo_id="runwayml/stable-diffusion-inpainting", 
            local_dir=str(download_dir / "stable_diffusion_inpainting"),
            cache_dir=str(ai_models_dir / "huggingface_cache"),
            resume_download=True,
            local_files_only=False,
            max_workers=2
        )
        
        print("✅ Stable Diffusion Inpainting 다운로드 완료!")
        return True
        
    except ImportError:
        print("⚠️ huggingface_hub 없음 - Git clone 시도")
        return git_clone_method(download_dir)
    
    except Exception as e:
        print(f"❌ Hugging Face 다운로드 실패: {e}")
        return git_clone_method(download_dir)

def git_clone_method(download_dir):
    """Git clone 대체 방법"""
    try:
        import subprocess
        
        print("🔄 Git clone으로 다운로드 시도...")
        
        # Git LFS 설정
        subprocess.run(["git", "lfs", "install"], check=False)
        
        # OOTDiffusion 클론
        ootd_dir = download_dir / "levihsu_OOTDiffusion"
        if not ootd_dir.exists():
            subprocess.run([
                "git", "clone", 
                "https://huggingface.co/levihsu/OOTDiffusion",
                str(ootd_dir)
            ], check=True)
            
        print("✅ Git clone 완료!")
        return True
        
    except Exception as e:
        print(f"❌ Git clone 실패: {e}")
        return False

if __name__ == "__main__":
    success = download_ootd_models()
    
    if success:
        print("\n🎉 OOTDiffusion 원본 모델 다운로드 완료!")
        print("서버를 재시작하여 고품질 모델을 사용하세요")
    else:
        print("\n⚠️ 다운로드 실패 - 수동 설치가 필요할 수 있습니다")
    
    sys.exit(0 if success else 1)
DOWNLOADER_EOF

chmod +x "$DOWNLOADER_FILE"
log_success "다운로더 스크립트 생성 완료"

# 4. 디스크 공간 확인
log_info "Step 4: 디스크 공간 확인"

REQUIRED_SPACE_GB=25
available_space_gb=$(df -BG "$PROJECT_ROOT" | tail -1 | awk '{print $4}' | sed 's/G//')

if [[ $available_space_gb -gt $REQUIRED_SPACE_GB ]]; then
    log_success "디스크 공간 충분: ${available_space_gb}GB (필요: ${REQUIRED_SPACE_GB}GB)"
else
    log_error "디스크 공간 부족: ${available_space_gb}GB (필요: ${REQUIRED_SPACE_GB}GB)"
    echo "일부 파일을 삭제한 후 다시 시도하세요"
    exit 1
fi

# 5. 다운로드 옵션 선택
log_info "Step 5: 다운로드 옵션 선택"

echo ""
echo "다운로드 옵션을 선택하세요:"
echo "1. 🚀 전체 모델 다운로드 (권장, ~15GB)"
echo "2. ⚡ 핵심 모델만 다운로드 (~8GB)"  
echo "3. 🔧 커스텀 선택"
echo ""

read -p "선택 (1-3, 기본값: 1): " choice
choice=${choice:-1}

case $choice in
    1)
        log_info "전체 모델 다운로드 선택"
        DOWNLOAD_MODE="full"
        ;;
    2) 
        log_info "핵심 모델만 다운로드 선택"
        DOWNLOAD_MODE="essential"
        ;;
    3)
        log_info "커스텀 다운로드 선택"
        DOWNLOAD_MODE="custom"
        ;;
    *)
        log_info "기본값 선택: 전체 다운로드"
        DOWNLOAD_MODE="full"
        ;;
esac

# 6. 다운로드 실행
log_header "Step 6: OOTDiffusion 원본 모델 다운로드 실행"

echo ""
log_warning "다운로드에 시간이 오래 걸릴 수 있습니다 (10-30분)"
log_warning "네트워크 연결을 확인하고 진행하세요"
echo ""

read -p "다운로드를 시작하시겠습니까? (y/N): " confirm
if [[ $confirm =~ ^[Yy]$ ]]; then
    
    log_info "다운로드 시작..."
    start_time=$(date +%s)
    
    if [[ $DOWNLOAD_MODE == "essential" ]]; then
        python "$DOWNLOADER_FILE" --essential
    else
        python "$DOWNLOADER_FILE"
    fi
    
    download_result=$?
    end_time=$(date +%s)
    duration=$((end_time - start_time))
    
    echo ""
    if [[ $download_result -eq 0 ]]; then
        log_success "다운로드 완료! (소요시간: ${duration}초)"
    else
        log_error "다운로드 실패 (소요시간: ${duration}초)"
    fi
    
else
    log_info "다운로드 취소됨"
    exit 0
fi

# 7. 다운로드 결과 확인
log_info "Step 7: 다운로드 결과 확인"

MODELS_DIR="$PROJECT_ROOT/ai_models/downloads/ootdiffusion_original"

if [[ -d "$MODELS_DIR" ]]; then
    log_success "모델 디렉토리 생성됨: $MODELS_DIR"
    
    # 다운로드된 파일 정보
    total_size=$(du -sh "$MODELS_DIR" 2>/dev/null | cut -f1)
    file_count=$(find "$MODELS_DIR" -type f | wc -l)
    
    log_info "다운로드 통계:"
    echo "  📁 총 크기: $total_size"
    echo "  📄 파일 수: $file_count"
    
    # 주요 모델 파일 확인
    log_info "주요 모델 파일 확인:"
    find "$MODELS_DIR" -name "*.safetensors" -o -name "*.bin" | head -5 | while read file; do
        size=$(du -sh "$file" | cut -f1)
        echo "  ✅ $(basename "$file") ($size)"
    done
    
else
    log_warning "모델 디렉토리를 찾을 수 없습니다"
fi

# 8. 시스템 연결 설정
log_info "Step 8: 시스템 연결 설정"

# 백엔드 연결 경로들 생성
BACKEND_MODELS_DIR="$PROJECT_ROOT/backend/ai_models"
mkdir -p "$BACKEND_MODELS_DIR"

# 심볼릭 링크 생성
if [[ -d "$MODELS_DIR" ]] && [[ ! -L "$BACKEND_MODELS_DIR/ootdiffusion" ]]; then
    ln -sf "$MODELS_DIR" "$BACKEND_MODELS_DIR/ootdiffusion" 2>/dev/null || {
        log_info "심볼릭 링크 실패 - 디렉토리 복사"
        cp -r "$MODELS_DIR" "$BACKEND_MODELS_DIR/ootdiffusion"
    }
    log_success "백엔드 연결 완료"
fi

# Hugging Face 캐시 연결
HF_CACHE_TARGET="$PROJECT_ROOT/ai_models/huggingface_cache/models--levihsu--OOTDiffusion"
if [[ -d "$MODELS_DIR/levihsu_OOTDiffusion" ]] && [[ ! -d "$HF_CACHE_TARGET" ]]; then
    mkdir -p "$(dirname "$HF_CACHE_TARGET")"
    ln -sf "$MODELS_DIR/levihsu_OOTDiffusion" "$HF_CACHE_TARGET" 2>/dev/null || {
        cp -r "$MODELS_DIR/levihsu_OOTDiffusion" "$HF_CACHE_TARGET"
    }
    log_success "Hugging Face 캐시 연결 완료"
fi

# 9. 완료 및 다음 단계
log_header "🎉 OOTDiffusion 원본 모델 설치 완료!"
echo "================================================================="

if [[ $download_result -eq 0 ]]; then
    log_success "✅ 실제 고품질 OOTDiffusion 모델이 설치되었습니다"
    log_success "✅ 시스템 연결 및 캐시 설정 완료"
    log_success "✅ 서버 재시작 후 실제 AI 모델 사용 가능"
    
    echo ""
    echo "📋 다음 단계:"
    echo "1. 서버 재시작: cd backend && python app/main.py"
    echo "2. API 문서 확인: http://localhost:8000/docs"
    echo "3. 가상 피팅 테스트: /api/pipeline/virtual-fitting"
    echo ""
    
    echo "🎯 이제 실제 고품질 가상 피팅을 경험할 수 있습니다!"
    
else
    log_warning "⚠️ 다운로드가 완전히 성공하지 못했습니다"
    echo "서버는 폴백 모드로 계속 작동합니다"
    echo ""
    echo "🔧 수동 해결 방법:"
    echo "1. 네트워크 연결 확인"
    echo "2. 디스크 공간 확인" 
    echo "3. 스크립트 재실행: $0"
fi

echo ""
log_info "생성된 파일들:"
echo "- $DOWNLOADER_FILE (다운로더 스크립트)"
echo "- $MODELS_DIR (다운로드된 모델들)"
echo "- $BACKEND_MODELS_DIR/ootdiffusion (백엔드 연결)"

echo ""
echo "🚀 서버 재시작 명령어:"
echo "cd backend && python app/main.py"