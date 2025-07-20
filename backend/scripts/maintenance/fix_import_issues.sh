#!/bin/bash
# MyCloset AI Import 문제 해결 스크립트 (M3 Max 최적화)
# 실행: cd backend && bash fix_import_issues.sh

set -e

echo "🔧 MyCloset AI Import 체인 수정 시작..."
echo "=================================================="

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}ℹ️  $1${NC}"; }
log_success() { echo -e "${GREEN}✅ $1${NC}"; }
log_warning() { echo -e "${YELLOW}⚠️  $1${NC}"; }
log_error() { echo -e "${RED}❌ $1${NC}"; }

# 1. 필수 __init__.py 파일 생성
log_info "Step 1: 필수 __init__.py 파일 생성"

# 핵심 __init__.py 파일들
touch app/__init__.py
touch app/api/__init__.py
touch app/core/__init__.py
touch app/models/__init__.py
touch app/services/__init__.py
touch app/utils/__init__.py
touch app/ai_pipeline/__init__.py
touch app/ai_pipeline/steps/__init__.py
touch app/ai_pipeline/utils/__init__.py

log_success "__init__.py 파일 생성 완료"

# 2. 핵심 config 파일 생성
log_info "Step 2: 핵심 config 파일 생성"

cat > app/core/config.py << 'EOF'
"""
MyCloset AI 핵심 설정 파일 - M3 Max 최적화
"""
import os
import sys
import logging
from pathlib import Path

# 디바이스 자동 감지
try:
    import torch
    if torch.backends.mps.is_available():
        DEVICE = "mps"
        DEVICE_NAME = "Apple M3 Max"
    elif torch.cuda.is_available():
        DEVICE = "cuda"
        DEVICE_NAME = "NVIDIA GPU"
    else:
        DEVICE = "cpu"
        DEVICE_NAME = "CPU"
except ImportError:
    DEVICE = "cpu"
    DEVICE_NAME = "CPU"

# 프로젝트 경로
PROJECT_ROOT = Path(__file__).parent.parent.parent
BACKEND_ROOT = PROJECT_ROOT / "backend"
AI_MODELS_PATH = BACKEND_ROOT / "ai_models"
STATIC_PATH = BACKEND_ROOT / "static"

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s'
)

logger = logging.getLogger(__name__)
logger.info(f"🔧 디바이스: {DEVICE_NAME} ({DEVICE})")
EOF

log_success "core/config.py 생성 완료"

# 3. 기본 schemas.py 생성
log_info "Step 3: 기본 데이터 스키마 생성"

cat > app/models/schemas.py << 'EOF'
"""
MyCloset AI 데이터 스키마 정의
"""
from typing import Optional, Dict, Any, List
from pydantic import BaseModel
import time

class BaseResponse(BaseModel):
    """기본 응답 스키마"""
    success: bool = True
    message: str = ""
    timestamp: float = time.time()

class ImageUploadRequest(BaseModel):
    """이미지 업로드 요청"""
    image_data: str  # base64 encoded
    image_type: Optional[str] = "person"
    options: Optional[Dict[str, Any]] = {}

class VirtualTryOnResponse(BaseResponse):
    """가상 피팅 응답"""
    fitted_image: Optional[str] = None  # base64 encoded
    fit_score: Optional[float] = None
    quality_score: Optional[float] = None
    processing_time: Optional[float] = None

class StepProcessResponse(BaseResponse):
    """단계별 처리 응답"""
    step_number: int
    step_name: str
    result_data: Optional[Dict[str, Any]] = {}
    confidence: Optional[float] = None

class SystemStatus(BaseModel):
    """시스템 상태"""
    device: str
    memory_usage: Dict[str, float]
    models_loaded: List[str]
    server_uptime: float
EOF

log_success "models/schemas.py 생성 완료"

# 4. 필수 디렉토리 생성
log_info "Step 4: 필수 디렉토리 구조 생성"

mkdir -p ai_models
mkdir -p static/uploads
mkdir -p static/results
mkdir -p logs
mkdir -p scripts/test
mkdir -p scripts/utils

# .gitkeep 파일 생성
touch ai_models/.gitkeep
touch static/uploads/.gitkeep
touch static/results/.gitkeep
touch logs/.gitkeep

log_success "디렉토리 구조 생성 완료"

# 5. 핵심 import 테스트 스크립트 생성
log_info "Step 5: Import 테스트 스크립트 생성"

cat > test_imports.py << 'EOF'
#!/usr/bin/env python3
"""Import 체인 테스트"""
import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
sys.path.insert(0, str(Path(__file__).parent))

print("🔍 Import 체인 테스트 시작...")

def test_import(module_name, description):
    try:
        exec(f"import {module_name}")
        print(f"✅ {description}: 성공")
        return True
    except Exception as e:
        print(f"❌ {description}: 실패 - {e}")
        return False

# 기본 라이브러리 테스트
test_import("torch", "PyTorch")
test_import("fastapi", "FastAPI")
test_import("PIL", "PIL/Pillow")
test_import("cv2", "OpenCV")

# 프로젝트 모듈 테스트
test_import("app.core.config", "Core Config")
test_import("app.models.schemas", "Data Schemas")

print("\n🎯 Import 테스트 완료!")
EOF

chmod +x test_imports.py

log_success "테스트 스크립트 생성 완료"

# 6. 실행 테스트
log_info "Step 6: Import 체인 테스트 실행"

if python test_imports.py; then
    log_success "모든 핵심 Import가 정상 작동합니다!"
else
    log_warning "일부 Import에 문제가 있습니다. 다음 단계를 진행하세요."
fi

echo ""
echo "=================================================="
echo "🎉 1단계 수정 완료!"
echo ""
echo "📋 다음 단계:"
echo "   1. python test_imports.py 실행하여 결과 확인"
echo "   2. 문제가 있다면 해당 모듈 설치: pip install [모듈명]"
echo "   3. 모든 Import가 성공하면 2단계 진행"
echo ""
echo "🔧 2단계 실행 명령:"
echo "   bash fix_step2_ai_pipeline.sh"
echo "=================================================="