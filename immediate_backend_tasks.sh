#!/bin/bash

# MyCloset AI 백엔드 즉시 완성 스크립트
echo "🚀 MyCloset AI 백엔드 완성 작업 시작..."

cd backend

# 1. 프론트엔드 인터페이스에 맞는 통합 API 생성
echo "📝 통합 API 라우터 생성 중..."

cat > app/api/unified_routes.py << 'EOF'
from fastapi import APIRouter, File, UploadFile, Form, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
import asyncio
import uuid
import time
import base64
from typing import Optional, Dict, Any
import logging

# 기존 서비스들 import
from app.services.real_working_ai_fitter import RealWorkingAIFitter
from app.services.human_analysis import HumanAnalyzer
from app.services.clothing_3d_modeling import ClothingAnalyzer
from app.services.model_manager import ModelManager
from app.utils.image_utils import validate_image, process_image
from app.models.schemas import TryOnRequest, TryOnResponse

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["virtual-tryon"])

# 서비스 인스턴스들 (싱글톤)
ai_fitter = RealWorkingAIFitter()
human_analyzer = HumanAnalyzer()
clothing_analyzer = ClothingAnalyzer()
model_manager = ModelManager()

# 태스크 상태 저장소 (실제로는 Redis 사용 권장)
task_storage: Dict[str, Dict[str, Any]] = {}

@router.post("/virtual-tryon")
async def virtual_tryon_endpoint(
    background_tasks: BackgroundTasks,
    person_image: UploadFile = File(...),
    clothing_image: UploadFile = File(...),
    height: float = Form(...),
    weight: float = Form(...),
    chest: Optional[float] = Form(None),
    waist: Optional[float] = Form(None),
    hips: Optional[float] = Form(None)
):
    """프론트엔드 인터페이스에 맞는 가상 피팅 API"""
    
    # 입력 검증
    if not validate_image(person_image):
        raise HTTPException(400, "잘못된 사용자 이미지 형식입니다.")
    
    if not validate_image(clothing_image):
        raise HTTPException(400, "잘못된 의류 이미지 형식입니다.")
    
    # 태스크 ID 생성
    task_id = str(uuid.uuid4())
    
    # 초기 태스크 상태 설정
    task_storage[task_id] = {
        "status": "processing",
        "progress": 0,
        "current_step": "initializing",
        "steps": [
            {"id": "analyzing_body", "name": "신체 분석", "status": "pending"},
            {"id": "analyzing_clothing", "name": "의류 분석", "status": "pending"},
            {"id": "checking_compatibility", "name": "호환성 검사", "status": "pending"},
            {"id": "generating_fitting", "name": "가상 피팅 생성", "status": "pending"},
            {"id": "post_processing", "name": "후처리", "status": "pending"}
        ],
        "result": None,
        "error": None,
        "created_at": time.time()
    }
    
    # 백그라운드 태스크로 실제 처리 시작
    background_tasks.add_task(
        process_virtual_fitting,
        task_id,
        await person_image.read(),
        await clothing_image.read(),
        {
            "height": height,
            "weight": weight,
            "chest": chest,
            "waist": waist,
            "hips": hips
        }
    )
    
    return {
        "task_id": task_id,
        "status": "processing",
        "message": "가상 피팅이 시작되었습니다.",
        "estimated_time": "15-30초"
    }

@router.get("/status/{task_id}")
async def get_task_status(task_id: str):
    """태스크 처리 상태 조회"""
    if task_id not in task_storage:
        raise HTTPException(404, "존재하지 않는 태스크입니다.")
    
    return task_storage[task_id]

@router.get("/result/{task_id}")
async def get_task_result(task_id: str):
    """태스크 결과 조회"""
    if task_id not in task_storage:
        raise HTTPException(404, "존재하지 않는 태스크입니다.")
    
    task = task_storage[task_id]
    
    if task["status"] == "processing":
        raise HTTPException(202, "아직 처리 중입니다.")
    
    if task["status"] == "error":
        raise HTTPException(400, task["error"])
    
    return task["result"]

@router.post("/analyze-body")
async def analyze_body_endpoint(image: UploadFile = File(...)):
    """신체 분석 단독 API"""
    try:
        image_bytes = await image.read()
        result = await human_analyzer.analyze_complete_body(
            image_bytes, {"height": 170, "weight": 60}  # 기본값
        )
        return {"success": True, "result": result}
    except Exception as e:
        logger.error(f"신체 분석 오류: {e}")
        raise HTTPException(400, f"신체 분석 실패: {str(e)}")

@router.post("/analyze-clothing")
async def analyze_clothing_endpoint(image: UploadFile = File(...)):
    """의류 분석 단독 API"""
    try:
        image_bytes = await image.read()
        result = await clothing_analyzer.analyze_clothing(image_bytes)
        return {"success": True, "result": result}
    except Exception as e:
        logger.error(f"의류 분석 오류: {e}")
        raise HTTPException(400, f"의류 분석 실패: {str(e)}")

@router.get("/models")
async def get_available_models():
    """사용 가능한 AI 모델 목록"""
    return {
        "models": [
            {
                "id": "ootd_diffusion",
                "name": "OOT-Diffusion",
                "description": "최신 Diffusion 기반 가상 피팅",
                "quality": "High",
                "speed": "Medium"
            },
            {
                "id": "viton_hd", 
                "name": "VITON-HD",
                "description": "고해상도 가상 피팅",
                "quality": "Very High",
                "speed": "Slow"
            }
        ],
        "default": "ootd_diffusion"
    }

async def process_virtual_fitting(
    task_id: str,
    person_image: bytes,
    clothing_image: bytes,
    measurements: Dict[str, Any]
):
    """백그라운드에서 실행되는 실제 가상 피팅 처리"""
    
    try:
        # Step 1: 신체 분석
        update_task_progress(task_id, "analyzing_body", 20)
        logger.info(f"[{task_id}] 신체 분석 시작...")
        
        body_analysis = await human_analyzer.analyze_complete_body(
            person_image, measurements
        )
        
        # Step 2: 의류 분석
        update_task_progress(task_id, "analyzing_clothing", 40)
        logger.info(f"[{task_id}] 의류 분석 시작...")
        
        clothing_analysis = await clothing_analyzer.analyze_clothing(
            clothing_image
        )
        
        # Step 3: 호환성 검사
        update_task_progress(task_id, "checking_compatibility", 60)
        logger.info(f"[{task_id}] 호환성 검사 시작...")
        
        compatibility_score = calculate_compatibility(body_analysis, clothing_analysis)
        
        # Step 4: AI 가상 피팅
        update_task_progress(task_id, "generating_fitting", 80)
        logger.info(f"[{task_id}] AI 가상 피팅 생성 시작...")
        
        fitting_result = await ai_fitter.generate_virtual_fitting(
            person_image=person_image,
            clothing_image=clothing_image,
            body_analysis=body_analysis,
            clothing_analysis=clothing_analysis
        )
        
        # Step 5: 후처리
        update_task_progress(task_id, "post_processing", 95)
        logger.info(f"[{task_id}] 후처리 시작...")
        
        # 결과 이미지를 base64로 인코딩
        result_image_b64 = base64.b64encode(fitting_result["fitted_image"]).decode()
        
        # 최종 결과
        final_result = {
            "fitted_image": result_image_b64,
            "confidence": fitting_result.get("confidence", 0.85),
            "processing_time": fitting_result.get("processing_time", 15.0),
            "body_analysis": {
                "measurements": body_analysis.get("measurements", {}),
                "pose_keypoints": body_analysis.get("pose_keypoints", []),
                "body_type": body_analysis.get("body_type", "average")
            },
            "clothing_analysis": {
                "category": clothing_analysis.get("category", "shirt"),
                "style": clothing_analysis.get("style", "casual"),
                "colors": clothing_analysis.get("colors", ["blue"]),
                "pattern": clothing_analysis.get("pattern", "solid")
            },
            "fit_score": compatibility_score,
            "recommendations": generate_recommendations(
                body_analysis, clothing_analysis, compatibility_score
            )
        }
        
        # 완료 상태 업데이트
        task_storage[task_id].update({
            "status": "completed",
            "progress": 100,
            "current_step": "completed",
            "result": final_result,
            "completed_at": time.time()
        })
        
        # 모든 단계를 completed로 변경
        for step in task_storage[task_id]["steps"]:
            step["status"] = "completed"
        
        logger.info(f"[{task_id}] ✅ 가상 피팅 완료!")
        
    except Exception as e:
        logger.error(f"[{task_id}] ❌ 가상 피팅 처리 중 오류: {e}")
        
        task_storage[task_id].update({
            "status": "error",
            "error": str(e),
            "failed_at": time.time()
        })

def update_task_progress(task_id: str, current_step: str, progress: int):
    """태스크 진행상황 업데이트"""
    if task_id in task_storage:
        task_storage[task_id]["progress"] = progress
        task_storage[task_id]["current_step"] = current_step
        
        # 현재 단계를 processing으로, 이전 단계들을 completed로 설정
        for i, step in enumerate(task_storage[task_id]["steps"]):
            if step["id"] == current_step:
                step["status"] = "processing"
            elif i < len(task_storage[task_id]["steps"]) and \
                 task_storage[task_id]["steps"][i]["id"] != current_step:
                # 이전 단계들은 completed로 설정
                step["status"] = "completed"

def calculate_compatibility(body_analysis: dict, clothing_analysis: dict) -> float:
    """신체와 의류 호환성 점수 계산"""
    # 간단한 호환성 계산 로직
    base_score = 0.8
    
    # 의류 카테고리별 호환성
    category = clothing_analysis.get("category", "shirt")
    if category in ["shirt", "t-shirt", "blouse"]:
        base_score += 0.1
    
    return min(base_score, 1.0)

def generate_recommendations(
    body_analysis: dict, 
    clothing_analysis: dict, 
    fit_score: float
) -> list:
    """개인화된 추천 생성"""
    recommendations = []
    
    if fit_score < 0.7:
        recommendations.append("더 맞는 사이즈를 고려해보세요.")
    
    if clothing_analysis.get("style") == "formal":
        recommendations.append("정장 스타일에 어울리는 신발을 추천합니다.")
    
    return recommendations
EOF

# 2. main.py에 새 라우터 추가
echo "📝 main.py 업데이트 중..."

cat > app/main.py << 'EOF'
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles
import logging

from app.api import virtual_tryon, health
from app.api.unified_routes import router as unified_router
from app.core.logging_config import setup_logging
from app.core.config import settings

# 로깅 설정
setup_logging()
logger = logging.getLogger(__name__)

# FastAPI 앱 생성
app = FastAPI(
    title="MyCloset AI Backend",
    description="AI 기반 가상 피팅 시스템",
    version="1.0.0"
)

# CORS 미들웨어 추가
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173", 
        "http://localhost:8080",
        "https://mycloset-ai.vercel.app"  # 배포용
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Gzip 압축 미들웨어
app.add_middleware(GZipMiddleware, minimum_size=1000)

# 정적 파일 서빙 (업로드된 이미지 등)
app.mount("/static", StaticFiles(directory="static"), name="static")

# 라우터 등록
app.include_router(unified_router)  # 새로운 통합 라우터
app.include_router(virtual_tryon.router, prefix="/api", tags=["legacy"])
app.include_router(health.router, prefix="/api", tags=["health"])

@app.get("/")
async def root():
    return {
        "message": "MyCloset AI Backend is running!",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/health"
    }

@app.on_event("startup")
async def startup_event():
    logger.info("🚀 MyCloset AI Backend 시작됨")
    logger.info(f"🔧 설정: {settings.APP_NAME} v{settings.APP_VERSION}")

@app.on_event("shutdown") 
async def shutdown_event():
    logger.info("🛑 MyCloset AI Backend 종료됨")
EOF

# 3. 환경설정 파일 확인/생성
echo "⚙️ 환경설정 확인 중..."

if [ ! -f ".env" ]; then
    cat > .env << 'EOF'
# Application Settings
APP_NAME=MyCloset AI Backend
APP_VERSION=1.0.0
DEBUG=true
HOST=0.0.0.0
PORT=8000

# CORS Settings  
CORS_ORIGINS=http://localhost:3000,http://localhost:5173

# File Upload Settings
MAX_UPLOAD_SIZE=52428800
ALLOWED_EXTENSIONS=jpg,jpeg,png,webp,bmp

# AI Model Settings
DEFAULT_MODEL=ootd_diffusion
USE_GPU=true
DEVICE=cuda
IMAGE_SIZE=512
MAX_WORKERS=4
BATCH_SIZE=1

# Logging
LOG_LEVEL=INFO
EOF
    echo "✅ .env 파일 생성됨"
else
    echo "✅ .env 파일 이미 존재"
fi

# 4. requirements.txt 확인/업데이트
echo "📦 dependencies 확인 중..."

cat >> requirements.txt << 'EOF'

# 추가 필요 패키지들
python-multipart==0.0.6
python-dotenv==1.0.0
aiofiles==23.2.1
websockets==11.0.3
redis==5.0.1
pydantic==2.5.0
EOF

# 5. 이미지 검증 유틸리티 생성
echo "🔧 유틸리티 함수 생성 중..."

cat > app/utils/validators.py << 'EOF'
from fastapi import UploadFile
from PIL import Image
import io
from typing import List

ALLOWED_EXTENSIONS = ["jpg", "jpeg", "png", "webp", "bmp"]
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB

def validate_image(file: UploadFile) -> bool:
    """이미지 파일 검증"""
    
    # 파일 확장자 검사
    if not file.filename:
        return False
        
    extension = file.filename.split(".")[-1].lower()
    if extension not in ALLOWED_EXTENSIONS:
        return False
    
    # MIME 타입 검사
    if not file.content_type or not file.content_type.startswith("image/"):
        return False
    
    return True

def validate_measurements(height: float, weight: float) -> bool:
    """신체 측정값 검증"""
    
    # 합리적인 범위 검사
    if not (100 <= height <= 250):  # cm
        return False
        
    if not (30 <= weight <= 300):   # kg
        return False
    
    return True

async def validate_image_content(image_bytes: bytes) -> bool:
    """이미지 내용 검증"""
    try:
        image = Image.open(io.BytesIO(image_bytes))
        
        # 이미지 크기 검사
        width, height = image.size
        if width < 100 or height < 100:
            return False
            
        if width > 4096 or height > 4096:
            return False
        
        return True
    except:
        return False
EOF

# 6. 테스트 실행
echo "🧪 백엔드 테스트 중..."

# 가상환경이 활성화되어 있는지 확인
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo "⚠️ 가상환경을 먼저 활성화하세요:"
    echo "source venv/bin/activate"
    exit 1
fi

# 필수 패키지 설치
pip install -q python-multipart python-dotenv aiofiles

# 서버 시작 테스트 (백그라운드에서 5초만)
echo "🚀 서버 시작 테스트 중..."
timeout 5s uvicorn app.main:app --reload --host 0.0.0.0 --port 8000 &
SERVER_PID=$!

sleep 3

# 헬스체크
if curl -s http://localhost:8000/ > /dev/null 2>&1; then
    echo "✅ 백엔드 서버 정상 동작!"
else
    echo "❌ 서버 시작 실패"
fi

# 백그라운드 서버 종료
kill $SERVER_PID 2>/dev/null

echo ""
echo "🎉 백엔드 완성 작업 완료!"
echo ""
echo "📋 완성된 항목들:"
echo "   ✅ 프론트엔드 호환 API 엔드포인트"
echo "   ✅ 백그라운드 태스크 처리" 
echo "   ✅ 실시간 진행상황 추적"
echo "   ✅ 입력 검증 및 에러 처리"
echo "   ✅ CORS 및 미들웨어 설정"
echo ""
echo "🚀 서버 실행:"
echo "   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000"
echo ""
echo "📚 API 문서:"
echo "   http://localhost:8000/docs"
echo ""
echo "🔗 주요 엔드포인트:"
echo "   POST /api/virtual-tryon      # 가상 피팅"
echo "   GET  /api/status/{task_id}   # 진행상황 조회"
echo "   GET  /api/result/{task_id}   # 결과 조회"
echo "   POST /api/analyze-body       # 신체 분석"
echo "   POST /api/analyze-clothing   # 의류 분석"
echo ""
echo "🎯 다음 단계:"
echo "   1. 프론트엔드와 연동 테스트"
echo "   2. AI 모델 성능 최적화"
echo "   3. WebSocket 실시간 업데이트 추가"