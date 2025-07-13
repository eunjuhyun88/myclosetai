# backend/app/main.py
"""
MyCloset AI 통합 백엔드 - 실제 AI 모델과 프론트엔드 완전 연동
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles
import logging
import json
from typing import Dict

# 기존 라우터들
from app.api import health
from app.api.unified_routes import router as unified_router
from app.core.logging_config import setup_logging
from app.core.config import settings

# 새로운 통합 서비스
from app.services.real_working_ai_fitter import RealWorkingAIFitter

# 로깅 설정
setup_logging()
logger = logging.getLogger(__name__)

# FastAPI 앱 생성
app = FastAPI(
    title="MyCloset AI Backend",
    description="실제 AI 모델 기반 가상 피팅 시스템",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS 미들웨어 추가 (프론트엔드 연동)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",    # React 개발 서버
        "http://localhost:5173",    # Vite 개발 서버
        "http://localhost:8080",    # 추가 포트
        "https://mycloset-ai.vercel.app",  # 배포용
        "https://*.vercel.app",     # Vercel 서브도메인
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# Gzip 압축 미들웨어
app.add_middleware(GZipMiddleware, minimum_size=1000)

# 정적 파일 서빙
app.mount("/static", StaticFiles(directory="static"), name="static")

# WebSocket 연결 관리자
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, task_id: str):
        await websocket.accept()
        self.active_connections[task_id] = websocket
        logger.info(f"📡 WebSocket 연결됨: {task_id}")

    def disconnect(self, task_id: str):
        if task_id in self.active_connections:
            del self.active_connections[task_id]
            logger.info(f"📡 WebSocket 연결 해제됨: {task_id}")

    async def send_progress_update(self, task_id: str, data: dict):
        if task_id in self.active_connections:
            try:
                await self.active_connections[task_id].send_text(
                    json.dumps(data, ensure_ascii=False)
                )
                logger.debug(f"📡 진행상황 전송: {task_id} - {data.get('progress', 0)}%")
            except Exception as e:
                logger.warning(f"📡 WebSocket 전송 실패: {task_id} - {e}")
                self.disconnect(task_id)

# 전역 연결 관리자
manager = ConnectionManager()

# AI 서비스 인스턴스
ai_fitter = RealWorkingAIFitter()

@app.on_event("startup")
async def startup_event():
    """앱 시작 시 실행"""
    logger.info("🚀 MyCloset AI Backend 시작됨")
    logger.info(f"🔧 설정: {settings.APP_NAME} v{settings.APP_VERSION}")
    
    # AI 모델 초기화
    try:
        await ai_fitter.initialize()
        logger.info("✅ AI 서비스 초기화 완료")
    except Exception as e:
        logger.error(f"❌ AI 서비스 초기화 실패: {e}")

@app.on_event("shutdown") 
async def shutdown_event():
    """앱 종료 시 실행"""
    logger.info("🛑 MyCloset AI Backend 종료됨")

# 라우터 등록
app.include_router(unified_router)
app.include_router(health.router, prefix="/api", tags=["health"])

# WebSocket 엔드포인트
@app.websocket("/ws/fitting/{task_id}")
async def websocket_endpoint(websocket: WebSocket, task_id: str):
    """실시간 가상 피팅 진행상황 WebSocket"""
    await manager.connect(websocket, task_id)
    try:
        while True:
            # 클라이언트로부터 메시지 대기 (연결 유지용)
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(task_id)
    except Exception as e:
        logger.error(f"❌ WebSocket 오류: {e}")
        manager.disconnect(task_id)

# 새로운 통합 API 엔드포인트들
@app.get("/")
async def root():
    """루트 엔드포인트"""
    return {
        "message": "MyCloset AI Backend v2.0 - 실제 AI 모델 통합 완료",
        "version": "2.0.0",
        "status": "running",
        "features": [
            "실제 AI 가상 피팅",
            "실시간 WebSocket 업데이트", 
            "고급 신체 분석",
            "다중 모델 지원"
        ],
        "docs": "/docs",
        "health": "/api/health"
    }

@app.get("/api/system/status")
async def get_system_status():
    """시스템 상태 조회"""
    try:
        model_status = await ai_fitter.get_model_status()
        return {
            "status": "healthy",
            "ai_service": model_status,
            "websocket_connections": len(manager.active_connections),
            "version": "2.0.0"
        }
    except Exception as e:
        logger.error(f"❌ 시스템 상태 조회 실패: {e}")
        return {
            "status": "error",
            "error": str(e)
        }

# backend/app/api/unified_routes.py 업데이트
"""
프론트엔드와 완전 호환되는 통합 API 라우터
실제 AI 모델 서비스와 연동
"""

from fastapi import APIRouter, File, UploadFile, Form, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
import asyncio
import uuid
import time
import base64
from typing import Optional, Dict, Any
import logging

# 실제 AI 서비스들 import
from app.services.real_working_ai_fitter import RealWorkingAIFitter
from app.services.human_analysis import HumanAnalyzer
from app.services.clothing_analysis import ClothingAnalyzer
from app.utils.validators import validate_image, validate_measurements
from app.models.schemas import TryOnRequest, TryOnResponse

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["virtual-tryon"])

# 서비스 인스턴스들 (싱글톤)
ai_fitter = RealWorkingAIFitter()
human_analyzer = HumanAnalyzer()
clothing_analyzer = ClothingAnalyzer()

# 태스크 상태 저장소 (실제로는 Redis 권장)
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
    """실제 AI 모델을 사용한 가상 피팅 API"""
    
    # 입력 검증
    if not validate_image(person_image):
        raise HTTPException(400, "잘못된 사용자 이미지 형식입니다.")
    
    if not validate_image(clothing_image):
        raise HTTPException(400, "잘못된 의류 이미지 형식입니다.")
    
    if not validate_measurements(height, weight):
        raise HTTPException(400, "잘못된 신체 측정값입니다.")
    
    # 태스크 ID 생성
    task_id = str(uuid.uuid4())
    
    # 초기 태스크 상태 설정
    task_storage[task_id] = {
        "status": "processing",
        "progress": 0,
        "current_step": "이미지 업로드 완료",
        "steps": [
            {"id": "analyzing_body", "name": "신체 분석", "status": "pending"},
            {"id": "analyzing_clothing", "name": "의류 분석", "status": "pending"},
            {"id": "checking_compatibility", "name": "호환성 검사", "status": "pending"},
            {"id": "generating_fitting", "name": "AI 가상 피팅 생성", "status": "pending"},
            {"id": "post_processing", "name": "품질 향상 및 후처리", "status": "pending"}
        ],
        "result": None,
        "error": None,
        "created_at": time.time()
    }
    
    logger.info(f"🎨 새로운 가상 피팅 요청: {task_id}")
    
    # 백그라운드 태스크로 실제 AI 처리 시작
    background_tasks.add_task(
        process_real_virtual_fitting,
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
        "message": "AI 가상 피팅이 시작되었습니다.",
        "estimated_time": "15-30초"
    }

async def process_real_virtual_fitting(
    task_id: str,
    person_image: bytes,
    clothing_image: bytes,
    measurements: Dict[str, Any]
):
    """실제 AI 모델을 사용한 가상 피팅 처리"""
    
    try:
        # WebSocket 매니저 import (순환 import 방지)
        from app.main import manager
        
        logger.info(f"🤖 [{task_id}] 실제 AI 가상 피팅 처리 시작...")
        
        # Step 1: 신체 분석
        await update_task_progress(task_id, "analyzing_body", 15, manager)
        logger.info(f"[{task_id}] 신체 분석 시작...")
        
        body_analysis = await human_analyzer.analyze_complete_body(
            person_image, measurements
        )
        
        # Step 2: 의류 분석  
        await update_task_progress(task_id, "analyzing_clothing", 30, manager)
        logger.info(f"[{task_id}] 의류 분석 시작...")
        
        clothing_analysis = await clothing_analyzer.analyze_clothing(
            clothing_image
        )
        
        # Step 3: 호환성 검사
        await update_task_progress(task_id, "checking_compatibility", 45, manager)
        logger.info(f"[{task_id}] 호환성 검사 시작...")
        
        compatibility_score = calculate_enhanced_compatibility(body_analysis, clothing_analysis)
        
        # Step 4: 실제 AI 가상 피팅 생성
        await update_task_progress(task_id, "generating_fitting", 70, manager)
        logger.info(f"[{task_id}] AI 가상 피팅 생성 시작...")
        
        # 실제 AI 서비스 호출
        fitting_result = await ai_fitter.generate_virtual_fitting(
            person_image=person_image,
            clothing_image=clothing_image,
            body_analysis=body_analysis,
            clothing_analysis=clothing_analysis
        )
        
        # Step 5: 후처리 및 품질 향상
        await update_task_progress(task_id, "post_processing", 90, manager)
        logger.info(f"[{task_id}] 후처리 시작...")
        
        # 최종 결과 구성
        final_result = {
            "fitted_image": fitting_result["fitted_image"],
            "confidence": fitting_result.get("confidence", 0.85),
            "processing_time": fitting_result.get("processing_time", 15.0),
            "model_used": fitting_result.get("model_used", "ootdiffusion"),
            "body_analysis": {
                "measurements": body_analysis.get("measurements", {}),
                "pose_keypoints": body_analysis.get("pose_analysis", {}).get("keypoints", []),
                "body_type": body_analysis.get("body_type", "보통"),
                "analysis_confidence": body_analysis.get("analysis_confidence", 0.8)
            },
            "clothing_analysis": {
                "category": clothing_analysis.get("category", "상의"),
                "style": clothing_analysis.get("style", "캐주얼"),
                "colors": clothing_analysis.get("colors", ["파란색"]),
                "pattern": clothing_analysis.get("pattern", "무지"),
                "material": clothing_analysis.get("material", "면")
            },
            "fit_score": compatibility_score,
            "recommendations": generate_enhanced_recommendations(
                body_analysis, clothing_analysis, compatibility_score
            ),
            "image_specs": fitting_result.get("image_specs", {
                "resolution": [512, 512],
                "format": "JPEG",
                "quality": 95
            }),
            "processing_stats": fitting_result.get("processing_stats", {})
        }
        
        # 완료 상태 업데이트
        task_storage[task_id].update({
            "status": "completed",
            "progress": 100,
            "current_step": "완료",
            "result": final_result,
            "completed_at": time.time()
        })
        
        # 모든 단계를 completed로 변경
        for step in task_storage[task_id]["steps"]:
            step["status"] = "completed"
        
        # WebSocket으로 완료 알림
        await manager.send_progress_update(task_id, {
            "status": "completed",
            "progress": 100,
            "result": final_result
        })
        
        logger.info(f"[{task_id}] ✅ 실제 AI 가상 피팅 완료!")
        
    except Exception as e:
        logger.error(f"[{task_id}] ❌ AI 가상 피팅 처리 중 오류: {e}")
        
        task_storage[task_id].update({
            "status": "error",
            "error": str(e),
            "failed_at": time.time()
        })
        
        # WebSocket으로 에러 알림
        await manager.send_progress_update(task_id, {
            "status": "error",
            "error": str(e)
        })

async def update_task_progress(task_id: str, current_step: str, progress: int, manager):
    """태스크 진행상황 업데이트 및 WebSocket 전송"""
    if task_id in task_storage:
        task_storage[task_id]["progress"] = progress
        task_storage[task_id]["current_step"] = current_step
        
        # 현재 단계를 processing으로, 이전 단계들을 completed로 설정
        step_order = ["analyzing_body", "analyzing_clothing", "checking_compatibility", 
                     "generating_fitting", "post_processing"]
        
        current_index = step_order.index(current_step) if current_step in step_order else -1
        
        for i, step in enumerate(task_storage[task_id]["steps"]):
            if i < current_index:
                step["status"] = "completed"
            elif step["id"] == current_step:
                step["status"] = "processing"
            else:
                step["status"] = "pending"
        
        # WebSocket으로 실시간 업데이트 전송
        await manager.send_progress_update(task_id, {
            "progress": progress,
            "current_step": current_step,
            "steps": task_storage[task_id]["steps"]
        })

def calculate_enhanced_compatibility(body_analysis: dict, clothing_analysis: dict) -> float:
    """향상된 호환성 점수 계산"""
    base_score = 0.75
    
    # 체형과 의류 스타일 매칭
    body_type = body_analysis.get("body_type", "보통")
    clothing_style = clothing_analysis.get("style", "캐주얼")
    
    # 체형별 스타일 점수
    style_compatibility = {
        "슬림": {"피트": 0.9, "캐주얼": 0.8, "포멀": 0.7},
        "보통": {"캐주얼": 0.9, "피트": 0.8, "포멀": 0.8},
        "통통": {"루즈": 0.9, "캐주얼": 0.8, "포멀": 0.7}
    }
    
    # 추가 점수 계산
    style_score = style_compatibility.get(body_type, {}).get(clothing_style, 0.7) * 0.2
    
    return min(base_score + style_score, 1.0)

def generate_enhanced_recommendations(
    body_analysis: dict, 
    clothing_analysis: dict, 
    fit_score: float
) -> list:
    """향상된 개인화 추천 생성"""
    recommendations = []
    
    # 핏 점수 기반 추천
    if fit_score < 0.7:
        recommendations.append("더 잘 맞는 사이즈나 스타일을 고려해보세요.")
    elif fit_score > 0.9:
        recommendations.append("완벽한 핏입니다! 이 스타일이 잘 어울려요.")
    
    # 체형별 추천
    body_type = body_analysis.get("body_type", "보통")
    if "슬림" in body_type:
        recommendations.append("피팅 스타일의 의류가 체형을 더욱 돋보이게 할 수 있어요.")
    elif "통통" in body_type:
        recommendations.append("A라인이나 루즈핏 스타일이 더 편안하고 멋스러울 수 있어요.")
    
    # 색상 추천
    colors = clothing_analysis.get("colors", [])
    if "검정" in colors or "블랙" in colors:
        recommendations.append("검정색은 슬림해 보이는 효과가 있어 다양한 체형에 잘 어울려요.")
    
    # 스타일 추천
    style = clothing_analysis.get("style", "캐주얼")
    if style == "포멀":
        recommendations.append("포멀한 스타일에는 깔끔한 신발과 액세서리를 매치해보세요.")
    
    return recommendations

@router.get("/status/{task_id}")
async def get_task_status(task_id: str):
    """태스크 처리 상태 조회 (프론트엔드 호환)"""
    if task_id not in task_storage:
        raise HTTPException(404, "존재하지 않는 태스크입니다.")
    
    return task_storage[task_id]

@router.get("/result/{task_id}")
async def get_task_result(task_id: str):
    """태스크 결과 조회 (프론트엔드 호환)"""
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
    """신체 분석 단독 API (프론트엔드 호환)"""
    try:
        image_bytes = await image.read()
        result = await human_analyzer.analyze_complete_body(
            image_bytes, {"height": 170, "weight": 60}
        )
        return {"success": True, "result": result}
    except Exception as e:
        logger.error(f"신체 분석 오류: {e}")
        raise HTTPException(400, f"신체 분석 실패: {str(e)}")

@router.post("/analyze-clothing")
async def analyze_clothing_endpoint(image: UploadFile = File(...)):
    """의류 분석 단독 API (프론트엔드 호환)"""
    try:
        image_bytes = await image.read()
        result = await clothing_analyzer.analyze_clothing(image_bytes)
        return {"success": True, "result": result}
    except Exception as e:
        logger.error(f"의류 분석 오류: {e}")
        raise HTTPException(400, f"의류 분석 실패: {str(e)}")

@router.get("/models")
async def get_available_models():
    """사용 가능한 AI 모델 목록 (프론트엔드 호환)"""
    try:
        model_status = await ai_fitter.get_model_status()
        return {
            "models": [
                {
                    "id": "ootdiffusion",
                    "name": "OOT-Diffusion",
                    "description": "최신 Diffusion 기반 고품질 가상 피팅",
                    "quality": "Very High",
                    "speed": "Medium",
                    "enabled": "ootdiffusion" in model_status.get("available_models", [])
                },
                {
                    "id": "viton_hd", 
                    "name": "VITON-HD",
                    "description": "고해상도 가상 피팅 모델",
                    "quality": "High",
                    "speed": "Fast",
                    "enabled": "viton_hd" in model_status.get("available_models", [])
                }
            ],
            "default": "ootdiffusion",
            "system_info": model_status
        }
    except Exception as e:
        logger.error(f"모델 정보 조회 실패: {e}")
        return {
            "models": [],
            "error": str(e)
        }