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
