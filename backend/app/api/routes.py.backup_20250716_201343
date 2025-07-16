"""
MyCloset AI Backend API 라우터
8단계 가상 피팅 시스템의 주요 API 엔드포인트
"""

import os
import asyncio
import time
from typing import Optional, Dict, Any
import base64
from io import BytesIO

from fastapi import APIRouter, File, UploadFile, Form, HTTPException, BackgroundTasks, Depends
from fastapi.responses import JSONResponse
from PIL import Image
import numpy as np

from app.models.schemas import (
    VirtualTryOnRequest, VirtualTryOnResponse, 
    ProcessingStatus, SystemStatus
)
from app.utils.image_utils import validate_image, save_uploaded_file, load_image_from_upload
from app.utils.validators import validate_measurements
from app.main import get_pipeline_manager, is_pipeline_ready

# API 라우터 생성
api_router = APIRouter()

# 진행 중인 작업들을 추적하는 딕셔너리 (실제로는 Redis나 데이터베이스 사용 권장)
active_tasks: Dict[str, Dict[str, Any]] = {}


@api_router.post("/virtual-tryon", response_model=VirtualTryOnResponse)
async def virtual_tryon_complete(
    background_tasks: BackgroundTasks,
    person_image: UploadFile = File(..., description="사용자 전신 사진"),
    clothing_image: UploadFile = File(..., description="의류 이미지"),
    height: float = Form(..., description="키 (cm)", ge=120, le=220),
    weight: float = Form(..., description="몸무게 (kg)", ge=30, le=200),
    chest: Optional[float] = Form(None, description="가슴둘레 (cm)", ge=70, le=150),
    waist: Optional[float] = Form(None, description="허리둘레 (cm)", ge=50, le=150),
    hips: Optional[float] = Form(None, description="엉덩이둘레 (cm)", ge=70, le=150),
    clothing_type: str = Form("shirt", description="의류 타입"),
    fabric_type: str = Form("cotton", description="재질"),
    quality_level: str = Form("high", description="품질 레벨 (fast/balanced/high/ultra)"),
    style_preferences: Optional[str] = Form(None, description="스타일 선호도 (JSON)"),
    save_intermediate: bool = Form(False, description="중간 결과 저장 여부"),
    async_processing: bool = Form(False, description="비동기 처리 여부")
):
    """
    완전한 8단계 가상 피팅 처리
    
    프론트엔드의 ProcessingStep과 연동되는 메인 API
    """
    
    # 파이프라인 준비 상태 확인
    if not is_pipeline_ready():
        raise HTTPException(
            status_code=503, 
            detail="AI 파이프라인이 준비되지 않았습니다. 잠시 후 다시 시도해주세요."
        )
    
    # 입력 검증
    await validate_image(person_image, "person_image")
    await validate_image(clothing_image, "clothing_image")
    
    # 신체 치수 검증
    body_measurements = validate_measurements({
        "height": height,
        "weight": weight,
        "chest": chest,
        "waist": waist,
        "hips": hips
    })
    
    # 스타일 선호도 파싱
    style_prefs = {}
    if style_preferences:
        try:
            import json
            style_prefs = json.loads(style_preferences)
        except:
            style_prefs = {}
    
    # 작업 ID 생성
    task_id = f"tryon_{int(time.time() * 1000)}"
    
    try:
        # 이미지 로드 및 전처리
        person_pil = await load_image_from_upload(person_image)
        clothing_pil = await load_image_from_upload(clothing_image)
        
        # 파이프라인 매니저 가져오기
        pipeline_manager = get_pipeline_manager()
        
        # 비동기 처리인 경우
        if async_processing:
            # 백그라운드 작업으로 등록
            active_tasks[task_id] = {
                "status": "processing",
                "progress": 0,
                "current_step": "초기화",
                "start_time": time.time(),
                "result": None,
                "error": None
            }
            
            background_tasks.add_task(
                process_virtual_fitting_background,
                task_id, pipeline_manager, person_pil, clothing_pil,
                body_measurements, clothing_type, fabric_type,
                quality_level, style_prefs, save_intermediate
            )
            
            return VirtualTryOnResponse(
                success=True,
                task_id=task_id,
                message="가상 피팅이 시작되었습니다. /api/status/{task_id}로 진행상황을 확인하세요.",
                processing_time=0.0,
                async_processing=True
            )
        
        # 동기 처리
        else:
            # 진행상황 콜백 함수
            async def progress_callback(stage: str, percentage: int):
                active_tasks[task_id] = {
                    "status": "processing",
                    "progress": percentage,
                    "current_step": stage,
                    "start_time": active_tasks.get(task_id, {}).get("start_time", time.time()),
                    "result": None,
                    "error": None
                }
            
            # 초기 상태 설정
            active_tasks[task_id] = {
                "status": "processing",
                "progress": 0,
                "current_step": "시작",
                "start_time": time.time(),
                "result": None,
                "error": None
            }
            
            # 파이프라인 실행
            result = await pipeline_manager.process_complete_virtual_fitting(
                person_image=person_pil,
                clothing_image=clothing_pil,
                body_measurements=body_measurements,
                clothing_type=clothing_type,
                fabric_type=fabric_type,
                style_preferences=style_prefs,
                quality_target=0.8,
                progress_callback=progress_callback,
                save_intermediate=save_intermediate,
                enable_auto_retry=True
            )
            
            # 결과 저장
            result_image_path = None
            result_image_base64 = None
            
            if result['success'] and result.get('result_image'):
                # 결과 이미지 저장
                result_image_path = await save_result_image(result['result_image'], task_id)
                
                # Base64 인코딩
                result_image_base64 = pil_to_base64(result['result_image'])
            
            # 최종 상태 업데이트
            active_tasks[task_id] = {
                "status": "completed" if result['success'] else "failed",
                "progress": 100,
                "current_step": "완료",
                "start_time": active_tasks[task_id]["start_time"],
                "result": result,
                "error": result.get('error') if not result['success'] else None
            }
            
            return VirtualTryOnResponse(
                success=result['success'],
                task_id=task_id,
                message="가상 피팅이 완료되었습니다." if result['success'] else f"가상 피팅 실패: {result.get('error', '알 수 없는 오류')}",
                processing_time=result.get('total_processing_time', 0.0),
                
                # 결과 이미지
                result_image_base64=result_image_base64,
                result_image_url=f"/static/results/{os.path.basename(result_image_path)}" if result_image_path else None,
                
                # 품질 메트릭
                quality_score=result.get('final_quality_score', 0.0),
                fit_score=result.get('fit_analysis', {}).get('overall_fit_score', 0.0),
                confidence=result.get('confidence', 0.0),
                
                # 처리 정보
                steps_completed=len(result.get('step_results_summary', {})),
                processing_details=result.get('processing_statistics', {}),
                
                # 개선 제안
                recommendations=result.get('improvement_suggestions', {}),
                
                # 중간 결과 (요청된 경우)
                intermediate_results=result.get('intermediate_results', {}) if save_intermediate else {},
                
                async_processing=False
            )
    
    except Exception as e:
        # 에러 상태 업데이트
        if task_id in active_tasks:
            active_tasks[task_id]["status"] = "failed"
            active_tasks[task_id]["error"] = str(e)
        
        raise HTTPException(status_code=500, detail=f"가상 피팅 처리 중 오류: {str(e)}")


@api_router.get("/status/{task_id}", response_model=ProcessingStatus)
async def get_processing_status(task_id: str):
    """작업 진행상황 조회"""
    
    if task_id not in active_tasks:
        raise HTTPException(status_code=404, detail="작업을 찾을 수 없습니다.")
    
    task_info = active_tasks[task_id]
    
    return ProcessingStatus(
        task_id=task_id,
        status=task_info["status"],
        progress=task_info["progress"],
        current_step=task_info["current_step"],
        elapsed_time=time.time() - task_info["start_time"],
        result=task_info["result"],
        error=task_info["error"]
    )


@api_router.post("/quick-fit")
async def quick_virtual_fitting(
    person_image: UploadFile = File(...),
    clothing_image: UploadFile = File(...),
    height: float = Form(170),
    weight: float = Form(65)
):
    """
    빠른 가상 피팅 (품질보다 속도 우선)
    """
    
    if not is_pipeline_ready():
        raise HTTPException(status_code=503, detail="AI 파이프라인이 준비되지 않았습니다.")
    
    try:
        # 이미지 로드
        person_pil = await load_image_from_upload(person_image)
        clothing_pil = await load_image_from_upload(clothing_image)
        
        # 파이프라인 매니저
        pipeline_manager = get_pipeline_manager()
        
        # 빠른 처리 설정
        result = await pipeline_manager.process_complete_virtual_fitting(
            person_image=person_pil,
            clothing_image=clothing_pil,
            body_measurements={"height": height, "weight": weight},
            clothing_type="shirt",
            fabric_type="cotton",
            quality_target=0.6,  # 낮은 품질 목표
            save_intermediate=False,
            enable_auto_retry=False
        )
        
        if result['success']:
            result_base64 = pil_to_base64(result['result_image'])
            
            return {
                "success": True,
                "result_image": result_base64,
                "quality_score": result.get('final_quality_score', 0.0),
                "processing_time": result.get('total_processing_time', 0.0)
            }
        else:
            raise HTTPException(status_code=500, detail=result.get('error', '처리 실패'))
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"빠른 피팅 처리 실패: {str(e)}")


@api_router.post("/parse-human")
async def parse_human_only(
    person_image: UploadFile = File(...)
):
    """인체 파싱만 수행 (테스트/디버깅용)"""
    
    if not is_pipeline_ready():
        raise HTTPException(status_code=503, detail="AI 파이프라인이 준비되지 않았습니다.")
    
    try:
        person_pil = await load_image_from_upload(person_image)
        pipeline_manager = get_pipeline_manager()
        
        # 1단계만 실행
        if hasattr(pipeline_manager, 'steps') and 'human_parsing' in pipeline_manager.steps:
            step = pipeline_manager.steps['human_parsing']
            
            # 이미지를 텐서로 변환
            person_tensor = pipeline_manager.data_converter.pil_to_tensor(person_pil, pipeline_manager.device)
            
            # 인체 파싱 실행
            result = await step.process(person_tensor)
            
            return {
                "success": True,
                "confidence": result.get('confidence', 0.0),
                "body_parts_detected": len(result.get('body_parts_detected', [])),
                "processing_time": result.get('processing_time', 0.0)
            }
        else:
            raise HTTPException(status_code=503, detail="인체 파싱 모델이 로드되지 않았습니다.")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"인체 파싱 실패: {str(e)}")


@api_router.get("/pipeline/status", response_model=SystemStatus)
async def get_pipeline_status():
    """파이프라인 상태 조회"""
    
    try:
        if is_pipeline_ready():
            pipeline_manager = get_pipeline_manager()
            status = await pipeline_manager.get_pipeline_status()
            
            return SystemStatus(
                pipeline_ready=True,
                pipeline_status=status,
                active_tasks=len(active_tasks),
                system_health="healthy"
            )
        else:
            return SystemStatus(
                pipeline_ready=False,
                pipeline_status={"error": "Pipeline not initialized"},
                active_tasks=len(active_tasks),
                system_health="degraded"
            )
    
    except Exception as e:
        return SystemStatus(
            pipeline_ready=False,
            pipeline_status={"error": str(e)},
            active_tasks=len(active_tasks),
            system_health="unhealthy"
        )


@api_router.delete("/tasks/{task_id}")
async def cancel_task(task_id: str):
    """작업 취소"""
    
    if task_id not in active_tasks:
        raise HTTPException(status_code=404, detail="작업을 찾을 수 없습니다.")
    
    task_info = active_tasks[task_id]
    
    if task_info["status"] == "processing":
        task_info["status"] = "cancelled"
        task_info["error"] = "사용자에 의해 취소됨"
        
        return {"success": True, "message": "작업이 취소되었습니다."}
    else:
        return {"success": False, "message": f"작업을 취소할 수 없습니다. 현재 상태: {task_info['status']}"}


@api_router.get("/tasks")
async def list_active_tasks():
    """활성 작업 목록 조회"""
    
    return {
        "active_tasks": len(active_tasks),
        "tasks": {
            task_id: {
                "status": info["status"],
                "progress": info["progress"],
                "current_step": info["current_step"],
                "elapsed_time": time.time() - info["start_time"]
            }
            for task_id, info in active_tasks.items()
        }
    }


# 백그라운드 작업 함수
async def process_virtual_fitting_background(
    task_id: str,
    pipeline_manager,
    person_image: Image.Image,
    clothing_image: Image.Image,
    body_measurements: Dict[str, Any],
    clothing_type: str,
    fabric_type: str,
    quality_level: str,
    style_preferences: Dict[str, Any],
    save_intermediate: bool
):
    """백그라운드에서 가상 피팅 처리"""
    
    try:
        # 진행상황 콜백
        async def progress_callback(stage: str, percentage: int):
            if task_id in active_tasks:
                active_tasks[task_id]["progress"] = percentage
                active_tasks[task_id]["current_step"] = stage
        
        # 파이프라인 실행
        result = await pipeline_manager.process_complete_virtual_fitting(
            person_image=person_image,
            clothing_image=clothing_image,
            body_measurements=body_measurements,
            clothing_type=clothing_type,
            fabric_type=fabric_type,
            style_preferences=style_preferences,
            quality_target=0.8,
            progress_callback=progress_callback,
            save_intermediate=save_intermediate,
            enable_auto_retry=True
        )
        
        # 결과 저장
        if result['success'] and result.get('result_image'):
            result_image_path = await save_result_image(result['result_image'], task_id)
            result['result_image_url'] = f"/static/results/{os.path.basename(result_image_path)}"
            result['result_image_base64'] = pil_to_base64(result['result_image'])
        
        # 완료 상태 업데이트
        active_tasks[task_id]["status"] = "completed"
        active_tasks[task_id]["progress"] = 100
        active_tasks[task_id]["current_step"] = "완료"
        active_tasks[task_id]["result"] = result
        
    except Exception as e:
        # 실패 상태 업데이트
        active_tasks[task_id]["status"] = "failed"
        active_tasks[task_id]["error"] = str(e)


# 유틸리티 함수들
async def save_result_image(image: Image.Image, task_id: str) -> str:
    """결과 이미지 저장"""
    from app.core.config import get_settings
    settings = get_settings()
    
    os.makedirs(settings.RESULT_DIR, exist_ok=True)
    
    filename = f"result_{task_id}.jpg"
    filepath = os.path.join(settings.RESULT_DIR, filename)
    
    image.save(filepath, "JPEG", quality=95)
    return filepath


def pil_to_base64(image: Image.Image) -> str:
    """PIL 이미지를 Base64로 변환"""
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=95)
    img_bytes = buffer.getvalue()
    return base64.b64encode(img_bytes).decode('utf-8')