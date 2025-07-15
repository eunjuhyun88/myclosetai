"""
단계별 파이프라인 API - 각 단계마다 결과를 반환하고 다음 단계로 진행
app/api/step_by_step_routes.py
"""

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
import asyncio
import time
import logging
import uuid
import base64
from typing import Dict, Any, Optional
from PIL import Image
import numpy as np
import cv2
from io import BytesIO

# 기존 파이프라인 import
from ..ai_pipeline.enhanced_pipeline_manager import EnhancedPipelineManager
from ..core.config import Config

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/step-by-step", tags=["Step-by-Step Pipeline"])

# 전역 매니저
pipeline_manager = EnhancedPipelineManager()

# 활성 세션 저장
active_sessions: Dict[str, Dict[str, Any]] = {}

class StepByStepResponse:
    """단계별 응답 모델"""
    def __init__(self, 
                 success: bool,
                 session_id: str,
                 current_step: int,
                 step_name: str,
                 step_title: str,
                 output_image: Optional[str] = None,
                 processing_time: float = 0.0,
                 quality_score: float = 0.0,
                 confidence: float = 0.0,
                 metrics: Dict[str, Any] = None,
                 can_proceed: bool = True,
                 error_message: Optional[str] = None):
        
        self.success = success
        self.session_id = session_id
        self.current_step = current_step
        self.step_name = step_name
        self.step_title = step_title
        self.output_image = output_image
        self.processing_time = processing_time
        self.quality_score = quality_score
        self.confidence = confidence
        self.metrics = metrics or {}
        self.can_proceed = can_proceed
        self.error_message = error_message
    
    def to_dict(self):
        return {
            "success": self.success,
            "session_id": self.session_id,
            "current_step": self.current_step,
            "step_name": self.step_name,
            "step_title": self.step_title,
            "output_image": self.output_image,
            "processing_time": self.processing_time,
            "quality_score": self.quality_score,
            "confidence": self.confidence,
            "metrics": self.metrics,
            "can_proceed": self.can_proceed,
            "error_message": self.error_message
        }

@router.post("/initialize")
async def initialize_session(
    person_image: UploadFile = File(...),
    clothing_image: UploadFile = File(...),
    height: float = Form(170),
    weight: float = Form(65)
):
    """
    새로운 단계별 세션 초기화
    """
    try:
        # 세션 ID 생성
        session_id = f"step_{uuid.uuid4().hex[:12]}"
        
        # 이미지 로드 및 전처리
        person_img = await load_and_preprocess_image(person_image)
        cloth_img = await load_and_preprocess_image(clothing_image)
        
        # 세션 데이터 저장
        session_data = {
            "session_id": session_id,
            "person_image": person_img,
            "clothing_image": cloth_img,
            "body_measurements": {
                "height": height,
                "weight": weight
            },
            "current_step": 0,
            "step_results": {},
            "created_at": time.time()
        }
        
        active_sessions[session_id] = session_data
        
        logger.info(f"새 세션 초기화: {session_id}")
        
        return JSONResponse(content={
            "success": True,
            "session_id": session_id,
            "message": "세션이 성공적으로 초기화되었습니다.",
            "total_steps": 8,
            "ready_for_step": 1
        })
        
    except Exception as e:
        logger.error(f"세션 초기화 실패: {str(e)}")
        raise HTTPException(status_code=500, detail=f"세션 초기화 실패: {str(e)}")

@router.post("/process-step/{session_id}/{step_number}")
async def process_step(
    session_id: str,
    step_number: int
):
    """
    특정 단계 처리
    """
    try:
        # 세션 확인
        if session_id not in active_sessions:
            raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다.")
        
        session_data = active_sessions[session_id]
        
        # 단계 번호 검증
        if step_number < 1 or step_number > 8:
            raise HTTPException(status_code=400, detail="잘못된 단계 번호입니다. (1-8)")
        
        # 이전 단계가 완료되었는지 확인
        if step_number > 1 and (step_number - 1) not in session_data["step_results"]:
            raise HTTPException(status_code=400, detail="이전 단계를 먼저 완료해주세요.")
        
        step_start_time = time.time()
        
        # 단계별 처리
        result = await process_pipeline_step(session_data, step_number)
        
        processing_time = time.time() - step_start_time
        
        # 결과 저장
        session_data["step_results"][step_number] = {
            "result": result,
            "processing_time": processing_time,
            "timestamp": time.time()
        }
        session_data["current_step"] = step_number
        
        # 응답 생성
        response = StepByStepResponse(
            success=True,
            session_id=session_id,
            current_step=step_number,
            step_name=get_step_name(step_number),
            step_title=get_step_title(step_number),
            output_image=result.get("output_image"),
            processing_time=processing_time,
            quality_score=result.get("quality_score", 0.85),
            confidence=result.get("confidence", 0.90),
            metrics=result.get("metrics", {}),
            can_proceed=step_number < 8
        )
        
        logger.info(f"Step {step_number} 완료 (세션: {session_id}, 시간: {processing_time:.2f}s)")
        
        return JSONResponse(content=response.to_dict())
        
    except Exception as e:
        logger.error(f"Step {step_number} 처리 실패 (세션: {session_id}): {str(e)}")
        
        error_response = StepByStepResponse(
            success=False,
            session_id=session_id,
            current_step=step_number,
            step_name=get_step_name(step_number),
            step_title=get_step_title(step_number),
            can_proceed=False,
            error_message=str(e)
        )
        
        return JSONResponse(content=error_response.to_dict(), status_code=500)

@router.get("/session/{session_id}/status")
async def get_session_status(session_id: str):
    """
    세션 상태 조회
    """
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다.")
    
    session_data = active_sessions[session_id]
    
    completed_steps = list(session_data["step_results"].keys())
    next_step = max(completed_steps) + 1 if completed_steps else 1
    
    return JSONResponse(content={
        "session_id": session_id,
        "current_step": session_data["current_step"],
        "completed_steps": completed_steps,
        "next_step": next_step if next_step <= 8 else None,
        "total_steps": 8,
        "progress_percentage": (len(completed_steps) / 8) * 100,
        "created_at": session_data["created_at"]
    })

@router.get("/session/{session_id}/step/{step_number}/result")
async def get_step_result(session_id: str, step_number: int):
    """
    특정 단계 결과 조회
    """
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다.")
    
    session_data = active_sessions[session_id]
    
    if step_number not in session_data["step_results"]:
        raise HTTPException(status_code=404, detail=f"Step {step_number} 결과를 찾을 수 없습니다.")
    
    step_result = session_data["step_results"][step_number]
    
    return JSONResponse(content={
        "session_id": session_id,
        "step_number": step_number,
        "step_name": get_step_name(step_number),
        "step_title": get_step_title(step_number),
        "result": step_result["result"],
        "processing_time": step_result["processing_time"],
        "timestamp": step_result["timestamp"]
    })

@router.post("/session/{session_id}/finalize")
async def finalize_session(session_id: str):
    """
    세션 완료 및 최종 결과 생성
    """
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다.")
    
    session_data = active_sessions[session_id]
    
    # 모든 단계가 완료되었는지 확인
    if len(session_data["step_results"]) < 8:
        raise HTTPException(status_code=400, detail="모든 단계를 완료해주세요.")
    
    try:
        # 최종 결과 통합
        final_result = compile_final_result(session_data)
        
        # 세션에 최종 결과 저장
        session_data["final_result"] = final_result
        session_data["completed_at"] = time.time()
        
        total_processing_time = sum(
            step_data["processing_time"] 
            for step_data in session_data["step_results"].values()
        )
        
        return JSONResponse(content={
            "success": True,
            "session_id": session_id,
            "final_result": final_result,
            "total_processing_time": total_processing_time,
            "step_summary": {
                step_num: {
                    "name": get_step_name(step_num),
                    "processing_time": step_data["processing_time"],
                    "quality_score": step_data["result"].get("quality_score", 0.0)
                }
                for step_num, step_data in session_data["step_results"].items()
            }
        })
        
    except Exception as e:
        logger.error(f"세션 완료 처리 실패 (세션: {session_id}): {str(e)}")
        raise HTTPException(status_code=500, detail=f"최종 결과 생성 실패: {str(e)}")

@router.delete("/session/{session_id}")
async def cleanup_session(session_id: str):
    """
    세션 정리
    """
    if session_id in active_sessions:
        del active_sessions[session_id]
        logger.info(f"세션 정리됨: {session_id}")
        return JSONResponse(content={"success": True, "message": "세션이 정리되었습니다."})
    else:
        raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다.")

# 헬퍼 함수들

async def load_and_preprocess_image(upload_file: UploadFile) -> np.ndarray:
    """업로드된 파일을 numpy 배열로 변환"""
    image_bytes = await upload_file.read()
    image_pil = Image.open(BytesIO(image_bytes)).convert('RGB')
    image_pil = image_pil.resize((512, 512))  # 표준 크기로 리사이즈
    return np.array(image_pil)

async def process_pipeline_step(session_data: Dict[str, Any], step_number: int) -> Dict[str, Any]:
    """실제 파이프라인 단계 처리"""
    
    person_image = session_data["person_image"]
    clothing_image = session_data["clothing_image"]
    
    # 이전 단계 결과 가져오기
    previous_results = session_data["step_results"]
    
    if step_number == 1:
        # Step 1: Human Parsing
        result = await process_human_parsing(person_image)
        
    elif step_number == 2:
        # Step 2: Pose Estimation
        result = await process_pose_estimation(person_image)
        
    elif step_number == 3:
        # Step 3: Cloth Segmentation
        result = await process_cloth_segmentation(clothing_image)
        
    elif step_number == 4:
        # Step 4: Geometric Matching
        parsing_result = previous_results[1]["result"]
        pose_result = previous_results[2]["result"]
        cloth_result = previous_results[3]["result"]
        result = await process_geometric_matching(parsing_result, pose_result, cloth_result)
        
    elif step_number == 5:
        # Step 5: Cloth Warping
        matching_result = previous_results[4]["result"]
        result = await process_cloth_warping(clothing_image, matching_result)
        
    elif step_number == 6:
        # Step 6: Virtual Fitting
        warping_result = previous_results[5]["result"]
        parsing_result = previous_results[1]["result"]
        result = await process_virtual_fitting(person_image, warping_result, parsing_result)
        
    elif step_number == 7:
        # Step 7: Post Processing
        fitting_result = previous_results[6]["result"]
        result = await process_post_processing(fitting_result)
        
    elif step_number == 8:
        # Step 8: Quality Assessment
        final_image = previous_results[7]["result"]
        result = await process_quality_assessment(person_image, final_image)
        
    else:
        raise ValueError(f"잘못된 단계 번호: {step_number}")
    
    return result

# 각 단계별 처리 함수들 (시뮬레이션)

async def process_human_parsing(person_image: np.ndarray) -> Dict[str, Any]:
    """Step 1: 인체 파싱"""
    await asyncio.sleep(1.0)  # 처리 시간 시뮬레이션
    
    # 더미 파싱 맵 생성
    h, w = person_image.shape[:2]
    parsing_map = np.random.randint(0, 20, (h, w), dtype=np.uint8)
    
    # 시각화 이미지 생성
    colored_map = cv2.applyColorMap((parsing_map * 12).astype(np.uint8), cv2.COLORMAP_JET)
    visualization = cv2.addWeighted(person_image, 0.6, colored_map, 0.4, 0)
    
    return {
        "parsing_map": parsing_map.tolist(),
        "output_image": image_to_base64(visualization),
        "quality_score": 0.95,
        "confidence": 0.92,
        "metrics": {
            "segments_detected": 20,
            "coverage_ratio": 0.88
        }
    }

async def process_pose_estimation(person_image: np.ndarray) -> Dict[str, Any]:
    """Step 2: 자세 추정"""
    await asyncio.sleep(0.8)
    
    h, w = person_image.shape[:2]
    
    # 더미 키포인트 생성
    keypoints = np.random.rand(18, 3) * [w, h, 1]
    
    # 키포인트 시각화
    visualization = person_image.copy()
    for i, (x, y, conf) in enumerate(keypoints):
        if conf > 0.5:
            cv2.circle(visualization, (int(x), int(y)), 5, (0, 255, 0), -1)
    
    return {
        "keypoints": keypoints.tolist(),
        "output_image": image_to_base64(visualization),
        "quality_score": 0.90,
        "confidence": 0.88,
        "metrics": {
            "keypoints_detected": 18,
            "average_confidence": 0.85
        }
    }

async def process_cloth_segmentation(clothing_image: np.ndarray) -> Dict[str, Any]:
    """Step 3: 의류 분할"""
    await asyncio.sleep(1.1)
    
    # 더미 세그멘테이션 (실제로는 U2-Net 등 사용)
    segmented = clothing_image.copy()
    
    return {
        "segmented_cloth": segmented.tolist(),
        "output_image": image_to_base64(segmented),
        "quality_score": 0.93,
        "confidence": 0.91,
        "metrics": {
            "edge_quality": 0.89,
            "background_removal": 0.95
        }
    }

async def process_geometric_matching(parsing_result, pose_result, cloth_result) -> Dict[str, Any]:
    """Step 4: 기하학적 매칭"""
    await asyncio.sleep(0.7)
    
    # 더미 매칭 포인트
    cloth_points = np.random.rand(32, 2) * 512
    body_points = np.random.rand(32, 2) * 512
    
    # 매칭 시각화
    canvas = np.zeros((512, 512, 3), dtype=np.uint8)
    for i in range(len(cloth_points)):
        cv2.circle(canvas, tuple(cloth_points[i].astype(int)), 3, (255, 0, 0), -1)
        cv2.circle(canvas, tuple(body_points[i].astype(int)), 3, (0, 255, 0), -1)
        cv2.line(canvas, tuple(cloth_points[i].astype(int)), tuple(body_points[i].astype(int)), (255, 255, 255), 1)
    
    return {
        "cloth_keypoints": cloth_points.tolist(),
        "body_keypoints": body_points.tolist(),
        "output_image": image_to_base64(canvas),
        "quality_score": 0.84,
        "confidence": 0.87,
        "metrics": {
            "matching_accuracy": 0.84,
            "control_points": 32
        }
    }

async def process_cloth_warping(clothing_image: np.ndarray, matching_result) -> Dict[str, Any]:
    """Step 5: 의류 워핑"""
    await asyncio.sleep(0.6)
    
    # 더미 워핑 (실제로는 TPS 변환 등 사용)
    warped = clothing_image.copy()
    
    return {
        "warped_cloth": warped.tolist(),
        "output_image": image_to_base64(warped),
        "quality_score": 0.80,
        "confidence": 0.83,
        "metrics": {
            "naturalness": 0.80,
            "deformation_quality": 0.78
        }
    }

async def process_virtual_fitting(person_image: np.ndarray, warping_result, parsing_result) -> Dict[str, Any]:
    """Step 6: 가상 피팅"""
    await asyncio.sleep(1.8)
    
    # 더미 피팅 (실제로는 HR-VITON 등 사용)
    fitted = person_image.copy()
    
    return {
        "fitted_image": fitted.tolist(),
        "output_image": image_to_base64(fitted),
        "quality_score": 0.88,
        "confidence": 0.90,
        "metrics": {
            "realism": 0.88,
            "fit_quality": 0.85
        }
    }

async def process_post_processing(fitting_result) -> Dict[str, Any]:
    """Step 7: 후처리"""
    await asyncio.sleep(0.5)
    
    # 더미 후처리
    enhanced = np.array(fitting_result["fitted_image"], dtype=np.uint8)
    
    return {
        "enhanced_image": enhanced.tolist(),
        "output_image": image_to_base64(enhanced),
        "quality_score": 0.91,
        "confidence": 0.89,
        "metrics": {
            "color_enhancement": 0.87,
            "edge_smoothing": 0.82
        }
    }

async def process_quality_assessment(person_image: np.ndarray, final_result) -> Dict[str, Any]:
    """Step 8: 품질 평가"""
    await asyncio.sleep(0.3)
    
    enhanced_image = np.array(final_result["enhanced_image"], dtype=np.uint8)
    
    quality_metrics = {
        "overall_score": 0.88,
        "fit_coverage": 0.85,
        "color_preservation": 0.92,
        "edge_quality": 0.89,
        "realism_score": 0.87
    }
    
    return {
        "final_image": enhanced_image.tolist(),
        "output_image": image_to_base64(enhanced_image),
        "quality_score": quality_metrics["overall_score"],
        "confidence": 0.91,
        "metrics": quality_metrics
    }

def compile_final_result(session_data: Dict[str, Any]) -> Dict[str, Any]:
    """최종 결과 컴파일"""
    
    final_step_result = session_data["step_results"][8]["result"]
    
    return {
        "fitted_image_base64": final_step_result["output_image"],
        "overall_quality": final_step_result["quality_score"],
        "confidence": final_step_result["confidence"],
        "quality_metrics": final_step_result["metrics"],
        "recommendations": [
            "✅ 전체적인 핏이 좋습니다!",
            "✅ 색상 조합이 자연스럽습니다.",
            "✅ 이 스타일이 잘 어울립니다."
        ]
    }

def get_step_name(step_number: int) -> str:
    """단계 이름 반환"""
    step_names = [
        "", "human_parsing", "pose_estimation", "cloth_segmentation",
        "geometric_matching", "cloth_warping", "virtual_fitting",
        "post_processing", "quality_assessment"
    ]
    return step_names[step_number] if 1 <= step_number <= 8 else "unknown"

def get_step_title(step_number: int) -> str:
    """단계 제목 반환"""
    step_titles = [
        "", "인체 파싱", "자세 추정", "의류 분할",
        "기하학적 매칭", "의류 워핑", "가상 피팅",
        "후처리", "품질 평가"
    ]
    return step_titles[step_number] if 1 <= step_number <= 8 else "알 수 없는 단계"

def image_to_base64(image: np.ndarray) -> str:
    """이미지를 base64 문자열로 변환"""
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    
    _, buffer = cv2.imencode('.png', image)
    return base64.b64encode(buffer).decode('utf-8')

# 라우터를 main app에 등록하기 위한 export
__all__ = ["router"]