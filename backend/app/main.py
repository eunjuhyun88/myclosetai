"""
MyCloset AI Virtual Try-On - FastAPI 메인 서버
실제로 작동하는 가상 피팅 웹 서비스
"""
import os
import asyncio
import logging
import time
import base64
import uuid
from datetime import datetime
from typing import Optional, Dict, Any
from io import BytesIO

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
import uvicorn

# 우리가 만든 완전한 파이프라인 import
from complete_virtual_fitting_pipeline import (
    CompleteVirtualFittingPipeline,
    get_global_pipeline,
    cleanup_global_pipeline
)

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# FastAPI 앱 생성
app = FastAPI(
    title="MyCloset AI Virtual Try-On",
    description="AI 기반 가상 피팅 시스템 - 사진 한 장으로 구현되는 초개인화된 핏",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS 설정 (프론트엔드와 연결)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # React 개발 서버
        "http://localhost:5173",  # Vite 개발 서버
        "http://localhost:8080",  # Vue 개발 서버
        "https://mycloset-ai.vercel.app",  # 배포 도메인 (예시)
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 정적 파일 서빙 (업로드된 이미지, 결과 이미지)
os.makedirs("static/uploads", exist_ok=True)
os.makedirs("static/results", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# 전역 변수
DEVICE = "mps" if os.environ.get("USE_MPS") == "true" else "cpu"
MAX_IMAGE_SIZE = 2048
ALLOWED_FORMATS = {"JPEG", "JPG", "PNG", "WEBP"}

# 진행 상황 추적을 위한 딕셔너리
processing_status = {}

@app.on_event("startup")
async def startup_event():
    """서버 시작 시 AI 모델 초기화"""
    logger.info("🚀 MyCloset AI 서버 시작 - AI 모델 로딩 중...")
    
    try:
        # 파이프라인 초기화
        pipeline_config = {
            'pipeline': {
                'quality_level': 'high',
                'enable_caching': True,
                'memory_optimization': True
            }
        }
        
        pipeline = await get_global_pipeline(device=DEVICE, config=pipeline_config)
        
        logger.info(f"✅ AI 모델 로딩 완료 - 디바이스: {DEVICE}")
        logger.info("🎯 가상 피팅 서비스 준비 완료!")
        
    except Exception as e:
        logger.error(f"❌ AI 모델 초기화 실패: {e}")
        raise e

@app.on_event("shutdown")
async def shutdown_event():
    """서버 종료 시 리소스 정리"""
    logger.info("🧹 서버 종료 - 리소스 정리 중...")
    await cleanup_global_pipeline()
    logger.info("✅ 리소스 정리 완료")

# ============================================================================
# 메인 API 엔드포인트들
# ============================================================================

@app.get("/", response_class=HTMLResponse)
async def root():
    """메인 페이지 (간단한 HTML 인터페이스)"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>MyCloset AI - Virtual Try-On</title>
        <meta charset="utf-8">
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
            .container { text-align: center; }
            .upload-section { margin: 20px 0; padding: 20px; border: 2px dashed #ccc; }
            .result-section { margin: 20px 0; }
            input[type="file"] { margin: 10px; }
            button { background: #007bff; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; }
            button:hover { background: #0056b3; }
            .progress { width: 100%; height: 20px; background: #f0f0f0; border-radius: 10px; overflow: hidden; }
            .progress-bar { height: 100%; background: #007bff; transition: width 0.3s; }
            .result-image { max-width: 100%; margin: 10px; border: 1px solid #ddd; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎯 MyCloset AI - Virtual Try-On</h1>
            <p>AI 기반 가상 피팅 시스템 - 사진 한 장으로 구현되는 초개인화된 핏</p>
            
            <div class="upload-section">
                <h3>📸 이미지 업로드</h3>
                <form id="uploadForm" enctype="multipart/form-data">
                    <div>
                        <label>사람 사진:</label>
                        <input type="file" id="personImage" name="person_image" accept="image/*" required>
                    </div>
                    <div>
                        <label>옷 사진:</label>
                        <input type="file" id="clothingImage" name="clothing_image" accept="image/*" required>
                    </div>
                    <div>
                        <label>키 (cm):</label>
                        <input type="number" id="height" name="height" value="170" min="140" max="220">
                        <label>몸무게 (kg):</label>
                        <input type="number" id="weight" name="weight" value="65" min="40" max="150">
                    </div>
                    <div>
                        <label>의류 타입:</label>
                        <select id="clothingType" name="clothing_type">
                            <option value="shirt">셔츠</option>
                            <option value="dress">원피스</option>
                            <option value="pants">바지</option>
                            <option value="jacket">재킷</option>
                        </select>
                        <label>소재:</label>
                        <select id="fabricType" name="fabric_type">
                            <option value="cotton">면</option>
                            <option value="denim">데님</option>
                            <option value="silk">실크</option>
                            <option value="wool">울</option>
                        </select>
                    </div>
                    <button type="submit">🎨 가상 피팅 시작</button>
                </form>
            </div>
            
            <div id="progressSection" style="display: none;">
                <h3>⏳ 처리 중...</h3>
                <div class="progress">
                    <div id="progressBar" class="progress-bar" style="width: 0%"></div>
                </div>
                <p id="progressText">초기화 중...</p>
            </div>
            
            <div id="resultSection" class="result-section" style="display: none;">
                <h3>✨ 가상 피팅 결과</h3>
                <div id="resultContent"></div>
            </div>
        </div>
        
        <script>
            document.getElementById('uploadForm').addEventListener('submit', async function(e) {
                e.preventDefault();
                
                const formData = new FormData(this);
                const progressSection = document.getElementById('progressSection');
                const resultSection = document.getElementById('resultSection');
                const progressBar = document.getElementById('progressBar');
                const progressText = document.getElementById('progressText');
                
                // UI 초기화
                progressSection.style.display = 'block';
                resultSection.style.display = 'none';
                progressBar.style.width = '0%';
                progressText.textContent = '가상 피팅 시작 중...';
                
                try {
                    // 가상 피팅 요청
                    const response = await fetch('/api/virtual-tryon', {
                        method: 'POST',
                        body: formData
                    });
                    
                    const result = await response.json();
                    
                    if (result.success) {
                        // 태스크 ID로 진행 상황 추적
                        await trackProgress(result.task_id);
                    } else {
                        throw new Error(result.message || '가상 피팅 실패');
                    }
                    
                } catch (error) {
                    alert('오류 발생: ' + error.message);
                    progressSection.style.display = 'none';
                }
            });
            
            async function trackProgress(taskId) {
                const progressBar = document.getElementById('progressBar');
                const progressText = document.getElementById('progressText');
                const resultSection = document.getElementById('resultSection');
                const resultContent = document.getElementById('resultContent');
                
                const checkStatus = async () => {
                    try {
                        const response = await fetch(`/api/status/${taskId}`);
                        const status = await response.json();
                        
                        // 진행률 업데이트
                        progressBar.style.width = status.progress + '%';
                        progressText.textContent = status.message;
                        
                        if (status.status === 'completed') {
                            // 완료 - 결과 표시
                            document.getElementById('progressSection').style.display = 'none';
                            resultSection.style.display = 'block';
                            
                            resultContent.innerHTML = `
                                <img src="data:image/jpeg;base64,${status.result.result_image_base64}" class="result-image" alt="가상 피팅 결과">
                                <div>
                                    <p><strong>전체 품질:</strong> ${(status.result.overall_quality * 100).toFixed(1)}%</p>
                                    <p><strong>핏 점수:</strong> ${(status.result.fit_analysis.fit_score * 100).toFixed(1)}%</p>
                                    <p><strong>처리 시간:</strong> ${status.result.processing_stats.total_processing_time.toFixed(2)}초</p>
                                    <div><strong>추천사항:</strong></div>
                                    <ul>${status.result.recommendations.map(r => `<li>${r}</li>`).join('')}</ul>
                                </div>
                            `;
                        } else if (status.status === 'failed') {
                            throw new Error(status.error || '처리 실패');
                        } else {
                            // 계속 진행 중
                            setTimeout(checkStatus, 1000);
                        }
                        
                    } catch (error) {
                        alert('상태 확인 실패: ' + error.message);
                        document.getElementById('progressSection').style.display = 'none';
                    }
                };
                
                checkStatus();
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.get("/health")
async def health_check():
    """헬스 체크"""
    try:
        pipeline = await get_global_pipeline()
        pipeline_info = await pipeline.get_pipeline_info()
        
        return {
            "status": "healthy",
            "message": "MyCloset AI Virtual Try-On API is running",
            "device": DEVICE,
            "models_loaded": pipeline_info["initialized"],
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy", 
                "message": f"서비스 초기화 중: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }
        )

@app.post("/api/virtual-tryon")
async def virtual_tryon_endpoint(
    background_tasks: BackgroundTasks,
    person_image: UploadFile = File(..., description="사용자 사진 (정면, 전신)"),
    clothing_image: UploadFile = File(..., description="의류 사진 (배경 제거 권장)"),
    height: float = Form(170, description="키 (cm)", ge=140, le=220),
    weight: float = Form(65, description="몸무게 (kg)", ge=40, le=150),
    chest: Optional[float] = Form(None, description="가슴둘레 (cm)", ge=70, le=150),
    waist: Optional[float] = Form(None, description="허리둘레 (cm)", ge=60, le=120),
    hips: Optional[float] = Form(None, description="엉덩이둘레 (cm)", ge=70, le=150),
    clothing_type: str = Form("shirt", description="의류 타입"),
    fabric_type: str = Form("cotton", description="천 소재")
):
    """메인 가상 피팅 API"""
    
    # 태스크 ID 생성
    task_id = str(uuid.uuid4())
    
    # 초기 상태 설정
    processing_status[task_id] = {
        "status": "started",
        "progress": 0,
        "message": "가상 피팅 시작 중...",
        "created_at": datetime.now()
    }
    
    # 입력 검증
    try:
        person_pil = await validate_and_load_image(person_image, "person")
        clothing_pil = await validate_and_load_image(clothing_image, "clothing")
    except Exception as e:
        processing_status[task_id] = {
            "status": "failed",
            "error": f"이미지 검증 실패: {str(e)}"
        }
        raise HTTPException(status_code=400, detail=str(e))
    
    # 신체 치수 구성
    body_measurements = {
        "height": height,
        "weight": weight,
        "bmi": weight / ((height/100) ** 2)
    }
    
    if chest:
        body_measurements["chest"] = chest
    if waist:
        body_measurements["waist"] = waist  
    if hips:
        body_measurements["hips"] = hips
    
    # 백그라운드에서 가상 피팅 실행
    background_tasks.add_task(
        process_virtual_fitting_task,
        task_id,
        person_pil,
        clothing_pil,
        body_measurements,
        clothing_type,
        fabric_type
    )
    
    return {
        "success": True,
        "task_id": task_id,
        "message": "가상 피팅이 시작되었습니다. 상태를 확인하려면 /api/status/{task_id}를 호출하세요.",
        "estimated_time": "15-30초"
    }

@app.get("/api/status/{task_id}")
async def get_task_status(task_id: str):
    """태스크 상태 확인"""
    
    if task_id not in processing_status:
        raise HTTPException(status_code=404, detail="태스크를 찾을 수 없습니다")
    
    status = processing_status[task_id]
    
    # 완료된 태스크는 일정 시간 후 삭제 (메모리 관리)
    if status["status"] in ["completed", "failed"]:
        created_at = status.get("created_at", datetime.now())
        if (datetime.now() - created_at).total_seconds() > 300:  # 5분 후 삭제
            del processing_status[task_id]
            raise HTTPException(status_code=410, detail="태스크 결과가 만료되었습니다")
    
    return status

@app.get("/api/pipeline-info")
async def get_pipeline_info():
    """파이프라인 정보 조회"""
    try:
        pipeline = await get_global_pipeline()
        info = await pipeline.get_pipeline_info()
        return info
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"파이프라인 정보 조회 실패: {str(e)}")

@app.post("/api/quick-demo")
async def quick_demo_endpoint(
    person_image: UploadFile = File(...),
    clothing_image: UploadFile = File(...)
):
    """빠른 데모 (간단한 합성)"""
    
    try:
        # 이미지 로드
        person_pil = await validate_and_load_image(person_image, "person")
        clothing_pil = await validate_and_load_image(clothing_image, "clothing") 
        
        # 간단한 합성
        result_image = create_simple_composite(person_pil, clothing_pil)
        
        # Base64 인코딩
        buffer = BytesIO()
        result_image.save(buffer, format="JPEG", quality=85)
        result_base64 = base64.b64encode(buffer.getvalue()).decode()
        
        return {
            "success": True,
            "result_image_base64": result_base64,
            "message": "간단한 데모 합성 완료",
            "quality": "demo",
            "processing_time": 0.5
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"데모 처리 실패: {str(e)}")

# ============================================================================
# 헬퍼 함수들
# ============================================================================

async def validate_and_load_image(upload_file: UploadFile, image_type: str) -> Image.Image:
    """이미지 검증 및 로드"""
    
    # 파일 크기 확인 (10MB 제한)
    content = await upload_file.read()
    if len(content) > 10 * 1024 * 1024:
        raise ValueError(f"{image_type} 이미지가 너무 큽니다 (최대 10MB)")
    
    # 이미지 로드
    try:
        image = Image.open(BytesIO(content))
    except Exception:
        raise ValueError(f"{image_type} 이미지 형식이 올바르지 않습니다")
    
    # 형식 확인
    if image.format not in ALLOWED_FORMATS:
        raise ValueError(f"지원하지 않는 이미지 형식: {image.format}")
    
    # RGB 변환
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # 크기 제한
    if max(image.width, image.height) > MAX_IMAGE_SIZE:
        # 비율 유지하며 리사이즈
        ratio = MAX_IMAGE_SIZE / max(image.width, image.height)
        new_width = int(image.width * ratio)
        new_height = int(image.height * ratio)
        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    return image

async def process_virtual_fitting_task(
    task_id: str,
    person_image: Image.Image,
    clothing_image: Image.Image,
    body_measurements: Dict[str, float],
    clothing_type: str,
    fabric_type: str
):
    """백그라운드에서 실행되는 가상 피팅 태스크"""
    
    try:
        # 진행률 콜백 함수
        async def progress_callback(status_msg: str, progress: int):
            processing_status[task_id] = {
                "status": "processing",
                "progress": progress,
                "message": status_msg,
                "created_at": processing_status[task_id]["created_at"]
            }
        
        # 파이프라인 실행
        pipeline = await get_global_pipeline()
        
        result = await pipeline.process_virtual_fitting(
            person_image=person_image,
            clothing_image=clothing_image,
            body_measurements=body_measurements,
            clothing_type=clothing_type,
            fabric_type=fabric_type,
            progress_callback=progress_callback
        )
        
        if result["success"]:
            # 결과 이미지를 Base64로 인코딩
            buffer = BytesIO()
            result["result_image"].save(buffer, format="JPEG", quality=90)
            result_base64 = base64.b64encode(buffer.getvalue()).decode()
            
            # 상태 업데이트
            processing_status[task_id] = {
                "status": "completed",
                "progress": 100,
                "message": "가상 피팅 완료!",
                "result": {
                    **result,
                    "result_image_base64": result_base64,
                    "result_image": None  # PIL 객체는 JSON 직렬화 불가하므로 제거
                },
                "created_at": processing_status[task_id]["created_at"]
            }
        else:
            processing_status[task_id] = {
                "status": "failed",
                "error": result.get("error_message", "알 수 없는 오류"),
                "created_at": processing_status[task_id]["created_at"]
            }
    
    except Exception as e:
        logger.error(f"가상 피팅 태스크 실패 (ID: {task_id}): {e}")
        processing_status[task_id] = {
            "status": "failed",
            "error": f"처리 중 오류 발생: {str(e)}",
            "created_at": processing_status[task_id]["created_at"]
        }

def create_simple_composite(person_image: Image.Image, clothing_image: Image.Image) -> Image.Image:
    """간단한 이미지 합성 (데모용)"""
    
    result = person_image.copy()
    
    # 의류 이미지 리사이즈 (가슴 부분 크기)
    clothing_size = min(result.width // 3, result.height // 3)
    clothing_resized = clothing_image.resize((clothing_size, clothing_size), Image.Resampling.LANCZOS)
    
    # 가슴 부분에 배치
    paste_x = (result.width - clothing_size) // 2
    paste_y = result.height // 4
    
    # 간단한 알파 블렌딩
    mask = Image.new('L', clothing_resized.size, 180)  # 70% 불투명도
    result.paste(clothing_resized, (paste_x, paste_y), mask)
    
    return result

# ============================================================================
# 서버 실행
# ============================================================================

if __name__ == "__main__":
    # 환경 변수에서 설정 읽기
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    debug = os.getenv("DEBUG", "true").lower() == "true"
    
    logger.info(f"🚀 MyCloset AI 서버 시작")
    logger.info(f"   - 주소: http://{host}:{port}")
    logger.info(f"   - 디바이스: {DEVICE}")
    logger.info(f"   - 디버그 모드: {debug}")
    logger.info(f"   - API 문서: http://{host}:{port}/docs")
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=debug,
        log_level="info" if debug else "warning"
    )