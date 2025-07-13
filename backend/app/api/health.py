# backend/app/api/health.py
"""
MyCloset AI Backend - 헬스체크 API
시스템 상태 모니터링 엔드포인트
"""

import time
import psutil
import torch
from datetime import datetime
from typing import Dict, Any
from fastapi import APIRouter, HTTPException
from pathlib import Path

from app.core.gpu_config import gpu_config, DEVICE_INFO
from app.core.config import settings

router = APIRouter()

# 서버 시작 시간
_startup_time = time.time()

@router.get("/")
async def health_check() -> Dict[str, Any]:
    """기본 헬스체크"""
    
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "uptime_seconds": round(time.time() - _startup_time, 2)
    }

@router.get("/detailed")
async def detailed_health_check() -> Dict[str, Any]:
    """상세 헬스체크"""
    
    # 시스템 메모리 정보
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage(str(settings.PROJECT_ROOT))
    
    # GPU 상태 확인
    gpu_status = {
        "device": gpu_config.device,
        "available": gpu_config.device != "cpu"
    }
    
    if gpu_config.device == "mps":
        gpu_status.update({
            "mps_available": torch.backends.mps.is_available(),
            "backend": "Metal Performance Shaders"
        })
    elif gpu_config.device == "cuda":
        gpu_status.update({
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0
        })
    
    # 디렉토리 상태 확인
    directories = {
        "upload_dir": settings.UPLOAD_DIR.exists(),
        "results_dir": settings.RESULTS_DIR.exists(),
        "models_dir": settings.AI_MODELS_DIR.exists(),
        "logs_dir": settings.LOGS_DIR.exists()
    }
    
    # AI 모델 상태
    model_status = {"status": "checking"}
    try:
        from app.services.model_manager import model_manager
        model_status = {
            "loaded": len(model_manager.loaded_models),
            "available": len(model_manager.available_models),
            "models": list(model_manager.loaded_models.keys())
        }
    except Exception as e:
        model_status = {"error": f"Model manager not ready: {e}"}
    
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": {
            "name": settings.APP_NAME,
            "version": settings.APP_VERSION,
            "debug": settings.DEBUG,
            "uptime_seconds": round(time.time() - _startup_time, 2)
        },
        "system": {
            "platform": DEVICE_INFO["platform"],
            "machine": DEVICE_INFO["machine"],
            "python_version": DEVICE_INFO["python_version"],
            "pytorch_version": DEVICE_INFO["pytorch_version"]
        },
        "memory": {
            "total_gb": round(memory.total / (1024**3), 1),
            "available_gb": round(memory.available / (1024**3), 1),
            "used_percent": memory.percent,
            "free_percent": round(100 - memory.percent, 1)
        },
        "disk": {
            "total_gb": round(disk.total / (1024**3), 1),
            "free_gb": round(disk.free / (1024**3), 1),
            "used_percent": round((disk.used / disk.total) * 100, 1)
        },
        "gpu": gpu_status,
        "directories": directories,
        "models": model_status,
        "config": {
            "max_upload_size_mb": settings.MAX_UPLOAD_SIZE // (1024*1024),
            "allowed_extensions": settings.ALLOWED_EXTENSIONS,
            "image_size": settings.IMAGE_SIZE,
            "batch_size": settings.BATCH_SIZE
        }
    }

@router.get("/gpu")
async def gpu_health() -> Dict[str, Any]:
    """GPU 상태 전용 체크"""
    
    # GPU 기본 정보
    gpu_info = {
        "device": gpu_config.device,
        "is_available": gpu_config.device != "cpu",
        "device_info": DEVICE_INFO,
        "timestamp": datetime.utcnow().isoformat()
    }
    
    # 디바이스별 상세 정보
    if gpu_config.device == "mps":
        try:
            gpu_info.update({
                "mps_available": torch.backends.mps.is_available(),
                "backend": "Metal Performance Shaders",
                "optimization": "Apple M3 Max",
                "memory_type": "Unified Memory"
            })
            
            # 간단한 연산 테스트
            device = torch.device("mps")
            test_tensor = torch.randn(100, 100, device=device)
            result = torch.mm(test_tensor, test_tensor)
            
            gpu_info["test_status"] = "passed"
            gpu_info["test_result_shape"] = list(result.shape)
            
        except Exception as e:
            gpu_info["test_status"] = "failed"
            gpu_info["test_error"] = str(e)
            
    elif gpu_config.device == "cuda":
        try:
            gpu_info.update({
                "cuda_available": torch.cuda.is_available(),
                "cuda_version": torch.version.cuda,
                "device_count": torch.cuda.device_count(),
            })
            
            if torch.cuda.is_available():
                # GPU 메모리 정보
                memory_allocated = torch.cuda.memory_allocated(0)
                memory_reserved = torch.cuda.memory_reserved(0)
                
                gpu_info.update({
                    "memory_allocated_mb": round(memory_allocated / (1024*1024), 1),
                    "memory_reserved_mb": round(memory_reserved / (1024*1024), 1),
                    "device_name": torch.cuda.get_device_name(0)
                })
                
                # 간단한 연산 테스트
                device = torch.device("cuda")
                test_tensor = torch.randn(100, 100, device=device)
                result = torch.mm(test_tensor, test_tensor)
                
                gpu_info["test_status"] = "passed"
                gpu_info["test_result_shape"] = list(result.shape)
            
        except Exception as e:
            gpu_info["test_status"] = "failed"
            gpu_info["test_error"] = str(e)
    
    else:
        gpu_info.update({
            "backend": "CPU",
            "note": "GPU acceleration not available"
        })
    
    return gpu_info

@router.get("/models")
async def models_health() -> Dict[str, Any]:
    """AI 모델 상태 체크"""
    
    try:
        from app.services.model_manager import model_manager
        
        # 모델 디렉토리 체크
        models_dir = settings.AI_MODELS_DIR / "checkpoints"
        available_models = []
        
        if models_dir.exists():
            for model_dir in models_dir.iterdir():
                if model_dir.is_dir():
                    file_count = len(list(model_dir.glob("**/*")))
                    available_models.append({
                        "name": model_dir.name,
                        "path": str(model_dir),
                        "files_count": file_count,
                        "size_mb": sum(f.stat().st_size for f in model_dir.glob("**/*") if f.is_file()) // (1024*1024)
                    })
        
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "models_directory": str(models_dir),
            "models_directory_exists": models_dir.exists(),
            "available_models": available_models,
            "loaded_models": list(model_manager.loaded_models.keys()) if hasattr(model_manager, 'loaded_models') else [],
            "total_available": len(available_models),
            "total_loaded": len(model_manager.loaded_models) if hasattr(model_manager, 'loaded_models') else 0
        }
        
    except ImportError:
        return {
            "status": "error",
            "timestamp": datetime.utcnow().isoformat(),
            "error": "Model manager not available",
            "models_directory": str(settings.AI_MODELS_DIR / "checkpoints"),
            "suggestion": "Run 'python scripts/download_ai_models.py' to download models"
        }
    except Exception as e:
        return {
            "status": "error",
            "timestamp": datetime.utcnow().isoformat(),
            "error": str(e)
        }

@router.get("/storage")
async def storage_health() -> Dict[str, Any]:
    """저장소 상태 체크"""
    
    def get_directory_info(path: Path) -> Dict[str, Any]:
        """디렉토리 정보 조회"""
        if not path.exists():
            return {"exists": False, "size_mb": 0, "files_count": 0}
        
        total_size = 0
        files_count = 0
        
        try:
            for file_path in path.glob("**/*"):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
                    files_count += 1
        except PermissionError:
            pass
        
        return {
            "exists": True,
            "size_mb": round(total_size / (1024*1024), 1),
            "files_count": files_count,
            "path": str(path)
        }
    
    # 주요 디렉토리들 체크
    directories = {
        "uploads": get_directory_info(settings.UPLOAD_DIR),
        "results": get_directory_info(settings.RESULTS_DIR),
        "models": get_directory_info(settings.AI_MODELS_DIR),
        "logs": get_directory_info(settings.LOGS_DIR),
        "temp": get_directory_info(settings.TEMP_DIR)
    }
    
    # 전체 프로젝트 크기
    total_project_size = sum(dir_info["size_mb"] for dir_info in directories.values())
    
    # 디스크 사용량
    disk_usage = psutil.disk_usage(str(settings.PROJECT_ROOT))
    
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "project_root": str(settings.PROJECT_ROOT),
        "directories": directories,
        "total_project_size_mb": round(total_project_size, 1),
        "disk_usage": {
            "total_gb": round(disk_usage.total / (1024**3), 1),
            "free_gb": round(disk_usage.free / (1024**3), 1),
            "used_gb": round(disk_usage.used / (1024**3), 1),
            "free_percent": round((disk_usage.free / disk_usage.total) * 100, 1)
        }
    }

@router.get("/benchmark")
async def run_benchmark() -> Dict[str, Any]:
    """GPU 벤치마크 실행"""
    
    try:
        # GPU 벤치마크 실행
        benchmark_result = gpu_config.benchmark_device(iterations=50)
        
        # AI 파이프라인 테스트
        pipeline_test = gpu_config.test_ai_pipeline()
        
        return {
            "status": "completed",
            "timestamp": datetime.utcnow().isoformat(),
            "gpu_benchmark": benchmark_result,
            "ai_pipeline_test": {
                "status": "passed" if pipeline_test else "failed",
                "result": pipeline_test
            },
            "device_info": DEVICE_INFO
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Benchmark failed: {str(e)}"
        )

@router.post("/cleanup")
async def cleanup_temp_files() -> Dict[str, Any]:
    """임시 파일 정리"""
    
    try:
        import shutil
        
        cleaned_dirs = []
        total_freed_mb = 0
        
        # 임시 디렉토리 정리
        temp_dirs = [
            settings.TEMP_DIR,
            settings.UPLOAD_DIR / "temp",
            settings.RESULTS_DIR / "temp"
        ]
        
        for temp_dir in temp_dirs:
            if temp_dir.exists():
                # 크기 계산
                dir_size = sum(f.stat().st_size for f in temp_dir.glob("**/*") if f.is_file())
                
                # 정리
                shutil.rmtree(temp_dir)
                temp_dir.mkdir(parents=True, exist_ok=True)
                
                cleaned_dirs.append({
                    "directory": str(temp_dir),
                    "freed_mb": round(dir_size / (1024*1024), 1)
                })
                total_freed_mb += dir_size / (1024*1024)
        
        # GPU 메모리 정리
        gpu_config.optimize_memory()
        
        return {
            "status": "completed",
            "timestamp": datetime.utcnow().isoformat(),
            "cleaned_directories": cleaned_dirs,
            "total_freed_mb": round(total_freed_mb, 1),
            "gpu_memory_cleared": True
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Cleanup failed: {str(e)}"
        )