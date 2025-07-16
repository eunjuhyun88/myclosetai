#!/usr/bin/env python3
"""
PyTorch 테스트 서버
설치 확인 및 간단한 AI 기능 테스트
"""

import sys
import os
from pathlib import Path

# 프로젝트 루트 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    import torch
    import torchvision.transforms as transforms
    from PIL import Image
    import numpy as np
    TORCH_AVAILABLE = True
    
    # 디바이스 선택
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        DEVICE = "mps"
        DEVICE_INFO = "Apple Silicon (Metal)"
    elif torch.cuda.is_available():
        DEVICE = "cuda"
        DEVICE_INFO = f"NVIDIA GPU ({torch.cuda.get_device_name()})"
    else:
        DEVICE = "cpu"
        DEVICE_INFO = "CPU"
        
    print(f"✅ PyTorch 로드 성공: {torch.__version__}")
    print(f"🎯 사용 디바이스: {DEVICE} ({DEVICE_INFO})")
    
except ImportError as e:
    print(f"❌ PyTorch 로드 실패: {e}")
    TORCH_AVAILABLE = False
    DEVICE = "none"
    DEVICE_INFO = "PyTorch 없음"

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from datetime import datetime
import time

app = FastAPI(
    title="PyTorch 테스트 서버",
    description="PyTorch 설치 확인 및 AI 기능 테스트",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {
        "message": "PyTorch 테스트 서버 🔥",
        "torch_available": TORCH_AVAILABLE,
        "torch_version": torch.__version__ if TORCH_AVAILABLE else "없음",
        "device": DEVICE,
        "device_info": DEVICE_INFO,
        "status": "ready" if TORCH_AVAILABLE else "pytorch_missing",
        "test_endpoints": {
            "tensor_test": "/test/tensor",
            "performance": "/test/performance",
            "memory": "/test/memory"
        }
    }

@app.get("/test/tensor")
async def test_tensor_operations():
    """기본 텐서 연산 테스트"""
    if not TORCH_AVAILABLE:
        raise HTTPException(503, "PyTorch가 설치되지 않았습니다")
    
    try:
        start_time = time.time()
        
        # 기본 텐서 생성
        x = torch.randn(100, 100, device=DEVICE)
        y = torch.randn(100, 100, device=DEVICE)
        
        # 행렬 곱셈
        z = torch.mm(x, y)
        
        # 통계 계산
        mean_val = torch.mean(z).item()
        std_val = torch.std(z).item()
        
        processing_time = time.time() - start_time
        
        return {
            "success": True,
            "device": DEVICE,
            "tensor_shape": list(z.shape),
            "statistics": {
                "mean": round(mean_val, 4),
                "std": round(std_val, 4)
            },
            "processing_time_ms": round(processing_time * 1000, 2),
            "message": "텐서 연산이 정상적으로 작동합니다!"
        }
        
    except Exception as e:
        raise HTTPException(500, f"텐서 연산 실패: {str(e)}")

@app.get("/test/performance")
async def test_performance():
    """성능 벤치마크 테스트"""
    if not TORCH_AVAILABLE:
        raise HTTPException(503, "PyTorch가 설치되지 않았습니다")
    
    try:
        results = {}
        
        # CPU vs 현재 디바이스 성능 비교
        for device in ["cpu", DEVICE] if DEVICE != "cpu" else ["cpu"]:
            start_time = time.time()
            
            # 1000x1000 행렬 곱셈 10회
            for _ in range(10):
                a = torch.randn(1000, 1000, device=device)
                b = torch.randn(1000, 1000, device=device)
                c = torch.mm(a, b)
            
            total_time = time.time() - start_time
            results[device] = {
                "total_time_seconds": round(total_time, 3),
                "avg_time_per_operation_ms": round(total_time * 100, 2),  # 10회 반복이므로 *100
                "operations_per_second": round(10 / total_time, 1)
            }
        
        return {
            "success": True,
            "torch_version": torch.__version__,
            "benchmark_results": results,
            "recommendation": f"{DEVICE} 사용을 권장합니다" if DEVICE != "cpu" else "GPU가 있다면 더 빠른 처리가 가능합니다"
        }
        
    except Exception as e:
        raise HTTPException(500, f"성능 테스트 실패: {str(e)}")

@app.get("/test/memory")
async def test_memory():
    """메모리 사용량 테스트"""
    if not TORCH_AVAILABLE:
        raise HTTPException(503, "PyTorch가 설치되지 않았습니다")
    
    try:
        results = {
            "device": DEVICE,
            "torch_version": torch.__version__
        }
        
        if DEVICE == "cuda":
            # CUDA 메모리 정보
            results["cuda_memory"] = {
                "total_mb": torch.cuda.get_device_properties(0).total_memory // 1024**2,
                "allocated_mb": torch.cuda.memory_allocated() // 1024**2,
                "cached_mb": torch.cuda.memory_reserved() // 1024**2
            }
        elif DEVICE == "mps":
            # MPS는 통합 메모리 사용
            results["mps_info"] = {
                "unified_memory": True,
                "note": "Apple Silicon은 통합 메모리를 사용합니다"
            }
        
        # 간단한 메모리 할당 테스트
        test_tensor = torch.randn(1000, 1000, device=DEVICE)
        tensor_size_mb = test_tensor.element_size() * test_tensor.nelement() / 1024**2
        
        results["test_allocation"] = {
            "tensor_shape": list(test_tensor.shape),
            "tensor_size_mb": round(tensor_size_mb, 2),
            "allocation_successful": True
        }
        
        # 메모리 정리
        del test_tensor
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        elif DEVICE == "mps":
            torch.mps.empty_cache()
            
        return {
            "success": True,
            "memory_info": results
        }
        
    except Exception as e:
        raise HTTPException(500, f"메모리 테스트 실패: {str(e)}")

if __name__ == "__main__":
    print("🔥 PyTorch 테스트 서버 시작...")
    print(f"✅ PyTorch: {'사용 가능' if TORCH_AVAILABLE else '사용 불가'}")
    print(f"🎯 디바이스: {DEVICE}")
    print("")
    print("📱 접속 주소: http://localhost:8001")
    print("📚 API 문서: http://localhost:8001/docs")
    print("🧪 텐서 테스트: http://localhost:8001/test/tensor")
    print("⚡ 성능 테스트: http://localhost:8001/test/performance")
    print("💾 메모리 테스트: http://localhost:8001/test/memory")
    print("")
    
    uvicorn.run(app, host="0.0.0.0", port=8001, reload=False)
