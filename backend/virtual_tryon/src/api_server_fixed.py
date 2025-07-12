from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
import cv2
import numpy as np
import os
from datetime import datetime
import sys

# 현재 디렉토리를 Python 경로에 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from basic_tryon import BasicVirtualTryOn

app = FastAPI(title="Virtual Try-On API")

# 모델 인스턴스
model = BasicVirtualTryOn()

@app.get("/")
def root():
    return {
        "message": "Virtual Try-On API is running!",
        "python_version": sys.version,
        "endpoints": {
            "docs": "/docs",
            "tryon": "/tryon"
        }
    }

@app.post("/tryon")
async def virtual_tryon(
    person: UploadFile = File(...),
    clothing: UploadFile = File(...)
):
    # 임시 파일 저장
    os.makedirs("temp", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    person_path = f"temp/person_{timestamp}.jpg"
    cloth_path = f"temp/cloth_{timestamp}.jpg"
    
    # 파일 저장
    with open(person_path, "wb") as f:
        f.write(await person.read())
    with open(cloth_path, "wb") as f:
        f.write(await clothing.read())
    
    # 처리
    result = model.process(person_path, cloth_path)
    
    if result is not None:
        result_path = f"results/result_{timestamp}.jpg"
        cv2.imwrite(result_path, result)
        
        # 임시 파일 정리
        os.remove(person_path)
        os.remove(cloth_path)
        
        return FileResponse(result_path)
    else:
        return {"error": "처리 실패"}

if __name__ == "__main__":
    import uvicorn
    print("🚀 서버 시작: http://localhost:8000")
    print("📚 API 문서: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)
