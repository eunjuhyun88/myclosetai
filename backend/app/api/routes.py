from fastapi import APIRouter

router = APIRouter()

@router.get("/test")
async def test_endpoint():
    return {"message": "API 테스트 성공!", "status": "working"}

@router.get("/models/status")
async def models_status():
    return {
        "models": {"ootd": False, "viton": False},
        "status": "development"
    }
