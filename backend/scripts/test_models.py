#!/usr/bin/env python3
"""
AI 모델 테스트 스크립트
"""
import asyncio
import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(str(Path(__file__).parent.parent))

from app.services.ai_models import model_manager
from app.services.real_working_ai_fitter import RealWorkingAIFitter
from PIL import Image

async def test_models():
    print("🧪 AI 모델 테스트 시작...")
    
    try:
        # 1. 모델 관리자 초기화
        await model_manager.initialize_models()
        print("✅ 모델 관리자 초기화 완료")
        
        # 2. 사용 가능한 모델 확인
        available_models = model_manager.get_available_models()
        print(f"📋 사용 가능한 모델: {available_models}")
        
        # 3. 더미 이미지로 테스트
        dummy_person = Image.new('RGB', (512, 512), color='white')
        dummy_clothing = Image.new('RGB', (512, 512), color='blue')
        
        if available_models:
            print("🎨 가상 피팅 테스트 중...")
            result_image, metadata = await model_manager.generate_virtual_fitting(
                dummy_person, dummy_clothing
            )
            print(f"✅ 가상 피팅 테스트 성공: {metadata}")
        
        # 4. AI 서비스 테스트
        ai_fitter = RealWorkingAIFitter()
        status = await ai_fitter.get_model_status()
        print(f"📊 AI 서비스 상태: {status}")
        
        print("🎉 모든 테스트 통과!")
        
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        return False
    
    return True

if __name__ == "__main__":
    result = asyncio.run(test_models())
    sys.exit(0 if result else 1)
