#!/usr/bin/env python3
"""
Step 7 가상 피팅 디버깅 스크립트
"""

import os
import sys
import logging
from pathlib import Path

# 백엔드 경로 추가
backend_root = Path(__file__).parent
if str(backend_root) not in sys.path:
    sys.path.insert(0, str(backend_root))

# 로깅 설정
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def check_ai_models():
    """AI 모델 파일 확인"""
    print("🔍 AI 모델 파일 확인 중...")
    
    ai_models_root = backend_root / "ai_models"
    step_06_path = ai_models_root / "step_06_virtual_fitting"
    
    print(f"📁 AI 모델 루트: {ai_models_root}")
    print(f"📁 Step 06 경로: {step_06_path}")
    
    if step_06_path.exists():
        print("✅ Step 06 디렉토리 존재")
        
        # OOTD Diffusion 체크
        ootd_path = step_06_path / "ootdiffusion"
        if ootd_path.exists():
            print("✅ OOTDiffusion 디렉토리 존재")
            
            # 주요 모델 파일들 체크
            model_files = [
                "checkpoints/ootd/ootd_hd/checkpoint-36000/unet_vton/diffusion_pytorch_model.safetensors",
                "diffusion_pytorch_model.bin",
                "pytorch_model.bin"
            ]
            
            for file_path in model_files:
                full_path = ootd_path / file_path
                if full_path.exists():
                    size_mb = full_path.stat().st_size / (1024*1024)
                    print(f"✅ {file_path}: {size_mb:.1f}MB")
                else:
                    print(f"❌ {file_path}: 없음")
        else:
            print("❌ OOTDiffusion 디렉토리 없음")
    else:
        print("❌ Step 06 디렉토리 없음")

def test_virtual_fitting_step():
    """VirtualFittingStep 테스트"""
    print("\n🧪 VirtualFittingStep 테스트 중...")
    
    try:
        from app.ai_pipeline.steps.step_06_virtual_fitting import VirtualFittingStep
        print("✅ VirtualFittingStep import 성공")
        
        # 인스턴스 생성
        step = VirtualFittingStep()
        print("✅ VirtualFittingStep 인스턴스 생성 성공")
        
        # 초기화
        if hasattr(step, 'initialize'):
            result = step.initialize()
            print(f"✅ initialize() 결과: {result}")
        
        # 상태 확인
        if hasattr(step, 'get_status'):
            status = step.get_status()
            print(f"📊 Step 상태: {status}")
        
        # AI 모델 로딩 상태 확인
        if hasattr(step, 'virtual_fitting_ai'):
            ai_status = step.virtual_fitting_ai.get_model_status() if hasattr(step.virtual_fitting_ai, 'get_model_status') else "Unknown"
            print(f"🤖 AI 모델 상태: {ai_status}")
        
    except Exception as e:
        print(f"❌ VirtualFittingStep 테스트 실패: {e}")
        import traceback
        traceback.print_exc()

def check_step_service():
    """StepService 상태 확인"""
    print("\n🔧 StepService 상태 확인 중...")
    
    try:
        from app.services.step_service import StepServiceManager
        print("✅ StepServiceManager import 성공")
        
        # 인스턴스 생성
        service = StepServiceManager()
        print("✅ StepServiceManager 인스턴스 생성 성공")
        
        # Step 7 프로세싱 테스트 (더미 데이터)
        print("🧪 Step 7 더미 테스트...")
        
    except Exception as e:
        print(f"❌ StepService 테스트 실패: {e}")
        import traceback
        traceback.print_exc()

def main():
    """메인 디버깅 함수"""
    print("🔥 Step 7 가상 피팅 디버깅 시작")
    print("=" * 60)
    
    # 1. AI 모델 파일 확인
    check_ai_models()
    
    # 2. VirtualFittingStep 테스트
    test_virtual_fitting_step()
    
    # 3. StepService 상태 확인
    check_step_service()
    
    print("\n" + "=" * 60)
    print("🎯 디버깅 완료")
    
    print("\n💡 해결 방법:")
    print("1. AI 모델 파일이 없다면: python download_ai_models.py")
    print("2. 모델 로딩 실패라면: 메모리 부족 또는 권한 문제")
    print("3. Step 초기화 실패라면: 의존성 문제")

if __name__ == "__main__":
    main()