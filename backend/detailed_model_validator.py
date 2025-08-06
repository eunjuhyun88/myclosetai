#!/usr/bin/env python3
"""
상세 AI 모델 검증 및 문제 해결 스크립트
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

import torch
from pathlib import Path
import logging
import json
from datetime import datetime

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def analyze_ootd_model():
    """OOTD 모델 크기 불일치 문제 분석"""
    print("\n🎭 OOTD 모델 분석:")
    print("=" * 50)
    
    ootd_paths = [
        "ai_models/OOTD/diffusion_pytorch_model.bin",
        "ai_models/OOTD/diffusion_pytorch_model.safetensors",
        "ai_models/OOTD/pytorch_model.bin"
    ]
    
    for path in ootd_paths:
        full_path = Path(path)
        if full_path.exists():
            size_mb = full_path.stat().st_size / (1024 * 1024)
            print(f"📁 {path} ({size_mb:.1f}MB)")
            
            try:
                if path.endswith('.safetensors'):
                    from safetensors import safe_open
                    with safe_open(full_path, framework="pt", device="cpu") as f:
                        keys = f.keys()
                        print(f"   🔑 키 수: {len(keys)}")
                        print(f"   📊 샘플 키: {list(keys)[:3]}")
                else:
                    checkpoint = torch.load(full_path, map_location='cpu', weights_only=True)
                    if isinstance(checkpoint, dict):
                        print(f"   🔑 키 수: {len(checkpoint)}")
                        print(f"   📊 샘플 키: {list(checkpoint.keys())[:3]}")
                        
                        # 크기 정보 분석
                        for key, tensor in list(checkpoint.items())[:5]:
                            if hasattr(tensor, 'shape'):
                                print(f"   📏 {key}: {tensor.shape}")
            except Exception as e:
                print(f"   ❌ 로딩 실패: {str(e)[:100]}")
        else:
            print(f"❌ {path} (없음)")

def analyze_gmm_model():
    """GMM 모델 타입 불일치 문제 분석"""
    print("\n🎯 GMM 모델 분석:")
    print("=" * 50)
    
    gmm_path = "ai_models/GMM/gmm_final.pth"
    if Path(gmm_path).exists():
        size_mb = Path(gmm_path).stat().st_size / (1024 * 1024)
        print(f"📁 {gmm_path} ({size_mb:.1f}MB)")
        
        try:
            checkpoint = torch.load(gmm_path, map_location='cpu', weights_only=True)
            if isinstance(checkpoint, dict):
                print(f"🔑 키 수: {len(checkpoint)}")
                
                # 텐서 타입 분석
                for key, tensor in list(checkpoint.items())[:5]:
                    if hasattr(tensor, 'dtype'):
                        print(f"📊 {key}: {tensor.dtype} - {tensor.shape}")
                        
                        # MPS 호환성 확인
                        if tensor.dtype == torch.float32:
                            try:
                                mps_tensor = tensor.to('mps')
                                print(f"   ✅ MPS 변환 가능")
                            except Exception as e:
                                print(f"   ❌ MPS 변환 실패: {e}")
        except Exception as e:
            print(f"❌ 로딩 실패: {str(e)[:100]}")
    else:
        print(f"❌ {gmm_path} (없음)")

def check_model_compatibility():
    """모델 호환성 검사"""
    print("\n🔧 모델 호환성 검사:")
    print("=" * 50)
    
    # PyTorch 버전 확인
    print(f"🐍 Python: {sys.version}")
    print(f"🔥 PyTorch: {torch.__version__}")
    print(f"📱 CUDA 사용 가능: {torch.cuda.is_available()}")
    print(f"🍎 MPS 사용 가능: {torch.backends.mps.is_available()}")
    
    if torch.backends.mps.is_available():
        print(f"🍎 MPS 디바이스: {torch.device('mps')}")
        
        # MPS 텐서 테스트
        try:
            test_tensor = torch.randn(2, 3).to('mps')
            print(f"✅ MPS 텐서 생성 성공: {test_tensor.dtype}")
        except Exception as e:
            print(f"❌ MPS 텐서 생성 실패: {e}")

def suggest_fixes():
    """문제 해결 방안 제시"""
    print("\n💡 문제 해결 방안:")
    print("=" * 50)
    
    print("1. 🎭 OOTD 모델 문제:")
    print("   - OOTD 체크포인트와 모델 아키텍처 불일치")
    print("   - 해결방안:")
    print("     a) 올바른 OOTD 체크포인트 다운로드")
    print("     b) 모델 아키텍처 수정")
    print("     c) 다른 가상 피팅 모델 사용 (HR-VITON, VITON HD)")
    
    print("\n2. 🎯 GMM 모델 문제:")
    print("   - MPS 백엔드와 PyTorch 텐서 타입 불일치")
    print("   - 해결방안:")
    print("     a) 모델을 CPU에서 로드 후 MPS로 변환")
    print("     b) 텐서 타입 강제 변환")
    print("     c) MPS 호환 모델 사용")
    
    print("\n3. 🌐 DPT 모델 문제:")
    print("   - Hugging Face 연결 실패")
    print("   - 해결방안:")
    print("     a) 네트워크 연결 확인")
    print("     b) 프록시 설정")
    print("     c) 로컬 캐시된 모델 사용")

def generate_model_report():
    """모델 상태 리포트 생성"""
    print("\n📋 모델 상태 리포트 생성:")
    print("=" * 50)
    
    report = {
        "timestamp": str(datetime.now()),
        "pytorch_version": torch.__version__,
        "mps_available": torch.backends.mps.is_available(),
        "cuda_available": torch.cuda.is_available(),
        "model_status": {}
    }
    
    # 각 스텝별 모델 상태 확인
    step_modules = [
        "step_01_human_parsing",
        "step_02_pose_estimation", 
        "step_03_cloth_segmentation",
        "step_04_geometric_matching",
        "step_05_cloth_warping",
        "step_06_virtual_fitting",
        "step_07_post_processing",
        "step_08_quality_assessment"
    ]
    
    for module in step_modules:
        try:
            module_path = f"app.ai_pipeline.steps.{module}"
            step_class = getattr(__import__(module_path, fromlist=[module.split('.')[-1]]), 
                               module.split('.')[-1].replace('step_', '').title().replace('_', '') + 'Step')
            
            step = step_class()
            if hasattr(step, '_load_ai_models_via_central_hub'):
                result = step._load_ai_models_via_central_hub()
                report["model_status"][module] = "success" if result else "failed"
            else:
                report["model_status"][module] = "no_load_method"
        except Exception as e:
            report["model_status"][module] = f"error: {str(e)[:100]}"
    
    # 리포트 저장
    report_path = "model_status_report.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"✅ 리포트 저장됨: {report_path}")

if __name__ == "__main__":
    print("🔍 MyCloset AI 상세 모델 검증")
    print("=" * 60)
    
    check_model_compatibility()
    analyze_ootd_model()
    analyze_gmm_model()
    suggest_fixes()
    generate_model_report()
    
    print("\n✅ 상세 모델 검증 완료!") 