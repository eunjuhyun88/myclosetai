#!/usr/bin/env python3
"""
🔧 누락된 라이브러리 설치 및 환경 수정 스크립트
"""

import subprocess
import sys
import os
from pathlib import Path

def log_info(msg: str):
    print(f"ℹ️  {msg}")

def log_success(msg: str):
    print(f"✅ {msg}")

def log_error(msg: str):
    print(f"❌ {msg}")

def install_package(package_name: str):
    """패키지 설치"""
    try:
        log_info(f"{package_name} 설치 중...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        log_success(f"{package_name} 설치 완료")
        return True
    except subprocess.CalledProcessError:
        log_error(f"{package_name} 설치 실패")
        return False

def main():
    """메인 실행"""
    print("🔧 MyCloset AI 라이브러리 의존성 수정")
    print("=" * 50)
    
    # 필수 패키지들
    required_packages = [
        "PyYAML",           # yaml 모듈
        "opencv-python",    # cv2 모듈
        "torch",            # PyTorch 최신 버전
        "torchvision",      # 컴퓨터 비전 
        "torchaudio",       # 오디오 처리
        "transformers",     # HuggingFace 모델
        "diffusers",        # Stable Diffusion
        "onnxruntime",      # ONNX 런타임
        "mediapipe",        # MediaPipe
        "pillow",           # 이미지 처리
        "numpy",            # 수치 계산
        "psutil",           # 시스템 정보
    ]
    
    log_info("필수 패키지 설치 시작...")
    
    failed_packages = []
    for package in required_packages:
        if not install_package(package):
            failed_packages.append(package)
    
    if failed_packages:
        log_error(f"설치 실패한 패키지: {failed_packages}")
        print("\n수동 설치 명령어:")
        for package in failed_packages:
            print(f"  pip install {package}")
    else:
        log_success("모든 패키지 설치 완료!")
    
    # PyTorch MPS 지원 확인
    print("\n🔍 PyTorch MPS 지원 확인")
    print("=" * 30)
    
    try:
        import torch
        print(f"PyTorch 버전: {torch.__version__}")
        
        if hasattr(torch.backends, 'mps'):
            if torch.backends.mps.is_available():
                log_success("MPS 지원 가능 - M3 Max 최적화 활성화")
            else:
                log_info("MPS 지원 불가능 - CPU 모드로 실행")
        else:
            log_info("구버전 PyTorch - MPS 지원 없음")
            log_info("PyTorch 업그레이드 권장:")
            print("  pip install --upgrade torch torchvision torchaudio")
            
    except ImportError:
        log_error("PyTorch 설치 확인 실패")
    
    # OpenCV 설치 확인
    print("\n🔍 OpenCV 설치 확인")
    print("=" * 20)
    
    try:
        import cv2
        print(f"OpenCV 버전: {cv2.__version__}")
        
        # cvtColor 함수 확인
        if hasattr(cv2, 'cvtColor'):
            log_success("OpenCV 정상 동작")
        else:
            log_error("OpenCV cvtColor 함수 없음")
            
    except ImportError:
        log_error("OpenCV 설치 실패")
        log_info("다시 설치: pip install opencv-python")
    
    print("\n🚀 설치 완료 후 다음 명령어로 테스트:")
    print("  python3 advanced_model_test.py")

if __name__ == "__main__":
    main()