#!/usr/bin/env python3
"""
FLUX.1-Kontext-dev 모델 설치 스크립트
"""

import os
import sys
import subprocess
import logging
from pathlib import Path

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_command(command, description):
    """명령어 실행 및 에러 처리"""
    try:
        logger.info(f"{description} 실행 중...")
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        logger.info(f"{description} 완료!")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"{description} 실패: {e.stderr}")
        return False

def check_python_version():
    """Python 버전 확인"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        logger.error("Python 3.8 이상이 필요합니다.")
        return False
    logger.info(f"Python 버전 확인: {version.major}.{version.minor}.{version.micro}")
    return True

def install_requirements():
    """필요한 패키지 설치"""
    requirements = [
        "torch",
        "torchvision", 
        "diffusers",
        "transformers",
        "accelerate",
        "huggingface_hub",
        "safetensors",
        "Pillow",
        "numpy"
    ]
    
    for package in requirements:
        if not run_command(f"pip install {package}", f"{package} 설치"):
            return False
    return True

def install_diffusers_latest():
    """diffusers 최신 버전 설치 (FLUX 지원을 위해)"""
    return run_command(
        "pip install git+https://github.com/huggingface/diffusers.git",
        "diffusers 최신 버전 설치"
    )

def create_model_directory():
    """모델 저장 디렉토리 생성"""
    model_dir = Path("./models/flux_kontext")
    model_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"모델 디렉토리 생성: {model_dir.absolute()}")
    return model_dir

def download_model():
    """모델 다운로드"""
    try:
        from huggingface_hub import snapshot_download
        
        model_name = "black-forest-labs/FLUX.1-Kontext-dev"
        cache_dir = "./models/flux_kontext"
        
        logger.info(f"모델 다운로드 시작: {model_name}")
        
        model_path = snapshot_download(
            repo_id=model_name,
            cache_dir=cache_dir,
            local_dir=os.path.join(cache_dir, "local"),
            local_dir_use_symlinks=False
        )
        
        logger.info(f"모델 다운로드 완료: {model_path}")
        return model_path
        
    except Exception as e:
        logger.error(f"모델 다운로드 실패: {str(e)}")
        return None

def test_installation():
    """설치 테스트"""
    try:
        from diffusers import FluxKontextPipeline
        import torch
        
        logger.info("설치 테스트 시작...")
        
        # 간단한 파이프라인 로드 테스트
        pipe = FluxKontextPipeline.from_pretrained(
            "./models/flux_kontext/local",
            torch_dtype=torch.float16,
            use_safetensors=True,
            low_cpu_mem_usage=True
        )
        
        logger.info("파이프라인 로드 성공!")
        return True
        
    except Exception as e:
        logger.error(f"설치 테스트 실패: {str(e)}")
        return False

def main():
    """메인 설치 프로세스"""
    logger.info("=== FLUX.1-Kontext-dev 모델 설치 시작 ===")
    
    # 1. Python 버전 확인
    if not check_python_version():
        sys.exit(1)
    
    # 2. 기본 패키지 설치
    if not install_requirements():
        logger.error("기본 패키지 설치 실패")
        sys.exit(1)
    
    # 3. diffusers 최신 버전 설치
    if not install_diffusers_latest():
        logger.error("diffusers 최신 버전 설치 실패")
        sys.exit(1)
    
    # 4. 모델 디렉토리 생성
    create_model_directory()
    
    # 5. 모델 다운로드
    model_path = download_model()
    if not model_path:
        logger.error("모델 다운로드 실패")
        sys.exit(1)
    
    # 6. 설치 테스트
    if not test_installation():
        logger.error("설치 테스트 실패")
        sys.exit(1)
    
    logger.info("=== 설치 완료! ===")
    logger.info(f"모델 경로: {model_path}")
    logger.info("사용 예제는 examples 폴더를 참조하세요.")

if __name__ == "__main__":
    main()
