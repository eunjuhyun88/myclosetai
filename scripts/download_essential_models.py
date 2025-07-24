#!/usr/bin/env python3
"""
🍎 MyCloset AI - 필수 모델 다운로드 스크립트 (conda 환경)
===============================================================
✅ conda 환경 최적화
✅ M3 Max 메모리 고려
✅ 안전한 다운로드
✅ 진행률 표시
✅ 에러 복구
"""

import os
import sys
import requests
import hashlib
from pathlib import Path
from typing import Dict, Any
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 프로젝트 경로
PROJECT_ROOT = Path(__file__).parent
AI_MODELS_DIR = PROJECT_ROOT / "backend" / "ai_models"

# 필수 모델 정보
ESSENTIAL_MODELS = {
    "step_01_human_parsing": {
        "file_name": "exp-schp-201908301523-atr.pth",
        "url": "https://drive.google.com/uc?id=1ruJg-hPABjf5_WW3WQ18E_1DdQWPpWGS",
        "size_mb": 255.1,
        "md5": None  # 필요 시 추가
    },
    "step_03_cloth_segmentation": {
        "file_name": "u2net.pth", 
        "url": "https://drive.google.com/uc?id=1ao1ovG1Qtx4b7EoskHXmi2E9rp5CHLcZ",
        "size_mb": 168.1,
        "md5": None
    },
    "step_06_virtual_fitting": {
        "file_name": "sam_vit_h_4b8939.pth",
        "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
        "size_mb": 2445.7,
        "md5": "4b8939a88964f0f4ff5f5b2642c598a6"
    }
}

def download_file(url: str, dest_path: Path, expected_size_mb: float = None) -> bool:
    """파일 다운로드 (진행률 표시)"""
    try:
        logger.info(f"다운로드 시작: {dest_path.name}")
        
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(dest_path, 'wb') as f:
            downloaded = 0
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"\r진행률: {percent:.1f}% ({downloaded // (1024*1024)}MB)", end='')
        
        print()  # 새 줄
        
        # 크기 검증
        actual_size_mb = dest_path.stat().st_size / (1024 * 1024)
        if expected_size_mb and abs(actual_size_mb - expected_size_mb) > expected_size_mb * 0.1:
            logger.warning(f"크기 불일치: 예상 {expected_size_mb}MB, 실제 {actual_size_mb:.1f}MB")
        
        logger.info(f"다운로드 완료: {dest_path.name} ({actual_size_mb:.1f}MB)")
        return True
        
    except Exception as e:
        logger.error(f"다운로드 실패 {dest_path.name}: {e}")
        if dest_path.exists():
            dest_path.unlink()
        return False

def main():
    """메인 다운로드 프로세스"""
    logger.info("🍎 MyCloset AI 필수 모델 다운로드 시작")
    
    # conda 환경 확인
    conda_env = os.environ.get('CONDA_DEFAULT_ENV')
    if conda_env:
        logger.info(f"✅ conda 환경: {conda_env}")
    else:
        logger.warning("⚠️ conda 환경이 활성화되지 않음")
    
    success_count = 0
    total_count = len(ESSENTIAL_MODELS)
    
    for step_name, model_info in ESSENTIAL_MODELS.items():
        step_dir = AI_MODELS_DIR / step_name
        model_path = step_dir / model_info["file_name"]
        
        # 이미 존재하는지 확인
        if model_path.exists():
            actual_size_mb = model_path.stat().st_size / (1024 * 1024)
            expected_size_mb = model_info["size_mb"]
            
            if abs(actual_size_mb - expected_size_mb) < expected_size_mb * 0.1:
                logger.info(f"✅ 이미 존재: {model_info['file_name']}")
                success_count += 1
                continue
            else:
                logger.info(f"🔄 크기 불일치로 재다운로드: {model_info['file_name']}")
                model_path.unlink()
        
        # 다운로드 시도
        if download_file(model_info["url"], model_path, model_info["size_mb"]):
            success_count += 1
    
    # 결과 요약
    logger.info(f"📊 다운로드 완료: {success_count}/{total_count}")
    
    if success_count == total_count:
        logger.info("🎉 모든 필수 모델 다운로드 성공!")
        
        # 빠른 검증
        logger.info("🔍 모델 로딩 테스트...")
        try:
            sys.path.insert(0, str(PROJECT_ROOT / "backend"))
            from app.ai_pipeline.utils.model_loader import ModelLoader
            
            loader = ModelLoader()
            loader.scan_available_models()
            logger.info("✅ 모델 로더 테스트 성공!")
            
        except Exception as e:
            logger.warning(f"⚠️ 모델 로더 테스트 실패: {e}")
    else:
        logger.error(f"❌ {total_count - success_count}개 모델 다운로드 실패")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
