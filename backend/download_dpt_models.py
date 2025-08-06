#!/usr/bin/env python3
"""
🔥 MyCloset AI - DPT 모델 다운로드 스크립트
============================================

Intel DPT 모델들을 로컬에 다운로드하여 HuggingFace 인증 문제를 해결합니다.

Author: MyCloset AI Team
Date: 2025-08-06
"""

import os
import sys
import requests
import logging
from pathlib import Path
from typing import Dict, List
import hashlib
import shutil
from tqdm import tqdm

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 프로젝트 경로 설정
PROJECT_ROOT = Path(__file__).parent
AI_MODELS_ROOT = PROJECT_ROOT / "ai_models"

# DPT 모델 다운로드 정보
DPT_MODELS = {
    "dpt_hybrid_midas": {
        "name": "DPT Hybrid Midas",
        "url": "https://huggingface.co/Intel/dpt-hybrid-midas/resolve/main/pytorch_model.bin",
        "target_path": AI_MODELS_ROOT / "checkpoints" / "pose_estimation" / "dpt_hybrid-midas-501f0c75.pt",
        "size_mb": 469.9,
        "config_url": "https://huggingface.co/Intel/dpt-hybrid-midas/resolve/main/config.json"
    },
    "dpt_large": {
        "name": "DPT Large",
        "url": "https://huggingface.co/Intel/dpt-large/resolve/main/pytorch_model.bin",
        "target_path": AI_MODELS_ROOT / "checkpoints" / "pose_estimation" / "dpt_large-501f0c75.pt",
        "size_mb": 1024.0,
        "config_url": "https://huggingface.co/Intel/dpt-large/resolve/main/config.json"
    }
}

def download_file(url: str, target_path: Path, expected_size_mb: float = None) -> bool:
    """파일 다운로드 함수"""
    try:
        # 디렉토리 생성
        target_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 이미 존재하는지 확인
        if target_path.exists():
            logger.info(f"✅ 이미 존재함: {target_path}")
            return True
        
        logger.info(f"📥 다운로드 시작: {url}")
        logger.info(f"📁 저장 경로: {target_path}")
        
        # 다운로드
        response = requests.get(url, stream=True, timeout=300)
        response.raise_for_status()
        
        # 파일 크기 확인
        total_size = int(response.headers.get('content-length', 0))
        if expected_size_mb and total_size > 0:
            expected_size_bytes = expected_size_mb * 1024 * 1024
            if abs(total_size - expected_size_bytes) > expected_size_bytes * 0.1:  # 10% 허용 오차
                logger.warning(f"⚠️ 파일 크기 불일치: 예상 {expected_size_mb}MB, 실제 {total_size/1024/1024:.1f}MB")
        
        # 진행률 표시와 함께 다운로드
        with open(target_path, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=target_path.name) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        
        logger.info(f"✅ 다운로드 완료: {target_path}")
        return True
        
    except Exception as e:
        logger.error(f"❌ 다운로드 실패: {e}")
        # 실패한 파일 삭제
        if target_path.exists():
            target_path.unlink()
        return False

def download_config(url: str, target_path: Path) -> bool:
    """설정 파일 다운로드"""
    try:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        
        if target_path.exists():
            logger.info(f"✅ 설정 파일 이미 존재: {target_path}")
            return True
        
        logger.info(f"📥 설정 파일 다운로드: {url}")
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        
        with open(target_path, 'w') as f:
            f.write(response.text)
        
        logger.info(f"✅ 설정 파일 다운로드 완료: {target_path}")
        return True
        
    except Exception as e:
        logger.error(f"❌ 설정 파일 다운로드 실패: {e}")
        return False

def main():
    """메인 실행 함수"""
    print("🔥 MyCloset AI - DPT 모델 다운로드 스크립트")
    print("=" * 50)
    
    # AI 모델 디렉토리 생성
    AI_MODELS_ROOT.mkdir(exist_ok=True)
    
    success_count = 0
    total_count = len(DPT_MODELS)
    
    print(f"📋 총 {total_count}개 DPT 모델 다운로드 예정")
    print()
    
    for model_key, model_info in DPT_MODELS.items():
        print(f"🔄 {model_info['name']} 다운로드 중...")
        
        # 메인 모델 파일 다운로드
        success = download_file(
            model_info['url'], 
            model_info['target_path'], 
            model_info['size_mb']
        )
        
        if success:
            # 설정 파일도 다운로드
            config_path = model_info['target_path'].parent / "config.json"
            config_success = download_config(model_info['config_url'], config_path)
            
            if config_success:
                success_count += 1
                print(f"✅ {model_info['name']} 완료")
            else:
                print(f"⚠️ {model_info['name']} 모델은 완료했지만 설정 파일 실패")
        else:
            print(f"❌ {model_info['name']} 실패")
        
        print()
    
    # 결과 요약
    print("=" * 50)
    print(f"📊 다운로드 완료: {success_count}/{total_count}개")
    
    if success_count == total_count:
        print("🎉 모든 DPT 모델 다운로드 성공!")
        print("이제 Step 5에서 DPT 모델을 정상적으로 사용할 수 있습니다.")
    else:
        print("⚠️ 일부 모델 다운로드 실패")
        print("실패한 모델은 기본 깊이 추정 모델로 대체됩니다.")
    
    return success_count == total_count

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 