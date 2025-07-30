#!/usr/bin/env python3
"""
🔥 MyCloset AI - 누락된 AI 모델 다운로드 및 복구 스크립트
================================================================

누락된 모델들:
- schp_atr 모델
- schp_lip 모델 
- atr_model 모델
- lip_model 모델
- 손상된 safetensors 파일들

Author: MyCloset AI Team
Date: 2025-07-30
"""

import os
import sys
import requests
import logging
from pathlib import Path
from typing import Dict, List
import hashlib
import shutil

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 프로젝트 경로 설정
PROJECT_ROOT = Path(__file__).parent
AI_MODELS_ROOT = PROJECT_ROOT / "backend" / "ai_models"

# 누락된 모델 URL 매핑
MISSING_MODELS = {
    "schp_atr": {
        "url": "https://github.com/PeikeLi/Self-Correction-Human-Parsing/releases/download/v1.0/exp-schp-201908261155-atr.pth",
        "target_path": AI_MODELS_ROOT / "checkpoints" / "step_01_human_parsing" / "exp-schp-201908261155-atr.pth",
        "size_mb": 255.1
    },
    "schp_lip": {
        "url": "https://github.com/PeikeLi/Self-Correction-Human-Parsing/releases/download/v1.0/exp-schp-201908261155-lip.pth", 
        "target_path": AI_MODELS_ROOT / "checkpoints" / "step_01_human_parsing" / "exp-schp-201908261155-lip.pth",
        "size_mb": 255.1
    },
    "body_pose_model": {
        "url": "http://posefs1.perception.cs.cmu.edu/OpenPose/models/pose/body_25/pose_iter_584000.caffemodel",
        "target_path": AI_MODELS_ROOT / "checkpoints" / "step_02_pose_estimation" / "body_pose_model.pth",
        "size_mb": 200.0
    },
    "sam_vit_h": {
        "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
        "target_path": AI_MODELS_ROOT / "checkpoints" / "step_03_cloth_segmentation" / "sam_vit_h_4b8939.pth", 
        "size_mb": 2400.0
    }
}

def download_file(url: str, target_path: Path, expected_size_mb: float = None):
    """파일 다운로드 함수"""
    try:
        # 디렉토리 생성
        target_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"🔄 다운로드 시작: {url}")
        logger.info(f"📁 저장 경로: {target_path}")
        
        # 파일이 이미 존재하고 크기가 맞으면 스킵
        if target_path.exists():
            file_size_mb = target_path.stat().st_size / (1024 * 1024)
            if expected_size_mb and abs(file_size_mb - expected_size_mb) < 10:
                logger.info(f"✅ 파일이 이미 존재함: {target_path} ({file_size_mb:.1f}MB)")
                return True
        
        # 다운로드 실행
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(target_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    # 진행률 표시
                    if total_size > 0:
                        progress = (downloaded / total_size) * 100
                        print(f"\r진행률: {progress:.1f}% ({downloaded/(1024*1024):.1f}MB/{total_size/(1024*1024):.1f}MB)", end='')
        
        print()  # 개행
        
        # 파일 크기 검증
        final_size_mb = target_path.stat().st_size / (1024 * 1024)
        logger.info(f"✅ 다운로드 완료: {target_path} ({final_size_mb:.1f}MB)")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 다운로드 실패: {url} - {e}")
        return False

def verify_and_fix_corrupted_files():
    """손상된 파일 탐지 및 수정"""
    corrupted_files = []
    
    # safetensors 파일들 검증
    safetensors_files = list(AI_MODELS_ROOT.rglob("*.safetensors"))
    
    for file_path in safetensors_files:
        try:
            # 파일 크기 확인 (너무 작으면 손상 의심)
            if file_path.stat().st_size < 1024:  # 1KB 미만
                corrupted_files.append(file_path)
                continue
                
            # safetensors 헤더 검증
            with open(file_path, 'rb') as f:
                header_size_bytes = f.read(8)
                if len(header_size_bytes) < 8:
                    corrupted_files.append(file_path)
                    
        except Exception as e:
            logger.warning(f"⚠️ 파일 검증 실패: {file_path} - {e}")
            corrupted_files.append(file_path)
    
    # 손상된 파일들 백업 후 삭제
    for corrupted_file in corrupted_files:
        backup_path = corrupted_file.with_suffix(corrupted_file.suffix + '.corrupted')
        try:
            shutil.move(str(corrupted_file), str(backup_path))
            logger.info(f"🗑️ 손상된 파일 백업: {corrupted_file} → {backup_path}")
        except Exception as e:
            logger.error(f"❌ 손상된 파일 처리 실패: {corrupted_file} - {e}")
    
    return corrupted_files

def create_model_symlinks():
    """모델 파일 심볼릭 링크 생성"""
    symlink_mappings = {
        # SCHP 모델들을 다른 이름으로도 접근 가능하게
        AI_MODELS_ROOT / "checkpoints" / "step_01_human_parsing" / "exp-schp-201908261155-atr.pth": [
            AI_MODELS_ROOT / "checkpoints" / "step_01_human_parsing" / "atr_model.pth",
            AI_MODELS_ROOT / "checkpoints" / "step_01_human_parsing" / "schp_atr.pth"
        ],
        AI_MODELS_ROOT / "checkpoints" / "step_01_human_parsing" / "exp-schp-201908261155-lip.pth": [
            AI_MODELS_ROOT / "checkpoints" / "step_01_human_parsing" / "lip_model.pth",
            AI_MODELS_ROOT / "checkpoints" / "step_01_human_parsing" / "schp_lip.pth"
        ]
    }
    
    for source_path, target_paths in symlink_mappings.items():
        if source_path.exists():
            for target_path in target_paths:
                try:
                    if not target_path.exists():
                        target_path.symlink_to(source_path)
                        logger.info(f"🔗 심볼릭 링크 생성: {target_path} → {source_path}")
                except Exception as e:
                    logger.warning(f"⚠️ 심볼릭 링크 생성 실패: {target_path} - {e}")

def main():
    """메인 실행 함수"""
    logger.info("🔥 MyCloset AI - 누락된 모델 다운로드 시작")
    logger.info("=" * 60)
    
    # 1. 손상된 파일 정리
    logger.info("🧹 손상된 파일 정리 중...")
    corrupted_files = verify_and_fix_corrupted_files()
    logger.info(f"🧹 손상된 파일 {len(corrupted_files)}개 처리 완료")
    
    # 2. 누락된 모델 다운로드
    logger.info("📥 누락된 모델 다운로드 중...")
    success_count = 0
    
    for model_name, model_info in MISSING_MODELS.items():
        logger.info(f"\n🎯 {model_name} 다운로드 중...")
        
        if download_file(
            model_info["url"], 
            model_info["target_path"], 
            model_info["size_mb"]
        ):
            success_count += 1
        
    # 3. 심볼릭 링크 생성
    logger.info("\n🔗 모델 심볼릭 링크 생성 중...")
    create_model_symlinks()
    
    # 4. 결과 요약
    logger.info("\n" + "=" * 60)
    logger.info(f"🎉 모델 다운로드 완료: {success_count}/{len(MISSING_MODELS)}개 성공")
    
    if success_count == len(MISSING_MODELS):
        logger.info("✅ 모든 모델 다운로드 성공!")
        logger.info("🚀 이제 python debug_model_loading.py를 다시 실행해보세요.")
    else:
        logger.warning("⚠️ 일부 모델 다운로드 실패. 수동 다운로드가 필요할 수 있습니다.")

if __name__ == "__main__":
    main()