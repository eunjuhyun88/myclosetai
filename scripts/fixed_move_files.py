#!/usr/bin/env python3
# fixed_move_files.py
"""수정된 체크포인트 파일 이동 스크립트"""

import os
import shutil
import logging
from pathlib import Path
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def move_checkpoint_files():
    """체크포인트 파일들을 올바른 위치로 이동"""
    
    ai_models_dir = Path("backend/ai_models")
    
    if not ai_models_dir.exists():
        logger.error(f"❌ AI 모델 디렉토리 없음: {ai_models_dir}")
        return False
    
    logger.info(f"🔍 AI 모델 디렉토리: {ai_models_dir}")
    
    # 이동할 파일 매핑 (코드에서 찾는 파일명 → 실제 존재하는 파일명)
    moves = {
        # Virtual Fitting에서 찾는 파일들을 복사
        "step_06_virtual_fitting/body_pose_model.pth": "step_02_pose_estimation/body_pose_model.pth",
        "step_06_virtual_fitting/hrviton_final_01.pth": "step_06_virtual_fitting/hrviton_final.pth",
        "step_06_virtual_fitting/exp-schp-201908261155-lip.pth": "step_01_human_parsing/exp-schp-201908261155-lip.pth",
        "step_06_virtual_fitting/exp-schp-201908301523-atr.pth": "step_01_human_parsing/exp-schp-201908301523-atr.pth",
        
        # Human Parsing 표준화
        "step_01_human_parsing/exp-schp-201908261155-lip_22.pth": "step_01_human_parsing/exp-schp-201908261155-lip.pth",
        "step_01_human_parsing/exp-schp-201908301523-atr_30.pth": "step_01_human_parsing/exp-schp-201908301523-atr.pth",
        
        # Pose Estimation 표준화
        "step_02_pose_estimation/body_pose_model_41.pth": "step_02_pose_estimation/body_pose_model.pth",
        
        # Cloth Warping
        "step_05_cloth_warping/tom_final_01.pth": "step_05_cloth_warping/tom_final.pth",
    }
    
    copied_count = 0
    failed_count = 0
    
    for target_path, source_path in moves.items():
        target_full = ai_models_dir / target_path
        source_full = ai_models_dir / source_path
        
        logger.info(f"🔄 처리 중: {source_path} → {target_path}")
        
        # 타겟이 이미 존재하면 스킵
        if target_full.exists():
            logger.info(f"✅ 이미 존재: {target_path}")
            continue
        
        # 소스 파일이 없으면 스킵
        if not source_full.exists():
            logger.warning(f"⚠️ 소스 파일 없음: {source_path}")
            failed_count += 1
            continue
        
        try:
            # 타겟 디렉토리 생성 (부모 디렉토리들도 생성)
            target_full.parent.mkdir(parents=True, exist_ok=True)
            logger.debug(f"📁 디렉토리 생성: {target_full.parent}")
            
            # 파일 복사
            shutil.copy2(source_full, target_full)
            
            size_mb = target_full.stat().st_size / (1024 * 1024)
            logger.info(f"✅ 파일 복사 완료: {target_path} ({size_mb:.1f}MB)")
            copied_count += 1
            
        except Exception as e:
            logger.error(f"❌ 파일 복사 실패 {source_path} → {target_path}: {e}")
            failed_count += 1
    
    # 결과 리포트
    logger.info(f"🎉 파일 이동 프로세스 완료!")
    logger.info(f"📊 통계:")
    logger.info(f"   - 복사 성공: {copied_count}개")
    logger.info(f"   - 복사 실패: {failed_count}개")
    
    # 현재 상태 확인
    logger.info("📋 현재 필수 파일 상태 확인:")
    required_files = [
        "step_06_virtual_fitting/body_pose_model.pth",
        "step_06_virtual_fitting/hrviton_final_01.pth",
        "step_01_human_parsing/exp-schp-201908261155-lip_22.pth",
        "step_01_human_parsing/graphonomy_08.pth",
        "step_02_pose_estimation/body_pose_model_41.pth",
        "step_02_pose_estimation/openpose_08.pth",
    ]
    
    existing_count = 0
    for req_file in required_files:
        file_path = ai_models_dir / req_file
        if file_path.exists():
            size_mb = file_path.stat().st_size / (1024 * 1024)
            logger.info(f"   ✅ {req_file} ({size_mb:.1f}MB)")
            existing_count += 1
        else:
            logger.warning(f"   ❌ {req_file}")
    
    logger.info(f"📊 필수 파일 현황: {existing_count}/{len(required_files)}개 존재")
    
    return failed_count == 0

if __name__ == "__main__":
    success = move_checkpoint_files()
    print(f"완료: {success}")
