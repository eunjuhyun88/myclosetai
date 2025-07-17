#!/usr/bin/env python3
"""
🚀 MyCloset AI - 실제 모델 최적화 실행 스크립트
74.4GB 절약! 201.7GB → 127.2GB

직접 실행: python quick_execute_optimizer.py
"""

import os
import shutil
import logging
from pathlib import Path
from datetime import datetime

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)

def confirm_execution():
    """실행 확인"""
    print("🚨 중요한 확인사항:")
    print("   - 74.4GB가 삭제됩니다")
    print("   - 11개 모델이 제거됩니다") 
    print("   - 되돌릴 수 없습니다 (백업에서만 복원 가능)")
    print("")
    
    response = input("정말 실행하시겠습니까? (yes/no): ")
    return response.lower() in ['yes', 'y']

def check_backups():
    """백업 확인"""
    backup_dirs = list(Path('.').glob('backup_essential_*'))
    if not backup_dirs:
        logger.error("❌ 백업을 찾을 수 없습니다!")
        logger.info("먼저 백업을 실행하세요: bash backup_script.sh")
        return False
    
    logger.info(f"✅ {len(backup_dirs)}개 백업 발견")
    for backup in backup_dirs:
        size = get_dir_size(backup)
        logger.info(f"   📦 {backup.name}: {size}")
    return True

def get_dir_size(path):
    """디렉토리 크기 계산"""
    try:
        result = os.popen(f'du -sh "{path}" 2>/dev/null').read().strip()
        return result.split('\t')[0] if result else "Unknown"
    except:
        return "Unknown"

def remove_model(model_name):
    """모델 제거"""
    model_path = Path(f"ai_models/checkpoints/{model_name}")
    
    if not model_path.exists():
        logger.info(f"✅ {model_name}: 이미 제거됨")
        return True
    
    try:
        if model_path.is_dir():
            shutil.rmtree(model_path)
        else:
            model_path.unlink()
        
        logger.info(f"✅ 제거 완료: {model_name}")
        return True
    except Exception as e:
        logger.error(f"❌ 제거 실패: {model_name} - {e}")
        return False

def execute_optimization():
    """실제 최적화 실행"""
    
    # 제거 대상 모델들 (분석 결과 기반)
    removal_targets = [
        "stable_diffusion_v15",     # 44.0GB
        "stable_diffusion_inpaint", # 14.2GB  
        "sam_vit_h",               # 7.2GB
        "clip-vit-large-patch14",  # 6.4GB
        "controlnet_openpose",     # 2.7GB
        "esrgan",                  # 0.0GB (빈 디렉토리)
        "gfpgan",                  # 0.0GB (빈 디렉토리)
        "rembg",                   # 0.0GB (빈 디렉토리)
        "viton_hd",                # 0.0GB (불완전)
        "densepose",               # 0.0GB (빈 디렉토리)
        "u2net_cloth"              # 0.0GB (빈 디렉토리)
    ]
    
    logger.info("🚀 실제 모델 제거 시작...")
    
    successful = 0
    failed = 0
    total_removed_size = 0
    
    for model in removal_targets:
        # 제거 전 크기 측정
        model_path = Path(f"ai_models/checkpoints/{model}")
        if model_path.exists():
            size_before = get_dir_size(model_path)
            logger.info(f"🗑️ 제거 중: {model} ({size_before})")
        
        # 실제 제거
        if remove_model(model):
            successful += 1
        else:
            failed += 1
    
    # 결과 요약
    logger.info(f"")
    logger.info(f"📊 제거 작업 완료:")
    logger.info(f"   ✅ 성공: {successful}개")
    logger.info(f"   ❌ 실패: {failed}개")
    logger.info(f"   📈 성공률: {successful/(successful+failed)*100:.1f}%")
    
    return successful, failed

def verify_optimization():
    """최적화 결과 확인"""
    logger.info("🔍 최적화 결과 확인 중...")
    
    # 현재 크기 측정
    current_size = get_dir_size("ai_models/checkpoints")
    
    # 남은 모델 수 계산
    remaining_models = len(list(Path("ai_models/checkpoints").iterdir()))
    
    logger.info(f"📊 최적화 후 상태:")
    logger.info(f"   💾 현재 크기: {current_size}")
    logger.info(f"   📦 남은 모델: {remaining_models}개")
    
    # 핵심 모델 확인
    essential_models = [
        "ootdiffusion", "ootdiffusion_hf",
        "human_parsing", "step_01_human_parsing", 
        "openpose", "step_02_pose_estimation",
        "u2net", "step_03_cloth_segmentation",
        "step_04_geometric_matching", "step_05_cloth_warping"
    ]
    
    logger.info(f"🎯 핵심 모델 상태 확인:")
    for model in essential_models:
        model_path = Path(f"ai_models/checkpoints/{model}")
        status = "✅" if model_path.exists() else "❌"
        logger.info(f"   {status} {model}")

def main():
    """메인 함수"""
    print("🚀 MyCloset AI - 실제 모델 최적화 실행")
    print("====================================")
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("")
    
    # 1. 백업 확인
    logger.info("📦 백업 상태 확인...")
    if not check_backups():
        return False
    
    print("")
    
    # 2. 실행 확인
    if not confirm_execution():
        logger.info("❌ 실행이 취소되었습니다")
        return False
    
    print("")
    
    # 3. 실제 최적화 실행
    successful, failed = execute_optimization()
    
    print("")
    
    # 4. 결과 확인
    verify_optimization()
    
    print("")
    
    # 5. 완료 메시지
    if failed == 0:
        logger.info("🎉 모델 최적화 완료!")
        logger.info("💡 예상 효과: 201.7GB → 127.2GB (74.4GB 절약)")
    else:
        logger.warning(f"⚠️ {failed}개 모델 제거 실패")
        logger.info("수동으로 확인이 필요합니다")
    
    print("")
    logger.info("🚀 다음 단계:")
    logger.info("   1. 서버 테스트: python app/main.py")
    logger.info("   2. 성능 확인: python scripts/test/test_models.py")
    
    return successful > 0

if __name__ == "__main__":
    main()