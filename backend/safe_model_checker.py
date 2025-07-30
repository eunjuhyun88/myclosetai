#!/usr/bin/env python3
"""
🔥 MyCloset AI - 안전한 모델 체커 v1.0
================================================================================
✅ 깨진 심볼릭 링크 및 접근 불가능한 파일 안전 처리
✅ 모델 파일 상태 검증
✅ 문제 파일 자동 정리
================================================================================
"""

import os
import sys
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def safe_file_check(file_path: Path) -> dict:
    """안전한 파일 상태 확인"""
    result = {
        'exists': False,
        'is_file': False,
        'is_symlink': False,
        'size_mb': 0,
        'accessible': False,
        'issue': None
    }
    
    try:
        # 기본 존재 확인
        result['exists'] = file_path.exists()
        
        if not result['exists']:
            result['issue'] = 'file_not_found'
            return result
        
        # 심볼릭 링크 확인
        result['is_symlink'] = file_path.is_symlink()
        
        if result['is_symlink']:
            # 심볼릭 링크가 가리키는 실제 파일 확인
            try:
                resolved = file_path.resolve()
                if not resolved.exists():
                    result['issue'] = 'broken_symlink'
                    return result
            except Exception as e:
                result['issue'] = f'symlink_resolution_error: {e}'
                return result
        
        # 파일 타입 확인
        result['is_file'] = file_path.is_file()
        
        if not result['is_file']:
            result['issue'] = 'not_a_file'
            return result
        
        # 크기 확인
        try:
            size_bytes = file_path.stat().st_size
            result['size_mb'] = round(size_bytes / (1024 * 1024), 2)
            result['accessible'] = True
            
            if size_bytes == 0:
                result['issue'] = 'zero_size'
            
        except (OSError, FileNotFoundError) as e:
            result['issue'] = f'stat_error: {e}'
        
    except Exception as e:
        result['issue'] = f'unexpected_error: {e}'
    
    return result

def clean_broken_files(ai_models_root: str = "ai_models"):
    """깨진 파일들 정리"""
    root_path = Path(ai_models_root)
    
    if not root_path.exists():
        logger.error(f"❌ AI 모델 디렉토리 없음: {root_path}")
        return
    
    logger.info(f"🔍 모델 파일 검사 시작: {root_path}")
    
    broken_files = []
    zero_size_files = []
    total_files = 0
    
    # 모든 모델 파일 확장자
    extensions = ['*.pth', '*.pt', '*.bin', '*.safetensors', '*.onnx']
    
    for ext in extensions:
        try:
            for file_path in root_path.rglob(ext):
                total_files += 1
                
                # 안전한 파일 검사
                check_result = safe_file_check(file_path)
                
                if check_result['issue']:
                    logger.warning(f"⚠️ 문제 파일: {file_path.relative_to(root_path)} - {check_result['issue']}")
                    
                    if check_result['issue'] in ['broken_symlink', 'not_a_file']:
                        broken_files.append(file_path)
                    elif check_result['issue'] == 'zero_size':
                        zero_size_files.append(file_path)
                
                elif check_result['accessible']:
                    logger.debug(f"✅ 정상: {file_path.relative_to(root_path)} ({check_result['size_mb']}MB)")
        
        except Exception as e:
            logger.warning(f"⚠️ 확장자 검색 실패 {ext}: {e}")
    
    # 깨진 파일들 정리
    if broken_files:
        logger.info(f"🗑️ 깨진 파일 {len(broken_files)}개 제거...")
        for file_path in broken_files:
            try:
                if file_path.is_symlink():
                    file_path.unlink()
                    logger.info(f"   🔗 심볼릭 링크 제거: {file_path}")
                elif file_path.exists():
                    file_path.unlink()
                    logger.info(f"   📁 파일 제거: {file_path}")
            except Exception as e:
                logger.error(f"   ❌ 제거 실패 {file_path}: {e}")
    
    # 크기 0 파일들 처리
    if zero_size_files:
        logger.info(f"📝 크기 0 파일 {len(zero_size_files)}개 발견:")
        for file_path in zero_size_files:
            logger.info(f"   📄 {file_path.relative_to(root_path)}")
            # 크기 0 파일은 삭제하지 말고 보고만 함
    
    # 결과 요약
    logger.info("=" * 60)
    logger.info("🎉 모델 파일 검사 완료!")
    logger.info(f"📊 총 검사 파일: {total_files}개")
    logger.info(f"🗑️ 제거된 깨진 파일: {len(broken_files)}개")
    logger.info(f"⚠️ 크기 0 파일: {len(zero_size_files)}개")
    logger.info("=" * 60)

def list_model_summary(ai_models_root: str = "ai_models"):
    """모델 요약 정보 출력"""
    root_path = Path(ai_models_root)
    
    if not root_path.exists():
        logger.error(f"❌ AI 모델 디렉토리 없음: {root_path}")
        return
    
    step_dirs = sorted([d for d in root_path.iterdir() if d.is_dir() and d.name.startswith('step_')])
    
    logger.info("📋 Step별 모델 현황:")
    logger.info("-" * 60)
    
    total_size_gb = 0
    total_files = 0
    
    for step_dir in step_dirs:
        # 각 step의 모델 파일들
        model_files = []
        for ext in ['*.pth', '*.pt', '*.bin', '*.safetensors']:
            model_files.extend(list(step_dir.rglob(ext)))
        
        step_size_mb = 0
        healthy_files = 0
        
        for model_file in model_files:
            check_result = safe_file_check(model_file)
            if check_result['accessible'] and not check_result['issue']:
                step_size_mb += check_result['size_mb']
                healthy_files += 1
        
        step_size_gb = step_size_mb / 1024
        total_size_gb += step_size_gb
        total_files += healthy_files
        
        status_icon = "✅" if healthy_files > 0 else "❌"
        logger.info(f"{status_icon} {step_dir.name}: {healthy_files}개 파일, {step_size_gb:.1f}GB")
    
    logger.info("-" * 60)
    logger.info(f"📊 전체: {total_files}개 파일, {total_size_gb:.1f}GB")

def main():
    """메인 함수"""
    print("🔥 MyCloset AI - 안전한 모델 체커 v1.0")
    print("=" * 60)
    
    # 1. 깨진 파일들 정리
    clean_broken_files()
    
    print()
    
    # 2. 모델 현황 요약
    list_model_summary()
    
    print()
    print("💡 다음 단계:")
    print("   1. python pytorch_compatibility_patch.py")
    print("   2. python analyze_model_status.py")

if __name__ == "__main__":
    main()