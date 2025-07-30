#!/usr/bin/env python3
"""
🔥 MyCloset AI - PyTorch 2.7+ 호환성 완전 패치 v2.0
================================================================================
✅ weights_only=True 기본값 변경 문제 해결
✅ Legacy .tar 형식 모델 자동 변환
✅ TorchScript 모델 호환성 처리
✅ 안전한 모델 로딩 함수 제공
✅ 대량 모델 일괄 처리
================================================================================
"""

import os
import torch
import logging
import warnings
from pathlib import Path
from typing import Optional, Dict, Any, List
import shutil
from datetime import datetime

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PyTorchCompatibilityPatcher:
    """PyTorch 2.7+ 호환성 패치 클래스"""
    
    def __init__(self, ai_models_root: str = "ai_models"):
        self.ai_models_root = Path(ai_models_root)
        self.fixed_count = 0
        self.failed_count = 0
        self.skipped_count = 0
        self.backup_dir = self.ai_models_root / "backup_originals"
        
        # 백업 디렉토리 생성
        self.backup_dir.mkdir(exist_ok=True)
        
        logger.info("🔥 PyTorch 호환성 패처 v2.0 초기화 완료")
        logger.info(f"📁 AI 모델 루트: {self.ai_models_root}")
        logger.info(f"💾 백업 디렉토리: {self.backup_dir}")
    
    def safe_load_checkpoint(self, model_path: Path, map_location: str = 'cpu') -> Optional[Dict[str, Any]]:
        """안전한 체크포인트 로딩 (3단계 폴백)"""
        try:
            # 1단계: 안전 모드 시도
            try:
                return torch.load(model_path, map_location=map_location, weights_only=True)
            except Exception as e1:
                logger.debug(f"안전 모드 실패: {e1}")
                
                # 2단계: 호환성 모드 시도  
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        return torch.load(model_path, map_location=map_location, weights_only=False)
                except Exception as e2:
                    logger.debug(f"호환성 모드 실패: {e2}")
                    
                    # 3단계: TorchScript 시도
                    try:
                        return torch.jit.load(model_path, map_location=map_location)
                    except Exception as e3:
                        logger.error(f"모든 로딩 방법 실패: {e3}")
                        return None
                        
        except Exception as e:
            logger.error(f"체크포인트 로딩 완전 실패: {e}")
            return None
    
    def fix_single_model(self, model_path: Path) -> bool:
        """단일 모델 파일 수정"""
        try:
            # 파일 존재 및 유효성 검사
            if not model_path.exists():
                logger.warning(f"⚠️ 파일 없음: {model_path}")
                self.skipped_count += 1
                return False
                
            if not model_path.is_file():
                logger.warning(f"⚠️ 파일이 아님: {model_path}")
                self.skipped_count += 1
                return False
                
            # 심볼릭 링크 처리
            if model_path.is_symlink():
                resolved_path = model_path.resolve()
                if not resolved_path.exists():
                    logger.warning(f"🔗 깨진 심볼릭 링크 제거: {model_path}")
                    model_path.unlink()
                    self.skipped_count += 1
                    return False
                model_path = resolved_path
            
            # 파일 크기 검사
            try:
                file_size = model_path.stat().st_size
                if file_size == 0:
                    logger.warning(f"⚠️ 파일 크기 0: {model_path}")
                    self.skipped_count += 1
                    return False
            except OSError as e:
                logger.warning(f"⚠️ 파일 상태 확인 실패: {model_path} - {e}")
                self.skipped_count += 1
                return False
            
            logger.info(f"🔧 수정 중: {model_path} ({file_size / (1024**2):.1f}MB)")
            
            # 백업 생성 (안전한 경로로)
            try:
                backup_path = self.backup_dir / f"{model_path.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.backup"
                shutil.copy2(model_path, backup_path)
                logger.debug(f"💾 백업 생성: {backup_path}")
            except Exception as backup_error:
                logger.warning(f"⚠️ 백업 실패 (계속 진행): {backup_error}")
            
            # 체크포인트 로드
            checkpoint = self.safe_load_checkpoint(model_path)
            if checkpoint is None:
                logger.error(f"❌ 로딩 실패: {model_path}")
                self.failed_count += 1
                return False
            
            # 안전한 형태로 재저장
            temp_path = model_path.with_suffix(f'.tmp_{os.getpid()}')
            
            try:
                # TorchScript 모델인 경우 특별 처리
                if hasattr(checkpoint, 'save'):
                    checkpoint.save(str(temp_path))
                else:
                    torch.save(
                        checkpoint, 
                        temp_path, 
                        _use_new_zipfile_serialization=True
                    )
                
                # 원본 교체
                if temp_path.exists():
                    model_path.unlink()
                    temp_path.rename(model_path)
                    
                    logger.info(f"✅ 수정 완료: {model_path}")
                    self.fixed_count += 1
                    return True
                else:
                    logger.error(f"❌ 임시 파일 생성 실패: {temp_path}")
                    self.failed_count += 1
                    return False
                    
            except Exception as save_error:
                logger.error(f"❌ 저장 실패 {model_path}: {save_error}")
                # 임시 파일 정리
                if temp_path.exists():
                    temp_path.unlink()
                self.failed_count += 1
                return False
            
        except Exception as e:
            logger.error(f"❌ 수정 실패 {model_path}: {e}")
            self.failed_count += 1
            return False
    
    def find_problematic_models(self) -> List[Path]:
        """문제가 있는 모델 파일들 찾기"""
        problematic_models = []
        
        # 알려진 문제 모델들
        known_problems = [
            "u2net.pth",
            "hrviton_final.pth", 
            "lpips_alex.pth",
            "graphonomy_damaged.pth",
            "ViT-L-14.pt",
            "ViT-B-32.pt"
        ]
        
        for pattern in known_problems:
            try:
                found_files = list(self.ai_models_root.rglob(pattern))
                for file_path in found_files:
                    if file_path.exists() and file_path.is_file():
                        problematic_models.append(file_path)
            except Exception as e:
                logger.warning(f"⚠️ 패턴 검색 실패 {pattern}: {e}")
                continue
        
        # 크기가 0이거나 접근할 수 없는 파일들
        for ext in ['*.pth', '*.pt', '*.bin', '*.safetensors']:
            try:
                for model_file in self.ai_models_root.rglob(ext):
                    try:
                        # 파일 존재 및 접근 가능 확인
                        if not model_file.exists() or not model_file.is_file():
                            continue
                            
                        # 심볼릭 링크인 경우 실제 파일 확인
                        if model_file.is_symlink():
                            if not model_file.resolve().exists():
                                logger.warning(f"🔗 깨진 심볼릭 링크 발견: {model_file}")
                                model_file.unlink()  # 깨진 링크 제거
                                continue
                        
                        # 파일 크기 확인
                        if model_file.stat().st_size == 0:
                            problematic_models.append(model_file)
                            
                    except (OSError, FileNotFoundError) as e:
                        logger.warning(f"⚠️ 파일 접근 실패 {model_file}: {e}")
                        continue
                        
            except Exception as e:
                logger.warning(f"⚠️ 확장자 검색 실패 {ext}: {e}")
                continue
        
        # 중복 제거
        unique_models = []
        seen_paths = set()
        for model in problematic_models:
            if model not in seen_paths:
                unique_models.append(model)
                seen_paths.add(model)
        
        return unique_models
    
    def patch_all_models(self) -> Dict[str, int]:
        """모든 모델 일괄 패치"""
        logger.info("🚀 PyTorch 호환성 패치 시작...")
        
        # 문제 모델들 찾기
        problematic_models = self.find_problematic_models()
        
        logger.info(f"📋 발견된 문제 모델: {len(problematic_models)}개")
        
        for model_path in problematic_models:
            logger.info(f"🔧 처리 중: {model_path.relative_to(self.ai_models_root)}")
            
            if model_path.stat().st_size == 0:
                logger.warning(f"🗑️ 크기 0 파일 삭제: {model_path}")
                model_path.unlink()
                self.skipped_count += 1
                continue
            
            self.fix_single_model(model_path)
        
        # 추가: Legacy .tar 형식 모델 변환
        self._convert_legacy_tar_models()
        
        # 결과 반환
        results = {
            'fixed': self.fixed_count,
            'failed': self.failed_count, 
            'skipped': self.skipped_count,
            'total_processed': len(problematic_models)
        }
        
        logger.info("=" * 60)
        logger.info("🎉 PyTorch 호환성 패치 완료!")
        logger.info(f"✅ 수정 성공: {self.fixed_count}개")
        logger.info(f"❌ 수정 실패: {self.failed_count}개") 
        logger.info(f"⚠️ 건너뜀: {self.skipped_count}개")
        logger.info(f"📊 총 처리: {len(problematic_models)}개")
        logger.info("=" * 60)
        
        return results
    
    def _convert_legacy_tar_models(self):
        """Legacy .tar 형식 모델들 변환"""
        logger.info("🔄 Legacy .tar 형식 모델 검사...")
        
        legacy_models = [
            "hrviton_final.pth",
            "lpips_alex.pth"
        ]
        
        for model_name in legacy_models:
            found_files = list(self.ai_models_root.rglob(model_name))
            for model_path in found_files:
                if model_path.exists() and model_path.stat().st_size > 0:
                    logger.info(f"🔄 Legacy 모델 변환: {model_path}")
                    self.fix_single_model(model_path)
    
    def create_safe_loading_wrapper(self):
        """안전한 로딩 래퍼 함수 생성"""
        wrapper_code = '''
def safe_torch_load(file_path, map_location='cpu', **kwargs):
    """PyTorch 2.7+ 호환 안전 로딩 함수"""
    import torch
    import warnings
    from pathlib import Path
    
    file_path = Path(file_path)
    
    try:
        # 1단계: 안전 모드
        return torch.load(file_path, map_location=map_location, weights_only=True, **kwargs)
    except Exception:
        try:
            # 2단계: 호환성 모드
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return torch.load(file_path, map_location=map_location, weights_only=False, **kwargs)
        except Exception:
            try:
                # 3단계: TorchScript
                return torch.jit.load(file_path, map_location=map_location)
            except Exception as e:
                raise RuntimeError(f"모든 로딩 방법 실패: {e}")

# 사용 예시:
# checkpoint = safe_torch_load("model.pth")
'''
        
        wrapper_file = self.ai_models_root / "safe_loading_utils.py"
        wrapper_file.write_text(wrapper_code)
        logger.info(f"✅ 안전 로딩 래퍼 생성: {wrapper_file}")

def main():
    """메인 실행 함수"""
    print("🔥 MyCloset AI - PyTorch 2.7+ 호환성 패치 v2.0")
    print("=" * 60)
    
    # 패치 실행
    patcher = PyTorchCompatibilityPatcher()
    results = patcher.patch_all_models()
    
    # 안전 로딩 래퍼 생성
    patcher.create_safe_loading_wrapper()
    
    print("\n📋 권장 다음 단계:")
    print("   1. python enhanced_model_loading_validator.py")
    print("   2. python test_complete_ai_inference.py")
    print("   3. 필요시 ./fix_missing_models.sh 실행")
    
    return results

if __name__ == "__main__":
    main()