#!/usr/bin/env python3
"""
PyTorch 2.1.2 MPS 호환성 수정 패치
M3 Max 환경에서 torch.backends.mps.empty_cache() 오류 해결
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any
import re

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('pytorch_mps_fix.log')
    ]
)
logger = logging.getLogger(__name__)

class MPSCompatibilityFixer:
    """PyTorch 2.1.2 MPS 호환성 수정기"""
    
    def __init__(self, project_root: str = None):
        self.project_root = Path(project_root) if project_root else Path(".")
        self.backend_root = self.project_root / "backend"
        self.fixed_files = []
        self.backup_files = []
        
        # 수정할 파일들 (우선순위 순)
        self.target_files = [
            "app/core/gpu_config.py",
            "app/services/model_manager.py", 
            "app/ai_pipeline/utils/memory_manager.py",
            "app/ai_pipeline/pipeline_manager.py",
            "app/api/pipeline_routes.py",
            "app/ai_pipeline/steps/step_08_quality_assessment.py"
        ]
        
        # 패치 패턴들
        self.patch_patterns = [
            # torch.backends.mps.empty_cache() → torch.mps.empty_cache()
            (
                r'torch\.backends\.mps\.empty_cache\(\)',
                'torch.mps.empty_cache() if hasattr(torch.mps, "empty_cache") else None'
            ),
            # if hasattr(torch.backends.mps, 'empty_cache') → if hasattr(torch.mps, 'empty_cache')
            (
                r'hasattr\(torch\.backends\.mps,\s*[\'"]empty_cache[\'"]?\)',
                'hasattr(torch.mps, "empty_cache")'
            ),
            # torch.backends.mps.is_available() → torch.backends.mps.is_available()  (유지)
            # torch.mps.synchronize() 추가 지원
        ]
        
    def create_backup(self, file_path: Path) -> bool:
        """파일 백업 생성"""
        try:
            backup_path = file_path.with_suffix(f"{file_path.suffix}.backup_mps_fix")
            if backup_path.exists():
                logger.info(f"🔄 기존 백업 파일 덮어쓰기: {backup_path}")
            
            backup_path.write_text(file_path.read_text(encoding='utf-8'))
            self.backup_files.append(str(backup_path))
            logger.info(f"💾 백업 생성: {backup_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ 백업 생성 실패 {file_path}: {e}")
            return False
    
    def generate_mps_compatibility_code(self) -> str:
        """MPS 호환성 코드 생성"""
        return '''
def safe_mps_empty_cache():
    """PyTorch 2.1.2 호환 MPS 메모리 정리"""
    try:
        import torch
        if hasattr(torch.mps, 'empty_cache'):
            torch.mps.empty_cache()
            return True
        elif hasattr(torch.mps, 'synchronize'):
            torch.mps.synchronize()
            return True
        else:
            import gc
            gc.collect()
            return False
    except Exception as e:
        logger.warning(f"MPS 메모리 정리 실패: {e}")
        return False
'''
    
    def fix_gpu_config_file(self, file_path: Path) -> bool:
        """gpu_config.py 수정"""
        try:
            content = file_path.read_text(encoding='utf-8')
            
            # 기존 잘못된 패턴 수정
            fixes = [
                # torch.backends.mps.empty_cache() 수정
                (
                    r'torch\.backends\.mps\.empty_cache\(\)',
                    'torch.mps.empty_cache()'
                ),
                # hasattr 체크 수정
                (
                    r'hasattr\(torch\.backends\.mps,\s*[\'"]empty_cache[\'"]?\)',
                    'hasattr(torch.mps, "empty_cache")'
                ),
                # 호환성 체크 로직 개선
                (
                    r'elif hasattr\(torch\.backends\.mps, \'empty_cache\'\):.*?torch\.backends\.mps\.empty_cache\(\)',
                    '''elif hasattr(torch.mps, 'empty_cache'):
                        torch.mps.empty_cache()
                        result["method"] = "mps_empty_cache"
                        logger.info("✅ torch.mps.empty_cache() 실행 완료")'''
                )
            ]
            
            modified = False
            for pattern, replacement in fixes:
                if re.search(pattern, content):
                    content = re.sub(pattern, replacement, content)
                    modified = True
            
            if modified:
                file_path.write_text(content, encoding='utf-8')
                logger.info(f"✅ GPU Config 수정 완료: {file_path}")
                return True
            else:
                logger.info(f"ℹ️ GPU Config 수정 불필요: {file_path}")
                return False
                
        except Exception as e:
            logger.error(f"❌ GPU Config 수정 실패 {file_path}: {e}")
            return False
    
    def fix_model_manager_file(self, file_path: Path) -> bool:
        """model_manager.py 수정"""
        try:
            content = file_path.read_text(encoding='utf-8')
            
            # 기존 잘못된 패턴들 수정
            fixes = [
                # torch.backends.mps.empty_cache() 수정
                (
                    r'torch\.backends\.mps\.empty_cache\(\)',
                    '''if hasattr(torch.mps, 'empty_cache'):
                        torch.mps.empty_cache()
                    elif hasattr(torch.mps, 'synchronize'):
                        torch.mps.synchronize()'''
                ),
                # hasattr 체크 수정
                (
                    r'if hasattr\(torch\.backends\.mps,\s*[\'"]empty_cache[\'"]?\):.*?torch\.backends\.mps\.empty_cache\(\)',
                    '''if hasattr(torch.mps, 'empty_cache'):
                        torch.mps.empty_cache()
                        logger.info("✅ MPS 캐시 정리 완료")'''
                ),
                # 정보 로그 수정
                (
                    r'logger\.info\("ℹ️ MPS empty_cache 미지원 \(PyTorch 2\.5\.1\)"\)',
                    'logger.info("ℹ️ MPS empty_cache 미지원 (PyTorch 2.1.2)")'
                )
            ]
            
            modified = False
            for pattern, replacement in fixes:
                if re.search(pattern, content):
                    content = re.sub(pattern, replacement, content, flags=re.DOTALL)
                    modified = True
            
            if modified:
                file_path.write_text(content, encoding='utf-8')
                logger.info(f"✅ Model Manager 수정 완료: {file_path}")
                return True
            else:
                logger.info(f"ℹ️ Model Manager 수정 불필요: {file_path}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Model Manager 수정 실패 {file_path}: {e}")
            return False
    
    def fix_memory_manager_file(self, file_path: Path) -> bool:
        """memory_manager.py 수정"""
        try:
            content = file_path.read_text(encoding='utf-8')
            
            # MPS 메모리 정리 부분 수정
            fixes = [
                # torch.mps.empty_cache() 호환성 체크
                (
                    r'if hasattr\(torch\.mps,\s*[\'"]empty_cache[\'"]?\):.*?torch\.mps\.empty_cache\(\)',
                    '''if hasattr(torch.mps, 'empty_cache'):
                        torch.mps.empty_cache()
                    elif hasattr(torch.mps, 'synchronize'):
                        torch.mps.synchronize()'''
                ),
                # 예외 처리 개선
                (
                    r'except:.*?pass',
                    '''except Exception as e:
                    logger.warning(f"MPS 메모리 정리 실패: {e}")'''
                )
            ]
            
            modified = False
            for pattern, replacement in fixes:
                if re.search(pattern, content):
                    content = re.sub(pattern, replacement, content, flags=re.DOTALL)
                    modified = True
            
            if modified:
                file_path.write_text(content, encoding='utf-8')
                logger.info(f"✅ Memory Manager 수정 완료: {file_path}")
                return True
            else:
                logger.info(f"ℹ️ Memory Manager 수정 불필요: {file_path}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Memory Manager 수정 실패 {file_path}: {e}")
            return False
    
    def fix_pipeline_manager_file(self, file_path: Path) -> bool:
        """pipeline_manager.py 수정"""
        try:
            content = file_path.read_text(encoding='utf-8')
            
            # torch.mps.empty_cache() 직접 호출 수정
            fixes = [
                (
                    r'torch\.mps\.empty_cache\(\)',
                    '''if hasattr(torch.mps, 'empty_cache'):
                torch.mps.empty_cache()
            elif hasattr(torch.mps, 'synchronize'):
                torch.mps.synchronize()'''
                ),
                # hasattr 체크 수정
                (
                    r'if hasattr\(torch\.backends,\s*[\'"]mps[\'"]?\).*?torch\.mps\.empty_cache\(\)',
                    '''if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                if hasattr(torch.mps, 'empty_cache'):
                    torch.mps.empty_cache()
                elif hasattr(torch.mps, 'synchronize'):
                    torch.mps.synchronize()'''
                )
            ]
            
            modified = False
            for pattern, replacement in fixes:
                if re.search(pattern, content):
                    content = re.sub(pattern, replacement, content, flags=re.DOTALL)
                    modified = True
            
            if modified:
                file_path.write_text(content, encoding='utf-8')
                logger.info(f"✅ Pipeline Manager 수정 완료: {file_path}")
                return True
            else:
                logger.info(f"ℹ️ Pipeline Manager 수정 불필요: {file_path}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Pipeline Manager 수정 실패 {file_path}: {e}")
            return False
    
    def fix_pipeline_routes_file(self, file_path: Path) -> bool:
        """pipeline_routes.py 수정"""
        try:
            content = file_path.read_text(encoding='utf-8')
            
            # torch.mp 관련 수정
            fixes = [
                (
                    r'torch\.mp',
                    'torch.mps'
                ),
                # 메모리 정리 로직 개선
                (
                    r'if pipeline\.device == \'mps\'.*?torch\.backends\.mps\.is_available\(\):.*?torch\.mp',
                    '''if pipeline.device == 'mps' and torch.backends.mps.is_available():
                if hasattr(torch.mps, 'empty_cache'):
                    torch.mps.empty_cache()
                elif hasattr(torch.mps, 'synchronize'):
                    torch.mps.synchronize()'''
                )
            ]
            
            modified = False
            for pattern, replacement in fixes:
                if re.search(pattern, content):
                    content = re.sub(pattern, replacement, content, flags=re.DOTALL)
                    modified = True
            
            if modified:
                file_path.write_text(content, encoding='utf-8')
                logger.info(f"✅ Pipeline Routes 수정 완료: {file_path}")
                return True
            else:
                logger.info(f"ℹ️ Pipeline Routes 수정 불필요: {file_path}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Pipeline Routes 수정 실패 {file_path}: {e}")
            return False
    
    def fix_quality_assessment_file(self, file_path: Path) -> bool:
        """step_08_quality_assessment.py 수정"""
        try:
            content = file_path.read_text(encoding='utf-8')
            
            # MPS 메모리 정리 부분 수정
            fixes = [
                # torch.backends.mps.empty_cache() 수정
                (
                    r'if hasattr\(torch\.backends\.mps,\s*[\'"]empty_cache[\'"]?\):.*?torch\.backends\.mps\.empty_cache\(\)',
                    '''if hasattr(torch.mps, 'empty_cache'):
                        torch.mps.empty_cache()'''
                ),
                # torch.mps.synchronize() 추가 지원
                (
                    r'elif hasattr\(torch\.mps,\s*[\'"]synchronize[\'"]?\):.*?torch\.mps\.synchronize\(\)',
                    '''elif hasattr(torch.mps, 'synchronize'):
                        torch.mps.synchronize()'''
                )
            ]
            
            modified = False
            for pattern, replacement in fixes:
                if re.search(pattern, content):
                    content = re.sub(pattern, replacement, content, flags=re.DOTALL)
                    modified = True
            
            if modified:
                file_path.write_text(content, encoding='utf-8')
                logger.info(f"✅ Quality Assessment 수정 완료: {file_path}")
                return True
            else:
                logger.info(f"ℹ️ Quality Assessment 수정 불필요: {file_path}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Quality Assessment 수정 실패 {file_path}: {e}")
            return False
    
    def create_mps_utility_module(self) -> bool:
        """MPS 유틸리티 모듈 생성"""
        try:
            utils_dir = self.backend_root / "app" / "utils"
            utils_dir.mkdir(exist_ok=True)
            
            mps_utils_path = utils_dir / "mps_utils.py"
            
            mps_utils_content = '''"""
MPS 유틸리티 모듈 - PyTorch 2.1.2 호환
M3 Max 환경에서 안전한 MPS 메모리 관리
"""

import logging
import gc
from typing import Dict, Any, Optional
import torch

logger = logging.getLogger(__name__)

class MPSMemoryManager:
    """M3 Max MPS 메모리 관리자"""
    
    def __init__(self):
        self.is_available = torch.backends.mps.is_available()
        self.supports_empty_cache = hasattr(torch.mps, 'empty_cache')
        self.supports_synchronize = hasattr(torch.mps, 'synchronize')
        
        logger.info(f"🍎 MPS 메모리 관리자 초기화")
        logger.info(f"   - MPS 사용 가능: {self.is_available}")
        logger.info(f"   - empty_cache 지원: {self.supports_empty_cache}")
        logger.info(f"   - synchronize 지원: {self.supports_synchronize}")
    
    def safe_empty_cache(self) -> Dict[str, Any]:
        """안전한 MPS 메모리 정리"""
        result = {
            "success": False,
            "method": "none",
            "message": "MPS 사용 불가"
        }
        
        if not self.is_available:
            return result
        
        try:
            if self.supports_empty_cache:
                torch.mps.empty_cache()
                result.update({
                    "success": True,
                    "method": "mps_empty_cache",
                    "message": "torch.mps.empty_cache() 실행 완료"
                })
                logger.info("✅ torch.mps.empty_cache() 실행 완료")
                
            elif self.supports_synchronize:
                torch.mps.synchronize()
                result.update({
                    "success": True,
                    "method": "mps_synchronize",
                    "message": "torch.mps.synchronize() 실행 완료"
                })
                logger.info("✅ torch.mps.synchronize() 실행 완료")
                
            else:
                gc.collect()
                result.update({
                    "success": True,
                    "method": "gc_collect",
                    "message": "가비지 컬렉션으로 대체"
                })
                logger.info("✅ 가비지 컬렉션으로 메모리 정리")
            
            return result
            
        except Exception as e:
            result.update({
                "success": False,
                "method": "error",
                "message": f"MPS 메모리 정리 실패: {e}"
            })
            logger.error(f"❌ MPS 메모리 정리 실패: {e}")
            return result
    
    def get_compatibility_info(self) -> Dict[str, Any]:
        """MPS 호환성 정보 조회"""
        return {
            "pytorch_version": torch.__version__,
            "mps_available": self.is_available,
            "mps_built": torch.backends.mps.is_built(),
            "empty_cache_support": self.supports_empty_cache,
            "synchronize_support": self.supports_synchronize,
            "recommended_method": (
                "mps_empty_cache" if self.supports_empty_cache 
                else "mps_synchronize" if self.supports_synchronize 
                else "gc_collect"
            )
        }

# 전역 인스턴스
_mps_manager = None

def get_mps_manager() -> MPSMemoryManager:
    """전역 MPS 관리자 반환"""
    global _mps_manager
    if _mps_manager is None:
        _mps_manager = MPSMemoryManager()
    return _mps_manager

def safe_mps_empty_cache() -> Dict[str, Any]:
    """안전한 MPS 메모리 정리 (함수형 인터페이스)"""
    return get_mps_manager().safe_empty_cache()

def get_mps_compatibility_info() -> Dict[str, Any]:
    """MPS 호환성 정보 조회 (함수형 인터페이스)"""
    return get_mps_manager().get_compatibility_info()
'''
            
            mps_utils_path.write_text(mps_utils_content, encoding='utf-8')
            logger.info(f"✅ MPS 유틸리티 모듈 생성: {mps_utils_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ MPS 유틸리티 모듈 생성 실패: {e}")
            return False
    
    def run_fix(self) -> Dict[str, Any]:
        """전체 수정 실행"""
        logger.info("🔧 PyTorch 2.1.2 MPS 호환성 수정 시작")
        
        results = {
            "success": False,
            "fixed_files": [],
            "failed_files": [],
            "backup_files": [],
            "total_files": len(self.target_files)
        }
        
        # 파일별 수정 함수 매핑
        fix_functions = {
            "app/core/gpu_config.py": self.fix_gpu_config_file,
            "app/services/model_manager.py": self.fix_model_manager_file,
            "app/ai_pipeline/utils/memory_manager.py": self.fix_memory_manager_file,
            "app/ai_pipeline/pipeline_manager.py": self.fix_pipeline_manager_file,
            "app/api/pipeline_routes.py": self.fix_pipeline_routes_file,
            "app/ai_pipeline/steps/step_08_quality_assessment.py": self.fix_quality_assessment_file
        }
        
        # 각 파일 수정
        for file_path_str in self.target_files:
            file_path = self.backend_root / file_path_str
            
            if not file_path.exists():
                logger.warning(f"⚠️ 파일 없음: {file_path}")
                results["failed_files"].append(file_path_str)
                continue
            
            # 백업 생성
            if not self.create_backup(file_path):
                results["failed_files"].append(file_path_str)
                continue
            
            # 파일 수정
            fix_function = fix_functions.get(file_path_str)
            if fix_function:
                if fix_function(file_path):
                    results["fixed_files"].append(file_path_str)
                else:
                    results["failed_files"].append(file_path_str)
            else:
                logger.warning(f"⚠️ 수정 함수 없음: {file_path_str}")
                results["failed_files"].append(file_path_str)
        
        # MPS 유틸리티 모듈 생성
        if self.create_mps_utility_module():
            results["fixed_files"].append("app/utils/mps_utils.py")
        
        # 결과 집계
        results["backup_files"] = self.backup_files
        results["success"] = len(results["failed_files"]) == 0
        
        logger.info(f"🎉 수정 완료: {len(results['fixed_files'])}/{results['total_files']}")
        logger.info(f"✅ 성공: {results['fixed_files']}")
        if results["failed_files"]:
            logger.warning(f"❌ 실패: {results['failed_files']}")
        
        return results
    
    def rollback(self) -> bool:
        """백업에서 롤백"""
        try:
            logger.info("🔄 롤백 시작")
            
            for backup_file_str in self.backup_files:
                backup_path = Path(backup_file_str)
                if not backup_path.exists():
                    continue
                
                original_path = backup_path.with_suffix('')
                original_path.write_text(backup_path.read_text(encoding='utf-8'))
                logger.info(f"🔄 롤백: {original_path}")
            
            logger.info("✅ 롤백 완료")
            return True
            
        except Exception as e:
            logger.error(f"❌ 롤백 실패: {e}")
            return False

def main():
    """메인 실행 함수"""
    try:
        # 프로젝트 루트 찾기
        current_dir = Path.cwd()
        project_root = current_dir
        
        # backend 디렉토리 찾기
        if not (project_root / "backend").exists():
            if "backend" in str(current_dir):
                project_root = current_dir.parent
            else:
                logger.error("❌ backend 디렉토리를 찾을 수 없습니다")
                return False
        
        # 수정 실행
        fixer = MPSCompatibilityFixer(project_root)
        results = fixer.run_fix()
        
        # 결과 출력
        print("\n" + "="*60)
        print("🎉 PyTorch 2.1.2 MPS 호환성 수정 완료!")
        print("="*60)
        print(f"✅ 성공: {len(results['fixed_files'])}/{results['total_files']}")
        print(f"📁 수정된 파일: {results['fixed_files']}")
        
        if results['failed_files']:
            print(f"❌ 실패한 파일: {results['failed_files']}")
        
        print(f"💾 백업 파일: {len(results['backup_files'])}개")
        print("\n🚀 이제 서버를 다시 시작해주세요:")
        print("   cd backend && python app/main.py")
        
        return results['success']
        
    except Exception as e:
        logger.error(f"❌ 수정 도구 실행 실패: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)