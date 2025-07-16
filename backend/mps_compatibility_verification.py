#!/usr/bin/env python3
"""
MPS 호환성 검증 및 테스트 도구
수정 후 시스템이 올바르게 작동하는지 검증
"""

import sys
import logging
import traceback
from pathlib import Path
from typing import Dict, Any, List, Tuple
import importlib.util
import subprocess

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)

class MPSCompatibilityTester:
    """MPS 호환성 테스터"""
    
    def __init__(self, project_root: str = None):
        self.project_root = Path(project_root) if project_root else Path(".")
        self.backend_root = self.project_root / "backend"
        self.test_results = {}
        
    def test_pytorch_mps_basic(self) -> Dict[str, Any]:
        """기본 PyTorch MPS 테스트"""
        logger.info("🧪 PyTorch MPS 기본 테스트 시작")
        
        try:
            import torch
            
            result = {
                "success": True,
                "pytorch_version": torch.__version__,
                "mps_available": torch.backends.mps.is_available(),
                "mps_built": torch.backends.mps.is_built(),
                "empty_cache_support": hasattr(torch.mps, 'empty_cache'),
                "synchronize_support": hasattr(torch.mps, 'synchronize'),
                "tests": {}
            }
            
            # 1. MPS 디바이스 생성 테스트
            if result["mps_available"]:
                try:
                    device = torch.device("mps")
                    test_tensor = torch.randn(100, 100, device=device)
                    result["tests"]["device_creation"] = {"success": True, "shape": list(test_tensor.shape)}
                    logger.info("✅ MPS 디바이스 생성 성공")
                except Exception as e:
                    result["tests"]["device_creation"] = {"success": False, "error": str(e)}
                    logger.error(f"❌ MPS 디바이스 생성 실패: {e}")
            
            # 2. 메모리 정리 테스트
            memory_cleanup_result = self.test_memory_cleanup()
            result["tests"]["memory_cleanup"] = memory_cleanup_result
            
            # 3. 연산 테스트
            if result["mps_available"]:
                try:
                    a = torch.randn(50, 50, device="mps")
                    b = torch.randn(50, 50, device="mps")
                    c = torch.mm(a, b)
                    result["tests"]["computation"] = {"success": True, "result_shape": list(c.shape)}
                    logger.info("✅ MPS 연산 테스트 성공")
                except Exception as e:
                    result["tests"]["computation"] = {"success": False, "error": str(e)}
                    logger.error(f"❌ MPS 연산 테스트 실패: {e}")
            
            logger.info("✅ PyTorch MPS 기본 테스트 완료")
            return result
            
        except ImportError as e:
            logger.error(f"❌ PyTorch 임포트 실패: {e}")
            return {"success": False, "error": f"PyTorch 임포트 실패: {e}"}
        except Exception as e:
            logger.error(f"❌ PyTorch MPS 테스트 실패: {e}")
            return {"success": False, "error": str(e)}
    
    def test_memory_cleanup(self) -> Dict[str, Any]:
        """메모리 정리 함수 테스트"""
        logger.info("🧹 메모리 정리 테스트 시작")
        
        try:
            import torch
            import gc
            
            if not torch.backends.mps.is_available():
                return {"success": False, "reason": "MPS 사용 불가"}
            
            result = {"success": True, "methods": {}}
            
            # 1. torch.mps.empty_cache() 테스트
            if hasattr(torch.mps, 'empty_cache'):
                try:
                    torch.mps.empty_cache()
                    result["methods"]["mps_empty_cache"] = {"success": True, "available": True}
                    logger.info("✅ torch.mps.empty_cache() 테스트 성공")
                except Exception as e:
                    result["methods"]["mps_empty_cache"] = {"success": False, "error": str(e), "available": True}
                    logger.error(f"❌ torch.mps.empty_cache() 실패: {e}")
            else:
                result["methods"]["mps_empty_cache"] = {"success": False, "available": False}
                logger.warning("⚠️ torch.mps.empty_cache() 지원 안됨")
            
            # 2. torch.mps.synchronize() 테스트
            if hasattr(torch.mps, 'synchronize'):
                try:
                    torch.mps.synchronize()
                    result["methods"]["mps_synchronize"] = {"success": True, "available": True}
                    logger.info("✅ torch.mps.synchronize() 테스트 성공")
                except Exception as e:
                    result["methods"]["mps_synchronize"] = {"success": False, "error": str(e), "available": True}
                    logger.error(f"❌ torch.mps.synchronize() 실패: {e}")
            else:
                result["methods"]["mps_synchronize"] = {"success": False, "available": False}
                logger.warning("⚠️ torch.mps.synchronize() 지원 안됨")
            
            # 3. gc.collect() 테스트 (폴백)
            try:
                gc.collect()
                result["methods"]["gc_collect"] = {"success": True, "available": True}
                logger.info("✅ gc.collect() 테스트 성공")
            except Exception as e:
                result["methods"]["gc_collect"] = {"success": False, "error": str(e), "available": True}
                logger.error(f"❌ gc.collect() 실패: {e}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ 메모리 정리 테스트 실패: {e}")
            return {"success": False, "error": str(e)}
    
    def test_mps_utils_module(self) -> Dict[str, Any]:
        """MPS 유틸리티 모듈 테스트"""
        logger.info("🔧 MPS 유틸리티 모듈 테스트 시작")
        
        try:
            # MPS 유틸리티 모듈 import 테스트
            sys.path.insert(0, str(self.backend_root))
            
            from app.utils.mps_utils import get_mps_manager, safe_mps_empty_cache, get_mps_compatibility_info
            
            result = {"success": True, "tests": {}}
            
            # 1. 매니저 생성 테스트
            try:
                manager = get_mps_manager()
                result["tests"]["manager_creation"] = {"success": True}
                logger.info("✅ MPS 매니저 생성 성공")
            except Exception as e:
                result["tests"]["manager_creation"] = {"success": False, "error": str(e)}
                logger.error(f"❌ MPS 매니저 생성 실패: {e}")
            
            # 2. 메모리 정리 함수 테스트
            try:
                cleanup_result = safe_mps_empty_cache()
                result["tests"]["safe_empty_cache"] = {"success": True, "result": cleanup_result}
                logger.info(f"✅ 안전한 메모리 정리 테스트 성공: {cleanup_result['method']}")
            except Exception as e:
                result["tests"]["safe_empty_cache"] = {"success": False, "error": str(e)}
                logger.error(f"❌ 안전한 메모리 정리 테스트 실패: {e}")
            
            # 3. 호환성 정보 테스트
            try:
                compat_info = get_mps_compatibility_info()
                result["tests"]["compatibility_info"] = {"success": True, "info": compat_info}
                logger.info("✅ 호환성 정보 조회 성공")
            except Exception as e:
                result["tests"]["compatibility_info"] = {"success": False, "error": str(e)}
                logger.error(f"❌ 호환성 정보 조회 실패: {e}")
            
            return result
            
        except ImportError as e:
            logger.error(f"❌ MPS 유틸리티 모듈 임포트 실패: {e}")
            return {"success": False, "error": f"모듈 임포트 실패: {e}"}
        except Exception as e:
            logger.error(f"❌ MPS 유틸리티 모듈 테스트 실패: {e}")
            return {"success": False, "error": str(e)}
    
    def test_core_modules_import(self) -> Dict[str, Any]:
        """핵심 모듈들 import 테스트"""
        logger.info("📦 핵심 모듈 임포트 테스트 시작")
        
        modules_to_test = [
            "app.core.gpu_config",
            "app.services.model_manager",
            "app.ai_pipeline.utils.memory_manager",
            "app.ai_pipeline.pipeline_manager",
            "app.api.pipeline_routes"
        ]
        
        result = {"success": True, "modules": {}}
        
        sys.path.insert(0, str(self.backend_root))
        
        for module_name in modules_to_test:
            try:
                module = importlib.import_module(module_name)
                result["modules"][module_name] = {
                    "success": True,
                    "file_path": getattr(module, '__file__', 'unknown')
                }
                logger.info(f"✅ {module_name} 임포트 성공")
            except Exception as e:
                result["modules"][module_name] = {
                    "success": False,
                    "error": str(e),
                    "traceback": traceback.format_exc()
                }
                result["success"] = False
                logger.error(f"❌ {module_name} 임포트 실패: {e}")
        
        return result
    
    def test_gpu_config_functionality(self) -> Dict[str, Any]:
        """GPU 설정 기능 테스트"""
        logger.info("🔧 GPU 설정 기능 테스트 시작")
        
        try:
            sys.path.insert(0, str(self.backend_root))
            from app.core.gpu_config import GPUManager
            
            result = {"success": True, "tests": {}}
            
            # GPU 매니저 생성 테스트
            try:
                gpu_manager = GPUManager()
                result["tests"]["gpu_manager_creation"] = {"success": True}
                logger.info("✅ GPU 매니저 생성 성공")
                
                # 메모리 정리 테스트
                try:
                    cleanup_result = gpu_manager.cleanup_memory()
                    result["tests"]["gpu_memory_cleanup"] = {"success": True, "result": cleanup_result}
                    logger.info(f"✅ GPU 메모리 정리 테스트 성공: {cleanup_result.get('method', 'unknown')}")
                except Exception as e:
                    result["tests"]["gpu_memory_cleanup"] = {"success": False, "error": str(e)}
                    logger.error(f"❌ GPU 메모리 정리 테스트 실패: {e}")
                
            except Exception as e:
                result["tests"]["gpu_manager_creation"] = {"success": False, "error": str(e)}
                result["success"] = False
                logger.error(f"❌ GPU 매니저 생성 실패: {e}")
            
            return result
            
        except ImportError as e:
            logger.error(f"❌ GPU 설정 모듈 임포트 실패: {e}")
            return {"success": False, "error": f"모듈 임포트 실패: {e}"}
        except Exception as e:
            logger.error(f"❌ GPU 설정 기능 테스트 실패: {e}")
            return {"success": False, "error": str(e)}
    
    def test_server_startup(self) -> Dict[str, Any]:
        """서버 시작 테스트"""
        logger.info("🚀 서버 시작 테스트 시작")
        
        try:
            # 서버 시작 명령어 준비
            main_py_path = self.backend_root / "app" / "main.py"
            
            if not main_py_path.exists():
                return {"success": False, "error": "main.py 파일을 찾을 수 없습니다"}
            
            # 서버 import 테스트 (실제 실행 대신)
            try:
                sys.path.insert(0, str(self.backend_root))
                import app.main
                
                result = {
                    "success": True,
                    "main_module_imported": True,
                    "file_path": str(main_py_path)
                }
                logger.info("✅ 서버 메인 모듈 임포트 성공")
                
                return result
                
            except Exception as e:
                logger.error(f"❌ 서버 메인 모듈 임포트 실패: {e}")
                return {
                    "success": False,
                    "error": str(e),
                    "traceback": traceback.format_exc()
                }
                
        except Exception as e:
            logger.error(f"❌ 서버 시작 테스트 실패: {e}")
            return {"success": False, "error": str(e)}
    
    def run_all_tests(self) -> Dict[str, Any]:
        """모든 테스트 실행"""
        logger.info("🧪 전체 호환성 테스트 시작")
        
        test_suite = {
            "pytorch_mps_basic": self.test_pytorch_mps_basic,
            "mps_utils_module": self.test_mps_utils_module,
            "core_modules_import": self.test_core_modules_import,
            "gpu_config_functionality": self.test_gpu_config_functionality,
            "server_startup": self.test_server_startup
        }
        
        results = {
            "overall_success": True,
            "tests": {},
            "summary": {
                "total_tests": len(test_suite),
                "passed": 0,
                "failed": 0
            }
        }
        
        for test_name, test_func in test_suite.items():
            logger.info(f"🔍 테스트 실행: {test_name}")
            
            try:
                test_result = test_func()
                results["tests"][test_name] = test_result
                
                if test_result.get("success", False):
                    results["summary"]["passed"] += 1
                    logger.info(f"✅ {test_name} 통과")
                else:
                    results["summary"]["failed"] += 1
                    results["overall_success"] = False
                    logger.error(f"❌ {test_name} 실패")
                
            except Exception as e:
                results["tests"][test_name] = {
                    "success": False,
                    "error": str(e),
                    "traceback": traceback.format_exc()
                }
                results["summary"]["failed"] += 1
                results["overall_success"] = False
                logger.error(f"❌ {test_name} 예외 발생: {e}")
        
        # 결과 요약
        logger.info(f"🎯 테스트 완료: {results['summary']['passed']}/{results['summary']['total_tests']} 통과")
        
        return results
    
    def generate_report(self, results: Dict[str, Any]) -> str:
        """테스트 결과 보고서 생성"""
        report = []
        report.append("="*80)
        report.append("🧪 PyTorch 2.1.2 MPS 호환성 테스트 결과")
        report.append("="*80)
        
        # 전체 결과
        status = "✅ 성공" if results["overall_success"] else "❌ 실패"
        report.append(f"📊 전체 결과: {status}")
        report.append(f"📈 통과: {results['summary']['passed']}/{results['summary']['total_tests']}")
        
        # 각 테스트 결과
        report.append("\n📋 세부 테스트 결과:")
        for test_name, test_result in results["tests"].items():
            status = "✅" if test_result.get("success", False) else "❌"
            report.append(f"   {status} {test_name}")
            
            if not test_result.get("success", False) and "error" in test_result:
                report.append(f"      오류: {test_result['error']}")
        
        # PyTorch 정보
        if "pytorch_mps_basic" in results["tests"]:
            pytorch_info = results["tests"]["pytorch_mps_basic"]
            if "pytorch_version" in pytorch_info:
                report.append(f"\n🔥 PyTorch 버전: {pytorch_info['pytorch_version']}")
                report.append(f"🍎 MPS 사용 가능: {pytorch_info.get('mps_available', 'unknown')}")
                report.append(f"🧹 empty_cache 지원: {pytorch_info.get('empty_cache_support', 'unknown')}")
                report.append(f"⚡ synchronize 지원: {pytorch_info.get('synchronize_support', 'unknown')}")
        
        # 권장사항
        report.append("\n💡 권장사항:")
        if results["overall_success"]:
            report.append("   - 모든 테스트가 통과했습니다!")
            report.append("   - 서버를 안전하게 실행할 수 있습니다.")
            report.append("   - 명령어: cd backend && python app/main.py")
        else:
            report.append("   - 일부 테스트가 실패했습니다.")
            report.append("   - 실패한 테스트를 확인하고 수정해주세요.")
            report.append("   - 필요시 수정 스크립트를 다시 실행하세요.")
        
        report.append("\n" + "="*80)
        
        return "\n".join(report)

def main():
    """메인 실행 함수"""
    try:
        print("🧪 PyTorch 2.1.2 MPS 호환성 테스트 시작")
        print("="*60)
        
        # 프로젝트 루트 찾기
        current_dir = Path.cwd()
        project_root = current_dir
        
        if not (project_root / "backend").exists():
            if "backend" in str(current_dir):
                project_root = current_dir.parent
            else:
                print("❌ backend 디렉토리를 찾을 수 없습니다")
                return False
        
        # 테스트 실행
        tester = MPSCompatibilityTester(project_root)
        results = tester.run_all_tests()
        
        # 보고서 생성 및 출력
        report = tester.generate_report(results)
        print(report)
        
        # 결과 파일 저장
        report_file = project_root / "mps_compatibility_test_report.txt"
        report_file.write_text(report, encoding='utf-8')
        print(f"📄 보고서 저장: {report_file}")
        
        return results["overall_success"]
        
    except Exception as e:
        logger.error(f"❌ 테스트 실행 실패: {e}")
        print(f"❌ 테스트 실행 실패: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)