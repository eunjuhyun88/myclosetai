#!/usr/bin/env python3
"""
🔧 MyCloset AI - 모델 로더 테스트 스크립트
✅ 체크포인트 분석 결과 기반 모델 로딩 테스트
✅ M3 Max 128GB 최적화
✅ 실제 AI 모델 로딩 검증
"""

import os
import sys
import time
import asyncio
import logging
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
import psutil

# 프로젝트 경로 설정
current_file = Path(__file__).resolve()
backend_dir = current_file.parent.parent
project_root = backend_dir.parent
sys.path.insert(0, str(backend_dir))

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from app.core.optimized_model_paths import (
        ANALYZED_MODELS, get_optimal_model_for_step,
        get_model_checkpoints, get_largest_checkpoint
    )
    OPTIMIZED_PATHS_AVAILABLE = True
except ImportError:
    OPTIMIZED_PATHS_AVAILABLE = False

try:
    from app.ai_pipeline.utils.checkpoint_model_loader import (
        CheckpointModelLoader, get_checkpoint_model_loader
    )
    CHECKPOINT_LOADER_AVAILABLE = True
except ImportError:
    CHECKPOINT_LOADER_AVAILABLE = False

class ModelLoaderTester:
    """모델 로더 테스트 클래스"""
    
    def __init__(self):
        self.backend_dir = backend_dir
        self.device = self._detect_device()
        self.test_results = {}
        
        logger.info(f"🔧 모델 로더 테스트 초기화")
        logger.info(f"📁 백엔드 디렉토리: {self.backend_dir}")
        logger.info(f"🔧 디바이스: {self.device}")
        
    def _detect_device(self) -> str:
        """최적 디바이스 탐지"""
        if TORCH_AVAILABLE:
            if torch.backends.mps.is_available():
                return "mps"
            elif torch.cuda.is_available():
                return "cuda"
        return "cpu"
    
    def test_optimized_paths(self) -> bool:
        """최적화된 경로 설정 테스트"""
        logger.info("📋 1. 최적화된 경로 설정 테스트")
        
        if not OPTIMIZED_PATHS_AVAILABLE:
            logger.error("❌ app.core.optimized_model_paths 모듈을 import할 수 없습니다")
            logger.info("💡 다음 명령어로 생성하세요: python scripts/checkpoint_analyzer.py")
            return False
        
        try:
            # 분석된 모델 정보 확인
            model_count = len(ANALYZED_MODELS)
            logger.info(f"   ✅ 분석된 모델: {model_count}개")
            
            # 단계별 최적 모델 확인
            steps = [
                "step_01_human_parsing",
                "step_02_pose_estimation", 
                "step_03_cloth_segmentation",
                "step_06_virtual_fitting"
            ]
            
            for step in steps:
                optimal_model = get_optimal_model_for_step(step)
                if optimal_model:
                    logger.info(f"   ✅ {step}: {optimal_model}")
                else:
                    logger.warning(f"   ⚠️ {step}: 최적 모델 없음")
            
            self.test_results["optimized_paths"] = True
            return True
            
        except Exception as e:
            logger.error(f"❌ 최적화된 경로 설정 테스트 실패: {e}")
            self.test_results["optimized_paths"] = False
            return False
    
    def test_checkpoint_files(self) -> bool:
        """체크포인트 파일 존재 확인"""
        logger.info("📋 2. 체크포인트 파일 존재 확인")
        
        if not OPTIMIZED_PATHS_AVAILABLE:
            logger.error("❌ 최적화된 경로 설정이 필요합니다")
            return False
        
        try:
            existing_models = 0
            missing_models = 0
            
            for model_name, model_info in ANALYZED_MODELS.items():
                if not model_info.get('ready', False):
                    continue
                    
                # 모델 경로 확인
                model_path = model_info.get('path')
                if model_path and Path(model_path).exists():
                    # 체크포인트 파일들 확인
                    checkpoints = model_info.get('checkpoints', [])
                    if checkpoints:
                        largest_checkpoint = max(checkpoints, key=lambda x: x.get('size_mb', 0))
                        checkpoint_path = Path(model_path) / largest_checkpoint['path']
                        
                        if checkpoint_path.exists():
                            size_mb = checkpoint_path.stat().st_size / (1024 * 1024)
                            logger.info(f"   ✅ {model_name}: {largest_checkpoint['name']} ({size_mb:.1f}MB)")
                            existing_models += 1
                        else:
                            logger.warning(f"   ❌ {model_name}: {checkpoint_path} 없음")
                            missing_models += 1
                    else:
                        logger.warning(f"   ⚠️ {model_name}: 체크포인트 정보 없음")
                        missing_models += 1
                else:
                    logger.warning(f"   ❌ {model_name}: {model_path} 경로 없음")
                    missing_models += 1
            
            logger.info(f"   📊 존재하는 모델: {existing_models}개, 누락된 모델: {missing_models}개")
            
            self.test_results["checkpoint_files"] = {
                "existing": existing_models,
                "missing": missing_models,
                "success": missing_models == 0
            }
            
            return missing_models == 0
            
        except Exception as e:
            logger.error(f"❌ 체크포인트 파일 확인 실패: {e}")
            self.test_results["checkpoint_files"] = {"success": False, "error": str(e)}
            return False
    
    async def test_model_loader_creation(self) -> bool:
        """모델 로더 생성 테스트"""
        logger.info("📋 3. 모델 로더 생성 테스트")
        
        if not CHECKPOINT_LOADER_AVAILABLE:
            logger.error("❌ CheckpointModelLoader를 import할 수 없습니다")
            logger.info("💡 다음 명령어로 생성하세요: python scripts/checkpoint_analyzer.py")
            return False
        
        try:
            # 모델 로더 생성
            logger.info("   🔧 CheckpointModelLoader 생성 중...")
            model_loader = CheckpointModelLoader(device=self.device)
            
            # 등록된 모델 확인
            registered_models = len(model_loader.models)
            logger.info(f"   ✅ 등록된 모델: {registered_models}개")
            
            # 글로벌 모델 로더 확인
            global_loader = get_checkpoint_model_loader(device=self.device)
            global_models = len(global_loader.models)
            logger.info(f"   ✅ 글로벌 로더 모델: {global_models}개")
            
            self.test_results["model_loader_creation"] = {
                "registered_models": registered_models,
                "global_models": global_models,
                "success": registered_models > 0
            }
            
            return registered_models > 0
            
        except Exception as e:
            logger.error(f"❌ 모델 로더 생성 실패: {e}")
            self.test_results["model_loader_creation"] = {"success": False, "error": str(e)}
            return False
    
    async def test_pytorch_model_loading(self) -> bool:
        """PyTorch 모델 로딩 테스트"""
        logger.info("📋 4. PyTorch 모델 로딩 테스트")
        
        if not TORCH_AVAILABLE:
            logger.error("❌ PyTorch가 설치되지 않았습니다")
            return False
        
        if not CHECKPOINT_LOADER_AVAILABLE:
            logger.error("❌ CheckpointModelLoader를 사용할 수 없습니다")
            return False
        
        try:
            model_loader = get_checkpoint_model_loader(device=self.device)
            
            # 테스트할 모델 선택 (가장 작은 모델부터)
            test_models = [
                "step_04_geometric_matching",
                "step_03_cloth_segmentation",
                "step_01_human_parsing"
            ]
            
            loaded_models = 0
            failed_models = 0
            
            for model_name in test_models:
                try:
                    logger.info(f"   🔧 {model_name} 로딩 시도...")
                    
                    # 메모리 사용량 체크
                    memory_before = psutil.virtual_memory().percent
                    
                    # 모델 로딩 시도
                    model = await model_loader.load_optimal_model_for_step(model_name)
                    
                    if model:
                        memory_after = psutil.virtual_memory().percent
                        memory_used = memory_after - memory_before
                        
                        logger.info(f"   ✅ {model_name} 로딩 성공")
                        logger.info(f"   📊 메모리 사용량: {memory_used:.1f}% 증가")
                        loaded_models += 1
                    else:
                        logger.warning(f"   ❌ {model_name} 로딩 실패 (None 반환)")
                        failed_models += 1
                    
                    # 메모리 정리
                    if TORCH_AVAILABLE and self.device == "mps":
                        torch.mps.empty_cache()
                    elif TORCH_AVAILABLE and self.device == "cuda":
                        torch.cuda.empty_cache()
                    
                except Exception as e:
                    logger.error(f"   ❌ {model_name} 로딩 실패: {e}")
                    failed_models += 1
                
                # 메모리 과부하 방지
                await asyncio.sleep(1)
            
            logger.info(f"   📊 로딩 성공: {loaded_models}개, 실패: {failed_models}개")
            
            self.test_results["pytorch_loading"] = {
                "loaded": loaded_models,
                "failed": failed_models,
                "success": loaded_models > 0
            }
            
            return loaded_models > 0
            
        except Exception as e:
            logger.error(f"❌ PyTorch 모델 로딩 테스트 실패: {e}")
            self.test_results["pytorch_loading"] = {"success": False, "error": str(e)}
            return False
    
    def test_memory_optimization(self) -> bool:
        """메모리 최적화 테스트"""
        logger.info("📋 5. 메모리 최적화 테스트")
        
        try:
            # 메모리 정보 확인
            memory_info = psutil.virtual_memory()
            total_gb = memory_info.total / (1024**3)
            available_gb = memory_info.available / (1024**3)
            used_percent = memory_info.percent
            
            logger.info(f"   📊 총 메모리: {total_gb:.1f}GB")
            logger.info(f"   📊 사용 가능: {available_gb:.1f}GB")
            logger.info(f"   📊 사용률: {used_percent:.1f}%")
            
            # M3 Max 최적화 확인
            is_m3_max = (
                sys.platform == "darwin" and 
                os.uname().machine == "arm64" and
                total_gb > 100  # 128GB 메모리
            )
            
            if is_m3_max:
                logger.info("   ✅ M3 Max 128GB 환경 감지됨")
                
                # MPS 사용 가능 확인
                if TORCH_AVAILABLE and torch.backends.mps.is_available():
                    logger.info("   ✅ MPS (Metal Performance Shaders) 사용 가능")
                    
                    # MPS 최적화 환경 변수 확인
                    mps_env = os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK', '0')
                    logger.info(f"   📊 MPS 폴백: {mps_env}")
                    
                    # Neural Engine 최적화 확인
                    omp_threads = os.environ.get('OMP_NUM_THREADS', '1')
                    logger.info(f"   📊 OpenMP 스레드: {omp_threads}")
                    
                else:
                    logger.warning("   ⚠️ MPS를 사용할 수 없습니다")
            else:
                logger.info("   ℹ️ 일반 시스템 환경")
            
            self.test_results["memory_optimization"] = {
                "total_gb": total_gb,
                "available_gb": available_gb,
                "used_percent": used_percent,
                "is_m3_max": is_m3_max,
                "mps_available": TORCH_AVAILABLE and torch.backends.mps.is_available() if TORCH_AVAILABLE else False,
                "success": available_gb > 10  # 최소 10GB 여유 메모리
            }
            
            return available_gb > 10
            
        except Exception as e:
            logger.error(f"❌ 메모리 최적화 테스트 실패: {e}")
            self.test_results["memory_optimization"] = {"success": False, "error": str(e)}
            return False
    
    def generate_test_report(self) -> Dict[str, Any]:
        """테스트 보고서 생성"""
        report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "device": self.device,
            "torch_available": TORCH_AVAILABLE,
            "test_results": self.test_results,
            "summary": {
                "total_tests": len(self.test_results),
                "passed_tests": sum(1 for result in self.test_results.values() 
                                  if isinstance(result, dict) and result.get('success', False) or result is True),
                "failed_tests": sum(1 for result in self.test_results.values() 
                                  if isinstance(result, dict) and not result.get('success', True) or result is False),
                "overall_success": all(
                    result.get('success', False) if isinstance(result, dict) else result 
                    for result in self.test_results.values()
                )
            }
        }
        
        # 보고서 저장
        report_path = self.backend_dir / "scripts" / "test" / "model_loader_test_report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ 테스트 보고서 저장: {report_path}")
        return report

async def main():
    """메인 테스트 실행"""
    logger.info("🔧 MyCloset AI - 모델 로더 테스트 시작")
    logger.info("=" * 60)
    
    tester = ModelLoaderTester()
    
    # 테스트 실행
    tests = [
        ("최적화된 경로 설정", tester.test_optimized_paths),
        ("체크포인트 파일 존재", tester.test_checkpoint_files),
        ("모델 로더 생성", tester.test_model_loader_creation),
        ("PyTorch 모델 로딩", tester.test_pytorch_model_loading),
        ("메모리 최적화", tester.test_memory_optimization)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            logger.info(f"\n📋 {test_name} 테스트 시작...")
            
            if asyncio.iscoroutinefunction(test_func):
                success = await test_func()
            else:
                success = test_func()
            
            if success:
                logger.info(f"✅ {test_name} 테스트 통과")
                passed += 1
            else:
                logger.error(f"❌ {test_name} 테스트 실패")
                failed += 1
                
        except Exception as e:
            logger.error(f"❌ {test_name} 테스트 예외: {e}")
            failed += 1
    
    # 테스트 보고서 생성
    report = tester.generate_test_report()
    
    # 결과 요약
    logger.info("\n" + "=" * 60)
    logger.info("📊 테스트 결과 요약")
    logger.info("=" * 60)
    logger.info(f"✅ 통과: {passed}개")
    logger.info(f"❌ 실패: {failed}개")
    logger.info(f"📊 성공률: {passed/(passed+failed)*100:.1f}%")
    
    if report['summary']['overall_success']:
        logger.info("🎉 전체 테스트 성공! 모델 로더가 정상 작동합니다.")
        
        logger.info("\n🚀 다음 단계:")
        logger.info("   python app/main.py  # 서버 실행")
        logger.info("   브라우저에서 http://localhost:8000 접속")
        
        return True
    else:
        logger.error("❌ 일부 테스트 실패. 문제를 해결해주세요.")
        
        logger.info("\n💡 해결 방법:")
        if not tester.test_results.get("optimized_paths", True):
            logger.info("   python scripts/checkpoint_analyzer.py  # 체크포인트 분석")
        if not tester.test_results.get("checkpoint_files", {}).get("success", True):
            logger.info("   python scripts/corrected_checkpoint_relocator.py  # 체크포인트 정리")
        
        return False

if __name__ == "__main__":
    import sys
    
    try:
        result = asyncio.run(main())
        sys.exit(0 if result else 1)
    except KeyboardInterrupt:
        logger.info("🛑 사용자에 의해 중단됨")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ 예기치 않은 오류: {e}")
        sys.exit(1)
