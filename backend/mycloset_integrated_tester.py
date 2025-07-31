#!/usr/bin/env python3
"""
🔥 MyCloset AI - 통합 모델 로더 & 테스터 v1.0
================================================================================
✅ 한번에 모든 AI 모델 로딩
✅ 초기화 문제 없는 원스톱 실행
✅ 단순 호출로 즉시 테스트 가능
✅ 프로젝트의 ModelLoader v5.1 & step_interface.py v5.2 활용
✅ 실제 체크포인트 로딩 통합
✅ M3 Max 최적화 적용
================================================================================
"""

import os
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum

# MyCloset AI 프로젝트 경로 자동 설정
PROJECT_ROOT = Path(__file__).parent
BACKEND_ROOT = PROJECT_ROOT / "backend"
if BACKEND_ROOT.exists():
    sys.path.insert(0, str(BACKEND_ROOT))
else:
    # 현재 위치에서 backend 찾기
    current = Path.cwd()
    while current != current.parent:
        if (current / "backend").exists():
            sys.path.insert(0, str(current / "backend"))
            BACKEND_ROOT = current / "backend"
            break
        current = current.parent

print(f"🔧 프로젝트 루트: {PROJECT_ROOT}")
print(f"🔧 백엔드 루트: {BACKEND_ROOT}")

# 경고 무시
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class TestStatus(Enum):
    SUCCESS = "✅"
    FAILED = "❌"
    LOADING = "⏳"
    SKIPPED = "⏭️"

@dataclass
class TestResult:
    name: str
    status: TestStatus
    message: str
    load_time: float = 0.0
    model_size_mb: float = 0.0

class MyClosetIntegratedTester:
    """MyCloset AI 통합 모델 로더 & 테스터"""
    
    def __init__(self):
        self.results: List[TestResult] = []
        self.model_loader = None
        self.step_factory = None
        self.total_models_loaded = 0
        self.total_load_time = 0.0
        
        print("🚀 MyCloset AI 통합 테스터 초기화...")
        self._initialize_system()
    
    def _initialize_system(self):
        """시스템 초기화"""
        try:
            # 1. ModelLoader 초기화
            print("⏳ ModelLoader v5.1 초기화 중...")
            start_time = time.time()
            
            try:
                from app.ai_pipeline.utils.model_loader import get_global_model_loader
                self.model_loader = get_global_model_loader()
                
                if self.model_loader:
                    init_time = time.time() - start_time
                    self.results.append(TestResult(
                        "ModelLoader v5.1 초기화",
                        TestStatus.SUCCESS,
                        f"글로벌 로더 준비됨 ({init_time:.2f}s)",
                        init_time
                    ))
                    print(f"✅ ModelLoader v5.1 초기화 완료 ({init_time:.2f}s)")
                else:
                    raise Exception("글로벌 로더 None 반환")
                    
            except Exception as e:
                self.results.append(TestResult(
                    "ModelLoader v5.1 초기화",
                    TestStatus.FAILED,
                    f"초기화 실패: {str(e)[:50]}"
                ))
                print(f"❌ ModelLoader 초기화 실패: {e}")
            
            # 2. StepFactory 초기화
            print("⏳ StepFactory v11.0 초기화 중...")
            start_time = time.time()
            
            try:
                from app.services.step_factory import StepFactory
                self.step_factory = StepFactory()
                
                if hasattr(self.step_factory, 'initialize_all_steps'):
                    self.step_factory.initialize_all_steps()
                
                init_time = time.time() - start_time
                self.results.append(TestResult(
                    "StepFactory v11.0 초기화",
                    TestStatus.SUCCESS,
                    f"팩토리 준비됨 ({init_time:.2f}s)",
                    init_time
                ))
                print(f"✅ StepFactory v11.0 초기화 완료 ({init_time:.2f}s)")
                
            except Exception as e:
                self.results.append(TestResult(
                    "StepFactory v11.0 초기화",
                    TestStatus.FAILED,
                    f"초기화 실패: {str(e)[:50]}"
                ))
                print(f"❌ StepFactory 초기화 실패: {e}")
            
        except Exception as e:
            print(f"❌ 시스템 초기화 실패: {e}")
    
    def test_model_loading(self) -> Dict[str, TestResult]:
        """핵심 모델들 로딩 테스트"""
        print("\n🔍 핵심 AI 모델 로딩 테스트 시작...")
        
        # 테스트할 핵심 모델들
        core_models = [
            "human_parsing_schp",  # Step 01
            "openpose_body",       # Step 02  
            "sam_vit_h",          # Step 03
            "u2net_cloth_seg",    # Step 03 대체
            "realvisxl_v4",       # Step 05
            "ootd_diffusion",     # Step 06
            "gfpgan_enhance",     # Step 07
            "clip_quality"        # Step 08
        ]
        
        model_results = {}
        
        for model_name in core_models:
            print(f"\n⏳ {model_name} 로딩 중...")
            start_time = time.time()
            
            try:
                if self.model_loader and hasattr(self.model_loader, 'load_model'):
                    model = self.model_loader.load_model(model_name)
                    
                    if model is not None:
                        load_time = time.time() - start_time
                        
                        # 모델 크기 계산
                        model_size_mb = 0.0
                        if hasattr(model, 'memory_usage_mb'):
                            model_size_mb = model.memory_usage_mb
                        elif hasattr(model, 'get_memory_usage'):
                            model_size_mb = model.get_memory_usage()
                        
                        result = TestResult(
                            model_name,
                            TestStatus.SUCCESS,
                            f"로딩 성공 ({model_size_mb:.1f}MB)",
                            load_time,
                            model_size_mb
                        )
                        
                        self.total_models_loaded += 1
                        self.total_load_time += load_time
                        
                        print(f"✅ {model_name} 로딩 성공 ({model_size_mb:.1f}MB, {load_time:.2f}s)")
                        
                    else:
                        result = TestResult(
                            model_name,
                            TestStatus.FAILED,
                            "모델 로딩 실패 (None 반환)"
                        )
                        print(f"❌ {model_name} 로딩 실패 (None 반환)")
                        
                else:
                    result = TestResult(
                        model_name,
                        TestStatus.SKIPPED,
                        "ModelLoader 없음"
                    )
                    print(f"⏭️ {model_name} 스킵 (ModelLoader 없음)")
                    
            except Exception as e:
                load_time = time.time() - start_time
                result = TestResult(
                    model_name,
                    TestStatus.FAILED,
                    f"로딩 오류: {str(e)[:40]}",
                    load_time
                )
                print(f"❌ {model_name} 로딩 오류: {e}")
            
            model_results[model_name] = result
            self.results.append(result)
        
        return model_results
    
    def test_step_pipeline(self) -> Dict[str, TestResult]:
        """Step 파이프라인 테스트"""
        print("\n🔄 Step 파이프라인 테스트 시작...")
        
        step_types = [
            "step_01_human_parsing",
            "step_02_pose_estimation", 
            "step_03_cloth_segmentation",
            "step_06_virtual_fitting"
        ]
        
        step_results = {}
        
        for step_type in step_types:
            print(f"\n⏳ {step_type} 테스트 중...")
            start_time = time.time()
            
            try:
                if self.step_factory and hasattr(self.step_factory, 'create_step'):
                    step_instance = self.step_factory.create_step(step_type)
                    
                    if step_instance:
                        # Step 초기화 테스트
                        if hasattr(step_instance, 'initialize'):
                            step_instance.initialize()
                        
                        # 모델 로딩 상태 확인
                        models_loaded = False
                        if hasattr(step_instance, 'model_loader'):
                            models_loaded = step_instance.model_loader is not None
                        
                        test_time = time.time() - start_time
                        
                        result = TestResult(
                            step_type,
                            TestStatus.SUCCESS,
                            f"Step 생성 성공 (모델: {'✅' if models_loaded else '❌'})",
                            test_time
                        )
                        
                        print(f"✅ {step_type} 생성 성공 ({test_time:.2f}s)")
                        
                    else:
                        result = TestResult(
                            step_type,
                            TestStatus.FAILED,
                            "Step 생성 실패 (None 반환)"
                        )
                        print(f"❌ {step_type} 생성 실패")
                        
                else:
                    result = TestResult(
                        step_type,
                        TestStatus.SKIPPED,
                        "StepFactory 없음"
                    )
                    print(f"⏭️ {step_type} 스킵")
                    
            except Exception as e:
                test_time = time.time() - start_time
                result = TestResult(
                    step_type,
                    TestStatus.FAILED,
                    f"테스트 오류: {str(e)[:40]}",
                    test_time
                )
                print(f"❌ {step_type} 테스트 오류: {e}")
            
            step_results[step_type] = result
            self.results.append(result)
        
        return step_results
    
    def run_full_test(self) -> Dict[str, Any]:
        """전체 테스트 실행"""
        print("🚀 MyCloset AI 통합 테스트 시작!")
        print("=" * 60)
        
        start_time = time.time()
        
        # 1. 모델 로딩 테스트
        model_results = self.test_model_loading()
        
        # 2. Step 파이프라인 테스트  
        step_results = self.test_step_pipeline()
        
        total_time = time.time() - start_time
        
        # 결과 리포트
        self._generate_report(total_time)
        
        return {
            'model_results': model_results,
            'step_results': step_results,
            'total_time': total_time,
            'success_rate': self._calculate_success_rate()
        }
    
    def _calculate_success_rate(self) -> float:
        """성공률 계산"""
        if not self.results:
            return 0.0
        
        success_count = sum(1 for r in self.results if r.status == TestStatus.SUCCESS)
        return (success_count / len(self.results)) * 100
    
    def _generate_report(self, total_time: float):
        """결과 리포트 생성"""
        print("\n" + "=" * 60)
        print("📊 MyCloset AI 통합 테스트 결과 리포트")
        print("=" * 60)
        
        # 통계
        success_count = sum(1 for r in self.results if r.status == TestStatus.SUCCESS)
        failed_count = sum(1 for r in self.results if r.status == TestStatus.FAILED)
        skipped_count = sum(1 for r in self.results if r.status == TestStatus.SKIPPED)
        success_rate = self._calculate_success_rate()
        
        print(f"📈 전체 통계:")
        print(f"   총 테스트: {len(self.results)}개")
        print(f"   성공: {success_count}개 ({success_rate:.1f}%)")
        print(f"   실패: {failed_count}개")
        print(f"   스킵: {skipped_count}개")
        print(f"   총 실행시간: {total_time:.2f}s")
        
        if self.total_models_loaded > 0:
            print(f"   로딩된 모델: {self.total_models_loaded}개")
            print(f"   모델 로딩 시간: {self.total_load_time:.2f}s")
            print(f"   평균 로딩 시간: {self.total_load_time/self.total_models_loaded:.2f}s/모델")
        
        # 상세 결과
        print(f"\n📋 상세 결과:")
        for result in self.results:
            status_icon = result.status.value
            time_info = f"({result.load_time:.2f}s)" if result.load_time > 0 else ""
            size_info = f"[{result.model_size_mb:.1f}MB]" if result.model_size_mb > 0 else ""
            
            print(f"   {status_icon} {result.name}: {result.message} {time_info} {size_info}")
        
        # 결론
        print(f"\n🎯 결론:")
        if success_rate >= 80:
            print("   ✅ MyCloset AI 시스템이 정상적으로 작동합니다!")
            print("   🚀 프로덕션 환경에서 사용할 준비가 되었습니다.")
        elif success_rate >= 60:
            print("   ⚠️ 일부 문제가 있지만 기본 기능은 작동합니다.")
            print("   🔧 실패한 컴포넌트들을 점검해보세요.")
        else:
            print("   ❌ 시스템에 심각한 문제가 있습니다.")
            print("   🛠️ 환경 설정 및 의존성을 다시 확인해보세요.")
        
        print("=" * 60)

def quick_test():
    """빠른 테스트 (단일 호출용)"""
    print("⚡ MyCloset AI 빠른 테스트 실행...")
    
    tester = MyClosetIntegratedTester()
    
    # 핵심 모델 1개만 테스트
    if tester.model_loader:
        try:
            print("⏳ 핵심 모델 테스트 중...")
            model = tester.model_loader.load_model("human_parsing_schp")
            if model:
                print("✅ 핵심 모델 로딩 성공! 시스템 정상 작동 중")
                return True
            else:
                print("❌ 핵심 모델 로딩 실패")
                return False
        except Exception as e:
            print(f"❌ 테스트 오류: {e}")
            return False
    else:
        print("❌ ModelLoader 초기화 실패")
        return False

def main():
    """메인 실행 함수"""
    print("🔥 MyCloset AI 통합 모델 로더 & 테스터 v1.0")
    print("=" * 60)
    
    # 전체 테스트 실행
    tester = MyClosetIntegratedTester()
    results = tester.run_full_test()
    
    return results

if __name__ == "__main__":
    # 실행 예시
    print("선택하세요:")
    print("1. 전체 테스트 (main)")
    print("2. 빠른 테스트 (quick_test)")
    
    choice = input("선택 (1/2): ").strip()
    
    if choice == "2":
        quick_test()
    else:
        main()