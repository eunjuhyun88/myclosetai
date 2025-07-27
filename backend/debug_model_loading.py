#!/usr/bin/env python3
"""
AI 모델 로딩 문제 해결 테스트 - 완전 수정판
backend/debug_model_loading.py

🔥 주요 수정사항:
✅ DeviceManager → memory_manager 변경 반영
✅ MPS 호환성 설정 수정
✅ 모든 import 오류 해결
✅ Step별 초기화 로직 완전 수정
✅ 체크포인트 파일 탐지 개선
✅ 에러 핸들링 강화
✅ M3 Max 최적화 설정 추가
"""

import sys
import os
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Any

# 프로젝트 루트 경로 추가
sys.path.append('.')
sys.path.append('./backend')

def test_model_loading_fixes():
    """모델 로딩 수정사항 테스트 - 완전 수정판"""
    
    print("🔧 AI 모델 로딩 수정사항 테스트 시작...")
    print("=" * 80)
    
    test_results = {
        "mps_setup": False,
        "memory_manager": False,
        "model_loader": False,
        "checkpoint_detection": False,
        "step_initialization": {},
        "total_errors": 0
    }
    
    # 1. MPS 환경 설정 테스트 (수정됨)
    print("\n🍎 1. MPS/Memory Manager 설정 테스트")
    try:
        # memory_manager 사용으로 변경
        from app.ai_pipeline.utils.memory_manager import get_memory_manager
        
        memory_manager = get_memory_manager()
        print("✅ Memory Manager 가져오기 성공")
        
        # MPS 최적화 시도
        if hasattr(memory_manager, 'optimize_for_mps'):
            memory_manager.optimize_for_mps()
            print("✅ MPS 최적화 설정 완료")
            test_results["mps_setup"] = True
        elif hasattr(memory_manager, 'optimize'):
            result = memory_manager.optimize()
            print(f"✅ 메모리 최적화 완료: {result.get('memory_freed_mb', 0)}MB")
            test_results["memory_manager"] = True
        else:
            print("ℹ️ MPS 최적화 메서드 없음 - 기본 설정 사용")
            test_results["mps_setup"] = True
            
    except ImportError as e:
        print(f"⚠️ Memory Manager import 실패: {e}")
        print("ℹ️ 대안: 직접 MPS 설정 시도")
        try:
            import torch
            if torch.backends.mps.is_available():
                os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
                os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
                print("✅ 직접 MPS 환경변수 설정 완료")
                test_results["mps_setup"] = True
        except Exception as mps_e:
            print(f"❌ 직접 MPS 설정도 실패: {mps_e}")
            test_results["total_errors"] += 1
    except Exception as e:
        print(f"❌ Memory Manager 설정 실패: {e}")
        test_results["total_errors"] += 1
    
    # 2. ModelLoader 테스트 (수정됨)
    print("\n🤖 2. ModelLoader 기능 테스트")
    try:
        from app.ai_pipeline.utils.model_loader import get_global_model_loader
        
        model_loader = get_global_model_loader()
        if model_loader:
            print("✅ Global ModelLoader 가져오기 성공")
            test_results["model_loader"] = True
            
            # _find_checkpoint_file 메서드 테스트
            if hasattr(model_loader, '_find_checkpoint_file'):
                print("✅ _find_checkpoint_file 메서드 존재")
                
                # 테스트할 모델들
                test_models = [
                    "cloth_segmentation_u2net",
                    "geometric_matching_model", 
                    "pose_estimation_openpose",
                    "human_parsing_graphonomy",
                    "virtual_fitting_viton"
                ]
                
                found_models = []
                for model_name in test_models:
                    try:
                        result = model_loader._find_checkpoint_file(model_name)
                        if result:
                            found_models.append(model_name)
                            print(f"   ✅ {model_name}: {result}")
                        else:
                            print(f"   ❌ {model_name}: 체크포인트 없음")
                    except Exception as find_e:
                        print(f"   ⚠️ {model_name}: 검색 실패 - {find_e}")
                
                print(f"📊 발견된 모델: {len(found_models)}/{len(test_models)}")
            else:
                print("❌ _find_checkpoint_file 메서드 없음")
        else:
            print("❌ Global ModelLoader 가져오기 실패")
            
    except ImportError as e:
        print(f"❌ ModelLoader import 실패: {e}")
        test_results["total_errors"] += 1
    except Exception as e:
        print(f"❌ ModelLoader 테스트 실패: {e}")
        test_results["total_errors"] += 1
    
    # 3. 체크포인트 파일 탐지 테스트 (개선됨)
    print("\n📁 3. 체크포인트 파일 탐지 테스트")
    try:
        # 여러 가능한 경로 확인
        possible_paths = [
            Path("ai_models"),
            Path("models"),
            Path("backend/ai_models"),
            Path("../ai_models"),
            Path("./ai_models")
        ]
        
        total_checkpoints = 0
        total_size_gb = 0
        large_files = []
        
        for ai_models_path in possible_paths:
            if ai_models_path.exists():
                print(f"✅ 모델 디렉토리 발견: {ai_models_path}")
                
                # 체크포인트 파일 확장자들
                checkpoint_extensions = ["*.pth", "*.safetensors", "*.bin", "*.pt", "*.ckpt"]
                checkpoint_files = []
                
                for ext in checkpoint_extensions:
                    checkpoint_files.extend(list(ai_models_path.rglob(ext)))
                
                if checkpoint_files:
                    total_checkpoints += len(checkpoint_files)
                    print(f"   📦 체크포인트 파일: {len(checkpoint_files)}개")
                    
                    # 파일 크기 분석
                    for file in checkpoint_files:
                        try:
                            size_bytes = file.stat().st_size
                            size_gb = size_bytes / (1024*1024*1024)
                            total_size_gb += size_gb
                            
                            if size_gb >= 0.1:  # 100MB 이상
                                large_files.append((file.name, size_gb))
                        except Exception:
                            continue
                
                break  # 첫 번째 유효한 경로에서 중단
        
        if total_checkpoints > 0:
            print(f"✅ 총 체크포인트 파일: {total_checkpoints}개")
            print(f"📊 총 크기: {total_size_gb:.2f}GB")
            test_results["checkpoint_detection"] = True
            
            # 큰 파일들 표시 (상위 10개)
            if large_files:
                large_files.sort(key=lambda x: x[1], reverse=True)
                print(f"🔥 주요 모델 파일 (상위 {min(10, len(large_files))}개):")
                for name, size in large_files[:10]:
                    print(f"   {name}: {size:.1f}GB")
        else:
            print("❌ 체크포인트 파일을 찾을 수 없음")
            print("   💡 ai_models 디렉토리가 올바른 위치에 있는지 확인하세요")
            
    except Exception as e:
        print(f"❌ 체크포인트 탐지 실패: {e}")
        test_results["total_errors"] += 1
    
    # 4. Step별 AI 모델 초기화 테스트 (완전 수정됨)
    print("\n🚀 4. Step별 AI 모델 초기화 테스트")
    
    test_steps = [
        {
            "name": "HumanParsingStep",
            "id": 1,
            "module": "app.ai_pipeline.steps.step_01_human_parsing",
            "class": "HumanParsingStep"
        },
        {
            "name": "PoseEstimationStep", 
            "id": 2,
            "module": "app.ai_pipeline.steps.step_02_pose_estimation",
            "class": "PoseEstimationStep"
        },
        {
            "name": "GeometricMatchingStep",
            "id": 4, 
            "module": "app.ai_pipeline.steps.step_04_geometric_matching",
            "class": "GeometricMatchingStep"
        },
        {
            "name": "VirtualFittingStep",
            "id": 6,
            "module": "app.ai_pipeline.steps.step_06_virtual_fitting", 
            "class": "VirtualFittingStep"
        }
    ]
    
    for step_info in test_steps:
        step_name = step_info["name"]
        print(f"\n   🔧 {step_name} 테스트 중...")
        
        try:
            # 동적 import
            module = __import__(step_info["module"], fromlist=[step_info["class"]])
            step_class = getattr(module, step_info["class"])
            
            # Step 인스턴스 생성 (device 파라미터 자동 처리)
            try:
                step = step_class(device='mps')
            except TypeError:
                # device 파라미터를 지원하지 않는 경우
                step = step_class()
            
            print(f"      ✅ {step_name} 인스턴스 생성 성공")
            
            # 초기화 테스트
            if hasattr(step, 'initialize'):
                try:
                    # async 메서드인지 확인
                    import asyncio
                    import inspect
                    
                    if inspect.iscoroutinefunction(step.initialize):
                        # async 메서드
                        try:
                            loop = asyncio.get_event_loop()
                        except RuntimeError:
                            loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(loop)
                        
                        init_result = loop.run_until_complete(step.initialize())
                    else:
                        # sync 메서드
                        init_result = step.initialize()
                    
                    if init_result:
                        print(f"      ✅ {step_name} 초기화 성공")
                        test_results["step_initialization"][step_name] = "success"
                    else:
                        print(f"      ⚠️ {step_name} 초기화 실패 (False 반환)")
                        test_results["step_initialization"][step_name] = "failed"
                        
                except Exception as init_e:
                    print(f"      ⚠️ {step_name} 초기화 오류: {init_e}")
                    test_results["step_initialization"][step_name] = f"error: {str(init_e)[:50]}"
            else:
                print(f"      ℹ️ {step_name} initialize 메서드 없음")
                test_results["step_initialization"][step_name] = "no_initialize_method"
            
        except ImportError as import_e:
            print(f"      ❌ {step_name} import 실패: {import_e}")
            test_results["step_initialization"][step_name] = f"import_error: {str(import_e)[:50]}"
            test_results["total_errors"] += 1
            
        except Exception as e:
            print(f"      ❌ {step_name} 테스트 실패: {e}")
            test_results["step_initialization"][step_name] = f"error: {str(e)[:50]}"
            test_results["total_errors"] += 1
    
    # 5. 추가 시스템 정보
    print("\n📊 5. 시스템 정보")
    try:
        import platform
        print(f"   OS: {platform.system()} {platform.release()}")
        print(f"   Python: {sys.version.split()[0]}")
        print(f"   Architecture: {platform.machine()}")
        
        # PyTorch 정보
        try:
            import torch
            print(f"   PyTorch: {torch.__version__}")
            print(f"   CUDA 사용 가능: {torch.cuda.is_available()}")
            print(f"   MPS 사용 가능: {torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False}")
        except ImportError:
            print("   PyTorch: 설치되지 않음")
        
        # 메모리 정보
        try:
            import psutil
            memory = psutil.virtual_memory()
            print(f"   총 메모리: {memory.total / (1024**3):.1f}GB")
            print(f"   사용 가능: {memory.available / (1024**3):.1f}GB")
        except ImportError:
            print("   메모리 정보: psutil 없음")
            
    except Exception as e:
        print(f"   시스템 정보 수집 실패: {e}")
    
    # 6. 결과 요약
    print("\n" + "=" * 80)
    print("🎉 테스트 결과 요약")
    print("=" * 80)
    
    success_count = sum([
        test_results["mps_setup"] or test_results["memory_manager"],
        test_results["model_loader"],
        test_results["checkpoint_detection"],
        len([v for v in test_results["step_initialization"].values() if v == "success"])
    ])
    
    total_tests = 3 + len(test_steps)
    
    print(f"✅ 성공한 테스트: {success_count}")
    print(f"❌ 실패한 테스트: {test_results['total_errors']}")
    print(f"📊 Step 초기화 성공: {len([v for v in test_results['step_initialization'].values() if v == 'success'])}/{len(test_steps)}")
    
    if test_results["total_errors"] == 0:
        print("\n🎊 모든 테스트 통과! AI 모델 로딩 시스템이 정상 작동합니다.")
    elif test_results["total_errors"] <= 2:
        print("\n⚠️ 일부 문제가 있지만 기본 기능은 작동합니다.")
    else:
        print("\n🚨 심각한 문제가 발견되었습니다. 설정을 확인해주세요.")
    
    # 7. 개선 제안사항
    print("\n💡 개선 제안사항:")
    
    suggestions = []
    
    if not (test_results["mps_setup"] or test_results["memory_manager"]):
        suggestions.append("- Memory Manager 설정 확인 필요")
    
    if not test_results["model_loader"]:
        suggestions.append("- ModelLoader 의존성 확인 필요")
    
    if not test_results["checkpoint_detection"]:
        suggestions.append("- ai_models 디렉토리 위치 및 파일 확인 필요")
    
    failed_steps = [k for k, v in test_results["step_initialization"].items() if v != "success"]
    if failed_steps:
        suggestions.append(f"- 다음 Step들의 의존성 확인 필요: {', '.join(failed_steps)}")
    
    if not suggestions:
        suggestions.append("- 모든 시스템이 정상 작동 중입니다! 🎉")
    
    for suggestion in suggestions:
        print(suggestion)
    
    print("\n🔧 테스트 완료!")
    return test_results

def run_quick_diagnostics():
    """빠른 진단 테스트"""
    print("⚡ 빠른 진단 실행 중...")
    
    diagnostics = {
        "python_path": sys.path[:3],
        "working_directory": os.getcwd(),
        "environment": dict(os.environ).keys(),
        "imports": {}
    }
    
    # 핵심 import 테스트
    critical_imports = [
        "torch",
        "app.ai_pipeline.utils.memory_manager",
        "app.ai_pipeline.utils.model_loader",
        "app.core.config"
    ]
    
    for module_name in critical_imports:
        try:
            __import__(module_name)
            diagnostics["imports"][module_name] = "✅ 성공"
        except ImportError as e:
            diagnostics["imports"][module_name] = f"❌ 실패: {e}"
        except Exception as e:
            diagnostics["imports"][module_name] = f"⚠️ 오류: {e}"
    
    print("📋 진단 결과:")
    for module, status in diagnostics["imports"].items():
        print(f"   {module}: {status}")
    
    return diagnostics

if __name__ == "__main__":
    try:
        # 빠른 진단 먼저 실행
        print("🔍 사전 진단 실행...")
        quick_results = run_quick_diagnostics()
        
        print("\n" + "="*50)
        
        # 메인 테스트 실행
        test_results = test_model_loading_fixes()
        
        # JSON 형태로 결과 저장 (선택사항)
        try:
            import json
            results_file = Path("debug_results.json")
            
            combined_results = {
                "timestamp": time.time(),
                "quick_diagnostics": quick_results,
                "main_tests": test_results
            }
            
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(combined_results, f, indent=2, ensure_ascii=False)
            
            print(f"\n📄 결과가 {results_file}에 저장되었습니다.")
            
        except Exception as save_e:
            print(f"\n⚠️ 결과 저장 실패: {save_e}")
        
    except KeyboardInterrupt:
        print("\n⚠️ 사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"\n❌ 테스트 실행 중 예외 발생: {e}")
        print(f"스택 트레이스:\n{traceback.format_exc()}")
    finally:
        print("\n👋 디버그 스크립트 종료")