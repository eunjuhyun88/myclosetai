#!/usr/bin/env python3
"""
🔍 MyCloset AI - 모델 로딩 체크 스크립트
실제 모델이 제대로 로딩되는지 빠르게 테스트
"""

import sys
import os
from pathlib import Path
import logging
import time

# 프로젝트 루트를 Python 경로에 추가
current_dir = Path(__file__).parent
backend_dir = current_dir.parent.parent.parent  # backend/
sys.path.insert(0, str(backend_dir))

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

def check_model_loading():
    """모델 로딩 체크"""
    print("🔍 MyCloset AI 모델 로딩 체크 시작")
    print("=" * 60)
    
    try:
        # ModelLoader 임포트 및 초기화
        print("📦 ModelLoader 임포트 중...")
        from app.ai_pipeline.utils.model_loader import get_global_model_loader, initialize_global_model_loader
        
        print("🔄 ModelLoader 초기화 중...")
        success = initialize_global_model_loader()
        
        if not success:
            print("❌ ModelLoader 초기화 실패")
            return False
        
        loader = get_global_model_loader()
        print(f"✅ ModelLoader 초기화 성공")
        print(f"   📁 모델 디렉토리: {loader.model_cache_dir}")
        print(f"   🔧 디바이스: {loader.device}")
        print(f"   🧠 메모리: {loader.memory_gb:.1f}GB")
        print(f"   🎯 최소 모델 크기: {loader.min_model_size_mb}MB")
        
        # 1. 사용 가능한 모델 목록 확인
        print("\n📋 사용 가능한 모델 목록:")
        available_models = loader.list_available_models()
        
        if not available_models:
            print("⚠️ 사용 가능한 모델이 없습니다")
            return False
        
        # 크기별 분류
        gb_models = [m for m in available_models if m["size_mb"] > 1000]
        large_models = [m for m in available_models if 500 <= m["size_mb"] <= 1000] 
        medium_models = [m for m in available_models if 100 <= m["size_mb"] < 500]
        small_models = [m for m in available_models if 50 <= m["size_mb"] < 100]
        
        print(f"   🔥 GB급 모델(1GB+): {len(gb_models)}개")
        print(f"   📦 대형 모델(500MB-1GB): {len(large_models)}개")
        print(f"   📁 중형 모델(100-500MB): {len(medium_models)}개")
        print(f"   📄 소형 모델(50-100MB): {len(small_models)}개")
        print(f"   ✅ 총 모델: {len(available_models)}개")
        
        # 상위 5개 모델 출력
        print(f"\n🏆 상위 5개 모델 (크기순):")
        for i, model in enumerate(available_models[:5]):
            size_gb = model["size_mb"] / 1024 if model["size_mb"] > 1000 else None
            size_str = f"{size_gb:.1f}GB" if size_gb else f"{model['size_mb']:.0f}MB"
            status = "🔥" if model["size_mb"] > 1000 else "📦" if model["size_mb"] > 500 else "📁"
            print(f"   {i+1}. {status} {model['name']}: {size_str} ({model['model_type']})")
        
        # 2. 성능 메트릭 확인
        print(f"\n📊 성능 메트릭:")
        metrics = loader.get_performance_metrics()
        
        if "error" not in metrics:
            print(f"   📝 등록된 모델: {metrics['model_counts']['registered']}개")
            print(f"   📋 사용 가능: {metrics['model_counts']['available']}개")
            print(f"   🔄 로드된 모델: {metrics['model_counts']['loaded']}개")
            print(f"   💾 총 메모리: {metrics['memory_usage']['total_mb']:.1f}MB")
            print(f"   🎯 검증률: {metrics['performance_stats']['validation_rate']:.1%}")
            print(f"   🚀 캐시 히트율: {metrics['performance_stats']['cache_hit_rate']:.1%}")
        
        # 3. 실제 모델 로딩 테스트 (상위 3개)
        print(f"\n🧪 실제 모델 로딩 테스트:")
        test_models = available_models[:3]  # 상위 3개만 테스트
        
        loading_results = []
        for i, model in enumerate(test_models):
            model_name = model["name"]
            print(f"   {i+1}. 테스트 중: {model_name} ({model['size_mb']:.0f}MB)...")
            
            start_time = time.time()
            try:
                checkpoint = loader.load_model(model_name)
                load_time = time.time() - start_time
                
                if checkpoint is not None:
                    # 체크포인트 타입 확인
                    checkpoint_type = type(checkpoint).__name__
                    is_dict = isinstance(checkpoint, dict)
                    param_count = len(checkpoint) if is_dict else "N/A"
                    
                    print(f"      ✅ 성공: {load_time:.1f}초, 타입: {checkpoint_type}, 파라미터: {param_count}")
                    loading_results.append((model_name, True, load_time, checkpoint_type))
                else:
                    print(f"      ❌ 실패: None 반환")
                    loading_results.append((model_name, False, load_time, "None"))
                    
            except Exception as e:
                load_time = time.time() - start_time
                print(f"      ❌ 오류: {str(e)[:50]}...")
                loading_results.append((model_name, False, load_time, "Error"))
        
        # 4. 결과 요약
        print(f"\n📋 테스트 결과 요약:")
        successful_loads = [r for r in loading_results if r[1]]
        failed_loads = [r for r in loading_results if not r[1]]
        
        print(f"   ✅ 성공: {len(successful_loads)}/{len(loading_results)}개")
        print(f"   ❌ 실패: {len(failed_loads)}/{len(loading_results)}개")
        
        if successful_loads:
            avg_load_time = sum(r[2] for r in successful_loads) / len(successful_loads)
            print(f"   ⏱️ 평균 로딩 시간: {avg_load_time:.1f}초")
        
        # 5. Step별 인터페이스 테스트
        print(f"\n🔗 Step 인터페이스 테스트:")
        step_names = [
            "HumanParsingStep",
            "PoseEstimationStep", 
            "ClothSegmentationStep",
            "VirtualFittingStep"
        ]
        
        interface_results = []
        for step_name in step_names:
            try:
                interface = loader.create_step_interface(step_name)
                available_models_for_step = interface.list_available_models()
                
                print(f"   ✅ {step_name}: {len(available_models_for_step)}개 모델")
                interface_results.append((step_name, True, len(available_models_for_step)))
            except Exception as e:
                print(f"   ❌ {step_name}: 오류 - {str(e)[:30]}...")
                interface_results.append((step_name, False, 0))
        
        # 최종 판정
        success_rate = len(successful_loads) / len(loading_results) if loading_results else 0
        interface_success_rate = len([r for r in interface_results if r[1]]) / len(interface_results)
        
        print(f"\n🎯 최종 판정:")
        print(f"   📦 모델 로딩 성공률: {success_rate:.1%}")
        print(f"   🔗 인터페이스 성공률: {interface_success_rate:.1%}")
        
        overall_success = success_rate >= 0.5 and interface_success_rate >= 0.75
        
        if overall_success:
            print(f"   🎉 전체 평가: 성공! 모델 로딩이 정상 작동합니다.")
            return True
        else:
            print(f"   ⚠️ 전체 평가: 부분 실패. 일부 문제가 있습니다.")
            return False
            
    except ImportError as e:
        print(f"❌ 임포트 오류: {e}")
        print("   💡 conda 환경이 활성화되어 있는지 확인하세요:")
        print("   conda activate mycloset-ai-clean")
        return False
        
    except Exception as e:
        print(f"❌ 예상치 못한 오류: {e}")
        import traceback
        print("📋 오류 스택:")
        traceback.print_exc()
        return False

def quick_model_count():
    """빠른 모델 파일 개수 체크"""
    print("🔍 빠른 모델 파일 개수 체크")
    print("=" * 40)
    
    # 기본 AI 모델 경로들
    potential_paths = [
        Path("/Users/gimdudeul/MVP/mycloset-ai/backend/ai_models"),
        Path("./ai_models"),
        Path("../ai_models"),
        Path("backend/ai_models")
    ]
    
    ai_models_path = None
    for path in potential_paths:
        if path.exists():
            ai_models_path = path
            break
    
    if not ai_models_path:
        print("❌ AI 모델 디렉토리를 찾을 수 없습니다")
        return
    
    print(f"📁 검색 경로: {ai_models_path}")
    
    # 모델 파일 확장자
    extensions = [".pth", ".pt", ".bin", ".safetensors", ".ckpt"]
    
    total_files = 0
    total_size_gb = 0
    large_files = []
    
    for ext in extensions:
        files = list(ai_models_path.rglob(f"*{ext}"))
        for file_path in files:
            if file_path.is_file():
                try:
                    size_mb = file_path.stat().st_size / (1024 * 1024)
                    if size_mb >= 50:  # 50MB 이상만
                        total_files += 1
                        total_size_gb += size_mb / 1024
                        
                        if size_mb > 500:  # 500MB 이상은 대형 파일
                            large_files.append((file_path.name, size_mb))
                except:
                    continue
    
    print(f"📊 총 모델 파일: {total_files}개 (50MB 이상)")
    print(f"💾 총 크기: {total_size_gb:.1f}GB")
    print(f"🔥 대형 파일: {len(large_files)}개 (500MB+)")
    
    # 상위 5개 대형 파일
    if large_files:
        large_files.sort(key=lambda x: x[1], reverse=True)
        print(f"\n🏆 상위 5개 대형 파일:")
        for i, (name, size_mb) in enumerate(large_files[:5]):
            size_gb = size_mb / 1024 if size_mb > 1000 else None
            size_str = f"{size_gb:.1f}GB" if size_gb else f"{size_mb:.0f}MB"
            print(f"   {i+1}. {name}: {size_str}")

if __name__ == "__main__":
    print("🚀 MyCloset AI 모델 체크 스크립트")
    print("=" * 60)
    
    # 인자에 따라 다른 체크 실행
    if len(sys.argv) > 1:
        if sys.argv[1] == "quick":
            quick_model_count()
        elif sys.argv[1] == "full":
            check_model_loading()
        else:
            print("❓ 사용법:")
            print("   python check_models.py quick  # 빠른 파일 개수 체크")
            print("   python check_models.py full   # 전체 모델 로딩 테스트")
    else:
        # 기본: 전체 체크
        success = check_model_loading()
        sys.exit(0 if success else 1)