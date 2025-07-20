#!/usr/bin/env python3
"""
🔍 Auto Detector 디버깅 스크립트
왜 923개 모델이 있는데 못 찾는지 정확한 원인 파악
"""

import os
import sys
from pathlib import Path

# 프로젝트 경로 설정 (현재 위치 고려)
current_dir = Path.cwd()
if current_dir.name == "backend":
    # backend 디렉토리 안에서 실행한 경우
    project_root = current_dir.parent
    sys.path.insert(0, str(current_dir))
else:
    # 프로젝트 루트에서 실행한 경우
    project_root = current_dir
    sys.path.insert(0, str(project_root / "backend"))

print(f"📁 현재 위치: {current_dir}")
print(f"📁 프로젝트 루트: {project_root}")

def debug_search_paths():
    """Auto Detector 검색 경로 디버깅"""
    
    print("🔍 Auto Detector 검색 경로 디버깅")
    print("=" * 50)
    
    # 예상 검색 경로들
    backend_dir = project_root / "backend"
    
    expected_paths = [
        backend_dir / "ai_models",
        backend_dir / "ai_models" / "checkpoints", 
        backend_dir / "app" / "ai_pipeline" / "models",
        backend_dir / "app" / "models",
        backend_dir / "checkpoints",
        backend_dir / "models",
        backend_dir / "weights",
        backend_dir.parent / "ai_models",
        Path.home() / ".cache" / "huggingface" / "hub",
    ]
    
    print("📂 Auto Detector가 검색할 경로들:")
    for i, path in enumerate(expected_paths, 1):
        exists = "✅" if path.exists() else "❌"
        if path.exists():
            file_count = len(list(path.rglob("*.pth")))
            print(f"   {i:2d}. {exists} {path} ({file_count}개 .pth 파일)")
        else:
            print(f"   {i:2d}. {exists} {path} (경로 없음)")
    
    return expected_paths

def debug_actual_auto_detector():
    """실제 Auto Detector 실행해보기"""
    
    print("\n🤖 실제 Auto Detector 실행 테스트")
    print("=" * 50)
    
    try:
        # Auto Detector import
        from backend.app.ai_pipeline.utils.auto_model_detector import RealWorldModelDetector
        
        print("✅ Auto Detector import 성공")
        
        # 디버그 모드로 실행
        detector = RealWorldModelDetector(
            enable_deep_scan=True,
            enable_pytorch_validation=False,  # 빠른 테스트를 위해 비활성화
            max_workers=1
        )
        
        print(f"✅ Auto Detector 초기화 성공")
        print(f"   검색 경로 수: {len(detector.search_paths)}")
        
        # 검색 경로 출력
        print("\n🔍 실제 검색 경로:")
        for i, path in enumerate(detector.search_paths, 1):
            exists = "✅" if path.exists() else "❌"
            if path.exists():
                file_count = len(list(path.rglob("*.pth")))
                print(f"   {i:2d}. {exists} {path} ({file_count}개 .pth)")
            else:
                print(f"   {i:2d}. {exists} {path} (없음)")
        
        # 실제 탐지 실행
        print("\n🔄 모델 탐지 실행 중...")
        detected_models = detector.detect_all_models(
            force_rescan=True,
            min_confidence=0.1,  # 낮은 임계값
            enable_detailed_analysis=False
        )
        
        if detected_models:
            print(f"🎉 탐지 성공: {len(detected_models)}개 모델 발견!")
            
            # 상위 10개 모델 출력
            print("\n📋 탐지된 모델들 (상위 10개):")
            for i, (name, model) in enumerate(list(detected_models.items())[:10], 1):
                print(f"   {i:2d}. {name}")
                print(f"       📁 {model.path}")
                print(f"       📊 {model.file_size_mb:.1f}MB")
                print(f"       🎯 {model.step_name}")
                print("")
        else:
            print("❌ 모델을 하나도 찾지 못함!")
            
            # 원인 분석
            print("\n🔍 원인 분석:")
            print("1. 검색 경로 문제인지 확인...")
            
            # 수동으로 확인
            manual_search_path = project_root / "backend/app/ai_pipeline/models/checkpoints"
            if manual_search_path.exists():
                manual_files = list(manual_search_path.rglob("*.pth"))
                print(f"   ✅ 수동 검색: {len(manual_files)}개 .pth 파일 발견")
                
                if manual_files:
                    print("   📋 첫 5개 파일:")
                    for file in manual_files[:5]:
                        print(f"      📄 {file.name} ({file.stat().st_size / 1024 / 1024:.1f}MB)")
            else:
                print("   ❌ 수동 검색 경로도 없음")
        
        return detected_models
        
    except ImportError as e:
        print(f"❌ Auto Detector import 실패: {e}")
        return None
    except Exception as e:
        print(f"❌ Auto Detector 실행 오류: {e}")
        import traceback
        traceback.print_exc()
        return None

def debug_model_files_directly():
    """모델 파일들 직접 확인"""
    
    print("\n📁 모델 파일 직접 확인")
    print("=" * 50)
    
    # 알려진 모델 위치
    model_locations = [
        project_root / "backend/app/ai_pipeline/models/checkpoints",
        project_root / "backend/app/ai_pipeline/models/ai_models/checkpoints",
        project_root / "backend/ai_models/checkpoints",
        project_root / "ai_models",
    ]
    
    total_files = 0
    
    for location in model_locations:
        if location.exists():
            pth_files = list(location.rglob("*.pth"))
            safetensors_files = list(location.rglob("*.safetensors"))
            bin_files = list(location.rglob("*.bin"))
            
            file_count = len(pth_files) + len(safetensors_files) + len(bin_files)
            total_files += file_count
            
            print(f"📂 {location}")
            print(f"   📄 .pth: {len(pth_files)}개")
            print(f"   📄 .safetensors: {len(safetensors_files)}개") 
            print(f"   📄 .bin: {len(bin_files)}개")
            print(f"   📊 총합: {file_count}개")
            
            if file_count > 0:
                # Step별 분석
                step_dirs = [d for d in location.iterdir() if d.is_dir() and d.name.startswith('step_')]
                if step_dirs:
                    print(f"   📁 Step 디렉토리: {len(step_dirs)}개")
                    for step_dir in step_dirs[:3]:  # 상위 3개만
                        step_files = len(list(step_dir.rglob("*.pth")))
                        print(f"      📁 {step_dir.name}: {step_files}개 파일")
            print("")
    
    print(f"🎯 총 발견된 모델 파일: {total_files}개")
    
    return total_files

def debug_auto_detector_filters():
    """Auto Detector 필터 설정 확인"""
    
    print("\n🔧 Auto Detector 필터 설정 확인")
    print("=" * 50)
    
    try:
        from backend.app.ai_pipeline.utils.auto_model_detector import RealWorldModelDetector
        
        # 필터 없이 모든 파일 탐지
        detector = RealWorldModelDetector(
            enable_deep_scan=True,
            enable_pytorch_validation=False,
            max_workers=1
        )
        
        # 내부 메서드 직접 호출
        search_path = project_root / "backend/app/ai_pipeline/models/checkpoints"
        
        if hasattr(detector, '_scan_path_for_enhanced_models'):
            print(f"🔍 직접 경로 스캔: {search_path}")
            
            # 모든 카테고리 허용
            from backend.app.ai_pipeline.utils.auto_model_detector import ModelCategory
            all_categories = list(ModelCategory)
            
            results = detector._scan_path_for_enhanced_models(
                model_type="all",
                pattern_info=None,
                search_path=search_path,
                categories_filter=all_categories,
                min_confidence=0.0,  # 최소 임계값
                enable_detailed_analysis=False
            )
            
            print(f"📊 직접 스캔 결과: {len(results)}개 모델")
            
            for name, model in list(results.items())[:5]:
                print(f"   📄 {name}: {model.path}")
        
        else:
            print("❌ _scan_path_for_enhanced_models 메서드 없음")
            
    except Exception as e:
        print(f"❌ 필터 디버깅 실패: {e}")

def suggest_fixes():
    """문제 해결 방안 제시"""
    
    print("\n💡 문제 해결 방안")
    print("=" * 50)
    
    print("🔧 가능한 원인들:")
    print("1. Auto Detector 필터가 너무 엄격함")
    print("2. 파일 권한 문제")
    print("3. 파일 이름 패턴 매칭 실패")
    print("4. PyTorch 검증 과정에서 오류")
    print("5. 메모리 부족으로 스캔 중단")
    
    print("\n🚀 즉시 시도할 해결책:")
    print("1. Auto Detector 강제 재스캔:")
    print("   detector.detect_all_models(force_rescan=True, min_confidence=0.0)")
    
    print("\n2. 직접 모델 등록:")
    print("   ModelLoader에 수동으로 모델 경로 등록")
    
    print("\n3. 환경 변수 설정:")
    print("   export AI_MODELS_ROOT=/path/to/models")

def main():
    """메인 디버깅 함수"""
    
    print("🔍 MyCloset AI Auto Detector 디버깅")
    print("=" * 60)
    print("923개 모델이 있는데 왜 못 찾는지 원인 파악")
    print("")
    
    # 1. 검색 경로 확인
    debug_search_paths()
    
    # 2. 모델 파일 직접 확인
    total_files = debug_model_files_directly()
    
    if total_files > 0:
        # 3. Auto Detector 실제 실행
        detected = debug_actual_auto_detector()
        
        # 4. 필터 설정 확인
        if not detected:
            debug_auto_detector_filters()
        
        # 5. 해결 방안 제시
        suggest_fixes()
    else:
        print("❌ 모델 파일을 전혀 찾을 수 없습니다!")
        print("tree backend/app/ai_pipeline/models 명령으로 확인하세요.")

if __name__ == "__main__":
    main()