#!/usr/bin/env python3
"""
🔍 MyCloset AI 체크포인트 자동 탐지 실행
conda 환경에서 실행하여 현재 모델 위치를 파악
"""

import os
import sys
from pathlib import Path

# 프로젝트 루트를 Python path에 추가
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "backend"))

print("🔍 MyCloset AI 자동 모델 탐지 시작...")
print(f"📁 프로젝트 루트: {project_root}")
print(f"🐍 Python 경로: {sys.executable}")
print(f"🐍 Conda 환경: {os.environ.get('CONDA_DEFAULT_ENV', '없음')}")
print("=" * 60)

try:
    # auto_model_detector import 시도
    print("📦 auto_model_detector 모듈 로드 중...")
    
    try:
        from backend.app.ai_pipeline.utils.auto_model_detector import (
            RealWorldModelDetector,
            create_real_world_detector
        )
        print("✅ auto_model_detector 로드 성공!")
    except ImportError as e:
        print(f"❌ auto_model_detector 로드 실패: {e}")
        print("💡 경로 문제일 수 있습니다. 다른 경로로 시도...")
        
        # 직접 경로 지정해서 시도
        backend_path = project_root / "backend"
        if backend_path.exists():
            sys.path.insert(0, str(backend_path))
            from app.ai_pipeline.utils.auto_model_detector import (
                RealWorldModelDetector,
                create_real_world_detector
            )
            print("✅ 직접 경로로 로드 성공!")
        else:
            raise ImportError("backend 디렉토리를 찾을 수 없습니다")
    
    # 실제 탐지 실행
    print("\n🔍 실제 AI 모델 탐지 시작...")
    
    detector = create_real_world_detector(
        enable_pytorch_validation=True,
        enable_deep_scan=True,
        max_workers=2  # 안전한 병렬 처리
    )
    
    print("🔄 모델 스캔 실행 중... (최대 5분 소요)")
    detected_models = detector.detect_all_models(
        force_rescan=False,  # 캐시 사용
        min_confidence=0.3,
        enable_detailed_analysis=False  # 빠른 탐지
    )
    
    print(f"\n✅ 탐지 완료! {len(detected_models)}개 모델 발견")
    print("=" * 60)
    
    # 탐지 결과 요약
    if detected_models:
        print("📊 탐지된 모델 요약:")
        
        total_size_gb = 0
        pytorch_valid_count = 0
        
        for i, (name, model) in enumerate(detected_models.items(), 1):
            size_gb = model.file_size_mb / 1024
            total_size_gb += size_gb
            
            if model.pytorch_valid:
                pytorch_valid_count += 1
            
            status_icon = "✅" if model.pytorch_valid else "❓"
            
            print(f"  {i:2d}. {status_icon} {name}")
            print(f"      📁 {model.path}")
            print(f"      📊 {size_gb:.2f}GB | Step: {model.step_name} | 신뢰도: {model.confidence_score:.2f}")
            
            if i >= 10:  # 상위 10개만 출력
                remaining = len(detected_models) - 10
                if remaining > 0:
                    print(f"      ... 그 외 {remaining}개 모델")
                break
        
        print(f"\n📈 통계:")
        print(f"   • 총 크기: {total_size_gb:.2f}GB")
        print(f"   • PyTorch 검증: {pytorch_valid_count}/{len(detected_models)}개")
        print(f"   • 평균 신뢰도: {sum(m.confidence_score for m in detected_models.values()) / len(detected_models):.3f}")
        
        # Step별 분포
        step_counts = {}
        for model in detected_models.values():
            step = model.step_name
            if step not in step_counts:
                step_counts[step] = 0
            step_counts[step] += 1
        
        print(f"\n🎯 Step별 분포:")
        for step, count in sorted(step_counts.items()):
            print(f"   • {step}: {count}개")
    
    else:
        print("⚠️ 탐지된 모델이 없습니다.")
        print("💡 다음을 확인해주세요:")
        print("   1. ai_models 디렉토리가 존재하는지")
        print("   2. 체크포인트 파일들이 올바른 위치에 있는지")
        print("   3. 파일 권한이 올바른지")
    
    print("\n" + "=" * 60)
    print("🎯 다음 단계:")
    print("   1. 탐지된 모델들이 올바른지 확인")
    print("   2. ModelLoader 설정 업데이트")
    print("   3. Step별 모델 매핑 확인")
    print("   4. 애플리케이션 테스트")

except Exception as e:
    print(f"❌ 오류 발생: {e}")
    import traceback
    print(f"📋 상세 오류:")
    traceback.print_exc()
    
    print(f"\n💡 해결 방법:")
    print(f"   1. conda 환경이 활성화되어 있는지 확인")
    print(f"   2. 필요한 패키지가 설치되어 있는지 확인: pip install torch")
    print(f"   3. 프로젝트 디렉토리에서 실행하고 있는지 확인")

print("\n🔍 탐지 완료!")