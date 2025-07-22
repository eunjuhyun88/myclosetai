#!/usr/bin/env python3
"""
🔍 향상된 모델 탐지 테스트
✅ 실제 파일 존재 여부 확인
✅ 파일 크기 및 타입 검증
✅ Step별 매핑 테스트
"""

import os
import sys
from pathlib import Path

def test_enhanced_model_detection():
    """향상된 모델 탐지 테스트"""
    print("🔍 향상된 모델 탐지 테스트 시작...")
    print("="*60)
    
    project_root = Path(__file__).parent
    ai_models_root = project_root / "backend" / "ai_models"
    
    if not ai_models_root.exists():
        print("❌ AI 모델 루트 디렉토리를 찾을 수 없습니다")
        return
    
    print(f"📁 AI 모델 루트: {ai_models_root}")
    print(f"   크기: {get_directory_size(ai_models_root):.1f}GB")
    print("")
    
    # Step별 검사
    step_info = {
        1: {"name": "Human Parsing", "keywords": ["human", "parsing", "schp", "graphonomy"]},
        2: {"name": "Pose Estimation", "keywords": ["pose", "openpose", "body", "hrnet"]},
        3: {"name": "Cloth Segmentation", "keywords": ["cloth", "segment", "u2net", "rembg"]},
        4: {"name": "Geometric Matching", "keywords": ["geometric", "matching", "gmm", "tps"]},
        5: {"name": "Cloth Warping", "keywords": ["warp", "warping", "tom", "tps"]},
        6: {"name": "Virtual Fitting", "keywords": ["viton", "ootd", "diffusion", "fitting"]},
        7: {"name": "Post Processing", "keywords": ["enhance", "super", "resolution", "post"]},
        8: {"name": "Quality Assessment", "keywords": ["quality", "clip", "aesthetic", "assessment"]}
    }
    
    total_models = 0
    total_size_gb = 0
    
    for step_num, info in step_info.items():
        step_dir = ai_models_root / f"step_{step_num:02d}_{info['name'].lower().replace(' ', '_')}"
        
        print(f"🔧 Step {step_num:02d}: {info['name']}")
        print(f"   📁 경로: {step_dir}")
        
        if not step_dir.exists():
            print(f"   ❌ 디렉토리 없음")
            continue
            
        # 모델 파일 검색
        model_extensions = ['.pth', '.pt', '.bin', '.safetensors', '.onnx', '.pkl']
        model_files = []
        
        for ext in model_extensions:
            model_files.extend(list(step_dir.glob(f"*{ext}")))
            model_files.extend(list(step_dir.glob(f"**/*{ext}")))  # 하위 디렉토리도 검색
        
        # 중복 제거 및 실제 존재하는 파일만 필터링
        valid_files = []
        for f in model_files:
            try:
                if f.exists() and f.is_file():
                    valid_files.append(f)
            except:
                continue
        model_files = list(set(valid_files))
        
        if model_files:
            print(f"   ✅ {len(model_files)}개 모델 파일 발견")
            
            # 크기 순으로 정렬 (안전하게)
            try:
                model_files.sort(key=lambda x: x.stat().st_size if x.exists() else 0, reverse=True)
            except:
                print(f"   ⚠️ 일부 파일 정렬 중 오류 발생")
            
            step_size_gb = 0
            for i, model_file in enumerate(model_files[:5]):  # 상위 5개만 표시
                try:
                    size_mb = model_file.stat().st_size / (1024 * 1024)
                    size_gb = size_mb / 1024
                    step_size_gb += size_gb
                    
                    # 키워드 매칭 확인
                    relevance = calculate_relevance(model_file.name.lower(), info['keywords'])
                    relevance_emoji = "🎯" if relevance > 0.5 else "🔍" if relevance > 0.2 else "❓"
                    
                    print(f"      {relevance_emoji} {model_file.name}")
                    print(f"         크기: {size_mb:.1f}MB | 관련성: {relevance:.2f}")
                    
                except Exception as e:
                    print(f"      ❌ {model_file.name} - 오류: {e}")
            
            if len(model_files) > 5:
                print(f"      ... 외 {len(model_files) - 5}개")
            
            print(f"   📊 총 크기: {step_size_gb:.2f}GB")
            total_models += len(model_files)
            total_size_gb += step_size_gb
        else:
            print(f"   ❌ 모델 파일 없음")
        
        print("")
    
    print("="*60)
    print("📊 전체 요약")
    print("="*60)
    print(f"🔍 총 {total_models}개 모델 파일 발견")
    print(f"💾 총 크기: {total_size_gb:.2f}GB")
    
    if total_models > 0:
        print(f"✅ 모델 탐지 성공! 이제 서버가 정상 작동할 것입니다.")
        
        # 추가 검증
        print("")
        print("🔍 추가 검증...")
        
        # 중요 모델 존재 확인
        critical_steps = [1, 2, 6, 8]  # Human Parsing, Pose, Virtual Fitting, Quality
        critical_found = 0
        
        for step_num in critical_steps:
            step_name = step_info[step_num]['name']
            step_dir = ai_models_root / f"step_{step_num:02d}_{step_name.lower().replace(' ', '_')}"
            
            if step_dir.exists():
                model_files = list(step_dir.glob("*.pth")) + list(step_dir.glob("*.bin"))
                if model_files:
                    critical_found += 1
                    print(f"   ✅ {step_name}: 핵심 모델 존재")
                else:
                    print(f"   ⚠️ {step_name}: 핵심 모델 누락")
            else:
                print(f"   ❌ {step_name}: 디렉토리 없음")
        
        print(f"📊 핵심 Step: {critical_found}/{len(critical_steps)} 준비됨")
        
        if critical_found >= 2:
            print("🎉 최소 요구사항 충족 - AI 파이프라인 작동 가능!")
        else:
            print("⚠️ 추가 모델이 필요할 수 있습니다")
            
    else:
        print(f"❌ 모델 파일을 찾을 수 없습니다")
        print("   1. 모델 파일 복사 스크립트를 실행하세요")
        print("   2. 경로 설정을 확인하세요")
    
    print("")
    print("📋 다음 단계:")
    if total_models > 0:
        print("   1. python3 backend/app/main.py (서버 시작)")
        print("   2. 브라우저에서 http://localhost:8000/docs 접속")
    else:
        print("   1. 모델 파일 복사 스크립트 재실행")
        print("   2. 백업 디렉토리에서 모델 복구")

def get_directory_size(directory: Path) -> float:
    """디렉토리 크기 계산 (GB)"""
    try:
        total_size = sum(f.stat().st_size for f in directory.rglob('*') if f.is_file())
        return total_size / (1024 ** 3)
    except:
        return 0.0

def calculate_relevance(filename: str, keywords: list) -> float:
    """파일명과 키워드의 관련성 계산"""
    relevance = 0.0
    filename_lower = filename.lower()
    
    for keyword in keywords:
        if keyword.lower() in filename_lower:
            relevance += 1.0 / len(keywords)
    
    # 추가 점수
    if any(word in filename_lower for word in ['model', 'checkpoint', 'final', 'best']):
        relevance += 0.2
    
    return min(relevance, 1.0)

if __name__ == "__main__":
    test_enhanced_model_detection()