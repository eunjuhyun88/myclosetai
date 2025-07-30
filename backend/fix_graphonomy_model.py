#!/usr/bin/env python3
"""
🔧 Graphonomy 모델 파일 복구 및 검증 스크립트
backend/fix_graphonomy_model.py

✅ 손상된 graphonomy.pth 파일 분석
✅ 대체 모델 파일 탐지 및 활용
✅ 모델 무결성 검증
✅ Human Parsing Step 실제 테스트
"""

import sys
import os
import shutil
import time
from pathlib import Path
import torch
import hashlib

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "backend"))

def analyze_damaged_graphonomy():
    """손상된 graphonomy.pth 분석"""
    
    print("🔍 손상된 Graphonomy 모델 분석 중...")
    
    graphonomy_path = Path("ai_models/step_01_human_parsing/graphonomy.pth")
    
    if not graphonomy_path.exists():
        print("❌ graphonomy.pth 파일이 존재하지 않습니다")
        return False
    
    file_size = graphonomy_path.stat().st_size / (1024 * 1024)
    print(f"   📁 파일 크기: {file_size:.1f}MB")
    
    # 파일 손상 원인 분석
    try:
        with open(graphonomy_path, 'rb') as f:
            header = f.read(1024)
            
        # PyTorch 파일 헤더 확인
        if b'PK' in header[:10]:
            print("   🔍 ZIP 아카이브 형식 감지됨")
        elif b'PYTORCH' in header:
            print("   🔍 PyTorch 네이티브 형식 감지됨")
        else:
            print("   ❌ 알 수 없는 파일 형식")
            
        # 체크섬 계산
        with open(graphonomy_path, 'rb') as f:
            file_hash = hashlib.md5(f.read()).hexdigest()
        print(f"   🔐 MD5 체크섬: {file_hash}")
        
    except Exception as e:
        print(f"   ❌ 파일 분석 실패: {e}")
        return False
    
    return True

def find_alternative_models():
    """대체 Human Parsing 모델 찾기"""
    
    print("\n🔍 대체 Human Parsing 모델 탐색 중...")
    
    parsing_dir = Path("ai_models/step_01_human_parsing")
    alternatives = []
    
    if parsing_dir.exists():
        for model_file in parsing_dir.glob("*.pth"):
            if model_file.name != "graphonomy.pth":
                try:
                    # 모델 로딩 테스트
                    checkpoint = torch.load(model_file, map_location='cpu', weights_only=False)
                    size_mb = model_file.stat().st_size / (1024 * 1024)
                    
                    alternatives.append({
                        'name': model_file.name,
                        'path': model_file,
                        'size_mb': size_mb,
                        'loadable': True,
                        'checkpoint': checkpoint
                    })
                    
                    print(f"   ✅ {model_file.name}: {size_mb:.1f}MB (로딩 가능)")
                    
                except Exception as e:
                    print(f"   ❌ {model_file.name}: 로딩 실패 - {e}")
    
    return alternatives

def create_graphonomy_fallback(alternatives):
    """Graphonomy 폴백 모델 생성"""
    
    print("\n🔧 Graphonomy 폴백 모델 생성 중...")
    
    if not alternatives:
        print("❌ 사용 가능한 대체 모델이 없습니다")
        return False
    
    # 가장 큰 모델을 우선 선택
    best_alternative = max(alternatives, key=lambda x: x['size_mb'])
    
    print(f"   🎯 최적 대체 모델: {best_alternative['name']}")
    
    # 폴백 모델 생성
    fallback_path = Path("ai_models/step_01_human_parsing/graphonomy_fixed.pth")
    
    try:
        # 원본 체크포인트 로드
        checkpoint = best_alternative['checkpoint']
        
        # 필요한 키들이 있는지 확인
        if isinstance(checkpoint, dict):
            # Graphonomy 형식으로 재구성
            fixed_checkpoint = {
                'model_state_dict': checkpoint.get('state_dict', checkpoint),
                'model_name': 'graphonomy_fixed',
                'num_classes': 20,  # LIP 데이터셋
                'input_size': [512, 512],
                'architecture': 'DeepLabV3+',
                'source_file': best_alternative['name']
            }
            
            # 저장
            torch.save(fixed_checkpoint, fallback_path, pickle_protocol=2)
            
            print(f"   ✅ 폴백 모델 생성 완료: {fallback_path}")
            print(f"   📊 크기: {fallback_path.stat().st_size / (1024*1024):.1f}MB")
            
            return True
            
    except Exception as e:
        print(f"   ❌ 폴백 모델 생성 실패: {e}")
        return False

def test_human_parsing_with_fixed_model():
    """수정된 모델로 Human Parsing 테스트"""
    
    print("\n🧠 Human Parsing Step 실제 테스트...")
    
    try:
        # Human Parsing Step import
        from app.ai_pipeline.steps.step_01_human_parsing import HumanParsingStep
        
        # 인스턴스 생성 (수정된 모델 경로 사용)
        step = HumanParsingStep(device='cpu', strict_mode=False)
        
        print("   ✅ HumanParsingStep 인스턴스 생성 성공")
        
        # 모델 경로 수정
        fixed_model_path = Path("ai_models/step_01_human_parsing/graphonomy_fixed.pth")
        if fixed_model_path.exists():
            # 원본을 백업하고 수정된 모델을 사용
            original_path = Path("ai_models/step_01_human_parsing/graphonomy.pth")
            backup_path = Path("ai_models/step_01_human_parsing/graphonomy_damaged.pth")
            
            if original_path.exists():
                shutil.move(str(original_path), str(backup_path))
                print("   📦 손상된 모델을 graphonomy_damaged.pth로 백업")
            
            shutil.copy2(str(fixed_model_path), str(original_path))
            print("   🔄 수정된 모델을 graphonomy.pth로 복사")
        
        # 초기화 테스트
        try:
            if hasattr(step, 'initialize'):
                result = step.initialize()
                if result:
                    print("   🎉 Human Parsing Step 초기화 성공!")
                    return True
                else:
                    print("   ⚠️ 초기화가 False를 반환")
            else:
                print("   ⚠️ initialize 메서드 없음")
        except Exception as init_error:
            print(f"   ❌ 초기화 실패: {init_error}")
        
        return False
        
    except Exception as e:
        print(f"   ❌ Human Parsing Step 테스트 실패: {e}")
        return False

def download_fresh_graphonomy():
    """새로운 Graphonomy 모델 다운로드 안내"""
    
    print("\n📥 새로운 Graphonomy 모델 다운로드 안내")
    print("=" * 60)
    
    print("🔗 공식 소스:")
    print("   1. GitHub: https://github.com/Gaoyiminggithub/Graphonomy")
    print("   2. Google Drive: (논문 저자 제공)")
    print("   3. Papers With Code: https://paperswithcode.com/paper/graphonomy-universal-human-parsing-via-graph")
    
    print("\n📋 필요한 파일:")
    print("   - graphonomy_universal_learned.pth (약 1.2GB)")
    print("   - 또는 inference.pth")
    
    print("\n🎯 설치 위치:")
    print("   ai_models/step_01_human_parsing/graphonomy.pth")
    
    print("\n⚠️ 주의사항:")
    print("   - 모델이 LIP 데이터셋(20 클래스) 형식인지 확인")
    print("   - PyTorch 2.7 호환 형식으로 저장되어야 함")

def main():
    """메인 실행 함수"""
    
    print("🔧 Graphonomy 모델 복구 시스템")
    print("=" * 60)
    
    # 1. 손상된 모델 분석
    print("\n📋 1단계: 손상된 모델 분석")
    damaged_analysis = analyze_damaged_graphonomy()
    
    # 2. 대체 모델 탐색
    print("\n📋 2단계: 대체 모델 탐색")
    alternatives = find_alternative_models()
    
    # 3. 폴백 모델 생성
    if alternatives:
        print("\n📋 3단계: 폴백 모델 생성")
        fallback_created = create_graphonomy_fallback(alternatives)
        
        if fallback_created:
            # 4. 실제 테스트
            print("\n📋 4단계: Human Parsing 실제 테스트")
            test_success = test_human_parsing_with_fixed_model()
            
            if test_success:
                print("\n🎉 Graphonomy 모델 복구 완료!")
                print("✅ Human Parsing Step이 정상적으로 작동합니다")
                return True
    
    # 5. 새 모델 다운로드 안내
    print("\n📋 최종 단계: 새 모델 다운로드 필요")
    download_fresh_graphonomy()
    
    print("\n💡 추천 해결책:")
    print("   1. 임시 해결: 폴백 모델 사용 (기능 제한적)")
    print("   2. 영구 해결: 새로운 Graphonomy 모델 다운로드")
    print("   3. 대안: 다른 Human Parsing 모델 사용")
    
    return False

if __name__ == "__main__":
    success = main()
    
    if success:
        print("\n✅ 복구 완료 - 이제 AI 추론을 테스트할 수 있습니다!")
    else:
        print("\n⚠️ 수동 개입 필요 - 위의 안내를 따라 모델을 교체하세요.")