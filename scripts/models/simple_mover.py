#!/usr/bin/env python3
"""
🔄 파이프라인 모델 간단 이동 스크립트
================================

파이프라인 내부 모델들을 backend/ai_models/로 이동하고
models 폴더를 완전히 제거합니다.

사용법:
    python simple_mover.py --preview    # 미리보기
    python simple_mover.py --move       # 실제 이동
"""

import os
import shutil
from pathlib import Path
import argparse

def main():
    project_root = Path.cwd()
    
    # 경로 설정
    pipeline_models = project_root / "backend" / "app" / "ai_pipeline" / "models" / "ai_models" / "checkpoints"
    target_base = project_root / "backend" / "ai_models"
    models_folder = project_root / "backend" / "app" / "ai_pipeline" / "models"
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--preview', action='store_true', help='미리보기만')
    parser.add_argument('--move', action='store_true', help='실제 이동')
    args = parser.parse_args()
    
    if not pipeline_models.exists():
        print("❌ 파이프라인 모델 폴더가 없습니다:")
        print(f"   {pipeline_models}")
        return
    
    print("🔍 파이프라인 모델 이동 계획")
    print("=" * 50)
    
    # 이동할 파일들 스캔
    move_tasks = []
    
    for step_dir in pipeline_models.iterdir():
        if step_dir.is_dir() and step_dir.name.startswith('step_'):
            for model_file in step_dir.iterdir():
                if model_file.is_file():
                    source = model_file
                    target = target_base / step_dir.name / model_file.name
                    size_mb = model_file.stat().st_size / (1024 * 1024)
                    
                    move_tasks.append({
                        'source': source,
                        'target': target,
                        'name': model_file.name,
                        'step': step_dir.name,
                        'size_mb': size_mb
                    })
    
    if not move_tasks:
        print("📄 이동할 모델이 없습니다.")
        return
    
    # 이동 계획 출력
    print(f"📦 이동할 모델: {len(move_tasks)}개")
    print()
    
    for i, task in enumerate(move_tasks, 1):
        print(f"{i:2d}. {task['name']} ({task['size_mb']:.1f}MB)")
        print(f"    📤 {task['source']}")
        print(f"    📥 {task['target']}")
        print(f"    🎯 {task['step']}")
        print()
    
    # 삭제될 폴더
    print("🗑️ 삭제될 폴더:")
    print(f"   {models_folder}")
    print()
    
    if args.preview:
        print("💡 실제 이동하려면: python simple_mover.py --move")
        return
    
    if args.move:
        # 확인
        response = input("⚠️ 위 계획대로 이동하시겠습니까? [y/N]: ")
        if response.lower() != 'y':
            print("❌ 취소되었습니다.")
            return
        
        print("🚀 모델 이동 시작...")
        
        # 모델 이동
        for i, task in enumerate(move_tasks, 1):
            try:
                # 대상 디렉토리 생성
                task['target'].parent.mkdir(parents=True, exist_ok=True)
                
                # 파일 이동
                shutil.move(str(task['source']), str(task['target']))
                
                print(f"✅ [{i}/{len(move_tasks)}] {task['name']}")
                
            except Exception as e:
                print(f"❌ [{i}/{len(move_tasks)}] {task['name']} 실패: {e}")
        
        # models 폴더 삭제
        print("🗑️ models 폴더 삭제 중...")
        try:
            shutil.rmtree(models_folder)
            print("✅ models 폴더 삭제 완료")
        except Exception as e:
            print(f"⚠️ models 폴더 삭제 실패: {e}")
        
        print("\n🎯 이동 완료!")
        print("📁 새 구조:")
        
        # 새 구조 확인
        for step in ['step_01_human_parsing', 'step_02_pose_estimation', 'step_03_cloth_segmentation',
                    'step_04_geometric_matching', 'step_05_cloth_warping', 'step_06_virtual_fitting',
                    'step_07_post_processing', 'step_08_quality_assessment']:
            step_path = target_base / step
            if step_path.exists():
                files = list(step_path.iterdir())
                print(f"   📁 {step}/ ({len(files)}개 파일)")
        
        print("\n🚀 다음 단계:")
        print("1. 파이프라인 코드에서 경로 수정 필요")
        print("2. 'models/ai_models' → 'backend/ai_models' 로 변경")
        print("3. 파이프라인 테스트 실행")
    
    else:
        print("💡 사용법:")
        print("   python simple_mover.py --preview    # 미리보기")
        print("   python simple_mover.py --move       # 실제 이동")

if __name__ == "__main__":
    main()