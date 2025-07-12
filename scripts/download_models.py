#!/usr/bin/env python3
import os
import sys
import subprocess
from pathlib import Path

def main():
    print("🤖 MyCloset AI 모델 다운로드 시작...")
    
    # 프로젝트 루트 찾기
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    backend_dir = project_root / "backend"
    ai_models_dir = backend_dir / "ai_models"
    
    # 디렉토리 생성
    ai_models_dir.mkdir(exist_ok=True)
    
    print(f"📁 모델 저장 경로: {ai_models_dir}")
    
    # Git 확인
    try:
        result = subprocess.run(["git", "--version"], check=True, capture_output=True, text=True)
        print(f"✅ Git 발견: {result.stdout.strip()}")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Git이 설치되어 있지 않습니다.")
        return False
    
    # 모델 저장소 목록
    repositories = [
        {
            "name": "OOTDiffusion",
            "url": "https://github.com/levihsu/OOTDiffusion.git"
        },
        {
            "name": "VITON-HD", 
            "url": "https://github.com/shadow2496/VITON-HD.git"
        }
    ]
    
    # 각 저장소 클론
    for repo in repositories:
        name = repo["name"]
        url = repo["url"]
        target_dir = ai_models_dir / name
        
        if target_dir.exists():
            print(f"✅ {name} 이미 존재함")
            continue
            
        print(f"📥 {name} 다운로드 중...")
        try:
            subprocess.run(
                ["git", "clone", url, str(target_dir)], 
                check=True,
                cwd=str(ai_models_dir.parent)
            )
            print(f"✅ {name} 다운로드 완료")
        except subprocess.CalledProcessError as e:
            print(f"❌ {name} 다운로드 실패: {e}")
    
    # 체크포인트 디렉토리 생성
    checkpoints_dir = ai_models_dir / "checkpoints"
    checkpoints_dir.mkdir(exist_ok=True)
    gitkeep_file = checkpoints_dir / ".gitkeep"
    gitkeep_file.touch()
    
    print("\n✅ 모델 다운로드 완료!")
    print("\n📝 다음 단계:")
    print("1. 사전 훈련된 가중치 수동 다운로드 필요")
    print("2. OOTDiffusion 체크포인트:")
    print("   https://huggingface.co/levihsu/OOTDiffusion")
    print("3. 가중치를 다음 경로에 저장:")
    print(f"   {checkpoints_dir}")
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 모델 설정 완료!")
    else:
        print("\n❌ 모델 설정 실패")
        sys.exit(1)
