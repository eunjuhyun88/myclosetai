# fix_model_paths.py
"""
AI 모델 경로 수정 스크립트
backend/ai_models 경로로 올바르게 설정
"""

import os
from pathlib import Path

def fix_model_paths():
    """모델 경로를 올바르게 수정"""
    
    # 현재 backend 디렉토리 확인
    backend_dir = Path(__file__).parent / "backend"
    ai_models_dir = backend_dir / "ai_models"
    
    print("🔍 모델 경로 수정 스크립트")
    print("=" * 50)
    
    # 1. 실제 경로 확인
    print(f"Backend 디렉토리: {backend_dir}")
    print(f"AI Models 디렉토리: {ai_models_dir}")
    print(f"AI Models 존재 여부: {ai_models_dir.exists()}")
    
    if ai_models_dir.exists():
        print("✅ backend/ai_models 디렉토리 발견!")
        
        # 2. 하위 디렉토리 확인
        subdirs = [d for d in ai_models_dir.iterdir() if d.is_dir()]
        print(f"📁 하위 디렉토리: {len(subdirs)}개")
        for subdir in subdirs:
            print(f"   - {subdir.name}")
        
        # 3. 모델 파일 확인
        model_files = []
        for ext in ['.pth', '.pt', '.bin', '.safetensors']:
            model_files.extend(list(ai_models_dir.rglob(f"*{ext}")))
        
        print(f"🤖 모델 파일: {len(model_files)}개")
        total_size = 0
        for model_file in model_files:
            size = model_file.stat().st_size / (1024**3)  # GB
            total_size += size
            print(f"   - {model_file.name}: {size:.2f}GB")
        
        print(f"💾 총 모델 크기: {total_size:.2f}GB")
        
    else:
        print("❌ backend/ai_models 디렉토리가 없습니다!")
        print("다음 명령으로 생성하세요:")
        print("mkdir -p backend/ai_models")
        return False
    
    # 4. 설정 파일들 수정
    config_files_to_fix = [
        "backend/app/core/config.py",
        "backend/app/ai_pipeline/utils/auto_model_detector.py",
        "backend/app/ai_pipeline/utils/model_loader.py"
    ]
    
    print("\n🔧 수정할 설정 파일들:")
    for config_file in config_files_to_fix:
        if Path(config_file).exists():
            print(f"✅ {config_file}")
        else:
            print(f"❌ {config_file} (파일 없음)")
    
    return True

def create_fixed_config():
    """수정된 설정 파일 생성"""
    
    config_content = '''# backend/app/core/fixed_config.py
"""
수정된 AI 모델 경로 설정
"""

from pathlib import Path
import os

# 프로젝트 루트 및 백엔드 경로
BACKEND_DIR = Path(__file__).parent.parent.parent
PROJECT_ROOT = BACKEND_DIR.parent
AI_MODELS_DIR = BACKEND_DIR / "ai_models"

# 환경 변수 설정
os.environ["MYCLOSET_AI_MODELS_PATH"] = str(AI_MODELS_DIR)
os.environ["MYCLOSET_BACKEND_PATH"] = str(BACKEND_DIR)

# 로깅용 경로 정보
MODEL_PATH_INFO = {
    "project_root": str(PROJECT_ROOT),
    "backend_dir": str(BACKEND_DIR), 
    "ai_models_dir": str(AI_MODELS_DIR),
    "ai_models_exists": AI_MODELS_DIR.exists(),
}

def get_ai_models_path():
    """AI 모델 디렉토리 경로 반환"""
    return AI_MODELS_DIR

def get_model_file_path(step_name: str, model_name: str = None):
    """특정 Step의 모델 파일 경로 반환"""
    step_dir = AI_MODELS_DIR / step_name
    
    if model_name:
        return step_dir / model_name
    
    # 모델 파일 자동 탐지
    if step_dir.exists():
        for ext in ['.pth', '.pt', '.bin', '.safetensors']:
            model_files = list(step_dir.glob(f"*{ext}"))
            if model_files:
                return model_files[0]  # 첫 번째 파일 반환
    
    return None

def validate_model_paths():
    """모델 경로 검증"""
    print("🔍 모델 경로 검증")
    print(f"AI Models 디렉토리: {AI_MODELS_DIR}")
    print(f"존재 여부: {AI_MODELS_DIR.exists()}")
    
    if AI_MODELS_DIR.exists():
        steps = [
            "step_01_human_parsing",
            "step_02_pose_estimation", 
            "step_03_cloth_segmentation",
            "step_04_geometric_matching",
            "step_05_cloth_warping",
            "step_06_virtual_fitting",
            "step_07_post_processing",
            "step_08_quality_assessment"
        ]
        
        print("\\n📁 Step별 디렉토리 확인:")
        for step in steps:
            step_dir = AI_MODELS_DIR / step
            print(f"   {step}: {'✅' if step_dir.exists() else '❌'}")
    
    return AI_MODELS_DIR.exists()

if __name__ == "__main__":
    validate_model_paths()
'''
    
    # 파일 저장
    with open("backend/app/core/fixed_config.py", "w", encoding="utf-8") as f:
        f.write(config_content)
    
    print("✅ fixed_config.py 생성 완료!")

def create_model_path_script():
    """모델 경로 설정을 위한 bash 스크립트 생성"""
    
    script_content = '''#!/bin/bash
# fix_model_paths.sh
# AI 모델 경로 수정 스크립트

echo "🔧 MyCloset AI 모델 경로 수정"
echo "=============================="

# 1. conda 환경 활성화
echo "🐍 conda 환경 활성화..."
conda activate mycloset-ai

# 2. 현재 경로 확인
echo "📍 현재 위치: $(pwd)"

# 3. backend/ai_models 디렉토리 확인
if [ -d "backend/ai_models" ]; then
    echo "✅ backend/ai_models 디렉토리 발견"
    echo "📊 디렉토리 내용:"
    ls -la backend/ai_models/
    
    # 모델 파일 크기 확인
    echo "💾 모델 파일 크기:"
    find backend/ai_models -name "*.pth" -o -name "*.pt" -o -name "*.bin" -o -name "*.safetensors" | while read file; do
        size=$(du -h "$file" | cut -f1)
        echo "   $file: $size"
    done
    
else
    echo "❌ backend/ai_models 디렉토리가 없습니다"
    echo "📁 디렉토리 생성 중..."
    mkdir -p backend/ai_models/{step_01_human_parsing,step_02_pose_estimation,step_03_cloth_segmentation,step_04_geometric_matching,step_05_cloth_warping,step_06_virtual_fitting,step_07_post_processing,step_08_quality_assessment}
    echo "✅ 디렉토리 생성 완료"
fi

# 4. 환경 변수 설정
export MYCLOSET_AI_MODELS_PATH="$(pwd)/backend/ai_models"
export MYCLOSET_BACKEND_PATH="$(pwd)/backend"

echo "🔧 환경 변수 설정:"
echo "   MYCLOSET_AI_MODELS_PATH=$MYCLOSET_AI_MODELS_PATH"
echo "   MYCLOSET_BACKEND_PATH=$MYCLOSET_BACKEND_PATH"

# 5. Python 경로 테스트
echo "🧪 Python 경로 테스트:"
cd backend
python -c "
from pathlib import Path
ai_models = Path('ai_models')
print(f'AI Models 경로: {ai_models.absolute()}')
print(f'존재 여부: {ai_models.exists()}')
if ai_models.exists():
    subdirs = [d.name for d in ai_models.iterdir() if d.is_dir()]
    print(f'하위 디렉토리: {subdirs}')
"

echo "✅ 모델 경로 수정 완료!"
echo "🚀 이제 서버를 실행하세요: python main.py"
'''
    
    # 스크립트 저장
    with open("fix_model_paths.sh", "w") as f:
        f.write(script_content)
    
    # 실행 권한 부여
    os.chmod("fix_model_paths.sh", 0o755)
    
    print("✅ fix_model_paths.sh 생성 완료!")

if __name__ == "__main__":
    print("🔧 AI 모델 경로 수정 도구")
    print("=" * 40)
    
    # 1. 현재 상태 확인
    if fix_model_paths():
        print("\n✅ 경로 확인 완료!")
        
        # 2. 수정된 설정 파일 생성
        create_fixed_config()
        
        # 3. bash 스크립트 생성  
        create_model_path_script()
        
        print("\n🚀 다음 단계:")
        print("1. chmod +x fix_model_paths.sh")
        print("2. ./fix_model_paths.sh")
        print("3. cd backend && python main.py")
        
    else:
        print("\n❌ 경로 수정이 필요합니다.")