#!/usr/bin/env python3
"""
🔧 MyCloset AI - 모델 경로 설정 업데이트 스크립트
conda 환경에서 실행 권장

이 스크립트는 다음을 수행합니다:
1. 현재 존재하는 모델 파일 스캔
2. auto_model_detector.py 경로 업데이트
3. 각 Step 클래스의 체크포인트 경로 수정
4. DeviceManager conda_env 속성 오류 수정
"""

import os
import sys
import shutil
from pathlib import Path
from typing import Dict, List, Optional
import re

def check_conda_env():
    """conda 환경 확인"""
    conda_env = os.environ.get('CONDA_DEFAULT_ENV')
    if conda_env != 'mycloset-ai':
        print(f"⚠️  권장 conda 환경: mycloset-ai")
        print(f"   현재 환경: {conda_env or 'None'}")
        print("   활성화: conda activate mycloset-ai")
        return False
    return True

def scan_existing_models(ai_models_root: Path) -> Dict[str, List[Path]]:
    """현재 존재하는 모델 파일들 스캔"""
    print("🔍 현재 모델 파일 스캔 중...")
    
    model_extensions = {'.pth', '.bin', '.ckpt', '.pt', '.safetensors'}
    found_models = {}
    
    for step_num in range(1, 9):
        step_name = f"step_{step_num:02d}"
        found_models[step_name] = []
        
        # 여러 경로에서 모델 찾기
        search_paths = [
            ai_models_root,
            ai_models_root / "checkpoints",
            ai_models_root / "organized",
        ]
        
        for search_path in search_paths:
            if not search_path.exists():
                continue
                
            for file_path in search_path.rglob("*"):
                if (file_path.suffix.lower() in model_extensions and 
                    file_path.is_file() and
                    step_name in str(file_path).lower()):
                    found_models[step_name].append(file_path)
    
    # 전체 모델도 스캔 (Step 분류되지 않은 것들)
    found_models["general"] = []
    for file_path in ai_models_root.rglob("*"):
        if (file_path.suffix.lower() in model_extensions and 
            file_path.is_file() and
            not any(f"step_{i:02d}" in str(file_path).lower() for i in range(1, 9))):
            found_models["general"].append(file_path)
    
    return found_models

def update_auto_model_detector(backend_root: Path, found_models: Dict[str, List[Path]]):
    """auto_model_detector.py 경로 업데이트"""
    print("🔄 auto_model_detector.py 업데이트 중...")
    
    detector_file = backend_root / "app/ai_pipeline/utils/auto_model_detector.py"
    
    if not detector_file.exists():
        print(f"❌ 파일을 찾을 수 없음: {detector_file}")
        return False
    
    # 백업 생성
    backup_file = detector_file.with_suffix('.py.backup')
    shutil.copy2(detector_file, backup_file)
    print(f"  💾 백업 생성: {backup_file}")
    
    # 새로운 경로 설정 생성
    organized_paths = []
    for step_name, model_files in found_models.items():
        if step_name != "general" and model_files:
            # 첫 번째 모델의 디렉토리를 기준으로 경로 생성
            model_dir = model_files[0].parent
            organized_paths.append(str(model_dir.resolve()))
    
    # 파일 내용 읽기
    with open(detector_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # ENHANCED_SEARCH_PATHS 업데이트
    new_paths_code = f"""
# ==============================================
# 🔧 MyCloset AI 업데이트된 모델 경로 (자동 생성)
# ==============================================

# 실제 존재하는 모델 경로들
UPDATED_MODEL_PATHS = [
{chr(10).join(f'    "{path}",' for path in organized_paths)}
]

# 기존 경로와 병합
if 'ENHANCED_SEARCH_PATHS' in locals():
    ENHANCED_SEARCH_PATHS.extend(UPDATED_MODEL_PATHS)
    # 중복 제거
    ENHANCED_SEARCH_PATHS = list(set(ENHANCED_SEARCH_PATHS))
else:
    ENHANCED_SEARCH_PATHS = UPDATED_MODEL_PATHS
"""
    
    # 파일 끝에 추가
    with open(detector_file, 'w', encoding='utf-8') as f:
        f.write(content + new_paths_code)
    
    print(f"  ✅ {len(organized_paths)}개 경로 추가됨")
    return True

def fix_device_manager_conda_env(backend_root: Path):
    """DeviceManager conda_env 속성 오류 수정"""
    print("🔧 DeviceManager conda_env 속성 오류 수정 중...")
    
    # 가능한 DeviceManager 파일들
    device_manager_files = [
        backend_root / "app/core/gpu_config.py",
        backend_root / "app/ai_pipeline/utils/utils.py",
        backend_root / "app/ai_pipeline/utils/model_loader.py",
    ]
    
    fixed_count = 0
    
    for file_path in device_manager_files:
        if not file_path.exists():
            continue
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # conda_env 속성 관련 오류 수정
            original_content = content
            
            # 패턴 1: conda_env 속성 누락 수정
            if "class DeviceManager" in content:
                # __init__ 메서드에 conda_env 속성 추가
                content = re.sub(
                    r'(class DeviceManager[^:]*:.*?def __init__\(self[^)]*\):.*?)(\n)',
                    r'\1\n        self.conda_env = os.environ.get("CONDA_DEFAULT_ENV", "none")\2',
                    content,
                    flags=re.DOTALL
                )
                
                # conda_env 사용 부분 안전하게 수정
                content = re.sub(
                    r'(\w+)\.conda_env',
                    r'getattr(\1, "conda_env", os.environ.get("CONDA_DEFAULT_ENV", "none"))',
                    content
                )
            
            # 패턴 2: 일반적인 conda_env 접근 수정
            content = re.sub(
                r'([a-zA-Z_][a-zA-Z0-9_]*)\.conda_env',
                r'getattr(\1, "conda_env", os.environ.get("CONDA_DEFAULT_ENV", "none"))',
                content
            )
            
            if content != original_content:
                # 백업 생성
                backup_file = file_path.with_suffix(file_path.suffix + '.backup')
                shutil.copy2(file_path, backup_file)
                
                # 수정된 내용 저장
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                print(f"  ✅ {file_path.name} 수정 완료")
                fixed_count += 1
        
        except Exception as e:
            print(f"  ⚠️  {file_path.name} 수정 실패: {e}")
    
    return fixed_count

def update_step_checkpoint_paths(backend_root: Path, found_models: Dict[str, List[Path]]):
    """Step 클래스들의 체크포인트 경로 업데이트"""
    print("🔄 Step 클래스 체크포인트 경로 업데이트 중...")
    
    steps_dir = backend_root / "app/ai_pipeline/steps"
    
    if not steps_dir.exists():
        print(f"❌ Steps 디렉토리를 찾을 수 없음: {steps_dir}")
        return False
    
    updated_count = 0
    
    for step_num in range(1, 9):
        step_name = f"step_{step_num:02d}"
        step_file = steps_dir / f"{step_name}_*.py"
        
        # 실제 파일 찾기
        step_files = list(steps_dir.glob(f"*{step_name}*.py"))
        
        for step_file in step_files:
            if not step_file.exists():
                continue
                
            try:
                with open(step_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                original_content = content
                
                # 이 Step에 해당하는 모델들
                step_models = found_models.get(step_name, [])
                
                if step_models:
                    # 가장 큰 모델을 주 모델로 사용
                    primary_model = max(step_models, key=lambda x: x.stat().st_size)
                    
                    # 체크포인트 경로 패턴 업데이트
                    model_path_patterns = [
                        (r'DEFAULT_CHECKPOINT_PATH\s*=\s*["\'][^"\']*["\']', 
                         f'DEFAULT_CHECKPOINT_PATH = "{primary_model.resolve()}"'),
                        (r'self\.checkpoint_path\s*=\s*["\'][^"\']*["\']',
                         f'self.checkpoint_path = "{primary_model.resolve()}"'),
                        (r'checkpoint_path\s*=\s*["\'][^"\']*["\']',
                         f'checkpoint_path = "{primary_model.resolve()}"'),
                    ]
                    
                    for pattern, replacement in model_path_patterns:
                        content = re.sub(pattern, replacement, content)
                    
                    # 모델 로딩 관련 오류 처리 개선
                    if "ModelLoader" in content:
                        # ModelLoader 인터페이스 설정 오류 수정
                        content = re.sub(
                            r'(self\.model_loader\.setup_interface\([^)]*\))',
                            r'try:\n            \1\n        except AttributeError as e:\n            self.logger.warning(f"ModelLoader 인터페이스 설정 실패: {e}")',
                            content
                        )
                
                if content != original_content:
                    # 백업 생성
                    backup_file = step_file.with_suffix('.py.backup')
                    shutil.copy2(step_file, backup_file)
                    
                    # 수정된 내용 저장
                    with open(step_file, 'w', encoding='utf-8') as f:
                        f.write(content)
                    
                    print(f"  ✅ {step_file.name} 업데이트 완료")
                    updated_count += 1
                    
            except Exception as e:
                print(f"  ⚠️  {step_file.name} 업데이트 실패: {e}")
    
    return updated_count

def create_model_registry_file(ai_models_root: Path, found_models: Dict[str, List[Path]]):
    """모델 레지스트리 파일 생성"""
    print("📝 모델 레지스트리 파일 생성 중...")
    
    registry_file = ai_models_root / "model_registry.py"
    
    registry_content = '''#!/usr/bin/env python3
"""
🤖 MyCloset AI - 모델 레지스트리 (자동 생성)
실제 존재하는 모델 파일들의 경로를 관리합니다.
"""

from pathlib import Path
from typing import Dict, List, Optional

# 프로젝트 루트
PROJECT_ROOT = Path(__file__).parent.parent
AI_MODELS_ROOT = Path(__file__).parent

# 실제 존재하는 모델들
EXISTING_MODELS = {
'''
    
    total_size = 0
    total_count = 0
    
    for step_name, model_files in found_models.items():
        if not model_files:
            continue
            
        registry_content += f'    "{step_name}": [\n'
        
        for model_file in model_files:
            try:
                file_size = model_file.stat().st_size / (1024 * 1024)  # MB
                total_size += file_size
                total_count += 1
                
                registry_content += f'        {{\n'
                registry_content += f'            "name": "{model_file.name}",\n'
                registry_content += f'            "path": "{model_file.resolve()}",\n'
                registry_content += f'            "size_mb": {file_size:.1f},\n'
                registry_content += f'            "relative_path": "{model_file.relative_to(ai_models_root)}"\n'
                registry_content += f'        }},\n'
            except Exception as e:
                print(f"    ⚠️  {model_file} 정보 수집 실패: {e}")
                
        registry_content += '    ],\n'
    
    registry_content += f'''}}

# 통계 정보
REGISTRY_STATS = {{
    "total_models": {total_count},
    "total_size_mb": {total_size:.1f},
    "total_size_gb": {total_size / 1024:.1f},
    "steps_with_models": {len([k for k, v in found_models.items() if v])},
    "generated_on": "$(date +'%Y-%m-%d %H:%M:%S')"
}}

def get_model_path(step_name: str, model_name: Optional[str] = None) -> Optional[Path]:
    """모델 경로 반환"""
    if step_name not in EXISTING_MODELS:
        return None
    
    step_models = EXISTING_MODELS[step_name]
    if not step_models:
        return None
    
    if model_name:
        for model in step_models:
            if model["name"] == model_name:
                return Path(model["path"])
    
    # 가장 큰 모델 반환 (기본값)
    largest_model = max(step_models, key=lambda x: x["size_mb"])
    return Path(largest_model["path"])

def get_all_model_paths() -> Dict[str, str]:
    """모든 모델 경로 반환"""
    all_paths = {{}}
    for step_name, models in EXISTING_MODELS.items():
        for model in models:
            all_paths[f"{{step_name}}_{{model['name']}}"] = model["path"]
    return all_paths

if __name__ == "__main__":
    print("🤖 MyCloset AI 모델 레지스트리")
    print(f"📊 총 모델: {{REGISTRY_STATS['total_models']}}개")
    print(f"📦 총 크기: {{REGISTRY_STATS['total_size_gb']:.1f}}GB")
    
    for step_name, models in EXISTING_MODELS.items():
        if models:
            print(f"  {{step_name}}: {{len(models)}}개 모델")
'''
    
    with open(registry_file, 'w', encoding='utf-8') as f:
        f.write(registry_content)
    
    print(f"  ✅ {registry_file} 생성 완료")
    print(f"  📊 총 {total_count}개 모델, {total_size/1024:.1f}GB")
    
    return registry_file

def main():
    """메인 실행 함수"""
    print("🔧 MyCloset AI 모델 경로 설정 업데이트 시작...")
    
    # conda 환경 확인
    if not check_conda_env():
        print("⚠️  conda 환경 확인 권장")
    
    # 프로젝트 경로 설정
    project_root = Path("/Users/gimdudeul/MVP/mycloset-ai")
    backend_root = project_root / "backend"
    ai_models_root = backend_root / "ai_models"
    
    if not ai_models_root.exists():
        print(f"❌ AI 모델 디렉토리가 없습니다: {ai_models_root}")
        return False
    
    print(f"📁 프로젝트 루트: {project_root}")
    print(f"🧠 AI 모델 루트: {ai_models_root}")
    
    try:
        # 1. 현재 모델 스캔
        found_models = scan_existing_models(ai_models_root)
        
        total_models = sum(len(models) for models in found_models.values())
        print(f"🔍 총 {total_models}개 모델 파일 발견")
        
        for step_name, models in found_models.items():
            if models:
                print(f"  {step_name}: {len(models)}개")
        
        # 2. auto_model_detector 업데이트
        if update_auto_model_detector(backend_root, found_models):
            print("✅ auto_model_detector 업데이트 완료")
        
        # 3. DeviceManager 오류 수정
        fixed_count = fix_device_manager_conda_env(backend_root)
        print(f"✅ DeviceManager 수정: {fixed_count}개 파일")
        
        # 4. Step 클래스 업데이트
        step_count = update_step_checkpoint_paths(backend_root, found_models)
        print(f"✅ Step 클래스 업데이트: {step_count}개 파일")
        
        # 5. 모델 레지스트리 생성
        registry_file = create_model_registry_file(ai_models_root, found_models)
        print(f"✅ 모델 레지스트리 생성: {registry_file}")
        
        print("\n🎉 모델 경로 설정 업데이트 완료!")
        print("\n📋 다음 단계:")
        print("  1. 서버 재시작: cd backend && python app/main.py")
        print("  2. 모델 검증: python ai_models/model_registry.py")
        print("  3. API 테스트: curl http://localhost:8000/api/ai/status")
        
        return True
        
    except KeyboardInterrupt:
        print("\n⏹️  사용자 중단")
        return False
    except Exception as e:
        print(f"\n❌ 예상치 못한 오류: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)