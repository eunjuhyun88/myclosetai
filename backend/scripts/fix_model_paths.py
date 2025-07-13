#!/usr/bin/env python3
# backend/scripts/fix_model_paths.py
"""
기존 AI 모델 경로를 인식하고 설정을 업데이트하는 스크립트
"""

import os
import yaml
from pathlib import Path
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelPathFixer:
    """기존 AI 모델 경로 수정 및 설정 업데이트"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.ai_models_dir = self.project_root / "ai_models"
        
        logger.info(f"🔍 AI 모델 디렉토리: {self.ai_models_dir}")
    
    def scan_existing_models(self):
        """기존 AI 모델 스캔"""
        
        logger.info("🔍 기존 AI 모델 스캔 중...")
        
        found_models = {}
        
        # 알려진 모델 이름들
        known_models = {
            "ootdiffusion": ["OOTDiffusion", "ootdiffusion_hf"],
            "viton_hd": ["VITON-HD", "HR-VITON"],
            "graphonomy": ["Graphonomy", "Self-Correction-Human-Parsing"],
            "openpose": ["openpose"],
            "detectron2": ["detectron2"]
        }
        
        # 각 모델 디렉토리 확인
        for model_key, possible_names in known_models.items():
            for name in possible_names:
                model_path = self.ai_models_dir / name
                if model_path.exists() and model_path.is_dir():
                    # 디렉토리 내용 확인
                    files = list(model_path.glob("**/*"))
                    file_count = len([f for f in files if f.is_file()])
                    
                    if file_count > 0:
                        found_models[model_key] = {
                            "name": name,
                            "path": str(model_path),
                            "files_count": file_count,
                            "size_mb": self._get_directory_size(model_path)
                        }
                        logger.info(f"✅ {model_key} 발견: {name} ({file_count}개 파일, {found_models[model_key]['size_mb']}MB)")
                        break
        
        # 추가 파일들 확인
        additional_files = {
            "gen.pth": "VITON-HD 생성 모델",
            "mtviton.pth": "MT-VITON 모델"
        }
        
        for filename, description in additional_files.items():
            file_path = self.ai_models_dir / filename
            if file_path.exists():
                logger.info(f"✅ 추가 파일 발견: {filename} ({description})")
        
        return found_models
    
    def _get_directory_size(self, path):
        """디렉토리 크기 계산 (MB)"""
        total_size = 0
        try:
            for file_path in path.glob("**/*"):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
        except:
            pass
        return round(total_size / (1024 * 1024), 1)
    
    def create_updated_config(self, found_models):
        """업데이트된 모델 설정 파일 생성"""
        
        logger.info("📝 모델 설정 파일 업데이트 중...")
        
        config = {
            "models": {},
            "processing": {
                "image_size": 512,
                "batch_size": 1,
                "num_inference_steps": 20,
                "guidance_scale": 7.5,
                "device": "mps"  # M3 Max 기본값
            },
            "paths": {
                "ai_models_root": str(self.ai_models_dir),
                "cache_dir": str(self.ai_models_dir / "cache"),
                "temp_dir": str(self.ai_models_dir / "temp")
            }
        }
        
        # 발견된 모델들 설정에 추가
        for model_key, model_info in found_models.items():
            config["models"][model_key] = {
                "name": model_info["name"],
                "path": model_info["path"],
                "enabled": True,
                "device": "mps",
                "files_count": model_info["files_count"],
                "size_mb": model_info["size_mb"]
            }
        
        # 설정 파일 저장
        config_path = self.ai_models_dir / "models_config_updated.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        
        logger.info(f"✅ 업데이트된 설정 파일 저장: {config_path}")
        
        # 기존 파일 백업 및 교체
        old_config = self.ai_models_dir / "model_config.yaml"
        if old_config.exists():
            backup_path = self.ai_models_dir / "model_config_backup.yaml"
            old_config.rename(backup_path)
            logger.info(f"📦 기존 설정 백업: {backup_path}")
        
        config_path.rename(old_config)
        logger.info("✅ 새 설정 파일 적용 완료")
        
        return config
    
    def create_model_manager_config(self, found_models):
        """모델 매니저용 Python 설정 파일 생성"""
        
        logger.info("🐍 Python 모델 매니저 설정 생성 중...")
        
        # 모델 매니저용 설정 파일
        config_content = f'''# backend/app/core/model_paths.py
"""
AI 모델 경로 설정 - 자동 생성됨
기존 다운로드된 모델들의 실제 경로
"""

from pathlib import Path

# 기본 경로
AI_MODELS_ROOT = Path(__file__).parent.parent.parent / "ai_models"

# 발견된 모델 경로들
MODEL_PATHS = {{
'''
        
        for model_key, model_info in found_models.items():
            config_content += f'''    "{model_key}": {{
        "name": "{model_info['name']}",
        "path": AI_MODELS_ROOT / "{model_info['name']}",
        "enabled": True,
        "files_count": {model_info['files_count']},
        "size_mb": {model_info['size_mb']}
    }},
'''
        
        config_content += '''}}

# 추가 파일 경로
ADDITIONAL_FILES = {
'''
        
        # 추가 파일들 확인
        additional_files = ["gen.pth", "mtviton.pth"]
        for filename in additional_files:
            file_path = self.ai_models_dir / filename
            if file_path.exists():
                config_content += f'''    "{filename}": AI_MODELS_ROOT / "{filename}",
'''
        
        config_content += '''}

def get_model_path(model_key: str) -> Path:
    """모델 경로 반환"""
    if model_key in MODEL_PATHS:
        return MODEL_PATHS[model_key]["path"]
    raise KeyError(f"Unknown model: {model_key}")

def is_model_available(model_key: str) -> bool:
    """모델 사용 가능 여부 확인"""
    if model_key in MODEL_PATHS:
        return MODEL_PATHS[model_key]["path"].exists()
    return False

def get_available_models() -> list:
    """사용 가능한 모델 목록 반환"""
    available = []
    for key, info in MODEL_PATHS.items():
        if info["path"].exists():
            available.append(key)
    return available
'''
        
        # 파일 저장
        config_path = self.project_root / "app" / "core" / "model_paths.py"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w') as f:
            f.write(config_content)
        
        logger.info(f"✅ Python 모델 설정 저장: {config_path}")
    
    def test_model_access(self, found_models):
        """모델 접근 테스트"""
        
        logger.info("🧪 모델 접근 테스트 중...")
        
        for model_key, model_info in found_models.items():
            model_path = Path(model_info["path"])
            
            try:
                # 디렉토리 접근 테스트
                files = list(model_path.glob("*"))
                logger.info(f"✅ {model_key}: {len(files)}개 파일 접근 가능")
                
                # 주요 파일 확인
                important_files = []
                for file_path in model_path.glob("**/*"):
                    if file_path.is_file() and file_path.suffix in ['.pth', '.pt', '.safetensors', '.json', '.bin']:
                        important_files.append(file_path.name)
                        if len(important_files) >= 3:  # 처음 3개만 표시
                            break
                
                if important_files:
                    logger.info(f"   주요 파일: {', '.join(important_files[:3])}")
                
            except Exception as e:
                logger.error(f"❌ {model_key}: 접근 실패 - {e}")
    
    def generate_usage_guide(self, found_models):
        """사용 가이드 생성"""
        
        guide_content = f"""# AI 모델 사용 가이드

## 📁 발견된 모델들

"""
        
        for model_key, model_info in found_models.items():
            guide_content += f"""### {model_info['name']} ({model_key})
- **경로**: `{model_info['path']}`
- **파일 수**: {model_info['files_count']}개
- **크기**: {model_info['size_mb']}MB
- **상태**: ✅ 사용 가능

"""
        
        guide_content += """
## 🚀 사용 방법

### Python에서 모델 경로 사용:
```python
from app.core.model_paths import get_model_path, is_model_available

# 모델 경로 가져오기
ootd_path = get_model_path("ootdiffusion")
print(f"OOTDiffusion 경로: {ootd_path}")

# 모델 사용 가능 여부 확인
if is_model_available("ootdiffusion"):
    print("✅ OOTDiffusion 사용 가능")
```

### FastAPI에서 사용:
```python
from app.core.model_paths import MODEL_PATHS

# 모든 사용 가능한 모델 확인
available_models = [key for key, info in MODEL_PATHS.items() 
                   if info["path"].exists()]
print(f"사용 가능한 모델: {available_models}")
```

## 🔧 다음 단계

1. **모델 로더 구현**: `app/services/model_manager.py`
2. **가상 피팅 API**: `app/api/virtual_tryon.py`
3. **GPU 최적화**: M3 Max Metal 가속 활용

"""
        
        guide_path = self.ai_models_dir / "USAGE_GUIDE.md"
        with open(guide_path, 'w') as f:
            f.write(guide_content)
        
        logger.info(f"✅ 사용 가이드 생성: {guide_path}")

def main():
    """메인 실행 함수"""
    
    print("🔧 MyCloset AI - 기존 모델 경로 수정 도구")
    print("=" * 50)
    
    fixer = ModelPathFixer()
    
    # 1. 기존 모델 스캔
    found_models = fixer.scan_existing_models()
    
    if not found_models:
        logger.warning("⚠️ AI 모델을 찾을 수 없습니다.")
        logger.info("AI 모델을 먼저 다운로드하세요.")
        return
    
    logger.info(f"🎉 총 {len(found_models)}개 모델 발견!")
    
    # 2. 설정 파일 업데이트
    config = fixer.create_updated_config(found_models)
    
    # 3. Python 설정 파일 생성
    fixer.create_model_manager_config(found_models)
    
    # 4. 모델 접근 테스트
    fixer.test_model_access(found_models)
    
    # 5. 사용 가이드 생성
    fixer.generate_usage_guide(found_models)
    
    print("\n🎉 모델 경로 수정 완료!")
    print("📋 생성된 파일:")
    print("  - ai_models/model_config.yaml (업데이트됨)")
    print("  - app/core/model_paths.py (새로 생성)")
    print("  - ai_models/USAGE_GUIDE.md (사용 가이드)")
    print("\n🚀 다음 단계:")
    print("  1. FastAPI 서버 실행: python app/main.py")
    print("  2. 모델 상태 확인: curl http://localhost:8000/health/models")
    print("  3. AI 모델 로더 구현")

if __name__ == "__main__":
    main()