#!/usr/bin/env python3
"""
MyCloset AI - AI 모델 체크포인트 자동 다운로드 스크립트
M3 Max 128GB 최적화 버전 - 실제 AI 모델 다운로드

필요한 모델들:
1. OOTDiffusion - 최신 가상 피팅 모델
2. Human Parsing - 인체 분할 모델 (Graphonomy, SCHP)
3. U²-Net - 배경 제거 및 의류 분할
4. Stable Diffusion - 기본 디퓨전 모델
5. CLIP - 텍스트-이미지 이해 모델

사용법:
    python3 backend/scripts/download_ai_models.py
"""

import os
import sys
import time
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Optional
import subprocess
import hashlib
import json

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

class AIModelDownloader:
    """AI 모델 자동 다운로드 시스템"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.models_dir = self.project_root / "ai_models"
        self.checkpoints_dir = self.models_dir / "checkpoints"
        self.configs_dir = self.models_dir / "configs"
        self.temp_dir = self.models_dir / "temp"
        
        # 디렉토리 생성
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.configs_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        # 다운로드할 모델 목록
        self.models = {
            "ootdiffusion": {
                "name": "OOTDiffusion",
                "repo_id": "levihsu/OOTDiffusion",
                "local_dir": "ootdiffusion_hf",
                "size_gb": 8.0,
                "priority": 1,
                "step": "step_06_virtual_fitting",
                "description": "최신 고품질 가상 피팅 모델",
                "required_files": ["checkpoints/ootd", "configs"]
            },
            "human_parsing": {
                "name": "Human Parsing (Graphonomy)",
                "repo_id": "mattmdjaga/human_parsing",
                "local_dir": "human_parsing",
                "size_gb": 0.5,
                "priority": 2,
                "step": "step_01_human_parsing",
                "description": "인체 분할 모델 (20개 부위)",
                "required_files": ["atr_model.pth", "lip_model.pth"]
            },
            "u2net": {
                "name": "U²-Net Background Removal",
                "repo_id": "skytnt/u2net",
                "local_dir": "u2net",
                "size_gb": 0.2,
                "priority": 3,
                "step": "step_03_cloth_segmentation",
                "description": "배경 제거 및 의류 분할 모델",
                "required_files": ["u2net.pth"]
            },
            "stable_diffusion": {
                "name": "Stable Diffusion v1.5",
                "repo_id": "runwayml/stable-diffusion-v1-5",
                "local_dir": "stable-diffusion-v1-5",
                "size_gb": 4.0,
                "priority": 4,
                "step": "step_06_virtual_fitting",
                "description": "기본 디퓨전 모델",
                "required_files": ["pytorch_model.bin", "config.json"]
            },
            "clip_vit_base": {
                "name": "CLIP ViT-B/32",
                "repo_id": "openai/clip-vit-base-patch32",
                "local_dir": "clip-vit-base-patch32",
                "size_gb": 0.6,
                "priority": 5,
                "step": "auxiliary",
                "description": "텍스트-이미지 이해 모델",
                "required_files": ["pytorch_model.bin", "config.json"]
            },
            "clip_vit_large": {
                "name": "CLIP ViT-L/14",
                "repo_id": "openai/clip-vit-large-patch14",
                "local_dir": "clip-vit-large-patch14",
                "size_gb": 1.6,
                "priority": 6,
                "step": "auxiliary",
                "description": "대형 CLIP 모델 (고성능)",
                "required_files": ["pytorch_model.bin", "config.json"]
            }
        }
        
        self.total_size_gb = sum(model["size_gb"] for model in self.models.values())
    
    def check_dependencies(self) -> bool:
        """필요한 의존성 확인"""
        logger.info("🔍 의존성 확인 중...")
        
        required_packages = [
            "transformers",
            "diffusers", 
            "torch",
            "huggingface_hub",
            "accelerate"
        ]
        
        missing_packages = []
        
        for package in required_packages:
            try:
                __import__(package)
                logger.info(f"✅ {package} 설치됨")
            except ImportError:
                missing_packages.append(package)
                logger.warning(f"❌ {package} 누락")
        
        if missing_packages:
            logger.error(f"❌ 누락된 패키지: {', '.join(missing_packages)}")
            logger.info("📝 다음 명령어로 설치하세요:")
            logger.info(f"pip install {' '.join(missing_packages)}")
            return False
        
        logger.info("✅ 모든 의존성 확인 완료")
        return True
    
    def check_disk_space(self) -> bool:
        """디스크 공간 확인"""
        logger.info("💾 디스크 공간 확인 중...")
        
        try:
            # 사용 가능한 디스크 공간 확인
            statvfs = os.statvfs(self.models_dir)
            free_bytes = statvfs.f_frsize * statvfs.f_bavail
            free_gb = free_bytes / (1024**3)
            
            logger.info(f"📊 사용 가능한 공간: {free_gb:.1f} GB")
            logger.info(f"📦 필요한 공간: {self.total_size_gb:.1f} GB")
            
            if free_gb < self.total_size_gb + 5:  # 5GB 여유 공간
                logger.warning("⚠️ 디스크 공간이 부족할 수 있습니다")
                return False
            
            logger.info("✅ 디스크 공간 충분")
            return True
            
        except Exception as e:
            logger.warning(f"⚠️ 디스크 공간 확인 실패: {e}")
            return True
    
    def download_model(self, model_key: str, model_info: Dict) -> bool:
        """개별 모델 다운로드"""
        logger.info(f"📥 {model_info['name']} 다운로드 시작...")
        logger.info(f"   📍 크기: {model_info['size_gb']} GB")
        logger.info(f"   📝 설명: {model_info['description']}")
        
        local_path = self.checkpoints_dir / model_info['local_dir']
        
        # 이미 다운로드되어 있는지 확인
        if local_path.exists() and self.verify_model_integrity(local_path, model_info):
            logger.info(f"✅ {model_info['name']} 이미 다운로드됨")
            return True
        
        try:
            # Hugging Face Hub에서 다운로드
            from huggingface_hub import snapshot_download
            
            logger.info(f"🔄 {model_info['name']} 다운로드 중...")
            start_time = time.time()
            
            # 다운로드 실행
            snapshot_download(
                repo_id=model_info['repo_id'],
                local_dir=str(local_path),
                resume_download=True,
                local_dir_use_symlinks=False
            )
            
            elapsed_time = time.time() - start_time
            logger.info(f"✅ {model_info['name']} 다운로드 완료 ({elapsed_time:.1f}초)")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ {model_info['name']} 다운로드 실패: {e}")
            return False
    
    def verify_model_integrity(self, model_path: Path, model_info: Dict) -> bool:
        """모델 무결성 확인"""
        try:
            # 필수 파일들이 있는지 확인
            required_files = model_info.get('required_files', [])
            
            for file_pattern in required_files:
                found_files = list(model_path.rglob(file_pattern))
                if not found_files:
                    logger.warning(f"⚠️ 필수 파일 누락: {file_pattern}")
                    return False
            
            # 폴더 크기 확인 (대략적)
            total_size = sum(f.stat().st_size for f in model_path.rglob('*') if f.is_file())
            size_gb = total_size / (1024**3)
            
            expected_size = model_info['size_gb']
            if size_gb < expected_size * 0.8:  # 80% 이상이면 OK
                logger.warning(f"⚠️ 모델 크기 부족: {size_gb:.1f}GB < {expected_size:.1f}GB")
                return False
            
            return True
            
        except Exception as e:
            logger.warning(f"⚠️ 무결성 확인 실패: {e}")
            return False
    
    def create_model_config(self):
        """모델 설정 파일 생성"""
        logger.info("📝 모델 설정 파일 생성 중...")
        
        # YAML 설정 생성
        config_content = {
            "models": {},
            "device_config": {
                "auto_detect": True,
                "preferred_device": "mps",  # M3 Max 최적화
                "fallback_device": "cpu",
                "use_fp16": True,
                "batch_size": 4
            },
            "pipeline_config": {
                "image_size": [512, 512],
                "quality_level": "high",
                "enable_caching": True,
                "max_cache_size_gb": 8.0
            }
        }
        
        # 각 모델 설정 추가
        for model_key, model_info in self.models.items():
            config_content["models"][model_key] = {
                "name": model_info["name"],
                "path": f"ai_models/checkpoints/{model_info['local_dir']}",
                "step": model_info["step"],
                "enabled": True,
                "priority": model_info["priority"],
                "size_gb": model_info["size_gb"]
            }
        
        # YAML 파일로 저장
        config_path = self.configs_dir / "models_config.yaml"
        with open(config_path, 'w') as f:
            import yaml
            yaml.dump(config_content, f, default_flow_style=False, allow_unicode=True)
        
        logger.info(f"✅ 설정 파일 생성: {config_path}")
        
        # Python 설정 파일도 생성
        python_config_path = self.project_root / "app" / "core" / "model_paths.py"
        self.create_python_config(python_config_path)
    
    def create_python_config(self, output_path: Path):
        """Python 설정 파일 생성"""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        content = '''# app/core/model_paths.py
"""
AI 모델 경로 설정 - 자동 생성됨
"""

from pathlib import Path
from typing import Dict, Optional

# 기본 경로
AI_MODELS_ROOT = Path(__file__).parent.parent.parent / "ai_models"
CHECKPOINTS_ROOT = AI_MODELS_ROOT / "checkpoints"

# 다운로드된 모델들
DOWNLOADED_MODELS = {
'''
        
        # 각 모델 정보 추가
        for model_key, model_info in self.models.items():
            content += f'''    "{model_key}": {{
        "name": "{model_info['name']}",
        "path": CHECKPOINTS_ROOT / "{model_info['local_dir']}",
        "step": "{model_info['step']}",
        "priority": {model_info['priority']},
        "size_gb": {model_info['size_gb']},
        "enabled": True
    }},
'''
        
        content += '''}

def get_model_path(model_key: str) -> Optional[Path]:
    """모델 경로 반환"""
    model_info = DOWNLOADED_MODELS.get(model_key)
    if model_info:
        return model_info["path"]
    return None

def is_model_available(model_key: str) -> bool:
    """모델 사용 가능 여부 확인"""
    model_path = get_model_path(model_key)
    return model_path and model_path.exists()

def get_step_model(step_name: str) -> Optional[str]:
    """특정 단계의 모델 반환"""
    for model_key, model_info in DOWNLOADED_MODELS.items():
        if model_info["step"] == step_name:
            return model_key
    return None

def get_all_available_models() -> Dict[str, Dict]:
    """사용 가능한 모든 모델 반환"""
    return {
        key: info for key, info in DOWNLOADED_MODELS.items()
        if is_model_available(key)
    }
'''
        
        with open(output_path, 'w') as f:
            f.write(content)
        
        logger.info(f"✅ Python 설정 파일 생성: {output_path}")
    
    def show_download_summary(self):
        """다운로드 요약 표시"""
        logger.info("📊 다운로드 요약:")
        logger.info("=" * 60)
        
        total_downloaded = 0
        for model_key, model_info in self.models.items():
            local_path = self.checkpoints_dir / model_info['local_dir']
            
            if local_path.exists():
                size = sum(f.stat().st_size for f in local_path.rglob('*') if f.is_file())
                size_gb = size / (1024**3)
                total_downloaded += size_gb
                
                logger.info(f"✅ {model_info['name']}: {size_gb:.1f} GB")
            else:
                logger.info(f"❌ {model_info['name']}: 다운로드 안됨")
        
        logger.info(f"📦 총 다운로드 크기: {total_downloaded:.1f} GB")
    
    def run(self):
        """전체 다운로드 프로세스 실행"""
        logger.info("🤖 MyCloset AI 모델 다운로드 시작")
        logger.info("=" * 60)
        
        # 1. 의존성 확인
        if not self.check_dependencies():
            logger.error("❌ 의존성 확인 실패")
            return False
        
        # 2. 디스크 공간 확인
        if not self.check_disk_space():
            response = input("⚠️ 디스크 공간이 부족할 수 있습니다. 계속하시겠습니까? (y/N): ")
            if response.lower() != 'y':
                logger.info("❌ 다운로드 취소됨")
                return False
        
        # 3. 다운로드 확인
        logger.info(f"📥 다음 {len(self.models)}개 모델을 다운로드합니다:")
        for model_key, model_info in self.models.items():
            logger.info(f"   - {model_info['name']} ({model_info['size_gb']} GB)")
        
        logger.info(f"📦 총 크기: {self.total_size_gb:.1f} GB")
        logger.info("⏰ 예상 소요 시간: 30분 ~ 2시간")
        
        response = input("\n계속하시겠습니까? (y/N): ")
        if response.lower() != 'y':
            logger.info("❌ 다운로드 취소됨")
            return False
        
        # 4. 모델 다운로드 (우선순위순)
        logger.info("🚀 모델 다운로드 시작...")
        start_time = time.time()
        
        success_count = 0
        sorted_models = sorted(self.models.items(), key=lambda x: x[1]['priority'])
        
        for model_key, model_info in sorted_models:
            if self.download_model(model_key, model_info):
                success_count += 1
            
            # 진행률 표시
            progress = (success_count / len(self.models)) * 100
            logger.info(f"📊 진행률: {progress:.1f}% ({success_count}/{len(self.models)})")
        
        # 5. 설정 파일 생성
        self.create_model_config()
        
        # 6. 결과 출력
        elapsed_time = time.time() - start_time
        logger.info("=" * 60)
        logger.info(f"🎉 다운로드 완료! ({elapsed_time:.1f}초)")
        logger.info(f"✅ 성공: {success_count}/{len(self.models)} 모델")
        
        if success_count == len(self.models):
            logger.info("🚀 모든 모델 다운로드 완료! 이제 실제 AI 추론이 가능합니다!")
        else:
            logger.warning(f"⚠️ {len(self.models) - success_count}개 모델 다운로드 실패")
        
        self.show_download_summary()
        
        # 7. 다음 단계 안내
        logger.info("\n📋 다음 단계:")
        logger.info("1. python3 app/main.py  # 서버 실행")
        logger.info("2. http://localhost:8000/docs  # API 문서 확인")
        logger.info("3. 실제 AI 가상 피팅 테스트!")
        
        return success_count == len(self.models)

def main():
    """메인 함수"""
    try:
        downloader = AIModelDownloader()
        success = downloader.run()
        
        if success:
            print("\n🎉 모든 AI 모델 다운로드 완료!")
            print("이제 실제 AI 추론이 가능합니다!")
        else:
            print("\n⚠️ 일부 모델 다운로드 실패")
            print("다시 실행하거나 수동으로 다운로드하세요.")
            
    except KeyboardInterrupt:
        print("\n❌ 사용자에 의해 중단됨")
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")

if __name__ == "__main__":
    main()