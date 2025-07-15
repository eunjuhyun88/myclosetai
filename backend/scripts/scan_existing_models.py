#!/usr/bin/env python3
"""
🔍 기존 AI 모델 스캔 및 설정 생성기
현재 다운로드된 모델들을 인식하고 우리 ModelLoader와 연동
"""

import os
import json
import yaml
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
import time

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ExistingModelScanner:
    """기존 AI 모델 스캔 및 설정"""
    
    def __init__(self, ai_models_path: str = "ai_models"):
        self.ai_models_dir = Path(ai_models_path)
        self.scanned_models = {}
        self.model_configs = {}
        
        logger.info(f"📁 AI 모델 디렉토리: {self.ai_models_dir.absolute()}")
        
        if not self.ai_models_dir.exists():
            logger.error(f"❌ AI 모델 디렉토리를 찾을 수 없습니다: {self.ai_models_dir}")
            raise FileNotFoundError(f"AI 모델 디렉토리 없음: {self.ai_models_dir}")
    
    def scan_all_models(self) -> Dict[str, Any]:
        """모든 AI 모델 스캔"""
        logger.info("🔍 기존 AI 모델 스캔 시작...")
        
        # 현재 디렉토리 구조 출력
        self._show_directory_structure()
        
        # 각 모델별 스캔
        self._scan_ootdiffusion()
        self._scan_hr_viton()
        self._scan_graphonomy()
        self._scan_openpose()
        self._scan_detectron2()
        self._scan_self_correction_parsing()
        self._scan_checkpoints()
        self._scan_additional_files()
        
        # 스캔 결과 요약
        self._show_scan_summary()
        
        return self.scanned_models
    
    def _show_directory_structure(self):
        """현재 디렉토리 구조 표시"""
        logger.info("📂 현재 AI 모델 디렉토리 구조:")
        logger.info("=" * 50)
        
        for item in sorted(self.ai_models_dir.iterdir()):
            if item.is_dir():
                file_count = len(list(item.rglob("*")))
                size_mb = self._get_directory_size(item)
                logger.info(f"📁 {item.name}/ ({file_count} 항목, {size_mb} MB)")
            else:
                size_mb = round(item.stat().st_size / (1024 * 1024), 1)
                logger.info(f"📄 {item.name} ({size_mb} MB)")
    
    def _scan_ootdiffusion(self):
        """OOTDiffusion 모델 스캔"""
        possible_dirs = ["OOTDiffusion", "oot_diffusion", "ootdiffusion"]
        
        for dir_name in possible_dirs:
            model_dir = self.ai_models_dir / dir_name
            if model_dir.exists():
                logger.info(f"✅ OOTDiffusion 발견: {dir_name}")
                
                # 체크포인트 파일 찾기
                checkpoints = self._find_checkpoints(model_dir, [
                    "*.bin", "*.pth", "*.ckpt", "pytorch_model.bin"
                ])
                
                # 설정 파일 찾기
                config_files = self._find_configs(model_dir, [
                    "config.json", "model_index.json", "*.yaml", "*.yml"
                ])
                
                self.scanned_models["ootdiffusion"] = {
                    "name": "OOTDiffusion",
                    "type": "diffusion",
                    "step": "step_06_virtual_fitting",
                    "path": str(model_dir),
                    "checkpoints": checkpoints,
                    "configs": config_files,
                    "size_mb": self._get_directory_size(model_dir),
                    "files_count": len(list(model_dir.rglob("*"))),
                    "ready": len(checkpoints) > 0,
                    "priority": 1  # 최우선
                }
                break
    
    def _scan_hr_viton(self):
        """HR-VITON 모델 스캔"""
        model_dir = self.ai_models_dir / "HR-VITON"
        if model_dir.exists():
            logger.info(f"✅ HR-VITON 발견")
            
            # 서브모델들 찾기
            submodels = {
                "gmm": self._find_checkpoints(model_dir, ["*gmm*.pth", "*GMM*.pth"]),
                "tom": self._find_checkpoints(model_dir, ["*tom*.pth", "*TOM*.pth"]),
                "full": self._find_checkpoints(model_dir, ["*final*.pth", "*complete*.pth"])
            }
            
            self.scanned_models["hr_viton"] = {
                "name": "HR-VITON",
                "type": "virtual_tryon",
                "step": "step_05_cloth_warping",
                "path": str(model_dir),
                "submodels": submodels,
                "size_mb": self._get_directory_size(model_dir),
                "files_count": len(list(model_dir.rglob("*"))),
                "ready": any(len(files) > 0 for files in submodels.values()),
                "priority": 2
            }
    
    def _scan_graphonomy(self):
        """Graphonomy 모델 스캔"""
        model_dir = self.ai_models_dir / "Graphonomy"
        if model_dir.exists():
            logger.info(f"✅ Graphonomy 발견")
            
            checkpoints = self._find_checkpoints(model_dir, [
                "*.pth", "*.pt", "*inference*.pth", "*final*.pth"
            ])
            
            self.scanned_models["graphonomy"] = {
                "name": "Graphonomy",
                "type": "human_parsing",
                "step": "step_01_human_parsing",
                "path": str(model_dir),
                "checkpoints": checkpoints,
                "size_mb": self._get_directory_size(model_dir),
                "files_count": len(list(model_dir.rglob("*"))),
                "ready": len(checkpoints) > 0,
                "priority": 3
            }
    
    def _scan_openpose(self):
        """OpenPose 모델 스캔"""
        model_dir = self.ai_models_dir / "openpose"
        if model_dir.exists():
            logger.info(f"✅ OpenPose 발견")
            
            # Caffe 모델 찾기
            caffe_models = self._find_checkpoints(model_dir, [
                "*.caffemodel", "*.prototxt"
            ])
            
            # PyTorch 모델 찾기
            pytorch_models = self._find_checkpoints(model_dir, [
                "*.pth", "*.pt"
            ])
            
            self.scanned_models["openpose"] = {
                "name": "OpenPose",
                "type": "pose_estimation",
                "step": "step_02_pose_estimation",
                "path": str(model_dir),
                "caffe_models": caffe_models,
                "pytorch_models": pytorch_models,
                "size_mb": self._get_directory_size(model_dir),
                "files_count": len(list(model_dir.rglob("*"))),
                "ready": len(caffe_models) > 0 or len(pytorch_models) > 0,
                "priority": 4
            }
    
    def _scan_detectron2(self):
        """Detectron2 모델 스캔"""
        model_dir = self.ai_models_dir / "detectron2"
        if model_dir.exists():
            logger.info(f"✅ Detectron2 발견")
            
            # 사전 훈련된 모델들 찾기
            models = self._find_checkpoints(model_dir, [
                "*.pkl", "*.pth", "*.pt"
            ])
            
            self.scanned_models["detectron2"] = {
                "name": "Detectron2",
                "type": "detection_segmentation",
                "step": "auxiliary",
                "path": str(model_dir),
                "models": models,
                "size_mb": self._get_directory_size(model_dir),
                "files_count": len(list(model_dir.rglob("*"))),
                "ready": len(models) > 0,
                "priority": 5
            }
    
    def _scan_self_correction_parsing(self):
        """Self-Correction Human Parsing 모델 스캔"""
        model_dir = self.ai_models_dir / "Self-Correction-Human-Parsing"
        if model_dir.exists():
            logger.info(f"✅ Self-Correction Human Parsing 발견")
            
            checkpoints = self._find_checkpoints(model_dir, [
                "*.pth", "*.pt", "*parsing*.pth"
            ])
            
            self.scanned_models["self_correction_parsing"] = {
                "name": "Self-Correction Human Parsing",
                "type": "human_parsing",
                "step": "step_01_human_parsing",
                "path": str(model_dir),
                "checkpoints": checkpoints,
                "size_mb": self._get_directory_size(model_dir),
                "files_count": len(list(model_dir.rglob("*"))),
                "ready": len(checkpoints) > 0,
                "priority": 6
            }
    
    def _scan_checkpoints(self):
        """checkpoints 디렉토리 스캔"""
        checkpoints_dir = self.ai_models_dir / "checkpoints"
        if checkpoints_dir.exists():
            logger.info(f"✅ checkpoints 디렉토리 발견")
            
            # 각 서브디렉토리 스캔
            checkpoint_models = {}
            for subdir in checkpoints_dir.iterdir():
                if subdir.is_dir():
                    files = self._find_checkpoints(subdir, [
                        "*.pth", "*.pt", "*.bin", "*.ckpt", "*.pkl", "*.caffemodel"
                    ])
                    if files:
                        checkpoint_models[subdir.name] = {
                            "path": str(subdir),
                            "files": files,
                            "size_mb": self._get_directory_size(subdir)
                        }
            
            if checkpoint_models:
                self.scanned_models["checkpoints"] = {
                    "name": "Additional Checkpoints",
                    "type": "mixed",
                    "step": "auxiliary",
                    "path": str(checkpoints_dir),
                    "models": checkpoint_models,
                    "size_mb": self._get_directory_size(checkpoints_dir),
                    "files_count": len(list(checkpoints_dir.rglob("*"))),
                    "ready": True,
                    "priority": 7
                }
    
    def _scan_additional_files(self):
        """루트 디렉토리의 추가 모델 파일들"""
        additional_files = {}
        
        # 알려진 모델 파일 패턴들
        patterns = ["*.pth", "*.pt", "*.bin", "*.ckpt", "*.pkl"]
        
        for pattern in patterns:
            for file_path in self.ai_models_dir.glob(pattern):
                if file_path.is_file():
                    size_mb = round(file_path.stat().st_size / (1024 * 1024), 1)
                    additional_files[file_path.name] = {
                        "path": str(file_path),
                        "size_mb": size_mb
                    }
        
        if additional_files:
            logger.info(f"✅ 루트 디렉토리 추가 파일들 발견: {len(additional_files)}개")
            
            self.scanned_models["additional_files"] = {
                "name": "Additional Model Files",
                "type": "mixed",
                "step": "auxiliary",
                "path": str(self.ai_models_dir),
                "files": additional_files,
                "ready": True,
                "priority": 8
            }
    
    def _find_checkpoints(self, directory: Path, patterns: List[str]) -> List[str]:
        """체크포인트 파일들 찾기"""
        found_files = []
        for pattern in patterns:
            for file_path in directory.rglob(pattern):
                if file_path.is_file():
                    found_files.append(str(file_path.relative_to(directory)))
        return found_files
    
    def _find_configs(self, directory: Path, patterns: List[str]) -> List[str]:
        """설정 파일들 찾기"""
        return self._find_checkpoints(directory, patterns)
    
    def _get_directory_size(self, directory: Path) -> float:
        """디렉토리 크기 계산 (MB)"""
        total_size = 0
        try:
            for file_path in directory.rglob("*"):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
        except:
            pass
        return round(total_size / (1024 * 1024), 1)
    
    def _show_scan_summary(self):
        """스캔 결과 요약"""
        logger.info("\n📊 스캔 결과 요약:")
        logger.info("=" * 60)
        
        total_models = len(self.scanned_models)
        ready_models = sum(1 for model in self.scanned_models.values() if model.get("ready", False))
        total_size = sum(model.get("size_mb", 0) for model in self.scanned_models.values())
        
        logger.info(f"📦 발견된 모델: {total_models}개")
        logger.info(f"✅ 사용 가능: {ready_models}개")
        logger.info(f"💾 총 크기: {total_size:.1f} MB ({total_size/1024:.1f} GB)")
        
        logger.info(f"\n📋 발견된 모델 목록:")
        for model_key, model_info in self.scanned_models.items():
            status = "✅" if model_info.get("ready", False) else "⚠️"
            name = model_info.get("name", model_key)
            size = model_info.get("size_mb", 0)
            step = model_info.get("step", "unknown")
            logger.info(f"   {status} {name} ({size:.1f} MB) - {step}")
    
    def create_model_paths_config(self):
        """app/core/model_paths.py 생성"""
        logger.info("🐍 Python 모델 경로 설정 파일 생성 중...")
        
        config_content = '''# app/core/model_paths.py
"""
AI 모델 경로 설정 - 자동 생성됨
기존 다운로드된 모델들의 실제 경로 매핑
"""

from pathlib import Path
from typing import Dict, Optional, List

# 기본 경로
AI_MODELS_ROOT = Path(__file__).parent.parent.parent / "ai_models"

# 스캔된 모델 정보
SCANNED_MODELS = {
'''
        
        # 스캔된 모델들 추가
        for model_key, model_info in self.scanned_models.items():
            config_content += f'''    "{model_key}": {{
        "name": "{model_info['name']}",
        "type": "{model_info['type']}",
        "step": "{model_info['step']}",
        "path": AI_MODELS_ROOT / "{Path(model_info['path']).name}",
        "ready": {model_info['ready']},
        "size_mb": {model_info['size_mb']},
        "priority": {model_info.get('priority', 99)}
    }},
'''
        
        config_content += '''}

# 단계별 모델 매핑
STEP_TO_MODELS = {
    "step_01_human_parsing": ["graphonomy", "self_correction_parsing"],
    "step_02_pose_estimation": ["openpose"],
    "step_03_cloth_segmentation": [],  # U2Net 등 추가 필요
    "step_04_geometric_matching": [],  # HR-VITON GMM
    "step_05_cloth_warping": ["hr_viton"],  # HR-VITON TOM
    "step_06_virtual_fitting": ["ootdiffusion", "hr_viton"],
    "step_07_post_processing": [],
    "step_08_quality_assessment": []
}

def get_model_path(model_key: str) -> Optional[Path]:
    """모델 경로 반환"""
    if model_key in SCANNED_MODELS:
        return SCANNED_MODELS[model_key]["path"]
    return None

def is_model_ready(model_key: str) -> bool:
    """모델 사용 가능 여부 확인"""
    if model_key in SCANNED_MODELS:
        model_info = SCANNED_MODELS[model_key]
        return model_info["ready"] and model_info["path"].exists()
    return False

def get_ready_models() -> List[str]:
    """사용 가능한 모델 목록"""
    return [key for key, info in SCANNED_MODELS.items() if info["ready"]]

def get_models_for_step(step: str) -> List[str]:
    """특정 단계에 사용 가능한 모델들"""
    available_models = []
    for model_key in STEP_TO_MODELS.get(step, []):
        if is_model_ready(model_key):
            available_models.append(model_key)
    return available_models

def get_primary_model_for_step(step: str) -> Optional[str]:
    """단계별 주요 모델 반환 (우선순위 기준)"""
    models = get_models_for_step(step)
    if not models:
        return None
    
    # 우선순위로 정렬
    models_with_priority = [(model, SCANNED_MODELS[model]["priority"]) for model in models]
    models_with_priority.sort(key=lambda x: x[1])
    
    return models_with_priority[0][0] if models_with_priority else None

def get_ootdiffusion_path() -> Optional[Path]:
    """OOTDiffusion 경로 반환"""
    return get_model_path("ootdiffusion")

def get_hr_viton_path() -> Optional[Path]:
    """HR-VITON 경로 반환"""
    return get_model_path("hr_viton")

def get_graphonomy_path() -> Optional[Path]:
    """Graphonomy 경로 반환"""
    return get_model_path("graphonomy")

def get_openpose_path() -> Optional[Path]:
    """OpenPose 경로 반환"""
    return get_model_path("openpose")

def get_model_info(model_key: str) -> Optional[Dict]:
    """모델 상세 정보 반환"""
    return SCANNED_MODELS.get(model_key)

def list_all_models() -> Dict[str, Dict]:
    """모든 모델 정보 반환"""
    return SCANNED_MODELS.copy()
'''
        
        # app/core 디렉토리 생성
        app_core_dir = Path("app/core")
        app_core_dir.mkdir(parents=True, exist_ok=True)
        
        config_path = app_core_dir / "model_paths.py"
        with open(config_path, 'w', encoding='utf-8') as f:
            f.write(config_content)
        
        logger.info(f"✅ Python 설정 파일 생성: {config_path}")
    
    def create_yaml_config(self):
        """YAML 설정 파일 생성"""
        logger.info("📝 YAML 설정 파일 생성 중...")
        
        config = {
            "version": "1.0",
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "ai_models_directory": str(self.ai_models_dir.absolute()),
            "scanned_models": self.scanned_models,
            "model_loader_config": {
                "device": "mps",  # M3 Max 기본값
                "use_fp16": True,
                "max_cached_models": 8,
                "lazy_loading": True,
                "optimization_enabled": True
            },
            "pipeline_config": {
                "default_image_size": [512, 512],
                "batch_size": 1,
                "quality_level": "balanced"
            }
        }
        
        # YAML 파일 저장
        config_path = self.ai_models_dir / "scanned_models_config.yaml"
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        
        logger.info(f"✅ YAML 설정 파일 생성: {config_path}")
    
    def create_json_summary(self):
        """JSON 요약 파일 생성"""
        logger.info("📋 JSON 요약 파일 생성 중...")
        
        summary = {
            "scan_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_models": len(self.scanned_models),
            "ready_models": sum(1 for model in self.scanned_models.values() if model.get("ready", False)),
            "total_size_mb": sum(model.get("size_mb", 0) for model in self.scanned_models.values()),
            "models": {}
        }
        
        # 간단한 모델 정보만
        for model_key, model_info in self.scanned_models.items():
            summary["models"][model_key] = {
                "name": model_info["name"],
                "type": model_info["type"],
                "ready": model_info["ready"],
                "size_mb": model_info["size_mb"],
                "path": model_info["path"]
            }
        
        # JSON 파일 저장
        summary_path = self.ai_models_dir / "model_scan_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ JSON 요약 파일 생성: {summary_path}")
    
    def update_model_loader_registry(self):
        """ModelLoader용 레지스트리 업데이트 파일 생성"""
        logger.info("🔧 ModelLoader 레지스트리 업데이트 파일 생성 중...")
        
        registry_update = '''# app/ai_pipeline/utils/model_registry_update.py
"""
스캔된 모델들을 ModelLoader에 자동 등록
"""

from app.ai_pipeline.utils.model_loader import ModelConfig, ModelType
from pathlib import Path

def update_model_registry(model_loader):
    """스캔된 모델들을 ModelLoader에 등록"""
    
    # 기본 경로
    ai_models_root = Path("ai_models")
    
    # 스캔된 모델들 등록
'''
        
        # 각 모델별 등록 코드 생성
        for model_key, model_info in self.scanned_models.items():
            if not model_info.get("ready", False):
                continue
                
            model_type = self._map_to_model_type(model_info["type"])
            if model_type:
                registry_update += f'''
    # {model_info["name"]}
    model_loader.register_model(
        "{model_key}",
        ModelConfig(
            name="{model_info['name']}",
            model_type=ModelType.{model_type},
            model_class="{self._get_model_class(model_info['type'])}",
            checkpoint_path=str(ai_models_root / "{Path(model_info['path']).name}"),
            input_size=(512, 512),
            device="mps"
        )
    )'''
        
        registry_update += '''

def get_available_models():
    """사용 가능한 모델 목록 반환"""
    return ['''
        
        for model_key, model_info in self.scanned_models.items():
            if model_info.get("ready", False):
                registry_update += f'"{model_key}", '
        
        registry_update += ''']

# 사용 예시:
# from app.ai_pipeline.utils.model_loader import ModelLoader
# from app.ai_pipeline.utils.model_registry_update import update_model_registry
# 
# loader = ModelLoader()
# update_model_registry(loader)
# model = await loader.load_model("ootdiffusion")
'''
        
        # 파일 저장
        registry_path = Path("app/ai_pipeline/utils/model_registry_update.py")
        registry_path.parent.mkdir(parents=True, exist_ok=True)
        with open(registry_path, 'w', encoding='utf-8') as f:
            f.write(registry_update)
        
        logger.info(f"✅ ModelLoader 레지스트리 업데이트 파일 생성: {registry_path}")
    
    def _map_to_model_type(self, scan_type: str) -> Optional[str]:
        """스캔 타입을 ModelType으로 매핑"""
        mapping = {
            "diffusion": "DIFFUSION",
            "virtual_tryon": "VIRTUAL_FITTING",
            "human_parsing": "HUMAN_PARSING",
            "pose_estimation": "POSE_ESTIMATION",
            "detection_segmentation": "SEGMENTATION"
        }
        return mapping.get(scan_type)
    
    def _get_model_class(self, scan_type: str) -> str:
        """스캔 타입에서 모델 클래스명 추출"""
        mapping = {
            "diffusion": "StableDiffusionPipeline",
            "virtual_tryon": "HRVITONModel",
            "human_parsing": "GraphonomyModel",
            "pose_estimation": "OpenPoseModel",
            "detection_segmentation": "DetectronModel"
        }
        return mapping.get(scan_type, "BaseModel")

def main():
    """메인 함수"""
    print("🔍 MyCloset AI - 기존 모델 스캔 및 설정 생성")
    print("=" * 50)
    
    try:
        # 현재 디렉토리에서 ai_models 찾기
        scanner = ExistingModelScanner("ai_models")
        
        # 모델 스캔
        scanned_models = scanner.scan_all_models()
        
        if not scanned_models:
            logger.warning("⚠️ 사용 가능한 AI 모델을 찾을 수 없습니다.")
            return False
        
        # 설정 파일들 생성
        scanner.create_model_paths_config()
        scanner.create_yaml_config()
        scanner.create_json_summary()
        scanner.update_model_loader_registry()
        
        print(f"\n🎉 모델 스캔 및 설정 생성 완료!")
        print(f"📊 총 {len(scanned_models)}개 모델 발견")
        print(f"✅ 사용 가능한 모델: {sum(1 for m in scanned_models.values() if m.get('ready', False))}개")
        
        # 다음 단계 안내
        print(f"\n📝 생성된 파일들:")
        print(f"   - app/core/model_paths.py (Python 설정)")
        print(f"   - ai_models/scanned_models_config.yaml (YAML 설정)")
        print(f"   - ai_models/model_scan_summary.json (JSON 요약)")
        print(f"   - app/ai_pipeline/utils/model_registry_update.py (ModelLoader 연동)")
        
        print(f"\n🚀 다음 단계:")
        print(f"   1. app/core/model_paths.py를 import 하여 모델 경로 사용")
        print(f"   2. ModelLoader에서 update_model_registry() 함수 호출")
        print(f"   3. 각 step에서 get_primary_model_for_step() 사용")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 스캔 실패: {e}")
        return False

if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)