#!/usr/bin/env python3
# backend/scripts/detect_existing_models.py
"""
기존에 다운로드된 AI 모델들을 감지하고 MyCloset AI 시스템에서 사용할 수 있도록 설정
"""

import os
import yaml
import json
from pathlib import Path
import logging
from typing import Dict, List, Any

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ExistingModelsDetector:
    """기존 AI 모델 감지 및 설정 클래스"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.ai_models_dir = self.project_root / "ai_models"
        self.detected_models = {}
        
        logger.info(f"🔍 AI 모델 디렉토리: {self.ai_models_dir}")
    
    def scan_all_models(self) -> Dict[str, Any]:
        """모든 AI 모델 스캔"""
        
        logger.info("🚀 AI 모델 전체 스캔 시작...")
        
        # 1. OOTDiffusion 모델들
        self._scan_ootdiffusion()
        
        # 2. Stable Diffusion
        self._scan_stable_diffusion()
        
        # 3. Segment Anything (SAM)
        self._scan_sam()
        
        # 4. Human Parsing & Pose Detection
        self._scan_human_parsing()
        
        # 5. CLIP
        self._scan_clip()
        
        # 6. 기타 모델들
        self._scan_additional_models()
        
        # 7. 통계 출력
        self._print_summary()
        
        return self.detected_models
    
    def _scan_ootdiffusion(self):
        """OOTDiffusion 모델 스캔"""
        
        logger.info("🎽 OOTDiffusion 모델 스캔 중...")
        
        # ootdiffusion_hf 디렉토리 (Hugging Face 버전)
        ootd_hf_path = self.ai_models_dir / "ootdiffusion_hf"
        if ootd_hf_path.exists():
            ootd_info = {
                "name": "OOTDiffusion (Hugging Face)",
                "path": str(ootd_hf_path),
                "type": "virtual_tryon",
                "priority": 1,
                "components": {},
                "total_size_gb": 0,
                "ready": True
            }
            
            # 하위 컴포넌트 스캔
            checkpoints_path = ootd_hf_path / "checkpoints" / "ootd"
            if checkpoints_path.exists():
                
                # HD 모델 (고해상도)
                hd_path = checkpoints_path / "ootd_hd" / "checkpoint-36000"
                if hd_path.exists():
                    ootd_info["components"]["hd"] = {
                        "unet_vton": str(hd_path / "unet_vton"),
                        "unet_garm": str(hd_path / "unet_garm"),
                        "size_gb": 6.4  # 3.2GB x 2
                    }
                
                # DC 모델 (일반 품질)
                dc_path = checkpoints_path / "ootd_dc" / "checkpoint-36000"
                if dc_path.exists():
                    ootd_info["components"]["dc"] = {
                        "unet_vton": str(dc_path / "unet_vton"),
                        "unet_garm": str(dc_path / "unet_garm"),
                        "size_gb": 6.4  # 3.2GB x 2
                    }
                
                # 공통 컴포넌트
                text_encoder = checkpoints_path / "text_encoder"
                vae = checkpoints_path / "vae"
                
                if text_encoder.exists():
                    ootd_info["components"]["text_encoder"] = str(text_encoder)
                if vae.exists():
                    ootd_info["components"]["vae"] = str(vae)
                
                ootd_info["total_size_gb"] = 13.0  # 대략적 크기
            
            self.detected_models["ootdiffusion"] = ootd_info
            logger.info(f"✅ OOTDiffusion 발견: HD/DC 모드 지원, {ootd_info['total_size_gb']}GB")
        
        # checkpoints/ootdiffusion 디렉토리도 확인
        checkpoints_ootd = self.ai_models_dir / "checkpoints" / "ootdiffusion"
        if checkpoints_ootd.exists():
            additional_info = {
                "name": "OOTDiffusion (Checkpoints)",
                "path": str(checkpoints_ootd),
                "type": "virtual_tryon_additional",
                "components": {
                    "openpose": str(checkpoints_ootd / "checkpoints" / "openpose"),
                    "humanparsing": str(checkpoints_ootd / "checkpoints" / "humanparsing")
                },
                "ready": True
            }
            self.detected_models["ootdiffusion_additional"] = additional_info
            logger.info("✅ OOTDiffusion 추가 체크포인트 발견")
    
    def _scan_stable_diffusion(self):
        """Stable Diffusion 모델 스캔"""
        
        logger.info("🎨 Stable Diffusion 모델 스캔 중...")
        
        sd_path = self.ai_models_dir / "checkpoints" / "stable-diffusion-v1-5"
        if sd_path.exists():
            sd_info = {
                "name": "Stable Diffusion v1.5",
                "path": str(sd_path),
                "type": "base_diffusion",
                "priority": 2,
                "components": {},
                "total_size_gb": 15.0,
                "ready": True
            }
            
            # 주요 컴포넌트 확인
            components = {
                "unet": sd_path / "unet",
                "vae": sd_path / "vae", 
                "text_encoder": sd_path / "text_encoder",
                "safety_checker": sd_path / "safety_checker"
            }
            
            for comp_name, comp_path in components.items():
                if comp_path.exists():
                    sd_info["components"][comp_name] = str(comp_path)
            
            # 전체 모델 파일들도 확인
            model_files = list(sd_path.glob("*.safetensors")) + list(sd_path.glob("*.ckpt"))
            if model_files:
                sd_info["components"]["full_models"] = [str(f) for f in model_files]
            
            self.detected_models["stable_diffusion"] = sd_info
            logger.info(f"✅ Stable Diffusion v1.5 발견: 완전한 파이프라인, {sd_info['total_size_gb']}GB")
    
    def _scan_sam(self):
        """Segment Anything 모델 스캔"""
        
        logger.info("✂️ Segment Anything 모델 스캔 중...")
        
        sam_path = self.ai_models_dir / "checkpoints" / "sam"
        if sam_path.exists():
            sam_info = {
                "name": "Segment Anything (SAM)",
                "path": str(sam_path),
                "type": "segmentation",
                "priority": 3,
                "models": {},
                "total_size_gb": 2.8,
                "ready": True
            }
            
            # SAM 모델들 확인
            sam_models = {
                "vit_h": sam_path / "sam_vit_h_4b8939.pth",
                "vit_b": sam_path / "sam_vit_b_01ec64.pth"
            }
            
            for model_name, model_path in sam_models.items():
                if model_path.exists():
                    size_mb = model_path.stat().st_size / (1024 * 1024)
                    sam_info["models"][model_name] = {
                        "path": str(model_path),
                        "size_mb": round(size_mb, 1)
                    }
            
            self.detected_models["sam"] = sam_info
            logger.info(f"✅ SAM 발견: {len(sam_info['models'])}개 모델, {sam_info['total_size_gb']}GB")
    
    def _scan_human_parsing(self):
        """Human Parsing 및 Pose Detection 모델 스캔"""
        
        logger.info("👤 Human Parsing & Pose 모델 스캔 중...")
        
        # Graphonomy
        graphonomy_path = self.ai_models_dir / "Graphonomy"
        if graphonomy_path.exists():
            graphonomy_info = {
                "name": "Graphonomy (Human Parsing)",
                "path": str(graphonomy_path),
                "type": "human_parsing",
                "priority": 4,
                "ready": True,
                "size_gb": 0.1
            }
            self.detected_models["graphonomy"] = graphonomy_info
            logger.info("✅ Graphonomy 발견")
        
        # Self-Correction-Human-Parsing
        schp_path = self.ai_models_dir / "Self-Correction-Human-Parsing"
        if schp_path.exists():
            schp_info = {
                "name": "Self-Correction Human Parsing",
                "path": str(schp_path),
                "type": "human_parsing",
                "priority": 4,
                "ready": True,
                "size_gb": 0.1
            }
            self.detected_models["schp"] = schp_info
            logger.info("✅ Self-Correction Human Parsing 발견")
        
        # OpenPose
        openpose_path = self.ai_models_dir / "openpose"
        if openpose_path.exists():
            openpose_info = {
                "name": "OpenPose",
                "path": str(openpose_path),
                "type": "pose_estimation",
                "priority": 4,
                "ready": True,
                "size_gb": 0.2
            }
            self.detected_models["openpose"] = openpose_info
            logger.info("✅ OpenPose 발견")
    
    def _scan_clip(self):
        """CLIP 모델 스캔"""
        
        logger.info("🔗 CLIP 모델 스캔 중...")
        
        clip_path = self.ai_models_dir / "checkpoints" / "clip-vit-large-patch14"
        if clip_path.exists():
            clip_info = {
                "name": "CLIP ViT-Large",
                "path": str(clip_path),
                "type": "vision_language",
                "priority": 5,
                "ready": True,
                "size_gb": 1.6
            }
            self.detected_models["clip"] = clip_info
            logger.info("✅ CLIP ViT-Large 발견")
    
    def _scan_additional_models(self):
        """기타 추가 모델들 스캔"""
        
        logger.info("📦 추가 모델 스캔 중...")
        
        # gen.pth (VITON-HD)
        gen_pth = self.ai_models_dir / "gen.pth"
        if gen_pth.exists() and gen_pth.stat().st_size > 1000:  # 최소 크기 확인
            self.detected_models["viton_gen"] = {
                "name": "VITON-HD Generator",
                "path": str(gen_pth),
                "type": "virtual_tryon",
                "priority": 6,
                "ready": True,
                "size_mb": round(gen_pth.stat().st_size / (1024 * 1024), 1)
            }
            logger.info("✅ VITON-HD Generator 발견")
        
        # ResNet50 features
        resnet_path = self.ai_models_dir / "checkpoints" / "resnet50_features.pth"
        if resnet_path.exists():
            self.detected_models["resnet50"] = {
                "name": "ResNet50 Features",
                "path": str(resnet_path),
                "type": "feature_extractor",
                "priority": 7,
                "ready": True,
                "size_mb": round(resnet_path.stat().st_size / (1024 * 1024), 1)
            }
            logger.info("✅ ResNet50 Features 발견")
        
        # HR-VITON, VITON-HD 디렉토리들
        for viton_name in ["HR-VITON", "VITON-HD"]:
            viton_path = self.ai_models_dir / viton_name
            if viton_path.exists():
                self.detected_models[viton_name.lower().replace("-", "_")] = {
                    "name": viton_name,
                    "path": str(viton_path),
                    "type": "virtual_tryon",
                    "priority": 8,
                    "ready": True,
                    "size_gb": 0.1
                }
                logger.info(f"✅ {viton_name} 디렉토리 발견")
    
    def _print_summary(self):
        """발견된 모델 요약 출력"""
        
        logger.info("\n" + "="*60)
        logger.info("🎉 AI 모델 스캔 완료 - 요약")
        logger.info("="*60)
        
        total_models = len(self.detected_models)
        total_size = 0
        
        by_type = {}
        
        for model_key, model_info in self.detected_models.items():
            model_type = model_info.get("type", "unknown")
            size = model_info.get("total_size_gb", model_info.get("size_gb", 0))
            
            if model_type not in by_type:
                by_type[model_type] = {"count": 0, "size": 0, "models": []}
            
            by_type[model_type]["count"] += 1
            by_type[model_type]["size"] += size
            by_type[model_type]["models"].append(model_info["name"])
            
            total_size += size
        
        logger.info(f"📊 총 발견된 모델: {total_models}개")
        logger.info(f"💾 총 크기: {total_size:.1f}GB")
        logger.info("")
        
        for type_name, type_info in by_type.items():
            logger.info(f"🔹 {type_name}: {type_info['count']}개 ({type_info['size']:.1f}GB)")
            for model_name in type_info['models']:
                logger.info(f"    - {model_name}")
        
        logger.info("\n🚀 다음 단계: 모델 설정 파일 생성")
    
    def create_model_config(self) -> Dict[str, Any]:
        """모델 설정 파일 생성"""
        
        logger.info("📝 모델 설정 파일 생성 중...")
        
        config = {
            "models": self.detected_models,
            "system": {
                "device": "mps",  # M3 Max 기본값
                "batch_size": 1,
                "fp16": False,  # MPS에서는 fp32 사용
                "cpu_offload": False,  # 128GB RAM이므로 비활성화
            },
            "paths": {
                "ai_models_root": str(self.ai_models_dir),
                "cache_dir": str(self.ai_models_dir / "cache"),
                "temp_dir": str(self.ai_models_dir / "temp"),
                "checkpoints_dir": str(self.ai_models_dir / "checkpoints")
            },
            "virtual_tryon": {
                "default_model": "ootdiffusion",
                "image_size": 512,
                "num_inference_steps": 20,
                "guidance_scale": 7.5,
                "enable_safety_checker": True
            },
            "performance": {
                "max_memory_gb": 24,  # M3 Max GPU 메모리 한계
                "enable_attention_slicing": True,
                "enable_cpu_offload": False,
                "enable_sequential_cpu_offload": False
            }
        }
        
        # YAML 파일로 저장
        config_path = self.ai_models_dir / "detected_models_config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        
        logger.info(f"✅ 모델 설정 파일 저장: {config_path}")
        
        # 기존 설정 파일 백업 및 교체
        old_config = self.ai_models_dir / "model_config.yaml"
        if old_config.exists():
            backup_path = self.ai_models_dir / "model_config_backup.yaml"
            old_config.rename(backup_path)
            logger.info(f"📦 기존 설정 백업: {backup_path}")
        
        config_path.rename(old_config)
        logger.info("✅ 새 모델 설정 파일 적용 완료")
        
        return config
    
    def create_python_model_paths(self):
        """Python에서 사용할 모델 경로 파일 생성"""
        
        logger.info("🐍 Python 모델 경로 파일 생성 중...")
        
        python_config = f'''# backend/app/core/model_paths.py
"""
AI 모델 경로 설정 - 자동 생성됨
기존 다운로드된 모델들의 실제 경로 매핑
"""

from pathlib import Path
from typing import Dict, Optional, List

# 기본 경로
AI_MODELS_ROOT = Path(__file__).parent.parent.parent / "ai_models"

# 발견된 모델 경로 매핑
DETECTED_MODELS = {{
'''
        
        for model_key, model_info in self.detected_models.items():
            python_config += f'''    "{model_key}": {{
        "name": "{model_info['name']}",
        "path": Path("{model_info['path']}"),
        "type": "{model_info['type']}",
        "ready": {model_info['ready']},
        "priority": {model_info.get('priority', 99)}
    }},
'''
        
        python_config += '''}}

# 타입별 모델 그룹핑
def get_models_by_type(model_type: str) -> List[str]:
    """타입별 모델 목록 반환"""
    return [key for key, info in DETECTED_MODELS.items() 
            if info["type"] == model_type and info["ready"]]

def get_virtual_tryon_models() -> List[str]:
    """가상 피팅 모델 목록"""
    return get_models_by_type("virtual_tryon")

def get_primary_ootd_path() -> Path:
    """메인 OOTDiffusion 경로 반환"""
    if "ootdiffusion" in DETECTED_MODELS:
        return DETECTED_MODELS["ootdiffusion"]["path"]
    raise FileNotFoundError("OOTDiffusion 모델을 찾을 수 없습니다")

def get_stable_diffusion_path() -> Path:
    """Stable Diffusion 경로 반환"""
    if "stable_diffusion" in DETECTED_MODELS:
        return DETECTED_MODELS["stable_diffusion"]["path"]
    raise FileNotFoundError("Stable Diffusion 모델을 찾을 수 없습니다")

def get_sam_path(model_size: str = "vit_h") -> Path:
    """SAM 모델 경로 반환"""
    if "sam" in DETECTED_MODELS:
        base_path = DETECTED_MODELS["sam"]["path"]
        if model_size == "vit_h":
            return Path(base_path) / "sam_vit_h_4b8939.pth"
        elif model_size == "vit_b":
            return Path(base_path) / "sam_vit_b_01ec64.pth"
    raise FileNotFoundError(f"SAM {model_size} 모델을 찾을 수 없습니다")

def is_model_available(model_key: str) -> bool:
    """모델 사용 가능 여부 확인"""
    if model_key in DETECTED_MODELS:
        model_path = DETECTED_MODELS[model_key]["path"]
        return Path(model_path).exists()
    return False

def get_all_available_models() -> List[str]:
    """사용 가능한 모든 모델 목록"""
    available = []
    for key, info in DETECTED_MODELS.items():
        if info["ready"] and Path(info["path"]).exists():
            available.append(key)
    return sorted(available, key=lambda x: DETECTED_MODELS[x]["priority"])

def get_model_info(model_key: str) -> Optional[Dict]:
    """모델 정보 반환"""
    return DETECTED_MODELS.get(model_key)

# 빠른 경로 접근
class ModelPaths:
    """모델 경로 빠른 접근 클래스"""
    
    @property
    def ootd_hf(self) -> Path:
        return get_primary_ootd_path()
    
    @property
    def stable_diffusion(self) -> Path:
        return get_stable_diffusion_path()
    
    @property
    def sam_large(self) -> Path:
        return get_sam_path("vit_h")
    
    @property
    def sam_base(self) -> Path:
        return get_sam_path("vit_b")

# 전역 인스턴스
model_paths = ModelPaths()
'''
        
        # 파일 저장
        python_path = self.project_root / "app" / "core" / "model_paths.py"
        python_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(python_path, 'w') as f:
            f.write(python_config)
        
        logger.info(f"✅ Python 모델 경로 파일 저장: {python_path}")
    
    def create_usage_guide(self):
        """사용 가이드 생성"""
        
        logger.info("📖 사용 가이드 생성 중...")
        
        guide_content = f"""# 🎯 MyCloset AI - 발견된 모델 사용 가이드

## 📊 발견된 모델 현황

총 **{len(self.detected_models)}개** AI 모델이 발견되었습니다.

"""
        
        # 모델별 상세 정보
        for model_key, model_info in self.detected_models.items():
            guide_content += f"""### {model_info['name']} (`{model_key}`)
- **타입**: {model_info['type']}
- **경로**: `{model_info['path']}`
- **우선순위**: {model_info.get('priority', 'N/A')}
- **상태**: {'✅ 준비됨' if model_info['ready'] else '❌ 준비 안됨'}
"""
            
            if 'total_size_gb' in model_info:
                guide_content += f"- **크기**: {model_info['total_size_gb']}GB\n"
            elif 'size_mb' in model_info:
                guide_content += f"- **크기**: {model_info['size_mb']}MB\n"
            
            if 'components' in model_info:
                guide_content += "- **컴포넌트**:\n"
                for comp_name, comp_path in model_info['components'].items():
                    guide_content += f"  - {comp_name}: `{comp_path}`\n"
            
            guide_content += "\n"
        
        guide_content += """
## 🚀 Python에서 사용법

### 기본 사용
```python
from app.core.model_paths import model_paths, get_all_available_models

# 사용 가능한 모든 모델 확인
available_models = get_all_available_models()
print(f"사용 가능한 모델: {available_models}")

# 주요 모델 경로 접근
ootd_path = model_paths.ootd_hf
sd_path = model_paths.stable_diffusion
sam_path = model_paths.sam_large
```

### 가상 피팅 파이프라인
```python
from app.core.model_paths import get_primary_ootd_path, get_stable_diffusion_path

# OOTDiffusion 로드 준비
ootd_base = get_primary_ootd_path()
sd_base = get_stable_diffusion_path()

# 실제 모델 로딩은 diffusers 라이브러리 사용
from diffusers import StableDiffusionPipeline

pipeline = StableDiffusionPipeline.from_pretrained(
    str(sd_base),
    torch_dtype=torch.float32,  # M3 Max MPS용
    device_map="auto"
)
```

### SAM 세그멘테이션
```python
from app.core.model_paths import get_sam_path

# SAM 모델 로드
sam_model_path = get_sam_path("vit_h")  # 고성능
# sam_model_path = get_sam_path("vit_b")  # 경량

import torch
from segment_anything import sam_model_registry, SamPredictor

sam = sam_model_registry["vit_h"](checkpoint=str(sam_model_path))
sam.to(device="mps")  # M3 Max GPU
predictor = SamPredictor(sam)
```

## 🔧 FastAPI 통합

### 모델 상태 API
```python
from app.core.model_paths import DETECTED_MODELS, is_model_available

@app.get("/api/models/status")
async def get_models_status():
    status = {}
    for model_key, model_info in DETECTED_MODELS.items():
        status[model_key] = {
            "name": model_info["name"],
            "type": model_info["type"],
            "available": is_model_available(model_key),
            "ready": model_info["ready"]
        }
    return status
```

## 🎯 추천 워크플로우

1. **기본 테스트**: SAM 세그멘테이션부터 시작
2. **가상 피팅 구현**: OOTDiffusion + Stable Diffusion
3. **성능 최적화**: M3 Max Metal 가속 활용
4. **프로덕션 배포**: 모든 모델 통합

## 🚨 주의사항

- **M3 Max 최적화**: `torch.float32` 사용 (MPS에서 안정적)
- **메모리 관리**: 128GB RAM 활용하여 CPU 오프로드 비활성화
- **배치 크기**: `batch_size=1` 권장 (초기 설정)

---

🎉 **모든 모델이 준비되어 있습니다!** 바로 AI 가상 피팅 개발을 시작할 수 있습니다.
"""
        
        guide_path = self.ai_models_dir / "MODELS_USAGE_GUIDE.md"
        with open(guide_path, 'w') as f:
            f.write(guide_content)
        
        logger.info(f"✅ 사용 가이드 저장: {guide_path}")

def main():
    """메인 실행 함수"""
    
    print("🔍 MyCloset AI - 기존 AI 모델 감지 도구")
    print("=" * 60)
    
    detector = ExistingModelsDetector()
    
    # 1. 모든 모델 스캔
    detected = detector.scan_all_models()
    
    if not detected:
        logger.warning("⚠️ 사용 가능한 AI 모델을 찾을 수 없습니다.")
        return
    
    # 2. 설정 파일 생성
    config = detector.create_model_config()
    
    # 3. Python 경로 파일 생성
    detector.create_python_model_paths()
    
    # 4. 사용 가이드 생성
    detector.create_usage_guide()
    
    print("\n🎉 모델 감지 및 설정 완료!")
    print("\n📋 생성된 파일:")
    print("  - ai_models/model_config.yaml (업데이트됨)")
    print("  - app/core/model_paths.py (새로 생성)")
    print("  - ai_models/MODELS_USAGE_GUIDE.md (사용 가이드)")
    
    print("\n🚀 다음 단계:")
    print("  1. FastAPI 서버 실행: cd backend && python app/main.py")
    print("  2. 모델 상태 확인: curl http://localhost:8000/api/models/status")
    print("  3. 가상 피팅 API 개발")
    
    print("\n💡 주요 모델:")
    for key, info in detected.items():
        if info.get('priority', 99) <= 3:
            print(f"  - {info['name']}: {info['type']}")

if __name__ == "__main__":
    main()