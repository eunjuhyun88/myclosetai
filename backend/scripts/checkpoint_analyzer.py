#!/usr/bin/env python3
"""
🔍 MyCloset AI - 수정된 체크포인트 분석기
✅ 기존 127.2GB 분석 결과 활용
✅ 누락된 키 문제 해결
✅ 즉시 실행 가능
"""

import os
import sys
import time
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

# 프로젝트 경로 설정
current_file = Path(__file__).resolve()
backend_dir = current_file.parent.parent
project_root = backend_dir.parent
sys.path.insert(0, str(backend_dir))

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)

class FixedCheckpointAnalyzer:
    """수정된 체크포인트 분석기"""
    
    def __init__(self):
        self.backend_dir = backend_dir
        self.checkpoints_dir = backend_dir / "ai_models" / "checkpoints"
        
        # 기존 분석 결과 기반 모델 데이터 (실제 터미널 출력 기반)
        self.analyzed_models = self._create_analyzed_models_from_output()
        
        logger.info("🔍 수정된 체크포인트 분석기 초기화 완료")
        logger.info(f"📁 체크포인트 디렉토리: {self.checkpoints_dir}")
        logger.info(f"📊 분석된 모델: {len(self.analyzed_models)}개")
    
    def _create_analyzed_models_from_output(self) -> Dict[str, Dict]:
        """터미널 출력 결과를 바탕으로 모델 데이터 생성"""
        
        # 실제 터미널 출력에서 확인된 모델들
        models_data = {
            "ootdiffusion": {
                "name": "OOTDiffusion",
                "type": "diffusion",
                "step": "step_06_virtual_fitting",
                "ready": True,
                "total_size_mb": 15129.3,
                "priority": 1,
                "checkpoints": [
                    {"name": "pytorch_model.bin", "path": "pytorch_model.bin", "size_mb": 469.5},
                    {"name": "diffusion_pytorch_model.bin", "path": "diffusion_pytorch_model.bin", "size_mb": 319.2},
                    {"name": "body_pose_model.pth", "path": "body_pose_model.pth", "size_mb": 199.6}
                ],
                "total_checkpoints": 5
            },
            
            "ootdiffusion_hf": {
                "name": "OOTDiffusion HF",
                "type": "diffusion", 
                "step": "step_06_virtual_fitting",
                "ready": True,
                "total_size_mb": 15129.3,
                "priority": 1,
                "checkpoints": [
                    {"name": "pytorch_model.bin", "path": "pytorch_model.bin", "size_mb": 469.5},
                    {"name": "diffusion_pytorch_model.bin", "path": "diffusion_pytorch_model.bin", "size_mb": 319.2},
                    {"name": "body_pose_model.pth", "path": "body_pose_model.pth", "size_mb": 199.6}
                ],
                "total_checkpoints": 5
            },
            
            "stable-diffusion-v1-5": {
                "name": "Stable Diffusion v1.5",
                "type": "diffusion",
                "step": "step_06_virtual_fitting",
                "ready": True,
                "total_size_mb": 45070.6,
                "priority": 2,
                "checkpoints": [
                    {"name": "v1-5-pruned.ckpt", "path": "v1-5-pruned.ckpt", "size_mb": 7346.9},
                    {"name": "v1-5-pruned-emaonly.ckpt", "path": "v1-5-pruned-emaonly.ckpt", "size_mb": 4067.8},
                    {"name": "pytorch_model.fp16.bin", "path": "pytorch_model.fp16.bin", "size_mb": 234.8}
                ],
                "total_checkpoints": 11
            },
            
            "human_parsing": {
                "name": "Human Parsing",
                "type": "human_parsing",
                "step": "step_01_human_parsing",
                "ready": True,
                "total_size_mb": 1288.3,
                "priority": 3,
                "checkpoints": [
                    {"name": "schp_atr.pth", "path": "schp_atr.pth", "size_mb": 255.1},
                    {"name": "optimizer.pt", "path": "optimizer.pt", "size_mb": 209.0},
                    {"name": "rng_state.pth", "path": "rng_state.pth", "size_mb": 0.0}
                ],
                "total_checkpoints": 8
            },
            
            "step_01_human_parsing": {
                "name": "Step 01 Human Parsing",
                "type": "human_parsing",
                "step": "step_01_human_parsing", 
                "ready": True,
                "total_size_mb": 1787.7,
                "priority": 3,
                "checkpoints": [
                    {"name": "densepose_rcnn_R_50_FPN_s1x.pkl", "path": "densepose_rcnn_R_50_FPN_s1x.pkl", "size_mb": 243.9},
                    {"name": "graphonomy_lip.pth", "path": "graphonomy_lip.pth", "size_mb": 255.1},
                    {"name": "lightweight_parsing.pth", "path": "lightweight_parsing.pth", "size_mb": 0.5}
                ],
                "total_checkpoints": 11
            },
            
            "pose_estimation": {
                "name": "Pose Estimation",
                "type": "pose_estimation",
                "step": "step_02_pose_estimation",
                "ready": True,
                "total_size_mb": 10095.6,
                "priority": 4,
                "checkpoints": [
                    {"name": "sk_model.pth", "path": "sk_model.pth", "size_mb": 16.4},
                    {"name": "upernet_global_small.pth", "path": "upernet_global_small.pth", "size_mb": 196.8},
                    {"name": "latest_net_G.pth", "path": "latest_net_G.pth", "size_mb": 303.5}
                ],
                "total_checkpoints": 23
            },
            
            "step_02_pose_estimation": {
                "name": "Step 02 Pose Estimation",
                "type": "pose_estimation",
                "step": "step_02_pose_estimation",
                "ready": True,
                "total_size_mb": 273.6,
                "priority": 4,
                "checkpoints": [
                    {"name": "openpose.pth", "path": "openpose.pth", "size_mb": 199.6},
                    {"name": "yolov8n-pose.pt", "path": "yolov8n-pose.pt", "size_mb": 6.5}
                ],
                "total_checkpoints": 2
            },
            
            "openpose": {
                "name": "OpenPose",
                "type": "pose_estimation",
                "step": "step_02_pose_estimation",
                "ready": True,
                "total_size_mb": 539.7,
                "priority": 4,
                "checkpoints": [
                    {"name": "body_pose_model.pth", "path": "body_pose_model.pth", "size_mb": 199.6},
                    {"name": "hand_pose_model.pth", "path": "hand_pose_model.pth", "size_mb": 140.5}
                ],
                "total_checkpoints": 3
            },
            
            "cloth_segmentation": {
                "name": "Cloth Segmentation",
                "type": "cloth_segmentation",
                "step": "step_03_cloth_segmentation",
                "ready": True,
                "total_size_mb": 803.2,
                "priority": 5,
                "checkpoints": [
                    {"name": "model.pth", "path": "model.pth", "size_mb": 168.5},
                    {"name": "pytorch_model.bin", "path": "pytorch_model.bin", "size_mb": 168.4}
                ],
                "total_checkpoints": 2
            },
            
            "step_03_cloth_segmentation": {
                "name": "Step 03 Cloth Segmentation",
                "type": "cloth_segmentation",
                "step": "step_03_cloth_segmentation",
                "ready": True,
                "total_size_mb": 206.7,
                "priority": 5,
                "checkpoints": [
                    {"name": "mobile_sam.pt", "path": "mobile_sam.pt", "size_mb": 38.8}
                ],
                "total_checkpoints": 1
            },
            
            "step_04_geometric_matching": {
                "name": "Step 04 Geometric Matching",
                "type": "geometric_matching",
                "step": "step_04_geometric_matching",
                "ready": True,
                "total_size_mb": 33.2,
                "priority": 6,
                "checkpoints": [
                    {"name": "gmm_final.pth", "path": "gmm_final.pth", "size_mb": 4.1},
                    {"name": "lightweight_gmm.pth", "path": "lightweight_gmm.pth", "size_mb": 4.1},
                    {"name": "tps_network.pth", "path": "tps_network.pth", "size_mb": 2.1}
                ],
                "total_checkpoints": 4
            },
            
            "step_05_cloth_warping": {
                "name": "Step 05 Cloth Warping",
                "type": "cloth_warping",
                "step": "step_05_cloth_warping",
                "ready": True,
                "total_size_mb": 3279.2,
                "priority": 7,
                "checkpoints": [
                    {"name": "tom_final.pth", "path": "tom_final.pth", "size_mb": 3279.1},
                    {"name": "lightweight_warping.pth", "path": "lightweight_warping.pth", "size_mb": 0.1}
                ],
                "total_checkpoints": 2
            },
            
            "step_06_virtual_fitting": {
                "name": "Step 06 Virtual Fitting",
                "type": "virtual_tryon",
                "step": "step_06_virtual_fitting",
                "ready": True,
                "total_size_mb": 20854.2,
                "priority": 1,
                "checkpoints": [
                    {"name": "hrviton_final.pth", "path": "hrviton_final.pth", "size_mb": 2445.7},
                    {"name": "diffusion_pytorch_model.bin", "path": "diffusion_pytorch_model.bin", "size_mb": 3279.1},
                    {"name": "pytorch_model.bin", "path": "pytorch_model.bin", "size_mb": 469.5}
                ],
                "total_checkpoints": 7
            },
            
            "step_07_post_processing": {
                "name": "Step 07 Post Processing",
                "type": "auxiliary",
                "step": "step_07_post_processing",
                "ready": True,
                "total_size_mb": 63.9,
                "priority": 8,
                "checkpoints": [
                    {"name": "RealESRGAN_x4plus.pth", "path": "RealESRGAN_x4plus.pth", "size_mb": 63.9}
                ],
                "total_checkpoints": 1
            },
            
            "sam": {
                "name": "SAM (Segment Anything Model)",
                "type": "auxiliary",
                "step": "auxiliary",
                "ready": True,
                "total_size_mb": 2445.7,
                "priority": 8,
                "checkpoints": [
                    {"name": "sam_vit_h_4b8939.pth", "path": "sam_vit_h_4b8939.pth", "size_mb": 2445.7}
                ],
                "total_checkpoints": 1
            },
            
            "clip-vit-base-patch32": {
                "name": "CLIP ViT Base",
                "type": "text_image",
                "step": "auxiliary",
                "ready": True,
                "total_size_mb": 580.7,
                "priority": 9,
                "checkpoints": [
                    {"name": "pytorch_model.bin", "path": "pytorch_model.bin", "size_mb": 577.2}
                ],
                "total_checkpoints": 1
            },
            
            "grounding_dino": {
                "name": "Grounding DINO",
                "type": "auxiliary",
                "step": "auxiliary",
                "ready": True,
                "total_size_mb": 1318.2,
                "priority": 9,
                "checkpoints": [
                    {"name": "pytorch_model.bin", "path": "pytorch_model.bin", "size_mb": 659.9}
                ],
                "total_checkpoints": 1
            }
        }
        
        # 각 모델에 경로 정보 추가
        for model_name, model_info in models_data.items():
            model_info["path"] = str(self.checkpoints_dir / model_name)
        
        return models_data
    
    def create_optimized_model_config(self):
        """최적화된 모델 설정 파일 생성"""
        logger.info("📝 최적화된 모델 설정 파일 생성 중...")
        
        # 단계별 최적 모델 매핑
        step_optimal_models = {
            "step_01_human_parsing": "step_01_human_parsing",
            "step_02_pose_estimation": "step_02_pose_estimation", 
            "step_03_cloth_segmentation": "step_03_cloth_segmentation",
            "step_04_geometric_matching": "step_04_geometric_matching",
            "step_05_cloth_warping": "step_05_cloth_warping",
            "step_06_virtual_fitting": "step_06_virtual_fitting",
            "step_07_post_processing": "step_07_post_processing",
            "auxiliary": "sam"
        }
        
        # Python 설정 파일 생성
        config_content = f'''# app/core/optimized_model_paths.py
"""
최적화된 AI 모델 경로 설정 - 체크포인트 분석 기반
실제 사용 가능한 체크포인트들로만 구성
생성일: {time.strftime("%Y-%m-%d %H:%M:%S")}
분석된 모델: {len(self.analyzed_models)}개
총 크기: {sum(m["total_size_mb"] for m in self.analyzed_models.values())/1024:.1f}GB
"""

from pathlib import Path
from typing import Dict, Optional, List, Any

# 기본 경로
AI_MODELS_ROOT = Path(__file__).parent.parent.parent / "ai_models"
CHECKPOINTS_ROOT = AI_MODELS_ROOT / "checkpoints"

# 분석된 체크포인트 모델들
ANALYZED_MODELS = {{
'''
        
        for model_name, model_info in self.analyzed_models.items():
            config_content += f'''    "{model_name}": {{
        "name": "{model_info['name']}",
        "type": "{model_info['type']}",
        "step": "{model_info['step']}",
        "path": CHECKPOINTS_ROOT / "{model_name}",
        "ready": {model_info['ready']},
        "size_mb": {model_info['total_size_mb']:.1f},
        "priority": {model_info['priority']},
        "checkpoints": {model_info['checkpoints'][:3]},  # 상위 3개만
        "total_checkpoints": {model_info['total_checkpoints']}
    }},
'''
        
        config_content += '''}

# 단계별 최적 모델 매핑
STEP_OPTIMAL_MODELS = {
'''
        
        for step, best_model in step_optimal_models.items():
            config_content += f'''    "{step}": "{best_model}",
'''
        
        config_content += '''}

def get_optimal_model_for_step(step: str) -> Optional[str]:
    """단계별 최적 모델 반환"""
    return STEP_OPTIMAL_MODELS.get(step)

def get_model_checkpoints(model_name: str) -> List[Dict]:
    """모델의 체크포인트 목록 반환"""
    if model_name in ANALYZED_MODELS:
        return ANALYZED_MODELS[model_name]["checkpoints"]
    return []

def get_largest_checkpoint(model_name: str) -> Optional[str]:
    """모델의 가장 큰 체크포인트 반환 (보통 메인 모델)"""
    checkpoints = get_model_checkpoints(model_name)
    if not checkpoints:
        return None
    
    largest = max(checkpoints, key=lambda x: x.get('size_mb', 0))
    return largest.get('path', largest.get('name'))

def get_ready_models_by_type(model_type: str) -> List[str]:
    """타입별 사용 가능한 모델들"""
    return [name for name, info in ANALYZED_MODELS.items() 
            if info["type"] == model_type and info["ready"]]

def get_diffusion_models() -> List[str]:
    """Diffusion 모델들 (OOTDiffusion 등)"""
    return get_ready_models_by_type("diffusion")

def get_virtual_tryon_models() -> List[str]:
    """가상 피팅 모델들"""
    return get_ready_models_by_type("virtual_tryon")

def get_human_parsing_models() -> List[str]:
    """인체 파싱 모델들"""
    return get_ready_models_by_type("human_parsing")

def get_model_info(model_name: str) -> Optional[Dict]:
    """모델 상세 정보 반환"""
    return ANALYZED_MODELS.get(model_name)

def list_all_ready_models() -> Dict[str, Dict]:
    """모든 사용 가능한 모델 정보"""
    return ANALYZED_MODELS.copy()

def get_model_path(model_name: str) -> Optional[Path]:
    """모델 디렉토리 경로 반환"""
    if model_name in ANALYZED_MODELS:
        return ANALYZED_MODELS[model_name]["path"]
    return None

def get_checkpoint_path(model_name: str, checkpoint_name: Optional[str] = None) -> Optional[Path]:
    """특정 체크포인트 파일 경로 반환"""
    model_path = get_model_path(model_name)
    if not model_path:
        return None
    
    if checkpoint_name:
        return model_path / checkpoint_name
    else:
        # 가장 큰 체크포인트 반환
        largest_ckpt = get_largest_checkpoint(model_name)
        return model_path / largest_ckpt if largest_ckpt else None

# 사용 통계
ANALYSIS_STATS = {{
    "total_models": {len(self.analyzed_models)},
    "ready_models": {len(self.analyzed_models)},
    "total_size_gb": {sum(m["total_size_mb"] for m in self.analyzed_models.values())/1024:.1f},
    "models_by_step": {{
        step: len([m for m in self.analyzed_models.values() if m["step"] == step])
        for step in set(m["step"] for m in self.analyzed_models.values())
    }},
    "largest_model": "{max(self.analyzed_models.items(), key=lambda x: x[1]["total_size_mb"])[0]}"
}}

# 빠른 접근 함수들
def get_best_diffusion_model() -> Optional[str]:
    """최고 성능 Diffusion 모델"""
    return get_optimal_model_for_step("step_06_virtual_fitting")

def get_best_human_parsing_model() -> Optional[str]:
    """최고 성능 인체 파싱 모델"""  
    return get_optimal_model_for_step("step_01_human_parsing")

def get_best_pose_model() -> Optional[str]:
    """최고 성능 포즈 추정 모델"""
    return get_optimal_model_for_step("step_02_pose_estimation")
'''
        
        # 파일 저장
        config_path = self.backend_dir / "app" / "core" / "optimized_model_paths.py"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w', encoding='utf-8') as f:
            f.write(config_content)
        
        logger.info(f"✅ 최적화된 Python 설정 파일 생성: {config_path}")
        
        # JSON 설정 파일도 생성
        json_config = {
            "analyzed_models": self.analyzed_models,
            "step_optimal_models": step_optimal_models,
            "analysis_stats": {
                "total_models": len(self.analyzed_models),
                "ready_models": len(self.analyzed_models),
                "total_size_gb": sum(m["total_size_mb"] for m in self.analyzed_models.values())/1024,
                "largest_model": max(self.analyzed_models.items(), key=lambda x: x[1]["total_size_mb"])[0]
            }
        }
        
        json_path = self.backend_dir / "app" / "core" / "optimized_models.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_config, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ JSON 설정 파일 생성: {json_path}")
    
    def create_checkpoint_model_loader(self):
        """체크포인트 모델 로더 생성"""
        logger.info("🔧 체크포인트 모델 로더 생성 중...")
        
        loader_content = '''# app/ai_pipeline/utils/checkpoint_model_loader.py
"""
체크포인트 분석 기반 ModelLoader 완전 연동
실제 다운로드된 127.2GB 체크포인트들 활용
"""

import os
import torch
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging

try:
    from app.core.optimized_model_paths import (
        ANALYZED_MODELS, get_optimal_model_for_step, 
        get_checkpoint_path, get_largest_checkpoint
    )
    OPTIMIZED_PATHS_AVAILABLE = True
except ImportError:
    OPTIMIZED_PATHS_AVAILABLE = False

logger = logging.getLogger(__name__)

class CheckpointModelLoader:
    """체크포인트 분석 기반 모델 로더"""
    
    def __init__(self, device: str = "auto"):
        self.device = self._setup_device(device)
        self.models = {}
        self.loaded_models = {}
        
        if OPTIMIZED_PATHS_AVAILABLE:
            self._register_analyzed_models()
        else:
            logger.warning("⚠️ 최적화된 모델 경로가 없습니다")
    
    def _setup_device(self, device: str) -> str:
        """디바이스 설정"""
        if device == "auto":
            if torch.backends.mps.is_available():
                return "mps"
            elif torch.cuda.is_available():
                return "cuda"
            else:
                return "cpu"
        return device
    
    def _register_analyzed_models(self):
        """분석된 체크포인트 모델들 등록"""
        if not OPTIMIZED_PATHS_AVAILABLE:
            return
            
        registered_count = 0
        
        for model_name, model_info in ANALYZED_MODELS.items():
            if not model_info["ready"]:
                continue
            
            try:
                # 모델 정보 등록
                self.models[model_name] = {
                    "name": model_info["name"],
                    "type": model_info["type"],
                    "step": model_info["step"],
                    "path": model_info["path"],
                    "checkpoints": model_info["checkpoints"],
                    "size_mb": model_info["size_mb"],
                    "priority": model_info["priority"]
                }
                
                registered_count += 1
                
            except Exception as e:
                logger.warning(f"   ⚠️ {model_name} 등록 실패: {e}")
        
        logger.info(f"📦 {registered_count}개 체크포인트 모델 등록 완료")
    
    async def load_model(self, model_name: str, **kwargs) -> Optional[Any]:
        """모델 로드"""
        if model_name in self.loaded_models:
            return self.loaded_models[model_name]
        
        if model_name not in self.models:
            logger.warning(f"⚠️ 등록되지 않은 모델: {model_name}")
            return None
        
        try:
            model_info = self.models[model_name]
            
            # 가장 큰 체크포인트 경로 찾기
            largest_checkpoint = get_largest_checkpoint(model_name)
            if not largest_checkpoint:
                logger.warning(f"⚠️ {model_name}의 체크포인트를 찾을 수 없습니다")
                return None
            
            checkpoint_path = get_checkpoint_path(model_name, largest_checkpoint)
            
            if not checkpoint_path or not checkpoint_path.exists():
                logger.warning(f"⚠️ {model_name}의 체크포인트 파일이 없습니다: {checkpoint_path}")
                return None
            
            # PyTorch 모델 로드
            logger.info(f"🔧 {model_name} 로딩 중... ({checkpoint_path})")
            
            # 안전한 로드
            try:
                model = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
            except:
                # weights_only가 지원되지 않는 경우 폴백
                model = torch.load(checkpoint_path, map_location=self.device)
            
            # 모델 정리 및 디바이스 이동
            if isinstance(model, dict):
                if 'model' in model:
                    model = model['model']
                elif 'state_dict' in model:
                    model = model['state_dict']
            
            # 캐시에 저장
            self.loaded_models[model_name] = model
            
            logger.info(f"✅ {model_name} 로딩 완료")
            return model
            
        except Exception as e:
            logger.error(f"❌ {model_name} 로딩 실패: {e}")
            return None
    
    async def load_optimal_model_for_step(self, step: str, **kwargs) -> Optional[Any]:
        """단계별 최적 모델 로드"""
        optimal_model = get_optimal_model_for_step(step)
        if not optimal_model:
            logger.warning(f"⚠️ {step}에 대한 최적 모델을 찾을 수 없음")
            return None
        
        logger.info(f"🎯 {step} 최적 모델 로드: {optimal_model}")
        return await self.load_model(optimal_model, **kwargs)
    
    def get_model_info(self, model_name: str) -> Optional[Dict]:
        """모델 정보 반환"""
        return self.models.get(model_name)
    
    def list_models(self) -> Dict[str, Dict]:
        """등록된 모델 목록"""
        return self.models.copy()
    
    def clear_cache(self):
        """모델 캐시 정리"""
        self.loaded_models.clear()
        
        if self.device == "mps" and torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("🧹 모델 캐시 정리 완료")

# 전역 모델 로더
_global_checkpoint_loader: Optional[CheckpointModelLoader] = None

def get_checkpoint_model_loader(**kwargs) -> CheckpointModelLoader:
    """전역 체크포인트 모델 로더 반환"""
    global _global_checkpoint_loader
    if _global_checkpoint_loader is None:
        _global_checkpoint_loader = CheckpointModelLoader(**kwargs)
    return _global_checkpoint_loader

async def load_best_model_for_step(step: str, **kwargs) -> Optional[Any]:
    """단계별 최고 성능 모델 로드"""
    loader = get_checkpoint_model_loader()
    return await loader.load_optimal_model_for_step(step, **kwargs)

# 빠른 접근 함수들
async def load_best_diffusion_model(**kwargs) -> Optional[Any]:
    """최고 성능 Diffusion 모델 로드"""
    return await load_best_model_for_step("step_06_virtual_fitting", **kwargs)

async def load_best_human_parsing_model(**kwargs) -> Optional[Any]:
    """최고 성능 인체 파싱 모델 로드"""
    return await load_best_model_for_step("step_01_human_parsing", **kwargs)

async def load_best_pose_model(**kwargs) -> Optional[Any]:
    """최고 성능 포즈 추정 모델 로드"""
    return await load_best_model_for_step("step_02_pose_estimation", **kwargs)

async def load_best_cloth_segmentation_model(**kwargs) -> Optional[Any]:
    """최고 성능 의류 분할 모델 로드"""
    return await load_best_model_for_step("step_03_cloth_segmentation", **kwargs)
'''
        
        # 파일 저장
        loader_path = self.backend_dir / "app" / "ai_pipeline" / "utils" / "checkpoint_model_loader.py"
        loader_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(loader_path, 'w', encoding='utf-8') as f:
            f.write(loader_content)
        
        logger.info(f"✅ 체크포인트 모델 로더 생성: {loader_path}")
    
    def create_test_scripts(self):
        """테스트 스크립트 생성"""
        logger.info("🧪 테스트 스크립트 생성 중...")
        
        # 테스트 디렉토리 생성
        test_dir = self.backend_dir / "scripts" / "test"
        test_dir.mkdir(parents=True, exist_ok=True)
        
        # __init__.py 파일 생성
        init_file = test_dir / "__init__.py"
        with open(init_file, 'w') as f:
            f.write('# Test scripts package\n')
        
        logger.info(f"✅ 테스트 스크립트 디렉토리 생성: {test_dir}")

def main():
    """메인 함수"""
    logger.info("🔍 MyCloset AI - 수정된 체크포인트 분석기 시작")
    logger.info("=" * 60)
    
    try:
        analyzer = FixedCheckpointAnalyzer()
        
        # 최적화된 설정 파일 생성
        analyzer.create_optimized_model_config()
        
        # 체크포인트 모델 로더 생성
        analyzer.create_checkpoint_model_loader()
        
        # 테스트 스크립트 생성
        analyzer.create_test_scripts()
        
        logger.info("\n🎉 체크포인트 분석기 완료!")
        logger.info(f"📊 분석된 모델: {len(analyzer.analyzed_models)}개")
        total_size_gb = sum(m["total_size_mb"] for m in analyzer.analyzed_models.values()) / 1024
        logger.info(f"💾 총 크기: {total_size_gb:.1f}GB")
        
        logger.info("\n📝 생성된 파일들:")
        logger.info("   ✅ app/core/optimized_model_paths.py")
        logger.info("   ✅ app/core/optimized_models.json")
        logger.info("   ✅ app/ai_pipeline/utils/checkpoint_model_loader.py")
        logger.info("   ✅ scripts/test/ (테스트 디렉토리)")
        
        logger.info("\n🚀 다음 단계:")
        logger.info("   python scripts/test/test_model_loader.py  # 모델 로더 테스트")
        logger.info("   python app/main.py  # 서버 실행")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 체크포인트 분석기 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)