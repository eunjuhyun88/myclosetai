#!/usr/bin/env python3
"""
🔍 완전한 AI 모델 체크 및 설치 스크립트
- 모든 필요한 모델 파일 확인
- 누락된 모델 자동 다운로드
- M3 Max 최적화 설정
"""

import os
import sys
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import subprocess
import time
from urllib.parse import urlparse
import hashlib

def log_info(msg: str):
    print(f"ℹ️  {msg}")

def log_success(msg: str):
    print(f"✅ {msg}")

def log_warning(msg: str):
    print(f"⚠️  {msg}")

def log_error(msg: str):
    print(f"❌ {msg}")

def log_download(msg: str):
    print(f"📥 {msg}")

class AIModelChecker:
    """AI 모델 체크 및 관리"""
    
    def __init__(self):
        self.base_dir = Path("ai_models")
        self.checkpoints_dir = self.base_dir / "checkpoints"
        self.missing_models = []
        self.available_models = []
        self.model_configs = self._get_model_requirements()
        
    def _get_model_requirements(self) -> Dict:
        """필요한 모델들과 요구사항 정의"""
        return {
            # Step 1: Human Parsing (인체 파싱)
            "segformer_human_parsing": {
                "path": "checkpoints/step_01_human_parsing/segformer_b2_clothes",
                "files": [
                    "config.json",
                    "pytorch_model.bin",
                    "preprocessor_config.json"
                ],
                "size_mb": 200,
                "description": "Segformer 인체 파싱 모델",
                "huggingface_id": "mattmdjaga/segformer_b2_clothes",
                "priority": "high"
            },
            
            "graphonomy_atr": {
                "path": "checkpoints/step_01_human_parsing",
                "files": ["graphonomy_atr.pth"],
                "size_mb": 85,
                "description": "Graphonomy ATR 모델",
                "download_url": "https://drive.google.com/file/d/1ruJg4lqR_jgQPj-9K0PP-L2vJERYOxLP/view",
                "priority": "high"
            },
            
            "graphonomy_lip": {
                "path": "checkpoints/step_01_human_parsing", 
                "files": ["graphonomy_lip.pth"],
                "size_mb": 85,
                "description": "Graphonomy LIP 모델",
                "priority": "medium"
            },
            
            # Step 2: Pose Estimation (포즈 추정)
            "mediapipe_pose": {
                "path": "checkpoints/step_02_pose_estimation",
                "files": [
                    "pose_landmarker.task",
                    "pose_landmark_full.tflite"
                ],
                "size_mb": 15,
                "description": "MediaPipe 포즈 추정",
                "priority": "high"
            },
            
            "yolov8_pose": {
                "path": "checkpoints/step_02_pose_estimation",
                "files": ["yolov8n-pose.pt"],
                "size_mb": 6,
                "description": "YOLOv8 포즈 모델",
                "priority": "medium"
            },
            
            # Step 3: Cloth Segmentation (의류 분할)
            "u2net_pytorch": {
                "path": "checkpoints/step_03_cloth_segmentation",
                "files": ["u2net.pth"],
                "size_mb": 176,
                "description": "U²-Net PyTorch 모델",
                "priority": "high"
            },
            
            "u2net_onnx": {
                "path": "checkpoints/step_03_cloth_segmentation",
                "files": ["u2net.onnx"],
                "size_mb": 176,
                "description": "U²-Net ONNX 모델",
                "priority": "high"
            },
            
            "mobile_sam": {
                "path": "checkpoints/step_03_cloth_segmentation",
                "files": ["mobile_sam.pt"],
                "size_mb": 40,
                "description": "Mobile SAM 모델",
                "priority": "medium"
            },
            
            # CLIP Models (텍스트-이미지 임베딩)
            "clip_vit_base": {
                "path": "clip-vit-base-patch32",  # checkpoints 밖에 있음
                "files": [
                    "config.json",
                    "model.safetensors",  # 또는 pytorch_model.bin
                    "preprocessor_config.json",
                    "tokenizer.json",
                    "vocab.json"
                ],
                "size_mb": 600,
                "description": "CLIP ViT-Base 모델",
                "huggingface_id": "openai/clip-vit-base-patch32",
                "priority": "critical"
            },
            
            "clip_vit_large": {
                "path": "checkpoints/clip-vit-large-patch14",
                "files": [
                    "config.json",
                    "model.safetensors",
                    "pytorch_model.bin"
                ],
                "size_mb": 1700,
                "description": "CLIP ViT-Large 모델",
                "priority": "medium"
            },
            
            # OOTDiffusion (가상 피팅)
            "ootdiffusion_checkpoints": {
                "path": "checkpoints/ootdiffusion/checkpoints/ootd",
                "files": [
                    "model_index.json",
                    "ootd_hd/checkpoint-36000/unet_garm/diffusion_pytorch_model.safetensors",
                    "ootd_hd/checkpoint-36000/unet_vton/diffusion_pytorch_model.safetensors"
                ],
                "size_mb": 3400,
                "description": "OOTDiffusion 체크포인트",
                "priority": "critical"
            },
            
            # Stable Diffusion Base
            "stable_diffusion_v15": {
                "path": "checkpoints/stable-diffusion-v1-5",
                "files": [
                    "model_index.json",
                    "unet/diffusion_pytorch_model.safetensors",
                    "vae/diffusion_pytorch_model.safetensors",
                    "text_encoder/pytorch_model.bin"
                ],
                "size_mb": 4000,
                "description": "Stable Diffusion v1.5",
                "huggingface_id": "runwayml/stable-diffusion-v1-5",
                "priority": "critical"
            },
            
            # Post Processing
            "realesrgan": {
                "path": "checkpoints/step_07_post_processing",
                "files": ["RealESRGAN_x4plus.pth"],
                "size_mb": 67,
                "description": "RealESRGAN 업스케일러",
                "priority": "low"
            }
        }
    
    def check_all_models(self) -> Dict:
        """모든 모델 체크"""
        log_info("AI 모델 완전 체크 시작...")
        print("=" * 60)
        
        results = {
            "critical_missing": [],
            "high_missing": [],
            "medium_missing": [],
            "low_missing": [],
            "available": [],
            "total_missing_size_gb": 0,
            "issues": []
        }
        
        for model_name, config in self.model_configs.items():
            status = self._check_single_model(model_name, config)
            
            if status["available"]:
                results["available"].append({
                    "name": model_name,
                    "description": config["description"],
                    "size_mb": status.get("actual_size_mb", 0)
                })
                log_success(f"{config['description']}: 정상")
            else:
                priority = config.get("priority", "medium")
                missing_info = {
                    "name": model_name,
                    "description": config["description"],
                    "size_mb": config.get("size_mb", 0),
                    "path": config["path"],
                    "missing_files": status["missing_files"],
                    "has_huggingface": "huggingface_id" in config
                }
                
                results[f"{priority}_missing"].append(missing_info)
                results["total_missing_size_gb"] += config.get("size_mb", 0) / 1024
                
                missing_files_str = ", ".join(status["missing_files"][:3])
                if len(status["missing_files"]) > 3:
                    missing_files_str += f" 외 {len(status['missing_files'])-3}개"
                
                log_error(f"{config['description']}: 누락 ({missing_files_str})")
        
        # CLIP 특별 체크 (경로 문제 때문에)
        self._special_clip_check(results)
        
        return results
    
    def _check_single_model(self, model_name: str, config: Dict) -> Dict:
        """개별 모델 체크"""
        model_path = self.base_dir / config["path"]
        missing_files = []
        available_files = []
        total_size = 0
        
        for required_file in config["files"]:
            file_path = model_path / required_file
            if file_path.exists():
                available_files.append(required_file)
                total_size += file_path.stat().st_size
            else:
                missing_files.append(required_file)
        
        # CLIP 모델의 경우 safetensors 또는 pytorch_model.bin 중 하나만 있어도 OK
        if model_name.startswith("clip") and "model.safetensors" in missing_files:
            pytorch_model = model_path / "pytorch_model.bin"
            if pytorch_model.exists():
                missing_files.remove("model.safetensors")
                available_files.append("pytorch_model.bin")
                total_size += pytorch_model.stat().st_size
        
        return {
            "available": len(missing_files) == 0,
            "missing_files": missing_files,
            "available_files": available_files,
            "actual_size_mb": total_size / (1024**2)
        }
    
    def _special_clip_check(self, results: Dict):
        """CLIP 모델 특별 체크 (여러 위치 확인)"""
        log_info("CLIP 모델 특별 체크...")
        
        possible_clip_paths = [
            "ai_models/clip-vit-base-patch32",
            "ai_models/checkpoints/clip-vit-base-patch32", 
            "ai_models/checkpoints/shared_encoder/clip-vit-base-patch32"
        ]
        
        clip_found = False
        for clip_path in possible_clip_paths:
            path = Path(clip_path)
            if path.exists():
                safetensors = path / "model.safetensors"
                pytorch_model = path / "pytorch_model.bin"
                config_file = path / "config.json"
                
                if (safetensors.exists() or pytorch_model.exists()) and config_file.exists():
                    log_success(f"CLIP 모델 발견: {clip_path}")
                    clip_found = True
                    
                    # 결과에서 CLIP missing 제거
                    for priority in ["critical_missing", "high_missing", "medium_missing"]:
                        results[priority] = [m for m in results[priority] if not m["name"].startswith("clip")]
                    
                    # available에 추가
                    results["available"].append({
                        "name": "clip_vit_base",
                        "description": "CLIP ViT-Base 모델 (발견됨)",
                        "size_mb": (safetensors.stat().st_size if safetensors.exists() 
                                  else pytorch_model.stat().st_size) / (1024**2),
                        "path": str(clip_path)
                    })
                    break
        
        if not clip_found:
            log_warning("CLIP 모델을 찾을 수 없습니다. 다운로드가 필요합니다.")
    
    def download_missing_models(self, results: Dict, priorities: List[str] = ["critical", "high"]):
        """누락된 모델 다운로드"""
        log_info("누락된 모델 다운로드 시작...")
        
        total_downloaded = 0
        
        for priority in priorities:
            missing_key = f"{priority}_missing"
            if missing_key not in results:
                continue
                
            for model_info in results[missing_key]:
                model_name = model_info["name"]
                config = self.model_configs[model_name]
                
                log_download(f"{model_info['description']} 다운로드 중...")
                
                if self._download_model(model_name, config):
                    log_success(f"{model_info['description']} 다운로드 완료")
                    total_downloaded += 1
                else:
                    log_error(f"{model_info['description']} 다운로드 실패")
        
        log_info(f"총 {total_downloaded}개 모델 다운로드 완료")
    
    def _download_model(self, model_name: str, config: Dict) -> bool:
        """개별 모델 다운로드"""
        try:
            # HuggingFace에서 다운로드 가능한 경우
            if "huggingface_id" in config:
                return self._download_from_huggingface(model_name, config)
            
            # 특별한 다운로드 로직이 필요한 모델들
            if model_name == "u2net_pytorch":
                return self._download_u2net()
            elif model_name == "mediapipe_pose":
                return self._download_mediapipe_pose()
            elif model_name.startswith("graphonomy"):
                return self._download_graphonomy(model_name)
            
            log_warning(f"{model_name}: 자동 다운로드 방법을 찾을 수 없습니다")
            return False
            
        except Exception as e:
            log_error(f"{model_name} 다운로드 오류: {e}")
            return False
    
    def _download_from_huggingface(self, model_name: str, config: Dict) -> bool:
        """HuggingFace에서 모델 다운로드"""
        try:
            from transformers import AutoModel, AutoProcessor, AutoTokenizer
            
            model_path = self.base_dir / config["path"]
            model_path.mkdir(parents=True, exist_ok=True)
            
            hf_id = config["huggingface_id"]
            
            # 모델 타입에 따라 다운로드
            if model_name == "segformer_human_parsing":
                from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
                model = SegformerForSemanticSegmentation.from_pretrained(hf_id)
                processor = SegformerImageProcessor.from_pretrained(hf_id)
                model.save_pretrained(model_path)
                processor.save_pretrained(model_path)
                
            elif model_name.startswith("clip"):
                from transformers import CLIPModel, CLIPProcessor
                model = CLIPModel.from_pretrained(hf_id)
                processor = CLIPProcessor.from_pretrained(hf_id)
                model.save_pretrained(model_path)
                processor.save_pretrained(model_path)
                
            elif model_name == "stable_diffusion_v15":
                from diffusers import StableDiffusionPipeline
                pipeline = StableDiffusionPipeline.from_pretrained(hf_id)
                pipeline.save_pretrained(model_path)
            
            return True
            
        except ImportError as e:
            log_error(f"필요한 라이브러리가 설치되지 않음: {e}")
            return False
        except Exception as e:
            log_error(f"HuggingFace 다운로드 실패: {e}")
            return False
    
    def _download_u2net(self) -> bool:
        """U²-Net 모델 다운로드"""
        try:
            import urllib.request
            
            url = "https://github.com/xuebinqin/U-2-Net/raw/master/saved_models/u2net/u2net.pth"
            save_path = self.checkpoints_dir / "step_03_cloth_segmentation/u2net.pth"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            log_download("U²-Net 다운로드 중...")
            urllib.request.urlretrieve(url, save_path)
            
            # ONNX 버전도 변환 (선택사항)
            self._convert_u2net_to_onnx(save_path)
            
            return True
            
        except Exception as e:
            log_error(f"U²-Net 다운로드 실패: {e}")
            return False
    
    def _download_mediapipe_pose(self) -> bool:
        """MediaPipe 포즈 모델 다운로드"""
        try:
            import urllib.request
            
            urls = {
                "pose_landmarker.task": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task",
                "pose_landmark_full.tflite": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.tflite"
            }
            
            save_dir = self.checkpoints_dir / "step_02_pose_estimation"
            save_dir.mkdir(parents=True, exist_ok=True)
            
            for filename, url in urls.items():
                save_path = save_dir / filename
                log_download(f"MediaPipe {filename} 다운로드 중...")
                urllib.request.urlretrieve(url, save_path)
            
            return True
            
        except Exception as e:
            log_error(f"MediaPipe 다운로드 실패: {e}")
            return False
    
    def _download_graphonomy(self, model_name: str) -> bool:
        """Graphonomy 모델 다운로드 (수동 안내)"""
        log_warning(f"{model_name}: 수동 다운로드가 필요합니다")
        log_info("Graphonomy 모델 다운로드 방법:")
        log_info("1. https://github.com/Gaoyiminggithub/Graphonomy 방문")
        log_info("2. pre-trained 모델 다운로드")
        log_info(f"3. ai_models/checkpoints/step_01_human_parsing/ 폴더에 저장")
        return False
    
    def _convert_u2net_to_onnx(self, pytorch_path: Path):
        """U²-Net PyTorch를 ONNX로 변환"""
        try:
            import torch
            import torch.onnx
            
            # 간단한 U²-Net 모델 로드 및 변환 로직
            # (실제 구현에서는 U²-Net 모델 구조가 필요)
            log_info("U²-Net ONNX 변환은 별도로 진행하세요")
            
        except Exception as e:
            log_warning(f"ONNX 변환 실패: {e}")
    
    def generate_install_script(self, results: Dict) -> str:
        """설치 스크립트 생성"""
        script_lines = [
            "#!/bin/bash",
            "# AI 모델 자동 설치 스크립트",
            "",
            "echo '🚀 MyCloset AI 모델 설치 시작'",
            "echo '================================'",
            "",
            "# Python 환경 확인",
            "if ! command -v python3 &> /dev/null; then",
            "    echo '❌ Python3가 설치되지 않았습니다'",
            "    exit 1",
            "fi",
            "",
            "# 필요한 라이브러리 설치",
            "echo '📦 라이브러리 설치 중...'",
            "pip install transformers diffusers torch torchvision onnxruntime",
            "pip install mediapipe opencv-python pillow",
            "",
            "# 모델 다운로드",
            "echo '📥 모델 다운로드 중...'",
            "python3 -c \"",
            "import sys",
            "sys.path.append('.')",
            "from complete_model_check import AIModelChecker",
            "checker = AIModelChecker()",
            "results = checker.check_all_models()",
            "checker.download_missing_models(results, ['critical', 'high'])",
            "\"",
            "",
            "echo '✅ 설치 완료!'"
        ]
        
        return "\n".join(script_lines)

def main():
    """메인 실행 함수"""
    print("🔍 MyCloset AI 모델 완전 체크")
    print("=" * 50)
    
    checker = AIModelChecker()
    
    # 1. 모든 모델 체크
    results = checker.check_all_models()
    
    # 2. 결과 요약
    print("\n📊 체크 결과 요약")
    print("=" * 30)
    print(f"✅ 사용 가능한 모델: {len(results['available'])}개")
    print(f"🔴 치명적 누락: {len(results['critical_missing'])}개")
    print(f"🟡 높은 우선순위 누락: {len(results['high_missing'])}개")
    print(f"🟢 중간 우선순위 누락: {len(results['medium_missing'])}개")
    print(f"📦 총 누락 용량: {results['total_missing_size_gb']:.1f} GB")
    
    # 3. 누락된 모델 상세 정보
    if results['critical_missing'] or results['high_missing']:
        print("\n🚨 우선 다운로드 필요한 모델들:")
        for model in results['critical_missing'] + results['high_missing']:
            print(f"   - {model['description']} ({model['size_mb']} MB)")
            if model['has_huggingface']:
                print(f"     📥 자동 다운로드 가능")
            else:
                print(f"     ⚠️  수동 다운로드 필요")
    
    # 4. 자동 다운로드 제안
    if results['critical_missing'] or results['high_missing']:
        print(f"\n🤖 자동 다운로드를 시작하시겠습니까? (y/n): ", end="")
        if input().lower() == 'y':
            checker.download_missing_models(results, ['critical', 'high'])
        else:
            log_info("수동으로 다음 명령어를 실행하세요:")
            log_info("python3 complete_model_check.py --download")
    
    # 5. 설치 스크립트 생성
    install_script = checker.generate_install_script(results)
    with open("install_ai_models.sh", "w") as f:
        f.write(install_script)
    os.chmod("install_ai_models.sh", 0o755)
    log_success("설치 스크립트 생성됨: install_ai_models.sh")
    
    # 6. 최종 안내
    print("\n🎯 다음 단계:")
    if results['critical_missing']:
        print("1. 치명적 모델들을 먼저 다운로드하세요")
        print("2. ./install_ai_models.sh 실행")
    print("3. python3 scripts/test/test_final_models.py 로 테스트")
    print("4. 서버 시작: python3 run_server.py")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="AI 모델 체크 및 다운로드")
    parser.add_argument("--download", action="store_true", help="자동 다운로드 실행")
    parser.add_argument("--priority", choices=["critical", "high", "medium", "low"], 
                       default="high", help="다운로드 우선순위")
    
    args = parser.parse_args()
    
    if args.download:
        checker = AIModelChecker()
        results = checker.check_all_models()
        priorities = ["critical", args.priority] if args.priority != "critical" else ["critical"]
        checker.download_missing_models(results, priorities)
    else:
        main()