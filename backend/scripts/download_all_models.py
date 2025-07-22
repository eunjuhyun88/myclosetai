# backend/scripts/download_all_models.py
import os
import torch
from pathlib import Path
from transformers import AutoModel, AutoTokenizer
from diffusers import StableDiffusionPipeline
import gdown
import subprocess

class ModelDownloader:
    def __init__(self, base_dir="./ai_models"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        
    async def download_all_models(self):
        """모든 필요한 AI 모델 다운로드"""
        
        # 1. OOTDiffusion (핵심 가상 피팅)
        await self.download_ootdiffusion()
        
        # 2. Human Parsing (Graphonomy)
        await self.download_human_parsing()
        
        # 3. Pose Estimation (OpenPose)
        await self.download_pose_estimation()
        
        # 4. Cloth Segmentation (U2Net)
        await self.download_cloth_segmentation()
        
        # 5. CLIP (품질 평가)
        await self.download_clip()
        
        # 6. Background Removal (rembg)
        await self.download_background_removal()
        
    async def download_ootdiffusion(self):
        """OOTDiffusion 모델 다운로드"""
        print("🤖 OOTDiffusion 다운로드 중...")
        
        # Hugging Face에서 다운로드
        model_path = self.base_dir / "OOTDiffusion"
        model_path.mkdir(exist_ok=True)
        
        # 기본 Stable Diffusion v1.5 다운로드
        pipeline = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            cache_dir=str(model_path)
        )
        pipeline.save_pretrained(str(model_path / "stable-diffusion-v1-5"))
        
        # OOTDiffusion 특화 가중치 (있다면)
        # TODO: 실제 OOTDiffusion 가중치 URL
        
    async def download_human_parsing(self):
        """Human Parsing 모델 다운로드"""
        print("👤 Human Parsing 모델 다운로드 중...")
        
        model_path = self.base_dir / "human_parsing"
        model_path.mkdir(exist_ok=True)
        
        # Self-Correction Human Parsing 다운로드
        # Google Drive 링크 (실제 다운로드 링크로 교체 필요)
        gdown.download(
            "https://drive.google.com/uc?id=1k4dllHpu0bdx38J7H28rVVLpU-kOHmnH",
            str(model_path / "exp-schp-201908261155-pascal.pth"),
            quiet=False
        )
        
    async def download_pose_estimation(self):
        """Pose Estimation 모델 다운로드"""
        print("🤸 Pose Estimation 모델 다운로드 중...")
        
        # OpenPose 모델은 mediapipe로 대체 (더 가벼움)
        # mediapipe는 pip 설치시 자동으로 모델 포함됨
        print("✅ MediaPipe Pose 모델 준비 완료")
        
    async def download_cloth_segmentation(self):
        """Cloth Segmentation 모델 다운로드"""
        print("👕 Cloth Segmentation 모델 다운로드 중...")
        
        model_path = self.base_dir / "cloth_segmentation"
        model_path.mkdir(exist_ok=True)
        
        # U2Net 모델 다운로드
        gdown.download(
            "https://drive.google.com/uc?id=1ao1ovG1Qtx4b7EoskHXmi2E9rp5CHLcZ",
            str(model_path / "u2net.pth"),
            quiet=False
        )
        
    async def download_clip(self):
        """CLIP 모델 다운로드"""
        print("🔍 CLIP 모델 다운로드 중...")
        
        model_path = self.base_dir / "clip"
        model_path.mkdir(exist_ok=True)
        
        # CLIP 모델 다운로드
        model = AutoModel.from_pretrained(
            "openai/clip-vit-base-patch32",
            cache_dir=str(model_path)
        )
        tokenizer = AutoTokenizer.from_pretrained(
            "openai/clip-vit-base-patch32",
            cache_dir=str(model_path)
        )
        
    async def download_background_removal(self):
        """배경 제거 모델 다운로드"""
        print("🎭 배경 제거 모델 다운로드 중...")
        
        # rembg 모델들은 첫 사용시 자동 다운로드됨
        from rembg import new_session
        new_session('u2net')  # 모델 초기화로 다운로드 트리거
        
        print("✅ 모든 AI 모델 다운로드 완료!")

if __name__ == "__main__":
    import asyncio
    downloader = ModelDownloader()
    asyncio.run(downloader.download_all_models())