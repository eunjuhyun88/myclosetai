#!/usr/bin/env python3
"""
🔥 MyCloset AI - 필수 모델 자동 다운로드
8단계 파이프라인 완성을 위한 우선순위 기반 다운로드
"""

import os
import sys
import gdown
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_priority_models():
    """우선순위 기반 필수 모델 다운로드"""
    
    print("🔥 MyCloset AI - 필수 모델 다운로드")
    print("=" * 50)
    
    base_dir = Path("ai_models/checkpoints")
    base_dir.mkdir(parents=True, exist_ok=True)
    
    # 우선순위 모델들
    models = [
        {
            "name": "U²-Net Human Segmentation",
            "step": "step_03_cloth_segmentation",
            "url": "https://drive.google.com/uc?id=1ao1ovG1Qtx4b7EoskHXmi2E9rp5CHLcZ", 
            "filename": "u2net.pth",
            "size_gb": 0.176,
            "priority": 1,
            "reason": "의류 세그멘테이션 필수 - 파이프라인 핵심"
        },
        {
            "name": "Graphonomy ATR weights",
            "step": "step_01_human_parsing",
            "url": "https://drive.google.com/uc?id=1ruJg4lqR_jgQPj-9K0PP-L2vJERYOxLP", 
            "filename": "graphonomy_atr.pth",
            "size_gb": 0.178,
            "priority": 1,
            "reason": "인체 파싱 정확도 향상"
        },
        {
            "name": "Graphonomy LIP weights",
            "step": "step_01_human_parsing",
            "url": "https://drive.google.com/uc?id=1k4dllHpu0bdx38J7H28rVVLpU-kOHmnH", 
            "filename": "graphonomy_lip.pth",
            "size_gb": 0.178,
            "priority": 1,
            "reason": "인체 파싱 대체 모델"
        },
        {
            "name": "HR-VITON GMM weights",
            "step": "step_04_geometric_matching",
            "url": "https://drive.google.com/uc?id=1WJkwlCJXFWsEgdNGWSoXDhpqtNmwcaVY", 
            "filename": "gmm_final.pth",
            "size_gb": 0.045,
            "priority": 2,
            "reason": "기하학적 매칭 정확도"
        },
        {
            "name": "HR-VITON TOM weights",
            "step": "step_05_cloth_warping",
            "url": "https://drive.google.com/uc?id=1YJU5kNNL8Y-CqaXq-hOjJlh2hZ3s2qY", 
            "filename": "tom_final.pth",
            "size_gb": 0.089,
            "priority": 2,
            "reason": "의류 워핑 품질 향상"
        },
        {
            "name": "Real-ESRGAN",
            "step": "step_07_post_processing",
            "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth", 
            "filename": "RealESRGAN_x4plus.pth",
            "size_gb": 0.067,
            "priority": 3,
            "reason": "이미지 품질 향상"
        },
    ]
    
    total_size = sum(model["size_gb"] for model in models)
    logger.info(f"📦 다운로드 예정: {len(models)}개 모델 ({total_size:.2f}GB)")
    
    success_count = 0
    
    for i, model in enumerate(models, 1):
        logger.info(f"\n[{i}/{len(models)}] {model['name']} 다운로드 중...")
        logger.info(f"   이유: {model['reason']}")
        logger.info(f"   크기: {model['size_gb']:.2f}GB")
        
        # 단계별 디렉토리 생성
        step_dir = base_dir / model["step"]
        step_dir.mkdir(exist_ok=True)
        output_path = step_dir / model["filename"]
        
        # 이미 존재하는지 확인
        if output_path.exists():
            file_size_mb = output_path.stat().st_size / (1024 * 1024)
            if file_size_mb > 10:  # 10MB 이상이면 이미 다운로드된 것으로 간주
                logger.info(f"   ✅ 이미 존재함: {model['name']} ({file_size_mb:.1f}MB)")
                success_count += 1
                continue
        
        try:
            if "drive.google.com" in model["url"]:
                # Google Drive 다운로드
                success = gdown.download(model["url"], str(output_path), quiet=False)
                if success:
                    logger.info(f"   ✅ {model['name']} 다운로드 완료")
                    success_count += 1
                else:
                    logger.error(f"   ❌ {model['name']} 다운로드 실패")
            else:
                logger.info(f"   ⚠️ 수동 다운로드 필요: {model['url']}")
                logger.info(f"      다운로드 후 {output_path}에 저장하세요")
                
        except Exception as e:
            logger.error(f"   ❌ {model['name']} 다운로드 실패: {e}")
    
    logger.info(f"\n🎉 다운로드 완료: {success_count}/{len(models)}개")
    
    if success_count >= len(models) * 0.8:  # 80% 이상 성공
        logger.info("✅ 필수 모델 다운로드 성공! 이제 파이프라인을 테스트할 수 있습니다.")
        logger.info("\n🚀 다음 단계:")
        logger.info("   python scripts/analyze_checkpoints.py  # 모델 재스캔")
        logger.info("   python scripts/test_loaded_models.py   # 파이프라인 테스트")
    else:
        logger.warning(f"⚠️ 일부 모델 다운로드 실패. 수동 다운로드가 필요할 수 있습니다.")

if __name__ == "__main__":
    try:
        # gdown 설치 확인
        import gdown
    except ImportError:
        print("❌ gdown이 설치되지 않았습니다.")
        print("설치: pip install gdown")
        sys.exit(1)
    
    download_priority_models()
