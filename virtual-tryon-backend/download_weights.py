#!/usr/bin/env python3
"""
가상 피팅 모델 가중치 다운로드 스크립트
"""

import os
import sys
from pathlib import Path

# gdown이 설치되어 있는지 확인
try:
    import gdown
except ImportError:
    print("⚠️  gdown이 설치되어 있지 않습니다. 설치 중...")
    os.system(f"{sys.executable} -m pip install gdown")
    import gdown

# requests 설치 확인
try:
    import requests
except ImportError:
    print("⚠️  requests가 설치되어 있지 않습니다. 설치 중...")
    os.system(f"{sys.executable} -m pip install requests")
    import requests

def download_viton_hd_weights():
    """VITON-HD 가중치 다운로드"""
    
    # 체크포인트 디렉토리 생성
    checkpoint_dir = Path("models/VITON-HD/checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = checkpoint_dir / "VITON-HD.pth"
    
    if output_path.exists():
        print("✅ VITON-HD 가중치가 이미 존재합니다!")
        return
    
    print("📥 VITON-HD 가중치 다운로드 중...")
    print("⚠️  원본 Google Drive 링크가 작동하지 않을 수 있습니다.")
    
    # 대체 다운로드 시도
    alternative_sources = [
        {
            "name": "원본 Google Drive",
            "type": "gdown",
            "id": "1Uc0DTTkSfr_PG0XBXQFEOlpEP0IQf4cK"
        },
        {
            "name": "Hugging Face 미러",
            "type": "url",
            "url": "https://huggingface.co/yisol/VITON-HD/resolve/main/viton_hd.pth"
        }
    ]
    
    for source in alternative_sources:
        print(f"\n시도 중: {source['name']}")
        
        try:
            if source['type'] == 'gdown':
                # Google Drive 다운로드
                url = f"https://drive.google.com/uc?id={source['id']}"
                gdown.download(url, str(output_path), quiet=False)
                
                if output_path.exists() and output_path.stat().st_size > 1000000:  # 1MB 이상
                    print(f"✅ {source['name']}에서 다운로드 성공!")
                    return
                    
            elif source['type'] == 'url':
                # 직접 URL 다운로드
                response = requests.get(source['url'], stream=True)
                if response.status_code == 200:
                    total_size = int(response.headers.get('content-length', 0))
                    
                    with open(output_path, 'wb') as f:
                        downloaded = 0
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                            downloaded += len(chunk)
                            if total_size > 0:
                                percent = (downloaded / total_size) * 100
                                print(f"\r진행: {percent:.1f}%", end='')
                    
                    print(f"\n✅ {source['name']}에서 다운로드 성공!")
                    return
                    
        except Exception as e:
            print(f"❌ {source['name']} 실패: {str(e)}")
            if output_path.exists():
                output_path.unlink()  # 실패한 파일 삭제
    
    print("\n❌ 모든 소스에서 다운로드 실패")
    print("\n대안:")
    print("1. 가중치 없이 작동하는 기본 모델 사용 (권장)")
    print("2. 다른 모델 사용 (CP-VTON+, ACGPN 등)")
    print("3. Hugging Face에서 직접 검색")

def setup_basic_model():
    """기본 모델 설정 (가중치 불필요)"""
    print("\n🔧 가중치가 필요 없는 기본 모델 설정 중...")
    
    # 필요한 패키지 설치
    packages = ['torch', 'torchvision', 'opencv-python', 'mediapipe', 'numpy', 'pillow']
    
    for package in packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            print(f"📦 {package} 설치 중...")
            os.system(f"{sys.executable} -m pip install {package}")
    
    # 기본 모델 파일 생성
    basic_model_code = '''import cv2
import numpy as np
import mediapipe as mp
import torch

class BasicVirtualTryOn:
    """가중치 파일 없이 작동하는 기본 가상 피팅 모델"""
    
    def __init__(self):
        # MediaPipe (자동 다운로드)
        self.mp_pose = mp.solutions.pose
        self.mp_selfie = mp.solutions.selfie_segmentation
        
        self.pose = self.mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            enable_segmentation=True
        )
        
        self.selfie_seg = self.mp_selfie.SelfieSegmentation(model_selection=1)
        
        print("✅ 기본 모델 준비 완료!")
    
    def process(self, person_img, clothing_img):
        """가상 피팅 처리"""
        # 포즈 검출
        pose_results = self.pose.process(cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB))
        
        # 세그멘테이션
        seg_results = self.selfie_seg.process(cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB))
        mask = seg_results.segmentation_mask
        
        # 간단한 합성
        h, w = person_img.shape[:2]
        clothing_resized = cv2.resize(clothing_img, (w, h))
        
        # 마스크 적용
        mask_3ch = np.stack([mask] * 3, axis=-1)
        result = person_img * (1 - mask_3ch * 0.7) + clothing_resized * mask_3ch * 0.7
        
        return result.astype(np.uint8)

# 사용 예시
if __name__ == "__main__":
    model = BasicVirtualTryOn()
    print("기본 모델을 사용할 준비가 되었습니다!")
'''
    
    # models 디렉토리 생성
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # basic_model.py 생성
    basic_model_path = models_dir / "basic_model.py"
    with open(basic_model_path, 'w', encoding='utf-8') as f:
        f.write(basic_model_code)
    
    print(f"✅ 기본 모델 파일 생성: {basic_model_path}")
    print("\n사용 방법:")
    print("from models.basic_model import BasicVirtualTryOn")
    print("model = BasicVirtualTryOn()")

def main():
    print("🚀 가상 피팅 모델 설정")
    print("=" * 50)
    
    # 1. VITON-HD 다운로드 시도
    download_viton_hd_weights()
    
    # 2. 기본 모델 설정
    print("\n" + "=" * 50)
    response = input("\n기본 모델을 설정하시겠습니까? (권장) [Y/n]: ").strip().lower()
    
    if response != 'n':
        setup_basic_model()
    
    print("\n✅ 설정 완료!")
    print("\n다음 단계:")
    print("1. python main.py 실행하여 서버 시작")
    print("2. http://localhost:8000/docs 에서 API 테스트")

if __name__ == "__main__":
    main()