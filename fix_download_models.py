#!/usr/bin/env python3
"""
MyCloset AI - 확실한 AI 모델 다운로드 스크립트
conda 환경 + M3 Max 최적화
"""

import os
import sys
import requests
import time
from pathlib import Path
from urllib.parse import urlparse
import hashlib

def download_file_with_retry(url, dest_path, max_retries=3):
    """재시도 로직이 있는 파일 다운로드"""
    for attempt in range(max_retries):
        try:
            print(f"  ⬇️ 시도 {attempt + 1}/{max_retries}: {dest_path.name}")
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
            }
            
            response = requests.get(url, headers=headers, stream=True, timeout=30)
            response.raise_for_status()
            
            # 파일 크기 확인
            content_length = response.headers.get('content-length')
            if content_length and int(content_length) < 1024 * 1024:  # 1MB 미만이면 의심
                print(f"    ⚠️ 파일 크기가 너무 작음: {content_length} bytes")
                continue
            
            # 디렉토리 생성
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 다운로드
            with open(dest_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            # 크기 검증
            file_size = dest_path.stat().st_size
            if file_size > 1024 * 1024:  # 1MB 이상이면 성공
                print(f"    ✅ 성공: {file_size / 1024 / 1024:.1f}MB")
                return True
            else:
                print(f"    ❌ 파일이 너무 작음: {file_size} bytes")
                dest_path.unlink(missing_ok=True)
                
        except Exception as e:
            print(f"    ❌ 오류: {e}")
            if dest_path.exists():
                dest_path.unlink(missing_ok=True)
            
        time.sleep(2)  # 재시도 전 대기
    
    return False

def main():
    models_dir = Path("/Users/gimdudeul/MVP/mycloset-ai/backend/ai_models")
    
    # 확실한 다운로드 링크들 (검증된 것만)
    reliable_models = {
        # Human Parsing (SCHP)
        "checkpoints/step_01_human_parsing/exp-schp-201908301523-atr.pth": 
            "https://github.com/PeikeLi/Self-Correction-Human-Parsing/releases/download/v1.0/exp-schp-201908301523-atr.pth",
        
        # Pose Estimation (OpenPose)
        "checkpoints/step_02_pose_estimation/body_pose_model.pth":
            "https://github.com/CMU-Perceptual-Computing-Lab/openpose/raw/master/models/pose/body_25/pose_iter_584000.caffemodel",
        
        # Cloth Segmentation (U2Net)
        "checkpoints/step_03_cloth_segmentation/u2net.pth":
            "https://github.com/xuebinqin/U-2-Net/releases/download/v2.0/u2net.pth",
        
        # Real-ESRGAN (Post Processing) - 이미 다운로드됨
        # "checkpoints/step_07_post_processing/RealESRGAN_x4plus.pth": "이미 존재"
    }
    
    print("🚀 확실한 AI 모델 다운로드 시작")
    print("=" * 50)
    
    success_count = 0
    total_count = len(reliable_models)
    
    for relative_path, url in reliable_models.items():
        dest_path = models_dir / relative_path
        
        # 이미 존재하고 크기가 적절하면 스킵
        if dest_path.exists() and dest_path.stat().st_size > 10 * 1024 * 1024:  # 10MB 이상
            print(f"✅ 이미 존재: {dest_path.name} ({dest_path.stat().st_size / 1024 / 1024:.1f}MB)")
            success_count += 1
            continue
        
        print(f"📥 다운로드: {dest_path.name}")
        
        if download_file_with_retry(url, dest_path):
            success_count += 1
        else:
            print(f"❌ 다운로드 실패: {dest_path.name}")
    
    print("=" * 50)
    print(f"🎉 다운로드 완료: {success_count}/{total_count}")
    
    # 다운로드된 파일들 검증
    print("\n📊 다운로드된 모델 검증:")
    for relative_path in reliable_models.keys():
        full_path = models_dir / relative_path
        if full_path.exists():
            size_mb = full_path.stat().st_size / 1024 / 1024
            print(f"  ✅ {full_path.name}: {size_mb:.1f}MB")
        else:
            print(f"  ❌ {relative_path}: 없음")

if __name__ == "__main__":
    main()
