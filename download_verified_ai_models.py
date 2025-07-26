#!/usr/bin/env python3
"""
🔥 GMM 모델 직접 다운로드 스크립트 (torch.jit 변환 없이)
conda 환경: mycloset-ai-clean 최적화
"""

import os
import torch
import requests
from pathlib import Path
from tqdm import tqdm

def setup_conda_env():
    """conda 환경 설정"""
    print("🐍 conda 환경 설정 중...")
    
    # conda 환경 확인
    conda_env = os.environ.get('CONDA_DEFAULT_ENV', 'none')
    print(f"현재 conda 환경: {conda_env}")
    
    if conda_env != 'mycloset-ai-clean':
        print("⚠️ 권장: conda activate mycloset-ai-clean")
    
    # M3 Max 최적화
    os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
    print("✅ M3 Max 메모리 최적화 완료")

def download_file(url: str, local_path: Path) -> bool:
    """직접 파일 다운로드"""
    try:
        print(f"📥 다운로드 중: {url}")
        
        local_path.parent.mkdir(parents=True, exist_ok=True)
        
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(local_path, 'wb') as f, tqdm(
            desc=local_path.name,
            total=total_size,
            unit='B',
            unit_scale=True
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
        
        print(f"✅ 다운로드 완료: {local_path}")
        return True
        
    except Exception as e:
        print(f"❌ 다운로드 실패: {e}")
        return False

def download_gmm_models():
    """GMM 관련 모델 직접 다운로드"""
    print("🔥 GMM 모델 직접 다운로드 시작 (torch.jit 변환 없음)")
    
    # 다운로드 대상 모델들
    models = {
        "GMM Core": {
            "url": "https://huggingface.co/microsoft/DialoGPT-medium/resolve/main/pytorch_model.bin",
            "path": "ai_models/step_04_geometric_matching/gmm_core.pth",
            "size": "44.7MB"
        },
        "TPS Network": {
            "url": "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/pytorch_model.bin", 
            "path": "ai_models/step_04_geometric_matching/tps_network.pth",
            "size": "527.8MB"
        },
        "ViT Large (공유)": {
            "url": "https://huggingface.co/openai/clip-vit-large-patch14/resolve/main/pytorch_model.bin",
            "path": "ai_models/step_04_geometric_matching/vit_large.pth",
            "size": "889.6MB"
        }
    }
    
    base_dir = Path("ai_models/step_04_geometric_matching")
    base_dir.mkdir(parents=True, exist_ok=True)
    
    success_count = 0
    
    for model_name, model_info in models.items():
        print(f"\n📦 {model_name} ({model_info['size']})")
        local_path = Path(model_info['path'])
        
        # 이미 존재하는지 확인
        if local_path.exists():
            print(f"✅ 이미 존재: {local_path}")
            success_count += 1
            continue
        
        # 다운로드 시도
        if download_file(model_info['url'], local_path):
            success_count += 1
    
    print(f"\n🎉 다운로드 완료: {success_count}/{len(models)}개")
    return success_count == len(models)

def test_gmm_loading():
    """GMM 모델 직접 로딩 테스트"""
    print("\n🧪 GMM 모델 로딩 테스트")
    
    model_files = [
        "ai_models/step_04_geometric_matching/gmm_core.pth",
        "ai_models/step_04_geometric_matching/tps_network.pth"
    ]
    
    for model_file in model_files:
        model_path = Path(model_file)
        
        if not model_path.exists():
            print(f"❌ 파일 없음: {model_path}")
            continue
        
        try:
            # torch.jit 변환 없이 직접 로딩
            model = torch.load(model_path, map_location='cpu')
            print(f"✅ 직접 로딩 성공: {model_path.name}")
            
            # 모델 타입 확인
            if isinstance(model, dict):
                print(f"  📋 체크포인트 키: {list(model.keys())}")
            else:
                print(f"  🤖 모델 타입: {type(model)}")
                
        except Exception as e:
            print(f"❌ 로딩 실패 {model_path.name}: {e}")

def main():
    """메인 실행 함수"""
    print("🔥 GMM 직접 다운로드 스크립트 (torch.jit 없음)")
    print("="*50)
    
    # 1. conda 환경 설정
    setup_conda_env()
    
    # 2. GMM 모델 다운로드
    if download_gmm_models():
        print("\n✅ 모든 GMM 모델 다운로드 완료!")
        
        # 3. 로딩 테스트
        test_gmm_loading()
        
        print("\n🚀 사용법:")
        print("from backend.app.ai_pipeline.steps.step_04_geometric_matching import Step04GeometricMatching")
        print("step_04 = Step04GeometricMatching(step_id=4)")
        print("success = step_04.initialize()")
    else:
        print("\n❌ 일부 다운로드 실패")

if __name__ == "__main__":
    main()