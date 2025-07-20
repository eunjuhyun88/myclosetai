#!/usr/bin/env python3
"""
🔍 현재 설정 검증 및 누락 모델 확인
"""

import os
import json
from pathlib import Path

def check_model_files():
    """탐지된 모델 파일들이 실제로 존재하는지 확인"""
    
    print("🔍 모델 파일 존재 여부 검증...")
    
    # 탐지 결과 기반 경로들
    model_paths = [
        # 탐지된 실제 경로들
        "/Users/gimdudeul/MVP/mycloset-ai/backend/app/ai_pipeline/models/downloads/ootdiffusion/snapshots/c79f9dd0585743bea82a39261cc09a24040bc4f9/checkpoints/humanparsing/exp-schp-201908301523-atr.pth",
        "/Users/gimdudeul/MVP/mycloset-ai/backend/app/ai_pipeline/models/ai_models/checkpoints/step_03_cloth_segmentation/u2net.pth", 
        "/Users/gimdudeul/MVP/mycloset-ai/backend/app/ai_pipeline/models/ai_models/downloads/models--runwayml--stable-diffusion-v1-5/snapshots/451f4fe16113bff5a5d2269ed5ad43b0592e9a14/unet/diffusion_pytorch_model.bin",
        "/Users/gimdudeul/MVP/mycloset-ai/backend/app/ai_pipeline/models/ai_models/checkpoints/step_02_pose_estimation/openpose_body.pth"
    ]
    
    existing_files = []
    missing_files = []
    
    for model_path in model_paths:
        path = Path(model_path)
        if path.exists():
            size_mb = path.stat().st_size / (1024 * 1024)
            existing_files.append({
                'path': str(path),
                'name': path.name,
                'size_mb': size_mb,
                'step': get_step_from_path(str(path))
            })
            print(f"   ✅ {path.name} ({size_mb:.1f}MB)")
        else:
            missing_files.append(str(path))
            print(f"   ❌ {path.name} - 파일 없음")
    
    return existing_files, missing_files

def get_step_from_path(path_str):
    """경로에서 Step 추정"""
    path_lower = path_str.lower()
    
    if 'human' in path_lower or 'parsing' in path_lower or 'schp' in path_lower:
        return 'HumanParsingStep'
    elif 'pose' in path_lower or 'openpose' in path_lower:
        return 'PoseEstimationStep'
    elif 'cloth' in path_lower or 'segmentation' in path_lower or 'u2net' in path_lower:
        return 'ClothSegmentationStep'
    elif 'diffusion' in path_lower or 'stable' in path_lower or 'virtual' in path_lower:
        return 'VirtualFittingStep'
    elif 'geometric' in path_lower or 'matching' in path_lower:
        return 'GeometricMatchingStep'
    elif 'warping' in path_lower or 'warp' in path_lower:
        return 'ClothWarpingStep'
    else:
        return 'UnknownStep'

def find_additional_models():
    """추가로 사용 가능한 모델들 찾기"""
    
    print("🔍 추가 모델 파일 탐색...")
    
    base_dir = Path("/Users/gimdudeul/MVP/mycloset-ai/backend/app/ai_pipeline/models")
    
    additional_models = []
    
    if base_dir.exists():
        # .pth, .pt, .bin, .safetensors 파일들 찾기
        for ext in ['*.pth', '*.pt', '*.bin', '*.safetensors']:
            for file_path in base_dir.rglob(ext):
                if file_path.is_file():
                    size_mb = file_path.stat().st_size / (1024 * 1024)
                    if size_mb > 10:  # 10MB 이상만
                        additional_models.append({
                            'path': str(file_path),
                            'name': file_path.name,
                            'size_mb': size_mb,
                            'step': get_step_from_path(str(file_path))
                        })
    
    # 크기순 정렬
    additional_models.sort(key=lambda x: x['size_mb'], reverse=True)
    
    return additional_models

def check_step_coverage():
    """각 Step별 모델 커버리지 확인"""
    
    print("📊 Step별 모델 커버리지 분석...")
    
    required_steps = {
        'HumanParsingStep': '사람 파싱',
        'PoseEstimationStep': '포즈 추정', 
        'ClothSegmentationStep': '옷 분할',
        'GeometricMatchingStep': '기하학적 매칭',
        'ClothWarpingStep': '옷 변형',
        'VirtualFittingStep': '가상 피팅',
        'PostProcessingStep': '후처리',
        'QualityAssessmentStep': '품질 평가'
    }
    
    existing_files, _ = check_model_files()
    additional_models = find_additional_models()
    
    all_models = existing_files + additional_models
    
    step_coverage = {}
    for step_name, step_desc in required_steps.items():
        models_for_step = [m for m in all_models if m['step'] == step_name]
        step_coverage[step_name] = {
            'description': step_desc,
            'models': models_for_step,
            'count': len(models_for_step),
            'covered': len(models_for_step) > 0
        }
    
    return step_coverage

def generate_download_guide():
    """누락된 모델 다운로드 가이드 생성"""
    
    step_coverage = check_step_coverage()
    
    uncovered_steps = [
        step for step, info in step_coverage.items() 
        if not info['covered']
    ]
    
    if uncovered_steps:
        print(f"\n📋 누락된 모델 다운로드 가이드:")
        print("=" * 50)
        
        download_commands = {
            'GeometricMatchingStep': 'wget https://github.com/shadow2496/HR-VITON/releases/download/3.0.0/gmm_final.pth',
            'ClothWarpingStep': 'wget https://github.com/shadow2496/HR-VITON/releases/download/3.0.0/tom_final.pth', 
            'PostProcessingStep': 'wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth',
            'QualityAssessmentStep': 'wget https://github.com/openai/CLIP/raw/main/clip/bpe_simple_vocab_16e6.txt'
        }
        
        for step in uncovered_steps:
            step_info = step_coverage[step]
            print(f"\n❌ {step} ({step_info['description']})")
            
            if step in download_commands:
                print(f"   💾 다운로드: {download_commands[step]}")
            else:
                print(f"   💡 수동 설치 필요")
    
    else:
        print(f"\n✅ 모든 Step에 모델이 준비되어 있습니다!")

def main():
    """메인 검증 함수"""
    
    print("🔍 MyCloset AI 모델 설정 검증 시작...")
    print("=" * 60)
    
    # 1. 기본 모델 파일 확인
    existing_files, missing_files = check_model_files()
    
    print(f"\n📊 기본 검증 결과:")
    print(f"   ✅ 존재하는 파일: {len(existing_files)}개")
    print(f"   ❌ 누락된 파일: {len(missing_files)}개")
    
    # 2. 추가 모델 탐색
    additional_models = find_additional_models()
    
    print(f"\n🔍 추가 모델 탐색 결과:")
    print(f"   📦 발견된 추가 모델: {len(additional_models)}개")
    
    if additional_models:
        print("   상위 5개 (크기순):")
        for model in additional_models[:5]:
            print(f"      📄 {model['name']} ({model['size_mb']:.1f}MB) - {model['step']}")
    
    # 3. Step 커버리지 분석
    step_coverage = check_step_coverage()
    
    print(f"\n📋 Step별 커버리지:")
    covered_count = 0
    for step_name, info in step_coverage.items():
        status = "✅" if info['covered'] else "❌"
        covered_count += 1 if info['covered'] else 0
        print(f"   {status} {step_name}: {info['count']}개 모델")
    
    print(f"\n📊 전체 요약:")
    print(f"   🎯 커버된 Step: {covered_count}/8개 ({covered_count/8*100:.1f}%)")
    
    total_models = len(existing_files) + len(additional_models)
    total_size = sum(m['size_mb'] for m in existing_files + additional_models)
    print(f"   📦 총 모델 수: {total_models}개")
    print(f"   💾 총 크기: {total_size/1024:.2f}GB")
    
    # 4. 다운로드 가이드
    generate_download_guide()
    
    print(f"\n🎯 권장 다음 단계:")
    print(f"   1. python update_model_config.py 실행")
    print(f"   2. 누락된 모델 다운로드 (필요시)")
    print(f"   3. ModelLoader 재시작 테스트")

if __name__ == "__main__":
    main()