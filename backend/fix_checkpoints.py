import sys
sys.path.append('.')
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

try:
    import torch
    print('✅ PyTorch 로드됨')
except:
    print('❌ PyTorch 없음')
    exit(1)

try:
    import safetensors.torch as st
    print('✅ Safetensors 로드됨')
    SAFETENSORS_AVAILABLE = True
except:
    print('⚠️ Safetensors 없음')
    SAFETENSORS_AVAILABLE = False

# AI 모델 경로 찾기
ai_models = Path('ai_models')
if not ai_models.exists():
    ai_models = Path('../ai_models') 
if not ai_models.exists():
    print('❌ ai_models 없음')
    exit(1)

print(f'📁 AI 모델 루트: {ai_models}')

# 실패한 파일들
failed_files = [
    'step_02_pose_estimation/ultra_models/diffusion_pytorch_model.safetensors',
    'checkpoints/step_03_cloth_segmentation/u2net_alternative.pth',
    'checkpoints/step_05_cloth_warping/RealVisXL_V4.0.safetensors',
    'step_06_virtual_fitting/ootdiffusion/checkpoints/ootd/ootd_hd/checkpoint-36000/unet_vton/diffusion_pytorch_model.safetensors',
    'step_06_virtual_fitting/ootdiffusion/checkpoints/ootd/ootd_hd/checkpoint-36000/unet_garm/diffusion_pytorch_model.safetensors',
    'step_06_virtual_fitting/unet/diffusion_pytorch_model.safetensors',
    'step_06_virtual_fitting/ootdiffusion/checkpoints/ootd/ootd_dc/checkpoint-36000/unet_vton/diffusion_pytorch_model.safetensors',
    'step_06_virtual_fitting/ootdiffusion/checkpoints/ootd/ootd_dc/checkpoint-36000/unet_garm/diffusion_pytorch_model.safetensors'
]

success = 0
total = len(failed_files)

for i, file_path in enumerate(failed_files):
    full_path = ai_models / file_path
    print(f'\\n[{i+1}/{total}] {file_path}')
    
    if not full_path.exists():
        print('  📁 파일 없음')
        continue
    
    size_mb = full_path.stat().st_size / (1024*1024)
    print(f'  📏 크기: {size_mb:.1f}MB')
    
    # Safetensors 테스트
    if full_path.suffix.lower() == '.safetensors' and SAFETENSORS_AVAILABLE:
        try:
            data = st.load_file(str(full_path))
            print('  ✅ Safetensors 성공!')
            success += 1
            continue
        except Exception as e:
            print(f'  ❌ Safetensors 실패: {str(e)[:50]}')
    
    # PyTorch 테스트
    methods = [
        ('weights_only=True', lambda: torch.load(full_path, map_location='cpu', weights_only=True)),
        ('weights_only=False', lambda: torch.load(full_path, map_location='cpu', weights_only=False)),
        ('legacy', lambda: torch.load(full_path, map_location='cpu'))
    ]
    
    loaded = False
    for method_name, loader in methods:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                data = loader()
            print(f'  ✅ PyTorch {method_name} 성공!')
            success += 1
            loaded = True
            break
        except Exception as e:
            print(f'  ❌ PyTorch {method_name} 실패: {str(e)[:50]}')
    
    if not loaded:
        print('  💥 모든 로딩 방법 실패')

print(f'\\n📊 결과: {success}/{total} 성공 ({success/total*100:.1f}%)')

if success < total:
    print('\\n💡 해결책:')
    print('  1. 실패한 파일들을 Hugging Face에서 재다운로드')
    print('  2. 손상된 파일 삭제 후 재다운로드')
    print('  3. 네트워크 문제로 중단된 다운로드 재시도')
