"""
FLUX.1-Kontext-dev 모델 사용 예제
"""

import torch
from diffusers import FluxKontextPipeline
from diffusers.utils import load_image
from PIL import Image
import numpy as np

def create_test_image():
    """테스트용 이미지 생성"""
    # 간단한 색상 이미지 생성
    test_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    return Image.fromarray(test_image)

def edit_image_with_flux():
    """FLUX 모델을 사용한 이미지 편집"""
    try:
        # 모델 로드
        print("모델 로드 중...")
        pipe = FluxKontextPipeline.from_pretrained(
            "./models/flux_kontext/local",
            torch_dtype=torch.float16,
            use_safetensors=True,
            low_cpu_mem_usage=True
        )
        
        # GPU 사용 설정
        if torch.cuda.is_available():
            pipe = pipe.to("cuda")
            print(f"GPU 사용: {torch.cuda.get_device_name()}")
        else:
            print("CPU 사용")
        
        # 테스트 이미지 생성
        input_image = create_test_image()
        input_image.save("input_test.jpg")
        print("입력 이미지 저장: input_test.jpg")
        
        # 이미지 편집
        print("이미지 편집 중...")
        result = pipe(
            image=input_image,
            prompt="Make the image more vibrant and colorful",
            guidance_scale=2.5,
            num_inference_steps=20
        ).images[0]
        
        # 결과 저장
        result.save("output_edited.jpg")
        print("편집된 이미지 저장: output_edited.jpg")
        
        return True
        
    except Exception as e:
        print(f"오류 발생: {str(e)}")
        return False

if __name__ == "__main__":
    print("=== FLUX 모델 사용 예제 ===")
    success = edit_image_with_flux()
    
    if success:
        print("✅ 이미지 편집 완료!")
    else:
        print("❌ 이미지 편집 실패")
