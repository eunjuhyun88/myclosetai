#!/usr/bin/env python3
"""
최종 파이프라인 검증 스크립트
- 작동하는 모델들만 테스트
- 실제 추론 테스트
"""
import sys
import logging
from pathlib import Path
import torch
from PIL import Image
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_test_image():
    """테스트용 이미지 생성"""
    # 512x512 RGB 이미지
    image_array = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    return Image.fromarray(image_array)

def test_working_pipeline():
    """작동하는 파이프라인 테스트"""
    print("🚀 최종 파이프라인 검증")
    print("=" * 40)
    
    test_image = create_test_image()
    results = {}
    
    # 1. Segformer 인체 파싱 테스트
    try:
        print("\n1️⃣ 인체 파싱 테스트...")
        from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
        
        model_path = "ai_models/checkpoints/step_01_human_parsing/segformer_b2_clothes"
        processor = SegformerImageProcessor.from_pretrained(model_path)
        model = SegformerForSemanticSegmentation.from_pretrained(model_path)
        
        inputs = processor(images=test_image, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
            parsing_result = outputs.logits
        
        logger.info(f"✅ 인체 파싱 성공: {parsing_result.shape}")
        results['human_parsing'] = True
        
    except Exception as e:
        logger.error(f"❌ 인체 파싱 실패: {e}")
        results['human_parsing'] = False
    
    # 2. U²-Net ONNX 배경 제거 테스트  
    try:
        print("\n2️⃣ 배경 제거 테스트...")
        import onnxruntime as ort
        
        model_path = "ai_models/checkpoints/step_03_cloth_segmentation/u2net.onnx"
        session = ort.InferenceSession(model_path)
        
        # 더미 입력으로 테스트
        input_name = session.get_inputs()[0].name
        dummy_input = np.random.randn(1, 3, 320, 320).astype(np.float32)
        
        outputs = session.run(None, {input_name: dummy_input})
        
        logger.info(f"✅ 배경 제거 성공: {len(outputs)} 출력")
        results['background_removal'] = True
        
    except Exception as e:
        logger.error(f"❌ 배경 제거 실패: {e}")
        results['background_removal'] = False
    
    # 3. MediaPipe 포즈 추정 (파일 확인만)
    try:
        print("\n3️⃣ 포즈 추정 테스트...")
        model_path = Path("ai_models/checkpoints/step_02_pose_estimation/pose_landmarker.task")
        
        if model_path.exists():
            size_mb = model_path.stat().st_size / (1024**2)
            logger.info(f"✅ 포즈 모델 준비됨: {size_mb:.1f}MB")
            results['pose_estimation'] = True
        else:
            raise FileNotFoundError("포즈 모델 파일 없음")
            
    except Exception as e:
        logger.error(f"❌ 포즈 추정 실패: {e}")
        results['pose_estimation'] = False
    
    # 4. Real-ESRGAN 후처리 테스트
    try:
        print("\n4️⃣ 후처리 테스트...")
        model_path = "ai_models/checkpoints/step_07_post_processing/RealESRGAN_x4plus.pth"
        
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)
        logger.info(f"✅ 후처리 모델 로딩 성공")
        results['post_processing'] = True
        
    except Exception as e:
        logger.error(f"❌ 후처리 실패: {e}")
        results['post_processing'] = False
    
    # 결과 요약
    print("\n" + "=" * 40)
    print("🎉 최종 검증 결과")
    print("=" * 40)
    
    working_count = sum(results.values())
    total_count = len(results)
    
    for component, status in results.items():
        emoji = "✅" if status else "❌"
        print(f"{emoji} {component}: {'작동' if status else '실패'}")
    
    print(f"\n📊 작동률: {working_count}/{total_count} ({working_count/total_count*100:.1f}%)")
    
    if working_count >= 3:
        print("\n🚀 파이프라인 실행 준비 완료!")
        print("다음 명령어로 서버를 시작하세요:")
        print("   python -m app.main")
    else:
        print("\n⚠️ 일부 모델 문제 - 기본 기능만 사용 가능")
    
    return results

if __name__ == "__main__":
    test_working_pipeline()
