#!/usr/bin/env python3
import sys
import os
sys.path.append(os.path.dirname(__file__))

from step_03_cloth_segmentation import create_cloth_segmentation_step
import numpy as np
import torch
import cv2

def test_model():
    # 스텝 생성 및 초기화
    step = create_cloth_segmentation_step()
    step.initialize()
    
    # 테스트 이미지 생성
    test_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
    test_image = cv2.resize(test_image, (512, 512))
    test_image = test_image.astype(np.float32) / 255.0
    
    # U2Net 전처리
    test_image = (test_image - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    test_tensor = torch.from_numpy(test_image).permute(2, 0, 1).unsqueeze(0).float()
    test_tensor = test_tensor.to(step.device)
    
    # 모델 가져오기
    model = step.segmentation_models['u2net_cloth']
    model.eval()
    
    print(f"🔍 모델 타입: {type(model)}")
    print(f"🔍 입력 텐서 shape: {test_tensor.shape}")
    
    # 추론 수행
    with torch.no_grad():
        output = model(test_tensor)
        print(f"🔍 U2Net 모델 출력 shape: {output.shape}")
        print(f"🔍 U2Net 모델 출력 타입: {type(output)}")
        print(f"🔍 U2Net 모델 출력 값 범위: {output.min().item():.3f} - {output.max().item():.3f}")
        
        # 마스크 생성
        mask = torch.sigmoid(output).cpu().numpy()[0, 0]
        mask = (mask > 0.5).astype(np.uint8)
        print(f"🔍 마스크 shape: {mask.shape}")
        print(f"🔍 마스크 값 범위: {mask.min()} - {mask.max()}")
        print(f"🔍 마스크 평균: {np.mean(mask):.3f}")

if __name__ == "__main__":
    test_model()
