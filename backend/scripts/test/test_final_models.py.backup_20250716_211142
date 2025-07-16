#!/usr/bin/env python3
"""
간단한 모델 테스트
"""
import os
import sys
from pathlib import Path

def test_models():
    """모델 로딩 테스트"""
    print("🚀 모델 테스트 시작")
    print("=" * 40)
    
    base_dir = Path("ai_models/checkpoints")
    
    # 1. Segformer 테스트
    segformer_path = base_dir / "step_01_human_parsing/segformer_b2_clothes"
    if segformer_path.exists():
        try:
            from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
            processor = SegformerImageProcessor.from_pretrained(str(segformer_path))
            model = SegformerForSemanticSegmentation.from_pretrained(str(segformer_path))
            print("✅ Segformer 인체 파싱: 정상")
        except Exception as e:
            print(f"❌ Segformer 인체 파싱: 실패 - {e}")
    else:
        print("❌ Segformer 인체 파싱: 파일 없음")
    
    # 2. U²-Net ONNX 테스트
    u2net_path = base_dir / "step_03_cloth_segmentation/u2net.onnx"
    if u2net_path.exists():
        try:
            import onnxruntime as ort
            session = ort.InferenceSession(str(u2net_path))
            print("✅ U²-Net ONNX: 정상")
        except Exception as e:
            print(f"❌ U²-Net ONNX: 실패 - {e}")
    else:
        print("❌ U²-Net ONNX: 파일 없음")
    
    # 3. MediaPipe 테스트
    mediapipe_path = base_dir / "step_02_pose_estimation/pose_landmarker.task"
    if mediapipe_path.exists():
        size_mb = mediapipe_path.stat().st_size / (1024**2)
        print(f"✅ MediaPipe 포즈: 정상 ({size_mb:.1f}MB)")
    else:
        print("❌ MediaPipe 포즈: 파일 없음")
    
    # 4. CLIP 테스트 (safetensors만)
    clip_path = base_dir / "shared_encoder/clip-vit-base-patch32"
    if clip_path.exists():
        safetensors_files = list(clip_path.glob("*.safetensors"))
        if safetensors_files:
            print(f"✅ CLIP (safetensors): 정상 ({len(safetensors_files)}개 파일)")
        else:
            print("⚠️ CLIP: safetensors 파일 없음")
    else:
        print("❌ CLIP: 디렉토리 없음")
    
    print("\n🎉 테스트 완료!")

if __name__ == "__main__":
    test_models()
