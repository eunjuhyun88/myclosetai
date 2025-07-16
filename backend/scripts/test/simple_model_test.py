#!/usr/bin/env python3
"""
🔧 간단한 모델 테스트
다운로드된 모델들이 제대로 로드되는지 확인
"""

import os
import sys
import logging
from pathlib import Path

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class SimpleModelTester:
    """간단한 모델 테스트"""
    
    def __init__(self):
        self.models_dir = Path("ai_models/checkpoints")
        self.test_results = {}
        
    def test_segformer_parsing(self) -> bool:
        """Segformer 인체 파싱 테스트"""
        logger.info("🎯 Segformer 인체 파싱 테스트...")
        
        try:
            model_path = self.models_dir / "step_01_human_parsing" / "segformer_b2_clothes"
            
            if not model_path.exists():
                logger.warning("❌ Segformer 모델 없음")
                return False
            
            # transformers 설치 확인
            try:
                from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
            except ImportError:
                logger.info("📦 transformers 설치 중...")
                import subprocess
                subprocess.check_call([sys.executable, "-m", "pip", "install", "transformers", "torch", "torchvision"])
                from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
            
            # 모델 로드 테스트
            processor = SegformerImageProcessor.from_pretrained(str(model_path))
            model = SegformerForSemanticSegmentation.from_pretrained(str(model_path))
            
            logger.info("✅ Segformer 로드 성공")
            return True
            
        except Exception as e:
            logger.error(f"❌ Segformer 테스트 실패: {e}")
            return False
    
    def test_u2net_onnx(self) -> bool:
        """U²-Net ONNX 테스트"""
        logger.info("🎯 U²-Net ONNX 테스트...")
        
        try:
            model_path = self.models_dir / "step_03_cloth_segmentation" / "u2net.onnx"
            
            if not model_path.exists():
                logger.warning("❌ U²-Net ONNX 모델 없음")
                return False
            
            # onnxruntime 설치 확인
            try:
                import onnxruntime as ort
            except ImportError:
                logger.info("📦 onnxruntime 설치 중...")
                import subprocess
                subprocess.check_call([sys.executable, "-m", "pip", "install", "onnxruntime"])
                import onnxruntime as ort
            
            # ONNX 세션 생성 테스트
            session = ort.InferenceSession(str(model_path))
            
            # 입력 정보 확인
            input_info = session.get_inputs()[0]
            logger.info(f"ONNX 입력: {input_info.name}, shape: {input_info.shape}")
            
            logger.info("✅ U²-Net ONNX 로드 성공")
            return True
            
        except Exception as e:
            logger.error(f"❌ U²-Net ONNX 테스트 실패: {e}")
            return False
    
    def test_mediapipe_pose(self) -> bool:
        """MediaPipe 포즈 테스트"""
        logger.info("🎯 MediaPipe 포즈 테스트...")
        
        try:
            model_path = self.models_dir / "step_02_pose_estimation" / "pose_landmarker.task"
            
            if not model_path.exists():
                logger.warning("❌ MediaPipe 포즈 모델 없음")
                return False
            
            # 파일 크기 확인
            size_mb = model_path.stat().st_size / 1024 / 1024
            logger.info(f"MediaPipe 모델 크기: {size_mb:.1f}MB")
            
            if size_mb > 1:  # 1MB 이상이면 정상
                logger.info("✅ MediaPipe 포즈 모델 정상")
                return True
            else:
                logger.warning("❌ MediaPipe 모델 크기 이상")
                return False
            
        except Exception as e:
            logger.error(f"❌ MediaPipe 테스트 실패: {e}")
            return False
    
    def test_real_esrgan(self) -> bool:
        """Real-ESRGAN 테스트"""
        logger.info("🎯 Real-ESRGAN 테스트...")
        
        try:
            model_path = self.models_dir / "step_07_post_processing" / "RealESRGAN_x4plus.pth"
            
            if not model_path.exists():
                logger.warning("❌ Real-ESRGAN 모델 없음")
                return False
            
            # PyTorch 설치 확인
            try:
                import torch
            except ImportError:
                logger.info("📦 torch 설치 중...")
                import subprocess
                subprocess.check_call([sys.executable, "-m", "pip", "install", "torch", "torchvision"])
                import torch
            
            # 모델 로드 테스트
            checkpoint = torch.load(model_path, map_location='cpu')
            
            if isinstance(checkpoint, dict):
                logger.info(f"Real-ESRGAN 체크포인트 키: {len(checkpoint)} 개")
                logger.info("✅ Real-ESRGAN 로드 성공")
                return True
            else:
                logger.warning("❌ Real-ESRGAN 체크포인트 형식 이상")
                return False
            
        except Exception as e:
            logger.error(f"❌ Real-ESRGAN 테스트 실패: {e}")
            return False
    
    def test_clip_model(self) -> bool:
        """CLIP 모델 테스트"""
        logger.info("🎯 CLIP 모델 테스트...")
        
        try:
            model_path = self.models_dir / "shared_encoder" / "clip-vit-base-patch32"
            
            if not model_path.exists():
                logger.warning("❌ CLIP 모델 없음")
                return False
            
            # transformers로 CLIP 테스트
            try:
                from transformers import CLIPProcessor, CLIPModel
            except ImportError:
                logger.info("📦 transformers 설치 중...")
                import subprocess
                subprocess.check_call([sys.executable, "-m", "pip", "install", "transformers"])
                from transformers import CLIPProcessor, CLIPModel
            
            # 모델 로드 테스트
            model = CLIPModel.from_pretrained(str(model_path))
            processor = CLIPProcessor.from_pretrained(str(model_path))
            
            logger.info("✅ CLIP 로드 성공")
            return True
            
        except Exception as e:
            logger.error(f"❌ CLIP 테스트 실패: {e}")
            return False
    
    def run_all_tests(self):
        """모든 테스트 실행"""
        print("🚀 간단한 모델 테스트 시작")
        print("=" * 40)
        
        tests = [
            ("Segformer 인체 파싱", self.test_segformer_parsing),
            ("U²-Net ONNX", self.test_u2net_onnx),
            ("MediaPipe 포즈", self.test_mediapipe_pose),
            ("Real-ESRGAN", self.test_real_esrgan),
            ("CLIP", self.test_clip_model)
        ]
        
        passed = 0
        total = len(tests)
        
        for test_name, test_func in tests:
            print(f"\n{'='*30}")
            
            try:
                result = test_func()
                self.test_results[test_name] = result
                
                if result:
                    passed += 1
                    logger.info(f"✅ {test_name}: 통과")
                else:
                    logger.warning(f"❌ {test_name}: 실패")
                    
            except Exception as e:
                logger.error(f"💥 {test_name}: 오류 - {e}")
                self.test_results[test_name] = False
        
        # 결과 요약
        success_rate = passed / total
        
        print(f"\n{'='*40}")
        print(f"🎉 테스트 완료!")
        print(f"✅ 통과: {passed}/{total} ({success_rate:.1%})")
        
        # 상세 결과
        print(f"\n📊 상세 결과:")
        for test_name, result in self.test_results.items():
            status = "✅ 통과" if result else "❌ 실패"
            print(f"   {test_name}: {status}")
        
        # 권장사항
        if success_rate >= 0.8:
            print(f"\n🚀 대부분의 모델이 정상입니다!")
            print(f"📋 서버를 실행할 수 있습니다:")
            print(f"   python -m app.main")
        elif success_rate >= 0.5:
            print(f"\n⚠️ 일부 모델에 문제가 있지만 기본 기능은 가능합니다.")
        else:
            print(f"\n❌ 많은 모델에 문제가 있습니다.")
            print(f"모델을 다시 다운로드하세요:")
            print(f"   python simple_model_downloader.py")
        
        return success_rate >= 0.5

def main():
    """메인 함수"""
    try:
        tester = SimpleModelTester()
        success = tester.run_all_tests()
        return success
        
    except KeyboardInterrupt:
        print("\n❌ 중단되었습니다.")
        return False
    except Exception as e:
        logger.error(f"❌ 테스트 오류: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)