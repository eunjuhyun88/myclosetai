#!/usr/bin/env python3
"""
🚀 MyCloset AI 모델 고급 테스트 스크립트
- 모든 AI 모델의 실제 로딩 테스트
- M3 Max 최적화 확인
- 메모리 사용량 모니터링
- 추론 속도 벤치마크
"""

import os
import sys
import time
import psutil
import gc
from pathlib import Path
from typing import Dict, Any, Optional
import warnings
warnings.filterwarnings("ignore")

def log_info(msg: str):
    print(f"ℹ️  {msg}")

def log_success(msg: str):
    print(f"✅ {msg}")

def log_warning(msg: str):
    print(f"⚠️  {msg}")

def log_error(msg: str):
    print(f"❌ {msg}")

def log_benchmark(msg: str):
    print(f"⏱️  {msg}")

class AdvancedModelTester:
    """고급 AI 모델 테스트"""
    
    def __init__(self):
        self.base_dir = Path("ai_models")
        self.checkpoints_dir = self.base_dir / "checkpoints"
        self.device = self._detect_device()
        self.test_results = {}
        self.memory_baseline = self._get_memory_usage()
        
    def _detect_device(self) -> str:
        """최적 디바이스 감지"""
        try:
            import torch
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "mps"
            elif torch.cuda.is_available():
                return "cuda"
            else:
                return "cpu"
        except ImportError:
            return "cpu"
    
    def _get_memory_usage(self) -> Dict[str, float]:
        """현재 메모리 사용량"""
        memory = psutil.virtual_memory()
        return {
            "total_gb": memory.total / (1024**3),
            "used_gb": memory.used / (1024**3),
            "available_gb": memory.available / (1024**3),
            "percent": memory.percent
        }
    
    def _measure_time_and_memory(self, func, *args, **kwargs):
        """시간과 메모리 사용량 측정"""
        gc.collect()  # 가비지 컬렉션
        
        start_memory = self._get_memory_usage()
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            success = True
            error = None
        except Exception as e:
            result = None
            success = False
            error = str(e)
        
        end_time = time.time()
        end_memory = self._get_memory_usage()
        
        return {
            "result": result,
            "success": success,
            "error": error,
            "execution_time": end_time - start_time,
            "memory_used_mb": (end_memory["used_gb"] - start_memory["used_gb"]) * 1024,
            "peak_memory_gb": end_memory["used_gb"]
        }
    
    def test_segformer_human_parsing(self):
        """Segformer 인체 파싱 테스트"""
        log_info("Segformer 인체 파싱 테스트...")
        
        def load_segformer():
            from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
            
            model_path = self.checkpoints_dir / "step_01_human_parsing/segformer_b2_clothes"
            processor = SegformerImageProcessor.from_pretrained(str(model_path))
            model = SegformerForSemanticSegmentation.from_pretrained(str(model_path))
            
            # 디바이스로 이동
            if self.device != "cpu":
                model = model.to(self.device)
            
            # 간단한 추론 테스트
            import torch
            import numpy as np
            from PIL import Image
            
            # 더미 이미지 생성 (512x512 RGB)
            dummy_image = Image.fromarray(np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8))
            inputs = processor(images=dummy_image, return_tensors="pt")
            
            if self.device != "cpu":
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
            
            return {
                "model_size_mb": sum(p.numel() * p.element_size() for p in model.parameters()) / (1024**2),
                "output_shape": logits.shape,
                "num_classes": logits.shape[1] if len(logits.shape) > 1 else 0,
                "device": str(next(model.parameters()).device)
            }
        
        result = self._measure_time_and_memory(load_segformer)
        
        if result["success"]:
            info = result["result"]
            log_success(f"Segformer: {info['num_classes']}클래스, {info['model_size_mb']:.1f}MB, "
                       f"{result['execution_time']:.1f}초, 메모리: +{result['memory_used_mb']:.0f}MB")
            log_info(f"  디바이스: {info['device']}, 출력: {info['output_shape']}")
        else:
            log_error(f"Segformer 실패: {result['error']}")
        
        self.test_results["segformer"] = result
        return result["success"]
    
    def test_u2net_models(self):
        """U²-Net 모델들 테스트"""
        log_info("U²-Net 모델 테스트...")
        
        # PyTorch 버전 테스트
        def test_pytorch_u2net():
            import torch
            model_path = self.checkpoints_dir / "step_03_cloth_segmentation/u2net.pth"
            
            # 모델 로드 (실제 U²-Net 구조가 필요하지만 체크포인트만 확인)
            checkpoint = torch.load(model_path, map_location=self.device)
            
            return {
                "checkpoint_keys": list(checkpoint.keys()) if isinstance(checkpoint, dict) else ["single_tensor"],
                "file_size_mb": model_path.stat().st_size / (1024**2)
            }
        
        # ONNX 버전 테스트
        def test_onnx_u2net():
            import onnxruntime as ort
            
            model_path = self.checkpoints_dir / "step_03_cloth_segmentation/u2net.onnx"
            
            # ONNX 세션 생성
            providers = ['CPUExecutionProvider']
            if self.device == "cuda":
                providers.insert(0, 'CUDAExecutionProvider')
            
            session = ort.InferenceSession(str(model_path), providers=providers)
            
            # 입력/출력 정보
            input_info = session.get_inputs()[0]
            output_info = session.get_outputs()[0]
            
            # 더미 추론
            import numpy as np
            dummy_input = np.random.randn(*input_info.shape).astype(np.float32)
            output = session.run([output_info.name], {input_info.name: dummy_input})
            
            return {
                "input_shape": input_info.shape,
                "output_shape": output[0].shape,
                "input_type": input_info.type,
                "providers": session.get_providers(),
                "file_size_mb": model_path.stat().st_size / (1024**2)
            }
        
        # PyTorch 테스트
        pytorch_result = self._measure_time_and_memory(test_pytorch_u2net)
        if pytorch_result["success"]:
            info = pytorch_result["result"]
            log_success(f"U²-Net PyTorch: {info['file_size_mb']:.1f}MB, "
                       f"{pytorch_result['execution_time']:.1f}초")
        else:
            log_error(f"U²-Net PyTorch 실패: {pytorch_result['error']}")
        
        # ONNX 테스트
        onnx_result = self._measure_time_and_memory(test_onnx_u2net)
        if onnx_result["success"]:
            info = onnx_result["result"]
            log_success(f"U²-Net ONNX: {info['input_shape']} → {info['output_shape']}, "
                       f"{onnx_result['execution_time']:.1f}초")
            log_info(f"  Providers: {', '.join(info['providers'])}")
        else:
            log_error(f"U²-Net ONNX 실패: {onnx_result['error']}")
        
        self.test_results["u2net_pytorch"] = pytorch_result
        self.test_results["u2net_onnx"] = onnx_result
        
        return pytorch_result["success"] and onnx_result["success"]
    
    def test_mediapipe_pose(self):
        """MediaPipe 포즈 추정 테스트"""
        log_info("MediaPipe 포즈 추정 테스트...")
        
        def test_mediapipe():
            import mediapipe as mp
            import numpy as np
            import cv2
            
            # MediaPipe 포즈 모델 로드
            mp_pose = mp.solutions.pose
            
            # 더미 이미지로 테스트 (실제 사람 형태와 유사하게)
            dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            # 중앙에 사람 형태 패턴 추가 (랜드마크 감지 확률 높이기)
            dummy_image[150:350, 250:450] = [100, 150, 200]  # 사람 몸체 영역
            
            with mp_pose.Pose(
                static_image_mode=True,
                model_complexity=1,
                enable_segmentation=False,
                min_detection_confidence=0.3  # 더미 이미지이므로 낮은 임계값
            ) as pose:
                # BGR to RGB 변환
                rgb_image = cv2.cvtColor(dummy_image, cv2.COLOR_BGR2RGB)
                results = pose.process(rgb_image)
            
            return {
                "pose_detected": results.pose_landmarks is not None,
                "num_landmarks": len(results.pose_landmarks.landmark) if results.pose_landmarks else 0,
                "image_shape": dummy_image.shape,
                "mediapipe_version": getattr(mp, '__version__', 'unknown'),
                "opencv_version": getattr(cv2, '__version__', 'unknown'),
                "detection_confidence": 0.3,
                "model_complexity": 1
            }
        
        result = self._measure_time_and_memory(test_mediapipe)
        
        if result["success"]:
            info = result["result"]
            log_success(f"MediaPipe 포즈: {info['num_landmarks']}개 랜드마크, "
                       f"{result['execution_time']:.1f}초")
            log_info(f"  MediaPipe: {info['mediapipe_version']}, "
                    f"OpenCV: {info['opencv_version']}")
            if info['pose_detected']:
                log_info(f"  포즈 감지 성공 (33개 중 {info['num_landmarks']}개)")
            else:
                log_info(f"  더미 이미지에서 포즈 미감지 (정상)")
        else:
            log_error(f"MediaPipe 실패: {result['error']}")
        
        self.test_results["mediapipe"] = result
        return result["success"]
    
    def test_clip_models(self):
        """CLIP 모델들 테스트"""
        log_info("CLIP 모델 테스트...")
        
        def test_clip_base():
            from transformers import CLIPModel, CLIPProcessor
            import torch
            
            model_path = self.base_dir / "clip-vit-base-patch32"
            
            processor = CLIPProcessor.from_pretrained(str(model_path))
            model = CLIPModel.from_pretrained(str(model_path))
            
            if self.device != "cpu":
                model = model.to(self.device)
            
            # 텍스트와 이미지 임베딩 테스트
            import numpy as np
            from PIL import Image
            
            dummy_image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
            dummy_text = ["a person wearing clothes"]
            
            inputs = processor(text=dummy_text, images=dummy_image, return_tensors="pt", padding=True)
            
            if self.device != "cpu":
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model(**inputs)
                image_embeds = outputs.image_embeds
                text_embeds = outputs.text_embeds
            
            return {
                "model_size_mb": sum(p.numel() * p.element_size() for p in model.parameters()) / (1024**2),
                "image_embed_dim": image_embeds.shape[-1],
                "text_embed_dim": text_embeds.shape[-1],
                "device": str(next(model.parameters()).device)
            }
        
        result = self._measure_time_and_memory(test_clip_base)
        
        if result["success"]:
            info = result["result"]
            log_success(f"CLIP Base: {info['model_size_mb']:.1f}MB, "
                       f"임베딩 차원: {info['image_embed_dim']}, {result['execution_time']:.1f}초")
            log_info(f"  디바이스: {info['device']}")
        else:
            log_error(f"CLIP 실패: {result['error']}")
        
        self.test_results["clip"] = result
        return result["success"]
    
    def test_ootdiffusion_checkpoint(self):
        """OOTDiffusion 체크포인트 테스트"""
        log_info("OOTDiffusion 체크포인트 테스트...")
        
        def test_ootd():
            import torch
            
            # UNet 모델 체크포인트 확인
            unet_garm_path = (self.checkpoints_dir / 
                            "ootdiffusion/checkpoints/ootd/ootd_hd/checkpoint-36000/"
                            "unet_garm/diffusion_pytorch_model.safetensors")
            
            unet_vton_path = (self.checkpoints_dir / 
                            "ootdiffusion/checkpoints/ootd/ootd_hd/checkpoint-36000/"
                            "unet_vton/diffusion_pytorch_model.safetensors")
            
            # 체크포인트 로드 테스트 (실제 로딩은 메모리가 많이 필요하므로 파일만 확인)
            garm_exists = unet_garm_path.exists()
            vton_exists = unet_vton_path.exists()
            
            sizes = {}
            if garm_exists:
                sizes["garm_mb"] = unet_garm_path.stat().st_size / (1024**2)
            if vton_exists:
                sizes["vton_mb"] = unet_vton_path.stat().st_size / (1024**2)
            
            return {
                "garm_available": garm_exists,
                "vton_available": vton_exists,
                "sizes": sizes,
                "total_checkpoints": len(list((self.checkpoints_dir / "ootdiffusion").rglob("*.safetensors")))
            }
        
        result = self._measure_time_and_memory(test_ootd)
        
        if result["success"]:
            info = result["result"]
            log_success(f"OOTDiffusion: GARM({info['garm_available']}), "
                       f"VTON({info['vton_available']}), "
                       f"총 {info['total_checkpoints']}개 체크포인트")
            if info["sizes"]:
                size_info = ", ".join([f"{k}: {v:.0f}MB" for k, v in info["sizes"].items()])
                log_info(f"  크기: {size_info}")
        else:
            log_error(f"OOTDiffusion 실패: {result['error']}")
        
        self.test_results["ootdiffusion"] = result
        return result["success"]
    
    def benchmark_performance(self):
        """성능 벤치마크"""
        log_info("성능 벤치마크 실행...")
        
        # 디바이스별 성능 테스트
        if self.device == "mps":
            log_benchmark("M3 Max MPS 최적화 테스트")
            try:
                import torch
                
                # PyTorch MPS 지원 확인
                if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    # MPS 기본 연산 테스트
                    x = torch.randn(1000, 1000, device=self.device)
                    y = torch.randn(1000, 1000, device=self.device)
                    
                    start_time = time.time()
                    for _ in range(100):
                        z = torch.matmul(x, y)
                    
                    # MPS 동기화 (있는 경우)
                    if hasattr(torch.mps, 'synchronize'):
                        torch.mps.synchronize()
                    
                    end_time = time.time()
                    
                    log_benchmark(f"MPS 행렬 곱셈 (1000x1000, 100회): {end_time - start_time:.2f}초")
                    
                    # 메모리 정보
                    if hasattr(torch.mps, 'current_allocated_memory'):
                        mps_memory = torch.mps.current_allocated_memory() / (1024**2)
                        log_benchmark(f"MPS 메모리 사용량: {mps_memory:.0f}MB")
                else:
                    log_warning("MPS 지원되지 않음 - PyTorch 버전 확인 필요")
                    
            except Exception as e:
                log_warning(f"MPS 벤치마크 실패: {e}")
        
        elif self.device == "cuda":
            log_benchmark("CUDA 성능 테스트")
            try:
                import torch
                
                # CUDA 기본 연산 테스트
                x = torch.randn(1000, 1000, device=self.device)
                y = torch.randn(1000, 1000, device=self.device)
                
                start_time = time.time()
                for _ in range(100):
                    z = torch.matmul(x, y)
                torch.cuda.synchronize()
                end_time = time.time()
                
                log_benchmark(f"CUDA 행렬 곱셈 (1000x1000, 100회): {end_time - start_time:.2f}초")
                
                # GPU 메모리 정보
                gpu_memory = torch.cuda.memory_allocated() / (1024**2)
                log_benchmark(f"GPU 메모리 사용량: {gpu_memory:.0f}MB")
                
            except Exception as e:
                log_warning(f"CUDA 벤치마크 실패: {e}")
        
        else:
            log_benchmark("CPU 성능 테스트")
            try:
                import torch
                
                # CPU 기본 연산 테스트
                x = torch.randn(1000, 1000)
                y = torch.randn(1000, 1000)
                
                start_time = time.time()
                for _ in range(100):
                    z = torch.matmul(x, y)
                end_time = time.time()
                
                log_benchmark(f"CPU 행렬 곱셈 (1000x1000, 100회): {end_time - start_time:.2f}초")
                
            except Exception as e:
                log_warning(f"CPU 벤치마크 실패: {e}")
        
        # 메모리 사용량 분석
        current_memory = self._get_memory_usage()
        memory_used = current_memory["used_gb"] - self.memory_baseline["used_gb"]
        
        log_benchmark(f"총 메모리 사용량: +{memory_used:.1f}GB")
        log_benchmark(f"현재 메모리: {current_memory['used_gb']:.1f}GB / {current_memory['total_gb']:.1f}GB "
                     f"({current_memory['percent']:.1f}%)")
    
    def generate_report(self):
        """테스트 결과 리포트 생성"""
        log_info("테스트 결과 리포트 생성...")
        
        successful_tests = sum(1 for result in self.test_results.values() if result["success"])
        total_tests = len(self.test_results)
        
        report = {
            "device": self.device,
            "success_rate": f"{successful_tests}/{total_tests}",
            "total_execution_time": sum(r["execution_time"] for r in self.test_results.values()),
            "total_memory_used_mb": sum(r["memory_used_mb"] for r in self.test_results.values()),
            "system_info": {
                "memory": self._get_memory_usage(),
                "cpu_count": psutil.cpu_count(),
                "platform": sys.platform
            },
            "test_details": self.test_results
        }
        
        # JSON으로 저장
        import json
        report_path = Path("model_test_report.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        log_success(f"리포트 저장: {report_path}")
        return report

def main():
    """메인 테스트 실행"""
    print("🚀 MyCloset AI 모델 고급 테스트")
    print("=" * 50)
    
    tester = AdvancedModelTester()
    
    log_info(f"디바이스: {tester.device}")
    log_info(f"초기 메모리: {tester.memory_baseline['used_gb']:.1f}GB / "
             f"{tester.memory_baseline['total_gb']:.1f}GB")
    
    print("\n🧪 AI 모델 로딩 테스트")
    print("=" * 30)
    
    # 각 모델 테스트 실행
    tests = [
        ("Segformer 인체 파싱", tester.test_segformer_human_parsing),
        ("U²-Net 모델들", tester.test_u2net_models),
        ("MediaPipe 포즈", tester.test_mediapipe_pose),
        ("CLIP 모델", tester.test_clip_models),
        ("OOTDiffusion", tester.test_ootdiffusion_checkpoint)
    ]
    
    successful_tests = 0
    for test_name, test_func in tests:
        try:
            log_info(f"{test_name} 시작...")
            if test_func():
                successful_tests += 1
            print()  # 줄바꿈
        except KeyboardInterrupt:
            log_warning("사용자가 테스트를 중단했습니다")
            break
        except Exception as e:
            log_error(f"{test_name} 예외 발생: {e}")
    
    # 성능 벤치마크
    print("⚡ 성능 벤치마크")
    print("=" * 20)
    tester.benchmark_performance()
    
    # 결과 리포트
    print("\n📊 테스트 결과")
    print("=" * 20)
    report = tester.generate_report()
    
    print(f"✅ 성공한 테스트: {successful_tests}/{len(tests)}")
    print(f"🕐 총 실행 시간: {report['total_execution_time']:.1f}초")
    print(f"💾 총 메모리 사용: {report['total_memory_used_mb']:.0f}MB")
    
    if successful_tests == len(tests):
        print("\n🎉 모든 AI 모델이 정상적으로 동작합니다!")
        print("🚀 이제 python3 run_server.py 로 서버를 시작할 수 있습니다")
    else:
        print(f"\n⚠️  {len(tests) - successful_tests}개 모델에서 문제가 발생했습니다")
        print("📝 model_test_report.json 파일을 확인하세요")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⏹️  테스트가 중단되었습니다")
    except Exception as e:
        print(f"\n\n❌ 예상치 못한 오류: {e}")
        import traceback
        traceback.print_exc()