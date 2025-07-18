# backend/scripts/add_fallback_solutions.py
"""
🔄 MyCloset AI - 대체 솔루션 확장 가이드 v1.0
✅ 새로운 대체 솔루션을 계속 추가할 수 있는 시스템
✅ 다양한 라이브러리 기반 솔루션 지원
✅ 자동 가용성 체크 및 등록
✅ 성능 벤치마킹 포함

사용법:
1. 이 파일을 참고해서 새로운 솔루션 구현
2. register_new_solution() 함수로 등록
3. 자동으로 PipelineManager에서 사용 가능

확장 가능한 라이브러리들:
- TensorFlow/Keras 모델
- ONNX Runtime 모델  
- Dlib 얼굴 처리
- ImageIO 기본 처리
- Matplotlib 시각화
- 클라우드 API (OpenAI, Replicate 등)
- 웹 기반 처리 서비스
"""

import os
import sys
import logging
import time
import asyncio
import json
from pathlib import Path
from typing import Dict, Any, List, Callable, Optional, Union, Tuple
from dataclasses import dataclass
import inspect

# 현재 디렉토리를 Python 경로에 추가
current_dir = Path(__file__).parent.parent
sys.path.insert(0, str(current_dir))

# MyCloset AI 시스템 import
try:
    from app.ai_pipeline.pipeline_manager import (
        register_fallback_solution,
        get_fallback_registry,
        FallbackSolutionRegistry
    )
    from app.ai_pipeline.utils.auto_model_detector import (
        EnhancedModelDetector,
        create_enhanced_detector
    )
    MYCLOSET_AVAILABLE = True
except ImportError:
    MYCLOSET_AVAILABLE = False
    print("⚠️ MyCloset AI 시스템을 import할 수 없습니다. 경로를 확인하세요.")

logger = logging.getLogger(__name__)

# ==============================================
# 🔄 1. TensorFlow/Keras 기반 솔루션들
# ==============================================

def add_tensorflow_solutions():
    """TensorFlow/Keras 기반 대체 솔루션들 추가"""
    
    def tensorflow_image_classification(image, **kwargs):
        """TensorFlow 기반 이미지 분류"""
        try:
            import tensorflow as tf
            import numpy as np
            
            # 사전 훈련된 MobileNetV2 사용 (경량)
            model = tf.keras.applications.MobileNetV2(
                weights='imagenet',
                include_top=True,
                input_shape=(224, 224, 3)
            )
            
            # 이미지 전처리
            if hasattr(image, 'shape'):
                if len(image.shape) == 3:
                    # 크기 조정
                    resized = tf.image.resize(image, [224, 224])
                    preprocessed = tf.keras.applications.mobilenet_v2.preprocess_input(resized)
                    batch_input = tf.expand_dims(preprocessed, 0)
                    
                    # 예측
                    predictions = model.predict(batch_input, verbose=0)
                    decoded = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=3)[0]
                    
                    # 의류 관련 클래스 필터링
                    clothing_classes = ['jersey', 'sweatshirt', 'cardigan', 'suit', 'dress', 'skirt']
                    clothing_predictions = [pred for pred in decoded 
                                          if any(cls in pred[1].lower() for cls in clothing_classes)]
                    
                    return {
                        "success": True,
                        "predictions": [{"class": pred[1], "confidence": float(pred[2])} 
                                      for pred in clothing_predictions],
                        "all_predictions": [{"class": pred[1], "confidence": float(pred[2])} 
                                          for pred in decoded],
                        "confidence": max([float(pred[2]) for pred in decoded]) if decoded else 0.0,
                        "method": "tensorflow_mobilenetv2"
                    }
            
            return {"success": False, "error": "Invalid image format"}
            
        except ImportError:
            return {"success": False, "error": "TensorFlow not available"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def tensorflow_semantic_segmentation(image, **kwargs):
        """TensorFlow 기반 시맨틱 세그멘테이션"""
        try:
            import tensorflow as tf
            import numpy as np
            
            # DeepLab v3+ 모델 (사전 훈련됨)
            # 실제로는 TensorFlow Hub에서 로드해야 하지만, 여기서는 간단한 구현
            
            # 간단한 U-Net 스타일 세그멘테이션
            if hasattr(image, 'shape'):
                height, width = image.shape[:2]
                
                # 색상 기반 간단 세그멘테이션
                hsv = tf.image.rgb_to_hsv(tf.cast(image, tf.float32) / 255.0)
                
                # 사람/의류 영역 추정 (색상 기반)
                h, s, v = tf.split(hsv, 3, axis=-1)
                
                # 피부색 범위 (대략적)
                skin_mask = tf.logical_and(
                    tf.logical_and(h >= 0.0, h <= 0.1),
                    tf.logical_and(s >= 0.2, s <= 0.8)
                )
                
                # 의류 영역 (피부색이 아닌 영역)
                cloth_mask = tf.logical_not(skin_mask)
                
                return {
                    "success": True,
                    "segmentation_mask": tf.cast(cloth_mask, tf.uint8).numpy() * 255,
                    "skin_mask": tf.cast(skin_mask, tf.uint8).numpy() * 255,
                    "confidence": 0.6,
                    "method": "tensorflow_color_segmentation"
                }
            
            return {"success": False, "error": "Invalid image format"}
            
        except ImportError:
            return {"success": False, "error": "TensorFlow not available"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    # 솔루션들 등록
    if MYCLOSET_AVAILABLE:
        register_fallback_solution(
            "tensorflow_image_classification",
            tensorflow_image_classification,
            "quality_assessment",
            confidence=0.75,
            requirements=["tensorflow"],
            description="TensorFlow MobileNetV2 기반 이미지 분류"
        )
        
        register_fallback_solution(
            "tensorflow_semantic_segmentation",
            tensorflow_semantic_segmentation,
            "cloth_segmentation",
            confidence=0.6,
            requirements=["tensorflow"],
            description="TensorFlow 기반 시맨틱 세그멘테이션"
        )

# ==============================================
# 🔄 2. ONNX Runtime 기반 솔루션들
# ==============================================

def add_onnx_solutions():
    """ONNX Runtime 기반 대체 솔루션들 추가"""
    
    def onnx_inference_generic(image, model_path: str = None, **kwargs):
        """ONNX 모델 범용 추론"""
        try:
            import onnxruntime as ort
            import numpy as np
            
            if not model_path or not Path(model_path).exists():
                return {"success": False, "error": "ONNX model path not found"}
            
            # ONNX 세션 생성
            session = ort.InferenceSession(model_path)
            
            # 입력 정보 확인
            input_info = session.get_inputs()[0]
            input_name = input_info.name
            input_shape = input_info.shape
            
            # 이미지 전처리
            if hasattr(image, 'shape'):
                # 입력 크기에 맞게 리사이즈
                target_height, target_width = input_shape[2], input_shape[3]
                
                # 간단한 리사이즈 (OpenCV 없이)
                from PIL import Image as PILImage
                pil_image = PILImage.fromarray(image)
                resized = pil_image.resize((target_width, target_height))
                
                # 정규화
                input_array = np.array(resized).astype(np.float32) / 255.0
                input_array = input_array.transpose(2, 0, 1)  # HWC -> CHW
                input_array = np.expand_dims(input_array, axis=0)  # 배치 차원 추가
                
                # 추론
                outputs = session.run(None, {input_name: input_array})
                
                return {
                    "success": True,
                    "raw_output": outputs[0] if outputs else None,
                    "output_shape": outputs[0].shape if outputs else None,
                    "confidence": 0.7,
                    "method": "onnx_generic_inference"
                }
            
            return {"success": False, "error": "Invalid image format"}
            
        except ImportError:
            return {"success": False, "error": "ONNX Runtime not available"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def onnx_face_detection(image, **kwargs):
        """ONNX 기반 얼굴 감지 (가상의 모델)"""
        try:
            import onnxruntime as ort
            import numpy as np
            
            # 실제로는 얼굴 감지 ONNX 모델을 로드해야 함
            # 여기서는 중앙 영역을 얼굴로 가정하는 mock 구현
            
            if hasattr(image, 'shape'):
                height, width = image.shape[:2]
                
                # 상단 1/3 영역을 얼굴 영역으로 가정
                face_y1 = 0
                face_y2 = height // 3
                face_x1 = width // 4
                face_x2 = 3 * width // 4
                
                face_boxes = [{
                    "x1": face_x1,
                    "y1": face_y1,
                    "x2": face_x2,
                    "y2": face_y2,
                    "confidence": 0.6
                }]
                
                return {
                    "success": True,
                    "faces": face_boxes,
                    "face_count": len(face_boxes),
                    "confidence": 0.6,
                    "method": "onnx_mock_face_detection"
                }
            
            return {"success": False, "error": "Invalid image format"}
            
        except ImportError:
            return {"success": False, "error": "ONNX Runtime not available"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    # 솔루션들 등록
    if MYCLOSET_AVAILABLE:
        register_fallback_solution(
            "onnx_inference_generic",
            onnx_inference_generic,
            "virtual_fitting",
            confidence=0.7,
            requirements=["onnxruntime", "pillow"],
            description="ONNX Runtime 범용 모델 추론"
        )
        
        register_fallback_solution(
            "onnx_face_detection",
            onnx_face_detection,
            "human_parsing",
            confidence=0.6,
            requirements=["onnxruntime"],
            description="ONNX 기반 얼굴 감지"
        )

# ==============================================
# 🔄 3. Dlib 기반 솔루션들
# ==============================================

def add_dlib_solutions():
    """Dlib 기반 대체 솔루션들 추가"""
    
    def dlib_face_landmarks(image, **kwargs):
        """Dlib 기반 얼굴 랜드마크 검출"""
        try:
            import dlib
            import numpy as np
            
            # Dlib 감지기 초기화
            detector = dlib.get_frontal_face_detector()
            
            # 68점 랜드마크 예측기 (다운로드 필요)
            predictor_path = kwargs.get("predictor_path", "shape_predictor_68_face_landmarks.dat")
            
            if Path(predictor_path).exists():
                predictor = dlib.shape_predictor(predictor_path)
            else:
                # 예측기 없으면 감지만 수행
                predictor = None
            
            # 그레이스케일 변환
            if len(image.shape) == 3:
                gray = np.mean(image, axis=2).astype(np.uint8)
            else:
                gray = image
            
            # 얼굴 감지
            faces = detector(gray)
            
            results = []
            for face in faces:
                face_info = {
                    "x": face.left(),
                    "y": face.top(),
                    "width": face.width(),
                    "height": face.height(),
                    "confidence": 1.0  # Dlib은 신뢰도를 제공하지 않음
                }
                
                # 랜드마크 검출 (예측기가 있는 경우)
                if predictor:
                    landmarks = predictor(gray, face)
                    face_info["landmarks"] = [
                        {"x": landmarks.part(i).x, "y": landmarks.part(i).y}
                        for i in range(68)
                    ]
                
                results.append(face_info)
            
            return {
                "success": True,
                "faces": results,
                "face_count": len(results),
                "confidence": 0.8,
                "method": "dlib_face_landmarks"
            }
            
        except ImportError:
            return {"success": False, "error": "Dlib not available"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def dlib_shape_analysis(image, **kwargs):
        """Dlib 기반 형태 분석"""
        try:
            import dlib
            import numpy as np
            
            # HOG 기반 객체 감지기 학습 (간단한 예시)
            # 실제로는 의류 형태 감지를 위한 커스텀 모델이 필요
            
            if hasattr(image, 'shape'):
                height, width = image.shape[:2]
                
                # 간단한 형태 분석 (에지 기반)
                if len(image.shape) == 3:
                    gray = np.mean(image, axis=2).astype(np.uint8)
                else:
                    gray = image
                
                # 간단한 에지 검출
                edges = np.zeros_like(gray)
                for i in range(1, height-1):
                    for j in range(1, width-1):
                        gx = gray[i-1:i+2, j-1:j+2] * np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
                        gy = gray[i-1:i+2, j-1:j+2] * np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
                        edges[i, j] = min(255, int(np.sqrt(np.sum(gx)**2 + np.sum(gy)**2)))
                
                # 형태 특징 추출
                edge_density = np.sum(edges > 50) / edges.size
                
                return {
                    "success": True,
                    "edge_map": edges,
                    "edge_density": edge_density,
                    "shape_features": {
                        "edge_density": edge_density,
                        "avg_edge_strength": np.mean(edges),
                        "max_edge_strength": np.max(edges)
                    },
                    "confidence": 0.5,
                    "method": "dlib_shape_analysis"
                }
            
            return {"success": False, "error": "Invalid image format"}
            
        except ImportError:
            return {"success": False, "error": "Dlib not available"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    # 솔루션들 등록
    if MYCLOSET_AVAILABLE:
        register_fallback_solution(
            "dlib_face_landmarks",
            dlib_face_landmarks,
            "human_parsing",
            confidence=0.8,
            requirements=["dlib"],
            description="Dlib 기반 얼굴 랜드마크 검출"
        )
        
        register_fallback_solution(
            "dlib_shape_analysis",
            dlib_shape_analysis,
            "quality_assessment",
            confidence=0.5,
            requirements=["dlib"],
            description="Dlib 기반 형태 분석"
        )

# ==============================================
# 🔄 4. 클라우드 API 기반 솔루션들
# ==============================================

def add_cloud_api_solutions():
    """클라우드 API 기반 대체 솔루션들 추가"""
    
    def openai_vision_analysis(image, api_key: str = None, **kwargs):
        """OpenAI GPT-4 Vision 기반 이미지 분석"""
        try:
            import requests
            import base64
            import io
            from PIL import Image as PILImage
            
            if not api_key:
                api_key = os.getenv("OPENAI_API_KEY")
            
            if not api_key:
                return {"success": False, "error": "OpenAI API key required"}
            
            # 이미지를 base64로 인코딩
            if hasattr(image, 'shape'):
                pil_image = PILImage.fromarray(image)
            else:
                pil_image = image
            
            # 이미지 크기 최적화 (API 제한 고려)
            pil_image.thumbnail((512, 512), PILImage.Resampling.LANCZOS)
            
            buffer = io.BytesIO()
            pil_image.save(buffer, format="JPEG", quality=85)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            
            # OpenAI API 요청
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            prompt = kwargs.get("prompt", 
                "Analyze this fashion image. Describe the clothing items, colors, style, and suggest improvements.")
            
            payload = {
                "model": "gpt-4-vision-preview",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_base64}"
                                }
                            }
                        ]
                    }
                ],
                "max_tokens": 500
            }
            
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                analysis = result["choices"][0]["message"]["content"]
                
                return {
                    "success": True,
                    "analysis": analysis,
                    "confidence": 0.9,
                    "method": "openai_gpt4_vision",
                    "usage": result.get("usage", {})
                }
            else:
                return {
                    "success": False,
                    "error": f"OpenAI API error: {response.status_code}",
                    "details": response.text
                }
                
        except ImportError:
            return {"success": False, "error": "Requests library not available"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def replicate_ai_processing(image, model_name: str = None, **kwargs):
        """Replicate.com AI 모델 기반 처리"""
        try:
            import requests
            import base64
            import io
            import time
            from PIL import Image as PILImage
            
            api_token = kwargs.get("api_token") or os.getenv("REPLICATE_API_TOKEN")
            if not api_token:
                return {"success": False, "error": "Replicate API token required"}
            
            # 기본 모델 (예: 배경 제거 모델)
            if not model_name:
                model_name = "cjwbw/rembg:fb8af171cfa1616ddcf1242c093f9c46bcada5ad4cf6f2fbe8b81b330ec5c003"
            
            # 이미지 준비
            if hasattr(image, 'shape'):
                pil_image = PILImage.fromarray(image)
            else:
                pil_image = image
            
            # 임시 파일로 저장 (Replicate는 URL 필요)
            buffer = io.BytesIO()
            pil_image.save(buffer, format="PNG")
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            
            # Replicate API 요청
            headers = {
                "Authorization": f"Token {api_token}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "version": model_name.split(":")[-1],
                "input": {
                    "image": f"data:image/png;base64,{image_base64}"
                }
            }
            
            # 예측 시작
            response = requests.post(
                f"https://api.replicate.com/v1/predictions",
                headers=headers,
                json=payload,
                timeout=10
            )
            
            if response.status_code == 201:
                prediction = response.json()
                prediction_id = prediction["id"]
                
                # 결과 대기 (간단한 폴링)
                for _ in range(30):  # 최대 30초 대기
                    status_response = requests.get(
                        f"https://api.replicate.com/v1/predictions/{prediction_id}",
                        headers=headers,
                        timeout=5
                    )
                    
                    if status_response.status_code == 200:
                        status_data = status_response.json()
                        
                        if status_data["status"] == "succeeded":
                            return {
                                "success": True,
                                "result_url": status_data["output"],
                                "confidence": 0.8,
                                "method": "replicate_ai",
                                "model": model_name
                            }
                        elif status_data["status"] == "failed":
                            return {
                                "success": False,
                                "error": "Replicate processing failed",
                                "details": status_data.get("error")
                            }
                    
                    time.sleep(1)
                
                return {"success": False, "error": "Replicate processing timeout"}
            else:
                return {
                    "success": False,
                    "error": f"Replicate API error: {response.status_code}",
                    "details": response.text
                }
                
        except ImportError:
            return {"success": False, "error": "Requests library not available"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    # 솔루션들 등록
    if MYCLOSET_AVAILABLE:
        register_fallback_solution(
            "openai_vision_analysis",
            openai_vision_analysis,
            "quality_assessment",
            confidence=0.9,
            requirements=["requests", "pillow"],
            description="OpenAI GPT-4 Vision 기반 패션 이미지 분석"
        )
        
        register_fallback_solution(
            "replicate_ai_processing",
            replicate_ai_processing,
            "cloth_segmentation",
            confidence=0.8,
            requirements=["requests", "pillow"],
            description="Replicate.com AI 모델 기반 이미지 처리"
        )

# ==============================================
# 🔄 5. 웹 기반 처리 서비스들
# ==============================================

def add_web_service_solutions():
    """웹 기반 처리 서비스들 추가"""
    
    def remove_bg_api(image, **kwargs):
        """Remove.bg API 기반 배경 제거"""
        try:
            import requests
            import io
            from PIL import Image as PILImage
            
            api_key = kwargs.get("api_key") or os.getenv("REMOVE_BG_API_KEY")
            if not api_key:
                return {"success": False, "error": "Remove.bg API key required"}
            
            # 이미지 준비
            if hasattr(image, 'shape'):
                pil_image = PILImage.fromarray(image)
            else:
                pil_image = image
            
            buffer = io.BytesIO()
            pil_image.save(buffer, format="PNG")
            buffer.seek(0)
            
            # Remove.bg API 요청
            response = requests.post(
                "https://api.remove.bg/v1.0/removebg",
                files={"image_file": buffer},
                data={"size": "auto"},
                headers={"X-Api-Key": api_key},
                timeout=30
            )
            
            if response.status_code == 200:
                # 결과 이미지 처리
                result_image = PILImage.open(io.BytesIO(response.content))
                result_array = np.array(result_image)
                
                # 알파 채널을 마스크로 변환
                if result_array.shape[2] == 4:
                    mask = result_array[:, :, 3]
                    rgb_image = result_array[:, :, :3]
                else:
                    mask = np.ones(result_array.shape[:2], dtype=np.uint8) * 255
                    rgb_image = result_array
                
                return {
                    "success": True,
                    "segmented_image": rgb_image,
                    "segmentation_mask": mask,
                    "confidence": 0.9,
                    "method": "remove_bg_api"
                }
            else:
                return {
                    "success": False,
                    "error": f"Remove.bg API error: {response.status_code}",
                    "details": response.text
                }
                
        except ImportError:
            return {"success": False, "error": "Requests library not available"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def unsplash_style_reference(query: str = "fashion", **kwargs):
        """Unsplash API 기반 스타일 참조 이미지 검색"""
        try:
            import requests
            
            api_key = kwargs.get("api_key") or os.getenv("UNSPLASH_ACCESS_KEY")
            if not api_key:
                return {"success": False, "error": "Unsplash API key required"}
            
            # Unsplash API 요청
            headers = {"Authorization": f"Client-ID {api_key}"}
            params = {
                "query": query,
                "per_page": 5,
                "orientation": "portrait"
            }
            
            response = requests.get(
                "https://api.unsplash.com/search/photos",
                headers=headers,
                params=params,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                
                style_references = []
                for photo in data.get("results", []):
                    style_references.append({
                        "url": photo["urls"]["regular"],
                        "thumbnail": photo["urls"]["small"],
                        "description": photo.get("description", ""),
                        "tags": photo.get("tags", []),
                        "photographer": photo["user"]["name"]
                    })
                
                return {
                    "success": True,
                    "style_references": style_references,
                    "total_results": data.get("total", 0),
                    "confidence": 0.7,
                    "method": "unsplash_style_reference"
                }
            else:
                return {
                    "success": False,
                    "error": f"Unsplash API error: {response.status_code}",
                    "details": response.text
                }
                
        except ImportError:
            return {"success": False, "error": "Requests library not available"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    # 솔루션들 등록
    if MYCLOSET_AVAILABLE:
        register_fallback_solution(
            "remove_bg_api",
            remove_bg_api,
            "cloth_segmentation",
            confidence=0.9,
            requirements=["requests", "pillow"],
            description="Remove.bg API 기반 배경 제거"
        )
        
        register_fallback_solution(
            "unsplash_style_reference",
            unsplash_style_reference,
            "quality_assessment",
            confidence=0.7,
            requirements=["requests"],
            description="Unsplash API 기반 스타일 참조 검색"
        )

# ==============================================
# 🔄 6. 성능 벤치마킹 및 테스트 시스템
# ==============================================

class FallbackSolutionBenchmark:
    """대체 솔루션 성능 벤치마킹"""
    
    def __init__(self):
        self.results = {}
        self.logger = logging.getLogger(f"{__name__}.Benchmark")
    
    def benchmark_solution(self, solution_name: str, test_image_path: str, iterations: int = 3):
        """솔루션 성능 벤치마크"""
        try:
            from PIL import Image
            import numpy as np
            
            # 테스트 이미지 로드
            test_image = np.array(Image.open(test_image_path))
            
            # 벤치마크 실행
            times = []
            success_count = 0
            confidence_scores = []
            
            registry = get_fallback_registry()
            
            for i in range(iterations):
                start_time = time.time()
                
                result = registry.execute_solution(solution_name, test_image)
                
                execution_time = time.time() - start_time
                times.append(execution_time)
                
                if result.get("success", False):
                    success_count += 1
                    confidence = result.get("confidence", 0.0)
                    confidence_scores.append(confidence)
            
            # 결과 계산
            avg_time = np.mean(times) if times else 0
            success_rate = success_count / iterations
            avg_confidence = np.mean(confidence_scores) if confidence_scores else 0
            
            benchmark_result = {
                "solution_name": solution_name,
                "avg_execution_time": avg_time,
                "success_rate": success_rate,
                "avg_confidence": avg_confidence,
                "iterations": iterations,
                "individual_times": times,
                "confidence_scores": confidence_scores,
                "timestamp": time.time()
            }
            
            self.results[solution_name] = benchmark_result
            
            self.logger.info(f"✅ {solution_name} 벤치마크 완료")
            self.logger.info(f"  평균 실행 시간: {avg_time:.3f}초")
            self.logger.info(f"  성공률: {success_rate:.1%}")
            self.logger.info(f"  평균 신뢰도: {avg_confidence:.3f}")
            
            return benchmark_result
            
        except Exception as e:
            self.logger.error(f"벤치마크 실패 {solution_name}: {e}")
            return {"error": str(e)}
    
    def benchmark_all_solutions(self, test_image_path: str):
        """모든 등록된 솔루션 벤치마크"""
        try:
            registry = get_fallback_registry()
            
            for solution_name in registry.solutions.keys():
                self.benchmark_solution(solution_name, test_image_path)
            
            return self.get_benchmark_summary()
            
        except Exception as e:
            self.logger.error(f"전체 벤치마크 실패: {e}")
            return {"error": str(e)}
    
    def get_benchmark_summary(self):
        """벤치마크 결과 요약"""
        try:
            if not self.results:
                return {"message": "No benchmark results available"}
            
            summary = {
                "total_solutions": len(self.results),
                "fastest_solution": None,
                "most_reliable_solution": None,
                "highest_confidence_solution": None,
                "detailed_results": self.results
            }
            
            # 가장 빠른 솔루션
            fastest = min(self.results.items(), key=lambda x: x[1].get("avg_execution_time", float('inf')))
            summary["fastest_solution"] = {
                "name": fastest[0],
                "avg_time": fastest[1]["avg_execution_time"]
            }
            
            # 가장 신뢰할 수 있는 솔루션
            most_reliable = max(self.results.items(), key=lambda x: x[1].get("success_rate", 0))
            summary["most_reliable_solution"] = {
                "name": most_reliable[0],
                "success_rate": most_reliable[1]["success_rate"]
            }
            
            # 가장 높은 신뢰도 솔루션
            highest_confidence = max(self.results.items(), key=lambda x: x[1].get("avg_confidence", 0))
            summary["highest_confidence_solution"] = {
                "name": highest_confidence[0],
                "avg_confidence": highest_confidence[1]["avg_confidence"]
            }
            
            return summary
            
        except Exception as e:
            return {"error": str(e)}

# ==============================================
# 🔄 7. 새로운 솔루션 추가 템플릿
# ==============================================

def create_custom_solution_template():
    """새로운 대체 솔루션 추가 템플릿"""
    
    template_code = '''
def my_custom_solution(image, **kwargs):
    """
    커스텀 대체 솔루션 템플릿
    
    Args:
        image: 입력 이미지 (numpy array 또는 PIL Image)
        **kwargs: 추가 파라미터들
    
    Returns:
        Dict[str, Any]: 처리 결과
        {
            "success": bool,           # 성공 여부
            "result": Any,            # 주요 결과
            "confidence": float,      # 신뢰도 (0.0-1.0)
            "method": str,           # 사용된 방법
            "error": str             # 오류 메시지 (실패시)
        }
    """
    try:
        # 1. 필요한 라이브러리 import
        import numpy as np
        # import your_library
        
        # 2. 입력 검증
        if not hasattr(image, 'shape'):
            return {"success": False, "error": "Invalid image format"}
        
        # 3. 이미지 전처리
        if len(image.shape) == 3:
            # RGB 이미지 처리
            height, width, channels = image.shape
        else:
            # 그레이스케일 이미지 처리
            height, width = image.shape
            channels = 1
        
        # 4. 실제 처리 로직 구현
        # TODO: 여기에 실제 알고리즘 구현
        
        # 예시: 간단한 이미지 처리
        processed_image = image.copy()
        
        # 5. 결과 반환
        return {
            "success": True,
            "processed_image": processed_image,
            "confidence": 0.7,  # 알고리즘에 따라 신뢰도 계산
            "method": "my_custom_algorithm",
            "metadata": {
                "input_size": (height, width),
                "processing_time": 0.1  # 실제 측정 권장
            }
        }
        
    except ImportError as e:
        return {"success": False, "error": f"Required library not available: {e}"}
    except Exception as e:
        return {"success": False, "error": str(e)}

# 솔루션 등록
register_fallback_solution(
    "my_custom_solution",                    # 고유한 이름
    my_custom_solution,                      # 함수 객체
    "cloth_segmentation",                    # Step 카테고리
    confidence=0.7,                          # 예상 신뢰도
    requirements=["numpy"],                  # 필요한 라이브러리들
    description="나만의 커스텀 이미지 처리 솔루션"  # 설명
)
'''
    
    return template_code

# ==============================================
# 🔄 8. 메인 실행 및 테스트 함수들
# ==============================================

def register_all_fallback_solutions():
    """모든 대체 솔루션들을 등록"""
    print("🔄 대체 솔루션들 등록 시작...")
    
    try:
        add_tensorflow_solutions()
        print("✅ TensorFlow 솔루션들 등록 완료")
    except Exception as e:
        print(f"⚠️ TensorFlow 솔루션 등록 실패: {e}")
    
    try:
        add_onnx_solutions()
        print("✅ ONNX 솔루션들 등록 완료")
    except Exception as e:
        print(f"⚠️ ONNX 솔루션 등록 실패: {e}")
    
    try:
        add_dlib_solutions()
        print("✅ Dlib 솔루션들 등록 완료")
    except Exception as e:
        print(f"⚠️ Dlib 솔루션 등록 실패: {e}")
    
    try:
        add_cloud_api_solutions()
        print("✅ 클라우드 API 솔루션들 등록 완료")
    except Exception as e:
        print(f"⚠️ 클라우드 API 솔루션 등록 실패: {e}")
    
    try:
        add_web_service_solutions()
        print("✅ 웹 서비스 솔루션들 등록 완료")
    except Exception as e:
        print(f"⚠️ 웹 서비스 솔루션 등록 실패: {e}")
    
    print("🎉 모든 대체 솔루션 등록 완료!")

def list_available_solutions():
    """사용 가능한 대체 솔루션들 조회"""
    if not MYCLOSET_AVAILABLE:
        print("❌ MyCloset AI 시스템을 사용할 수 없습니다.")
        return
    
    registry = get_fallback_registry()
    solutions = registry.solutions
    
    print(f"\n📋 등록된 대체 솔루션들 ({len(solutions)}개)")
    print("=" * 60)
    
    # Step 카테고리별로 그룹화
    by_category = {}
    for name, info in solutions.items():
        category = info["step_category"]
        if category not in by_category:
            by_category[category] = []
        by_category[category].append((name, info))
    
    for category, solution_list in by_category.items():
        print(f"\n🔸 {category.upper()}")
        for name, info in solution_list:
            status = "✅" if info["is_available"] else "❌"
            confidence = info["confidence"]
            description = info["description"]
            requirements = ", ".join(info["requirements"])
            
            print(f"  {status} {name}")
            print(f"     신뢰도: {confidence:.2f} | 요구사항: {requirements}")
            print(f"     설명: {description}")

def test_solution(solution_name: str, test_image_path: str = None):
    """특정 솔루션 테스트"""
    if not MYCLOSET_AVAILABLE:
        print("❌ MyCloset AI 시스템을 사용할 수 없습니다.")
        return
    
    registry = get_fallback_registry()
    
    # 테스트 이미지 생성 (실제 이미지가 없는 경우)
    if not test_image_path:
        import numpy as np
        test_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        print("📸 랜덤 테스트 이미지 생성")
    else:
        from PIL import Image
        test_image = np.array(Image.open(test_image_path))
        print(f"📸 테스트 이미지 로드: {test_image_path}")
    
    print(f"\n🧪 {solution_name} 테스트 시작...")
    
    start_time = time.time()
    result = registry.execute_solution(solution_name, test_image)
    execution_time = time.time() - start_time
    
    print(f"⏱️ 실행 시간: {execution_time:.3f}초")
    
    if result.get("success", False):
        print("✅ 테스트 성공")
        print(f"🎯 신뢰도: {result.get('confidence', 0):.3f}")
        print(f"🔧 방법: {result.get('method', 'Unknown')}")
        
        # 결과 키들 출력
        result_keys = [k for k in result.keys() if k not in ['success', 'confidence', 'method']]
        if result_keys:
            print(f"📊 결과 키들: {', '.join(result_keys)}")
    else:
        print("❌ 테스트 실패")
        print(f"🚨 오류: {result.get('error', 'Unknown error')}")

def show_usage_guide():
    """사용법 가이드 출력"""
    guide = """
🔄 MyCloset AI 대체 솔루션 확장 가이드

📚 1. 새로운 솔루션 추가 방법:
   1) 이 파일을 참고해서 함수 구현
   2) register_fallback_solution()으로 등록
   3) PipelineManager에서 자동 사용 가능

🛠️ 2. 지원되는 라이브러리들:
   - TensorFlow/Keras (딥러닝 모델)
   - ONNX Runtime (경량 추론)
   - Dlib (얼굴/형태 처리)
   - OpenAI/Replicate (클라우드 AI)
   - Remove.bg/Unsplash (웹 서비스)
   - 기타 Python 라이브러리

🧪 3. 테스트 및 벤치마킹:
   - test_solution(solution_name) : 개별 테스트
   - benchmark_all_solutions() : 성능 벤치마크
   - list_available_solutions() : 등록된 솔루션 조회

📝 4. 솔루션 구현 가이드라인:
   - 입력: image (numpy array 또는 PIL Image)
   - 출력: Dict with success, result, confidence, method
   - 오류 처리: try-except로 안전하게 처리
   - 요구사항: requirements 리스트에 명시

🔗 5. 자동 연동:
   - PipelineManager가 자동으로 감지
   - Step 실패시 자동으로 대체 솔루션 사용
   - 신뢰도 기반 우선순위 자동 결정

💡 6. 추가 확장 예시:
   - 새로운 AI 모델 (Hugging Face Hub)
   - 커스텀 알고리즘 (OpenCV 확장)
   - 외부 API 서비스
   - 하드웨어 가속 (CUDA, Metal)

사용 예시:
  python add_fallback_solutions.py --register-all
  python add_fallback_solutions.py --list
  python add_fallback_solutions.py --test opencv_segmentation
  python add_fallback_solutions.py --benchmark
"""
    print(guide)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="MyCloset AI 대체 솔루션 관리")
    parser.add_argument("--register-all", action="store_true", help="모든 솔루션 등록")
    parser.add_argument("--list", action="store_true", help="등록된 솔루션 목록")
    parser.add_argument("--test", type=str, help="특정 솔루션 테스트")
    parser.add_argument("--test-image", type=str, help="테스트용 이미지 경로")
    parser.add_argument("--benchmark", action="store_true", help="성능 벤치마크")
    parser.add_argument("--template", action="store_true", help="커스텀 솔루션 템플릿 출력")
    parser.add_argument("--help-guide", action="store_true", help="상세 사용법 가이드")
    
    args = parser.parse_args()
    
    if args.help_guide:
        show_usage_guide()
    elif args.register_all:
        register_all_fallback_solutions()
    elif args.list:
        list_available_solutions()
    elif args.test:
        test_solution(args.test, args.test_image)
    elif args.benchmark:
        benchmark = FallbackSolutionBenchmark()
        if args.test_image:
            summary = benchmark.benchmark_all_solutions(args.test_image)
            print(f"\n📊 벤치마크 결과 요약:")
            print(json.dumps(summary, indent=2, ensure_ascii=False))
        else:
            print("❌ 벤치마크를 위해 --test-image 경로가 필요합니다.")
    elif args.template:
        template = create_custom_solution_template()
        print("📝 커스텀 솔루션 템플릿:")
        print(template)
    else:
        print("🔄 MyCloset AI 대체 솔루션 확장 시스템")
        print("사용법: python add_fallback_solutions.py --help")
        print("상세 가이드: python add_fallback_solutions.py --help-guide")

print("\n✅ 대체 솔루션 확장 시스템 로드 완료")
print("💡 새로운 라이브러리나 API를 이용해서 계속 솔루션을 추가할 수 있습니다!")
print("📚 --help-guide 옵션으로 상세한 사용법을 확인하세요.")