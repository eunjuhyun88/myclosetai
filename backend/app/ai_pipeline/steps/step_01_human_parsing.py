# backend/app/ai_pipeline/steps/step_01_human_parsing.py
"""
1단계: 인체 파싱 (Human Parsing) - M3 Max 최적화 버전
- Graphonomy 모델 기반 20개 부위 분할
- M3 Max Neural Engine 활용 최적화
- 메모리 효율적 처리 및 실시간 캐싱
- 기존 AI 파이프라인 구조와 완벽 호환
"""

import os
import logging
import time
import asyncio
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import cv2
from concurrent.futures import ThreadPoolExecutor

# 기존 유틸리티 import
from ..utils.model_loader import ModelLoader
from ..utils.memory_manager import MemoryManager
from ..utils.data_converter import DataConverter

# Apple Metal Performance Shaders 지원
try:
    import torch.backends.mps
    MPS_AVAILABLE = torch.backends.mps.is_available()
except ImportError:
    MPS_AVAILABLE = False

# CoreML 지원 (선택사항)
try:
    import coremltools as ct
    COREML_AVAILABLE = True
except ImportError:
    COREML_AVAILABLE = False

logger = logging.getLogger(__name__)

class HumanParsingStep:
    """1단계: 인체 파싱 - M3 Max 최적화"""
    
    # LIP (Look Into Person) 데이터셋 기반 20개 부위 라벨
    BODY_PARTS = {
        0: "Background",
        1: "Hat", 2: "Hair", 3: "Glove", 4: "Sunglasses",
        5: "Upper-clothes", 6: "Dress", 7: "Coat", 8: "Socks",
        9: "Pants", 10: "Jumpsuits", 11: "Scarf", 12: "Skirt",
        13: "Face", 14: "Left-arm", 15: "Right-arm",
        16: "Left-leg", 17: "Right-leg", 18: "Left-shoe", 19: "Right-shoe"
    }
    
    # 의류 카테고리 매핑 (다음 단계들을 위한)
    CLOTHING_CATEGORIES = {
        "upper": [5, 7],      # Upper-clothes, Coat
        "lower": [9, 12],     # Pants, Skirt
        "dress": [6],         # Dress
        "full_body": [10],    # Jumpsuits
        "accessories": [1, 3, 4, 8, 11, 18, 19]  # Hat, Glove, etc.
    }
    
    def __init__(self, model_loader: ModelLoader, device: str, config: Dict[str, Any] = None):
        """
        Args:
            model_loader: 모델 로더 인스턴스 (기존 파이프라인 호환)
            device: 사용할 디바이스 ('cpu', 'cuda', 'mps')
            config: 설정 딕셔너리
        """
        self.model_loader = model_loader
        self.memory_manager = MemoryManager(device)
        self.data_converter = DataConverter()
        
        # M3 Max 최적화 설정
        self.device = self._get_optimal_device(device)
        self.use_coreml = config.get('use_coreml', True) and COREML_AVAILABLE
        self.enable_quantization = config.get('enable_quantization', True)
        
        # 모델 설정
        self.input_size = config.get('input_size', (512, 512))
        self.num_classes = config.get('num_classes', 20)
        self.model_name = config.get('model_name', 'graphonomy')
        self.model_path = config.get('model_path', 'ai_models/checkpoints/human_parsing')
        
        # 전처리 파라미터 (ImageNet 표준)
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])
        
        # 모델 인스턴스
        self.pytorch_model = None
        self.coreml_model = None
        self.is_initialized = False
        
        # 성능 최적화
        self.batch_size = config.get('batch_size', 1)
        self.cache_size = config.get('cache_size', 50)
        self.result_cache = {}
        
        # 병렬 처리
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # 성능 통계
        self.stats = {
            "total_inferences": 0,
            "average_time": 0.0,
            "cache_hits": 0,
            "model_switches": 0
        }
        
        logger.info(f"🎯 1단계 인체 파싱 초기화 - 디바이스: {self.device}")
    
    def _get_optimal_device(self, preferred_device: str) -> str:
        """M3 Max에 최적화된 디바이스 선택"""
        if preferred_device == "mps" and MPS_AVAILABLE:
            logger.info("🚀 Apple Metal Performance Shaders 활성화")
            return "mps"
        elif preferred_device == "cuda" and torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"
    
    async def initialize(self) -> bool:
        """모델 초기화 (비동기)"""
        try:
            logger.info("🔄 1단계: 인체 파싱 모델 로드 중...")
            
            # CoreML 모델 우선 로드
            if self.use_coreml:
                coreml_loaded = await self._load_coreml_model()
                if coreml_loaded:
                    logger.info("✅ CoreML 모델 로드 성공 (Neural Engine 가속)")
            
            # PyTorch 모델 로드 (백업 또는 병행)
            pytorch_loaded = await self._load_pytorch_model()
            
            if not (self.coreml_model or self.pytorch_model):
                raise RuntimeError("사용 가능한 모델이 없습니다.")
            
            # 모델 워밍업
            await self._warmup_models()
            
            self.is_initialized = True
            logger.info("✅ 1단계 인체 파싱 모델 로드 완료")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ 1단계 모델 로드 실패: {e}")
            self.is_initialized = False
            return False
    
    async def _load_coreml_model(self) -> bool:
        """CoreML 모델 로드 (Neural Engine 활용)"""
        coreml_path = os.path.join(self.model_path, "human_parser_optimized.mlpackage")
        
        if not COREML_AVAILABLE:
            logger.warning("⚠️ CoreML 지원 안됨")
            return False
        
        if os.path.exists(coreml_path):
            try:
                def _load_coreml():
                    return ct.models.MLModel(coreml_path)
                
                loop = asyncio.get_event_loop()
                self.coreml_model = await loop.run_in_executor(self.executor, _load_coreml)
                return True
                
            except Exception as e:
                logger.warning(f"⚠️ CoreML 모델 로드 실패: {e}")
        else:
            logger.info("📦 CoreML 모델 없음 - PyTorch 우선 사용")
        
        return False
    
    async def _load_pytorch_model(self) -> bool:
        """PyTorch 모델 로드"""
        try:
            # 모델 로더를 통한 로드 (기존 파이프라인 호환)
            model_config = {
                "model_type": "human_parsing",
                "model_name": self.model_name,
                "num_classes": self.num_classes,
                "input_size": self.input_size
            }
            
            # 실제 Graphonomy 모델이 있다면 로드, 없다면 데모 모델 생성
            try:
                self.pytorch_model = await self.model_loader.load_model(
                    "human_parsing", 
                    self.model_path
                )
            except FileNotFoundError:
                logger.warning("⚠️ Graphonomy 모델 없음 - 데모 모델 생성")
                self.pytorch_model = self._create_demo_model()
            
            # 디바이스로 이동
            self.pytorch_model = self.pytorch_model.to(self.device)
            
            # M3 Max 최적화
            if self.device == "mps":
                # MPS 최적화
                self.pytorch_model = self._optimize_for_mps(self.pytorch_model)
            elif self.enable_quantization and self.device == "cpu":
                # CPU 양자화
                self.pytorch_model = self._quantize_model(self.pytorch_model)
            
            self.pytorch_model.eval()
            return True
            
        except Exception as e:
            logger.error(f"❌ PyTorch 모델 로드 실패: {e}")
            return False
    
    def _create_demo_model(self):
        """데모용 경량 인체 파싱 모델"""
        
        class EfficientHumanParser(torch.nn.Module):
            """MobileNet 기반 경량 인체 파싱 모델"""
            
            def __init__(self, num_classes=20):
                super().__init__()
                
                # Efficient backbone (MobileNetV3 스타일)
                self.backbone = torch.nn.Sequential(
                    # Initial conv
                    torch.nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
                    torch.nn.BatchNorm2d(32),
                    torch.nn.ReLU(inplace=True),
                    
                    # Inverted residual blocks
                    self._make_inverted_block(32, 64, 3, 2, 1),
                    self._make_inverted_block(64, 64, 3, 1, 2),
                    self._make_inverted_block(64, 128, 5, 2, 2),
                    self._make_inverted_block(128, 128, 5, 1, 3),
                    self._make_inverted_block(128, 256, 3, 2, 3),
                    self._make_inverted_block(256, 256, 3, 1, 2),
                )
                
                # Feature Pyramid Network (FPN) 스타일 디코더
                self.decoder = torch.nn.ModuleList([
                    torch.nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
                    torch.nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
                    torch.nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
                    torch.nn.ConvTranspose2d(32, num_classes, 4, stride=2, padding=1)
                ])
                
                # 활성화 함수
                self.relu = torch.nn.ReLU(inplace=True)
                
            def _make_inverted_block(self, in_ch, out_ch, kernel_size, stride, repeat):
                layers = []
                for i in range(repeat):
                    s = stride if i == 0 else 1
                    layers.extend([
                        torch.nn.Conv2d(in_ch, out_ch, kernel_size, s, kernel_size//2, bias=False),
                        torch.nn.BatchNorm2d(out_ch),
                        torch.nn.ReLU(inplace=True)
                    ])
                    in_ch = out_ch
                return torch.nn.Sequential(*layers)
            
            def forward(self, x):
                # Encoder
                features = self.backbone(x)
                
                # Decoder
                out = features
                for decoder_layer in self.decoder:
                    out = self.relu(decoder_layer(out))
                
                # 최종 크기 맞춤
                out = F.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=False)
                
                return out
        
        return EfficientHumanParser(self.num_classes)
    
    def _optimize_for_mps(self, model):
        """M3 Max MPS 최적화"""
        # MPS에 최적화된 설정 적용
        if hasattr(model, 'eval'):
            model.eval()
        
        # Mixed precision을 위한 준비 (M3 Max는 FP16 지원)
        for param in model.parameters():
            param.requires_grad_(False)
        
        return model
    
    def _quantize_model(self, model):
        """모델 양자화 (메모리 효율성)"""
        try:
            quantized_model = torch.quantization.quantize_dynamic(
                model, 
                {torch.nn.Linear, torch.nn.Conv2d}, 
                dtype=torch.qint8
            )
            logger.info("✅ 모델 양자화 완료")
            return quantized_model
        except Exception as e:
            logger.warning(f"⚠️ 양자화 실패: {e}")
            return model
    
    async def _warmup_models(self):
        """모델 워밍업 (첫 추론 최적화)"""
        logger.info("🔥 1단계 모델 워밍업 중...")
        
        try:
            # 더미 입력 생성
            dummy_input = torch.randn(1, 3, *self.input_size)
            
            # PyTorch 모델 워밍업
            if self.pytorch_model:
                dummy_input = dummy_input.to(self.device)
                with torch.no_grad():
                    _ = self.pytorch_model(dummy_input)
            
            # CoreML 모델 워밍업
            if self.coreml_model:
                dummy_np = dummy_input.cpu().numpy()
                _ = self.coreml_model.predict({"input": dummy_np})
            
            logger.info("✅ 1단계 워밍업 완료")
            
        except Exception as e:
            logger.warning(f"⚠️ 워밍업 중 오류: {e}")
    
    async def process(self, person_image_tensor: torch.Tensor) -> Dict[str, Any]:
        """
        1단계: 인체 파싱 처리 (비동기)
        
        Args:
            person_image_tensor: 입력 이미지 텐서 [1, 3, H, W]
            
        Returns:
            Dict[str, Any]: {
                "parsing_map": np.ndarray,           # 파싱 맵 (H, W)
                "body_masks": Dict[str, np.ndarray], # 부위별 마스크
                "clothing_regions": Dict[str, Any],  # 의류 영역 정보
                "confidence": float,                 # 전체 신뢰도
                "processing_time": float,            # 처리 시간
                "step_info": Dict[str, Any]         # 단계 정보
            }
        """
        if not self.is_initialized:
            raise RuntimeError("1단계 모델이 초기화되지 않았습니다.")
        
        start_time = time.time()
        
        try:
            # 캐시 확인
            cache_key = self._get_cache_key(person_image_tensor)
            if cache_key in self.result_cache:
                self.stats["cache_hits"] += 1
                logger.info("💾 1단계: 캐시에서 결과 반환")
                cached_result = self.result_cache[cache_key].copy()
                cached_result["from_cache"] = True
                return cached_result
            
            # 입력 전처리
            preprocessed_tensor = await self._preprocess_input(person_image_tensor)
            
            # 모델 추론
            parsing_output = await self._run_inference(preprocessed_tensor)
            
            # 후처리 및 결과 생성
            result = await self._postprocess_result(
                parsing_output, 
                person_image_tensor.shape[2:],
                start_time
            )
            
            # 캐시 저장
            self._update_cache(cache_key, result)
            
            # 통계 업데이트
            self._update_stats(time.time() - start_time)
            
            logger.info(f"✅ 1단계 완료 - {result['processing_time']:.3f}초")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ 1단계 처리 실패: {e}")
            raise
    
    async def _preprocess_input(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """입력 전처리"""
        def _preprocess():
            # 크기 조정
            if image_tensor.shape[2:] != self.input_size:
                resized = F.interpolate(
                    image_tensor, 
                    size=self.input_size, 
                    mode='bilinear', 
                    align_corners=False
                )
            else:
                resized = image_tensor
            
            # 정규화 (0-1 범위 가정)
            if resized.max() > 1.0:
                resized = resized / 255.0
            
            # ImageNet 정규화
            mean = torch.tensor(self.mean).view(1, 3, 1, 1)
            std = torch.tensor(self.std).view(1, 3, 1, 1)
            
            if self.device != "cpu":
                mean = mean.to(self.device)
                std = std.to(self.device)
            
            normalized = (resized - mean) / std
            
            return normalized.to(self.device)
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, _preprocess)
    
    async def _run_inference(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """모델 추론 실행"""
        # CoreML 모델 우선 사용 (Neural Engine 가속)
        if self.coreml_model:
            try:
                def _coreml_inference():
                    input_np = input_tensor.cpu().numpy()
                    result = self.coreml_model.predict({"input": input_np})
                    # CoreML 출력 키는 모델에 따라 다를 수 있음
                    output_key = list(result.keys())[0]
                    return torch.from_numpy(result[output_key]).to(self.device)
                
                loop = asyncio.get_event_loop()
                output = await loop.run_in_executor(self.executor, _coreml_inference)
                logger.debug("🚀 CoreML 추론 완료")
                return output
                
            except Exception as e:
                logger.warning(f"⚠️ CoreML 추론 실패, PyTorch로 전환: {e}")
                self.stats["model_switches"] += 1
        
        # PyTorch 모델 사용
        if self.pytorch_model:
            with torch.no_grad():
                output = self.pytorch_model(input_tensor)
                logger.debug("🔥 PyTorch 추론 완료")
                return output
        
        raise RuntimeError("사용 가능한 모델이 없습니다.")
    
    async def _postprocess_result(self, output: torch.Tensor, original_size: Tuple[int, int], 
                                 start_time: float) -> Dict[str, Any]:
        """결과 후처리"""
        def _postprocess():
            # 확률을 클래스 인덱스로 변환
            parsing_map = torch.argmax(output, dim=1).squeeze().cpu().numpy()
            
            # 원본 크기로 복원
            if parsing_map.shape != original_size:
                parsing_map = cv2.resize(
                    parsing_map.astype(np.uint8),
                    (original_size[1], original_size[0]),
                    interpolation=cv2.INTER_NEAREST
                )
            
            # 노이즈 제거 (모폴로지 연산)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            parsing_map = cv2.morphologyEx(parsing_map, cv2.MORPH_CLOSE, kernel)
            parsing_map = cv2.morphologyEx(parsing_map, cv2.MORPH_OPEN, kernel)
            
            return parsing_map
        
        loop = asyncio.get_event_loop()
        parsing_map = await loop.run_in_executor(self.executor, _postprocess)
        
        # 부위별 마스크 생성
        body_masks = self._create_body_masks(parsing_map)
        
        # 의류 영역 분석
        clothing_regions = self._analyze_clothing_regions(parsing_map)
        
        # 신뢰도 계산
        confidence = self._calculate_confidence(output)
        
        processing_time = time.time() - start_time
        
        return {
            "parsing_map": parsing_map.astype(np.uint8),
            "body_masks": body_masks,
            "clothing_regions": clothing_regions,
            "confidence": float(confidence),
            "body_parts_detected": self._get_detected_parts(parsing_map),
            "processing_time": processing_time,
            "step_info": {
                "step_name": "human_parsing",
                "step_number": 1,
                "model_used": "CoreML" if self.coreml_model else "PyTorch",
                "device": self.device,
                "input_size": self.input_size,
                "num_classes": self.num_classes
            },
            "from_cache": False
        }
    
    def _create_body_masks(self, parsing_map: np.ndarray) -> Dict[str, np.ndarray]:
        """신체 부위별 마스크 생성"""
        body_masks = {}
        
        for part_id, part_name in self.BODY_PARTS.items():
            if part_id == 0:  # 배경 제외
                continue
            
            mask = (parsing_map == part_id).astype(np.uint8)
            if mask.sum() > 0:
                # 키 이름 정규화
                key_name = part_name.lower().replace('-', '_').replace(' ', '_')
                body_masks[key_name] = mask
        
        return body_masks
    
    def _analyze_clothing_regions(self, parsing_map: np.ndarray) -> Dict[str, Any]:
        """의류 영역 분석 (다음 단계들을 위한)"""
        analysis = {
            "categories_detected": [],
            "coverage_ratio": {},
            "bounding_boxes": {},
            "dominant_category": None
        }
        
        total_pixels = parsing_map.size
        max_coverage = 0.0
        
        for category, part_ids in self.CLOTHING_CATEGORIES.items():
            category_mask = np.zeros_like(parsing_map, dtype=bool)
            
            for part_id in part_ids:
                category_mask |= (parsing_map == part_id)
            
            if category_mask.sum() > 0:
                coverage = category_mask.sum() / total_pixels
                
                analysis["categories_detected"].append(category)
                analysis["coverage_ratio"][category] = coverage
                analysis["bounding_boxes"][category] = self._get_bounding_box(category_mask)
                
                if coverage > max_coverage:
                    max_coverage = coverage
                    analysis["dominant_category"] = category
        
        return analysis
    
    def _calculate_confidence(self, output: torch.Tensor) -> float:
        """전체 신뢰도 계산"""
        max_probs = torch.max(F.softmax(output, dim=1), dim=1)[0]
        confidence = torch.mean(max_probs).item()
        return confidence
    
    def _get_detected_parts(self, parsing_map: np.ndarray) -> Dict[str, Any]:
        """감지된 신체 부위 정보"""
        detected_parts = {}
        
        for part_id, part_name in self.BODY_PARTS.items():
            if part_id == 0:
                continue
            
            mask = (parsing_map == part_id)
            pixel_count = mask.sum()
            
            if pixel_count > 0:
                key_name = part_name.lower().replace('-', '_').replace(' ', '_')
                detected_parts[key_name] = {
                    "pixel_count": int(pixel_count),
                    "percentage": float(pixel_count / parsing_map.size * 100),
                    "bounding_box": self._get_bounding_box(mask),
                    "part_id": part_id
                }
        
        return detected_parts
    
    def _get_bounding_box(self, mask: np.ndarray) -> Dict[str, int]:
        """바운딩 박스 계산"""
        coords = np.where(mask)
        if len(coords[0]) == 0:
            return {"x": 0, "y": 0, "width": 0, "height": 0}
        
        y_min, y_max = coords[0].min(), coords[0].max()
        x_min, x_max = coords[1].min(), coords[1].max()
        
        return {
            "x": int(x_min),
            "y": int(y_min),
            "width": int(x_max - x_min),
            "height": int(y_max - y_min)
        }
    
    def _get_cache_key(self, tensor: torch.Tensor) -> str:
        """캐시 키 생성"""
        tensor_hash = hash(tensor.cpu().numpy().tobytes())
        return f"step01_{tensor_hash}_{self.input_size[0]}x{self.input_size[1]}"
    
    def _update_cache(self, key: str, result: Dict[str, Any]):
        """결과 캐싱 (LRU 방식)"""
        if len(self.result_cache) >= self.cache_size:
            # 가장 오래된 항목 제거
            oldest_key = next(iter(self.result_cache))
            del self.result_cache[oldest_key]
        
        # 캐시에 저장할 때는 무거운 데이터는 복사본 생성
        cached_result = {
            k: v.copy() if isinstance(v, (np.ndarray, dict)) else v 
            for k, v in result.items()
        }
        self.result_cache[key] = cached_result
    
    def _update_stats(self, processing_time: float):
        """성능 통계 업데이트"""
        self.stats["total_inferences"] += 1
        current_avg = self.stats["average_time"]
        new_avg = (current_avg * (self.stats["total_inferences"] - 1) + 
                  processing_time) / self.stats["total_inferences"]
        self.stats["average_time"] = new_avg
    
    def get_clothing_mask(self, parsing_map: np.ndarray, category: str) -> np.ndarray:
        """특정 의류 카테고리의 통합 마스크 반환 (다음 단계들을 위한)"""
        if category not in self.CLOTHING_CATEGORIES:
            raise ValueError(f"지원하지 않는 카테고리: {category}")
        
        combined_mask = np.zeros_like(parsing_map, dtype=np.uint8)
        
        for part_id in self.CLOTHING_CATEGORIES[category]:
            combined_mask |= (parsing_map == part_id).astype(np.uint8)
        
        return combined_mask
    
    def visualize_parsing(self, parsing_map: np.ndarray) -> np.ndarray:
        """파싱 결과 시각화 (디버깅용)"""
        # 20개 부위별 색상 매핑
        colors = np.array([
            [0, 0, 0],       # 0: Background
            [128, 0, 0],     # 1: Hat
            [255, 0, 0],     # 2: Hair
            [0, 85, 0],      # 3: Glove
            [170, 0, 51],    # 4: Sunglasses
            [255, 85, 0],    # 5: Upper-clothes
            [0, 0, 85],      # 6: Dress
            [0, 119, 221],   # 7: Coat
            [85, 85, 0],     # 8: Socks
            [0, 85, 85],     # 9: Pants
            [85, 51, 0],     # 10: Jumpsuits
            [52, 86, 128],   # 11: Scarf
            [0, 128, 0],     # 12: Skirt
            [0, 0, 255],     # 13: Face
            [51, 170, 221],  # 14: Left-arm
            [0, 255, 255],   # 15: Right-arm
            [85, 255, 170],  # 16: Left-leg
            [170, 255, 85],  # 17: Right-leg
            [255, 255, 0],   # 18: Left-shoe
            [255, 170, 0]    # 19: Right-shoe
        ])
        
        colored_parsing = colors[parsing_map]
        return colored_parsing.astype(np.uint8)
    
    async def get_step_stats(self) -> Dict[str, Any]:
        """1단계 성능 통계 반환"""
        return {
            "step_name": "human_parsing",
            "step_number": 1,
            "performance": self.stats,
            "cache_size": len(self.result_cache),
            "device": self.device,
            "models_available": {
                "pytorch": self.pytorch_model is not None,
                "coreml": self.coreml_model is not None
            },
            "memory_usage": await self.memory_manager.get_usage_stats(),
            "configuration": {
                "input_size": self.input_size,
                "num_classes": self.num_classes,
                "cache_limit": self.cache_size,
                "use_quantization": self.enable_quantization
            }
        }
    
    async def cleanup(self):
        """리소스 정리"""
        logger.info("🧹 1단계: 리소스 정리 중...")
        
        # 모델 정리
        if self.pytorch_model:
            del self.pytorch_model
            self.pytorch_model = None
        
        if self.coreml_model:
            del self.coreml_model
            self.coreml_model = None
        
        # 캐시 정리
        self.result_cache.clear()
        
        # 스레드 풀 정리
        self.executor.shutdown(wait=True)
        
        # 메모리 정리
        await self.memory_manager.cleanup()
        
        if self.device == "mps":
            torch.mps.empty_cache()
        elif self.device == "cuda":
            torch.cuda.empty_cache()
        
        self.is_initialized = False
        logger.info("✅ 1단계 리소스 정리 완료")


# 팩토리 함수 (기존 파이프라인 호환)
async def create_human_parsing_step(
    model_loader: ModelLoader, 
    device: str = "auto",
    config: Dict[str, Any] = None
) -> HumanParsingStep:
    """1단계 인체 파싱 스텝 생성"""
    
    if device == "auto":
        # M3 Max 자동 감지
        if MPS_AVAILABLE:
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
    
    default_config = {
        "use_coreml": True,
        "enable_quantization": True,
        "input_size": (512, 512),
        "num_classes": 20,
        "cache_size": 50,
        "batch_size": 1,
        "model_name": "graphonomy",
        "model_path": "ai_models/checkpoints/human_parsing"
    }
    
    final_config = {**default_config, **(config or {})}
    
    step = HumanParsingStep(model_loader, device, final_config)
    
    if not await step.initialize():
        raise RuntimeError("1단계 인체 파싱 초기화 실패")
    
    return step


# 테스트 코드
if __name__ == "__main__":
    import asyncio
    from ..utils.model_loader import ModelLoader
    
    async def test_human_parsing():
        """1단계 테스트"""
        logger.info("🧪 1단계 인체 파싱 테스트 시작")
        
        # 모델 로더 초기화
        model_loader = ModelLoader()
        
        # 1단계 생성
        step1 = await create_human_parsing_step(model_loader)
        
        # 테스트 이미지 생성
        test_image = torch.randn(1, 3, 512, 512)
        
        # 처리
        result = await step1.process(test_image)
        
        # 결과 확인
        print(f"✅ 처리 완료 - 소요시간: {result['processing_time']:.3f}초")
        print(f"🎯 신뢰도: {result['confidence']:.3f}")
        print(f"👤 감지된 부위: {len(result['body_parts_detected'])}개")
        print(f"👕 의류 카테고리: {result['clothing_regions']['categories_detected']}")
        
        # 통계 확인
        stats = await step1.get_step_stats()
        print(f"📊 성능 통계: {stats['performance']}")
        
        # 정리
        await step1.cleanup()
        
        logger.info("✅ 1단계 테스트 완료")
    
    # 테스트 실행
    asyncio.run(test_human_parsing())