"""
1단계: 인체 파싱 (Human Parsing) - 20개 부위 분할
Graphonomy 또는 Self-Correction for Human Parsing 모델 사용
"""
import os
import logging
import time
from typing import Dict, Any, Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import cv2

# Graphonomy 모델 관련 임포트 (실제 구현 시 필요)
try:
    # 실제 구현에서는 Graphonomy 레포지토리의 모듈들을 임포트
    # from models.graphonomy import Graphonomy
    # from utils.transforms import get_affine_transform
    pass
except ImportError:
    logging.warning("Graphonomy 모듈을 찾을 수 없습니다. 데모 모드로 실행됩니다.")

logger = logging.getLogger(__name__)

class HumanParsingStep:
    """인체 파싱 스텝 - 20개 신체 부위 분할"""
    
    # LIP (Look Into Person) 데이터셋 기반 20개 부위 라벨
    BODY_PARTS = {
        0: "Background",
        1: "Hat",
        2: "Hair", 
        3: "Glove",
        4: "Sunglasses",
        5: "Upper-clothes",
        6: "Dress",
        7: "Coat",
        8: "Socks",
        9: "Pants",
        10: "Jumpsuits",
        11: "Scarf",
        12: "Skirt",
        13: "Face",
        14: "Left-arm",
        15: "Right-arm",
        16: "Left-leg",
        17: "Right-leg",
        18: "Left-shoe",
        19: "Right-shoe"
    }
    
    def __init__(self, model_loader, device: str, config: Dict[str, Any] = None):
        """
        Args:
            model_loader: 모델 로더 인스턴스
            device: 사용할 디바이스 ('cpu', 'cuda', 'mps')
            config: 설정 딕셔너리
        """
        self.model_loader = model_loader
        self.device = device
        self.config = config or {}
        
        # 기본 설정
        self.input_size = self.config.get('input_size', (512, 512))
        self.num_classes = self.config.get('num_classes', 20)
        self.model_name = self.config.get('model_name', 'graphonomy')
        
        # 모델 관련
        self.model = None
        self.is_initialized = False
        
        # 전처리 설정
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])
        
        logger.info(f"🎯 인체 파싱 스텝 초기화 - 디바이스: {device}, 입력 크기: {self.input_size}")
    
    async def initialize(self) -> bool:
        """모델 초기화"""
        try:
            logger.info("🔄 인체 파싱 모델 로드 중...")
            
            # 모델 파일 경로 설정
            model_path = self._get_model_path()
            
            if os.path.exists(model_path):
                # 실제 모델 로드
                self.model = await self._load_real_model(model_path)
            else:
                logger.warning(f"⚠️ 모델 파일을 찾을 수 없음: {model_path}")
                # 데모용 모델 생성
                self.model = self._create_demo_model()
            
            # 모델을 디바이스로 이동 및 최적화
            self.model = self.model_loader.optimize_model(self.model, 'human_parsing')
            
            # 평가 모드로 설정
            self.model.eval()
            
            self.is_initialized = True
            logger.info("✅ 인체 파싱 모델 로드 완료")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ 인체 파싱 모델 로드 실패: {e}")
            self.is_initialized = False
            return False
    
    def _get_model_path(self) -> str:
        """모델 파일 경로 반환"""
        # 실제 구현에서는 다운로드된 모델 경로
        model_dir = self.config.get('model_dir', 'app/models/ai_models/graphonomy')
        model_file = self.config.get('model_file', 'graphonomy_universal.pth')
        return os.path.join(model_dir, model_file)
    
    async def _load_real_model(self, model_path: str):
        """실제 Graphonomy 모델 로드"""
        try:
            # 실제 구현에서는 Graphonomy 모델 로드
            # model = Graphonomy(num_classes=self.num_classes)
            # checkpoint = torch.load(model_path, map_location=self.device)
            # model.load_state_dict(checkpoint['state_dict'])
            
            # 현재는 데모용 모델 반환
            return self._create_demo_model()
            
        except Exception as e:
            logger.error(f"실제 모델 로드 실패: {e}")
            return self._create_demo_model()
    
    def _create_demo_model(self):
        """데모용 세그멘테이션 모델 생성"""
        class DemoHumanParsingModel(torch.nn.Module):
            def __init__(self, num_classes=20):
                super().__init__()
                # 간단한 CNN 아키텍처
                self.backbone = torch.nn.Sequential(
                    torch.nn.Conv2d(3, 64, 3, padding=1),
                    torch.nn.ReLU(),
                    torch.nn.Conv2d(64, 128, 3, padding=1),
                    torch.nn.ReLU(),
                    torch.nn.MaxPool2d(2),
                    torch.nn.Conv2d(128, 256, 3, padding=1),
                    torch.nn.ReLU(),
                    torch.nn.MaxPool2d(2),
                    torch.nn.Conv2d(256, 512, 3, padding=1),
                    torch.nn.ReLU(),
                )
                
                self.decoder = torch.nn.Sequential(
                    torch.nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),
                    torch.nn.ReLU(),
                    torch.nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
                    torch.nn.ReLU(),
                    torch.nn.Conv2d(128, num_classes, 1)
                )
            
            def forward(self, x):
                features = self.backbone(x)
                output = self.decoder(features)
                return F.interpolate(output, size=x.shape[2:], mode='bilinear', align_corners=False)
        
        return DemoHumanParsingModel(self.num_classes).to(self.device)
    
    def process(self, person_image_tensor: torch.Tensor) -> Dict[str, Any]:
        """
        인체 파싱 처리
        
        Args:
            person_image_tensor: 사용자 이미지 텐서 [1, 3, H, W]
            
        Returns:
            처리 결과 딕셔너리
        """
        if not self.is_initialized:
            raise RuntimeError("인체 파싱 모델이 초기화되지 않았습니다.")
        
        start_time = time.time()
        
        try:
            # 입력 전처리
            input_tensor = self._preprocess_image(person_image_tensor)
            
            # 모델 추론
            with torch.no_grad():
                parsing_output = self.model(input_tensor)
                
                # 확률을 클래스 인덱스로 변환
                parsing_map = torch.argmax(parsing_output, dim=1).squeeze().cpu().numpy()
            
            # 후처리
            parsing_result = self._postprocess_parsing(parsing_map, person_image_tensor.shape[2:])
            
            # 신체 부위별 마스크 생성
            body_masks = self._create_body_masks(parsing_map)
            
            # 신뢰도 계산
            confidence = self._calculate_confidence(parsing_output)
            
            processing_time = time.time() - start_time
            
            result = {
                "parsing_map": parsing_map.astype(np.uint8),
                "body_masks": body_masks,
                "confidence": float(confidence),
                "body_parts_detected": self._get_detected_parts(parsing_map),
                "processing_time": processing_time,
                "input_size": self.input_size,
                "num_classes": self.num_classes
            }
            
            logger.info(f"✅ 인체 파싱 완료 - 처리시간: {processing_time:.3f}초, 신뢰도: {confidence:.3f}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ 인체 파싱 처리 실패: {e}")
            raise
    
    def _preprocess_image(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """이미지 전처리"""
        # 크기 조정
        if image_tensor.shape[2:] != self.input_size:
            image_tensor = F.interpolate(
                image_tensor, 
                size=self.input_size, 
                mode='bilinear', 
                align_corners=False
            )
        
        # 정규화 (0-1 범위라고 가정)
        if image_tensor.max() > 1.0:
            image_tensor = image_tensor / 255.0
        
        # ImageNet 정규화
        mean = torch.tensor(self.mean).view(1, 3, 1, 1).to(self.device)
        std = torch.tensor(self.std).view(1, 3, 1, 1).to(self.device)
        
        normalized = (image_tensor - mean) / std
        
        return normalized
    
    def _postprocess_parsing(self, parsing_map: np.ndarray, original_size: Tuple[int, int]) -> np.ndarray:
        """파싱 결과 후처리"""
        # 원본 크기로 복원
        if parsing_map.shape != original_size:
            parsing_map = cv2.resize(
                parsing_map.astype(np.uint8), 
                (original_size[1], original_size[0]), 
                interpolation=cv2.INTER_NEAREST
            )
        
        # 작은 노이즈 제거 (모폴로지 연산)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        parsing_map = cv2.morphologyEx(parsing_map, cv2.MORPH_CLOSE, kernel)
        parsing_map = cv2.morphologyEx(parsing_map, cv2.MORPH_OPEN, kernel)
        
        return parsing_map
    
    def _create_body_masks(self, parsing_map: np.ndarray) -> Dict[str, np.ndarray]:
        """신체 부위별 마스크 생성"""
        body_masks = {}
        
        for part_id, part_name in self.BODY_PARTS.items():
            if part_id == 0:  # 배경 제외
                continue
            
            mask = (parsing_map == part_id).astype(np.uint8)
            if mask.sum() > 0:  # 해당 부위가 감지된 경우만 추가
                body_masks[part_name.lower().replace('-', '_')] = mask
        
        return body_masks
    
    def _calculate_confidence(self, parsing_output: torch.Tensor) -> float:
        """파싱 결과 신뢰도 계산"""
        # 최대 확률값들의 평균으로 신뢰도 계산
        max_probs = torch.max(F.softmax(parsing_output, dim=1), dim=1)[0]
        confidence = torch.mean(max_probs).item()
        
        return confidence
    
    def _get_detected_parts(self, parsing_map: np.ndarray) -> Dict[str, Any]:
        """감지된 신체 부위 정보"""
        detected_parts = {}
        
        for part_id, part_name in self.BODY_PARTS.items():
            if part_id == 0:  # 배경 제외
                continue
            
            mask = (parsing_map == part_id)
            pixel_count = mask.sum()
            
            if pixel_count > 0:
                # 부위별 통계
                detected_parts[part_name.lower().replace('-', '_')] = {
                    "pixel_count": int(pixel_count),
                    "percentage": float(pixel_count / parsing_map.size * 100),
                    "bounding_box": self._get_bounding_box(mask)
                }
        
        return detected_parts
    
    def _get_bounding_box(self, mask: np.ndarray) -> Dict[str, int]:
        """마스크의 바운딩 박스 계산"""
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
    
    def get_body_part_mask(self, parsing_map: np.ndarray, part_names: list) -> np.ndarray:
        """특정 신체 부위들의 통합 마스크 반환"""
        combined_mask = np.zeros_like(parsing_map, dtype=np.uint8)
        
        for part_name in part_names:
            # 부위 이름으로 ID 찾기
            for part_id, name in self.BODY_PARTS.items():
                if name.lower().replace('-', '_') == part_name.lower():
                    combined_mask |= (parsing_map == part_id).astype(np.uint8)
                    break
        
        return combined_mask
    
    def visualize_parsing(self, parsing_map: np.ndarray) -> np.ndarray:
        """파싱 결과 시각화"""
        # 컬러맵 정의 (각 부위별 색상)
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
        
        # 파싱 맵을 컬러 이미지로 변환
        colored_parsing = colors[parsing_map]
        
        return colored_parsing.astype(np.uint8)
    
    async def get_model_info(self) -> Dict[str, Any]:
        """모델 정보 반환"""
        return {
            "model_name": self.model_name,
            "input_size": self.input_size,
            "num_classes": self.num_classes,
            "device": self.device,
            "initialized": self.is_initialized,
            "body_parts": list(self.BODY_PARTS.values()),
            "model_parameters": sum(p.numel() for p in self.model.parameters()) if self.model else 0
        }
    
    async def cleanup(self):
        """리소스 정리"""
        if self.model:
            del self.model
            self.model = None
        
        self.is_initialized = False
        logger.info("🧹 인체 파싱 스텝 리소스 정리 완료")