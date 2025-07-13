"""
MyCloset AI 1단계: 인체 파싱 (Human Parsing)
20개 신체 부위를 정확히 분할하는 고정밀 파싱 시스템
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import cv2
from PIL import Image
import logging
from typing import Dict, List, Optional, Tuple, Any
import asyncio
from pathlib import Path

# Graphonomy 기반 인체 파싱 모델
class GraphonomyNet(nn.Module):
    """Graphonomy 기반 인체 파싱 네트워크"""
    
    def __init__(self, num_classes: int = 20):
        super().__init__()
        self.num_classes = num_classes
        
        # ResNet-101 백본 (경량화 버전)
        self.backbone = self._create_backbone()
        
        # ASPP (Atrous Spatial Pyramid Pooling)
        self.aspp = ASPP(2048, 256)
        
        # Graph Reasoning Module
        self.graph_module = GraphReasoningModule(256, num_classes)
        
        # 출력 헤드
        self.classifier = nn.Conv2d(256, num_classes, 1)
        
        # Edge Detection Branch
        self.edge_detector = EdgeDetectionBranch(256)

    def _create_backbone(self):
        """경량화된 ResNet-101 백본 생성"""
        # 실제로는 torchvision.models.resnet101을 사용하되 일부 레이어 제거
        backbone = nn.Sequential(
            # Conv1
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
            
            # ResNet blocks (간소화)
            self._make_layer(64, 256, 3, stride=1),
            self._make_layer(256, 512, 4, stride=2),
            self._make_layer(512, 1024, 6, stride=2),
            self._make_layer(1024, 2048, 3, stride=1, dilation=2)
        )
        return backbone

    def _make_layer(self, inplanes, planes, blocks, stride=1, dilation=1):
        """ResNet 레이어 생성"""
        layers = []
        layers.append(BasicBlock(inplanes, planes, stride, dilation))
        for _ in range(1, blocks):
            layers.append(BasicBlock(planes, planes, 1, dilation))
        return nn.Sequential(*layers)

    def forward(self, x):
        # 백본 특징 추출
        features = self.backbone(x)
        
        # ASPP 적용
        aspp_features = self.aspp(features)
        
        # 그래프 추론
        graph_features = self.graph_module(aspp_features)
        
        # 분류 헤드
        parsing_output = self.classifier(graph_features)
        
        # 엣지 검출
        edge_output = self.edge_detector(graph_features)
        
        # 업샘플링
        parsing_output = F.interpolate(
            parsing_output, size=x.shape[2:], mode='bilinear', align_corners=True
        )
        edge_output = F.interpolate(
            edge_output, size=x.shape[2:], mode='bilinear', align_corners=True
        )
        
        return parsing_output, edge_output

class BasicBlock(nn.Module):
    """ResNet BasicBlock"""
    def __init__(self, inplanes, planes, stride=1, dilation=1):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride, dilation, dilation)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, dilation, dilation)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or inplanes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inplanes, planes, 1, stride),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # 다양한 dilation rate로 컨볼루션
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1)
        self.conv6 = nn.Conv2d(in_channels, out_channels, 3, padding=6, dilation=6)
        self.conv12 = nn.Conv2d(in_channels, out_channels, 3, padding=12, dilation=12)
        self.conv18 = nn.Conv2d(in_channels, out_channels, 3, padding=18, dilation=18)
        
        # Global Average Pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_gap = nn.Conv2d(in_channels, out_channels, 1)
        
        # 출력 컨볼루션
        self.conv_out = nn.Conv2d(out_channels * 5, out_channels, 1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        size = x.shape[2:]
        
        # 각 브랜치 계산
        conv1 = self.conv1(x)
        conv6 = self.conv6(x)
        conv12 = self.conv12(x)
        conv18 = self.conv18(x)
        
        # Global Average Pooling 브랜치
        gap = self.global_avg_pool(x)
        gap = self.conv_gap(gap)
        gap = F.interpolate(gap, size=size, mode='bilinear', align_corners=True)
        
        # 모든 브랜치 결합
        concat = torch.cat([conv1, conv6, conv12, conv18, gap], dim=1)
        output = self.relu(self.bn(self.conv_out(concat)))
        
        return output

class GraphReasoningModule(nn.Module):
    """그래프 추론 모듈 - 신체 부위 간 관계 모델링"""
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.num_classes = num_classes
        
        # 노드 특징 추출
        self.node_proj = nn.Conv2d(in_channels, in_channels // 2, 1)
        
        # 그래프 어텐션
        self.attention = nn.MultiheadAttention(in_channels // 2, 8, batch_first=True)
        
        # 출력 프로젝션
        self.output_proj = nn.Conv2d(in_channels // 2, in_channels, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        
        # 노드 특징 추출
        node_features = self.node_proj(x)  # [B, C//2, H, W]
        
        # 공간 차원을 시퀀스로 변환
        node_features = node_features.view(B, C // 2, H * W).permute(0, 2, 1)  # [B, HW, C//2]
        
        # 그래프 어텐션 적용
        attended_features, _ = self.attention(node_features, node_features, node_features)
        
        # 원래 형태로 복원
        attended_features = attended_features.permute(0, 2, 1).view(B, C // 2, H, W)
        
        # 출력 프로젝션
        output = self.output_proj(attended_features)
        
        return output + x  # 잔차 연결

class EdgeDetectionBranch(nn.Module):
    """엣지 검출 브랜치"""
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // 2, 3, padding=1)
        self.conv2 = nn.Conv2d(in_channels // 2, in_channels // 4, 3, padding=1)
        self.conv3 = nn.Conv2d(in_channels // 4, 1, 1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = torch.sigmoid(self.conv3(x))
        return x

class HumanParsingStep:
    """1단계: 인체 파싱 실행 클래스"""
    
    # 20개 신체 부위 라벨 정의
    BODY_PARTS = {
        0: "background",
        1: "head",
        2: "hair", 
        3: "glove",
        4: "sunglasses",
        5: "upper_clothes",
        6: "dress", 
        7: "coat",
        8: "socks",
        9: "pants",
        10: "jumpsuits",
        11: "scarf",
        12: "skirt",
        13: "face",
        14: "left_arm",
        15: "right_arm",
        16: "left_leg", 
        17: "right_leg",
        18: "left_shoe",
        19: "right_shoe"
    }
    
    def __init__(self, config, device, model_loader):
        self.config = config
        self.device = device
        self.model_loader = model_loader
        self.logger = logging.getLogger(__name__)
        
        # 모델 초기화
        self.model = None
        self.model_loaded = False
        
        # 전처리 변환
        self.transform = transforms.Compose([
            transforms.Resize((config.image_size, config.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # 후처리 설정
        self.smoothing_kernel_size = 5
        self.confidence_threshold = 0.5

    async def process(self, input_tensor: torch.Tensor) -> Dict[str, Any]:
        """인체 파싱 메인 처리"""
        try:
            # 모델 로드 (필요시)
            if not self.model_loaded:
                await self._load_model()
            
            # 전처리
            processed_input = await self._preprocess(input_tensor)
            
            # 추론
            with torch.no_grad():
                parsing_result, edge_result = await self._inference(processed_input)
            
            # 후처리
            final_result = await self._postprocess(parsing_result, edge_result)
            
            return {
                "parsing_map": final_result["parsing_map"],
                "edge_map": final_result["edge_map"], 
                "body_parts": final_result["body_parts"],
                "confidence_scores": final_result["confidence_scores"],
                "metadata": {
                    "num_parts": len(self.BODY_PARTS),
                    "image_size": self.config.image_size,
                    "processing_time": final_result["processing_time"]
                }
            }
            
        except Exception as e:
            self.logger.error(f"인체 파싱 처리 중 오류: {str(e)}")
            raise

    async def _load_model(self):
        """모델 로드 및 초기화"""
        try:
            # 캐시된 모델 확인
            cached_model = self.model_loader.memory_manager.get_cached_model("human_parsing")
            
            if cached_model is not None:
                self.model = cached_model
                self.logger.info("캐시된 인체 파싱 모델 로드")
            else:
                # 새 모델 생성
                self.model = GraphonomyNet(num_classes=len(self.BODY_PARTS))
                
                # 사전 훈련된 가중치 로드 (실제 환경에서는 체크포인트 로드)
                checkpoint_path = Path("models/checkpoints/human_parsing_graphonomy.pth")
                if checkpoint_path.exists():
                    checkpoint = torch.load(checkpoint_path, map_location=self.device)
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                    self.logger.info("사전 훈련된 가중치 로드 완료")
                else:
                    self.logger.warning("사전 훈련된 가중치를 찾을 수 없습니다. 랜덤 초기화 사용")
                
                # 모델을 디바이스로 이동
                self.model = self.model.to(self.device)
                
                # FP16 최적화 (M3 Max)
                if self.config.use_fp16 and self.device.type == "mps":
                    self.model = self.model.half()
                
                # 평가 모드 설정
                self.model.eval()
                
                # 모델 캐싱
                self.model_loader.memory_manager.cache_model("human_parsing", self.model)
            
            self.model_loaded = True
            self.logger.info("인체 파싱 모델 로드 완료")
            
        except Exception as e:
            self.logger.error(f"모델 로드 실패: {str(e)}")
            raise

    async def _preprocess(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """입력 전처리"""
        # 이미 텐서인 경우 정규화만 수행
        if input_tensor.dim() == 4:  # [B, C, H, W]
            # 정규화
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)
            
            if self.config.use_fp16 and self.device.type == "mps":
                mean = mean.half()
                std = std.half()
                input_tensor = input_tensor.half()
            
            normalized = (input_tensor - mean) / std
            return normalized
        else:
            raise ValueError(f"예상하지 못한 입력 텐서 형태: {input_tensor.shape}")

    async def _inference(self, input_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """모델 추론"""
        # 메모리 효율적 추론
        if self.config.use_fp16 and self.device.type == "mps":
            with torch.autocast(device_type='cpu'):  # MPS는 autocast 제한적 지원
                parsing_output, edge_output = self.model(input_tensor)
        else:
            parsing_output, edge_output = self.model(input_tensor)
        
        return parsing_output, edge_output

    async def _postprocess(self, parsing_output: torch.Tensor, edge_output: torch.Tensor) -> Dict[str, Any]:
        """결과 후처리"""
        import time
        start_time = time.time()
        
        # 소프트맥스 적용하여 확률 분포 계산
        parsing_probs = F.softmax(parsing_output, dim=1)
        
        # 최대 확률 클래스 선택
        parsing_map = torch.argmax(parsing_probs, dim=1)  # [B, H, W]
        
        # 신뢰도 점수 계산
        confidence_scores = torch.max(parsing_probs, dim=1)[0]  # [B, H, W]
        
        # 엣지 맵 이진화
        edge_map = (edge_output > 0.5).float()
        
        # CPU로 이동 및 numpy 변환
        parsing_map_np = parsing_map.cpu().numpy()
        edge_map_np = edge_map.squeeze(1).cpu().numpy()
        confidence_np = confidence_scores.cpu().numpy()
        
        # 각 신체 부위별 마스크 생성
        body_parts_masks = {}
        for part_id, part_name in self.BODY_PARTS.items():
            mask = (parsing_map_np == part_id).astype(np.uint8)
            if mask.sum() > 0:  # 해당 부위가 존재하는 경우만
                body_parts_masks[part_name] = mask
        
        # 형태학적 연산으로 노이즈 제거
        parsing_map_cleaned = self._apply_morphology(parsing_map_np[0])
        
        # 신뢰도 기반 필터링
        parsing_map_filtered = self._confidence_filtering(
            parsing_map_cleaned, confidence_np[0]
        )
        
        processing_time = time.time() - start_time
        
        return {
            "parsing_map": parsing_map_filtered,
            "edge_map": edge_map_np[0],
            "body_parts": body_parts_masks,
            "confidence_scores": confidence_np[0],
            "processing_time": processing_time
        }

    def _apply_morphology(self, parsing_map: np.ndarray) -> np.ndarray:
        """형태학적 연산으로 파싱 결과 정제"""
        # 각 클래스별로 형태학적 연산 적용
        cleaned_map = parsing_map.copy()
        
        for class_id in np.unique(parsing_map):
            if class_id == 0:  # 배경 제외
                continue
                
            # 해당 클래스 마스크 추출
            mask = (parsing_map == class_id).astype(np.uint8)
            
            # 노이즈 제거 (opening)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            mask_opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            
            # 홀 채우기 (closing)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask_closed = cv2.morphologyEx(mask_opened, cv2.MORPH_CLOSE, kernel)
            
            # 결과 적용
            cleaned_map[mask_closed == 0] = np.where(
                cleaned_map[mask_closed == 0] == class_id, 0, cleaned_map[mask_closed == 0]
            )
        
        return cleaned_map

    def _confidence_filtering(self, parsing_map: np.ndarray, confidence_map: np.ndarray) -> np.ndarray:
        """신뢰도 기반 파싱 결과 필터링"""
        filtered_map = parsing_map.copy()
        
        # 낮은 신뢰도 영역을 배경으로 설정
        low_confidence_mask = confidence_map < self.confidence_threshold
        filtered_map[low_confidence_mask] = 0
        
        return filtered_map

    def get_body_part_statistics(self, parsing_result: Dict[str, Any]) -> Dict[str, Any]:
        """신체 부위별 통계 정보"""
        parsing_map = parsing_result["parsing_map"]
        confidence_scores = parsing_result["confidence_scores"]
        
        stats = {}
        total_pixels = parsing_map.size
        
        for part_id, part_name in self.BODY_PARTS.items():
            mask = (parsing_map == part_id)
            pixel_count = mask.sum()
            
            if pixel_count > 0:
                stats[part_name] = {
                    "pixel_count": int(pixel_count),
                    "area_ratio": float(pixel_count / total_pixels),
                    "avg_confidence": float(confidence_scores[mask].mean()),
                    "min_confidence": float(confidence_scores[mask].min()),
                    "max_confidence": float(confidence_scores[mask].max())
                }
        
        return stats

    def visualize_parsing_result(self, parsing_map: np.ndarray, save_path: Optional[str] = None) -> np.ndarray:
        """파싱 결과 시각화"""
        # 컬러 팔레트 정의 (각 신체 부위별 고유 색상)
        color_palette = [
            [0, 0, 0],      # 0: background
            [255, 0, 0],    # 1: head
            [255, 255, 0],  # 2: hair
            [255, 0, 255],  # 3: glove
            [0, 255, 255],  # 4: sunglasses
            [0, 255, 0],    # 5: upper_clothes
            [0, 0, 255],    # 6: dress
            [255, 128, 0],  # 7: coat
            [255, 0, 128],  # 8: socks
            [128, 255, 0],  # 9: pants
            [128, 0, 255],  # 10: jumpsuits
            [0, 128, 255],  # 11: scarf
            [255, 128, 128], # 12: skirt
            [128, 255, 128], # 13: face
            [128, 128, 255], # 14: left_arm
            [255, 255, 128], # 15: right_arm
            [255, 128, 255], # 16: left_leg
            [128, 255, 255], # 17: right_leg
            [64, 128, 128],  # 18: left_shoe
            [128, 64, 128]   # 19: right_shoe
        ]
        
        # 컬러 맵 생성
        colored_map = np.zeros((parsing_map.shape[0], parsing_map.shape[1], 3), dtype=np.uint8)
        
        for class_id, color in enumerate(color_palette):
            if class_id < len(self.BODY_PARTS):
                mask = (parsing_map == class_id)
                colored_map[mask] = color
        
        # 저장 (옵션)
        if save_path:
            Image.fromarray(colored_map).save(save_path)
        
        return colored_map

    async def warmup(self, dummy_input: torch.Tensor):
        """모델 워밍업"""
        if not self.model_loaded:
            await self._load_model()
        
        with torch.no_grad():
            _ = await self._inference(dummy_input)
        
        self.logger.info("인체 파싱 모델 워밍업 완료")

    def cleanup(self):
        """리소스 정리"""
        if self.model is not None:
            del self.model
            self.model = None
            self.model_loaded = False
        
        self.logger.info("인체 파싱 모델 리소스 정리 완료")

# 유틸리티 함수들
class HumanParsingUtils:
    """인체 파싱 유틸리티 함수들"""
    
    @staticmethod
    def extract_clothing_regions(parsing_map: np.ndarray) -> Dict[str, np.ndarray]:
        """의류 영역만 추출"""
        clothing_parts = ["upper_clothes", "dress", "coat", "pants", "jumpsuits", "skirt"]
        clothing_regions = {}
        
        for part_name in clothing_parts:
            if part_name in HumanParsingStep.BODY_PARTS.values():
                part_id = list(HumanParsingStep.BODY_PARTS.keys())[
                    list(HumanParsingStep.BODY_PARTS.values()).index(part_name)
                ]
                mask = (parsing_map == part_id).astype(np.uint8)
                if mask.sum() > 0:
                    clothing_regions[part_name] = mask
        
        return clothing_regions

    @staticmethod
    def get_skin_regions(parsing_map: np.ndarray) -> np.ndarray:
        """피부 영역 추출"""
        skin_parts = ["head", "face", "left_arm", "right_arm", "left_leg", "right_leg"]
        skin_mask = np.zeros_like(parsing_map, dtype=np.uint8)
        
        for part_name in skin_parts:
            if part_name in HumanParsingStep.BODY_PARTS.values():
                part_id = list(HumanParsingStep.BODY_PARTS.keys())[
                    list(HumanParsingStep.BODY_PARTS.values()).index(part_name)
                ]
                skin_mask = np.logical_or(skin_mask, parsing_map == part_id)
        
        return skin_mask.astype(np.uint8)

    @staticmethod
    def calculate_body_proportions(parsing_map: np.ndarray) -> Dict[str, float]:
        """신체 비율 계산"""
        total_body_pixels = 0
        part_pixels = {}
        
        # 신체 부위별 픽셀 수 계산
        for part_id, part_name in HumanParsingStep.BODY_PARTS.items():
            if part_id == 0:  # 배경 제외
                continue
            
            mask = (parsing_map == part_id)
            pixel_count = mask.sum()
            part_pixels[part_name] = pixel_count
            total_body_pixels += pixel_count
        
        # 비율 계산
        proportions = {}
        if total_body_pixels > 0:
            for part_name, pixel_count in part_pixels.items():
                proportions[part_name] = pixel_count / total_body_pixels
        
        return proportions

# 사용 예시
async def example_usage():
    """인체 파싱 사용 예시"""
    from ..utils.memory_manager import GPUMemoryManager
    from ..utils.model_loader import ModelLoader
    
    # 설정
    class Config:
        image_size = 512
        use_fp16 = True
    
    config = Config()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # 메모리 매니저 및 모델 로더
    memory_manager = GPUMemoryManager(device, 16.0)
    model_loader = ModelLoader(device, True)
    model_loader.memory_manager = memory_manager
    
    # 인체 파싱 단계 초기화
    human_parsing = HumanParsingStep(config, device, model_loader)
    
    # 더미 입력 생성
    dummy_input = torch.randn(1, 3, 512, 512).to(device)
    
    # 처리
    result = await human_parsing.process(dummy_input)
    
    print(f"파싱 완료 - {len(result['body_parts'])}개 신체 부위 감지")
    print(f"처리 시간: {result['metadata']['processing_time']:.2f}초")
    
    # 시각화
    colored_map = human_parsing.visualize_parsing_result(
        result['parsing_map'], 
        "parsing_result.png"
    )
    
    # 통계 정보
    stats = human_parsing.get_body_part_statistics(result)
    for part_name, stat in stats.items():
        print(f"{part_name}: {stat['area_ratio']:.2%} (신뢰도: {stat['avg_confidence']:.2f})")

if __name__ == "__main__":
    asyncio.run(example_usage())