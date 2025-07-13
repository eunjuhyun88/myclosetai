"""
MyCloset AI 3단계: 의류 세그멘테이션 (Clothing Segmentation)
U²-Net + Cloth-Segmentation 기반 배경 제거 및 의류 추출 시스템
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import cv2
import numpy as np
from PIL import Image
import logging
from typing import Dict, List, Optional, Tuple, Any
import asyncio
from pathlib import Path

class DoubleConv(nn.Module):
    """U²-Net의 더블 컨볼루션 블록"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class RSU(nn.Module):
    """Residual U-block (RSU)"""
    def __init__(self, height, in_ch, mid_ch, out_ch, dilated=False):
        super().__init__()
        self.height = height
        
        # 입력 컨볼루션
        self.conv_in = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        
        # 인코더
        self.encoder_convs = nn.ModuleList()
        self.encoder_pools = nn.ModuleList()
        
        for i in range(height):
            if i == 0:
                self.encoder_convs.append(
                    nn.Conv2d(out_ch, mid_ch, 3, padding=1)
                )
            else:
                self.encoder_convs.append(
                    nn.Conv2d(mid_ch, mid_ch, 3, padding=1)
                )
            
            if i < height - 1:
                if dilated:
                    self.encoder_pools.append(
                        nn.Conv2d(mid_ch, mid_ch, 3, padding=2, dilation=2)
                    )
                else:
                    self.encoder_pools.append(nn.MaxPool2d(2, stride=2))
        
        # 바텀 컨볼루션
        if dilated:
            self.conv_bottom = nn.Conv2d(mid_ch, mid_ch, 3, padding=4, dilation=4)
        else:
            self.conv_bottom = nn.Conv2d(mid_ch, mid_ch, 3, padding=1)
        
        # 디코더
        self.decoder_convs = nn.ModuleList()
        self.decoder_upsamples = nn.ModuleList()
        
        for i in range(height):
            self.decoder_convs.append(
                nn.Conv2d(mid_ch * 2 if i == 0 else mid_ch, mid_ch, 3, padding=1)
            )
            
            if i < height - 1:
                if dilated:
                    self.decoder_upsamples.append(nn.Identity())
                else:
                    self.decoder_upsamples.append(nn.Upsample(scale_factor=2, mode='bilinear'))
        
        # 출력 컨볼루션
        self.conv_out = nn.Conv2d(mid_ch, out_ch, 3, padding=1)

    def forward(self, x):
        # 입력 처리
        h_in = self.conv_in(x)
        
        # 인코더
        h_enc = [h_in]
        h = h_in
        
        for i in range(self.height):
            h = F.relu(self.encoder_convs[i](h))
            h_enc.append(h)
            
            if i < self.height - 1:
                h = self.encoder_pools[i](h)
        
        # 바텀
        h = F.relu(self.conv_bottom(h))
        
        # 디코더
        for i in range(self.height):
            if i < self.height - 1:
                h = self.decoder_upsamples[i](h)
            
            h = torch.cat([h, h_enc[-(i+2)]], dim=1)
            h = F.relu(self.decoder_convs[i](h))
        
        # 출력
        h_out = self.conv_out(h)
        
        # 잔차 연결
        return h_out + h_in

class U2Net(nn.Module):
    """U²-Net 세그멘테이션 네트워크"""
    
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        
        # 인코더
        self.stage1 = RSU(7, in_channels, 32, 64)
        self.pool12 = nn.MaxPool2d(2, stride=2)
        
        self.stage2 = RSU(6, 64, 32, 128)
        self.pool23 = nn.MaxPool2d(2, stride=2)
        
        self.stage3 = RSU(5, 128, 64, 256)
        self.pool34 = nn.MaxPool2d(2, stride=2)
        
        self.stage4 = RSU(4, 256, 128, 512)
        self.pool45 = nn.MaxPool2d(2, stride=2)
        
        self.stage5 = RSU(4, 512, 256, 512, dilated=True)
        self.pool56 = nn.MaxPool2d(2, stride=2)
        
        self.stage6 = RSU(4, 512, 256, 512, dilated=True)
        
        # 디코더
        self.stage5d = RSU(4, 1024, 256, 512, dilated=True)
        self.stage4d = RSU(4, 1024, 128, 256)
        self.stage3d = RSU(5, 512, 64, 128)
        self.stage2d = RSU(6, 256, 32, 64)
        self.stage1d = RSU(7, 128, 16, 64)
        
        # 사이드 출력
        self.side1 = nn.Conv2d(64, out_channels, 3, padding=1)
        self.side2 = nn.Conv2d(64, out_channels, 3, padding=1)
        self.side3 = nn.Conv2d(128, out_channels, 3, padding=1)
        self.side4 = nn.Conv2d(256, out_channels, 3, padding=1)
        self.side5 = nn.Conv2d(512, out_channels, 3, padding=1)
        self.side6 = nn.Conv2d(512, out_channels, 3, padding=1)
        
        # 출력 융합
        self.outconv = nn.Conv2d(6 * out_channels, out_channels, 1)

    def forward(self, x):
        h, w = x.size()[2:]
        
        # 인코더
        h1 = self.stage1(x)
        h = self.pool12(h1)
        
        h2 = self.stage2(h)
        h = self.pool23(h2)
        
        h3 = self.stage3(h)
        h = self.pool34(h3)
        
        h4 = self.stage4(h)
        h = self.pool45(h4)
        
        h5 = self.stage5(h)
        h = self.pool56(h5)
        
        h6 = self.stage6(h)
        
        # 디코더
        h5d = self.stage5d(torch.cat([h5, F.upsample(h6, size=h5.size()[2:], mode='bilinear')], 1))
        h4d = self.stage4d(torch.cat([h4, F.upsample(h5d, size=h4.size()[2:], mode='bilinear')], 1))
        h3d = self.stage3d(torch.cat([h3, F.upsample(h4d, size=h3.size()[2:], mode='bilinear')], 1))
        h2d = self.stage2d(torch.cat([h2, F.upsample(h3d, size=h2.size()[2:], mode='bilinear')], 1))
        h1d = self.stage1d(torch.cat([h1, F.upsample(h2d, size=h1.size()[2:], mode='bilinear')], 1))
        
        # 사이드 출력
        d1 = self.side1(h1d)
        d2 = F.upsample(self.side2(h2d), size=(h, w), mode='bilinear')
        d3 = F.upsample(self.side3(h3d), size=(h, w), mode='bilinear')
        d4 = F.upsample(self.side4(h4d), size=(h, w), mode='bilinear')
        d5 = F.upsample(self.side5(h5d), size=(h, w), mode='bilinear')
        d6 = F.upsample(self.side6(h6), size=(h, w), mode='bilinear')
        
        # 출력 융합
        d0 = self.outconv(torch.cat([d1, d2, d3, d4, d5, d6], 1))
        
        return torch.sigmoid(d0), torch.sigmoid(d1), torch.sigmoid(d2), torch.sigmoid(d3), torch.sigmoid(d4), torch.sigmoid(d5), torch.sigmoid(d6)

class ClothTypeClassifier(nn.Module):
    """의류 종류 분류기"""
    
    def __init__(self, num_classes=8):
        super().__init__()
        
        # ResNet-50 백본 (경량화)
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
            
            # ResNet 블록들 (간소화)
            self._make_layer(64, 128, 2, stride=1),
            self._make_layer(128, 256, 2, stride=2),
            self._make_layer(256, 512, 2, stride=2),
            
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # 분류 헤드
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def _make_layer(self, inplanes, planes, blocks, stride=1):
        layers = []
        layers.append(nn.Conv2d(inplanes, planes, 3, stride, 1))
        layers.append(nn.BatchNorm2d(planes))
        layers.append(nn.ReLU(inplace=True))
        
        for _ in range(1, blocks):
            layers.append(nn.Conv2d(planes, planes, 3, 1, 1))
            layers.append(nn.BatchNorm2d(planes))
            layers.append(nn.ReLU(inplace=True))
        
        return nn.Sequential(*layers)

    def forward(self, x):
        features = self.backbone(x)
        features = features.view(features.size(0), -1)
        output = self.classifier(features)
        return output

class ClothSegmentationStep:
    """3단계: 의류 세그멘테이션 실행 클래스"""
    
    # 의류 종류 정의
    CLOTH_TYPES = {
        0: "상의",
        1: "하의", 
        2: "원피스",
        3: "외투",
        4: "신발",
        5: "가방",
        6: "모자",
        7: "액세서리"
    }
    
    def __init__(self, config, device, model_loader):
        self.config = config
        self.device = device
        self.model_loader = model_loader
        self.logger = logging.getLogger(__name__)
        
        # 모델 초기화
        self.segmentation_model = None
        self.classifier_model = None
        self.model_loaded = False
        
        # 전처리 변환
        self.transform = transforms.Compose([
            transforms.Resize((config.image_size, config.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # 후처리 파라미터
        self.mask_threshold = 0.5
        self.min_area_ratio = 0.01  # 최소 영역 비율

    async def process(self, input_tensor: torch.Tensor) -> Dict[str, Any]:
        """의류 세그멘테이션 메인 처리"""
        try:
            # 모델 로드 (필요시)
            if not self.model_loaded:
                await self._load_models()
            
            # 전처리
            processed_input = await self._preprocess(input_tensor)
            
            # 세그멘테이션 추론
            with torch.no_grad():
                segmentation_result = await self._segmentation_inference(processed_input)
            
            # 의류 종류 분류
            with torch.no_grad():
                classification_result = await self._classification_inference(processed_input)
            
            # 후처리
            final_result = await self._postprocess(
                segmentation_result, 
                classification_result, 
                input_tensor.shape[2:]
            )
            
            return {
                "cloth_mask": final_result["cloth_mask"],
                "segmented_cloth": final_result["segmented_cloth"],
                "cloth_type": final_result["cloth_type"],
                "cloth_confidence": final_result["cloth_confidence"],
                "background_removed": final_result["background_removed"],
                "bounding_box": final_result["bounding_box"],
                "metadata": {
                    "mask_quality": final_result["mask_quality"],
                    "cloth_area_ratio": final_result["cloth_area_ratio"],
                    "processing_time": final_result["processing_time"]
                }
            }
            
        except Exception as e:
            self.logger.error(f"의류 세그멘테이션 처리 중 오류: {str(e)}")
            raise

    async def _load_models(self):
        """모델 로드 및 초기화"""
        try:
            # 세그멘테이션 모델 로드
            cached_seg_model = self.model_loader.memory_manager.get_cached_model("cloth_segmentation")
            
            if cached_seg_model is not None:
                self.segmentation_model = cached_seg_model
                self.logger.info("캐시된 의류 세그멘테이션 모델 로드")
            else:
                # 새 U²-Net 모델 생성
                self.segmentation_model = U2Net(in_channels=3, out_channels=1)
                
                # 사전 훈련된 가중치 로드
                checkpoint_path = Path("models/checkpoints/cloth_segmentation_u2net.pth")
                if checkpoint_path.exists():
                    checkpoint = torch.load(checkpoint_path, map_location=self.device)
                    self.segmentation_model.load_state_dict(checkpoint['model_state_dict'])
                    self.logger.info("U²-Net 가중치 로드 완료")
                else:
                    self.logger.warning("U²-Net 가중치를 찾을 수 없습니다. 랜덤 초기화 사용")
                
                # 모델을 디바이스로 이동
                self.segmentation_model = self.segmentation_model.to(self.device)
                
                # FP16 최적화
                if self.config.use_fp16 and self.device.type == "mps":
                    self.segmentation_model = self.segmentation_model.half()
                
                # 평가 모드
                self.segmentation_model.eval()
                
                # 모델 캐싱
                self.model_loader.memory_manager.cache_model("cloth_segmentation", self.segmentation_model)
            
            # 분류 모델 로드
            cached_cls_model = self.model_loader.memory_manager.get_cached_model("cloth_classifier")
            
            if cached_cls_model is not None:
                self.classifier_model = cached_cls_model
                self.logger.info("캐시된 의류 분류 모델 로드")
            else:
                # 새 분류 모델 생성
                self.classifier_model = ClothTypeClassifier(num_classes=len(self.CLOTH_TYPES))
                
                # 사전 훈련된 가중치 로드
                classifier_checkpoint_path = Path("models/checkpoints/cloth_classifier.pth")
                if classifier_checkpoint_path.exists():
                    checkpoint = torch.load(classifier_checkpoint_path, map_location=self.device)
                    self.classifier_model.load_state_dict(checkpoint['model_state_dict'])
                    self.logger.info("의류 분류기 가중치 로드 완료")
                else:
                    self.logger.warning("의류 분류기 가중치를 찾을 수 없습니다. 랜덤 초기화 사용")
                
                # 모델을 디바이스로 이동
                self.classifier_model = self.classifier_model.to(self.device)
                
                # FP16 최적화
                if self.config.use_fp16 and self.device.type == "mps":
                    self.classifier_model = self.classifier_model.half()
                
                # 평가 모드
                self.classifier_model.eval()
                
                # 모델 캐싱
                self.model_loader.memory_manager.cache_model("cloth_classifier", self.classifier_model)
            
            self.model_loaded = True
            self.logger.info("의류 세그멘테이션 모델 로드 완료")
            
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

    async def _segmentation_inference(self, input_tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
        """U²-Net 세그멘테이션 추론"""
        # 메모리 효율적 추론
        if self.config.use_fp16 and self.device.type == "mps":
            with torch.autocast(device_type='cpu'):  # MPS는 autocast 제한적 지원
                outputs = self.segmentation_model(input_tensor)
        else:
            outputs = self.segmentation_model(input_tensor)
        
        # 다중 스케일 출력 처리
        if isinstance(outputs, tuple):
            main_output = outputs[0]  # 메인 출력
            side_outputs = outputs[1:]  # 사이드 출력들
        else:
            main_output = outputs
            side_outputs = []
        
        return {
            "main_mask": main_output,
            "side_masks": side_outputs
        }

    async def _classification_inference(self, input_tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
        """의류 종류 분류 추론"""
        # 메모리 효율적 추론
        if self.config.use_fp16 and self.device.type == "mps":
            with torch.autocast(device_type='cpu'):
                class_logits = self.classifier_model(input_tensor)
        else:
            class_logits = self.classifier_model(input_tensor)
        
        # 소프트맥스 적용
        class_probs = F.softmax(class_logits, dim=1)
        
        return {
            "class_logits": class_logits,
            "class_probs": class_probs
        }

    async def _postprocess(
        self, 
        segmentation_result: Dict[str, torch.Tensor], 
        classification_result: Dict[str, torch.Tensor],
        original_size: Tuple[int, int]
    ) -> Dict[str, Any]:
        """결과 후처리"""
        import time
        start_time = time.time()
        
        # 메인 마스크 처리
        main_mask = segmentation_result["main_mask"].squeeze(0).squeeze(0)  # [H, W]
        
        # 이진 마스크 생성
        binary_mask = (main_mask > self.mask_threshold).float()
        
        # 원본 크기로 리사이즈
        if binary_mask.shape != original_size:
            binary_mask = F.interpolate(
                binary_mask.unsqueeze(0).unsqueeze(0),
                size=original_size,
                mode='bilinear',
                align_corners=True
            ).squeeze(0).squeeze(0)
        
        # 형태학적 연산으로 노이즈 제거
        binary_mask_np = binary_mask.cpu().numpy()
        cleaned_mask = self._morphological_cleanup(binary_mask_np)
        
        # 의류 종류 예측
        class_probs = classification_result["class_probs"].cpu().numpy()[0]
        predicted_class = np.argmax(class_probs)
        confidence = class_probs[predicted_class]
        
        # 바운딩 박스 계산
        bounding_box = self._calculate_bounding_box(cleaned_mask)
        
        # 마스크 품질 평가
        mask_quality = self._evaluate_mask_quality(cleaned_mask)
        
        # 의류 영역 비율
        cloth_area_ratio = np.sum(cleaned_mask) / (cleaned_mask.shape[0] * cleaned_mask.shape[1])
        
        # 배경 제거된 이미지 생성 (원본 이미지 필요시)
        background_removed = self._apply_mask_to_image(cleaned_mask)
        
        processing_time = time.time() - start_time
        
        return {
            "cloth_mask": cleaned_mask,
            "segmented_cloth": background_removed,
            "cloth_type": self.CLOTH_TYPES[predicted_class],
            "cloth_confidence": float(confidence),
            "background_removed": background_removed,
            "bounding_box": bounding_box,
            "mask_quality": mask_quality,
            "cloth_area_ratio": float(cloth_area_ratio),
            "processing_time": processing_time
        }

    def _morphological_cleanup(self, mask: np.ndarray) -> np.ndarray:
        """형태학적 연산으로 마스크 정제"""
        # 노이즈 제거 (opening)
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask_opened = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel_small)
        
        # 홀 채우기 (closing)
        kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask_closed = cv2.morphologyEx(mask_opened, cv2.MORPH_CLOSE, kernel_large)
        
        # 작은 영역 제거
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_closed)
        
        # 최소 영역 임계치 계산
        total_area = mask.shape[0] * mask.shape[1]
        min_area = total_area * self.min_area_ratio
        
        # 큰 영역만 유지
        cleaned_mask = np.zeros_like(mask_closed)
        for i in range(1, num_labels):  # 0은 배경
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                cleaned_mask[labels == i] = 1
        
        return cleaned_mask.astype(np.float32)

    def _calculate_bounding_box(self, mask: np.ndarray) -> Dict[str, int]:
        """마스크에서 바운딩 박스 계산"""
        # 마스크에서 0이 아닌 픽셀 찾기
        coords = np.where(mask > 0)
        
        if len(coords[0]) == 0:  # 마스크가 비어있는 경우
            return {"x": 0, "y": 0, "width": 0, "height": 0}
        
        # 바운딩 박스 좌표 계산
        y_min, y_max = np.min(coords[0]), np.max(coords[0])
        x_min, x_max = np.min(coords[1]), np.max(coords[1])
        
        return {
            "x": int(x_min),
            "y": int(y_min),
            "width": int(x_max - x_min + 1),
            "height": int(y_max - y_min + 1)
        }

    def _evaluate_mask_quality(self, mask: np.ndarray) -> Dict[str, float]:
        """마스크 품질 평가"""
        # 연결성 평가
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
        
        if num_labels <= 1:  # 배경만 있는 경우
            return {
                "connectivity_score": 0.0,
                "smoothness_score": 0.0,
                "completeness_score": 0.0,
                "overall_score": 0.0
            }
        
        # 가장 큰 연결 영역의 비율 (연결성 점수)
        largest_area = np.max(stats[1:, cv2.CC_STAT_AREA])  # 배경 제외
        total_mask_area = np.sum(mask)
        connectivity_score = largest_area / total_mask_area if total_mask_area > 0 else 0
        
        # 경계 매끄러움 평가
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        smoothness_score = 0.0
        
        if contours:
            # 가장 큰 컨투어 선택
            largest_contour = max(contours, key=cv2.contourArea)
            
            # 컨투어 근사화로 매끄러움 평가
            epsilon = 0.02 * cv2.arcLength(largest_contour, True)
            approx = cv2.approxPolyDP(largest_contour, epsilon, True)
            
            # 근사화 정도로 매끄러움 평가 (점이 적을수록 매끄러움)
            smoothness_score = max(0, 1.0 - len(approx) / 100.0)
        
        # 완성도 평가 (바운딩 박스 대비 마스크 면적)
        bbox = self._calculate_bounding_box(mask)
        bbox_area = bbox["width"] * bbox["height"]
        completeness_score = total_mask_area / bbox_area if bbox_area > 0 else 0
        
        # 전체 점수 계산
        overall_score = (connectivity_score * 0.4 + smoothness_score * 0.3 + completeness_score * 0.3)
        
        return {
            "connectivity_score": float(connectivity_score),
            "smoothness_score": float(smoothness_score),
            "completeness_score": float(completeness_score),
            "overall_score": float(overall_score)
        }

    def _apply_mask_to_image(self, mask: np.ndarray, original_image: Optional[np.ndarray] = None) -> np.ndarray:
        """마스크를 이미지에 적용하여 배경 제거"""
        if original_image is None:
            # 원본 이미지가 없으면 마스크만 반환
            return mask
        
        # 3채널 마스크 생성
        if len(mask.shape) == 2:
            mask_3ch = np.stack([mask, mask, mask], axis=2)
        else:
            mask_3ch = mask
        
        # 배경을 투명하게 만들기 위해 알파 채널 추가
        if original_image.shape[2] == 3:
            alpha_channel = mask * 255
            background_removed = np.concatenate([
                original_image * mask_3ch,
                alpha_channel[..., np.newaxis]
            ], axis=2)
        else:
            background_removed = original_image * mask_3ch
        
        return background_removed.astype(np.uint8)

    def get_cloth_statistics(self, segmentation_result: Dict[str, Any]) -> Dict[str, Any]:
        """의류 세그멘테이션 통계 정보"""
        cloth_mask = segmentation_result["cloth_mask"]
        cloth_type = segmentation_result["cloth_type"]
        cloth_confidence = segmentation_result["cloth_confidence"]
        bounding_box = segmentation_result["bounding_box"]
        
        # 마스크 통계
        mask_area = np.sum(cloth_mask)
        total_area = cloth_mask.shape[0] * cloth_mask.shape[1]
        area_ratio = mask_area / total_area
        
        # 바운딩 박스 통계
        bbox_area = bounding_box["width"] * bounding_box["height"]
        bbox_ratio = bbox_area / total_area
        
        # 종횡비 계산
        aspect_ratio = bounding_box["width"] / bounding_box["height"] if bounding_box["height"] > 0 else 0
        
        return {
            "cloth_type": cloth_type,
            "classification_confidence": cloth_confidence,
            "mask_area_pixels": int(mask_area),
            "mask_area_ratio": float(area_ratio),
            "bounding_box": bounding_box,
            "bbox_area_ratio": float(bbox_ratio),
            "aspect_ratio": float(aspect_ratio),
            "mask_quality": segmentation_result["metadata"]["mask_quality"],
            "segmentation_success": area_ratio > self.min_area_ratio
        }

    def visualize_segmentation(
        self, 
        original_image: np.ndarray, 
        segmentation_result: Dict[str, Any], 
        save_path: Optional[str] = None
    ) -> np.ndarray:
        """세그멘테이션 결과 시각화"""
        
        cloth_mask = segmentation_result["cloth_mask"]
        bounding_box = segmentation_result["bounding_box"]
        cloth_type = segmentation_result["cloth_type"]
        
        # 시각화 이미지 생성
        vis_image = original_image.copy()
        
        # 마스크 오버레이 (반투명)
        mask_colored = np.zeros_like(original_image)
        mask_colored[:, :, 1] = cloth_mask * 255  # 녹색 채널
        
        # 마스크 블렌딩
        alpha = 0.3
        vis_image = cv2.addWeighted(vis_image, 1-alpha, mask_colored, alpha, 0)
        
        # 바운딩 박스 그리기
        if bounding_box["width"] > 0 and bounding_box["height"] > 0:
            x, y, w, h = bounding_box["x"], bounding_box["y"], bounding_box["width"], bounding_box["height"]
            cv2.rectangle(vis_image, (x, y), (x + w, y + h), (0, 255, 255), 2)
            
            # 의류 타입 라벨
            label = f"{cloth_type} ({segmentation_result['cloth_confidence']:.2f})"
            cv2.putText(vis_image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, (0, 255, 255), 2)
        
        # 마스크 윤곽선 그리기
        contours, _ = cv2.findContours(
            cloth_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(vis_image, contours, -1, (255, 0, 0), 2)
        
        # 저장 (옵션)
        if save_path:
            cv2.imwrite(save_path, vis_image)
        
        return vis_image

    def extract_cloth_features(self, segmentation_result: Dict[str, Any]) -> Dict[str, Any]:
        """의류 특징 추출"""
        cloth_mask = segmentation_result["cloth_mask"]
        bounding_box = segmentation_result["bounding_box"]
        
        # 컨투어 추출
        contours, _ = cv2.findContours(
            cloth_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        if not contours:
            return {"error": "컨투어를 찾을 수 없습니다"}
        
        # 가장 큰 컨투어 선택
        largest_contour = max(contours, key=cv2.contourArea)
        
        # 기하학적 특징 계산
        area = cv2.contourArea(largest_contour)
        perimeter = cv2.arcLength(largest_contour, True)
        
        # 원형도 (4π * 면적 / 둘레²)
        circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
        
        # 볼록도 (컨투어 면적 / 볼록 껍질 면적)
        hull = cv2.convexHull(largest_contour)
        hull_area = cv2.contourArea(hull)
        convexity = area / hull_area if hull_area > 0 else 0
        
        # 종횡비
        x, y, w, h = cv2.boundingRect(largest_contour)
        aspect_ratio = w / h if h > 0 else 0
        
        # 사각형도 (컨투어 면적 / 바운딩 박스 면적)
        rect_area = w * h
        rectangularity = area / rect_area if rect_area > 0 else 0
        
        # 타원 피팅
        if len(largest_contour) >= 5:
            ellipse = cv2.fitEllipse(largest_contour)
            ellipse_center = ellipse[0]
            ellipse_axes = ellipse[1]
            ellipse_angle = ellipse[2]
        else:
            ellipse_center = (0, 0)
            ellipse_axes = (0, 0)
            ellipse_angle = 0
        
        return {
            "area": float(area),
            "perimeter": float(perimeter),
            "circularity": float(circularity),
            "convexity": float(convexity),
            "aspect_ratio": float(aspect_ratio),
            "rectangularity": float(rectangularity),
            "ellipse_center": ellipse_center,
            "ellipse_axes": ellipse_axes,
            "ellipse_angle": float(ellipse_angle),
            "num_vertices": len(largest_contour)
        }

    async def warmup(self, dummy_input: torch.Tensor):
        """모델 워밍업"""
        if not self.model_loaded:
            await self._load_models()
        
        with torch.no_grad():
            # 세그멘테이션 워밍업
            _ = await self._segmentation_inference(dummy_input)
            
            # 분류 워밍업
            _ = await self._classification_inference(dummy_input)
        
        self.logger.info("의류 세그멘테이션 모델 워밍업 완료")

    def cleanup(self):
        """리소스 정리"""
        if self.segmentation_model is not None:
            del self.segmentation_model
            self.segmentation_model = None
        
        if self.classifier_model is not None:
            del self.classifier_model
            self.classifier_model = None
            
        self.model_loaded = False
        self.logger.info("의류 세그멘테이션 모델 리소스 정리 완료")

# 유틸리티 함수들
class ClothSegmentationUtils:
    """의류 세그멘테이션 유틸리티 함수들"""
    
    @staticmethod
    def refine_mask_with_grabcut(image: np.ndarray, initial_mask: np.ndarray) -> np.ndarray:
        """GrabCut을 사용한 마스크 정제"""
        try:
            # GrabCut 초기화
            mask = np.where(initial_mask > 0.5, cv2.GC_FGD, cv2.GC_BGD).astype(np.uint8)
            
            # 바운딩 박스 계산
            coords = np.where(initial_mask > 0.5)
            if len(coords[0]) == 0:
                return initial_mask
            
            y_min, y_max = np.min(coords[0]), np.max(coords[0])
            x_min, x_max = np.min(coords[1]), np.max(coords[1])
            rect = (x_min, y_min, x_max - x_min, y_max - y_min)
            
            # GrabCut 실행
            bgd_model = np.zeros((1, 65), np.float64)
            fgd_model = np.zeros((1, 65), np.float64)
            
            cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_MASK)
            
            # 결과 마스크 생성
            refined_mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 1, 0).astype(np.float32)
            
            return refined_mask
            
        except Exception as e:
            logging.warning(f"GrabCut 정제 실패: {e}")
            return initial_mask

    @staticmethod
    def remove_small_objects(mask: np.ndarray, min_size: int = 100) -> np.ndarray:
        """작은 객체 제거"""
        # 연결 요소 분석
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask.astype(np.uint8), connectivity=8
        )
        
        # 큰 객체만 유지
        cleaned_mask = np.zeros_like(mask)
        for i in range(1, num_labels):  # 0은 배경
            if stats[i, cv2.CC_STAT_AREA] >= min_size:
                cleaned_mask[labels == i] = 1
        
        return cleaned_mask.astype(np.float32)

    @staticmethod
    def fill_holes(mask: np.ndarray) -> np.ndarray:
        """마스크 내부 홀 채우기"""
        # 컨투어 찾기
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        # 가장 큰 컨투어로 마스크 채우기
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            filled_mask = np.zeros_like(mask)
            cv2.fillPoly(filled_mask, [largest_contour], 1)
            return filled_mask.astype(np.float32)
        
        return mask

    @staticmethod
    def smooth_mask_boundary(mask: np.ndarray, kernel_size: int = 5) -> np.ndarray:
        """마스크 경계 스무딩"""
        # 가우시안 블러 적용
        smoothed = cv2.GaussianBlur(mask.astype(np.float32), (kernel_size, kernel_size), 0)
        
        # 이진화
        smoothed_binary = (smoothed > 0.5).astype(np.float32)
        
        return smoothed_binary

# 사용 예시
async def example_usage():
    """의류 세그멘테이션 사용 예시"""
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
    
    # 의류 세그멘테이션 단계 초기화
    cloth_segmentation = ClothSegmentationStep(config, device, model_loader)
    
    # 더미 입력 생성 (의류 이미지)
    dummy_input = torch.randn(1, 3, 512, 512).to(device)
    
    # 처리
    result = await cloth_segmentation.process(dummy_input)
    
    print(f"세그멘테이션 완료 - 의류 종류: {result['cloth_type']}")
    print(f"분류 신뢰도: {result['cloth_confidence']:.3f}")
    print(f"처리 시간: {result['metadata']['processing_time']:.2f}초")
    
    # 통계 정보
    stats = cloth_segmentation.get_cloth_statistics(result)
    print(f"마스크 영역 비율: {stats['mask_area_ratio']:.2%}")
    print(f"바운딩 박스: {stats['bounding_box']}")
    
    # 특징 추출
    features = cloth_segmentation.extract_cloth_features(result)
    print("의류 기하학적 특징:")
    for feature_name, feature_value in features.items():
        if isinstance(feature_value, float):
            print(f"  {feature_name}: {feature_value:.3f}")

if __name__ == "__main__":
    asyncio.run(example_usage())