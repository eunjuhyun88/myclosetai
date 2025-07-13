"""
3단계: 의류 세그멘테이션 (Cloth Segmentation) - 배경 제거
U²-Net, rembg, 또는 커스텀 세그멘테이션 모델 사용
"""
import os
import logging
import time
from typing import Dict, Any, Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from PIL import Image

# rembg 임포트 (실제 구현)
try:
    from rembg import remove, new_session
    REMBG_AVAILABLE = True
except ImportError:
    REMBG_AVAILABLE = False
    logging.warning("rembg를 찾을 수 없습니다. 데모 모드로 실행됩니다.")

logger = logging.getLogger(__name__)

class ClothSegmentationStep:
    """의류 세그멘테이션 스텝 - 배경 제거 및 의류 영역 분할"""
    
    # 의류 카테고리
    CLOTHING_CATEGORIES = {
        'upper': ['shirt', 't-shirt', 'blouse', 'sweater', 'jacket', 'coat', 'dress'],
        'lower': ['pants', 'jeans', 'skirt', 'shorts', 'trousers'],
        'full': ['dress', 'jumpsuit', 'overall'],
        'accessories': ['hat', 'scarf', 'gloves', 'shoes', 'bag']
    }
    
    def __init__(self, model_loader, device: str, config: Dict[str, Any] = None):
        """
        Args:
            model_loader: 모델 로더 인스턴스
            device: 사용할 디바이스
            config: 설정 딕셔너리
        """
        self.model_loader = model_loader
        self.device = device
        self.config = config or {}
        
        # 기본 설정
        self.model_type = self.config.get('model_type', 'rembg')  # 'rembg', 'u2net', 'custom'
        self.input_size = self.config.get('input_size', (320, 320))
        self.confidence_threshold = self.config.get('confidence_threshold', 0.5)
        
        # 모델 관련
        self.segmentation_model = None
        self.rembg_session = None
        self.is_initialized = False
        
        logger.info(f"🎯 의류 세그멘테이션 스텝 초기화 - 모델: {self.model_type}, 디바이스: {device}")
    
    async def initialize(self) -> bool:
        """세그멘테이션 모델 초기화"""
        try:
            logger.info("🔄 의류 세그멘테이션 모델 로드 중...")
            
            if self.model_type == 'rembg' and REMBG_AVAILABLE:
                # rembg 세션 초기화
                self.rembg_session = self._initialize_rembg()
            elif self.model_type == 'u2net':
                # U²-Net 모델 로드
                self.segmentation_model = await self._initialize_u2net()
            else:
                # 커스텀 세그멘테이션 모델
                self.segmentation_model = await self._initialize_custom_model()
            
            self.is_initialized = True
            logger.info("✅ 의류 세그멘테이션 모델 로드 완료")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ 의류 세그멘테이션 모델 로드 실패: {e}")
            self.is_initialized = False
            return False
    
    def _initialize_rembg(self):
        """rembg 세션 초기화"""
        try:
            # 의류에 특화된 모델 선택
            model_name = self.config.get('rembg_model', 'u2net')  # u2net, silueta, etc.
            session = new_session(model_name)
            logger.info(f"✅ rembg 세션 초기화 완료 - 모델: {model_name}")
            return session
        except Exception as e:
            logger.warning(f"rembg 초기화 실패: {e}")
            return None
    
    async def _initialize_u2net(self):
        """U²-Net 모델 초기화"""
        try:
            # 실제 구현에서는 U²-Net 모델 로드
            model = self._create_u2net_model()
            
            # 모델 가중치 로드 (사전 훈련된 가중치)
            model_path = self._get_u2net_model_path()
            if os.path.exists(model_path):
                checkpoint = torch.load(model_path, map_location=self.device)
                model.load_state_dict(checkpoint['model_state_dict'])
                logger.info(f"✅ U²-Net 가중치 로드: {model_path}")
            else:
                logger.warning(f"⚠️ U²-Net 가중치 파일 없음: {model_path}")
            
            # 모델 최적화
            model = self.model_loader.optimize_model(model, 'cloth_segmentation')
            model.eval()
            
            return model
            
        except Exception as e:
            logger.warning(f"U²-Net 초기화 실패: {e}")
            return self._create_demo_segmentation_model()
    
    async def _initialize_custom_model(self):
        """커스텀 세그멘테이션 모델 초기화"""
        return self._create_demo_segmentation_model()
    
    def _create_u2net_model(self):
        """U²-Net 모델 아키텍처 생성"""
        class U2NetSegmentation(torch.nn.Module):
            def __init__(self, in_ch=3, out_ch=1):
                super(U2NetSegmentation, self).__init__()
                
                # 인코더
                self.encoder1 = self._conv_block(in_ch, 64)
                self.encoder2 = self._conv_block(64, 128)
                self.encoder3 = self._conv_block(128, 256)
                self.encoder4 = self._conv_block(256, 512)
                
                # 중간 레이어
                self.middle = self._conv_block(512, 1024)
                
                # 디코더
                self.decoder4 = self._conv_block(1024 + 512, 512)
                self.decoder3 = self._conv_block(512 + 256, 256)
                self.decoder2 = self._conv_block(256 + 128, 128)
                self.decoder1 = self._conv_block(128 + 64, 64)
                
                # 출력 레이어
                self.final = torch.nn.Conv2d(64, out_ch, 1)
                
                # 풀링 및 업샘플링
                self.pool = torch.nn.MaxPool2d(2)
                self.upsample = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            
            def _conv_block(self, in_ch, out_ch):
                return torch.nn.Sequential(
                    torch.nn.Conv2d(in_ch, out_ch, 3, padding=1),
                    torch.nn.BatchNorm2d(out_ch),
                    torch.nn.ReLU(inplace=True),
                    torch.nn.Conv2d(out_ch, out_ch, 3, padding=1),
                    torch.nn.BatchNorm2d(out_ch),
                    torch.nn.ReLU(inplace=True)
                )
            
            def forward(self, x):
                # 인코더
                e1 = self.encoder1(x)
                e2 = self.encoder2(self.pool(e1))
                e3 = self.encoder3(self.pool(e2))
                e4 = self.encoder4(self.pool(e3))
                
                # 중간
                m = self.middle(self.pool(e4))
                
                # 디코더
                d4 = self.decoder4(torch.cat([self.upsample(m), e4], dim=1))
                d3 = self.decoder3(torch.cat([self.upsample(d4), e3], dim=1))
                d2 = self.decoder2(torch.cat([self.upsample(d3), e2], dim=1))
                d1 = self.decoder1(torch.cat([self.upsample(d2), e1], dim=1))
                
                # 출력
                output = torch.sigmoid(self.final(d1))
                
                return output
        
        return U2NetSegmentation().to(self.device)
    
    def _create_demo_segmentation_model(self):
        """데모용 세그멘테이션 모델"""
        class DemoSegmentationModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = torch.nn.Conv2d(3, 64, 3, padding=1)
                self.conv2 = torch.nn.Conv2d(64, 32, 3, padding=1)
                self.conv3 = torch.nn.Conv2d(32, 1, 1)
                
            def forward(self, x):
                x = torch.relu(self.conv1(x))
                x = torch.relu(self.conv2(x))
                x = torch.sigmoid(self.conv3(x))
                return x
        
        return DemoSegmentationModel().to(self.device)
    
    def _get_u2net_model_path(self) -> str:
        """U²-Net 모델 파일 경로"""
        model_dir = self.config.get('model_dir', 'app/models/ai_models/u2net')
        model_file = self.config.get('model_file', 'u2net_cloth.pth')
        return os.path.join(model_dir, model_file)
    
    def process(self, clothing_image_tensor: torch.Tensor) -> Dict[str, Any]:
        """
        의류 세그멘테이션 처리
        
        Args:
            clothing_image_tensor: 의류 이미지 텐서
            
        Returns:
            처리 결과 딕셔너리
        """
        if not self.is_initialized:
            raise RuntimeError("의류 세그멘테이션 모델이 초기화되지 않았습니다.")
        
        start_time = time.time()
        
        try:
            # 텐서를 PIL 이미지로 변환
            pil_image = self._tensor_to_pil(clothing_image_tensor)
            
            # 세그멘테이션 수행
            if self.model_type == 'rembg' and self.rembg_session:
                segmented_image, mask = self._segment_with_rembg(pil_image)
            else:
                segmented_image, mask = self._segment_with_model(clothing_image_tensor)
            
            # 의류 타입 분류
            clothing_type = self._classify_clothing_type(pil_image, mask)
            
            # 세그멘테이션 품질 평가
            quality_metrics = self._evaluate_segmentation_quality(mask)
            
            # 후처리
            processed_mask = self._postprocess_mask(mask)
            refined_image = self._refine_segmentation(pil_image, processed_mask)
            
            processing_time = time.time() - start_time
            
            result = {
                "segmented_image": self._pil_to_base64(refined_image),
                "mask": self._array_to_base64(processed_mask),
                "raw_mask": self._array_to_base64(mask),
                "clothing_type": clothing_type,
                "quality_metrics": quality_metrics,
                "confidence": float(quality_metrics.get('confidence', 0.8)),
                "processing_time": processing_time,
                "background_removed": True,
                "mask_area_ratio": float(np.sum(processed_mask > 0) / processed_mask.size)
            }
            
            logger.info(f"✅ 의류 세그멘테이션 완료 - 처리시간: {processing_time:.3f}초, 타입: {clothing_type}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ 의류 세그멘테이션 처리 실패: {e}")
            raise
    
    def _tensor_to_pil(self, tensor: torch.Tensor) -> Image.Image:
        """PyTorch 텐서를 PIL 이미지로 변환"""
        # [1, 3, H, W] -> [3, H, W]
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0)
        
        # [3, H, W] -> [H, W, 3]
        if tensor.shape[0] == 3:
            tensor = tensor.permute(1, 2, 0)
        
        # 정규화 해제
        if tensor.max() <= 1.0:
            tensor = tensor * 255
        
        # NumPy로 변환
        array = tensor.cpu().numpy().astype(np.uint8)
        
        # PIL 이미지로 변환
        return Image.fromarray(array)
    
    def _segment_with_rembg(self, image: Image.Image) -> Tuple[Image.Image, np.ndarray]:
        """rembg를 사용한 세그멘테이션"""
        try:
            # rembg로 배경 제거
            if self.rembg_session:
                segmented = remove(image, session=self.rembg_session)
            else:
                segmented = remove(image)
            
            # 마스크 추출 (알파 채널)
            if segmented.mode == 'RGBA':
                mask = np.array(segmented)[:, :, 3]  # 알파 채널
                # RGB로 변환
                segmented = segmented.convert('RGB')
            else:
                # 알파 채널이 없는 경우 임시 마스크 생성
                mask = np.ones((segmented.height, segmented.width), dtype=np.uint8) * 255
            
            return segmented, mask
            
        except Exception as e:
            logger.error(f"rembg 세그멘테이션 실패: {e}")
            # 실패 시 원본 이미지와 전체 마스크 반환
            mask = np.ones((image.height, image.width), dtype=np.uint8) * 255
            return image, mask
    
    def _segment_with_model(self, image_tensor: torch.Tensor) -> Tuple[Image.Image, np.ndarray]:
        """딥러닝 모델을 사용한 세그멘테이션"""
        try:
            # 입력 전처리
            input_tensor = self._preprocess_for_model(image_tensor)
            
            # 모델 추론
            with torch.no_grad():
                mask_pred = self.segmentation_model(input_tensor)
                
                # 마스크 후처리
                mask = mask_pred.squeeze().cpu().numpy()
                mask = (mask * 255).astype(np.uint8)
            
            # 원본 크기로 복원
            original_size = image_tensor.shape[2:]
            if mask.shape != original_size:
                mask = cv2.resize(mask, (original_size[1], original_size[0]), interpolation=cv2.INTER_NEAREST)
            
            # 세그멘테이션된 이미지 생성
            pil_image = self._tensor_to_pil(image_tensor)
            segmented_image = self._apply_mask_to_image(pil_image, mask)
            
            return segmented_image, mask
            
        except Exception as e:
            logger.error(f"모델 세그멘테이션 실패: {e}")
            # 실패 시 데모 마스크 생성
            pil_image = self._tensor_to_pil(image_tensor)
            mask = self._create_demo_mask(pil_image)
            segmented_image = self._apply_mask_to_image(pil_image, mask)
            return segmented_image, mask
    
    def _preprocess_for_model(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """모델 입력을 위한 전처리"""
        # 크기 조정
        if image_tensor.shape[2:] != self.input_size:
            image_tensor = F.interpolate(
                image_tensor, 
                size=self.input_size, 
                mode='bilinear', 
                align_corners=False
            )
        
        # 정규화
        if image_tensor.max() > 1.0:
            image_tensor = image_tensor / 255.0
        
        # ImageNet 정규화
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)
        
        normalized = (image_tensor - mean) / std
        
        return normalized
    
    def _create_demo_mask(self, image: Image.Image) -> np.ndarray:
        """데모용 마스크 생성 (중앙 영역)"""
        w, h = image.size
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # 중앙 영역을 의류로 가정
        center_x, center_y = w // 2, h // 2
        radius_x, radius_y = w // 3, h // 3
        
        y, x = np.ogrid[:h, :w]
        center_mask = ((x - center_x) ** 2 / radius_x ** 2 + 
                      (y - center_y) ** 2 / radius_y ** 2) <= 1
        
        mask[center_mask] = 255
        
        return mask
    
    def _apply_mask_to_image(self, image: Image.Image, mask: np.ndarray) -> Image.Image:
        """마스크를 이미지에 적용하여 배경 제거"""
        # PIL 이미지를 RGBA로 변환
        if image.mode != 'RGBA':
            image = image.convert('RGBA')
        
        # 마스크를 알파 채널로 사용
        image_array = np.array(image)
        image_array[:, :, 3] = mask  # 알파 채널 설정
        
        # 배경을 투명하게 만든 이미지 생성
        segmented = Image.fromarray(image_array, 'RGBA')
        
        # RGB 배경 (흰색)으로 변환
        rgb_image = Image.new('RGB', image.size, (255, 255, 255))
        rgb_image.paste(segmented, mask=segmented.split()[3])
        
        return rgb_image
    
    def _classify_clothing_type(self, image: Image.Image, mask: np.ndarray) -> str:
        """의류 타입 분류"""
        try:
            # 마스크 영역 분석
            h, w = mask.shape
            mask_coords = np.where(mask > 0)
            
            if len(mask_coords[0]) == 0:
                return "unknown"
            
            # 바운딩 박스 계산
            y_min, y_max = mask_coords[0].min(), mask_coords[0].max()
            x_min, x_max = mask_coords[1].min(), mask_coords[1].max()
            
            # 종횡비 계산
            height_ratio = (y_max - y_min) / h
            width_ratio = (x_max - x_min) / w
            aspect_ratio = (y_max - y_min) / (x_max - x_min) if (x_max - x_min) > 0 else 1
            
            # 위치 분석
            vertical_center = (y_min + y_max) / 2 / h
            
            # 의류 타입 분류 규칙
            if height_ratio > 0.7 and aspect_ratio > 1.5:
                return "dress"  # 전체 길이, 세로로 긴 형태
            elif vertical_center < 0.4 and height_ratio < 0.6:
                return "shirt"  # 상단 위치, 높이 제한적
            elif vertical_center > 0.6 and height_ratio < 0.6:
                return "pants"  # 하단 위치
            elif height_ratio > 0.8:
                return "full"  # 전체 길이
            elif aspect_ratio > 2.0:
                return "skirt"  # 매우 세로로 긴 형태
            else:
                return "shirt"  # 기본값
            
        except Exception as e:
            logger.warning(f"의류 타입 분류 실패: {e}")
            return "unknown"
    
    def _evaluate_segmentation_quality(self, mask: np.ndarray) -> Dict[str, float]:
        """세그멘테이션 품질 평가"""
        metrics = {}
        
        try:
            # 마스크 영역 비율
            mask_area = np.sum(mask > 0) / mask.size
            metrics["mask_area_ratio"] = float(mask_area)
            
            # 마스크 연결성 분석
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            metrics["num_components"] = len(contours)
            
            # 가장 큰 연결 요소의 비율
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                largest_area = cv2.contourArea(largest_contour)
                metrics["largest_component_ratio"] = float(largest_area / np.sum(mask > 0)) if np.sum(mask > 0) > 0 else 0
            else:
                metrics["largest_component_ratio"] = 0
            
            # 경계 부드러움 (경계선 길이 대비 면적)
            if contours:
                total_perimeter = sum(cv2.arcLength(c, True) for c in contours)
                total_area = np.sum(mask > 0)
                if total_perimeter > 0:
                    compactness = 4 * np.pi * total_area / (total_perimeter ** 2)
                    metrics["compactness"] = float(min(1.0, compactness))
                else:
                    metrics["compactness"] = 0
            else:
                metrics["compactness"] = 0
            
            # 전체 신뢰도 계산
            confidence = (
                metrics["mask_area_ratio"] * 0.3 +
                (1 / max(1, metrics["num_components"])) * 0.3 +
                metrics["largest_component_ratio"] * 0.2 +
                metrics["compactness"] * 0.2
            )
            metrics["confidence"] = float(min(1.0, max(0.0, confidence)))
            
        except Exception as e:
            logger.warning(f"품질 평가 중 오류: {e}")
            metrics = {"confidence": 0.5, "mask_area_ratio": 0.5}
        
        return metrics
    
    def _postprocess_mask(self, mask: np.ndarray) -> np.ndarray:
        """마스크 후처리"""
        try:
            # 노이즈 제거
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            
            # 모폴로지 연산으로 구멍 메우기
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            
            # 작은 노이즈 제거
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            
            # 가장자리 부드럽게 만들기
            mask = cv2.GaussianBlur(mask, (3, 3), 0)
            
            # 이진화
            _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
            
            return mask
            
        except Exception as e:
            logger.warning(f"마스크 후처리 실패: {e}")
            return mask
    
    def _refine_segmentation(self, image: Image.Image, mask: np.ndarray) -> Image.Image:
        """세그멘테이션 결과 개선"""
        try:
            # 마스크를 부드럽게 만들기 위한 추가 처리
            soft_mask = cv2.GaussianBlur(mask, (5, 5), 0)
            soft_mask = soft_mask / 255.0  # 0-1 범위로 정규화
            
            # 원본 이미지에 부드러운 마스크 적용
            image_array = np.array(image)
            
            # 배경색 설정 (흰색)
            background = np.ones_like(image_array) * 255
            
            # 부드러운 블렌딩
            refined_array = (image_array * soft_mask[:, :, np.newaxis] + 
                           background * (1 - soft_mask[:, :, np.newaxis]))
            
            refined_image = Image.fromarray(refined_array.astype(np.uint8))
            
            return refined_image
            
        except Exception as e:
            logger.warning(f"세그멘테이션 개선 실패: {e}")
            return self._apply_mask_to_image(image, mask)
    
    def _pil_to_base64(self, image: Image.Image) -> str:
        """PIL 이미지를 base64로 변환"""
        import io
        import base64
        
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG', quality=90)
        img_str = base64.b64encode(buffer.getvalue()).decode()
        
        return img_str
    
    def _array_to_base64(self, array: np.ndarray) -> str:
        """NumPy 배열을 base64로 변환"""
        import io
        import base64
        
        # 마스크를 이미지로 변환
        mask_image = Image.fromarray(array)
        
        buffer = io.BytesIO()
        mask_image.save(buffer, format='PNG')
        img_str = base64.b64encode(buffer.getvalue()).decode()
        
        return img_str
    
    async def get_model_info(self) -> Dict[str, Any]:
        """모델 정보 반환"""
        return {
            "model_type": self.model_type,
            "input_size": self.input_size,
            "confidence_threshold": self.confidence_threshold,
            "device": self.device,
            "initialized": self.is_initialized,
            "rembg_available": REMBG_AVAILABLE,
            "clothing_categories": self.CLOTHING_CATEGORIES,
            "supported_formats": ["RGB", "RGBA"]
        }
    
    async def cleanup(self):
        """리소스 정리"""
        if self.segmentation_model:
            del self.segmentation_model
            self.segmentation_model = None
        
        if self.rembg_session:
            # rembg 세션 정리 (필요한 경우)
            self.rembg_session = None
        
        self.is_initialized = False
        logger.info("🧹 의류 세그멘테이션 스텝 리소스 정리 완료")