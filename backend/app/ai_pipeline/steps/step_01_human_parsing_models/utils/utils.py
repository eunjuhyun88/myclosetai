"""
유틸리티 관련 메서드들 - 기존 step.py의 모든 기능 복원
"""
import logging
from typing import Dict, Any, Optional, Tuple, List
import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image
import time

logger = logging.getLogger(__name__)

class Utils:
    """유틸리티 관련 메서드들을 담당하는 클래스 - 기존 step.py의 모든 기능 복원"""
    
    def __init__(self, step_instance):
        self.step = step_instance
        self.logger = logging.getLogger(f"{__name__}.Utils")
    
    def safe_tensor_to_scalar(self, tensor_value) -> Any:
        """텐서를 안전하게 스칼라로 변환하는 메서드"""
        try:
            if isinstance(tensor_value, torch.Tensor):
                if tensor_value.numel() == 1:
                    return tensor_value.item()
                else:
                    # 텐서의 평균값 사용
                    return tensor_value.mean().item()
            else:
                return float(tensor_value)
        except Exception as e:
            self.logger.warning(f"⚠️ 텐서 변환 실패: {e}")
            return 0.8  # 기본값
    
    def safe_extract_tensor_from_list(self, data_list) -> Any:
        """리스트에서 안전하게 텐서를 추출하는 메서드"""
        try:
            if not isinstance(data_list, list) or len(data_list) == 0:
                return None
            
            first_element = data_list[0]
            
            # 직접 텐서인 경우
            if isinstance(first_element, torch.Tensor):
                return first_element
            
            # 딕셔너리인 경우 텐서 찾기
            elif isinstance(first_element, dict):
                # 🔥 우선순위 키 순서로 텐서 찾기
                priority_keys = ['parsing_pred', 'parsing_output', 'output', 'parsing']
                for key in priority_keys:
                    if key in first_element and isinstance(first_element[key], torch.Tensor):
                        return first_element[key]
                
                # 🔥 모든 값에서 텐서 찾기
                for key, value in first_element.items():
                    if isinstance(value, torch.Tensor):
                        return value
            
            return None
        except Exception as e:
            self.logger.warning(f"⚠️ 리스트에서 텐서 추출 실패: {e}")
            return None
    
    def safe_convert_to_numpy(self, data) -> np.ndarray:
        """데이터를 안전하게 NumPy 배열로 변환하는 메서드"""
        try:
            if isinstance(data, np.ndarray):
                return data
            elif isinstance(data, torch.Tensor):
                # 🔥 그래디언트 문제 해결: detach() 사용
                return data.detach().cpu().numpy()
            elif isinstance(data, list):
                tensor = self.safe_extract_tensor_from_list(data)
                if tensor is not None:
                    return tensor.detach().cpu().numpy()
            elif isinstance(data, dict):
                for key in ['parsing', 'parsing_pred', 'output', 'parsing_output']:
                    if key in data and isinstance(data[key], torch.Tensor):
                        return data[key].detach().cpu().numpy()
            
            # 기본값 반환
            return np.zeros((512, 512), dtype=np.uint8)
        except Exception as e:
            self.logger.warning(f"⚠️ NumPy 변환 실패: {e}")
            return np.zeros((512, 512), dtype=np.uint8)
    
    def standardize_tensor_sizes(self, tensors: List[torch.Tensor], target_size: Optional[Tuple[int, int]] = None) -> List[torch.Tensor]:
        """텐서 크기 표준화"""
        try:
            if not tensors:
                return tensors
            
            if target_size is None:
                # 첫 번째 텐서의 크기를 기준으로 사용
                target_size = (tensors[0].shape[-2], tensors[0].shape[-1])
            
            standardized_tensors = []
            for tensor in tensors:
                if tensor.shape[-2:] != target_size:
                    # F.interpolate를 사용하여 크기 조정
                    tensor = torch.nn.functional.interpolate(
                        tensor.unsqueeze(0) if tensor.dim() == 3 else tensor,
                        size=target_size,
                        mode='bilinear',
                        align_corners=False
                    )
                    if tensor.dim() == 4:
                        tensor = tensor.squeeze(0)
                standardized_tensors.append(tensor)
            
            return standardized_tensors
        except Exception as e:
            self.logger.warning(f"⚠️ 텐서 크기 표준화 실패: {e}")
            return tensors
    
    def get_service_from_central_hub(self, service_key: str):
        """Central Hub에서 서비스 가져오기"""
        try:
            from app.ai_pipeline.utils.common_imports import _get_central_hub_container
            container = _get_central_hub_container()
            if container:
                return container.get(service_key)
            return None
        except Exception as e:
            self.logger.warning(f"⚠️ Central Hub 서비스 가져오기 실패: {e}")
            return None
    
    def assess_image_quality(self, image: np.ndarray) -> float:
        """이미지 품질 평가"""
        try:
            if image is None or image.size == 0:
                return 0.0
            
            # 그레이스케일 변환
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image
            
            # 1. 선명도 평가 (Laplacian variance)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            sharpness_score = min(laplacian_var / 1000.0, 1.0)
            
            # 2. 대비 평가
            contrast_score = gray.std() / 255.0
            
            # 3. 밝기 평가
            brightness_score = 1.0 - abs(gray.mean() - 128) / 128.0
            
            # 4. 노이즈 평가 (간단한 방법)
            noise_score = 1.0 - min(gray.std() / 50.0, 1.0)
            
            # 종합 점수 계산
            quality_score = (sharpness_score + contrast_score + brightness_score + noise_score) / 4.0
            
            return max(0.0, min(1.0, quality_score))
        except Exception as e:
            self.logger.warning(f"⚠️ 이미지 품질 평가 실패: {e}")
            return 0.5
    
    def memory_efficient_resize(self, image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """메모리 효율적인 리사이즈"""
        try:
            if image is None:
                return np.zeros(target_size + (3,), dtype=np.uint8)
            
            # PIL을 사용한 메모리 효율적인 리사이즈
            if len(image.shape) == 3:
                pil_image = Image.fromarray(image)
                resized_pil = pil_image.resize(target_size[::-1], Image.LANCZOS)
                return np.array(resized_pil)
            else:
                # 그레이스케일 이미지
                pil_image = Image.fromarray(image, mode='L')
                resized_pil = pil_image.resize(target_size[::-1], Image.LANCZOS)
                return np.array(resized_pil)
        except Exception as e:
            self.logger.warning(f"⚠️ 메모리 효율적인 리사이즈 실패: {e}")
            return cv2.resize(image, target_size[::-1]) if image is not None else np.zeros(target_size + (3,), dtype=np.uint8)
    
    def normalize_lighting(self, image: np.ndarray) -> np.ndarray:
        """조명 정규화"""
        try:
            if image is None:
                return image
            
            # LAB 색공간으로 변환
            if len(image.shape) == 3:
                lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
                l, a, b = cv2.split(lab)
                
                # L 채널에 CLAHE 적용
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                l = clahe.apply(l)
                
                # 다시 RGB로 변환
                lab = cv2.merge([l, a, b])
                normalized = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
                return normalized
            else:
                # 그레이스케일 이미지
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                return clahe.apply(image)
        except Exception as e:
            self.logger.warning(f"⚠️ 조명 정규화 실패: {e}")
            return image
    
    def correct_colors(self, image: np.ndarray) -> np.ndarray:
        """색상 보정"""
        try:
            if image is None or len(image.shape) != 3:
                return image
            
            # 자동 색상 보정
            # 1. 화이트 밸런스 적용
            def white_balance(img):
                result = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
                avg_a = np.average(result[:, :, 1])
                avg_b = np.average(result[:, :, 2])
                result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
                result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
                result = cv2.cvtColor(result, cv2.COLOR_LAB2RGB)
                return result
            
            corrected = white_balance(image)
            
            # 2. 감마 보정
            gamma = 1.1
            inv_gamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
            corrected = cv2.LUT(corrected, table)
            
            return corrected
        except Exception as e:
            self.logger.warning(f"⚠️ 색상 보정 실패: {e}")
            return image
    
    def detect_roi(self, image: np.ndarray) -> Dict[str, Any]:
        """관심 영역 감지"""
        try:
            if image is None:
                return {'roi_detected': False, 'roi_bbox': None, 'confidence': 0.0}
            
            # 그레이스케일 변환
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image
            
            # 1. 엣지 감지
            edges = cv2.Canny(gray, 50, 150)
            
            # 2. 윤곽선 찾기
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return {'roi_detected': False, 'roi_bbox': None, 'confidence': 0.0}
            
            # 3. 가장 큰 윤곽선 선택
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            
            # 4. 바운딩 박스 계산
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            # 5. 신뢰도 계산
            image_area = image.shape[0] * image.shape[1]
            confidence = min(area / image_area * 10, 1.0)
            
            return {
                'roi_detected': True,
                'roi_bbox': (x, y, w, h),
                'confidence': confidence,
                'area': area
            }
        except Exception as e:
            self.logger.warning(f"⚠️ ROI 감지 실패: {e}")
            return {'roi_detected': False, 'roi_bbox': None, 'confidence': 0.0}
    
    def create_safe_input_tensor(self, image: np.ndarray, device_str: str) -> torch.Tensor:
        """안전한 입력 텐서 생성"""
        try:
            if image is None:
                # 기본 텐서 생성
                return torch.zeros((1, 3, 512, 512), device=device_str)
            
            # 이미지 전처리
            if len(image.shape) == 2:
                # 그레이스케일을 RGB로 변환
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            
            # 크기 조정
            if image.shape[:2] != (512, 512):
                image = self.memory_efficient_resize(image, (512, 512))
            
            # 정규화
            image = image.astype(np.float32) / 255.0
            
            # 텐서 변환
            tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
            
            return tensor.to(device_str)
        except Exception as e:
            self.logger.warning(f"⚠️ 안전한 입력 텐서 생성 실패: {e}")
            return torch.zeros((1, 3, 512, 512), device=device_str)
    
    def create_fallback_parsing(self, image: np.ndarray) -> np.ndarray:
        """폴백 파싱 생성"""
        try:
            if image is None:
                return np.zeros((512, 512), dtype=np.uint8)
            
            # 간단한 폴백 파싱 생성
            height, width = image.shape[:2]
            parsing = np.zeros((height, width), dtype=np.uint8)
            
            # 중앙 영역을 배경으로 설정
            center_y, center_x = height // 2, width // 2
            radius = min(height, width) // 4
            
            # 원형 마스크 생성
            y, x = np.ogrid[:height, :width]
            mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
            
            # 배경 (0)과 전경 (1) 설정
            parsing[mask] = 1
            
            return parsing
        except Exception as e:
            self.logger.warning(f"⚠️ 폴백 파싱 생성 실패: {e}")
            return np.zeros((512, 512), dtype=np.uint8)
    
    def extract_input_image(self, input_data: Dict[str, Any]) -> Optional[np.ndarray]:
        """입력 데이터에서 이미지 추출"""
        try:
            if not input_data:
                return None
            
            # 다양한 키에서 이미지 찾기
            image_keys = ['image', 'input_image', 'file_path', 'file', 'data']
            
            for key in image_keys:
                if key in input_data:
                    data = input_data[key]
                    
                    # NumPy 배열인 경우
                    if isinstance(data, np.ndarray):
                        return data
                    
                    # 파일 경로인 경우
                    elif isinstance(data, str):
                        if data.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                            try:
                                image = cv2.imread(data)
                                if image is not None:
                                    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                            except Exception as e:
                                self.logger.warning(f"⚠️ 이미지 로딩 실패: {e}")
                    
                    # PIL 이미지인 경우
                    elif hasattr(data, 'convert'):
                        try:
                            return np.array(data.convert('RGB'))
                        except Exception as e:
                            self.logger.warning(f"⚠️ PIL 이미지 변환 실패: {e}")
            
            return None
        except Exception as e:
            self.logger.warning(f"⚠️ 입력 이미지 추출 실패: {e}")
            return None
    
    def preprocess_image_for_model(self, image: np.ndarray, model_name: str) -> torch.Tensor:
        """모델용 이미지 전처리"""
        try:
            if image is None:
                return torch.zeros((1, 3, 512, 512))
            
            # 이미지 전처리
            processed_image = self.memory_efficient_resize(image, (512, 512))
            processed_image = self.normalize_lighting(processed_image)
            processed_image = self.correct_colors(processed_image)
            
            # 정규화
            processed_image = processed_image.astype(np.float32) / 255.0
            
            # 텐서 변환
            tensor = torch.from_numpy(processed_image).permute(2, 0, 1).unsqueeze(0)
            
            return tensor
        except Exception as e:
            self.logger.warning(f"⚠️ 모델용 이미지 전처리 실패: {e}")
            return torch.zeros((1, 3, 512, 512))
    
    def calculate_confidence(self, parsing_probs, parsing_logits=None, edge_output=None, mode='advanced') -> float:
        """신뢰도 계산"""
        try:
            if parsing_probs is None:
                return 0.5
            
            if isinstance(parsing_probs, torch.Tensor):
                # 텐서인 경우
                if parsing_probs.dim() == 4:
                    # 배치 차원이 있는 경우
                    probs = parsing_probs.squeeze(0)
                else:
                    probs = parsing_probs
                
                # 최대 확률값 사용
                max_probs = torch.max(probs, dim=0)[0]
                confidence = torch.mean(max_probs).item()
            else:
                # NumPy 배열인 경우
                if len(parsing_probs.shape) == 4:
                    probs = parsing_probs.squeeze(0)
                else:
                    probs = parsing_probs
                
                max_probs = np.max(probs, axis=0)
                confidence = np.mean(max_probs)
            
            return max(0.0, min(1.0, confidence))
        except Exception as e:
            self.logger.warning(f"⚠️ 신뢰도 계산 실패: {e}")
            return 0.5
    
    def create_error_response(self, error_message: str) -> Dict[str, Any]:
        """에러 응답 생성"""
        return {
            'success': False,
            'error': error_message,
            'parsing_map': np.zeros((512, 512), dtype=np.uint8),
            'confidence': 0.0,
            'processing_time': 0.0,
            'model_used': 'none',
            'quality_metrics': {
                'overall_quality': 0.0,
                'sharpness': 0.0,
                'contrast': 0.0,
                'brightness': 0.0
            }
        }
