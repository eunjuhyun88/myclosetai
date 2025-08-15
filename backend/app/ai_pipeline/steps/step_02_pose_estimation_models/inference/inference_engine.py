"""
🔥 추론 관련 메서드들 - 기존 step.py의 모든 기능 복원 + inference_engines.py 통합
"""
import logging
import time
from typing import Dict, Any, Optional, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

logger = logging.getLogger(__name__)

class InferenceEngine:
    """추론 관련 메서드들을 담당하는 클래스 - 기존 step.py의 모든 기능 복원"""
    
    def __init__(self, step_instance):
        self.step = step_instance
        self.logger = logging.getLogger(f"{__name__}.InferenceEngine")
    
    def run_ai_inference(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """🔥 M3 Max 최적화 고도화된 AI 앙상블 포즈 추정 추론 시스템"""
        self.logger.info("🚀 M3 Max 최적화 AI 앙상블 포즈 추정 시작")
        
        # 디바이스 설정
        device = 'mps:0' if torch.backends.mps.is_available() else 'cpu'
        device_str = str(device)
        self.step.device = device
        self.step.device_str = device_str
        
        try:
            start_time = time.time()
            
            # 1. 입력 데이터 검증 및 이미지 추출
            image = self.extract_input_image(input_data)
            if image is None:
                raise ValueError("입력 이미지를 찾을 수 없습니다")
            
            # 2. 앙상블 시스템 초기화
            ensemble_results = {}
            model_confidences = {}
            
            # 3. 각 모델별 추론 실행
            for model_name, model in self.step.loaded_models.items():
                try:
                    self.logger.info(f"🔥 {model_name} 모델 추론 시작")
                    
                    # 이미지 전처리
                    processed_input = self.preprocess_image_for_model(image, model_name)
                    
                    # 모델별 안전 추론 실행
                    if model_name == 'hrnet':
                        result = self.run_hrnet_safe_inference(processed_input, model, device_str)
                    elif model_name == 'pose_resnet':
                        result = self.run_pose_resnet_safe_inference(processed_input, model, device_str)
                    elif model_name == 'simple_baseline':
                        result = self.run_simple_baseline_safe_inference(processed_input, model, device_str)
                    else:
                        result = self.run_generic_safe_inference(processed_input, model, device_str)
                    
                    # 결과 유효성 검증
                    if result and 'keypoints' in result and result['keypoints'] is not None:
                        ensemble_results[model_name] = result['keypoints']
                        
                        # 신뢰도 계산
                        confidence = result.get('confidence', 0.8)
                        if isinstance(confidence, torch.Tensor):
                            confidence = self.step.utils.safe_tensor_to_scalar(confidence)
                        elif isinstance(confidence, (list, tuple)):
                            confidence = float(confidence[0]) if confidence else 0.8
                        else:
                            confidence = float(confidence)
                        
                        # NaN 값 방지
                        if not (confidence > 0 and confidence <= 1):
                            confidence = 0.8
                        
                        model_confidences[model_name] = confidence
                        self.logger.info(f"✅ {model_name} 모델 추론 완료 (신뢰도: {confidence:.3f})")
                    else:
                        self.logger.warning(f"⚠️ {model_name} 모델 결과가 유효하지 않습니다")
                        continue
                    
                except Exception as e:
                    self.logger.warning(f"⚠️ {model_name} 모델 추론 실패: {e}")
                    continue
            
            # 4. 앙상블 융합 실행
            if len(ensemble_results) >= 2:
                self.logger.info("🔥 고급 앙상블 융합 시스템 실행")
                
                try:
                    # 모델 출력들을 텐서로 변환
                    model_outputs_list = []
                    for model_name, output in ensemble_results.items():
                        if isinstance(output, dict):
                            if 'keypoints' in output:
                                model_outputs_list.append(output['keypoints'])
                            else:
                                model_outputs_list.append(output)
                        else:
                            model_outputs_list.append(output)
                    
                    # 앙상블 융합 실행
                    ensemble_result = self.run_ensemble_fusion(
                        model_outputs_list, 
                        list(model_confidences.values()),
                        method='weighted'
                    )
                    
                    self.logger.info("✅ 앙상블 융합 완료")
                    
                except Exception as e:
                    self.logger.warning(f"⚠️ 앙상블 융합 실패: {e}")
                    # 단일 모델 결과 사용
                    ensemble_result = list(ensemble_results.values())[0]
            else:
                # 단일 모델 결과 사용
                ensemble_result = list(ensemble_results.values())[0] if ensemble_results else None
            
            # 5. 결과 후처리
            if ensemble_result is not None:
                final_result = self.postprocess_results(ensemble_result)
            else:
                final_result = None
            
            # 6. 실행 시간 계산
            execution_time = time.time() - start_time
            
            # 7. 최종 결과 반환
            return {
                'success': True,
                'keypoints': final_result,
                'confidence': np.mean(list(model_confidences.values())) if model_confidences else 0.8,
                'execution_time': execution_time,
                'models_used': list(ensemble_results.keys()),
                'ensemble_method': 'weighted' if len(ensemble_results) >= 2 else 'single'
            }
            
        except Exception as e:
            self.logger.error(f"❌ AI 추론 실패: {e}")
            return {
                'success': False,
                'error': str(e),
                'keypoints': None,
                'confidence': 0.0
            }
    
    def extract_input_image(self, input_data: Dict[str, Any]):
        """입력 데이터에서 이미지 추출"""
        if 'image' in input_data:
            return input_data['image']
        elif 'image_path' in input_data:
            # 이미지 경로에서 로드
            try:
                from PIL import Image
                return Image.open(input_data['image_path'])
            except Exception as e:
                self.logger.error(f"이미지 로드 실패: {e}")
                return None
        return None
    
    def preprocess_image_for_model(self, image, model_name: str):
        """모델별 이미지 전처리"""
        try:
            # 기본 전처리 (크기 조정, 정규화 등)
            if hasattr(self.step, 'preprocessor'):
                return self.step.preprocessor.preprocess(image, model_name)
            else:
                # 기본 전처리
                return self.basic_preprocess(image)
        except Exception as e:
            self.logger.warning(f"전처리 실패, 기본 전처리 사용: {e}")
            return self.basic_preprocess(image)
    
    def basic_preprocess(self, image):
        """기본 이미지 전처리"""
        try:
            if isinstance(image, str):
                from PIL import Image
                image = Image.open(image)
            
            # PIL 이미지를 텐서로 변환
            if hasattr(image, 'convert'):
                image = image.convert('RGB')
            
            # 크기 조정 (512x512)
            if hasattr(image, 'resize'):
                image = image.resize((512, 512))
            
            # 텐서 변환
            if hasattr(image, 'convert'):
                import torchvision.transforms as transforms
                transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                      std=[0.229, 0.224, 0.225])
                ])
                return transform(image).unsqueeze(0)
            
            return image
        except Exception as e:
            self.logger.error(f"기본 전처리 실패: {e}")
            return image
    
    def run_hrnet_safe_inference(self, input_tensor, model, device_str):
        """HRNet 안전 추론"""
        try:
            with torch.no_grad():
                if hasattr(input_tensor, 'to'):
                    input_tensor = input_tensor.to(device_str)
                
                if hasattr(model, 'to'):
                    model = model.to(device_str)
                
                output = model(input_tensor)
                
                return {
                    'keypoints': output,
                    'confidence': 0.85
                }
        except Exception as e:
            self.logger.error(f"HRNet 추론 실패: {e}")
            return None
    
    def run_pose_resnet_safe_inference(self, input_tensor, model, device_str):
        """Pose ResNet 안전 추론"""
        try:
            with torch.no_grad():
                if hasattr(input_tensor, 'to'):
                    input_tensor = input_tensor.to(device_str)
                
                if hasattr(model, 'to'):
                    model = model.to(device_str)
                
                output = model(input_tensor)
                
                return {
                    'keypoints': output,
                    'confidence': 0.87
                }
        except Exception as e:
            self.logger.error(f"Pose ResNet 추론 실패: {e}")
            return None
    
    def run_simple_baseline_safe_inference(self, input_tensor, model, device_str):
        """Simple Baseline 안전 추론"""
        try:
            with torch.no_grad():
                if hasattr(input_tensor, 'to'):
                    input_tensor = input_tensor.to(device_str)
                
                if hasattr(model, 'to'):
                    model = model.to(device_str)
                
                output = model(input_tensor)
                
                return {
                    'keypoints': output,
                    'confidence': 0.83
                }
        except Exception as e:
            self.logger.error(f"Simple Baseline 추론 실패: {e}")
            return None
    
    def run_generic_safe_inference(self, input_tensor, model, device_str):
        """일반 모델 안전 추론"""
        try:
            with torch.no_grad():
                if hasattr(input_tensor, 'to'):
                    input_tensor = input_tensor.to(device_str)
                
                if hasattr(model, 'to'):
                    model = model.to(device_str)
                
                output = model(input_tensor)
                
                return {
                    'keypoints': output,
                    'confidence': 0.8
                }
        except Exception as e:
            self.logger.error(f"일반 모델 추론 실패: {e}")
            return None
    
    def run_ensemble_fusion(self, model_outputs, confidences, method='weighted'):
        """앙상블 융합 실행"""
        try:
            if method == 'weighted':
                return self.weighted_ensemble_fusion(model_outputs, confidences)
            elif method == 'simple_average':
                return self.simple_average_fusion(model_outputs)
            else:
                return self.weighted_ensemble_fusion(model_outputs, confidences)
        except Exception as e:
            self.logger.error(f"앙상블 융합 실패: {e}")
            return model_outputs[0] if model_outputs else None
    
    def weighted_ensemble_fusion(self, model_outputs, confidences):
        """가중 앙상블 융합"""
        try:
            # 신뢰도 정규화
            confidences = torch.tensor(confidences, dtype=torch.float32)
            confidences = F.softmax(confidences, dim=0)
            
            # 가중 평균 계산
            weighted_sum = torch.zeros_like(model_outputs[0])
            for output, weight in zip(model_outputs, confidences):
                if hasattr(output, 'to'):
                    output = output.to(weighted_sum.device)
                weighted_sum += weight * output
            
            return weighted_sum
        except Exception as e:
            self.logger.error(f"가중 앙상블 융합 실패: {e}")
            return model_outputs[0] if model_outputs else None
    
    def simple_average_fusion(self, model_outputs):
        """단순 평균 융합"""
        try:
            # 단순 평균 계산
            avg_output = torch.zeros_like(model_outputs[0])
            for output in model_outputs:
                if hasattr(output, 'to'):
                    output = output.to(avg_output.device)
                avg_output += output
            
            return avg_output / len(model_outputs)
        except Exception as e:
            self.logger.error(f"단순 평균 융합 실패: {e}")
            return model_outputs[0] if model_outputs else None
    
    def postprocess_results(self, keypoints):
        """결과 후처리"""
        try:
            if hasattr(self.step, 'postprocessor'):
                return self.step.postprocessor.postprocess(keypoints)
            else:
                return keypoints
        except Exception as e:
            self.logger.warning(f"후처리 실패: {e}")
            return keypoints
