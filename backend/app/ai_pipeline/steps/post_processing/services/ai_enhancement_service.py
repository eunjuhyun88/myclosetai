"""
🔥 AI 향상 서비스 (AI 모델 통합)
=================================

전통적 기법 + AI 모델을 통합한 향상 서비스:
1. 전통적 이미지 처리 기법
2. ESRGAN Super-resolution
3. SwinIR Image restoration
4. Face Enhancement
5. 하이브리드 향상 파이프라인

Author: MyCloset AI Team
Date: 2025-08-14
Version: 1.0
"""

import logging
import numpy as np
import cv2
from typing import Dict, Any, Optional, Tuple, List
import torch

# AI 모델 매니저 import
from ..models.ai_model_manager import AIModelManager

logger = logging.getLogger(__name__)

class AIEnhancementService:
    """AI 모델 통합 향상 서비스"""
    
    def __init__(self, device: str = 'cpu'):
        self.logger = logging.getLogger(f"{__name__}.AIEnhancementService")
        
        # AI 모델 매니저 초기화
        self.ai_manager = AIModelManager(device)
        
        # 서비스 통계
        self.service_stats = {
            'total_enhancements': 0,
            'traditional_enhancements': 0,
            'ai_enhancements': 0,
            'hybrid_enhancements': 0,
            'average_enhancement_time': 0.0
        }
        
        # AI 모델 로드 시도
        self._load_ai_models()
    
    def _load_ai_models(self):
        """AI 모델 로드"""
        try:
            self.logger.info("🚀 AI 모델 로드 시작")
            
            if self.ai_manager.load_all_models():
                self.logger.info("✅ AI 모델 로드 완료")
            else:
                self.logger.warning("⚠️ AI 모델 로드 실패 - 전통적 기법만 사용")
                
        except Exception as e:
            self.logger.error(f"❌ AI 모델 로드 중 오류: {e}")
    
    def enhance_image_hybrid(self, 
                           image: np.ndarray, 
                           enhancement_type: str = "comprehensive",
                           use_ai: bool = True) -> Dict[str, Any]:
        """하이브리드 이미지 향상 (전통적 + AI)"""
        try:
            import time
            start_time = time.time()
            
            self.logger.info(f"🚀 하이브리드 이미지 향상 시작: {enhancement_type}")
            
            results = {
                'original': image.copy(),
                'traditional_enhanced': None,
                'ai_enhanced': None,
                'final_result': None,
                'enhancement_type': enhancement_type,
                'processing_time': 0.0,
                'methods_used': []
            }
            
            # 1. 전통적 향상
            traditional_enhanced = self._traditional_enhancement(image, enhancement_type)
            results['traditional_enhanced'] = traditional_enhanced
            results['methods_used'].append('traditional')
            self.service_stats['traditional_enhancements'] += 1
            
            # 2. AI 향상 (가능한 경우)
            if use_ai and self.ai_manager.get_available_models():
                try:
                    ai_results = self.ai_manager.run_comprehensive_enhancement(image)
                    results['ai_enhanced'] = ai_results
                    results['methods_used'].extend(ai_results.get('model_usage', []))
                    self.service_stats['ai_enhancements'] += 1
                    self.logger.info("✅ AI 향상 완료")
                except Exception as e:
                    self.logger.warning(f"⚠️ AI 향상 실패, 전통적 기법만 사용: {e}")
            
            # 3. 최종 결과 결정
            if results['ai_enhanced'] and 'esrgan_super_res' in results['ai_enhanced']['enhanced']:
                # AI 결과가 있으면 그것을 사용
                results['final_result'] = results['ai_enhanced']['enhanced']['esrgan_super_res']
                self.service_stats['hybrid_enhancements'] += 1
            else:
                # AI 결과가 없으면 전통적 결과 사용
                results['final_result'] = results['traditional_enhanced']
            
            # 처리 시간 계산
            processing_time = time.time() - start_time
            results['processing_time'] = processing_time
            
            # 통계 업데이트
            self._update_service_stats(processing_time)
            
            self.logger.info(f"✅ 하이브리드 이미지 향상 완료 ({processing_time:.2f}s)")
            return results
            
        except Exception as e:
            self.logger.error(f"❌ 하이브리드 이미지 향상 실패: {e}")
            self._update_service_stats(0.0)
            return {
                'original': image,
                'traditional_enhanced': image,
                'ai_enhanced': None,
                'final_result': image,
                'enhancement_type': enhancement_type,
                'processing_time': 0.0,
                'methods_used': ['traditional'],
                'error': str(e)
            }
    
    def _traditional_enhancement(self, image: np.ndarray, enhancement_type: str) -> np.ndarray:
        """전통적 이미지 향상"""
        try:
            enhanced = image.copy()
            
            if enhancement_type == "comprehensive":
                # 노이즈 제거
                enhanced = cv2.fastNlMeansDenoisingColored(enhanced, None, 10, 10, 7, 21)
                
                # 선명도 향상
                kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
                enhanced = cv2.filter2D(enhanced, -1, kernel)
                
                # 대비 향상
                lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
                l = clahe.apply(l)
                enhanced = cv2.merge([l, a, b])
                enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
                
            elif enhancement_type == "noise_reduction":
                enhanced = cv2.fastNlMeansDenoisingColored(enhanced, None, 10, 10, 7, 21)
                
            elif enhancement_type == "sharpness":
                kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
                enhanced = cv2.filter2D(enhanced, -1, kernel)
                
            elif enhancement_type == "contrast":
                lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
                l = clahe.apply(l)
                enhanced = cv2.merge([l, a, b])
                enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
            
            return enhanced
            
        except Exception as e:
            self.logger.error(f"❌ 전통적 향상 실패: {e}")
            return image
    
    def enhance_with_ai_only(self, image: np.ndarray) -> Dict[str, Any]:
        """AI 모델만 사용한 향상"""
        try:
            if not self.ai_manager.get_available_models():
                raise RuntimeError("사용 가능한 AI 모델이 없습니다")
            
            self.logger.info("🚀 AI 전용 향상 시작")
            
            results = self.ai_manager.run_comprehensive_enhancement(image)
            
            self.service_stats['ai_enhancements'] += 1
            self.logger.info("✅ AI 전용 향상 완료")
            
            return results
            
        except Exception as e:
            self.logger.error(f"❌ AI 전용 향상 실패: {e}")
            return {
                'original': image,
                'enhanced': {},
                'processing_time': 0.0,
                'model_usage': [],
                'error': str(e)
            }
    
    def get_enhancement_options(self) -> Dict[str, str]:
        """향상 옵션 반환"""
        options = {
            "comprehensive": "종합 향상 (전통적 + AI)",
            "traditional_only": "전통적 기법만",
            "ai_only": "AI 모델만",
            "hybrid": "하이브리드 (전통적 + AI)"
        }
        
        # AI 모델이 사용 가능한 경우에만 AI 옵션 추가
        if self.ai_manager.get_available_models():
            options.update({
                "esrgan": "ESRGAN Super-resolution",
                "swinir": "SwinIR Image Restoration",
                "face_enhancement": "Face Enhancement"
            })
        
        return options
    
    def get_available_ai_models(self) -> List[str]:
        """사용 가능한 AI 모델 목록 반환"""
        return self.ai_manager.get_available_models()
    
    def get_ai_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """AI 모델 정보 반환"""
        return self.ai_manager.get_model_info(model_name)
    
    def get_all_ai_models_info(self) -> Dict[str, Dict[str, Any]]:
        """모든 AI 모델 정보 반환"""
        return self.ai_manager.get_all_models_info()
    
    def get_ai_model_stats(self) -> Dict[str, Any]:
        """AI 모델 통계 반환"""
        return self.ai_manager.get_model_stats()
    
    def _update_service_stats(self, processing_time: float):
        """서비스 통계 업데이트"""
        try:
            self.service_stats['total_enhancements'] += 1
            
            # 평균 처리 시간 업데이트
            total = self.service_stats['total_enhancements']
            current_avg = self.service_stats['average_enhancement_time']
            new_avg = (current_avg * (total - 1) + processing_time) / total
            self.service_stats['average_enhancement_time'] = new_avg
            
        except Exception as e:
            self.logger.error(f"❌ 서비스 통계 업데이트 실패: {e}")
    
    def get_service_stats(self) -> Dict[str, Any]:
        """서비스 통계 반환"""
        return self.service_stats.copy()
    
    def reset_service_stats(self):
        """서비스 통계 초기화"""
        self.service_stats = {
            'total_enhancements': 0,
            'traditional_enhancements': 0,
            'ai_enhancements': 0,
            'hybrid_enhancements': 0,
            'average_enhancement_time': 0.0
        }
