"""
Final Output Quality Assessment
최종 출력 품질을 평가하고 검증합니다.
"""

import logging
import numpy as np
from typing import Dict, Any, Optional, Tuple
from pathlib import Path

# 로깅 설정
logger = logging.getLogger(__name__)

class FinalOutputQualityAssessment:
    """최종 출력 품질 평가기"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.quality_thresholds = self.config.get('quality_thresholds', {
            'min_resolution': (256, 256),
            'min_confidence': 0.7,
            'max_noise_level': 0.1
        })
        
    def assess_output_quality(self, output_data: Dict[str, Any]) -> Dict[str, Any]:
        """출력 품질 평가"""
        try:
            quality_score = 0.0
            quality_issues = []
            
            # 해상도 검사
            resolution_score = self._check_resolution(output_data)
            quality_score += resolution_score
            
            # 신뢰도 검사
            confidence_score = self._check_confidence(output_data)
            quality_score += confidence_score
            
            # 노이즈 레벨 검사
            noise_score = self._check_noise_level(output_data)
            quality_score += noise_score
            
            # 품질 등급 결정
            quality_grade = self._determine_quality_grade(quality_score)
            
            return {
                'quality_score': quality_score,
                'quality_grade': quality_grade,
                'quality_issues': quality_issues,
                'resolution_score': resolution_score,
                'confidence_score': confidence_score,
                'noise_score': noise_score,
                'assessment_passed': quality_score >= 2.0
            }
            
        except Exception as e:
            logger.error(f"❌ 품질 평가 실패: {e}")
            return {
                'quality_score': 0.0,
                'quality_grade': 'F',
                'quality_issues': [f'품질 평가 오류: {str(e)}'],
                'assessment_passed': False
            }
    
    def _check_resolution(self, output_data: Dict[str, Any]) -> float:
        """해상도 검사"""
        try:
            image = output_data.get('final_output_image')
            if image is None:
                return 0.0
            
            # 이미지 크기 확인
            if hasattr(image, 'shape'):
                height, width = image.shape[:2]
            elif hasattr(image, 'size'):
                width, height = image.size
            else:
                return 0.0
            
            min_width, min_height = self.quality_thresholds['min_resolution']
            
            if width >= min_width and height >= min_height:
                return 1.0
            else:
                return 0.5
                
        except Exception as e:
            logger.warning(f"⚠️ 해상도 검사 실패: {e}")
            return 0.0
    
    def _check_confidence(self, output_data: Dict[str, Any]) -> float:
        """신뢰도 검사"""
        try:
            confidence = output_data.get('confidence', 0.0)
            min_confidence = self.quality_thresholds['min_confidence']
            
            if confidence >= min_confidence:
                return 1.0
            elif confidence >= min_confidence * 0.8:
                return 0.7
            else:
                return 0.3
                
        except Exception as e:
            logger.warning(f"⚠️ 신뢰도 검사 실패: {e}")
            return 0.0
    
    def _check_noise_level(self, output_data: Dict[str, Any]) -> float:
        """노이즈 레벨 검사"""
        try:
            image = output_data.get('final_output_image')
            if image is None:
                return 0.0
            
            # 간단한 노이즈 레벨 계산
            if hasattr(image, 'shape'):
                # NumPy 배열인 경우
                if len(image.shape) == 3:
                    # RGB 이미지
                    gray = np.mean(image, axis=2)
                else:
                    gray = image
                
                # 노이즈 레벨 계산 (간단한 방법)
                noise_level = np.std(gray) / 255.0
            else:
                # PIL 이미지인 경우
                noise_level = 0.05  # 기본값
            
            max_noise = self.quality_thresholds['max_noise_level']
            
            if noise_level <= max_noise:
                return 1.0
            elif noise_level <= max_noise * 1.5:
                return 0.7
            else:
                return 0.3
                
        except Exception as e:
            logger.warning(f"⚠️ 노이즈 레벨 검사 실패: {e}")
            return 0.0
    
    def _determine_quality_grade(self, quality_score: float) -> str:
        """품질 등급 결정"""
        if quality_score >= 2.7:
            return 'A'
        elif quality_score >= 2.3:
            return 'B'
        elif quality_score >= 2.0:
            return 'C'
        elif quality_score >= 1.5:
            return 'D'
        else:
            return 'F'
    
    def validate_output(self, output_data: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """출력 검증"""
        try:
            assessment_result = self.assess_output_quality(output_data)
            is_valid = assessment_result['assessment_passed']
            
            return is_valid, assessment_result
            
        except Exception as e:
            logger.error(f"❌ 출력 검증 실패: {e}")
            return False, {
                'quality_score': 0.0,
                'quality_grade': 'F',
                'quality_issues': [f'검증 오류: {str(e)}'],
                'assessment_passed': False
            }
    
    def get_quality_report(self, output_data: Dict[str, Any]) -> str:
        """품질 보고서 생성"""
        try:
            assessment_result = self.assess_output_quality(output_data)
            
            report = f"""
🎯 최종 출력 품질 평가 보고서
================================
📊 품질 점수: {assessment_result['quality_score']:.2f}/3.0
🏆 품질 등급: {assessment_result['quality_grade']}
✅ 검증 통과: {'예' if assessment_result['assessment_passed'] else '아니오'}

📈 세부 점수:
  - 해상도: {assessment_result['resolution_score']:.1f}/1.0
  - 신뢰도: {assessment_result['confidence_score']:.1f}/1.0
  - 노이즈: {assessment_result['noise_score']:.1f}/1.0

🔍 품질 이슈: {assessment_result['quality_issues']}
================================
            """
            
            return report.strip()
            
        except Exception as e:
            logger.error(f"❌ 품질 보고서 생성 실패: {e}")
            return f"품질 보고서 생성 실패: {str(e)}"
