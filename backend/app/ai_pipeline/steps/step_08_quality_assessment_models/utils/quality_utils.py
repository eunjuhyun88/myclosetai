#!/usr/bin/env python3
"""
Quality Assessment 유틸리티 모듈
품질 평가를 위한 핵심 유틸리티 클래스들을 제공합니다.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
import os
import json
from datetime import datetime

# Pandas import (선택적)
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None

# NumPy import (선택적)
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

# PIL import (선택적)
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    Image = None

# OpenCV import (선택적)
try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False
    cv2 = None

# SciPy import (선택적)
try:
    import scipy
    from scipy import ndimage
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    scipy = None
    ndimage = None

# scikit-image import (선택적)
try:
    from skimage import measure, filters, morphology
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    measure = None
    filters = None
    morphology = None


class QualityUtils:
    """품질 평가 유틸리티 클래스"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.QualityUtils")
        self._validate_dependencies()
    
    def _validate_dependencies(self):
        """의존성 검증"""
        if not NUMPY_AVAILABLE:
            self.logger.warning("NumPy가 설치되지 않았습니다. 일부 기능이 제한됩니다.")
        if not PIL_AVAILABLE:
            self.logger.warning("PIL이 설치되지 않았습니다. 이미지 처리 기능이 제한됩니다.")
        if not OPENCV_AVAILABLE:
            self.logger.warning("OpenCV가 설치되지 않았습니다. 컴퓨터 비전 기능이 제한됩니다.")
    
    def analyze_image_quality(self, image: Any) -> Dict[str, Any]:
        """이미지 품질 분석"""
        try:
            if not NUMPY_AVAILABLE:
                return {"error": "NumPy가 필요합니다"}
            
            if isinstance(image, str):
                # 파일 경로인 경우
                if not os.path.exists(image):
                    return {"error": f"파일이 존재하지 않습니다: {image}"}
                
                if PIL_AVAILABLE:
                    img = Image.open(image)
                    img_array = np.array(img)
                elif OPENCV_AVAILABLE:
                    img_array = cv2.imread(image)
                    if img_array is not None:
                        img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
                    else:
                        return {"error": "이미지를 읽을 수 없습니다"}
                else:
                    return {"error": "이미지 읽기 라이브러리가 필요합니다"}
            elif hasattr(image, 'shape'):  # NumPy 배열
                img_array = image
            else:
                return {"error": "지원되지 않는 이미지 형식입니다"}
            
            # 기본 품질 메트릭 계산
            quality_metrics = self._calculate_basic_quality_metrics(img_array)
            
            # 고급 품질 메트릭 계산
            if NUMPY_AVAILABLE and img_array.size > 0:
                advanced_metrics = self._calculate_advanced_quality_metrics(img_array)
                quality_metrics.update(advanced_metrics)
            
            return {
                "success": True,
                "metrics": quality_metrics,
                "image_info": self._get_image_info(img_array)
            }
            
        except Exception as e:
            self.logger.error(f"이미지 품질 분석 실패: {e}")
            return {"error": str(e)}
    
    def _calculate_basic_quality_metrics(self, img_array: np.ndarray) -> Dict[str, float]:
        """기본 품질 메트릭 계산"""
        metrics = {}
        
        try:
            if not NUMPY_AVAILABLE:
                return metrics
            
            # 기본 통계
            metrics['mean_intensity'] = float(np.mean(img_array))
            metrics['std_intensity'] = float(np.std(img_array))
            metrics['min_intensity'] = float(np.min(img_array))
            metrics['max_intensity'] = float(np.max(img_array))
            
            # 동적 범위
            metrics['dynamic_range'] = metrics['max_intensity'] - metrics['min_intensity']
            
            # 이미지 크기
            metrics['width'] = float(img_array.shape[1])
            metrics['height'] = float(img_array.shape[0])
            metrics['total_pixels'] = float(img_array.size)
            
            # 채널 수
            if len(img_array.shape) == 3:
                metrics['channels'] = float(img_array.shape[2])
            else:
                metrics['channels'] = 1.0
            
        except Exception as e:
            self.logger.error(f"기본 메트릭 계산 실패: {e}")
        
        return metrics
    
    def _calculate_advanced_quality_metrics(self, img_array: np.ndarray) -> Dict[str, float]:
        """고급 품질 메트릭 계산"""
        metrics = {}
        
        try:
            if not NUMPY_AVAILABLE:
                return metrics
            
            # 그레이스케일 변환 (컬러 이미지인 경우)
            if len(img_array.shape) == 3:
                if PIL_AVAILABLE:
                    pil_img = Image.fromarray(img_array)
                    gray_img = pil_img.convert('L')
                    gray_array = np.array(gray_img)
                elif OPENCV_AVAILABLE:
                    gray_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
                else:
                    # 간단한 그레이스케일 변환
                    gray_array = np.mean(img_array, axis=2).astype(np.uint8)
            else:
                gray_array = img_array
            
            # 밝기 메트릭
            metrics['brightness'] = float(np.mean(gray_array))
            
            # 대비 메트릭
            metrics['contrast'] = float(np.std(gray_array))
            
            # 선명도 메트릭 (간단한 방법)
            if gray_array.shape[0] > 2 and gray_array.shape[1] > 2:
                # 라플라시안 기반 선명도
                laplacian = cv2.Laplacian(gray_array, cv2.CV_64F) if OPENCV_AVAILABLE else None
                if laplacian is not None:
                    metrics['sharpness'] = float(np.var(laplacian))
                else:
                    # 간단한 차이 기반 선명도
                    diff_x = np.diff(gray_array, axis=1)
                    diff_y = np.diff(gray_array, axis=0)
                    metrics['sharpness'] = float(np.mean(np.abs(diff_x)) + np.mean(np.abs(diff_y)))
            
            # 노이즈 레벨 (간단한 추정)
            if gray_array.shape[0] > 4 and gray_array.shape[1] > 4:
                # 작은 블록의 표준편차 평균으로 노이즈 추정
                block_size = min(8, gray_array.shape[0] // 4, gray_array.shape[1] // 4)
                if block_size > 1:
                    blocks = []
                    for i in range(0, gray_array.shape[0] - block_size, block_size):
                        for j in range(0, gray_array.shape[1] - block_size, block_size):
                            block = gray_array[i:i+block_size, j:j+block_size]
                            blocks.append(np.std(block))
                    if blocks:
                        metrics['noise_level'] = float(np.mean(blocks))
            
        except Exception as e:
            self.logger.error(f"고급 메트릭 계산 실패: {e}")
        
        return metrics
    
    def _get_image_info(self, img_array: np.ndarray) -> Dict[str, Any]:
        """이미지 정보 추출"""
        info = {}
        
        try:
            if not NUMPY_AVAILABLE:
                return info
            
            info['shape'] = img_array.shape
            info['dtype'] = str(img_array.dtype)
            info['size_bytes'] = img_array.nbytes
            
            # 메모리 사용량
            if hasattr(img_array, 'nbytes'):
                info['memory_mb'] = round(img_array.nbytes / (1024 * 1024), 2)
            
            # 데이터 타입 정보
            if img_array.dtype == np.uint8:
                info['bit_depth'] = 8
                info['value_range'] = [0, 255]
            elif img_array.dtype == np.uint16:
                info['bit_depth'] = 16
                info['value_range'] = [0, 65535]
            elif img_array.dtype == np.float32:
                info['bit_depth'] = 32
                info['value_range'] = [-1.0, 1.0]
            elif img_array.dtype == np.float64:
                info['bit_depth'] = 64
                info['value_range'] = [-1.0, 1.0]
            
        except Exception as e:
            self.logger.error(f"이미지 정보 추출 실패: {e}")
        
        return info
    
    def validate_image_format(self, image_path: str) -> Dict[str, Any]:
        """이미지 형식 검증"""
        try:
            if not os.path.exists(image_path):
                return {"valid": False, "error": "파일이 존재하지 않습니다"}
            
            if not PIL_AVAILABLE:
                return {"valid": False, "error": "PIL이 필요합니다"}
            
            # 이미지 열기 시도
            with Image.open(image_path) as img:
                format_info = {
                    "valid": True,
                    "format": img.format,
                    "mode": img.mode,
                    "size": img.size,
                    "info": img.info
                }
                
                # 추가 검증
                if img.size[0] == 0 or img.size[1] == 0:
                    format_info["valid"] = False
                    format_info["error"] = "이미지 크기가 유효하지 않습니다"
                
                return format_info
                
        except Exception as e:
            return {"valid": False, "error": str(e)}
    
    def get_quality_score(self, metrics: Dict[str, float]) -> Dict[str, Any]:
        """품질 점수 계산"""
        try:
            if not metrics:
                return {"score": 0.0, "grade": "F", "details": "메트릭이 없습니다"}
            
            score = 0.0
            max_score = 100.0
            details = []
            
            # 밝기 점수 (0-25점)
            if 'brightness' in metrics:
                brightness = metrics['brightness']
                if 50 <= brightness <= 200:  # 적절한 밝기 범위
                    brightness_score = 25.0
                elif 30 <= brightness <= 220:  # 허용 가능한 범위
                    brightness_score = 20.0
                else:
                    brightness_score = 10.0
                score += brightness_score
                details.append(f"밝기: {brightness_score:.1f}/25.0")
            
            # 대비 점수 (0-25점)
            if 'contrast' in metrics:
                contrast = metrics['contrast']
                if contrast >= 30:  # 좋은 대비
                    contrast_score = 25.0
                elif contrast >= 20:  # 적당한 대비
                    contrast_score = 20.0
                else:
                    contrast_score = 10.0
                score += contrast_score
                details.append(f"대비: {contrast_score:.1f}/25.0")
            
            # 선명도 점수 (0-25점)
            if 'sharpness' in metrics:
                sharpness = metrics['sharpness']
                if sharpness >= 100:  # 매우 선명
                    sharpness_score = 25.0
                elif sharpness >= 50:  # 선명
                    sharpness_score = 20.0
                else:
                    sharpness_score = 10.0
                score += sharpness_score
                details.append(f"선명도: {sharpness_score:.1f}/25.0")
            
            # 노이즈 점수 (0-25점)
            if 'noise_level' in metrics:
                noise = metrics['noise_level']
                if noise <= 5:  # 노이즈 적음
                    noise_score = 25.0
                elif noise <= 15:  # 노이즈 보통
                    noise_score = 20.0
                else:
                    noise_score = 10.0
                score += noise_score
                details.append(f"노이즈: {noise_score:.1f}/25.0")
            
            # 등급 결정
            if score >= 90:
                grade = "A"
            elif score >= 80:
                grade = "B"
            elif score >= 70:
                grade = "C"
            elif score >= 60:
                grade = "D"
            else:
                grade = "F"
            
            return {
                "score": round(score, 1),
                "grade": grade,
                "max_score": max_score,
                "details": details,
                "percentage": round((score / max_score) * 100, 1)
            }
            
        except Exception as e:
            self.logger.error(f"품질 점수 계산 실패: {e}")
            return {"score": 0.0, "grade": "F", "details": [f"오류: {str(e)}"]}
    
    def compare_images(self, image1: Any, image2: Any) -> Dict[str, Any]:
        """두 이미지 비교"""
        try:
            if not NUMPY_AVAILABLE:
                return {"error": "NumPy가 필요합니다"}
            
            # 두 이미지의 품질 분석
            quality1 = self.analyze_image_quality(image1)
            quality2 = self.analyze_image_quality(image2)
            
            if "error" in quality1 or "error" in quality2:
                return {"error": "이미지 분석 실패"}
            
            # 품질 점수 계산
            score1 = self.get_quality_score(quality1["metrics"])
            score2 = self.get_quality_score(quality2["metrics"])
            
            # 비교 결과
            comparison = {
                "image1": {
                    "quality": quality1,
                    "score": score1
                },
                "image2": {
                    "quality": quality2,
                    "score": score2
                },
                "comparison": {
                    "score_difference": round(score1["score"] - score2["score"], 1),
                    "better_image": "image1" if score1["score"] > score2["score"] else "image2",
                    "quality_ratio": round(score1["score"] / score2["score"], 2) if score2["score"] > 0 else 0
                }
            }
            
            return comparison
            
        except Exception as e:
            self.logger.error(f"이미지 비교 실패: {e}")
            return {"error": str(e)}
    
    def generate_quality_report(self, image_path: str, output_path: str = None) -> Dict[str, Any]:
        """품질 보고서 생성"""
        try:
            # 이미지 품질 분석
            quality_result = self.analyze_image_quality(image_path)
            
            if "error" in quality_result:
                return {"error": quality_result["error"]}
            
            # 품질 점수 계산
            score_result = self.get_quality_score(quality_result["metrics"])
            
            # 보고서 생성
            report = {
                "image_path": image_path,
                "analysis_timestamp": str(pd.Timestamp.now()) if PANDAS_AVAILABLE else str(datetime.now()),
                "quality_metrics": quality_result["metrics"],
                "quality_score": score_result,
                "image_info": quality_result["image_info"],
                "recommendations": self._generate_recommendations(score_result, quality_result["metrics"])
            }
            
            # 파일로 저장 (선택사항)
            if output_path:
                try:
                    with open(output_path, 'w', encoding='utf-8') as f:
                        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
                    report["report_saved"] = output_path
                except Exception as e:
                    self.logger.warning(f"보고서 저장 실패: {e}")
            
            return report
            
        except Exception as e:
            self.logger.error(f"품질 보고서 생성 실패: {e}")
            return {"error": str(e)}
    
    def _generate_recommendations(self, score_result: Dict[str, Any], metrics: Dict[str, float]) -> List[str]:
        """개선 권장사항 생성"""
        recommendations = []
        
        try:
            score = score_result.get("score", 0)
            
            if score < 60:
                recommendations.append("전체적인 이미지 품질이 낮습니다. 촬영 환경과 설정을 개선해주세요.")
            
            # 밝기 관련 권장사항
            if 'brightness' in metrics:
                brightness = metrics['brightness']
                if brightness < 50:
                    recommendations.append("이미지가 너무 어둡습니다. 조명을 개선하거나 노출을 높여주세요.")
                elif brightness > 200:
                    recommendations.append("이미지가 너무 밝습니다. 노출을 낮추거나 조명을 조절해주세요.")
            
            # 대비 관련 권장사항
            if 'contrast' in metrics:
                contrast = metrics['contrast']
                if contrast < 20:
                    recommendations.append("이미지 대비가 낮습니다. 촬영 각도나 조명을 조절해주세요.")
            
            # 선명도 관련 권장사항
            if 'sharpness' in metrics:
                sharpness = metrics['sharpness']
                if sharpness < 50:
                    recommendations.append("이미지가 흐릿합니다. 초점을 맞추거나 삼각대를 사용해주세요.")
            
            # 노이즈 관련 권장사항
            if 'noise_level' in metrics:
                noise = metrics['noise_level']
                if noise > 15:
                    recommendations.append("이미지 노이즈가 높습니다. ISO를 낮추거나 조명을 개선해주세요.")
            
            if not recommendations:
                recommendations.append("이미지 품질이 양호합니다. 현재 설정을 유지하세요.")
                
        except Exception as e:
            self.logger.error(f"권장사항 생성 실패: {e}")
            recommendations.append("권장사항을 생성할 수 없습니다.")
        
        return recommendations


# 모듈 레벨 함수들
def create_quality_analyzer() -> QualityUtils:
    """품질 분석기 인스턴스 생성"""
    return QualityUtils()


def analyze_image_quality_quick(image_path: str) -> Dict[str, Any]:
    """빠른 이미지 품질 분석"""
    analyzer = QualityUtils()
    return analyzer.analyze_image_quality(image_path)


def get_quality_score_quick(metrics: Dict[str, float]) -> Dict[str, Any]:
    """빠른 품질 점수 계산"""
    analyzer = QualityUtils()
    return analyzer.get_quality_score(metrics)


# 모듈 초기화
if __name__ == "__main__":
    # 테스트 코드
    analyzer = QualityUtils()
    print("✅ QualityUtils 모듈이 성공적으로 로드되었습니다!")
    print(f"NumPy 사용 가능: {NUMPY_AVAILABLE}")
    print(f"PIL 사용 가능: {PIL_AVAILABLE}")
    print(f"OpenCV 사용 가능: {OPENCV_AVAILABLE}")
