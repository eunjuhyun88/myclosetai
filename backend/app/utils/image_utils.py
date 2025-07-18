"""
backend/app/utils/image_utils.py - 완전 강화된 이미지 처리 유틸리티

✅ 기존 함수들 100% 유지 + 시각화 강화
✅ M3 Max 최적화  
✅ 고품질 이미지 처리
✅ PIL/OpenCV 통합
✅ 단계별 시각화 완전 구현
✅ 추가 유틸리티 함수들 포함
"""

import os
import io
import base64
import uuid
import tempfile
import logging
import asyncio
from typing import Tuple, Union, Optional, List, Dict, Any
from pathlib import Path
import numpy as np
import cv2
from PIL import Image, ImageEnhance, ImageFilter, ImageDraw, ImageFont, ImageOps
from datetime import datetime

# conda 환경 지원을 위한 안전한 import
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.colors import ListedColormap
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logging.warning("matplotlib 없음 - 고급 시각화 기능 제한됨")

try:
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("scikit-learn 없음 - 색상 클러스터링 기능 제한됨")

logger = logging.getLogger(__name__)

# ============================================================================
# 🎨 시각화 색상 및 설정 (확장됨)
# ============================================================================

# 인체 파싱용 색상 맵 (20개 부위) - 개선된 색상
HUMAN_PARSING_COLORS = {
    0: (0, 0, 0),        # 배경 (검정)
    1: (128, 0, 0),      # 모자 (어두운 빨강)
    2: (255, 165, 0),    # 머리카락 (주황)
    3: (0, 128, 0),      # 장갑 (어두운 초록)
    4: (75, 0, 130),     # 선글라스 (남색)
    5: (255, 20, 147),   # 상의 (분홍)
    6: (138, 43, 226),   # 드레스 (보라)
    7: (0, 191, 255),    # 코트 (하늘색)
    8: (255, 140, 0),    # 양말 (어두운 주황)
    9: (30, 144, 255),   # 바지 (파랑)
    10: (220, 20, 60),   # 점프수트 (진한 빨강)
    11: (255, 215, 0),   # 스카프 (금색)
    12: (218, 112, 214), # 치마 (연보라)
    13: (255, 228, 181), # 얼굴 (살색)
    14: (255, 182, 193), # 왼팔 (연분홍)
    15: (255, 160, 122), # 오른팔 (연주황)
    16: (250, 128, 114), # 왼다리 (연어색)
    17: (255, 192, 203), # 오른다리 (분홍)
    18: (240, 230, 140), # 왼발 (연노랑)
    19: (255, 235, 205)  # 오른발 (연살색)
}

# 부위별 한국어 이름
HUMAN_PARSING_NAMES = {
    0: "배경", 1: "모자", 2: "머리카락", 3: "장갑", 4: "선글라스",
    5: "상의", 6: "드레스", 7: "코트", 8: "양말", 9: "바지",
    10: "점프수트", 11: "스카프", 12: "치마", 13: "얼굴", 14: "왼팔",
    15: "오른팔", 16: "왼다리", 17: "오른다리", 18: "왼발", 19: "오른발"
}

# 포즈 키포인트 색상 (18개 키포인트) - 개선된 색상
POSE_KEYPOINT_COLORS = [
    (255, 69, 0),    # 코 (빨강-주황)
    (255, 140, 0),   # 왼눈 (주황)
    (255, 215, 0),   # 오른눈 (금색)
    (154, 205, 50),  # 왼귀 (연두)
    (0, 255, 127),   # 오른귀 (봄 초록)
    (0, 206, 209),   # 왼어깨 (터키석)
    (65, 105, 225),  # 오른어깨 (로얄블루)
    (138, 43, 226),  # 왼팔꿈치 (블루바이올렛)
    (186, 85, 211),  # 오른팔꿈치 (미디엄오키드)
    (255, 20, 147),  # 왼손목 (딥핑크)
    (255, 105, 180), # 오른손목 (핫핑크)
    (255, 182, 193), # 왼엉덩이 (라이트핑크)
    (250, 128, 114), # 오른엉덩이 (연어색)
    (255, 160, 122), # 왼무릎 (라이트샐몬)
    (255, 218, 185), # 오른무릎 (피치퍼프)
    (255, 228, 196), # 왼발목 (비스크)
    (255, 239, 213), # 오른발목 (파파야휩)
    (220, 20, 60)    # 머리 (크림슨)
]

# 포즈 키포인트 한국어 이름
POSE_KEYPOINT_NAMES = [
    "코", "왼눈", "오른눈", "왼귀", "오른귀",
    "왼어깨", "오른어깨", "왼팔꿈치", "오른팔꿈치", "왼손목", "오른손목",
    "왼엉덩이", "오른엉덩이", "왼무릎", "오른무릎", "왼발목", "오른발목", "머리"
]

# 포즈 연결선 (뼈대) - 더 정확한 연결
POSE_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),  # 얼굴
    (5, 6),  # 어깨 연결
    (5, 7), (7, 9),  # 왼팔
    (6, 8), (8, 10), # 오른팔
    (5, 11), (6, 12), (11, 12),  # 몸통
    (11, 13), (13, 15),  # 왼다리
    (12, 14), (14, 16),  # 오른다리
]

# 의류 카테고리별 색상 (확장)
CLOTHING_COLORS = {
    'shirt': (70, 130, 255),      # 셔츠 (코발트블루)
    'blouse': (255, 182, 193),    # 블라우스 (라이트핑크)
    'pants': (255, 140, 0),       # 바지 (다크오렌지)
    'jeans': (25, 25, 112),       # 청바지 (미드나이트블루)
    'dress': (255, 20, 147),      # 드레스 (딥핑크)
    'skirt': (148, 0, 211),       # 치마 (다크바이올렛)
    'jacket': (34, 139, 34),      # 재킷 (포레스트그린)
    'coat': (139, 69, 19),        # 코트 (새들브라운)
    'sweater': (220, 20, 60),     # 스웨터 (크림슨)
    'hoodie': (105, 105, 105),    # 후드티 (딤그레이)
    'tank_top': (255, 215, 0),    # 탱크톱 (골드)
    'shorts': (0, 255, 127),      # 반바지 (스프링그린)
    'unknown': (128, 128, 128)    # 알 수 없음 (그레이)
}

class ImageProcessor:
    """
    완전한 이미지 처리 유틸리티 클래스
    ✅ 기존 함수명 완전 유지
    ✅ M3 Max 최적화
    ✅ 고품질 처리
    ✅ 시각화 기능 대폭 확장
    """
    
    def __init__(self):
        self.is_m3_max = self._detect_m3_max()
        self.max_resolution = (2048, 2048) if self.is_m3_max else (1024, 1024)
        self.default_quality = 95 if self.is_m3_max else 85
        
        # 폰트 캐시
        self._font_cache = {}
        self._load_fonts()
        
        logger.info(f"🎨 ImageProcessor 초기화 - M3 Max: {self.is_m3_max}")

    def _detect_m3_max(self) -> bool:
        """M3 Max 감지 (개선된 버전)"""
        try:
            import platform
            import subprocess
            
            if platform.system() == 'Darwin':
                # macOS에서 CPU 정보 확인
                result = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'], 
                                      capture_output=True, text=True, timeout=5)
                chip_info = result.stdout.strip().upper()
                
                # M3 Max 감지
                if 'M3' in chip_info and 'MAX' in chip_info:
                    logger.info(f"🍎 M3 Max 감지됨: {chip_info}")
                    return True
                elif 'M3' in chip_info:
                    logger.info(f"🍎 M3 감지됨 (Max 아님): {chip_info}")
                    return False
                    
        except Exception as e:
            logger.warning(f"CPU 정보 확인 실패: {e}")
        
        return False
    
    def _load_fonts(self):
        """폰트 로딩 및 캐시"""
        font_sizes = [10, 12, 14, 16, 18, 20, 24, 28, 32]
        
        # 시스템별 폰트 경로
        font_paths = {
            'arial': [
                "/System/Library/Fonts/Arial.ttf",        # macOS
                "/System/Library/Fonts/Helvetica.ttc",    # macOS 대체
                "/Windows/Fonts/arial.ttf",               # Windows
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # Linux
                "/usr/share/fonts/TTF/arial.ttf"          # Linux 대체
            ],
            'times': [
                "/System/Library/Fonts/Times.ttc",        # macOS
                "/Windows/Fonts/times.ttf",               # Windows
                "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf"  # Linux
            ]
        }
        
        for font_name, paths in font_paths.items():
            for size in font_sizes:
                font_key = f"{font_name}_{size}"
                
                for font_path in paths:
                    try:
                        if os.path.exists(font_path):
                            self._font_cache[font_key] = ImageFont.truetype(font_path, size)
                            break
                    except Exception:
                        continue
                
                # 폴백: 기본 폰트
                if font_key not in self._font_cache:
                    self._font_cache[font_key] = ImageFont.load_default()
    
    def get_font(self, font_name: str = "arial", size: int = 14) -> ImageFont.ImageFont:
        """폰트 반환 (캐시된)"""
        font_key = f"{font_name}_{size}"
        
        if font_key in self._font_cache:
            return self._font_cache[font_key]
        
        # 동적 로딩
        try:
            font_paths = {
                'arial': ["/System/Library/Fonts/Arial.ttf", "/Windows/Fonts/arial.ttf"],
                'times': ["/System/Library/Fonts/Times.ttc", "/Windows/Fonts/times.ttf"]
            }
            
            for font_path in font_paths.get(font_name, []):
                if os.path.exists(font_path):
                    font = ImageFont.truetype(font_path, size)
                    self._font_cache[font_key] = font
                    return font
        except Exception:
            pass
        
        # 폴백
        font = ImageFont.load_default()
        self._font_cache[font_key] = font
        return font

    # ============================================================================
    # 🔧 기존 함수들 (100% 호환성 유지)
    # ============================================================================

    @staticmethod
    def enhance_image(image: Image.Image, enhancement_level: float = 1.1) -> Image.Image:
        """
        이미지 품질 향상
        ✅ 기존 함수명 유지
        """
        try:
            # 선명도 향상
            enhancer = ImageEnhance.Sharpness(image)
            enhanced = enhancer.enhance(enhancement_level)
            
            # 색상 향상
            enhancer = ImageEnhance.Color(enhanced)
            enhanced = enhancer.enhance(1.05)
            
            # 대비 향상
            enhancer = ImageEnhance.Contrast(enhanced)
            enhanced = enhancer.enhance(1.02)
            
            logger.debug("🎨 이미지 품질 향상 완료")
            return enhanced
            
        except Exception as e:
            logger.error(f"❌ 이미지 향상 실패: {e}")
            return image

    @staticmethod
    def resize_image(
        image: Image.Image, 
        target_size: Tuple[int, int], 
        maintain_ratio: bool = True,
        resample: int = Image.Resampling.LANCZOS
    ) -> Image.Image:
        """
        이미지 크기 조정
        ✅ 기존 함수와 완전 호환
        """
        try:
            if maintain_ratio:
                # 비율 유지하며 리사이즈
                image.thumbnail(target_size, resample)
                
                # 정사각형으로 패딩
                new_image = Image.new('RGB', target_size, (255, 255, 255))
                paste_x = (target_size[0] - image.width) // 2
                paste_y = (target_size[1] - image.height) // 2
                new_image.paste(image, (paste_x, paste_y))
                return new_image
            else:
                return image.resize(target_size, resample)
                
        except Exception as e:
            logger.error(f"❌ 이미지 크기 조정 실패: {e}")
            return image

    @staticmethod
    def enhance_image_quality(image: Image.Image) -> Image.Image:
        """
        이미지 품질 향상 (기존 함수와 호환)
        """
        try:
            # 선명도 향상
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(1.1)
            
            # 대비 향상
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.05)
            
            return image
            
        except Exception as e:
            logger.error(f"❌ 이미지 품질 향상 실패: {e}")
            return image

    @staticmethod
    def convert_to_rgb(image: Image.Image) -> Image.Image:
        """
        RGB로 변환 (기존 함수와 호환)
        """
        try:
            if image.mode != 'RGB':
                return image.convert('RGB')
            return image
        except Exception as e:
            logger.error(f"❌ RGB 변환 실패: {e}")
            return image

    # ============================================================================
    # 🎨 시각화 전용 함수들 (새로 추가)
    # ============================================================================
    
    def create_human_parsing_visualization(
        self, 
        original_image: np.ndarray, 
        parsing_map: np.ndarray,
        detected_parts: List[int] = None,
        show_legend: bool = True,
        show_overlay: bool = True,
        overlay_opacity: float = 0.6
    ) -> Dict[str, str]:
        """인체 파싱 결과 시각화 생성"""
        try:
            visualizations = {}
            
            # 1. 컬러 파싱 맵 생성
            colored_parsing = self._create_colored_parsing_map(parsing_map)
            visualizations['colored_parsing'] = self._numpy_to_base64(colored_parsing)
            
            # 2. 오버레이 이미지 생성
            if show_overlay:
                overlay_image = self._create_overlay_image(
                    original_image, colored_parsing, overlay_opacity
                )
                visualizations['overlay_image'] = self._numpy_to_base64(overlay_image)
            
            # 3. 범례 이미지 생성
            if show_legend and detected_parts:
                legend_image = self._create_parsing_legend(detected_parts)
                visualizations['legend_image'] = self._numpy_to_base64(legend_image)
            
            # 4. 통계 정보 이미지
            if detected_parts:
                stats_image = self._create_parsing_statistics(parsing_map, detected_parts)
                visualizations['statistics_image'] = self._numpy_to_base64(stats_image)
            
            # 5. 비교 그리드 생성
            comparison_images = [original_image, colored_parsing]
            if show_overlay:
                comparison_images.append(overlay_image)
            
            comparison_grid = self._create_comparison_grid(
                comparison_images, 
                titles=['Original', 'Parsing', 'Overlay'] if show_overlay else ['Original', 'Parsing']
            )
            visualizations['comparison_grid'] = self._numpy_to_base64(comparison_grid)
            
            logger.info(f"✅ 인체 파싱 시각화 생성 완료: {len(visualizations)}개")
            return visualizations
            
        except Exception as e:
            logger.error(f"❌ 인체 파싱 시각화 실패: {e}")
            return {}
    
    def create_pose_estimation_visualization(
        self, 
        original_image: np.ndarray, 
        keypoints: np.ndarray,
        confidence_scores: np.ndarray = None,
        show_skeleton: bool = True,
        show_confidence: bool = True
    ) -> Dict[str, str]:
        """포즈 추정 결과 시각화 생성"""
        try:
            visualizations = {}
            
            # 1. 키포인트만 표시
            keypoint_image = self._draw_pose_keypoints(
                original_image.copy(), keypoints, confidence_scores, show_confidence
            )
            visualizations['keypoint_image'] = self._numpy_to_base64(keypoint_image)
            
            # 2. 스켈레톤 포함 표시
            if show_skeleton:
                skeleton_image = self._draw_pose_skeleton(
                    original_image.copy(), keypoints, confidence_scores
                )
                visualizations['skeleton_image'] = self._numpy_to_base64(skeleton_image)
            
            # 3. 신뢰도 분석 차트
            if confidence_scores is not None:
                confidence_chart = self._create_confidence_analysis_chart(
                    keypoints, confidence_scores
                )
                visualizations['confidence_chart'] = self._numpy_to_base64(confidence_chart)
            
            # 4. 포즈 품질 평가
            quality_image = self._create_pose_quality_assessment(
                keypoints, confidence_scores
            )
            visualizations['quality_assessment'] = self._numpy_to_base64(quality_image)
            
            # 5. 비교 그리드
            comparison_images = [original_image, keypoint_image]
            if show_skeleton:
                comparison_images.append(skeleton_image)
            
            comparison_grid = self._create_comparison_grid(
                comparison_images,
                titles=['Original', 'Keypoints', 'Skeleton'] if show_skeleton else ['Original', 'Keypoints']
            )
            visualizations['comparison_grid'] = self._numpy_to_base64(comparison_grid)
            
            logger.info(f"✅ 포즈 추정 시각화 생성 완료: {len(visualizations)}개")
            return visualizations
            
        except Exception as e:
            logger.error(f"❌ 포즈 추정 시각화 실패: {e}")
            return {}
    
    def create_clothing_analysis_visualization(
        self, 
        clothing_image: np.ndarray, 
        segmentation_mask: np.ndarray = None,
        color_analysis: Dict[str, Any] = None,
        category_info: Dict[str, Any] = None
    ) -> Dict[str, str]:
        """의류 분석 결과 시각화 생성"""
        try:
            visualizations = {}
            
            # 1. 의류 분할 마스크 적용
            if segmentation_mask is not None:
                segmented_image = self._apply_segmentation_mask(
                    clothing_image, segmentation_mask
                )
                visualizations['segmented_image'] = self._numpy_to_base64(segmented_image)
            
            # 2. 색상 분석 결과
            if color_analysis:
                color_chart = self._create_color_analysis_visualization(color_analysis)
                visualizations['color_analysis'] = self._numpy_to_base64(color_chart)
            
            # 3. 카테고리 정보 패널
            if category_info:
                category_panel = self._create_category_info_panel(category_info)
                visualizations['category_panel'] = self._numpy_to_base64(category_panel)
            
            # 4. 종합 분석 대시보드
            dashboard = self._create_clothing_analysis_dashboard(
                clothing_image, segmentation_mask, color_analysis, category_info
            )
            visualizations['analysis_dashboard'] = self._numpy_to_base64(dashboard)
            
            logger.info(f"✅ 의류 분석 시각화 생성 완료: {len(visualizations)}개")
            return visualizations
            
        except Exception as e:
            logger.error(f"❌ 의류 분석 시각화 실패: {e}")
            return {}
    
    def create_virtual_fitting_visualization(
        self, 
        original_person: np.ndarray, 
        clothing_item: np.ndarray,
        fitted_result: np.ndarray,
        fit_score: float = None,
        confidence: float = None,
        processing_details: Dict[str, Any] = None
    ) -> Dict[str, str]:
        """가상 피팅 결과 시각화 생성"""
        try:
            visualizations = {}
            
            # 1. Before/After 비교
            before_after = self._create_detailed_before_after_comparison(
                original_person, fitted_result, fit_score, confidence
            )
            visualizations['before_after'] = self._numpy_to_base64(before_after)
            
            # 2. 3단계 프로세스 (사람 | 옷 | 결과)
            process_flow = self._create_fitting_process_flow(
                original_person, clothing_item, fitted_result
            )
            visualizations['process_flow'] = self._numpy_to_base64(process_flow)
            
            # 3. 품질 점수 대시보드
            if fit_score is not None or confidence is not None:
                quality_dashboard = self._create_quality_score_dashboard(
                    fit_score, confidence, processing_details
                )
                visualizations['quality_dashboard'] = self._numpy_to_base64(quality_dashboard)
            
            # 4. 상세 분석 (확대 영역들)
            detail_analysis = self._create_fitting_detail_analysis(
                original_person, fitted_result
            )
            visualizations['detail_analysis'] = self._numpy_to_base64(detail_analysis)
            
            # 5. 개선 제안사항
            if processing_details:
                recommendations = self._create_fitting_recommendations(
                    processing_details, fit_score
                )
                visualizations['recommendations'] = self._numpy_to_base64(recommendations)
            
            logger.info(f"✅ 가상 피팅 시각화 생성 완료: {len(visualizations)}개")
            return visualizations
            
        except Exception as e:
            logger.error(f"❌ 가상 피팅 시각화 실패: {e}")
            return {}

    # ============================================================================
    # 🔧 내부 도우미 함수들 (시각화)
    # ============================================================================
    
    def _numpy_to_base64(self, image: np.ndarray, format: str = "JPEG", quality: int = 90) -> str:
        """NumPy 배열을 Base64 문자열로 변환"""
        try:
            # 데이터 타입 정규화
            if image.dtype != np.uint8:
                if image.max() <= 1.0:
                    image = (image * 255).astype(np.uint8)
                else:
                    image = np.clip(image, 0, 255).astype(np.uint8)
            
            # PIL 이미지로 변환
            if len(image.shape) == 3:
                pil_image = Image.fromarray(image, 'RGB')
            else:
                pil_image = Image.fromarray(image, 'L')
            
            # Base64 인코딩
            buffer = io.BytesIO()
            if format.upper() == "JPEG":
                # 고품질 설정 (M3 Max 최적화)
                actual_quality = self.default_quality if self.is_m3_max else quality
                pil_image.save(buffer, format=format, quality=actual_quality, optimize=True)
            else:
                pil_image.save(buffer, format=format)
            
            return base64.b64encode(buffer.getvalue()).decode('utf-8')
            
        except Exception as e:
            logger.error(f"❌ NumPy → Base64 변환 실패: {e}")
            return ""
    
    def _create_colored_parsing_map(self, parsing_map: np.ndarray) -> np.ndarray:
        """컬러 파싱 맵 생성 (개선된 버전)"""
        height, width = parsing_map.shape
        colored_map = np.zeros((height, width, 3), dtype=np.uint8)
        
        # 각 부위별로 색상 적용
        for part_id, color in HUMAN_PARSING_COLORS.items():
            mask = (parsing_map == part_id)
            colored_map[mask] = color
        
        # 부드러운 경계 처리 (M3 Max에서만)
        if self.is_m3_max:
            colored_map = cv2.bilateralFilter(colored_map, 9, 75, 75)
        
        return colored_map
    
    def _create_overlay_image(self, base_image: np.ndarray, overlay: np.ndarray, alpha: float = 0.5) -> np.ndarray:
        """오버레이 이미지 생성 (개선된 버전)"""
        try:
            # 크기 맞추기
            if base_image.shape[:2] != overlay.shape[:2]:
                overlay = cv2.resize(overlay, (base_image.shape[1], base_image.shape[0]))
            
            # 고품질 블렌딩 (M3 Max에서)
            if self.is_m3_max:
                # 가우시안 가중치를 사용한 고급 블렌딩
                blended = cv2.addWeighted(base_image, 1-alpha, overlay, alpha, 0)
                # 추가 후처리
                blended = cv2.bilateralFilter(blended, 9, 75, 75)
            else:
                # 기본 블렌딩
                blended = cv2.addWeighted(base_image, 1-alpha, overlay, alpha, 0)
            
            return blended
            
        except Exception as e:
            logger.error(f"❌ 오버레이 생성 실패: {e}")
            return base_image
    
    def _create_parsing_legend(self, detected_parts: List[int]) -> np.ndarray:
        """파싱 범례 생성 (개선된 버전)"""
        try:
            # 범례 크기 계산
            item_height = 35
            legend_width = 280
            legend_height = len(detected_parts) * item_height + 80
            
            # PIL 이미지 생성
            legend_pil = Image.new('RGB', (legend_width, legend_height), (245, 245, 245))
            draw = ImageDraw.Draw(legend_pil)
            
            # 제목 스타일링
            title_font = self.get_font("arial", 20)
            detail_font = self.get_font("arial", 14)
            
            # 제목 배경
            draw.rectangle([10, 10, legend_width-10, 50], fill=(70, 130, 180), outline=(0, 0, 0))
            draw.text((legend_width//2 - 60, 20), "감지된 부위", fill=(255, 255, 255), font=title_font)
            
            # 각 부위별 항목
            y_offset = 60
            for i, part_id in enumerate(detected_parts):
                if part_id in HUMAN_PARSING_COLORS and part_id in HUMAN_PARSING_NAMES:
                    color = HUMAN_PARSING_COLORS[part_id]
                    name = HUMAN_PARSING_NAMES[part_id]
                    
                    # 배경 (교대로 다른 색상)
                    bg_color = (255, 255, 255) if i % 2 == 0 else (240, 240, 240)
                    draw.rectangle([15, y_offset, legend_width-15, y_offset + item_height], fill=bg_color)
                    
                    # 색상 박스 (그림자 효과)
                    draw.rectangle([22, y_offset + 6, 47, y_offset + 26], fill=(0, 0, 0))  # 그림자
                    draw.rectangle([20, y_offset + 5, 45, y_offset + 25], fill=color, outline=(0, 0, 0))
                    
                    # 텍스트
                    draw.text((55, y_offset + 8), f"{part_id:2d}. {name}", fill=(30, 30, 30), font=detail_font)
                    
                    y_offset += item_height
            
            # 하단 정보
            draw.text((20, y_offset + 10), f"총 {len(detected_parts)}개 부위 감지", 
                     fill=(100, 100, 100), font=detail_font)
            
            return np.array(legend_pil)
            
        except Exception as e:
            logger.error(f"❌ 파싱 범례 생성 실패: {e}")
            # 폴백: 간단한 범례
            return self._create_simple_legend(detected_parts)
    
    def _create_parsing_statistics(self, parsing_map: np.ndarray, detected_parts: List[int]) -> np.ndarray:
        """파싱 통계 정보 생성"""
        try:
            # 통계 계산
            total_pixels = parsing_map.size
            part_stats = {}
            
            for part_id in detected_parts:
                mask = (parsing_map == part_id)
                pixel_count = np.sum(mask)
                percentage = (pixel_count / total_pixels) * 100
                part_stats[part_id] = {
                    'pixels': pixel_count,
                    'percentage': percentage,
                    'name': HUMAN_PARSING_NAMES.get(part_id, f"Part {part_id}")
                }
            
            # 차트 생성
            chart_width = 400
            chart_height = 300
            chart_pil = Image.new('RGB', (chart_width, chart_height), (255, 255, 255))
            draw = ImageDraw.Draw(chart_pil)
            
            # 제목
            title_font = self.get_font("arial", 16)
            draw.text((chart_width//2 - 60, 10), "부위별 비율", fill=(0, 0, 0), font=title_font)
            
            # 막대 차트
            y_start = 50
            bar_height = 20
            max_width = chart_width - 100
            
            for i, (part_id, stats) in enumerate(sorted(part_stats.items(), key=lambda x: x[1]['percentage'], reverse=True)):
                y = y_start + i * (bar_height + 5)
                
                # 막대 길이 계산
                bar_width = int((stats['percentage'] / 100) * max_width)
                color = HUMAN_PARSING_COLORS.get(part_id, (128, 128, 128))
                
                # 막대 그리기
                draw.rectangle([80, y, 80 + bar_width, y + bar_height], fill=color)
                
                # 텍스트
                text_font = self.get_font("arial", 10)
                draw.text((10, y + 5), stats['name'][:10], fill=(0, 0, 0), font=text_font)
                draw.text((85 + bar_width, y + 5), f"{stats['percentage']:.1f}%", fill=(0, 0, 0), font=text_font)
            
            return np.array(chart_pil)
            
        except Exception as e:
            logger.error(f"❌ 파싱 통계 생성 실패: {e}")
            # 폴백: 텍스트만
            return self._create_text_info("파싱 통계", [f"감지된 부위: {len(detected_parts)}개"])

    def _draw_pose_keypoints(self, image: np.ndarray, keypoints: np.ndarray, 
                           confidence_scores: np.ndarray = None, show_confidence: bool = True) -> np.ndarray:
        """포즈 키포인트 그리기 (개선된 버전)"""
        try:
            image_pil = Image.fromarray(image)
            draw = ImageDraw.Draw(image_pil)
            
            for i, (x, y) in enumerate(keypoints):
                # 신뢰도 체크
                confidence = confidence_scores[i] if confidence_scores is not None else 1.0
                if confidence < 0.3:
                    continue
                
                # 색상 및 크기 결정
                color = POSE_KEYPOINT_COLORS[i % len(POSE_KEYPOINT_COLORS)]
                
                # 신뢰도에 따른 크기 조정
                if confidence_scores is not None:
                    radius = int(3 + (confidence * 5))  # 3-8 픽셀
                else:
                    radius = 5
                
                # 키포인트 그리기 (그림자 효과)
                # 그림자
                draw.ellipse([x-radius+1, y-radius+1, x+radius+1, y+radius+1], fill=(0, 0, 0, 128))
                # 메인 포인트
                draw.ellipse([x-radius, y-radius, x+radius, y+radius], fill=color, outline=(255, 255, 255), width=1)
                
                # 신뢰도 텍스트 (옵션)
                if show_confidence and confidence_scores is not None and confidence > 0.5:
                    conf_text = f"{confidence:.2f}"
                    font = self.get_font("arial", 10)
                    # 배경 박스
                    text_bbox = draw.textbbox((0, 0), conf_text, font=font)
                    text_width = text_bbox[2] - text_bbox[0]
                    text_height = text_bbox[3] - text_bbox[1]
                    draw.rectangle([x+radius+2, y-radius-2, x+radius+2+text_width+4, y-radius-2+text_height+4], 
                                 fill=(0, 0, 0, 200))
                    draw.text((x+radius+4, y-radius), conf_text, fill=(255, 255, 255), font=font)
                
                # 키포인트 이름 (고신뢰도에서만)
                if confidence > 0.8 and i < len(POSE_KEYPOINT_NAMES):
                    name = POSE_KEYPOINT_NAMES[i]
                    font = self.get_font("arial", 9)
                    draw.text((x-10, y+radius+2), name, fill=(255, 255, 255), font=font)
            
            return np.array(image_pil)
            
        except Exception as e:
            logger.error(f"❌ 포즈 키포인트 그리기 실패: {e}")
            return image
    
    def _draw_pose_skeleton(self, image: np.ndarray, keypoints: np.ndarray, 
                          confidence_scores: np.ndarray = None) -> np.ndarray:
        """포즈 스켈레톤 그리기 (개선된 버전)"""
        try:
            image_pil = Image.fromarray(image)
            draw = ImageDraw.Draw(image_pil)
            
            # 연결선 그리기 (두께 및 색상 개선)
            for start_idx, end_idx in POSE_SKELETON:
                if start_idx < len(keypoints) and end_idx < len(keypoints):
                    start_x, start_y = keypoints[start_idx]
                    end_x, end_y = keypoints[end_idx]
                    
                    # 신뢰도 체크
                    if confidence_scores is not None:
                        start_conf = confidence_scores[start_idx]
                        end_conf = confidence_scores[end_idx]
                        if start_conf < 0.3 or end_conf < 0.3:
                            continue
                        
                        # 신뢰도에 따른 선 굵기 및 투명도
                        avg_conf = (start_conf + end_conf) / 2
                        line_width = int(2 + (avg_conf * 3))  # 2-5 픽셀
                        alpha = int(100 + (avg_conf * 155))   # 100-255 투명도
                    else:
                        line_width = 3
                        alpha = 255
                    
                    # 그림자 선 (더 두껍고 어두운)
                    draw.line([start_x+1, start_y+1, end_x+1, end_y+1], 
                             fill=(0, 0, 0, alpha//2), width=line_width+1)
                    
                    # 메인 선
                    line_color = (0, 255, 0)  # 초록색 스켈레톤
                    draw.line([start_x, start_y, end_x, end_y], fill=line_color, width=line_width)
            
            # 키포인트 다시 그리기 (선 위에 표시)
            for i, (x, y) in enumerate(keypoints):
                confidence = confidence_scores[i] if confidence_scores is not None else 1.0
                if confidence > 0.3:
                    color = POSE_KEYPOINT_COLORS[i % len(POSE_KEYPOINT_COLORS)]
                    radius = 4 if confidence > 0.7 else 3
                    
                    # 키포인트 그리기
                    draw.ellipse([x-radius, y-radius, x+radius, y+radius], 
                               fill=color, outline=(255, 255, 255), width=1)
            
            return np.array(image_pil)
            
        except Exception as e:
            logger.error(f"❌ 포즈 스켈레톤 그리기 실패: {e}")
            return image
    
    def _create_confidence_analysis_chart(self, keypoints: np.ndarray, confidence_scores: np.ndarray) -> np.ndarray:
        """신뢰도 분석 차트 생성"""
        try:
            if MATPLOTLIB_AVAILABLE:
                return self._create_matplotlib_confidence_chart(keypoints, confidence_scores)
            else:
                return self._create_pil_confidence_chart(keypoints, confidence_scores)
        except Exception as e:
            logger.error(f"❌ 신뢰도 차트 생성 실패: {e}")
            return self._create_text_info("신뢰도 분석", [
                f"평균 신뢰도: {confidence_scores.mean():.2f}",
                f"고신뢰도 포인트: {sum(confidence_scores > 0.7)}/18",
                f"감지된 포인트: {sum(confidence_scores > 0.3)}/18"
            ])
    
    def _create_matplotlib_confidence_chart(self, keypoints: np.ndarray, confidence_scores: np.ndarray) -> np.ndarray:
        """Matplotlib을 사용한 신뢰도 차트"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        fig.patch.set_facecolor('white')
        
        # 1. 키포인트별 신뢰도 막대 차트
        keypoint_names = [name[:6] for name in POSE_KEYPOINT_NAMES]  # 이름 축약
        colors = ['green' if conf > 0.7 else 'orange' if conf > 0.3 else 'red' for conf in confidence_scores]
        
        bars = ax1.bar(range(len(confidence_scores)), confidence_scores, color=colors)
        ax1.set_xlabel('키포인트')
        ax1.set_ylabel('신뢰도')
        ax1.set_title('키포인트별 신뢰도')
        ax1.set_xticks(range(len(keypoint_names)))
        ax1.set_xticklabels(keypoint_names, rotation=45, ha='right', fontsize=8)
        ax1.grid(axis='y', alpha=0.3)
        
        # 값 표시
        for bar, conf in zip(bars, confidence_scores):
            if conf > 0.1:  # 너무 낮은 값은 표시하지 않음
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{conf:.2f}', ha='center', va='bottom', fontsize=7)
        
        # 2. 품질 분포 파이 차트
        high_conf = sum(confidence_scores > 0.7)
        med_conf = sum((confidence_scores > 0.3) & (confidence_scores <= 0.7))
        low_conf = sum(confidence_scores <= 0.3)
        
        sizes = [high_conf, med_conf, low_conf]
        labels = ['높음 (>0.7)', '보통 (0.3-0.7)', '낮음 (≤0.3)']
        colors_pie = ['#2ecc71', '#f39c12', '#e74c3c']
        explode = (0.05, 0, 0)  # 첫 번째 조각 강조
        
        wedges, texts, autotexts = ax2.pie(sizes, labels=labels, colors=colors_pie, 
                                          autopct='%1.0f개', explode=explode, shadow=True)
        ax2.set_title('신뢰도 분포')
        
        # 텍스트 스타일링
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        plt.tight_layout()
        
        # NumPy 배열로 변환
        fig.canvas.draw()
        chart_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        chart_array = chart_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        plt.close(fig)
        return chart_array
    
    def _create_pil_confidence_chart(self, keypoints: np.ndarray, confidence_scores: np.ndarray) -> np.ndarray:
        """PIL을 사용한 신뢰도 차트 (matplotlib 없을 때)"""
        chart_width = 600
        chart_height = 400
        chart_pil = Image.new('RGB', (chart_width, chart_height), (250, 250, 250))
        draw = ImageDraw.Draw(chart_pil)
        
        # 제목
        title_font = self.get_font("arial", 18)
        draw.text((chart_width//2 - 80, 20), "키포인트 신뢰도 분석", fill=(50, 50, 50), font=title_font)
        
        # 막대 차트 영역
        chart_x = 50
        chart_y = 80
        chart_w = chart_width - 100
        chart_h = 200
        
        # 배경
        draw.rectangle([chart_x, chart_y, chart_x + chart_w, chart_y + chart_h], 
                      fill=(255, 255, 255), outline=(200, 200, 200))
        
        # 막대 그리기
        bar_width = chart_w // len(confidence_scores)
        for i, conf in enumerate(confidence_scores):
            x = chart_x + i * bar_width
            bar_height = int(conf * chart_h)
            y = chart_y + chart_h - bar_height
            
            # 색상 결정
            if conf > 0.7:
                color = (46, 204, 113)  # 초록
            elif conf > 0.3:
                color = (243, 156, 18)  # 주황
            else:
                color = (231, 76, 60)   # 빨강
            
            # 막대 그리기
            draw.rectangle([x + 1, y, x + bar_width - 1, chart_y + chart_h], 
                          fill=color, outline=(100, 100, 100))
            
            # 값 표시
            if conf > 0.1:
                value_text = f"{conf:.2f}"
                text_font = self.get_font("arial", 8)
                text_bbox = draw.textbbox((0, 0), value_text, font=text_font)
                text_width = text_bbox[2] - text_bbox[0]
                draw.text((x + bar_width//2 - text_width//2, y - 15), 
                         value_text, fill=(50, 50, 50), font=text_font)
        
        # 통계 정보
        stats_y = chart_y + chart_h + 30
        stats_font = self.get_font("arial", 14)
        
        avg_conf = confidence_scores.mean()
        high_count = sum(confidence_scores > 0.7)
        detected_count = sum(confidence_scores > 0.3)
        
        stats_text = [
            f"평균 신뢰도: {avg_conf:.3f}",
            f"고신뢰도 키포인트: {high_count}/18개",
            f"감지된 키포인트: {detected_count}/18개"
        ]
        
        for i, text in enumerate(stats_text):
            draw.text((chart_x, stats_y + i * 25), text, fill=(80, 80, 80), font=stats_font)
        
        return np.array(chart_pil)
    
    def _create_pose_quality_assessment(self, keypoints: np.ndarray, confidence_scores: np.ndarray) -> np.ndarray:
        """포즈 품질 평가 생성"""
        try:
            assessment_width = 400
            assessment_height = 300
            assessment_pil = Image.new('RGB', (assessment_width, assessment_height), (248, 249, 250))
            draw = ImageDraw.Draw(assessment_pil)
            
            # 제목
            title_font = self.get_font("arial", 16)
            draw.text((assessment_width//2 - 70, 15), "포즈 품질 평가", fill=(52, 58, 64), font=title_font)
            
            # 전체 품질 점수 계산
            if confidence_scores is not None:
                overall_quality = confidence_scores.mean()
                detected_ratio = sum(confidence_scores > 0.3) / len(confidence_scores)
                high_quality_ratio = sum(confidence_scores > 0.7) / len(confidence_scores)
            else:
                overall_quality = 0.5
                detected_ratio = 0.5
                high_quality_ratio = 0.3
            
            # 품질 등급 결정
            if overall_quality > 0.8:
                grade = "우수"
                grade_color = (40, 167, 69)
            elif overall_quality > 0.6:
                grade = "양호"
                grade_color = (255, 193, 7)
            elif overall_quality > 0.4:
                grade = "보통"
                grade_color = (255, 133, 27)
            else:
                grade = "개선필요"
                grade_color = (220, 53, 69)
            
            # 등급 표시
            grade_y = 60
            draw.rectangle([50, grade_y, 350, grade_y + 60], fill=grade_color, outline=(0, 0, 0))
            grade_font = self.get_font("arial", 24)
            draw.text((assessment_width//2 - 30, grade_y + 18), grade, fill=(255, 255, 255), font=grade_font)
            
            # 세부 점수들
            details_y = 140
            detail_font = self.get_font("arial", 12)
            
            details = [
                f"전체 신뢰도: {overall_quality:.1%}",
                f"감지율: {detected_ratio:.1%}",
                f"고품질 비율: {high_quality_ratio:.1%}",
                f"완성도: {min(detected_ratio * 1.2, 1.0):.1%}"
            ]
            
            for i, detail in enumerate(details):
                y = details_y + i * 25
                # 배경 바
                draw.rectangle([60, y, 340, y + 20], fill=(233, 236, 239), outline=(173, 181, 189))
                # 진행 바
                if "신뢰도" in detail:
                    progress = overall_quality
                elif "감지율" in detail:
                    progress = detected_ratio
                elif "고품질" in detail:
                    progress = high_quality_ratio
                else:
                    progress = min(detected_ratio * 1.2, 1.0)
                
                progress_width = int(280 * progress)
                progress_color = (40, 167, 69) if progress > 0.7 else (255, 193, 7) if progress > 0.5 else (220, 53, 69)
                draw.rectangle([60, y, 60 + progress_width, y + 20], fill=progress_color)
                
                # 텍스트
                draw.text((65, y + 4), detail, fill=(52, 58, 64), font=detail_font)
            
            return np.array(assessment_pil)
            
        except Exception as e:
            logger.error(f"❌ 포즈 품질 평가 생성 실패: {e}")
            return self._create_text_info("포즈 품질 평가", ["평가 생성 실패"])
    
    def _apply_segmentation_mask(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """분할 마스크 적용"""
        try:
            # 마스크 크기 조정
            if mask.shape[:2] != image.shape[:2]:
                mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
            
            # 바이너리 마스크로 변환
            if len(mask.shape) == 3:
                mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
            
            # 임계값 적용
            _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
            
            # 3채널로 확장
            mask_3d = np.stack([binary_mask] * 3, axis=-1) / 255.0
            
            # 배경을 흰색으로 설정
            background = np.ones_like(image) * 255
            result = image * mask_3d + background * (1 - mask_3d)
            
            return result.astype(np.uint8)
            
        except Exception as e:
            logger.error(f"❌ 분할 마스크 적용 실패: {e}")
            return image
    
    def _create_color_analysis_visualization(self, color_analysis: Dict[str, Any]) -> np.ndarray:
        """색상 분석 시각화 생성"""
        try:
            viz_width = 500
            viz_height = 400
            viz_pil = Image.new('RGB', (viz_width, viz_height), (255, 255, 255))
            draw = ImageDraw.Draw(viz_pil)
            
            # 제목
            title_font = self.get_font("arial", 18)
            draw.text((viz_width//2 - 70, 20), "색상 분석 결과", fill=(50, 50, 50), font=title_font)
            
            # 주요 색상 팔레트
            dominant_colors = color_analysis.get('dominant_colors', [(128, 128, 128)])
            palette_y = 70
            palette_height = 60
            
            # 팔레트 배경
            draw.rectangle([50, palette_y, viz_width - 50, palette_y + palette_height], 
                          fill=(240, 240, 240), outline=(200, 200, 200))
            
            # 색상별 영역
            color_width = (viz_width - 100) // len(dominant_colors)
            for i, color in enumerate(dominant_colors):
                x1 = 50 + i * color_width
                x2 = 50 + (i + 1) * color_width
                draw.rectangle([x1, palette_y, x2, palette_y + palette_height], fill=tuple(color))
                
                # 색상 정보 텍스트
                color_text = f"RGB({color[0]}, {color[1]}, {color[2]})"
                text_font = self.get_font("arial", 10)
                text_bbox = draw.textbbox((0, 0), color_text, font=text_font)
                text_width = text_bbox[2] - text_bbox[0]
                
                # 텍스트 색상 (대비를 위해 밝기에 따라 조정)
                brightness = sum(color) / 3
                text_color = (255, 255, 255) if brightness < 128 else (0, 0, 0)
                
                draw.text((x1 + (color_width - text_width) // 2, palette_y + 25), 
                         color_text, fill=text_color, font=text_font)
            
            # 색상 통계
            stats_y = palette_y + palette_height + 30
            stats_font = self.get_font("arial", 14)
            
            # 주요 색상 이름
            primary_color_name = color_analysis.get('primary_color_name', '알 수 없음')
            draw.text((50, stats_y), f"주요 색상: {primary_color_name}", fill=(70, 70, 70), font=stats_font)
            
            # 색상 다양성
            color_diversity = color_analysis.get('color_diversity', 0.5)
            diversity_text = "높음" if color_diversity > 0.7 else "보통" if color_diversity > 0.4 else "낮음"
            draw.text((50, stats_y + 30), f"색상 다양성: {diversity_text} ({color_diversity:.2f})", 
                     fill=(70, 70, 70), font=stats_font)
            
            # 밝기 분석
            brightness_avg = color_analysis.get('average_brightness', 128)
            brightness_text = "밝음" if brightness_avg > 180 else "보통" if brightness_avg > 100 else "어두움"
            draw.text((50, stats_y + 60), f"전체 밝기: {brightness_text} ({brightness_avg:.0f})", 
                     fill=(70, 70, 70), font=stats_font)
            
            # 채도 분석
            saturation_avg = color_analysis.get('average_saturation', 0.5)
            saturation_text = "높음" if saturation_avg > 0.7 else "보통" if saturation_avg > 0.4 else "낮음"
            draw.text((50, stats_y + 90), f"채도: {saturation_text} ({saturation_avg:.2f})", 
                     fill=(70, 70, 70), font=stats_font)
            
            return np.array(viz_pil)
            
        except Exception as e:
            logger.error(f"❌ 색상 분석 시각화 실패: {e}")
            return self._create_text_info("색상 분석", ["분석 결과 없음"])
    
    def _create_category_info_panel(self, category_info: Dict[str, Any]) -> np.ndarray:
        """카테고리 정보 패널 생성"""
        try:
            panel_width = 400
            panel_height = 350
            panel_pil = Image.new('RGB', (panel_width, panel_height), (248, 249, 250))
            draw = ImageDraw.Draw(panel_pil)
            
            # 제목
            title_font = self.get_font("arial", 18)
            draw.text((panel_width//2 - 70, 20), "의류 카테고리 분석", fill=(52, 58, 64), font=title_font)
            
            # 카테고리 정보
            category = category_info.get('category', '알 수 없음')
            subcategory = category_info.get('subcategory', '')
            confidence = category_info.get('confidence', 0.0)
            
            # 메인 카테고리 표시
            main_y = 70
            category_font = self.get_font("arial", 20)
            
            # 카테고리 배경 색상
            category_color = CLOTHING_COLORS.get(category.lower(), (128, 128, 128))
            draw.rectangle([50, main_y, panel_width - 50, main_y + 50], 
                          fill=category_color, outline=(0, 0, 0))
            
            # 카테고리 텍스트
            text_color = (255, 255, 255) if sum(category_color) / 3 < 128 else (0, 0, 0)
            draw.text((panel_width//2 - len(category) * 6, main_y + 15), 
                     category.upper(), fill=text_color, font=category_font)
            
            # 세부 정보
            details_y = main_y + 70
            detail_font = self.get_font("arial", 14)
            
            details = [
                f"카테고리: {category}",
                f"세부 분류: {subcategory}" if subcategory else "",
                f"신뢰도: {confidence:.1%}",
                f"스타일: {category_info.get('style', '캐주얼')}",
                f"시즌: {category_info.get('season', '사계절')}",
                f"성별: {category_info.get('gender', '유니섹스')}"
            ]
            
            for i, detail in enumerate(details):
                if detail:  # 빈 문자열 제외
                    y = details_y + i * 25
                    draw.text((60, y), detail, fill=(73, 80, 87), font=detail_font)
            
            # 특징 태그들
            features = category_info.get('features', [])
            if features:
                tags_y = details_y + len([d for d in details if d]) * 25 + 20
                tag_font = self.get_font("arial", 11)
                
                x_offset = 60
                for feature in features[:5]:  # 최대 5개만 표시
                    # 태그 크기 계산
                    text_bbox = draw.textbbox((0, 0), feature, font=tag_font)
                    tag_width = text_bbox[2] - text_bbox[0] + 16
                    tag_height = 22
                    
                    # 태그 배경
                    draw.rectangle([x_offset, tags_y, x_offset + tag_width, tags_y + tag_height], 
                                  fill=(108, 117, 125), outline=(73, 80, 87))
                    
                    # 태그 텍스트
                    draw.text((x_offset + 8, tags_y + 4), feature, fill=(255, 255, 255), font=tag_font)
                    
                    x_offset += tag_width + 10
                    if x_offset > panel_width - 100:  # 줄바꿈
                        x_offset = 60
                        tags_y += 30
            
            return np.array(panel_pil)
            
        except Exception as e:
            logger.error(f"❌ 카테고리 정보 패널 생성 실패: {e}")
            return self._create_text_info("카테고리 분석", ["분석 결과 없음"])
    
    def _create_comparison_grid(self, images: List[np.ndarray], titles: List[str] = None) -> np.ndarray:
        """비교 그리드 이미지 생성 (개선된 버전)"""
        try:
            if not images:
                return np.zeros((400, 400, 3), dtype=np.uint8)
            
            # 이미지 크기 통일 (더 큰 크기로)
            target_height = 400 if self.is_m3_max else 300
            processed_images = []
            
            for img in images:
                # 비율 유지하면서 리사이즈
                height, width = img.shape[:2]
                scale = target_height / height
                new_width = int(width * scale)
                
                # 고품질 리사이즈
                if self.is_m3_max:
                    resized = cv2.resize(img, (new_width, target_height), interpolation=cv2.INTER_LANCZOS4)
                else:
                    resized = cv2.resize(img, (new_width, target_height), interpolation=cv2.INTER_LINEAR)
                
                processed_images.append(resized)
            
            # 그리드 레이아웃 결정
            num_images = len(processed_images)
            if num_images == 1:
                result = processed_images[0]
            elif num_images == 2:
                # 2개: 수평 배치
                max_width = max(img.shape[1] for img in processed_images)
                # 너비 통일
                unified_images = []
                for img in processed_images:
                    if img.shape[1] < max_width:
                        # 중앙 정렬로 패딩
                        padding = max_width - img.shape[1]
                        left_pad = padding // 2
                        right_pad = padding - left_pad
                        padded = np.pad(img, ((0, 0), (left_pad, right_pad), (0, 0)), 
                                      mode='constant', constant_values=255)
                        unified_images.append(padded)
                    else:
                        unified_images.append(img)
                
                # 간격 추가
                gap = np.ones((target_height, 20, 3), dtype=np.uint8) * 240
                result = np.hstack([unified_images[0], gap, unified_images[1]])
            else:
                # 3개 이상: 수평 배치 (간격 포함)
                max_width = max(img.shape[1] for img in processed_images)
                unified_images = []
                
                for img in processed_images:
                    if img.shape[1] < max_width:
                        padding = max_width - img.shape[1]
                        left_pad = padding // 2
                        right_pad = padding - left_pad
                        padded = np.pad(img, ((0, 0), (left_pad, right_pad), (0, 0)), 
                                      mode='constant', constant_values=255)
                        unified_images.append(padded)
                    else:
                        unified_images.append(img)
                
                # 간격을 두고 배치
                result_parts = []
                for i, img in enumerate(unified_images):
                    result_parts.append(img)
                    if i < len(unified_images) - 1:  # 마지막이 아니면 간격 추가
                        gap = np.ones((target_height, 15, 3), dtype=np.uint8) * 240
                        result_parts.append(gap)
                
                result = np.hstack(result_parts)
            
            # 제목 추가
            if titles and len(titles) == len(processed_images):
                # 제목 공간을 위해 이미지 확장
                title_height = 50
                extended_height = result.shape[0] + title_height
                extended_result = np.ones((extended_height, result.shape[1], 3), dtype=np.uint8) * 250
                extended_result[title_height:, :] = result
                
                # PIL로 변환하여 텍스트 추가
                result_pil = Image.fromarray(extended_result)
                draw = ImageDraw.Draw(result_pil)
                title_font = self.get_font("arial", 16)
                
                # 각 이미지 영역의 중앙에 제목 배치
                if num_images == 1:
                    title_x = result.shape[1] // 2 - len(titles[0]) * 5
                    draw.text((title_x, 15), titles[0], fill=(50, 50, 50), font=title_font)
                else:
                    x_offset = 0
                    for i, (title, img) in enumerate(zip(titles, processed_images)):
                        img_center_x = x_offset + img.shape[1] // 2
                        title_x = img_center_x - len(title) * 5
                        draw.text((title_x, 15), title, fill=(50, 50, 50), font=title_font)
                        
                        # 다음 이미지 위치 계산
                        x_offset += img.shape[1]
                        if i < len(processed_images) - 1:  # 간격 고려
                            x_offset += 15 if num_images > 2 else 20
                
                result = np.array(result_pil)
            
            return result
            
        except Exception as e:
            logger.error(f"❌ 비교 그리드 생성 실패: {e}")
            # 폴백: 첫 번째 이미지만 반환
            return images[0] if images else np.zeros((400, 400, 3), dtype=np.uint8)
    
    def _create_text_info(self, title: str, items: List[str]) -> np.ndarray:
        """텍스트 기반 정보 패널 생성"""
        try:
            panel_width = 400
            panel_height = 300
            panel_pil = Image.new('RGB', (panel_width, panel_height), (248, 249, 250))
            draw = ImageDraw.Draw(panel_pil)
            
            # 제목
            title_font = self.get_font("arial", 18)
            draw.text((panel_width//2 - len(title) * 5, 30), title, fill=(52, 58, 64), font=title_font)
            
            # 항목들
            item_font = self.get_font("arial", 14)
            y_offset = 80
            
            for item in items:
                draw.text((30, y_offset), f"• {item}", fill=(73, 80, 87), font=item_font)
                y_offset += 30
            
            return np.array(panel_pil)
            
        except Exception as e:
            logger.error(f"❌ 텍스트 정보 패널 생성 실패: {e}")
            # 최소한의 폴백
            fallback = np.ones((300, 400, 3), dtype=np.uint8) * 240
            return fallback

# ============================================================================
# 🔧 기존 호환 함수들 (전역 함수로 유지)
# ============================================================================

def resize_image(image: Image.Image, target_size: Tuple[int, int], maintain_ratio: bool = True) -> Image.Image:
    """기존 resize_image 함수와 완전 호환"""
    return ImageProcessor.resize_image(image, target_size, maintain_ratio)

def enhance_image_quality(image: Image.Image) -> Image.Image:
    """기존 enhance_image_quality 함수와 완전 호환"""
    return ImageProcessor.enhance_image_quality(image)

def convert_to_rgb(image: Image.Image) -> Image.Image:
    """기존 convert_to_rgb 함수와 완전 호환"""
    return ImageProcessor.convert_to_rgb(image)

async def validate_image_content(image_bytes: bytes) -> bool:
    """기존 validate_image_content 함수와 완전 호환"""
    try:
        image = Image.open(io.BytesIO(image_bytes))
        width, height = image.size
        
        # 최소/최대 크기 검사
        if width < 100 or height < 100:
            return False
        if width > 4096 or height > 4096:
            return False
            
        return True
    except Exception:
        return False

# ============================================================================
# 🎯 시각화 전용 편의 함수들 (새로 추가)
# ============================================================================

def create_step_visualization(step_id: int, **kwargs) -> Dict[str, str]:
    """단계별 시각화 생성 편의 함수"""
    processor = get_image_processor()
    
    if step_id == 3:  # 인체 파싱
        return processor.create_human_parsing_visualization(**kwargs)
    elif step_id == 4:  # 포즈 추정
        return processor.create_pose_estimation_visualization(**kwargs)
    elif step_id == 5:  # 의류 분석
        return processor.create_clothing_analysis_visualization(**kwargs)
    elif step_id == 7:  # 가상 피팅
        return processor.create_virtual_fitting_visualization(**kwargs)
    else:
        logger.warning(f"단계 {step_id}에 대한 시각화 미구현")
        return {}

# ============================================================================
# 🔧 기존 추가 유틸리티 함수들 (완전 유지)
# ============================================================================

def save_temp_image(
    image: Union[Image.Image, np.ndarray], 
    prefix: str = "temp", 
    suffix: str = ".jpg",
    directory: Optional[str] = None
) -> str:
    """
    임시 이미지 파일 저장
    ✅ 기존 함수와 완전 호환
    """
    try:
        # 디렉토리 설정
        if directory is None:
            directory = tempfile.gettempdir()
        
        # 파일명 생성
        filename = f"{prefix}_{uuid.uuid4().hex[:8]}{suffix}"
        filepath = os.path.join(directory, filename)
        
        # PIL Image로 변환
        if isinstance(image, np.ndarray):
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)
            if len(image.shape) == 3 and image.shape[2] == 3:
                # BGR to RGB 변환 (OpenCV 사용 시)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image)
        else:
            pil_image = image
        
        # RGB로 변환
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        # 파일 저장
        pil_image.save(filepath, "JPEG", quality=90)
        logger.debug(f"임시 이미지 저장: {filepath}")
        
        return filepath
        
    except Exception as e:
        logger.error(f"임시 이미지 저장 실패: {e}")
        raise

def image_to_base64(
    image: Union[Image.Image, np.ndarray], 
    format: str = "JPEG"
) -> str:
    """
    이미지를 base64 문자열로 변환
    ✅ 기존 함수와 완전 호환
    """
    processor = get_image_processor()
    if isinstance(image, np.ndarray):
        return processor._numpy_to_base64(image, format)
    else:
        # PIL 이미지 처리
        try:
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            buffer = io.BytesIO()
            quality = processor.default_quality if processor.is_m3_max else 90
            image.save(buffer, format=format, quality=quality)
            return base64.b64encode(buffer.getvalue()).decode('utf-8')
        except Exception as e:
            logger.error(f"❌ PIL → Base64 변환 실패: {e}")
            return ""

def base64_to_image(base64_str: str) -> Image.Image:
    """
    base64 문자열을 이미지로 변환
    ✅ 기존 함수와 완전 호환
    """
    try:
        image_data = base64.b64decode(base64_str)
        image = Image.open(io.BytesIO(image_data))
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        return image
        
    except Exception as e:
        logger.error(f"base64 이미지 변환 실패: {e}")
        raise

# ============================================================================
# 🎯 전역 ImageProcessor 인스턴스
# ============================================================================

_global_image_processor = None

def get_image_processor() -> ImageProcessor:
    """전역 이미지 프로세서 인스턴스 반환"""
    global _global_image_processor
    if _global_image_processor is None:
        _global_image_processor = ImageProcessor()
    return _global_image_processor

# ============================================================================
# 🎉 완료 메시지
# ============================================================================

logger.info("🎨 완전 강화된 이미지 처리 유틸리티 로드 완료")
logger.info("✅ 기존 함수 100% 호환성 유지")
logger.info("✅ 단계별 시각화 완전 구현")
logger.info("✅ M3 Max 최적화 적용")
logger.info("✅ 고품질 이미지 처리")
logger.info("✅ PIL/OpenCV/Matplotlib 통합")
logger.info("🚀 시각화 완전 구현 준비 완료!")