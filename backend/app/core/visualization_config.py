"""
backend/app/core/visualization_config.py - 시각화 설정 및 관리

✅ 단계별 시각화 설정
✅ M3 Max 최적화 파라미터
✅ 색상 및 스타일 관리
✅ 메모리 효율적 설정
✅ 환경별 설정 지원
"""

import logging
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
import os

logger = logging.getLogger(__name__)

# ============================================================================
# 🎨 시각화 품질 및 모드 설정
# ============================================================================

class VisualizationQuality(Enum):
    """시각화 품질 레벨"""
    LOW = "low"          # 빠른 처리, 낮은 품질
    MEDIUM = "medium"    # 균형잡힌 품질
    HIGH = "high"        # 고품질, 느린 처리
    ULTRA = "ultra"      # 최고 품질 (M3 Max 전용)

class VisualizationMode(Enum):
    """시각화 모드"""
    DEVELOPMENT = "development"  # 개발용 (모든 시각화)
    PRODUCTION = "production"    # 프로덕션 (핵심만)
    DEBUG = "debug"             # 디버그용 (상세 정보)
    DEMO = "demo"               # 데모용 (시각적 효과)

# ============================================================================
# 🔧 단계별 시각화 설정 클래스들
# ============================================================================

@dataclass
class BaseVisualizationConfig:
    """기본 시각화 설정"""
    enabled: bool = True
    quality: VisualizationQuality = VisualizationQuality.MEDIUM
    output_format: str = "JPEG"
    jpeg_quality: int = 90
    max_image_size: Tuple[int, int] = (1024, 1024)
    enable_caching: bool = True
    cache_size: int = 50

@dataclass
class HumanParsingVisualizationConfig(BaseVisualizationConfig):
    """인체 파싱 시각화 설정"""
    # 시각화 요소 활성화
    show_colored_parsing: bool = True
    show_overlay: bool = True
    show_legend: bool = True
    show_comparison: bool = True
    show_statistics: bool = False
    
    # 시각적 스타일
    overlay_opacity: float = 0.6
    part_boundary_thickness: int = 2
    enable_smooth_edges: bool = True
    
    # 색상 설정
    use_custom_colors: bool = False
    custom_color_map: Dict[int, Tuple[int, int, int]] = field(default_factory=dict)
    
    # 범례 설정
    legend_position: str = "right"  # right, bottom, overlay
    legend_font_size: int = 12
    show_part_percentages: bool = True

@dataclass
class PoseEstimationVisualizationConfig(BaseVisualizationConfig):
    """포즈