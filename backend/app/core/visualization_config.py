# backend/app/core/visualization_config.py
"""
🎨 MyCloset AI - 시각화 설정 관리 시스템
✅ 단계별 시각화 설정
✅ 성능 최적화 설정  
✅ M3 Max GPU 최적화
✅ 메모리 효율적 시각화
✅ 품질 레벨별 설정
"""

import logging
import os
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)

# ==============================================
# 🎨 시각화 설정 데이터 구조
# ==============================================

class VisualizationQuality(Enum):
    """시각화 품질 레벨"""
    LOW = "low"           # 512x512, JPEG 품질 60
    MEDIUM = "medium"     # 768x768, JPEG 품질 75  
    HIGH = "high"         # 1024x1024, JPEG 품질 85
    ULTRA = "ultra"       # 1536x1536, JPEG 품질 95

class VisualizationFormat(Enum):
    """시각화 출력 형식"""
    JPEG = "JPEG"
    PNG = "PNG"
    WEBP = "WEBP"

class PerformanceMode(Enum):
    """성능 모드"""
    FAST = "fast"         # 속도 우선
    BALANCED = "balanced" # 균형
    QUALITY = "quality"   # 품질 우선

@dataclass
class VisualizationConfig:
    """시각화 설정 클래스"""
    enabled: bool = True
    quality: VisualizationQuality = VisualizationQuality.HIGH
    format: VisualizationFormat = VisualizationFormat.JPEG
    max_size: Tuple[int, int] = (1024, 1024)
    compression_quality: int = 85
    enable_caching: bool = True
    memory_efficient: bool = True
    gpu_accelerated: bool = True
    show_confidence: bool = True
    show_timing: bool = False
    custom_settings: Dict[str, Any] = field(default_factory=dict)

# ==============================================
# 🎯 단계별 시각화 설정
# ==============================================

STEP_VISUALIZATION_CONFIG = {
    # 1단계: 이미지 업로드 검증
    1: VisualizationConfig(
        enabled=True,
        quality=VisualizationQuality.HIGH,
        format=VisualizationFormat.JPEG,
        max_size=(800, 600),
        compression_quality=85,
        custom_settings={
            "show_quality_metrics": True,
            "show_comparison": True,
            "show_recommendations": True,
            "thumbnail_size": (200, 200)
        }
    ),
    
    # 2단계: 신체 측정 검증
    2: VisualizationConfig(
        enabled=True,
        quality=VisualizationQuality.MEDIUM,
        format=VisualizationFormat.PNG,
        max_size=(600, 400),
        compression_quality=75,
        custom_settings={
            "show_bmi_chart": True,
            "show_body_type": True,
            "show_recommendations": True,
            "chart_style": "modern"
        }
    ),
    
    # 3단계: 인간 파싱
    3: VisualizationConfig(
        enabled=True,
        quality=VisualizationQuality.HIGH,
        format=VisualizationFormat.PNG,
        max_size=(1024, 1024),
        compression_quality=90,
        custom_settings={
            "show_segmentation_overlay": True,
            "show_part_labels": True,
            "overlay_opacity": 0.6,
            "color_scheme": "rainbow",
            "show_legend": True
        }
    ),
    
    # 4단계: 포즈 추정
    4: VisualizationConfig(
        enabled=True,
        quality=VisualizationQuality.HIGH,
        format=VisualizationFormat.JPEG,
        max_size=(1024, 1024),
        compression_quality=85,
        custom_settings={
            "show_keypoints": True,
            "show_skeleton": True,
            "show_confidence_scores": True,
            "keypoint_radius": 3,
            "skeleton_thickness": 2,
            "confidence_threshold": 0.5
        }
    ),
    
    # 5단계: 의류 분석
    5: VisualizationConfig(
        enabled=True,
        quality=VisualizationQuality.HIGH,
        format=VisualizationFormat.JPEG,
        max_size=(1024, 1024),
        compression_quality=85,
        custom_settings={
            "show_color_analysis": True,
            "show_segmentation": True,
            "show_material_info": True,
            "color_palette_size": 5,
            "segmentation_alpha": 0.7
        }
    ),
    
    # 6단계: 기하학적 매칭
    6: VisualizationConfig(
        enabled=True,
        quality=VisualizationQuality.MEDIUM,
        format=VisualizationFormat.JPEG,
        max_size=(800, 600),
        compression_quality=80,
        custom_settings={
            "show_matching_points": True,
            "show_transformation": True,
            "show_alignment_grid": False,
            "point_size": 4,
            "connection_thickness": 1,
            "max_points_display": 20
        }
    ),
    
    # 7단계: 가상 피팅
    7: VisualizationConfig(
        enabled=True,
        quality=VisualizationQuality.ULTRA,
        format=VisualizationFormat.JPEG,
        max_size=(1536, 1536),
        compression_quality=95,
        custom_settings={
            "show_before_after": True,
            "show_process_flow": True,
            "show_quality_metrics": True,
            "enable_zoom": True,
            "comparison_layout": "side_by_side",
            "quality_indicators": True
        }
    ),
    
    # 8단계: 품질 평가
    8: VisualizationConfig(
        enabled=True,
        quality=VisualizationQuality.MEDIUM,
        format=VisualizationFormat.PNG,
        max_size=(800, 600),
        compression_quality=80,
        custom_settings={
            "show_score_breakdown": True,
            "show_improvement_suggestions": True,
            "show_comparison_charts": True,
            "chart_type": "radar",
            "score_visualization": "gauge"
        }
    )
}

# ==============================================
# 🚀 성능 모드별 설정
# ==============================================

PERFORMANCE_MODE_SETTINGS = {
    PerformanceMode.FAST: {
        "default_quality": VisualizationQuality.LOW,
        "default_format": VisualizationFormat.JPEG,
        "compression_quality": 60,
        "max_concurrent_visualizations": 1,
        "enable_gpu_acceleration": False,
        "cache_size_mb": 50,
        "processing_timeout": 5.0
    },
    
    PerformanceMode.BALANCED: {
        "default_quality": VisualizationQuality.MEDIUM,
        "default_format": VisualizationFormat.JPEG,
        "compression_quality": 75,
        "max_concurrent_visualizations": 2,
        "enable_gpu_acceleration": True,
        "cache_size_mb": 128,
        "processing_timeout": 10.0
    },
    
    PerformanceMode.QUALITY: {
        "default_quality": VisualizationQuality.HIGH,
        "default_format": VisualizationFormat.PNG,
        "compression_quality": 90,
        "max_concurrent_visualizations": 4,
        "enable_gpu_acceleration": True,
        "cache_size_mb": 256,
        "processing_timeout": 20.0
    }
}

# ==============================================
# 🍎 M3 Max 특화 설정
# ==============================================

M3_MAX_VISUALIZATION_SETTINGS = {
    "enable_metal_acceleration": True,
    "memory_pool_size_mb": 2048,  # 2GB 전용 메모리 풀
    "max_texture_size": (4096, 4096),
    "enable_neural_engine": True,
    "batch_processing": True,
    "memory_mapping": True,
    "async_processing": True,
    "gpu_memory_fraction": 0.6,  # GPU 메모리의 60% 사용
    "optimization_level": "maximum"
}

# ==============================================
# 🔧 전역 시각화 설정
# ==============================================

GLOBAL_VISUALIZATION_CONFIG = {
    "enabled": True,
    "performance_mode": PerformanceMode.BALANCED,
    "m3_max_optimized": True,
    "debug_mode": False,
    "save_intermediate_results": False,
    "enable_analytics": True,
    "default_output_directory": "static/visualizations",
    "cache_directory": "static/cache/visualizations",
    "max_cache_size_gb": 2.0,
    "cache_ttl_hours": 24,
    "enable_compression": True,
    "enable_watermarking": False,
    "max_visualization_time_seconds": 30.0
}

# ==============================================
# 🔧 설정 관리 함수들
# ==============================================

def get_visualization_config() -> Dict[str, Any]:
    """전체 시각화 설정 반환"""
    try:
        return {
            "global_settings": GLOBAL_VISUALIZATION_CONFIG,
            "step_configs": {k: _config_to_dict(v) for k, v in STEP_VISUALIZATION_CONFIG.items()},
            "performance_modes": {k.value: v for k, v in PERFORMANCE_MODE_SETTINGS.items()},
            "m3_max_settings": M3_MAX_VISUALIZATION_SETTINGS,
            "quality_levels": {q.value: _get_quality_settings(q) for q in VisualizationQuality},
            "supported_formats": [f.value for f in VisualizationFormat]
        }
        
    except Exception as e:
        logger.error(f"❌ 시각화 설정 조회 실패: {e}")
        return _get_default_config()

def get_step_visualization_config(step_id: int) -> VisualizationConfig:
    """단계별 시각화 설정 반환"""
    try:
        if step_id in STEP_VISUALIZATION_CONFIG:
            return STEP_VISUALIZATION_CONFIG[step_id]
        else:
            logger.warning(f"⚠️ Step {step_id} 설정 없음 - 기본 설정 사용")
            return _get_default_step_config()
            
    except Exception as e:
        logger.error(f"❌ Step {step_id} 설정 조회 실패: {e}")
        return _get_default_step_config()

def is_visualization_enabled(step_id: int) -> bool:
    """단계별 시각화 활성화 여부 확인"""
    try:
        # 전역 설정 확인
        if not GLOBAL_VISUALIZATION_CONFIG.get("enabled", True):
            return False
        
        # 단계별 설정 확인
        step_config = get_step_visualization_config(step_id)
        return step_config.enabled
        
    except Exception as e:
        logger.error(f"❌ Step {step_id} 시각화 활성화 여부 확인 실패: {e}")
        return False

def update_visualization_config(step_id: int, **kwargs) -> bool:
    """시각화 설정 업데이트"""
    try:
        if step_id not in STEP_VISUALIZATION_CONFIG:
            logger.warning(f"⚠️ Step {step_id} 설정이 존재하지 않습니다")
            return False
        
        config = STEP_VISUALIZATION_CONFIG[step_id]
        updated_fields = []
        
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
                updated_fields.append(key)
            elif key in config.custom_settings:
                config.custom_settings[key] = value
                updated_fields.append(f"custom_settings.{key}")
            else:
                logger.warning(f"⚠️ 알 수 없는 설정 필드: {key}")
        
        if updated_fields:
            logger.info(f"✅ Step {step_id} 시각화 설정 업데이트: {', '.join(updated_fields)}")
            return True
        else:
            logger.warning(f"⚠️ Step {step_id} 업데이트할 설정이 없습니다")
            return False
            
    except Exception as e:
        logger.error(f"❌ Step {step_id} 시각화 설정 업데이트 실패: {e}")
        return False

def set_performance_mode(mode: PerformanceMode) -> bool:
    """성능 모드 설정"""
    try:
        if mode not in PERFORMANCE_MODE_SETTINGS:
            logger.error(f"❌ 지원되지 않는 성능 모드: {mode}")
            return False
        
        GLOBAL_VISUALIZATION_CONFIG["performance_mode"] = mode
        
        # 모든 단계별 설정에 성능 모드 적용
        mode_settings = PERFORMANCE_MODE_SETTINGS[mode]
        
        for step_id, config in STEP_VISUALIZATION_CONFIG.items():
            if "default_quality" in mode_settings:
                config.quality = mode_settings["default_quality"]
            if "default_format" in mode_settings:
                config.format = VisualizationFormat(mode_settings["default_format"])
            if "compression_quality" in mode_settings:
                config.compression_quality = mode_settings["compression_quality"]
            if "enable_gpu_acceleration" in mode_settings:
                config.gpu_accelerated = mode_settings["enable_gpu_acceleration"]
        
        logger.info(f"✅ 성능 모드 '{mode.value}'로 설정 완료")
        return True
        
    except Exception as e:
        logger.error(f"❌ 성능 모드 설정 실패: {e}")
        return False

def enable_m3_max_optimization() -> bool:
    """M3 Max 최적화 활성화"""
    try:
        GLOBAL_VISUALIZATION_CONFIG.update(M3_MAX_VISUALIZATION_SETTINGS)
        GLOBAL_VISUALIZATION_CONFIG["m3_max_optimized"] = True
        
        # 모든 단계에 M3 Max 최적화 적용
        for config in STEP_VISUALIZATION_CONFIG.values():
            config.gpu_accelerated = True
            config.memory_efficient = True
            
            # 고품질 설정 적용 (M3 Max의 강력한 성능 활용)
            if config.quality == VisualizationQuality.LOW:
                config.quality = VisualizationQuality.MEDIUM
            
        logger.info("✅ M3 Max 최적화 활성화 완료")
        return True
        
    except Exception as e:
        logger.error(f"❌ M3 Max 최적화 활성화 실패: {e}")
        return False

def get_optimal_settings_for_device() -> Dict[str, Any]:
    """디바이스에 최적화된 설정 반환"""
    try:
        import platform
        import subprocess
        
        # M3 Max 감지
        is_m3_max = False
        if platform.system() == "Darwin":  # macOS
            try:
                result = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'], 
                                      capture_output=True, text=True, timeout=5)
                if "M3 Max" in result.stdout:
                    is_m3_max = True
            except:
                pass
        
        if is_m3_max:
            # M3 Max 최적화 설정
            enable_m3_max_optimization()
            return {
                "device_type": "M3_Max",
                "performance_mode": PerformanceMode.QUALITY,
                "recommended_settings": M3_MAX_VISUALIZATION_SETTINGS,
                "optimization_applied": True
            }
        else:
            # 일반 시스템 설정
            set_performance_mode(PerformanceMode.BALANCED)
            return {
                "device_type": "Generic",
                "performance_mode": PerformanceMode.BALANCED,
                "recommended_settings": PERFORMANCE_MODE_SETTINGS[PerformanceMode.BALANCED],
                "optimization_applied": False
            }
            
    except Exception as e:
        logger.error(f"❌ 디바이스 최적화 설정 실패: {e}")
        return {
            "device_type": "Unknown",
            "performance_mode": PerformanceMode.FAST,
            "optimization_applied": False,
            "error": str(e)
        }

# ==============================================
# 🔧 헬퍼 함수들
# ==============================================

def _config_to_dict(config: VisualizationConfig) -> Dict[str, Any]:
    """VisualizationConfig를 딕셔너리로 변환"""
    return {
        "enabled": config.enabled,
        "quality": config.quality.value,
        "format": config.format.value,
        "max_size": config.max_size,
        "compression_quality": config.compression_quality,
        "enable_caching": config.enable_caching,
        "memory_efficient": config.memory_efficient,
        "gpu_accelerated": config.gpu_accelerated,
        "show_confidence": config.show_confidence,
        "show_timing": config.show_timing,
        "custom_settings": config.custom_settings
    }

def _get_quality_settings(quality: VisualizationQuality) -> Dict[str, Any]:
    """품질 레벨별 설정 반환"""
    quality_map = {
        VisualizationQuality.LOW: {
            "resolution": (512, 512),
            "compression": 60,
            "processing_time_limit": 3.0
        },
        VisualizationQuality.MEDIUM: {
            "resolution": (768, 768),
            "compression": 75,
            "processing_time_limit": 8.0
        },
        VisualizationQuality.HIGH: {
            "resolution": (1024, 1024),
            "compression": 85,
            "processing_time_limit": 15.0
        },
        VisualizationQuality.ULTRA: {
            "resolution": (1536, 1536),
            "compression": 95,
            "processing_time_limit": 30.0
        }
    }
    return quality_map.get(quality, quality_map[VisualizationQuality.MEDIUM])

def _get_default_step_config() -> VisualizationConfig:
    """기본 Step 설정 반환"""
    return VisualizationConfig(
        enabled=True,
        quality=VisualizationQuality.MEDIUM,
        format=VisualizationFormat.JPEG,
        max_size=(800, 600),
        compression_quality=75
    )

def _get_default_config() -> Dict[str, Any]:
    """기본 전체 설정 반환"""
    return {
        "global_settings": {
            "enabled": True,
            "performance_mode": "balanced",
            "m3_max_optimized": False
        },
        "step_configs": {},
        "error": "설정 로드 실패 - 기본 설정 사용"
    }

def get_visualization_stats() -> Dict[str, Any]:
    """시각화 통계 정보 반환"""
    try:
        enabled_steps = sum(1 for config in STEP_VISUALIZATION_CONFIG.values() if config.enabled)
        
        return {
            "total_steps": len(STEP_VISUALIZATION_CONFIG),
            "enabled_steps": enabled_steps,
            "disabled_steps": len(STEP_VISUALIZATION_CONFIG) - enabled_steps,
            "global_enabled": GLOBAL_VISUALIZATION_CONFIG["enabled"],
            "performance_mode": GLOBAL_VISUALIZATION_CONFIG["performance_mode"].value,
            "m3_max_optimized": GLOBAL_VISUALIZATION_CONFIG["m3_max_optimized"],
            "cache_enabled": any(config.enable_caching for config in STEP_VISUALIZATION_CONFIG.values()),
            "gpu_acceleration_enabled": any(config.gpu_accelerated for config in STEP_VISUALIZATION_CONFIG.values())
        }
        
    except Exception as e:
        logger.error(f"❌ 시각화 통계 조회 실패: {e}")
        return {"error": str(e)}

# ==============================================
# 🎉 모듈 초기화
# ==============================================

def _initialize_visualization_config():
    """시각화 설정 모듈 초기화"""
    try:
        # 디바이스 최적화 설정 적용
        device_settings = get_optimal_settings_for_device()
        
        logger.info("✅ 시각화 설정 모듈 초기화 완료")
        logger.info(f"🔧 디바이스: {device_settings.get('device_type', 'Unknown')}")
        logger.info(f"⚡ 성능 모드: {GLOBAL_VISUALIZATION_CONFIG['performance_mode'].value}")
        logger.info(f"🍎 M3 Max 최적화: {'✅' if GLOBAL_VISUALIZATION_CONFIG.get('m3_max_optimized') else '❌'}")
        logger.info(f"📊 활성화된 Step: {len([c for c in STEP_VISUALIZATION_CONFIG.values() if c.enabled])}개")
        
    except Exception as e:
        logger.error(f"❌ 시각화 설정 초기화 실패: {e}")

# 모듈 로드 시 자동 초기화
_initialize_visualization_config()

# ==============================================
# 🎉 Export
# ==============================================

__all__ = [
    # 데이터 클래스
    'VisualizationConfig',
    'VisualizationQuality',
    'VisualizationFormat', 
    'PerformanceMode',
    
    # 설정 딕셔너리
    'STEP_VISUALIZATION_CONFIG',
    'GLOBAL_VISUALIZATION_CONFIG',
    'M3_MAX_VISUALIZATION_SETTINGS',
    
    # 주요 함수들
    'get_visualization_config',
    'get_step_visualization_config',
    'is_visualization_enabled',
    'update_visualization_config',
    'set_performance_mode',
    'enable_m3_max_optimization',
    'get_optimal_settings_for_device',
    'get_visualization_stats'
]

logger.info("🎨 시각화 설정 시스템 로드 완료!")
logger.info(f"📊 {len(STEP_VISUALIZATION_CONFIG)}개 Step 시각화 설정 준비")
logger.info("✅ M3 Max 최적화 지원")
logger.info("🔧 동적 성능 모드 지원")