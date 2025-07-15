# backend/app/core/__init__.py
"""
핵심 설정 및 구성 모듈 - 통합 개선 버전
- 개선된 설정 시스템 통합
- 최적 생성자 패턴 파이프라인 설정
- GPU 설정, 보안, 설정 관리 등
- M3 Max 최적화 지원
- 하위 호환성 100% 보장
"""

import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# =======================================
# 🔧 개선된 설정 시스템 import
# =======================================

# 메인 설정 먼저 import
try:
    from .config import (
        # 기본 설정 인스턴스들
        get_settings, 
        settings,
        config,  # 하위 호환성
        app_config,  # 하위 호환성
        ai_config,  # 하위 호환성
        
        # 팩토리 함수들
        create_development_config,
        create_production_config,
        create_testing_config,
        create_m3_max_config,
        
        # 유틸리티 함수들
        validate_environment,
        get_configuration_summary,
        ensure_directories,
        cleanup_temp_files
    )
    
    CONFIG_SYSTEM_AVAILABLE = True
    logger.info("✅ 개선된 설정 시스템 로드 완료")
    
except ImportError as e:
    logger.error(f"❌ 개선된 설정 시스템 import 실패: {e}")
    CONFIG_SYSTEM_AVAILABLE = False
    
    # 폴백: 기본 설정만 사용
    try:
        from .config import get_settings, settings
        logger.warning("⚠️ 기본 설정만 사용")
    except ImportError:
        logger.critical("💥 설정 시스템을 전혀 로드할 수 없습니다")
        raise

# =======================================
# 🎯 개선된 파이프라인 설정 import
# =======================================

try:
    from .pipeline_config import (
        # 핵심 클래스들
        PipelineConfig,
        OptimalConstructorBase,
        SystemConfig,
        
        # 팩토리 함수들 (기본)
        get_pipeline_config,
        get_step_configs,
        get_model_paths,
        create_custom_config,
        create_optimized_config,
        
        # 생성자 패턴 함수들
        create_optimal_pipeline_config,  # 새로운 최적 방식
        create_legacy_pipeline_config,   # 기존 호환성
        create_advanced_pipeline_config, # 고급 설정
        
        # 환경별 설정 함수들
        configure_for_development,
        configure_for_production,
        configure_for_testing,
        configure_for_m3_max,
        configure_for_low_memory,
        configure_for_high_performance,
        
        # 설정 관리 유틸리티
        compare_configs,
        merge_configs,
        create_config_profile,
        get_config_templates,
        create_config_from_template,
        
        # 검증 및 호환성
        validate_optimal_constructor_compatibility
    )
    
    PIPELINE_CONFIG_AVAILABLE = True
    logger.info("✅ 최적 생성자 패턴 파이프라인 설정 로드 완료")
    
except ImportError as e:
    logger.error(f"❌ 파이프라인 설정 import 실패: {e}")
    PIPELINE_CONFIG_AVAILABLE = False
    
    # 폴백: 기본 파이프라인 설정
    class FallbackPipelineConfig:
        def __init__(self, quality_level="high", device="auto", **kwargs):
            self.quality_level = quality_level
            self.device = device
            self.config = {"pipeline": {"quality_level": quality_level}}
        
        def get_step_config(self, step_name):
            return {}
        
        def get_model_path(self, model_name):
            return f"models/{model_name}"
    
    def get_pipeline_config(**kwargs):
        return FallbackPipelineConfig(**kwargs)
    
    def create_optimal_pipeline_config(**kwargs):
        return FallbackPipelineConfig(**kwargs)
    
    logger.warning("⚠️ 폴백 파이프라인 설정 사용")

# =======================================
# 🖥️ GPU 설정 import (안전하게 + 통합)
# =======================================

try:
    from .gpu_config import (
        gpu_config, 
        DEVICE, 
        DEVICE_INFO, 
        MODEL_CONFIG,
        get_device,
        get_device_info,
        get_optimal_settings,
        optimize_memory,
        check_memory_available
    )
    
    GPU_CONFIG_AVAILABLE = True
    logger.info("✅ GPU 설정 로드 완료")
    
    # GPU 설정과 메인 설정 통합
    if CONFIG_SYSTEM_AVAILABLE:
        try:
            main_settings = get_settings()
            
            # 메인 설정의 디바이스 설정을 GPU 설정과 동기화
            if hasattr(main_settings, 'DEVICE') and main_settings.DEVICE != "auto":
                # 메인 설정이 명시적 디바이스를 지정한 경우 사용
                DEVICE = main_settings.DEVICE
                logger.info(f"🔄 메인 설정에서 디바이스 동기화: {DEVICE}")
            
        except Exception as e:
            logger.warning(f"⚠️ GPU/메인 설정 동기화 실패: {e}")
    
except ImportError as e:
    # 폴백 GPU 설정 (기존 로직 유지)
    logger.warning(f"⚠️ GPU 설정 import 실패: {e}")
    GPU_CONFIG_AVAILABLE = False
    
    # 메인 설정에서 디바이스 정보 가져오기 시도
    if CONFIG_SYSTEM_AVAILABLE:
        try:
            main_settings = get_settings()
            DEVICE = getattr(main_settings, 'DEVICE', 'cpu')
            DEVICE_INFO = main_settings.get_device_info() if hasattr(main_settings, 'get_device_info') else {"device": DEVICE}
            MODEL_CONFIG = {"device": DEVICE, "batch_size": getattr(main_settings, 'BATCH_SIZE', 1)}
        except Exception:
            DEVICE = "cpu"
            DEVICE_INFO = {"device": "cpu", "error": str(e)}
            MODEL_CONFIG = {"device": "cpu", "batch_size": 1}
    else:
        DEVICE = "cpu"
        DEVICE_INFO = {"device": "cpu", "error": str(e)}
        MODEL_CONFIG = {"device": "cpu", "batch_size": 1}
    
    def get_device():
        return DEVICE
    
    def get_device_info():
        return DEVICE_INFO
    
    def get_optimal_settings():
        return MODEL_CONFIG
    
    def optimize_memory():
        logger.info("💾 폴백 메모리 최적화 실행")
        import gc
        gc.collect()
    
    def check_memory_available(required_gb=4.0):
        return True
    
    # 더미 gpu_config 객체 (개선된 버전)
    class DummyGPUConfig:
        def __init__(self):
            self.device = DEVICE
            self.device_info = DEVICE_INFO
            self.model_config = MODEL_CONFIG
        
        def get_model_config(self):
            return self.model_config
        
        def get_device_config(self):
            return DEVICE_INFO
        
        def optimize_memory(self):
            optimize_memory()
        
        def get_system_info(self):
            return {
                "device": self.device,
                "available": False,
                "fallback_mode": True
            }
    
    gpu_config = DummyGPUConfig()

# =======================================
# 🔧 통합 설정 헬퍼 함수들
# =======================================

def get_integrated_config() -> Dict[str, Any]:
    """모든 설정을 통합하여 반환"""
    integrated = {
        "system_status": {
            "config_system": CONFIG_SYSTEM_AVAILABLE,
            "pipeline_config": PIPELINE_CONFIG_AVAILABLE,
            "gpu_config": GPU_CONFIG_AVAILABLE
        }
    }
    
    # 메인 설정 정보
    if CONFIG_SYSTEM_AVAILABLE:
        try:
            main_settings = get_settings()
            integrated["main_settings"] = {
                "app_name": main_settings.APP_NAME,
                "version": main_settings.APP_VERSION,
                "device": main_settings.DEVICE,
                "debug": main_settings.DEBUG,
                "m3_max_optimized": getattr(main_settings, '_is_m3_max', False)
            }
        except Exception as e:
            integrated["main_settings"] = {"error": str(e)}
    
    # 파이프라인 설정 정보
    if PIPELINE_CONFIG_AVAILABLE:
        try:
            pipeline_config = get_pipeline_config()
            integrated["pipeline_config"] = {
                "quality_level": pipeline_config.quality_level,
                "device": pipeline_config.device,
                "optimization_enabled": pipeline_config.optimization_enabled,
                "constructor_pattern": "optimal"
            }
        except Exception as e:
            integrated["pipeline_config"] = {"error": str(e)}
    
    # GPU 설정 정보
    integrated["gpu_config"] = {
        "device": DEVICE,
        "device_info": DEVICE_INFO,
        "available": GPU_CONFIG_AVAILABLE
    }
    
    return integrated

def validate_all_configs() -> Dict[str, Any]:
    """모든 설정 시스템 검증"""
    validation_result = {
        "overall_valid": True,
        "errors": [],
        "warnings": [],
        "system_checks": {}
    }
    
    # 메인 설정 검증
    if CONFIG_SYSTEM_AVAILABLE:
        try:
            env_validation = validate_environment()
            validation_result["system_checks"]["environment"] = env_validation
            if not env_validation.get("settings_valid", True):
                validation_result["overall_valid"] = False
                validation_result["errors"].append("메인 설정 검증 실패")
        except Exception as e:
            validation_result["errors"].append(f"환경 검증 실패: {e}")
    
    # 파이프라인 설정 검증
    if PIPELINE_CONFIG_AVAILABLE:
        try:
            pipeline_config = get_pipeline_config()
            pipeline_validation = pipeline_config.validate_config()
            validation_result["system_checks"]["pipeline"] = pipeline_validation
            if not pipeline_validation.get("valid", True):
                validation_result["overall_valid"] = False
                validation_result["errors"].extend(pipeline_validation.get("errors", []))
        except Exception as e:
            validation_result["errors"].append(f"파이프라인 설정 검증 실패: {e}")
    
    # 호환성 검증
    if PIPELINE_CONFIG_AVAILABLE:
        try:
            compatibility = validate_optimal_constructor_compatibility()
            validation_result["system_checks"]["compatibility"] = compatibility
            if not compatibility.get("overall_compatible", True):
                validation_result["warnings"].append("최적 생성자 패턴 호환성 문제")
        except Exception as e:
            validation_result["warnings"].append(f"호환성 검증 실패: {e}")
    
    return validation_result

def create_unified_config(
    environment: str = "development",
    device: Optional[str] = None,
    quality_level: str = "high",
    **kwargs
) -> Dict[str, Any]:
    """통합된 설정 생성"""
    
    # 환경별 메인 설정 생성
    if CONFIG_SYSTEM_AVAILABLE:
        if environment == "development":
            main_config = create_development_config(**kwargs)
        elif environment == "production":
            main_config = create_production_config(**kwargs)
        elif environment == "testing":
            main_config = create_testing_config(**kwargs)
        elif environment == "m3_max":
            main_config = create_m3_max_config(**kwargs)
        else:
            main_config = get_settings()
    else:
        main_config = None
    
    # 환경별 파이프라인 설정 생성
    if PIPELINE_CONFIG_AVAILABLE:
        if environment == "development":
            pipeline_config = configure_for_development(device=device, **kwargs)
        elif environment == "production":
            pipeline_config = configure_for_production(device=device, **kwargs)
        elif environment == "testing":
            pipeline_config = configure_for_testing(device=device, **kwargs)
        elif environment == "m3_max":
            pipeline_config = configure_for_m3_max(device=device, **kwargs)
        else:
            pipeline_config = get_pipeline_config(quality_level=quality_level, device=device, **kwargs)
    else:
        pipeline_config = get_pipeline_config(quality_level=quality_level, device=device)
    
    return {
        "main_config": main_config,
        "pipeline_config": pipeline_config,
        "environment": environment,
        "unified": True
    }

def optimize_system_memory():
    """시스템 메모리 최적화 (통합)"""
    try:
        # GPU 설정의 메모리 최적화
        optimize_memory()
        
        # 메인 설정의 임시 파일 정리
        if CONFIG_SYSTEM_AVAILABLE:
            cleanup_temp_files()
        
        # 파이프라인 설정의 메모리 최적화
        if PIPELINE_CONFIG_AVAILABLE:
            try:
                pipeline_config = get_pipeline_config()
                if hasattr(pipeline_config, 'config') and 'memory' in pipeline_config.config:
                    memory_config = pipeline_config.get_memory_config()
                    if memory_config.get('optimization', False):
                        import gc
                        gc.collect()
                        logger.info("🧹 파이프라인 메모리 최적화 실행")
            except Exception as e:
                logger.warning(f"⚠️ 파이프라인 메모리 최적화 실패: {e}")
        
        logger.info("✅ 시스템 메모리 최적화 완료")
        return True
        
    except Exception as e:
        logger.error(f"❌ 시스템 메모리 최적화 실패: {e}")
        return False

def get_system_status() -> Dict[str, Any]:
    """시스템 전체 상태 조회"""
    return {
        "timestamp": __import__('datetime').datetime.now().isoformat(),
        "integrated_config": get_integrated_config(),
        "validation": validate_all_configs(),
        "memory_status": {
            "optimization_available": True,
            "cleanup_available": CONFIG_SYSTEM_AVAILABLE
        }
    }

# =======================================
# 📤 모듈 내보내기 (기존 + 새로운)
# =======================================

__all__ = [
    # =======================================
    # 🔧 기존 호환성 (100% 유지)
    # =======================================
    # 설정
    "get_settings",
    "settings",
    
    # GPU 설정
    "gpu_config",
    "DEVICE", 
    "DEVICE_INFO",
    "MODEL_CONFIG",
    "get_device",
    "get_device_info", 
    "get_optimal_settings",
    "optimize_memory",
    "check_memory_available",
    
    # =======================================
    # 🆕 새로운 통합 설정 시스템
    # =======================================
    # 하위 호환성 설정 인스턴스들
    "config",
    "app_config", 
    "ai_config",
    
    # 메인 설정 팩토리 함수들
    "create_development_config",
    "create_production_config",
    "create_testing_config", 
    "create_m3_max_config",
    
    # 메인 설정 유틸리티
    "validate_environment",
    "get_configuration_summary",
    "ensure_directories",
    "cleanup_temp_files",
    
    # =======================================
    # 🎯 파이프라인 설정 시스템
    # =======================================
    # 핵심 클래스들
    "PipelineConfig",
    "OptimalConstructorBase",
    "SystemConfig",
    
    # 파이프라인 팩토리 함수들
    "get_pipeline_config",
    "get_step_configs", 
    "get_model_paths",
    "create_custom_config",
    "create_optimized_config",
    "create_optimal_pipeline_config",
    "create_legacy_pipeline_config",
    "create_advanced_pipeline_config",
    
    # 환경별 파이프라인 설정
    "configure_for_development",
    "configure_for_production",
    "configure_for_testing",
    "configure_for_m3_max",
    "configure_for_low_memory",
    "configure_for_high_performance",
    
    # 파이프라인 유틸리티
    "compare_configs",
    "merge_configs", 
    "create_config_profile",
    "get_config_templates",
    "create_config_from_template",
    "validate_optimal_constructor_compatibility",
    
    # =======================================
    # 🔗 통합 시스템 함수들
    # =======================================
    "get_integrated_config",
    "validate_all_configs",
    "create_unified_config",
    "optimize_system_memory",
    "get_system_status",
    
    # =======================================
    # 🔍 시스템 상태 플래그들
    # =======================================
    "CONFIG_SYSTEM_AVAILABLE",
    "PIPELINE_CONFIG_AVAILABLE", 
    "GPU_CONFIG_AVAILABLE"
]

# =======================================
# 📊 초기화 로깅
# =======================================

def _log_initialization_status():
    """초기화 상태 로깅"""
    logger.info("=" * 70)
    logger.info("🎯 MyCloset AI Core 모듈 초기화 완료")
    logger.info("=" * 70)
    logger.info(f"✅ 메인 설정 시스템: {'활성화' if CONFIG_SYSTEM_AVAILABLE else '❌ 비활성화'}")
    logger.info(f"✅ 파이프라인 설정: {'활성화' if PIPELINE_CONFIG_AVAILABLE else '❌ 비활성화'}")
    logger.info(f"✅ GPU 설정: {'활성화' if GPU_CONFIG_AVAILABLE else '❌ 비활성화'}")
    logger.info(f"🔧 통합 함수: {len([x for x in __all__ if x.startswith(('get_', 'create_', 'validate_'))])}개")
    logger.info(f"📤 총 내보내기: {len(__all__)}개")
    
    # 디바이스 정보
    try:
        device_info = get_device_info()
        if isinstance(device_info, dict):
            device_name = device_info.get('device', 'unknown')
        else:
            device_name = str(device_info)
        logger.info(f"🖥️ 디바이스: {device_name}")
    except Exception:
        logger.info(f"🖥️ 디바이스: {DEVICE}")
    
    # M3 Max 정보  
    if CONFIG_SYSTEM_AVAILABLE:
        try:
            main_settings = get_settings()
            is_m3_max = getattr(main_settings, '_is_m3_max', False)
            memory_gb = getattr(main_settings, '_system_memory_gb', 0)
            logger.info(f"🍎 M3 Max: {'✅ 감지됨' if is_m3_max else '❌ 일반 시스템'} ({memory_gb}GB)")
        except Exception:
            logger.info("🍎 M3 Max: ❓ 감지 불가")
    
    logger.info("=" * 70)

# 초기화 상태 로깅 실행
_log_initialization_status()

# 초기 시스템 검증 (옵션)
try:
    if CONFIG_SYSTEM_AVAILABLE and PIPELINE_CONFIG_AVAILABLE:
        validation = validate_all_configs()
        if not validation.get("overall_valid", True):
            logger.warning("⚠️ 설정 시스템 검증에서 문제 발견됨")
            for error in validation.get("errors", []):
                logger.warning(f"   - {error}")
        else:
            logger.info("✅ 모든 설정 시스템 검증 통과")
except Exception as e:
    logger.warning(f"⚠️ 초기 검증 실패: {e}")

logger.info("🚀 MyCloset AI Core 모듈 준비 완료")